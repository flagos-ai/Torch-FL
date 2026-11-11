# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Python fast-launch replay for the FlagGems route.

The FlagGems Python route costs 60-90 us of host time per operator on H100, of
which ~95% is the wrapper's own Python work plus Triton's Python launcher.  A
measured 29 us of that is removable without touching a kernel: cache the compiled
kernel launcher and its argument layout once, then replay it per call patching
only the tensor slots.  Measured on the lean replay path (two wheels, four
implementation x wheel combinations):

* pointwise         12-16 us  = 1.4-1.8x vendor, 5.4-8.0x faster than the wrapper
* single-launch reduction, ``sum.dim``  10.75-11.69 us = 1.06-1.16x vendor (parity)
* single-launch reduction, ``_softmax`` 10.54-11.80 us = 1.24-1.39x
* two-launch full-reduce ``sum``        16.45-18.49 us = 1.17-1.29x

This module is a *launch* optimisation, not a kernel: it keeps FlagGems' compiled
kernel and its numerics bit-identical, composes with the existing ``flaggems``
routing key as an internal fast path, and falls back to the untouched callable for
every case it cannot replay.

Correctness rules (each one is a measured silent-wrongness class - wrong numbers,
no exception raised - so they are invariants here, not tuning):

1. Slots are classified by **first appearance**, never by ``data_ptr`` equality
   with the returned tensor: FlagGems launches into an internal buffer and
   returns ``buffer.view(...)``, so pointer equality fails.  A non-input slot
   first seen in the **last** launch is an output; earlier ones are
   intermediates.  The rule is op-agnostic but *not* algorithm-agnostic: it
   requires the output to be introduced in the final launch, which is exactly why
   a two-launch full-reduce replays and a ping-pong ``sort`` (21 launches) cannot.
2. Intermediates are deduplicated **per distinct pointer**, one persistent buffer
   each, shared by every launch that passes it.  Allocating per slot occurrence
   instead let launch 2 read a fresh empty buffer rather than launch 1's partial
   sums (measured ``max_err`` 2312.5 on a 1M-element sum).
3. Capture keys include shape, strides and contiguity, and a mismatched replay is
   **refused**, never attempted: replaying a 1024^2 capture on a 512^2 input
   wrote past the smaller buffer (``max_err`` 31.38, corrupted allocator).
4. Intermediates are persistent and **never re-initialised**: launch 0 fully
   overwrites them, and zeroing measured +30 us on a 16.5 us op for no accuracy
   change.
5. The output buffer is allocated at the **result** shape and the captured
   shape/strides are supplied through the argument wrapper, rather than
   allocating at the captured buffer shape and calling ``.view()`` per call -
   measured 1.75-2.35 us per call on the squeezed-view case.
6. The replay path patches pointers into a **prebuilt** argument list; it rebuilds
   no argument list and resolves no index per call (measured ~2 us per call).

Together rules 5 and 6 were worth ~4 us/op, which is larger than the entire
remaining gap between ``sum.dim`` and the vendor kernel, so a naive
implementation of this module would leave most of the win on the table.

Scope of the skeleton: functional (non-``out=``) calls with positional tensor
arguments only; anything else falls back.  The production switch is expected to
be a ``FLAGOS_*`` variable registered in ``torch_fl/_env.py`` with its mirror in
``docs/reference/environment-variables.md``; this module deliberately reads no
environment so that it stays importable (and unit-testable) without torch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

__all__ = [
    "CapturedLaunch",
    "LaunchPlan",
    "OutSpec",
    "Plan",
    "PlanCache",
    "Provider",
    "Signature",
    "Stats",
    "Unsupported",
    "build_plan",
    "make_replay",
    "plans",
    "signature_of",
    "wrap_callable",
]

# A capture this long is an algorithm with ping-pong buffers rather than a
# straight-line kernel sequence (``sort`` measured 21 launches over 9 buffers).
MAX_LAUNCHES = 4
# More output slots than this has not been characterised (``max.dim`` has two).
MAX_OUTPUTS = 4


class Unsupported(Exception):
    """Raised when a capture cannot be replayed safely; the caller falls back."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Slot:
    """One tensor-valued argument position inside a captured launch."""

    position: int
    """Index into the launch's positional arguments."""
    kind: str
    """"in", "out" or "scratch"."""
    key: int
    """Input index (``in``) or ``data_ptr`` (``out``/``scratch``)."""


@dataclass
class CapturedLaunch:
    """One recorded kernel launch: the bound launcher and the arguments it saw."""

    call: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaunchPlan:
    """Replay state for a single launch; ``arg_list`` is built once, then patched."""

    call: Callable[..., Any]
    kwargs: dict[str, Any]
    template: tuple[Any, ...]
    arg_list: list[Any]
    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class OutSpec:
    """How one output buffer is allocated and exposed for a given plan."""

    key: int
    rank: int
    """Position of this output in the callable's return value (call order)."""
    dtype: Any
    device: Any
    buffer_shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    contiguous: bool

    @property
    def flat(self) -> bool:
        """True when the captured buffer can be allocated at the result shape."""
        return self.contiguous and _prod(self.buffer_shape) == _prod(self.result_shape)


@dataclass
class Plan:
    """A replayable capture for one operator signature."""

    op: str
    launches: tuple[LaunchPlan, ...]
    outs: tuple[OutSpec, ...]
    scratch: dict[int, Any]
    cohort: tuple[Any, ...]
    launch_count: int
    returns_tuple: bool = False

    def replay(
        self, inputs: Sequence[Any], provider: "Provider"
    ) -> list[Any] | tuple[Any, ...]:
        """Replay the plan; returns the fresh output tensors in call order.

        ``self.outs`` is sorted by ``OutSpec.rank``, so the returned buffers line up
        with the order the wrapped callable returns them.
        """
        buffers: list[Any] = []
        for spec in self.outs:
            if spec.flat:
                buffers.append(
                    provider.empty(spec.result_shape, spec.dtype, spec.device)
                )
            else:
                base = provider.empty_like(self._out_template(spec.key))
                buffers.append(base.view(spec.result_shape))
        by_key = dict(zip((spec.key for spec in self.outs), buffers, strict=True))
        for launch in self.launches:
            for slot in launch.slots:
                if slot.kind == "in":
                    source = inputs[slot.key]
                elif slot.kind == "out":
                    source = by_key[slot.key]
                else:
                    source = self.scratch[slot.key]
                launch.arg_list[slot.position] = provider.rebuild(
                    launch.template[slot.position], source
                )
            launch.call(*launch.arg_list, **launch.kwargs)
        return tuple(buffers) if self.returns_tuple else buffers

    def _out_template(self, key: int) -> Any:
        for launch in self.launches:
            for slot in launch.slots:
                if slot.kind == "out" and slot.key == key:
                    return launch.template[slot.position]
        raise AssertionError(f"no template for output {key}")  # pragma: no cover


def _prod(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


@dataclass(frozen=True)
class Signature:
    """Cache key: everything that changes the launched kernel or its arguments."""

    op: str
    tensors: tuple[tuple[str, tuple[int, ...], tuple[int, ...], bool], ...]
    kwargs: tuple[tuple[str, str], ...] = ()

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{self.op}({len(self.tensors)} tensors)"


@dataclass
class Stats:
    """Counters for tests and for the eventual ``FLAGOS_LOG`` integration."""

    hits: int = 0
    misses: int = 0
    fallbacks: int = 0
    captures: int = 0
    warmups: int = 0


class Provider(Protocol):
    """Everything device- and framework-specific, injected so tests stay hermetic."""

    def capture(self, call: Callable[[], Any]) -> tuple[Any, list[CapturedLaunch]]:
        """Run ``call`` once, returning its result and the launches it performed."""

    def unwrap(self, arg: Any) -> Any | None:
        """Return the tensor behind a launcher argument, or None for a scalar."""

    def storage_ptr(self, tensor: Any) -> int:
        """Base-storage pointer, so a ``view`` result matches the buffer it views."""

    def rebuild(self, template: Any, tensor: Any) -> Any:
        """Re-create a launcher argument of ``template``'s flavour around ``tensor``."""

    def empty(self, shape: Sequence[int], dtype: Any, device: Any = None) -> Any:
        """Allocate a tensor with ``shape``/``dtype`` on ``device`` (rule 5)."""

    def empty_like(self, template: Any) -> Any:
        """Allocate a tensor with ``template``'s captured shape, dtype and layout."""

    def cohort(self) -> tuple[Any, ...]:
        """Identity of the kernel cohort; a change invalidates every cached plan."""


class PlanCache:
    """Signature -> Plan, invalidated wholesale when the cohort changes."""

    def __init__(self) -> None:
        self._plans: dict[Signature, Plan] = {}
        self._cohort: tuple[Any, ...] | None = None

    def get(self, signature: Signature, cohort: tuple[Any, ...]) -> Plan | None:
        if self._cohort is not None and cohort != self._cohort:
            self.clear()
            return None
        return self._plans.get(signature)

    def put(self, signature: Signature, plan: Plan) -> None:
        self._cohort = plan.cohort
        self._plans[signature] = plan

    def clear(self) -> None:
        self._plans.clear()
        self._cohort = None

    def __len__(self) -> int:
        return len(self._plans)

    def items(self) -> list[tuple[Signature, Plan]]:
        return list(self._plans.items())


def _capture_safe_kwargs(kwargs: dict[str, Any], provider: Provider) -> bool:
    """True when every keyword is a scalar constant that the plan can bake in.

    The C++ bridge passes ``alpha``, ``dtype``, ``keepdim`` and friends as keyword
    arguments (100 of the generated kernels use ``CallPythonOp_GenericKw``), so
    rejecting keyword calls outright would exclude that whole surface. A scalar
    keyword is safe because it is part of the cache key: a different value produces
    a different plan and therefore a fresh capture. A tensor keyword (``out=``) is
    not: its storage would move between calls without changing the key.
    """
    for value in kwargs.values():
        if provider.unwrap(value) is not None:
            return False
        try:
            hash(value)
        except TypeError:
            return False
    return True


def signature_of(
    op: str,
    args: Sequence[Any],
    provider: Provider,
    kwargs: dict[str, Any] | None = None,
) -> Signature:
    """Build the cache key from every tensor operand and scalar keyword."""
    tensors = []
    for arg in args:
        tensor = provider.unwrap(arg)
        if tensor is None:
            continue
        tensors.append(
            (
                str(tensor.dtype),
                tuple(int(dim) for dim in tensor.shape),
                tuple(int(step) for step in tensor.stride()),
                bool(tensor.is_contiguous()),
            )
        )
    keyword_key = ()
    if kwargs:
        keyword_key = tuple(
            sorted((str(name), repr(value)) for name, value in kwargs.items())
        )
    return Signature(op=op, tensors=tuple(tensors), kwargs=keyword_key)


def build_plan(
    op: str,
    captured: Sequence[CapturedLaunch],
    inputs: Sequence[Any],
    provider: Provider,
    results: Sequence[Any],
    returns_tuple: bool = False,
) -> Plan:
    """Classify a capture into a replayable plan, or raise :class:`Unsupported`.

    ``inputs`` are the unwrapped input tensors of the call that was captured;
    ``results`` are the tensors that call returned, in call order.
    """
    if not captured:
        raise Unsupported("no kernel launches captured")
    if len(captured) > MAX_LAUNCHES:
        raise Unsupported(
            f"{len(captured)} launches: ping-pong algorithms cannot be replayed"
        )

    # Identity, not data_ptr: two inputs may alias one storage (``f(a, a.view(...))``)
    # and every zero-element tensor reports data_ptr() == 0, which would collapse the
    # whole table into one bogus entry.
    in_ids: dict[int, int] = {}
    for index, tensor in enumerate(inputs):
        in_ids.setdefault(id(tensor), index)
    for tensor in inputs:
        if tensor.data_ptr() == 0:
            raise Unsupported(
                "an input is a zero-element tensor (data_ptr 0 is ambiguous)"
            )

    # A launcher often receives a tensor that is not the caller's object but shares
    # its storage: FlagGems hands the kernel a normalized view of the input. Measured
    # on max_dim, where the input arrived as a distinct tensor with identical
    # shape/strides, so identity alone classified it as an output ("3 output slots
    # for 2 results"). Accept a storage match, but only when shape, strides and dtype
    # all agree - a reshaped view then fails closed instead of being patched with a
    # tensor of the wrong shape.
    in_by_storage: dict[int, int] = {}
    ambiguous: set[int] = set()
    for index, tensor in enumerate(inputs):
        storage = provider.storage_ptr(tensor)
        if storage not in in_by_storage:
            in_by_storage[storage] = index
            continue
        previous = inputs[in_by_storage[storage]]
        if (tuple(previous.shape), tuple(previous.stride()), previous.dtype) != (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        ):
            ambiguous.add(storage)

    def as_input(tensor: Any) -> int | None:
        exact = in_ids.get(id(tensor))
        if exact is not None:
            return exact
        storage = provider.storage_ptr(tensor)
        index = in_by_storage.get(storage)
        if index is None or storage in ambiguous:
            return None
        candidate = inputs[index]
        if (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype) != (
            tuple(candidate.shape),
            tuple(candidate.stride()),
            candidate.dtype,
        ):
            raise Unsupported("a launch argument is a reshaped view of an input")
        return index

    last = len(captured) - 1

    # Rule 1: first appearance decides, never pointer equality with the result.
    first_seen: dict[int, int] = {}
    for index, launch in enumerate(captured):
        for arg in launch.args:
            tensor = provider.unwrap(arg)
            if tensor is None or as_input(tensor) is not None:
                continue
            ptr = tensor.data_ptr()
            if ptr not in first_seen:
                first_seen[ptr] = index

    out_ptrs = [ptr for ptr, index in first_seen.items() if index == last]
    if not out_ptrs:
        raise Unsupported(
            "no output slot in the final launch: replay would write nowhere"
        )
    if len(out_ptrs) > MAX_OUTPUTS:
        raise Unsupported(f"{len(out_ptrs)} outputs is beyond the characterised set")
    if len(out_ptrs) != len(results):
        raise Unsupported(f"{len(out_ptrs)} output slots for {len(results)} results")

    # Which output slot produced which returned tensor? Pair on the BASE storage
    # pointer: a result is the captured buffer or a view of it, so
    # untyped_storage().data_ptr() matches while result.data_ptr() (storage_offset)
    # may not. Pointer *order* is unrelated to call order, so pairing
    # sorted(out_ptrs) with result_shapes would silently swap ``max.dim``'s
    # values/indices - the container then holds int64 where float32 belongs.
    result_by_storage: dict[int, int] = {}
    for rank, result in enumerate(results):
        storage = provider.storage_ptr(result)
        if storage in result_by_storage:
            raise Unsupported("two results share one storage: cannot pair outputs")
        result_by_storage[storage] = rank
    pairs: list[tuple[int, int]] = []
    for ptr in out_ptrs:
        rank = result_by_storage.get(ptr)
        if rank is None:
            raise Unsupported(
                "an output slot has no matching result storage; refusing to guess"
            )
        pairs.append((ptr, rank))
    if len({rank for _, rank in pairs}) != len(pairs):
        raise Unsupported("output pairing is not one-to-one")

    launches: list[LaunchPlan] = []
    scratch: dict[int, Any] = {}
    for launch in captured:
        slots: list[Slot] = []
        for position, arg in enumerate(launch.args):
            tensor = provider.unwrap(arg)
            if tensor is None:
                continue
            ptr = tensor.data_ptr()
            input_index = as_input(tensor)
            if input_index is not None:
                slots.append(Slot(position, "in", input_index))
            elif first_seen.get(ptr) == last:
                slots.append(Slot(position, "out", ptr))
            else:
                # Rule 2: one persistent buffer per distinct pointer.
                if ptr not in scratch:
                    # Rule 4: allocated once, never initialised, never re-created.
                    scratch[ptr] = provider.empty_like(tensor)
                slots.append(Slot(position, "scratch", ptr))
        if not slots:
            raise Unsupported("a captured launch has no tensor arguments")
        launches.append(
            LaunchPlan(
                call=launch.call,
                kwargs=dict(launch.kwargs),
                template=tuple(launch.args),
                arg_list=list(launch.args),
                slots=tuple(slots),
            )
        )

    outs: list[OutSpec] = []
    for key, rank in pairs:
        template = _find_template(launches, key, provider)
        outs.append(
            OutSpec(
                key=key,
                rank=rank,
                dtype=template.dtype,
                device=getattr(template, "device", None),
                buffer_shape=tuple(int(dim) for dim in template.shape),
                result_shape=tuple(int(dim) for dim in results[rank].shape),
                contiguous=bool(template.is_contiguous()),
            )
        )
    outs.sort(key=lambda spec: spec.rank)

    return Plan(
        op=op,
        launches=tuple(launches),
        outs=tuple(outs),
        scratch=scratch,
        cohort=provider.cohort(),
        launch_count=len(captured),
        returns_tuple=returns_tuple,
    )


def _find_template(launches: Sequence[LaunchPlan], key: int, provider: Provider) -> Any:
    for launch in launches:
        for slot in launch.slots:
            if slot.kind == "out" and slot.key == key:
                tensor = provider.unwrap(launch.template[slot.position])
                if tensor is not None:
                    return tensor
    raise Unsupported("output template not found")  # pragma: no cover


def make_replay(plan: Plan, provider: Provider) -> Callable[[Sequence[Any]], list[Any]]:
    """Return a closure that replays ``plan`` for a given input tuple."""
    return lambda inputs: plan.replay(inputs, provider)


_ACTIVE = threading.local()


def _in_capture() -> bool:
    return getattr(_ACTIVE, "capturing", False)


def wrap_callable(
    name: str,
    callable_: Callable[..., Any],
    *,
    provider: Provider,
    cache: PlanCache,
    stats: Stats | None = None,
    unsupported: set[str] | None = None,
    state: dict[str, str] | None = None,
) -> Callable[..., Any]:
    """Wrap one FlagGems entry point with warm-once, capture-once, replay-thereafter.

    The warm step is not optional: the *first* call of a new (shape, layout) class can
    trigger libtuner benchmarking, which emits ~19 kernel launches. Capturing that
    call would classify the operator as an unsupported multi-launch algorithm. So the
    first call runs untouched (absorbing tuning and codegen), the second captures the
    now-steady-state single launch, and later calls replay. The cost is one extra
    unoptimised execution per operator, paid once per process.
    """
    stats = stats if stats is not None else Stats()
    unsupported = unsupported if unsupported is not None else set()
    state = state if state is not None else {}

    def fast_launch(*args: Any, **kwargs: Any) -> Any:
        if _in_capture() or name in unsupported:
            return callable_(*args, **kwargs)
        if kwargs and not _capture_safe_kwargs(kwargs, provider):
            return callable_(*args, **kwargs)
        tensors = [t for t in (provider.unwrap(arg) for arg in args) if t is not None]
        if not tensors:
            return callable_(*args, **kwargs)

        # Warm first: the cold call may still be tuning, so it must not be captured.
        if state.get(name, "cold") == "cold":
            state[name] = "warm"
            stats.warmups += 1
            return callable_(*args, **kwargs)

        signature = signature_of(name, args, provider, kwargs)
        plan = cache.get(signature, provider.cohort())
        if plan is not None:
            stats.hits += 1
            try:
                result = plan.replay(tensors, provider)
            except Exception:  # noqa: BLE001 - a broken replay must not reach the user
                unsupported.add(name)
                stats.fallbacks += 1
                return callable_(*args, **kwargs)
            return _as_container(result, len(plan.outs))

        stats.misses += 1
        _ACTIVE.capturing = True
        try:
            result, captured = provider.capture(lambda: callable_(*args, **kwargs))
        finally:
            _ACTIVE.capturing = False
        stats.captures += 1

        results = list(result) if isinstance(result, (list, tuple)) else [result]
        try:
            plan = build_plan(
                name,
                captured,
                tensors,
                provider,
                results,
                returns_tuple=isinstance(result, tuple),
            )
        except Unsupported:
            # Mark the operator, not just this signature: an unsupported
            # algorithm shape stays unsupported for every shape.
            unsupported.add(name)
            stats.fallbacks += 1
            return result

        cache.put(signature, plan)
        state[name] = "ready"
        return result

    fast_launch.__name__ = getattr(callable_, "__name__", name)
    fast_launch.__doc__ = getattr(callable_, "__doc__", None)
    fast_launch.__wrapped__ = callable_  # type: ignore[attr-defined]
    return fast_launch


def _as_container(items: list[Any], count: int) -> Any:
    """Match the container the original callable returns (``max.dim`` returns a tuple)."""
    return items[0] if count == 1 else items


def install(module: Any, name: str, wrapper: Callable[..., Any]) -> Callable[..., Any]:
    """Replace ``module.name`` with ``wrapper``; returns the original callable."""
    original = getattr(module, name)
    setattr(module, name, wrapper)
    return original


class FlagGemsProvider:
    """The real provider: FlagGems' compiled launcher, flagos tensors, Triton.

    Everything here imports lazily, so importing this module never requires torch,
    triton or flag_gems - which is what keeps the plan/replay logic unit-testable on
    a machine with none of them. ``enable()`` is the only entry point that pays the
    import cost, and it raises a clear error when the stack is absent.
    """

    def __init__(self) -> None:
        try:
            import flag_gems  # noqa: F401
            import torch
            from flag_gems.utils.tensor_wrapper import StridedBuffer
            from triton.backends.nvidia.driver import CudaLauncher
        except ImportError as error:  # pragma: no cover - needs the device stack
            raise RuntimeError(
                "fast_launch needs torch, triton and flag_gems on a working "
                "accelerator; call enable() on the target box"
            ) from error
        self._torch = torch
        self._buffer_cls = StridedBuffer
        self._launcher_cls = CudaLauncher
        self._original_call = CudaLauncher.__call__
        self._sink: list[CapturedLaunch] | None = None

    # -- capture ---------------------------------------------------------
    def capture(self, call: Callable[[], Any]) -> tuple[Any, list[CapturedLaunch]]:
        sink: list[CapturedLaunch] = []
        original = self._original_call

        def patched(launcher: Any, *args: Any, **kwargs: Any) -> Any:
            bound = original.__get__(launcher, type(launcher))
            sink.append(CapturedLaunch(call=bound, args=args, kwargs=kwargs))
            return original(launcher, *args, **kwargs)

        self._launcher_cls.__call__ = patched  # type: ignore[method-assign]
        self._sink = sink
        try:
            result = call()
        finally:
            self._launcher_cls.__call__ = original  # type: ignore[method-assign]
            self._sink = None
        return result, sink

    # -- argument marshalling --------------------------------------------
    def unwrap(self, arg: Any) -> Any | None:
        torch = self._torch
        if isinstance(arg, torch.Tensor):
            return arg
        if isinstance(arg, self._buffer_cls):
            return arg._base
        for attr in ("tensor", "buf", "data", "_tensor"):
            candidate = getattr(arg, attr, None)
            if isinstance(candidate, torch.Tensor):
                return candidate
        return None

    def storage_ptr(self, tensor: Any) -> int:
        """Base-storage pointer: ``result.data_ptr()`` moves with ``storage_offset``."""
        return tensor.untyped_storage().data_ptr()

    def rebuild(self, template: Any, tensor: Any) -> Any:
        """Re-create ``template``'s argument flavour around ``tensor``.

        The captured shape/strides are carried over, which is what lets the output
        be allocated at the *result* shape (rule 5) while the kernel still sees the
        captured layout.
        """
        torch = self._torch
        if isinstance(template, torch.Tensor):
            return tensor
        if isinstance(template, self._buffer_cls):
            # A captured StridedBuffer's ``shape``/``stride()`` are TASK-space values
            # (collapsed ``(numel,)`` / ``(1,)``), not the tensor's logical shape.
            # Passing the logical shape back is silently wrong, so replay exactly what
            # was captured, including the dtype.
            return self._buffer_cls(
                tensor,
                shape=tuple(template.shape),
                strides=tuple(template.stride()),
                dtype=template.dtype,
            )
        duplicate = type(template).__new__(type(template))
        duplicate.__dict__.update(getattr(template, "__dict__", {}))
        for attr in ("tensor", "buf", "data", "_tensor"):
            if isinstance(getattr(duplicate, attr, None), torch.Tensor):
                setattr(duplicate, attr, tensor)
                return duplicate
        raise Unsupported(f"cannot rebuild argument of type {type(template)!r}")

    # -- allocation ------------------------------------------------------
    def empty(self, shape: Sequence[int], dtype: Any, device: Any = None) -> Any:
        torch = self._torch
        return torch.empty(
            tuple(shape), dtype=dtype, device=device or self._default_device()
        )

    def empty_like(self, template: Any) -> Any:
        torch = self._torch
        tensor = self.unwrap(template)
        if tensor is None:  # pragma: no cover - template is always a tensor here
            raise Unsupported("no tensor behind the allocation template")
        return torch.empty_like(tensor)

    def _default_device(self) -> Any:
        torch = self._torch
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            return torch.accelerator.current_device_index()
        return getattr(torch, "default_device", None) or "cpu"

    def cohort(self) -> tuple[Any, ...]:
        try:
            import flag_gems
            import torch
            import triton
        except ImportError:  # pragma: no cover - guarded by __init__
            return ()
        return (
            torch.__version__,
            triton.__version__,
            getattr(flag_gems, "__version__", "?"),
        )


_INSTALLED: dict[tuple[str, str], Any] = {}
_STATS = Stats()
_UNSUPPORTED: set[str] = set()
_CACHE = PlanCache()
_PROVIDER: FlagGemsProvider | None = None


def enable(names: Sequence[str], package: Any = None) -> Stats:
    """Install the fast path over ``flag_gems`` entry points named in ``names``.

    Both bindings are replaced: the package attribute and the ``flag_gems.ops``
    submodule attribute. They start out as the *same* function object, but they are
    two separate module-level bindings, so wrapping one leaves the other pointing at
    the original - and the C++ bridge resolves the package-level name
    (``CallPythonOp_Generic("flag_gems.mul", ...)``). Replacing only ``flag_gems.ops``
    makes the shim look correct while never running on the routed path, which is
    exactly what the hardware run caught.
    """
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = FlagGemsProvider()
    import importlib

    if package is None:
        package = importlib.import_module("flag_gems")
    try:
        ops_module = importlib.import_module(f"{package.__name__}.ops")
    except ModuleNotFoundError:  # a flat package: wrap the one namespace
        ops_module = package
    for name in names:
        if name in _UNSUPPORTED:
            continue
        package_target = getattr(package, name, None)
        ops_target = getattr(ops_module, name, None)
        target = package_target if callable(package_target) else ops_target
        if target is None or not callable(target):
            continue
        wrapper = wrap_callable(
            name,
            target,
            provider=_PROVIDER,
            cache=_CACHE,
            stats=_STATS,
            unsupported=_UNSUPPORTED,
        )
        for module, current in ((package, package_target), (ops_module, ops_target)):
            if not callable(current):
                continue
            key = (module.__name__, name)
            if key in _INSTALLED:
                setattr(module, name, wrapper)
                continue
            _INSTALLED[key] = install(module, name, wrapper)
    return _STATS


def disable() -> None:
    """Restore every replaced callable (used by tests and by failure drills)."""
    import importlib
    import sys

    for (module_name, name), original in list(_INSTALLED.items()):
        module = sys.modules.get(module_name) or importlib.import_module(module_name)
        setattr(module, name, original)
        _INSTALLED.pop((module_name, name), None)
    _CACHE.clear()
    _UNSUPPORTED.clear()


def stats() -> Stats:
    """Counters for the installed fast path (hits, misses, fallbacks, captures)."""
    return _STATS


def plans(cache: PlanCache | None = None) -> list[dict[str, Any]]:
    """Describe every cached plan, for the integration assertions.

    The decisive integrity check is not a timing but the **slot census**: a plan whose
    ``launch_arg_kinds`` has no ``"out"`` never writes its result (that defect
    withdrew a headline number during the campaign), and a plan whose output slot is
    ``"scratch"`` is the same failure wearing a correct-looking error.
    """
    described = []
    for signature, plan in (cache or _CACHE).items():
        described.append(
            {
                "op": plan.op,
                "signature": signature.tensors,
                "launch_count": plan.launch_count,
                "launch_arg_kinds": [
                    [slot.kind for slot in launch.slots] for launch in plan.launches
                ],
                "outputs": len(plan.outs),
                "output_ranks": [spec.rank for spec in plan.outs],
                "output_dtypes": [str(spec.dtype) for spec in plan.outs],
                "scratch_buffers": len(plan.scratch),
                "returns_tuple": plan.returns_tuple,
            }
        )
    return described
