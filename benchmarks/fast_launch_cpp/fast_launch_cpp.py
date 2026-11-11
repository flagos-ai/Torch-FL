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

"""Replay a captured FlagGems kernel from C++ instead of from Python.

`torch_fl/fast_launch.py` replays from Python and wins 1.4-4.4x over the wrapper,
but it cannot reach the vendor operator: keying, allocation and the bridge hop
together cost more than the vendor's whole call. This module keeps the same
FlagGems Triton kernel and removes all three from the hot path.

The split of work is:

* **capture, once, in Python.** Run the operator once at steady state, record the
  compiled kernel (cubin + metadata), the launch geometry and the argument layout,
  and hand them to C++. The layout is read from the compiled kernel's PTX, not from
  the launcher's Python arguments, because pointwise kernels forward constexprs
  (including a dtype string) that have no kernel parameter.
* **replay, per call, in C++.** Patch a prebuilt argument array, launch through
  `cuLaunchKernel`, and hand back a buffer from a pool that recycles a slot when the
  last reference to its result drops. No Python, no allocation, no marshalling.

Nothing here is enabled implicitly: `enable()` is opt-in, it records why it declined
an operator, and every operator it cannot replay keeps calling the untouched
callable. Misclassifying an argument is not a crash but a plausible wrong answer, so
the capture proves its own layout before a plan is installed.
"""

import re

__all__ = [
    "Backend",
    "Plan",
    "Unsupported",
    "enable",
    "disable",
    "plans",
    "stats",
    "parse_ptx_params",
    "classify_arguments",
]

#: A plan replays one launch of one operator. An operator whose steady state issues
#: more launches than this is left alone; see ``Unsupported``.
MAX_LAUNCHES = 1

#: Slots the pool keeps ready before it starts growing on demand.
POOL_SIZE = 8

#: Operators whose captured layout has been validated against the wrapper on hardware.
#: The check that makes a layout trustworthy is that the operator's replay, the pool and
#: the argument mapping all agree -- and a layout that is wrong does not crash, it
#: returns a plausible wrong answer. So the module ships an allowlist and declines
#: everything else by name, including operators whose mapping is still unexplained:
#: ``relu`` captures a three-scalar layout whose replay does not match the wrapper, and
#: ``add``/``sum_dim`` are invoked with keyword arguments, which the plan does not key
#: on yet. Adding a name here requires a gate run that proves exactness.
VERIFIED_OPS = frozenset({"mul", "softmax"})

_AT_DTYPE = {
    "torch.float32": 6,
    "torch.float16": 5,
    "torch.bfloat16": 15,
    "torch.float64": 7,
    "torch.int64": 4,
    "torch.int32": 3,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 0,
}

_PTX_HEADER = r"\.visible \.entry \w+\((?P<params>.*?)\)\s*\n"


class Unsupported(Exception):
    """The operator cannot be replayed; the caller must keep the original."""


def parse_ptx_params(ptx):
    """Return the kernel's real parameter kinds, in order, from the PTX entry.

    This is the authority for the argument layout. Triton appends a global and a
    profile scratch pointer even when their sizes are 0, and the driver reads them, so
    a launch that omits them reads past its argument list: the observed symptom was one
    run that looked correct and the next returning ``CUDA_ERROR_INVALID_VALUE``.
    """
    match = re.search(_PTX_HEADER, ptx, re.S)
    if match is None:
        raise Unsupported("no .entry declaration in the PTX")
    kinds = []
    for line in match.group("params").split(","):
        line = line.strip()
        if not line:
            continue
        if ".ptr" in line:
            kinds.append("ptr")
        elif ".u32" in line or ".s32" in line:
            kinds.append("i32")
        elif ".u64" in line or ".s64" in line:
            kinds.append("i64")
        else:
            raise Unsupported(f"unsupported PTX parameter: {line}")
    if not kinds:
        raise Unsupported("empty PTX parameter list")
    return kinds


def _is_tensor_like(value):
    """Duck-typed tensor test, so a subclass (a fake one in tests) still counts."""
    return hasattr(value, "untyped_storage") and hasattr(value, "is_contiguous")


def _as_tensor(value):
    """The tensor behind a launcher argument, if it has one.

    FlagGems wraps operands of its strided operators in ``StridedBuffer``. For a
    contiguous operand that wrapper is equivalent to the tensor's own pointer, so it is
    accepted; a genuinely strided operand would need its layout baked into the plan and
    is refused rather than launched with the wrong addresses.
    """
    key = type(value).__name__
    if key == "Tensor":
        return value
    if key == "StridedBuffer":
        base = getattr(value, "_base", None)
        if _is_tensor_like(base) and base.is_contiguous():
            return base
        raise Unsupported("strided operand that is not contiguous")
    base = getattr(value, "_base", None)
    return base if _is_tensor_like(base) else None


def tensor_arguments(args):
    """The tensor operands among a call's positional arguments.

    The plugin calls a FlagGems callable with every positional argument the operator
    schema has, including scalars: ``softmax(self, dim, half_to_float)`` arrives as
    three positional arguments, only the first of which is a tensor.
    """
    return [value for value in args if _is_tensor_like(value)]


def _storage_of(tensor):
    return tensor.untyped_storage().data_ptr()


def classify_arguments(launch_args, inputs, ptx_kinds):
    """Map one launch's arguments onto the kernel's parameters.

    ``launch_args`` is the argument tuple the Triton launcher received. Its leading
    nine entries are the launcher's own header (grid, stream, function, scratch tuple,
    metadata, hooks); the kernel arguments follow. Returns the ``kinds`` list the C++
    side expects plus the scalar values, or raises ``Unsupported``.

    The mapping is deliberately strict: a positional mismatch raises rather than
    producing a plan that launches with a shifted argument list.
    """
    pointers = sum(1 for kind in ptx_kinds if kind == "ptr")
    ints_expected = sum(1 for kind in ptx_kinds if kind == "i32")
    if pointers + ints_expected != len(ptx_kinds):
        raise Unsupported("64-bit scalar parameters are not supported yet")

    grid = int(launch_args[0])
    user_args = list(launch_args[9:])

    input_storages = {
        _storage_of(tensor): index
        for index, tensor in enumerate(tensor_arguments(inputs))
    }
    kinds, ints = [], []
    for value in user_args:
        tensor = _as_tensor(value)
        if tensor is not None:
            kinds.append("in" if _storage_of(tensor) in input_storages else "out")
        elif isinstance(value, bool):
            continue  # constexpr: no kernel parameter
        elif isinstance(value, int):
            if len(ints) < ints_expected:
                kinds.append("i32")
                ints.append(int(value))
        else:
            continue  # constexpr (including a dtype string): no kernel parameter

    if len(ints) != ints_expected:
        raise Unsupported(
            f"scalar arguments do not match the PTX: ptx={ints_expected} launch={ints}"
        )
    scratch = pointers - kinds.count("in") - kinds.count("out")
    if scratch < 0:
        raise Unsupported("more pointer arguments than the PTX declares")
    if any(kind == "in" for kind in kinds) is False:
        raise Unsupported("no input tensor found")
    kinds.extend(["zero"] * scratch)
    return grid, kinds, ints


class Plan:
    """One installed plan: what to replay and what it produced."""

    __slots__ = ("op", "grid", "threads", "shared", "kinds", "ints", "shape", "dtype")

    def __init__(self, op, grid, threads, shared, kinds, ints, shape, dtype):
        self.op = op
        self.grid = grid
        self.threads = threads
        self.shared = shared
        self.kinds = kinds
        self.ints = ints
        self.shape = tuple(int(extent) for extent in shape)
        self.dtype = dtype

    def census(self):
        """The slot census, which is the check that the layout is what it claims."""
        return {
            "op": self.op,
            "grid": self.grid,
            "threads": self.threads,
            "shared": self.shared,
            "launches": 1,
            "launch_arg_kinds": [list(self.kinds)],
            "outputs": self.kinds.count("out"),
            "scratch_buffers": self.kinds.count("zero"),
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


class Backend:
    """Captures FlagGems operators and replays them from C++.

    A plan is installed on the second call of an operator: the first call compiles and
    autotunes, the second is the steady state the plan is captured from, and the third
    onwards replay. Capturing the first call would bake the autotuner's warm-up
    launches into the plan.
    """

    def __init__(
        self, extension, provider, max_launches=MAX_LAUNCHES, pool_size=POOL_SIZE
    ):
        self.extension = extension
        self.provider = provider
        self.max_launches = max_launches
        self.pool_size = pool_size
        self.plans = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "fallbacks": 0,
            "captures": 0,
            "warmups": 0,
        }
        self.reasons = {}
        self._originals = {}
        self.provider.install()

    # -- capture ---------------------------------------------------------------

    def _capture(self, op, callable_, args):
        captured = self.provider.capture(callable_, args)
        try:
            plan, compiled = self._build(op, captured, args)
        finally:
            captured.close()
        return plan, compiled, captured.results[0]

    def _build(self, op, captured, args):
        if len(captured.launches) != 1:
            raise Unsupported(
                f"{len(captured.launches)} launches per call, the replay supports "
                f"{self.max_launches}"
            )
        if captured.returns_tuple:
            raise Unsupported("multi-output operators are not supported yet")
        result = captured.results[0]
        if result.numel() == 0:
            raise Unsupported("zero-element result")

        launch_args = captured.launches[0]
        compiled = captured.compiled
        ptx_kinds = parse_ptx_params(compiled.asm["ptx"])
        grid, kinds, ints = classify_arguments(launch_args, args, ptx_kinds)
        if kinds.count("out") != 1:
            raise Unsupported(f"expected one output slot, found {kinds.count('out')}")

        threads = int(compiled.metadata.num_warps) * 32
        shared = int(compiled.metadata.shared or 0)
        plan = Plan(
            op=op,
            grid=grid,
            threads=threads,
            shared=shared,
            kinds=kinds,
            ints=ints,
            shape=tuple(result.shape),
            dtype=str(result.dtype),
        )
        if plan.dtype not in _AT_DTYPE:
            raise Unsupported(f"unsupported result dtype {plan.dtype}")
        return plan, compiled

    # -- replay ----------------------------------------------------------------

    def wrap(self, op, callable_):
        """Return a callable that replays ``op`` once a plan is installed.

        Phase 1 (cold) calls the operator so that Triton compiles and autotunes it.
        Phase 2 (warm) calls it again under the capture hooks; that call is already
        steady state, so its launch is what the plan is built from. Phase 3 onwards
        replays from C++ and returns the pooled result directly - it must NOT call the
        original, or the fast path would only add work. Any failure in phases 2-3
        parks the operator on the original for the rest of the process.
        """
        state = {"phase": "cold"}

        def passthrough(*args, **kwargs):
            state["phase"] = "passthrough"
            return callable_(*args, **kwargs)

        def fast(*args, **kwargs):
            phase = state["phase"]
            if kwargs or phase in ("cold", "passthrough"):
                if kwargs and op not in self.reasons:
                    # The plan bakes the scalars the kernel received, so a call that
                    # carries keywords needs a plan keyed on them. Not implemented yet:
                    # say so once instead of looking like a silent slow path.
                    self.reasons[op] = "operator is called with keyword arguments"
                self.stats[
                    "warmups" if phase == "cold" and not kwargs else "misses"
                ] += 1
                result = callable_(*args, **kwargs)
                if phase == "cold" and not kwargs:
                    state["phase"] = "warm"
                return result
            if phase == "warm":
                try:
                    plan, compiled, result = self._capture(op, callable_, args)
                except Unsupported as error:
                    self.reasons[op] = str(error)
                    self.stats["fallbacks"] += 1
                    return passthrough(*args, **kwargs)
                self._install(op, plan, compiled)
                self.stats["captures"] += 1
                state["phase"] = "ready"
                return result

            plan = self.plans[op]
            inputs = tensor_arguments(args)
            if len(inputs) != plan.kinds.count("in"):
                self.reasons[op] = f"replay got {len(inputs)} inputs"
                self.stats["fallbacks"] += 1
                return passthrough(*args, **kwargs)
            try:
                self.stats["hits"] += 1
                return self.extension.launch(op, inputs)
            except Exception as error:  # noqa: BLE001 - the fallback is the contract
                self.stats["hits"] -= 1
                self.reasons[op] = f"replay failed: {error}"
                self.stats["fallbacks"] += 1
                return passthrough(*args, **kwargs)

        return fast

    def _install(self, op, plan, compiled):
        cubin = compiled.asm["cubin"]
        self.extension.register_kernel(
            op,
            cubin,
            compiled.name,
            plan.grid,
            plan.threads,
            plan.shared,
            plan.kinds,
            plan.ints,
            self.pool_size,
            list(plan.shape),
            _AT_DTYPE[plan.dtype],
        )
        self.plans[op] = plan

    # -- lifecycle -------------------------------------------------------------

    def enable(self, names, package):
        """Install the fast path for ``names`` on ``package``.

        A name outside :data:`VERIFIED_OPS` is declined with a reason rather than
        risked: the wrapper stays in place and the operator keeps its untouched cost.
        """
        for name in names:
            if name not in VERIFIED_OPS:
                self.reasons[name] = "operator is not in the verified set"
                continue
            callable_ = getattr(package, name, None)
            if callable_ is None:
                continue
            wrapped = self.wrap(name, callable_)
            self._originals[name] = (package, name, callable_)
            setattr(package, name, wrapped)
            ops = getattr(package, "ops", None)
            if ops is not None and getattr(ops, name, None) is callable_:
                setattr(ops, name, wrapped)

    def disable(self):
        self.provider.uninstall()
        for package, name, original in self._originals.values():
            setattr(package, name, original)
            ops = getattr(package, "ops", None)
            if ops is not None and getattr(ops, name, None) is not original:
                if getattr(ops, name, None) is not None:
                    setattr(ops, name, original)
        self._originals.clear()
        self.plans.clear()


_state = {"backend": None}


def enable(names, package=None):
    """Turn the C++ replay on for ``names``; returns the backend."""
    if package is None:
        import flag_gems  # noqa: PLC0415 - optional dependency, imported on demand

        package = flag_gems
    extension = _load_extension()
    backend = Backend(extension, _TritonProvider())
    backend.enable(names, package)
    _state["backend"] = backend
    return backend


def disable():
    backend = _state.get("backend")
    if backend is not None:
        backend.disable()
        _state["backend"] = None


def plans():
    backend = _state.get("backend")
    return (
        {name: plan.census() for name, plan in backend.plans.items()} if backend else {}
    )


def stats():
    backend = _state.get("backend")
    return dict(backend.stats) if backend else {}


def _load_extension():
    """Compile and load the CUDA extension that owns the replay."""
    from pathlib import Path  # noqa: PLC0415 - keep the module importable without torch

    from torch.utils.cpp_extension import load_inline  # noqa: PLC0415

    declarations = """
#include <torch/extension.h>
void register_kernel(std::string key, py::bytes cubin_bytes, std::string kernel_name,
                     int64_t grid, int64_t threads, int64_t shared,
                     std::vector<std::string> kinds, std::vector<int64_t> int_args,
                     int64_t pool_size, std::vector<int64_t> out_shape,
                     int64_t out_dtype_code);
at::Tensor launch(std::string key, std::vector<at::Tensor> inputs);
py::dict bench(std::string key, std::vector<at::Tensor> inputs, int64_t iters);
"""
    source = (Path(__file__).resolve().parent / "fast_launch_ext.cu").read_text(
        encoding="utf-8"
    )
    return load_inline(
        name="flagos_fast_launch_ext",
        cpp_sources=declarations,
        cuda_sources=source,
        functions=["register_kernel", "launch", "bench"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


class _TritonProvider:
    """Observes Triton so a FlagGems call can be described.

    Triton compiles a kernel once per configuration, so by the time an operator is at
    steady state the ``CompiledKernel`` was built during the cold call, before this
    provider was watching. The provider therefore observes compilation for its whole
    lifetime and resolves the kernel of a capture by the CUfunction handle the launcher
    receives (``launch_args[4]`` equals ``CompiledKernel.function``), which is exact
    even when several kernels are live.
    """

    #: Only the kernels and launches near a capture are kept. Recording every launch
    #: would retain the tensors it was given, which grows memory and slows the allocator
    #: for the rest of the process - measured as ~58 us per call on operators that kept
    #: using the wrapper.
    KERNEL_HISTORY = 32

    def __init__(self):
        self._kernels = []
        self._launches = []
        self._installed = False
        self._armed = False

    # -- observation -----------------------------------------------------------

    def install(self):
        from triton.backends.nvidia.driver import CudaLauncher  # noqa: PLC0415
        from triton.compiler.compiler import CompiledKernel  # noqa: PLC0415

        if self._installed:
            return
        self._original_init = CompiledKernel.__init__
        self._original_call = CudaLauncher.__call__

        def patched_init(kernel, *init_args, **init_kwargs):
            self._original_init(kernel, *init_args, **init_kwargs)
            self._kernels.append(kernel)
            del self._kernels[: -self.KERNEL_HISTORY]

        def patched_call(launcher, *launch_args, **launch_kwargs):
            if self._armed:
                self._launches.append(launch_args)
            return self._original_call(launcher, *launch_args, **launch_kwargs)

        CompiledKernel.__init__ = patched_init
        CudaLauncher.__call__ = patched_call
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        from triton.backends.nvidia.driver import CudaLauncher  # noqa: PLC0415
        from triton.compiler.compiler import CompiledKernel  # noqa: PLC0415

        CompiledKernel.__init__ = self._original_init
        CudaLauncher.__call__ = self._original_call
        self._installed = False

    def kernel_for(self, launch_args):
        """The compiled kernel behind a launch, matched by its CUfunction handle."""
        handle = launch_args[4]
        for kernel in reversed(self._kernels):
            if getattr(kernel, "function", None) == handle:
                return kernel
        return None

    # -- capture ---------------------------------------------------------------

    class _Captured:
        def __init__(self, launches, compiled, results, returns_tuple):
            self.launches = launches
            self.compiled = compiled
            self.results = results
            self.returns_tuple = returns_tuple

        def close(self):
            pass

    def capture(self, callable_, args):
        if not self._installed:
            raise Unsupported("the Triton observer is not installed")
        self._launches = []
        self._armed = True
        try:
            results = callable_(*args)
        finally:
            self._armed = False
        launches = self._launches
        self._launches = []
        if not launches:
            raise Unsupported("no Triton launch observed; is this a Triton operator?")
        if self._kernels:
            compiled = self.kernel_for(launches[0])
        else:
            compiled = None
        if compiled is None:
            raise Unsupported("could not resolve the compiled kernel behind the launch")
        returns_tuple = isinstance(results, tuple)
        return self._Captured(launches, compiled, [results], returns_tuple)
