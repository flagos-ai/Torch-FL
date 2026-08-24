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

"""
FlagGems compatibility layer for Enflame GCU.

FlagGems' Triton kernels reach the GCU through Enflame's ``triton_gcu`` plugin
(the vendor Triton backend; ``pip install triton-gcu`` from the FlagOS enflame
index, plus the ``triton-gcu`` deb that installs the ``/opt/triton_gcu``
compiler toolchain). That plugin was written against Enflame's own ``torch_gcu``
plugin, which claims PrivateUse1 and names the device ``gcu``. torch_fl claims
PrivateUse1 first and names it ``flagos``, and a process can rename PrivateUse1
only once -- so every place triton_gcu says "gcu" has to be redirected here.

None of this changes what the kernels compute; it only makes the vendor backend
agree with torch_fl about what the device is called and how to time a kernel.

Requires, on top of the shims below:
  * ``GEMS_VENDOR=enflame``  -- FlagGems' vendor autodetect keys off
    ``hasattr(torch, "gcu")``, which torch_fl does not provide. Set
    automatically once the vendor stack is confirmed present.
  * ``COMPILE_ARCH=gcu300``/``gcu400`` -- selects the arch without asking the
    driver for a stream handle (see GCUDriver.__init__). Set automatically from
    the driver's reported arch when unset.
  * a libstdc++ new enough for FlagGems' sqlite3/sqlalchemy import -- conda's
    ``$CONDA_PREFIX/lib/libstdc++.so.6`` works where a system one may not.

Two further mismatches are corrected here rather than in the vendor stack, both
because a FlagGems op silently calls something other than what it looks like:
``bind_vendor_ops_in_generic_modules`` (a generic op reaching a generic sub-op
past the vendor's override) and ``device_guarded_config`` (a kernel launching on
the current device rather than its operand's).
"""

import functools
import os
import time


_gcu_generators = {}

_UINT64_MASK = (1 << 64) - 1


def _as_int64_bits(value):
    """Reinterpret a uint64 seed as the signed value with the same bit pattern.

    ``torch.initial_seed()`` and CUDA's philox seed are unsigned 64-bit, so a
    default seed routinely exceeds ``int64`` max. ``torch.tensor(..., int64)``
    raises ``ValueError: Overflow when unpacking long long`` on those values,
    which would break every RNG call on the device.
    """
    value &= _UINT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


class _GcuPhiloxGenerator:
    """Minimal generator implementing FlagGems' seed/offset protocol."""

    def __init__(self, seed):
        self._seed = int(seed) & _UINT64_MASK
        self._offset = 0

    def get_state(self):
        import torch

        # 16 bytes of seed+offset, the same layout a CUDA generator exposes.
        return torch.tensor(
            [_as_int64_bits(self._seed), _as_int64_bits(self._offset)],
            dtype=torch.int64,
            device="cpu",
        ).view(torch.uint8)

    def set_state(self, state):
        import torch

        values = state.reshape(-1)
        if values.dtype != torch.int64:
            values = values.view(torch.int64)
        if values.numel() != 2:
            raise ValueError("GCU philox state must contain seed and offset")
        # The state carries the unsigned bit pattern in a signed container.
        self._seed, self._offset = (
            int(values[0].item()) & _UINT64_MASK,
            int(values[1].item()) & _UINT64_MASK,
        )

    def manual_seed(self, seed):
        self._seed = int(seed) & _UINT64_MASK
        self._offset = 0
        return self

    def initial_seed(self):
        return self._seed


def _get_gcu_generator(index):
    import torch

    generator = _gcu_generators.get(index)
    if generator is None:
        generator = _GcuPhiloxGenerator(torch.initial_seed())
        _gcu_generators[index] = generator
    return generator


class _GcuDefaultGenerators:
    """List-like per-device philox generators shared by GCU RNG paths."""

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index):
        n = len(self)
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(n)))
        index = int(index)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"device index {index} out of range for {n} device(s)")
        return _get_gcu_generator(index)

    def __len__(self):
        from torch_fl import flagos

        return max(flagos.device_count(), 1)


def install_gcu_rng_generators():
    """Expose one philox-shaped generator set to FlagGems and native GCU.

    The native PrivateUse1 generators use CPU MT19937 state and are not a
    valid source for FlagGems' ``seed, offset`` unpacking. These lightweight
    philox state objects are intentionally separate and are shared through both
    ``torch.cuda`` (the C++ bridge) and ``torch.flagos`` (FlagGems' runtime).
    """
    import torch

    from torch_fl import flagos

    generators = _GcuDefaultGenerators()
    torch.cuda.default_generators = generators
    flagos.default_generators = generators
    native_manual_seed = flagos.manual_seed
    native_manual_seed_all = flagos.manual_seed_all

    def manual_seed(seed):
        native_manual_seed(seed)
        _get_gcu_generator(flagos.current_device()).manual_seed(seed)

    def manual_seed_all(seed):
        native_manual_seed_all(seed)
        for index in range(len(generators)):
            _get_gcu_generator(index).manual_seed(seed)

    torch.cuda.manual_seed = manual_seed
    torch.cuda.manual_seed_all = manual_seed_all

    # Direct calls through torch.flagos.manual_seed are also part of the
    # public backend API and must reset the shared philox stream.
    flagos.manual_seed = manual_seed
    flagos.manual_seed_all = manual_seed_all


def is_triton_gcu_available() -> bool:
    """True when both halves of the vendor Triton stack are installed.

    The Python plugin alone is not enough: kernel compilation shells out to
    ``$TRITON_GCU_PATH/bin/gcu-compiler-{opt,compile}``, which ship in a
    separate deb. Checking both here keeps the failure at import time (where it
    is actionable) instead of at the first kernel launch.
    """
    import importlib.util

    if importlib.util.find_spec("triton_gcu") is None:
        return False
    if importlib.util.find_spec("triton") is None:
        return False
    datadir = os.environ.get("TRITON_GCU_PATH") or "/opt/triton_gcu"
    return all(
        os.path.exists(os.path.join(datadir, "bin", tool))
        for tool in ("gcu-compiler-opt", "gcu-compiler-compile")
    )


class _WallClockEvent:
    """Wall-clock stand-in for a CUDA-style timing Event.

    ``triton.testing.do_bench`` -- which FlagGems' autotuner calls -- needs
    ``Event(enable_timing=True)`` with ``record``/``elapsed_time``. torch_fl's
    ``flagos.Event`` derives from ``torch.cuda.Event``, which on a CPU-only
    torch wheel is a dummy base class that raises on instantiation.

    Autotuning only compares candidate configs against each other, so relative
    wall-clock timings pick the same winner. Every GCU op path synchronizes
    before returning, so the measured interval does bracket real device work.
    """

    def __init__(self, enable_timing: bool = True):
        self._t = None

    def record(self, *args, **kwargs):
        self._t = time.perf_counter()

    def synchronize(self):
        pass

    def wait(self, *args, **kwargs):
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, end) -> float:
        if self._t is None or end._t is None:
            return 0.0
        return (end._t - self._t) * 1000.0


def _patch_vendor_device_name() -> None:
    """Make FlagGems' enflame descriptor report the device as ``flagos``.

    FlagGems' _enflame backend declares ``device_name="gcu"``, which surfaces as
    ``flag_gems.runtime.device.name``. Its ops use that string two ways -- as a
    torch device for intermediate allocations (``torch.empty(..., device=device)``)
    and as an identity check against their inputs
    (``assert X.device.type == device``, in 20 op files including maximum/minimum
    and the upsample family). Both want the name torch_fl actually registered.

    Patched on ``VendorDescriptor`` before ``flag_gems`` is imported, because
    ``DeviceDetector`` copies the value out of the descriptor at construction and
    is a singleton -- after that first import the name is fixed. ``backend_utils``
    is a top-level module, so importing it does not pull in flag_gems itself.
    """
    try:
        import backend_utils
    except ImportError:
        return

    descriptor = getattr(backend_utils, "VendorDescriptor", None)
    if descriptor is None:
        # flag_gems < 5.3 calls it VendorInfoBase.
        descriptor = getattr(backend_utils, "VendorInfoBase", None)
    if descriptor is None or getattr(descriptor, "_flagos_patched", False):
        return

    original_init = descriptor.__init__

    def _init(self, *args, **kwargs):
        if kwargs.get("device_name") == "gcu":
            kwargs["device_name"] = "flagos"
        original_init(self, *args, **kwargs)

    descriptor.__init__ = _init
    descriptor._flagos_patched = True


def patch_triton_gcu_for_flagos() -> bool:
    """Redirect Enflame's triton_gcu backend onto torch_fl's flagos device.

    Returns False (having changed nothing) when the vendor stack is not
    installed, so callers can fall through to the topsaten/CPU paths.
    """
    if not is_triton_gcu_available():
        return False

    import torch

    from torch_fl import flagos

    # 1. triton_gcu's driver calls torch.gcu.current_device()/current_stream().
    #    torch_fl registers the same surface as torch.flagos.
    if not hasattr(torch, "gcu"):
        torch.gcu = flagos

    # 2. toolkit/backend build torch device strings from a module-level
    #    device_name = "gcu". PrivateUse1 is already named flagos.
    import triton_gcu.triton.backend as _backend
    import triton_gcu.triton.toolkit as _toolkit

    _toolkit.device_name = "flagos"
    _backend.device_name = "flagos"

    # 3. The autotuner's L2-flush buffer is allocated with a hardcoded
    #    device='gcu' (driver.py get_empty_cache_for_benchmark).
    #
    #    The index matters as much as the name. ``do_bench`` allocates this
    #    buffer once and then calls ``clear_cache(cache)`` -- i.e. ``cache
    #    .zero_()`` -- between timing runs, interleaved with the kernel it is
    #    autotuning. Written as device="flagos" the buffer lands on whatever
    #    device was current at allocation, so autotuning an op on flagos:1
    #    while flagos:0 is current has zero_ writing device-0 memory from a
    #    device-1 context. The tops runtime does not reject that: the SIP
    #    faults asynchronously and the process aborts at the next
    #    synchronization ("Receive Sip error message" / "Detected context
    #    error!!!"), several ops after the one that caused it. Naming the
    #    current device pins the buffer to the same device as the kernel.
    import triton_gcu.triton.driver as _driver

    _driver._GCUDriver.get_empty_cache_for_benchmark = lambda self: torch.empty(
        256, dtype=torch.int, device=torch.device("flagos", flagos.current_device())
    )

    # 4. torch_fl's native Tops event supports both Triton timing and stream
    #    ordering, so keep it installed for FSDP and other stream consumers.
    torch.gcu = flagos

    # 5. Setting COMPILE_ARCH (below) takes a GCUDriver constructor branch that
    #    hardcodes `get_current_device = lambda: 0`, so a kernel operating on
    #    flagos:1 would launch against device 0 and read another device's memory
    #    -- in practice it hangs, since a tops pointer only resolves against the
    #    current device (see the tops-pointers-are-device-scoped note).
    #
    #    Both driver classes need fixing: _GCUDriver.__init__ copies the inner
    #    GCUDriver's bound method onto the instance, so patching only the wrapper
    #    class would be undone by the next construction. get_current_stream keeps
    #    the branch's 0 (the default stream), which is what the launcher wants.
    #    get_active_torch_device also builds a torch.device("gcu", ...).
    #    Patching the classes is not enough, and this is the subtle part:
    #    GCUDriver.__init__ assigns `self.get_current_device = lambda idx=0: 0`
    #    as an *instance attribute*, which shadows anything set on the class, and
    #    _GCUDriver.__init__ then copies that attribute onto itself. _GCUDriver
    #    caches only its instance (in __new__), so __init__ -- and with it a fresh
    #    GCUDriver -- re-runs on every `_GCUDriver()` call, undoing a class patch.
    #    So wrap both constructors and re-assign the attribute afterwards.
    #
    #    Symptom when this is wrong: >=3 kernel launches on device 0 followed by
    #    one on another device kills the driver outright ("Receive Sip error
    #    message", then "Receive Abort message from KMD: Sip exception"), taking
    #    the process with it. Fewer than three launches survive, which is what
    #    made this look like a bug in whichever op happened to run first.
    _current_device = staticmethod(lambda: flagos.current_device())
    _driver.GCUDriver.get_current_device = _current_device
    _driver._GCUDriver.get_current_device = _current_device
    _driver._GCUDriver.get_active_torch_device = lambda self: torch.device(
        "flagos", flagos.current_device()
    )

    def _patch_init(cls):
        original = cls.__init__
        if getattr(original, "_flagos_patched", False):
            return

        @functools.wraps(original)
        def __init__(self, *args, **kwargs):
            original(self, *args, **kwargs)
            # Drop the instance attribute so the class-level staticmethod above
            # is what lookups find. get_current_stream keeps the constructor's 0
            # (the default stream), which is what the launcher wants.
            self.__dict__.pop("get_current_device", None)

        __init__._flagos_patched = True
        cls.__init__ = __init__

    _patch_init(_backend.GCUDriver)
    _patch_init(_driver._GCUDriver)

    #    An instance built before this point still holds the copied attribute.
    _existing = getattr(_driver._GCUDriver, "instance", None)
    if _existing is not None:
        _existing.__dict__.pop("get_current_device", None)
        if getattr(_existing, "_driver", None) is not None:
            _existing._driver.__dict__.pop("get_current_device", None)

    # COMPILE_ARCH lets GCUDriver skip the torch.gcu stream lookup during its
    # own construction. Derive it from the arch the driver reports
    # ("dtu-enflame-tops--gcu300" -> "gcu300") rather than hardcoding a chip.
    if "COMPILE_ARCH" not in os.environ:
        import re

        try:
            arch = _driver._GCUDriver().get_arch()
            match = re.search(r"gcu\d+", arch)
            if match:
                os.environ["COMPILE_ARCH"] = match.group(0)
        except Exception:
            # Leave COMPILE_ARCH unset: the driver then resolves the arch from
            # the live device, which works once the shims above are in place.
            pass

    return True


def patch_flaggems_device_name() -> bool:
    """Point FlagGems' own device name at ``flagos``.

    FlagGems takes the device name from its enflame vendor descriptor, so
    ``flag_gems.runtime.device.name`` is ``"gcu"`` while tensors here report
    ``"flagos"``. Ops that compare the two (``maximum``/``minimum`` do
    ``assert X.device.type == device``) then fail on correct input.

    Unlike the triton_gcu shims this runs *after* FlagGems is imported, so it
    has two things to fix. ``DeviceDetector`` is a singleton, so correcting
    ``.name`` on it covers every later reader. But the op modules do
    ``device = device.name`` at module scope, and importing any part of
    ``flag_gems.runtime`` eagerly imports them -- those already hold the old
    literal, so each such module global is rewritten as well.

    Call this after ``import flag_gems`` and before ``flag_gems.enable()``.

    Returns False if FlagGems is not installed or is not on the enflame vendor.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("flag_gems") is None:
        return False
    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        # Older FlagGems releases keep DeviceDetector in backend.device.
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            return False

    detector = DeviceDetector()
    if detector.vendor_name != "enflame":
        return False
    stale = detector.name
    if stale == "flagos":
        return True
    detector.name = "flagos"
    # Not filtered by module name: the vendor's arch-specific overrides are
    # loaded under bare keys like "gcu300.ops.maximum", outside the flag_gems
    # package path. Match on the stale value instead, which is specific enough.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "device", None) == stale:
            module.device = "flagos"
    return True


def _is_int64_arithmetic(args, kwargs) -> bool:
    """True when every tensor operand is int64, i.e. int64 *arithmetic*.

    The GCU has no int64 kernels, which FlagGems' own enflame descriptor states
    (``int64_enabled=False``) and then never enforces -- nothing reads
    ``DeviceDetector.support_int64``. An int64 tensor therefore reaches a Triton
    kernel that cannot legalize the type: a compile error at best ("failed to
    legalize operation 'arith.extsi'"), a wrong result at worst.

    The condition is deliberately "all tensor operands", not "any". Passing int64
    *indices* alongside float data is normal and works -- ``embedding(idx, w)``
    and ``index_select`` are verified correct on device, and diverting them would
    cost real performance. What fails is arithmetic carried out in int64, which is
    exactly the all-int64 case: ``torch.diff`` on a ``torch.arange`` (int64 by
    default) reaches the vendor's ``sub``, correct for fp32/int32 and broken for
    int64 -- so the op cannot simply be excluded either.
    """
    import torch

    seen = False
    for value in list(args) + list(kwargs.values()):
        items = value if isinstance(value, (list, tuple)) else (value,)
        for item in items:
            if isinstance(item, torch.Tensor):
                if item.dtype != torch.int64:
                    return False
                seen = True
    return seen


def device_guarded_config(flag_gems):
    """Return FlagGems' op table with every kernel wrapped in a device guard.

    FlagGems allocates its intermediates with ``device=input.device`` but
    launches Triton on the *current* device. Call an op on a tensor living on
    flagos:1 while flagos:0 is current and the launch reads across devices --
    the tops runtime rejects it ("DeviceId[1] of memory VA ... is not match for
    DeviceId[0] of stream"), surfacing as ``topsErrorInvalidValue`` or a fault
    rather than anything actionable. ``embedding`` is the case that shows up in
    practice, through the ``index_select`` it decomposes to.

    The C++ dispatch path solves this with a ``c10::OptionalDeviceGuard`` per
    caller (python_op_caller.cc), but ops that ``flag_gems.enable()`` registers
    straight onto PrivateUse1 never pass through it. This is the same guard, in
    Python, applied at the one place every such op goes through.

    The device is resolved from the first flagos tensor operand that names an
    index -- matching the C++ DeviceOfArgs -- so leading non-tensor arguments
    (``topk(values, k)``) resolve correctly. Ops with no such operand run
    unguarded on the current device, as before.

    Pass the result as ``flag_gems.enable``'s config. It reads
    ``flag_gems._FULL_CONFIG`` directly, so callers replace that attribute for
    the duration of the call.
    """
    import functools

    import torch

    from torch_fl import flagos

    def _first_device(args, kwargs):
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor):
                if value.device.type in ("flagos", "privateuseone"):
                    return value.device.index
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, torch.Tensor) and item.device.type in (
                        "flagos",
                        "privateuseone",
                    ):
                        return item.device.index
        return None

    def _cpu_compute(aten_op, args, kwargs):
        """Run an op on CPU operands and put flagos results back on the device.

        Used only for the int64 case below. An op registered into
        ``TORCH_LIBRARY_IMPL(aten, PrivateUse1)`` cannot defer to the C++ kernel
        for the same key -- returning NotImplemented re-enters this wrapper -- so
        the fallback is performed here rather than delegated.

        ``aten_op`` is called rather than the FlagGems function: a vendor
        override launches its Triton kernel regardless of where its operands
        live, so handing it CPU tensors still compiles an int64 kernel and still
        fails. Going through the aten op on CPU operands reaches the CPU kernel.
        """
        import torch

        device = None

        def to_cpu(value):
            nonlocal device
            if isinstance(value, torch.Tensor):
                if device is None and value.device.type in ("flagos", "privateuseone"):
                    device = value.device
                return value.cpu()
            if isinstance(value, (list, tuple)):
                return type(value)(to_cpu(item) for item in value)
            return value

        cpu_args = tuple(to_cpu(a) for a in args)
        cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
        out = aten_op(*cpu_args, **cpu_kwargs)
        if device is None:
            return out

        def back(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, (list, tuple)):
                return type(value)(back(item) for item in value)
            return value

        return back(out)

    def _aten_op(aten_name):
        """Resolve "sub.Tensor" to torch.ops.aten.sub.Tensor, or None."""
        import torch

        parts = str(aten_name).split(".")
        op = getattr(torch.ops.aten, parts[0], None)
        if op is None:
            return None
        if len(parts) > 1:
            op = getattr(op, parts[1], None)
        elif hasattr(op, "default"):
            op = op.default
        return op

    def _guard(func, aten_name):
        # In-place and out= ops write through their operand, which a CPU copy
        # would not propagate, so they keep the device path even for int64.
        name = getattr(func, "__name__", "")
        aten_op = None
        if not (name.endswith("_") or name.endswith("_out")):
            aten_op = _aten_op(aten_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if aten_op is not None and _is_int64_arithmetic(args, kwargs):
                return _cpu_compute(aten_op, args, kwargs)
            index = _first_device(args, kwargs)
            if index is None or index == flagos.current_device():
                return func(*args, **kwargs)
            with flagos.device(index):
                return func(*args, **kwargs)

        return wrapper

    # functools.wraps carries __name__ and __module__ over, which the exclusion
    # filter and the vendor-override detection both key on.
    return tuple(
        (entry[0], _guard(entry[1], entry[0])) + tuple(entry[2:])
        for entry in flag_gems._FULL_CONFIG
    )


def bind_vendor_ops_in_generic_modules(flag_gems) -> int:
    """Make generic FlagGems ops call the vendor kernel for their sub-ops.

    A few generic ops are written on top of other ops, imported by value at
    module scope -- ``flag_gems/ops/linear_backward.py`` does
    ``from .mm import mm``, ``baddbmm.py`` does ``from .bmm import bmm``. That
    binding is to the *generic* implementation and is fixed at import time, so
    it survives the vendor's override: aten::mm dispatches to Enflame's
    ``gcu300.ops.mm``, but ``linear_backward`` still calls ``flag_gems.ops.mm``.

    On GCU the two are not interchangeable. The generic ``mm_kernel_general``
    casts its index arithmetic with ``.to(tl.int64)`` (mm.py:101-104), and the
    GCU Triton backend rejects the resulting widening -- "failed to legalize
    operation 'arith.extsi' that was explicitly marked illegal", the same int64
    limitation as the rest of the tops stack. The vendor ``mm_kernel`` exists
    precisely because of that, and stays on int32 indices. So every backward
    pass through ``nn.Linear`` failed to compile while a direct ``torch.mm``
    worked, on the same shapes.

    Excluding ``linear_backward`` instead is not an option: aten has no CPU
    kernel for it, so leaving it unregistered turns the failure into a hard
    NotImplementedError rather than a fallback.

    The vendor overrides are discovered from ``flag_gems._FULL_CONFIG`` (the
    op table FlagGems is about to register) by their module not being under
    ``flag_gems`` -- the arch-specific package is imported under a bare name
    like ``gcu300.ops.mm``. Only names a generic module imported from a sibling
    generic op module are rebound; its own definitions are left alone.

    Call after ``import flag_gems``. Returns the number of names rebound.
    """
    import sys
    import types

    config = getattr(flag_gems, "_FULL_CONFIG", None)
    if not config:
        return 0

    vendor = {}
    for entry in config:
        func = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else None
        if not isinstance(func, types.FunctionType):
            continue
        module = getattr(func, "__module__", "") or ""
        if module == "flag_gems" or module.startswith("flag_gems."):
            continue
        vendor.setdefault(func.__name__, func)
    if not vendor:
        return 0

    rebound = 0
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("flag_gems.ops."):
            continue
        for attr, value in list(vars(module).items()):
            if not isinstance(value, types.FunctionType):
                continue
            # Only names imported from a *sibling* generic op module: an op's
            # own definitions are what the vendor overrides, not what it calls.
            origin = getattr(value, "__module__", "") or ""
            if origin == name or not origin.startswith("flag_gems.ops."):
                continue
            replacement = vendor.get(value.__name__)
            if replacement is None or replacement is value:
                continue
            setattr(module, attr, replacement)
            rebound += 1
    return rebound
