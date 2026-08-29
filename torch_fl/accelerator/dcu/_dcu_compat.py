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

"""torch.cuda + RNG compatibility for Hygon DCU (DTK).

DCU uses DTK's hipified libtorch.  Its CUDA kernels therefore consume the
CUDA-shaped generators exposed by ``torch.cuda.default_generators``, while the
PrivateUse1 device module created by torch_fl owns a separate generator set.
Keep the public ``torch.flagos`` seed/state API synchronized with the generators
that the kernels actually consume.

Two entry points:

``install_dcu_rng_bridge()``
    Points ``torch.flagos``'s seed/state API at ``torch.cuda``'s generators.
    Correct on its own when DTK's own torch wheel is in front, because there
    ``torch.cuda`` is fully functional.

``patch_torch_cuda_for_dcu()``
    Needed for a decoupled DCU wheel, where the Python-visible ``torch.cuda`` API
    comes from the official ``torch+cpu`` wheel and reports
    ``is_available() == False``: its ``libtorch_python.so`` was compiled without
    CUDA, so ``torch._C._cuda_*`` does not exist and ``_lazy_init()`` raises
    "Torch not compiled with CUDA enabled".  The *kernels* are fine -- they live
    in DTK's ``libtorch_hip.so`` under the CUDA dispatch key, which the boxing
    path reaches through C++ without going near ``torch.cuda``.  What breaks is
    everything that asks Python whether a CUDA device exists: triton's hcu
    backend (``is_active()``: ``torch.cuda.is_available() and torch.version.hip
    is not None``), hence all of FlagGems, and inductor's device-property probes.
    This shim answers those questions from the flagos runtime and the HIP driver
    instead, mirroring ``metax/_metax_compat.py`` and ``cuda/_cuda_compat.py``.
"""

import ctypes
from dataclasses import dataclass, field

_patched = False
_cuda_patched = False
_hip = None
_props_cache = {}
_cuda_generators = {}

# HIP driver candidates, in DTK's own preference order. libgalaxyhip.so.5 is what
# DTK's libtorch_hip.so links against (measured DT_NEEDED); the amdhip64 name is
# the upstream ROCm spelling and exists on some DTK layouts.
_HIP_LIBS = ("libgalaxyhip.so.5", "libamdhip64.so", "libamdhip64.so.6")

# hipDeviceAttribute_t values from
# /opt/dtk/hip/include/hip/hip_runtime_defines.h. Passing the id rather than
# parsing hipDeviceProp_t keeps this independent of the struct layout, which is
# not stable across DTK releases. Verified against DTK 2604's own
# torch.cuda.get_device_properties(0) on a BW card: every value below matches.
_ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 25
_ATTR_WARP_SIZE = 27
_ATTR_MAX_REGISTERS_PER_BLOCK = 28
_ATTR_MULTIPROCESSOR_COUNT = 32
_ATTR_L2_CACHE_SIZE = 34
_ATTR_MAX_THREADS_PER_MULTIPROCESSOR = 35
_ATTR_COMPUTE_CAPABILITY_MAJOR = 36
_ATTR_COMPUTE_CAPABILITY_MINOR = 37
_ATTR_IS_MULTI_GPU_BOARD = 41
_ATTR_INTEGRATED = 42
_ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 10002  # AmdSpecificBegin + 2


def _device_index(device) -> int:
    import torch

    if isinstance(device, torch.device):
        return 0 if device.index is None else device.index
    if isinstance(device, str):
        return int(device.rsplit(":", 1)[1]) if ":" in device else 0
    if device is None:
        return 0
    return int(device)


def _load_hip():
    """The HIP driver, for device queries torch.cuda cannot answer here."""
    global _hip
    if _hip is not None:
        return _hip
    for name in _HIP_LIBS:
        try:
            _hip = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            return _hip
        except OSError:
            continue
    return None


def _hip_attribute(hip, attr, device_index, default=0):
    value = ctypes.c_int()
    try:
        if hip.hipDeviceGetAttribute(ctypes.byref(value), attr, device_index) != 0:
            return default
    except (AttributeError, OSError):
        return default
    return value.value


def _hip_device_name(hip, device_index):
    buf = ctypes.create_string_buffer(256)
    try:
        if hip.hipDeviceGetName(buf, 256, device_index) != 0:
            return ""
    except (AttributeError, OSError):
        return ""
    return buf.value.decode("utf-8", "replace")


def _hip_gcn_arch_name(hip, device_index):
    """The gfx target string, e.g. "gfx936:sramecc+:xnack-".

    hipDeviceGetAttribute(hipDeviceAttributeGcnArchName) returns hipErrorInvalidValue
    on DTK 2604, so read it out of the hipDeviceProp_t blob instead. We do not
    know the struct layout (it changes between DTK releases and the field is not
    at a fixed offset), but the value is a NUL-terminated "gfx*" string inside an
    over-allocated buffer, so a regex over the raw bytes finds it safely. Used
    only as a cache-key/target string, so a miss is not fatal.
    """
    import re

    buf = ctypes.create_string_buffer(8192)
    try:
        if hip.hipGetDeviceProperties(buf, device_index) != 0:
            return ""
    except (AttributeError, OSError):
        return ""
    match = re.search(rb"gfx[0-9a-zA-Z:+\-]*", bytes(buf))
    return match.group(0).decode("ascii", "replace") if match else ""


def _hip_total_memory(hip, device_index):
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    try:
        # hipMemGetInfo reports the *current* device, so bind it first.
        hip.hipSetDevice(device_index)
        if hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
            return 0
    except (AttributeError, OSError):
        return 0
    return total.value


@dataclass
class _DcuDeviceProperties:
    """torch.cuda._CudaDeviceProperties stand-in, filled from the HIP driver.

    Field set follows metax/_metax_compat.py: inductor and FlagGems read these to
    size Triton launch grids, pick autotune configs and build cache keys, so the
    values have to be the device's real ones rather than CUDA-shaped guesses (a
    zero L2_cache_size, for instance, gives inductor's benchmark path an empty
    flush buffer).
    """

    name: str = ""
    major: int = 9
    minor: int = 3
    total_memory: int = 0
    multi_processor_count: int = 0
    warp_size: int = 64
    L2_cache_size: int = 0
    max_threads_per_multi_processor: int = 2560
    regs_per_multiprocessor: int = 196608
    shared_memory_per_block: int = 65536
    shared_memory_per_block_optin: int = 65536
    shared_memory_per_multiprocessor: int = 65536
    is_integrated: bool = False
    is_multi_gpu_board: bool = False
    # Inductor's cache fingerprint (codecache.py CacheBase.get_system) reads
    # gcnArchName whenever torch.version.cuda is None -- upstream only reaches
    # that branch on ROCm, where the attribute exists. Without it every
    # compile_fx dies with AttributeError while hashing the graph.
    gcnArchName: str = ""
    uuid: str = field(default="")


def _query_device_properties(device_index):
    hip = _load_hip()
    if hip is None:
        # No driver reachable: return CUDA-shaped defaults rather than failing, so
        # a properties probe cannot take down an otherwise working process.
        return _DcuDeviceProperties(name=f"Hygon DCU {device_index}")

    name = _hip_device_name(hip, device_index) or f"Hygon DCU {device_index}"
    arch = _hip_gcn_arch_name(hip, device_index) or name
    props = _DcuDeviceProperties(name=name, gcnArchName=arch)
    props.major = _hip_attribute(
        hip, _ATTR_COMPUTE_CAPABILITY_MAJOR, device_index, props.major
    )
    props.minor = _hip_attribute(
        hip, _ATTR_COMPUTE_CAPABILITY_MINOR, device_index, props.minor
    )
    props.total_memory = _hip_total_memory(hip, device_index)
    props.multi_processor_count = _hip_attribute(
        hip, _ATTR_MULTIPROCESSOR_COUNT, device_index, props.multi_processor_count
    )
    props.warp_size = _hip_attribute(
        hip, _ATTR_WARP_SIZE, device_index, props.warp_size
    )
    props.L2_cache_size = _hip_attribute(
        hip, _ATTR_L2_CACHE_SIZE, device_index, props.L2_cache_size
    )
    props.max_threads_per_multi_processor = _hip_attribute(
        hip,
        _ATTR_MAX_THREADS_PER_MULTIPROCESSOR,
        device_index,
        props.max_threads_per_multi_processor,
    )
    props.regs_per_multiprocessor = _hip_attribute(
        hip,
        _ATTR_MAX_REGISTERS_PER_BLOCK,
        device_index,
        props.regs_per_multiprocessor,
    )
    shared = _hip_attribute(
        hip,
        _ATTR_MAX_SHARED_MEMORY_PER_BLOCK,
        device_index,
        props.shared_memory_per_block,
    )
    props.shared_memory_per_block = shared
    props.shared_memory_per_block_optin = shared
    props.shared_memory_per_multiprocessor = _hip_attribute(
        hip,
        _ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        device_index,
        shared,
    )
    props.is_multi_gpu_board = bool(
        _hip_attribute(hip, _ATTR_IS_MULTI_GPU_BOARD, device_index, 0)
    )
    props.is_integrated = bool(_hip_attribute(hip, _ATTR_INTEGRATED, device_index, 0))
    return props


def _get_cuda_generator(idx):
    """One CUDA generator per device -- the FlagGems RNG source.

    FlagGems' hygon backend uses ``device_name="cuda"``, so its
    ``philox_backend_seed_offset`` reads ``torch.cuda.default_generators[dev]``
    and unpacks the state as 2x int64 ``(seed, offset)`` -- the CUDA generator's
    philox layout. The flagos PrivateUse1 generator is a CPUGeneratorImpl whose
    state is the ~5KB mt19937 blob, which would blow that unpack up with "too
    many values to unpack". So expose CUDA generators here, not the flagos ones.
    Constructing one requires libcaffe2_nvrtc.so to be preloaded; see
    _dcu_libtorch_link._DEVICE_LOAD_ORDER.
    """
    import torch

    gen = _cuda_generators.get(idx)
    if gen is None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(torch.initial_seed())
        _cuda_generators[idx] = gen
    return gen


class _CudaDefaultGenerators:
    """list-like stand-in for ``torch.cuda.default_generators``.

    Upstream declares it as a *tuple*, so callers may iterate, slice or wrap it in
    ``list()``. Bounds-checking ``__getitem__`` is what makes that safe: with no
    ``__iter__``, Python falls back to calling ``__getitem__(0, 1, 2, ...)`` until
    IndexError, so an unchecked index turns ``for g in default_generators`` into
    an infinite loop allocating a generator per step.
    """

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, idx):
        n = len(self)
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(n)))
        idx = int(idx)
        if idx < 0:  # negative indices wrap, as on the tuple this replaces
            idx += n
        if not 0 <= idx < n:
            raise IndexError(
                f"device index {idx} out of range for {n} flagos device(s)"
            )
        return _get_cuda_generator(idx)

    def __len__(self):
        import torch

        try:
            return max(torch.flagos.device_count(), 1)
        except Exception:
            return 1


class _DcuStreamShim:
    """Minimal stream object exposing ``.cuda_stream`` for triton-hcu.

    Handle 0 is the default/null stream, consistent with the boxing path, where
    the caching allocator is handed ``stream=nullptr`` and the DTK kernels run on
    the default stream. Triton's GPUDriver reads either
    ``torch._C._cuda_getCurrentRawStream(idx)`` or
    ``torch.cuda.current_stream(idx).cuda_stream``; both are patched to agree
    with this.
    """

    def __init__(self, index=0):
        self.cuda_stream = 0
        self.device_index = index

    def synchronize(self):
        import torch

        torch.flagos.synchronize()

    def wait_stream(self, other):
        return None

    def query(self):
        return True


def is_dcu_available() -> bool:
    """True when a DCU device is reachable through the flagos runtime."""
    import torch

    try:
        return torch.flagos.device_count() > 0
    except Exception:
        return False


def patch_torch_cuda_for_dcu() -> bool:
    """Make the official wheel's ``torch.cuda`` usable on DCU.

    Idempotent. Returns False when no DCU device is reachable, or when a real DTK
    torch is in front (``torch.cuda.is_available()`` already True), in which case
    the native implementation is strictly better and must not be shadowed.

    Must be called before FlagGems / triton import anything that captures
    ``torch.cuda`` attributes, which is why ``_patch_flaggems_codegen_config()``
    in ``torch_fl/__init__.py`` calls it ahead of ``install_dcu_rng_bridge()``.
    """
    global _cuda_patched
    if _cuda_patched:
        return True

    import torch

    try:
        if torch.cuda.is_available():
            return False  # a real DTK torch: leave its native torch.cuda alone.
    except Exception:
        pass

    if not is_dcu_available():
        return False

    def _device_index(device=None):
        if device is None:
            return _current_device()
        if isinstance(device, torch.device):
            return 0 if device.index is None else device.index
        if isinstance(device, str):
            return int(device.rsplit(":", 1)[1]) if ":" in device else 0
        return int(device)

    def _current_device():
        try:
            return torch.flagos.current_device()
        except Exception:
            return 0

    def _get_device_properties(device=None):
        idx = _device_index(device)
        if idx not in _props_cache:
            _props_cache[idx] = _query_device_properties(idx)
        return _props_cache[idx]

    torch.cuda.get_device_properties = _get_device_properties
    torch.cuda.get_device_name = lambda device=None: _get_device_properties(device).name
    torch.cuda.get_device_capability = lambda device=None: (
        _get_device_properties(device).major,
        _get_device_properties(device).minor,
    )

    # Availability. The +cpu wheel's torch.cuda.is_available() is False and
    # _lazy_init() raises AssertionError("Torch not compiled with CUDA enabled"),
    # which is what triton's hcu is_active() and torch.cuda.init() trip over. The
    # compute itself never needs these -- it goes through the C++ boxing path into
    # DTK's CUDA-key kernels.
    torch.cuda.is_available = lambda: True

    # The CPU wheel leaves torch.cuda.Event as a dummy base class. Triton's
    # autotuner constructs Event(enable_timing=True) while selecting a kernel,
    # so merely making is_available() true is not enough: construction otherwise
    # raises "Tried to instantiate dummy base class Event". flagos.Event provides
    # the host-timed fallback when no native CUDA event is available (and uses the
    # native event in legacy DTK mode), so expose it through the CUDA namespace.
    flagos_event = getattr(torch, "flagos", None)
    flagos_event = getattr(flagos_event, "Event", None)
    if flagos_event is not None:
        torch.cuda.Event = flagos_event
    torch.cuda._lazy_init = lambda: None
    torch.cuda.device_count = lambda: max(torch.flagos.device_count(), 1)
    if hasattr(torch.cuda, "_initialized"):
        torch.cuda._initialized = True
    if hasattr(torch.cuda, "_queued_calls"):
        torch.cuda._queued_calls.clear()

    # Route set_device/current_device to the flagos runtime. Stock
    # torch.cuda.set_device calls torch._C._cuda_setDevice, which does not exist
    # on the CPU wheel; without this, torch.cuda.set_device(rank) either raises or
    # silently leaves the DTK runtime on device 0 while the tensors live on device
    # r -- the failure mode measured on MetaX as FlagCX "invalid resource handle".
    _orig_cuda_set_device = getattr(torch._C, "_cuda_setDevice", None)

    def _set_device(device):
        idx = _device_index(device)
        if _orig_cuda_set_device is not None:
            try:
                _orig_cuda_set_device(idx)  # keep torch's own counter aligned
            except Exception:
                pass
        try:
            torch.flagos.set_device(idx)
        except Exception:
            pass

    torch.cuda.set_device = _set_device
    torch.cuda.current_device = _current_device
    torch.cuda.synchronize = lambda device=None: torch.flagos.synchronize(device)

    def _exchange_device(idx):
        if idx < 0:
            return -1
        prev = _current_device()
        _set_device(idx)
        return prev

    torch.cuda._exchange_device = _exchange_device
    torch.cuda._maybe_exchange_device = _exchange_device

    torch.cuda.current_stream = lambda device=None: _DcuStreamShim(
        _device_index(device)
    )
    torch.cuda.default_stream = lambda device=None: _DcuStreamShim(
        _device_index(device)
    )
    # triton's GPUDriver prefers torch._C._cuda_getCurrentRawStream when it
    # imports; keep the two answers identical (the DTK default stream, handle 0).
    try:
        torch._C._cuda_getCurrentRawStream = lambda idx=0: 0
    except Exception:
        pass

    # Seeding. With is_available()=True, torch.manual_seed() now calls
    # torch.cuda.manual_seed_all(), which walks torch.cuda.default_generators --
    # an empty tuple on the CPU wheel, so IndexError. Install real per-device CUDA
    # generators (16-byte philox state, exactly what FlagGems unpacks) and route
    # CUDA seeding to them.
    def _manual_seed_all(seed):
        seed = int(seed)
        for i in range(max(torch.flagos.device_count(), 1)):
            try:
                _get_cuda_generator(i).manual_seed(seed)
            except Exception:
                pass

    def _manual_seed(seed):
        try:
            _get_cuda_generator(_current_device()).manual_seed(int(seed))
        except Exception:
            pass

    torch.cuda.manual_seed_all = _manual_seed_all
    torch.cuda.manual_seed = _manual_seed
    try:
        torch.cuda.default_generators = _CudaDefaultGenerators()
    except Exception:
        pass

    _cuda_patched = True
    return True


def install_dcu_rng_bridge() -> bool:
    """Make ``torch.flagos`` expose DTK's real CUDA RNG streams.

    The PrivateUse1 generator remains available for explicit ``flagos``
    generators, but generator-less DCU kernels use the CUDA generator injected
    by the boxing/codegen path.  Delegating state and seeding here makes both
    paths observe one public stream without replacing DTK's CUDA generator
    collection or its native ``torch.cuda`` methods.
    """
    global _patched
    if _patched:
        return True

    import torch

    from torch_fl import flagos

    generators = getattr(torch.cuda, "default_generators", None)
    if generators is None:
        return False
    try:
        if len(generators) == 0:
            # Native path (a DTK torch in front): torch.cuda.init() populates the
            # collection. Decoupled path: patch_torch_cuda_for_dcu() already
            # replaced it with _CudaDefaultGenerators, whose len() is the device
            # count, so this branch is not reached -- and torch.cuda.init() would
            # be a no-op there anyway.
            torch.cuda.init()
            generators = torch.cuda.default_generators
    except (AssertionError, AttributeError, RuntimeError, TypeError):
        return False
    if len(generators) == 0:
        return False

    native_manual_seed = flagos.manual_seed
    native_manual_seed_all = flagos.manual_seed_all

    def manual_seed(seed):
        seed = int(seed)
        native_manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def manual_seed_all(seed):
        seed = int(seed)
        native_manual_seed_all(seed)
        torch.cuda.manual_seed_all(seed)

    def get_rng_state(device="flagos"):
        return generators[_device_index(device)].get_state()

    def set_rng_state(state, device="flagos"):
        generators[_device_index(device)].set_state(state)

    # Keep the native PrivateUse1 seed bookkeeping in sync while exposing the
    # state object that the actual DCU CUDA kernels consume.
    flagos.manual_seed = manual_seed
    flagos.manual_seed_all = manual_seed_all
    flagos.get_rng_state = get_rng_state
    flagos.set_rng_state = set_rng_state
    flagos.default_generators = generators

    _patched = True
    return True
