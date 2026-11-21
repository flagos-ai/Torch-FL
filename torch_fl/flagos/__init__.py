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

import sys
import time
import types

import torch

from .. import _C  # type: ignore[misc]
from .._platform import build_info, detect_platform
from ..compat import triton as _triton_compat

from . import _streams
from . import meta  # noqa: F401


_initialized = False


def _platform() -> str:
    return detect_platform()


def _has_cuda_runtime() -> bool:
    """Whether torch.cuda is backed by a real CUDA runtime on this build.

    Every vendor branch below (Ascend ACL, Enflame tops, MUSA) exists for
    accelerators that have no CUDA runtime at all. The obvious test for "no CUDA
    runtime" -- ``torch._C._cuda_getCurrentStream`` -- is really a test for "torch
    was compiled with CUDA", and the CPU torch wheel that torch_fl drives against
    an external libtorch_cuda.so fails it while the runtime underneath is
    perfectly real. Reading it as a vendor signal sends NVIDIA runs down the
    Ascend path, where constructing a stream raises "libascendcl.so not found"
    and every autotuned Triton kernel is misreported as failing to compile.
    """
    if hasattr(torch._C, "_cuda_getCurrentStream"):
        return True
    try:
        from torch_fl.accelerator.nvidia._cuda_compat import cuda_shim_active
    except Exception:
        return False
    return cuda_shim_active()


def _cuda_stream_shim(device):
    """A CUDA stream stand-in for a build whose torch has no CUDA stream class.

    Same object ``torch.cuda.current_stream`` returns on such a build, so both
    spellings denote the same (null) stream the boxing kernels submit to.
    """
    from torch_fl.accelerator.nvidia._cuda_compat import _StreamShim, _device_index

    return _StreamShim(_device_index(device))


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        _C._set_device(self.prev_idx)
        return False


def is_available():
    return _C._get_device_count() > 0


def device_count() -> int:
    return _C._get_device_count()


def current_device():
    return _C._get_device()


def set_device(device) -> None:
    return _C._set_device(device)


def synchronize(device=None):
    r"""Waits for all operations on the flagos device to complete.

    Args:
        device (torch.device or int, optional): device to synchronize.
            It uses the current device, given by :func:`~torch_fl.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if device is not None:
        idx = torch.accelerator._get_device_index(device, optional=True)
        prev_idx = _C._exchangeDevice(idx)
        try:
            _C._synchronize()
        finally:
            _C._set_device(prev_idx)
    else:
        _C._synchronize()


def init():
    _lazy_init()


def is_initialized():
    return _initialized


def _lazy_init():
    global _initialized
    if is_initialized():
        from ..compat.flaggems.autotune import refresh_tune_space

        # A lane whose first device touch happened after `import flag_gems`
        # reaches the tune-space repair only here; see the function.
        refresh_tune_space(build_info().vendor)
        return
    _C._init()
    _initialized = True

    from ..compat.cuda import is_active
    from ..compat.flaggems.runtime import initialize

    initialize(build_info().vendor, is_active())

    # Advanced indexing on this device takes the C++ dispatcher and needs no
    # Python patch here. `index.Tensor` is registered on PrivateUse1 from
    # csrc/aten/generated/register.inc, so `x[:, tensor_idx]` and its siblings
    # reach `IndexTensorKernelCuda` directly.
    #
    # Do not reintroduce a `torch.Tensor.__getitem__` assignment. It is a
    # CPython special method, so assigning to it also populates the type's
    # `sq_item` slot and `PySequence_Check()` flips from 0 to 1 for *every*
    # Tensor, on every device. `torch.tensor([tensor(1.), tensor(2.)])` then
    # takes the sequence-protocol branch in
    # `torch/csrc/utils/tensor_new.cpp::compute_sizes`, calls
    # `PySequence_Length`, and raises `TypeError: len() of a 0-d tensor` --
    # including for CPU tensors, in a process that merely touched a non-CPU
    # device. Restoring the attribute afterwards does not undo it; only never
    # assigning it does.


from .random import *  # noqa: F403, E402


# default_generators: list of one Generator per device, required by FlagGems
class _DefaultGenerators:
    """Lazy list-like accessor for per-device default generators.

    Bounds-checked so iteration terminates: with no ``__iter__``, Python's
    legacy protocol calls ``__getitem__(0, 1, 2, ...)`` until IndexError, and
    an out-of-range index otherwise surfaces as a RuntimeError from C++ (which
    aborts iteration instead of ending it).
    """

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, device):
        n = len(self)
        if isinstance(device, slice):
            return tuple(self[i] for i in range(*device.indices(n)))
        device = int(device)
        if device < 0:  # negative indices wrap, as on a list
            device += n
        if not 0 <= device < n:
            raise IndexError(
                f"device index {device} out of range for {n} flagos device(s)"
            )
        return _C._get_default_generator(device)

    def __len__(self):
        return device_count()


default_generators = _DefaultGenerators()


# ---------------------------------------------------------------------------
# Caching Allocator APIs
# ---------------------------------------------------------------------------


def empty_cache():
    """Release all unoccupied cached memory held by the caching allocator.

    This frees GPU memory that is reserved but not currently used by tensors,
    making it available for other GPU applications or new allocations.
    """
    _C._empty_cache()


def memory_stats(device=None):
    """Return a dictionary of memory allocator statistics for the given device.

    Args:
        device: device index (int) or None for current device.

    Returns:
        dict with keys: allocated_bytes, reserved_bytes, peak_allocated_bytes,
        peak_reserved_bytes, num_alloc_calls, num_free_calls,
        num_device_malloc, num_device_free, num_alloc_retries.
    """
    if device is None:
        device = current_device()
    return _C._memory_stats(device)


def memory_allocated(device=None):
    """Return the current GPU memory occupied by tensors in bytes.

    Args:
        device: device index (int) or None for current device.
    """
    if device is None:
        device = current_device()
    return _C._memory_allocated(device)


def memory_reserved(device=None):
    """Return the current GPU memory managed by the caching allocator in bytes.

    This includes both used and cached (free) memory.

    Args:
        device: device index (int) or None for current device.
    """
    if device is None:
        device = current_device()
    return _C._memory_reserved(device)


def mem_get_info(device=None):
    """Return physical free and total device memory in bytes."""
    return _C._mem_get_info(_device_index(device))


def reset_peak_memory_stats(device=None):
    """Reset the peak memory statistics tracked by the allocator.

    Args:
        device: device index (int) or None for current device.
    """
    if device is None:
        device = current_device()
    _C._reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Stream API required by FSDP
# Since flagos shares the same GPU as CUDA, we proxy to torch.cuda streams.
# ---------------------------------------------------------------------------


class Stream(torch.cuda.Stream):
    """Stream facade for the wheel's device runtime."""

    def __new__(cls, device=None, priority=0, **kwargs):
        if device is None:
            device = current_device()
        try:
            return super().__new__(cls, device=device, priority=priority, **kwargs)
        except RuntimeError as e:
            # torch.cuda.Stream unavailable (CPU-only build on Ascend, etc.)
            if "cuda" in str(e).lower():
                return object.__new__(cls)
            raise

    def __init__(self, device=None, priority=0, **kwargs):
        native = _streams.create_stream(device, priority)
        if native is not None:
            self._stream = native

    @property
    def device(self):
        if hasattr(self, "_stream"):
            return self._stream.device
        return super().device

    def __repr__(self):
        if hasattr(self, "_stream"):
            return f"<torch_fl.flagos.Stream device={self.device} native_stream={self._stream.handle:#x}>"
        return super().__repr__()

    def wait_stream(self, other):
        if hasattr(self, "_stream"):
            return self._stream.wait_stream(other)
        return super().wait_stream(other)

    def wait_event(self, event):
        if hasattr(self, "_stream"):
            return self._stream.wait_event(event)
        return super().wait_event(event)

    def record_event(self, event=None):
        if hasattr(self, "_stream"):
            return self._stream.record_event(event)
        return super().record_event(event)

    def synchronize(self):
        if hasattr(self, "_stream"):
            return self._stream.synchronize()
        return super().synchronize()

    def query(self):
        if hasattr(self, "_stream"):
            return self._stream.query()
        return super().query()

    @property
    def cuda_stream(self):
        if hasattr(self, "_stream"):
            return self._stream.handle
        return super().cuda_stream

    @property
    def gcu_stream(self):
        if hasattr(self, "_stream"):
            return self._stream.handle
        return self.cuda_stream

    @property
    def musa_stream(self):
        if hasattr(self, "_stream"):
            return self._stream.handle
        return self.cuda_stream

    @property
    def mlu_stream(self):
        """Raw cnrt queue handle, the name Triton's mlu driver reads.

        ``triton/backends/mlu/driver.py`` does
        ``torch.mlu.current_stream(idx).mlu_stream``. On this platform torch.mlu
        is torch_fl's own shim over this module (see
        torch_fl/accelerator/cambricon/_cambricon_compat.py), so the attribute
        belongs here next to the musa/gcu spellings rather than being arranged
        around in the shim.
        """
        if hasattr(self, "_stream"):
            return self._stream.handle
        return self.cuda_stream

    @property
    def device_index(self):
        if hasattr(self, "_stream"):
            return self._stream.device_index
        return super().device_index

    @property
    def stream_id(self):
        if hasattr(self, "_stream"):
            return self._stream.stream_id
        return super().stream_id

    @property
    def device_type(self):
        if hasattr(self, "_stream"):
            return self._stream.device_type
        return super().device_type


def _real_current_stream(device=None):
    return _streams.current_stream(device)


class _HostTimedEvent:
    """Host-clock event for backends with no CUDA event under the hood.

    On Ascend ``torch.cuda.Event`` is a dummy base class, so instantiating it
    raises "Tried to instantiate dummy base class Event". Triton's autotuner
    calls ``Event(enable_timing=True)`` in ``testing.do_bench`` to time candidate
    configs, so without this every gems kernel that autotunes over more than one
    config dies -- that is what broke ``sum`` with multiple dims, ``all.dim``,
    ``any.dim``, ``amax`` and ``prod.dim_int``.

    ``record`` synchronizes the device and reads the host clock. That measures
    wall time around a drained queue rather than true device time, which is
    exactly what do_bench needs (it only compares candidates against each other),
    and it makes ``wait``/``query`` trivially correct because nothing is
    outstanding once ``record`` returns.
    """

    def __init__(
        self, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        self.enable_timing = enable_timing
        self._t = None

    def record(self, stream=None):
        synchronize()
        self._t = time.perf_counter()

    def wait(self, stream=None):
        return None

    def synchronize(self):
        synchronize()

    def query(self):
        return self._t is not None

    def elapsed_time(self, end_event):
        if self._t is None or end_event._t is None:
            raise RuntimeError("elapsed_time called on an unrecorded event")
        return (end_event._t - self._t) * 1000.0  # ms, matching torch.cuda.Event

    def hardware_time(self, end_event):
        """Device-side duration in nanoseconds, the name Triton's mlu bench uses.

        ``triton/testing.py``'s Cambricon ``do_bench`` estimates a kernel from
        ``start_event.hardware_time(end_event) / 1000 / 5`` and expects
        nanoseconds, which on real torch_mlu comes from
        ``cnrtNotifierElapsedTime``. This event has no device timestamp to read --
        ``record`` drains the queue and reads the host clock -- so the host-clock
        duration is reported in the unit the caller expects. That is a wall-time
        estimate of a drained launch, which is what an autotuner compares
        candidates by; it is not device-only occupancy time.
        """
        return self.elapsed_time(end_event) * 1e6


class Event(torch.cuda.Event):
    """Flagos event: a real CUDA event where there is a CUDA runtime.

    Where there is not, this is the vendor's event (Enflame tops, Ascend ACL) or,
    on MUSA and any accelerator without one, the host-clock
    ``_HostTimedEvent``.
    """

    def __new__(
        cls, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        if not _has_cuda_runtime():
            # One spelling of the host-clock fallback for the arms below, so they
            # differ only in which vendor class they try first.
            host_timed = dict(
                enable_timing=enable_timing,
                blocking=blocking,
                interprocess=interprocess,
                external=external,
            )
            if _platform() == "gcu":
                try:
                    from torch_fl.accelerator.enflame.tops_stream import TopsEvent

                    obj = object.__new__(cls)
                    obj._event = TopsEvent(**host_timed)
                    return obj
                except RuntimeError:
                    return _HostTimedEvent(**host_timed)
            if _platform() == "ascend":
                try:
                    from torch_fl.accelerator.ascend.acl_stream import AclEvent

                    obj = object.__new__(cls)
                    obj._event = AclEvent(
                        enable_timing=enable_timing,
                        blocking=blocking,
                        external=external,
                    )
                    return obj
                except RuntimeError:
                    return _HostTimedEvent(**host_timed)
            if _platform() == "mps":
                # Apple Silicon: MTLSharedEvent is a real device event, so timing
                # is measured on the queue rather than on the host clock.
                from torch_fl.accelerator.apple.mps_stream import MpsEvent

                obj = object.__new__(cls)
                obj._event = MpsEvent(**host_timed)
                return obj
            if _platform() == "iluvatar":
                # Corex events carry a device timestamp, so timing is measured
                # on the queue rather than read from the host clock -- the same
                # reason the MPS arm above takes this path.
                from torch_fl.accelerator.iluvatar.ilu_stream import IluEvent

                obj = object.__new__(cls)
                obj._event = IluEvent(**host_timed)
                return obj
            # MUSA, whose vendor events live behind the shared C ABI: its timing
            # flags are not musa_runtime's, so there is nothing here to wrap and
            # the host clock is the clock that is left. Any other accelerator with
            # no event class takes the same answer, rather than being sent to
            # Ascend for one. See torch_fl/accelerator/mthreads/musa_stream.py.
            return _HostTimedEvent(**host_timed)
        if not hasattr(torch._C, "_cuda_getCurrentStream"):
            # CPU torch wheel over a real CUDA runtime: torch.cuda.Event is the
            # wheel's dummy base class and no C++ CUDA event is reachable, so the
            # host-clock event is the only thing left to time with.
            return _HostTimedEvent(
                enable_timing=enable_timing,
                blocking=blocking,
                interprocess=interprocess,
                external=external,
            )
        return super().__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
            external=external,
        )

    def __init__(
        self, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        del enable_timing, blocking, interprocess, external

    def record(self, stream=None):
        if hasattr(self, "_event"):
            native = getattr(stream, "_stream", stream)
            return self._event.record(native)
        return super().record(_streams.cuda_event_stream(stream))

    def wait(self, stream=None):
        if hasattr(self, "_event"):
            native = getattr(stream, "_stream", stream)
            return self._event.wait(native)
        return super().wait(_streams.cuda_event_stream(stream))

    def synchronize(self):
        if hasattr(self, "_event"):
            return self._event.synchronize()
        return super().synchronize()

    def query(self):
        if hasattr(self, "_event"):
            return self._event.query()
        return super().query()

    def elapsed_time(self, end_event):
        if hasattr(self, "_event"):
            other = getattr(end_event, "_event", end_event)
            return self._event.elapsed_time(other)
        return super().elapsed_time(end_event)


def current_stream(device=None):
    """Return a snapshot of the selected stream for the given device."""
    return _real_current_stream(device)


def stream(s):
    """Select a stream for the region and restore the previous stream on exit."""
    return _streams.stream_scope(s)


def get_amp_supported_dtype():
    """Return list of supported dtypes for AMP (Automatic Mixed Precision).

    Required by torch.autocast for custom device backends.
    """
    return [torch.float16, torch.bfloat16]


class _DeviceProperties:
    """Minimal device properties object for compiler/runtime compatibility."""

    def __init__(self, device_id):
        if _platform() == "tsingmicro":
            # The TX runtime exposes no device-properties query at all -- the
            # only tx* calls reachable from Python are device count/get/set,
            # synchronize, malloc/free and the stream/event set -- so these are
            # constants, matching the ones FlagGems' own tsingmicro descriptor
            # reports (`_tsingmicro.get_device_properties`). Falling through to
            # the Ascend arm below would name the device "Ascend NPU {id}" and
            # report its AICore count as multi_processor_count, which inductor
            # uses for its reduction-split heuristics and for its cache key.
            self.name = "TX81"
            self.multi_processor_count = 16
            self.total_memory = 64 * 1024**3
            self.major = 8
            self.minor = 1
            self.gcnArchName = self.name
            self.warp_size = 16
            self.L2_cache_size = 3 * 1024 * 1024
            self.max_threads_per_multi_processor = 2048
            self.regs_per_multiprocessor = 65536
            self.shared_memory_per_multiprocessor = 0
            return

        get_musa_properties = getattr(_C, "_get_musa_device_properties", None)
        if get_musa_properties is not None:
            props = get_musa_properties(device_id)
            self.name = props["name"]
            self.multi_processor_count = props["multi_processor_count"]
            self.total_memory = props["total_memory"]
            self.major = props["major"]
            self.minor = props["minor"]
            # Inductor's cache fingerprint reads this ROCm-shaped optional
            # attribute even for custom GPU devices. Keep it descriptive rather
            # than pretending that MUSA is a GCN target.
            self.gcnArchName = self.name
            # Compiler-facing fields, all reported by musaGetDeviceProperties.
            # Inductor sizes its Triton launch grids and its benchmarking L2
            # flush buffer from these, so they must be the device's real values
            # (an invented zero L2 size yields an empty buffer that mudnn's
            # Fill rejects). Older builds of the extension did not return them.
            self.warp_size = props.get("warp_size", 128)
            self.L2_cache_size = props.get("l2_cache_size", 0)
            self.max_threads_per_multi_processor = props.get(
                "max_threads_per_multi_processor", 2048
            )
            self.regs_per_multiprocessor = props.get("regs_per_block", 65536)
            self.shared_memory_per_multiprocessor = props.get(
                "shared_memory_per_multiprocessor", 0
            )
            return

        # Iluvatar corex ships a complete cuda_runtime.h, so the same fields come
        # from cudaGetDeviceProperties. It is not the MUSA getter because that one
        # is compiled only into a MUSA build; the dict shape is deliberately the
        # same.
        get_ilu_properties = getattr(_C, "_get_iluvatar_device_properties", None)
        if get_ilu_properties is not None:
            props = get_ilu_properties(device_id)
            self.name = props["name"]
            self.multi_processor_count = props["multi_processor_count"]
            self.total_memory = props["total_memory"]
            self.major = props["major"]
            self.minor = props["minor"]
            # Same reason as the MUSA arm above: Inductor's cache fingerprint
            # reads gcnArchName whenever torch.version.cuda is None, which is
            # every CPU-torch build.
            self.gcnArchName = self.name
            self.warp_size = props.get("warp_size", 32)
            self.L2_cache_size = props.get("l2_cache_size", 0)
            self.max_threads_per_multi_processor = props.get(
                "max_threads_per_multi_processor", 1024
            )
            self.regs_per_multiprocessor = props.get("regs_per_block", 65536)
            self.shared_memory_per_multiprocessor = props.get(
                "shared_memory_per_multiprocessor", 0
            )
            return

        # Kunlunxin: the XPU runtime has no per-device property struct, only the
        # individual xpu_device_get_attr queries the binding wraps. The XPU
        # reports no compute-capability pair at all, and nothing downstream asks
        # it for one -- FlagOSDeviceInterface reads the arch FlagTree's XPU
        # backend compiles for (_kunlunxin_arch) instead -- so major/minor stay
        # the CUDA-shaped sentinels torch.utils._triton's probes expect.
        get_kunlunxin_properties = getattr(_C, "_get_kunlunxin_device_properties", None)
        if get_kunlunxin_properties is not None:
            props = get_kunlunxin_properties(device_id)
            self.name = props["name"]
            self.multi_processor_count = props["multi_processor_count"]
            self.total_memory = props["total_memory"]
            self.major = 8
            self.minor = 0
            # Inductor sizes its benchmarking L2 flush buffer from this field;
            # a zero leaves it empty. The binding reports the on-die L3, the
            # only cache capacity the XPU SDK exposes.
            self.L2_cache_size = props.get("l2_cache_size", 0)
            self.gcnArchName = self.name
            return

        # Ascend analogue: use the AICore count reported by the Ascend Triton
        # backend. The module path is the same on FlagTree and triton-ascend.
        self.name = f"Ascend NPU {device_id}"
        self.multi_processor_count = _aicore_count()
        self.total_memory = mem_get_info(device_id)[1]
        # CUDA-style sentinels required by torch.utils._triton probes.
        self.major = 8
        self.minor = 0

        # Inductor's cache key reads the device name from `gcnArchName` whenever
        # `torch.version.cuda` is None (codecache.py CacheBase.get_system), which
        # is the case for the CPU torch wheel these backends run on -- upstream
        # only reaches that branch on ROCm, where the field exists. Without it
        # every compile_fx dies with AttributeError while hashing the graph.
        # The value is used purely as a string in that hash, so the device name
        # is the honest answer.
        self.gcnArchName = self.name


def _aicore_count(_cache=[]):
    """AICore count for this chip, queried once from the Ascend Triton backend."""
    if not _cache:
        try:
            from triton.backends.ascend.driver import NPUUtils

            _cache.append(int(NPUUtils().get_aicore_num()))
        except Exception:
            _cache.append(32)
    return _cache[0]


def _device_index(device):
    """Resolve a device spec to an index, accepting what `torch.cuda` accepts.

    `get_device_properties` is published as `torch.cuda.get_device_properties`
    (and as `torch.musa`/`torch.npu`), so callers pass every form the CUDA
    function documents: an index, a `torch.device`, a `"cuda:1"`/`"musa"` style
    string, or nothing at all for the current device. FlagGems adds a fifth, its
    own duck-typed device descriptor, and reaches the string form through
    `flag_gems.ops.cumsum`'s module-level `device = device.name`.

    Only the index matters here: the flagos backend has a single device type, so
    the type portion of the spec carries no information beyond "not the CPU".
    """
    if device is None:
        return current_device()
    if isinstance(device, torch.device):
        return current_device() if device.index is None else device.index
    if isinstance(device, str):
        # "flagos", "musa", "cuda", "flagos:1" -- only the suffix is meaningful.
        return int(device.rpartition(":")[2]) if ":" in device else current_device()
    if isinstance(device, int):
        return device
    # FlagGems' DeviceDetector and similar duck-typed specs.
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    raise TypeError(
        f"get_device_properties expects an int, str, torch.device or None, "
        f"but got {type(device).__name__}"
    )


def get_device_properties(device=None):
    """Return device properties for the given device.

    Args:
        device: device index, `torch.device`, `"<type>[:<index>]"` string, or
            None for the current device. Matches `torch.cuda.get_device_properties`.
    """
    return _DeviceProperties(_device_index(device))


def _utils_get_device_index(device, optional=False, allow_cpu=False):
    """`torch.cuda._utils._get_device_index`, for the privateuse1 device module.

    This exists for one caller that is not ours.
    `torch.serialization._validate_device` resolves a `map_location` by asking
    the device module for `_utils._get_device_index`, and its docstring names
    that attribute as the contract a privateuse1 backend has to implement. With
    no `_utils` it takes an `else` branch that calls `torch.device(location)`
    directly, so `map_location="cuda:3"` deserializes onto a **cuda** device
    rather than onto flagos -- which on a CPU-only torch is not a device at all,
    and on a boxing build is the wrong one. Most of the ecosystem checkpoints
    with `map_location="cuda"`, so the CUDA spellings are the ones that matter
    (issue #531).

    `_device_index` already accepts every form the CUDA function documents,
    including the `"cuda:1"` string, and returns only the index -- which is the
    whole answer here, because flagos has a single device type. The two flags
    are the CUDA signature's, honoured rather than ignored: `allow_cpu` lets a
    caller pass `"cpu"` and get -1, and `optional` is what `None` means. A
    non-optional `None` is an error in the CUDA function and stays one here.
    """
    if allow_cpu:
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(device, torch.device) and device.type == "cpu":
            return -1
    if device is None and not optional:
        raise ValueError(
            "_get_device_index: `device` is None and `optional` is False; pass a "
            "device index, a torch.device, a '<type>[:<index>]' string, or set "
            "optional=True to mean the current device"
        )
    return _device_index(device)


#: The submodule `torch.serialization._validate_device` looks for. Built here
#: rather than shipped as a `torch_fl/flagos/_utils.py` file because it is one
#: function and it has to be an attribute of the registered device module --
#: a real submodule would also be importable as `torch_fl.flagos._utils`, which
#: would advertise a second, unowned entry point to the same shim.
_utils = types.ModuleType(f"{__name__}._utils")
_utils.__doc__ = (
    "The privateuse1 device-module hook torch.serialization._validate_device "
    "reads. See _utils_get_device_index."
)
_utils._get_device_index = _utils_get_device_index


__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "initial_seed",  # noqa: F405
    "is_available",
    "init",
    "is_initialized",
    "manual_seed",  # noqa: F405
    "manual_seed_all",  # noqa: F405
    "get_rng_state",  # noqa: F405
    "set_rng_state",  # noqa: F405
    "_is_in_bad_fork",  # noqa: F405  (torch.random._seed_custom_device probes it)
    "get_amp_supported_dtype",
    "Stream",
    "Event",
    "current_stream",
    "stream",
    "default_generators",
    "empty_cache",
    "memory_stats",
    "memory_allocated",
    "memory_reserved",
    "mem_get_info",
    "reset_peak_memory_stats",
    "get_device_properties",
    "_utils",
]

# Kunlunxin's Triton backend inherits GPUDriver's torch.cuda stream accessor,
# which cannot exist on the CPU-torch wheel that backend runs on, so its first
# kernel launch dies in torch.cuda._lazy_init. Bound here, at import, rather
# than in _lazy_init: torch_fl's device init is itself lazy and nothing in the
# eager FlagGems path is guaranteed to have triggered it before the first
# kernel, while this only needs the module objects. See the function for the
# mechanism and the measurement.
# The launcher needs this before it is dlopen'd, and it is dlopen'd on the
# first kernel -- so it has to happen here, not inside a kernel path.
_triton_compat.preload_kunlunxin_runtime(_platform())
_triton_compat.bind_kunlunxin_stream(_platform(), sys.modules[__name__])
