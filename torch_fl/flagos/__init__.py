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

import contextlib
import os
import time

import torch

from .. import _C  # type: ignore[misc]

from . import meta  # noqa: F401


_initialized = False


def _platform() -> str:
    marker = os.path.join(os.path.dirname(__file__), "..", "lib", "flagos_platform")
    try:
        with open(marker) as stream:
            return stream.read().strip()
    except OSError:
        return ""


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
        return
    _C._init()
    _initialized = True

    # Eagerly import FlagGems to avoid deep import chain during dispatch.
    # FlagGems has a deep lazy import chain (fused → FLA → utils → models → sqlalchemy)
    # that can exceed Python's recursion limit when triggered inside PyTorch dispatch.
    #
    # Not just ImportError: FlagGems' vendor autodetect raises RuntimeError("No
    # device were detected on your machine") when it is installed but cannot
    # identify the backend -- which is the normal state on a vendor box whose
    # Triton plugin is absent, and must not take down device init. The FlagGems
    # kernels are optional everywhere; whatever is not registered runs on the
    # native or CPU path.
    try:
        import flag_gems  # noqa: F401
    except Exception:
        pass  # FlagGems unavailable or undetectable here, skip

    # Monkey-patch Tensor.__getitem__ to work around PyTorch C++ dispatch issue
    # with advanced indexing on custom devices. The C++ __getitem__ fails for
    # patterns like x[:, tensor_idx] but torch.ops.aten.index.Tensor works.
    import torch

    _original_getitem = torch.Tensor.__getitem__

    _Tensor = torch.Tensor
    _full_slice = slice(None, None, None)
    _aten_index = torch.ops.aten.index.Tensor

    def _patched_getitem(self, indices):
        # Fast path: the workaround only applies to a tuple of indices that
        # contains at least one Tensor. Anything else (the vast majority of
        # __getitem__ calls, e.g. x[:, -1:]) returns immediately, avoiding the
        # device property access and any tuple scan.
        if type(indices) is not tuple:
            return _original_getitem(self, indices)

        has_tensor = False
        for idx in indices:
            if isinstance(idx, _Tensor):
                has_tensor = True
                break
        if not has_tensor:
            return _original_getitem(self, indices)

        # Only patch for our device
        if self.device.type not in ("privateuseone", "flagos"):
            return _original_getitem(self, indices)

        # Convert to list for aten.index.Tensor
        indices_list = []
        for idx in indices:
            if isinstance(idx, _Tensor):
                indices_list.append(idx)
            elif idx is _full_slice or idx == _full_slice:
                indices_list.append(None)
            else:
                # Non-trivial slice / int / other — fall back to original
                return _original_getitem(self, indices)

        # Use aten.index.Tensor which works correctly
        return _aten_index(self, indices_list)

    torch.Tensor.__getitem__ = _patched_getitem


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
    """Flagos stream that wraps a CUDA stream (same GPU memory)."""

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
        # On Ascend with no CUDA, __new__ returns object.__new__(cls) and we need
        # a full native implementation. Import ACL late to avoid crashing on CUDA.
        if not hasattr(torch._C, "_cuda_getCurrentStream"):
            if _platform() == "gcu":
                from torch_fl.accelerator.gcu.tops_stream import TopsStream

                self._stream = TopsStream(device, priority)
            else:
                from torch_fl.accelerator.ascend.acl_stream import AclStream

                self._stream = AclStream(device, priority)
            # Set device attribute for __repr__ compatibility with torch.cuda.Stream
            self.device = self._stream.device
        else:
            # Real torch.cuda.Stream path: __new__ constructed it, do nothing.
            pass

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


class _DefaultStreamHandle:
    """Minimal stream stand-in for backends with no CUDA runtime.

    Vendor Triton launchers (and FlagGems through them) want an object exposing
    a raw stream handle, under whichever name their vendor uses -- ``cuda_stream``
    for CUDA-derived backends, ``gcu_stream`` for Enflame's triton_gcu. On
    Enflame GCU, Ascend and MUSA the kernels go to the vendor's default stream,
    which every one of them denotes with handle 0. The synchronization those
    backends need already happens in their own op paths, so
    ``synchronize``/``wait_stream`` have nothing to do here.
    """

    __slots__ = ("cuda_stream", "gcu_stream", "musa_stream", "device_index")

    def __init__(self, device_index: int = 0):
        self.cuda_stream = 0
        self.gcu_stream = 0
        get_musa_stream = getattr(_C, "_get_musa_current_raw_stream", None)
        self.musa_stream = get_musa_stream(device_index) if get_musa_stream else 0
        self.device_index = device_index

    def __int__(self) -> int:
        return 0

    def synchronize(self):
        pass

    def wait_stream(self, other):
        pass

    def query(self) -> bool:
        return True

    def wait_event(self, event):
        return None


def _real_current_stream(device=None):
    """Resolve the actual current CUDA stream as a real ``torch.cuda.Stream``.

    Under MetaX boxing, ``torch.cuda.current_stream`` is monkey-patched to a
    lightweight shim (only ``.cuda_stream``, for triton/FlagGems launch), which
    is NOT a real Stream and makes ``torch.cuda.Event.record()`` fail with
    "invalid StreamId". We instead read the true stream tuple straight from the
    C++ runtime (``_cuda_getCurrentStream``) and rebuild a real Stream from it,
    so events record/wait on the same physical (default) stream the boxing
    kernels submit to.

    On a vendor backend with no CUDA runtime at all (Enflame GCU, Ascend, MUSA)
    ``_cuda_getCurrentStream`` does not exist and ``torch.cuda`` cannot even be
    lazily initialized ("Torch not compiled with CUDA enabled"). There is no CUDA
    stream to describe, so return a stand-in carrying the vendor's default stream
    handle: callers such as FlagGems' Triton launcher only read ``.cuda_stream``
    off the result, and those backends submit to their default stream.
    """
    idx = current_device() if device is None else int(device)
    if hasattr(_C, "_get_musa_current_raw_stream"):
        return _DefaultStreamHandle(idx)
    if not hasattr(torch._C, "_cuda_getCurrentStream"):
        if _platform() == "gcu":
            from torch_fl.accelerator.gcu.tops_stream import current_tops_stream

            return current_tops_stream(idx)
        from torch_fl.accelerator.ascend.acl_stream import current_acl_stream

        try:
            return current_acl_stream(idx)
        except RuntimeError:
            return _DefaultStreamHandle(idx)
    stream_id, device_index, device_type = torch._C._cuda_getCurrentStream(idx)
    return torch.cuda.Stream(
        stream_id=stream_id, device_index=device_index, device_type=device_type
    )


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


class Event(torch.cuda.Event):
    """Flagos event backed by CUDA or native ACL runtime state."""

    def __new__(
        cls, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        if not hasattr(torch._C, "_cuda_getCurrentStream"):
            try:
                obj = object.__new__(cls)
                if _platform() == "gcu":
                    from torch_fl.accelerator.gcu.tops_stream import TopsEvent

                    obj._event = TopsEvent(
                        enable_timing=enable_timing,
                        blocking=blocking,
                        interprocess=interprocess,
                        external=external,
                    )
                else:
                    from torch_fl.accelerator.ascend.acl_stream import AclEvent

                    obj._event = AclEvent(
                        enable_timing=enable_timing,
                        blocking=blocking,
                        external=external,
                    )
                return obj
            except RuntimeError:
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
        if stream is None:
            stream = _real_current_stream()
        return super().record(stream)

    def wait(self, stream=None):
        if hasattr(self, "_event"):
            native = getattr(stream, "_stream", stream)
            return self._event.wait(native)
        if stream is None:
            stream = _real_current_stream()
        return super().wait(stream)

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
    """Return the currently selected stream for the given device.

    Returns a real ``torch.cuda.Stream`` (bypassing the boxing shim on
    ``torch.cuda.current_stream``) so it is usable for event record/wait and
    stream ordering, not just triton launch.
    """
    return _real_current_stream(device)


@contextlib.contextmanager
def stream(s):
    """Context-manager that selects a given stream.

    Reimplemented instead of delegating to ``torch.cuda.stream`` because the
    latter saves/restores via ``torch.cuda.current_stream``, which boxing
    replaces with a non-Stream shim (no ``.device``) -> AttributeError. We
    save/restore the real stream through ``_cuda_setStream`` directly.
    """
    if s is None:
        yield
        return

    # On native runtimes with no CUDA, switch the thread-local stream registry.
    if not hasattr(torch._C, "_cuda_setStream"):
        native = getattr(s, "_stream", s)
        if _platform() == "gcu":
            from torch_fl.accelerator.gcu.tops_stream import TopsStream

            native_type = TopsStream
        else:
            from torch_fl.accelerator.ascend.acl_stream import AclStream

            native_type = AclStream
        if isinstance(native, native_type):
            previous = _real_current_stream(native.device_index)
            native.set_current()
            try:
                yield
            finally:
                previous.set_current()
            return
        yield
        return

    # CUDA path: real stream switching via _cuda_setStream
    prev = _real_current_stream(s.device.index)
    torch._C._cuda_setStream(
        stream_id=s.stream_id,
        device_index=s.device_index,
        device_type=s.device_type,
    )
    try:
        yield
    finally:
        torch._C._cuda_setStream(
            stream_id=prev.stream_id,
            device_index=prev.device_index,
            device_type=prev.device_type,
        )


def get_amp_supported_dtype():
    """Return list of supported dtypes for AMP (Automatic Mixed Precision).

    Required by torch.autocast for custom device backends.
    """
    return [torch.float16, torch.bfloat16]


class _DeviceProperties:
    """Minimal device properties object for compiler/runtime compatibility."""

    def __init__(self, device_id):
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

        # Ascend analogue: use the AICore count reported by triton-ascend.
        self.name = f"Ascend NPU {device_id}"
        self.multi_processor_count = _aicore_count()
        self.total_memory = _C._memory_reserved(device_id)
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
    """AICore count for this chip, queried once from triton-ascend."""
    if not _cache:
        try:
            from triton.backends.ascend.driver import NPUUtils

            _cache.append(int(NPUUtils().get_aicore_num()))
        except Exception:
            _cache.append(32)
    return _cache[0]


def get_device_properties(device=None):
    """Return device properties for the given device.

    Args:
        device (int or None): device index, or None for current device.
    """
    if device is None:
        device = current_device()
    return _DeviceProperties(device)


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
    "reset_peak_memory_stats",
    "get_device_properties",
]
