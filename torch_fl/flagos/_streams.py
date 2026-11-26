# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Stream access shared by the device module and compiler launchers."""

from contextlib import contextmanager, nullcontext
from functools import cache
from importlib import import_module

import torch

from .. import _C


class _StreamHandle:
    """A view of a shared queue with the handle spellings used by launchers."""

    def __init__(self, index, handle=0):
        self.device_index = index
        self.handle = self.cuda_stream = handle
        self.gcu_stream = self.musa_stream = self.mlu_stream = self.txda_stream = handle
        self.stream_id = handle
        self.device_type = 1

    def __int__(self):
        return self.handle

    @property
    def device(self):
        return torch.device("flagos", self.device_index)

    def synchronize(self):
        pass

    def query(self):
        return True

    def wait_stream(self, other):
        pass

    def wait_event(self, event):
        pass

    def record_event(self, event=None):
        from . import Event

        if event is None:
            event = Event()
        getattr(event, "_event", event).record(self)
        return event


class _SharedStreams:
    """The single-queue stream protocol used by legacy device shims."""

    def __init__(self, getter=None, factory=None):
        self._get_handle = getter
        self._factory = factory

    def create(self, index, priority):
        if self._factory is not None:
            return self._factory(index, priority)
        return self.current(index)

    def current(self, index):
        return _StreamHandle(index, self.raw(index))

    def raw(self, index):
        return self._get_handle(index) if self._get_handle is not None else 0

    def set(self, stream):
        pass

    def set_by_id(self, stream_id, device_index, device_type):
        pass

    def scope(self, stream):
        return nullcontext()


class _CudaStreams:
    def create(self, index, priority):
        from . import _cuda_stream_shim, _has_cuda_runtime

        if hasattr(torch._C, "_cuda_getCurrentStream"):
            return None  # Stream.__new__ already created the CUDA stream.
        if _has_cuda_runtime():
            return _cuda_stream_shim(index)
        return self.current(index)

    def current(self, index):
        from . import _cuda_stream_shim, _has_cuda_runtime

        query = getattr(torch._C, "_cuda_getCurrentStream", None)
        if query is None:
            if _has_cuda_runtime():
                return _cuda_stream_shim(index)
            return _StreamHandle(index, self.raw(index))
        stream_id, device_index, device_type = query(index)
        return torch.cuda.Stream(
            stream_id=stream_id, device_index=device_index, device_type=device_type
        )

    def raw(self, index):
        query = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        return query(index) if query is not None else 0

    def set(self, stream):
        torch.cuda.set_stream(stream)

    def set_by_id(self, stream_id, device_index, device_type):
        torch.cuda._set_stream_by_id(
            stream_id=stream_id, device_index=device_index, device_type=device_type
        )

    @contextmanager
    def scope(self, stream):
        set_stream = getattr(torch._C, "_cuda_setStream", None)
        if set_stream is None:
            yield
            return
        previous = self.current(stream.device.index)
        set_stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )
        try:
            yield
        finally:
            set_stream(
                stream_id=previous.stream_id,
                device_index=previous.device_index,
                device_type=previous.device_type,
            )


class _DeviceStreams:
    """Adapter for a device library's owning stream and borrowed view."""

    def __init__(self, stream_type, current, raw=None):
        self._type = stream_type
        self.current = current
        self._raw = raw

    def create(self, index, priority):
        return self._type(index, priority)

    def raw(self, index):
        if self._raw is not None:
            return self._raw(index)
        return self.current(index).cuda_stream

    def set(self, stream):
        stream.set_current()

    def set_by_id(self, stream_id, device_index, device_type):
        self._type.borrowed(stream_id, device=device_index).set_current()

    @contextmanager
    def scope(self, stream):
        if not isinstance(stream, self._type):
            yield
            return
        previous = self.current(stream.device_index)
        if not isinstance(previous, self._type):
            raise RuntimeError(f"Current stream is unavailable for {stream.device}")
        self.set(stream)
        try:
            yield
        finally:
            self.set(previous)


class _GuardStream(torch.Stream):
    """Borrowed view of a PrivateUse1 stream whose ID encodes its raw handle."""

    def __new__(cls, stream):
        return super().__new__(
            cls, stream.stream_id, stream.device_index, stream.device_type
        )

    @property
    def handle(self):
        return self.stream_id

    cuda_stream = handle
    gcu_stream = handle
    musa_stream = handle
    mlu_stream = handle
    txda_stream = handle

    def record_event(self, event=None):
        from . import Event

        if event is None:
            event = Event()
        event.record(self)
        return event

    def wait_event(self, event):
        return event.wait(self)

    def wait_stream(self, stream):
        coerced = coerce_privateuse1_stream(stream, self.device_index)
        return super().wait_stream(stream if coerced is None else coerced)


_HANDLE_ATTRS = (
    "handle",
    "cuda_stream",
    "gcu_stream",
    "musa_stream",
    "mlu_stream",
    "txda_stream",
    "stream_id",
)


@cache
def _privateuse1_device_type() -> int:
    for code in (20, *range(32)):
        try:
            if torch.Stream(0, 0, code).device.type in ("flagos", "privateuseone"):
                return code
        except Exception:
            continue
    return 20


def coerce_privateuse1_stream(stream, fallback_device_index: int = 0):
    """Return a borrowed PrivateUse1 ``torch.Stream`` for ``stream``, or ``None``.

    PyTorch C++ ``THPVariable_record_stream`` (``Tensor.record_stream``) requires
    a ``torch.Stream`` instance, and ``WrapperRecordStream`` reads the raw vendor
    queue pointer from ``c10::Stream::id()`` on a ``PrivateUse1`` device. Native
    accelerators without owned C++ guard streams (GCU ``TopsStream``, Iluvatar
    ``IluStream``, Kunlunxin ``_StreamHandle``) and CPU-torch ``flagos.Stream``
    wrappers expose raw handles in Python instead of inheriting ``torch.Stream``,
    while ``torch.cuda.Stream`` packs a CUDA pool index into ``stream_id`` rather
    than the raw ``cudaStream_t``. Wrapping the raw handle in a borrowed
    ``PrivateUse1`` ``torch.Stream`` satisfies both contracts while leaving
    non-stream objects untouched so upstream's error still fires.
    """
    inner = getattr(stream, "_stream", stream)
    real = getattr(inner, "_real", None)
    if real is not None:
        inner = real

    if isinstance(inner, torch.Stream):
        if inner.device.type in ("flagos", "privateuseone"):
            return inner
        if isinstance(inner, torch.cuda.Stream):
            dev_idx = (
                fallback_device_index
                if inner.device_index < 0
                else int(inner.device_index)
            )
            return torch.Stream(
                int(inner.cuda_stream),
                dev_idx,
                _privateuse1_device_type(),
            )
        return inner

    if isinstance(inner, (int, float, str, bytes, torch.Tensor)):
        return None
    if not callable(getattr(inner, "synchronize", None)):
        return None

    raw_handle = None
    found_attr = False
    for attr in _HANDLE_ATTRS:
        if not hasattr(inner, attr):
            continue
        val = getattr(inner, attr)
        if val is None:
            raw_handle = 0
            found_attr = True
            break
        if isinstance(val, int) and not isinstance(val, bool):
            raw_handle = int(val)
            found_attr = True
            break
    if not found_attr:
        return None

    dev_idx = getattr(inner, "device_index", None)
    if dev_idx is None:
        dev = getattr(inner, "device", None)
        dev_idx = getattr(dev, "index", None)
    dev_idx = fallback_device_index if dev_idx is None else int(dev_idx)
    return torch.Stream(raw_handle, dev_idx, _privateuse1_device_type())


class _GuardStreams:
    """Stream identity and selection through the registered PyTorch guard."""

    def create(self, index, priority):
        # The flagos facade on these builds has always been a borrowed view.
        # torch.Stream(device="flagos") requests a separate runtime stream.
        return self.current(index)

    def current(self, index):
        return _GuardStream(torch.accelerator.current_stream(index))

    def raw(self, index):
        return _C._get_native_current_raw_stream(index)

    def set(self, stream):
        torch.accelerator.set_stream(stream)

    def set_by_id(self, stream_id, device_index, device_type):
        self.set(torch.Stream(stream_id, device_index, device_type))

    @contextmanager
    def scope(self, stream):
        with stream:
            yield


def create_guard_stream(index, priority):
    return _GuardStream(
        torch.Stream(device=torch.device("flagos", index), priority=priority)
    )


# Load only the device library selected by the wheel.
_DEVICE_STREAMS = {
    "ascend": ("ascend.acl_stream", "AclStream", "current_acl_stream"),
    "gcu": ("enflame.tops_stream", "TopsStream", "current_tops_stream"),
    "iluvatar": ("iluvatar.ilu_stream", "IluStream", "current_ilu_stream"),
    "mps": ("apple.mps_stream", "MpsStream", "current_mps_stream"),
}
_CUDA_STREAMS = _CudaStreams()
_GUARD_STREAMS = _GuardStreams()


@cache
def _device_streams(platform):
    if platform == "musa":
        from ..accelerator.mthreads.musa_stream import MusaStream

        return _SharedStreams(_C._get_musa_current_raw_stream, MusaStream)
    if platform == "cambricon":
        return _SharedStreams(_C._get_cambricon_current_raw_stream)
    if platform not in _DEVICE_STREAMS:
        return _SharedStreams()

    name, stream_type, current = _DEVICE_STREAMS[platform]
    module = import_module(f"torch_fl.accelerator.{name}")
    get_current = getattr(module, current)
    if platform == "ascend":
        # Keep eager callers' existing fallback. Compiled launchers require
        # the actual ACL queue, so their query propagates a registry failure.
        def current_with_fallback(index):
            try:
                return get_current(index)
            except RuntimeError:
                return _StreamHandle(index)

        return _DeviceStreams(
            module.AclStream, current_with_fallback, module.current_acl_raw_stream
        )
    if platform == "iluvatar":
        return _DeviceStreams(
            module.IluStream, get_current, _C._get_iluvatar_current_raw_stream
        )
    return _DeviceStreams(getattr(module, stream_type), get_current)


def _runtime():
    from . import _has_cuda_runtime, _platform

    if hasattr(_C, "_get_native_current_raw_stream"):
        return _GUARD_STREAMS
    if _has_cuda_runtime():
        return _CUDA_STREAMS
    platform = _platform()
    if platform in ("cuda", "dcu", "metax", "ppu"):
        return _CUDA_STREAMS
    return _device_streams(platform)


def create_stream(device, priority):
    from . import _device_index

    return _runtime().create(_device_index(device), priority)


def current_stream(device=None):
    from . import _device_index

    return _runtime().current(_device_index(device))


def cuda_event_stream(stream):
    if stream is None:
        stream = current_stream()
    stream = getattr(stream, "_stream", stream)
    if (
        hasattr(_C, "_get_native_current_raw_stream")
        and isinstance(stream, torch.Stream)
        and stream.device.type == "flagos"
    ):
        # CUDA events need a CUDA stream object. The guard's ID is the raw
        # handle, which ExternalStream borrows with the correct CUDA metadata.
        return torch.cuda.ExternalStream(stream.stream_id, device=stream.device_index)
    return stream


def get_raw_stream(device=None) -> int:
    """Return the current device-library stream for a compiled launch."""
    from . import _device_index

    index = _device_index(device)
    getter = getattr(_C, "_get_current_raw_stream", None)
    if getter is not None:
        return getter(index)
    return _runtime().raw(index)


def set_stream(stream):
    if stream is not None:
        _runtime().set(getattr(stream, "_stream", stream))


def set_stream_by_id(stream_id, device_index, device_type):
    _runtime().set_by_id(stream_id, device_index, device_type)


def stream_scope(stream):
    if stream is None:
        return nullcontext()
    return _runtime().scope(getattr(stream, "_stream", stream))
