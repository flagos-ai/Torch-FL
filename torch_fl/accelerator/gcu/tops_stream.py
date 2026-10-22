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

"""Native Tops stream and event wrappers backed by the public FlagOS C ABI."""

import ctypes
import os

import torch


_SUCCESS = 0
_NOT_READY = 2
_NOT_SUPPORTED = 5
_EVENT_DISABLE_TIMING = 0
_EVENT_ENABLE_TIMING = 1


def _check(result: int, name: str) -> None:
    if result != _SUCCESS:
        raise RuntimeError(f"{name} failed with code {result}")


def _handle(value) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        return value
    raw = getattr(value, "handle", value)
    if raw is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(int(raw))


def _load_api():
    import torch_fl

    path = os.path.join(os.path.dirname(torch_fl.__file__), "lib", "libflagos.so")
    lib = ctypes.CDLL(path)
    void_p = ctypes.c_void_p
    void_pp = ctypes.POINTER(void_p)

    lib.StreamCreateWithPriority.argtypes = [void_pp, ctypes.c_uint, ctypes.c_int]
    lib.StreamCreateWithPriority.restype = ctypes.c_int
    lib.StreamDestroy.argtypes = [void_p]
    lib.StreamDestroy.restype = ctypes.c_int
    lib.StreamQuery.argtypes = [void_p]
    lib.StreamQuery.restype = ctypes.c_int
    lib.StreamSynchronize.argtypes = [void_p]
    lib.StreamSynchronize.restype = ctypes.c_int
    lib.StreamWaitEvent.argtypes = [void_p, void_p, ctypes.c_uint]
    lib.StreamWaitEvent.restype = ctypes.c_int
    lib.GetCurrentStreamForDevice.argtypes = [ctypes.c_int]
    lib.GetCurrentStreamForDevice.restype = void_p
    lib.SetCurrentStreamForDevice.argtypes = [ctypes.c_int, void_p]
    lib.SetCurrentStreamForDevice.restype = ctypes.c_int

    lib.EventCreateWithFlags.argtypes = [void_pp, ctypes.c_uint]
    lib.EventCreateWithFlags.restype = ctypes.c_int
    lib.EventDestroy.argtypes = [void_p]
    lib.EventDestroy.restype = ctypes.c_int
    lib.EventRecord.argtypes = [void_p, void_p]
    lib.EventRecord.restype = ctypes.c_int
    lib.EventSynchronize.argtypes = [void_p]
    lib.EventSynchronize.restype = ctypes.c_int
    lib.EventQuery.argtypes = [void_p]
    lib.EventQuery.restype = ctypes.c_int
    lib.EventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), void_p, void_p]
    lib.EventElapsedTime.restype = ctypes.c_int
    return lib


_api = _load_api()


class TopsEvent:
    """An owning Tops event with stream-ordering semantics."""

    def __init__(
        self, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        del blocking, interprocess, external
        self.enable_timing = bool(enable_timing)
        self._handle = ctypes.c_void_p()
        flags = _EVENT_ENABLE_TIMING if self.enable_timing else _EVENT_DISABLE_TIMING
        _check(
            _api.EventCreateWithFlags(ctypes.byref(self._handle), flags),
            "EventCreateWithFlags",
        )

    def __del__(self):
        # ``_api`` is a module global and is torn down to None before the last
        # TopsEvent is collected at interpreter exit, so read it defensively.
        api = globals().get("_api")
        handle = getattr(self, "_handle", None)
        if api is not None and handle is not None and handle.value:
            api.EventDestroy(handle)
            self._handle = ctypes.c_void_p()

    @property
    def handle(self):
        return self._handle.value

    def record(self, stream=None):
        if stream is None:
            stream = current_tops_stream()
        stream = getattr(stream, "_stream", stream)
        _check(_api.EventRecord(self._handle, _handle(stream)), "EventRecord")
        return self

    def wait(self, stream=None):
        if stream is None:
            stream = current_tops_stream()
        stream = getattr(stream, "_stream", stream)
        stream.wait_event(self)

    def synchronize(self):
        _check(_api.EventSynchronize(self._handle), "EventSynchronize")

    def query(self) -> bool:
        result = _api.EventQuery(self._handle)
        if result == _NOT_READY:
            return False
        if result == _NOT_SUPPORTED:
            # No topsEventQuery in the installed runtime (801 for every event).
            # The GCU op path drains the device after each op, so nothing is
            # outstanding for this to report -- see TopsStream.query.
            return True
        _check(result, "EventQuery")
        return True

    def elapsed_time(self, end_event) -> float:
        end_event = getattr(end_event, "_event", end_event)
        elapsed = ctypes.c_float()
        _check(
            _api.EventElapsedTime(
                ctypes.byref(elapsed), self._handle, _handle(end_event)
            ),
            "EventElapsedTime",
        )
        return float(elapsed.value)


class TopsStream:
    """An owning or borrowed Tops stream associated with one GCU device."""

    def __init__(self, device=None, priority=0, handle=None, owns_handle=True):
        from torch_fl.flagos import current_device, set_device

        if device is None:
            device = current_device()
        elif isinstance(device, torch.device):
            device = device.index if device.index is not None else current_device()
        else:
            device = int(device)
        self.device_index = device
        self.device = torch.device("flagos", device)
        self._handle = ctypes.c_void_p()
        self._owns_handle = owns_handle

        if handle is not None:
            self._handle = _handle(handle)
            self._owns_handle = False
            return

        previous_device = current_device()
        if previous_device != device:
            set_device(device)
        try:
            _check(
                _api.StreamCreateWithPriority(
                    ctypes.byref(self._handle), 0, int(priority)
                ),
                "StreamCreateWithPriority",
            )
        finally:
            if previous_device != device:
                set_device(previous_device)

    @classmethod
    def borrowed(cls, handle, device=None):
        return cls(device=device, handle=handle, owns_handle=False)

    def __del__(self):
        # ``_api`` is a module global and is torn down to None before the last
        # TopsStream is collected at interpreter exit, so read it defensively.
        api = globals().get("_api")
        handle = getattr(self, "_handle", None)
        if (
            api is not None
            and handle is not None
            and handle.value
            and getattr(self, "_owns_handle", False)
        ):
            api.StreamDestroy(handle)
            self._handle = ctypes.c_void_p()

    @property
    def handle(self):
        return self._handle.value

    @property
    def cuda_stream(self):
        return self._handle.value

    @property
    def gcu_stream(self):
        return self._handle.value

    @property
    def stream_id(self):
        return self._handle.value

    def synchronize(self):
        _check(_api.StreamSynchronize(self._handle), "StreamSynchronize")

    def query(self) -> bool:
        result = _api.StreamQuery(self._handle)
        if result == _NOT_READY:
            return False
        if result == _NOT_SUPPORTED:
            # The installed tops runtime ships no topsStreamQuery: it answers 801
            # (topsErrorNotSupported) for every handle, freshly created or not. It
            # cannot say whether work is outstanding, and the GCU op path drains
            # the device after each op, so the honest answer is "nothing is" --
            # the same policy MusaStream.query documents for the same reason.
            return True
        _check(result, "StreamQuery")
        return True

    def wait_stream(self, other):
        other = getattr(other, "_stream", other)
        if not isinstance(other, TopsStream):
            raise TypeError("Tops streams can only wait on another Tops stream")
        event = TopsEvent()
        event.record(other)
        self.wait_event(event)

    def wait_event(self, event):
        event = getattr(event, "_event", event)
        if not isinstance(event, TopsEvent):
            raise TypeError("Tops stream requires a Tops event")
        _check(
            _api.StreamWaitEvent(self._handle, _handle(event), 0),
            "StreamWaitEvent",
        )

    def record_event(self, event=None):
        if event is None:
            event = TopsEvent()
        event = getattr(event, "_event", event)
        if not isinstance(event, TopsEvent):
            raise TypeError("Tops stream requires a Tops event")
        return event.record(self)

    def set_current(self):
        _check(
            _api.SetCurrentStreamForDevice(self.device_index, self._handle),
            "SetCurrentStreamForDevice",
        )


def current_tops_stream(device=None):
    from torch_fl.flagos import current_device

    index = current_device() if device is None else int(device)
    handle = _api.GetCurrentStreamForDevice(index)
    if not handle:
        raise RuntimeError("FlagOS current Tops stream is unavailable")
    return TopsStream.borrowed(handle, device=index)
