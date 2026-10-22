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

"""Iluvatar corex streams and events over the FlagOS C ABI.

``torch.flagos.Stream`` needs a native implementation on iluvatar because the
platform links the *CPU* torch wheel: there is no ``torch.cuda.Stream`` to inherit
from, and no ``c10::cuda`` anywhere in the build (see
``csrc/runtime/accelerator/iluvatar/stream.cc`` for why that is on purpose).

Unlike MUSA, whose single queue every op runs on, iluvatar really creates
streams. Corex's runtime is CUDA-shaped, so ``StreamCreateWithPriority`` maps to
``cudaStreamCreateWithPriority`` and each handle is a distinct device queue. The
one stream the backend *shares* rather than creates is the per-device current one
in the runtime's registry, which is what the eager kernels submit to
(``IluvatarCurrentStream``) and what ``get_iluvatar_current_raw_stream`` hands
FlagTree -- ``stream(handle)`` below borrows exactly that handle, so FSDP's
collectives and the eager op path are ordered against each other.

Like ``mps_stream.py`` and ``tops_stream.py`` this goes through the C ABI rather
than a bound C++ class, because the ABI is what ``libflagos.so`` exports.
"""

import ctypes
import os

import torch


_SUCCESS = 0
_NOT_READY = 2
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

    lib_dir = os.path.join(os.path.dirname(torch_fl.__file__), "lib")
    lib = ctypes.CDLL(os.path.join(lib_dir, "libflagos.so"))
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
    lib.DeviceGetStreamPriorityRange.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.DeviceGetStreamPriorityRange.restype = ctypes.c_int
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


def _priority_range():
    """``(least, greatest)`` device stream priorities, or ``None``.

    Corex -- like CUDA, whose convention it follows -- gives the *numerically
    smallest* value the highest priority. Queried rather than assumed, because
    the range is a property of the device and not of the ABI.
    """
    least = ctypes.c_int()
    greatest = ctypes.c_int()
    if _api.DeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)):
        return None
    return least.value, greatest.value


class IluEvent:
    """An owning corex event with stream-ordering semantics."""

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
        # ``_api`` is torn down to None before the last event is collected at
        # interpreter exit, so read it defensively.
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
            stream = current_ilu_stream()
        stream = getattr(stream, "_stream", stream)
        _check(_api.EventRecord(self._handle, _handle(stream)), "EventRecord")
        return self

    def wait(self, stream=None):
        if stream is None:
            stream = current_ilu_stream()
        stream = getattr(stream, "_stream", stream)
        stream.wait_event(self)

    def synchronize(self):
        _check(_api.EventSynchronize(self._handle), "EventSynchronize")

    def query(self) -> bool:
        result = _api.EventQuery(self._handle)
        if result == _NOT_READY:
            return False
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


class IluStream:
    """A corex stream: created from priority, or borrowed from the registry."""

    def __init__(self, device=None, priority=0, handle=None, owns_handle=True):
        from torch_fl.flagos import current_device, set_device

        if device is None:
            device = current_device()
        elif isinstance(device, torch.device):
            device = device.index if device.index is not None else current_device()
        elif isinstance(device, str):
            # "flagos:3", "flagos" and a bare "3" all reach here; the index is
            # the part after the colon, as everywhere else in torch.
            device = int(device.split(":")[1]) if ":" in device else current_device()
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

        # ``torch.cuda.Stream``'s ``priority`` is a linear index -- 0 is the
        # default band, lower values are the higher-priority streams FSDP asks
        # for -- while corex takes the device's own range. Map one onto the
        # other; a device that reports no range creates the stream at its default
        # priority, which is what the index 0 means anyway.
        corex_priority = 0
        if priority:
            priority_range = _priority_range()
            if priority_range is not None:
                least, greatest = priority_range
                corex_priority = max(greatest, min(least, int(priority)))

        # The runtime creates on the current device, so a stream for another one
        # has to make that one current first -- and put the old one back.
        previous_device = current_device()
        if previous_device != device:
            set_device(device)
        try:
            _check(
                _api.StreamCreateWithPriority(
                    ctypes.byref(self._handle), 0, corex_priority
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

    def __repr__(self) -> str:
        return (
            f"<IluStream device=flagos:{self.device_index} "
            f"handle={self.handle:#x} owns={self._owns_handle}>"
        )

    @property
    def handle(self):
        return self._handle.value

    @property
    def cuda_stream(self):
        # FlagGems' Triton launcher and FlagTree both read ``.cuda_stream`` off
        # whatever ``current_stream()`` returns on CUDA-derived backends. Corex's
        # stream *is* a cudaStream_t, so this is that handle rather than a name
        # borrowed for convenience.
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
        _check(result, "StreamQuery")
        return True

    def wait_stream(self, other):
        other = getattr(other, "_stream", other)
        if not isinstance(other, IluStream):
            raise TypeError("Iluvatar streams can only wait on another Iluvatar stream")
        event = IluEvent()
        event.record(other)
        self.wait_event(event)

    def wait_event(self, event):
        event = getattr(event, "_event", event)
        if not isinstance(event, IluEvent):
            raise TypeError("Iluvatar streams require an Iluvatar event")
        _check(
            _api.StreamWaitEvent(self._handle, _handle(event), 0),
            "StreamWaitEvent",
        )

    def record_event(self, event=None):
        if event is None:
            event = IluEvent()
        event = getattr(event, "_event", event)
        if not isinstance(event, IluEvent):
            raise TypeError("Iluvatar streams require an Iluvatar event")
        return event.record(self)

    def set_current(self):
        _check(
            _api.SetCurrentStreamForDevice(self.device_index, self._handle),
            "SetCurrentStreamForDevice",
        )


def current_ilu_stream(device=None):
    """The runtime's current stream for ``device``, as a borrowed ``IluStream``.

    This is the handle the eager iluvatar kernels submit to, so it is the stream
    a vendor Triton kernel must launch on -- and the one the C++ ``StreamGuard``
    switches when Python enters ``flagos.stream(...)``.
    """
    from torch_fl.flagos import current_device

    index = current_device() if device is None else int(device)
    handle = _api.GetCurrentStreamForDevice(index)
    if not handle:
        raise RuntimeError(
            f"FlagOS current iluvatar stream for device {index} is unavailable"
        )
    return IluStream.borrowed(handle, device=index)
