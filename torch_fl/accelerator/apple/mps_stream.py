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

"""Native Metal stream and event wrappers over the public FlagOS C ABI.

``torch.flagos.Stream`` needs a native implementation on Apple Silicon because
the CPU torch wheel has no ``torch.cuda.Stream`` to inherit from, which is what
every other arm of that dispatcher assumes. This module is that implementation,
and unlike ``MusaStream`` it really creates a stream: ``StreamCreateWithPriority``
maps to ``-[MTLDevice newCommandQueue]`` and each handle owns a distinct
``MTLCommandQueue``. Events wrap ``MTLSharedEvent`` handles created by
``EventCreateWithFlags`` and recorded with a monotonically increasing signal
value, so ``EventElapsedTime`` reports device time.

The wrappers go through the C ABI rather than a bound C++ module because the ABI
is what ``libflagos.so`` exports and what the runtime's own conformance suite
exercises; ``tops_stream.py`` takes the same route for the same reason.
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


def _runtime_library_path(torch_fl_module) -> str:
    """Locate the accelerator runtime library (``libflagos``).

    Apple Silicon names shared libraries ``.dylib``; every other platform the
    native build targets -- and the whole ``lib/`` loader elsewhere in the tree
    -- uses ``.so``. The MPS backend only builds on Darwin, so ``.dylib`` is the
    expected hit, but probing both keeps the loader from breaking if a future
    CMake change flips the suffix back.
    """
    lib_dir = os.path.join(os.path.dirname(torch_fl_module.__file__), "lib")
    for suffix in (".dylib", ".so"):
        candidate = os.path.join(lib_dir, f"libflagos{suffix}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(lib_dir, "libflagos.dylib")


def _load_api():
    import torch_fl

    path = _runtime_library_path(torch_fl)
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


class MpsEvent:
    """An owning Metal (``MTLSharedEvent``) event with stream-ordering semantics."""

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
        # MpsEvent is collected at interpreter exit, so read it defensively.
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
            stream = current_mps_stream()
        stream = getattr(stream, "_stream", stream)
        _check(_api.EventRecord(self._handle, _handle(stream)), "EventRecord")
        return self

    def wait(self, stream=None):
        if stream is None:
            stream = current_mps_stream()
        stream = getattr(stream, "_stream", stream)
        stream.wait_event(self)

    def synchronize(self):
        _check(_api.EventSynchronize(self._handle), "EventSynchronize")

    def query(self) -> bool:
        # ``EventQuery`` returns ErrorNotReady while the recorded signal is
        # outstanding and Success once the queue has passed it, so the status is
        # the answer -- nothing to reinterpret the way the Tops runtime needs.
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


class MpsStream:
    """An owning (or borrowed) Metal command queue on the single flagos device."""

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

        # ``priority`` is passed through, but the Metal runtime reports 0/0 from
        # DeviceGetStreamPriorityRange (MTLCommandQueue has no priority knob), so
        # it changes nothing today; it is forwarded rather than dropped so the
        # argument keeps its meaning if Metal grows one.
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
        # FlagGems' Triton launcher reads ``.cuda_stream`` off whatever
        # ``current_stream()`` returns on CUDA-derived backends. There is no
        # Triton on Metal, but a caller that reaches this attribute still wants
        # the queue the work is submitted to, which is what it returns.
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
        if not isinstance(other, MpsStream):
            raise TypeError("MPS streams can only wait on another MPS stream")
        event = MpsEvent()
        event.record(other)
        self.wait_event(event)

    def wait_event(self, event):
        event = getattr(event, "_event", event)
        if not isinstance(event, MpsEvent):
            raise TypeError("MPS stream requires an MPS event")
        _check(
            _api.StreamWaitEvent(self._handle, _handle(event), 0),
            "StreamWaitEvent",
        )

    def record_event(self, event=None):
        if event is None:
            event = MpsEvent()
        event = getattr(event, "_event", event)
        if not isinstance(event, MpsEvent):
            raise TypeError("MPS stream requires an MPS event")
        return event.record(self)

    def set_current(self):
        _check(
            _api.SetCurrentStreamForDevice(self.device_index, self._handle),
            "SetCurrentStreamForDevice",
        )


def current_mps_stream(device=None):
    from torch_fl.flagos import current_device

    index = current_device() if device is None else int(device)
    handle = _api.GetCurrentStreamForDevice(index)
    if not handle:
        raise RuntimeError("FlagOS current MPS stream is unavailable")
    return MpsStream.borrowed(handle, device=index)
