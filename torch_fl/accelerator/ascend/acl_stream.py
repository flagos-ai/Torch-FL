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

"""Native ACL stream and event wrappers for Ascend on CPU-only torch builds."""

import ctypes
import os

import torch


try:
    _acl = ctypes.CDLL("libascendcl.so")
except OSError:
    _acl = None


def _configure_acl() -> None:
    if _acl is None:
        return
    void_p = ctypes.c_void_p
    int_p = ctypes.POINTER(ctypes.c_int)
    _acl.aclrtCreateStream.argtypes = [ctypes.POINTER(void_p)]
    _acl.aclrtCreateStream.restype = ctypes.c_int
    _acl.aclrtDestroyStream.argtypes = [void_p]
    _acl.aclrtDestroyStream.restype = ctypes.c_int
    _acl.aclrtSynchronizeStream.argtypes = [void_p]
    _acl.aclrtSynchronizeStream.restype = ctypes.c_int
    _acl.aclrtStreamQuery.argtypes = [void_p, int_p]
    _acl.aclrtStreamQuery.restype = ctypes.c_int
    _acl.aclrtCreateEventWithFlag.argtypes = [ctypes.POINTER(void_p), ctypes.c_uint]
    _acl.aclrtCreateEventWithFlag.restype = ctypes.c_int
    _acl.aclrtDestroyEvent.argtypes = [void_p]
    _acl.aclrtDestroyEvent.restype = ctypes.c_int
    _acl.aclrtRecordEvent.argtypes = [void_p, void_p]
    _acl.aclrtRecordEvent.restype = ctypes.c_int
    _acl.aclrtStreamWaitEvent.argtypes = [void_p, void_p]
    _acl.aclrtStreamWaitEvent.restype = ctypes.c_int
    _acl.aclrtSynchronizeEvent.argtypes = [void_p]
    _acl.aclrtSynchronizeEvent.restype = ctypes.c_int
    query_event = getattr(_acl, "aclrtQueryEventStatus", None)
    if query_event is None:
        query_event = _acl.aclrtQueryEventWaitStatus
    query_event.argtypes = [void_p, ctypes.POINTER(ctypes.c_int)]
    query_event.restype = ctypes.c_int
    _acl._flagos_query_event = query_event
    _acl.aclrtEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        void_p,
        void_p,
    ]
    _acl.aclrtEventElapsedTime.restype = ctypes.c_int


_configure_acl()

_ACL_EVENT_SYNC = 1
_ACL_STREAM_STATUS_COMPLETE = 0
_ACL_EVENT_WAIT_STATUS_COMPLETE = 0


def _handle(value) -> ctypes.c_void_p:
    if isinstance(value, ctypes.c_void_p):
        return value
    raw = getattr(value, "handle", value)
    if raw is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(int(raw))


def _check(ret: int, name: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{name} failed with code {ret}")


_stream_api_cache = []


def _stream_api():
    """Load torch_fl's stream registry API when it is available.

    Cached: current_stream() sits on the per-op and per-module FSDP2/DDP path,
    so the dlopen and argtypes setup must not be repeated per call.
    """
    if _stream_api_cache:
        return _stream_api_cache[0]
    try:
        import torch_fl

        path = os.path.join(os.path.dirname(torch_fl.__file__), "lib", "libflagos.so")
        lib = ctypes.CDLL(path)
    except (ImportError, OSError):
        _stream_api_cache.append(None)
        return None
    lib.GetCurrentStream.argtypes = [ctypes.c_int]
    lib.GetCurrentStream.restype = ctypes.c_void_p
    lib.SetCurrentStream.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.SetCurrentStream.restype = None
    _stream_api_cache.append(lib)
    return lib


class AclEvent:
    """An owning ACL event with stream-ordering semantics."""

    def __init__(self, enable_timing=False, blocking=False, external=False):
        del blocking, external
        if _acl is None:
            raise RuntimeError("libascendcl.so not found. ACL runtime is required")
        self.enable_timing = bool(enable_timing)
        self._handle = ctypes.c_void_p()
        flags = 0 if self.enable_timing else _ACL_EVENT_SYNC
        _check(
            _acl.aclrtCreateEventWithFlag(ctypes.byref(self._handle), flags),
            "aclrtCreateEventWithFlag",
        )

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.value and _acl is not None:
            _acl.aclrtDestroyEvent(handle)
            self._handle = ctypes.c_void_p()

    @property
    def handle(self):
        return self._handle.value

    def record(self, stream=None):
        if stream is None:
            from torch_fl.flagos import current_stream

            stream = current_stream()
        _check(
            _acl.aclrtRecordEvent(self._handle, _handle(stream)),
            "aclrtRecordEvent",
        )
        return self

    def wait(self, stream=None):
        if stream is None:
            from torch_fl.flagos import current_stream

            stream = current_stream()
        _check(
            _acl.aclrtStreamWaitEvent(_handle(stream), self._handle),
            "aclrtStreamWaitEvent",
        )

    def synchronize(self):
        _check(_acl.aclrtSynchronizeEvent(self._handle), "aclrtSynchronizeEvent")

    def query(self) -> bool:
        status = ctypes.c_int()
        _check(
            _acl._flagos_query_event(self._handle, ctypes.byref(status)),
            "aclrtQueryEventWaitStatus",
        )
        return status.value == _ACL_EVENT_WAIT_STATUS_COMPLETE

    def elapsed_time(self, end_event) -> float:
        elapsed = ctypes.c_float()
        _check(
            _acl.aclrtEventElapsedTime(
                ctypes.byref(elapsed), self._handle, _handle(end_event)
            ),
            "aclrtEventElapsedTime",
        )
        return float(elapsed.value)


class AclStream:
    """An ACL stream, optionally borrowing an existing runtime handle."""

    def __init__(self, device=None, priority=0, handle=None, owns_handle=True):
        del priority
        from torch_fl.flagos import current_device, set_device

        if device is None:
            device = current_device()
        elif isinstance(device, torch.device):
            device = device.index if device.index is not None else 0
        else:
            device = int(device)
        self.device_index = device
        self.device = torch.device(f"flagos:{device}")
        self._handle = None
        self._owns_handle = owns_handle
        if _acl is None:
            raise RuntimeError("libascendcl.so not found. ACL runtime is required")

        if handle is not None:
            self._handle = int(
                handle.value if isinstance(handle, ctypes.c_void_p) else handle
            )
            self._owns_handle = False
            return

        prev_device = current_device()
        if prev_device != device:
            set_device(device)
        try:
            handle_obj = ctypes.c_void_p()
            _check(
                _acl.aclrtCreateStream(ctypes.byref(handle_obj)), "aclrtCreateStream"
            )
            self._handle = handle_obj.value
        finally:
            if prev_device != device:
                set_device(prev_device)

    @classmethod
    def borrowed(cls, handle, device=None):
        return cls(device=device, handle=handle, owns_handle=False)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle and getattr(self, "_owns_handle", False) and _acl is not None:
            _acl.aclrtDestroyStream(_handle(handle))
            self._handle = None

    @property
    def handle(self):
        return self._handle

    @property
    def cuda_stream(self):
        return self._handle

    @property
    def stream_id(self):
        return self._handle

    def synchronize(self):
        if self._handle:
            _check(
                _acl.aclrtSynchronizeStream(_handle(self)),
                "aclrtSynchronizeStream",
            )

    def query(self) -> bool:
        if not self._handle:
            return True
        status = ctypes.c_int()
        _check(
            _acl.aclrtStreamQuery(_handle(self), ctypes.byref(status)),
            "aclrtStreamQuery",
        )
        return status.value == _ACL_STREAM_STATUS_COMPLETE

    def wait_stream(self, other):
        if hasattr(other, "_stream"):
            other = other._stream
        if not isinstance(other, AclStream):
            raise TypeError("ACL streams can only wait on another ACL stream")
        event = AclEvent()
        try:
            event.record(other)
            self.wait_event(event)
        finally:
            del event

    def wait_event(self, event):
        if hasattr(event, "_event"):
            event = event._event
        if not isinstance(event, AclEvent):
            raise TypeError("ACL stream requires an AclEvent")
        event.wait(self)

    def record_event(self, event=None):
        if event is None:
            event = AclEvent()
        if hasattr(event, "_event"):
            event = event._event
        if not isinstance(event, AclEvent):
            raise TypeError("ACL stream requires an AclEvent")
        return event.record(self)

    def set_current(self):
        api = _stream_api()
        if api is None:
            raise RuntimeError("torch_fl ACL stream registry is unavailable")
        api.SetCurrentStream(self.device_index, _handle(self))


def current_acl_stream(device=None):
    from torch_fl.flagos import current_device

    idx = current_device() if device is None else int(device)
    api = _stream_api()
    if api is None:
        raise RuntimeError("torch_fl ACL stream registry is unavailable")
    handle = api.GetCurrentStream(idx)
    if not handle:
        raise RuntimeError("torch_fl ACL current stream is unavailable")
    return AclStream.borrowed(handle, device=idx)


def current_acl_raw_stream(device=None) -> int:
    """Return the current aclrtStream for `device` as a plain int.

    This is the handle Triton's generated launcher passes to rtKernelLaunch, so
    it must be the *same* stream torch_fl's aclnn ops run on: torch_fl creates
    its own stream, and a kernel launched on rt stream 0 has no ordering against
    the ops producing its inputs (see the nan-loss regression documented in
    scripts/patch_triton_ascend.py).

    Kept separate from `current_acl_stream` because callers on the launch path
    only need the integer and must not pay for an AclStream wrapper per kernel.
    """
    from torch_fl.flagos import current_device

    idx = current_device() if device is None else int(device)
    api = _stream_api()
    if api is None:
        raise RuntimeError("torch_fl ACL stream registry is unavailable")
    handle = api.GetCurrentStream(idx)
    if not handle:
        raise RuntimeError("torch_fl ACL current stream is unavailable")
    return int(handle)
