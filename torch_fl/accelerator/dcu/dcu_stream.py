# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
"""Real DCU streams/events backed by the vendor c10 HIP registry.

Subclass generic torch types so Tensor.record_stream and autograd understand
these objects on an official CPU Torch wheel. IDs are NOT raw HIP pointers.
"""

import contextlib
import operator
import struct
import threading
import torch
from torch_fl import _C


def device_index(device=None):
    if device is None:
        return torch.flagos.current_device()
    if isinstance(device, int):
        if device < 0:
            return torch.flagos.current_device()
        return device
    d = torch.device(device)
    if d.type not in ("flagos", "cuda", "privateuseone"):
        raise ValueError(f"Expected a DCU device, got {d}")
    return torch.flagos.current_device() if d.index is None else d.index


class DcuStream(torch.Stream):
    def __new__(cls, device=None, priority=0, **kwargs):
        if kwargs:
            allowed = {"stream_id", "device_index", "device_type"}
            if set(kwargs) != allowed:
                raise TypeError(f"Unsupported stream arguments: {kwargs}")
            obj = super().__new__(cls, **kwargs)
            if obj.device.type != "flagos":
                raise ValueError("DcuStream requires a flagos stream tag")
        else:
            obj = super().__new__(
                cls, device=f"flagos:{device_index(device)}", priority=priority
            )
        # Publish the per-object TLS before callers can share this Stream.
        # Lazy assignment in __enter__ races on the same object across threads.
        obj._contexts = threading.local()
        return obj

    @property
    def cuda_stream(self):
        return _C._dcu_raw_stream(self.stream_id, self.device_index)

    @property
    def handle(self):
        return self.cuda_stream

    @property
    def priority(self):
        return _C._dcu_stream_priority(self.stream_id, self.device_index)

    @staticmethod
    def priority_range():
        return _C._dcu_priority_range()

    def set_current(self):
        set_stream(self)

    def record_event(self, event=None):
        if event is None:
            event = DcuEvent()
        if not isinstance(event, torch.Event):
            raise TypeError("Expected a torch.Event")
        return super().record_event(event)

    def wait_event(self, event):
        if not isinstance(event, torch.Event):
            raise TypeError("Expected a torch.Event")
        return super().wait_event(event)

    def wait_stream(self, other):
        if not isinstance(other, torch.Stream) or other.device.type != "flagos":
            raise TypeError("Expected a flagos torch.Stream")
        return super().wait_stream(other)

    def __enter__(self):
        if not hasattr(self._contexts, "stack"):
            self._contexts.stack = []
        ctx = stream_context(self)
        ctx.__enter__()
        try:
            self._contexts.stack.append(ctx)
        except BaseException as error:
            try:
                ctx.__exit__(type(error), error, error.__traceback__)
            except BaseException as restore_error:
                _annotate_restore_errors(error, [restore_error])
            raise
        return self

    def __exit__(self, *exc):
        return self._contexts.stack.pop().__exit__(*exc)


class DcuEvent(torch.Event):
    def __new__(
        cls, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        if blocking or interprocess or external:
            raise NotImplementedError(
                "DCU generic Event does not support blocking, IPC or graph-external flags"
            )
        return super().__new__(cls, device="flagos", enable_timing=enable_timing)

    def record(self, stream=None):
        stream = current_stream() if stream is None else stream
        if not isinstance(stream, torch.Stream) or stream.device.type != "flagos":
            raise TypeError("Expected a flagos torch.Stream")
        return super().record(stream)

    def wait(self, stream=None):
        stream = current_stream() if stream is None else stream
        if not isinstance(stream, torch.Stream) or stream.device.type != "flagos":
            raise TypeError("Expected a flagos torch.Stream")
        return super().wait(stream)


def _existing(device, use_default):
    sid, idx, device_type = _C._dcu_stream_info(device_index(device), use_default)
    return DcuStream(stream_id=sid, device_index=idx, device_type=device_type)


def current_stream(device=None):
    return _existing(device, False)


def default_stream(device=None):
    return _existing(device, True)


def set_stream(stream):
    if stream is None:
        return
    if not isinstance(stream, torch.Stream) or stream.device.type != "flagos":
        raise TypeError("Expected a flagos torch.Stream")
    previous_device = torch.flagos.current_device()
    previous_stream = current_stream(stream.device_index)
    try:
        torch.flagos.set_device(stream.device_index)
        _C._dcu_set_stream(stream.stream_id, stream.device_index)
    except BaseException as error:
        _restore_state(previous_stream, previous_device, error)
        raise


def set_stream_by_id(stream_id, device_index, device_type):
    # Inductor's CUDA facade may supply the CUDA tag (1). The ID still belongs
    # to the same vendor registry; only its public tag needs normalizing.
    _, _, public_type = _C._dcu_stream_info(device_index, False)
    if device_type not in (1, public_type):
        raise ValueError(f"Expected a flagos/CUDA stream tag, got {device_type}")
    stream = DcuStream(
        stream_id=stream_id, device_index=device_index, device_type=public_type
    )
    set_stream(stream)


def _external_stream_args(raw, device):
    if isinstance(raw, bool):
        raise TypeError("External stream pointer must be an integer, not bool")
    raw = operator.index(raw)
    if not 0 <= raw < (1 << (8 * struct.calcsize("P"))):
        raise ValueError("External stream pointer is outside the native pointer range")
    sid, idx = _C._dcu_external_stream(raw, device_index(device))
    _, _, public_type = _C._dcu_stream_info(idx, False)
    return dict(stream_id=sid, device_index=idx, device_type=public_type)


class DcuExternalStream(DcuStream):
    """Borrow an external HIP stream without taking ownership of its lifetime."""

    def __new__(cls, stream_ptr, device=None):
        return super().__new__(cls, **_external_stream_args(stream_ptr, device))


def external_stream(raw, device=None):
    # Preserve the existing internal positional/raw factory API.
    return DcuExternalStream(stream_ptr=raw, device=device)


def _annotate_restore_errors(error, errors):
    if callable(getattr(error, "add_note", None)):
        for restore_error in errors:
            error.add_note(f"DCU stream/device restoration failed: {restore_error!r}")
    else:
        # Python 3.10 has no BaseException.add_note. Retain diagnostics while
        # preserving the primary exception rather than masking a model failure.
        error._flagos_restore_errors = getattr(
            error, "_flagos_restore_errors", ()
        ) + tuple(errors)


def _restore_state(
    previous_stream, previous_device, primary_error=None, source_stream=None
):
    errors = []
    try:
        _C._dcu_set_stream(previous_stream.stream_id, previous_stream.device_index)
    except BaseException as error:
        errors.append(error)
    if source_stream is not None:
        try:
            _C._dcu_set_stream(source_stream.stream_id, source_stream.device_index)
        except BaseException as error:
            errors.append(error)
    try:
        torch.flagos.set_device(previous_device)
    except BaseException as error:
        errors.append(error)
    if errors:
        if primary_error is not None:
            _annotate_restore_errors(primary_error, errors)
        else:
            if len(errors) > 1:
                _annotate_restore_errors(errors[0], errors[1:])
            raise errors[0]


@contextlib.contextmanager
def stream_context(stream):
    if stream is None:
        yield
        return
    if not isinstance(stream, torch.Stream) or stream.device.type != "flagos":
        raise TypeError("Expected a flagos torch.Stream")
    previous_device = torch.flagos.current_device()
    previous_stream = current_stream(stream.device_index)
    source_stream = (
        current_stream(previous_device)
        if previous_device != stream.device_index
        else None
    )
    primary_error = None
    try:
        set_stream(stream)
        yield stream
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # Includes a failed __enter__. Always attempt device restoration even
        # if restoring the target device's stream fails.
        _restore_state(previous_stream, previous_device, primary_error, source_stream)
