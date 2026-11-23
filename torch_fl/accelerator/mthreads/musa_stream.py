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

"""Borrowed streams for MUSA builds with the legacy single-queue runtime.

Wheels with the PrivateUse1 stream registry use torch_fl.flagos._streams.
"""

import torch


def _shared_handle(device_index: int) -> int:
    """The ``musaStream_t`` the device's ops run on.

    ``_C._get_musa_current_raw_stream`` answers for the current device only: the
    index is checked against ``c10::flagos::CurrentDevice()``, so asking about
    another device's stream would describe the wrong one. A stream built for
    another device therefore makes that device current for the lookup and puts
    the previous one back, mirroring ``TopsStream``, which swaps devices around
    its own ``StreamCreateWithPriority`` call for the same reason.
    """
    from torch_fl.flagos import _C, current_device, set_device

    previous = current_device()
    if previous == device_index:
        return int(_C._get_musa_current_raw_stream(device_index))
    set_device(device_index)
    try:
        return int(_C._get_musa_current_raw_stream(device_index))
    finally:
        set_device(previous)


class MusaStream:
    """A borrowed handle to the device's single MUSA stream.

    The surface is what ``torch_fl.flagos.Stream`` delegates to, and no more.
    FSDP builds four of these at startup and hands them to the collectives, so
    what is used is the handle for the launch and
    ``wait_stream``/``wait_event``/``record_event`` for the ordering around it.
    Nothing here owns the handle, so there is no ``__del__``: the stream belongs
    to the legacy runtime and outlives every Python object.
    """

    def __init__(self, device=None, priority=0, **kwargs):
        # ``priority`` is dropped rather than handed to
        # musaStreamCreateWithPriority, which this class never calls. The one
        # stream the process has was created without one, and FSDP's ``-1`` for
        # its three collectives streams is a scheduling hint with no second stream
        # to apply to.
        del priority, kwargs
        from torch_fl.flagos import current_device

        if device is None:
            index = current_device()
        elif isinstance(device, torch.device):
            index = current_device() if device.index is None else device.index
        else:
            index = int(device)

        self.device_index = index
        self.device = torch.device("flagos", index)
        self._handle = _shared_handle(index)

    def __repr__(self) -> str:
        return (
            f"<MusaStream device=flagos:{self.device_index} handle={self._handle:#x}>"
        )

    @property
    def handle(self) -> int:
        """The raw handle, under the name ``flagos.Stream.__repr__`` reads."""
        return self._handle

    @property
    def stream_id(self) -> int:
        """The handle, under the name ``torch.cuda.Stream`` uses for one."""
        return self._handle

    @property
    def musa_stream(self) -> int:
        """The handle, under the vendor name MUSA's own launchers use."""
        return self._handle

    @property
    def cuda_stream(self) -> int:
        """The handle, under the CUDA-derived name FlagGems reads.

        FlagGems' Triton launcher reads ``.cuda_stream`` off whatever
        ``current_stream()`` returns on the backends that derive from CUDA. There
        is no CUDA stream on MUSA, so this names the MUSA one: it is the queue the
        kernels are submitted to, which is the only thing the caller wants the
        number for. It is the same value as ``musa_stream``.
        """
        return self._handle

    def synchronize(self) -> None:
        from torch_fl.flagos import synchronize

        synchronize(self.device_index)

    def query(self) -> bool:
        """Whether the queue has drained.

        Always true, for the reason ``flagos._streams._StreamHandle.query`` is: the
        mudnn exit path drains the device after each op, so by the time a caller
        can ask, the op path has left nothing outstanding. A FlagGems Triton
        kernel launched since then is not covered by that -- there is no binding
        for ``musaStreamQuery`` in ``torch_fl._C`` to ask the driver directly.
        """
        return True

    def wait_stream(self, other) -> None:
        """No-op: ``other`` is this queue, or a stream on it."""
        del other

    def wait_event(self, event) -> None:
        """No-op: an event this process recorded was recorded on this queue."""
        del event

    def record_event(self, event=None):
        """Return an event over the work submitted so far.

        FSDP wants an event it can hand back to ``wait_event`` and
        ``synchronize``. There is nothing for the wait to order on a single queue,
        so what is left is the synchronize, which is all ``flagos.Event`` does:
        its ``record`` drains the device and reads the host clock. FSDP asks the
        event for ordering, not for a device-time measurement, so the host clock
        is enough -- see ``flagos._HostTimedEvent``.
        """
        from torch_fl.flagos import Event

        if event is None:
            event = Event()
        native = getattr(event, "_event", event)
        native.record(self)
        return event

    def set_current(self) -> None:
        """No-op: the shared stream is the current stream already."""
