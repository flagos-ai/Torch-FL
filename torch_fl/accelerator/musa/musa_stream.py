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

"""The one MUSA stream, and why FSDP's four streams are all the same handle.

``torch.flagos.Stream`` needs a native implementation on MUSA because the CPU
torch wheel has no ``torch.cuda.Stream`` to inherit from, which is what every
other arm of that dispatcher assumes. This module is that implementation, and it
deliberately does not create a stream.

``GetDefaultMusaStream()`` (``csrc/runtime/accelerator/musa/musa_stream.cc``) is
the stream every device-side op in the process runs on. ``GetMudnnHandle`` binds
each device's mudnn handle to it when that handle is first built and never
re-reads it, so mudnn stays there for the life of the process, and the Triton
kernels FlagGems launches reach the same handle through
``flagtree_shim.get_musa_current_raw_stream``. ``musa_stream.h`` records what
happened while two "default" streams existed: ``libtorch_bindings.so`` and
``libtorch_fl.so`` each instantiated their own function-local static, so a kernel
and the mudnn op consuming its output were never ordered against each other -- a
wrong answer, not a slow one.

``MusaStream`` therefore denotes the stream that already exists rather than
adding a second one. FSDP asks for four streams before it starts
(``all_gather_copy_in_stream``, ``all_gather_stream``, ``reduce_scatter_stream``,
``all_reduce_stream``); it gets four handles to this one queue, and the copies it
meant to overlap run one after another. That is slower than FSDP on CUDA, and it
is correct. Handing out genuinely separate streams would be neither, because the
mudnn ops those copies feed would keep running on the default stream, leaving an
op and the copy that produced its input ordered either way -- the defect above,
reintroduced. Real overlap needs mudnn moved off ``GetDefaultMusaStream`` first,
which is a change to the op layer and not to this class.

Everything FSDP waits on was submitted to this queue, so ``wait_stream`` and
``wait_event`` have nothing to order. ``synchronize`` is real: it drains the
device through ``flagos.synchronize``. ``flagos.current_stream`` needs no
counterpart here -- ``flagos._DefaultStreamHandle`` already carries this same
handle on MUSA.
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
    to ``GetDefaultMusaStream()`` and outlives every Python object.
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

        Always true, for the reason ``flagos._DefaultStreamHandle.query`` is: the
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
