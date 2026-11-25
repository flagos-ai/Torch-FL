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

"""PyTorch Distributed Checkpoint (DCP) staging adapter for FlagOS tensors.

``torch.distributed.checkpoint.filesystem._OverlappingCpuLoader`` determines
its accelerator device type via ``stream.device_type`` when an explicit stream
is supplied or ``torch._utils._get_available_device_type()`` when ``stream`` is
``None`` (the default path in ``FileSystemWriter``). Two mismatches break
overlapping D2H staging on FlagOS:

1. ``_get_available_device_type()`` checks ``torch.cuda.is_available()`` before
   ``privateuse1``. On CUDA-shimmed FlagOS platforms (DCU, MetaX, NVIDIA CPU
   wheels, Kunlunxin, Iluvatar, GCU, MUSA alias), ``torch.cuda.is_available()``
   returns ``True``, so ``_OverlappingCpuLoader`` sets ``self.device_type =
   "cuda"`` and ``self.device_module = torch.cuda``. In ``_refill()``,
   ``tensor.device.type == "cuda"`` is ``False`` for ``flagos:0`` tensors,
   leaving them on device until ``FileSystemWriter`` aborts with
   ``AssertionError: Tensor must be on CPU`` (issue #533).
2. When a caller passes a ``torch.Stream`` / ``torch.flagos.Stream`` explicitly,
   ``stream.device_type`` is the integer ``c10::DeviceType`` enum value (``20``
   for ``PrivateUse1``) rather than the string ``"flagos"``, causing
   ``_get_device_module(stream.device_type)`` to fail ``getattr(torch, 20)``
   and ``tensor.device.type == self.device_type`` to miss.

This adapter points ``_OverlappingCpuLoader`` at ``torch.flagos`` when
``torch.flagos.is_available()`` is true (or when a ``flagos`` stream is passed),
preserving the caller's current stream across ``_refill()`` and staging both
``flagos`` and any ``cuda`` tensors to CPU before serialization.
"""

import functools

_FLAGOS_DEVICE_TYPES = ("flagos", "privateuseone")
_installed = False


def _resolve_flagos_device_type() -> str:
    import torch

    name = torch._C._get_privateuse1_backend_name()
    if name and hasattr(torch, name):
        return name
    return "flagos"


def _stream_device_type_name(stream) -> str | None:
    device = getattr(stream, "device", None)
    dev_type = getattr(device, "type", None)
    if isinstance(dev_type, str) and dev_type:
        return dev_type
    raw = getattr(stream, "device_type", None)
    if isinstance(raw, str) and raw:
        return raw
    return None


def install_dcp_compat() -> bool:
    """Patch DCP's ``_OverlappingCpuLoader`` to stage ``flagos`` tensors to host."""
    global _installed
    if _installed:
        return True

    try:
        import torch
        import torch.distributed.checkpoint.filesystem as _fs
    except ImportError:
        return False

    loader_cls = getattr(_fs, "_OverlappingCpuLoader", None)
    if loader_cls is None:
        return False

    _orig_init = loader_cls.__init__
    _orig_refill = loader_cls._refill

    @functools.wraps(_orig_init)
    def _patched_init(
        self,
        resolve_fun,
        stream=None,
        inflight_threshhold: int = 1_000_000,
    ) -> None:
        flagos_mod = getattr(torch, "flagos", None)
        flagos_available = False
        if flagos_mod is not None:
            try:
                flagos_available = bool(flagos_mod.is_available())
            except Exception:
                flagos_available = False

        stream_dev_type = (
            _stream_device_type_name(stream) if stream is not None else None
        )
        use_flagos = stream_dev_type in _FLAGOS_DEVICE_TYPES or (
            stream is None and flagos_available
        )
        if not use_flagos:
            if (
                stream is not None
                and isinstance(getattr(stream, "device_type", None), int)
                and stream_dev_type is not None
            ):
                # Normalize integer c10::DeviceType on non-flagos torch.Stream
                # instances before delegating to stock DCP.
                self.resolve_fun = resolve_fun
                self.items = []
                self.inflight_threshhold = inflight_threshhold
                self.in_flight_data = 0
                self.current_items = _fs.collections.deque()
                self.idx = 0
                self.started = False
                self.device_type = stream_dev_type
                self.device_module = _fs._get_device_module(stream_dev_type)
                self.stream = stream
                cur = self.device_module.current_stream()
                if self.stream != cur:
                    self.stream.wait_stream(cur)
                return
            return _orig_init(
                self,
                resolve_fun,
                stream=stream,
                inflight_threshhold=inflight_threshhold,
            )

        dev_type = (
            stream_dev_type
            if stream_dev_type and hasattr(torch, stream_dev_type)
            else _resolve_flagos_device_type()
        )
        dev_mod = getattr(torch, dev_type, flagos_mod)
        if dev_mod is None:
            return _orig_init(
                self,
                resolve_fun,
                stream=stream,
                inflight_threshhold=inflight_threshhold,
            )

        self.resolve_fun = resolve_fun
        self.items = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items = _fs.collections.deque()
        self.idx = 0
        self.started = False
        self.device_type = dev_type
        self.device_module = dev_mod
        current = dev_mod.current_stream()
        self.stream = stream if stream is not None else current
        if self.stream != current:
            self.stream.wait_stream(current)

    @functools.wraps(_orig_refill)
    def _patched_refill(self) -> None:
        if getattr(self, "device_type", None) not in _FLAGOS_DEVICE_TYPES:
            # Even when an explicit non-flagos stream was supplied, stage any
            # flagos tensor encountered so FileSystemWriter never receives a
            # device tensor.
            orig_resolve = self.resolve_fun

            def _resolve_with_flagos_fallback(obj):
                tensor = orig_resolve(obj)
                if tensor.device.type in _FLAGOS_DEVICE_TYPES:
                    return tensor.detach().to(device="cpu", non_blocking=False)
                return tensor

            self.resolve_fun = _resolve_with_flagos_fallback
            try:
                return _orig_refill(self)
            finally:
                self.resolve_fun = orig_resolve

        with self.device_module.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if tensor.device.type in _FLAGOS_DEVICE_TYPES:
                    tensor = tensor.to(device="cpu", non_blocking=True)
                elif tensor.device.type == "cpu":
                    if (
                        tensor.untyped_storage().size()
                        != tensor.numel() * tensor.itemsize
                    ):
                        tensor = tensor.clone()
                else:
                    # Mixed-device checkpoint (e.g. a CUDA tensor alongside
                    # FlagOS tensors): move synchronously to host so the writer
                    # invariant (`tensor.is_cpu`) still holds.
                    tensor = tensor.to(device="cpu", non_blocking=False)

                self.current_items.append((tensor, obj))
                self.in_flight_data += tensor.numel() * tensor.element_size()

    loader_cls.__init__ = _patched_init
    loader_cls._refill = _patched_refill
    _installed = True
    return True
