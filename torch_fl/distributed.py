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

"""
Distributed training utilities for flagos device.

Architecture (after import torch_fl)
--------------------------------------
``import torch_fl`` automatically:

1. Registers ``"flagos"`` as a ``torch.distributed`` backend for the
   ``privateuseone`` device type (via ``torch_fl.comm.ProcessGroupFlagOS``).
   The ProcessGroup wraps NCCL/FlagCX and converts privateuseone tensors to
   CUDA views internally — no monkeypatching of ``torch.distributed.*`` needed.

2. Patches ``torch.nn.parallel.DistributedDataParallel.__init__`` so that
   models on flagos devices automatically use ``python_reducer`` mode and
   flagos-compatible grad hooks.

After those two registrations, **standard PyTorch distributed APIs work as-is**:

    import torch_fl                          # registration happens here
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("flagos")        # or backend="nccl"/"auto"
    # — or auto-detect —
    dist.init_process_group(device_id=torch.device("privateuseone:0"))

    model = DistributedDataParallel(model)   # standard DDP, no wrapper needed

This module is kept for backward compatibility and convenience utilities.
``init_process_group`` is a thin alias; ``DistributedDataParallel`` is
deprecated in favour of the standard class.
"""

import warnings

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as _DDP


# ---------------------------------------------------------------------------
# Public API: init_process_group (compat alias)
# ---------------------------------------------------------------------------


def init_process_group(backend: str = "auto", **kwargs):
    """Initialize distributed process group for flagos device.

    Thin alias over ``torch.distributed.init_process_group``.  The
    ``"flagos"`` backend (registered by ``import torch_fl``) handles
    privateuseone tensors natively via ``ProcessGroupFlagOS``.

    Args:
        backend: ``"auto"`` or ``"flagos"`` (both map to the flagos backend),
            ``"nccl"`` (NVIDIA/MetaX direct), or ``"hccl"`` (Ascend).
            Default ``"auto"`` uses ``"flagos"`` which internally falls back
            to the vendor-native backend when FlagCX is unavailable.
        **kwargs: Forwarded to ``torch.distributed.init_process_group``.
    """
    actual = "flagos" if backend in ("auto", "flagos") else backend
    dist.init_process_group(backend=actual, **kwargs)


# ---------------------------------------------------------------------------
# Public API: DistributedDataParallel (deprecated — use standard DDP)
# ---------------------------------------------------------------------------


class DistributedDataParallel(_DDP):
    """Deprecated DDP wrapper for flagos models.

    .. deprecated::
        ``import torch_fl`` now patches
        ``torch.nn.parallel.DistributedDataParallel`` directly so that any
        model on a ``privateuseone`` device automatically gets the correct
        ``python_reducer`` mode and grad hooks.  Use the standard class:

            from torch.nn.parallel import DistributedDataParallel
            model = DistributedDataParallel(model)

        This subclass is kept for backward compatibility only and will be
        removed in a future release.
    """

    def __init__(self, module, **kwargs):
        warnings.warn(
            "torch_fl.distributed.DistributedDataParallel is deprecated. "
            "Use torch.nn.parallel.DistributedDataParallel directly — "
            "import torch_fl patches it automatically for flagos devices.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(module, **kwargs)


# ---------------------------------------------------------------------------
# Public API: move_buffers_to_device
# ---------------------------------------------------------------------------


def move_buffers_to_device(module, device):
    """Recursively move all module buffers to the specified device.

    When loading a model to the flagos device, some buffers (e.g. causal masks)
    may remain on CPU or CUDA. This ensures all buffers are on the correct
    device.

    Args:
        module: ``torch.nn.Module`` to process.
        device: Target device (e.g. ``"flagos:0"``).
    """
    for name, buf in module._buffers.items():
        if buf is not None and buf.device.type not in ("privateuseone", "flagos"):
            module._buffers[name] = buf.to(device)
    for child in module.children():
        move_buffers_to_device(child, device)


__all__ = [
    "init_process_group",
    "DistributedDataParallel",
    "move_buffers_to_device",
]
