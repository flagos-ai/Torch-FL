# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""FlashAttention 3 inference on the NVIDIA backend."""

from contextlib import contextmanager

import torch

from ... import _C
from ...flagos import _streams

_library = None


@contextmanager
def _cuda_stream(device_index):
    with torch.cuda.device(device_index):
        stream = torch.cuda.ExternalStream(
            _streams.get_raw_stream(device_index), device=device_index
        )
        # The Python CUDA shim reports the FlagOS queue. Save the CUDA
        # dispatcher's actual stream so the caller gets it back afterwards.
        previous = torch._C._cuda_getCurrentStream(device_index)
        torch.cuda.set_stream(stream)
        try:
            yield stream
        finally:
            torch.cuda._set_stream_by_id(*previous)


def _cuda_view(value, device):
    if not isinstance(value, torch.Tensor):
        return value
    if value.device != device:
        raise RuntimeError("FlashAttention 3 expects all tensors on the query device")
    return _C._flagos_to_cuda_view(value.resolve_conj().resolve_neg())


def _forward(q, k, v, *args, **kwargs):
    op = torch.ops.flash_attn_3._flash_attn_forward.default
    options = {
        argument.name: value for argument, value in zip(op._schema.arguments[3:], args)
    }
    options.update(kwargs)
    # The CUDA kernel can write through these arguments despite the functional
    # forward schema. This route implements read-only dense and varlen attention.
    if any(
        options.get(name) is not None
        for name in ("out_", "k_new", "v_new", "scheduler_metadata")
    ):
        raise NotImplementedError(
            "FlashAttention 3 on FlagOS supports read-only forward without an "
            "output buffer, KV-cache updates, or external scheduler metadata"
        )
    if q.device.type != "flagos":
        raise RuntimeError("FlashAttention 3 expects a FlagOS query tensor")

    outputs = []
    with _cuda_stream(q.device.index) as stream:
        inputs = tuple(_cuda_view(tensor, q.device) for tensor in (q, k, v))
        native_options = {
            name: _cuda_view(value, q.device) for name, value in options.items()
        }
        try:
            for native in op(*inputs, **native_options):
                if native is None:
                    outputs.append(None)
                    continue
                output = torch.empty_strided(
                    native.shape, native.stride(), dtype=native.dtype, device=q.device
                )
                outputs.append(output)
                _C._flagos_to_cuda_view(output).copy_(native)
        finally:
            # The views borrow FlagOS storage. Drain the shared queue before
            # releasing the borrows. Outputs retain the FlagOS allocator and
            # the CUDA result strides, including the logsumexp layout.
            stream.synchronize()
    return tuple(outputs)


def register():
    """Install the forward kernel without importing the optional CUDA extension."""
    global _library
    if _library is not None:
        return
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::empty.memory_format", "CUDA"
    ):
        return
    name = "_flash_attn_forward"
    if hasattr(
        torch.ops.flash_attn_3, name
    ) and torch._C._dispatch_has_kernel_for_dispatch_key(
        f"flash_attn_3::{name}", "PrivateUse1"
    ):
        return

    library = torch.library.Library("flash_attn_3", "IMPL", "PrivateUse1")
    library.impl(name, _forward)
    _library = library
