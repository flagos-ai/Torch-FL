# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convolution through the optional NVIDIA cuDNN frontend."""

from functools import cache, lru_cache, partial
from threading import local
from weakref import finalize

import torch

from torch_fl.flagos._streams import get_raw_stream


_devices = local()


@cache
def _frontend():
    try:
        import cudnn
    except (ImportError, OSError):
        return None
    return cudnn


def _plan(cudnn, handle, input_meta, weight_meta, output_meta, params, deterministic):
    padding, stride, dilation = params
    dtype = cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    image = graph.tensor(dim=input_meta[0], stride=input_meta[1], name="image")
    filter_ = graph.tensor(dim=weight_meta[0], stride=weight_meta[1], name="filter")
    output = graph.conv_fprop(
        image, filter_, padding=padding, stride=stride, dilation=dilation
    )
    output.set_output(True).set_data_type(dtype).set_stride(output_meta[1])
    try:
        graph.validate()
        graph.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return None
    notes = [cudnn.numerical_note.DOWN_CONVERT_INPUTS]
    if deterministic:
        notes.append(cudnn.numerical_note.NONDETERMINISTIC)
    for mode in (cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK):
        try:
            graph.create_execution_plans([mode])
            graph.deselect_numeric_notes(notes)
            graph.check_support()
            graph.build_plans()
        except cudnn.cudnnGraphNotSupportedError:
            continue
        return graph, image, filter_, output
    return None


class _DevicePlans:
    def __init__(self, cudnn):
        self.handle = cudnn.create_handle()
        self.convolution = lru_cache(maxsize=128)(partial(_plan, cudnn, self.handle))
        self._close = finalize(self, cudnn.destroy_handle, self.handle)


def convolution(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    rank = input.dim()
    if (
        transposed
        or any(output_padding)
        or rank not in (4, 5)
        or weight.dim() != rank
        or weight.device != input.device
        or input.dtype != torch.bfloat16
        or weight.dtype != input.dtype
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.enabled
        or torch.is_grad_enabled()
        and any(t.requires_grad for t in (input, weight, bias) if t is not None)
    ):
        return None
    cudnn = _frontend()
    if cudnn is None:
        return None
    shape = (input.size(0), weight.size(0)) + tuple(
        (i + 2 * p - d * (k - 1) - 1) // s + 1
        for i, k, p, d, s in zip(
            input.shape[2:], weight.shape[2:], padding, dilation, stride, strict=True
        )
    )
    if min(shape) <= 0:
        return None
    # The convolution API infers groups from the input and filter dimensions.
    if groups * weight.size(1) != input.size(1):
        return None
    channels_last = torch.channels_last_3d if rank == 5 else torch.channels_last
    memory_format = (
        channels_last
        if any(
            t.is_contiguous(memory_format=channels_last) and not t.is_contiguous()
            for t in (input, weight)
        )
        else torch.contiguous_format
    )
    image = input.contiguous(memory_format=memory_format)
    filter_ = weight.contiguous(memory_format=memory_format)
    result = torch.empty(
        shape, dtype=input.dtype, device=input.device, memory_format=memory_format
    )
    device = input.device.index
    if not hasattr(_devices, "plans"):
        _devices.plans = {}
    state = _devices.plans.get(device)
    if state is None:
        state = _devices.plans[device] = _DevicePlans(cudnn)
    cudnn.set_stream(state.handle, get_raw_stream(device))
    plan = state.convolution(
        (tuple(image.shape), image.stride()),
        (tuple(filter_.shape), filter_.stride()),
        (shape, result.stride()),
        (tuple(padding), tuple(stride), tuple(dilation)),
        torch.backends.cudnn.deterministic
        or torch.are_deterministic_algorithms_enabled(),
    )
    if plan is None:
        return None
    graph, x, w, y = plan
    scratch = torch.empty(
        graph.get_workspace_size(), dtype=torch.uint8, device=input.device
    )
    graph.execute(
        {x: image.data_ptr(), w: filter_.data_ptr(), y: result.data_ptr()},
        scratch.data_ptr(),
        handle=state.handle,
    )
    if bias is not None:
        result.add_(bias.reshape(1, -1, *(1,) * (rank - 2)))
    return result
