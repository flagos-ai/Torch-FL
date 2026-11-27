# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Convolution and fused attention through the NVIDIA cuDNN frontend."""

import math
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


def _compile_graph(cudnn, graph, deterministic):
    try:
        graph.validate()
        graph.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return False
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
        return True
    return False


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
    if _compile_graph(cudnn, graph, deterministic):
        return graph, image, filter_, output
    return None


def _attention_plan(
    cudnn, handle, query, key, value, output_strides, causal, scale, deterministic
):
    dtype = cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    q, k, v = (
        graph.tensor(dim=shape, stride=strides)
        for shape, strides in (query, key, value)
    )
    try:
        result, _ = graph.sdpa(
            q=q,
            k=k,
            v=v,
            attn_scale=scale,
            generate_stats=False,
            diagonal_band_right_bound=0 if causal else None,
        )
    except (TypeError, cudnn.cudnnGraphNotSupportedError):
        return None
    result.set_output(True).set_data_type(dtype).set_dim(
        (*query[0][:-1], value[0][-1])
    ).set_stride(output_strides)
    try:
        if _compile_graph(cudnn, graph, deterministic):
            return graph, q, k, v, result, graph.get_workspace_size()
    except (RuntimeError, ValueError):
        return None
    return None


class _DevicePlans:
    def __init__(self, cudnn):
        self.handle = cudnn.create_handle()
        self.convolution = lru_cache(maxsize=128)(partial(_plan, cudnn, self.handle))
        self.attention = lru_cache(maxsize=128)(
            partial(_attention_plan, cudnn, self.handle)
        )
        self._close = finalize(self, cudnn.destroy_handle, self.handle)


def _device_plans(cudnn, device):
    if not hasattr(_devices, "plans"):
        _devices.plans = {}
    state = _devices.plans.get(device)
    if state is None:
        state = _devices.plans[device] = _DevicePlans(cudnn)
    cudnn.set_stream(state.handle, get_raw_stream(device))
    return state


def attention(query, key, value, is_causal, scale, enable_gqa):
    shape = query.shape
    # Keep single-query attention on the existing GPU composition. Fused
    # decode and multi-query plans use different accumulation orders.
    if query.ndim != 4 or shape[2] == 1:
        return None
    if (
        any(
            t.ndim != 4
            or t.device != query.device
            or t.dtype != torch.bfloat16
            or not t.numel()
            or t.stride(-1) != 1
            for t in (query, key, value)
        )
        or key.shape[:3] != value.shape[:3]
        or key.size(0) != shape[0]
        or key.size(3) != shape[3]
        or shape[1] % key.size(1)
        or (not enable_gqa and shape[1] != key.size(1))
        or any(
            width < 8 or width > 128 or width % 8 for width in (shape[3], value.size(3))
        )
    ):
        return None
    cudnn = _frontend()
    if cudnn is None or not hasattr(cudnn.pygraph, "sdpa"):
        return None
    scale = 1.0 / math.sqrt(shape[3]) if scale is None else scale
    if not math.isfinite(scale):
        return None
    output = torch.empty(
        (shape[0], shape[2], shape[1], value.size(3)),
        dtype=query.dtype,
        device=query.device,
    ).transpose(1, 2)
    state = _device_plans(cudnn, query.device.index)
    plan = state.attention(
        *((tuple(t.shape), t.stride()) for t in (query, key, value)),
        output.stride(),
        is_causal,
        scale,
        torch.are_deterministic_algorithms_enabled(),
    )
    if plan is None:
        return None
    graph, q, k, v, result, workspace = plan
    scratch = torch.empty(workspace, dtype=torch.uint8, device=query.device)
    graph.execute(
        {
            q: query.data_ptr(),
            k: key.data_ptr(),
            v: value.data_ptr(),
            result: output.data_ptr(),
        },
        scratch.data_ptr(),
        handle=state.handle,
    )
    return output


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
    state = _device_plans(cudnn, input.device.index)
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
