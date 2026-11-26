# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Indexed reductions with ATen's first-NaN and first-extremum semantics."""

import math
import operator

import torch
import triton
import triton.language as tl
from flag_gems import runtime

from torch_fl import flagos


@triton.jit
def _offsets(positions, shape: tl.constexpr, strides: tl.constexpr):
    offsets = tl.full(positions.shape, 0, positions.dtype)
    for i in tl.static_range(len(shape) - 1, 0, -1):
        offsets += (positions % shape[i]) * strides[i]
        positions = positions // shape[i]
    if len(shape) > 0:
        offsets += positions * strides[0]
    return offsets


@triton.jit
def _copy_flat(
    inp,
    out,
    size: tl.constexpr,
    shape: tl.constexpr,
    strides: tl.constexpr,
    wide_offsets: tl.constexpr,
    block: tl.constexpr,
):
    offset_type: tl.constexpr = tl.int64 if wide_offsets else tl.int32
    positions = tl.program_id(0).to(offset_type) * block + tl.arange(0, block)
    values = tl.load(inp + _offsets(positions, shape, strides), positions < size, 0)
    tl.store(out + positions, values, positions < size)


@triton.jit
def _select_positions(
    inp,
    positions,
    rows: tl.constexpr,
    width: tl.constexpr,
    row_shape: tl.constexpr,
    row_strides: tl.constexpr,
    col_stride: tl.constexpr,
    identity: tl.constexpr,
    minimum: tl.constexpr,
    wide_offsets: tl.constexpr,
    chunk_start: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    offset_type: tl.constexpr = tl.int64 if wide_offsets else tl.int32
    blocks: tl.constexpr = tl.cdiv(width, block_cols)
    if width <= block_cols:
        chunk = 0
    else:
        chunk = tl.program_id(1).to(offset_type) + chunk_start
    row = tl.program_id(0).to(offset_type) * block_rows + tl.arange(0, block_rows)
    lane = tl.arange(0, block_cols)
    col = chunk * block_cols + lane
    row_offset = _offsets(row, row_shape, row_strides)
    col_offset = col * col_stride
    valid = (row[:, None] < rows) & (col[None, :] < width)
    values = tl.load(inp + row_offset[:, None] + col_offset[None, :], valid, 0)
    if values.dtype == tl.float16 or values.dtype == tl.bfloat16:
        values = values.to(tl.float32)
    nan = values != values
    ordered = valid & ~nan
    comparable = tl.where(ordered, values, identity)
    if minimum:
        extreme = tl.min(comparable, 1)
    else:
        extreme = tl.max(comparable, 1)

    # A NaN lane ranks before any ordered extremum, with ties broken by
    # position. If a finite reduction identity beats a tile of all losing
    # infinities, the sentinel maps to the first lane in that tile.
    rank = tl.where(
        valid & nan,
        lane[None, :],
        tl.where(
            ordered & (values == extreme[:, None]),
            block_cols + lane[None, :],
            2 * block_cols,
        ),
    )
    selected = chunk * block_cols + (tl.min(rank, 1) & (block_cols - 1))
    out_offset = row * blocks + chunk
    tl.store(positions + out_offset, selected, row < rows)


@triton.jit
def _gather_winners(
    inp,
    positions,
    previous_indices,
    values_out,
    indices_out,
    rows: tl.constexpr,
    blocks: tl.constexpr,
    row_shape: tl.constexpr,
    row_strides: tl.constexpr,
    col_stride: tl.constexpr,
    write_values: tl.constexpr,
    from_partials: tl.constexpr,
    wide_offsets: tl.constexpr,
    chunk_start: tl.constexpr,
    block_rows: tl.constexpr,
):
    offset_type: tl.constexpr = tl.int64 if wide_offsets else tl.int32
    chunk = tl.program_id(1).to(offset_type) + chunk_start
    row = tl.program_id(0).to(offset_type) * block_rows + tl.arange(0, block_rows)
    out_offset = row * blocks + chunk
    selected = tl.load(positions + out_offset, row < rows, 0).to(offset_type)
    selected_offset = _offsets(row, row_shape, row_strides) + selected * col_stride
    if from_partials:
        index = tl.load(previous_indices + selected_offset, row < rows, 0)
        tl.store(indices_out + out_offset, index, row < rows)
    if write_values:
        value = tl.load(inp + selected_offset, row < rows, 0)
        tl.store(values_out + out_offset, value, row < rows)


def _last_offset(inp):
    return sum((size - 1) * stride for size, stride in zip(inp.shape, inp.stride()))


def _flatten(inp, size):
    if inp.is_contiguous():
        return inp.view(-1)

    # Materialize logical row-major order on devices with a host-backed clone.
    flat = torch.empty(size, dtype=inp.dtype, device=inp.device)
    block = min(triton.next_power_of_2(size), 64)
    wide = max(_last_offset(inp), size + block) >= 2**31
    with flagos.device(inp.device):
        _copy_flat[(triton.cdiv(size, block),)](
            inp, flat, size, tuple(inp.shape), inp.stride(), wide, block
        )
    return flat


def _launch(
    inp,
    rows,
    width,
    row_shape,
    row_strides,
    col_stride,
    values,
    indices,
    *,
    minimum,
    write_values,
):
    if inp.dtype.is_floating_point:
        identity = float("inf") if minimum else float("-inf")
    elif inp.dtype == torch.bool:
        identity = minimum
    else:
        bounds = torch.iinfo(inp.dtype)
        identity = bounds.max if minimum else bounds.min

    largest_offset = _last_offset(inp)
    previous = None
    with flagos.device(inp.device):
        while True:
            block_cols = min(triton.next_power_of_2(width), 64)
            block_rows = min(triton.next_power_of_2(rows), 8, 64 // block_cols)
            blocks = triton.cdiv(width, block_cols)
            final = blocks == 1
            if final:
                next_values, next_indices = values, indices
            else:
                shape = (rows, blocks)
                next_values = torch.empty(shape, dtype=inp.dtype, device=inp.device)
                next_indices = torch.empty(
                    shape, dtype=indices.dtype, device=inp.device
                )
            positions = (
                next_indices if previous is None else torch.empty_like(next_indices)
            )
            row_groups = triton.cdiv(rows, block_rows)
            wide = (
                max(
                    largest_offset,
                    row_groups * block_rows,
                    rows * blocks,
                    width + block_cols,
                )
                >= 2**31
            )
            # A separate grid axis keeps division out of the address
            # calculation. Split it at the CUDA limit on the second dimension.
            grid_limit = 65535
            for start in range(0, blocks, grid_limit):
                grid = (row_groups, min(blocks - start, grid_limit))
                _select_positions[grid](
                    inp,
                    positions,
                    rows,
                    width,
                    row_shape,
                    row_strides,
                    col_stride,
                    identity,
                    minimum,
                    wide,
                    start,
                    block_rows,
                    block_cols,
                )
                # Keep the indexed load separate from the position reduction.
                # Some compilers miscompile a fused reduction and gather.
                if not final or write_values or previous is not None:
                    _gather_winners[grid](
                        inp,
                        positions,
                        next_indices if previous is None else previous,
                        next_values,
                        next_indices,
                        rows,
                        blocks,
                        row_shape,
                        row_strides,
                        col_stride,
                        not final or write_values,
                        previous is not None,
                        wide,
                        start,
                        block_rows,
                    )
            if final:
                return
            inp, previous = next_values, next_indices
            width = blocks
            row_shape, row_strides = (rows,), (width,)
            col_stride = 1
            largest_offset = rows * width - 1


def _reduce(inp, dim, keepdim, *, minimum, return_values, op):
    rank = inp.ndim
    if dim is None:
        row_shape = row_strides = ()
        width = inp.numel()
        if width == 0:
            raise IndexError(
                f"{op}(): Expected reduction dim to be specified for input.numel() == 0."
            )
        inp = _flatten(inp, width)
        col_stride = 1
        output_shape = (1,) * rank if keepdim else ()
    else:
        dim = operator.index(dim)
        dims = max(rank, 1)
        if not -dims <= dim < dims:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-dims}, {dims - 1}], but got {dim})"
            )
        axis = dim % dims
        shape, strides = (tuple(inp.shape), inp.stride()) if rank else ((1,), (0,))
        width = shape[axis]
        if width == 0:
            raise IndexError(
                f"{op}(): Expected reduction dim {axis} to have non-zero size."
            )
        row_shape = shape[:axis] + shape[axis + 1 :]
        row_strides = strides[:axis] + strides[axis + 1 :]
        col_stride = strides[axis]
        output_shape = (
            shape[:axis] + (1,) + shape[axis + 1 :] if keepdim and rank else row_shape
        )

    rows = math.prod(row_shape)
    narrow_indices = rows != 0 and not runtime.device.support_int64
    index_dtype = torch.int32 if narrow_indices else torch.int64
    indices = torch.empty(output_shape, dtype=index_dtype, device=inp.device)
    values = (
        torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        if return_values
        else indices
    )
    if rows:
        _launch(
            inp,
            rows,
            width,
            row_shape,
            row_strides,
            col_stride,
            values,
            indices,
            minimum=minimum,
            write_values=return_values,
        )
        if narrow_indices:
            indices = indices.to(torch.int64)
    return (values, indices) if return_values else indices


def argmin(inp, dim=None, keepdim=False):
    return _reduce(inp, dim, keepdim, minimum=True, return_values=False, op="argmin")


def max_dim(inp, dim, keepdim=False):
    return _reduce(inp, dim, keepdim, minimum=False, return_values=True, op="max")


def min_dim(inp, dim, keepdim=False):
    return _reduce(inp, dim, keepdim, minimum=True, return_values=True, op="min")
