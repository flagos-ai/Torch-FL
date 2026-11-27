# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Test reductions against the same PyTorch reference on each device."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch_fl  # noqa: F401


@pytest.mark.anyplatform
@pytest.mark.parametrize("dim", [0, 1])
def test_empty_mean_keeps_reduction_shape(dim):
    host = torch.empty((0, 4), dtype=torch.float32)
    data = host.to("flagos:0")
    result = data.mean(dim=dim, keepdim=True)
    assert result.device == data.device
    torch.testing.assert_close(
        result.cpu(), host.mean(dim=dim, keepdim=True), equal_nan=True
    )


DEVICE = torch.device("flagos:0")
OPERATORS = {"argmin": "argmin", "max": "max.dim", "min": "min.dim"}


def _assert_reduction(op, input_cpu, dim, keepdim=False, *, transpose=False):
    reduce = getattr(torch, op)
    device_input = input_cpu.to(DEVICE)
    if transpose:
        input_cpu = input_cpu.T
        # Create the strided view without invoking a second operator backend.
        device_input = device_input.as_strided(
            input_cpu.shape, input_cpu.stride(), input_cpu.storage_offset()
        )
    expected = reduce(input_cpu, dim=dim, keepdim=keepdim)
    actual = reduce(device_input, dim=dim, keepdim=keepdim)
    if op == "argmin":
        actual, expected = (actual,), (expected,)
    for result, reference in zip(actual, expected, strict=True):
        assert result.device == DEVICE
        torch.testing.assert_close(
            result.cpu(), reference, atol=0, rtol=0, equal_nan=True
        )


def _check_cases(op):
    # A fresh interpreter makes the dispatch log independent of earlier tests.
    torch.ops.flagos.strict_host_routes(True)
    nan, inf = float("nan"), float("inf")
    ties = torch.tensor([[4.0, -2, -2, 4], [-3, -3, 6, 6], [2, 2, 2, 2]])
    nans = torch.tensor([[inf, -inf, nan, nan], [nan, 4, -2, -2], [nan] * 4])
    extremes = torch.tensor([[inf, -1, inf], [-inf] * 3, [inf] * 3])
    rank_three = nans.reshape(2, 2, 3)
    for keepdim in (False, True):
        _assert_reduction(op, ties, -1, keepdim)
        _assert_reduction(op, nans, -1, keepdim)
        _assert_reduction(op, extremes, -1, keepdim)
        _assert_reduction(op, nans, 0, keepdim, transpose=True)
        _assert_reduction(op, rank_three, 0, keepdim)
        _assert_reduction(op, rank_three, 2, keepdim)
        _assert_reduction(op, torch.empty(0, 3), 1, keepdim)

    # The later blocks must neither displace an earlier tie nor miss a NaN.
    wide = torch.full((2, 2051), 3.0)
    wide[0, [17, 1537]] = -4.0
    wide[1, [1025, 2050]] = nan
    _assert_reduction(op, wide, 1)
    tail = torch.full((2, 20), 1000.0)
    tail[0, -1] = 1.0
    tail[1, :12] = -1000.0
    _assert_reduction(op, tail, 1)
    _assert_reduction(op, torch.tensor([[3.0, nan, -4, -4, 9, 9, nan]]), 1)
    _assert_reduction(op, torch.tensor([[2.0]]), 1)
    _assert_reduction(op, torch.tensor([2.0, nan, 2.0]), 0)
    _assert_reduction(op, torch.tensor(nan), 0)
    if op == "argmin":
        _assert_reduction(op, nans, None, transpose=True)
        _assert_reduction(op, nans, None, True, transpose=True)

    for device in ("cpu", DEVICE):
        empty = torch.empty(2, 0, 3, device=device)
        with pytest.raises(IndexError, match="reduction dim 1 to have non-zero size"):
            getattr(torch, op)(empty, dim=1)


@pytest.mark.anyplatform
@pytest.mark.parametrize("op", OPERATORS)
def test_indexed_reduction(op):
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys, torch_fl; runpy.run_path(sys.argv[1])['_check_cases'](sys.argv[2])",
            str(Path(__file__).resolve()),
            op,
        ],
        env={**os.environ, "FLAGOS_LOG": "dispatch,fallback"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "[flagos cpu_fallback]" not in process.stderr, process.stderr
    assert f"[flagos dispatch] {OPERATORS[op]} -> " in process.stderr, process.stderr


@pytest.mark.anyplatform
@pytest.mark.parametrize("shape", [(0, 16, 128), (2, 16, 0)])
def test_mean_empty_input(shape):
    inputs = torch.empty(shape, dtype=torch.float32)
    expected = inputs.mean(-1, keepdim=True)
    result = inputs.to(DEVICE).mean(-1, keepdim=True)
    assert result.device == DEVICE
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.anyplatform
@pytest.mark.parametrize("inplace", [False, True])
def test_scatter_add_expanded_index(inplace):
    rows, columns = 16, 12
    inputs = torch.full((rows + 2, columns), 0.5, dtype=torch.bfloat16)
    source = torch.full((rows, columns), 0.25, dtype=inputs.dtype)
    storage = torch.zeros(rows * columns, dtype=torch.long)
    storage[:rows] = torch.arange(1, rows + 1)
    index = storage[:rows].view(rows, 1).expand_as(source)
    device_index = storage.to(DEVICE)[:rows].view(rows, 1).expand_as(source)
    assert device_index.stride() == (1, 0)
    expected = inputs.scatter_add(0, index, source)

    destination = inputs.to(DEVICE)
    operation = destination.scatter_add_ if inplace else destination.scatter_add
    result = operation(0, device_index, source.to(DEVICE))
    assert result.device == DEVICE
    assert result.dtype == inputs.dtype
    assert (result is destination) == inplace
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        destination.cpu(), expected if inplace else inputs, rtol=0, atol=0
    )
