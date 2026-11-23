# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Test indexed reductions with the same PyTorch reference on each device."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch_fl  # noqa: F401

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
    if op == "argmin":
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
