# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify NVIDIA device copies with strict operator dispatch."""

import pytest
import torch

import torch_fl  # noqa: F401

DEVICE = "flagos:0"


@pytest.mark.nvidia
@pytest.mark.flaggems_python
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex64, torch.complex128])
def test_nvidia_same_dtype_transpose_copy_without_cpu_fallback(dtype):
    from torch_fl import _build_config

    if not _build_config.NATIVE_HOSTS:
        pytest.skip("NVIDIA native host copy is required")
    host = torch.tensor([[1.25, -1.125, 0.5], [0.0, 2.5, -16777217.0]], dtype=dtype)
    source = host.to(DEVICE)
    out = torch.empty((3, 2), dtype=dtype, device=DEVICE).t()
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        out.copy_(source)
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    torch.testing.assert_close(out.cpu(), host, rtol=0, atol=0)


@pytest.mark.nvidia
@pytest.mark.flaggems_python
def test_nvidia_disjoint_strided_views_copy_without_cpu_fallback():
    expected = torch.arange(2 * 5 * 96, dtype=torch.float32).reshape(2, 5, 96)
    actual = expected.to(DEVICE)
    expected[0, :, 1::3].copy_(expected[1, :, 1::3])

    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        actual[0, :, 1::3].copy_(actual[1, :, 1::3])
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.nvidia
@pytest.mark.flaggems_python
def test_nvidia_uint32_integer_cast_without_cpu_fallback():
    source = torch.tensor([-1, 0, 2**31, 2**32 + 1], dtype=torch.int64)
    actual = source.to(DEVICE)

    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        round_trip = actual.to(torch.uint32).to(torch.int64)
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    torch.testing.assert_close(
        round_trip.cpu(), source.to(torch.uint32).to(torch.int64)
    )
