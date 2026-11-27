# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401
from torch_fl._platform import detect_platform


requires_nvidia = pytest.mark.skipif(
    detect_platform() != "cuda", reason="requires NVIDIA's strict GPU routes"
)


@pytest.mark.anyplatform
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_concat_strided_tensors(dtype):
    host = torch.arange(24, dtype=dtype).reshape(4, 6)
    device = host.to("flagos:0")
    expected = torch.cat((host[:, ::2], host[:, 1::2]), dim=-2)
    actual = torch.cat((device[:, ::2], device[:, 1::2]), dim=-2)
    assert actual.device == device.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.anyplatform
def test_nonzero_resolves_grid_for_expanded_mask():
    host = (torch.arange(31) % 7 == 0).reshape(1, 31, 1)
    device = host.to("flagos:0").expand(1, 31, 257)
    actual = device.nonzero()
    assert actual.device == device.device
    torch.testing.assert_close(actual.cpu(), host.expand(1, 31, 257).nonzero())


@pytest.mark.anyplatform
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32, torch.int64]
)
def test_zero_pad_strided_tensor(dtype):
    host = torch.arange(-12, 12, dtype=dtype).reshape(4, 6)
    device = host.to("flagos:0")
    pad = (2, 1, 1, 0)
    expected = F.pad(host[:, ::2], pad, value=0.0)
    actual = F.pad(device[:, ::2], pad, value=0.0)
    assert actual.device == device.device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@requires_nvidia
@pytest.mark.flaggems_python
def test_equal_and_mul_out_without_cpu_fallback():
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    left = source.to("flagos:0")
    right = torch.tensor([2.0, 3.0, 4.0]).to(left.device)
    out = torch.empty((3, 4), device=left.device).transpose(0, 1)
    expected = (source * right.cpu()).to(left.device)
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        assert torch.mul(left, right, out=out) is out
        assert torch.equal(out, expected)
        assert not torch.equal(out, left)
        assert not torch.equal(out, expected.flatten())
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)


@requires_nvidia
@pytest.mark.flaggems_python
def test_float_arithmetic_accepts_cpu_scalar_tensors():
    host = torch.linspace(-2, 2, 30).reshape(5, 6)
    device = host.to("flagos:0").t()
    host = host.t()
    scale = torch.tensor(1.01)
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        results = (
            (device - scale, host - scale),
            (scale - device, scale - host),
            (device / scale, host * (1 / scale.item())),
        )
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    for actual, expected in results:
        assert actual.device == device.device
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@requires_nvidia
@pytest.mark.flaggems_python
def test_prod_preserves_large_long_values_without_cpu_fallback():
    host = torch.tensor([[2**30 + 1, 3, 5], [2**27 + 1, 11, 17]])
    source = host.to("flagos:0")
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        result = source.prod(dim=-1, keepdim=True)
        empty = source[:, :0].prod(dim=-1)
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    torch.testing.assert_close(result.cpu(), host.prod(dim=-1, keepdim=True))
    torch.testing.assert_close(empty.cpu(), host[:, :0].prod(dim=-1))


@requires_nvidia
@pytest.mark.flaggems_python
def test_index_selection_and_advanced_indexing_without_cpu_fallback():
    host = torch.arange(60).reshape(3, 4, 5)
    source = host.to("flagos:0")
    rows = torch.tensor([[0], [-1]])
    columns = torch.tensor([[4, 0]])
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        result = source[rows.to(source.device), :, columns.to(source.device)]
        selected = source.index_select(1, torch.tensor([3, 1]).to(source.device))
        values = torch.arange(16).reshape(2, 2, 4) - 32
        source[rows.to(source.device), :, columns.to(source.device)] = values.to(
            source.device
        )
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    torch.testing.assert_close(result.cpu(), host[rows, :, columns])
    torch.testing.assert_close(
        selected.cpu(), host.index_select(1, torch.tensor([3, 1]))
    )
    host[rows, :, columns] = values
    torch.testing.assert_close(source.cpu(), host)


@pytest.mark.anyplatform
def test_index_copy_preserves_output_layout():
    host = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    device = host.to("flagos:0")
    index = torch.tensor([2, 0])
    updates = torch.arange(-12, 0, dtype=host.dtype).reshape(2, 6).t()
    out = torch.empty((4, 6), device=device.device, dtype=host.dtype).t()
    actual = torch.index_copy(
        device, -1, index.to(device.device), updates.to(device.device), out=out
    )
    assert actual is out
    assert out.stride() == host.stride()
    torch.testing.assert_close(out.cpu(), host.index_copy(-1, index, updates))


@pytest.mark.anyplatform
def test_linear_solve_accepts_vector_and_matrix_rhs():
    matrix = torch.tensor([[2.0, 1.0], [1.0, -1.0]])
    vector = torch.tensor([3.0, 0.0])
    matrix_rhs = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    batch = torch.stack((matrix, matrix + torch.eye(2)))
    for lhs, rhs in ((matrix, vector), (matrix, matrix_rhs), (batch, matrix_rhs)):
        expected = torch.linalg.solve(lhs, rhs)
        gpu_lhs = lhs.to("flagos:0")
        actual = torch.linalg.solve(gpu_lhs, rhs.to(gpu_lhs.device))
        assert actual.device == gpu_lhs.device
        torch.testing.assert_close(actual.cpu(), expected)


@requires_nvidia
@pytest.mark.flaggems_python
def test_index_copy_preserves_last_write_without_cpu_fallback():
    host = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).t()
    device = host.to("flagos:0")
    index = torch.tensor([2, 0, 2], device=device.device)
    updates = torch.arange(-18, 0, dtype=host.dtype).reshape(3, 6).t()
    out = torch.empty((4, 6), device=device.device, dtype=host.dtype).t()
    expected = host.clone()
    expected[:, 0] = updates[:, 1]
    expected[:, 2] = updates[:, 2]
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        assert (
            torch.index_copy(device, -1, index, updates.to(device.device), out=out)
            is out
        )
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    assert out.stride() == host.stride()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@requires_nvidia
@pytest.mark.flaggems_python
def test_small_addmm_writes_to_the_supplied_output_on_device():
    bias = torch.arange(4, dtype=torch.bfloat16)
    left = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    right = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    arguments = [tensor.to("flagos:0") for tensor in (bias, left, right)]
    out = torch.empty((4, 4), device="flagos:0", dtype=bias.dtype)
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        assert torch.addmm(*arguments, out=out) is out
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    torch.testing.assert_close(
        out.cpu(), torch.addmm(bias, left, right), rtol=0, atol=0
    )


@requires_nvidia
@pytest.mark.flaggems_python
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_nearest_exact_uses_device_indices(layout):
    host = torch.arange(30, dtype=torch.bfloat16).reshape(1, 2, 3, 5)
    host = host.contiguous(memory_format=layout)
    source = host.to("flagos:0")
    was_allowed = torch.ops.flagos.cpu_fallback_allowed()
    torch.ops.flagos.strict_host_routes(True)
    try:
        result = F.interpolate(source, scale_factor=(2.0, 1.5), mode="nearest-exact")
    finally:
        torch.ops.flagos.strict_host_routes(not was_allowed)
    expected = F.interpolate(host, scale_factor=(2.0, 1.5), mode="nearest-exact")
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
