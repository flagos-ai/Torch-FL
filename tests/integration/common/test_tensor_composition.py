# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


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


@pytest.mark.nvidia
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
