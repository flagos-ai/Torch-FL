# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_fl


@pytest.mark.kunlunxin
@pytest.mark.kunlunxin_xblaslt
def test_bf16_linear_keeps_vendor_scale_rounding():
    shape = (1, 16, 16)
    source = torch.zeros(shape, dtype=torch.bfloat16)
    source[..., 0] = 1.0078125 * 2**-14
    source[..., 1] = 2**-22
    weights = torch.zeros((16, 16), dtype=torch.bfloat16)
    weights[0, :2] = 0.5
    incoming = torch.zeros(shape, dtype=torch.bfloat16)
    incoming[..., 0] = 0.5

    # The unscaled sum is an exact BF16 midpoint. XPU's stock FP32 scale
    # places it below the midpoint.
    expected = torch.zeros_like(source)
    expected[..., 0] = 1.0078125 * 2**-15
    input_grad = torch.zeros_like(source)
    input_grad[..., :2] = 0.25
    weight_grad = torch.zeros_like(weights)
    weight_grad[0, 0] = 1.0078125 * 2**-11
    weight_grad[0, 1] = 2**-19

    device = "flagos:0"
    x = source.to(device).requires_grad_()
    w = weights.to(device).requires_grad_()
    with torch_fl.native_host.execution_scope(0):
        y = torch.nn.functional.linear(x, w)
        dx, dw = torch.autograd.grad(y, (x, w), incoming.to(device))
    torch.testing.assert_close(y.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(dx.cpu(), input_grad, atol=0, rtol=0)
    torch.testing.assert_close(dw.cpu(), weight_grad, atol=0, rtol=0)
