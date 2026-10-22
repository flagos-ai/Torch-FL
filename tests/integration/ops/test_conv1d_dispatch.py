# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
conv1d / optional-bias boxing tests

Regression coverage for two boxing bugs exposed by Qwen3.5's linear-attention
depthwise causal conv1d:

1. Optional Tensor inputs were not boxed. gen_functional_pure only boxed plain
   at::Tensor args, silently skipping optional<Tensor> (conv/linear bias, loss
   weights, clamp min/max, batchnorm weight/bias, ...). When the optional tensor
   lived on flagos, the backend op got a mix of boxed (input/weight) and unboxed
   (bias) tensors -> "tensor does not have a device" / segfault. Now every
   functional kernel materializes optional tensors into a holder and boxes them.

2. Undefined tensors in *_backward output tuples crashed unboxing.
   convolution_backward with output_mask[2]==false (no bias grad) returns an
   undefined bias-grad tensor; SetTensorDevice dereferenced its null TensorImpl.
   It now skips undefined tensors.

Usage:
    pytest tests/integration/ops/test_conv1d_dispatch.py -v
"""

import pytest

import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


class TestConv1dForward:
    @pytest.mark.anyplatform
    def test_conv1d_with_bias(self):
        # optional bias present -> must be boxed alongside input/weight.
        x = torch.randn(1, 8, 32, device=DEVICE)
        w = torch.randn(16, 8, 4, device=DEVICE)
        b = torch.randn(16, device=DEVICE)
        y = F.conv1d(x, w, b, padding=3)
        assert tuple(y.shape) == (1, 16, 35)

    @pytest.mark.anyplatform
    def test_conv1d_depthwise_with_bias(self):
        C = 16
        x = torch.randn(1, C, 32, device=DEVICE)
        w = torch.randn(C, 1, 4, device=DEVICE)
        b = torch.randn(C, device=DEVICE)
        y = F.conv1d(x, w, b, padding=3, groups=C)
        assert tuple(y.shape) == (1, C, 35)

    @pytest.mark.anyplatform
    def test_conv1d_matches_cpu(self):
        torch.manual_seed(0)
        x = torch.randn(1, 8, 32)
        w = torch.randn(16, 8, 4)
        b = torch.randn(16)
        y_fl = F.conv1d(x.to(DEVICE), w.to(DEVICE), b.to(DEVICE), padding=3)
        y_cpu = F.conv1d(x, w, b, padding=3)
        # GPU conv accumulates differently from CPU; loose tolerance.
        torch.testing.assert_close(y_fl.cpu(), y_cpu, rtol=1e-3, atol=1e-2)


class TestConv1dBackward:
    @pytest.mark.anyplatform
    def test_conv1d_backward_with_bias(self):
        C = 32
        x = torch.randn(1, C, 32, device=DEVICE, requires_grad=True)
        w = torch.randn(C, 1, 4, device=DEVICE, requires_grad=True)
        b = torch.randn(C, device=DEVICE, requires_grad=True)
        F.conv1d(x, w, b, padding=3, groups=C).sum().backward()
        assert x.grad is not None and w.grad is not None and b.grad is not None

    @pytest.mark.anyplatform
    def test_conv1d_backward_no_bias(self):
        # bias=None -> convolution_backward returns an UNDEFINED bias grad
        # (output_mask[2]==false). Unboxing that undefined tensor used to crash.
        C = 32
        x = torch.randn(1, C, 32, device=DEVICE, requires_grad=True)
        w = torch.randn(C, 1, 4, device=DEVICE, requires_grad=True)
        F.conv1d(x, w, None, padding=3, groups=C).sum().backward()
        assert x.grad is not None and w.grad is not None

    @pytest.mark.anyplatform
    def test_conv1d_backward_large_depthwise_no_bias(self):
        # Qwen3.5 linear_attn conv1d: depthwise, huge channel count, no bias.
        C = 6144
        x = torch.randn(1, C, 32, device=DEVICE, requires_grad=True)
        w = torch.randn(C, 1, 4, device=DEVICE, requires_grad=True)
        F.conv1d(x, w, None, padding=3, groups=C)[:, :, :32].sum().backward()
        assert tuple(x.grad.shape) == (1, C, 32)


class TestOptionalBiasBoxing:
    """Other ops with optional Tensor args that were also unboxed."""

    @pytest.mark.anyplatform
    def test_clamp_tensor_min_max(self):
        # clamp.Tensor takes optional<Tensor> min/max.
        x = torch.randn(4, 8, device=DEVICE)
        lo = torch.full((4, 8), -0.5, device=DEVICE)
        hi = torch.full((4, 8), 0.5, device=DEVICE)
        y = torch.clamp(x, min=lo, max=hi)
        assert y.max().cpu().item() <= 0.5 + 1e-5
        assert y.min().cpu().item() >= -0.5 - 1e-5

    @pytest.mark.anyplatform
    def test_binary_cross_entropy_with_weight(self):
        # binary_cross_entropy takes optional<Tensor> weight.
        p = torch.rand(16, device=DEVICE).clamp(0.05, 0.95)
        t = torch.randint(0, 2, (16,), device=DEVICE).float()
        w = torch.rand(16, device=DEVICE)
        loss = F.binary_cross_entropy(p, t, weight=w)
        assert torch.isfinite(loss.cpu()).item()
