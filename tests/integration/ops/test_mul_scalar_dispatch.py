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
mul.Scalar dispatch tests

mul.Scalar is a CompositeImplicitAutograd op — PyTorch decomposes it to
mul.Tensor before reaching PrivateUse1 dispatch. We only verify correctness.

Usage:
    pytest tests/integration/ops/test_mul_scalar_dispatch.py -v
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


class TestMulScalarCorrectness:
    """torch.mul(tensor, scalar) correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_mul_scalar_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.mul(a, 3.0)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_mul_scalar_correctness(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.mul(a, 4.0)
        ref = a.cpu() * 4.0
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_mul_scalar_dtype(self, dtype):
        torch.manual_seed(3)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        out = torch.mul(a, 2.0)
        ref = a.cpu().float() * 2.0
        torch.testing.assert_close(out.cpu().float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.kunlunxin
@pytest.mark.parametrize("scale", [0.08838834764831845, -0.18731])
def test_kunlunxin_bf16_mul_retains_fp32_scalar(scale):
    source = torch.arange(-512, 512, dtype=torch.float32).div(8).bfloat16()
    expected = (source.float() * scale).bfloat16()
    result = torch.mul(source.to(DEVICE), scale)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
