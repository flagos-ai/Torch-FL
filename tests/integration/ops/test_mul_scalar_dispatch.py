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

    @pytest.mark.cuda
    def test_mul_scalar_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.mul(a_cuda, 5.0)
        a = a_cuda.to(DEVICE)
        out = torch.mul(a, 5.0)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_mul_scalar_dtype(self, dtype):
        torch.manual_seed(3)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        out = torch.mul(a, 2.0)
        ref = a.cpu().float() * 2.0
        torch.testing.assert_close(out.cpu().float(), ref, rtol=1e-2, atol=1e-2)
