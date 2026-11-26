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
div.Scalar dispatch tests

div.Scalar is a CompositeImplicitAutograd op — PyTorch decomposes it to
mul.Tensor (multiply by reciprocal) before reaching PrivateUse1 dispatch.
We only verify correctness.

Usage:
    pytest tests/integration/ops/test_div_scalar_dispatch.py -v
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


class TestDivScalarCorrectness:
    """torch.div(tensor, scalar) correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_div_scalar_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.div(a, 3.0)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_div_scalar_correctness(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.div(a, 4.0)
        ref = a.cpu() / 4.0
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_div_scalar_dtype(self, dtype):
        torch.manual_seed(3)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        out = torch.div(a, 2.0)
        ref = a.cpu().float() / 2.0
        torch.testing.assert_close(out.cpu().float(), ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.nvidia
    @pytest.mark.flaggems_python
    @pytest.mark.parametrize("operand", [127.5, -127.5, 7.0])
    @pytest.mark.parametrize("overload", ["Scalar", "Tensor"])
    def test_fp32_host_scalar_matches_cuda_bits(self, operand, overload):
        if not torch.version.cuda or not torch.cuda.is_available():
            pytest.skip("This parity test requires CUDA PyTorch")
        if overload == "Tensor":
            operand = torch.tensor(operand, dtype=torch.float64)
        cpu = torch.arange(-255, 257, dtype=torch.float32).reshape(16, 32).T
        divide = getattr(torch.ops.aten.div, overload)
        divide_inplace = getattr(torch.ops.aten.div_, overload)
        cuda = cpu.to("cuda:0")
        flagos = cpu.to(DEVICE)
        expected = divide(cuda, operand).cpu()
        actual = divide(flagos, operand).cpu()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        assert divide_inplace(flagos, operand) is flagos
        divide_inplace(cuda, operand)
        assert torch.equal(flagos.cpu().view(torch.int32), cuda.cpu().view(torch.int32))
