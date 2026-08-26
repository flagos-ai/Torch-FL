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

"""Software dispatch for low-precision matrix operators.

FP8/FP4 tensors carry scalar types that the vendor GEMM route does not accept.
The generated wrappers therefore select the device-side software implementation
for the supported matrix operators instead of sending the tensors to a vendor
library or the generic CPU fallback. Operators without a software implementation
remain gated explicitly.

Usage:
    pytest tests/integration/ops/test_soft_lowp_gate_dispatch.py -v
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

FP8_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
]
LOWP_DTYPES = FP8_DTYPES + [torch.float4_e2m1fn_x2]


class TestSoftLowpMatrixDispatch:
    """Supported matrix operators use the software low-precision path."""

    @pytest.mark.parametrize("dtype", LOWP_DTYPES)
    @pytest.mark.dcu
    def test_mm_executes_on_device(self, dtype):
        a = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        b = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        result = torch.mm(a, b)
        assert result.shape == (4, 4)
        assert result.dtype == torch.bfloat16
        assert result.device.type == "flagos"

    @pytest.mark.parametrize("dtype", LOWP_DTYPES)
    @pytest.mark.dcu
    def test_bmm_executes_on_device(self, dtype):
        a = torch.empty(2, 4, 4, device=DEVICE, dtype=dtype)
        b = torch.empty(2, 4, 4, device=DEVICE, dtype=dtype)
        result = torch.bmm(a, b)
        assert result.shape == (2, 4, 4)
        assert result.dtype == torch.bfloat16
        assert result.device.type == "flagos"

    @pytest.mark.parametrize("dtype", LOWP_DTYPES)
    @pytest.mark.dcu
    def test_addmm_executes_on_device(self, dtype):
        bias = torch.zeros(4, 4, device=DEVICE, dtype=torch.bfloat16)
        a = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        b = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        result = torch.addmm(bias, a, b)
        assert result.shape == (4, 4)
        assert result.dtype == torch.bfloat16
        assert result.device.type == "flagos"

    @pytest.mark.parametrize("dtype", LOWP_DTYPES)
    @pytest.mark.dcu
    def test_mm_out_executes_on_device(self, dtype):
        a = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        b = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        out = torch.empty(4, 4, device=DEVICE, dtype=torch.bfloat16)
        result = torch.mm(a, b, out=out)
        assert result is out
        assert result.dtype == torch.bfloat16
        assert result.device.type == "flagos"

    @pytest.mark.dcu
    def test_mm_dtype_and_dtype_out_execute_on_device(self):
        dtype = torch.float4_e2m1fn_x2
        a = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        b = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        result = torch.mm(a, b, torch.float32)
        assert result.shape == (4, 4)
        assert result.dtype == torch.float32

        out = torch.empty(4, 4, device=DEVICE, dtype=torch.float32)
        result = torch.mm(a, b, torch.float32, out=out)
        assert result is out
        assert result.dtype == torch.float32

    @pytest.mark.dcu
    def test_matmul_uses_the_mm_software_path(self):
        dtype = torch.float8_e4m3fn
        a = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        b = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        result = torch.matmul(a, b)
        assert result.shape == (4, 4)
        assert result.dtype == torch.bfloat16
        assert result.device.type == "flagos"

    @pytest.mark.dcu
    def test_scaled_mm_without_software_kernel_is_gated(self):
        dtype = torch.float8_e4m3fn
        a = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        b = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        scale = torch.ones((), device=DEVICE, dtype=torch.float32)
        with pytest.raises(
            RuntimeError, match="soft-lowp _scaled_mm requires a device kernel"
        ):
            torch._scaled_mm(a, b, scale, scale, out_dtype=torch.bfloat16)


class TestSoftLowpMatrixNumerics:
    """The composite FP4 decoder preserves the packed nibble values."""

    @pytest.mark.dcu
    def test_fp4_nibble_decode_mm(self):
        # 0x21 decodes to (low=0.5, high=1.0); 0x11 decodes to (0.5, 0.5).
        raw_a = torch.tensor([0x21], dtype=torch.uint8, device=DEVICE)
        raw_b = torch.tensor([0x11], dtype=torch.uint8, device=DEVICE)
        a = raw_a.view(torch.float4_e2m1fn_x2).reshape(1, 1)
        b = raw_b.view(torch.float4_e2m1fn_x2).reshape(1, 1)

        result = torch.mm(a, b)
        torch.testing.assert_close(
            result.cpu(), torch.tensor([[0.75]], dtype=torch.bfloat16), atol=0, rtol=0
        )


class TestSoftLowpGateLeavesFloatPathsAlone:
    """The software path does not disturb ordinary vendor GEMM dtypes."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
    )
    @pytest.mark.anyplatform
    def test_mm_still_runs(self, dtype):
        torch.manual_seed(0)
        a = torch.randn(32, 48, device=DEVICE, dtype=dtype)
        b = torch.randn(48, 16, device=DEVICE, dtype=dtype)
        out = torch.mm(a, b)
        assert out.shape == (32, 16)
        assert out.dtype == dtype
        assert out.device.type == "flagos"

        expected = torch.mm(a.cpu().float(), b.cpu().float())
        # fp16/bf16 accumulate differently on device than a float32 CPU reference;
        # the tolerance follows the mantissa width rather than the op.
        tol = {torch.float32: 1e-3, torch.float64: 1e-5}.get(dtype, 0.5)
        torch.testing.assert_close(
            out.cpu().float(), expected, rtol=tol, atol=tol, check_dtype=False
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_bmm_and_addmm_still_run(self, dtype):
        torch.manual_seed(0)
        a = torch.randn(2, 8, 12, device=DEVICE, dtype=dtype)
        b = torch.randn(2, 12, 4, device=DEVICE, dtype=dtype)
        assert torch.bmm(a, b).shape == (2, 8, 4)

        bias = torch.zeros(8, 4, device=DEVICE, dtype=dtype)
        assert torch.addmm(bias, a[0], b[0]).shape == (8, 4)
