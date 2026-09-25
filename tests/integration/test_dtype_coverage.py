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

"""Comprehensive dtype support validation across operator categories.

This test suite verifies that torch-fl backends correctly handle all major
PyTorch dtypes across representative operator categories: creation, unary,
binary, reduction, and indexing operations.

Most of it is portable: the assertions are about the dtype a call returns, which
every backend is expected to preserve. The handful that compare against the CPU
*implementation* rather than its dtype are marked per platform below, because a
backend only matches the reference where it declines the operand and reaches
reference code -- a routing decision, not a dtype contract.
"""

import pytest
import torch
import torch_fl  # noqa: F401
from platform_support import detect_platform


DEVICE = "flagos:0"

# Core floating-point dtypes for numerical computation
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]

# Integer dtypes for indexing, counting, and discrete operations
INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]

# Boolean dtype for masks and logical operations
BOOL_DTYPE = [torch.bool]

# All dtypes combined for exhaustive checks
ALL_DTYPES = FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPE

# `neg` is the one op whose *reference* behaviour, rather than its dtype, the
# suite pins, and matching that behaviour is a routing decision the backends do
# not agree on.
#
# Ascend is the one that agrees. `FlagGemsRejectsOpDtype` in
# `csrc/aten/common.cc` routes bool `neg` away from FlagGems -- whose
# `flag_gems/ops/neg.py` is a bare pointwise `-x` that BiShengIR refuses to
# verify over `i1` -- and into the vendor slot, whose aclnn kernel falls back to
# `at::neg` on a CPU copy for the dtypes it does not cover. That fallback is the
# reference call, so it raises the reference error.
#
# Elsewhere the operand reaches FlagGems' kernel and comes back as a bool tensor
# (MetaX), or reaches a vendor kernel that rejects it in its own words (GCU's
# `topsatenNeg failed: NOT_SUPPORT`, which does not name the dtype and so does
# not match the reference message either). Both are backend gaps worth closing,
# not contracts worth asserting here, so the reference-error case is xfailed off
# Ascend instead of reddening the other platform jobs over a FlagGems
# divergence rather than a torch-fl one.
_REFERENCE_NEG_IS_ASCEND_ONLY = detect_platform() != "ascend"
_GCU = detect_platform() == "gcu"


class TestFactoryDtypeSupport:
    """Tensor creation operations must preserve requested dtype."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_empty_preserves_dtype(self, dtype):
        result = torch.empty(4, 4, device=DEVICE, dtype=dtype)
        assert result.dtype == dtype
        assert result.device.type == "flagos"

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_zeros_preserves_dtype(self, dtype):
        result = torch.zeros(4, 4, device=DEVICE, dtype=dtype)
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), torch.zeros(4, 4, dtype=dtype))

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_ones_preserves_dtype(self, dtype):
        result = torch.ones(4, 4, device=DEVICE, dtype=dtype)
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), torch.ones(4, 4, dtype=dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_randn_preserves_float_dtype(self, dtype):
        torch.manual_seed(42)
        result = torch.randn(16, device=DEVICE, dtype=dtype)
        assert result.dtype == dtype
        assert result.numel() == 16


class TestUnaryDtypeSupport:
    """Unary operations must handle all appropriate dtypes."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_neg_preserves_dtype(self, dtype):
        x = torch.ones(8, device=DEVICE, dtype=dtype)
        result = torch.neg(x)
        assert result.dtype == dtype

    @pytest.mark.xfail(
        _GCU,
        reason="GCU routes neg to topsatenNeg, which admits uint8 through the "
        "dtype gate and then rejects it with NOT_SUPPORT instead of computing "
        "the reference wraparound",
        strict=False,
    )
    def test_neg_uint8_matches_cpu_wraparound(self):
        values = torch.tensor([0, 1, 2, 200], dtype=torch.uint8)
        result = torch.neg(values.to(DEVICE)).cpu()
        torch.testing.assert_close(result, torch.neg(values))

    @pytest.mark.xfail(
        _REFERENCE_NEG_IS_ASCEND_ONLY,
        reason="bool neg only raises the reference error on a backend that "
        "declines the operand; elsewhere FlagGems' pointwise neg returns a bool "
        "tensor, and topsatenNeg rejects it without naming the dtype",
        strict=False,
    )
    def test_neg_bool_matches_cpu_error(self):
        with pytest.raises(RuntimeError, match="bool"):
            torch.neg(torch.ones(8, device=DEVICE, dtype=torch.bool))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_abs_preserves_dtype(self, dtype):
        if dtype.is_floating_point:
            x = torch.randn(8, device=DEVICE, dtype=dtype)
        else:
            x = torch.tensor([-1, -2, 3, 4], device=DEVICE, dtype=dtype)
        result = torch.abs(x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sin_preserves_float_dtype(self, dtype):
        x = torch.randn(8, device=DEVICE, dtype=dtype)
        result = torch.sin(x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_exp_preserves_float_dtype(self, dtype):
        x = torch.randn(8, device=DEVICE, dtype=dtype)
        result = torch.exp(x)
        assert result.dtype == dtype


class TestBinaryDtypeSupport:
    """Binary operations must handle dtype promotion correctly."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_add_same_dtype(self, dtype):
        a = torch.ones(4, device=DEVICE, dtype=dtype)
        b = torch.ones(4, device=DEVICE, dtype=dtype)
        result = torch.add(a, b)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_mul_same_dtype(self, dtype):
        a = torch.ones(4, device=DEVICE, dtype=dtype)
        b = torch.ones(4, device=DEVICE, dtype=dtype) * 2
        result = torch.mul(a, b)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matmul_preserves_float_dtype(self, dtype):
        a = torch.randn(4, 4, device=DEVICE, dtype=dtype)
        b = torch.randn(4, 4, device=DEVICE, dtype=dtype)
        result = torch.matmul(a, b)
        assert result.dtype == dtype

    def test_float64_matmul_matches_cpu_fallback(self):
        a = torch.randn(4, 4, dtype=torch.float64)
        b = torch.randn(4, 4, dtype=torch.float64)
        result = torch.matmul(a.to(DEVICE), b.to(DEVICE)).cpu()
        torch.testing.assert_close(result, torch.matmul(a, b), rtol=0, atol=0)

    def test_mixed_float_promotion(self):
        """Tensor-tensor arithmetic follows PyTorch promotion rules."""
        a = torch.ones(4, device=DEVICE, dtype=torch.float16)
        b = torch.ones(4, device=DEVICE, dtype=torch.float32)
        result = torch.add(a, b)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result.cpu(), torch.add(a.cpu(), b.cpu()))

    def test_integer_promotion(self):
        a = torch.ones(4, device=DEVICE, dtype=torch.int16)
        b = torch.ones(4, device=DEVICE, dtype=torch.int64)
        assert torch.add(a, b).dtype == torch.int64

    def test_integer_true_division_promotes_to_default_float(self):
        a = torch.full((4,), 3, device=DEVICE, dtype=torch.int16)
        b = torch.full((4,), 2, device=DEVICE, dtype=torch.int64)
        result = torch.div(a, b)
        assert result.dtype == torch.float32
        torch.testing.assert_close(result.cpu(), torch.div(a.cpu(), b.cpu()))


class TestReductionDtypeSupport:
    """Reduction operations may change dtype for numerical stability."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sum_float_dtype(self, dtype):
        x = torch.randn(8, 8, device=DEVICE, dtype=dtype)
        result = torch.sum(x)
        # Sum may upcast for accumulation, accept that
        assert result.dtype in (dtype, torch.float32, torch.float64)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_sum_int_dtype(self, dtype):
        x = torch.ones(8, device=DEVICE, dtype=dtype)
        result = torch.sum(x)
        # Integer sum typically promotes to int64
        assert result.dtype in (dtype, torch.int64)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_returns_float(self, dtype):
        if dtype is torch.float64 and _GCU:
            pytest.xfail(
                "FlagGems' GCU mean kernel does not compile for float64: the "
                "Enflame Triton pipeline fails in PassManager execution"
            )
        x = torch.randn(8, 8, device=DEVICE, dtype=dtype)
        result = torch.mean(x)
        # Mean stays floating-point
        assert result.dtype.is_floating_point

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_max_preserves_dtype(self, dtype):
        if dtype.is_floating_point:
            x = torch.randn(8, device=DEVICE, dtype=dtype)
        else:
            x = torch.randint(0, 10, (8,), device=DEVICE, dtype=dtype)
        result = torch.max(x)
        assert result.dtype == dtype


class TestCopyDtypeSupport:
    """Copy operations must preserve or correctly convert dtype."""

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_clone_preserves_dtype(self, dtype):
        x = torch.ones(4, device=DEVICE, dtype=dtype)
        result = torch.clone(x)
        assert result.dtype == dtype

    @pytest.mark.parametrize("src_dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("dst_dtype", FLOAT_DTYPES)
    def test_to_dtype_conversion(self, src_dtype, dst_dtype):
        x = torch.ones(4, device=DEVICE, dtype=src_dtype)
        result = x.to(dtype=dst_dtype)
        assert result.dtype == dst_dtype

    def test_float64_cpu_device_roundtrip(self):
        cpu = torch.tensor([1.0 + 2**-40, 1e300], dtype=torch.float64)
        device = cpu.to(DEVICE)
        assert device.dtype == torch.float64
        torch.testing.assert_close(device.cpu(), cpu, rtol=0, atol=0)

    def test_float32_to_float64_on_device(self):
        result = torch.ones(4, device=DEVICE, dtype=torch.float32).to(torch.float64)
        assert result.dtype == torch.float64

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cpu_device_transfer_preserves_dtype(self, dtype):
        cpu_tensor = torch.ones(4, dtype=dtype)
        device_tensor = cpu_tensor.to(DEVICE)
        assert device_tensor.dtype == dtype
        back_to_cpu = device_tensor.cpu()
        assert back_to_cpu.dtype == dtype


class TestIndexingDtypeSupport:
    """Indexing operations must preserve source dtype."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_slice_preserves_dtype(self, dtype):
        x = torch.ones(8, 8, device=DEVICE, dtype=dtype)
        result = x[2:6, 1:5]
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_index_select_preserves_dtype(self, dtype):
        x = torch.randn(8, 8, device=DEVICE, dtype=dtype)
        indices = torch.tensor([0, 2, 4], device=DEVICE, dtype=torch.int64)
        result = torch.index_select(x, 0, indices)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_masked_select_preserves_dtype(self, dtype):
        if dtype.is_floating_point:
            x = torch.randn(8, device=DEVICE, dtype=dtype)
        else:
            x = torch.randint(0, 10, (8,), device=DEVICE, dtype=dtype)
        mask = torch.tensor(
            [True, False, True, False, True, False, True, False], device=DEVICE
        )
        result = torch.masked_select(x, mask)
        assert result.dtype == dtype


class TestComparisonDtypeSupport:
    """Comparison operations always return bool regardless of input dtype."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_eq_returns_bool(self, dtype):
        a = torch.ones(4, device=DEVICE, dtype=dtype)
        b = torch.ones(4, device=DEVICE, dtype=dtype)
        result = torch.eq(a, b)
        assert result.dtype == torch.bool

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_gt_returns_bool(self, dtype):
        if dtype.is_floating_point:
            a = torch.randn(4, device=DEVICE, dtype=dtype)
            b = torch.randn(4, device=DEVICE, dtype=dtype)
        else:
            a = torch.randint(0, 10, (4,), device=DEVICE, dtype=dtype)
            b = torch.randint(0, 10, (4,), device=DEVICE, dtype=dtype)
        result = torch.gt(a, b)
        assert result.dtype == torch.bool
