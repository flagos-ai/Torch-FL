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
*implementation* rather than its dtype are marked per platform below, for one of
two reasons: a backend only matches the reference where it declines the operand
and reaches reference code -- a routing decision, not a dtype contract -- and a
backend that computes float32 in a reduced-precision format matches neither the
reference nor the precision the dtype names.
"""

import os

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

# The other divergence the suite found is float64, which no two backends serve
# the same way. GCU's FlagGems mean kernel does not compile for it and MUSA's
# muDNN Binary has no float64 MUL mode, so those two cases are marked; the
# float64 `matmul` case is asserted with a tolerance instead, because there the
# disagreement is arithmetic rather than a missing kernel.
_GCU = detect_platform() == "gcu"
_MUSA = detect_platform() == "musa"

# The float32 counterpart of the float64 matmul case below: an fp32 product must
# be computed in fp32, not in a 10-bit-mantissa format that is only *stored* in
# fp32. The two are ~900x apart, whichever op and shape is measured, so a
# dimensionless bound separates them without a per-shape constant -- measured on
# a MetaX C550 over 5 shapes x 20 seeds: 6.0e-08 to 2.9e-07 relative to an fp64
# CPU reference when the computation is fp32, 2.6e-04 to 3.3e-04 when the GEMM
# takes its TF32 kernel. 1e-5 is ~35x above the worst faithful measurement and
# ~26x below the best reduced-precision one.
FP32_MATMUL_RELATIVE_ERROR = 1e-5

# Ascend is the one backend that cannot meet that bound, and it does not claim
# to: the cube takes an HF32 (10-bit mantissa) input format by request --
# `get_cube_math_type(true)` in `csrc/aten/backends/ascend/matmul.cc`, the
# switch `scripts/tools/verify_flaggems_ascend.py` reads as "~5e-3 on a K=128
# matmul" and this file's TestMatrixOps sibling in `tests/integration/test_ops.py`
# carries as its `MM_RTOL`/`MM_ATOL` comment, flat at ~1.5e-4 relative. The
# switch the cases below are about -- `torch.backends.cuda.matmul.allow_tf32`
# and the `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` that seeds it -- has no effect on
# that path, so xfail rather than assert: a precision mode the backend chose is
# not a contract worth reddening its job over, and xfail turns into an XPASS the
# day Ascend starts computing fp32 in fp32.
_ASCEND = detect_platform() == "ascend"

_ASCEND_HF32_REASON = (
    "Ascend's cube takes the reduced-precision HF32 input format by request "
    "(get_cube_math_type(true) in csrc/aten/backends/ascend/matmul.cc), and "
    "the TF32 switches this case is about do not reach that path"
)


def _relative_error(got, reference):
    """Frobenius relative error of an fp32 device result over its fp64 CPU pair.

    Per-element relative error is not usable here, which is why this case does
    not use ``torch.testing.assert_close``. A random matmul's reference has
    entries near zero, so cancellation alone puts ~5.6e-01 between a *correct*
    fp32 product and its fp64 reference at 64x128 @ 128x32 -- larger than the
    reduced-precision gap the case is about, which would make the assertion
    vacuous. The norm ratio has neither that pole nor a shape-dependent scale:
    it is ~2e-07 for fp32 and ~3e-04 for a 10-bit input format at every shape
    measured, so one constant covers them all.
    """
    return float((got.double() - reference).norm() / reference.norm())


def _precision_failure(op, relative):
    """Name the numbers and both switches, so a failure reads on its own."""
    try:
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    except RuntimeError as error:
        # A process that set `fp32_precision` without `allow_tf32` is left in a
        # mixed-API state where the legacy accessor raises, and a failure message
        # must not turn a failed assertion into an error.
        allow_tf32 = f"unreadable ({error})"
    return (
        f"fp32 {op} differs from the fp64 CPU reference by {relative:.3e} "
        f"relative (limit {FP32_MATMUL_RELATIVE_ERROR:.0e}); a 10-bit-mantissa "
        f"matmul lands at ~3e-04 and fp32 at ~2e-07. "
        f"torch.backends.cuda.matmul.allow_tf32={allow_tf32}; "
        f"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="
        f"{os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')!r}"
    )


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
        if dtype is torch.float64 and _MUSA:
            pytest.xfail(
                "MUSA routes mul to muDNN's Binary, which reports "
                "NOT_SUPPORTED for a float64 MUL; its ADD mode takes float64, "
                "which is why the add case above passes"
            )
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

    def test_float64_matmul_matches_cpu_reference(self):
        """The product agrees with the CPU reference to float64 precision.

        Not bit-for-bit, which is what this used to assert. Ascend reaches the
        reference by construction -- aclnn takes matmul only up to float32, so
        `IsMatmulDtypeSupported` in `csrc/aten/backends/ascend/dtype_support.h`
        sends float64 through `at::matmul` on a CPU copy -- but off Ascend a
        real device float64 GEMM computes it (CUDA/DCU/MUSA all reach FlagGems'
        tiled Triton `mm`), and a different summation order over 4x4 inputs of
        order 1 lands 2 ulp away: the CI jobs measured 4.4e-16 absolute and
        4.1e-16 relative, identically on all three. That is reordering, not
        error, and requiring exactness here made the case a proxy for Ascend's
        routing decision. 1e-12 is ~2000x the observed deviation and still five
        orders of magnitude tighter than a silent float32 computation would be
        (~1e-7), so the claim this test is worth making -- the device answer is
        a float64 answer -- is kept.
        """
        a = torch.randn(4, 4, dtype=torch.float64)
        b = torch.randn(4, 4, dtype=torch.float64)
        result = torch.matmul(a.to(DEVICE), b.to(DEVICE)).cpu()
        torch.testing.assert_close(result, torch.matmul(a, b), rtol=1e-12, atol=1e-12)

    @pytest.mark.xfail(_ASCEND, reason=_ASCEND_HF32_REASON, strict=False)
    def test_float32_mm_is_computed_in_float32(self):
        """An fp32 `mm` is an fp32 computation, not a reduced-precision one.

        The float64 case above asks whether the device answer is a float64
        answer. This one asks the same of float32, and it is the case that
        notices when an accelerator replaces the fp32 arithmetic with a 10-bit
        input format: issue #253 is exactly that, one environment variable in
        the MetaX image away, silently turning every numeric assertion in the
        job into a TF32 assertion -- and issue #409 is the same loss chosen
        deliberately on Ascend. Shape is `tests/integration/test_ops.py`'s
        `TestMatrixOps::test_mm`, whose 1e-3/1e-2 bound is wide enough to admit
        either mode; this one is not.
        """
        torch.manual_seed(0)
        a = torch.randn(64, 128)
        b = torch.randn(128, 32)
        got = torch.mm(a.to(DEVICE), b.to(DEVICE)).cpu()
        relative = _relative_error(got, torch.mm(a.double(), b.double()))
        assert relative < FP32_MATMUL_RELATIVE_ERROR, _precision_failure("mm", relative)

    @pytest.mark.xfail(_ASCEND, reason=_ASCEND_HF32_REASON, strict=False)
    def test_float32_bmm_is_computed_in_float32(self):
        """The batched case, which is where attention loses the bits it needs.

        `bmm` is the op the issue's decoder models fail on: a forward pass that
        keeps every fp32 GEMM at ~2e-07 still diverges in `out.logits` at
        ~2e-04 once the attention `bmm` products run reduced-precision, and the
        accumulated error is the whole gap. Same bound and same shape as
        `tests/integration/test_ops.py`'s `test_bmm`, so the two files agree on
        what the batched product is measured against.
        """
        torch.manual_seed(0)
        a = torch.randn(4, 64, 128)
        b = torch.randn(4, 128, 32)
        got = torch.bmm(a.to(DEVICE), b.to(DEVICE)).cpu()
        relative = _relative_error(got, torch.bmm(a.double(), b.double()))
        assert relative < FP32_MATMUL_RELATIVE_ERROR, _precision_failure(
            "bmm", relative
        )

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
