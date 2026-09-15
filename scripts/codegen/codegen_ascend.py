#!/usr/bin/env python3
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
Codegen for torch_fl Ascend (aclnn) operators.

Unlike the CUDA codegen (scripts/codegen/codegen_ops.py), which emits a one-line
`at::op(args)` boxing body and relies on the dedicated CUDA dispatch key,
Ascend has no independent key to box into (torch_npu shares PrivateUse1 with
flagos). So every Ascend kernel must call CANN aclnn (libopapi.so) directly.

Because the aclnn call shape is uniform per *category*, this generator is
category-driven: a mapping table classifies each op, and a per-category
template emits the argument marshalling + output allocation + EXEC_ASCEND_CMD.

Categories implemented:
    unary               1 Tensor in -> 1 Tensor out (same shape/dtype)
                          aclnn<Name>(self, out)
    binary              2 Tensor in, broadcast, preserve dtype
                          aclnn<Name>(self, other, out)
    binary_alpha        binary + trailing Scalar alpha (sub.Tensor)
                          aclnn<Name>(self, other, alpha, out)
    binary_cmp          2 Tensor in, broadcast, bool out (eq/lt/logical_and..)
                          aclnn<Name>(self, other, out)
    binary_scalar_alpha Tensor + Scalar other + Scalar alpha (add/sub.Scalar)
                          aclnn<Name>s(self, other, alpha, out)
    binary_scalar_cmp   Tensor + Scalar, bool out (eq/lt.Scalar..)
                          aclnn<Name>(self, other, out)

Reuses:
  - scripts/codegen/codegen_ops.py:schema_to_cpp_name  (symbol names must match the
    dispatcher declarations already emitted into generated/ops.h)
  - the dispatchers/DECLARE_DISPATCHER already present in generated/ops.h
    (we only add the Backend::kAscend slot; we declare nothing new)

Generates:
  - csrc/aten/backends/ascend/generated/ascend_kernels.cc
  - appends newly-covered ops to torch_fl/configs/backends_ascend.conf

Validation:
  - each derived aclnn<Name>/<Name>GetWorkspaceSize symbol must exist in
    libopapi.so, else the op is skipped with a warning.
  - handwritten ops (SKIP set) are never re-emitted (would double-register
    kAscend and crash at import).

Known limitations (documented, acceptable for the current op set):
  - binary/binary_alpha use PyTorch's result-type promotion before marshalling
    inputs. True division additionally promotes integral inputs to the default
    floating dtype, matching aten semantics.
  - No category may pass a by-value `float`/`double` argument to aclnn.
    EXEC_ASCEND_CMD calls the aclnn entry through a variadic `int(*)(...)`
    pointer, and on AArch64 a by-value float goes through varargs promotion
    (float->double, wrong register class) so aclnn reads a garbage value.
    Scalars must be marshalled as `aclScalar*`; int64/bool pass through fine.
    (This is why smooth_l1_loss, which has a `float beta`, is left long-tail.)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Reuse the authoritative symbol-naming from the CUDA codegen so the emitted
# REGISTER_IMPL_TO_DISPATCHER(FnType, dispatcher, ...) matches the
# DECLARE_DISPATCHER in generated/ops.h exactly (else the build won't link).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegen_ops import schema_to_cpp_name  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
OUT_CC = REPO / "csrc/aten/backends/ascend/generated/ascend_kernels.cc"
OUT_INC = REPO / "csrc/aten/backends/ascend/generated/ascend_register.inc"
REGISTER_INC = REPO / "csrc/aten/generated/register.inc"
# Handwritten kernels also claim Backend::kAscend slots, so the coverage set is
# the union of what this script generates and what those files register. Scanned
# rather than restated: a new handwritten kernel must not silently drop out of
# the registration list (that would route it to cpu_fallback despite existing).
HANDWRITTEN_DIRS = (
    REPO / "csrc/aten/backends/ascend",
    REPO / "csrc/aten",
)

# --------------------------------------------------------------------------
# Op registry: schema op name -> (category, aclnn-name override or None).
#
# Default aclnn name = "aclnn" + PascalCase(op base). A non-None override
# replaces that stem for irregular spellings (e.g. eq.Tensor -> aclnnEqTensor).
#
# Ops already handwritten in csrc/aten/backends/ascend/*.cc must NOT appear
# here (they already own the kAscend slot; double-register crashes at import).
# --------------------------------------------------------------------------
OPS = {
    # ---- unary: aclnn<Name>(self, out), out = self.shape/dtype ----
    "sqrt": ("unary", None),
    "exp": ("unary", None),
    "tanh": ("unary", None),
    "sigmoid": ("unary", None),
    "reciprocal": ("unary", None),
    "log": ("unary", None),
    "floor": ("unary", None),
    "ceil": ("unary", None),
    "erf": ("unary", None),
    "erfc": ("unary", None),
    "expm1": ("unary", None),
    "log2": ("unary", None),
    "log10": ("unary", None),
    "log1p": ("unary", None),
    "round": ("unary", None),
    "trunc": ("unary", None),
    "frac": ("unary", None),
    "sign": ("unary", None),
    "relu": ("unary", None),
    "cosh": ("unary", None),
    "sinh": ("unary", None),
    "asin": ("unary", None),
    "atan": ("unary", None),
    "tan": ("unary", None),  # needed by cauchy_'s inverse-transform sampling
    "asinh": ("unary", None),
    "acosh": ("unary", None),
    "atanh": ("unary", None),
    "logical_not": ("unary", None),
    "bitwise_not": ("unary", None),
    # migrated from handwritten seed kernels (bodies were identical to T_UNARY):
    "abs": ("unary", None),
    "acos": ("unary", None),
    "cos": ("unary", None),
    "sin": ("unary", None),
    "neg": ("unary", None),
    "rsqrt": ("unary", None),
    "silu": ("unary", None),
    # ---- binary: aclnn<Name>(self, other, out), broadcast, preserve dtype ----
    "div.Tensor": ("binary", "Div"),
    # migrated from handwritten seeds:
    "mul.Tensor": ("binary", "Mul"),
    "bitwise_and.Tensor": ("binary", "BitwiseAndTensor"),
    "pow.Tensor_Tensor": ("binary", "PowTensorTensor"),
    "atan2": ("binary", None),
    "maximum": ("binary", None),
    "minimum": ("binary", None),
    "bitwise_or.Tensor": ("binary", "BitwiseOrTensor"),
    "bitwise_xor.Tensor": ("binary", "BitwiseXorTensor"),
    # ---- binary_alpha: aclnn<Name>(self, other, alpha, out) ----
    "sub.Tensor": ("binary_alpha", "Sub"),
    "add.Tensor": ("binary_alpha", "Add"),  # migrated seed
    # ---- binary_scalar: aclnn<Name>(self, scalar, out), no alpha (migrated seeds) ----
    "mul.Scalar": ("binary_scalar", "Muls"),
    "div.Scalar": ("binary_scalar", "Divs"),
    # ---- binary_cmp: bool out ----
    "eq.Tensor": ("binary_cmp", "EqTensor"),
    "ne.Tensor": ("binary_cmp", "NeTensor"),
    "gt.Tensor": ("binary_cmp", "GtTensor"),
    "lt.Tensor": ("binary_cmp", "LtTensor"),
    "ge.Tensor": ("binary_cmp", "GeTensor"),
    "logical_and": ("binary_cmp", None),
    "logical_or": ("binary_cmp", None),
    # ---- binary_scalar_alpha: aclnn<Name>s(self, other, alpha, out) ----
    "add.Scalar": ("binary_scalar_alpha", "Adds"),
    "sub.Scalar": ("binary_scalar_alpha", "Subs"),
    "rsub.Scalar": ("binary_scalar_alpha", "Rsubs"),
    # ---- binary_scalar_cmp: bool out, aclnn<Name>(self, other, out) ----
    "eq.Scalar": ("binary_scalar_cmp", "EqScalar"),
    "ne.Scalar": ("binary_scalar_cmp", "NeScalar"),
    "gt.Scalar": ("binary_scalar_cmp", "GtScalar"),
    "lt.Scalar": ("binary_scalar_cmp", "LtScalar"),
    "ge.Scalar": ("binary_scalar_cmp", "GeScalar"),
    "le.Scalar": ("binary_scalar_cmp", "LeScalar"),
    # ---- reduce_dims: (Tensor, IntArrayRef dim, bool keepdim), same dtype ----
    "amax": ("reduce_dims", None),
    "amin": ("reduce_dims", None),
    # ---- reduce_dim_bool: (Tensor, int64_t dim, bool keepdim), bool out ----
    "any.dim": ("reduce_dim_bool", "Any"),
    # ---- cumsum: (Tensor, int64_t dim, optional<ScalarType> dtype) ----
    "cumsum": ("cumsum", None),
    # ---- cumprod: like cumsum but aclnn takes dim as aclScalar*, not int64_t ----
    "cumprod": ("cumprod", None),
    # ---- unary_bool: (Tensor) -> bool out ----
    "isinf": ("unary_bool", "IsInf"),
    # ---- unary_scalar: (Tensor, Scalar) -> same shape ----
    "leaky_relu": ("unary_scalar", None),
    "clamp_min": ("unary_scalar", None),
    "clamp_max": ("unary_scalar", None),
    "fmod.Scalar": ("unary_scalar", "FmodScalar"),
    "pow.Tensor_Scalar": ("unary_scalar", "PowTensorScalar"),  # migrated seed
    # ---- unary_two_scalar: (Tensor, Scalar, Scalar) -> same shape ----
    "softplus": ("unary_two_scalar", None),
    "threshold": ("unary_two_scalar", None),
    # ---- unary_int: (Tensor, int64_t) -> same shape ----
    "tril": ("unary_int", None),
    "triu": ("unary_int", None),
    # ---- unary_dims: (Tensor, IntArrayRef) -> same shape ----
    "flip": ("unary_dims", None),
    # ---- addcmul: (self, t1, t2, Scalar value) -> broadcast ----
    "addcmul": ("addcmul", None),
    "addcdiv": ("addcmul", None),
    # ---- binary (tensor-tensor, preserve dtype) additions ----
    "fmod.Tensor": ("binary", "FmodTensor"),
    "floor_divide": ("binary", None),
    "logical_xor": ("binary_cmp", None),
    # ---- act_backward: (grad_output, output) -> grad_input ----
    "tanh_backward": ("act_backward", None),
    "sigmoid_backward": ("act_backward", None),
    # ---- threshold_backward: (grad_output, self, Scalar threshold) ----
    "threshold_backward": ("threshold_backward", None),
    # ---- activation backward: forward was already routed, backward was not,
    # so inference worked and .backward() died at runtime. See the template
    # block for the two ops whose CANN name is NOT the derived PascalCase one.
    "hardshrink_backward": ("grad_scalar_backward", None),
    "softshrink_backward": ("grad_scalar_backward", None),
    "softplus_backward": ("grad_two_scalar_backward", None),
    "hardtanh_backward": ("grad_two_scalar_backward", None),
    "leaky_relu_backward": ("leaky_relu_backward", None),
    "elu_backward": ("elu_backward", None),
    # CANN spells these two differently from the derived name:
    #   native_dropout_backward -> aclnnDropoutBackward   (not NativeDropout*)
    #   _prelu_kernel_backward  -> aclnnPreluBackward     (not PreluKernel*)
    # composed from routed ops, not aclnnDropoutBackward (mask format
    # mismatch -- see the template). Listed in NO_ACLNN_CATEGORIES.
    "native_dropout_backward": ("dropout_backward", None),
    "_prelu_kernel": ("binary", "Prelu"),
    "_prelu_kernel_backward": ("prelu_backward", "PreluBackward"),
    # ---- pow_scalar_tensor: (Scalar self, Tensor exponent) ----
    "pow.Scalar": ("pow_scalar_tensor", "PowScalarTensor"),
    # ---- reduce_max_dim: (Tensor, int64_t dim, bool keepdim) -> (values, indices) ----
    "max.dim": ("reduce_max_dim", "MaxDim"),
    "min.dim": ("reduce_max_dim", "MinDim"),
    # ---- more unary_scalar activations ----
    "celu": ("unary_scalar", None),
    "softshrink": ("unary_scalar", None),
    "hardshrink": ("unary_scalar", None),
    # ---- more unary_two_scalar (min/max clip) ----
    "hardtanh": ("unary_two_scalar", None),
    # ---- elu: (Tensor, alpha, scale, input_scale) ----
    "elu": ("elu", None),
    # ---- loss: (self, target, reduction) -> scalar/elementwise ----
    "mse_loss": ("loss", "MseLoss"),
    # smooth_l1_loss/l1_loss(*): smooth_l1 needs a by-value `float beta`. The
    # EXEC_ASCEND_CMD macro calls the aclnn entry through a variadic
    # `int(*)(...)` pointer; on AArch64 a by-value float passed through varargs
    # is promoted to double and lands in the wrong register class, so aclnn
    # reads beta as 0 (result collapses to pure L1). Left long-tail until the
    # macro grows a typed-call path for by-value floats.
    # ---- cummax/cummin: (Tensor, dim) -> tuple(values, indices) ----
    "cummax": ("cummax_cummin", "Cummax"),
    "cummin": ("cummax_cummin", "Cummin"),
    # ---- aminmax: (Tensor, optional dim, keepdim) -> tuple(min, max) ----
    "aminmax": ("aminmax", "Aminmax"),
    # ---- prod: (Tensor, optional dtype) -> scalar ----
    "prod": ("prod", "Prod"),
    # ---- gemm family (cubeMathType) ----
    # mm/bmm: functional + .out variants. aclnn mm=aclnnMm, bmm=aclnnBatchMatMul.
    #   The .out variant is a generic codegen capability (any op whose .out kernel
    #   just writes into a caller-shaped out& can reuse T_MATMUL_OUT-style pairs).
    "mm": ("matmul", "Mm"),
    "mm.out": ("matmul_out", "Mm"),
    "bmm": ("matmul", "BatchMatMul"),
    "bmm.out": ("matmul_out", "BatchMatMul"),
    # cat: TensorList concat (aclCreateTensorList), functional + .out variants.
    "cat": ("cat", "Cat"),
    "cat.out": ("cat_out", "Cat"),
    # stack: TensorList concat along a NEW dim (aclCreateTensorList).
    "stack": ("stack", "Stack"),
    # factory ops: at::empty + device-side zero_/fill_ (no direct aclnn call).
    "zeros": ("zeros", None),
    "ones": ("ones", None),
    "scalar_tensor": ("scalar_tensor", None),
    "ones_like": ("ones_like", None),
    "zeros_like": ("zeros_like", None),
    "empty_like": ("empty_like", None),
    "full": ("full", None),
    "full_like": ("full_like", None),
    "new_ones": ("new_ones", None),
    "addmm": ("gemm_addmm", "Addmm"),
    "baddbmm": ("gemm_baddbmm", "Baddbmm"),
    "mv": ("mv", "Mv"),
    "dot": ("dot", "Dot"),
    # addmv: mat(n,m) x vec(m) -> (n,); aclnn arg order is (self,mat,vec,ALPHA,BETA).
    "addmv": ("gemm_addmv", "Addmv"),
    # addr: outer(vec1(n), vec2(m)) -> (n,m); no cubeMathType.
    "addr": ("gemm_addr", "Addr"),
    # NOTE addbmm left out: hf32 cube accumulation over the batch dim inflates
    #   rel-err to ~1e-2 (single addmm is ~1e-4).
    # ---- CNN training closure: pool bwd + batch norm ----
    "max_pool2d_with_indices_backward": (
        "max_pool2d_indices_backward",
        "MaxPool2dWithIndicesBackward",
    ),
    "native_batch_norm": ("native_batch_norm", "BatchNorm"),
    "native_batch_norm_backward": ("native_batch_norm_backward", "BatchNormBackward"),
    "avg_pool2d_backward": ("avg_pool2d_backward", "AvgPool2dBackward"),
    "_adaptive_avg_pool2d_backward": (
        "adaptive_avg_pool2d_backward",
        "AdaptiveAvgPool2dBackward",
    ),
    "native_layer_norm_backward": ("native_layer_norm_backward", "LayerNormBackward"),
    "native_group_norm_backward": ("native_group_norm_backward", "GroupNormBackward"),
    # ---- Transformer indexing / masking (aclnn masked_fill is INPLACE-only) ----
    "masked_fill.Scalar": ("masked_fill_scalar", "InplaceMaskedFillScalar"),
    "masked_fill.Tensor": ("masked_fill_tensor", "InplaceMaskedFillTensor"),
    "gather": ("gather", "Gather"),
    "index_select": ("index_select", "IndexSelect"),
    # ---- in-place zero/fill (aclnn Inplace* ops); device-side, no h2d.
    #   Factory ops (zeros/ones_like/new_ones/scalar_tensor) call these. ----
    "zero_": ("inplace_zero", "InplaceZero"),
    "fill_.Scalar": ("inplace_fill_scalar", "InplaceFillScalar"),
    "fill_.Tensor": ("inplace_fill_tensor", "InplaceFillTensor"),
    "add_.Tensor": ("inplace_add_tensor", "InplaceAdd"),
    "add_.Scalar": ("inplace_add_scalar", "InplaceAdds"),
    "mul_.Tensor": ("inplace_mul_tensor", "InplaceMul"),
    "mul_.Scalar": ("inplace_mul_scalar", "InplaceMuls"),
    "div_.Tensor": ("inplace_div_tensor", "InplaceDiv"),
    # bitwise_{and,or,xor}_.Tensor have the same (self&, other) shape as
    # mul_.Tensor; reuse that category with the Inplace* aclnn override.
    # torch's allclose()->isclose() decomposition combines nan/close masks
    # with these in-place, so their absence cascades into ~every
    # allclose-based test failing.
    "bitwise_and_.Tensor": ("inplace_mul_tensor", "InplaceBitwiseAndTensor"),
    "bitwise_or_.Tensor": ("inplace_mul_tensor", "InplaceBitwiseOrTensor"),
    "bitwise_xor_.Tensor": ("inplace_mul_tensor", "InplaceBitwiseXorTensor"),
    "addcmul_": ("inplace_addcmul", "InplaceAddcmul"),
    "addcdiv_": ("inplace_addcdiv", "InplaceAddcdiv"),
    "sqrt_": ("inplace_sqrt", "InplaceSqrt"),
    "lerp_.Scalar": ("inplace_lerp_scalar", "InplaceLerps"),
    # ---- the rest of the in-place arithmetic surface.
    #
    # aten spells these ops twice and only the out-of-place half was routed, so
    # `x.neg_()` died with "backend not registered" while `x.neg()` worked. The
    # asymmetry is per-overload and only shows up at runtime, which also forced
    # composed kernels (rng.cc's inverse-transform samplers) to spell themselves
    # as `at::neg(x)` + `zero_().add_()` -- an extra kernel and an extra
    # allocation each. Every entry below is gated on the aclnn symbol actually
    # being in libopapi.so, so a CANN version without one drops that op rather
    # than emitting a kernel that fails to link.
    #
    # NOTE: every in-place op needs an explicit "Inplace*" override. The default
    # aclnn name drops the trailing underscore ("neg_" -> aclnnNeg), which is
    # the OUT-of-place entry point and would silently write to the wrong
    # argument slot. ----
    "acos_": ("inplace_unary", "InplaceAcos"),
    "acosh_": ("inplace_unary", "InplaceAcosh"),
    "asin_": ("inplace_unary", "InplaceAsin"),
    "asinh_": ("inplace_unary", "InplaceAsinh"),
    "atan_": ("inplace_unary", "InplaceAtan"),
    "atanh_": ("inplace_unary", "InplaceAtanh"),
    "ceil_": ("inplace_unary", "InplaceCeil"),
    "cos_": ("inplace_unary", "InplaceCos"),
    "cosh_": ("inplace_unary", "InplaceCosh"),
    "erf_": ("inplace_unary", "InplaceErf"),
    "erfc_": ("inplace_unary", "InplaceErfc"),
    "erfinv_": ("inplace_unary", "InplaceErfinv"),
    "exp_": ("inplace_unary", "InplaceExp"),
    "exp2_": ("inplace_unary", "InplaceExp2"),
    "expm1_": ("inplace_unary", "InplaceExpm1"),
    "floor_": ("inplace_unary", "InplaceFloor"),
    "frac_": ("inplace_unary", "InplaceFrac"),
    "hardsigmoid_": ("inplace_unary", "InplaceHardsigmoid"),
    "hardswish_": ("inplace_unary", "InplaceHardswish"),
    "log_": ("inplace_unary", "InplaceLog"),
    "log10_": ("inplace_unary", "InplaceLog10"),
    "log1p_": ("inplace_unary", "InplaceLog1p"),
    "log2_": ("inplace_unary", "InplaceLog2"),
    "logical_not_": ("inplace_unary", "InplaceLogicalNot"),
    "mish_": ("inplace_unary", "InplaceMish"),
    "neg_": ("inplace_unary", "InplaceNeg"),
    "reciprocal_": ("inplace_unary", "InplaceReciprocal"),
    "relu_": ("inplace_unary", "InplaceRelu"),
    "round_": ("inplace_unary", "InplaceRound"),
    "rsqrt_": ("inplace_unary", "InplaceRsqrt"),
    "sigmoid_": ("inplace_unary", "InplaceSigmoid"),
    "sin_": ("inplace_unary", "InplaceSin"),
    "sinh_": ("inplace_unary", "InplaceSinh"),
    "tan_": ("inplace_unary", "InplaceTan"),
    "tanh_": ("inplace_unary", "InplaceTanh"),
    "trunc_": ("inplace_unary", "InplaceTrunc"),
    "round_.decimals": ("inplace_int64", "InplaceRoundDecimals"),
    # (self&, Scalar). The "s"-suffixed aclnn names (Divs/Subs/FloorDivides) are
    # the scalar variants; the unsuffixed ones take a Tensor and would misread
    # an aclScalar* as an aclTensor*.
    "div_.Scalar": ("inplace_unary_scalar", "InplaceDivs"),
    "floor_divide_.Scalar": ("inplace_unary_scalar", "InplaceFloorDivides"),
    "fmod_.Scalar": ("inplace_unary_scalar", "InplaceFmodScalar"),
    "eq_.Scalar": ("inplace_unary_scalar", "InplaceEqScalar"),
    "ne_.Scalar": ("inplace_unary_scalar", "InplaceNeScalar"),
    "lt_.Scalar": ("inplace_unary_scalar", "InplaceLtScalar"),
    "gt_.Scalar": ("inplace_unary_scalar", "InplaceGtScalar"),
    "le_.Scalar": ("inplace_unary_scalar", "InplaceLeScalar"),
    "ge_.Scalar": ("inplace_unary_scalar", "InplaceGeScalar"),
    "bitwise_and_.Scalar": ("inplace_unary_scalar", "InplaceBitwiseAndScalar"),
    "bitwise_or_.Scalar": ("inplace_unary_scalar", "InplaceBitwiseOrScalar"),
    "bitwise_xor_.Scalar": ("inplace_unary_scalar", "InplaceBitwiseXorScalar"),
    "celu_": ("inplace_unary_scalar", "InplaceCelu"),
    "leaky_relu_": ("inplace_unary_scalar", "InplaceLeakyRelu"),
    # (self&, Tensor other).
    "eq_.Tensor": ("inplace_binary_tensor", "InplaceEqTensor"),
    "ne_.Tensor": ("inplace_binary_tensor", "InplaceNeTensor"),
    "lt_.Tensor": ("inplace_binary_tensor", "InplaceLtTensor"),
    "gt_.Tensor": ("inplace_binary_tensor", "InplaceGtTensor"),
    "le_.Tensor": ("inplace_binary_tensor", "InplaceLeTensor"),
    "ge_.Tensor": ("inplace_binary_tensor", "InplaceGeTensor"),
    "fmod_.Tensor": ("inplace_binary_tensor", "InplaceFmodTensor"),
    "floor_divide_.Tensor": ("inplace_binary_tensor", "InplaceFloorDivide"),
    "logical_and_": ("inplace_binary_tensor", "InplaceLogicalAnd"),
    "logical_or_": ("inplace_binary_tensor", "InplaceLogicalOr"),
    "atan2_": ("inplace_binary_tensor", "InplaceAtan2"),
    # sub_ carries the trailing alpha that mul_/div_ do not.
    "sub_.Tensor": ("inplace_sub_tensor", "InplaceSub"),
    "sub_.Scalar": ("inplace_sub_scalar", "InplaceSubs"),
    # (self&, Scalar, Scalar) / (self&, int64) / activation with 3 scalars.
    "threshold_": ("inplace_two_scalar", "InplaceThreshold"),
    "hardtanh_": ("inplace_two_scalar", "InplaceHardtanh"),
    "tril_": ("inplace_int64", "InplaceTril"),
    "triu_": ("inplace_int64", "InplaceTriu"),
    "elu_": ("inplace_elu", "InplaceElu"),
    "masked_fill_.Scalar": (
        "inplace_masked_fill_scalar",
        "InplaceMaskedFillScalar",
    ),
    "masked_fill_.Tensor": (
        "inplace_masked_fill_tensor",
        "InplaceMaskedFillTensor",
    ),
    # ---- clamp: the in-place spellings plus the two missing out-of-place
    # Tensor-bound siblings. CANN's in-place coverage is asymmetric here
    # (ClampMax yes / ClampMin no, both Tensor forms yes / neither Scalar form
    # of the two-sided op), so the templates differ per spelling -- see the
    # T_INPLACE_CLAMP* block for which ones compose from an out-of-place call.
    "clamp_": ("inplace_clamp", "Clamp"),
    "clamp_.Tensor": ("inplace_clamp_tensor", "ClampTensor"),
    "clamp_min_": ("inplace_clamp_bound", "ClampMin"),
    "clamp_max_": ("inplace_unary_scalar", "InplaceClampMax"),
    "clamp_min_.Tensor": ("inplace_clamp_bound_tensor", "InplaceClampMinTensor"),
    "clamp_max_.Tensor": ("inplace_clamp_bound_tensor", "InplaceClampMaxTensor"),
    "clamp_min.Tensor": ("clamp_bound_tensor", "ClampMinTensor"),
    "clamp_max.Tensor": ("clamp_bound_tensor", "ClampMaxTensor"),
    # ---- long-tail gaps found alongside the clamp family ----
    # linspace: used by test_clamp_dispatch to build broadcast bounds, and a
    # common way to build a schedule/grid in user code.
    "linspace": ("linspace", "Linspace"),
    # mse_loss was routed but its backward was not, so any training loop using
    # nn.MSELoss died on .backward().
    "mse_loss_backward": ("mse_loss_backward", "MseLossBackward"),
    # ---- foreach (TensorList) family: needed by torch.optim.AdamW's default
    # foreach=True path (aten's _multi_tensor_adam). All void-returning
    # in-place ops except _foreach_sqrt (returns new Tensor[]). ----
    "_foreach_mul_.Scalar": ("foreach_inplace_scalar", "ForeachMulScalarV2"),
    "_foreach_add_.Scalar": ("foreach_inplace_scalar", "ForeachAddScalarV2"),
    "_foreach_add_.List": ("foreach_inplace_add_list", None),
    "_foreach_lerp_.Scalar": ("foreach_inplace_lerp_scalar", "ForeachLerpScalar"),
    "_foreach_addcmul_.Scalar": (
        "foreach_inplace_addcmul_scalar",
        "ForeachAddcmulScalarV2",
    ),
    "_foreach_sqrt": ("foreach_sqrt", "ForeachSqrt"),
    "_foreach_div_.ScalarList": (
        "foreach_inplace_div_scalarlist",
        "ForeachDivScalarList",
    ),
    "_foreach_addcdiv_.ScalarList": (
        "foreach_inplace_addcdiv_scalarlist",
        "ForeachAddcdivScalarList",
    ),
    # ---- foreach ops behind torch.nn.utils.clip_grad_* ----
    # clip_grad_norm_ needs _foreach_norm; clip_grad_value_ needs
    # _foreach_clamp_min_. CANN has no ForeachClampMin* spelling at all, but
    # clamp_min == elementwise maximum against a scalar, so the Maximum entry
    # is exactly equivalent. Use the V2 form: non-V2 declares its scalar as
    # aclTensor*, V2 as aclScalar*.
    "_foreach_norm.Scalar": ("foreach_norm", "ForeachNorm"),
    "_foreach_clamp_min_.Scalar": (
        "foreach_inplace_maximum_scalar",
        "ForeachMaximumScalarV2",
    ),
    # clamp_max == elementwise minimum against a scalar; same template, other
    # direction. clip_grad_value_ issues both clamp_min_ and clamp_max_.
    "_foreach_clamp_max_.Scalar": (
        "foreach_inplace_maximum_scalar",
        "ForeachMinimumScalarV2",
    ),
    # linalg_vector_norm backs torch.norm / F.normalize / cosine_similarity and
    # is what clip_grad_norm_ reduces each per-tensor norm with.
    "linalg_vector_norm": ("linalg_vector_norm", "LinalgVectorNorm"),
    # clip_grad_norm_ finishes by scaling every gradient by one clip coefficient.
    "_foreach_mul_.Tensor": ("foreach_inplace_mul_tensor", "ForeachMulList"),
    # ---- embedding + pad (single-aclnn-call, migrated from handwritten) ----
    "embedding": ("embedding", "Embedding"),
    "embedding_dense_backward": ("embedding_dense_backward", "EmbeddingDenseBackward"),
    "constant_pad_nd": ("constant_pad_nd", "ConstantPadNd"),
    # ---- BCE loss family: optional weight, int reduction (0=none/1=mean/2=sum) ----
    "binary_cross_entropy": ("bce", "BinaryCrossEntropy"),
    "binary_cross_entropy_backward": ("bce_backward", "BinaryCrossEntropyBackward"),
    "binary_cross_entropy_with_logits": ("bce_logits", "BinaryCrossEntropyWithLogits"),
    # ---- norm family: tuple(out, mean, rstd), optional weight/bias ----
    "native_layer_norm": ("layer_norm", "LayerNorm"),
    "native_group_norm": ("group_norm", "GroupNorm"),
    # ---- gelu / softmax family (transformer backbone, fwd + bwd) ----
    # gelu: v1 aclnnGelu hardcodes tanh; use V2 (int64 approximate 0=none/1=tanh)
    # to honor PyTorch's default approximate="none" (erf form).
    "gelu": ("gelu", "GeluV2"),
    "gelu_backward": ("gelu_backward", "GeluBackwardV2"),
    # _log_softmax mirrors handwritten softmax.cc (aclnnLogSoftmax(self,dim,out)).
    "_log_softmax": ("log_softmax", "LogSoftmax"),
    # backward: aclnn names lack the aten "_data" suffix.
    "_softmax_backward_data": ("softmax_backward", "SoftmaxBackward"),
    "_log_softmax_backward_data": ("softmax_backward", "LogSoftmaxBackward"),
    # ---- migrated seeds needing dedicated categories ----
    "silu_backward": ("act_backward_self", "SiluBackward"),
    "where.self": ("where", "SWhere"),
    "_softmax": ("softmax_fwd", "Softmax"),
    "all": ("reduce_all", "All"),
    "any": ("reduce_all", "Any"),
    "sum.dim_IntList": ("reduce_sum_dtype", "ReduceSum"),
    "sum": ("reduce_sum_all", "ReduceSum"),
    "max": ("reduce_minmax_all", "Max"),
    "min": ("reduce_minmax_all", "Min"),
    "mean.dim": ("reduce_mean_dtype", "MeanV2"),
    "mean": ("mean_all", "Mean"),
    "clamp": ("clamp", "Clamp"),
    "clamp.Tensor": ("clamp_tensor", "ClampTensor"),
    # ---- conv/pool family (each carries an output-shape formula) ----
    "_adaptive_avg_pool2d": ("adaptive_avg_pool2d", "AdaptiveAvgPool2d"),
    "avg_pool2d": ("avg_pool2d", "AvgPool2d"),
    "max_pool2d_with_indices": ("max_pool2d_indices", "MaxPool2dWithIndices"),
    "convolution": ("convolution", "Convolution"),
    "convolution_backward": ("convolution_backward", "ConvolutionBackward"),
    # GradScaler's device-side finite check and unscale operation. Ascend has
    # no generated aclnn variant in this release, so use the CPU reference
    # while preserving the in-place mutation contract.
    "_amp_foreach_non_finite_check_and_unscale_": ("amp_unscale", None),
    "_amp_foreach_non_finite_check_and_unscale.out": ("amp_unscale_out", None),
}

# Ops with a handwritten kAscend kernel — never regenerate (double-register).
# Kept as a guard even though none currently overlap OPS above.
SKIP = {
    # le.Tensor stays handwritten: aclnnLe symbol is absent, needs runtime
    # multi-version probing (aclnnLe / aclnnLeTensor / aclnnLessEqual).
    "le.Tensor",
}

# --------------------------------------------------------------------------
# Per-category kernel body templates. Placeholders: {kernel} {aclnn} {fn} {disp}
# --------------------------------------------------------------------------
T_UNARY = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  if (!ascend::IsUnaryDtypeSupported(self.scalar_type())) {{
    auto cpu_result = at::{cpu_op}(self.cpu());
    return ascend::MoveCpuResultToDevice(std::move(cpu_result), self);
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Shared broadcast+dtype prologue for tensor-tensor categories.
#
# `other` is coerced to self's DEVICE and dtype: the aten .Tensor overloads
# (add.Tensor/sub.Tensor/...) are what the Python operators lower a scalar
# argument to (e.g. `x - 3.0` -> aten::sub.Tensor with a CPU scalar tensor),
# so `other` may live on CPU. Building an aclTensor over CPU storage as if it
# were NPU memory yields garbage/nan, so we must migrate it to self's device
# first (mirrors the handwritten add.cc). Both operands are then expanded +
# materialized to the broadcast shape so aclnn (which does not always
# broadcast) sees matching ND-contiguous inputs. All steps are no-ops when
# device/dtype/shape already match.
# The prologue body (everything after the `namespace ascend` alias). Split out
# so the cached templates can inject a scalar fast-path branch before it while
# still sharing one namespace alias.
_BINARY_PROLOGUE_BODY = """\
  auto result_dtype = ascend::BinaryResultType("{aclnn}", self, other);
  auto self_c = self.scalar_type() == result_dtype
      ? self
      : self.to(self.options().dtype(result_dtype));
  auto other_c = other.is_privateuseone()
      ? (other.scalar_type() == result_dtype ? other : other.to(result_dtype))
      : other.to(self.options().dtype(result_dtype));
  auto out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  auto result_options = self.options().dtype(result_dtype);
  // aclnn binary ops broadcast and honor strides internally (verified), so we
  // pass self/other_c straight through instead of expand().contiguous(). This
  // avoids up to two device strided-copies + host view construction per op on
  // the eager decode hot path. Only materialize a contiguous copy when the
  // tensor is genuinely non-contiguous AND the aclnn path would otherwise need
  // it — measured unnecessary for the common same-shape/contiguous case, which
  // is the overwhelming majority in Qwen3.
  const at::Tensor& self_b = self_c;
  const at::Tensor& other_b = other_c;
"""

_BINARY_PROLOGUE = (
    """\
  namespace ascend = at::native::flagos::ascend;
"""
    + _BINARY_PROLOGUE_BODY
)

# Scalar fast-path branches injected at the top of the cached binary kernels.
# When `other` is a wrapped CPU scalar (a python float/int, materialized by
# PyTorch as a 0-dim CPU tensor), the default path's `other.to(self.options())`
# does a per-call H2D copy (~22us measured -- the single biggest torch_fl vs
# torch_npu host gap: add.Tensor 49us vs 13us). Diverting to the aclnn scalar
# variant (aclnn<Name>s, which takes an aclScalar* by value) skips the H2D
# entirely. Only emitted for ops that actually ship an <Name>s symbol
# (add/sub/mul/div); the scalar value is folded into the executor-cache key.
_SCALAR_FASTPATH_NOALPHA = """\
  if (self.is_privateuseone() && !other.is_privateuseone() && other.numel() == 1) {{
    at::Scalar sc = other.item();
    auto out = ascend::OpPreparation::apply_tensor_without_format(
        self.sizes(), self.options());
    ascend::AclScalarWrapper acl_sc(sc, self.scalar_type());
    static void* sOpAddr = nullptr; static void* sWsAddr = nullptr;
    ascend::SigHasher hsh; hsh.tensor(self);
    {{ double sv = sc.toDouble(); hsh.val(sv); }}
    ascend::ExecAscendCached(
        "{aclnn_s}", "{aclnn_s}GetWorkspaceSize", sOpAddr, sWsAddr, hsh.h,
        {{&self}}, {{&out}},
        [&](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
            std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
          return gws(in[0].acl_tensor, acl_sc.get(), out_t[0].acl_tensor, pws, pex);
        }});
    return out;
  }}
"""

_SCALAR_FASTPATH_ALPHA = """\
  if (self.is_privateuseone() && !other.is_privateuseone() && other.numel() == 1) {{
    at::Scalar sc = other.item();
    auto out = ascend::OpPreparation::apply_tensor_without_format(
        self.sizes(), self.options());
    ascend::AclScalarWrapper acl_sc(sc, self.scalar_type());
    ascend::AclScalarWrapper acl_alpha_s(alpha, self.scalar_type());
    static void* sOpAddr = nullptr; static void* sWsAddr = nullptr;
    ascend::SigHasher hsh; hsh.tensor(self);
    {{ double sv = sc.toDouble(); hsh.val(sv); double av = alpha.toDouble(); hsh.val(av); }}
    ascend::ExecAscendCached(
        "{aclnn_s}", "{aclnn_s}GetWorkspaceSize", sOpAddr, sWsAddr, hsh.h,
        {{&self}}, {{&out}},
        [&](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
            std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
          return gws(in[0].acl_tensor, acl_sc.get(), acl_alpha_s.get(), out_t[0].acl_tensor, pws, pex);
        }});
    return out;
  }}
"""

T_FOREACH_INPLACE_ADD_LIST = """
void {kernel}(at::TensorList self, at::TensorList other, const at::Scalar& alpha) {{
  TORCH_CHECK(self.size() == other.size(), "{disp}: tensor lists must match in length");
  for (size_t i = 0; i < self.size(); ++i) {{
    at::Tensor dst = self[i];
    dst.add_(other[i], alpha);
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_AMP_UNSCALE_OUT = """
void {kernel}(at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale, at::TensorList out) {{
  TORCH_CHECK(self.size() == out.size(), "{disp}: tensor lists must match in length");
  std::vector<at::Tensor> cpu_self;
  cpu_self.reserve(self.size());
  for (const auto& tensor : self) {{
    cpu_self.push_back(tensor.cpu());
  }}
  auto cpu_found_inf = found_inf.cpu();
  auto cpu_inv_scale = inv_scale.cpu();
  at::_amp_foreach_non_finite_check_and_unscale_outf(
      cpu_self, cpu_found_inf, cpu_inv_scale, cpu_self);
  for (size_t i = 0; i < out.size(); ++i) {{
    at::Tensor dst = out[i];
    dst.copy_(cpu_self[i]);
  }}
  found_inf.copy_(cpu_found_inf);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_AMP_UNSCALE = """
void {kernel}(at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale) {{
  std::vector<at::Tensor> cpu_self;
  cpu_self.reserve(self.size());
  for (const auto& tensor : self) {{
    cpu_self.push_back(tensor.cpu());
  }}
  auto cpu_found_inf = found_inf.cpu();
  auto cpu_inv_scale = inv_scale.cpu();
  at::_amp_foreach_non_finite_check_and_unscale_(cpu_self, cpu_found_inf, cpu_inv_scale);
  for (size_t i = 0; i < self.size(); ++i) {{
    at::Tensor dst = self[i];
    dst.copy_(cpu_self[i]);
  }}
  found_inf.copy_(cpu_found_inf);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_BINARY = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, result_options);

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

T_BINARY_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, result_options);

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclScalarWrapper acl_alpha(alpha, result_dtype);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_alpha.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

T_BINARY_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

T_BINARY_SCALAR_ALPHA = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_alpha.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_BINARY_SCALAR_CMP = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Shared dim-normalization + reduced-shape prologue for reduce categories.
# Mirrors the handwritten sum.cc: negative dims are wrapped to [0,ndim); an
# empty dim list means "reduce all"; the output shape drops (or, with keepdim,
# sets to 1) each reduced dim. Dims are erased high-to-low so earlier erases do
# not shift later indices.
_REDUCE_DIMS_PROLOGUE = """\
  namespace ascend = at::native::flagos::ascend;
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  if (!dim.empty()) {{
    for (int64_t d : dim) norm_dims.push_back(d < 0 ? d + ndim : d);
  }} else {{
    for (int64_t d = 0; d < ndim; ++d) norm_dims.push_back(d);
  }}
  auto out_shape = self.sizes().vec();
  std::vector<int64_t> sorted_dims(norm_dims);
  std::sort(sorted_dims.rbegin(), sorted_dims.rend());
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
  }}
"""

# reduce_dims: (Tensor, IntArrayRef dim, bool keepdim) -> reduced, same dtype.
#   aclnn<Name>(self, dim, keepdim, out)   e.g. amax/amin
T_REDUCE_DIMS = (
    """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {{
"""
    + _REDUCE_DIMS_PROLOGUE
    + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(norm_dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# reduce_dim_bool: (Tensor, int64_t dim, bool keepdim) -> bool out.
#   aclnn<Name>(self, dim_list, keepdim, out)   e.g. any.dim
# aclnn takes a dim *list*, so the single dim is wrapped into a one-element vec.
T_REDUCE_DIM_BOOL = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool keepdim) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  std::vector<int64_t> dims{{d}};
  auto out_shape = self.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cumsum: (Tensor, int64_t dim, optional<ScalarType> dtype) -> same shape.
#   aclnn<Name>(self, dim, dtype, out)
# Integral promotion: with no explicit dtype, PyTorch promotes any integral input
# (incl. bool) to int64; float dtypes pass through. aclnn also rejects a bool
# `self` (err 161002), so cast the input tensor to the promoted dtype too.
T_CUMSUM = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, ::std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_dtype = dtype.value_or(
      at::isIntegralType(self.scalar_type(), /*includeBool=*/true)
          ? at::kLong : self.scalar_type());
  auto in = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      in.sizes(), in.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(in);
  ascend::AclTensorWrapper acl_out(out);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cumprod: like cumsum, but aclnnCumprod takes dim as an aclScalar* (int64), not
# a plain int64_t. Otherwise identical: (Tensor, int64 dim, optional dtype).
# Same integral->int64 promotion + input cast as cumsum.
T_CUMPROD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, ::std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_dtype = dtype.value_or(
      at::isIntegralType(self.scalar_type(), /*includeBool=*/true)
          ? at::kLong : self.scalar_type());
  auto in = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      in.sizes(), in.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(in);
  ascend::AclScalarWrapper acl_dim(at::Scalar(d), at::kLong);
  ascend::AclTensorWrapper acl_out(out);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_bool: (Tensor) -> bool out, same shape.  aclnn<Name>(self, out)
#   e.g. isinf (always bool regardless of input dtype)
T_UNARY_BOOL = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_scalar: (Tensor, Scalar) -> same shape/dtype.
#   aclnn<Name>(self, scalar, out)   e.g. leaky_relu/clamp_min/clamp_max/fmod.Scalar
# The scalar is packed at self's dtype so aclnn sees matching operand types.
T_UNARY_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& s) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_s(s, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_s.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_scalar CACHED: same as T_UNARY_SCALAR but through the repeatable-executor
# cache. The Scalar is baked into the executor at build time (aclnn reads it
# during GetWorkspaceSize), so it MUST be part of the cache key -- a different
# scalar value needs a distinct executor. On the decode hot path pow.Tensor_Scalar
# (x^2 in RMSNorm, 113/step, measured 50us uncached) is the big win.
T_UNARY_SCALAR_CACHED = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& s) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());
  ascend::AclScalarWrapper acl_s(s, self.scalar_type());

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self);
  {{ double sv = s.toDouble(); hsh.val(sv); }}
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self}}, {{&out}},
      [&](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
          std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, acl_s.get(), out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_two_scalar: (Tensor, Scalar, Scalar) -> same shape/dtype.
#   aclnn<Name>(self, s1, s2, out)   e.g. softplus(beta,threshold)/threshold(threshold,value)
T_UNARY_TWO_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& s1, const at::Scalar& s2) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_s1(s1, self.scalar_type());
  ascend::AclScalarWrapper acl_s2(s2, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_s1.get(), acl_s2.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_int: (Tensor, int64_t) -> same shape/dtype.  aclnn<Name>(self, i, out)
#   e.g. tril/triu (diagonal offset)
T_UNARY_INT = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t i) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), i, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# unary_dims: (Tensor, IntArrayRef dims) -> same shape/dtype.
#   aclnn<Name>(self, dims, out)   e.g. flip
T_UNARY_DIMS = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef dims) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  std::vector<int64_t> dims_v(dims.begin(), dims.end());
  ascend::AclIntArrayWrapper acl_dims(dims_v);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dims.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# addcmul/addcdiv: (self, t1, t2, Scalar value) -> broadcast(self,t1,t2), self dtype.
#   aclnn<Name>(self, t1, t2, value, out)
# t1/t2 are migrated to self's device+dtype (same rationale as _BINARY_PROLOGUE:
# a CPU-resident operand read as NPU storage would produce garbage).
T_ADDCMUL = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto opts = self.options();
  auto t1 = tensor1.is_privateuseone() ? tensor1.to(opts.dtype(tensor1.scalar_type())) : tensor1.to(opts);
  auto t2 = tensor2.is_privateuseone() ? tensor2.to(opts.dtype(tensor2.scalar_type())) : tensor2.to(opts);
  auto out_shape = at::infer_size(at::infer_size(self.sizes(), t1.sizes()), t2.sizes());
  auto self_b = self.expand(out_shape).contiguous();
  auto t1_b = t1.expand(out_shape).contiguous();
  auto t2_b = t2.expand(out_shape).contiguous();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, opts);

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_t1(t1_b);
  ascend::AclTensorWrapper acl_t2(t2_b);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_t1.get(), acl_t2.get(), acl_value.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# act_backward: (grad_output, output) -> grad_input, output shape/dtype.
#   aclnn<Name>(gradOutput, output, gradInput)   e.g. tanh_backward/sigmoid_backward
T_ACT_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& output) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      output.sizes(), output.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_output(output);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_output.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# threshold_backward: (grad_output, self, Scalar threshold) -> grad_input.
#   aclnn<Name>(gradOutput, self, threshold, gradInput)
T_THRESHOLD_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_threshold(threshold, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_threshold.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# pow_scalar_tensor: (Scalar self, Tensor exponent) -> exponent shape.
#   aclnn<Name>(selfScalar, exponent, out)   e.g. pow.Scalar
T_POW_SCALAR_TENSOR = """\
at::Tensor {kernel}(const at::Scalar& self, const at::Tensor& exponent) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      exponent.sizes(), exponent.options());

  ascend::AclScalarWrapper acl_self(self, exponent.scalar_type());
  ascend::AclTensorWrapper acl_exp(exponent);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_exp.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_max_dim: (Tensor, int64_t dim, bool keepdim) -> tuple(values, indices).
#   aclnn<Name>(self, dim, keepdim, valuesOut, indicesOut)   e.g. max.dim/min.dim
# values keep self dtype; indices are int64.
T_REDUCE_MAX_DIM = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, int64_t dim, bool keepdim) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_shape = self.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);

  auto values = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_values(values);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, keepdim, acl_values.get(), acl_indices.get());
  return std::make_tuple(values, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# elu: (Tensor, Scalar alpha, Scalar scale, Scalar input_scale) -> same shape.
#   aclnn<Name>(self, alpha, scale, inputScale, out)
T_ELU = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclScalarWrapper acl_scale(scale, self.scalar_type());
  ascend::AclScalarWrapper acl_input_scale(input_scale, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_alpha.get(), acl_scale.get(), acl_input_scale.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# loss: (self, target, int64_t reduction) -> reduction==None(0) elementwise,
# Mean(1)/Sum(2) scalar.  aclnn<Name>(self, target, reduction, out)
# e.g. mse_loss / l1_loss. target is a genuine on-device tensor (no CPU-scalar
# lowering as with the binary categories), so no device coercion is needed.
T_LOSS = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar for Mean/Sum
  if (reduction == 0) out_shape = self.sizes().vec();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_target.get(), reduction, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cummax/cummin: (Tensor, int64_t dim) -> tuple(values, indices), same shape.
#   aclnn<Name>(self, dim, valuesOut, indicesOut)
T_CUMMAX_CUMMIN = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, int64_t dim) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto values = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_values(values);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_values.get(), acl_indices.get());
  return std::make_tuple(values, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# aminmax: (Tensor, optional<int64_t> dim, bool keepdim) -> tuple(min, max).
#   aclnn<Name>(self, dim_list, keepDim, minOut, maxOut)
# nullopt dim reduces all dims (scalar out); a given dim reduces that one.
T_AMINMAX = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, ::std::optional<int64_t> dim, bool keepdim) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> dims;
  std::vector<int64_t> out_shape;
  if (dim.has_value()) {{
    int64_t d = dim.value() < 0 ? dim.value() + self.dim() : dim.value();
    dims.push_back(d);
    out_shape = self.sizes().vec();
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
  }} else {{
    for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i);
  }}
  auto min_out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());
  auto max_out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_min(min_out);
  ascend::AclTensorWrapper acl_max(max_out);
  ascend::AclIntArrayWrapper acl_dim(dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_min.get(), acl_max.get());
  return std::make_tuple(min_out, max_out);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# prod: (Tensor, optional<ScalarType> dtype) -> scalar.
#   aclnn<Name>(self, dtype, out)
T_PROD = """\
at::Tensor {kernel}(const at::Tensor& self, ::std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = dtype.value_or(self.scalar_type());
  std::vector<int64_t> out_shape;   // scalar
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gemm_addmm: (self, mat1, mat2, Scalar beta, Scalar alpha) -> (m1.rows, m2.cols).
#   aclnn<Name>(self, mat1, mat2, beta, alpha, out, cubeMathType)   e.g. addmm
# self broadcasts into the matmul result; aclnn handles the broadcast.
T_GEMM_ADDMM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{mat1.size(0), mat2.size(1)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat1(mat1);
  ascend::AclTensorWrapper acl_mat2(mat2);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_mat1.get(), acl_mat2.get(), acl_beta.get(), acl_alpha.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gemm_baddbmm: batched addmm -> (b, b1.rows, b2.cols).
#   aclnn<Name>(self, batch1, batch2, beta, alpha, out, cubeMathType)
T_GEMM_BADDBMM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{batch1.size(0), batch1.size(1), batch2.size(2)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_batch1(batch1);
  ascend::AclTensorWrapper acl_batch2(batch2);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_batch1.get(), acl_batch2.get(), acl_beta.get(), acl_alpha.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# matmul: (self, mat2) -> matmul result.  aclnn<Name>(self, mat2, out, cubeMathType)
#   mm:  2-D x 2-D -> (self.rows, mat2.cols)
#   bmm: 3-D x 3-D -> (batch, self.rows, mat2.cols)  [aclnnBatchMatMul]
# out_shape widens the batch dims of self then appends mat2's trailing dim, so a
# single template covers both the 2-D and batched cases.
T_MATMUL = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  if (!ascend::IsMatmulDtypeSupported(self.scalar_type()) ||
      self.scalar_type() != mat2.scalar_type()) {{
    auto cpu_result = at::{cpu_op}(self.cpu(), mat2.cpu());
    return ascend::MoveCpuResultToDevice(std::move(cpu_result), self);
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat2(mat2);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_mat2.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# matmul_out: .out variant of matmul.  Writes self @ mat2 into caller-provided out&.
#   Shares the aclnn call with T_MATMUL; the framework has already shaped `out`.
T_MATMUL_OUT = """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat2(mat2);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_mat2.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# mv: (self (n,m), vec (m,)) -> (n,).  aclnn<Name>(self, vec, out, cubeMathType)
T_MV = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& vec) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{self.size(0)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_vec(vec);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_vec.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# dot: (self, tensor) both 1-D -> scalar.  aclnn<Name>(self, tensor, out)
T_DOT = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& tensor) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_tensor(tensor);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_tensor.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cat: (ITensorListRef tensors, dim) -> concatenation along dim.
#   aclnn<Name>(aclTensorList, dim, out). Filters numel==0 tensors (avoids
#   dim-mismatch) and short-circuits the 0/1-valid-tensor cases like the ref.
#   NOTE: aclCreateTensorList's aclTensor* are still owned by the AclTensorWrapper
#   RAII objects, so we must NOT call aclDestroyTensorList.
T_CAT = """\
at::Tensor {kernel}(const at::ITensorListRef& tensors, int64_t dim) {{
  namespace ascend = at::native::flagos::ascend;

  auto materialized = tensors.materialize();
  TORCH_CHECK(!materialized.empty(), "cat: expected a non-empty list of tensors");

  std::vector<at::Tensor> valid_tensors;
  for (const auto& t : materialized) {{
    if (t.get().numel() > 0) {{
      valid_tensors.push_back(t.get());
    }}
  }}

  if (valid_tensors.empty()) {{
    return materialized[0].get().clone();
  }}
  if (valid_tensors.size() == 1) {{
    return valid_tensors[0].clone();
  }}

  auto& first = valid_tensors[0];
  auto ndim = first.dim();
  if (dim < 0) dim += ndim;

  std::vector<int64_t> out_sizes(first.sizes().begin(), first.sizes().end());
  for (size_t i = 1; i < valid_tensors.size(); ++i) {{
    out_sizes[dim] += valid_tensors[i].size(dim);
  }}

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, first.options());

  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(valid_tensors.size());
  for (auto& t : valid_tensors) {{
    wrappers.emplace_back(t);
  }}

  std::vector<const aclTensor*> acl_tensors;
  acl_tensors.reserve(valid_tensors.size());
  for (auto& w : wrappers) {{
    acl_tensors.push_back(w.get());
  }}

  aclTensorList* tensor_list = aclCreateTensorList(
      acl_tensors.data(), acl_tensors.size());

  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, tensor_list, dim, acl_out.get());

  (void)tensor_list;  // aclTensor* owned by wrappers; do not aclDestroyTensorList
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# cat.out: same aclnn call and filtering as T_CAT, but writes into the
# caller-provided output. The framework has already sized `out`.
T_CAT_OUT = """\
at::Tensor& {kernel}(
    const at::ITensorListRef& tensors,
    int64_t dim,
    at::Tensor& out) {{
  namespace ascend = at::native::flagos::ascend;

  auto materialized = tensors.materialize();
  TORCH_CHECK(!materialized.empty(), "cat.out: expected a non-empty list of tensors");

  std::vector<at::Tensor> valid_tensors;
  for (const auto& t : materialized) {{
    if (t.get().numel() > 0) {{
      valid_tensors.push_back(t.get());
    }}
  }}

  if (valid_tensors.empty()) {{
    out.copy_(materialized[0].get());
    return out;
  }}
  if (valid_tensors.size() == 1) {{
    out.copy_(valid_tensors[0]);
    return out;
  }}

  auto& first = valid_tensors[0];
  auto ndim = first.dim();
  if (dim < 0) dim += ndim;

  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(valid_tensors.size());
  for (auto& t : valid_tensors) {{
    wrappers.emplace_back(t);
  }}

  std::vector<const aclTensor*> acl_tensors;
  acl_tensors.reserve(valid_tensors.size());
  for (auto& w : wrappers) {{
    acl_tensors.push_back(w.get());
  }}

  aclTensorList* tensor_list = aclCreateTensorList(
      acl_tensors.data(), acl_tensors.size());

  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, tensor_list, dim, acl_out.get());

  (void)tensor_list;  // aclTensor* owned by wrappers; do not aclDestroyTensorList
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# stack: (TensorList tensors, dim) -> new-dim concatenation (unlike cat, no
# existing dim is merged; each input keeps its own shape and dim is inserted).
#   aclnn<Name>(aclTensorList, dim, out). dim is normalized against the OUTPUT
#   rank (input rank + 1), matching torch's `maybe_wrap_dim(dim, ndim + 1)`.
T_STACK = """\
at::Tensor {kernel}(at::TensorList tensors, int64_t dim) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(!tensors.empty(), "stack: expected a non-empty list of tensors");

  auto& first = tensors[0];
  int64_t out_ndim = first.dim() + 1;
  if (dim < 0) dim += out_ndim;

  std::vector<int64_t> out_sizes(first.sizes().begin(), first.sizes().end());
  out_sizes.insert(out_sizes.begin() + dim, static_cast<int64_t>(tensors.size()));

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, first.options());

  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(tensors.size());
  for (const auto& t : tensors) {{
    wrappers.emplace_back(t);
  }}

  std::vector<const aclTensor*> acl_tensors;
  acl_tensors.reserve(tensors.size());
  for (auto& w : wrappers) {{
    acl_tensors.push_back(w.get());
  }}

  aclTensorList* tensor_list = aclCreateTensorList(
      acl_tensors.data(), acl_tensors.size());

  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, tensor_list, dim, acl_out.get());

  (void)tensor_list;  // aclTensor* owned by wrappers; do not aclDestroyTensorList
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# --- foreach ops needed by torch.optim.AdamW's foreach=True path (aten's
# _multi_tensor_adam). void return, in-place on `self`'s TensorList: build an
# aclTensorList for each TensorList arg, execute in-place-style (out == x),
# then leave the input tensors mutated (matches PyTorch's _foreach_*_ inplace
# semantics: the storage is written in place, no new Tensors are returned).
#   NOTE: aclTensorList's aclTensor* are owned by the AclTensorWrapper RAII
#   vector, so — same as cat/stack — we must NOT aclDestroyTensorList.

# _foreach_mul_.Scalar / _foreach_add_.Scalar: (self[]&, scalar) -> void.
#   aclnnForeachMulScalarV2/aclnnForeachAddScalarV2(x, scalar, out=x).
T_FOREACH_INPLACE_SCALAR = """\
static void {kernel}Chunk(at::TensorList self, const at::Scalar& scalar) {{
  namespace ascend = at::native::flagos::ascend;

  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(self.size());
  for (const auto& t : self) {{
    wrappers.emplace_back(t);
  }}
  std::vector<const aclTensor*> acl_tensors;
  acl_tensors.reserve(self.size());
  for (auto& w : wrappers) {{
    acl_tensors.push_back(w.get());
  }}
  aclTensorList* tensor_list = aclCreateTensorList(acl_tensors.data(), acl_tensors.size());
  // aic-ops-info: ForeachMulScalar/ForeachAddScalar's `scalar` dtype tracks x's
  // EXCEPT bf16 x, which requires a float32 scalar (no bf16 scalar entry).
  auto scalar_dtype = self[0].scalar_type() == at::kBFloat16 ? at::kFloat : self[0].scalar_type();
  ascend::AclScalarWrapper acl_scalar(scalar, scalar_dtype);

  EXEC_ASCEND_CMD({aclnn}, tensor_list, acl_scalar.get(), tensor_list);

  (void)tensor_list;  // aclTensor* owned by wrappers; do not aclDestroyTensorList
}}

void {kernel}(at::TensorList self, const at::Scalar& scalar) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), scalar);
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# CANN's aclnnForeach* kernels only process the FIRST 50 entries of an
# aclTensorList. Past that they either error (the ScalarList variants return
# 561002/161002) or -- worse -- return success while leaving entries >= 50
# untouched, so the bug is silent. Measured: entry 50 is the first wrong one for
# Mul/Add/Addcmul/Lerp/Sqrt alike, independent of each tensor's numel
# (8 .. 65536) and dtype (fp16/fp32/bf16). AdamW on Qwen3-0.6B passes 310
# tensors, so every foreach kernel slices its lists into sub-50 chunks;
# elementwise semantics make the split exact.
FOREACH_CHUNK = 32  # aclnn processes at most 50 entries per call

# _foreach_lerp_.Scalar: (self[]&, tensors1[], weight) -> void, self += weight*(tensors1-self).
#   aclnnForeachLerpScalar(x1=self, x2=tensors1, weight, out=self).
T_FOREACH_INPLACE_LERP_SCALAR = """\
static void {kernel}Chunk(at::TensorList self, at::TensorList tensors1, const at::Scalar& weight) {{
  namespace ascend = at::native::flagos::ascend;

  std::vector<ascend::AclTensorWrapper> self_w, t1_w;
  self_w.reserve(self.size());
  t1_w.reserve(tensors1.size());
  for (const auto& t : self) self_w.emplace_back(t);
  for (const auto& t : tensors1) t1_w.emplace_back(t);

  std::vector<const aclTensor*> self_ptrs, t1_ptrs;
  self_ptrs.reserve(self.size());
  t1_ptrs.reserve(tensors1.size());
  for (auto& w : self_w) self_ptrs.push_back(w.get());
  for (auto& w : t1_w) t1_ptrs.push_back(w.get());

  aclTensorList* self_list = aclCreateTensorList(self_ptrs.data(), self_ptrs.size());
  aclTensorList* t1_list = aclCreateTensorList(t1_ptrs.data(), t1_ptrs.size());
  // aic-ops-info: ForeachLerpScalar's `weight` is ALWAYS float32, regardless
  // of x1/x2's dtype (unlike mul_/add_.Scalar, which track x except for bf16).
  ascend::AclScalarWrapper acl_weight(weight, at::kFloat);

  EXEC_ASCEND_CMD({aclnn}, self_list, t1_list, acl_weight.get(), self_list);

  (void)self_list; (void)t1_list;  // owned by *_w; do not aclDestroyTensorList
}}

void {kernel}(at::TensorList self, at::TensorList tensors1, const at::Scalar& weight) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  TORCH_CHECK(self.size() == tensors1.size(), "{disp}: tensor lists must match in length");
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), tensors1.slice(off, n), weight);
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _foreach_addcmul_.Scalar: (self[]&, tensor1[], tensor2[], value) -> void,
#   self += value * tensor1 * tensor2.
#   aclnnForeachAddcmulScalarV2(x1=self, x2=tensor1, x3=tensor2, scalar=value, out=self).
T_FOREACH_INPLACE_ADDCMUL_SCALAR = """\
static void {kernel}Chunk(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;

  std::vector<ascend::AclTensorWrapper> self_w, t1_w, t2_w;
  self_w.reserve(self.size());
  t1_w.reserve(tensor1.size());
  t2_w.reserve(tensor2.size());
  for (const auto& t : self) self_w.emplace_back(t);
  for (const auto& t : tensor1) t1_w.emplace_back(t);
  for (const auto& t : tensor2) t2_w.emplace_back(t);

  std::vector<const aclTensor*> self_ptrs, t1_ptrs, t2_ptrs;
  self_ptrs.reserve(self.size());
  t1_ptrs.reserve(tensor1.size());
  t2_ptrs.reserve(tensor2.size());
  for (auto& w : self_w) self_ptrs.push_back(w.get());
  for (auto& w : t1_w) t1_ptrs.push_back(w.get());
  for (auto& w : t2_w) t2_ptrs.push_back(w.get());

  aclTensorList* self_list = aclCreateTensorList(self_ptrs.data(), self_ptrs.size());
  aclTensorList* t1_list = aclCreateTensorList(t1_ptrs.data(), t1_ptrs.size());
  aclTensorList* t2_list = aclCreateTensorList(t2_ptrs.data(), t2_ptrs.size());
  // aic-ops-info: ForeachAddcmulScalar's `scalar` dtype tracks x EXCEPT bf16 x,
  // which requires a float32 scalar (same rule as mul_/add_.Scalar).
  auto value_dtype = self[0].scalar_type() == at::kBFloat16 ? at::kFloat : self[0].scalar_type();
  ascend::AclScalarWrapper acl_value(value, value_dtype);

  EXEC_ASCEND_CMD({aclnn}, self_list, t1_list, t2_list, acl_value.get(), self_list);

  (void)self_list; (void)t1_list; (void)t2_list;  // owned by *_w
}}

void {kernel}(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar& value) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  TORCH_CHECK(self.size() == tensor1.size() && self.size() == tensor2.size(),
      "{disp}: tensor lists must match in length");
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), tensor1.slice(off, n), tensor2.slice(off, n), value);
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _foreach_sqrt: (self[]) -> Tensor[] (NOT in-place — returns new tensors).
#   aclnnForeachSqrt(x, out) with out a freshly-allocated TensorList.
T_FOREACH_SQRT = """\
static void {kernel}Chunk(at::TensorList self, at::TensorList outs) {{
  namespace ascend = at::native::flagos::ascend;


  std::vector<ascend::AclTensorWrapper> in_w, out_w;
  in_w.reserve(self.size());
  out_w.reserve(outs.size());
  for (const auto& t : self) in_w.emplace_back(t);
  for (const auto& t : outs) out_w.emplace_back(t);

  std::vector<const aclTensor*> in_ptrs, out_ptrs;
  in_ptrs.reserve(self.size());
  out_ptrs.reserve(outs.size());
  for (auto& w : in_w) in_ptrs.push_back(w.get());
  for (auto& w : out_w) out_ptrs.push_back(w.get());

  aclTensorList* in_list = aclCreateTensorList(in_ptrs.data(), in_ptrs.size());
  aclTensorList* out_list = aclCreateTensorList(out_ptrs.data(), out_ptrs.size());

  EXEC_ASCEND_CMD({aclnn}, in_list, out_list);

  (void)in_list; (void)out_list;  // owned by in_w/out_w
}}

::std::vector<at::Tensor> {kernel}(at::TensorList self) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  std::vector<at::Tensor> outs;
  outs.reserve(self.size());
  for (const auto& t : self) outs.push_back(at::empty_like(t));
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), at::TensorList(outs).slice(off, n));
  }}
  return outs;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _foreach_div_.ScalarList / _foreach_addcdiv_.ScalarList: per-tensor scalar
# list. aclnn's ScalarList variant takes an aclScalarList (div), while its
# addcdiv counterpart's "scalars" param is (per the header) an aclTensor* —
# both are boxed as a plain list of aclScalar* built from the ArrayRef<Scalar>.
T_FOREACH_INPLACE_DIV_SCALARLIST = """\
static void {kernel}Chunk(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {{
  namespace ascend = at::native::flagos::ascend;

  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(self.size());
  for (const auto& t : self) wrappers.emplace_back(t);
  std::vector<const aclTensor*> acl_tensors;
  acl_tensors.reserve(self.size());
  for (auto& w : wrappers) acl_tensors.push_back(w.get());
  aclTensorList* tensor_list = aclCreateTensorList(acl_tensors.data(), acl_tensors.size());

  // aic-ops-info: ForeachDivScalarList's `scalars` is ALWAYS float32,
  // regardless of x's dtype (same rule as ForeachLerpScalar's weight).
  std::vector<ascend::AclScalarWrapper> scalar_wrappers;
  scalar_wrappers.reserve(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i) {{
    scalar_wrappers.emplace_back(scalars[i], at::kFloat);
  }}
  std::vector<const aclScalar*> acl_scalars;
  acl_scalars.reserve(scalar_wrappers.size());
  for (auto& sw : scalar_wrappers) acl_scalars.push_back(sw.get());
  aclScalarList* scalar_list = aclCreateScalarList(acl_scalars.data(), acl_scalars.size());

  EXEC_ASCEND_CMD({aclnn}, tensor_list, scalar_list, tensor_list);

  aclDestroyScalarList(scalar_list);
  (void)tensor_list;  // aclTensor* owned by wrappers; do not aclDestroyTensorList
}}

void {kernel}(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  TORCH_CHECK(self.size() == scalars.size(), "{disp}: scalars must match tensor list length");
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), scalars.slice(off, n));
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_FOREACH_INPLACE_ADDCDIV_SCALARLIST = """\
static void {kernel}Chunk(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {{
  namespace ascend = at::native::flagos::ascend;

  std::vector<ascend::AclTensorWrapper> self_w, t1_w, t2_w;
  self_w.reserve(self.size());
  t1_w.reserve(tensor1.size());
  t2_w.reserve(tensor2.size());
  for (const auto& t : self) self_w.emplace_back(t);
  for (const auto& t : tensor1) t1_w.emplace_back(t);
  for (const auto& t : tensor2) t2_w.emplace_back(t);

  std::vector<const aclTensor*> self_ptrs, t1_ptrs, t2_ptrs;
  self_ptrs.reserve(self.size());
  t1_ptrs.reserve(tensor1.size());
  t2_ptrs.reserve(tensor2.size());
  for (auto& w : self_w) self_ptrs.push_back(w.get());
  for (auto& w : t1_w) t1_ptrs.push_back(w.get());
  for (auto& w : t2_w) t2_ptrs.push_back(w.get());

  aclTensorList* self_list = aclCreateTensorList(self_ptrs.data(), self_ptrs.size());
  aclTensorList* t1_list = aclCreateTensorList(t1_ptrs.data(), t1_ptrs.size());
  aclTensorList* t2_list = aclCreateTensorList(t2_ptrs.data(), t2_ptrs.size());

  // aclnnForeachAddcdivScalarList's "scalars" param is a plain device aclTensor
  // (1-D, one element per list entry), NOT an aclScalarList -- unlike div's
  // ScalarList variant. Materialize scalars on host in self[0]'s dtype, then
  // move to device once. The dtype MUST match self (a float32 scalars tensor
  // against fp16 inputs returns 161002), which costs up to 1 ulp versus CPU,
  // where the divisor stays a full-precision Scalar.
  at::Tensor scalars_cpu = at::empty({{static_cast<int64_t>(scalars.size())}},
      at::TensorOptions().dtype(self[0].scalar_type()));
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, self[0].scalar_type(),
      "{disp}_scalars", [&] {{
    auto* ptr = scalars_cpu.data_ptr<scalar_t>();
    for (size_t i = 0; i < scalars.size(); ++i) {{
      ptr[i] = scalars[i].to<scalar_t>();
    }}
  }});
  at::Tensor scalars_dev = scalars_cpu.to(self[0].device());
  ascend::AclTensorWrapper acl_scalars(scalars_dev);

  EXEC_ASCEND_CMD({aclnn}, self_list, t1_list, t2_list, acl_scalars.get(), self_list);

  (void)self_list; (void)t1_list; (void)t2_list;  // owned by *_w
}}

void {kernel}(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  TORCH_CHECK(self.size() == tensor1.size() && self.size() == tensor2.size() && self.size() == scalars.size(),
      "{disp}: tensor/scalar lists must match in length");
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), tensor1.slice(off, n), tensor2.slice(off, n),
        scalars.slice(off, n));
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# --- foreach ops needed by torch.nn.utils.clip_grad_* ----------------------
#
# Gradient clipping is near-universal in real training scripts and both of its
# entry points were dead on Ascend:
#   clip_grad_norm_  -> _foreach_norm.Scalar        (aclnnForeachNorm)
#   clip_grad_value_ -> _foreach_clamp_min_.Scalar  (no aclnnForeachClampMin*)
#
# For the second one CANN has no clamp_min spelling at all, but clamp_min is
# elementwise max against a scalar, so aclnnForeachMaximumScalarV2 is exactly
# equivalent. Use the V2 entry: the non-V2 aclnnForeachMaximumScalar declares
# its scalar as `const aclTensor*`, while V2 takes a proper `const aclScalar*`.
#
# Both chunk at FOREACH_CHUNK for the same reason as the AdamW foreach ops --
# aclnn silently drops list entries past 50.

# _foreach_norm.Scalar: (self[], Scalar ord, optional<ScalarType>) -> Tensor[].
#   aclnnForeachNorm(x, scalar, out) with out a list of 0-d per-tensor norms.
# Only ord==2 is supported by CANN's kernel; anything else must not silently
# return a 2-norm, so it is rejected loudly here.
T_FOREACH_NORM = """\
static void {kernel}Chunk(at::TensorList self, at::TensorList outs, const at::Scalar& ord) {{
  namespace ascend = at::native::flagos::ascend;

  std::vector<ascend::AclTensorWrapper> in_w, out_w;
  in_w.reserve(self.size());
  out_w.reserve(outs.size());
  for (const auto& t : self) in_w.emplace_back(t);
  for (const auto& t : outs) out_w.emplace_back(t);

  std::vector<const aclTensor*> in_ptrs, out_ptrs;
  in_ptrs.reserve(self.size());
  out_ptrs.reserve(outs.size());
  for (auto& w : in_w) in_ptrs.push_back(w.get());
  for (auto& w : out_w) out_ptrs.push_back(w.get());

  aclTensorList* in_list = aclCreateTensorList(in_ptrs.data(), in_ptrs.size());
  aclTensorList* out_list = aclCreateTensorList(out_ptrs.data(), out_ptrs.size());
  // aic-ops-info: ForeachNorm's `scalar` is float32 regardless of x's dtype.
  ascend::AclScalarWrapper acl_ord(ord, at::kFloat);

  EXEC_ASCEND_CMD({aclnn}, in_list, acl_ord.get(), out_list);

  (void)in_list; (void)out_list;  // owned by in_w/out_w
}}

::std::vector<at::Tensor> {kernel}(at::TensorList self, const at::Scalar& ord, ::std::optional<at::ScalarType> dtype) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  TORCH_CHECK(ord.toDouble() == 2.0,
      "{disp}: only ord=2 is supported on Ascend (aclnnForeachNorm), got ", ord.toDouble());
  std::vector<at::Tensor> outs;
  outs.reserve(self.size());
  for (const auto& t : self) {{
    auto opts = dtype.has_value() ? t.options().dtype(dtype.value()) : t.options();
    outs.push_back(at::native::flagos::ascend::OpPreparation::apply_tensor_without_format(
        {{}}, opts));
  }}
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), at::TensorList(outs).slice(off, n), ord);
  }}
  return outs;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _foreach_clamp_min_.Scalar: (self[]&, Scalar) -> void, self = max(self, scalar).
#   aclnnForeachMaximumScalarV2(x, scalar, out=x)
T_FOREACH_INPLACE_MAXIMUM_SCALAR = """\
static void {kernel}Chunk(at::TensorList self, const at::Scalar& scalar) {{
  namespace ascend = at::native::flagos::ascend;

  std::vector<ascend::AclTensorWrapper> wrappers;
  wrappers.reserve(self.size());
  for (const auto& t : self) wrappers.emplace_back(t);

  std::vector<const aclTensor*> acl_tensors;
  acl_tensors.reserve(self.size());
  for (auto& w : wrappers) acl_tensors.push_back(w.get());

  aclTensorList* tensor_list = aclCreateTensorList(acl_tensors.data(), acl_tensors.size());
  // Same bf16 carve-out as ForeachMulScalar/ForeachAddScalar: no bf16 scalar
  // entry in aic-ops-info, so a bf16 x needs its scalar as float32.
  auto scalar_dtype = self[0].scalar_type() == at::kBFloat16 ? at::kFloat : self[0].scalar_type();
  ascend::AclScalarWrapper acl_scalar(scalar, scalar_dtype);

  EXEC_ASCEND_CMD({aclnn}, tensor_list, acl_scalar.get(), tensor_list);

  (void)tensor_list;  // aclTensor* owned by wrappers; do not aclDestroyTensorList
}}

void {kernel}(at::TensorList self, const at::Scalar& scalar) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), scalar);
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _foreach_mul_.Tensor: (self[]&, Tensor other) -> void, every entry scaled by
# the SAME single `other`. clip_grad_norm_ ends here: it computes one clip
# coefficient tensor and multiplies every gradient by it.
#
# aclnnForeachMulList requires x2[i] to have the SAME SHAPE as x1[i] ("The
# input 1 shape should be same with input 0"), so a list of n aliases of a 0-d
# `other` is rejected. Broadcast `other` to each entry's shape first. The
# expand is a stride-0 view, but aclnn needs dense memory, so it is
# materialized -- one temporary per entry, sized like that entry.
#
# `other` is a 1-element device tensor in the clip_grad_norm_ path, which is
# the only caller that matters here; a non-scalar `other` would broadcast the
# same way.
T_FOREACH_INPLACE_MUL_TENSOR = """\
static void {kernel}Chunk(at::TensorList self, const at::Tensor& other) {{
  namespace ascend = at::native::flagos::ascend;

  // aclnn compares x1[i]/x2[i] shapes elementwise, so materialize `other`
  // at each entry's shape rather than aliasing one 0-d tensor n times.
  std::vector<at::Tensor> others;
  others.reserve(self.size());
  for (const auto& t : self) {{
    auto o = other.scalar_type() == t.scalar_type() ? other : other.to(t.scalar_type());
    others.push_back(o.sizes().equals(t.sizes()) ? o.contiguous()
                                                 : o.expand(t.sizes()).contiguous());
  }}

  std::vector<ascend::AclTensorWrapper> self_w, other_w;
  self_w.reserve(self.size());
  other_w.reserve(others.size());
  for (const auto& t : self) self_w.emplace_back(t);
  for (const auto& t : others) other_w.emplace_back(t);

  std::vector<const aclTensor*> self_ptrs, other_ptrs;
  self_ptrs.reserve(self.size());
  other_ptrs.reserve(others.size());
  for (auto& w : self_w) self_ptrs.push_back(w.get());
  for (auto& w : other_w) other_ptrs.push_back(w.get());

  aclTensorList* self_list = aclCreateTensorList(self_ptrs.data(), self_ptrs.size());
  aclTensorList* other_list = aclCreateTensorList(other_ptrs.data(), other_ptrs.size());

  EXEC_ASCEND_CMD({aclnn}, self_list, other_list, self_list);

  (void)self_list; (void)other_list;  // owned by self_w/other_w
}}

void {kernel}(at::TensorList self, const at::Tensor& other) {{
  TORCH_CHECK(!self.empty(), "{disp}: expected a non-empty list of tensors");
  for (size_t off = 0; off < self.size(); off += {chunk}) {{
    size_t n = std::min<size_t>({chunk}, self.size() - off);
    {kernel}Chunk(self.slice(off, n), other);
  }}
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# --- factory ops: at::empty(...) on the PrivateUse1 device + device-side fill ---
# These build TensorOptions on-host then fill via zero_/fill_, which are themselves
# device-side aclnn kernels (aclnnInplaceZero / aclnnInplaceFillScalar), so the
# whole op stays on-device with no h2d. No aclnn override (fill is the dispatcher).

# zeros: (IntArrayRef size, dtype?, layout?, device?, pin?) -> zero tensor.
T_ZEROS = """\
at::Tensor {kernel}(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(at::kFloat))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
    .pinned_memory(pin_memory.value_or(false));
  auto result = at::empty(size, options);
  result.zero_();
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# scalar_tensor: (Scalar s, dtype?, layout?, device?, pin?) -> 0-dim tensor filled s.
T_SCALAR_TENSOR = """\
at::Tensor {kernel}(const at::Scalar& s, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(at::ScalarType::Float))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
    .pinned_memory(pin_memory.value_or(false));
  auto result = at::empty({{}}, options);
  result.fill_(s);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# ones_like: (self, dtype?, layout?, device?, pin?, memory_format?) -> ones w/ self's meta.
T_ONES_LIKE = """\
at::Tensor {kernel}(const at::Tensor& self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(self.scalar_type()))
    .layout(layout.value_or(self.layout()))
    .device(device.value_or(self.device()))
    .pinned_memory(pin_memory.value_or(false));
  auto fmt = memory_format.value_or(at::MemoryFormat::Contiguous);
  // Preserve replicates self's strides when it can; see T_EMPTY_LIKE for why
  // suggest_memory_format() is not a valid stand-in.
  auto result = (fmt == at::MemoryFormat::Preserve && self.is_non_overlapping_and_dense())
    ? at::empty_strided(self.sizes(), self.strides(), options)
    : at::empty(self.sizes(), options,
                fmt == at::MemoryFormat::Preserve ? self.suggest_memory_format() : fmt);
  result.fill_(1);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# zeros_like: (self, dtype?, layout?, device?, pin?, memory_format?) -> zeros w/ self's meta.
#   Identical to ones_like but fills 0. Used by optimizers (Adam exp_avg state).
T_ZEROS_LIKE = """\
at::Tensor {kernel}(const at::Tensor& self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(self.scalar_type()))
    .layout(layout.value_or(self.layout()))
    .device(device.value_or(self.device()))
    .pinned_memory(pin_memory.value_or(false));
  auto fmt = memory_format.value_or(at::MemoryFormat::Contiguous);
  // Preserve replicates self's strides when it can; see T_EMPTY_LIKE for why
  // suggest_memory_format() is not a valid stand-in.
  auto result = (fmt == at::MemoryFormat::Preserve && self.is_non_overlapping_and_dense())
    ? at::empty_strided(self.sizes(), self.strides(), options)
    : at::empty(self.sizes(), options,
                fmt == at::MemoryFormat::Preserve ? self.suggest_memory_format() : fmt);
  result.zero_();
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# empty_like: (self, dtype?, layout?, device?, pin?, memory_format?) -> uninit tensor
#   with self's meta. Same shape as ones_like but no fill_ (contents undefined).
#   FlagGems' pointwise_dynamic allocates its outputs via torch.empty_like, so this
#   must exist on the ascend backend for any op routed to flagos_python.
#
#   Preserve must replicate self's strides, not fall back to suggest_memory_format().
#   For a transposed 2D tensor suggest_memory_format() answers Contiguous, so the
#   output came back row-major while gems' pointwise fast path -- taken whenever the
#   operands are non-overlapping and dense (use_fast_path in
#   utils/pointwise_dynamic.py) -- writes elements in the INPUT's physical order into
#   a flat view of it. Every pointwise gems kernel then returned a transposed-wrong
#   result for a transposed input (neg/abs/sin/exp/sqrt/sigmoid/tanh/rsqrt measured
#   98% of elements wrong); a strided slice was unaffected because it is not dense
#   and takes the slow path. Mirrors upstream at::native::empty_like.
T_EMPTY_LIKE = """\
at::Tensor {kernel}(const at::Tensor& self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(self.scalar_type()))
    .layout(layout.value_or(self.layout()))
    .device(device.value_or(self.device()))
    .pinned_memory(pin_memory.value_or(false));
  auto fmt = memory_format.value_or(at::MemoryFormat::Preserve);
  if (fmt == at::MemoryFormat::Preserve) {{
    if (self.is_non_overlapping_and_dense()) {{
      return at::empty_strided(self.sizes(), self.strides(), options);
    }}
    fmt = self.suggest_memory_format();
  }}
  return at::empty(self.sizes(), options, fmt);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# full: (IntArrayRef size, Scalar fill, dtype?, layout?, device?, pin?) -> filled tensor.
#   Default dtype: if a fill value is integral and no dtype given, torch uses long;
#   but transformers' generate always passes an explicit dtype, and value_or(kFloat)
#   matches zeros/ones behaviour, so keep it simple and consistent with T_ZEROS.
T_FULL = """\
at::Tensor {kernel}(at::IntArrayRef size, const at::Scalar& fill, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(at::kFloat))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
    .pinned_memory(pin_memory.value_or(false));
  auto result = at::empty(size, options);
  result.fill_(fill);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# full_like: (self, Scalar fill, dtype?, layout?, device?, pin?, memory_format?) -> self-shaped, filled.
T_FULL_LIKE = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& fill, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(self.scalar_type()))
    .layout(layout.value_or(self.layout()))
    .device(device.value_or(self.device()))
    .pinned_memory(pin_memory.value_or(false));
  auto fmt = memory_format.value_or(at::MemoryFormat::Preserve);
  if (fmt == at::MemoryFormat::Preserve) {{
    fmt = self.suggest_memory_format();
  }}
  auto result = at::empty(self.sizes(), options, fmt);
  result.fill_(fill);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# ones: (IntArrayRef size, dtype?, layout?, device?, pin?) -> tensor of ones.
#   Same shape as T_ZEROS; fill_(1) instead of zero_(). transformers' generate()
#   uses torch.ones(batch_size, device=...) for unfinished_sequences bookkeeping.
T_ONES = """\
at::Tensor {kernel}(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(at::kFloat))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
    .pinned_memory(pin_memory.value_or(false));
  auto result = at::empty(size, options);
  result.fill_(1);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# new_ones: (self, IntArrayRef size, dtype?, layout?, device?, pin?) -> ones w/ self's meta.
T_NEW_ONES = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {{
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(self.scalar_type()))
    .layout(layout.value_or(self.layout()))
    .device(device.value_or(self.device()))
    .pinned_memory(pin_memory.value_or(false));
  auto result = at::empty(size, options);
  result.fill_(1);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gemm_addmv: (self, mat(n,m), vec(m), beta, alpha) -> (n,). NOTE the aclnn arg
#   order is (self, mat, vec, ALPHA, BETA) -- alpha before beta, unlike addmm.
T_GEMM_ADDMV = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  std::vector<int64_t> out_shape = {{mat.size(0)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mat(mat);
  ascend::AclTensorWrapper acl_vec(vec);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_mat.get(), acl_vec.get(), acl_alpha.get(), acl_beta.get(), acl_out.get(), cube_math_type);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gemm_addr: (self, vec1(n), vec2(m), beta, alpha) -> (n,m) outer product.
#   aclnnAddr(self, vec1, vec2, beta, alpha, out) -- no cubeMathType.
T_GEMM_ADDR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& vec1, const at::Tensor& vec2, const at::Scalar& beta, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape = {{vec1.size(0), vec2.size(0)}};
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_vec1(vec1);
  ascend::AclTensorWrapper acl_vec2(vec2);
  ascend::AclScalarWrapper acl_beta(beta, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_vec1.get(), acl_vec2.get(), acl_beta.get(), acl_alpha.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# bce: (self, target, optional weight, int reduction) -> Tensor.
#   aclnnBinaryCrossEntropy(self, target, weight, reduction, out).
#   reduction 0=none -> out=self.shape; 1=mean/2=sum -> scalar. weight may be
#   undefined -> AclTensorWrapper yields nullptr (aclnn treats as absent).
T_BCE = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& target, const ::std::optional<at::Tensor>& weight, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar for mean/sum
  if (reduction == 0) out_shape = self.sizes().vec();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_target.get(), acl_weight.get(), reduction, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# bce_backward: (grad_output, self, target, optional weight, int reduction).
#   aclnnBinaryCrossEntropyBackward(grad, self, target, weight, reduction, out).
#   grad_input = self.shape.
T_BCE_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, const ::std::optional<at::Tensor>& weight, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(self.sizes(), self.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_self.get(), acl_target.get(), acl_weight.get(), reduction, acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# bce_logits: (self, target, optional weight, optional pos_weight, reduction).
#   aclnnBinaryCrossEntropyWithLogits(self, target, weight, posWeight, reduction, out).
T_BCE_LOGITS = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& target, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& pos_weight, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> out_shape;   // scalar for mean/sum
  if (reduction == 0) out_shape = self.sizes().vec();
  auto out = ascend::OpPreparation::apply_tensor_without_format(out_shape, self.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor pos_weight_t = pos_weight.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_pos_weight(pos_weight_t);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_target.get(), acl_weight.get(), acl_pos_weight.get(), reduction, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_layer_norm: (input, IntArrayRef normalized_shape, optional weight,
#   optional bias, double eps) -> tuple(out, mean, rstd).
#   aclnn<Name>(input, normShape, weight, bias, eps, out, meanOut, rstdOut)
# out = input shape; mean/rstd = input.shape[:begin_axis] + 1s, where
# begin_axis = input.dim() - normalized_shape.size(). weight/bias may be
# undefined -> AclTensorWrapper yields nullptr (aclnn treats as absent).
T_LAYER_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, double eps) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t begin_axis = input.dim() - static_cast<int64_t>(normalized_shape.size());
  auto stat_shape = input.sizes().vec();
  for (int64_t i = begin_axis; i < input.dim(); ++i) stat_shape[i] = 1;

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto mean = ascend::OpPreparation::apply_tensor_without_format(
      stat_shape, input.options());
  auto rstd = ascend::OpPreparation::apply_tensor_without_format(
      stat_shape, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_input(input);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);

  std::vector<int64_t> ns(normalized_shape.begin(), normalized_shape.end());
  ascend::AclIntArrayWrapper acl_ns(ns);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_ns.get(), acl_weight.get(), acl_bias.get(), eps, acl_out.get(), acl_mean.get(), acl_rstd.get());
  return std::make_tuple(out, mean, rstd);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_group_norm: (input, optional weight, optional bias, int64 N, int64 C,
#   int64 HxW, int64 group, double eps) -> tuple(out, mean, rstd).
#   aclnn<Name>(self, gamma, beta, N, C, HxW, group, eps, out, meanOut, rstdOut)
# out = input shape; mean/rstd = (N, group).
T_GROUP_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& input, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto mean = ascend::OpPreparation::apply_tensor_without_format(
      {{N, group}}, input.options());
  auto rstd = ascend::OpPreparation::apply_tensor_without_format(
      {{N, group}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_input(input);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_weight.get(), acl_bias.get(), N, C, HxW, group, eps, acl_out.get(), acl_mean.get(), acl_rstd.get());
  return std::make_tuple(out, mean, rstd);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gelu: (self, c10::string_view approximate) -> Tensor. NOTE: aclnnGelu (v1)
# hardcodes the *tanh* approximation, but PyTorch's default is approximate=
# "none" (erf form, used by qwen3 et al). So we use aclnnGeluV2, whose
# int64_t approximate selects 0="none"/1="tanh" (int is varargs-safe). out =
# self shape/dtype.
T_GELU = """\
at::Tensor {kernel}(const at::Tensor& self, c10::string_view approximate) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t approx = (approximate == "tanh") ? 1 : 0;
  TORCH_CHECK(approximate == "none" || approximate == "tanh",
      "gelu: unsupported approximate='", approximate, "'");
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), approx, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gelu_backward: (grad_output, self, approximate) -> Tensor. aclnnGeluBackwardV2
# takes the approximation as a `char*` string ("none"/"tanh"); a pointer is
# varargs-safe. grad_input = self shape. (v1 aclnnGeluBackward is tanh-only,
# same mismatch as forward.)
T_GELU_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(approximate == "none" || approximate == "tanh",
      "gelu_backward: unsupported approximate='", approximate, "'");
  std::string approx_str(approximate);
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_self.get(), approx_str.data(), acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _log_softmax: (self, int64 dim, bool half_to_float) -> Tensor. Mirrors the
# handwritten softmax.cc: aclnn<Name>(self, dim, out); half_to_float promotes
# the output dtype to float. out = self shape.
T_LOG_SOFTMAX = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), dim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _softmax_backward_data / _log_softmax_backward_data:
#   (grad_output, output, int64 dim, at::ScalarType input_dtype) -> Tensor.
#   aclnn<Name>(gradOutput, output, dim, gradInput). grad_input = grad_output
#   shape, dtype = input_dtype (the dtype the forward input had).
T_SOFTMAX_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      grad_output.sizes(), grad_output.options().dtype(input_dtype));

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_output(output);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_output.get(), dim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# binary_scalar: (Tensor, Scalar) -> same shape/dtype. Scalar op, NO alpha.
#   aclnn<Name>(self, scalar, out)   e.g. mul.Scalar->aclnnMuls, div.Scalar->aclnnDivs
# (aclnn headers for Muls/Divs are absent but the symbols exist; arg marshaling
#  confirmed from the handwritten mul_scalar.cc / div_scalar.cc.)
T_BINARY_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# act_backward_self: (grad_output, self) -> grad_input. Differs from act_backward
#   (grad, output) in taking `self` as the second tensor. e.g. silu_backward.
#   aclnn<Name>(gradOutput, self, gradInput)
T_ACT_BACKWARD_SELF = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_grad_input(grad_input);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# where: where.self(cond, self, other) -> broadcast(cond, self, other), self dtype.
#   aclnn<Name>(condition, self, other, out). aclnn does not broadcast, so all
#   three operands are expanded+contiguous to the common shape.
T_WHERE = """\
at::Tensor {kernel}(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = at::infer_size(self.sizes(), other.sizes());
  out_shape = at::infer_size(condition.sizes(), out_shape);

  auto cond_b = condition.expand(out_shape).contiguous();
  auto self_b = self.expand(out_shape).contiguous();
  auto other_b = other.expand(out_shape).contiguous();

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_cond(cond_b);
  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_other(other_b);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_cond.get(), acl_self.get(), acl_other.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# clamp: (self, Scalar? min, Scalar? max) -> Tensor, self's shape/dtype.
#   aclnnClamp(self, clipValueMin, clipValueMax, out). Either bound may be
#   absent (torch allows min=None or max=None, just not both); AclScalarWrapper's
#   default ctor leaves the acl_scalar null, which aclnn reads as "not supplied".
T_CLAMP = """\
at::Tensor {kernel}(const at::Tensor& self, const ::std::optional<at::Scalar>& min, const ::std::optional<at::Scalar>& max) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_min = min.has_value()
      ? ascend::AclScalarWrapper(min.value(), self.scalar_type())
      : ascend::AclScalarWrapper();
  ascend::AclScalarWrapper acl_max = max.has_value()
      ? ascend::AclScalarWrapper(max.value(), self.scalar_type())
      : ascend::AclScalarWrapper();
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_min.get(), acl_max.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# clamp.Tensor: (self, Tensor? min, Tensor? max) -> Tensor, broadcast shape.
#   aclnnClampTensor(self, minT, maxT, out). AclTensorWrapper already maps an
#   undefined at::Tensor to a null aclTensor*, matching an absent bound.
T_CLAMP_TENSOR = """\
at::Tensor {kernel}(const at::Tensor& self, const ::std::optional<at::Tensor>& min, const ::std::optional<at::Tensor>& max) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = self.sizes().vec();
  if (min.has_value()) out_shape = at::infer_size(out_shape, min.value().sizes());
  if (max.has_value()) out_shape = at::infer_size(out_shape, max.value().sizes());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_min(min.value_or(at::Tensor()));
  ascend::AclTensorWrapper acl_max(max.value_or(at::Tensor()));
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_min.get(), acl_max.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# softmax_fwd: _softmax(self, int64 dim, bool half_to_float) -> same shape.
#   aclnn<Name>(self, dim, out). half_to_float promotes the output dtype to float.
T_SOFTMAX_FWD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), dim, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# softmax_fwd CACHED: same as T_SOFTMAX_FWD but through the repeatable-executor
# cache. `dim` is baked into the executor at GetWorkspaceSize, so it MUST be in
# the key. half_to_float only changes the output dtype, which is already part of
# the out-tensor signature, but fold it in too for safety. Uncached softmax was
# measured at a FLAT ~38us/call regardless of shape (pure GetWorkspaceSize +
# aclCreateTensor build cost) vs ~14us on torch_npu; the decode attention shape
# is fixed so caching drops it to the aclnn-execute floor.
T_SOFTMAX_FWD_CACHED = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(out_dtype));

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self); hsh.val(dim);
  {{ int8_t h2f = half_to_float ? 1 : 0; hsh.val(h2f); }}
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self}}, {{&out}},
      [&](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
          std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, dim, out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_all: all(self) -> bool scalar over ALL elements. aclnnAll reduces along
#   a dim list, so flatten to 1-D and reduce dim=0 to a 0-d bool out.
#   aclnn<Name>(self_flat, dim_list, keepdim=false, out)
T_REDUCE_ALL = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto input = self.contiguous().reshape({{-1}});
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {{}}, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(input);
  ascend::AclTensorWrapper acl_out(out);

  int64_t dim_val = 0;
  std::vector<int64_t> dims{{dim_val}};
  ascend::AclIntArrayWrapper acl_dim(dims);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), false, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Shared prologue for the dtype-aware reduce categories (sum.dim_IntList /
# mean.dim). OptionalIntArrayRef dim (None/empty = reduce all) + optional dtype.
# Mirrors the handwritten sum.cc dim-normalization and reduced-shape logic.
_REDUCE_DTYPE_PROLOGUE = """\
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = dtype.has_value() ? dtype.value() : self.scalar_type();
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  if (dim.has_value() && !dim.value().empty()) {{
    for (int64_t d : dim.value()) norm_dims.push_back(d < 0 ? d + ndim : d);
  }} else {{
    for (int64_t d = 0; d < ndim; ++d) norm_dims.push_back(d);
  }}
  auto out_shape = self.sizes().vec();
  std::vector<int64_t> sorted_dims(norm_dims);
  std::sort(sorted_dims.rbegin(), sorted_dims.rend());
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(out_dtype));
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(norm_dims);
"""

# reduce_sum_dtype: sum.dim_IntList(self, int[]? dim, keepdim, ScalarType? dtype).
#   aclnnReduceSum(self, dims, keepdim, aclDataType, out)
T_REDUCE_SUM_DTYPE = (
    """\
at::Tensor {kernel}(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {{
"""
    + _REDUCE_DTYPE_PROLOGUE
    + """\
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# reduce_sum_dtype CACHED. dims/keepdim/out_dtype are baked into the executor at
# build (aclnnReduceSum reads them during GetWorkspaceSize), so all three go in
# the cache key alongside the input tensor signature.
T_REDUCE_SUM_DTYPE_CACHED = (
    """\
at::Tensor {kernel}(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {{
"""
    + _REDUCE_DTYPE_PROLOGUE
    + """\
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self);
  for (int64_t d : norm_dims) hsh.val(d);
  hsh.val(keepdim);
  {{ int32_t dtk = static_cast<int32_t>(acl_dtype); hsh.val(dtk); }}
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self}}, {{&out}},
      [&](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
          std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, acl_dim.get(), keepdim, acl_dtype, out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# reduce_sum_all: sum(self, ScalarType? dtype) -> full reduction to a 0-d tensor.
#   Reuses aclnnReduceSum over every axis with keepdim=false. transformers'
#   fast_all() calls tensor.sum() on the causal-mask bool tensor.
T_REDUCE_SUM_ALL = """\
at::Tensor {kernel}(const at::Tensor& self, std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  // Integral/bool inputs promote to int64 when no dtype given (matches torch).
  at::ScalarType out_dtype = dtype.has_value()
      ? dtype.value()
      : (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)
             ? at::kLong : self.scalar_type());
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  for (int64_t d = 0; d < ndim; ++d) norm_dims.push_back(d);
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {{}}, self.options().dtype(out_dtype));
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(norm_dims);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), false, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# mean_all: mean(self, ScalarType? dtype) -> full reduction to a 0-d tensor.
#   aclnnMean(self, dims=all, keepdim=false, aclDataType, out). Unlike sum,
#   mean does NOT integer-promote (undefined for int/bool without a given
#   dtype); out_dtype defaults to self's own float dtype.
T_MEAN_ALL = """\
at::Tensor {kernel}(const at::Tensor& self, std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  at::ScalarType out_dtype = dtype.has_value() ? dtype.value() : self.scalar_type();
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  for (int64_t d = 0; d < ndim; ++d) norm_dims.push_back(d);
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {{}}, self.options().dtype(out_dtype));
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclIntArrayWrapper acl_dim(norm_dims);
  aclDataType acl_dtype = ascend::ToAclDataType(out_dtype);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), false, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_minmax_all: max(self) / min(self) -> 0-d tensor over ALL elements.
#   aclnn<Max/Min>(self, out); out keeps self's dtype. transformers' generate()
#   loop calls unfinished_sequences.max() to test the stop condition.
T_REDUCE_MINMAX_ALL = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      {{}}, self.options());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# reduce_mean_dtype: mean.dim(self, int[]? dim, keepdim, ScalarType? dtype).
#   aclnnMeanV2(self, dims, keepdim, int32 dtype, out)  -- MeanV2 for CANN 8.5.
T_REDUCE_MEAN_DTYPE = (
    """\
at::Tensor {kernel}(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {{
"""
    + _REDUCE_DTYPE_PROLOGUE
    + """\
  auto acl_dtype = static_cast<int32_t>(ascend::ToAclDataType(out_dtype));

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_dim.get(), keepdim, acl_dtype, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# reduce_mean_dtype CACHED. Like the sum variant: dims/keepdim/dtype baked into
# the executor, so keyed on all three. mean.dim is 113/step in RMSNorm variance
# (measured 57us uncached) -- one of the largest remaining host lines.
T_REDUCE_MEAN_DTYPE_CACHED = (
    """\
at::Tensor {kernel}(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {{
"""
    + _REDUCE_DTYPE_PROLOGUE
    + """\
  auto acl_dtype = static_cast<int32_t>(ascend::ToAclDataType(out_dtype));

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self);
  for (int64_t d : norm_dims) hsh.val(d);
  hsh.val(keepdim);
  hsh.val(acl_dtype);
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self}}, {{&out}},
      [&](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
          std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, acl_dim.get(), keepdim, acl_dtype, out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# ==========================================================================
# conv / pool family. These need an explicit output-shape formula (aclnn wants
# the output pre-allocated), so each carries a small shape helper in its body.
# Shared pooling out-dim formula (matches PyTorch): for ceil_mode the division
# rounds up, else down; a ceil-mode start beyond padding is clamped back.
# ==========================================================================
_POOL_OUT_DIM = """\
  auto pool_out_dim = [](int64_t in, int64_t k, int64_t s, int64_t p,
                         int64_t d, bool ceil) -> int64_t {{
    int64_t num = in + 2 * p - d * (k - 1) - 1;
    int64_t out = (ceil ? (num + s - 1) / s : num / s) + 1;
    if (ceil && (out - 1) * s >= in + p) out -= 1;   // last window all-padding
    return out;
  }};
"""

# adaptive_avg_pool2d: (self, SymInt[2] output_size) -> Tensor.
#   out shape = self leading dims + output_size. aclnn<Name>(self, outSize, out).
T_ADAPTIVE_AVG_POOL2D = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef output_size) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = self.sizes().vec();
  int64_t r = out_shape.size();
  out_shape[r - 2] = output_size[0];
  out_shape[r - 1] = output_size[1];

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  // aclnn pooling rejects ND 4-D; tag as NCHW.
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclIntArrayWrapper acl_osize(output_size);
  ascend::AclTensorWrapper acl_out(out, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_osize.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# avg_pool2d: (self, kernel_size, stride, padding, ceil_mode, count_include_pad,
#   divisor_override) -> Tensor. stride defaults to kernel_size when empty.
#   aclnnAvgPool2d(self, k, s, p, ceil, countPad, divOverride, cubeMathType, out).
T_AVG_POOL2D = (
    """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());
"""
    + _POOL_OUT_DIM
    + """\
  auto out_shape = self.sizes().vec();
  int64_t r = out_shape.size();
  out_shape[r - 2] = pool_out_dim(self.size(r - 2), k[0], s[0], p[0], 1, ceil_mode);
  out_shape[r - 1] = pool_out_dim(self.size(r - 1), k[1], s[1], p[1], 1, ceil_mode);

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  int64_t div_override = divisor_override.value_or(0);
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclTensorWrapper acl_out(out, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_k.get(), acl_s.get(), acl_p.get(),
      ceil_mode, count_include_pad, div_override, (int8_t)0, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# max_pool2d_with_indices: (self, k, stride, padding, dilation, ceil_mode) ->
#   tuple(out, int64 indices). stride defaults to kernel_size when empty.
#   aclnn<Name>(self, k, s, p, dil, ceil, out, indices).
T_MAX_POOL2D_INDICES = (
    """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());
  std::vector<int64_t> dil(dilation.begin(), dilation.end());
"""
    + _POOL_OUT_DIM
    + """\
  auto out_shape = self.sizes().vec();
  int64_t r = out_shape.size();
  out_shape[r - 2] = pool_out_dim(self.size(r - 2), k[0], s[0], p[0], dil[0], ceil_mode);
  out_shape[r - 1] = pool_out_dim(self.size(r - 1), k[1], s[1], p[1], dil[1], ceil_mode);

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclIntArrayWrapper acl_dil(dil);
  ascend::AclTensorWrapper acl_out(out);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_k.get(), acl_s.get(), acl_p.get(),
      acl_dil.get(), ceil_mode, acl_out.get(), acl_indices.get());
  return std::make_tuple(out, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# masked_fill.Scalar: (self, mask, value) -> Tensor (= self broadcast mask).
#   aclnn only ships the INPLACE variant (aclnnInplaceMaskedFillScalar), so
#   clone self (broadcast to mask if needed) and fill in place. mask is bool.
T_MASKED_FILL_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = at::infer_size(self.sizes(), mask.sizes());
  // avoid clone()/empty_like (not registered for ascend): alloc + copy_.
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  out.copy_(self.expand(out_shape));
  auto mask_b = mask.expand(out_shape).contiguous();

  ascend::AclTensorWrapper acl_self(out);
  ascend::AclTensorWrapper acl_mask(mask_b);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());

  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_mask.get(),
      acl_value.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# masked_fill.Tensor: (self, mask, value) -> Tensor. value is a 0-dim tensor;
#   coerce to self's device/dtype (CPU scalar-tensor path, cf. binary prologue).
T_MASKED_FILL_TENSOR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_shape = at::infer_size(self.sizes(), mask.sizes());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  out.copy_(self.expand(out_shape));
  auto mask_b = mask.expand(out_shape).contiguous();
  auto value_c = value.is_privateuseone()
      ? (value.scalar_type() == self.scalar_type() ? value : value.to(self.scalar_type()))
      : value.to(self.options());

  ascend::AclTensorWrapper acl_self(out);
  ascend::AclTensorWrapper acl_mask(mask_b);
  ascend::AclTensorWrapper acl_value(value_c);

  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_mask.get(),
      acl_value.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# gather: (self, dim, index, sparse_grad) -> Tensor (= index shape, self dtype).
#   aclnnGather(self, dim, index, out). dim normalized for negatives.
T_GATHER = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index, bool sparse_grad) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      index.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_index(index);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_index.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# index_select: (self, dim, index) -> Tensor (= self shape w/ dim replaced by
#   index.numel()). index is 1-D. aclnnIndexSelect(self, dim, index, out).
T_INDEX_SELECT = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
  out_shape[d] = index.numel();
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_index(index);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), d, acl_index.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# zero_: (self) -> self&, in-place. aclnnInplaceZero(selfRef). selfRef is both
#   input and output; return the same tensor. Factory ops (zeros/*_like) build
#   the storage then call this, so it must be device-side aclnn (no h2d).
T_INPLACE_ZERO = """\
at::Tensor& {kernel}(at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()));
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# fill_.Scalar: (self, value) -> self&, in-place. aclnnInplaceFillScalar(
#   selfRef, value). Scalar coerced to self's dtype.
T_INPLACE_FILL_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# fill_.Tensor: (self, value) -> self&, in-place. aclnnInplaceFillTensor(
#   selfRef, value). value is a 0-dim tensor; coerce to self device/dtype
#   (CPU scalar-tensor path, cf. masked_fill.Tensor prologue).
T_INPLACE_FILL_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto value_c = value.is_privateuseone()
      ? (value.scalar_type() == self.scalar_type() ? value : value.to(self.scalar_type()))
      : value.to(self.options());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_value(value_c);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# add_.Tensor: (self&, other, alpha) -> self&, in-place self += alpha*other.
#   aclnnInplaceAdd(selfRef, other, alpha). other may broadcast against self and
#   is coerced to self's device/dtype (mirrors the out-of-place add prologue).
T_INPLACE_ADD_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  auto other_c = other.is_privateuseone()
      ? (other.scalar_type() == self.scalar_type() ? other : other.to(self.scalar_type()))
      : other.to(self.options());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_other(other_c);
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_other.get(),
      acl_alpha.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# mul_.Tensor: (self&, other) -> self&, in-place self *= other.
#   aclnnInplaceMul(selfRef, other). other coerced to self device/dtype.
T_INPLACE_MUL_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto other_c = other.is_privateuseone()
      ? (other.scalar_type() == self.scalar_type() ? other : other.to(self.scalar_type()))
      : other.to(self.options());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_other(other_c);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_other.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# div_.Tensor: (self&, other) -> self&, in-place self /= other.
#   aclnnInplaceDiv(selfRef, other). Same shape as mul_.Tensor.
T_INPLACE_DIV_TENSOR = T_INPLACE_MUL_TENSOR

# mul_.Scalar: (self&, other) -> self&, in-place self *= scalar.
#   aclnnInplaceMuls(selfRef, aclScalar).
T_INPLACE_MUL_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& other) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_other.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# add_.Scalar: (self&, other, alpha) -> self&, in-place self += alpha*scalar.
#   aclnnInplaceAdds(selfRef, otherScalar, alphaScalar).
T_INPLACE_ADD_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_other.get(),
      acl_alpha.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# addcmul_ / addcdiv_: (self&, tensor1, tensor2, value) -> self&, in-place
#   self += value * (tensor1 {{*,/}} tensor2).
#   aclnn<Name>(selfRef, tensor1, tensor2, value). tensor1/tensor2 coerced to
#   self's dtype when they live on device.
T_INPLACE_ADDCMUL = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto t1 = tensor1.scalar_type() == self.scalar_type() ? tensor1 : tensor1.to(self.scalar_type());
  auto t2 = tensor2.scalar_type() == self.scalar_type() ? tensor2 : tensor2.to(self.scalar_type());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_t1(t1);
  ascend::AclTensorWrapper acl_t2(t2);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_t1.get(),
      acl_t2.get(), acl_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_INPLACE_ADDCDIV = T_INPLACE_ADDCMUL

# lerp_.Scalar: (self&, end, weight) -> self&, in-place self += weight*(end-self).
#   aclnnInplaceLerps(selfRef, end, weightScalar). end coerced to self dtype.
T_INPLACE_LERP_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight) {{
  namespace ascend = at::native::flagos::ascend;
  auto end_c = end.scalar_type() == self.scalar_type() ? end : end.to(self.scalar_type());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_end(end_c);
  ascend::AclScalarWrapper acl_weight(weight, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_end.get(),
      acl_weight.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# sqrt_: (self&) -> self&, in-place self = sqrt(self). aclnnInplaceSqrt(selfRef).
T_INPLACE_SQRT = """\
at::Tensor& {kernel}(at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()));
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# --------------------------------------------------------------------------
# Generic in-place families.
#
# aten spells almost every arithmetic op twice -- out-of-place (`neg`) and
# in-place (`neg_`) -- and routing only the first half is a trap: the failure
# surfaces at runtime on the first call ("neg_: backend not registered"), never
# at build time, and it silently pushes composed kernels into allocating a
# temporary + writing it back. CANN ships 138 `aclnnInplace*` entry points, so
# the second half is mostly a table entry rather than new code.
#
# All of them share the same contract: mutate `selfRef` and return `self&`. The
# `const_cast` is what every existing in-place template does -- some aclnn
# headers declare selfRef `const aclTensor*` and some `aclTensor*`, and
# EXEC_ASCEND_CMD goes through a variadic pointer either way.
# --------------------------------------------------------------------------

# inplace_unary: (self&) -> self&. aclnnInplace<Name>(selfRef).
#   Same shape as T_INPLACE_SQRT; named generically since ~35 ops use it.
T_INPLACE_UNARY = """\
at::Tensor& {kernel}(at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()));
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_unary_scalar: (self&, Scalar) -> self&. aclnnInplace<Name>(selfRef, s).
#   Covers eq_/ne_/lt_/gt_/le_/ge_.Scalar, clamp_max_, celu_, leaky_relu_,
#   fmod_.Scalar, div_.Scalar, bitwise_*_.Scalar.
T_INPLACE_UNARY_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& other) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_other(other, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_other.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_binary_tensor: (self&, other) -> self&. aclnnInplace<Name>(selfRef, other).
#   `other` is coerced to self's device/dtype for the same reason the
#   out-of-place binary prologue does it: the .Tensor overloads are what a
#   python scalar lowers to, so `other` may be a 0-dim CPU tensor, and building
#   an aclTensor over host storage yields garbage.
#
#   Comparison ops (eq_/ne_/lt_/...) are in this family too. Their aten result
#   dtype is bool, but the IN-PLACE spelling writes back into `self`, so aten
#   itself requires self to already be bool -- no dtype juggling needed here.
T_INPLACE_BINARY_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& other) {{
  namespace ascend = at::native::flagos::ascend;
  auto other_c = other.is_privateuseone()
      ? (other.scalar_type() == self.scalar_type() ? other : other.to(self.scalar_type()))
      : other.to(self.options());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_other(other_c);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_other.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_two_scalar: (self&, Scalar a, Scalar b) -> self&.
#   aclnnInplace<Name>(selfRef, a, b). threshold_ and hardtanh_.
T_INPLACE_TWO_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& a, const at::Scalar& b) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_a(a, self.scalar_type());
  ascend::AclScalarWrapper acl_b(b, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_a.get(),
      acl_b.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_sub_tensor: (self&, other, alpha) -> self&, self -= alpha*other.
#   aclnnInplaceSub(selfRef, other, alpha). Same shape as T_INPLACE_ADD_TENSOR;
#   separate name so the OPS table reads by intent.
T_INPLACE_SUB_TENSOR = T_INPLACE_ADD_TENSOR

# inplace_sub_scalar: (self&, Scalar other, Scalar alpha) -> self&.
#   aclnnInplaceSubs(selfRef, other, alpha). Same shape as T_INPLACE_ADD_SCALAR.
T_INPLACE_SUB_SCALAR = T_INPLACE_ADD_SCALAR

# inplace_int64: (self&, int64) -> self&. aclnnInplace<Name>(selfRef, n).
#   tril_/triu_ (diagonal) and round_.decimals. int64 passes through
#   EXEC_ASCEND_CMD's varargs fine -- only float/double are unsafe there.
T_INPLACE_INT64 = """\
at::Tensor& {kernel}(at::Tensor& self, int64_t n) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), n);
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_elu: (self&, alpha, scale, input_scale) -> self&.
#   aclnnInplaceElu(selfRef, alpha, scale, inputScale).
T_INPLACE_ELU = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {{
  namespace ascend = at::native::flagos::ascend;
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_alpha(alpha, self.scalar_type());
  ascend::AclScalarWrapper acl_scale(scale, self.scalar_type());
  ascend::AclScalarWrapper acl_input_scale(input_scale, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_alpha.get(),
      acl_scale.get(), acl_input_scale.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_masked_fill_scalar: (self&, mask, value) -> self&.
#   aclnnInplaceMaskedFillScalar(selfRef, mask, value). mask is broadcast to
#   self's shape and materialized: aclnn wants a matching ND-contiguous mask.
T_INPLACE_MASKED_FILL_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto mask_b = mask.sizes().equals(self.sizes())
      ? mask.contiguous()
      : mask.expand(self.sizes()).contiguous();
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mask(mask_b);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_mask.get(),
      acl_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_masked_fill_tensor: (self&, mask, value) -> self&. value is a 0-dim
#   tensor that may live on CPU; coerce it like the out-of-place variant does.
T_INPLACE_MASKED_FILL_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto mask_b = mask.sizes().equals(self.sizes())
      ? mask.contiguous()
      : mask.expand(self.sizes()).contiguous();
  auto value_c = value.is_privateuseone()
      ? (value.scalar_type() == self.scalar_type() ? value : value.to(self.scalar_type()))
      : value.to(self.options());
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_mask(mask_b);
  ascend::AclTensorWrapper acl_value(value_c);
  EXEC_ASCEND_CMD({aclnn}, const_cast<aclTensor*>(acl_self.get()), acl_mask.get(),
      acl_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# --------------------------------------------------------------------------
# clamp in-place family.
#
# CANN's coverage here is asymmetric and the gaps are NOT guessable from the
# header names, so each spelling is handled explicitly:
#     aclnnInplaceClampMax        present   (Scalar)
#     aclnnInplaceClampMin        ABSENT    -> compose from aclnnClampMin + copy
#     aclnnInplaceClampMinTensor  present
#     aclnnInplaceClampMaxTensor  present
#     aclnnInplaceClamp           ABSENT    -> compose from aclnnClamp + copy
#     aclnnInplaceClampTensor     ABSENT    -> compose from aclnnClampTensor
# The composed forms allocate one temporary and copy_ back. That is still
# strictly better than the status quo (op unrouted -> hard runtime error), and
# copy_ is device-side (aclnnInplaceCopy) so there is no host round-trip.
# --------------------------------------------------------------------------

# inplace_clamp: (self&, Scalar? min, Scalar? max) -> self&, via out-of-place
#   aclnnClamp into a temporary. Null aclScalar means "bound not supplied".
T_INPLACE_CLAMP = """\
at::Tensor& {kernel}(at::Tensor& self, const ::std::optional<at::Scalar>& min, const ::std::optional<at::Scalar>& max) {{
  namespace ascend = at::native::flagos::ascend;
  auto tmp = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_min = min.has_value()
      ? ascend::AclScalarWrapper(min.value(), self.scalar_type())
      : ascend::AclScalarWrapper();
  ascend::AclScalarWrapper acl_max = max.has_value()
      ? ascend::AclScalarWrapper(max.value(), self.scalar_type())
      : ascend::AclScalarWrapper();
  ascend::AclTensorWrapper acl_tmp(tmp);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_min.get(), acl_max.get(), acl_tmp.get());
  self.copy_(tmp);
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_clamp_tensor: (self&, Tensor? min, Tensor? max) -> self&, via
#   out-of-place aclnnClampTensor. An undefined at::Tensor maps to a null
#   aclTensor*, which aclnn reads as an absent bound.
#
#   The bounds broadcast against self in the out-of-place op, but the in-place
#   spelling cannot change self's shape, so the temporary is self-shaped and any
#   bound wider than self would be an aten-level error before reaching here.
T_INPLACE_CLAMP_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const ::std::optional<at::Tensor>& min, const ::std::optional<at::Tensor>& max) {{
  namespace ascend = at::native::flagos::ascend;
  auto tmp = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());
  auto coerce = [&](const ::std::optional<at::Tensor>& t) {{
    if (!t.has_value()) return at::Tensor();
    const auto& v = t.value();
    return v.is_privateuseone()
        ? (v.scalar_type() == self.scalar_type() ? v : v.to(self.scalar_type()))
        : v.to(self.options());
  }};
  auto min_c = coerce(min);
  auto max_c = coerce(max);

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_min(min_c);
  ascend::AclTensorWrapper acl_max(max_c);
  ascend::AclTensorWrapper acl_tmp(tmp);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_min.get(), acl_max.get(), acl_tmp.get());
  self.copy_(tmp);
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_clamp_bound: (self&, Scalar bound) -> self&, via the out-of-place
#   aclnnClampMin/aclnnClampMax into a temporary. Only needed for clamp_min_
#   (aclnnInplaceClampMin is absent); clamp_max_ has a real in-place entry and
#   uses T_INPLACE_UNARY_SCALAR instead.
T_INPLACE_CLAMP_BOUND = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& bound) {{
  namespace ascend = at::native::flagos::ascend;
  auto tmp = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_bound(bound, self.scalar_type());
  ascend::AclTensorWrapper acl_tmp(tmp);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_bound.get(), acl_tmp.get());
  self.copy_(tmp);
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# clamp_bound_tensor: (self, Tensor bound) -> Tensor, OUT-of-place.
#   aclnnClampMinTensor/aclnnClampMaxTensor(self, bound, out). The single-bound
#   Tensor siblings of clamp.Tensor; bound broadcasts against self.
T_CLAMP_BOUND_TENSOR = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& bound) {{
  namespace ascend = at::native::flagos::ascend;
  auto bound_c = bound.is_privateuseone()
      ? (bound.scalar_type() == self.scalar_type() ? bound : bound.to(self.scalar_type()))
      : bound.to(self.options());
  auto out_shape = at::infer_size(self.sizes(), bound_c.sizes());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_bound(bound_c);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_bound.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# inplace_clamp_bound_tensor: (self&, Tensor bound) -> self&.
#   aclnnInplaceClampMinTensor/MaxTensor(selfRef, bound). Real in-place entries,
#   no temporary needed. Distinct from T_INPLACE_BINARY_TENSOR only in intent,
#   but kept separate so the clamp family reads as one block.
T_INPLACE_CLAMP_BOUND_TENSOR = T_INPLACE_BINARY_TENSOR

# linspace: (start, end, steps, dtype?, layout?, device?, pin?) -> 1-D Tensor.
#   aclnnLinspace(start, end, steps, out). A factory, so the output options come
#   from the kwargs; torch's default for linspace is float (NOT the input
#   Scalars' type -- linspace(0, 1, 5) is a float tensor).
T_LINSPACE = """\
at::Tensor {kernel}(const at::Scalar& start, const at::Scalar& end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {{
  namespace ascend = at::native::flagos::ascend;
  auto options = at::TensorOptions()
      .dtype(dtype.value_or(at::get_default_dtype_as_scalartype()))
      .layout(layout.value_or(at::kStrided))
      .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
      .pinned_memory(pin_memory.value_or(false));
  auto out = ascend::OpPreparation::apply_tensor_without_format({{steps}}, options);
  if (steps == 0) return out;

  ascend::AclScalarWrapper acl_start(start, options.dtype().toScalarType());
  ascend::AclScalarWrapper acl_end(end, options.dtype().toScalarType());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_start.get(), acl_end.get(), steps, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# mse_loss_backward: (grad_output, self, target, reduction) -> grad_input.
#   aclnnMseLossBackward(gradOutput, self, target, reduction, out). Output has
#   self's shape/dtype for every reduction mode.
T_MSE_LOSS_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction) {{
  namespace ascend = at::native::flagos::ascend;
  auto target_c = target.scalar_type() == self.scalar_type()
      ? target : target.to(self.scalar_type());
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_target(target_c);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_target.get(),
      reduction, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# --- activation backward family --------------------------------------------
#
# These close the "forward routed, backward not" gap: the forward half of each
# of these activations already dispatches to aclnn, so inference works while
# .backward() dies at runtime with "<op>: backend not registered". Same failure
# shape as the in-place gap -- invisible at build time, only training hits it.
#
# NOTE on aclnn naming: the derived PascalCase name is right for most of these
# (elu_backward -> aclnnEluBackward), but NOT for the two most valuable ones.
# CANN spells them aclnnDropoutBackward (not NativeDropoutBackward) and
# aclnnRmsNormGrad (not RmsNormBackward), so both need an explicit override.
# Grep libopapi.so per spelling; the derived name is a guess, not a contract.

# grad_scalar_backward: (grad_output, self, Scalar) -> grad_input, self-shaped.
#   aclnn<Name>(gradOutput, self, scalar, gradInput)
# e.g. hardshrink_backward (lambd) / softshrink_backward (lambda).
T_GRAD_SCALAR_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_value.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# grad_two_scalar_backward: (grad_output, self, Scalar a, Scalar b) -> grad_input.
#   aclnn<Name>(gradOutput, self, a, b, gradInput)
# e.g. softplus_backward (beta, threshold) / hardtanh_backward (min, max).
T_GRAD_TWO_SCALAR_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& a, const at::Scalar& b) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_a(a, self.scalar_type());
  ascend::AclScalarWrapper acl_b(b, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_a.get(), acl_b.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# leaky_relu_backward: (grad_output, self, Scalar negative_slope, bool self_is_result).
#   aclnnLeakyReluBackward(gradOutput, self, negativeSlope, selfIsResult, out)
# `self_is_result` passes through as a bool, which survives varargs promotion
# (unlike float -- see EXEC_ASCEND_CMD_SIG).
T_LEAKY_RELU_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& negative_slope, bool self_is_result) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclScalarWrapper acl_slope(negative_slope, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_slope.get(),
      self_is_result, acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# elu_backward: (grad_output, alpha, scale, input_scale, is_result, self_or_result).
#   aclnnEluBackward(gradOutput, alpha, scale, inputScale, isResult, selfOrResult, gradInput)
# Argument order matches aten's exactly, which is unusual -- most backward
# entries lead with (gradOutput, self).
T_ELU_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale, bool is_result, const at::Tensor& self_or_result) {{
  namespace ascend = at::native::flagos::ascend;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self_or_result.sizes(), self_or_result.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclScalarWrapper acl_alpha(alpha, self_or_result.scalar_type());
  ascend::AclScalarWrapper acl_scale(scale, self_or_result.scalar_type());
  ascend::AclScalarWrapper acl_input_scale(input_scale, self_or_result.scalar_type());
  ascend::AclTensorWrapper acl_self(self_or_result);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_alpha.get(), acl_scale.get(),
      acl_input_scale.get(), is_result, acl_self.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_dropout_backward: (grad_output, mask, double scale) -> grad_input.
#
# NOT routed through aclnnDropoutBackward, despite that symbol existing.
# aclnn's dropout pair speaks a BIT-PACKED uint8 mask (1 bit per element,
# 128-byte aligned) while aten's schema promises a bool tensor, and our
# native_dropout forward (rng.cc) already returns the bool form because that is
# what the schema requires. Feeding it to aclnnDropoutBackward fails the shape
# check -- "Size of maskOut has to be 64, but current is 512" for a 512-element
# input, i.e. it wants the 64-byte packed buffer, not 512 bools.
#
# Round-tripping bool -> packed -> aclnn would mean re-packing on device for no
# gain: the backward is just `grad * mask * scale`, two already-routed aclnn
# ops. Compose it instead. Stays on device, no CPU round-trip.
T_DROPOUT_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& mask, double scale) {{
  auto m = mask.scalar_type() == grad_output.scalar_type()
      ? mask : mask.to(grad_output.scalar_type());
  return grad_output * m * scale;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _prelu_kernel_backward: (grad_output, self, weight) -> (grad_self, grad_weight).
#   aclnnPreluBackward(gradOutput, self, weight, gradInput, gradWeight)
#
# TRAP: the two sides disagree on grad_weight's shape.
#   aclnn wants it WEIGHT-shaped -- it reduces internally, and rejects anything
#     else with ret=161002 ("Expected tensor for gradWeight to have same size
#     as [4], but got [8, 4]").
#   aten's contract is the UN-reduced, self-shaped per-element gradient; the
#     sum over broadcast dims is autograd's job downstream.
# Allocating either shape alone is wrong: aclnn's shape breaks aten's callers,
# aten's shape breaks the aclnn call. So take aclnn's reduced result and expand
# it back along the broadcast dims to satisfy aten.
#
# The weight broadcasts over dim 1 for ndim>=2 (the channel dim) and is a
# single element otherwise, which is exactly how the forward broadcasts it.
# _prelu_kernel_backward: (grad_output, self, weight) -> (grad_self, grad_weight).
#   aclnnPreluBackward(gradOutput, self, weight, gradInput, gradWeight)
#
# aclnn is only usable here for HALF of this op. The two sides disagree on
# grad_weight and the disagreement is not a reshape:
#   aclnn  returns grad_weight already SUMMED over the broadcast dims, flat,
#          shaped exactly [weight.numel()].
#   aten's contract is the UN-reduced per-element gradient, shaped like `self`;
#          autograd sums it afterwards.
# Broadcasting aclnn's reduced vector back to self's shape does not recover the
# per-element values -- it repeats the total, so autograd's later sum inflates
# each entry by the number of rows (measured: exactly 4x on a 4-row input).
#
# grad_weight's true per-element value is just `self < 0 ? self * grad_output
# : 0`, which is two already-routed device ops, so compute grad_input with
# aclnn (it is correct and matches CPU exactly) and grad_weight directly.
# No CPU round-trip either way.
T_PRELU_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_self = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());
  // aclnn's gradWeight is reduced, and its shape check is self-contradictory
  // across the two weight kinds -- measured against CANN 9.0.0:
  //   weight [1, 4] (channel-wise, as autograd broadcasts it) -> demands [4]
  //     "Expected tensor for gradWeight to have same size as [4], but got [1,4]"
  //   weight [1, 1] (single shared slope)                     -> demands [1, 1]
  //     "Expected tensor for weight to have same size as tensor for
  //      gradWeight, but [1,1] does not equal [1]"
  // So a single rule cannot satisfy both: flat [numel] for the multi-element
  // case, weight.sizes() verbatim for the scalar case. Its contents are not
  // what aten wants either way; see the note above.
  const int64_t w_numel = weight.numel();
  auto grad_weight_reduced = w_numel == 1
      ? ascend::OpPreparation::apply_tensor_without_format(
            weight.sizes(), weight.options())
      : ascend::OpPreparation::apply_tensor_without_format(
            {{w_numel}}, weight.options());

  ascend::AclTensorWrapper acl_grad(grad_output);
  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_weight(weight);
  ascend::AclTensorWrapper acl_grad_self(grad_self);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight_reduced);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_weight.get(),
      const_cast<aclTensor*>(acl_grad_self.get()),
      const_cast<aclTensor*>(acl_grad_weight.get()));

  // aten's grad_weight: d/dw of (x<0 ? w*x : x) = (x<0 ? x : 0), times the
  // incoming gradient. Both ops are aclnn-routed, so this stays on device.
  auto grad_weight = at::where(self < 0, self, at::zeros({{}}, self.options())) * grad_output;
  return std::make_tuple(grad_self, grad_weight);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# linalg_vector_norm: (self, Scalar ord, dims?, keepdim, dtype?) -> Tensor.
#   aclnnLinalgVectorNorm(self, ord, dims, keepDims, dtype, out)
# Reached by torch.norm / F.normalize / cosine_similarity / clip_grad_norm_.
# `dims` absent means "all dims"; aclnn wants an explicit aclIntArray, so
# materialize 0..ndim-1 rather than passing null.
T_LINALG_VECTOR_NORM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_dtype = dtype.value_or(
      at::isFloatingType(self.scalar_type()) ? self.scalar_type() : at::kFloat);
  auto x = self.scalar_type() == out_dtype ? self : self.to(out_dtype);

  std::vector<int64_t> dims;
  if (dim.has_value() && !dim.value().empty()) {{
    dims.assign(dim.value().begin(), dim.value().end());
    for (auto& d : dims) if (d < 0) d += x.dim();
  }} else {{
    for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
  }}

  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < x.dim(); ++i) {{
    bool reduced = std::find(dims.begin(), dims.end(), i) != dims.end();
    if (!reduced) out_shape.push_back(x.size(i));
    else if (keepdim) out_shape.push_back(1);
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, x.options().dtype(out_dtype));

  ascend::AclTensorWrapper acl_self(x);
  ascend::AclScalarWrapper acl_ord(ord, at::kFloat);
  ascend::AclIntArrayWrapper acl_dims(dims);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_ord.get(), acl_dims.get(), keepdim,
      ascend::ToAclDataType(out_dtype),
      const_cast<aclTensor*>(acl_out.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# embedding: (weight, indices, padding_idx, scale_grad_by_freq, sparse) -> Tensor.
#   aclnnEmbedding(weight, indices, out) uses only weight+indices; the trailing
#   three args are ignored by aclnn. Output = indices.sizes() + [weight.size(1)].
T_EMBEDDING = """\
at::Tensor {kernel}(const at::Tensor& weight, const at::Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {{
  namespace ascend = at::native::flagos::ascend;
  auto out_sizes = indices.sizes().vec();
  out_sizes.push_back(weight.size(1));
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, weight.options());

  ascend::AclTensorWrapper acl_weight(weight);
  ascend::AclTensorWrapper acl_indices(indices);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_weight.get(), acl_indices.get(), acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# embedding_dense_backward: (grad, indices, num_weights, padding_idx,
#   scale_grad_by_freq) -> grad_weight = {num_weights, grad.size(-1)}.
#   aclnnEmbeddingDenseBackward(grad, indices, numWeights, paddingIdx,
#   scaleGradByFreq, out); numWeights/paddingIdx passed as int64 (aclnn uint64).
T_EMBEDDING_DENSE_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      {{num_weights, grad_output.size(-1)}}, grad_output.options());

  ascend::AclTensorWrapper acl_grad_output(grad_output);
  ascend::AclTensorWrapper acl_indices(indices);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight);

  EXEC_ASCEND_CMD({aclnn}, acl_grad_output.get(), acl_indices.get(),
      num_weights, padding_idx, scale_grad_by_freq, acl_grad_weight.get());
  return grad_weight;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# constant_pad_nd: (self, pad, value) -> Tensor. aclnnConstantPadNd(self, pad,
#   value, out). Output widens the trailing dims by pad pairs (last-dim-first,
#   matching torch's pad ordering).
T_CONSTANT_PAD_ND = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value) {{
  namespace ascend = at::native::flagos::ascend;
  auto input_sizes = self.sizes().vec();
  auto ndim = input_sizes.size();
  auto pad_size = pad.size();
  std::vector<int64_t> out_sizes(input_sizes.begin(), input_sizes.end());
  for (size_t i = 0; i < pad_size / 2; ++i) {{
    auto dim = ndim - 1 - i;
    out_sizes[dim] += pad[2 * i] + pad[2 * i + 1];
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, self.options());

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclIntArrayWrapper acl_pad(pad);
  ascend::AclScalarWrapper acl_value(value, self.scalar_type());
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD({aclnn}, acl_self.get(), acl_pad.get(), acl_value.get(),
      acl_out.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# avg_pool2d_backward: (grad_output, self, k, stride, padding, ceil_mode,
#   count_include_pad, divisor_override) -> grad_input (= self shape). Same
#   NCHW-format requirement as the forward. cubeMathType=0 (KEEP_DTYPE).
T_AVG_POOL2D_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  int64_t div_override = divisor_override.value_or(0);
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_k.get(), acl_s.get(),
      acl_p.get(), ceil_mode, count_include_pad, div_override, (int8_t)0,
      acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# _adaptive_avg_pool2d_backward: (grad_output, self) -> grad_input (= self
#   shape). Minimal aclnn(grad, self, out). NCHW format like the forward.
T_ADAPTIVE_AVG_POOL2D_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_layer_norm_backward: (grad_out, input, normalized_shape, mean, rstd,
#   weight?, bias?, output_mask[3]) -> (grad_input, grad_weight, grad_bias).
#   grad_input = input shape; grad_weight/grad_bias = normalized_shape.
T_NATIVE_LAYER_NORM_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_out, const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& mean, const at::Tensor& rstd, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      normalized_shape, input.options());
  auto grad_bias = ascend::OpPreparation::apply_tensor_without_format(
      normalized_shape, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_grad(grad_out);
  ascend::AclTensorWrapper acl_input(input);
  ascend::AclIntArrayWrapper acl_nshape(normalized_shape);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight);
  ascend::AclTensorWrapper acl_grad_bias(grad_bias);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_nshape.get(),
      acl_mean.get(), acl_rstd.get(), acl_weight.get(), acl_bias.get(),
      acl_mask.get(), acl_grad_input.get(), acl_grad_weight.get(),
      acl_grad_bias.get());
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_group_norm_backward: (grad_out, input, mean, rstd, weight?, N, C, HxW,
#   group, output_mask[3]) -> (grad_input, grad_gamma, grad_beta). grad_input =
#   input shape; grad_gamma/grad_beta = [C].
T_NATIVE_GROUP_NORM_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean, const at::Tensor& rstd, const ::std::optional<at::Tensor>& weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_gamma = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());
  auto grad_beta = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_grad(grad_out);
  ascend::AclTensorWrapper acl_input(input);
  ascend::AclTensorWrapper acl_mean(mean);
  ascend::AclTensorWrapper acl_rstd(rstd);
  ascend::AclTensorWrapper acl_gamma(weight_t);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input);
  ascend::AclTensorWrapper acl_grad_gamma(grad_gamma);
  ascend::AclTensorWrapper acl_grad_beta(grad_beta);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_mean.get(),
      acl_rstd.get(), acl_gamma.get(), N, C, HxW, group, acl_mask.get(),
      acl_grad_input.get(), acl_grad_gamma.get(), acl_grad_beta.get());
  return std::make_tuple(grad_input, grad_gamma, grad_beta);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# max_pool2d_with_indices_backward: (grad_output, self, kernel_size, stride,
#   padding, dilation, ceil_mode, indices) -> grad_input (= self shape).
#   NOTE aten arg order != aclnn: aclnn puts `indices` right after `self`
#   (aclnn<Name>(grad, self, indices, k, s, p, dil, ceil, gradInput)).
T_MAX_POOL2D_INDICES_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor& indices) {{
  namespace ascend = at::native::flagos::ascend;
  std::vector<int64_t> k(kernel_size.begin(), kernel_size.end());
  std::vector<int64_t> s = stride.empty() ? k : std::vector<int64_t>(stride.begin(), stride.end());
  std::vector<int64_t> p(padding.begin(), padding.end());
  std::vector<int64_t> dil(dilation.begin(), dilation.end());

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  // aclnn max_pool2d backward (unlike forward) rejects ND 4-D; tag NCHW.
  // aclnn also wants indices as int32, but the forward emits int64 -> cast.
  aclFormat fmt = self.dim() == 4 ? ACL_FORMAT_NCHW : ACL_FORMAT_NCL;
  auto indices_i32 = indices.scalar_type() == at::kInt ? indices : indices.to(at::kInt);
  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_self(self, fmt);
  ascend::AclTensorWrapper acl_indices(indices_i32, fmt);
  ascend::AclIntArrayWrapper acl_k(k);
  ascend::AclIntArrayWrapper acl_s(s);
  ascend::AclIntArrayWrapper acl_p(p);
  ascend::AclIntArrayWrapper acl_dil(dil);
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_self.get(), acl_indices.get(),
      acl_k.get(), acl_s.get(), acl_p.get(), acl_dil.get(), ceil_mode,
      acl_grad_input.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_batch_norm: (input, weight?, bias?, running_mean?, running_var?,
#   training, momentum, eps) -> (output, save_mean, save_invstd). runningMean/
#   runningVar are updated in-place by aclnn (non-const). save_mean/save_invstd
#   are [C]. Needs NCHW/NCL/NCDHW format like conv. momentum/eps are by-value
#   double (varargs-safe: aarch64 already passes double, cf. group_norm).
T_NATIVE_BATCH_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& input, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& bias, const ::std::optional<at::Tensor>& running_mean, const ::std::optional<at::Tensor>& running_var, bool training, double momentum, double eps) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t rank = input.dim();
  int64_t C = input.size(1);
  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : (rank == 5 ? ACL_FORMAT_NCDHW : ACL_FORMAT_ND));

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto save_mean = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());
  auto save_invstd = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor bias_t = bias.value_or(at::Tensor());
  at::Tensor rmean_t = running_mean.value_or(at::Tensor());
  at::Tensor rvar_t = running_var.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclTensorWrapper acl_rmean(rmean_t);
  ascend::AclTensorWrapper acl_rvar(rvar_t);
  ascend::AclTensorWrapper acl_out(out, fmt);
  ascend::AclTensorWrapper acl_save_mean(save_mean);
  ascend::AclTensorWrapper acl_save_invstd(save_invstd);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_weight.get(), acl_bias.get(),
      const_cast<aclTensor*>(acl_rmean.get()), const_cast<aclTensor*>(acl_rvar.get()),
      training, momentum, eps, acl_out.get(), acl_save_mean.get(), acl_save_invstd.get());
  return std::make_tuple(out, save_mean, save_invstd);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# native_batch_norm_backward: (grad_out, input, weight?, running_mean?,
#   running_var?, save_mean?, save_invstd?, train, eps, output_mask[3]) ->
#   (grad_input, grad_weight, grad_bias). grad_weight/grad_bias are [C].
T_NATIVE_BATCH_NORM_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_out, const at::Tensor& input, const ::std::optional<at::Tensor>& weight, const ::std::optional<at::Tensor>& running_mean, const ::std::optional<at::Tensor>& running_var, const ::std::optional<at::Tensor>& save_mean, const ::std::optional<at::Tensor>& save_invstd, bool train, double eps, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  int64_t rank = input.dim();
  int64_t C = input.size(1);
  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : (rank == 5 ? ACL_FORMAT_NCDHW : ACL_FORMAT_ND));

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());
  auto grad_bias = ascend::OpPreparation::apply_tensor_without_format(
      {{C}}, input.options());

  at::Tensor weight_t = weight.value_or(at::Tensor());
  at::Tensor rmean_t = running_mean.value_or(at::Tensor());
  at::Tensor rvar_t = running_var.value_or(at::Tensor());
  at::Tensor smean_t = save_mean.value_or(at::Tensor());
  at::Tensor sinvstd_t = save_invstd.value_or(at::Tensor());

  ascend::AclTensorWrapper acl_grad(grad_out, fmt);
  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight_t);
  ascend::AclTensorWrapper acl_rmean(rmean_t);
  ascend::AclTensorWrapper acl_rvar(rvar_t);
  ascend::AclTensorWrapper acl_smean(smean_t);
  ascend::AclTensorWrapper acl_sinvstd(sinvstd_t);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight);
  ascend::AclTensorWrapper acl_grad_bias(grad_bias);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_weight.get(),
      acl_rmean.get(), acl_rvar.get(), acl_smean.get(), acl_sinvstd.get(),
      train, eps, acl_mask.get(), acl_grad_input.get(), acl_grad_weight.get(),
      acl_grad_bias.get());
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# convolution: (input, weight, bias?, stride, padding, dilation, transposed,
#   output_padding, groups) -> Tensor. Non-transposed conv only (transposed
#   uses output_padding + a different out formula -> left to a later batch).
#   out[N, Cout, *spatial] where Cout=weight.size(0). aclnnConvolution wants
#   NCHW/NCL/NCDHW format (ND 4-D is rejected, same as pooling). cubeMathType=0
#   (KEEP_DTYPE) keeps full fp32 -- type=1 (ALLOW_FP32_DOWN_PRECISION) loses
#   ~2.5e-3 vs CPU on the cube unit.
T_CONVOLUTION = """\
at::Tensor {kernel}(const at::Tensor& input, const at::Tensor& weight, const ::std::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(!transposed, "convolution codegen kernel: transposed not supported");
  int64_t rank = input.dim();
  int64_t nspatial = rank - 2;
  auto out_shape = std::vector<int64_t>{{input.size(0), weight.size(0)}};
  for (int64_t i = 0; i < nspatial; ++i) {{
    int64_t in = input.size(i + 2), k = weight.size(i + 2);
    int64_t st = stride.size() == 1 ? stride[0] : stride[i];
    int64_t pd = padding.size() == 1 ? padding[0] : padding[i];
    int64_t dl = dilation.size() == 1 ? dilation[0] : dilation[i];
    out_shape.push_back((in + 2 * pd - dl * (k - 1) - 1) / st + 1);
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, input.options());

  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : ACL_FORMAT_NCDHW);
  at::Tensor bias_t = bias.value_or(at::Tensor());
  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight, fmt);
  ascend::AclTensorWrapper acl_bias(bias_t);
  ascend::AclIntArrayWrapper acl_stride(stride);
  ascend::AclIntArrayWrapper acl_padding(padding);
  ascend::AclIntArrayWrapper acl_dilation(dilation);
  ascend::AclIntArrayWrapper acl_outpad(output_padding);
  ascend::AclTensorWrapper acl_out(out, fmt);

  EXEC_ASCEND_CMD({aclnn}, acl_input.get(), acl_weight.get(), acl_bias.get(),
      acl_stride.get(), acl_padding.get(), acl_dilation.get(), transposed,
      acl_outpad.get(), groups, acl_out.get(), (int8_t)0);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# convolution_backward: (grad_output, input, weight, bias_sizes?, stride,
#   padding, dilation, transposed, output_padding, groups, output_mask[3]) ->
#   (grad_input, grad_weight, grad_bias). aclnn allocates all three; unwanted
#   ones (output_mask=false) are still passed but ignored. grad_bias has shape
#   [Cout] = weight.size(0).
T_CONVOLUTION_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool, 3> output_mask) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(!transposed, "convolution_backward codegen kernel: transposed not supported");
  int64_t rank = input.dim();
  aclFormat fmt = rank == 4 ? ACL_FORMAT_NCHW : (rank == 3 ? ACL_FORMAT_NCL : ACL_FORMAT_NCDHW);

  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input.sizes(), input.options());
  auto grad_weight = ascend::OpPreparation::apply_tensor_without_format(
      weight.sizes(), weight.options());
  // grad_bias is always allocated and passed, even when output_mask[2] is
  // false (aclnn writes nothing to it then). But its shape must still be
  // valid: aclnnConvolutionBackward rejects an empty biasSizes, or one whose
  // product is 0, with 161002 (ACLNN_ERR_PARAM_INVALID). For bias=None
  // autograd hands us [0] (and an empty list is possible too), so in either
  // case substitute the real bias length [Cout] = weight.size(0).
  std::vector<int64_t> bias_shape = std::vector<int64_t>{{weight.size(0)}};
  if (bias_sizes.has_value() && !bias_sizes.value().empty()) {{
    const auto bs = bias_sizes.value();
    int64_t numel = 1;
    for (auto d : bs) {{ numel *= d; }}
    if (numel > 0) {{
      bias_shape = bs.vec();
    }}
  }}
  auto grad_bias = ascend::OpPreparation::apply_tensor_without_format(
      bias_shape, weight.options());

  ascend::AclTensorWrapper acl_grad(grad_output, fmt);
  ascend::AclTensorWrapper acl_input(input, fmt);
  ascend::AclTensorWrapper acl_weight(weight, fmt);
  ascend::AclIntArrayWrapper acl_bias_sizes(bias_shape);
  ascend::AclIntArrayWrapper acl_stride(stride);
  ascend::AclIntArrayWrapper acl_padding(padding);
  ascend::AclIntArrayWrapper acl_dilation(dilation);
  ascend::AclIntArrayWrapper acl_outpad(output_padding);
  ascend::AclBoolArrayWrapper acl_mask(at::ArrayRef<bool>(output_mask.data(), output_mask.size()));
  ascend::AclTensorWrapper acl_grad_input(grad_input, fmt);
  ascend::AclTensorWrapper acl_grad_weight(grad_weight, fmt);
  ascend::AclTensorWrapper acl_grad_bias(grad_bias);

  EXEC_ASCEND_CMD({aclnn}, acl_grad.get(), acl_input.get(), acl_weight.get(),
      acl_bias_sizes.get(), acl_stride.get(), acl_padding.get(),
      acl_dilation.get(), transposed, acl_outpad.get(), (int)groups,
      acl_mask.get(), (int8_t)0, acl_grad_input.get(), acl_grad_weight.get(),
      acl_grad_bias.get());
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# ==========================================================================
# Cached (repeatable-executor) variants of the hot pure-tensor categories.
#
# These mirror the plain templates but route through ascend::ExecAscendCached,
# which caches the aclOpExecutor keyed by (op, tensor signature, scalar value).
# On a cache hit (constant shapes -- the eager decode steady state) it skips
# aclnn<Op>GetWorkspaceSize + aclCreateTensor and only rebinds the tensor data
# addresses, matching torch_npu's per-op host cost. Only categories whose aclnn
# call is purely (tensors..., [scalars baked into key], out) are cached; scalars
# are folded into the key because they are baked into the executor and are NOT
# rebindable (verified on CANN 9.0.0). The `build` lambda replays the exact
# GetWorkspaceSize arg order on a miss using the cache-owned aclTensors.
# ==========================================================================

T_UNARY_CACHED = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  namespace ascend = at::native::flagos::ascend;
  if (!ascend::IsUnaryDtypeSupported(self.scalar_type())) {{
    auto cpu_result = at::{cpu_op}(self.cpu());
    return ascend::MoveCpuResultToDevice(std::move(cpu_result), self);
  }}
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self);
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self}}, {{&out}},
      [](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
         std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

T_BINARY_CACHED = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  namespace ascend = at::native::flagos::ascend;
{scalar_fastpath}"""
    + _BINARY_PROLOGUE_BODY
    + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, result_options);

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self_b); hsh.tensor(other_b);
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self_b, &other_b}}, {{&out}},
      [](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
         std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, in[1].acl_tensor, out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

T_BINARY_ALPHA_CACHED = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
  namespace ascend = at::native::flagos::ascend;
{scalar_fastpath}"""
    + _BINARY_PROLOGUE_BODY
    + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, result_options);
  ascend::AclScalarWrapper acl_alpha(alpha, result_dtype);

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self_b); hsh.tensor(other_b);
  {{ double av = alpha.toDouble(); hsh.val(av); }}
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self_b, &other_b}}, {{&out}},
      [&](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
          std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, in[1].acl_tensor, acl_alpha.get(), out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

T_BINARY_CMP_CACHED = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kBool));

  static void* opApiFuncAddr = nullptr;
  static void* getWsFuncAddr = nullptr;
  ascend::SigHasher hsh; hsh.tensor(self_b); hsh.tensor(other_b);
  ascend::ExecAscendCached(
      "{aclnn}", "{aclnn}GetWorkspaceSize", opApiFuncAddr, getWsFuncAddr, hsh.h,
      {{&self_b, &other_b}}, {{&out}},
      [](ascend::GwsFunc gws, std::vector<ascend::AclTensorWrapper>& in,
         std::vector<ascend::AclTensorWrapper>& out_t, uint64_t* pws, aclOpExecutor** pex) {{
        return gws(in[0].acl_tensor, in[1].acl_tensor, out_t[0].acl_tensor, pws, pex);
      }});
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
)

# Map each cacheable category to its cached template. Gated by the env var
# FLAGOS_EXEC_CACHE (default ON); set FLAGOS_EXEC_CACHE=0 to regenerate the
# plain uncached kernels (bisection / correctness fallback).
CACHED_CATEGORIES = {
    "unary": T_UNARY_CACHED,
    "binary": T_BINARY_CACHED,
    "binary_alpha": T_BINARY_ALPHA_CACHED,
    "binary_cmp": T_BINARY_CMP_CACHED,
    "unary_scalar": T_UNARY_SCALAR_CACHED,
    "reduce_sum_dtype": T_REDUCE_SUM_DTYPE_CACHED,
    "reduce_mean_dtype": T_REDUCE_MEAN_DTYPE_CACHED,
    "softmax_fwd": T_SOFTMAX_FWD_CACHED,
}

# Maps a Tensor-Tensor binary op to its aclnn scalar variant (<Name>s) for the
# CPU-scalar fast path. The value is the aclnn base name (without "aclnn"); the
# variant kind ("noalpha"/"alpha") selects which fast-path template to inject.
# Only ops whose <Name>s symbol exists in libopapi.so are listed; the presence
# check at codegen time is a hard gate on top of this map.
SCALAR_VARIANT = {
    "mul.Tensor": ("Muls", "noalpha"),
    "div.Tensor": ("Divs", "noalpha"),
    "add.Tensor": ("Adds", "alpha"),
    "sub.Tensor": ("Subs", "alpha"),
}

CATEGORIES = {
    "unary": T_UNARY,
    "binary": T_BINARY,
    "binary_alpha": T_BINARY_ALPHA,
    "binary_cmp": T_BINARY_CMP,
    "binary_scalar_alpha": T_BINARY_SCALAR_ALPHA,
    "binary_scalar_cmp": T_BINARY_SCALAR_CMP,
    "reduce_dims": T_REDUCE_DIMS,
    "reduce_dim_bool": T_REDUCE_DIM_BOOL,
    "cumsum": T_CUMSUM,
    "cumprod": T_CUMPROD,
    "unary_bool": T_UNARY_BOOL,
    "unary_scalar": T_UNARY_SCALAR,
    "unary_two_scalar": T_UNARY_TWO_SCALAR,
    "unary_int": T_UNARY_INT,
    "unary_dims": T_UNARY_DIMS,
    "addcmul": T_ADDCMUL,
    "act_backward": T_ACT_BACKWARD,
    "threshold_backward": T_THRESHOLD_BACKWARD,
    "pow_scalar_tensor": T_POW_SCALAR_TENSOR,
    "reduce_max_dim": T_REDUCE_MAX_DIM,
    "elu": T_ELU,
    "loss": T_LOSS,
    "cummax_cummin": T_CUMMAX_CUMMIN,
    "aminmax": T_AMINMAX,
    "prod": T_PROD,
    "gemm_addmm": T_GEMM_ADDMM,
    "gemm_baddbmm": T_GEMM_BADDBMM,
    "gemm_addmv": T_GEMM_ADDMV,
    "gemm_addr": T_GEMM_ADDR,
    "matmul": T_MATMUL,
    "matmul_out": T_MATMUL_OUT,
    "cat": T_CAT,
    "cat_out": T_CAT_OUT,
    "stack": T_STACK,
    "mv": T_MV,
    "dot": T_DOT,
    "bce": T_BCE,
    "bce_backward": T_BCE_BACKWARD,
    "bce_logits": T_BCE_LOGITS,
    "layer_norm": T_LAYER_NORM,
    "group_norm": T_GROUP_NORM,
    "gelu": T_GELU,
    "gelu_backward": T_GELU_BACKWARD,
    "log_softmax": T_LOG_SOFTMAX,
    "softmax_backward": T_SOFTMAX_BACKWARD,
    "binary_scalar": T_BINARY_SCALAR,
    "act_backward_self": T_ACT_BACKWARD_SELF,
    "where": T_WHERE,
    "clamp": T_CLAMP,
    "clamp_tensor": T_CLAMP_TENSOR,
    "softmax_fwd": T_SOFTMAX_FWD,
    "reduce_all": T_REDUCE_ALL,
    "reduce_sum_dtype": T_REDUCE_SUM_DTYPE,
    "reduce_sum_all": T_REDUCE_SUM_ALL,
    "mean_all": T_MEAN_ALL,
    "reduce_minmax_all": T_REDUCE_MINMAX_ALL,
    "reduce_mean_dtype": T_REDUCE_MEAN_DTYPE,
    "adaptive_avg_pool2d": T_ADAPTIVE_AVG_POOL2D,
    "avg_pool2d": T_AVG_POOL2D,
    "max_pool2d_indices": T_MAX_POOL2D_INDICES,
    "convolution": T_CONVOLUTION,
    "convolution_backward": T_CONVOLUTION_BACKWARD,
    "max_pool2d_indices_backward": T_MAX_POOL2D_INDICES_BACKWARD,
    "native_batch_norm": T_NATIVE_BATCH_NORM,
    "native_batch_norm_backward": T_NATIVE_BATCH_NORM_BACKWARD,
    "avg_pool2d_backward": T_AVG_POOL2D_BACKWARD,
    "adaptive_avg_pool2d_backward": T_ADAPTIVE_AVG_POOL2D_BACKWARD,
    "native_layer_norm_backward": T_NATIVE_LAYER_NORM_BACKWARD,
    "native_group_norm_backward": T_NATIVE_GROUP_NORM_BACKWARD,
    "masked_fill_scalar": T_MASKED_FILL_SCALAR,
    "masked_fill_tensor": T_MASKED_FILL_TENSOR,
    "gather": T_GATHER,
    "index_select": T_INDEX_SELECT,
    "inplace_zero": T_INPLACE_ZERO,
    "inplace_fill_scalar": T_INPLACE_FILL_SCALAR,
    "inplace_fill_tensor": T_INPLACE_FILL_TENSOR,
    "inplace_add_tensor": T_INPLACE_ADD_TENSOR,
    "inplace_add_scalar": T_INPLACE_ADD_SCALAR,
    "inplace_mul_tensor": T_INPLACE_MUL_TENSOR,
    "inplace_mul_scalar": T_INPLACE_MUL_SCALAR,
    "inplace_div_tensor": T_INPLACE_DIV_TENSOR,
    "inplace_addcmul": T_INPLACE_ADDCMUL,
    "inplace_addcdiv": T_INPLACE_ADDCDIV,
    "inplace_sqrt": T_INPLACE_SQRT,
    "inplace_lerp_scalar": T_INPLACE_LERP_SCALAR,
    "inplace_unary": T_INPLACE_UNARY,
    "inplace_unary_scalar": T_INPLACE_UNARY_SCALAR,
    "inplace_binary_tensor": T_INPLACE_BINARY_TENSOR,
    "inplace_two_scalar": T_INPLACE_TWO_SCALAR,
    "inplace_sub_tensor": T_INPLACE_SUB_TENSOR,
    "inplace_sub_scalar": T_INPLACE_SUB_SCALAR,
    "inplace_int64": T_INPLACE_INT64,
    "inplace_elu": T_INPLACE_ELU,
    "inplace_masked_fill_scalar": T_INPLACE_MASKED_FILL_SCALAR,
    "inplace_masked_fill_tensor": T_INPLACE_MASKED_FILL_TENSOR,
    "inplace_clamp": T_INPLACE_CLAMP,
    "inplace_clamp_tensor": T_INPLACE_CLAMP_TENSOR,
    "inplace_clamp_bound": T_INPLACE_CLAMP_BOUND,
    "inplace_clamp_bound_tensor": T_INPLACE_CLAMP_BOUND_TENSOR,
    "clamp_bound_tensor": T_CLAMP_BOUND_TENSOR,
    "linspace": T_LINSPACE,
    "mse_loss_backward": T_MSE_LOSS_BACKWARD,
    "grad_scalar_backward": T_GRAD_SCALAR_BACKWARD,
    "grad_two_scalar_backward": T_GRAD_TWO_SCALAR_BACKWARD,
    "leaky_relu_backward": T_LEAKY_RELU_BACKWARD,
    "elu_backward": T_ELU_BACKWARD,
    "dropout_backward": T_DROPOUT_BACKWARD,
    "prelu_backward": T_PRELU_BACKWARD,
    "linalg_vector_norm": T_LINALG_VECTOR_NORM,
    "foreach_inplace_scalar": T_FOREACH_INPLACE_SCALAR,
    "foreach_inplace_lerp_scalar": T_FOREACH_INPLACE_LERP_SCALAR,
    "foreach_inplace_addcmul_scalar": T_FOREACH_INPLACE_ADDCMUL_SCALAR,
    "foreach_sqrt": T_FOREACH_SQRT,
    "foreach_inplace_div_scalarlist": T_FOREACH_INPLACE_DIV_SCALARLIST,
    "foreach_inplace_addcdiv_scalarlist": T_FOREACH_INPLACE_ADDCDIV_SCALARLIST,
    "foreach_norm": T_FOREACH_NORM,
    "foreach_inplace_maximum_scalar": T_FOREACH_INPLACE_MAXIMUM_SCALAR,
    "foreach_inplace_mul_tensor": T_FOREACH_INPLACE_MUL_TENSOR,
    "embedding": T_EMBEDDING,
    "embedding_dense_backward": T_EMBEDDING_DENSE_BACKWARD,
    "constant_pad_nd": T_CONSTANT_PAD_ND,
    "zeros": T_ZEROS,
    "ones": T_ONES,
    "scalar_tensor": T_SCALAR_TENSOR,
    "ones_like": T_ONES_LIKE,
    "zeros_like": T_ZEROS_LIKE,
    "empty_like": T_EMPTY_LIKE,
    "full": T_FULL,
    "full_like": T_FULL_LIKE,
    "new_ones": T_NEW_ONES,
    "amp_unscale": T_AMP_UNSCALE,
    "amp_unscale_out": T_AMP_UNSCALE_OUT,
    "foreach_inplace_add_list": T_FOREACH_INPLACE_ADD_LIST,
}

# Categories whose kernels do NOT issue a direct aclnn call (they build tensors
# on-host and fill via zero_/fill_, which are themselves device-side aclnn ops).
# The symbol-validation guard is skipped for these; their OPS override is unused.
NO_ACLNN_CATEGORIES = {
    "dropout_backward",
    "zeros",
    "ones",
    "scalar_tensor",
    "ones_like",
    "zeros_like",
    "empty_like",
    "full",
    "full_like",
    "new_ones",
    "amp_unscale",
    "amp_unscale_out",
    "foreach_inplace_add_list",
}

# Categories whose template splits its TensorList args into chunks to stay under
# the CANN per-kernel list-length cap (see FOREACH_CHUNK above). Maps the
# category to the chunk size substituted into the template's {chunk} slot.
FOREACH_CHUNKED_CATEGORIES = {
    "foreach_inplace_scalar": FOREACH_CHUNK,
    "foreach_inplace_lerp_scalar": FOREACH_CHUNK,
    "foreach_inplace_addcmul_scalar": FOREACH_CHUNK,
    "foreach_sqrt": FOREACH_CHUNK,
    "foreach_inplace_div_scalarlist": FOREACH_CHUNK,
    "foreach_inplace_addcdiv_scalarlist": FOREACH_CHUNK,
    "foreach_norm": FOREACH_CHUNK,
    "foreach_inplace_maximum_scalar": FOREACH_CHUNK,
    "foreach_inplace_mul_tensor": FOREACH_CHUNK,
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_ascend.py -- DO NOT EDIT.
//
// aclnn kernels for the Ascend backend, generated per-category. Each kernel
// marshals aten tensors into aclTensors and issues the two-phase aclnn call
// via EXEC_ASCEND_CMD. Dispatchers are declared in generated/ops.h (shared
// with the CUDA codegen); here we only fill the Backend::kAscend slot.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/acos.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/cosh.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erfc.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/frac.h>
#include <ATen/ops/log.h>
#include <ATen/ops/log10.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/round.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sinh.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/tan.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/trunc.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <vector>
#include "../op_preparation.h"
#include "../op_api_common.h"
#include "../dtype_support.h"

namespace at::native::flagos {

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"


def aclnn_name(op_base: str, override) -> str:
    if override:
        return "aclnn" + override
    pascal = "".join(w.capitalize() for w in op_base.split("_") if w)
    return "aclnn" + pascal


def libopapi_path() -> Path:
    ah = os.environ.get("ASCEND_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
    return Path(ah) / "lib64" / "libopapi.so"


def symbols(lib: Path):
    if not lib.exists():
        print(f"[warn] {lib} not found; skipping symbol validation", file=sys.stderr)
        return None
    out = subprocess.run(["nm", "-D", str(lib)], capture_output=True, text=True)
    syms = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts:
            syms.add(parts[-1])
    return syms


def _handwritten_dispatchers() -> set:
    """Dispatcher stems that a handwritten .cc registers to Backend::kAscend."""
    pat = re.compile(
        r"REGISTER_IMPL_TO_DISPATCHER\(\s*\w+\s*,\s*(\w+)_dispatcher\s*,"
        r"\s*Backend::kAscend\b"
    )
    found = set()
    for d in HANDWRITTEN_DIRS:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.cc")):
            found |= set(pat.findall(path.read_text()))
    return found


def emit_register_inc(covered: list) -> int:
    """Write the m.impl() subset naming exactly Ascend's kernel coverage set.

    Ascend used to include the full generated/register.inc, claiming all ~2000
    PrivateUse1 ops while owning kernels for a few hundred. Every uncovered op
    then reached the dispatcher with an empty kAscend slot, so routing it to
    `none` raised instead of falling back. Registering only the covered ops
    makes `none` mean what the conf says it means: unregistered, so the call
    reaches cpu_fallback. Same contract as gcu_register.inc.
    """
    if not REGISTER_INC.exists():
        print(
            f"[inc] skipped: {REGISTER_INC.relative_to(REPO)} missing "
            "(run scripts/codegen/codegen_ops.py first)",
            file=sys.stderr,
        )
        return 0
    wrappers = dict(
        re.findall(
            r'^\s*m\.impl\("([^"]+)",\s*(\w+)\);',
            REGISTER_INC.read_text(),
            flags=re.MULTILINE,
        )
    )
    disp_to_op = {}
    for op in wrappers:
        _, disp = schema_to_cpp_name(op)
        disp_to_op[disp[: -len("_dispatcher")]] = op

    ops = {op for op, _, _ in covered}
    missing = []
    for stem in _handwritten_dispatchers():
        if stem in disp_to_op:
            ops.add(disp_to_op[stem])
        else:
            missing.append(stem)
    if missing:
        print(
            f"[inc] warning: {len(missing)} handwritten dispatcher(s) not in "
            f"register.inc, omitted: {', '.join(sorted(missing)[:8])}",
            file=sys.stderr,
        )

    lines = [
        "// Copyright (c) 2026, BAAI. All rights reserved.",
        "//",
        "// @generated by scripts/codegen/codegen_ascend.py -- DO NOT EDIT.",
        "//",
        "// m.impl() lines for the ops that have an Ascend kernel, generated or",
        "// handwritten. Included by register.cc inside",
        "// TORCH_LIBRARY_IMPL(aten, PrivateUse1) when USE_ASCEND is set, in place",
        "// of the full generated/register.inc list: an op registered on",
        "// PrivateUse1 without a kernel behind it fails the dispatcher's",
        '// "backend not registered" check, whereas an unregistered op simply',
        "// reaches the cpu_fallback. So this file is exactly the Ascend coverage",
        "// set, and `none` in backends_ascend.conf is honest.",
        "",
    ]
    for op in sorted(ops):
        lines.append(f'  m.impl("{op}", {wrappers[op]});')
    OUT_INC.parent.mkdir(parents=True, exist_ok=True)
    OUT_INC.write_text("\n".join(lines) + "\n")
    return len(ops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--category",
        default="all",
        choices=["all"] + list(CATEGORIES),
        help="restrict generation to one category (default: all)",
    )
    ap.add_argument(
        "--no-conf",
        action="store_true",
        help="do not append covered ops to backends_ascend.conf",
    )
    args = ap.parse_args()

    syms = symbols(libopapi_path())

    # Repeatable-executor cache: on for the cacheable categories unless disabled.
    exec_cache = os.environ.get("FLAGOS_EXEC_CACHE", "1") != "0"

    bodies = []
    covered = []  # (op, aclnn, category)
    skipped = []  # (op, reason)
    cached_ops = []  # ops emitted with the cached template
    scalar_fastpath_ops = []  # ops that got the CPU-scalar diversion

    for op, (cat, override) in OPS.items():
        if args.category != "all" and cat != args.category:
            continue
        if op in SKIP:
            skipped.append((op, "handwritten"))
            continue
        base = op.split(".")[0]
        acl = aclnn_name(base, override)
        if syms is not None and cat not in NO_ACLNN_CATEGORIES:
            if (acl not in syms) or (acl + "GetWorkspaceSize" not in syms):
                skipped.append((op, f"{acl} not in libopapi.so"))
                continue
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelAscend"  # SqrtFn -> SqrtKernelAscend
        template = CATEGORIES[cat]
        fmt = dict(kernel=kernel, aclnn=acl, fn=fn, disp=disp, cpu_op=base)
        if cat in FOREACH_CHUNKED_CATEGORIES:
            fmt["chunk"] = FOREACH_CHUNKED_CATEGORIES[cat]
        if exec_cache and cat in CACHED_CATEGORIES:
            template = CACHED_CATEGORIES[cat]
            cached_ops.append(op)
            # binary/binary_alpha cached templates carry a {scalar_fastpath}
            # slot. Fill it with the CPU-scalar diversion when the op has an
            # aclnn scalar variant present in libopapi.so; otherwise leave empty.
            if cat in ("binary", "binary_alpha"):
                sf = ""
                if exec_cache and op in SCALAR_VARIANT:
                    sname, kind = SCALAR_VARIANT[op]
                    acl_s = "aclnn" + sname
                    if syms is None or (
                        acl_s in syms and acl_s + "GetWorkspaceSize" in syms
                    ):
                        tmpl = (
                            _SCALAR_FASTPATH_ALPHA
                            if kind == "alpha"
                            else _SCALAR_FASTPATH_NOALPHA
                        )
                        sf = tmpl.format(aclnn_s=acl_s)
                        scalar_fastpath_ops.append(op)
                fmt["scalar_fastpath"] = sf
        bodies.append(template.format(**fmt))
        covered.append((op, acl, cat))

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(FILE_HEADER + "\n".join(bodies) + FILE_FOOTER)

    # Report grouped by category.
    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({len(covered)} kernels)")
    if exec_cache:
        print(
            f"    [exec-cache] ON for {len(cached_ops)} op(s): {', '.join(cached_ops)}"
        )
        if scalar_fastpath_ops:
            print(
                f"    [scalar-fastpath] {len(scalar_fastpath_ops)} op(s): {', '.join(scalar_fastpath_ops)}"
            )
    else:
        print("    [exec-cache] OFF (FLAGOS_EXEC_CACHE=0)")
    by_cat = {}
    for op, acl, cat in covered:
        by_cat.setdefault(cat, []).append((op, acl))
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if items:
            print(f"    [{cat}] {len(items)}")
            for op, acl in items:
                print(f"       + {op} -> {acl}")
    for op, why in skipped:
        print(f"       - {op} skipped ({why})")

    if not args.no_conf and covered:
        n = emit_register_inc(covered)
        if n:
            print(f"[inc] {OUT_INC.relative_to(REPO)}  ({n} registered ops)")
        # backends_ascend.conf is NOT written here. It is generated in full by
        # scripts/codegen/gen_vendor_confs.py, which reads this .inc to learn the Ascend
        # kernel set. Appending here as well made two scripts writers of one
        # file, and the appended `op = ascend` lines could not express the
        # FlagGems-first priority the generated conf applies.


if __name__ == "__main__":
    main()
