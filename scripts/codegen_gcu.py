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
Codegen for torch_fl Enflame GCU (topsaten) operators.

Same problem as Ascend: there is no vendor dispatch key to box into (no CUDA
runtime exists on GCU at all), so every kernel must call the vendor op library
-- libtopsaten.so -- directly. And as with Ascend the call shape is uniform per
*category*, so this generator is category-driven.

topsaten is simpler than aclnn: one direct call, no workspace/executor phase.

    topsaten::topsatenAdd(out, lhs, rhs, alpha, stream)

Generates:
  - csrc/aten/backends/gcu/generated/gcu_kernels.cc
      the kernels + REGISTER_IMPL_TO_DISPATCHER(..., Backend::kGcu, ...)
  - csrc/aten/backends/gcu/generated/gcu_register.inc
      the m.impl() subset for register.cc. GCU registers PrivateUse1 ONLY for
      ops it has a kernel for; everything else stays unregistered and reaches
      the cpu_fallback (registering all 2033 ops would instead hit the
      dispatcher's "backend not registered" check). FlagGems Python kernels are
      compiled alongside GCU, but only overloads in this coverage set own a
      PrivateUse1 wrapper and may select that dispatcher slot.
  - appends `<op> = gcu` to torch_fl/configs/backends_gcu.conf

Validation:
  - the derived topsaten<Name> must exist in libtopsaten.so or the op is
    skipped with a warning. Symbols are C++-mangled (namespace topsaten), so
    they are read via `nm -DC` and matched as `topsaten::topsaten<Name>`.
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
from codegen_ops import schema_to_cpp_name

REPO = Path(__file__).resolve().parent.parent
OUT_CC = REPO / "csrc/aten/backends/gcu/generated/gcu_kernels.cc"
OUT_INC = REPO / "csrc/aten/backends/gcu/generated/gcu_register.inc"
REGISTER_INC = REPO / "csrc/aten/generated/register.inc"
CONF = REPO / "torch_fl/configs/backends_gcu.conf"

# --------------------------------------------------------------------------
# Op registry: schema op name -> (category, topsaten-name override or None).
#
# Default topsaten name = "topsaten" + PascalCase(op base); a non-None override
# replaces that stem for irregular spellings (_softmax -> SoftmaxForward).
# --------------------------------------------------------------------------
OPS = {
    # ---- unary: topsaten<Name>(out, self) ----
    "abs": ("unary", None),
    "sqrt": ("unary", None),
    "rsqrt": ("unary", None),
    "exp": ("unary", None),
    "expm1": ("unary", None),
    "log": ("unary", None),
    "log2": ("unary", None),
    "log10": ("unary", None),
    "log1p": ("unary", None),
    "sin": ("unary", None),
    "cos": ("unary", None),
    "sinh": ("unary", None),
    "cosh": ("unary", None),
    "asin": ("unary", None),
    "acos": ("unary", None),
    "atan": ("unary", None),
    "tanh": ("unary", None),
    "sigmoid": ("unary", None),
    "silu": ("unary", None),
    "relu": ("unary", None),
    "neg": ("unary", None),
    "reciprocal": ("unary", None),
    "erf": ("unary", None),
    "floor": ("unary", None),
    "ceil": ("unary", None),
    "trunc": ("unary", None),
    "sign": ("unary", None),
    # ---- binary: topsaten<Name>(out, self, other) ----
    "mul.Tensor": ("binary", None),
    "div.Tensor": ("binary", None),
    "maximum": ("binary", None),
    "minimum": ("binary", None),
    "remainder.Tensor": ("binary", None),
    "fmod.Tensor": ("binary", None),
    "pow.Tensor_Tensor": ("binary", None),
    # ---- binary_alpha: topsaten<Name>(out, self, other, alpha) ----
    "add.Tensor": ("binary_alpha", None),
    "sub.Tensor": ("binary_alpha", None),
    # ---- binary_cmp: bool out ----
    "eq.Tensor": ("binary_cmp", None),
    "ne.Tensor": ("binary_cmp", None),
    "lt.Tensor": ("binary_cmp", None),
    "gt.Tensor": ("binary_cmp", None),
    "le.Tensor": ("binary_cmp", None),
    "ge.Tensor": ("binary_cmp", None),
    "logical_and": ("binary_cmp", None),
    "logical_or": ("binary_cmp", None),
    # ---- binary_scalar: topsaten<Name>(out, self, scalar) ----
    "pow.Tensor_Scalar": ("binary_scalar", None),
    "remainder.Scalar": ("binary_scalar", None),
    "fmod.Scalar": ("binary_scalar", None),
    # ---- scalar staged as a device tensor (see T_BINARY_SCALAR_AS_TENSOR) ----
    "mul.Scalar": ("binary_scalar_as_tensor", None),
    "div.Scalar": ("binary_scalar_as_tensor", None),
    "add.Scalar": ("binary_scalar_alpha_as_tensor", None),
    "sub.Scalar": ("binary_scalar_alpha_as_tensor", None),
    # ---- binary_scalar_cmp: Tensor + Scalar -> bool ----
    "eq.Scalar": ("binary_scalar_cmp", None),
    "ne.Scalar": ("binary_scalar_cmp", None),
    "lt.Scalar": ("binary_scalar_cmp", None),
    "gt.Scalar": ("binary_scalar_cmp", None),
    "le.Scalar": ("binary_scalar_cmp", None),
    "ge.Scalar": ("binary_scalar_cmp", None),
    # ---- matmul ----
    "mm": ("matmul", None),
    "bmm": ("matmul", None),
    "mm.out": ("matmul_out", None),
    "bmm.out": ("matmul_out", None),
    # ---- reduce over dims with optional out dtype ----
    "sum.dim_IntList": ("reduce_dims_dtype", None),
    "mean.dim": ("reduce_dims_dtype", None),
    # ---- reduce whole tensor ----
    "sum": ("reduce_all_dtype", None),
    "mean": ("reduce_all_dtype", None),
    # ---- reduce over a required dim list (no dtype arg) ----
    "amax": ("reduce_dims_plain", None),
    "amin": ("reduce_dims_plain", None),
    # ---- shape-preserving with one int64 arg ----
    "tril": ("unary_int", None),
    "triu": ("unary_int", None),
    # ---- shape-preserving with a dim list ----
    "flip": ("unary_dims", None),
    # ---- misc ----
    "gelu": ("gelu", None),
    "_softmax": ("softmax_fwd", "SoftmaxForward"),
    "clamp": ("clamp", None),
    "addmm": ("addmm", None),
    "addmm.out": ("addmm_out", None),
    "cat": ("cat", None),
    "zeros_like": ("full_like", "ZerosLike"),
    "ones_like": ("full_like", "OnesLike"),
    "native_layer_norm": ("layer_norm", None),
    # native_layer_norm_backward is deliberately absent: topsaten's kernel
    # rejects every output_mask ("LNB Output mask is not supported now!") and,
    # worse, still returns TOPSATEN_STATUS_SUCCESS while writing nothing to the
    # output buffers, so a caller sees uninitialized gradients rather than an
    # error. Left unregistered -> cpu_fallback. The forward is fine.
    "_softmax_backward_data": ("softmax_bwd", "SoftmaxBackwardData"),
    "silu_backward": ("binary_grad", None),
    "mse_loss": ("loss", None),
    "mse_loss_backward": ("loss_backward", None),
    # ---- AMP GradScaler unscale/check ----
    # topsaten provides both overloads; the generated kernels keep the native
    # path for eligible contiguous floating-point lists and preserve the CPU
    # fallback for unsupported layouts/dtypes.
    "_amp_foreach_non_finite_check_and_unscale_": ("amp_unscale", None),
    "_amp_foreach_non_finite_check_and_unscale.out": ("amp_unscale_out", None),
    # ---- convolution ----
    # aten::convolution routes PrivateUse1 to convolution_overrideable, which
    # has no composite fallback: without a kernel here conv raises
    # NotImplementedError. conv is one of the three autocast lower-precision
    # policies, so AMP needs it.
    "convolution_overrideable": ("convolution", "Convolution"),
    "convolution_backward_overrideable": (
        "convolution_backward",
        "ConvolutionBackward",
    ),
    # ---- foreach: the body of every optimizer step ----
    "_foreach_add_.Scalar": ("foreach_scalar_alpha_inplace", "ForeachAdd"),
    "_foreach_add_.List": ("foreach_list_alpha_inplace", "ForeachAdd"),
    "_foreach_add_.Tensor": ("foreach_tensor_alpha_inplace", "ForeachAdd"),
    "_foreach_sub_.Scalar": ("foreach_scalar_alpha_inplace", "ForeachSub"),
    "_foreach_sub_.List": ("foreach_list_alpha_inplace", "ForeachSub"),
    "_foreach_mul_.Scalar": ("foreach_scalar_inplace", "ForeachMul"),
    "_foreach_mul_.List": ("foreach_list_inplace", "ForeachMul"),
    "_foreach_div_.Scalar": ("foreach_scalar_inplace", "ForeachDiv"),
    "_foreach_div_.List": ("foreach_list_inplace", "ForeachDiv"),
    "_foreach_div_.ScalarList": ("foreach_scalarlist_inplace", "ForeachDiv"),
    "_foreach_neg_": ("foreach_unary_inplace", "ForeachNeg"),
    "_foreach_sqrt_": ("foreach_unary_inplace", "ForeachSqrt"),
    "_foreach_sqrt": ("foreach_unary", "ForeachSqrt"),
    "_foreach_lerp_.Scalar": ("foreach_lerp_scalar_inplace", "ForeachLerp"),
    "_foreach_addcmul_.Scalar": (
        "foreach_ternary_scalar_inplace",
        "ForeachAddcmul",
    ),
    "_foreach_addcmul_.ScalarList": (
        "foreach_ternary_scalarlist_inplace",
        "ForeachAddcmul",
    ),
    "_foreach_addcdiv_.Scalar": (
        "foreach_ternary_scalar_inplace",
        "ForeachAddcdiv",
    ),
    "_foreach_addcdiv_.ScalarList": (
        "foreach_ternary_scalarlist_inplace",
        "ForeachAddcdiv",
    ),
}

# Ops handwritten elsewhere for GCU would double-register the kGcu slot (which
# crashes at import), so they must be excluded here. None yet.
SKIP: set = set()

# Handwritten kernels live in a separate translation unit when the vendor API
# does not fit one of the category templates. They still belong in the generated
# PrivateUse1 coverage include, so list their ATen overloads here.
HANDWRITTEN_OPS = {
    "bernoulli",
    "bernoulli_.float",
    "exponential",
    "exponential_",
    "multinomial",
    "poisson",
    "randn",
    "randn.generator",
    "randn_like.generator",
    "randn_like.generator_out",
    "randint.generator",
    "randint.low_generator",
    "randperm.generator",
    "random_",
    "random_.to",
}

# The int64 CPU-fallback path in each kernel calls back into at::<name>. That is
# the op base name except where the base is not a real at:: function.
AT_OP_OVERRIDES = {
    "mm.out": "mm",
    "bmm.out": "bmm",
    "_softmax": "_softmax",
    # The foreach fallbacks loop over the list applying the equivalent per-tensor
    # Tensor method, rather than calling at::_foreach_* -- the latter would
    # re-enter this same kernel and recurse forever.
    "_foreach_add_.Scalar": "add_",
    "_foreach_add_.List": "add_",
    "_foreach_add_.Tensor": "add_",
    "_foreach_sub_.Scalar": "sub_",
    "_foreach_sub_.List": "sub_",
    "_foreach_mul_.Scalar": "mul_",
    "_foreach_mul_.List": "mul_",
    "_foreach_div_.Scalar": "div_",
    "_foreach_div_.List": "div_",
    "_foreach_div_.ScalarList": "div_",
    "_foreach_neg_": "neg_",
    "_foreach_sqrt_": "sqrt_",
    "_foreach_sqrt": "sqrt",
    "_foreach_lerp_.Scalar": "lerp_",
    "_foreach_addcmul_.Scalar": "addcmul_",
    "_foreach_addcmul_.ScalarList": "addcmul_",
    "_foreach_addcdiv_.Scalar": "addcdiv_",
    "_foreach_addcdiv_.ScalarList": "addcdiv_",
}

# ==========================================================================
# Templates
# ==========================================================================

T_UNARY = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# topsaten does not broadcast for us, so inputs are expanded (and made
# contiguous, since an expanded view has 0-strides) to the common shape first.
# Output dtype follows at::result_type, matching PyTorch's promotion.
#
# `other` may be a CPU tensor: PyTorch wraps a Python number operand into a
# 0-dim CPU tensor and dispatches through the Tensor overload (a * 3.0 ->
# mul.Tensor). Handing that host pointer to topsaten fails in the driver, so any
# non-device operand is moved onto self's device first.
_BINARY_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto other_c = other.to(self.device(), result_dtype);
  auto out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  auto self_b = self_c.expand(out_shape).contiguous();
  auto other_b = other_c.expand(out_shape).contiguous();
"""

T_BINARY = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu(), alpha).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));
  auto t_alpha = gcu::ToTopsatenScalar(alpha, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get(), t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(other.scalar_type())) {{
    return at::{at_op}(self.cpu(), other.cpu()).to(self.device());
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options().dtype(at::kBool));

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

# A Scalar participates in promotion only via its category (integral scalars do
# not widen a float tensor), which is exactly at::result_type(Tensor, Scalar).
_SCALAR_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto t_other = gcu::ToTopsatenScalar(other, result_dtype);
"""

T_BINARY_SCALAR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_SCALAR_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other, alpha).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));
  auto t_alpha = gcu::ToTopsatenScalar(alpha, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other, t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

# add/sub/mul/div reject topsaten's tensor-with-scalar overload in-process (the
# scalar is staged in host memory the driver will not accept), so the scalar is
# materialized as a device tensor and the tensor-with-tensor overload is used.
_SCALAR_AS_TENSOR_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = (self.scalar_type() == result_dtype ? self : self.to(result_dtype))
                    .contiguous();
  auto other_t = gcu::ScalarToDeviceTensor(
      other, self_c.sizes(), self_c.options());
  auto out = at::empty(self_c.sizes(), self.options().dtype(result_dtype));
"""

T_BINARY_SCALAR_AS_TENSOR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_AS_TENSOR_PROLOGUE
    + """\
  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_other(other_t);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_SCALAR_ALPHA_AS_TENSOR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other, alpha).to(self.device());
  }}
"""
    + _SCALAR_AS_TENSOR_PROLOGUE
    + """\
  auto t_alpha = gcu::ToTopsatenScalar(alpha, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_other(other_t);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other.get(), t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_SCALAR_CMP = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
  auto t_other = gcu::ToTopsatenScalar(other, self.scalar_type());

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_other);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_MATMUL = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  auto out = at::empty(out_shape, self.options());

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_mat2(mat2);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_mat2.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_MATMUL_OUT = """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat2.scalar_type())) {{
    out.copy_(at::{at_op}(self.cpu(), mat2.cpu()));
    return out;
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_mat2(mat2);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_mat2.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# sum/mean over an optional dim list. An absent or empty list reduces every
# dim; negative dims are wrapped. Dims are erased high-to-low so an earlier
# erase does not shift a later index. sum promotes integral inputs to int64,
# matching PyTorch.
T_REDUCE_DIMS_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {{
  // Integral reductions accumulate in int64 (PyTorch's rule), which topsaten
  // cannot express, so they take the CPU path along with int64 inputs.
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      ({promote_integral} && at::isIntegralType(self.scalar_type(), true)) ||
      (dtype.has_value() && !gcu::TopsatenSupportsDtype(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
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

  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = at::empty(out_shape, self.options().dtype(out_dtype));
  gcu::TopsatenSizeWrapper t_dims(norm_dims);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_dims.get(),
      keepdim, gcu::ToTopsatenDataType(out_dtype));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_REDUCE_ALL_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self, ::std::optional<at::ScalarType> dtype) {{
  // See T_REDUCE_DIMS_DTYPE: int64 accumulation is not available on topsaten.
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      ({promote_integral} && at::isIntegralType(self.scalar_type(), true)) ||
      (dtype.has_value() && !gcu::TopsatenSupportsDtype(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dtype).to(self.device());
  }}
  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = at::empty({{}}, self.options().dtype(out_dtype));

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(),
      gcu::ToTopsatenDataType(out_dtype));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_GELU = """\
at::Tensor {kernel}(const at::Tensor& self, c10::string_view approximate) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), approximate).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
  std::string approx(approximate);

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), approx.c_str());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_SOFTMAX_FWD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, half_to_float).to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_dtype = half_to_float ? at::kFloat : self.scalar_type();
  auto out = at::empty(self.sizes(), self.options().dtype(out_dtype));

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), d, half_to_float);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# amax/amin: reduce over a required (non-optional, possibly empty) dim list, no
# dtype argument and no integral promotion -- the output dtype is the input's.
T_REDUCE_DIMS_PLAIN = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, keepdim).to(self.device());
  }}
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
    if (keepdim) {{
      out_shape[d] = 1;
    }} else {{
      out_shape.erase(out_shape.begin() + d);
    }}
  }}
  auto out = at::empty(out_shape, self.options());

  gcu::TopsatenSizeWrapper t_dims(norm_dims);
  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, self, t_out.get(), t_self.get(), t_dims.get(), keepdim);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# tril/triu: same shape as the input plus one int64 (the diagonal).
T_UNARY_INT = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t k) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), k).to(self.device());
  }}
  auto self_c = self.contiguous();
  auto out = at::empty(self_c.sizes(), self_c.options());
  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), k);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# flip: same shape as the input plus a dim list. Negative dims are wrapped.
T_UNARY_DIMS = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef dims) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), dims).to(self.device());
  }}
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  for (int64_t d : dims) norm_dims.push_back(d < 0 ? d + ndim : d);
  auto self_c = self.contiguous();
  auto out = at::empty(self_c.sizes(), self_c.options());

  gcu::TopsatenSizeWrapper t_dims(norm_dims);
  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_dims.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# clamp: both bounds are optional. topsaten takes two required scalars, so an
# absent bound becomes the dtype's limit, which is a no-op clamp on that side.
T_CLAMP = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), min, max).to(self.device());
  }}
  auto self_c = self.contiguous();
  auto out = at::empty(self_c.sizes(), self_c.options());
  auto dtype = self.scalar_type();
  auto lo = min.has_value()
      ? gcu::ToTopsatenScalar(min.value(), dtype)
      : gcu::ToTopsatenScalar(gcu::DtypeLowest(dtype), dtype);
  auto hi = max.has_value()
      ? gcu::ToTopsatenScalar(max.value(), dtype)
      : gcu::ToTopsatenScalar(gcu::DtypeHighest(dtype), dtype);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), lo, hi);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# addmm: out = beta * self + alpha * (mat1 @ mat2). `self` (the bias) broadcasts
# in PyTorch but not in topsaten, so it is expanded to the product's shape.
_ADDMM_PROLOGUE = """\
  std::vector<int64_t> out_shape{{mat1.size(0), mat2.size(1)}};
  auto self_b = self.expand(out_shape).contiguous();
  auto mat1_c = mat1.contiguous();
  auto mat2_c = mat2.contiguous();
  auto t_beta = gcu::ToTopsatenScalar(beta, self.scalar_type());
  auto t_alpha = gcu::ToTopsatenScalar(alpha, self.scalar_type());
"""

T_ADDMM = (
    """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat1.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
        .to(self.device());
  }}
"""
    + _ADDMM_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self.options());

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_mat1(mat1_c);
  gcu::TopsatenTensorWrapper t_mat2(mat2_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, self, t_out.get(), t_self.get(), t_mat1.get(), t_mat2.get(),
      t_beta, t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_ADDMM_OUT = (
    """\
at::Tensor& {kernel}(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat1.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(mat2.scalar_type())) {{
    out.copy_(at::{at_op}(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha));
    return out;
  }}
"""
    + _ADDMM_PROLOGUE
    + """\
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_mat1(mat1_c);
  gcu::TopsatenTensorWrapper t_mat2(mat2_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, self, t_out.get(), t_self.get(), t_mat1.get(), t_mat2.get(),
      t_beta, t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

# cat: topsaten takes a std::vector<topsatenTensor>, so the wrappers are held in
# a vector to keep each one's sizes/strides alive for the duration of the call.
# An empty tensor is skipped, matching PyTorch's treatment of it as absent.
T_CAT = """\
at::Tensor {kernel}(const at::ITensorListRef& tensors, int64_t dim) {{
  std::vector<at::Tensor> inputs;
  bool cpu_path = false;
  for (const at::Tensor& t : tensors) {{
    if (t.numel() == 0 && t.dim() == 1) continue;
    if (!gcu::TopsatenSupportsDtype(t.scalar_type())) cpu_path = true;
    inputs.push_back(t);
  }}
  TORCH_CHECK(!inputs.empty(), "flagos cat: expected at least one tensor");
  if (cpu_path) {{
    std::vector<at::Tensor> host;
    host.reserve(inputs.size());
    for (const auto& t : inputs) host.push_back(t.cpu());
    return at::{at_op}(host, dim).to(inputs[0].device());
  }}
  int64_t ndim = inputs[0].dim();
  int64_t d = dim < 0 ? dim + ndim : dim;

  auto out_shape = inputs[0].sizes().vec();
  int64_t total = 0;
  for (const auto& t : inputs) total += t.size(d);
  out_shape[d] = total;
  auto out = at::empty(out_shape, inputs[0].options());

  std::vector<at::Tensor> contig;
  contig.reserve(inputs.size());
  std::vector<std::unique_ptr<gcu::TopsatenTensorWrapper>> keep;
  keep.reserve(inputs.size());
  std::vector<topsatenTensor> tops_in;
  tops_in.reserve(inputs.size());
  for (const auto& t : inputs) {{
    contig.push_back(t.contiguous());
    keep.push_back(
        std::make_unique<gcu::TopsatenTensorWrapper>(contig.back()));
    tops_in.push_back(keep.back()->get());
  }}
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, inputs[0], t_out.get(), tops_in, d);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# zeros_like / ones_like: the dtype/layout/device/pin/memory_format options
# describe the *result*, so a request for a different device or a dtype topsaten
# cannot express is handed to the generic implementation instead.
T_FULL_LIKE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    ::std::optional<at::MemoryFormat> memory_format) {{
  auto out_dtype = dtype.value_or(self.scalar_type());
  auto target_device = device.value_or(self.device());
  if (target_device != self.device() ||
      !gcu::TopsatenSupportsDtype(out_dtype) ||
      !gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    // Built on a CPU tensor so the call lands on the CPU kernel: passing the
    // flagos tensor back to at::{at_op} would re-enter this kernel.
    auto host = at::{at_op}(
        self.cpu(), out_dtype, layout, at::kCPU, pin_memory, memory_format);
    return target_device.type() == at::kCPU ? host : host.to(target_device);
  }}
  // at::empty rejects MemoryFormat::Preserve (autograd passes it for every
  // seed gradient), so resolve it against the input first. A non-contiguous
  // result would leave topsaten writing through strides it does not honour for
  // this op, so those go to the host path.
  auto fmt = memory_format.value_or(at::MemoryFormat::Contiguous);
  if (fmt == at::MemoryFormat::Preserve) {{
    fmt = self.suggest_memory_format();
  }}
  if (fmt != at::MemoryFormat::Contiguous) {{
    auto host = at::{at_op}(
        self.cpu(), out_dtype, layout, at::kCPU, pin_memory, memory_format);
    return host.to(target_device);
  }}
  auto options = self.options().dtype(out_dtype);
  auto out = at::empty(self.sizes(), options, fmt);

  auto self_c = self.contiguous();
  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, self, t_out.get(), t_self.get(),
      gcu::ToTopsatenDataType(out_dtype), TOPSATEN_LAYOUT_STRIDED,
      TOPSATEN_MEMORY_CONTIGUOUS);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# native_layer_norm -> (out, mean, rstd). mean/rstd keep the leading (un-
# normalized) dims and carry 1s for the normalized tail, which is the shape
# topsaten writes even though PyTorch's public shape drops the trailing 1s.
# weight/bias are optional in PyTorch but required by topsaten, so they are
# materialized as ones/zeros when absent.
T_LAYER_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    double eps) {{
  if (!gcu::TopsatenSupportsDtype(input.scalar_type())) {{
    auto r = at::{at_op}(
        input.cpu(),
        normalized_shape,
        weight.has_value() ? ::std::optional<at::Tensor>(weight->cpu())
                           : ::std::nullopt,
        bias.has_value() ? ::std::optional<at::Tensor>(bias->cpu())
                         : ::std::nullopt,
        eps);
    return {{std::get<0>(r).to(input.device()),
            std::get<1>(r).to(input.device()),
            std::get<2>(r).to(input.device())}};
  }}
  auto input_c = input.contiguous();
  int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  int64_t outer = input_c.dim() - norm_ndim;

  auto stat_shape = input_c.sizes().vec();
  for (int64_t i = outer; i < input_c.dim(); ++i) stat_shape[i] = 1;

  auto out = at::empty(input_c.sizes(), input_c.options());
  auto mean = at::empty(stat_shape, input_c.options());
  auto rstd = at::empty(stat_shape, input_c.options());

  auto w = weight.has_value() && weight->defined()
      ? weight->contiguous()
      : at::ones(normalized_shape, input_c.options());
  auto b = bias.has_value() && bias->defined()
      ? bias->contiguous()
      : at::zeros(normalized_shape, input_c.options());

  gcu::TopsatenSizeWrapper t_shape(normalized_shape);
  gcu::TopsatenTensorWrapper t_in(input_c);
  gcu::TopsatenTensorWrapper t_w(w);
  gcu::TopsatenTensorWrapper t_b(b);
  gcu::TopsatenTensorWrapper t_out(out);
  gcu::TopsatenTensorWrapper t_mean(mean);
  gcu::TopsatenTensorWrapper t_rstd(rstd);
  auto t_eps = gcu::ToTopsatenScalar(at::Scalar(eps), input.scalar_type());
  EXEC_TOPSATEN_CMD(
      {tops}, input, t_out.get(), t_mean.get(), t_rstd.get(), t_in.get(),
      t_shape.get(), t_w.get(), t_b.get(), t_eps);

  // PyTorch's mean/rstd drop the normalized dims entirely.
  std::vector<int64_t> pt_stat(
      input_c.sizes().begin(), input_c.sizes().begin() + outer);
  pt_stat.push_back(1);
  return {{out, mean.reshape(pt_stat), rstd.reshape(pt_stat)}};
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _softmax_backward_data(grad_output, output, dim, input_dtype)
T_SOFTMAX_BWD = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {{
  if (!gcu::TopsatenSupportsDtype(grad_output.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(input_dtype)) {{
    return at::{at_op}(grad_output.cpu(), output.cpu(), dim, input_dtype)
        .to(grad_output.device());
  }}
  int64_t d = dim < 0 ? dim + output.dim() : dim;
  auto grad_c = grad_output.contiguous();
  auto out_c = output.contiguous();
  auto result = at::empty(out_c.sizes(), out_c.options().dtype(input_dtype));

  gcu::TopsatenTensorWrapper t_grad(grad_c);
  gcu::TopsatenTensorWrapper t_out(out_c);
  gcu::TopsatenTensorWrapper t_result(result);
  EXEC_TOPSATEN_CMD(
      {tops}, grad_output, t_result.get(), t_grad.get(), t_out.get(), d,
      gcu::ToTopsatenDataType(input_dtype));
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# silu_backward(grad_output, self) -> grad_input, all the same shape.
T_BINARY_GRAD = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(grad_output.scalar_type())) {{
    return at::{at_op}(grad_output.cpu(), self.cpu()).to(self.device());
  }}
  auto grad_c = grad_output.contiguous();
  auto self_c = self.contiguous();
  auto grad_input = at::empty(self_c.sizes(), self_c.options());

  gcu::TopsatenTensorWrapper t_grad(grad_c);
  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_gi(grad_input);
  EXEC_TOPSATEN_CMD({tops}, self, t_gi.get(), t_grad.get(), t_self.get());
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# mse_loss(self, target, reduction). reduction 0=none keeps the input shape;
# 1=mean and 2=sum produce a scalar.
T_LOSS = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(target.scalar_type())) {{
    return at::{at_op}(self.cpu(), target.cpu(), reduction).to(self.device());
  }}
  auto self_c = self.contiguous();
  auto target_c = target.to(self.device()).contiguous();
  auto out = reduction == 0
      ? at::empty(self_c.sizes(), self_c.options())
      : at::empty({{}}, self_c.options());

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_target(target_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, self, t_out.get(), t_self.get(), t_target.get(), reduction);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# mse_loss_backward(grad_output, self, target, reduction) -> grad_input.
T_LOSS_BACKWARD = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {{
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(target.scalar_type())) {{
    return at::{at_op}(
               grad_output.cpu(), self.cpu(), target.cpu(), reduction)
        .to(self.device());
  }}
  auto self_c = self.contiguous();
  auto target_c = target.to(self.device()).contiguous();
  // topsaten does not broadcast the (scalar) grad for a reduced loss.
  auto grad_c = grad_output.to(self.device())
                    .expand(self_c.sizes())
                    .contiguous();
  auto grad_input = at::empty(self_c.sizes(), self_c.options());

  gcu::TopsatenTensorWrapper t_grad(grad_c);
  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_target(target_c);
  gcu::TopsatenTensorWrapper t_gi(grad_input);
  EXEC_TOPSATEN_CMD(
      {tops}, self, t_gi.get(), t_grad.get(), t_self.get(), t_target.get(),
      reduction);
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# ---------------------------------------------------------------------------
# foreach. These are what an optimizer step is made of, so keeping them off the
# host is worth more than any single elementwise op: one AdamW step over a
# 2-block transformer issued ~20 foreach calls, each of which was copying every
# parameter to the CPU and back.
#
# The in-place variants pass the same list as both source and destination --
# verified on hardware that topsaten accepts an aliasing output -- so the
# parameters are updated in place with no extra copy. A list that is not
# uniformly contiguous/supported goes to the generic implementation via
# at::_foreach_*, which is why every kernel opens with IsForeachEligible().
# ---------------------------------------------------------------------------

# convolution_overrideable: topsatenConvolution(out, input, weight, bias, ...).
# topsaten requires a bias tensor, and signals "no bias" with a default-
# constructed topsatenTensor (dtype TOPSATEN_DATA_NONE) rather than a null
# pointer, so an absent bias is materialized as zeros. Output shape comes from
# ATen's own conv shape math so the result matches the CPU reference exactly.
T_CONVOLUTION = """\
at::Tensor {kernel}(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ::std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {{
  if (!gcu::TopsatenSupportsDtype(input.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(weight.scalar_type()) || transposed) {{
    auto r = at::convolution(
        input.cpu(), weight.cpu(),
        bias.has_value() && bias->defined()
            ? ::std::optional<at::Tensor>(bias->cpu())
            : ::std::nullopt,
        stride, padding, dilation, transposed, output_padding, groups);
    return r.to(input.device());
  }}
  auto input_c = input.contiguous();
  auto weight_c = weight.contiguous();

  auto out_sizes = at::native::conv_output_size(
      input_c.sizes(), weight_c.sizes(), padding, stride, dilation);
  auto out = at::empty(out_sizes, input_c.options());

  // topsaten has no "absent bias" sentinel usable from here, so a zero bias
  // reproduces the unbiased result.
  at::Tensor bias_c = bias.has_value() && bias->defined()
      ? bias->contiguous()
      : at::zeros({{weight_c.size(0)}}, weight_c.options());

  gcu::TopsatenTensorWrapper t_out(out);
  gcu::TopsatenTensorWrapper t_input(input_c);
  gcu::TopsatenTensorWrapper t_weight(weight_c);
  gcu::TopsatenTensorWrapper t_bias(bias_c);
  gcu::TopsatenSizeWrapper w_stride(stride);
  gcu::TopsatenSizeWrapper w_padding(padding);
  gcu::TopsatenSizeWrapper w_dilation(dilation);
  gcu::TopsatenSizeWrapper w_output_padding(output_padding);
  EXEC_TOPSATEN_CMD(
      {tops}, input_c, t_out.get(), t_input.get(), t_weight.get(),
      t_bias.get(), w_stride.get(), w_padding.get(), w_dilation.get(),
      transposed, w_output_padding.get(), groups);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# convolution_backward_overrideable: like the forward, autograd has no composite
# fallback for it, so without a kernel here any backward through a conv raises
# NotImplementedError.
#
# topsatenConvolutionBackward is declared by the SDK headers and exported by
# libtopsaten.so.3, but on the measured S60 it returns NOT_SUPPORT for every
# input we tried (fp32 and fp16, grouped and ungrouped, with and without
# padding, and with both the caller's output_mask and an all-true mask), with no
# vendor-side diagnostic. So this route is a correctness-first CPU fallback:
# the grads are computed by the reference kernel and copied back to the device.
# Switch it to the native call once a TopsRider release implements it.
T_CONVOLUTION_BACKWARD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {{
  std::vector<int64_t> bias_sizes{{weight.size(0)}};
  auto r = at::convolution_backward(
      grad_output.cpu(), input.cpu(), weight.cpu(),
      ::std::optional<at::IntArrayRef>(at::IntArrayRef(bias_sizes)),
      stride, padding, dilation, transposed, output_padding, groups,
      output_mask);
  auto to_dev = [&](const at::Tensor& t) {{
    return t.defined() ? t.to(input.device()) : t;
  }};
  return {{to_dev(std::get<0>(r)), to_dev(std::get<1>(r)),
          to_dev(std::get<2>(r))}};
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_neg_ / (any in-place unary foreach): topsatenForeachX(out, in)
T_AMP_UNSCALE = """\
void {kernel}(at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale) {{
  if (self.empty() || !gcu::IsForeachEligible(self) ||
      !found_inf.defined() || !inv_scale.defined() ||
      !found_inf.is_contiguous() || !inv_scale.is_contiguous() ||
      found_inf.device() != self[0].device() ||
      inv_scale.device() != self[0].device() ||
      !gcu::TopsatenSupportsDtype(found_inf.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(inv_scale.scalar_type())) {{
    std::vector<at::Tensor> cpu_self;
    cpu_self.reserve(self.size());
    for (const auto& tensor : self) cpu_self.push_back(tensor.cpu());
    auto cpu_found_inf = found_inf.cpu();
    auto cpu_inv_scale = inv_scale.cpu();
    at::_amp_foreach_non_finite_check_and_unscale_(
        cpu_self, cpu_found_inf, cpu_inv_scale);
    for (size_t i = 0; i < self.size(); ++i) self[i].copy_(cpu_self[i]);
    found_inf.copy_(cpu_found_inf);
    return;
  }}

  std::vector<at::Tensor> out;
  out.reserve(self.size());
  for (const auto& tensor : self) out.push_back(at::empty_like(tensor));
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(out);
  gcu::TopsatenTensorWrapper t_found_inf(found_inf);
  gcu::TopsatenTensorWrapper t_inv_scale(inv_scale);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_found_inf.get(), t_self.get(),
      t_inv_scale.get());
  for (size_t i = 0; i < self.size(); ++i) self[i].copy_(out[i]);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_AMP_UNSCALE_OUT = """\
void {kernel}(at::TensorList self, at::Tensor& found_inf,
             const at::Tensor& inv_scale, at::TensorList out) {{
  TORCH_CHECK(self.size() == out.size(),
              "{disp}: tensor lists must match in length");
  if (self.empty() || !gcu::IsForeachEligible(self) ||
      !gcu::IsForeachEligible(out) || !found_inf.defined() ||
      !inv_scale.defined() || !found_inf.is_contiguous() ||
      !inv_scale.is_contiguous() || found_inf.device() != self[0].device() ||
      inv_scale.device() != self[0].device() ||
      !gcu::TopsatenSupportsDtype(found_inf.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(inv_scale.scalar_type())) {{
    std::vector<at::Tensor> cpu_self;
    cpu_self.reserve(self.size());
    for (const auto& tensor : self) cpu_self.push_back(tensor.cpu());
    auto cpu_found_inf = found_inf.cpu();
    auto cpu_inv_scale = inv_scale.cpu();
    std::vector<at::Tensor> cpu_out;
    cpu_out.reserve(out.size());
    for (const auto& tensor : out) cpu_out.push_back(tensor.cpu());
    at::_amp_foreach_non_finite_check_and_unscale_outf(
        cpu_self, cpu_found_inf, cpu_inv_scale, cpu_out);
    for (size_t i = 0; i < out.size(); ++i) out[i].copy_(cpu_out[i]);
    // The reference leaves found_inf alone for this overload; propagate only
    // what it actually wrote so both routes agree.
    found_inf.copy_(cpu_found_inf);
    return;
  }}

  // The CPU reference for the .out overload writes the unscaled values but
  // leaves found_inf untouched; topsaten updates it. Give the vendor call a
  // scratch flag so the observable contract matches the other backends.
  // GradScaler uses the in-place overload for overflow detection.
  auto scratch_found_inf = at::empty_like(found_inf);
  scratch_found_inf.copy_(found_inf);
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(out);
  gcu::TopsatenTensorWrapper t_found_inf(scratch_found_inf);
  gcu::TopsatenTensorWrapper t_inv_scale(inv_scale);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_found_inf.get(), t_self.get(),
      t_inv_scale.get());
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_FOREACH_UNARY_INPLACE = """\
void {kernel}(at::TensorList self) {{
  if (!gcu::IsForeachEligible(self)) {{
    for (const at::Tensor& t : self) t.{at_op}();
    return;
  }}
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD({tops}, self[0], t_out.get(), t_self.get());
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_sqrt: out-of-place, returns a fresh list.
T_FOREACH_UNARY = """\
::std::vector<at::Tensor> {kernel}(at::TensorList self) {{
  if (!gcu::IsForeachEligible(self)) {{
    std::vector<at::Tensor> out;
    out.reserve(self.size());
    for (const at::Tensor& t : self) out.push_back(t.{at_op}());
    return out;
  }}
  std::vector<at::Tensor> out;
  out.reserve(self.size());
  for (const at::Tensor& t : self) {{
    out.push_back(at::empty(t.sizes(), t.options()));
  }}
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self[0], t_out.get(), t_self.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_mul_.Scalar / _foreach_div_.Scalar: one scalar for the whole list.
T_FOREACH_SCALAR_INPLACE = """\
void {kernel}(at::TensorList self, const at::Scalar& scalar) {{
  if (!gcu::IsForeachEligible(self)) {{
    for (const at::Tensor& t : self) t.{at_op}(scalar);
    return;
  }}
  auto t_scalar = gcu::ToTopsatenScalar(scalar, self[0].scalar_type());
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD({tops}, self[0], t_out.get(), t_self.get(), t_scalar);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_add_.Scalar: same as above but the overload takes a trailing alpha.
T_FOREACH_SCALAR_ALPHA_INPLACE = """\
void {kernel}(at::TensorList self, const at::Scalar& scalar) {{
  if (!gcu::IsForeachEligible(self)) {{
    for (const at::Tensor& t : self) t.{at_op}(scalar);
    return;
  }}
  auto t_scalar = gcu::ToTopsatenScalar(scalar, self[0].scalar_type());
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD({tops}, self[0], t_out.get(), t_self.get(), t_scalar);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_div_.ScalarList: one scalar per tensor.
T_FOREACH_SCALARLIST_INPLACE = """\
void {kernel}(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {{
  if (!gcu::IsForeachEligible(self) || scalars.size() != self.size()) {{
    for (size_t i = 0; i < self.size(); ++i) self[i].{at_op}(scalars[i]);
    return;
  }}
  std::vector<topsatenScalar_t> t_scalars;
  t_scalars.reserve(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i) {{
    t_scalars.push_back(
        gcu::ToTopsatenScalar(scalars[i], self[i].scalar_type()));
  }}
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD({tops}, self[0], t_out.get(), t_self.get(), t_scalars);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_mul_.List / _foreach_div_.List: elementwise against a second list.
T_FOREACH_LIST_INPLACE = """\
void {kernel}(at::TensorList self, at::TensorList other) {{
  if (!gcu::IsForeachEligible(self) || !gcu::IsForeachEligible(other) ||
      self.size() != other.size()) {{
    for (size_t i = 0; i < self.size(); ++i) self[i].{at_op}(other[i]);
    return;
  }}
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_other(other);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_self.get(), t_other.get());
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_add_.List: two lists plus alpha.
T_FOREACH_LIST_ALPHA_INPLACE = """\
void {kernel}(
    at::TensorList self,
    at::TensorList other,
    const at::Scalar& alpha) {{
  if (!gcu::IsForeachEligible(self) || !gcu::IsForeachEligible(other) ||
      self.size() != other.size()) {{
    for (size_t i = 0; i < self.size(); ++i) {{
      self[i].{at_op}(other[i], alpha);
    }}
    return;
  }}
  auto t_alpha = gcu::ToTopsatenScalar(alpha, self[0].scalar_type());
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_other(other);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_self.get(), t_other.get(), t_alpha);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_add_.Tensor: a single tensor broadcast across the whole list.
T_FOREACH_TENSOR_ALPHA_INPLACE = """\
void {kernel}(
    at::TensorList self,
    const at::Tensor& other,
    const at::Scalar& alpha) {{
  // aten requires `other` to be 0-dim here (it is typically a device-resident
  // learning rate). topsaten rejects rank-0 shapes, but the wrapper presents
  // those as shape {{1}} and the op broadcasts a 1-element rhs across every
  // list tensor -- verified on hardware.
  if (!gcu::IsForeachEligible(self) || !other.defined() ||
      other.numel() != 1 || !other.is_contiguous() ||
      !gcu::TopsatenSupportsDtype(other.scalar_type()) ||
      other.device() != self[0].device()) {{
    for (const at::Tensor& t : self) t.{at_op}(other, alpha);
    return;
  }}
  auto t_alpha = gcu::ToTopsatenScalar(alpha, self[0].scalar_type());
  gcu::TopsatenTensorWrapper t_other(other);
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_self.get(), t_other.get(), t_alpha);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_lerp_.Scalar: self + weight * (other - self), over two lists.
T_FOREACH_LERP_SCALAR_INPLACE = """\
void {kernel}(
    at::TensorList self,
    at::TensorList tensors1,
    const at::Scalar& weight) {{
  if (!gcu::IsForeachEligible(self) || !gcu::IsForeachEligible(tensors1) ||
      self.size() != tensors1.size()) {{
    for (size_t i = 0; i < self.size(); ++i) {{
      self[i].{at_op}(tensors1[i], weight);
    }}
    return;
  }}
  auto t_weight = gcu::ToTopsatenScalar(weight, self[0].scalar_type());
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_end(tensors1);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_self.get(), t_end.get(), t_weight);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_addcmul_.Scalar / _foreach_addcdiv_.Scalar:
#   self += value * (tensor1 op tensor2)
T_FOREACH_TERNARY_SCALAR_INPLACE = """\
void {kernel}(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    const at::Scalar& value) {{
  if (!gcu::IsForeachEligible(self) || !gcu::IsForeachEligible(tensor1) ||
      !gcu::IsForeachEligible(tensor2) || self.size() != tensor1.size() ||
      self.size() != tensor2.size()) {{
    for (size_t i = 0; i < self.size(); ++i) {{
      self[i].{at_op}(tensor1[i], tensor2[i], value);
    }}
    return;
  }}
  auto t_value = gcu::ToTopsatenScalar(value, self[0].scalar_type());
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_t1(tensor1);
  gcu::TopsatenTensorList t_t2(tensor2);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_self.get(), t_t1.get(), t_t2.get(),
      t_value);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _foreach_addcdiv_.ScalarList / _foreach_addcmul_.ScalarList: per-tensor value.
T_FOREACH_TERNARY_SCALARLIST_INPLACE = """\
void {kernel}(
    at::TensorList self,
    at::TensorList tensor1,
    at::TensorList tensor2,
    at::ArrayRef<at::Scalar> scalars) {{
  if (!gcu::IsForeachEligible(self) || !gcu::IsForeachEligible(tensor1) ||
      !gcu::IsForeachEligible(tensor2) || self.size() != tensor1.size() ||
      self.size() != tensor2.size() || scalars.size() != self.size()) {{
    for (size_t i = 0; i < self.size(); ++i) {{
      self[i].{at_op}(tensor1[i], tensor2[i], scalars[i]);
    }}
    return;
  }}
  std::vector<topsatenScalar_t> t_scalars;
  t_scalars.reserve(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i) {{
    t_scalars.push_back(
        gcu::ToTopsatenScalar(scalars[i], self[i].scalar_type()));
  }}
  gcu::TopsatenTensorList t_self(self);
  gcu::TopsatenTensorList t_t1(tensor1);
  gcu::TopsatenTensorList t_t2(tensor2);
  gcu::TopsatenTensorList t_out(self);
  EXEC_TOPSATEN_CMD(
      {tops}, self[0], t_out.get(), t_self.get(), t_t1.get(), t_t2.get(),
      t_scalars);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

CATEGORIES = {
    "unary": T_UNARY,
    "binary": T_BINARY,
    "binary_alpha": T_BINARY_ALPHA,
    "binary_cmp": T_BINARY_CMP,
    "binary_scalar": T_BINARY_SCALAR,
    "binary_scalar_alpha": T_BINARY_SCALAR_ALPHA,
    "binary_scalar_as_tensor": T_BINARY_SCALAR_AS_TENSOR,
    "binary_scalar_alpha_as_tensor": T_BINARY_SCALAR_ALPHA_AS_TENSOR,
    "binary_scalar_cmp": T_BINARY_SCALAR_CMP,
    "matmul": T_MATMUL,
    "matmul_out": T_MATMUL_OUT,
    "reduce_dims_dtype": T_REDUCE_DIMS_DTYPE,
    "reduce_all_dtype": T_REDUCE_ALL_DTYPE,
    "reduce_dims_plain": T_REDUCE_DIMS_PLAIN,
    "unary_int": T_UNARY_INT,
    "unary_dims": T_UNARY_DIMS,
    "clamp": T_CLAMP,
    "addmm": T_ADDMM,
    "addmm_out": T_ADDMM_OUT,
    "cat": T_CAT,
    "full_like": T_FULL_LIKE,
    "layer_norm": T_LAYER_NORM,
    "softmax_bwd": T_SOFTMAX_BWD,
    "binary_grad": T_BINARY_GRAD,
    "loss": T_LOSS,
    "loss_backward": T_LOSS_BACKWARD,
    "amp_unscale": T_AMP_UNSCALE,
    "amp_unscale_out": T_AMP_UNSCALE_OUT,
    "convolution": T_CONVOLUTION,
    "convolution_backward": T_CONVOLUTION_BACKWARD,
    "gelu": T_GELU,
    "softmax_fwd": T_SOFTMAX_FWD,
    "foreach_unary": T_FOREACH_UNARY,
    "foreach_unary_inplace": T_FOREACH_UNARY_INPLACE,
    "foreach_scalar_inplace": T_FOREACH_SCALAR_INPLACE,
    "foreach_scalar_alpha_inplace": T_FOREACH_SCALAR_ALPHA_INPLACE,
    "foreach_scalarlist_inplace": T_FOREACH_SCALARLIST_INPLACE,
    "foreach_list_inplace": T_FOREACH_LIST_INPLACE,
    "foreach_list_alpha_inplace": T_FOREACH_LIST_ALPHA_INPLACE,
    "foreach_tensor_alpha_inplace": T_FOREACH_TENSOR_ALPHA_INPLACE,
    "foreach_lerp_scalar_inplace": T_FOREACH_LERP_SCALAR_INPLACE,
    "foreach_ternary_scalar_inplace": T_FOREACH_TERNARY_SCALAR_INPLACE,
    "foreach_ternary_scalarlist_inplace": T_FOREACH_TERNARY_SCALARLIST_INPLACE,
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_gcu.py -- DO NOT EDIT.
//
// topsaten kernels for the Enflame GCU backend, generated per-category. Each
// kernel wraps its aten tensors into topsatenTensors and issues one direct
// topsaten call via EXEC_TOPSATEN_CMD. Dispatchers are declared in
// generated/ops.h (shared with the CUDA codegen); here we only fill the
// Backend::kGcu slot.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/zeros.h>
#include <ATen/native/ConvUtils.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <string>
#include <vector>
#include "../topsaten_common.h"

namespace at::native::flagos {

namespace gcu = at::native::flagos::gcu;

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"

INC_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_gcu.py -- DO NOT EDIT.
//
// m.impl() lines for the ops that have a topsaten kernel. Included by
// register.cc inside TORCH_LIBRARY_IMPL(aten, PrivateUse1) when USE_GCU is
// set, in place of the full generated/register.inc list: an op registered on
// PrivateUse1 without a kernel behind it fails the dispatcher's
// "backend not registered" check, whereas an unregistered op simply reaches
// the cpu_fallback. So this file is exactly the GCU coverage set.

"""


def topsaten_name(op_base: str, override) -> str:
    if override:
        return "topsaten" + override
    pascal = "".join(w.capitalize() for w in op_base.lstrip("_").split("_") if w)
    return "topsaten" + pascal


def libtopsaten_path() -> Path:
    env = os.environ.get("TOPSATEN_LIB")
    if env:
        return Path(env)
    for cand in ("/usr/lib/libtopsaten.so", "/opt/tops/lib/libtopsaten.so"):
        if Path(cand).exists():
            return Path(cand)
    return Path("/usr/lib/libtopsaten.so")


def symbols(lib: Path):
    """topsaten entry points are C++ functions in `namespace topsaten`, so the
    mangled names must be demangled (nm -C) before matching."""
    if not lib.exists():
        print(f"[warn] {lib} not found; skipping symbol validation", file=sys.stderr)
        return None
    out = subprocess.run(
        ["nm", "-DC", "--defined-only", str(lib)],
        capture_output=True,
        text=True,
        check=False,
    )
    return set(re.findall(r"topsaten::(topsaten\w+)", out.stdout))


def wrapper_map():
    """op name -> Wrapper<Name>, read back from the CUDA codegen's register.inc
    so the m.impl() subset we emit cannot drift from the wrappers that exist."""
    if not REGISTER_INC.exists():
        return {}
    text = REGISTER_INC.read_text()
    return dict(
        re.findall(
            r'^\s*m\.impl\("([^"]+)",\s*(\w+)\);',
            text,
            flags=re.MULTILINE,
        )
    )


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
        help="do not append covered ops to backends_gcu.conf",
    )
    args = ap.parse_args()

    syms = symbols(libtopsaten_path())
    wrappers = wrapper_map()

    bodies = []
    covered = []  # (op, tops, category)
    skipped = []  # (op, reason)

    for op, (cat, override) in OPS.items():
        if args.category != "all" and cat != args.category:
            continue
        if op in SKIP:
            skipped.append((op, "handwritten"))
            continue
        if op in HANDWRITTEN_OPS:
            if op not in wrappers:
                skipped.append((op, "no wrapper in generated/register.inc"))
                continue
            covered.append((op, "handwritten", "handwritten"))
            continue
        base = op.split(".")[0]
        tops = topsaten_name(base, override)
        if syms is not None and tops not in syms:
            skipped.append((op, f"{tops} not in {libtopsaten_path().name}"))
            continue
        if op not in wrappers:
            skipped.append((op, "no wrapper in generated/register.inc"))
            continue
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelGcu"  # SqrtFn -> SqrtKernelGcu
        bodies.append(
            CATEGORIES[cat].format(
                kernel=kernel,
                tops=tops,
                fn=fn,
                disp=disp,
                at_op=AT_OP_OVERRIDES.get(op, base),
                promote_integral="true" if base == "sum" else "false",
            )
        )
        covered.append((op, tops, cat))

    for op in sorted(HANDWRITTEN_OPS):
        if op in wrappers and not any(item[0] == op for item in covered):
            covered.append((op, "handwritten", "handwritten"))

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(FILE_HEADER + "\n".join(bodies) + FILE_FOOTER)

    impls = "".join(
        f'  m.impl("{op}", {wrappers[op]});\n' for op, _, _ in sorted(covered)
    )
    OUT_INC.write_text(INC_HEADER + impls)

    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({len(covered)} kernels)")
    print(f"[gen] {OUT_INC.relative_to(REPO)}  ({len(covered)} m.impl lines)")
    by_cat = {}
    for op, tops, cat in covered:
        by_cat.setdefault(cat, []).append((op, tops))
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if items:
            print(f"    [{cat}] {len(items)}")
            for op, tops in items:
                print(f"       + {op} -> {tops}")
    for op, why in skipped:
        print(f"       - {op} skipped ({why})")

    if not args.no_conf and covered:
        existing = CONF.read_text() if CONF.exists() else ""
        # Strip any prior codegen block so re-runs stay idempotent.
        marker = "\n# --- generated by codegen_gcu.py"
        if marker in existing:
            existing = existing[: existing.index(marker)].rstrip() + "\n"
        lines = []
        for op, _, _ in covered:
            if f"\n{op} = " not in ("\n" + existing):
                lines.append(f"{op} = gcu")
        new = existing.rstrip() + "\n"
        if lines:
            new += "\n# --- generated by codegen_gcu.py ---\n"
            new += "\n".join(lines) + "\n"
        CONF.write_text(new)
        print(f"[conf] wrote {len(lines)} generated op(s) to {CONF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
