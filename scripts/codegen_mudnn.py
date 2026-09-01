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
Codegen for torch_fl Moore Threads MUSA (mudnn) operators.

Same problem as Ascend and GCU: there is no vendor dispatch key to box into (the
MUSA toolkit ships no CUDA runtime), so every kernel calls the vendor op library
-- libmudnn.so -- directly. The call shape is uniform per *category*, so this
generator is category-driven, mirroring scripts/codegen_gcu.py.

This replaces the earlier codegen_musa.py, which emitted 1186 passthroughs to
torch_musa's flat `at::musa::*` API. That API lives in libmusa_python.so, which
links against torch and therefore embeds torch's C++ object layout -- pinning
the plugin to one exact torch build (sizeof(c10::MessageLogger) went 408 -> 400
between 2.9.1 and 2.10, which corrupts the vendor .so's stack). mudnn pulls in
no torch symbols at all, so this route is torch-version-agnostic.

Unlike topsaten, mudnn is configured then run rather than called in one shot:

    Unary op;
    op.SetMode(Unary::Mode::ADD);
    op.SetAlpha(alpha);
    op.Run(handle, out, self);

Two mudnn properties shape the templates, both verified directly against the
library rather than assumed:

  - mudnn Tensors carry strides on *both* operands, and honour 0-strides. So
    broadcasting is expressed with expand() alone (a view) and non-contiguous
    inputs are read in place -- no `.contiguous()` materialization anywhere,
    which is where the GCU templates have to spend an extra copy.
  - int64 works across Unary/Binary/Reduce/MatMul. topsaten has no int64
    kernels at all, so the GCU templates carry an int64 CPU-fallback branch;
    here only genuinely unmapped dtypes (complex, quantized) fall back.

Generates:
  - csrc/aten/backends/musa/generated/musa_kernels.cc
      the kernels + REGISTER_IMPL_TO_DISPATCHER(..., Backend::kMusa, ...)
  - csrc/aten/backends/musa/generated/musa_register.inc
      the m.impl() subset for register.cc. MUSA registers PrivateUse1 ONLY for
      ops it has a kernel for; everything else stays unregistered and reaches
      the cpu_fallback (registering an op with no kernel behind it would instead
      hit the dispatcher's "backend not registered" check).
  - appends `<op> = musa` to torch_fl/configs/backends_musa.conf
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

REPO = Path(__file__).resolve().parent.parent
OUT_CC = REPO / "csrc/aten/backends/musa/generated/musa_kernels.cc"
OUT_INC = REPO / "csrc/aten/backends/musa/generated/musa_register.inc"
REGISTER_INC = REPO / "csrc/aten/generated/register.inc"
CONF = REPO / "torch_fl/configs/backends_musa.conf"

# --------------------------------------------------------------------------
# Op registry: schema op name -> (category, mudnn mode).
#
# The mode is the enum entry inside the category's op class, e.g. "ABS" means
# Unary::Mode::ABS. Operand order for the alpha-carrying modes was confirmed on
# device: SUB/DIV/POW/FLOORMOD are `self OP alpha`, and *_BY_ALPHA is the
# reverse -- so no scalar overload silently computes backwards.
# --------------------------------------------------------------------------
OPS = {
    # ---- unary: Unary::Mode::<M>, Run(out, self) ----
    "abs": ("unary", "ABS"),
    "sqrt": ("unary", "SQRT"),
    "rsqrt": ("unary", "RSQRT"),
    "exp": ("unary", "EXP"),
    "log": ("unary", "LOG"),
    "log2": ("unary", "LOG2"),
    "log10": ("unary", "LOG10"),
    "log1p": ("unary", "LOG1P"),
    "sin": ("unary", "SIN"),
    "cos": ("unary", "COS"),
    "acos": ("unary", "ACOS"),
    "atan": ("unary", "ATAN"),
    "tanh": ("unary", "TANH"),
    "sigmoid": ("unary", "SIGMOID"),
    "silu": ("unary", "SILU"),
    "relu": ("unary", "RELU"),
    "reciprocal": ("unary", "RECIPROCAL"),
    "erf": ("unary", "ERF"),
    "floor": ("unary", "FLOOR"),
    "ceil": ("unary", "CEIL"),
    "sign": ("unary", "SIGN"),
    # ---- unary composed from a mode + a fixed alpha (see notes below) ----
    # mudnn has no NEG or TRUNC mode. Both fall out of an alpha op exactly:
    # neg == self * -1, trunc == truncating-divide by 1. Verified against CPU
    # on float and int64 (-3 -1 0 1 2 5 -> 3 1 0 -1 -2 -5; -2.7 -> -2).
    "neg": ("unary_alpha_const", ("MUL", "-1")),
    "trunc": ("unary_alpha_const", ("TRUNCATEDIV", "1")),
    # expm1 == exp(self) - 1, two passes over the output. Matches CPU expm1 to
    # printed precision across [-2.7, 2.7].
    "expm1": ("unary_two_pass", ("EXP", "SUB", "1")),
    # ---- binary: Binary::Mode::<M>, Run(out, self, other) ----
    "mul.Tensor": ("binary", "MUL"),
    "div.Tensor": ("binary", "TRUEDIV"),
    "maximum": ("binary", "MAX"),
    "minimum": ("binary", "MIN"),
    "remainder.Tensor": ("binary", "FLOORMOD"),
    "fmod.Tensor": ("binary", "TRUNCATEMOD"),
    "pow.Tensor_Tensor": ("binary", "POW"),
    # ---- binary_alpha: ADD_ALPHA/SUB_ALPHA carry aten's `alpha` ----
    "add.Tensor": ("binary_alpha", "ADD_ALPHA"),
    "sub.Tensor": ("binary_alpha", "SUB_ALPHA"),
    # ---- binary_cmp: bool out ----
    "eq.Tensor": ("binary_cmp", "EQ"),
    "ne.Tensor": ("binary_cmp", "NE"),
    "lt.Tensor": ("binary_cmp", "LT"),
    "gt.Tensor": ("binary_cmp", "GT"),
    "le.Tensor": ("binary_cmp", "LE"),
    "ge.Tensor": ("binary_cmp", "GE"),
    "logical_and": ("binary_cmp", "LOGICAL_AND"),
    "logical_or": ("binary_cmp", "LOGICAL_OR"),
    # ---- scalar overloads: Unary + SetAlpha, no device tensor needed ----
    # GCU has to stage these scalars into a full-size device tensor because the
    # topsaten scalar path fails in its driver; mudnn takes the scalar directly.
    "mul.Scalar": ("unary_scalar", "MUL"),
    "div.Scalar": ("unary_scalar", "TRUEDIV"),
    "pow.Tensor_Scalar": ("unary_scalar", "POW"),
    "remainder.Scalar": ("unary_scalar", "FLOORMOD"),
    "fmod.Scalar": ("unary_scalar", "TRUNCATEMOD"),
    # add.Scalar/sub.Scalar also take aten's `alpha`, which folds into the
    # scalar (self + other*alpha), so one Unary call still suffices.
    "add.Scalar": ("unary_scalar_alpha", "ADD"),
    "sub.Scalar": ("unary_scalar_alpha", "SUB"),
    # ---- scalar comparisons -> bool out ----
    "eq.Scalar": ("unary_scalar_cmp", "EQ"),
    "ne.Scalar": ("unary_scalar_cmp", "NE"),
    "lt.Scalar": ("unary_scalar_cmp", "LT"),
    "gt.Scalar": ("unary_scalar_cmp", "GT"),
    "le.Scalar": ("unary_scalar_cmp", "LE"),
    "ge.Scalar": ("unary_scalar_cmp", "GE"),
    # ---- matmul ----
    "mm": ("matmul", "MatMul"),
    "bmm": ("matmul", "BatchMatMul"),
    "mm.out": ("matmul_out", "MatMul"),
    "bmm.out": ("matmul_out", "BatchMatMul"),
    # ---- reduce over dims with optional out dtype ----
    "sum.dim_IntList": ("reduce_dims_dtype", "ADD"),
    "mean.dim": ("reduce_dims_dtype", "MEAN"),
    # ---- reduce whole tensor ----
    "sum": ("reduce_all_dtype", "ADD"),
    "mean": ("reduce_all_dtype", "MEAN"),
    # ---- misc ----
    "gelu": ("gelu", "GELU"),
    "_softmax": ("softmax_fwd", "SOFTMAX"),
    # ---- P1: further unary modes (all verified against CPU) ----
    "tan": ("unary", "TAN"),
    # mudnn's ROUND is half-to-even, which is what aten does too: -2.5 -> -2,
    # 0.5 -> 0, 3.5 -> 4. No correction needed.
    "round": ("unary", "ROUND"),
    "mish": ("unary", "MISH"),
    "hardswish": ("unary", "HARDSWISH"),
    # IS_NAN/IS_INF write a BOOL output, so they use the permissive predicate.
    "isnan": ("unary_cmp", "IS_NAN"),
    "isinf": ("unary_cmp", "IS_INF"),
    "hardsigmoid": ("unary_two_const", ("HARDSIGMOID", "1.0 / 6.0", "0.5")),
    "leaky_relu": ("unary_param", ("LEAKY_RELU", "negative_slope")),
    "elu": ("elu", "ELU"),
    "softplus": ("softplus", "SOFTPLUS"),
    "clamp": ("clamp", "CLIP"),
    "clamp_min": (
        "clamp_one_sided",
        ("CLIP", "min", "min.to<double>()", "std::numeric_limits<double>::infinity()"),
    ),
    "clamp_max": (
        "clamp_one_sided",
        ("CLIP", "max", "-std::numeric_limits<double>::infinity()", "max.to<double>()"),
    ),
    "logical_xor": ("binary_cmp", "LOGICAL_XOR"),
    "floor_divide": ("binary", "FLOORDIV"),
    # ---- P1: activation backwards ----
    # SIGMOID_BW/TANH_BW take (grad, output); aten passes `output` in that slot
    # too, so the template is shared. The rest take (grad, input).
    "sigmoid_backward": ("binary_bw", "SIGMOID_BW"),
    "tanh_backward": ("binary_bw", "TANH_BW"),
    "silu_backward": ("binary_bw", "SILU_BW"),
    "gelu_backward": ("gelu_bw", "GELU_NONE_BW"),
    "threshold_backward": ("threshold_bw", "THRESHOLD_BW"),
    "leaky_relu_backward": ("leaky_relu_bw", "LEAKY_RELU_BW"),
    # ---- P1: ternary ----
    "addcmul": ("ternary_value", "ADDCMUL_ALPHA"),
    "addcdiv": ("ternary_value", "ADDCDIV_ALPHA"),
    "where.self": ("where", "SELECT"),
    # ---- P1: addmm family (three-branch, see T_ADDMM) ----
    "addmm": ("addmm", "MatMul"),
    "baddbmm": ("addmm", "BatchMatMul"),
    # ---- P2: softmax family ----
    # The forward template already carries the mode, so log_softmax is a table
    # entry rather than a new shape. Both backwards verified against
    # y*(g - sum(g*y)) on device.
    "_log_softmax": ("softmax_fwd", "LOGSOFTMAX"),
    "_softmax_backward_data": ("softmax_bwd", "SOFTMAX"),
    "_log_softmax_backward_data": ("softmax_bwd", "LOGSOFTMAX"),
    # ---- P2: the rest of the Reduce modes ----
    "amax": ("reduce_dims_plain", "MAX"),
    "amin": ("reduce_dims_plain", "MIN"),
    # PROD, not MUL -- MUL returns all zeros on v3300. See T_REDUCE_PROD.
    "prod.dim_int": ("reduce_prod", "PROD"),
    "any.dim": ("reduce_bool", "OR"),
    "all.dim": ("reduce_bool", "AND"),
    "var.correction": ("reduce_correction", "VARIANCE"),
    "std.correction": ("reduce_correction", "STD"),
    "linalg_vector_norm": ("reduce_norm", "NORM"),
    # ---- P2: layer norm ----
    "native_layer_norm": ("layer_norm", "LayerNorm"),
    "native_layer_norm_backward": ("layer_norm_bwd", "LayerNorm"),
    # ---- P3: concat family ----
    # The tuple is (class, aten list type, needs-unsqueeze). `stack` is
    # cat-of-unsqueezed, which mudnn expresses as the same Concat call. The
    # list types differ in more than the name: cat takes ITensorListRef by
    # const reference, stack takes TensorList by value, and the registered
    # dispatcher's function-pointer type must match exactly.
    "cat": ("concat", ("Concat", "const at::ITensorListRef&", False)),
    "stack": ("concat", ("Concat", "at::TensorList", True)),
    # ---- P3: gather family ----
    # GATHER_ELEMENTS is aten's `gather` (index shaped like the output, selecting
    # along one dim); GATHER is aten's `index_select`/`embedding` (a 1-D index
    # picking whole slices).
    "gather": ("gather", "GATHER_ELEMENTS"),
    "index_select": ("index_select", "GATHER"),
    "embedding": ("embedding", "GATHER"),
    # ---- P3: scatter family ----
    "scatter.src": ("scatter", "UPDATE_ONLY"),
    "scatter.value": ("scatter_value", "UPDATE_ONLY"),
    "scatter_add": ("scatter", "ADD"),
    # ---- P3: fill family ----
    "fill_.Scalar": (
        "fill_scalar",
        (
            "Fill",
            ", const at::Scalar& value",
            "value",
            "value.to<int64_t>()",
            "value.to<double>()",
        ),
    ),
    "zero_": ("fill_scalar", ("Fill", "", "", "0", "0.0")),
    "masked_fill.Scalar": ("masked_fill", "Fill"),
    # ---- P3: sort / topk ----
    "topk": ("topk", "TopK"),
    "sort": ("sort", ("Sort", "", "", "false")),
    "sort.stable": (
        "sort",
        (
            "Sort",
            "\n    ::std::optional<bool> stable,",
            " stable,",
            "stable.has_value() && stable.value()",
        ),
    ),
    # ---- P3: cumulative ----
    # Cum has only ADD and MUL, so cummax/cummin/logcumsumexp stay on fallback.
    "cumsum": ("cum", "ADD"),
    "cumprod": ("cum", "MUL"),
    # ---- P3: reductions that also return indices ----
    "max.dim": ("reduce_with_indices", "MAX"),
    "min.dim": ("reduce_with_indices", "MIN"),
    "argmax": ("reduce_indices", "MAX"),
    "argmin": ("reduce_indices", "MIN"),
}

# Ops in the GCU coverage set with no mudnn equivalent. Deliberately absent from
# OPS rather than registered-and-broken: an unregistered op reaches the
# cpu_fallback and stays correct, so these keep working, just on the host.
#
# mudnn's Unary::Mode has SIN/COS/TAN/ACOS/ATAN but no SINH/COSH/ASIN, and
# composing them from EXP (sinh = (e^x - e^-x)/2) would need several passes with
# worse accuracy than the host, so it is not worth it for these three.
# relu6 is CompositeImplicitAutograd -- it has no wrapper in register.inc and
# decomposes into hardtanh, so mudnn's RELU6 mode is unreachable from here.
NO_MUDNN_EQUIVALENT = ["sinh", "cosh", "asin"]

# Ops handwritten elsewhere for MUSA would double-register the kMusa slot (which
# crashes at import), so they must be excluded here. MudnnCopy is not listed: it
# is called directly by copy_ops.cc/contiguous_ops.cc, not registered as a
# kernel.
SKIP: set = set()

# Handwritten kernels that still need an m.impl() line emitted into
# musa_register.inc. The two `*_overrideable` convolution ops cannot be left to
# the cpu_fallback like other uncovered ops -- ATen's default for them is a
# raising TORCH_CHECK, not something boxable -- so mudnn_conv.cc implements them
# and they are registered here. Kernel bodies live in
# csrc/aten/backends/musa/mudnn_conv.cc, so they are absent from OPS.
HANDWRITTEN_REGISTRATIONS = [
    "convolution_overrideable",
    "convolution_backward_overrideable",
    # Native muRAND/mudnn RNG kernels live in backends/musa/rng.cc.
    "rand",
    "rand.generator",
    "rand.out",
    "rand.names_out",
    "rand_like",
    "rand_like.generator",
    "rand_like.out",
    "randn",
    "randn.generator",
    "randn.names_out",
    "randn_like",
    "randn_like.generator",
    "randn_like.out",
    "randint",
    "randint.generator",
    "randint.low",
    "randint.low_generator",
    "randint.out",
    "randint.low_out",
    "normal_",
    "uniform_",
    "random_",
    "random_.from",
    "random_.to",
    "native_dropout",
    "native_dropout_backward",
]

# The CPU-fallback path in each kernel calls back into at::<name>. That is the
# op base name except where the base is not a real at:: function.
AT_OP_OVERRIDES = {
    "mm.out": "mm",
    "bmm.out": "bmm",
    "_softmax": "_softmax",
}

# ==========================================================================
# Templates
#
# `musa_ops` is the helper namespace from ../mudnn_common.h. EXEC_MUDNN_CMD
# takes the whole Run() expression (not op + args) because mudnn's entry points
# vary: Run, Run-with-workspace, RunWithIndices. `_mudnn_h` is the cached
# per-device Handle the macro binds.
# ==========================================================================

T_UNARY = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# A unary aten op that mudnn only expresses as a mode plus a constant alpha
# (neg -> MUL by -1, trunc -> TRUNCATEDIV by 1). SetAlpha is overloaded on
# double vs int64_t and the op reads the member back as the tensor's dtype, so
# an integral tensor must set the integral overload.
T_UNARY_ALPHA_CONST = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  if (at::isIntegralType(self.scalar_type(), true)) {{
    op.SetAlpha(static_cast<int64_t>({alpha}));
  }} else {{
    op.SetAlpha(static_cast<double>({alpha}));
  }}
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# Two mudnn passes for one aten op (expm1 = EXP then SUB alpha=1). The second
# pass reads and writes `out`, which mudnn allows for elementwise unary.
T_UNARY_TWO_PASS = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary first;
  first.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self, first.Run(_mudnn_h, t_out.get(), t_self.get()));
  musa_ops::MudnnTensorWrapper t_out2(out);
  musa_ops::MudnnTensorWrapper t_in2(out);
  musa_ops::mudnn::Unary second;
  second.SetMode(musa_ops::mudnn::Unary::Mode::{mode2});
  if (at::isIntegralType(self.scalar_type(), true)) {{
    second.SetAlpha(static_cast<int64_t>({alpha}));
  }} else {{
    second.SetAlpha(static_cast<double>({alpha}));
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", self, second.Run(_mudnn_h, t_out2.get(), t_in2.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# Output dtype follows at::result_type, matching PyTorch's promotion.
#
# `self` is normally the device tensor, but that assumption breaks for
# rsub.Scalar: its sub.Tensor decomposition is
# sub(wrapped_scalar_tensor(other), self, alpha), which puts the CPU wrapped
# scalar in the `self` slot and the real device tensor in `other`. Picking the
# compute device from whichever operand is not CPU (instead of hard-coding
# self.device()) handles both orderings, and .to() is a no-op when a tensor
# already has the target device/dtype, so the ordinary self-is-device path
# pays no extra copy (issue #238).
#
# `other` may be a CPU tensor: PyTorch wraps a Python number operand into a
# 0-dim CPU tensor and dispatches through the Tensor overload (a * 3.0 ->
# mul.Tensor). Handing that host pointer to mudnn would fault, so any non-device
# operand is moved onto the compute device first.
#
# expand() to the broadcast shape stays a view (it only introduces 0-strides),
# and mudnn reads 0-strides correctly -- verified: (2,3) + (3,) via strides
# {0,1} gives 11 22 33 14 25 36. So unlike the GCU templates there is no
# `.contiguous()` here and broadcasting costs no allocation.
_BINARY_PROLOGUE = """\
  auto compute_device = self.is_cpu() ? other.device() : self.device();
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.to(compute_device, result_dtype);
  auto other_c = other.to(compute_device, result_dtype);
  auto out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  auto self_b = self_c.expand(out_shape);
  auto other_b = other_c.expand(out_shape);
"""

T_BINARY = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type())) {{
    auto fallback_device = self.is_cpu() ? other.device() : self.device();
    return at::{at_op}(self.cpu(), other.cpu()).to(fallback_device);
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self_c.options());
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self_c,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

T_BINARY_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type())) {{
    auto fallback_device = self.is_cpu() ? other.device() : self.device();
    return at::{at_op}(self.cpu(), other.cpu(), alpha).to(fallback_device);
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self_c.options());
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(alpha.to<int64_t>());
  }} else {{
    op.SetAlpha(alpha.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", self_c,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

T_BINARY_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type())) {{
    auto fallback_device = self.is_cpu() ? other.device() : self.device();
    return at::{at_op}(self.cpu(), other.cpu()).to(fallback_device);
  }}
"""
    + _BINARY_PROLOGUE
    + """\
  auto out = at::empty(out_shape, self_c.options().dtype(at::kBool));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self_c,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# A Scalar participates in promotion only via its category (integral scalars do
# not widen a float tensor), which is exactly at::result_type(Tensor, Scalar).
#
# The scalar goes straight into Unary::SetAlpha -- one kernel, no staging
# tensor. Operand order is `self OP alpha` for every mode used here (SUB, DIV,
# POW, FLOORMOD, TRUNCATEMOD all confirmed on device; the reverse spellings are
# the separate *_BY_ALPHA modes, which we do not use).
_SCALAR_PROLOGUE = """\
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
"""

# Braces stay doubled: this fragment is concatenated into a template that is
# .format()ed once, at generation time.
_SCALAR_SET_ALPHA = """\
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(other.to<int64_t>());
  }} else {{
    op.SetAlpha(other.to<double>());
  }}
"""

T_UNARY_SCALAR = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
"""
    + _SCALAR_SET_ALPHA
    + """\
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# add.Scalar/sub.Scalar: aten's alpha scales the scalar operand, so
# self + other*alpha folds into a single ADD/SUB alpha. Folded on the host in
# the scalar's own type to avoid a needless int->double round trip.
T_UNARY_SCALAR_ALPHA = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other, alpha).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(other.to<int64_t>() * alpha.to<int64_t>());
  }} else {{
    op.SetAlpha(other.to<double>() * alpha.to<double>());
  }}
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# Tensor-vs-Scalar comparison. The comparison itself happens in the promoted
# type, but the result is always bool -- confirmed on device (GT alpha=3 over
# 1..6 gives 0 0 0 1 1 1 into a BOOL tensor).
T_UNARY_SCALAR_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(at::result_type(self, other))) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
"""
    + _SCALAR_PROLOGUE
    + """\
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
"""
    + _SCALAR_SET_ALPHA
    + """\
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""
)

# mm/bmm. mudnn's MatMul and BatchMatMul both want a workspace maintainer (the
# query returned 0 bytes for the shapes probed, but the argument is mandatory).
# Operands are made contiguous here: unlike the elementwise ops, a GEMM's inner
# layout requirements are not something the stride descriptor alone guarantees,
# so this keeps mm(a.t(), b) correct rather than fast.
T_MATMUL = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  auto out = at::empty(out_shape, self.options());
{empty_guard}  auto self_c = self.contiguous();
  auto mat2_c = mat2.contiguous();

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_mat2(mat2_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::{mode} op;
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_mat2.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

T_MATMUL_OUT = """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat2.scalar_type())) {{
    out.copy_(at::{at_op}(self.cpu(), mat2.cpu()));
    return out;
  }}
  std::vector<int64_t> out_shape = self.sizes().vec();
  out_shape.back() = mat2.size(-1);
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}
  auto self_c = self.contiguous();
  auto mat2_c = mat2.contiguous();

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_mat2(mat2_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::{mode} op;
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_mat2.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# sum/mean over an optional dim list. An absent or empty list reduces every
# dim; negative dims are wrapped. Dims are erased high-to-low so an earlier
# erase does not shift a later index. sum promotes integral inputs to int64,
# matching PyTorch -- and unlike topsaten, mudnn *can* reduce in int64
# (verified: ADD over int64 {1..6} by dim gives 6, 15), so integral sums stay
# on device instead of taking a CPU fallback.
T_REDUCE_DIMS_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      (dtype.has_value() && !musa_ops::{dtype_pred}(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  int64_t ndim = self.dim();
  std::vector<int64_t> norm_dims;
  if (dim.has_value() && !dim.value().empty()) {{
    for (int64_t d : dim.value()) norm_dims.push_back(d < 0 ? d + ndim : d);
  }} else {{
    for (int64_t d = 0; d < ndim; ++d) norm_dims.push_back(d);
  }}
  // aten's answer shape, and the *squeezed* shape mudnn must be handed. mudnn
  // silently drops all but the first output element when the output still
  // carries the reduced axes as extent-1 dims and the input is non-contiguous
  // (measured on v3300: (4,5) stride (0,1) reduced over dim 0 into a [1,5]
  // output writes only out[0]). Reducing into the squeezed shape is correct in
  // every configuration probed, so keepdim is restored with a view afterwards.
  auto out_shape = self.sizes().vec();
  std::vector<int64_t> sorted_dims(norm_dims);
  std::sort(sorted_dims.rbegin(), sorted_dims.rend());
  auto squeezed_shape = self.sizes().vec();
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
    squeezed_shape.erase(squeezed_shape.begin() + d);
  }}

  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  // mudnn multi-dim Reduce (more than one axis at a time) silently ignores
  // strides and reads the input as if contiguous (verified: (4,5) stride (0,1)
  // reduced over all dims gives 210 = sum(1..20) instead of 60 = 4*sum(1..5)).
  // Single-dim reduces honour strides correctly, except on a fully-broadcast
  // input (a 0-stride view of one element), where a single-dim reduce
  // intermittently writes only out[0]. MudnnReduceNeedsContiguous catches
  // that; together the two branches cover the stride bug, the SIGFPE and the
  // partial write.
  if (norm_dims.size() > 1 && !self_c.is_contiguous()) {{
    self_c = self_c.contiguous();
  }} else if (musa_ops::MudnnReduceNeedsContiguous(self_c)) {{
    self_c = self_c.contiguous();
  }}
  auto out = at::empty(squeezed_shape, self.options().dtype(out_dtype));
{empty_guard}  std::vector<int> mudnn_dims = musa_ops::ToMudnnDims(norm_dims, ndim);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(static_cast<int>(mudnn_dims.size()), mudnn_dims.data());
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return keepdim ? out.view(out_shape) : out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

T_REDUCE_ALL_DTYPE = """\
at::Tensor {kernel}(
    const at::Tensor& self, ::std::optional<at::ScalarType> dtype) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      (dtype.has_value() && !musa_ops::{dtype_pred}(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dtype).to(self.device());
  }}
  auto out_dtype = dtype.value_or(
      {promote_integral} && at::isIntegralType(self.scalar_type(), true)
          ? at::kLong
          : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  std::vector<int> mudnn_dims;
  for (int64_t d = 0; d < self.dim(); ++d) {{
    mudnn_dims.push_back(static_cast<int>(d));
  }}
  // Reducing every dim at once is a multi-dim Reduce, which mudnn runs as if
  // the input were contiguous -- it ignores strides outright, and faults on a
  // fully-broadcast input, which also breaks the single-dim path. Materializing
  // once covers all of it. Same measurement as in T_REDUCE_DIMS_DTYPE.
  if (mudnn_dims.size() > 1 && !self_c.is_contiguous()) {{
    self_c = self_c.contiguous();
  }} else if (musa_ops::MudnnReduceNeedsContiguous(self_c)) {{
    self_c = self_c.contiguous();
  }}
  auto out = at::empty({{}}, self.options().dtype(out_dtype));
{empty_guard}
  // Reducing an empty input into a scalar: mudnn rejects the zero-element
  // operand, but aten's answer is the reduction's identity ({identity_note}).
  // Fill it on-device rather than launching or detouring through the host.
  if (self_c.numel() == 0) {{
    out.fill_({identity});
    return out;
  }}

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(static_cast<int>(mudnn_dims.size()), mudnn_dims.data());
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# aten spells the variant as a string; mudnn has separate modes for it.
T_GELU = """\
at::Tensor {kernel}(const at::Tensor& self, c10::string_view approximate) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), approximate).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(
      approximate == "tanh" ? musa_ops::mudnn::Unary::Mode::GELU_TANH
                            : musa_ops::mudnn::Unary::Mode::GELU);
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# `half_to_float` asks for a float output from a half input, which mudnn's
# Softmax does not express (out dtype must match in), so that combination runs
# on the host. ACCURATE is the max-subtracting algorithm, matching aten.
T_SOFTMAX_FWD = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool half_to_float) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) || half_to_float) {{
    return at::{at_op}(self.cpu(), dim, half_to_float).to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Softmax op;
  op.SetMode(musa_ops::mudnn::Softmax::Mode::{mode});
  op.SetAlgorithm(musa_ops::mudnn::Softmax::Algorithm::ACCURATE);
  op.SetDim(static_cast<int>(d));
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. Softmax/LogSoftmax backward. Probed on v3300: RunBwd's operands are
# (gradInput, output, gradOutput) and the numbers match aten's
# y*(g - sum(g*y)) exactly. `input_dtype` only tells us what the forward input
# was; when it differs from the gradient's dtype aten wants a converting
# backward, which mudnn does not express, so that combination goes to the host.
T_SOFTMAX_BWD = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(output.scalar_type()) ||
      input_dtype != grad_output.scalar_type()) {{
    return at::{at_op}(grad_output.cpu(), output.cpu(), dim, input_dtype)
        .to(grad_output.device());
  }}
  int64_t d = dim < 0 ? dim + output.dim() : dim;
  auto grad_input = at::empty(output.sizes(), output.options());

  musa_ops::MudnnTensorWrapper t_go(grad_output);
  musa_ops::MudnnTensorWrapper t_out(output);
  musa_ops::MudnnTensorWrapper t_gi(grad_input);
  musa_ops::mudnn::Softmax op;
  op.SetMode(musa_ops::mudnn::Softmax::Mode::{mode});
  op.SetAlgorithm(musa_ops::mudnn::Softmax::Algorithm::ACCURATE);
  op.SetDim(static_cast<int>(d));
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.RunBwd(_mudnn_h, t_gi.get(), t_out.get(), t_go.get(),
                musa_ops::MudnnWorkspaceFor(grad_input)));
  return grad_input;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. Reduce over a single dim with no dtype argument (amax/amin), and the
# bool-producing any/all. Kept apart from T_REDUCE_DIMS_DTYPE because the aten
# signatures differ; the Reduce call itself is the same shape. Single-dim
# reduces honour strides on v3300, so no materialization is needed -- but
# `any`/`all` over several dims would, hence the shared contiguous guard.
T_REDUCE_DIMS_PLAIN = """\
at::Tensor {kernel}(
    const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
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
  auto squeezed_shape = self.sizes().vec();
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
    squeezed_shape.erase(squeezed_shape.begin() + d);
  }}
  auto self_c = self;
  if (norm_dims.size() > 1 && !self_c.is_contiguous()) {{
    self_c = self_c.contiguous();
  }} else if (musa_ops::MudnnReduceNeedsContiguous(self_c)) {{
    self_c = self_c.contiguous();
  }}
  auto out = at::empty(squeezed_shape, self.options());
{empty_guard}  std::vector<int> mudnn_dims = musa_ops::ToMudnnDims(norm_dims, ndim);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(static_cast<int>(mudnn_dims.size()), mudnn_dims.data());
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return keepdim ? out.view(out_shape) : out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. any.dim/all.dim: one int64 dim, bool output regardless of input dtype.
# mudnn's AND/OR accept a BOOL input and write BOOL, so the input is cast to
# bool first (aten defines any/all as "nonzero", which `!= 0` expresses).
T_REDUCE_BOOL = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, bool keepdim) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, keepdim).to(self.device());
  }}
  int64_t ndim = self.dim();
  int64_t d = dim < 0 ? dim + ndim : dim;
  auto out_shape = self.sizes().vec();
  auto squeezed_shape = self.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);
  squeezed_shape.erase(squeezed_shape.begin() + d);

  auto self_b = self.scalar_type() == at::kBool ? self : self.ne(0);
  auto out = at::empty(squeezed_shape, self.options().dtype(at::kBool));
{empty_guard}  int mudnn_dim = static_cast<int>(d);

  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(1, &mudnn_dim);
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return keepdim ? out.view(out_shape) : out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. prod.dim_int -- single int64 dim plus an out dtype.
#
# NOTE the mode is PROD, not MUL: `Reduce::Mode::MUL` returns SUCCESS and writes
# all zeros on v3300 (measured on [[1,2,3,4],[5,6,7,8]] reduced over dim 1: MUL
# gave 0 0, PROD gave the correct 24 1680). The plausibly-named mode is the
# broken one.
T_REDUCE_PROD = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      (dtype.has_value() && !musa_ops::{dtype_pred}(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dim, keepdim, dtype).to(self.device());
  }}
  int64_t ndim = self.dim();
  int64_t d = dim < 0 ? dim + ndim : dim;
  auto out_shape = self.sizes().vec();
  auto squeezed_shape = self.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);
  squeezed_shape.erase(squeezed_shape.begin() + d);

  auto out_dtype = dtype.value_or(
      at::isIntegralType(self.scalar_type(), true) ? at::kLong
                                                   : self.scalar_type());
  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = at::empty(squeezed_shape, self.options().dtype(out_dtype));
{empty_guard}  int mudnn_dim = static_cast<int>(d);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(1, &mudnn_dim);
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return keepdim ? out.view(out_shape) : out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. var/std with aten's `correction`. mudnn's VARIANCE/STD default to
# correction=1 (measured: var of 1..4 came back 1.66667 = 5/3, the unbiased
# value, not the biased 1.25), and SetCorrection takes an int, so aten's
# optional Scalar maps straight across with 1 as the default.
T_REDUCE_CORRECTION = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const ::std::optional<at::Scalar>& correction,
    bool keepdim) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !at::isFloatingType(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, correction, keepdim).to(self.device());
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
  auto squeezed_shape = self.sizes().vec();
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
    squeezed_shape.erase(squeezed_shape.begin() + d);
  }}
  auto self_c = self;
  if (norm_dims.size() > 1 && !self_c.is_contiguous()) {{
    self_c = self_c.contiguous();
  }} else if (musa_ops::MudnnReduceNeedsContiguous(self_c)) {{
    self_c = self_c.contiguous();
  }}
  auto out = at::empty(squeezed_shape, self.options());
{empty_guard}  std::vector<int> mudnn_dims = musa_ops::ToMudnnDims(norm_dims, ndim);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(static_cast<int>(mudnn_dims.size()), mudnn_dims.data());
  op.SetCorrection(
      correction.has_value() ? static_cast<int>(correction.value().to<double>())
                             : 1);
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return keepdim ? out.view(out_shape) : out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. linalg_vector_norm. Reduce::NORM with SetNormOrd; verified ord=2 over
# 1..4 gives 5.47723. ord 0 and +-inf are separate reductions in aten (count of
# nonzeros, max/min |x|) that NORM does not express, so they stay on the host.
T_REDUCE_NORM = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Scalar& ord,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {{
  double ord_d = ord.to<double>();
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !at::isFloatingType(self.scalar_type()) ||
      (dtype.has_value() && dtype.value() != self.scalar_type()) ||
      ord_d == 0.0 || !std::isfinite(ord_d)) {{
    return at::{at_op}(self.cpu(), ord, dim, keepdim, dtype).to(self.device());
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
  auto squeezed_shape = self.sizes().vec();
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
    squeezed_shape.erase(squeezed_shape.begin() + d);
  }}
  auto self_c = self;
  if (norm_dims.size() > 1 && !self_c.is_contiguous()) {{
    self_c = self_c.contiguous();
  }} else if (musa_ops::MudnnReduceNeedsContiguous(self_c)) {{
    self_c = self_c.contiguous();
  }}
  auto out = at::empty(squeezed_shape, self.options());
{empty_guard}  std::vector<int> mudnn_dims = musa_ops::ToMudnnDims(norm_dims, ndim);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(static_cast<int>(mudnn_dims.size()), mudnn_dims.data());
  op.SetNormOrd(static_cast<float>(ord_d));
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return keepdim ? out.view(out_shape) : out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. native_layer_norm. mudnn's LayerNorm::Run emits (out, mean, inv_var) in
# one shot, and the third output is exactly aten's rstd: measured
# 1/sqrt(var+eps) = 0.89442 for var=1.25, eps=1e-5, so no conversion is needed.
# SetAxis takes the trailing axes that normalized_shape names, matching aten.
#
# gamma/beta are optional in aten but not in mudnn, so a missing one becomes
# ones/zeros. aten also requires mean/rstd be float32 even for a half input,
# which mudnn will not do, so half runs on the host.
T_LAYER_NORM = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    double eps) {{
  if (!musa_ops::{dtype_pred}(input.scalar_type()) ||
      input.scalar_type() != at::kFloat || normalized_shape.empty()) {{
    auto r = at::{at_op}(
        input.cpu(), normalized_shape,
        weight.has_value() ? ::std::optional<at::Tensor>(weight.value().cpu())
                           : ::std::nullopt,
        bias.has_value() ? ::std::optional<at::Tensor>(bias.value().cpu())
                         : ::std::nullopt,
        eps);
    return ::std::make_tuple(::std::get<0>(r).to(input.device()),
                             ::std::get<1>(r).to(input.device()),
                             ::std::get<2>(r).to(input.device()));
  }}
  auto input_c = input.contiguous();
  const int64_t ndim = input_c.dim();
  const int64_t naxes = static_cast<int64_t>(normalized_shape.size());
  // aten's stat shape keeps the leading dims and 1s in the normalized ones.
  std::vector<int64_t> stat_shape;
  for (int64_t d = 0; d < ndim - naxes; ++d) {{
    stat_shape.push_back(input_c.size(d));
  }}
  auto out = at::empty(input_c.sizes(), input_c.options());
  auto mean = at::empty(stat_shape, input_c.options());
  auto rstd = at::empty(stat_shape, input_c.options());

  auto w = weight.has_value() && weight.value().defined()
      ? weight.value().contiguous()
      : at::ones(normalized_shape, input_c.options());
  auto b = bias.has_value() && bias.value().defined()
      ? bias.value().contiguous()
      : at::zeros(normalized_shape, input_c.options());

  std::vector<int> axes;
  for (int64_t d = ndim - naxes; d < ndim; ++d) axes.push_back(static_cast<int>(d));

  musa_ops::MudnnTensorWrapper t_in(input_c);
  musa_ops::MudnnTensorWrapper t_w(w);
  musa_ops::MudnnTensorWrapper t_b(b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::MudnnTensorWrapper t_mean(mean);
  musa_ops::MudnnTensorWrapper t_rstd(rstd);
  musa_ops::mudnn::LayerNorm op;
  op.SetEpsilon(eps);
  op.SetAxis(axes.size(), axes.data());
  EXEC_MUDNN_CMD(
      "{at_op}", input,
      op.Run(_mudnn_h, t_out.get(), t_mean.get(), t_rstd.get(), t_in.get(),
             t_w.get(), t_b.get(), musa_ops::MudnnWorkspaceFor(out)));
  // aten reports the stats with the normalized axes kept as extent-1 dims.
  std::vector<int64_t> aten_stat_shape(stat_shape);
  for (int64_t i = 0; i < naxes; ++i) aten_stat_shape.push_back(1);
  return ::std::make_tuple(out, mean.view(aten_stat_shape),
                           rstd.view(aten_stat_shape));
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# P2. native_layer_norm_backward. RunBwd emits (dX, dGamma, dBeta) from
# (dY, in, mean, invVar, gamma) -- verified against CPU: with dY=1 and gamma=2,
# dGamma came back as 2*xhat summed over rows and dBeta as the row count, both
# matching, and dX was 0 as the algebra requires.
#
# aten's output_mask can switch any of the three off; mudnn always writes all
# three, so the masked-out ones are computed and dropped (an undefined Tensor is
# what aten expects back). That costs nothing extra beyond the allocation.
T_LAYER_NORM_BWD = """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    ::std::array<bool, 3> output_mask) {{
  if (!musa_ops::{dtype_pred}(input.scalar_type()) ||
      input.scalar_type() != at::kFloat || normalized_shape.empty()) {{
    auto r = at::{at_op}(
        grad_out.cpu(), input.cpu(), normalized_shape, mean.cpu(), rstd.cpu(),
        weight.has_value() ? ::std::optional<at::Tensor>(weight.value().cpu())
                           : ::std::nullopt,
        bias.has_value() ? ::std::optional<at::Tensor>(bias.value().cpu())
                         : ::std::nullopt,
        output_mask);
    auto to_dev = [&](const at::Tensor& t) {{
      return t.defined() ? t.to(input.device()) : t;
    }};
    return ::std::make_tuple(to_dev(::std::get<0>(r)), to_dev(::std::get<1>(r)),
                             to_dev(::std::get<2>(r)));
  }}
  auto grad_c = grad_out.contiguous();
  auto input_c = input.contiguous();
  auto mean_c = mean.contiguous();
  auto rstd_c = rstd.contiguous();
  const int64_t ndim = input_c.dim();
  const int64_t naxes = static_cast<int64_t>(normalized_shape.size());

  auto w = weight.has_value() && weight.value().defined()
      ? weight.value().contiguous()
      : at::ones(normalized_shape, input_c.options());
  auto d_input = at::empty(input_c.sizes(), input_c.options());
  auto d_weight = at::empty(normalized_shape, input_c.options());
  auto d_bias = at::empty(normalized_shape, input_c.options());

  std::vector<int> axes;
  for (int64_t d = ndim - naxes; d < ndim; ++d) axes.push_back(static_cast<int>(d));

  musa_ops::MudnnTensorWrapper t_dy(grad_c);
  musa_ops::MudnnTensorWrapper t_in(input_c);
  musa_ops::MudnnTensorWrapper t_mean(mean_c);
  musa_ops::MudnnTensorWrapper t_rstd(rstd_c);
  musa_ops::MudnnTensorWrapper t_w(w);
  musa_ops::MudnnTensorWrapper t_dx(d_input);
  musa_ops::MudnnTensorWrapper t_dw(d_weight);
  musa_ops::MudnnTensorWrapper t_db(d_bias);
  musa_ops::mudnn::LayerNorm op;
  op.SetAxis(axes.size(), axes.data());
  EXEC_MUDNN_CMD(
      "{at_op}", input,
      op.RunBwd(_mudnn_h, t_dx.get(), t_dw.get(), t_db.get(), t_dy.get(),
                t_in.get(), t_mean.get(), t_rstd.get(), t_w.get(),
                musa_ops::MudnnWorkspaceFor(d_input)));
  return ::std::make_tuple(output_mask[0] ? d_input : at::Tensor(),
                           output_mask[1] ? d_weight : at::Tensor(),
                           output_mask[2] ? d_bias : at::Tensor());
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

#
# Which saved tensor is the second operand differs per mode and is NOT visible
# in the header -- both orders return SUCCESS, only the numbers differ. Measured
# against CPU formulas: SIGMOID_BW/TANH_BW consume the op's *output*
# (g*y*(1-y), g*(1-y^2)), every other mode here consumes the *input*. aten
# passes exactly that tensor in the same position either way, so `self` maps
# straight through; the distinction is only documented, never branched on.
T_BINARY_BW = """\
at::Tensor {kernel}(const at::Tensor& grad_output, const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(grad_output.cpu(), self.cpu()).to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# gelu_backward: aten spells the variant as a string, mudnn as two modes.
T_GELU_BW = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(grad_output.cpu(), self.cpu(), approximate)
        .to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(
      approximate == "tanh" ? musa_ops::mudnn::Binary::Mode::GELU_TANH_BW
                            : musa_ops::mudnn::Binary::Mode::GELU_NONE_BW);
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# threshold_backward(grad_output, self, threshold). mudnn's THRESHOLD_BW takes
# the threshold as alpha; measured to match aten (grad passes where self >
# threshold, 0 elsewhere).
T_THRESHOLD_BW = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(grad_output.cpu(), self.cpu(), threshold)
        .to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(threshold.to<int64_t>());
  }} else {{
    op.SetAlpha(threshold.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# leaky_relu_backward(grad_output, self, negative_slope, self_is_result).
# LEAKY_RELU_BW's alpha is the slope. `self_is_result` only tells autograd
# whether `self` is the output rather than the input; for a leaky ReLU the
# gradient test (`> 0`) gives the same answer either way, so it is unused.
T_LEAKY_RELU_BW = """\
at::Tensor {kernel}(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result) {{
  if (!musa_ops::{dtype_pred}(grad_output.scalar_type()) ||
      !musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(
               grad_output.cpu(), self.cpu(), negative_slope, self_is_result)
        .to(grad_output.device());
  }}
  auto result_dtype = at::result_type(grad_output, self);
  auto grad_c = grad_output.scalar_type() == result_dtype
      ? grad_output
      : grad_output.to(result_dtype);
  auto self_c = self.to(grad_output.device(), result_dtype);
  auto out_shape = at::infer_size(grad_c.sizes(), self_c.sizes());
  auto grad_b = grad_c.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto out = at::empty(out_shape, grad_output.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_grad(grad_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Binary op;
  op.SetMode(musa_ops::mudnn::Binary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(negative_slope.to<int64_t>());
  }} else {{
    op.SetAlpha(negative_slope.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", grad_output,
      op.Run(_mudnn_h, t_out.get(), t_grad.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# Ternary ops. Measured to map onto aten 1:1 with Run(out, self, t1, t2):
# ADDCMUL_ALPHA is self + value*t1*t2, ADDCDIV_ALPHA is self + value*t1/t2.
# All three operands broadcast against each other, and mudnn honours the
# resulting 0-strides, so expand() alone is enough.
T_TERNARY_VALUE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(tensor1.scalar_type()) ||
      !musa_ops::{dtype_pred}(tensor2.scalar_type())) {{
    return at::{at_op}(self.cpu(), tensor1.cpu(), tensor2.cpu(), value)
        .to(self.device());
  }}
  auto result_dtype = at::result_type(self, tensor1);
  result_dtype = at::promoteTypes(result_dtype, tensor2.scalar_type());
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto t1_c = tensor1.to(self.device(), result_dtype);
  auto t2_c = tensor2.to(self.device(), result_dtype);
  auto out_shape = at::infer_size(self_c.sizes(), t1_c.sizes());
  out_shape = at::infer_size(out_shape, t2_c.sizes());
  auto self_b = self_c.expand(out_shape);
  auto t1_b = t1_c.expand(out_shape);
  auto t2_b = t2_c.expand(out_shape);
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_t1(t1_b);
  musa_ops::MudnnTensorWrapper t_t2(t2_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Ternary op;
  op.SetMode(musa_ops::mudnn::Ternary::Mode::{mode});
  if (at::isIntegralType(result_dtype, true)) {{
    op.SetAlpha(value.to<int64_t>());
  }} else {{
    op.SetAlpha(value.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(), t_t1.get(), t_t2.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# where.self(condition, self, other) -> Ternary::SELECT(mask, x, y). The
# condition is bool and stays bool; only the value operands promote.
T_WHERE = """\
at::Tensor {kernel}(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(other.scalar_type()) ||
      condition.scalar_type() != at::kBool) {{
    return at::{at_op}(condition.cpu(), self.cpu(), other.cpu())
        .to(self.device());
  }}
  auto result_dtype = at::result_type(self, other);
  auto self_c = self.to(condition.device(), result_dtype);
  auto other_c = other.to(condition.device(), result_dtype);
  auto out_shape = at::infer_size(condition.sizes(), self_c.sizes());
  out_shape = at::infer_size(out_shape, other_c.sizes());
  auto cond_b = condition.expand(out_shape);
  auto self_b = self_c.expand(out_shape);
  auto other_b = other_c.expand(out_shape);
  auto out = at::empty(out_shape, self.options().dtype(result_dtype));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_cond(cond_b);
  musa_ops::MudnnTensorWrapper t_self(self_b);
  musa_ops::MudnnTensorWrapper t_other(other_b);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Ternary op;
  op.SetMode(musa_ops::mudnn::Ternary::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_cond.get(), t_self.get(),
             t_other.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# addmm/baddbmm: out = beta*self + alpha*(mat1 @ mat2).
#
# mudnn computes d = alpha*a@b + beta*c + gamma*bias, and which slot `self`
# takes depends on its shape -- this mirrors torch_musa's own three-branch
# dispatch (csrc/aten/ops/Matmul.cpp), verified numerically here:
#   * self shaped like out -> the C slot, aten's beta -> SetBeta
#   * self is 1-D of length N -> the bias slot with c aliasing d, so aten's
#     beta must ride on *gamma* instead (SetBeta stays 0, or it would fold in
#     the output buffer's prior contents)
#   * anything else (a scalar, [M,1]) -> plain Run, then add the bias on the
#     host side with a normal aten add.
#
# MatMul rejects non-contiguous operands ("MatMulRun only support contiguous
# tensor"), unlike the elementwise ops, so mat1/mat2/self are materialized.
T_ADDMM = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat1.scalar_type()) ||
      !musa_ops::{dtype_pred}(mat2.scalar_type())) {{
    return at::{at_op}(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
        .to(self.device());
  }}
  std::vector<int64_t> out_shape = mat1.sizes().vec();
  out_shape.back() = mat2.size(-1);
  auto out = at::empty(out_shape, mat1.options());
{empty_guard}  auto mat1_c = mat1.contiguous();
  auto mat2_c = mat2.contiguous();

  musa_ops::MudnnTensorWrapper t_mat1(mat1_c);
  musa_ops::MudnnTensorWrapper t_mat2(mat2_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::{mode} op;

  const bool self_is_out_shaped = self.sizes().equals(out_shape);
  const bool self_is_vector =
      self.dim() == 1 && self.size(0) == out_shape.back();

  if (self_is_out_shaped) {{
    auto self_c = self.to(mat1.device(), mat1.scalar_type()).contiguous();
    musa_ops::MudnnTensorWrapper t_self(self_c);
    musa_ops::mudnn::Tensor empty_bias;
    op.SetAlpha(alpha.to<double>());
    op.SetBeta(beta.to<double>());
    op.SetGamma(1.0);
    EXEC_MUDNN_CMD(
        "{at_op}", self,
        op.RunWithBiasAdd(_mudnn_h, t_out.get(), t_mat1.get(), t_mat2.get(),
                          t_self.get(), empty_bias,
                          musa_ops::MudnnWorkspaceFor(out)));
  }} else if (self_is_vector) {{
    auto self_c = self.to(mat1.device(), mat1.scalar_type()).contiguous();
    musa_ops::MudnnTensorWrapper t_self(self_c);
    // c aliases d here, so beta must stay 0 and aten's beta rides on gamma.
    op.SetAlpha(alpha.to<double>());
    op.SetBeta(0.0);
    op.SetGamma(beta.to<double>());
    EXEC_MUDNN_CMD(
        "{at_op}", self,
        op.RunWithBiasAdd(_mudnn_h, t_out.get(), t_mat1.get(), t_mat2.get(),
                          t_out.get(), t_self.get(),
                          musa_ops::MudnnWorkspaceFor(out)));
  }} else {{
    op.SetAlpha(alpha.to<double>());
    op.SetBeta(0.0);
    op.SetGamma(1.0);
    EXEC_MUDNN_CMD(
        "{at_op}", self,
        op.Run(_mudnn_h, t_out.get(), t_mat1.get(), t_mat2.get(),
               musa_ops::MudnnWorkspaceFor(out)));
    out.add_(self, beta);
  }}
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# A unary predicate: same call shape as T_UNARY but the output is bool
# (isnan, isinf). Verified that mudnn accepts a BOOL destination for these.
T_UNARY_CMP = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# A unary mode configured by two fixed constants rather than by aten arguments.
# mudnn's HARDSIGMOID is clamp(alpha*x + beta, 0, 1) with both defaulting to 0,
# so leaving them unset returns all zeros. aten's hardsigmoid is alpha=1/6,
# beta=0.5 (verified against CPU).
T_UNARY_TWO_CONST = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(static_cast<double>({alpha}));
  op.SetBeta(static_cast<double>({beta}));
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# leaky_relu(self, negative_slope): one aten Scalar straight into alpha.
T_UNARY_PARAM = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& {param}) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), {param}).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha({param}.to<double>());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# clamp_min / clamp_max map onto CLIP, whose alpha is the lower bound and beta
# the upper. Both default to 0 in mudnn, so the unused side must be set to an
# explicit infinity or the op would clip against 0.
T_CLAMP_ONE_SIDED = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& {param}) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), {param}).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha({lo});
  op.SetBeta({hi});
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# clamp(self, min?, max?) -- both bounds optional; an absent bound becomes the
# corresponding infinity.
T_CLAMP = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), min, max).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(
      min.has_value() ? min.value().to<double>()
                      : -std::numeric_limits<double>::infinity());
  op.SetBeta(
      max.has_value() ? max.value().to<double>()
                      : std::numeric_limits<double>::infinity());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# softplus(self, beta, threshold). mudnn's setter names are inverted relative
# to aten: SetAlpha carries aten's `beta`, SetBeta carries aten's `threshold`.
# Leaving either unset returns inf.
T_SOFTPLUS = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), beta, threshold).to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(beta.to<double>());
  op.SetBeta(threshold.to<double>());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# elu(self, alpha, scale, input_scale) computes
# scale * (max(0,x) + min(0, alpha*(exp(input_scale*x)-1))). mudnn's ELU takes
# only alpha, so a non-unit scale or input_scale has no expression here and
# takes the CPU fallback rather than silently ignoring the argument.
T_ELU = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      scale.to<double>() != 1.0 || input_scale.to<double>() != 1.0) {{
    return at::{at_op}(self.cpu(), alpha, scale, input_scale)
        .to(self.device());
  }}
  auto out = at::empty(self.sizes(), self.options());
{empty_guard}  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Unary op;
  op.SetMode(musa_ops::mudnn::Unary::Mode::{mode});
  op.SetAlpha(alpha.to<double>());
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# ==========================================================================
# P3: shape / index / sort / scan, plus the Reduce variants that also return
# indices.
#
# Every mudnn class used below was probed against CPU-computable answers before
# being wired up (/tmp/probe_p3): Concat on both axes, Fill with and without a
# mask, GatherX in GATHER and GATHER_ELEMENTS modes, Scatter in UPDATE_ONLY and
# ADD, TopK, Sort, Cum in ADD and MUL, and Reduce's RunWithIndices/RunIndices.
# All matched exactly, in float32 and int64, with no operand-order surprises.
#
# One measured limit shapes what is NOT here: mudnn hangs the device on a
# negative-strided input (Permute over a reversed dim returned `4 5 6 0 0 0` in
# isolation and wedged the queue when run in sequence), so flip/roll get no
# kernel. permute/transpose/t are absent for a different reason -- they are
# CompositeExplicitAutograd metadata ops that already alias correctly and run in
# ~2us via csrc/aten/strided_ops.cc, so they never reach the fallback at all.
# ==========================================================================

# cat/stack. mudnn's Concat takes a C array of Tensors and one axis. The
# operands must be materialized: MudnnTensorWrapper describes strides faithfully
# but Concat reads each input as a dense block, and aten allows a cat of
# arbitrary views. `stack` is cat-of-unsqueezed, so the same template serves both
# with the unsqueeze folded in via {pre_unsqueeze}.
T_CONCAT = """\
at::Tensor {kernel}({list_type} tensors, int64_t dim) {{
  std::vector<at::Tensor> ins;
  for (const at::Tensor& t : tensors) ins.push_back(t);
{pre_unsqueeze}
  // aten's cat ignores empty tensors, and lets them disagree about ndim: the
  // `torch.cat([empty_cache, value_states], dim=-2)` in transformers' KV
  // cache is legal when the empty operand has the same rank as the real one.
  // Dropping every zero-element operand before looking at sizes keeps the shape
  // arithmetic on the real operands and avoids mudnn's unsupported empty input.
  // If every operand is empty, retain aten's handling in the fallback below.
  std::vector<at::Tensor> kept;
  for (const auto& t : ins) {{
    if (t.numel() != 0) kept.push_back(t);
  }}
  if (kept.empty()) {{
    // Every operand is empty, so mudnn has nothing to concatenate and the
    // result carries no elements. Re-dispatching at::{at_op} here would land
    // back in this kernel and recurse until the stack is exhausted, so shape
    // and dtype come from a host concat of zero-element operands (no data
    // moves) and the result is placed back on the device.
    std::vector<at::Tensor> cpu_empty;
    for (const auto& t : ins) cpu_empty.push_back(t.cpu());
    return at::cat(cpu_empty, dim).to(ins[0].device());
  }}
  bool ok = true;
  for (const auto& t : kept) {{
    if (!musa_ops::{dtype_pred}(t.scalar_type()) ||
        t.scalar_type() != kept[0].scalar_type()) {{
      ok = false;
      break;
    }}
  }}
  if (!ok) {{
    std::vector<at::Tensor> cpu_ins;
    for (const auto& t : kept) cpu_ins.push_back(t.cpu());
    return at::cat(cpu_ins, dim).to(kept[0].device());
  }}
  int64_t ndim = kept[0].dim();
  int64_t d = dim < 0 ? dim + ndim : dim;
  auto out_shape = kept[0].sizes().vec();
  int64_t total = 0;
  for (const auto& t : kept) total += t.size(d);
  out_shape[d] = total;
  auto out = at::empty(out_shape, kept[0].options());
{empty_guard}
  std::vector<at::Tensor> ins_c;
  std::vector<std::unique_ptr<musa_ops::MudnnTensorWrapper>> wraps;
  std::vector<musa_ops::mudnn::Tensor> raw;
  for (const auto& t : kept) {{
    ins_c.push_back(t.contiguous());
    wraps.push_back(
        std::make_unique<musa_ops::MudnnTensorWrapper>(ins_c.back()));
    raw.push_back(wraps.back()->get());
  }}
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Concat op;
  op.SetAxis(static_cast<int>(d));
  EXEC_MUDNN_CMD(
      "{at_op}", kept[0],
      op.Run(_mudnn_h, t_out.get(), static_cast<int>(raw.size()), raw.data()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# gather(self, dim, index, sparse_grad) -> GatherX::GATHER_ELEMENTS, which is
# aten's element-wise gather (index has the output's shape and selects along
# `dim` only). Probed: axis 1 over [[1,2,3],[4,5,6]] with index [[2,1,0],[0,1,2]]
# gave [3,2,1,4,5,6]. NB the Run argument order is (out, index, in).
T_GATHER = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      index.scalar_type() != at::kLong || sparse_grad ||
      index.device() != self.device()) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), sparse_grad)
        .to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto self_c = self.contiguous();
  auto index_c = index.contiguous();
  auto out = at::empty(index.sizes(), self.options());
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_index(index_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::GatherX op;
  op.SetMode(musa_ops::mudnn::GatherX::Mode::{mode});
  op.SetAxis(static_cast<int>(d));
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_index.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# index_select(self, dim, index) -> GatherX::GATHER: a 1-D index picks whole
# slices along `dim`, so the output takes self's shape with dim replaced by
# index.numel(). Probed on axis 0: index [1,0] over [[1,2,3],[4,5,6]] gave
# [4,5,6,1,2,3].
T_INDEX_SELECT = """\
at::Tensor {kernel}(
    const at::Tensor& self, int64_t dim, const at::Tensor& index) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      index.scalar_type() != at::kLong || index.dim() > 1 ||
      index.device() != self.device()) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto self_c = self.contiguous();
  auto index_c = index.contiguous();
  auto out_shape = self.sizes().vec();
  out_shape[d] = index.numel();
  auto out = at::empty(out_shape, self.options());
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_index(index_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::GatherX op;
  op.SetMode(musa_ops::mudnn::GatherX::Mode::{mode});
  op.SetAxis(static_cast<int>(d));
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_index.get(), t_self.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# embedding(weight, indices, ...) is index_select along dim 0 with an
# arbitrarily-shaped index, so the output is indices.shape + weight.shape[1:].
# padding_idx only affects the backward, and scale_grad_by_freq/sparse likewise,
# so the forward ignores them exactly as the dense CUDA kernel does.
T_EMBEDDING = """\
at::Tensor {kernel}(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {{
  if (!musa_ops::{dtype_pred}(weight.scalar_type()) ||
      (indices.scalar_type() != at::kLong &&
       indices.scalar_type() != at::kInt) ||
      sparse || indices.device() != weight.device()) {{
    return at::{at_op}(
               weight.cpu(), indices.cpu(), padding_idx, scale_grad_by_freq,
               sparse)
        .to(weight.device());
  }}
  auto weight_c = weight.contiguous();
  auto index_c = indices.reshape({{-1}}).contiguous();
  std::vector<int64_t> out_shape = indices.sizes().vec();
  for (int64_t i = 1; i < weight.dim(); ++i) {{
    out_shape.push_back(weight.size(i));
  }}
  std::vector<int64_t> flat_shape = weight.sizes().vec();
  flat_shape[0] = index_c.numel();
  auto out = at::empty(flat_shape, weight.options());
{empty_guard}
  musa_ops::MudnnTensorWrapper t_weight(weight_c);
  musa_ops::MudnnTensorWrapper t_index(index_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::GatherX op;
  op.SetMode(musa_ops::mudnn::GatherX::Mode::{mode});
  op.SetAxis(0);
  EXEC_MUDNN_CMD(
      "{at_op}", weight,
      op.Run(_mudnn_h, t_out.get(), t_index.get(), t_weight.get()));
  return out.view(out_shape);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# scatter.src / scatter_add: out = self with index/src scattered along dim.
# mudnn's in-place Run(self, idx, update, dim) mutates its first argument, and
# aten's out-of-place form must not touch self, so the kernel clones first. The
# clone is what the functional op would allocate anyway. Probed both modes.
#
# NB on duplicate indices: when the same (dim-slice, index) pair appears twice,
# aten's own docs call scatter's result nondeterministic -- "which value ends up
# there is unspecified" -- so mudnn's winner may differ from CPU's. That is
# within spec and is not a correctness gap; scatter_add is unambiguous because
# it sums, and scatter.value is unambiguous because every write is the same
# value. Both match CPU exactly.
T_SCATTER = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      !musa_ops::{dtype_pred}(src.scalar_type()) ||
      self.scalar_type() != src.scalar_type() ||
      index.scalar_type() != at::kLong ||
      index.device() != self.device() ||
      src.device() != self.device()) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), src.cpu())
        .to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out = self.contiguous().clone();
  auto index_c = index.contiguous();
  auto src_c = src.contiguous();

  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::MudnnTensorWrapper t_index(index_c);
  musa_ops::MudnnTensorWrapper t_src(src_c);
  musa_ops::mudnn::Scatter op;
  op.SetMode(musa_ops::mudnn::Scatter::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_index.get(), t_src.get(),
             static_cast<int>(d), musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# scatter.value: same as above with a scalar `src`, expanded to the index shape.
# mudnn has no scalar-update form, so the scalar is materialized once at
# index.sizes() -- small relative to the clone the op already does.
T_SCATTER_VALUE = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      index.scalar_type() != at::kLong ||
      index.device() != self.device()) {{
    return at::{at_op}(self.cpu(), dim, index.cpu(), value)
        .to(self.device());
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out = self.contiguous().clone();
  auto index_c = index.contiguous();
  auto src_c = at::full(index.sizes(), value, self.options());

  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::MudnnTensorWrapper t_index(index_c);
  musa_ops::MudnnTensorWrapper t_src(src_c);
  musa_ops::mudnn::Scatter op;
  op.SetMode(musa_ops::mudnn::Scatter::Mode::{mode});
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_index.get(), t_src.get(),
             static_cast<int>(d), musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# fill_.Scalar / zero_: in-place, no input tensor. Fill::SetValue is overloaded
# on double vs int64_t and the integral overload must be chosen for integral
# dtypes, or an int64 fill of 3 writes the bit pattern of 3.0.
T_FILL_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self{value_param}) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) || !self.is_contiguous()) {{
    auto cpu = self.cpu();
    cpu.{at_op}({value_arg_cpu});
    self.copy_(cpu);
    return self;
  }}
  // mudnn rejects zero-element tensors; the in-place result is already the
  // correct device tensor, so do not launch Fill or fall back through the host.
  if (self.numel() == 0) return self;
  musa_ops::MudnnTensorWrapper t_self(self);
  musa_ops::mudnn::Fill op;
  if (at::isIntegralType(self.scalar_type(), true)) {{
    op.SetValue(static_cast<int64_t>({value_int}));
  }} else {{
    op.SetValue(static_cast<double>({value_double}));
  }}
  EXEC_MUDNN_CMD("{at_op}", self, op.Run(_mudnn_h, t_self.get()));
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# masked_fill.Scalar -> Fill's masked Run, which writes `value` only where the
# mask is true and leaves the rest untouched. Probed: value 9 with mask
# [1,0,1,0,1,0] over [1..6] gave 9 2 9 4 9 6. aten's masked_fill is
# out-of-place, so the kernel clones self first; the mask broadcasts to self.
T_MASKED_FILL = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      mask.scalar_type() != at::kBool ||
      mask.device() != self.device()) {{
    return at::{at_op}(self.cpu(), mask.cpu(), value).to(self.device());
  }}
  auto out_shape = at::infer_size(self.sizes(), mask.sizes());
  auto out = self.expand(out_shape).contiguous();
  auto mask_c = mask.expand(out_shape).contiguous();

  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::MudnnTensorWrapper t_mask(mask_c);
  musa_ops::mudnn::Fill op;
  if (at::isIntegralType(self.scalar_type(), true)) {{
    op.SetValue(value.to<int64_t>());
  }} else {{
    op.SetValue(value.to<double>());
  }}
  EXEC_MUDNN_CMD(
      "{at_op}", self, op.Run(_mudnn_h, t_out.get(), t_mask.get()));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# topk -> (values, indices). mudnn's TopK matches aten's argument set exactly
# (k / dim / largest / sorted), but its dtype support is narrower than the rest
# of the library: FLOAT/HALF/BFLOAT16/DOUBLE/INT32 work, while INT64, INT16,
# INT8 and UINT8 are rejected outright ("Unsupported in data type: INT64,
# TopKPreProcess failed"). Sort accepts all of them, so the restriction is
# TopK's alone and is spelled out here rather than in the shared predicate.
T_TOPK = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {{
  auto st = self.scalar_type();
  bool topk_dtype = st == at::kFloat || st == at::kHalf ||
                    st == at::kBFloat16 || st == at::kDouble || st == at::kInt;
  if (!musa_ops::{dtype_pred}(st) || !topk_dtype) {{
    auto r = at::{at_op}(self.cpu(), k, dim, largest, sorted);
    return std::make_tuple(
        std::get<0>(r).to(self.device()), std::get<1>(r).to(self.device()));
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto self_c = self.contiguous();
  auto out_shape = self.sizes().vec();
  out_shape[d] = k;
  auto values = at::empty(out_shape, self.options());
  auto indices = at::empty(out_shape, self.options().dtype(at::kLong));

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_values(values);
  musa_ops::MudnnTensorWrapper t_indices(indices);
  musa_ops::mudnn::TopK op;
  op.SetK(static_cast<int>(k));
  op.SetDim(static_cast<int>(d));
  op.SetLargest(largest);
  op.SetSorted(sorted);
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_values.get(), t_indices.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(values)));
  return std::make_tuple(values, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# sort / sort.stable -> (values, indices). aten's `stable` is an optional that
# defaults to false; mudnn takes it as a plain flag. {stable_param}/{stable_expr}
# absorb the difference between the two overloads.
#
# On ties: with stable=false (aten's default) the returned *values* always match
# CPU, but the *indices* chosen among equal elements need not -- measured on
# randint(-9,9) inputs across int64/int32/float32/float16. aten documents that
# order as unspecified unless stable=true, and with stable=true mudnn's indices
# do match CPU exactly, so this is conformant rather than a gap.
T_SORT = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& self,{stable_param}
    int64_t dim,
    bool descending) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    auto r = at::{at_op}(self.cpu(),{stable_cpu_arg} dim, descending);
    return std::make_tuple(
        std::get<0>(r).to(self.device()), std::get<1>(r).to(self.device()));
  }}
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto self_c = self.contiguous();
  auto values = at::empty(self.sizes(), self.options());
  auto indices = at::empty(self.sizes(), self.options().dtype(at::kLong));

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_values(values);
  musa_ops::MudnnTensorWrapper t_indices(indices);
  musa_ops::mudnn::Sort op;
  op.SetDim(static_cast<int>(d));
  op.SetDescending(descending);
  op.SetStable({stable_expr});
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_values.get(), t_indices.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(values)));
  return std::make_tuple(values, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# cumsum / cumprod -> Cum::ADD / Cum::MUL. mudnn's Cum has only these two modes,
# so cummax/cummin/logcumsumexp stay on the fallback rather than being faked.
T_CUM = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    int64_t dim,
    ::std::optional<at::ScalarType> dtype) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type()) ||
      (dtype.has_value() && !musa_ops::{dtype_pred}(dtype.value()))) {{
    return at::{at_op}(self.cpu(), dim, dtype).to(self.device());
  }}
  auto acc = dtype.has_value() ? dtype.value() : self.scalar_type();
  auto self_c = self.to(acc).contiguous();
  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out = at::empty(self.sizes(), self.options().dtype(acc));
{empty_guard}
  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_out(out);
  musa_ops::mudnn::Cum op;
  op.SetMode(musa_ops::mudnn::Cum::Mode::{mode});
  op.SetDim(static_cast<int>(d));
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.Run(_mudnn_h, t_out.get(), t_self.get(),
             musa_ops::MudnnWorkspaceFor(out)));
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# max.dim / min.dim -> Reduce::RunWithIndices, which fills the values and the
# argmax/argmin in one pass. Probed against both MAX and MIN; ties resolve to the
# lowest index, as aten does. The 0-strided guard is the same one the plain
# reductions carry (see MudnnReduceNeedsContiguous).
T_REDUCE_WITH_INDICES = """\
::std::tuple<at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& self, int64_t dim, bool keepdim) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    auto r = at::{at_op}(self.cpu(), dim, keepdim);
    return std::make_tuple(
        std::get<0>(r).to(self.device()), std::get<1>(r).to(self.device()));
  }}
  int64_t ndim = self.dim();
  int64_t d = dim < 0 ? dim + ndim : dim;
  auto out_shape = self.sizes().vec();
  auto squeezed_shape = self.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);
  squeezed_shape.erase(squeezed_shape.begin() + d);

  auto self_c = self;
  if (musa_ops::MudnnReduceNeedsContiguous(self_c)) {{
    self_c = self_c.contiguous();
  }}
  auto values = at::empty(squeezed_shape, self.options());
  auto indices = at::empty(squeezed_shape, self.options().dtype(at::kLong));
  int mudnn_dim = static_cast<int>(d);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_values(values);
  musa_ops::MudnnTensorWrapper t_indices(indices);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(1, &mudnn_dim);
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.RunWithIndices(_mudnn_h, t_values.get(), t_indices.get(),
                        t_self.get(), musa_ops::MudnnWorkspaceFor(values)));
  if (keepdim) {{
    return std::make_tuple(values.view(out_shape), indices.view(out_shape));
  }}
  return std::make_tuple(values, indices);
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

# argmax / argmin -> Reduce::RunIndices, the indices-only form. aten allows a
# nullopt dim meaning "over the flattened tensor", which is a reduce over dim 0
# of self.reshape(-1).
T_REDUCE_INDICES = """\
at::Tensor {kernel}(
    const at::Tensor& self, ::std::optional<int64_t> dim, bool keepdim) {{
  if (!musa_ops::{dtype_pred}(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, keepdim).to(self.device());
  }}
  auto self_r = dim.has_value() ? self : self.reshape({{-1}});
  int64_t ndim = self_r.dim();
  int64_t d = dim.has_value() ? (dim.value() < 0 ? dim.value() + ndim
                                                 : dim.value())
                              : 0;
  auto out_shape = self_r.sizes().vec();
  auto squeezed_shape = self_r.sizes().vec();
  if (keepdim) out_shape[d] = 1;
  else out_shape.erase(out_shape.begin() + d);
  squeezed_shape.erase(squeezed_shape.begin() + d);

  auto self_c = self_r;
  if (musa_ops::MudnnReduceNeedsContiguous(self_c)) {{
    self_c = self_c.contiguous();
  }}
  auto indices = at::empty(squeezed_shape, self.options().dtype(at::kLong));
  int mudnn_dim = static_cast<int>(d);

  musa_ops::MudnnTensorWrapper t_self(self_c);
  musa_ops::MudnnTensorWrapper t_indices(indices);
  musa_ops::mudnn::Reduce op;
  op.SetMode(musa_ops::mudnn::Reduce::Mode::{mode});
  op.SetDim(1, &mudnn_dim);
  EXEC_MUDNN_CMD(
      "{at_op}", self,
      op.RunIndices(_mudnn_h, t_indices.get(), t_self.get(),
                    musa_ops::MudnnWorkspaceFor(indices)));
  return keepdim ? indices.view(out_shape) : indices;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kMusa, {kernel})
"""

CATEGORIES = {
    "unary": T_UNARY,
    "unary_alpha_const": T_UNARY_ALPHA_CONST,
    "unary_two_pass": T_UNARY_TWO_PASS,
    "binary": T_BINARY,
    "binary_alpha": T_BINARY_ALPHA,
    "binary_cmp": T_BINARY_CMP,
    "unary_scalar": T_UNARY_SCALAR,
    "unary_scalar_alpha": T_UNARY_SCALAR_ALPHA,
    "unary_scalar_cmp": T_UNARY_SCALAR_CMP,
    "matmul": T_MATMUL,
    "matmul_out": T_MATMUL_OUT,
    "reduce_dims_dtype": T_REDUCE_DIMS_DTYPE,
    "reduce_all_dtype": T_REDUCE_ALL_DTYPE,
    "gelu": T_GELU,
    "softmax_fwd": T_SOFTMAX_FWD,
    "unary_cmp": T_UNARY_CMP,
    "unary_two_const": T_UNARY_TWO_CONST,
    "unary_param": T_UNARY_PARAM,
    "clamp_one_sided": T_CLAMP_ONE_SIDED,
    "clamp": T_CLAMP,
    "softplus": T_SOFTPLUS,
    "elu": T_ELU,
    "binary_bw": T_BINARY_BW,
    "gelu_bw": T_GELU_BW,
    "threshold_bw": T_THRESHOLD_BW,
    "leaky_relu_bw": T_LEAKY_RELU_BW,
    "ternary_value": T_TERNARY_VALUE,
    "where": T_WHERE,
    "addmm": T_ADDMM,
    "softmax_bwd": T_SOFTMAX_BWD,
    "reduce_dims_plain": T_REDUCE_DIMS_PLAIN,
    "reduce_bool": T_REDUCE_BOOL,
    "reduce_prod": T_REDUCE_PROD,
    "reduce_correction": T_REDUCE_CORRECTION,
    "reduce_norm": T_REDUCE_NORM,
    "layer_norm": T_LAYER_NORM,
    "layer_norm_bwd": T_LAYER_NORM_BWD,
    # ---- P3: shape / index / sort / scan ----
    "concat": T_CONCAT,
    "gather": T_GATHER,
    "index_select": T_INDEX_SELECT,
    "embedding": T_EMBEDDING,
    "scatter": T_SCATTER,
    "scatter_value": T_SCATTER_VALUE,
    "fill_scalar": T_FILL_SCALAR,
    "masked_fill": T_MASKED_FILL,
    "topk": T_TOPK,
    "sort": T_SORT,
    "cum": T_CUM,
    "reduce_with_indices": T_REDUCE_WITH_INDICES,
    "reduce_indices": T_REDUCE_INDICES,
}

# Categories whose mudnn mode is arithmetic, and therefore rejects bool operands
# ("Unsupported binary mode: MUL, with left data type: BOOL"). PyTorch does
# define bool arithmetic, so these kernels use the stricter predicate and let
# bool take the CPU fallback. The comparison and logical categories accept bool
# natively, and the reductions cast to the accumulate dtype before running.
ARITHMETIC_CATEGORIES = {
    "unary",
    "unary_alpha_const",
    "unary_two_pass",
    "binary",
    "binary_alpha",
    "unary_scalar",
    "unary_scalar_alpha",
    "matmul",
    "matmul_out",
    "gelu",
    "softmax_fwd",
    "unary_two_const",
    "unary_param",
    "clamp_one_sided",
    "clamp",
    "softplus",
    "elu",
    "binary_bw",
    "gelu_bw",
    "threshold_bw",
    "leaky_relu_bw",
    "ternary_value",
    "addmm",
    # P2. LayerNorm and the numeric reductions are float-only in practice; the
    # arithmetic predicate keeps bool off them, matching the other reduces.
    "softmax_bwd",
    "reduce_dims_plain",
    "reduce_prod",
    "reduce_correction",
    "reduce_norm",
    "layer_norm",
    "layer_norm_bwd",
    # P3. The data-movement categories (concat/gather/scatter/fill/sort/topk)
    # are deliberately absent: they copy elements rather than compute on them,
    # so bool is accepted there and they use the permissive predicate. Cum and
    # the Reduce index-returning variants do arithmetic/compares and stay
    # strict, like the other reductions.
    "cum",
    "reduce_with_indices",
    "reduce_indices",
}

# The mudnn class each category configures, for symbol validation.
CATEGORY_CLASS = {
    "unary": "Unary",
    "unary_alpha_const": "Unary",
    "unary_two_pass": "Unary",
    "binary": "Binary",
    "binary_alpha": "Binary",
    "binary_cmp": "Binary",
    "unary_scalar": "Unary",
    "unary_scalar_alpha": "Unary",
    "unary_scalar_cmp": "Unary",
    "matmul": None,  # mode names the class itself (MatMul / BatchMatMul)
    "matmul_out": None,
    "reduce_dims_dtype": "Reduce",
    "reduce_all_dtype": "Reduce",
    "gelu": "Unary",
    "softmax_fwd": "Softmax",
    "unary_cmp": "Unary",
    "unary_two_const": "Unary",
    "unary_param": "Unary",
    "clamp_one_sided": "Unary",
    "clamp": "Unary",
    "softplus": "Unary",
    "elu": "Unary",
    "binary_bw": "Binary",
    "gelu_bw": "Binary",
    "threshold_bw": "Binary",
    "leaky_relu_bw": "Binary",
    "ternary_value": "Ternary",
    "where": "Ternary",
    "addmm": None,  # mode names the class itself (MatMul / BatchMatMul)
    "softmax_bwd": "Softmax",
    "reduce_dims_plain": "Reduce",
    "reduce_bool": "Reduce",
    "reduce_prod": "Reduce",
    "reduce_correction": "Reduce",
    "reduce_norm": "Reduce",
    "layer_norm": "LayerNorm",
    "layer_norm_bwd": "LayerNorm",
    # P3. Concat/Fill/Permute/Reduce live in mudnn_tensor.h rather than
    # mudnn_ops.h, but they are the same ImplBase-derived classes and the symbol
    # gate resolves them identically.
    "concat": "Concat",
    "gather": "GatherX",
    "index_select": "GatherX",
    "embedding": "GatherX",
    "scatter": "Scatter",
    "scatter_value": "Scatter",
    "fill_scalar": None,  # Fill has no Mode enum -- SetValue is the whole config
    "masked_fill": None,
    "topk": None,  # TopK has no Mode enum either
    "sort": None,
    "cum": "Cum",
    "reduce_with_indices": "Reduce",
    "reduce_indices": "Reduce",
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_mudnn.py -- DO NOT EDIT.
//
// mudnn kernels for the Moore Threads MUSA backend, generated per-category.
// Each kernel describes its aten tensors as musa::dnn::Tensors, configures a
// mudnn op object and issues one Run via EXEC_MUDNN_CMD. Dispatchers are
// declared in generated/ops.h (shared with the CUDA codegen); here we only fill
// the Backend::kMusa slot.
//
// mudnn links against musart only -- no torch symbols -- so unlike the previous
// at::musa::* passthroughs these kernels are independent of the torch version.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/result_type.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include "../mudnn_common.h"

namespace at::native::flagos {

namespace musa_ops = at::native::flagos::musa_ops;

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"

INC_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen_mudnn.py -- DO NOT EDIT.
//
// m.impl() lines for the ops that have a mudnn kernel. Included by register.cc
// inside TORCH_LIBRARY_IMPL(aten, PrivateUse1) when USE_MUSA is set, in place
// of the full generated/register.inc list: an op registered on PrivateUse1
// without a kernel behind it fails the dispatcher's "backend not registered"
// check, whereas an unregistered op simply reaches the cpu_fallback. So this
// file is exactly the MUSA coverage set.
//
// The list also covers the handwritten mudnn convolution and native
// muRAND/mudnn RNG kernels. Unsupported RNG schemas stay unregistered and
// continue through cpu_fallback.

"""


def libmudnn_path() -> Path:
    env = os.environ.get("MUDNN_LIB")
    if env:
        return Path(env)
    musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
    return Path(musa_home) / "lib/libmudnn.so"


def symbols(lib: Path):
    """mudnn ops are C++ classes in `namespace musa::dnn`, so the mangled names
    are demangled (nm -C) and matched as `musa::dnn::<Class>::`."""
    if not lib.exists():
        print(f"[warn] {lib} not found; skipping symbol validation", file=sys.stderr)
        return None
    out = subprocess.run(
        ["nm", "-DC", "--defined-only", str(lib)], capture_output=True, text=True
    )
    return set(re.findall(r"musa::dnn::(\w+)::", out.stdout))


# mudnn rejects zero-element operands outright -- every mode probed on v3300
# (Unary, Binary, Reduce, Fill, MatMul) answers NOT_SUPPORTED rather than
# no-opping, and mudnn's UncontigHandle additionally faults. Empty tensors are
# not an exotic input: a zero-length slice and its backward produce them, so the
# kernels must handle them instead of erroring (issue #214). Where the *output*
# is empty the answer carries no elements, so returning the allocation
# unwritten is exactly aten's result and no launch is needed. This must stay on
# device: a cpu() detour would silently move the result off the accelerator.
#
# A dim-wise reduce over an empty axis into a non-empty output is a separate
# case, and mudnn already answers it correctly (measured on v3300: (4,0) reduced
# over dim 1 gives sum 0, prod 1, mean nan, any false, all true, norm 0, all
# matching CPU), so it is left alone. The whole-tensor reduce is the exception
# handled by EMPTY_ALL_REDUCE_IDENTITY below: its output is a non-empty scalar,
# so the guard here cannot fire, yet the empty input still trips mudnn.
def empty_guard(ret_expr: str) -> str:
    return (
        "  // mudnn rejects zero-element operands (NOT_SUPPORTED); an empty\n"
        "  // output holds no elements, so the allocation above is already the\n"
        "  // answer. Return it on-device without launching.\n"
        f"  if (out.numel() == 0) return {ret_expr};\n"
    )


# What each guarded template returns, so the guard's early return matches the
# kernel's own final return (keepdim views included) instead of assuming `out`.
EMPTY_GUARD_RETURN = {
    "embedding": "out.view(out_shape)",
    "reduce_bool": "keepdim ? out.view(out_shape) : out",
    "reduce_correction": "keepdim ? out.view(out_shape) : out",
    "reduce_dims_dtype": "keepdim ? out.view(out_shape) : out",
    "reduce_dims_plain": "keepdim ? out.view(out_shape) : out",
    "reduce_norm": "keepdim ? out.view(out_shape) : out",
    "reduce_prod": "keepdim ? out.view(out_shape) : out",
}

# aten's whole-tensor reduction of an empty input, measured on CPU: sum 0,
# mean nan (0/0), prod 1. Only these three ops use reduce_all_dtype.
EMPTY_ALL_REDUCE_IDENTITY = {
    "ADD": ("0", "sum -> 0"),
    "MEAN": ("std::numeric_limits<double>::quiet_NaN()", "mean -> nan"),
    "PROD": ("1", "prod -> 1"),
}


def wrapper_map():
    """op name -> Wrapper<Name>, read back from the CUDA codegen's register.inc
    so the m.impl() subset we emit cannot drift from the wrappers that exist."""
    if not REGISTER_INC.exists():
        return {}
    text = REGISTER_INC.read_text()
    return dict(re.findall(r'^\s*m\.impl\("([^"]+)",\s*(\w+)\);', text, flags=re.M))


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
        help="do not append covered ops to backends_musa.conf",
    )
    args = ap.parse_args()

    syms = symbols(libmudnn_path())
    wrappers = wrapper_map()

    bodies = []
    covered = []  # (op, mode, category)
    skipped = []  # (op, reason)

    for op, (cat, mode) in OPS.items():
        if args.category != "all" and cat != args.category:
            continue
        if op in SKIP:
            skipped.append((op, "handwritten"))
            continue
        base = op.split(".")[0]
        # Composed categories carry a tuple of (mode, ...) rather than one mode.
        if isinstance(mode, tuple):
            mode_name, extra = mode[0], mode[1:]
        else:
            mode_name, extra = mode, ()
        cls = CATEGORY_CLASS[cat] or mode_name
        if syms is not None and cls not in syms:
            skipped.append((op, f"musa::dnn::{cls} not in {libmudnn_path().name}"))
            continue
        if op not in wrappers:
            skipped.append((op, "no wrapper in generated/register.inc"))
            continue
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelMusa"  # SqrtFn -> SqrtKernelMusa
        fmt = dict(
            kernel=kernel,
            mode=mode_name,
            fn=fn,
            disp=disp,
            at_op=AT_OP_OVERRIDES.get(op, base),
            empty_guard=empty_guard(EMPTY_GUARD_RETURN.get(cat, "out")),
            promote_integral="true" if base == "sum" else "false",
            dtype_pred=(
                "MudnnSupportsArithmeticDtype"
                if cat in ARITHMETIC_CATEGORIES
                else "MudnnSupportsDtype"
            ),
        )
        if cat == "reduce_all_dtype":
            if mode_name not in EMPTY_ALL_REDUCE_IDENTITY:
                raise SystemExit(
                    f"{op}: reduce_all_dtype mode {mode_name} has no empty-input "
                    "identity; add it to EMPTY_ALL_REDUCE_IDENTITY"
                )
            ident, note = EMPTY_ALL_REDUCE_IDENTITY[mode_name]
            fmt["identity"], fmt["identity_note"] = ident, note
        if cat == "unary_alpha_const":
            fmt["alpha"] = extra[0]
        elif cat == "unary_two_pass":
            fmt["mode2"], fmt["alpha"] = extra[0], extra[1]
        elif cat == "unary_two_const":
            fmt["alpha"], fmt["beta"] = extra[0], extra[1]
        elif cat == "unary_param":
            fmt["param"] = extra[0]
        elif cat == "clamp_one_sided":
            fmt["param"], fmt["lo"], fmt["hi"] = extra[0], extra[1], extra[2]
        elif cat == "concat":
            # cat takes an ITensorListRef, stack a plain TensorList, and stack
            # has to unsqueeze every operand before concatenating.
            fmt["list_type"] = extra[0]
            fmt["pre_unsqueeze"] = (
                "  for (auto& t : ins) t = t.unsqueeze(dim);" if extra[1] else ""
            )
        elif cat == "fill_scalar":
            # zero_ has no value argument; fill_.Scalar takes a Scalar.
            fmt["value_param"] = extra[0]
            fmt["value_arg_cpu"] = extra[1]
            fmt["value_int"] = extra[2]
            fmt["value_double"] = extra[3]
        elif cat == "sort":
            # sort.stable carries an extra optional<bool> ahead of `dim`.
            fmt["stable_param"] = extra[0]
            fmt["stable_cpu_arg"] = extra[1]
            fmt["stable_expr"] = extra[2]
        bodies.append(CATEGORIES[cat].format(**fmt))
        covered.append((op, mode_name, cat))

    for op in NO_MUDNN_EQUIVALENT:
        skipped.append((op, "no mudnn mode; stays on cpu_fallback"))

    # Handwritten kernels contribute an m.impl() line but no generated body.
    handwritten = []
    if args.category == "all":
        for op in HANDWRITTEN_REGISTRATIONS:
            if op in wrappers:
                handwritten.append(op)
            else:
                skipped.append((op, "handwritten, but no wrapper in register.inc"))

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(FILE_HEADER + "\n".join(bodies) + FILE_FOOTER)

    impl_ops = sorted([op for op, _, _ in covered] + handwritten)
    impls = "".join(f'  m.impl("{op}", {wrappers[op]});\n' for op in impl_ops)
    OUT_INC.write_text(INC_HEADER + impls)

    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({len(covered)} kernels)")
    print(f"[gen] {OUT_INC.relative_to(REPO)}  ({len(impl_ops)} m.impl lines)")
    for op in handwritten:
        print(f"       * {op} (handwritten in mudnn_conv.cc)")
    by_cat = {}
    for op, mode, cat in covered:
        by_cat.setdefault(cat, []).append((op, mode))
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if items:
            print(f"    [{cat}] {len(items)}")
            for op, mode in items:
                print(f"       + {op} -> {mode}")
    for op, why in skipped:
        print(f"       - {op} skipped ({why})")

    if not args.no_conf and covered:
        existing = CONF.read_text() if CONF.exists() else ""
        # Strip any prior codegen block so re-runs stay idempotent.
        marker = "\n# --- generated by codegen_mudnn.py"
        if marker in existing:
            existing = existing[: existing.index(marker)].rstrip() + "\n"
        lines = []
        for op in [o for o, _, _ in covered] + handwritten:
            if f"\n{op} = " not in ("\n" + existing):
                lines.append(f"{op} = musa")
        new = existing.rstrip() + "\n"
        if lines:
            new += "\n# --- generated by codegen_mudnn.py ---\n"
            new += "\n".join(lines) + "\n"
        CONF.write_text(new)
        print(f"[conf] wrote {len(lines)} generated op(s) to {CONF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
