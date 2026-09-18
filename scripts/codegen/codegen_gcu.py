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

REPO = Path(__file__).resolve().parent.parent.parent
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
    # ---- the out= forms, which are also what the in-place ops box onto ----
    "mul.out": ("binary_out", None),
    "add.out": ("binary_alpha_out", None),
    "sub.out": ("binary_alpha_out", None),
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
    # ---- ordered reduction over an optional dim list ----
    "linalg_vector_norm": ("linalg_vector_norm", None),
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
    # arange: all three overloads share one template; the per-overload scalars,
    # the locals the shared body reads and ATen's dtype-inference predicate for
    # that overload live in ARANGE_OVERLOADS.
    "arange": ("arange", None),
    "arange.start": ("arange", None),
    "arange.start_step": ("arange", None),
    # The in-place half of the factories (zeros/ones/full) plus the masked write
    # every attention path builds its mask out of.
    "zero_": ("zero_inplace", "Zero"),
    "fill_.Scalar": ("fill_inplace_scalar", "Fill_"),
    "fill_.Tensor": ("fill_inplace_tensor", "Fill_"),
    "masked_fill_.Scalar": ("masked_fill_inplace_scalar", "Masked_fill"),
    "masked_fill_.Tensor": ("masked_fill_inplace_tensor", "Masked_fill"),
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
    # ---- indexed reads and writes, and the embedding gradient ----
    # The inference census never reaches these; a *training* step does. Every
    # overload is listed, including the out= and non-in-place spellings, because
    # an overload left out keeps its `= none` route and reaches cpu_fallback on
    # its own.
    #
    # The four index_fill overloads already sit in FLAGGEMS_PENDING_NATIVE_OPS,
    # and that entry is what leaves them on these kernels: the set is subtracted
    # from FlagGems' Python coverage in gen_vendor_confs.py, so with a native
    # wrapper registered the route falls through to `gcu` instead of coming back
    # `flaggems  # gcu`. Leaving the four entries in place is also what keeps the
    # Ascend and MUSA confs -- which read the same set -- unchanged.
    #
    # The in-place pair is listed first because the generator emits kernels in
    # this order: the three non-in-place spellings are thin wrappers around
    # IndexFillInplace*KernelGcu, and a definition has to precede its use.
    "index_select": ("index_select", None),
    "index_select.out": ("index_select_out", None),
    "index_fill_.int_Scalar": ("index_fill_inplace_scalar", None),
    "index_fill_.int_Tensor": ("index_fill_inplace_tensor", None),
    "index_fill.int_Scalar": ("index_fill_scalar", None),
    "index_fill.int_Scalar_out": ("index_fill_scalar_out", None),
    "index_fill.int_Tensor": ("index_fill_tensor", None),
    "index_fill.int_Tensor_out": ("index_fill_tensor_out", None),
    "embedding_dense_backward": ("embedding_dense_backward", None),
    "embedding_dense_backward.out": ("embedding_dense_backward_out", None),
    # The VAE's upsampling block. The name needs no override: topsaten_name
    # strips the leading underscore before PascalCasing, so the default already
    # spells topsatenUpsampleNearestExact2d.
    "_upsample_nearest_exact2d": ("upsample_nearest_exact2d", None),
    "_upsample_nearest_exact2d.out": ("upsample_nearest_exact2d_out", None),
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
    # ---- the rest of the cpu_fallback census ----
    # The five ops below are what a Qwen-Image denoise step still reaches
    # cpu_fallback on after the indexed/embedding/upsample work (measured with
    # FLAGOS_LOG=fallback on the S60: 56 @where, 2 @nonzero, 2 @index, 2 @all
    # per run). Each has a topsaten entry point, measured on card 4 rather than
    # assumed -- see the templates for what each kernel encodes.
    #
    # Listing them here is what makes them routable at all, in two steps that are
    # easy to confuse: this dict emits the m.impl() into gcu_register.inc, which
    # is the registration gen_vendor_confs.route() gates every route on, and the
    # conf then places the op. `where.self`, `all` and `nonzero` are already in
    # the GCU NATIVE_TRITON_GAPS set and `nonzero_static` in
    # FLAGGEMS_PENDING_NATIVE_OPS, so the FlagGems route is closed for them and
    # they land on `gcu`; `index.Tensor` is in no FlagGems set at all.
    #
    # index.Tensor needs a word of warning: ATen's advanced indexing is one op
    # covering a rank-1 index (== index_select), several index tensors at once,
    # a bool mask and a 0-dim index, and only the first of those is what the
    # vendor's index kernel implements. The kernel recognises that case and
    # delegates; everything else takes the host path with ATen's own semantics.
    "where.self": ("where_self", None),
    # The out= spelling is claimed for a different reason than the out-of-place
    # one: not a cpu_fallback, but the last op of ATen's eager `_safe_softmax`,
    # which FlagGems serves at (1, 24, 4114, 4114) in 1098.6 ms against the
    # vendor's 23.7 ms for the same operation. See T_WHERE_SELF_OUT.
    "where.self_out": ("where_self_out", None),
    "all": ("all_whole", None),
    "index.Tensor": ("index_tensor", None),
    "nonzero": ("nonzero", None),
    # Built on topsatenNonzero plus a count, an i32 staging tensor and a cast;
    # the override names the entry point the op is built on, which is also the
    # symbol the generator validates against libtopsaten.so (there is no
    # topsatenNonzeroStatic).
    "nonzero_static": ("nonzero_static", "Nonzero"),
}

# Ops handwritten elsewhere for GCU would double-register the kGcu slot (which
# crashes at import), so they must be excluded here. None yet.
SKIP: set = set()

# Pure-metadata view ops. There is no topsaten entry point for either (there
# cannot be one: they reinterpret size/stride/dtype over the *same* storage, so
# there is nothing for a device kernel to compute), and every category template
# below ends in a `topsaten::` call, so the generator cannot express them. They
# are implemented next to the other stride ops in csrc/aten/strided_ops.cc --
# the same file that already carries alias/detach/t/permute/_conj for Ascend --
# and listed here only so this generator still emits their m.impl() into
# gcu_register.inc and their `= gcu` route into the conf.
#
# The registration is not optional even though the ops need no compute:
# gen_vendor_confs.route() gates on the PrivateUse1 registration, so an op left
# out of gcu_register.inc routes to `none` and reaches ATen's cpu_fallback. A
# view op cannot survive that -- the fallback copies, so the result stops
# aliasing the input's storage -- and, in the Qwen-Image RoPE path, view_as_real
# and view_as_complex are called 488 times each per denoise step.
METADATA_OPS = {
    "view_as_complex",
    "view_as_real",
}

# Handwritten kernels live in a separate translation unit when the vendor API
# does not fit one of the category templates. They still belong in the generated
# PrivateUse1 coverage include, so list their ATen overloads here.
HANDWRITTEN_OPS = {
    "_sample_dirichlet",
    "_standard_gamma",
    "bernoulli",
    "bernoulli_.float",
    "bernoulli_.Tensor",
    "binomial",
    "exponential",
    "exponential_",
    "multinomial",
    "native_dropout",
    "native_dropout_backward",
    "normal_",
    "normal.float_float",
    "normal.Tensor_float",
    "normal.Tensor_Tensor",
    "poisson",
    "rand",
    "rand.generator",
    "rand.names_out",
    "rand.out",
    "rand_like",
    "rand_like.generator",
    "rand_like.out",
    "randn",
    "randn.generator",
    "randn.names_out",
    "randn_like",
    "randn_like.generator",
    "randn_like.generator_out",
    "randn_like.out",
    "randint",
    "randint.generator",
    "randint.low",
    "randint.low_generator",
    "randint.low_out",
    "randint.out",
    "randint_like",
    "randint_like.low_dtype",
    "randint_like.low_dtype_out",
    "randint_like.out",
    "randperm",
    "randperm.generator",
    "randperm.out",
    "random_",
    "random_.from",
    "random_.to",
    "uniform_",
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
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto other_c = other.to(self.device(), result_dtype);
  auto out_shape = at::infer_size(self_c.sizes(), other_c.sizes());
  auto self_b = self_c.expand(out_shape).contiguous();
  auto other_b = other_c.expand(out_shape).contiguous();
"""


# The dtype test has to be `at::result_type`, not `other.scalar_type()`.
#
# A Python number operand reaches these kernels as a *wrapped scalar tensor*: a
# real 0-dim tensor carrying the number's own dtype (f64 for a float, i64 for an
# int), flagged `is_wrapped_number`, which promotion deliberately stops from
# widening `self`. So `other.scalar_type()` is f64 for the most ordinary spelling
# there is -- `x * 2.0` -- while the arithmetic is f32 end to end. Testing that
# dtype rejected the operand, and the guard's host fallback then moved the whole
# tensor to the CPU and back:
#
#   (1,24,4114,4114) f32, 1.51 GiB     a * 2.0                        575 ms
#                                      a * 2                          616 ms
#                                      a * <f32 0-dim device tensor>   20 ms
#                                      a * <f32 0-dim CPU tensor>      22 ms
#   torch.profiler on `a * 2.0`: 2 x topsMemcpy (219 ms each), 0 topsLaunchKernel.
#
# `at::result_type` is wrapped-number aware (`ResultTypeState::wrappedResult`),
# so it reports f32 for `mul(f32_tensor, wrapped_f64)` and for
# `mul(f32_tensor, wrapped_i64)` -- the type the arithmetic is actually done in.
# A genuine f64 operand, wrapped or not, still promotes to f64 and still falls
# back, which is what the CPU does as well: topsaten has no float64 kernels.
#
# `TopsatenSupportsDtype(self.scalar_type())` stays. `self` is read as well as
# written, so an i64 self has to fall back even when its promotion lands on f32.
#
# Both helpers below emit braces, and their output is concatenated into a
# template that `CATEGORIES[cat].format(...)` expands later, so every brace is
# doubled here -- the same convention the templates themselves use.
def _binary_dtype_guard(host_body: str) -> str:
    """Emit `result_dtype` and the unsupported-dtype fallback for a binary op.

    `host_body` is the fallback itself, indented for the `if` body, and may use
    `result_dtype`.
    """
    return (
        "  auto result_dtype = at::result_type(self, other);\n"
        "  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||\n"
        "      !gcu::TopsatenSupportsDtype(result_dtype)) {{\n" + host_body + "  }}\n"
    )


def _binary_host_fallback(at_op: str, alpha: bool = False, out: bool = False) -> str:
    """The fallback body for `_binary_dtype_guard`, as an indented block."""
    args = "self.cpu(), other.cpu(), alpha" if alpha else "self.cpu(), other.cpu()"
    if not out:
        return f"    return at::{at_op}({args}).to(self.device());\n"
    return (
        f"    auto host = at::{at_op}({args});\n"
        "    if (!out.sizes().equals(host.sizes())) {{\n"
        "      out.resize_(host.sizes());\n"
        "    }}\n"
        "    out.copy_(host);\n"
        "    return out;\n"
    )


# Every `*_out` kernel below may grow a caller-supplied `out`, and that resize
# has to happen with the right device selected. `resize_` takes no device
# argument: `TensorImpl::resize_` allocates through the allocator, which
# resolves the pool from the *current* device
# (`caching_device_allocator.cc` calls `backend_->get_device_index()`), and it
# installs no guard of its own. The factories do (`at::empty(..., device=...)`
# guards for its own duration), which is why an ordinary device tensor is fine,
# but a multi-device process sitting on device 0 will grow `out` out of device
# 0's pool while the tensor still reports flagos:N, and topsaten then refuses the
# descriptor:
#
#   FindMemObj DeviceId[0] of memory VA[0x...] is not match for DeviceId[6] of
#   stream
#
# Measured: `torch.add(a, a, out=torch.empty(0, device="flagos:6"))` failed with
# the current device at 0 and passed once device 6 was current. The guard is a
# no-op when the current device already matches, so it costs nothing on the path
# that was already working.
_OUT_DEVICE_GUARD = """\
  gcu::TopsDeviceGuard out_guard(self);
"""


# ATen refuses an `out` it cannot be cast into rather than truncating into it, and
# the in-place spellings inherit that rule: `add_.Tensor` is a composite that
# redispatches onto `add.out` with out=self, so `int32_t.add_(fp32_tensor)` raises
# on the CPU with
#
#   RuntimeError: result type Float can't be cast to the desired output type Int
#
# (`TensorIterator` makes the same `canCast` check, and ATen's out= parsing
# applies it before any element is touched). The host path below ends in an
# ordinary `copy_`, which casts, so without this check a device call would
# silently narrow where the CPU raises -- `torch.mul(fp32_a, fp32_b,
# out=int32_tensor)` measured `max|diff| 1.19e9` instead of an error.
#
# The check sits ahead of the unsupported-dtype branch so that path is covered
# too, and ahead of the resize so a rejected call leaves `out` untouched.
def _out_cast_check(result_expr: str, define: bool = False) -> str:
    """The `canCast` guard every `*_out` template runs before it writes.

    `define` emits the `result_dtype` local the caller's later dtype comparisons
    need; without it the expression is used only for the check, because the
    prologue that follows already defines that local.
    """
    check = "result_dtype" if define else result_expr
    body = (
        "  TORCH_CHECK(\n"
        f"      c10::canCast({check}, out.scalar_type()),\n"
        '      "result type ",\n'
        f"      {check},\n"
        '      " can\'t be cast to the desired output type ",\n'
        "      out.scalar_type());\n"
    )
    return f"  auto result_dtype = {result_expr};\n" + body if define else body


T_BINARY = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
"""
    + _binary_dtype_guard(_binary_host_fallback("{at_op}"))
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
"""
    + _binary_dtype_guard(_binary_host_fallback("{at_op}", alpha=True))
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

# topsaten<Name>(out, self, other [, alpha]) with the output supplied by the
# caller. ATen gives the in-place arithmetic ops no PrivateUse1 registration of
# their own: `add_.Tensor`, `sub_.Tensor` and `mul_.Tensor` are structured
# in-place ops whose composite redispatches onto `add.out` / `sub.out` /
# `mul.out` with out=self. The fallback log prints the *schema* name, not the
# overload, so every one of them reads as `aten::add` -- which is why the census
# below had to be read as "the in-place arithmetic", not as "the out-of-place
# op that is already native".
#
# Measured on a two-step Adam run over a five-parameter model, run once with
# `foreach=True` and once with `foreach=False`: 25 `aten::add` and 10 `aten::mul`
# cpu_fallback calls in total, all of them from the in-place state updates
# (`exp_avg.lerp_`, `exp_avg_sq.mul_(beta2)`, `denom.add_(eps)`) -- a
# device->host->device round trip per parameter per step, and the whole of the
# fallback traffic the optimizer itself produces. topsatenAdd/Mul/Sub write into
# an out that may alias an input; `zero_`, `fill_` and `masked_fill_` above
# already rely on that, so one template serves both the explicit `out=` spelling
# and the in-place one.
#
# Unlike the out-of-place categories the host path has to land in `out`, so it
# resizes `out` to the result shape first and then copies: the ATen contract for
# an out= op is that the kernel sizes the output, and a caller that passes a
# zero-size or differently-shaped `out` (which is legal, and is what `out=`
# callers in this stack do after `.resize_()`) would otherwise get the shape it
# passed rather than the result's. `out` is the write target, so a dtype it
# cannot hold, a non-contiguous it, or a device that disagrees with self all
# take the host path rather than handing topsaten a descriptor it would write
# through with the wrong width.
T_BINARY_OUT = (
    """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {{
"""
    + _OUT_DEVICE_GUARD
    + _out_cast_check("at::result_type(self, other)")
    + _binary_dtype_guard(_binary_host_fallback("{at_op}", out=True))
    + _BINARY_PROLOGUE
    + """\
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  if (out.scalar_type() != result_dtype || !out.is_contiguous() ||
      out.device() != self.device()) {{
    out.copy_(at::{at_op}(self.cpu(), other.cpu()));
    return out;
  }}

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, out, t_out.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_ALPHA_OUT = (
    """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {{
"""
    + _OUT_DEVICE_GUARD
    + _out_cast_check("at::result_type(self, other)")
    + _binary_dtype_guard(_binary_host_fallback("{at_op}", alpha=True, out=True))
    + _BINARY_PROLOGUE
    + """\
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  if (out.scalar_type() != result_dtype || !out.is_contiguous() ||
      out.device() != self.device()) {{
    out.copy_(at::{at_op}(self.cpu(), other.cpu(), alpha));
    return out;
  }}
  auto t_alpha = gcu::ToTopsatenScalar(alpha, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, out, t_out.get(), t_self.get(), t_other.get(), t_alpha);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_BINARY_CMP = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
"""
    + _binary_dtype_guard(_binary_host_fallback("{at_op}"))
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
  auto result_dtype = at::result_type(self, other);
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(result_dtype)) {{
    return at::{at_op}(self.cpu(), other).to(self.device());
  }}
  // The comparison happens in the promoted type, not in self's. `int32 < 0.5`
  // promotes to f32 and the CPU compares 0 < 0.5, so converting the scalar into
  // self's dtype first would truncate the bound to 0 and answer False for every
  // value in [0, 1). Casting self instead is what makes the two agree.
  auto self_c = self.scalar_type() == result_dtype ? self : self.to(result_dtype);
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
  auto t_other = gcu::ToTopsatenScalar(other, result_dtype);

  gcu::TopsatenTensorWrapper t_self(self_c);
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

T_MATMUL_OUT = (
    """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {{
"""
    + _OUT_DEVICE_GUARD
    + _out_cast_check("at::result_type(self, mat2)", define=True)
    + """\
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
  if (out.scalar_type() != result_dtype || !out.is_contiguous() ||
      out.device() != self.device()) {{
    out.copy_(at::{at_op}(self.cpu(), mat2.cpu()));
    return out;
  }}

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_mat2(mat2);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_mat2.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

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

# linalg_vector_norm is the kernel under tensor.norm and F.normalize, which
# diffusers' QwenImageRMS_norm calls on the VAE decode as
# F.normalize(x, dim=1) over (1, 128, 1, 1024, 1024). FlagGems cannot serve it
# (its Triton grid.x overflows at that size -- see NATIVE_TRITON_GAPS), so this
# is the route that keeps the op off cpu_fallback.
#
# Two details shape the body. topsaten rejects a rank-0 tensor, which is exactly
# what dim=None with keepdim=False produces, so the vendor call always runs with
# keepdim=true and the result is reshaped afterwards -- a metadata view over a
# buffer this kernel just allocated, not a copy. And the accumulation dtype:
# ATen's reduction widens half/bfloat16 to float through acc_type and casts the
# result back, which is a contract the vendor kernel makes no promise about, so
# only float32 takes the vendor path and every other dtype goes to the host,
# where that widening is guaranteed. The pipeline calls this at float32 only --
# F.normalize in diffusers' QwenImageRMS_norm gets x.float() first.
#
# An absent or empty dim list means "every dim" on both sides, but the dims are
# enumerated anyway so the two cannot disagree about reduce_dim's empty form.
T_LINALG_VECTOR_NORM = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Scalar& ord,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {{
  auto out_dtype = dtype.value_or(self.scalar_type());
  if (self.scalar_type() != at::kFloat ||
      !gcu::TopsatenSupportsDtype(out_dtype) ||
      !self.is_contiguous() || self.numel() == 0) {{
    return at::{at_op}(self.cpu(), ord, dim, keepdim, dtype).to(self.device());
  }}
  int64_t ndim = self.dim();
  std::vector<int64_t> reduce_dims;
  if (dim.has_value() && !dim.value().empty()) {{
    for (int64_t d : dim.value()) reduce_dims.push_back(d < 0 ? d + ndim : d);
  }} else {{
    for (int64_t d = 0; d < ndim; ++d) reduce_dims.push_back(d);
  }}

  auto out_shape = self.sizes().vec();
  std::vector<int64_t> sorted_dims(reduce_dims);
  std::sort(sorted_dims.rbegin(), sorted_dims.rend());
  for (int64_t d : sorted_dims) {{
    if (keepdim) out_shape[d] = 1;
    else out_shape.erase(out_shape.begin() + d);
  }}
  // The buffer the vendor writes is the keepdim=true shape; out_shape is what
  // ATen promises and is reached by a (metadata-only) reshape.
  auto kept_shape = self.sizes().vec();
  for (int64_t d : reduce_dims) kept_shape[d] = 1;

  auto self_c = self.scalar_type() == out_dtype ? self : self.to(out_dtype);
  auto out = at::empty(kept_shape, self.options().dtype(out_dtype));
  gcu::TopsatenSizeWrapper t_dims(reduce_dims);
  // `ord` is a real order (2 for F.normalize, +/-inf for max/min norms), so it
  // is staged as a float whatever integral form the Python caller passed.
  auto t_ord = gcu::ToTopsatenScalar(ord, at::kFloat);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_ord, t_dims.get(),
      true, gcu::ToTopsatenDataType(out_dtype));
  return out.reshape(out_shape);
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
# The bounds are promoted into the result the way ATen's clamp meta promotes
# them, so a float bound on an integral tensor computes in float -- see
# gcu::ClampComputeDtype.
T_CLAMP = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) {{
  auto result_dtype = gcu::ClampComputeDtype(self, min, max);
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenSupportsDtype(result_dtype)) {{
    return at::{at_op}(self.cpu(), min, max).to(self.device());
  }}
  auto self_c = (self.scalar_type() == result_dtype ? self : self.to(result_dtype))
                    .contiguous();
  auto out = at::empty(self_c.sizes(), self_c.options());
  auto lo = min.has_value()
      ? gcu::ToTopsatenScalar(min.value(), result_dtype)
      : gcu::ToTopsatenScalar(gcu::DtypeLowest(result_dtype), result_dtype);
  auto hi = max.has_value()
      ? gcu::ToTopsatenScalar(max.value(), result_dtype)
      : gcu::ToTopsatenScalar(gcu::DtypeHighest(result_dtype), result_dtype);

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
"""
    + _OUT_DEVICE_GUARD
    + _out_cast_check(
        "c10::promoteTypes(at::result_type(self, mat1), mat2.scalar_type())",
        define=True,
    )
    + """\
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
  if (out.scalar_type() != result_dtype || !out.is_contiguous() ||
      out.device() != self.device()) {{
    out.copy_(at::{at_op}(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha));
    return out;
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

# arange is the one creation op that takes scalars instead of a shape, and
# topsaten's entry point takes no size at all: it fills an output tensor that the
# caller has already sized. So the length has to be computed here, and it has to
# be ATen's rule or the two disagree on the tail of a non-divisible range --
# at::native::compute_arange_size is that rule, shared with the CPU and CUDA
# kernels, and it also carries ATen's bound checks (zero step, sign mismatch,
# non-finite bounds), so a caller sees the same errors on both devices.
#
# The three overloads differ only in how many scalars they take, so the template
# is parameterised per overload ({params}, {bind}, {dtype_infer}) rather than
# triplicated; the body always works with locals start/end/step.
#
# dtype inference mirrors the generated FlagGems Python kernels (see
# ArangeKernelPython in csrc/aten/generated/flaggems_python_kernels.cc): an
# explicitly requested dtype wins, otherwise any floating-point scalar makes the
# result the default float type and all-integral scalars make it int64.
#
# On the vendor path a fractional step does not land on the CPU kernel's bits:
# measured on S60, the vendor evaluates `(T)start + i * (T)step` with one
# rounding to the output dtype T, where the CPU kernel's formulation is not
# length-invariant at all -- `torch.arange(0., 1., 0.1)` and
# `torch.arange(0., 1000., 0.1)` give different values for the *same*
# mathematical element (0.8999999761581421 against 0.9000000357627869 at index
# 9), so there is no single set of bits to match. The vendor's own error is
# bounded: casting the step to T perturbs it by at most 2**-24 relative, so over
# i elements the result is off by at most about one ULP of the largest value in
# the range. Integer-valued steps -- which is every arange the Qwen-Image graph
# issues, `arange(0, dim, 2)` and `arange(4096)` included -- are bit-exact,
# because nothing rounds.
#
# int64 -- which is what `torch.arange(n)` infers and what the Qwen-Image text
# path asks for explicitly -- cannot use the vendor kernel at all
# (TopsatenSupportsDtype rejects kLong), so it keeps a host round trip. That is
# measured at 0.075 ms/call over a few hundred elements, against the full
# activation a cpu_fallback of a compute op would copy.
T_ARANGE = """\
at::Tensor {kernel}(
    {params},
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {{
  {bind}
  auto out_dtype = dtype.value_or({dtype_infer}
      ? at::typeMetaToScalarType(at::get_default_dtype())
      : at::kLong);
  // An absent (or index-less) device means the current one, exactly as
  // csrc/aten/backends/flagos/python_op_caller.cc resolves it for the other
  // factories: naming index 0 here would allocate on device 0 while the op runs
  // on the current device, which on GCU is a silent cross-device write.
  auto target = (device.has_value() && device->has_index())
      ? *device
      : at::Device(at::kPrivateUse1, static_cast<int>(c10::flagos::CurrentDevice()));
  TORCH_CHECK(
      target.is_privateuseone(), "arange on GCU requires a flagos device, got ", target);
  // Same contract as at::empty on a device (csrc/aten/empty.cc).
  TORCH_CHECK(
      !pin_memory.value_or(false), "Pin memory can only be on CPU");

  auto length = out_dtype == at::kLong
      ? at::native::compute_arange_size<int64_t>(start, end, step)
      : at::native::compute_arange_size<double>(start, end, step);

  if (!gcu::TopsatenArangeDtype(out_dtype) ||
      layout.value_or(at::kStrided) != at::kStrided || length == 0) {{
    // Built with at::kCPU and the caller's layout, so the call lands on the CPU
    // kernel (which re-raises for a non-strided layout the same way ATen does)
    // instead of re-entering this one.
    auto host = at::arange(start, end, step, out_dtype, layout, at::kCPU, false);
    return host.to(target);
  }}

  auto out = at::empty(
      {{length}},
      at::TensorOptions().dtype(out_dtype).device(target).pinned_memory(false));
  auto t_start = gcu::ToTopsatenScalar(start, out_dtype);
  auto t_end = gcu::ToTopsatenScalar(end, out_dtype);
  auto t_step = gcu::ToTopsatenScalar(step, out_dtype);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, out, t_out.get(), t_start, t_end, t_step,
      gcu::ToTopsatenDataType(out_dtype), TOPSATEN_LAYOUT_STRIDED, false);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# The three arange overloads, as T_ARANGE's per-overload parameters: the scalar
# parameters they declare, the locals that bind the ones they do not take, and
# ATen's dtype-inference predicate for that overload (mirrored from the FlagGems
# Python kernels -- see T_ARANGE).
ARANGE_OVERLOADS = {
    "arange": (
        "const at::Scalar& end",
        "at::Scalar start(0), step(1);",
        "end.isFloatingPoint()",
    ),
    "arange.start": (
        "const at::Scalar& start, const at::Scalar& end",
        "at::Scalar step(1);",
        "start.isFloatingPoint() || end.isFloatingPoint()",
    ),
    "arange.start_step": (
        "const at::Scalar& start, const at::Scalar& end, const at::Scalar& step",
        "",
        "start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()",
    ),
}

# zero_ / fill_ / masked_fill_ are the in-place writes the rest of the stack is
# built out of, and the two ATen factories decompose straight into them: on a
# non-CPU device torch.zeros / torch.ones / torch.full are empty(...) followed by
# zero_() or fill_(...), so a cpu_fallback here turns every factory call in a
# model into a device->host->device round trip -- and it copies a tensor that is
# still uninitialized at that point. A Qwen-Image denoise step issues ~1150 of
# them. topsatenZero / topsatenFill_ / topsatenMasked_fill write through the
# existing buffer, so the cost collapses to one launch.
#
# These are in-place, so unlike the out-of-place categories there is no output
# tensor to allocate, and the unsupported-dtype path has nowhere to put a host
# result except back into self. self.cpu() is a plain CPU tensor, so the method
# call on it runs the CPU kernel instead of re-entering this one.
#
# numel() == 0 short-circuits: topsaten rejects rank-0 and empty shapes, and the
# host path would round-trip nothing.
T_ZERO_INPLACE = """\
at::Tensor& {kernel}(at::Tensor& self) {{
  if (self.numel() == 0) {{
    return self;
  }}
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !self.is_contiguous()) {{
    auto host = self.cpu();
    host.zero_();
    self.copy_(host);
    return self;
  }}
  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD({tops}, self, t_self.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_FILL_INPLACE_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Scalar& value) {{
  if (self.numel() == 0) {{
    return self;
  }}
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !self.is_contiguous()) {{
    auto host = self.cpu();
    host.fill_(value);
    self.copy_(host);
    return self;
  }}
  auto t_value = gcu::ToTopsatenScalar(value, self.scalar_type());

  gcu::TopsatenTensorWrapper t_self(self);
  EXEC_TOPSATEN_CMD({tops}, self, t_self.get(), t_value);
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# The Tensor overload is how a Python number reaches the op (PyTorch wraps the
# operand into a 0-dim CPU tensor), so `value` may be a host tensor and is moved
# onto self's device and dtype first. topsaten has no rank-0 support, which is
# also why the 1-element reshape is not optional.
T_FILL_INPLACE_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& value) {{
  if (self.numel() == 0) {{
    return self;
  }}
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !self.is_contiguous() || value.numel() != 1) {{
    auto host = self.cpu();
    host.fill_(value.cpu());
    self.copy_(host);
    return self;
  }}
  auto value_c = value.to(self.device(), self.scalar_type()).reshape({{1}});

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_value(value_c);
  EXEC_TOPSATEN_CMD({tops}, self, t_self.get(), t_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_MASKED_FILL_INPLACE_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {{
  if (self.numel() == 0) {{
    return self;
  }}
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !self.is_contiguous() || mask.scalar_type() != at::kBool) {{
    auto host = self.cpu();
    host.masked_fill_(mask.cpu(), value);
    self.copy_(host);
    return self;
  }}
  auto mask_c = mask.to(self.device()).contiguous();
  auto t_value = gcu::ToTopsatenScalar(value, self.scalar_type());

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_mask(mask_c);
  EXEC_TOPSATEN_CMD({tops}, self, t_self.get(), t_self.get(), t_mask.get(), t_value);
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_MASKED_FILL_INPLACE_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {{
  if (self.numel() == 0) {{
    return self;
  }}
  if (!gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !self.is_contiguous() || mask.scalar_type() != at::kBool ||
      value.numel() != 1) {{
    auto host = self.cpu();
    host.masked_fill_(mask.cpu(), value.cpu());
    self.copy_(host);
    return self;
  }}
  auto mask_c = mask.to(self.device()).contiguous();
  auto value_c = value.to(self.device(), self.scalar_type()).reshape({{1}});

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_mask(mask_c);
  gcu::TopsatenTensorWrapper t_value(value_c);
  EXEC_TOPSATEN_CMD({tops}, self, t_self.get(), t_self.get(), t_mask.get(), t_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# --------------------------------------------------------------------------
# Indexed reads and writes, plus the embedding gradient that pairs with them.
#
# These are what is left of the cpu_fallback census. The Qwen-Image census is
# inference-only -- attention masks, the RoPE views and the image encoder's
# arange -- and its six families are all covered above. A *training* step
# reaches a different set: `index_select` and `embedding_dense_backward` through
# the embedding backward, `index_fill_` through every in-place index write, and
# `nonzero_static` through the static-shape nonzero a compiled graph uses in
# place of a sync. Measured over a two-step Adam run (foreach on and off) plus
# one explicit index_fill_ pair: 4 index_select, 4 embedding_dense_backward,
# 2 index_fill_ and 1 nonzero_static calls reach cpu_fallback, with every hot
# op from the inference census already native.
#
# topsaten has all three: topsatenIndexSelect, topsatenIndexFill (scalar and
# tensor value) and topsatenEmbeddingDenseBackward. nonzero_static is *not*
# here -- see its entry in OPS.
#
# The out= overloads delegate to the out-of-place kernel and copy into `out`
# rather than growing a second copy of every guard: the callers that reach them
# are rare (the census above found none), and an error raised inside the fill
# then leaves `out` untouched, which is what ATen does. The cost is one extra
# device tensor and one device copy on a spelling that is already paying for an
# output allocation.
# --------------------------------------------------------------------------
T_INDEX_SELECT = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index) {{
  // ATen raises IndexError for an out-of-range dim and for an index that is not
  // a vector or a scalar, and its messages are the useful ones, so both cases
  // are the host path's job. A 0-dim index is a legal one-element vector
  // (measured: shape (1, 4) for `index_select(t, 0, tensor(1))` on a (3, 4)),
  // which is why the flatten below is a reshape and not a rank check.
  const bool dim_ok =
      self.dim() != 0 && dim >= -self.dim() && dim < self.dim();
  if (!dim_ok || index.dim() > 1 ||
      !gcu::TopsatenSupportsDtype(self.scalar_type()) ||
      !gcu::TopsatenIndexDtype(index.scalar_type())) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}
  const int64_t dim_n = dim < 0 ? dim + self.dim() : dim;
  auto self_c = self.contiguous();
  auto index_c = index.to(self.device()).contiguous().reshape({{-1}});
  std::vector<int64_t> out_shape(self_c.sizes().vec());
  out_shape[dim_n] = index_c.size(0);
  auto out = at::empty(out_shape, self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  if (!gcu::TopsatenIndexSelectFits({{self_c, index_c, out}})) {{
    return at::{at_op}(self.cpu(), dim, index.cpu()).to(self.device());
  }}

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_index(index_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, out, t_out.get(), t_self.get(), dim_n, t_index.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# ATen refuses an `out` whose dtype is not the result's, with this message.
# Note that index_select is the one out= op here that does *not* use the generic
# "Expected out tensor to have dtype ..." wording, and that it does not care
# about out's contiguity -- a strided out is legal and is copied into.
T_INDEX_SELECT_OUT = (
    """\
at::Tensor& {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out) {{
"""
    + _OUT_DEVICE_GUARD
    + """\
  TORCH_CHECK(out.scalar_type() == self.scalar_type(),
              "index_select(): self and result must have the same scalar type");
  auto result = IndexSelectKernelGcu(self, dim, index);
  if (!out.sizes().equals(result.sizes())) {{
    out.resize_(result.sizes());
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  out.copy_(result);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

# index_fill_ is the primitive: ATen documents `index_fill.int_Scalar` as
# `self.clone(at::MemoryFormat::Preserve).index_fill_(...)`, and the clone is
# what lets the out-of-place and out= spellings below delegate here instead of
# carrying a second copy of the guards -- including the host path, which writes
# through self and so needs no out-specific handling.
#
# The index dtype is not negotiable and cannot be folded into the empty-index
# short circuit: ATen raises `index_fill_(): Expected dtype int64 for index.`
# for an int32 index *even when the index is empty* (measured), so the dtype
# test runs first and an empty index is only a no-op once it has passed.
#
# The vendor wants the opposite dtype from ATen here, and its bounds check is
# unsafe, so the index goes through `TopsatenIndexFillIndex`: the narrowing also
# validates, wrapping a negative index and refusing an out-of-range one. That
# helper returns an undefined tensor for a call the vendor must not be handed,
# which is folded into the same host path as the operand checks above -- one
# condition, one message, ATen's own.
T_INDEX_FILL_INPLACE_SCALAR = """\
at::Tensor& {kernel}(at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value) {{
  // Each way ATen rejects this call has a message worth keeping, and only the
  // host path raises it, so the checks come first: an out-of-range dim, an
  // index that is not a vector or a scalar, and an index that is not int64 --
  // the last applied even to an empty index (measured), which is why it cannot
  // be folded into the empty-index shortcut below.
  const bool dim_ok =
      self.dim() != 0 && dim >= -self.dim() && dim < self.dim();
  const bool index_ok = index.dim() <= 1 && index.scalar_type() == at::kLong;
  const bool self_ok =
      self.numel() != 0 && gcu::TopsatenSupportsDtype(self.scalar_type()) &&
      self.is_contiguous();
  const int64_t dim_n = dim_ok ? (dim < 0 ? dim + self.dim() : dim) : 0;
  at::Tensor index_c;
  if (dim_ok && index_ok && self_ok) {{
    index_c = gcu::TopsatenIndexFillIndex(index, self.size(dim_n));
  }}
  if (!index_c.defined()) {{
    auto host = self.cpu();
    host.index_fill_(dim, index.cpu(), value);
    self.copy_(host);
    return self;
  }}
  // Only reachable with dim and index both valid, so an empty index is the
  // no-op ATen makes it, and costs nothing here.
  if (index.numel() == 0) {{
    return self;
  }}
  auto t_value = gcu::ToTopsatenScalar(value, self.scalar_type());

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_index(index_c);
  EXEC_TOPSATEN_CMD({tops}, self, t_self.get(), t_self.get(), dim_n, t_index.get(), t_value);
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# The Tensor overload is how a Python number reaches this op when it is passed as
# a tensor. The value must be 0-dimensional: ATen takes nothing else, not even a
# one-element 1-D tensor (measured: `index_fill_ only supports a 0-dimensional
# value tensor, but got tensor with 1 dimension(s).`), so the rank test is a
# guard rather than a broadcast case, and a 1-D value takes the host path where
# ATen raises that message. The value may be a host tensor and is moved onto
# self's device and dtype first; the 0-dim-to-(1,) reshape afterwards is not
# optional because topsaten has no rank-0 support. index_fill_ is also the one
# index op whose tensor value topsaten takes directly, so no scalar staging is
# needed.
T_INDEX_FILL_INPLACE_TENSOR = """\
at::Tensor& {kernel}(at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& value) {{
  const bool dim_ok =
      self.dim() != 0 && dim >= -self.dim() && dim < self.dim();
  const bool index_ok = index.dim() <= 1 && index.scalar_type() == at::kLong;
  const bool self_ok =
      self.numel() != 0 && gcu::TopsatenSupportsDtype(self.scalar_type()) &&
      self.is_contiguous();
  const int64_t dim_n = dim_ok ? (dim < 0 ? dim + self.dim() : dim) : 0;
  at::Tensor index_c;
  if (dim_ok && index_ok && self_ok && value.dim() == 0) {{
    index_c = gcu::TopsatenIndexFillIndex(index, self.size(dim_n));
  }}
  if (!index_c.defined()) {{
    auto host = self.cpu();
    host.index_fill_(dim, index.cpu(), value.cpu());
    self.copy_(host);
    return self;
  }}
  if (index.numel() == 0) {{
    return self;
  }}
  auto value_c = value.to(self.device(), self.scalar_type()).reshape({{1}});

  gcu::TopsatenTensorWrapper t_self(self);
  gcu::TopsatenTensorWrapper t_index(index_c);
  gcu::TopsatenTensorWrapper t_value(value_c);
  EXEC_TOPSATEN_CMD({tops}, self, t_self.get(), t_self.get(), dim_n, t_index.get(), t_value.get());
  return self;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# The out-of-place index_fill is `self.clone(at::MemoryFormat::Preserve)
# .index_fill_(...)` in ATen's own composite, so the clone is the whole body:
# the in-place kernel above already carries the guards, the host path and the
# dim/dtype errors, and duplicating them here would be two places to keep in
# step.
T_INDEX_FILL_SCALAR = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value) {{
  auto result = self.clone();
  IndexFillInplaceIntScalarKernelGcu(result, dim, index, value);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_INDEX_FILL_TENSOR = """\
at::Tensor {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& value) {{
  auto result = self.clone();
  IndexFillInplaceIntTensorKernelGcu(result, dim, index, value);
  return result;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# index_fill's out= writes through the caller's tensor, which is where it
# differs from the clone above: nothing is allocated, so a rejected dtype has to
# be caught before the fill runs. The message is ATen's -- index_fill is one of
# the out= ops that refuse a widening dtype outright rather than casting into it
# (measured: out=double against a float result raises).
#
# The resize has to be followed by a write of the *whole* result, not just by the
# fill: an out that was grown from empty, or resized away from a different shape,
# holds no copy of `self` in the elements the fill does not touch, and ATen's own
# out= leaves those elements equal to `self`'s (measured on the host: out=(2,3,5)
# and out=(24,) both come back holding the filled result in full). So the result
# comes from the out-of-place kernel -- one clone and one fill -- and is copied in
# whole, which is also what the index_select and embedding_dense_backward out=
# kernels below do.
T_INDEX_FILL_SCALAR_OUT = (
    """\
at::Tensor& {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value, at::Tensor& out) {{
"""
    + _OUT_DEVICE_GUARD
    + """\
  TORCH_CHECK(out.scalar_type() == self.scalar_type(),
              "Expected out tensor to have dtype ", self.scalar_type(),
              ", but got ", out.scalar_type(), " instead");
  auto result = IndexFillIntScalarKernelGcu(self, dim, index, value);
  if (!out.sizes().equals(result.sizes())) {{
    out.resize_(result.sizes());
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  out.copy_(result);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

T_INDEX_FILL_TENSOR_OUT = (
    """\
at::Tensor& {kernel}(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& value, at::Tensor& out) {{
"""
    + _OUT_DEVICE_GUARD
    + """\
  TORCH_CHECK(out.scalar_type() == self.scalar_type(),
              "Expected out tensor to have dtype ", self.scalar_type(),
              ", but got ", out.scalar_type(), " instead");
  auto result = IndexFillIntTensorKernelGcu(self, dim, index, value);
  if (!out.sizes().equals(result.sizes())) {{
    out.resize_(result.sizes());
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  out.copy_(result);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

# embedding_dense_backward: the gradient of an embedding lookup, one row of
# [num_weights, embed_dim] accumulated per (index, grad row) pair with the
# padded row skipped. topsatenEmbeddingDenseBackward does the whole op --
# zeroing the buffer, the accumulation, and the 1/frequency scaling when
# scale_grad_by_freq is set -- so this kernel is operand preparation.
#
# Two operand rules are the vendor's rather than ATen's. The index has to be
# narrowed to int32 for it -- see `TopsatenEmbeddingIndex`, which also validates
# it, because a plain cast would fold an out-of-range index back into range. And
# the buffer must be shaped [numel, embed_dim] while ATen's grad may carry any
# leading dims -- the reshape below is a view whenever grad is contiguous, which
# it is after the contiguous() call.
T_EMBEDDING_DENSE_BACKWARD = """\
at::Tensor {kernel}(const at::Tensor& grad, const at::Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {{
  // A padding_idx outside [-1, num_weights) has no vendor equivalent to lean
  // on. ATen compares it against the *flat* index, which is never negative, so
  // a value below -1 skips no row at all (measured: -4 and -1 both leave a
  // 6-row table untouched) and a value at or above num_weights matches no row
  // either, while topsaten's parameter names the row to skip. -1 is the "no
  // padding" sentinel both sides share -- it is topsaten's documented default.
  if (grad.dim() == 0 || num_weights <= 0 || padding_idx < -1 ||
      padding_idx >= num_weights ||
      !gcu::TopsatenEmbeddingDenseBackwardDtype(grad.scalar_type()) ||
      (indices.scalar_type() != at::kLong && indices.scalar_type() != at::kInt)) {{
    return at::{at_op}(grad.cpu(), indices.cpu(), num_weights, padding_idx, scale_grad_by_freq)
        .to(grad.device());
  }}
  auto grad_c = grad.contiguous();
  const int64_t embed_dim = grad_c.size(-1);
  auto indices_i32 = gcu::TopsatenEmbeddingIndex(indices, num_weights, grad.device());
  // topsaten rejects a mismatch with a status error, ATen raises its own
  // message for it, and ATen's is the one a caller can act on.
  const int64_t num_indices = indices_i32.defined() ? indices_i32.numel() : -1;
  if (!indices_i32.defined() || grad_c.numel() != num_indices * embed_dim) {{
    return at::{at_op}(grad.cpu(), indices.cpu(), num_weights, padding_idx, scale_grad_by_freq)
        .to(grad.device());
  }}
  auto out = at::empty({{num_weights, embed_dim}}, grad_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  if (num_indices == 0) {{
    // topsatenEmbeddingDenseBackward zeroes its buffer as step 1 of its own
    // flow, so this is the one case that has to be zeroed here: no index means
    // no row is written, and ATen returns zeros.
    out.zero_();
    return out;
  }}
  auto grad_2d = grad_c.reshape({{-1, embed_dim}});
  auto indices_2d = indices_i32.reshape({{-1, 1}});

  gcu::TopsatenTensorWrapper t_grad(grad_2d);
  gcu::TopsatenTensorWrapper t_indices(indices_2d);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, grad, t_out.get(), t_grad.get(), t_indices.get(), num_weights, padding_idx, scale_grad_by_freq);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_EMBEDDING_DENSE_BACKWARD_OUT = """\
// The same guard _OUT_DEVICE_GUARD installs, spelled out because that snippet
// names `self` and this kernel's first parameter is `grad`. See its comment for
// why the resize below needs the device selected.
at::Tensor& {kernel}(const at::Tensor& grad, const at::Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, at::Tensor& out) {{
  gcu::TopsDeviceGuard out_guard(grad);
  TORCH_CHECK(out.scalar_type() == grad.scalar_type(),
              "Expected out tensor to have dtype ", grad.scalar_type(),
              ", but got ", out.scalar_type(), " instead");
  auto result = EmbeddingDenseBackwardKernelGcu(
      grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  if (!out.sizes().equals(result.sizes())) {{
    out.resize_(result.sizes());
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  out.copy_(result);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# _upsample_nearest_exact2d: the nearest-neighbour resize the Qwen-Image VAE's
# upsampling blocks use.
#
# Both scale arguments must be present. They are what topsaten interpolates
# with -- its `size` parameter only names the output's spatial extents, which
# the output tensor already carries in its shape -- and the callers that reach
# this op always have them: `nn.Upsample(scale_factor=2.0, mode="nearest-exact")`
# goes through F.interpolate, which derives output_size from the scale factor
# and passes both along. An output_size-only call (F.interpolate(size=...)) has
# no scale to hand over and takes the host path. A rank-3 input is declined for
# the same reason: it says nothing about how the vendor maps the batch-free
# form, and every caller in the stack passes NCHW.
T_UPSAMPLE_NEAREST_EXACT2D = """\
at::Tensor {kernel}(const at::Tensor& self, at::IntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {{
  if (self.dim() != 4 || output_size.size() != 2 || !scales_h.has_value() ||
      !scales_w.has_value() ||
      !gcu::TopsatenSupportsDtype(self.scalar_type())) {{
    return at::{at_op}(self.cpu(), output_size, scales_h, scales_w).to(self.device());
  }}
  auto self_c = self.contiguous();
  auto out = at::empty(
      {{self_c.size(0), self_c.size(1), output_size[0], output_size[1]}},
      self_c.options());
  if (out.numel() == 0) {{
    return out;
  }}
  auto t_scale_h = gcu::ToTopsatenScalar(at::Scalar(*scales_h), at::kDouble);
  auto t_scale_w = gcu::ToTopsatenScalar(at::Scalar(*scales_w), at::kDouble);
  gcu::TopsatenSizeWrapper t_size(output_size);

  gcu::TopsatenTensorWrapper t_self(self_c);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, self, t_out.get(), t_self.get(), t_size.get(), t_scale_h, t_scale_w);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

T_UPSAMPLE_NEAREST_EXACT2D_OUT = (
    """\
at::Tensor& {kernel}(const at::Tensor& self, at::IntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor& out) {{
"""
    + _OUT_DEVICE_GUARD
    + """\
  TORCH_CHECK(out.scalar_type() == self.scalar_type(),
              "Expected out tensor to have dtype ", self.scalar_type(),
              ", but got ", out.scalar_type(), " instead");
  auto result =
      PrivUpsampleNearestExact2dKernelGcu(self, output_size, scales_h, scales_w);
  if (!out.sizes().equals(result.sizes())) {{
    out.resize_(result.sizes());
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  out.copy_(result);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""
)

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

# where.self is the busiest of the five ops this file's last census still
# reached cpu_fallback on: the Qwen-Image denoise loop issues 56 of them per
# step. Three of the call sites are the rotary shift selection
# (transformer_qwenimage.py:709-711, `where(index_expanded == 0, shift_0, shift_1)`)
# and two build position ids out of the encoder mask (197, 199).
#
# topsatenWhere is also the one entry point here that takes every dtype it can
# represent, f64 and i64 included, so there is no operand guard at all -- only
# the two things that are measured to fail: a non-PRED condition (a u8 condition
# is BAD_PARAM at the vendor, while ATen casts it and proceeds, so that case is a
# host call rather than an error) and a dtype with no topsaten mapping, which
# would raise from the wrapper instead of falling back.
#
# The dtype promotion mirrors _BINARY_PROLOGUE, and for the same reason: where
# promotes its two value operands, and the vendor's op does not. The condition is
# moved to self's device for the same reason it is everywhere else -- a Python
# boolean mask arrives as a host tensor, and a host pointer in a topsaten
# descriptor fails in the driver rather than returning wrong values.
T_WHERE_SELF = """\
at::Tensor {kernel}(
    const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {{
  auto result_dtype = at::result_type(self, other);
  // The operands are read on `dev`, not on `self.device()`. `self` is a 0-dim
  // host tensor whenever a composite spells a scalar that way -- which is how
  // ATen's eager `_safe_softmax` builds its zero -- and the earlier spelling of
  // this template then ran the whole call, including the vendor call, against
  // the host. See TopsatenComputeDevice.
  auto dev = gcu::TopsatenComputeDevice(condition, self, other);
  // ATen's own `where` takes a bool condition, and a byte one by casting it;
  // every other condition dtype it rejects outright, which is the error the host
  // path below reproduces verbatim. A byte condition is therefore not a reason to
  // leave the card: it takes the same `.to(dev, at::kBool)` that the vendor's
  // PRED operand needs.
  if ((condition.scalar_type() != at::kBool &&
       condition.scalar_type() != at::kByte) ||
      !gcu::TopsatenWhereDtype(result_dtype) || dev.is_cpu()) {{
    return at::{at_op}(condition.cpu(), self.cpu(), other.cpu()).to(dev);
  }}
  auto cond_c = condition.to(dev, at::kBool);
  auto self_c = self.to(dev, result_dtype);
  auto other_c = other.to(dev, result_dtype);
  // All three operands are expanded, not just the two values: the vendor does not
  // broadcast, and ATen broadcasts the condition against the values' common
  // shape, which is not the same thing when the condition is the wider one.
  auto out_shape = at::infer_size(
      at::infer_size(cond_c.sizes(), self_c.sizes()), other_c.sizes());
  auto cond_b = cond_c.expand(out_shape).contiguous();
  auto self_b = self_c.expand(out_shape).contiguous();
  auto other_b = other_c.expand(out_shape).contiguous();
  auto out = at::empty(out_shape, self_b.options());
  if (out.numel() == 0) {{
    return out;
  }}

  gcu::TopsatenTensorWrapper t_cond(cond_b);
  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, out, t_out.get(), t_cond.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# `where.self_out` is the same operation through the out= contract, and it is
# the single largest remaining cost in the Qwen-Image denoise step. ATen's eager
# `_safe_softmax` -- which the SDPA math path runs once per attention layer on
# this platform -- is a CompositeExplicitAutograd C++ kernel whose last op is
# exactly this overload, so the whole op inherits FlagGems' cost here.
#
# Measured on the S60 (card 1) at (1, 24, 4114, 4114) fp32 with a
# (1, 24, 4114, 1) bool condition, drain on both sides of every call:
#
#   flag_gems.where_self_out(cond, tensor(0), x, out=o)       1098.6 ms
#   flag_gems.where_self_out(cond, full-size 0, x, out=o)       69.8 ms
#   gcu                     where.self(cond, tensor(0), x)      23.7 ms
#
# The 0-dim value operand is what falls off the cliff, not its device: the same
# call measures 1105.2 ms with the scalar on the host and 1098.6 ms with it on
# the card, while a full-size second operand is 15x faster at the same shape.
# Summed over the composite, one `_safe_softmax` costs 1135 ms through FlagGems
# and 37.8 ms once this op is taken out of it, and at that shape the op costs the
# same whether it runs alone or after the other four -- so this is the whole of
# the gap, not an amplifier of it.
#
# The body is the out-of-place template plus the out= contract T_BINARY_OUT
# spells out: the kernel sizes `out`, and everything the vendor cannot take
# lands on the host path *in* `out`. Two orderings differ from T_BINARY_OUT on
# purpose. That template resizes before testing `out`'s device, but `resize_`
# takes no device argument and allocates out of the *current* device's pool, so a
# mismatched `out` would be grown out of the wrong card's pool while still
# reporting its own. Testing the device first and copying into `out` afterwards
# keeps a caller's mistake on the host path instead of in the allocator -- which
# is also why the device guard here is installed on `out` rather than on `self`:
# `out` is both the tensor being grown and the one that names the device the work
# belongs on, and `self` is a host scalar for the composite this kernel exists
# for.
T_WHERE_SELF_OUT = """\
at::Tensor& {kernel}(
    const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other,
    at::Tensor& out) {{
  gcu::TopsDeviceGuard out_guard(out);
  auto result_dtype = at::result_type(self, other);
  TORCH_CHECK(
      c10::canCast(result_dtype, out.scalar_type()),
      "result type ",
      result_dtype,
      " can't be cast to the desired output type ",
      out.scalar_type());
  // Compute where `out` lives, which is also the device the guard just selected.
  // `self` is a 0-dim host tensor for the composite that this kernel exists for,
  // so its own device says nothing about where the work belongs; see
  // TopsatenComputeDevice.
  auto dev = gcu::TopsatenComputeDevice(out, self, other);
  // The host cases are the out-of-place template's, plus `out`'s: a condition
  // dtype ATen itself rejects, a dtype with no topsaten mapping that would raise
  // out of the wrapper, and a destination topsaten would write through with the
  // wrong width, in the wrong layout, or on another device. Each of those is a
  // host call rather than a descriptor.
  if ((condition.scalar_type() != at::kBool &&
       condition.scalar_type() != at::kByte) ||
      !gcu::TopsatenWhereDtype(result_dtype) || out.device() != dev ||
      out.scalar_type() != result_dtype || !out.is_contiguous()) {{
    auto host = at::{at_op}(condition.cpu(), self.cpu(), other.cpu());
    if (!out.sizes().equals(host.sizes())) {{
      out.resize_(host.sizes());
    }}
    out.copy_(host);
    return out;
  }}
  // All three operands are expanded, not just the two values: the vendor does not
  // broadcast, and ATen broadcasts the condition against the values' common
  // shape, which is not the same thing when the condition is the wider one.
  auto out_shape = at::infer_size(
      at::infer_size(condition.sizes(), self.sizes()), other.sizes());
  if (!out.sizes().equals(out_shape)) {{
    out.resize_(out_shape);
  }}
  if (out.numel() == 0) {{
    return out;
  }}
  auto cond_b = condition.to(dev, at::kBool).expand(out_shape).contiguous();
  auto self_b = self.to(dev, result_dtype).expand(out_shape).contiguous();
  auto other_b = other.to(dev, result_dtype).expand(out_shape).contiguous();

  gcu::TopsatenTensorWrapper t_cond(cond_b);
  gcu::TopsatenTensorWrapper t_self(self_b);
  gcu::TopsatenTensorWrapper t_other(other_b);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD(
      {tops}, out, t_out.get(), t_cond.get(), t_self.get(), t_other.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# The whole-tensor `all` -- `torch.all(t)`, no dim -- which is the spelling
# pipeline_qwenimage.py:265 uses to ask whether the prompt-embedding mask has any
# zero in it. The dim/dims overloads are a different category and are already on
# another backend, so this template is reached for one schema only.
#
# The output is a 0-dim bool, and the vendor needs a (1,) PRED destination: a
# bare rank-0 descriptor aborts it. TopsatenTensorWrapper already rewrites a
# rank-0 tensor to a 1-element vector, so the 0-dim result is passed straight
# through and no separate buffer or reshape is needed -- `at::Tensor::reshape`
# and `squeeze` are both dispatcher calls that GCU does not carry a kernel for.
#
# Both empty inputs and a PRED operand are measured working at the vendor, with
# the same vacuous True ATen gives, so neither needs a special case. An empty
# input is still routed to the host: an empty device tensor's data pointer is not
# worth handing to the driver for a result that costs nothing to compute here.
T_ALL_WHOLE = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  auto operand = gcu::TopsatenAllOperand(self);
  if (!operand.defined() || self.numel() == 0) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  // ATen's dtype inference is not uniformly bool here: `all` on a uint8
  // operand is uint8, on every other dtype bool. That is the *meta* function,
  // not a CPU kernel quirk, so a backend that answers bool is the deviant one.
  // The vendor writes the correct 0/1 into either descriptor (measured on card
  // 4), so the declared dtype is the one to allocate.
  auto out = at::empty(
      {{}}, self.options().dtype(
                self.scalar_type() == at::kByte ? at::kByte : at::kBool));

  gcu::TopsatenTensorWrapper t_self(operand);
  gcu::TopsatenTensorWrapper t_out(out);
  EXEC_TOPSATEN_CMD({tops}, out, t_out.get(), t_self.get());
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# index.Tensor is ATen's advanced indexing, and it is one dispatch for several
# different operations: a single rank-1 index tensor (which is index_select on
# dim 0), several index tensors at once, a bool mask, and a 0-dim index. The
# vendor's kernel implements the first of those, so the template recognises it
# and delegates to the index_select kernel this file already generates; every
# other spelling takes the host path with ATen's own semantics.
#
# The recognition is the whole content of the template. A rank-1 index is
# index_select on dim 0 only when it is the *only* index (measured against
# ATen: `t[[i]] == index_select(t, 0, i)` is True), and only when every
# coordinate is non-negative -- ATen wraps a negative index into the tensor,
# while the vendor resolves it outside, which is what `TopsatenIndexNonNegative`
# reads back to check. A 0-dim index is excluded by the rank test and is not an
# oversight: `t[tensor(1)]` drops the indexed dimension (shape (4,) for a (3,4))
# where index_select keeps it ((1,4)), so it is a different operation. A bool
# index is excluded by the dtype test and is a mask, not a coordinate list.
T_INDEX_TENSOR = """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices) {{
  if (indices.size() == 1 && indices[0].has_value() &&
      indices[0]->defined()) {{
    const at::Tensor& index = indices[0].value();
    if (index.dim() == 1 && gcu::TopsatenIndexDtype(index.scalar_type()) &&
        gcu::TopsatenIndexNonNegative(index)) {{
      // The delegate carries its own guard: a 0-dim self, an unsupported
      // operand dtype or a non-contiguous operand all fall out of it onto the
      // same host path this template uses below, so there is nothing to test
      // here that index_select does not already test.
      return IndexSelectKernelGcu(self, 0, index);
    }}
  }}
  // The indices travel with the operand: ATen's CPU advanced indexing runs on
  // whichever device holds the data, and a device index would otherwise reach it
  // as a device pointer. This is the same marshalling register.cc's
  // WrapperIndexPut_ does.
  c10::List<::std::optional<at::Tensor>> indices_cpu;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {{
    auto opt = indices.get(i);
    if (opt.has_value() && opt->defined()) {{
      indices_cpu.push_back(opt->cpu());
    }} else {{
      indices_cpu.push_back(::std::nullopt);
    }}
  }}
  return at::{at_op}(self.cpu(), indices_cpu).to(self.device());
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# nonzero / nonzero_static share one vendor chain, and it is three calls rather
# than the one an op of this shape would expect: the output's first dimension is
# the *count* of nonzeros, so nothing about the answer is known before the data
# has been read once.
#
#   1. topsatenCountNonzero writes the count into a device int32 -- a device
#      buffer, not a host one: the vendor writes it as a device value, and the
#      status alone is not enough to trust it, because on a dtype it does not
#      support the call returns SUCCESS and a wrong count.
#   2. topsatenNonzero writes the coordinates as int32, and only ever into an
#      (nnz, rank) output -- a partial write is refused outright (see
#      TopsatenRowView).
#   3. topsatenTo widens int32 to int64, since ATen's result is int64 and the
#      vendor writes int32 coordinates.
#
# Steps 2 and 3 are the same for both ops; what differs is the output size, which
# is why there are two templates rather than one with a flag.
T_NONZERO = """\
at::Tensor {kernel}(const at::Tensor& self) {{
  auto operand = gcu::TopsatenNonzeroOperand(self);
  // A rank-0 operand is not a coverage gap: ATen's result for one is (1, 0), a
  // nonzero number of rows with zero columns, which no topsaten output can
  // describe. An empty operand is declined for the same kind of reason -- there
  // is nothing to read, and a zero-element descriptor is not worth handing to
  // the driver.
  if (!operand.defined() || self.dim() == 0 || self.numel() == 0) {{
    return at::{at_op}(self.cpu()).to(self.device());
  }}
  const int64_t rank = self.dim();
  auto count = at::empty({{1}}, self.options().dtype(at::kInt));
  gcu::TopsatenTensorWrapper t_operand(operand);
  EXEC_TOPSATEN_CMD(
      topsatenCountNonzero, self, reinterpret_cast<int32_t*>(count.data_ptr()),
      t_operand.get());
  const auto count_host = count.cpu();
  const int64_t nnz = count_host.const_data_ptr<int32_t>()[0];

  auto out = at::empty({{nnz, rank}}, self.options().dtype(at::kLong));
  if (nnz == 0) {{
    return out;
  }}
  auto staging = at::empty({{nnz, rank}}, self.options().dtype(at::kInt));
  gcu::TopsatenTensorWrapper t_staging(staging);
  EXEC_TOPSATEN_CMD({tops}, self, t_staging.get(), t_operand.get());

  topsatenDataType_t want = TOPSATEN_DATA_I64;
  gcu::TopsatenTensorWrapper t_dst(out);
  EXEC_TOPSATEN_CMD(
      topsatenTo, self, t_dst.get(), t_staging.get(), want, false, false,
      TOPSATEN_MEMORY_PRESERVE);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

# nonzero_static is `nonzero` with the first dimension fixed by the caller: the
# result is always (size, rank), coordinates first, and `fill_value` wherever
# there were fewer than `size` nonzeros. That padding is why this needs a staging
# buffer of `max(size, nnz)` rows -- the tail has to be filled, and the fill
# lands in the int32 staging because the vendor has no int64 fill.
#
# The whole design is pinned by one measured property of topsatenNonzero: it
# refuses a partial write. Handed an output narrower than the count, it returns
# BAD_PARAM and writes nothing -- (2, 3) over a rank-3 input with four nonzeros
# failed, while (4, 3) and (6, 3) both succeeded. So the coordinates always go
# into an (nnz, rank) description of the staging and the read-back is what gets
# shortened, in either direction. See TopsatenRowView.
T_NONZERO_STATIC = """\
at::Tensor {kernel}(
    const at::Tensor& self, int64_t size, int64_t fill_value) {{
  if (size < 0) {{
    // ATen's own message -- "nonzero_static: 'size' must be an non-negative
    // integer" -- and its own kernel is the only spelling of it that cannot
    // drift from ATen.
    return at::{at_op}(self.cpu(), size, fill_value).to(self.device());
  }}
  auto operand = gcu::TopsatenNonzeroOperand(self);
  if (!operand.defined() || self.dim() == 0 || self.numel() == 0) {{
    return at::{at_op}(self.cpu(), size, fill_value).to(self.device());
  }}
  const int64_t rank = self.dim();
  if (size == 0) {{
    return at::empty({{size, rank}}, self.options().dtype(at::kLong));
  }}
  auto count = at::empty({{1}}, self.options().dtype(at::kInt));
  gcu::TopsatenTensorWrapper t_operand(operand);
  EXEC_TOPSATEN_CMD(
      topsatenCountNonzero, self, reinterpret_cast<int32_t*>(count.data_ptr()),
      t_operand.get());
  const auto count_host = count.cpu();
  const int64_t nnz = count_host.const_data_ptr<int32_t>()[0];

  // Nothing to fill when the coordinates already reach `size`, so the staging is
  // exactly as wide as the call needs.
  const bool fill = size > nnz;
  if (fill && (fill_value < std::numeric_limits<int32_t>::min() ||
               fill_value > std::numeric_limits<int32_t>::max())) {{
    // The padding lands in an int32 staging buffer, so a fill_value that does
    // not survive the round trip would be truncated on the way in and widened
    // back wrong. ATen accepts any int64, so that case takes the host path
    // rather than producing a narrowed tail.
    return at::{at_op}(self.cpu(), size, fill_value).to(self.device());
  }}
  const int64_t rows = fill ? size : nnz;
  auto staging = at::empty({{rows, rank}}, self.options().dtype(at::kInt));
  if (fill) {{
    gcu::TopsatenTensorWrapper t_staging(staging);
    topsatenScalar_t value = gcu::ToTopsatenScalar(at::Scalar(fill_value), at::kInt);
    EXEC_TOPSATEN_CMD(topsatenFill_, staging, t_staging.get(), value);
  }}
  if (nnz > 0) {{
    gcu::TopsatenRowView stage_rows(staging, nnz, rank);
    EXEC_TOPSATEN_CMD({tops}, self, stage_rows.get(), t_operand.get());
  }}

  auto out = at::empty({{size, rank}}, self.options().dtype(at::kLong));
  topsatenDataType_t want = TOPSATEN_DATA_I64;
  gcu::TopsatenTensorWrapper t_dst(out);
  // Equal to the staging when size >= nnz and a prefix of it when size < nnz;
  // both are one copy of the first `size` rows.
  gcu::TopsatenRowView stage_out(staging, size, rank);
  EXEC_TOPSATEN_CMD(
      topsatenTo, self, t_dst.get(), stage_out.get(), want, false, false,
      TOPSATEN_MEMORY_PRESERVE);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kGcu, {kernel})
"""

CATEGORIES = {
    "unary": T_UNARY,
    "binary": T_BINARY,
    "binary_alpha": T_BINARY_ALPHA,
    "binary_out": T_BINARY_OUT,
    "binary_alpha_out": T_BINARY_ALPHA_OUT,
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
    "linalg_vector_norm": T_LINALG_VECTOR_NORM,
    "unary_int": T_UNARY_INT,
    "unary_dims": T_UNARY_DIMS,
    "clamp": T_CLAMP,
    "addmm": T_ADDMM,
    "addmm_out": T_ADDMM_OUT,
    "cat": T_CAT,
    "full_like": T_FULL_LIKE,
    "arange": T_ARANGE,
    "zero_inplace": T_ZERO_INPLACE,
    "fill_inplace_scalar": T_FILL_INPLACE_SCALAR,
    "fill_inplace_tensor": T_FILL_INPLACE_TENSOR,
    "masked_fill_inplace_scalar": T_MASKED_FILL_INPLACE_SCALAR,
    "masked_fill_inplace_tensor": T_MASKED_FILL_INPLACE_TENSOR,
    "index_select": T_INDEX_SELECT,
    "index_select_out": T_INDEX_SELECT_OUT,
    "index_fill_inplace_scalar": T_INDEX_FILL_INPLACE_SCALAR,
    "index_fill_inplace_tensor": T_INDEX_FILL_INPLACE_TENSOR,
    "index_fill_scalar": T_INDEX_FILL_SCALAR,
    "index_fill_tensor": T_INDEX_FILL_TENSOR,
    "index_fill_scalar_out": T_INDEX_FILL_SCALAR_OUT,
    "index_fill_tensor_out": T_INDEX_FILL_TENSOR_OUT,
    "embedding_dense_backward": T_EMBEDDING_DENSE_BACKWARD,
    "embedding_dense_backward_out": T_EMBEDDING_DENSE_BACKWARD_OUT,
    "upsample_nearest_exact2d": T_UPSAMPLE_NEAREST_EXACT2D,
    "upsample_nearest_exact2d_out": T_UPSAMPLE_NEAREST_EXACT2D_OUT,
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
    "where_self": T_WHERE_SELF,
    "where_self_out": T_WHERE_SELF_OUT,
    "all_whole": T_ALL_WHOLE,
    "index_tensor": T_INDEX_TENSOR,
    "nonzero": T_NONZERO,
    "nonzero_static": T_NONZERO_STATIC,
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_gcu.py -- DO NOT EDIT.
//
// topsaten kernels for the Enflame GCU backend, generated per-category. Each
// kernel wraps its aten tensors into topsatenTensors and issues one direct
// topsaten call via EXEC_TOPSATEN_CMD. Dispatchers are declared in
// generated/ops.h (shared with the CUDA codegen); here we only fill the
// Backend::kGcu slot.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/RangeUtils.h>
#include <ATen/ops/all.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/convolution_backward.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/nonzero_static.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_upsample_nearest_exact2d.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/core/List.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "../topsaten_common.h"
#include "runtime/functions.h"

namespace at::native::flagos {

namespace gcu = at::native::flagos::gcu;

"""

FILE_FOOTER = "\n} // namespace at::native::flagos\n"

INC_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_gcu.py -- DO NOT EDIT.
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
        if op in METADATA_OPS:
            if op not in wrappers:
                skipped.append((op, "no wrapper in generated/register.inc"))
                continue
            covered.append((op, "metadata", "metadata"))
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
        # Only T_ARANGE's three instantiations read these; every other template
        # ignores the extra keywords.
        params, bind, dtype_infer = ARANGE_OVERLOADS.get(op, ("", "", ""))
        bodies.append(
            CATEGORIES[cat].format(
                kernel=kernel,
                tops=tops,
                fn=fn,
                disp=disp,
                at_op=AT_OP_OVERRIDES.get(op, base),
                promote_integral="true" if base == "sum" else "false",
                params=params,
                bind=bind,
                dtype_infer=dtype_infer,
            )
        )
        covered.append((op, tops, cat))

    for op in sorted(HANDWRITTEN_OPS):
        if op in wrappers and not any(item[0] == op for item in covered):
            covered.append((op, "handwritten", "handwritten"))

    # No kernel body: these emit an m.impl() only, so that
    # csrc/aten/strided_ops.cc's kGcu registrations are reachable.
    for op in sorted(METADATA_OPS):
        if op in wrappers and not any(item[0] == op for item in covered):
            covered.append((op, "strided_ops.cc", "metadata"))

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(FILE_HEADER + "\n".join(bodies) + FILE_FOOTER)

    impls = "".join(
        f'  m.impl("{op}", {wrappers[op]});\n' for op, _, _ in sorted(covered)
    )
    OUT_INC.write_text(INC_HEADER + impls)

    # `covered` is the PrivateUse1 coverage set. Only the generated categories
    # carry a body in OUT_CC (and a foreach category emits several bodies per op,
    # so this is the op count, not a function count); metadata ops live in
    # strided_ops.cc and handwritten ones in their own translation unit.
    generated = sum(1 for _, _, cat in covered if cat in CATEGORIES)
    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({generated} ops)")
    print(f"[gen] {OUT_INC.relative_to(REPO)}  ({len(covered)} m.impl lines)")
    by_cat = {}
    for op, tops, cat in covered:
        by_cat.setdefault(cat, []).append((op, tops))
    for cat in list(CATEGORIES) + ["metadata", "handwritten"]:
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
