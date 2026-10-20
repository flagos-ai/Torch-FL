#!/usr/bin/env python3
# Copyright (c) 2026, BAAI. All rights reserved.
#
# Generate the Iluvatar BI-V150 (corex) native operator layer.
#
# This is the vendor-native, NON-boxing half of the iluvatar backend. It sits
# alongside the FlagGems-primary path rather than replacing it: backends_iluvatar.conf
# routes an op to `flaggems` where FlagGems covers it and to `iluvatar` where it
# does not, and the ops this file claims are exactly the `iluvatar` set. The two
# registration files (iluvatar_flaggems_register.inc and this generator's
# iluvatar_register.inc) are both included into the same PrivateUse1
# TORCH_LIBRARY_IMPL body, so they must be disjoint: an op in both is a duplicate
# m.impl, which torch rejects at library load. render() enforces that against the
# committed FlagGems .inc (see `flaggems_claimed`).
#
# The vendor API is corex's CUDA-*named* C ABI. `/usr/local/corex/include/cudnn.h`
# is a thin umbrella over `IX/mapping_cudnn.h` + `IX/ixdnn/ixdnn.h`, and
# `cublas_v2.h` the same for `IX/ixblas/`. The mapping headers `#define` every
# CUDA symbol to its ix* counterpart, so a kernel here writes ordinary
# cudnnConvolutionForward / cublasGemmEx calls and the linker resolves them to the
# 198 `T ixdnn*` symbols in libcudnn.so.7 and the `ixblas*_v2` symbols in
# libcublas.so.10. The tensor's data_ptr() is passed straight through: a flagos
# tensor's storage *is* corex device memory (proven on ub22: same-pointer
# flagos->cuda view, ixdnn conv2d matches CPU bit-for-bit), so no device metadata
# is retagged and no CUDA dispatch key is entered.
#
# Unlike codegen_kunlunxin.py, which owns a fixed ten-op set, this generator's op
# set is the first cohort only -- the ops that both (a) currently route to `none`
# in backends_iluvatar.conf and (b) have a usable ixdnn/ixblas entry point. The
# pipeline it proves is end-to-end: conf route -> PrivateUse1 registration ->
# kIluvatar dispatcher slot -> vendor call -> CPU-comparison pass.
#
# Run it to (re)generate csrc/aten/backends/iluvatar/generated/; `--check`
# verifies the committed artifacts are current without writing, the mode CI uses.

import argparse
import os
import re
import sys
from pathlib import Path

# Reuse the authoritative naming from the CUDA codegen so the emitted
# REGISTER_IMPL_TO_DISPATCHER(FnType, dispatcher, ...) matches the
# DECLARE_DISPATCHER in generated/ops.h exactly (otherwise the build won't link).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegen_ops import schema_to_cpp_name

REPO = Path(__file__).resolve().parent.parent.parent
OUT_CC = REPO / "csrc/aten/backends/iluvatar/generated/iluvatar_kernels.cc"
OUT_INC = REPO / "csrc/aten/backends/iluvatar/generated/iluvatar_register.inc"
REGISTER_INC = REPO / "csrc/aten/generated/register.inc"
# The FlagGems .inc is included into the same TORCH_LIBRARY_IMPL body, so the
# claim set below is intersected against it. Read it as text rather than through
# gen_vendor_confs, which the FlagGems generator's own --check avoids importing
# for the same reason: keep this generator runnable in the lean codegen-checks CI
# job without pulling in gen_vendor_confs' full module graph.
FLAGGEMS_INC = (
    REPO / "csrc/aten/backends/iluvatar/generated/iluvatar_flaggems_register.inc"
)

# ---------------------------------------------------------------------------
# Op registry: schema op name -> category.
#
# The cohort is the ops RECON found a usable vendor entry point *and* an existing
# privateuseone Wrapper for:
#
#   * `_convolution`, `convolution`               -> conv  (ixdnn forward)
#   * `native_batch_norm`,
#     `_native_batch_norm_legit`,
#     `_native_batch_norm_legit_no_training`      -> batchnorm (ixdnn train/infer)
#   * `mm`, `bmm`, `addmm`                        -> gemm  (ixblas GemmEx)
#
# The frozen contract named "matmul (GEMM)" as a cohort op, but there is no
# `m.impl("matmul", ...)`, no `WrapperMatmul` and no `MatmulFn` in the generated
# artifacts: matmul is a composite that decomposes into mm/bmm. Binding the
# leaves is the only route the Wrapper mechanism admits without a hand-written
# wrapper -- see the conflict note in the PR.
#
# KERNELS vs CLAIMS. Every op here gets a kernel in iluvatar_kernels.cc and a
# kIluvatar dispatcher registration, because each is reachable by an explicit
# override -- FLAGOS_FORCE_BACKEND=vendor, or a per-op FLAGOS_OP_<name>=iluvatar.
# But only the ops FlagGems does *not* already claim get an m.impl line in
# iluvatar_register.inc: the FlagGems .inc is included into the same library
# body, and torch rejects a duplicate m.impl at load. The GEMM trio and the two
# `_native_batch_norm_legit*` variants are already claimed there (route=flaggems
# today), so they keep their FlagGems registration and their kernel here is
# reached only by an override. That the conf cannot move them is intended --
# FlagGems is the primary path and `route()` gives it priority.
#
# The runtime dtype escape is a THIRD mechanism and it does NOT reach them:
# ResolveFn substitutes the vendor slot only when FlagGemsRejectsArgs() fires,
# and that predicate is compiled per vendor with a `return false` fallback --
# there is no USE_ILUVATAR arm, so the escape never fires on this build. Adding
# one is a per-dtype decision that needs a measured BI-V150 FlagGems gap.
# ---------------------------------------------------------------------------
OPS = {
    "_convolution": "conv",
    "convolution": "conv_public",
    "native_batch_norm": "batchnorm",
    "_native_batch_norm_legit": "batchnorm_legit",
    "_native_batch_norm_legit_no_training": "batchnorm_legit_no_training",
    "mm": "mm",
    "bmm": "bmm",
    "addmm": "addmm",
}

# The per-op dtype predicate each kernel gates on (iluvatar_common.h). The base
# `IluvatarSupportsDtype` is the vendor scalar-type map (float/half/double/
# bfloat16); the per-op gates narrow it to the dtypes the first cohort was
# measured on. A kernel tests its op's predicate first and takes the CPU
# round-trip for anything outside it, so the op stays correct.
SUPPORTS = {
    "conv": "IluvatarSupportsConvDtype",
    "conv_public": "IluvatarSupportsConvDtype",
    "batchnorm": "IluvatarSupportsBatchNormDtype",
    "batchnorm_legit": "IluvatarSupportsBatchNormDtype",
    "batchnorm_legit_no_training": "IluvatarSupportsBatchNormDtype",
    "mm": "IluvatarSupportsGemmDtype",
    "bmm": "IluvatarSupportsGemmDtype",
    "addmm": "IluvatarSupportsGemmDtype",
}

# `{at_op}` names the aten function the CPU round-trip calls. For the private
# families it is the private leaf (`_convolution`, `_native_batch_norm_legit`,
# `_native_batch_norm_legit_no_training`), for the public ones the public leaf;
# mm/bmm/addmm call themselves.
AT_OP = {
    "_convolution": "_convolution",
    "convolution": "convolution",
    "native_batch_norm": "native_batch_norm",
    "_native_batch_norm_legit": "_native_batch_norm_legit",
    "_native_batch_norm_legit_no_training": "_native_batch_norm_legit_no_training",
    "mm": "mm",
    "bmm": "bmm",
    "addmm": "addmm",
}

# ==========================================================================
# Templates
#
# `{kernel}`, `{fn}`, `{disp}` come from schema_to_cpp_name; `{supports}` from
# SUPPORTS; `{at_op}` from AT_OP; the batch-norm body keys from _BN_ARGS. Every
# other brace is C++ and is doubled so str.format leaves it alone.
# ==========================================================================


def _gemm_call(alpha, beta, a_ptr, b_ptr, c_ptr, compute_type, m, n, k):
    """The column-major cublas call computing a row-major C(m,n) = A(m,k)*B(k,n).

    cublas is column-major, so C^T(n,m) = B^T(n,k) * A^T(k,m): pass B as the
    first operand (n x k, lda = n) and A as the second (k x m, ldb = k), and the
    product lands row-major in C (n x m column-major = m x n row-major). GemmEx
    takes the operand type and the compute type separately, so half inputs
    accumulate in float32 (compute_type, per IluvatarGemmComputeType).
    """
    return (
        "  ILUVATAR_EXEC(\n"
        "      cublasGemmEx(\n"
        "          handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>({n}),\n"
        "          static_cast<int>({m}), static_cast<int>({k}), {alpha}, {b_ptr},\n"
        "          ab_type, static_cast<int>({n}), {a_ptr}, ab_type,\n"
        "          static_cast<int>({k}), {beta}, {c_ptr}, c_type,\n"
        "          static_cast<int>({n}), {compute_type}, CUBLAS_GEMM_DEFAULT),\n"
        '      "cublasGemmEx");\n'
    ).format(
        alpha=alpha,
        beta=beta,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        c_ptr=c_ptr,
        compute_type=compute_type,
        m=m,
        n=n,
        k=k,
    )


# --- convolution -----------------------------------------------------------

T_CONV = """\
at::Tensor {kernel}(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ::std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32) {{
  // Transposed, non-4-D and non-float/half convolutions take the host
  // round-trip: ixdnn's forward path is the NCHW spatial correlation only, and
  // claiming more than has been measured is what the batch-norm CRASH on the
  // FlagGems side warns against. The device comparison is part of the gate for
  // the same reason as the batch-norm stats check: ixdnn takes a raw pointer,
  // so a CPU weight/bias would be read as device memory. Composite C++ code
  // does not always move an op's operands together.
  if (!iluvatar::{supports}(input.scalar_type()) || input.dim() != 4 ||
      input.scalar_type() != weight.scalar_type() || transposed ||
      weight.device() != input.device()) {{
    const ::std::optional<at::Tensor> bias_cpu =
        iluvatar::IluvatarDefined(bias)
        ? ::std::optional<at::Tensor>(bias->cpu())
        : ::std::nullopt;
    return at::{at_op}(
               input.cpu(), weight.cpu(), bias_cpu, stride, padding, dilation,
               transposed, output_padding, groups, false, false, false, false)
        .to(input.device());
  }}
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t H = input.size(2);
  const int64_t W = input.size(3);
  const int64_t K = weight.size(0);
  const int64_t R = weight.size(2);
  const int64_t S = weight.size(3);
  const int64_t pad_h = padding.empty() ? 0 : padding[0];
  const int64_t pad_w = padding.size() > 1 ? padding[1] : pad_h;
  const int64_t str_h = stride.empty() ? 1 : stride[0];
  const int64_t str_w = stride.size() > 1 ? stride[1] : str_h;
  const int64_t dil_h = dilation.empty() ? 1 : dilation[0];
  const int64_t dil_w = dilation.size() > 1 ? dilation[1] : dil_h;
  const int64_t OH = (H + 2 * pad_h - dil_h * (R - 1) - 1) / str_h + 1;
  const int64_t OW = (W + 2 * pad_w - dil_w * (S - 1) - 1) / str_w + 1;
  const int64_t Cpg = C / groups;

  // The guard opens before the .contiguous() copies and before at::empty: both
  // resolve the device from whatever is current, so a non-contiguous operand on
  // a device other than the current one would be copied onto the wrong card and
  // the output allocated there. Opening it first keeps every allocation on the
  // input's own device.
  iluvatar::IluvatarDeviceGuard guard(input);
  at::Tensor input_c = input.contiguous();
  at::Tensor weight_c = weight.contiguous();
  const int device = iluvatar::IluvatarDeviceIndex(input_c);
  cudnnHandle_t handle = iluvatar::GetCudnnHandle(device);
  const cudnnDataType_t dt = iluvatar::ToCudnnDataType(input_c.scalar_type());
  // Compute in float even for half input: ixdnn's half-precision accumulate is
  // not what torch's CPU reference does, so a float accumulate keeps the
  // CPU-comparison gate meaningful. alpha/beta are therefore plain float.
  const cudnnDataType_t compute = CUDNN_DATA_FLOAT;

  at::Tensor out = at::empty({{N, K, OH, OW}}, input_c.options());

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t c_desc = nullptr;
  ILUVATAR_EXEC(cudnnCreateTensorDescriptor(&x_desc),
                "cudnnCreateTensorDescriptor");
  ILUVATAR_EXEC(cudnnCreateTensorDescriptor(&y_desc),
                "cudnnCreateTensorDescriptor");
  ILUVATAR_EXEC(cudnnCreateFilterDescriptor(&w_desc),
                "cudnnCreateFilterDescriptor");
  ILUVATAR_EXEC(cudnnCreateConvolutionDescriptor(&c_desc),
                "cudnnCreateConvolutionDescriptor");
  ILUVATAR_EXEC(
      cudnnSetTensor4dDescriptor(
          x_desc, CUDNN_TENSOR_NCHW, dt, static_cast<int>(N),
          static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)),
      "cudnnSetTensor4dDescriptor");
  ILUVATAR_EXEC(
      cudnnSetTensor4dDescriptor(
          y_desc, CUDNN_TENSOR_NCHW, dt, static_cast<int>(N),
          static_cast<int>(K), static_cast<int>(OH), static_cast<int>(OW)),
      "cudnnSetTensor4dDescriptor");
  ILUVATAR_EXEC(
      cudnnSetFilter4dDescriptor(
          w_desc, dt, CUDNN_TENSOR_NCHW, static_cast<int>(K),
          static_cast<int>(Cpg), static_cast<int>(R), static_cast<int>(S)),
      "cudnnSetFilter4dDescriptor");
  ILUVATAR_EXEC(
      cudnnSetConvolution2dDescriptor(
          c_desc, static_cast<int>(pad_h), static_cast<int>(pad_w),
          static_cast<int>(str_h), static_cast<int>(str_w),
          static_cast<int>(dil_h), static_cast<int>(dil_w),
          CUDNN_CROSS_CORRELATION, compute),
      "cudnnSetConvolution2dDescriptor");
  if (groups > 1) {{
    ILUVATAR_EXEC(
        cudnnSetConvolutionGroupCount(c_desc, static_cast<int>(groups)),
        "cudnnSetConvolutionGroupCount");
  }}

  // Forward is a two-stage call: pick an algorithm, ask it how much workspace it
  // needs, allocate that from our caching allocator, then run it.
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  ILUVATAR_EXEC(
      cudnnGetConvolutionForwardAlgorithm(
          handle, x_desc, w_desc, c_desc, y_desc,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
          /*memoryLimitInBytes=*/0, &algo),
      "cudnnGetConvolutionForwardAlgorithm");
  size_t ws_size = 0;
  ILUVATAR_EXEC(
      cudnnGetConvolutionForwardWorkspaceSize(
          handle, x_desc, w_desc, c_desc, y_desc, algo, &ws_size),
      "cudnnGetConvolutionForwardWorkspaceSize");
  at::Tensor workspace = iluvatar::IluvatarWorkspaceTensor(ws_size, out);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  ILUVATAR_EXEC(
      cudnnConvolutionForward(
          handle, &alpha, x_desc, input_c.const_data_ptr(), w_desc,
          weight_c.const_data_ptr(), c_desc, algo,
          ws_size == 0 ? nullptr : workspace.data_ptr(), ws_size, &beta,
          y_desc, out.data_ptr()),
      "cudnnConvolutionForward");

  ILUVATAR_EXEC(cudnnDestroyConvolutionDescriptor(c_desc),
                "cudnnDestroyConvolutionDescriptor");
  ILUVATAR_EXEC(cudnnDestroyFilterDescriptor(w_desc),
                "cudnnDestroyFilterDescriptor");
  ILUVATAR_EXEC(cudnnDestroyTensorDescriptor(y_desc),
                "cudnnDestroyTensorDescriptor");
  ILUVATAR_EXEC(cudnnDestroyTensorDescriptor(x_desc),
                "cudnnDestroyTensorDescriptor");
  iluvatar::IluvatarSynchronizeStream(device);

  // ixdnn takes no bias, so it is added on the device afterwards; reshape makes
  // the (C,) bias broadcast along the channel dim of the NCHW output.
  //
  // `has_value()` alone is not the test. A `Tensor?` argument arriving from
  // C++ composite code -- aten::conv2d passes its own `Tensor()` placeholder
  // straight through when the caller supplied no bias -- is *present but
  // undefined*, so has_value() is true while the tensor is not. Dereferencing
  // it reaches at::native::reshape_symint, whose symbol argument lands on the
  // SymInt overload of aten::view, and that overload has no PrivateUse1 kernel
  // (measured on ub22: bias-less `aten::conv2d` dies with "no tensor arguments
  // ... no fallback function is registered for schema aten::view", while the
  // same call with a real bias succeeds). Requiring defined() keeps the
  // undefined placeholder on the no-bias path where it belongs.
  if (iluvatar::IluvatarDefined(bias)) {{
    out.add_(bias->reshape({{1, K, 1, 1}}));
  }}

  (void)benchmark;
  (void)deterministic;
  (void)cudnn_enabled;
  (void)allow_tf32;
  (void)output_padding;
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kIluvatar, {kernel})
"""

T_CONV_PUBLIC = """\
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
  if (!iluvatar::{supports}(input.scalar_type()) || input.dim() != 4 ||
      input.scalar_type() != weight.scalar_type() || transposed ||
      weight.device() != input.device()) {{
    const ::std::optional<at::Tensor> bias_cpu =
        iluvatar::IluvatarDefined(bias)
        ? ::std::optional<at::Tensor>(bias->cpu())
        : ::std::nullopt;
    return at::{at_op}(
               input.cpu(), weight.cpu(), bias_cpu, stride, padding, dilation,
               transposed, output_padding, groups)
        .to(input.device());
  }}
  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  const int64_t H = input.size(2);
  const int64_t W = input.size(3);
  const int64_t K = weight.size(0);
  const int64_t R = weight.size(2);
  const int64_t S = weight.size(3);
  const int64_t pad_h = padding.empty() ? 0 : padding[0];
  const int64_t pad_w = padding.size() > 1 ? padding[1] : pad_h;
  const int64_t str_h = stride.empty() ? 1 : stride[0];
  const int64_t str_w = stride.size() > 1 ? stride[1] : str_h;
  const int64_t dil_h = dilation.empty() ? 1 : dilation[0];
  const int64_t dil_w = dilation.size() > 1 ? dilation[1] : dil_h;
  const int64_t OH = (H + 2 * pad_h - dil_h * (R - 1) - 1) / str_h + 1;
  const int64_t OW = (W + 2 * pad_w - dil_w * (S - 1) - 1) / str_w + 1;
  const int64_t Cpg = C / groups;

  iluvatar::IluvatarDeviceGuard guard(input);
  at::Tensor input_c = input.contiguous();
  at::Tensor weight_c = weight.contiguous();
  const int device = iluvatar::IluvatarDeviceIndex(input_c);
  cudnnHandle_t handle = iluvatar::GetCudnnHandle(device);
  const cudnnDataType_t dt = iluvatar::ToCudnnDataType(input_c.scalar_type());
  const cudnnDataType_t compute = CUDNN_DATA_FLOAT;

  at::Tensor out = at::empty({{N, K, OH, OW}}, input_c.options());

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t c_desc = nullptr;
  ILUVATAR_EXEC(cudnnCreateTensorDescriptor(&x_desc),
                "cudnnCreateTensorDescriptor");
  ILUVATAR_EXEC(cudnnCreateTensorDescriptor(&y_desc),
                "cudnnCreateTensorDescriptor");
  ILUVATAR_EXEC(cudnnCreateFilterDescriptor(&w_desc),
                "cudnnCreateFilterDescriptor");
  ILUVATAR_EXEC(cudnnCreateConvolutionDescriptor(&c_desc),
                "cudnnCreateConvolutionDescriptor");
  ILUVATAR_EXEC(
      cudnnSetTensor4dDescriptor(
          x_desc, CUDNN_TENSOR_NCHW, dt, static_cast<int>(N),
          static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)),
      "cudnnSetTensor4dDescriptor");
  ILUVATAR_EXEC(
      cudnnSetTensor4dDescriptor(
          y_desc, CUDNN_TENSOR_NCHW, dt, static_cast<int>(N),
          static_cast<int>(K), static_cast<int>(OH), static_cast<int>(OW)),
      "cudnnSetTensor4dDescriptor");
  ILUVATAR_EXEC(
      cudnnSetFilter4dDescriptor(
          w_desc, dt, CUDNN_TENSOR_NCHW, static_cast<int>(K),
          static_cast<int>(Cpg), static_cast<int>(R), static_cast<int>(S)),
      "cudnnSetFilter4dDescriptor");
  ILUVATAR_EXEC(
      cudnnSetConvolution2dDescriptor(
          c_desc, static_cast<int>(pad_h), static_cast<int>(pad_w),
          static_cast<int>(str_h), static_cast<int>(str_w),
          static_cast<int>(dil_h), static_cast<int>(dil_w),
          CUDNN_CROSS_CORRELATION, compute),
      "cudnnSetConvolution2dDescriptor");
  if (groups > 1) {{
    ILUVATAR_EXEC(
        cudnnSetConvolutionGroupCount(c_desc, static_cast<int>(groups)),
        "cudnnSetConvolutionGroupCount");
  }}

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  ILUVATAR_EXEC(
      cudnnGetConvolutionForwardAlgorithm(
          handle, x_desc, w_desc, c_desc, y_desc,
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
          /*memoryLimitInBytes=*/0, &algo),
      "cudnnGetConvolutionForwardAlgorithm");
  size_t ws_size = 0;
  ILUVATAR_EXEC(
      cudnnGetConvolutionForwardWorkspaceSize(
          handle, x_desc, w_desc, c_desc, y_desc, algo, &ws_size),
      "cudnnGetConvolutionForwardWorkspaceSize");
  at::Tensor workspace = iluvatar::IluvatarWorkspaceTensor(ws_size, out);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  ILUVATAR_EXEC(
      cudnnConvolutionForward(
          handle, &alpha, x_desc, input_c.const_data_ptr(), w_desc,
          weight_c.const_data_ptr(), c_desc, algo,
          ws_size == 0 ? nullptr : workspace.data_ptr(), ws_size, &beta,
          y_desc, out.data_ptr()),
      "cudnnConvolutionForward");

  ILUVATAR_EXEC(cudnnDestroyConvolutionDescriptor(c_desc),
                "cudnnDestroyConvolutionDescriptor");
  ILUVATAR_EXEC(cudnnDestroyFilterDescriptor(w_desc),
                "cudnnDestroyFilterDescriptor");
  ILUVATAR_EXEC(cudnnDestroyTensorDescriptor(y_desc),
                "cudnnDestroyTensorDescriptor");
  ILUVATAR_EXEC(cudnnDestroyTensorDescriptor(x_desc),
                "cudnnDestroyTensorDescriptor");
  iluvatar::IluvatarSynchronizeStream(device);

  // defined() as well as has_value() -- see the T_CONV note on the undefined
  // `Tensor?` placeholder.
  if (iluvatar::IluvatarDefined(bias)) {{
    out.add_(bias->reshape({{1, K, 1, 1}}));
  }}
  (void)output_padding;
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kIluvatar, {kernel})
"""


# --- batch norm ------------------------------------------------------------
#
# One shared body, three signatures. The variants differ only in how the running
# statistics are obtained (optionals for `native_batch_norm`, mutable references
# for `_native_batch_norm_legit`, constants for `..._no_training`) and in whether
# training is possible; the vendor call sequence, the descriptor set-up and the
# tuple return are identical.
#
# ixdnn's batch-norm takes scale/bias pointers and a running-stats pointer pair,
# so the shared body synthesises ones/zeros when weight/bias are absent and
# passes the running stats through directly when they are present (mutated in
# place on the training path, read on the inference path).
_BN_BODY = """\
  const ::std::optional<at::Tensor> weight_cpu =
      iluvatar::IluvatarDefined(weight)
          ? ::std::optional<at::Tensor>(weight->cpu())
          : ::std::nullopt;
  const ::std::optional<at::Tensor> bias_cpu =
      iluvatar::IluvatarDefined(bias)
          ? ::std::optional<at::Tensor>(bias->cpu())
          : ::std::nullopt;
  // `training_in` rather than `training`: for the variants whose training flag
  // IS the parameter, `const bool training = training;` would read the
  // not-yet-initialised inner binding, not the argument.
  const bool training_in = {training_expr};
  at::Tensor rm = {rm_expr};
  at::Tensor rv = {rv_expr};

  // The vendor path needs float, contiguous running statistics (the derived
  // bnScaleBiasMeanVar descriptor is float32), a 4-D NCHW input and a mapped
  // dtype. Anything else keeps the op correct through the host round-trip.
  //
  // The device check is not decoration. ixdnn takes a raw pointer, so a CPU
  // running_mean -- which C++ composite code and `nn.BatchNorm2d` hand over
  // without complaint, since neither `load_state_dict` nor a bare module moves
  // buffers -- would be dereferenced as device memory: measured on ub22, a
  // host-resident running stat yields eval output off by 1.2e+03 with no error
  // from ixdnn, and the training path *writes* through that host pointer. CUDA
  // torch rejects the same call rather than corrupting. Sending the mismatch to
  // the round-trip below keeps it correct -- and the fallback's own
  // `running_mean->copy_` still lands on the caller's (host) buffer.
  const bool stats_ok = rm.defined() && rv.defined() &&
      rm.device() == input.device() && rv.device() == input.device() &&
      rm.scalar_type() == at::kFloat && rv.scalar_type() == at::kFloat &&
      rm.is_contiguous() && rv.is_contiguous();
  if (!iluvatar::{supports}(input.scalar_type()) || input.dim() != 4 ||
      !stats_ok) {{
{fallback}
  }}

  iluvatar::IluvatarDeviceGuard guard(input);
  at::Tensor input_c = input.contiguous();
  const int device = iluvatar::IluvatarDeviceIndex(input_c);
  cudnnHandle_t handle = iluvatar::GetCudnnHandle(device);
  const cudnnDataType_t dt = iluvatar::ToCudnnDataType(input_c.scalar_type());
  const int64_t N = input_c.size(0);
  const int64_t C = input_c.size(1);
  const int64_t H = input_c.size(2);
  const int64_t W = input_c.size(3);

  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnTensorDescriptor_t bnmv_desc = nullptr;
  ILUVATAR_EXEC(cudnnCreateTensorDescriptor(&x_desc),
                "cudnnCreateTensorDescriptor");
  ILUVATAR_EXEC(cudnnCreateTensorDescriptor(&y_desc),
                "cudnnCreateTensorDescriptor");
  ILUVATAR_EXEC(cudnnCreateTensorDescriptor(&bnmv_desc),
                "cudnnCreateTensorDescriptor");
  ILUVATAR_EXEC(
      cudnnSetTensor4dDescriptor(
          x_desc, CUDNN_TENSOR_NCHW, dt, static_cast<int>(N),
          static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)),
      "cudnnSetTensor4dDescriptor");
  ILUVATAR_EXEC(
      cudnnSetTensor4dDescriptor(
          y_desc, CUDNN_TENSOR_NCHW, dt, static_cast<int>(N),
          static_cast<int>(C), static_cast<int>(H), static_cast<int>(W)),
      "cudnnSetTensor4dDescriptor");
  ILUVATAR_EXEC(
      cudnnDeriveBNTensorDescriptor(bnmv_desc, x_desc, CUDNN_BATCHNORM_SPATIAL),
      "cudnnDeriveBNTensorDescriptor");

  at::Tensor out = at::empty_like(input_c);
  at::Tensor scale = iluvatar::IluvatarDefined(weight)
      ? weight->to(input_c.device(), input_c.scalar_type()).contiguous()
      : at::ones({{C}}, input_c.options());
  at::Tensor bias_t = iluvatar::IluvatarDefined(bias)
      ? bias->to(input_c.device(), input_c.scalar_type()).contiguous()
      : at::zeros({{C}}, input_c.options());

  // alpha/beta are the tensor's own type (half stays half); MakeIluvatarScalar
  // writes the right scalar width for the descriptor's dtype.
  auto alpha = MakeIluvatarScalar(dt, 1.0);
  auto beta = MakeIluvatarScalar(dt, 0.0);

  at::Tensor save_mean;
  at::Tensor save_invstd;
  if (training_in) {{
    // aten's native_batch_norm contract is that save_mean/save_invstd are rank-1
    // (C,) -- what the CPU kernel returns and what the fallback arm above returns
    // untouched -- because native_batch_norm_backward and autograd consume them
    // as (C,) and would broadcast a (1,C,1,1) wrongly. ixdnn's spatial training
    // writes them as (1,C,1,1), so the buffers are allocated (C,) and passed to
    // the vendor call through a (1,C,1,1) view of the same storage; the rank-1
    // shape is what leaves this function. (Measured on ub22: without this,
    // result[1].shape is (1,C,1,1) against the CPU's (C,).)
    save_mean = at::empty({{C}}, input_c.options());
    save_invstd = at::empty({{C}}, input_c.options());
    ILUVATAR_EXEC(
        cudnnBatchNormalizationForwardTraining(
            handle, CUDNN_BATCHNORM_SPATIAL, alpha.ptr(), beta.ptr(), x_desc,
            input_c.const_data_ptr(), y_desc, out.data_ptr(), bnmv_desc,
            scale.const_data_ptr(), bias_t.const_data_ptr(), momentum,
            rm.data_ptr(), rv.data_ptr(), eps,
            save_mean.view({{1, C, 1, 1}}).data_ptr(),
            save_invstd.view({{1, C, 1, 1}}).data_ptr()),
        "cudnnBatchNormalizationForwardTraining");
  }} else {{
    ILUVATAR_EXEC(
        cudnnBatchNormalizationForwardInference(
            handle, CUDNN_BATCHNORM_SPATIAL, alpha.ptr(), beta.ptr(), x_desc,
            input_c.const_data_ptr(), y_desc, out.data_ptr(), bnmv_desc,
            scale.const_data_ptr(), bias_t.const_data_ptr(),
            rm.const_data_ptr(), rv.const_data_ptr(), eps),
        "cudnnBatchNormalizationForwardInference");
    save_mean = at::empty({{0}}, input_c.options());
    save_invstd = at::empty({{0}}, input_c.options());
  }}

  ILUVATAR_EXEC(cudnnDestroyTensorDescriptor(bnmv_desc),
                "cudnnDestroyTensorDescriptor");
  ILUVATAR_EXEC(cudnnDestroyTensorDescriptor(y_desc),
                "cudnnDestroyTensorDescriptor");
  ILUVATAR_EXEC(cudnnDestroyTensorDescriptor(x_desc),
                "cudnnDestroyTensorDescriptor");
  iluvatar::IluvatarSynchronizeStream(device);
  return {{out, save_mean, save_invstd}};
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kIluvatar, {kernel})
"""

T_BATCHNORM = (
    """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const ::std::optional<at::Tensor>& running_mean,
    const ::std::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {{
"""
    + _BN_BODY
)

T_BATCHNORM_LEGIT = (
    """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps) {{
"""
    + _BN_BODY
)

T_BATCHNORM_LEGIT_NO_TRAINING = (
    """\
::std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& input,
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    double momentum,
    double eps) {{
"""
    + _BN_BODY
)

# The host round-trip for each batch-norm variant: move the operands to CPU, call
# the very same aten op there (which resolves to the CPU kernel, not this one),
# then move the outputs -- and any in-place updated running statistics -- back.
#
# These three are inserted into _BN_BODY as the *value* of its `{fallback}`
# field, so they are already-formatted text: a brace here is a brace in the
# output, NOT a substitution slot to escape. They are deliberately single-braced
# while the surrounding template is double-braced -- str.format does not rescan
# a substituted value.
_BN_FALLBACK_NATIVE = """\
    const ::std::optional<at::Tensor> rm_cpu =
        iluvatar::IluvatarDefined(running_mean)
            ? ::std::optional<at::Tensor>(running_mean->cpu())
            : ::std::nullopt;
    const ::std::optional<at::Tensor> rv_cpu =
        iluvatar::IluvatarDefined(running_var)
            ? ::std::optional<at::Tensor>(running_var->cpu())
            : ::std::nullopt;
    auto res = at::native_batch_norm(
        input.cpu(), weight_cpu, bias_cpu, rm_cpu, rv_cpu, training, momentum,
        eps);
    if (iluvatar::IluvatarDefined(running_mean)) {
      running_mean->copy_(*rm_cpu);
    }
    if (iluvatar::IluvatarDefined(running_var)) {
      running_var->copy_(*rv_cpu);
    }
    return {std::get<0>(res).to(input.device()),
            std::get<1>(res).to(input.device()),
            std::get<2>(res).to(input.device())};"""

_BN_FALLBACK_LEGIT = """\
    at::Tensor rm_cpu = running_mean.cpu();
    at::Tensor rv_cpu = running_var.cpu();
    auto res = at::_native_batch_norm_legit(
        input.cpu(), weight_cpu, bias_cpu, rm_cpu, rv_cpu, training, momentum,
        eps);
    running_mean.copy_(rm_cpu);
    running_var.copy_(rv_cpu);
    return {std::get<0>(res).to(input.device()),
            std::get<1>(res).to(input.device()),
            std::get<2>(res).to(input.device())};"""

_BN_FALLBACK_LEGIT_NO_TRAINING = """\
    auto res = at::_native_batch_norm_legit_no_training(
        input.cpu(), weight_cpu, bias_cpu, running_mean.cpu(),
        running_var.cpu(), momentum, eps);
    return {std::get<0>(res).to(input.device()),
            std::get<1>(res).to(input.device()),
            std::get<2>(res).to(input.device())};"""


# --- gemm ------------------------------------------------------------------
#
# `mm`/`bmm`/`addmm` all land on ixblas GemmEx through the shared call above. The
# operands are made contiguous and the accumulation type is float32 for half.

T_MM = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  if (!iluvatar::{supports}(self.scalar_type()) ||
      self.scalar_type() != mat2.scalar_type() || self.dim() != 2 ||
      mat2.dim() != 2 || mat2.size(0) != self.size(1) ||
      mat2.device() != self.device()) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  iluvatar::IluvatarDeviceGuard guard(self);
  at::Tensor a = self.contiguous();
  at::Tensor b = mat2.contiguous();
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  const int device = iluvatar::IluvatarDeviceIndex(a);
  cublasHandle_t handle = iluvatar::GetCublasHandle(device);
  at::Tensor out = at::empty({{m, n}}, a.options());
  const ixblasDataType_t ab_type = iluvatar::ToBlasDataType(a.scalar_type());
  const ixblasDataType_t c_type = iluvatar::ToBlasDataType(out.scalar_type());
  const ixblasDataType_t compute_type =
      iluvatar::IluvatarGemmComputeType(a.scalar_type());
  const float alpha = 1.0f;
  const float beta = 0.0f;
"""
    + _gemm_call(
        alpha="&alpha",
        beta="&beta",
        a_ptr="a.const_data_ptr()",
        b_ptr="b.const_data_ptr()",
        c_ptr="out.data_ptr()",
        compute_type="compute_type",
        m="m",
        n="n",
        k="k",
    )
    + """\
  iluvatar::IluvatarSynchronizeStream(device);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kIluvatar, {kernel})
"""
)

T_BMM = (
    """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& mat2) {{
  if (!iluvatar::{supports}(self.scalar_type()) ||
      self.scalar_type() != mat2.scalar_type() || self.dim() != 3 ||
      mat2.dim() != 3 || mat2.size(0) != self.size(0) ||
      mat2.size(1) != self.size(2) || mat2.device() != self.device()) {{
    return at::{at_op}(self.cpu(), mat2.cpu()).to(self.device());
  }}
  iluvatar::IluvatarDeviceGuard guard(self);
  at::Tensor a = self.contiguous();
  at::Tensor b = mat2.contiguous();
  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = b.size(2);
  const int device = iluvatar::IluvatarDeviceIndex(a);
  cublasHandle_t handle = iluvatar::GetCublasHandle(device);
  at::Tensor out = at::empty({{batch, m, n}}, a.options());
  const ixblasDataType_t ab_type = iluvatar::ToBlasDataType(a.scalar_type());
  const ixblasDataType_t c_type = iluvatar::ToBlasDataType(out.scalar_type());
  const ixblasDataType_t compute_type =
      iluvatar::IluvatarGemmComputeType(a.scalar_type());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int64_t a_stride = m * k;
  const int64_t b_stride = k * n;
  const int64_t c_stride = m * n;
  const char* a_base = static_cast<const char*>(a.const_data_ptr());
  const char* b_base = static_cast<const char*>(b.const_data_ptr());
  char* c_base = static_cast<char*>(out.data_ptr());
  // One GemmEx per batch slice. The batched strided entry point is not in the
  // symbol set this cohort was measured against, so the loop composes the
  // unbatched call rather than guessing at an unverified symbol.
  for (int64_t i = 0; i < batch; ++i) {{
    const void* a_ptr = a_base + i * a_stride * a.element_size();
    const void* b_ptr = b_base + i * b_stride * b.element_size();
    void* c_ptr = c_base + i * c_stride * out.element_size();
"""
    + _gemm_call(
        alpha="&alpha",
        beta="&beta",
        a_ptr="a_ptr",
        b_ptr="b_ptr",
        c_ptr="c_ptr",
        compute_type="compute_type",
        m="m",
        n="n",
        k="k",
    )
    .replace("\n", "\n  ")
    .replace("  ILUVATAR_EXEC", "    ILUVATAR_EXEC", 1)
    + """\
  }}
  iluvatar::IluvatarSynchronizeStream(device);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kIluvatar, {kernel})
"""
)

T_ADDMM = (
    """\
at::Tensor {kernel}(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {{
  if (!iluvatar::{supports}(mat1.scalar_type()) ||
      mat1.scalar_type() != mat2.scalar_type() || mat1.dim() != 2 ||
      mat2.dim() != 2 || mat2.size(0) != mat1.size(1) ||
      mat2.device() != mat1.device() || self.device() != mat1.device() ||
      self.scalar_type() != mat1.scalar_type()) {{
    return at::{at_op}(self.cpu(), mat1.cpu(), mat2.cpu(), beta, alpha)
        .to(self.device());
  }}
  iluvatar::IluvatarDeviceGuard guard(mat1);
  at::Tensor a = mat1.contiguous();
  at::Tensor b = mat2.contiguous();
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  const int device = iluvatar::IluvatarDeviceIndex(a);
  cublasHandle_t handle = iluvatar::GetCublasHandle(device);
  at::Tensor out = at::empty({{m, n}}, a.options());
  const ixblasDataType_t ab_type = iluvatar::ToBlasDataType(a.scalar_type());
  const ixblasDataType_t c_type = iluvatar::ToBlasDataType(out.scalar_type());
  const ixblasDataType_t compute_type =
      iluvatar::IluvatarGemmComputeType(a.scalar_type());
  // beta*self is folded in first, by broadcasting self into the output and then
  // calling the GEMM with beta = 1 so it accumulates onto that.
  out.copy_(self);
  const double beta_v = beta.to<double>();
  if (beta_v != 1.0) {{
    out.mul_(beta);
  }}
  const float alpha_v = alpha.to<float>();
  const float beta_gemm = 1.0f;
  if (alpha_v != 0.0f) {{
"""
    + _gemm_call(
        alpha="&alpha_v",
        beta="&beta_gemm",
        a_ptr="a.const_data_ptr()",
        b_ptr="b.const_data_ptr()",
        c_ptr="out.data_ptr()",
        compute_type="compute_type",
        m="m",
        n="n",
        k="k",
    )
    .replace("\n", "\n  ")
    .replace("  ILUVATAR_EXEC", "    ILUVATAR_EXEC", 1)
    + """\
  }}
  iluvatar::IluvatarSynchronizeStream(device);
  return out;
}}

REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kIluvatar, {kernel})
"""
)

CATEGORIES = {
    "conv": T_CONV,
    "conv_public": T_CONV_PUBLIC,
    "batchnorm": T_BATCHNORM,
    "batchnorm_legit": T_BATCHNORM_LEGIT,
    "batchnorm_legit_no_training": T_BATCHNORM_LEGIT_NO_TRAINING,
    "mm": T_MM,
    "bmm": T_BMM,
    "addmm": T_ADDMM,
}

# `{training_expr}` / `{rm_expr}` / `{rv_expr}` / `{fallback}` fill the shared
# batch-norm body; every other brace is C++ and doubled.
_BN_ARGS = {
    "batchnorm": {
        "training_expr": "training",
        "rm_expr": "iluvatar::IluvatarDefined(running_mean) ? *running_mean : at::Tensor()",
        "rv_expr": "iluvatar::IluvatarDefined(running_var) ? *running_var : at::Tensor()",
        "fallback": _BN_FALLBACK_NATIVE,
    },
    "batchnorm_legit": {
        "training_expr": "training",
        "rm_expr": "running_mean",
        "rv_expr": "running_var",
        "fallback": _BN_FALLBACK_LEGIT,
    },
    "batchnorm_legit_no_training": {
        "training_expr": "false",
        "rm_expr": "running_mean",
        "rv_expr": "running_var",
        "fallback": _BN_FALLBACK_LEGIT_NO_TRAINING,
    },
}

FILE_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_iluvatar.py -- DO NOT EDIT.
//
// corex ixdnn/ixblas kernels for the Iluvatar BI-V150 native, non-boxing
// backend. Each kernel opens a device guard, builds the vendor descriptors from
// the flagos tensor's sizes/strides, calls the vendor C API directly against
// data_ptr() (through the CUDA-named ABI the corex mapping headers provide) and
// returns the output it allocated. Dispatchers are declared in generated/ops.h
// (shared with the CUDA codegen); here we only fill the Backend::kIluvatar slot.
//
// This backend is NOT the CUDA-boxing path. A flagos tensor's storage is corex
// device memory, so data_ptr() is passed straight through with no device
// retagging and no CUDA dispatch key -- the CUDA *names* are only the spelling
// corex re-exports for its own libraries.
//
// Two sets are generated from one op registry. Every op gets a kernel and a
// kIluvatar dispatcher registration here, so all of them are reachable by an
// explicit override (FLAGOS_FORCE_BACKEND=vendor or FLAGOS_OP_<name>=iluvatar);
// the runtime dtype escape does not apply to this build, since
// FlagGemsRejectsArgs() has no USE_ILUVATAR arm. Only
// the ops FlagGems does not already claim get an m.impl in iluvatar_register.inc
// -- the FlagGems .inc is included into the same library body and a duplicate
// m.impl is rejected at load. Whatever routes here without a kernel falls
// through to cpu_fallback.

#include "../../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/_convolution.h>
#include <ATen/ops/_native_batch_norm_legit.h>
#include <ATen/ops/_native_batch_norm_legit_no_training.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <c10/core/Scalar.h>
#include <algorithm>
#include <cstring>
#include <optional>
#include <tuple>
#include <vector>
#include "../iluvatar_common.h"

namespace at::native::flagos {
// Reopened, not aliased: `namespace iluvatar = at::native::flagos::iluvatar;`
// inside at::native::flagos would redeclare the namespace as an alias of
// itself, which the compiler rejects. Reopening it means the kernels' own
// `iluvatar::X` spellings resolve against the header's declarations
// unambiguously, from inside the same namespace.
namespace iluvatar {

namespace {

// A host scalar whose width matches a cudnn descriptor's dtype, for the
// alpha/beta pointers cudnnBatchNormalizationForward* take. This cohort claims
// float and half only (IluvatarSupportsBatchNormDtype), and the values are
// always exactly 1 and 0, so the half/bfloat16 arms write the well-known IEEE
// bit patterns rather than pulling in cuda_fp16.h for two constants.
struct IluvatarScalar {
  alignas(8) unsigned char bytes[8];
  void* ptr() { return bytes; }
};

inline IluvatarScalar MakeIluvatarScalar(cudnnDataType_t type, double value) {
  IluvatarScalar s{};
  switch (type) {
    case CUDNN_DATA_DOUBLE:
      *reinterpret_cast<double*>(s.bytes) = value;
      break;
    case CUDNN_DATA_HALF: {
      const unsigned short h = (value == 0.0) ? 0x0000u : 0x3C00u;
      std::memcpy(s.bytes, &h, sizeof(h));
      break;
    }
    case CUDNN_DATA_BFLOAT16: {
      const unsigned short h = (value == 0.0) ? 0x0000u : 0x3F80u;
      std::memcpy(s.bytes, &h, sizeof(h));
      break;
    }
    default:
      *reinterpret_cast<float*>(s.bytes) = static_cast<float>(value);
      break;
  }
  return s;
}

} // namespace

"""

FILE_FOOTER = "\n} // namespace iluvatar\n} // namespace at::native::flagos\n"

INC_HEADER = """\
// Copyright (c) 2026, BAAI. All rights reserved.
//
// @generated by scripts/codegen/codegen_iluvatar.py -- DO NOT EDIT.
//
// m.impl() lines for the ops with a corex ixdnn/ixblas kernel that FlagGems does
// not already claim. Included by register.cc inside
// TORCH_LIBRARY_IMPL(aten, PrivateUse1) when USE_ILUVATAR and
// FLAGOS_ILUVATAR_KERNEL are set. It is disjoint from
// iluvatar_flaggems_register.inc -- both are included into the same body, so an
// op in both is a duplicate m.impl and a compile/load error. An op FlagGems
// claims keeps its FlagGems registration and is reached only by an explicit
// override (FLAGOS_FORCE_BACKEND=vendor / FLAGOS_OP_<name>=iluvatar); the
// cohort's kernel is still generated in iluvatar_kernels.cc. Which ops those
// are is read back from the committed iluvatar_flaggems_register.inc, so the
// cohort changes (mm moved here on 2026-10-20 when it left the FlagGems route
// for issue #570), not this comment.
//
// This file is also what scripts/codegen/gen_vendor_confs.py reads to decide
// which ops backends_iluvatar.conf may route to `iluvatar`, and a `flaggems`
// route competing with it takes precedence (the mm measurement below predates
// the FlagGems route mm lost on 2026-10-20). An op that is FlagGems-claimed is
// left out here on the FlagGems-primary build, where only one .inc is
// included, for the duplicate-registration reason above.

"""


def wrapper_map():
    """op name -> Wrapper<Name>, read back from the CUDA codegen's register.inc
    so the m.impl() subset we emit cannot drift from the wrappers that exist."""
    if not REGISTER_INC.exists():
        return {}
    return dict(
        re.findall(
            r'^\s*m\.impl\("([^"]+)",\s*(\w+)\);',
            REGISTER_INC.read_text(),
            flags=re.MULTILINE,
        )
    )


def flaggems_claimed():
    """Ops the FlagGems .inc already claims on PrivateUse1. The native .inc must
    not repeat any of them: both .inc files are included into one library body."""
    if not FLAGGEMS_INC.exists():
        return set()
    return set(re.findall(r'm\.impl\(\s*"([^"]+)"', FLAGGEMS_INC.read_text()))


def render():
    """(kernels.cc text, register.inc text, [(op, category, claimed)]).

    Raises on an op with no wrapper, so a stale register.inc fails loudly instead
    of emitting a file that will not link."""
    wrappers = wrapper_map()
    claimed_elsewhere = flaggems_claimed()
    bodies = []
    covered = []
    for op, category in OPS.items():
        wrapper = wrappers.get(op)
        if wrapper is None:
            raise SystemExit(
                f"error: {op} has no wrapper in {REGISTER_INC.relative_to(REPO)}; "
                "run scripts/codegen/codegen_ops.py first"
            )
        fn, disp = schema_to_cpp_name(op)
        kernel = fn[:-2] + "KernelIluvatar"  # PrivConvolutionFn -> ...KernelIluvatar
        fmt = dict(
            kernel=kernel,
            fn=fn,
            disp=disp,
            supports=SUPPORTS[category],
            at_op=AT_OP[op],
        )
        if category in _BN_ARGS:
            fmt.update(_BN_ARGS[category])
        bodies.append(CATEGORIES[category].format(**fmt))
        covered.append((op, category, op not in claimed_elsewhere))

    claims = [(op, wrappers[op]) for op, _, claimed in covered if claimed]
    return (
        FILE_HEADER + "\n".join(bodies) + FILE_FOOTER,
        INC_HEADER + "".join(f'  m.impl("{op}", {w});\n' for op, w in sorted(claims)),
        covered,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artifacts are current instead of writing them",
    )
    args = ap.parse_args()

    cc_text, inc_text, covered = render()
    withheld = [op for op, _, claimed in covered if not claimed]

    if args.check:
        stale = []
        for target, text in ((OUT_CC, cc_text), (OUT_INC, inc_text)):
            current = target.read_text() if target.exists() else ""
            if current != text:
                stale.append(target)
        if stale:
            for target in stale:
                print(f"[check] stale: {target.relative_to(REPO)}", file=sys.stderr)
            print(
                "error: run scripts/codegen/codegen_iluvatar.py to regenerate",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(
            f"[check] {len(covered)} kernel(s) up to date "
            f"({len(covered) - len(withheld)} claimed, {len(withheld)} via escape)"
        )
        return

    OUT_CC.parent.mkdir(parents=True, exist_ok=True)
    OUT_CC.write_text(cc_text)
    OUT_INC.write_text(inc_text)

    print(f"[gen] {OUT_CC.relative_to(REPO)}  ({len(covered)} kernels)")
    print(
        f"[gen] {OUT_INC.relative_to(REPO)}  "
        f"({len(covered) - len(withheld)} m.impl lines)"
    )
    for op, category, claimed in covered:
        mark = "" if claimed else "  (FlagGems-claimed; escape only)"
        print(f"       + {op} [{category}]{mark}")


if __name__ == "__main__":
    main()
