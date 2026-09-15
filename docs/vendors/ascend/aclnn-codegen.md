# Ascend aclnn operator codegen

> Goal: grow the hand-written aclnn kernels in `csrc/aten/backends/ascend/` to full coverage via
> code generation, covering the inference/training backbone operators at low maintenance cost.
>
> Status: design plus a unary-category prototype (verified on real hardware). 2026-07-20, torch
> 2.11 branch.

## 1. Why the CUDA codegen cannot be copied

On the CUDA side, `scripts/codegen/codegen_ops.py` generates `generated/cuda_kernels.cc` whose kernel
body is a single `at::op(args)` — `DeviceBoxingGuard` rewrites the device metadata of a flagos
(PrivateUse1) tensor to CUDA, reusing PyTorch's already-registered CUDA kernel directly. That
shortcut does not exist on Ascend:

- torch_npu and flagos both occupy PrivateUse1, so there is no separate key to box into (see
  [[ascend-libtorch-npu-fallback-fails]] and [[ascend-route1-intercept-fails]], both measured and
  closed off).
- So an Ascend kernel body must **call CANN aclnn itself** (the public C ABI of `libopapi.so`),
  which means each operator has to marshal its arguments into
  `AclTensorWrapper`/`AclScalarWrapper`/`AclIntArrayWrapper`, **allocate its own output and infer
  the shape/dtype**, and then run the two-stage `GetWorkspaceSize` + `Execute`.

| | CUDA codegen | aclnn codegen |
|---|---|---|
| Kernel body | one line, `at::op(args)` | argument marshalling + output allocation + `EXEC_ASCEND_CMD(aclnn<Name>, ...)` |
| Source of information | entirely in the aten schema | the schema **does not carry** the aclnn API name or the marshalling rules |
| Coverage strategy | enumerate every CUDA operator | **by category + mapping table**, expanded category by category |

The key insight: the aclnn calling convention is highly uniform (`EXEC_ASCEND_CMD` already
abstracts the two stages), and the real variation is only in "how arguments are marshalled and
how the output is allocated" — which is **completely consistent within a category**. So the
codegen is organized around **categories**.

## 2. Relationship to existing infrastructure

- **Dispatcher declarations are reused**: `generated/ops.h` already has the full set of `XxxFn`
  typedefs plus `DECLARE_DISPATCHER`, `ops.cc` already has `ADD_IMPL_TO_DISPATCHER`, and
  `register.inc` already has `m.impl("op", WrapperXxx)` binding the aten operator to
  `xxx_dispatcher`. **The ascend codegen declares no dispatchers of its own**; it only emits
  `REGISTER_IMPL_TO_DISPATCHER(XxxFn, xxx_dispatcher, Backend::kAscend, XxxKernelAscend)` to hang
  the kernel in the `kAscend` slot of the existing dispatcher.
- **Symbol names must agree**: the generator reuses `codegen_ops.py:schema_to_cpp_name()` so that
  `XxxFn`/`xxx_dispatcher` align exactly with the CUDA codegen (otherwise linking fails).
- **Runtime selection**: lines reading `op = ascend` in `torch_fl/configs/backends_ascend.conf`
  make `GetBackendForOp` route that op to the `kAscend` slot at runtime. The codegen also writes
  the generated operators into that conf.

## 3. Output location and build

Generated file: `csrc/aten/backends/ascend/generated/ascend_kernels.cc`

- That path is already excluded automatically from non-ascend builds by `csrc/CMakeLists.txt`
  (`if(NOT ASCEND_KERNEL) EXCLUDE ".*/aten/backends/ascend/.*"`); no new CMake rule is needed.
- Includes:
  ```cpp
  #include "../../../generated/ops.h"   // Fn typedefs + DECLARE_DISPATCHER
  #include "../op_preparation.h"        // OpPreparation::apply_tensor_without_format
  #include "../op_api_common.h"         // AclTensorWrapper / EXEC_ASCEND_CMD
  ```
- **Mutually exclusive with hand-written kernels**: an op is either hand-written or generated; it
  cannot register `kAscend` twice (duplicate registration is an error). The codegen reads a skip
  list to exclude already hand-written ops.
- **Principle (2026-07)**: any operator expressible by some codegen category goes through codegen;
  hand-writing is reserved for the bespoke cases codegen cannot express. The early "seed"
  operators written by hand before the codegen existed (abs/cos/add.Tensor/mul.Scalar/where/
  softmax/sum/mean and 19 others) have been migrated to codegen and their hand-written files
  deleted. `SKIP` has shrunk to just `le.Tensor` (the aclnnLe symbol is missing and needs runtime
  multi-version probing) and `mm`/`bmm` (which additionally register out variants the codegen does
  not produce). Currently 17 hand-written registrations and 122 generated.

## 4. The category system (expanded category by category)

63 categories are implemented, covering 138 operators (all verified against CPU on real
hardware):

| category | Criterion | Output shape / dtype | Kernel body template |
|---|---|---|---|
| `unary` | 1 Tensor in, Tensor out | = input | `aclnn<Name>(self, out)` |
| `unary_bool` | 1 Tensor in, predicate | input shape, **bool out** | `aclnn<Name>(self, out)` |
| `unary_scalar` | Tensor + Scalar | = input | `aclnn<Name>(self, s, out)` |
| `unary_two_scalar` | Tensor + 2 Scalars | = input | `aclnn<Name>(self, s1, s2, out)` |
| `unary_int` | Tensor + `int64_t` | = input | `aclnn<Name>(self, i, out)` |
| `unary_dims` | Tensor + `IntArrayRef` | = input | `aclnn<Name>(self, dims, out)` |
| `binary` | 2 Tensors in | broadcast(self, other), self dtype | `aclnn<Name>(self, other, out)` |
| `binary_alpha` | 2 Tensors + Scalar alpha | broadcast | `aclnn<Name>(self, other, alpha, out)` |
| `binary_cmp` | 2 Tensors in, comparison | broadcast, **bool out** | `aclnn<Name>(self, other, out)` |
| `binary_scalar_alpha` | Tensor + Scalar other + Scalar alpha | = input | `aclnn<Name>s(self, other, alpha, out)` |
| `binary_scalar_cmp` | Tensor + Scalar, comparison | input shape, **bool out** | `aclnn<Name>(self, other, out)` |
| `addcmul` | self + t1 + t2 + Scalar value | broadcast(3) | `aclnn<Name>(self, t1, t2, value, out)` |
| `pow_scalar_tensor` | Scalar self + Tensor exponent | = exponent | `aclnn<Name>(self, exp, out)` |
| `reduce_dims` | Tensor + `IntArrayRef dim` + keepdim | reduced along dim | `aclnn<Name>(self, dim, keepdim, out)` |
| `reduce_dim_bool` | Tensor + `int64_t dim` + keepdim | reduced along one dim, **bool out** | `aclnn<Name>(self, dim_list, keepdim, out)` |
| `reduce_max_dim` | Tensor + `int64_t dim` + keepdim | **tuple(values, indices)** | `aclnn<Name>(self, dim, keepdim, val, idx)` |
| `cumsum` | Tensor + `int64_t dim` + optional dtype | input shape (scan) | `aclnn<Name>(self, dim, dtype, out)` |
| `cumprod` | same as cumsum, but dim passed as `aclScalar*` | input shape (scan) | `aclnn<Name>(self, &dim, dtype, out)` |
| `act_backward` | grad_output + output | = output | `aclnn<Name>(grad, output, grad_in)` |
| `threshold_backward` | grad_output + self + Scalar threshold | = self | `aclnn<Name>(grad, self, thr, grad_in)` |
| `elu` | Tensor + 3 Scalars (alpha/scale/input_scale) | = input | `aclnn<Name>(self, a, s, is, out)` |
| `loss` | self + target + `int64 reduction` | None→input / Mean·Sum→scalar | `aclnn<Name>(self, target, reduction, out)` |
| `cummax_cummin` | Tensor + `int64_t dim` | **tuple(values, indices)**, input shape | `aclnn<Name>(self, dim, val, idx)` |
| `aminmax` | Tensor + optional dim + keepdim | **tuple(min, max)** | `aclnn<Name>(self, dim_list, keepdim, min, max)` |
| `prod` | Tensor + optional dtype | scalar | `aclnn<Name>(self, dtype, out)` |
| `gemm_addmm` | self + mat1 + mat2 + beta + alpha | (m1.rows, m2.cols) | `aclnn<Name>(self,m1,m2,beta,alpha,out,cubeMathType)` |
| `gemm_baddbmm` | self + batch1 + batch2 + beta + alpha | (b, b1.rows, b2.cols) | same as above (batched) |
| `mv` | self (n,m) + vec (m,) | (n,) | `aclnn<Name>(self, vec, out, cubeMathType)` |
| `dot` | self + tensor (both 1-D) | scalar | `aclnn<Name>(self, tensor, out)` |
| `layer_norm` | input + normalized_shape + optional weight/bias + eps | **tuple(out, mean, rstd)**; out=input, stat=leading dims + 1 | `aclnn<Name>(input, ns, weight, bias, eps, out, mean, rstd)` |
| `group_norm` | input + optional weight/bias + N/C/HxW/group + eps | **tuple(out, mean, rstd)**; out=input, stat=(N,group) | `aclnn<Name>(input, weight, bias, N, C, HxW, group, eps, out, mean, rstd)` |
| `gelu` | Tensor + `approximate` string_view | = input | `aclnnGeluV2(self, approx_int, out)` (int64 0=none/1=tanh) |
| `gelu_backward` | grad_output + self + `approximate` | = self | `aclnnGeluBackwardV2(grad, self, approx_str, grad_in)` (char\* string) |
| `log_softmax` | Tensor + `int64_t dim` + half_to_float | = input (half_to_float→float out) | `aclnn<Name>(self, dim, out)` |
| `softmax_backward` | grad_output + output + `int64_t dim` + input_dtype | grad_output shape, dtype=input_dtype | `aclnn<Name>(grad, output, dim, grad_in)` |
| `binary_scalar` | Tensor + Scalar, no alpha | = input | `aclnn<Name>(self, scalar, out)` (mul.Scalar→Muls / div.Scalar→Divs) |
| `act_backward_self` | grad_output + self | = self | `aclnn<Name>(grad, self, grad_in)` (silu_backward; differs from act_backward's grad+output) |
| `where` | cond + self + other (3 Tensors) | broadcast(3), self dtype | `aclnn<Name>(cond, self, other, out)` |
| `softmax_fwd` | Tensor + `int64 dim` + half_to_float | input shape; half_to_float→float | `aclnn<Name>(self, dim, out)` |
| `reduce_all` | Tensor (whole-tensor reduction) | bool scalar | `aclnn<Name>(flat, dim_list, false, out)` (flatten to 1-D and reduce) |
| `reduce_sum_dtype` | Tensor + `OptionalIntArrayRef dim` + keepdim + optional dtype | reduced along dim, dtype promoted | `aclnn<Name>(self, dims, keepdim, aclDataType, out)` |
| `reduce_mean_dtype` | same as above | same as above | `aclnnMeanV2(self, dims, keepdim, int32 dtype, out)` |
| `adaptive_avg_pool2d` | self + `SymInt[2] output_size` | leading dims + output_size; **NCHW format** | `aclnn<Name>(self, outSize, out)` |
| `avg_pool2d` | self + k/stride/pad + ceil/countPad + divOverride | pooling formula; **NCHW format** | `aclnn<Name>(self, k, s, p, ceil, cntPad, div, cubeType, out)` |
| `max_pool2d_indices` | self + k/stride/pad/dil + ceil | **tuple(out, int64 indices)**, pooling formula | `aclnn<Name>(self, k, s, p, dil, ceil, out, idx)` |
| `convolution` | input + weight + bias? + stride/pad/dil + transposed + outPad + groups | conv formula, Cout=weight.size(0); **NCHW/NCL/NCDHW format** | `aclnn<Name>(in, w, b, s, p, d, tr, oPad, g, out, cubeType=0)` |
| `convolution_backward` | grad_out + input + weight + biasSizes? + … + output_mask[3] | **tuple(gInput, gWeight, gBias)** | `aclnn<Name>(gOut, in, w, bSz, s, p, d, tr, oPad, g, mask, cubeType=0, gIn, gW, gB)` |
| `max_pool2d_indices_backward` | grad_out + self + k/s/p/dil + ceil + indices | = self; **NCHW format**, indices cast to int32 | `aclnn<Name>(gOut, self, idx_i32, k, s, p, dil, ceil, gIn)` |
| `native_batch_norm` | input + weight?/bias?/rMean?/rVar? + training + momentum + eps | **tuple(out, saveMean, saveInvstd)**; **NCHW format** | `aclnn<Name>(in, w, b, rMean, rVar, train, mom, eps, out, sMean, sInvstd)` |
| `native_batch_norm_backward` | grad_out + input + weight? + rMean?/rVar?/sMean?/sInvstd? + train + eps + output_mask[3] | **tuple(gInput, gWeight, gBias)** | `aclnn<Name>(gOut, in, w, rMean, rVar, sMean, sInvstd, train, eps, mask, gIn, gW, gB)` |
| `avg_pool2d_backward` | grad_out + self + k/s/p + ceil/countPad + divOverride | = self; **NCHW format** | `aclnn<Name>(gOut, self, k, s, p, ceil, cntPad, div, cubeType=0, gIn)` |
| `adaptive_avg_pool2d_backward` | grad_out + self | = self; **NCHW format** | `aclnn<Name>(gOut, self, gIn)` |
| `native_layer_norm_backward` | grad_out + input + normShape + mean + rstd + weight?/bias? + output_mask[3] | **tuple(gInput, gWeight, gBias)** | `aclnn<Name>(gOut, in, nShape, mean, rstd, w, b, mask, gIn, gW, gB)` |
| `native_group_norm_backward` | grad_out + input + mean + rstd + weight? + N/C/HxW/group + output_mask[3] | **tuple(gInput, gGamma, gBeta)** | `aclnn<Name>(gOut, in, mean, rstd, gamma, N, C, HxW, group, mask, gIn, gG, gB)` |
| `masked_fill_scalar` | self + mask + Scalar value | broadcast(self,mask) | copy→`aclnnInplaceMaskedFillScalar(out, mask, value)` |
| `masked_fill_tensor` | self + mask + Tensor value (0-dim) | broadcast(self,mask) | same as above (tensor value, must be device-aligned) |
| `gather` | self + `int64 dim` + index | index shape, self dtype | `aclnnGather(self, dim, index, out)` |
| `index_select` | self + `int64 dim` + index (1-D) | self shape with dim replaced by index.numel() | `aclnnIndexSelect(self, dim, index, out)` |
| `gemm_addmv` | self + mat(n,m) + vec(m) + beta + alpha | (n,) | `aclnn<Name>(self,mat,vec,ALPHA,BETA,out,cubeMathType)` (**alpha comes before beta**) |
| `gemm_addr` | self + vec1(n) + vec2(m) + beta + alpha | (n,m) outer product | `aclnn<Name>(self,vec1,vec2,beta,alpha,out)` (no cubeMathType) |
| `bce` | self + target + optional weight + int reduction | None→input / Mean·Sum→scalar | `aclnn<Name>(self,target,weight,reduction,out)` |
| `bce_backward` | grad_output + self + target + optional weight + reduction | = self | `aclnn<Name>(grad,self,target,weight,reduction,grad_in)` |
| `bce_logits` | self + target + optional weight + optional pos_weight + reduction | None→input / Mean·Sum→scalar | `aclnn<Name>(self,target,weight,posWeight,reduction,out)` |

- **unary (28)**: sqrt/exp/tanh/sigmoid/reciprocal/log/floor/ceil/erf/erfc/expm1/
  log2/log10/log1p/round/trunc/frac/sign/relu/cosh/sinh/asin/atan/asinh/acosh/atanh/
  logical_not/bitwise_not
- **unary_bool (1)**: isinf
- **unary_scalar (4)**: leaky_relu/clamp_min/clamp_max/fmod.Scalar
- **unary_two_scalar (2)**: softplus(beta,threshold)/threshold(threshold,value)
- **unary_int (2)**: tril/triu (diagonal offset)
- **unary_dims (1)**: flip
- **binary (9)**: div.Tensor/pow.Tensor_Tensor/atan2/maximum/minimum/bitwise_or/bitwise_xor/
  fmod.Tensor/floor_divide
- **binary_alpha (1)**: sub.Tensor
- **binary_cmp (8)**: eq/ne/gt/lt/ge.Tensor + logical_and/logical_or/logical_xor
- **binary_scalar_alpha (2)**: add.Scalar/sub.Scalar
- **binary_scalar_cmp (6)**: eq/ne/gt/lt/ge/le.Scalar
- **addcmul (2)**: addcmul/addcdiv
- **pow_scalar_tensor (1)**: pow.Scalar
- **reduce_dims (2)**: amax/amin (reusing the dim normalization and shape reduction logic from
  `sum.cc`)
- **reduce_dim_bool (1)**: any.dim (the single dim is wrapped in a one-element list for aclnn)
- **reduce_max_dim (2)**: max.dim/min.dim (tuple returning values + int64 indices)
- **cumsum (1)**: cumsum; **cumprod (1)**: cumprod
- **act_backward (2)**: tanh_backward/sigmoid_backward (training)
- **threshold_backward (1)**: threshold_backward (relu backward, training)
- **unary_scalar, additional (3)**: celu/softshrink/hardshrink; **unary_two_scalar, additional
  (1)**: hardtanh
- **elu (1)**: elu
- **loss (1)**: mse_loss (reduction decides scalar vs elementwise output)
- **cummax_cummin (2)**: cummax/cummin (tuple return, same-shape scan)
- **aminmax (1)**: aminmax (tuple(min,max), optional dim)
- **prod (1)**: prod (reduces to a scalar)
- **gemm family (6)**: addmm/baddbmm (cube_math_type)/mv/dot/addmv (alpha and beta swapped)/addr
  (no cube_math_type)
- **bce family (3)**: binary_cross_entropy (+ optional weight)/binary_cross_entropy_backward/
  binary_cross_entropy_with_logits (+ optional pos_weight)
- **layer_norm (1)**: native_layer_norm (tuple(out,mean,rstd); the transformer backbone)
- **group_norm (1)**: native_group_norm (tuple(out,mean,rstd))
- **gelu (1)**: gelu (uses aclnnGeluV2 to support both none and tanh approximations; see "the
  gelu V2 trap" below)
- **gelu_backward (1)**: gelu_backward (aclnnGeluBackwardV2, char\* approximate)
- **log_softmax (1)**: _log_softmax (following the hand-written softmax.cc pattern)
- **softmax_backward (2)**: _softmax_backward_data/_log_softmax_backward_data (training; the
  aclnn name drops aten's `_data` suffix)

Long tail not yet integrated (deferred or hand-written):

- **SDPA / flash-attention** (`_scaled_dot_product_efficient_attention` forward):
  **implemented and verified** (2026-07-21).
  - Hand-written `csrc/aten/backends/ascend/scaled_dot_product_attention.cc`, calling
    `aclnnFlashAttentionScore` directly.
  - Parameters: inputLayout="BNSD", scaleValue=1.0/sqrt(D), keepProb=1-dropout_p,
    headNum=num_heads.
  - **Key mapping**: aclnn outputs softmaxMax/Sum as `[B,N,S,8]` (8 is the online softmax tiling),
    while PyTorch needs `logsumexp [B,N,S]`. The implementation takes the first element of the
    last dimension of softmaxMax/Sum (`[:,:,:,0]`) and computes
    `log(softmaxSum) + softmaxMax` to get logsumexp.
  - **attenMask semantics**: `true=MASK_OUT`, `false=KEEP` — the opposite of what the
    documentation says. Causal attention uses `triu(..., diagonal=1)` to produce the upper
    triangular mask.
  - Verification: non-causal max_err=3.34e-06, causal max_err=7.15e-07 (compared against CPU on
    real hardware).
  - **Backward not implemented**: `_scaled_dot_product_efficient_attention_backward` is registered
    as NotImplemented. Reason: the aclnn backward needs softmaxMax and softmaxSum passed
    separately, but the PyTorch forward returns only a single logsumexp tensor; fixing this means
    either changing the forward to save max/sum or recomputing in the backward — a multi-day
    effort.
- var/std.correction, norm.ScalarOpt_dim (correction/p arguments), argmax/argmin/logsumexp/isnan/
  remainder/relu6 (no aclnn symbol, or they need special derivation), transposed conv, 3D
  conv/pool, upsample/interpolate, pad (reflection/replication/constant),
  scatter/scatter_add/index_put, sort/topk.
- **addbmm**: the symbol exists and runs, but hf32 cube accumulation along the batch dimension
  amplifies the relative error to ~1e-2 (a single addmm is only ~1e-4). Deferred until fp32
  accumulation is allowed or cubeMathType is lowered.
- **native_batch_norm**: `aclnnBatchNorm` works for 2D (N,C) input but returns
  `ACLNN_ERR_INNER_NULLPTR` (561103) for 4D NCHW input — it fails at the GetWorkspaceSize stage,
  suggesting it needs a specific format or a BatchNormV2/BatchNormReduce combination instead.
  Deferred to be done with the dedicated conv/pool batch.

**Key trap (the gelu V2)**: `aclnnGelu` (v1) hardcodes the **tanh** approximation, whereas
PyTorch's `gelu` defaults to `approximate="none"` (the erf form, which qwen3 and other backbones
use). Using v1 directly makes the default gelu silently produce a systematic error of ~4.5e-4 —
not precision jitter, but a different approximation. The fix is `aclnnGeluV2`
(`int64_t approximate`: 0=none/1=tanh; an int is varargs-safe) and `aclnnGeluBackwardV2`
(`char* approximate` string — pointer passing is also varargs-safe and immune to the by-value
float trap below).

**Key trap (aclFormat for conv/pool)**: `AclTensorWrapper` marks aclTensors as `ACL_FORMAT_ND` by
default. `aclnnAvgPool2d`/`aclnnAdaptiveAvgPool2d`/`aclnnConvolution` reject 4-D ND tensors and
`GetWorkspaceSize` returns `161002` (PARAM_INVALID) — not a shape or dtype problem. The fix: give
`AclTensorWrapper` an optional `aclFormat fmt` argument (defaulting to ND for backward
compatibility), and have the conv/pool templates pass `ACL_FORMAT_NCHW` (4-D) / `NCL` (3-D) /
`NCDHW` (5-D) by rank. `aclnnMaxPool2dWithIndices`, by contrast, does not care about format and
accepts ND — so this is a per-aclnn requirement, not a global one. Also, conv's `cubeMathType`
must be **0 (KEEP_DTYPE)**; passing 1 (ALLOW_FP32_DOWN_PRECISION) loses ~2.5e-3 of precision in
the cube unit.

**Key trap (max_pool backward's format + indices dtype)**: `aclnnMaxPool2dWithIndices` (forward)
does not care about format and outputs int64 indices, but
`aclnnMaxPool2dWithIndicesBackward` (backward) requires **both NCHW format and int32 indices** —
feeding the forward's int64 indices straight in yields 161002. In the backward template, an
`indices.to(at::kInt)` cast plus the NCHW tag is enough. So forward and backward can have
different format/dtype requirements; each aclnn's `@param` comments in the header must be checked
individually.

**Key trap (out-of-place ops built on an inplace aclnn — do not use clone)**: aclnn's masked_fill
only has inplace variants (`aclnnInplaceMaskedFillScalar/Tensor`, with a non-const selfRef). To
implement the out-of-place aten `masked_fill`, self must first be copied and then filled in
place. But **`self.clone()` cannot be used** — clone goes through `empty_like`, and the ascend
backend does not register `empty_like` (`RuntimeError: empty_like: backend not registered`).
Instead, allocate with `OpPreparation::apply_tensor_without_format(out_shape, opts)` and do
`out.copy_(self.expand(out_shape))`. The same applies to any generated operator that needs
"copy first, then modify in place".

**Key trap (SDPA attenMask semantics are inverted)**: `aclnnFlashAttentionScore`'s attenMask
semantics are **`true=MASK_OUT` (mask this position out), `false=KEEP`**, the opposite of both the
CANN documentation and published findings ("true=KEEP"). Measured: causal attention needs
`torch.triu(torch.ones(...), diagonal=1).bool()` (upper triangle true) as the mask to correctly
mask out future positions. Inverting it (lower triangle true) produces enormous numerical error
(err~4.0). Also, attenMask must be 2D `[S,S]` or 4D `[B,1,S,S]` broadcast — 3D is not accepted.

**Key trap (SDPA logsumexp mapping)**: PyTorch's `_scaled_dot_product_efficient_attention` returns
`logsumexp [B,N,S]` (a single tensor), but `aclnnFlashAttentionScore` outputs softmaxMax and
softmaxSum each as `[B,N,S,8]` (8 being the online softmax tiling factor). The mapping: take the
first element of the last dimension (`softmaxMax[:,:,:,0]` and `softmaxSum[:,:,:,0]`, both
`[B,N,S]`) and compute `logsumexp = torch.log(softmaxSum_0) + softmaxMax_0`. This mapping matches
CPU in both the non-causal and causal cases (max_err ≤3.34e-06). **Why backward is blocked**:
`aclnnFlashAttentionScoreGrad` needs softmaxMax and softmaxSum passed separately (as saved tensors
from the forward), but PyTorch autograd saves only the single logsumexp tensor, from which the
original max and sum cannot be recovered (the log-sum is not invertible). A solution would require
changing the forward's context to save max/sum as well, or recomputing in the backward — a
multi-day effort. The current version implements the forward only.

**Key trap (batch_norm's save_invstd semantics)**: `aclnnBatchNorm`'s forward `output` and
`saveMean` match CPU bit for bit, but `saveInvstd` is defined differently from PyTorch CPU (CPU
uses `1/sqrt(var+eps)`; aclnn returns another form, measured to differ by ~0.18). **This does not
affect training correctness**: the backward `aclnnBatchNormBackward` consumes the same NPU-side
`saveInvstd`, and all three gradients (grad_input/weight/bias) match CPU (err≤4e-6). Only
comparing save_invstd directly as a final result shows a "mismatch"; the end-to-end BN training
loop is correct.

**Key trap (varargs float)**: `EXEC_ASCEND_CMD` calls aclnn through a
`typedef int (*)(...)` variadic function pointer. On aarch64, passing a `float` by value triggers
default argument promotion (float→double) plus the wrong register class, so aclnn reads garbage.
Passing every scalar as an `aclScalar*` pointer or an `int64_t` is safe; the sole exception is
smooth_l1_loss's `float beta`, which is passed by value — measured, beta was always 0 (degenerating
to plain L1). So smooth_l1_loss stays in the long tail, and any aclnn needing a by-value float
argument must go through an explicit non-variadic dlsym call (see the hand-written `le.cc`).

### The shared prologue for binary categories (a key trap)

Every tensor-tensor category shares one prologue that does three things:

```cpp
auto result_dtype = self.scalar_type();
// (1) other must be aligned in BOTH device and dtype -- not just dtype!
auto other_c = other.is_privateuseone()
    ? (other.scalar_type() == result_dtype ? other : other.to(result_dtype))
    : other.to(self.options());           // CPU other -> move to flagos (NPU)
auto out_shape = at::infer_size(self.sizes(), other_c.sizes());
auto self_b  = self.expand(out_shape).contiguous();   // aclnn does not always broadcast itself
auto other_b = other_c.expand(out_shape).contiguous();
```

**The trap hit here**: a "tensor + python scalar" expression such as `torch.sub(x, 3.0)` gets the
scalar wrapped into a **CPU scalar tensor** by PyTorch and dispatched to `aten::sub.Tensor` (not
sub.Scalar). If the prologue aligns dtype but not device, `AclTensorWrapper` reads the CPU storage
as an NPU device address → all nan. The fix is the `other.to(self.options())` above (matching the
hand-written `add.cc`). This one change also fixed every other "scalar takes the Tensor overload"
path: sub, div, pow, and the rest.

## 5. Deriving aclnn names

- Default: `op_name` snake_case → `aclnn` + PascalCase (`sqrt`→`aclnnSqrt`, `floor`→`aclnnFloor`).
- Irregular cases: the `OPS` dict overrides the stem per op (`eq.Tensor`→`EqTensor`,
  `div.Tensor`→`Div`, `pow.Tensor_Tensor`→`PowTensorTensor`, `add.Scalar`→`Adds`, …).
- Before generating, `nm -D libopapi.so` verifies both the `aclnn<Name>` and
  `aclnn<Name>GetWorkspaceSize` symbols; if either is missing the op is skipped with a warning
  (e.g. `square`/`isnan`/`isfinite` are excluded automatically for lack of a dispatcher or a
  symbol).

## 6. The generator, `scripts/codegen/codegen_ascend.py`

Structure:

- `OPS` dict: `schema op name → (category, aclnn-name override)`. The only hand-maintained point.
- `SKIP` set: ops already hand-written for kAscend (abs/cos/add.Tensor/mul.Tensor/le.Tensor, …),
  never re-emitted (a duplicate kAscend registration crashes at import).
- `CATEGORIES` dict: `category → kernel body template string`. Adding a category = one template
  plus a batch of OPS entries.
- Reuses `codegen_ops.py:schema_to_cpp_name` so `XxxFn`/`xxx_dispatcher` align exactly with
  `ops.h`.

Usage: `python scripts/codegen/codegen_ascend.py [--category unary] [--no-conf]` (default: all).

Output:

- `csrc/aten/backends/ascend/generated/ascend_kernels.cc`
- An idempotent rewrite of the `# --- generated ---` block at the end of `backends_ascend.conf`
  (appending newly covered ops)

## 7. Verification loop

Build with
`ACCELERATOR=ascend ASCEND_KERNEL=1 FLAGGEMS_PYTHON=1 CUDA_KERNEL=0 FLAGGEMS_KERNEL=0`, set
`FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_ascend.conf`, and compare each op against CPU.
51/51 pass (unary max_err≤4.4e-5, binary ≤4.7e-6, comparisons match exactly).

**Note**: the full set of env vars must be passed explicitly at build time
(`ACCELERATOR=ascend …`), or setup.py defaults to `ACCELERATOR=cuda` and the cmake configure stage
fails.

## 8. Related

See `docs/vendors/ascend/npu-plan.md` (the overall plan). The bare aclnn prototype has been
deleted; see `docs/ascend_aclnn_codegen_prototype.cc` in git history.
Memories: [[ascend-aclnn-codegen-plan]], [[ascend-backend-broken-on-2.11]].
