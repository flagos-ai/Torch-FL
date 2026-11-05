// Copyright (c) 2026, BAAI. All rights reserved.
//
// `embedding_dense_backward` on Cambricon, through CNNL.
//
// FlagGems cannot serve it: its dense function compares `device.type` against the
// vendor's device name, which torch_fl registers as "flagos", so the check always
// rejects and the op fell through to cpu_fallback on every training step. CNNL's
// dedicated `cnnlEmbeddingBackward` has no such problem.
//
// Two aten semantics the CNNL entry point does not take as arguments have to be
// reproduced here:
//
//   * Indices outside [0, num_weights) are *ignored* by aten, including values
//     beyond int32 range. Narrowing int64 to int32 before that decision aliases
//     `2**32 + 1` onto row 1 and silently corrupts its gradient, so the range
//     check is done on the original width and invalid rows are redirected to a
//     dedicated slot appended past the real table.
//   * `padding_idx = -1` means "no row is skipped". CNNL takes the padding as a
//     plain int, and its treatment of a negative value is not documented, so an
//     absent padding is turned into the same discarded slot rather than relying
//     on whatever a negative index does.
//
// The discarded slot is one row of extra output: every invalid or padded row
// lands there, its gradient is computed normally, and the row is dropped before
// returning. That keeps the real rows' frequency counts exact, which
// `scale_grad_by_freq` divides by.

#include "aten/backends/cambricon/cambricon_common.h"
#include "aten/backends/cambricon/attention.h"

#ifdef USE_CAMBRICON

#include "aten/generated/ops.h"
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

namespace at::native::flagos::cambricon {

namespace {

// CNNL documents its index range as [-2^24, 2^24], and the redirect slot has to
// be representable in the int32 index tensor, so a table past this bound cannot
// be served in one call.
constexpr int64_t kMaxNumWeights = int64_t{1} << 24;

at::Tensor NarrowIndices(const at::Tensor& indices, cnnlHandle_t handle) {
  if (indices.scalar_type() == at::kInt) {
    return indices;
  }
  auto narrowed = at::empty(indices.sizes(), indices.options().dtype(at::kInt));
  TensorDesc from, to;
  from.Set(indices);
  to.Set(narrowed);
  CheckCnnl(
      cnnlCastDataType(handle, from.get(), indices.const_data_ptr(),
                       CNNL_CAST_INT64_TO_INT32, to.get(),
                       narrowed.mutable_data_ptr()),
      "cnnlCastDataType (embedding indices)");
  return narrowed;
}

} // namespace

at::Tensor EmbeddingDenseBackward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  TORCH_CHECK(
      grad_output.device().type() == c10::DeviceType::PrivateUse1 &&
          indices.device() == grad_output.device(),
      "flagos cambricon: embedding_dense_backward needs both tensors on the "
      "same flagos device");
  const auto dtype = grad_output.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
      "flagos cambricon: embedding_dense_backward does not serve ", dtype);
  TORCH_CHECK(
      indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt,
      "flagos cambricon: embedding indices must be int32 or int64, got ",
      indices.scalar_type());
  TORCH_CHECK(num_weights >= 0, "flagos cambricon: negative num_weights");
  TORCH_CHECK(
      num_weights <= kMaxNumWeights,
      "flagos cambricon: embedding tables larger than 2^24 rows are not served "
      "natively yet, got ",
      num_weights);

  const int64_t width = grad_output.dim() == 0 ? 0 : grad_output.size(-1);
  const int64_t rows = indices.numel();
  TORCH_CHECK(
      grad_output.numel() == rows * width,
      "flagos cambricon: embedding_dense_backward got ",
      grad_output.numel(), " gradient elements for ", rows, " indices of width ",
      width);

  auto result = at::zeros(
      {num_weights, width}, grad_output.options().dtype(dtype));
  if (result.numel() == 0 || rows == 0) {
    // Nothing can be scattered; the zero table is already the answer, and CNNL
    // is not called on an empty descriptor.
    return result;
  }

  CambriconDeviceGuard guard(grad_output);
  const auto handle =
      HandleForDevice(static_cast<int>(grad_output.device().index()));

  // Device-only normalization: everything below runs on the card. The gradient
  // is viewed as one row per index.
  const auto grad_dense = grad_output.reshape({rows, width}).contiguous();
  auto flat = indices.reshape({rows});

  // Redirect every row that must not reach a real entry -- padding, negative
  // values and values past the table -- to one slot past the end. This is the
  // range decision aten makes, evaluated at the original index width.
  const bool active_padding = padding_idx >= 0 && padding_idx < num_weights;
  const int64_t discarded = num_weights;
  // Compare at the index tensor's own width so values past int32 -- which aten
  // ignores -- keep their magnitude here and are rejected rather than aliased.
  auto compare = flat;
  if (compare.scalar_type() == at::kInt) {
    compare = compare.to(at::kLong);
  }
  auto clamped = at::where(compare < 0, discarded, compare);
  clamped = at::where(clamped >= num_weights, discarded, clamped);
  if (active_padding) {
    clamped = at::where(clamped == padding_idx, discarded, clamped);
  }
  const auto target = NarrowIndices(clamped.contiguous(), handle).contiguous();

  // The discarded row enlarges the output's leading axis by one and is dropped
  // afterwards. CNNL requires `output.dim == diff.dim - indices.dim + 1`, so the
  // index tensor stays 1-D: an (N,) index, an (N, D) gradient and a (W + 1, D)
  // output. The discarded slot absorbs padding and out-of-range rows.
  const int64_t seq = rows;
  const bool widen = dtype != at::kFloat;
  auto wide = at::zeros(
      {num_weights + 1, width}, grad_output.options().dtype(at::kFloat));

  auto flat_indices = target.reshape({seq});
  TensorDesc idx_desc;
  idx_desc.Set(flat_indices);
  TensorDesc grad_wide_desc, out_wide_desc;
  at::Tensor grad_fp32;
  if (widen) {
    grad_fp32 = at::empty({seq, width}, grad_output.options().dtype(at::kFloat));
    TensorDesc from, to;
    from.Set(grad_dense);
    to.Set(grad_fp32);
    CheckCnnl(
        cnnlCastDataType(handle, from.get(), grad_dense.const_data_ptr(),
                         dtype == at::kHalf ? CNNL_CAST_HALF_TO_FLOAT
                                            : CNNL_CAST_BFLOAT16_TO_FLOAT,
                         to.get(), grad_fp32.mutable_data_ptr()),
        "cnnlCastDataType (embedding gradient)");
    grad_wide_desc.Set(grad_fp32);
  } else {
    grad_wide_desc.Set(grad_dense);
  }
  out_wide_desc.Set(wide);

  size_t workspace_size = 0;
  CheckCnnl(
      cnnlGetEmbeddingBackwardWorkspaceSize(
          handle, grad_wide_desc.get(), out_wide_desc.get(), scale_grad_by_freq,
          &workspace_size),
      "cnnlGetEmbeddingBackwardWorkspaceSize");
  at::Tensor workspace;
  if (workspace_size > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        grad_output.options().dtype(at::kByte));
  }

  // Every index reaching CNNL is a valid table row: the padding and
  // out-of-range rows already sit on the discarded slot, so no row is skipped.
  CheckCnnl(
      cnnlEmbeddingBackward(
          handle, /*padding_idx=*/-1, scale_grad_by_freq, idx_desc.get(),
          target.const_data_ptr(), grad_wide_desc.get(),
          widen ? grad_fp32.const_data_ptr() : grad_dense.const_data_ptr(),
          workspace_size > 0 ? workspace.mutable_data_ptr() : nullptr,
          workspace_size, out_wide_desc.get(), wide.mutable_data_ptr()),
      "cnnlEmbeddingBackward");

  const auto kept =
      wide.narrow(0, 0, num_weights).reshape({num_weights, width}).contiguous();
  if (widen) {
    result.copy_(kept);
    return result;
  }
  return kept;
}

} // namespace at::native::flagos::cambricon

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    EmbeddingDenseBackwardFn, embedding_dense_backward_dispatcher,
    Backend::kCambricon, cambricon::EmbeddingDenseBackward);

} // namespace at::native::flagos

#endif // USE_CAMBRICON