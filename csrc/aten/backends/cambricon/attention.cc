// Copyright (c) 2026, BAAI. All rights reserved.
//
// Fused scaled-dot-product attention on Cambricon, through CNNL's flash
// attention.
//
// Why this is a vendor kernel and not a FlagGems route. The decomposition every
// other path here takes is `q @ k^T -> softmax -> @ v`, which materialises the
// B*H*S*S score matrix in HBM twice per layer -- 256 MB per layer at batch 2,
// 4 heads and seq 4096, and it reads it back for the second matmul. The fused
// kernel tiles over the sequence and never writes it out.
//
// FlagGems does ship a cambricon attention under
// runtime/backend/_cambricon/ops/attention.py, but it cannot load on this
// platform: importing it pulls in fused/FLA, which assumes torch_mlu's device
// management and takes the process down with it. That is the same root cause as
// flagos-ai/FlagGems#8448 -- the backend is written against torch_mlu, and this
// install is torch+cpu plus a PrivateUse1 shim.
//
// The CNNL entry point is `cnnlFlashAttentionForward_v2` from libcnnl_extra.
// Its only prior art is torch_mlu's own
// csrc/aten/operators/cnnl/internal/attention_internal.cpp; the argument shapes
// below (the descriptor order, the mandatory cu_seqlens even in the packed
// non-varlen case, and softmax_lse carrying heads on dim 0) were taken from
// there and then confirmed against a host reference before this file was
// written.

#include "aten/backends/cambricon/cambricon_common.h"
#include "aten/backends/cambricon/attention.h"

#ifdef USE_CAMBRICON
#ifdef FLAGOS_CAMBRICON_CNNL_EXTRA

#include <ATen/ops/empty.h>

#include <cnnl_extra.h>

#include <cmath>

namespace at::native::flagos::cambricon {

namespace {

// The layout cnnlFlashAttentionForward_v2 describes is [total_tokens, heads, D]
// with heads packed into the token axis (PACKED + is_pack_mode), which is the
// one torch_mlu uses for every non-varlen call. aten's SDPA hands us
// [batch, heads, seq, D], so the operands are transposed into the packed form
// and the result is transposed back.
at::Tensor ToPacked(const at::Tensor& src) {
  // (B, H, S, D) -> (B*S, H, D)
  return src.permute({0, 2, 1, 3}).reshape({src.size(0) * src.size(2), src.size(1), src.size(3)});
}

at::Tensor FromPacked(const at::Tensor& src, int64_t batch, int64_t heads,
                      int64_t seq, int64_t head_dim) {
  // (B*S, H, D) -> (B, H, S, D)
  return src.reshape({batch, seq, heads, head_dim}).permute({0, 2, 1, 3});
}

} // namespace

at::Tensor ScaledDotProductAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double scale, bool is_causal) {
  TORCH_CHECK(
      query.dim() == 4,
      "flagos cambricon: scaled_dot_product_attention expects 4-D query "
      "(batch, heads, seq, head_dim), got ",
      query.dim(),
      "-D");
  TORCH_CHECK(
      query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
      "flagos cambricon: fused attention serves half and bfloat16, got ",
      query.scalar_type());
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          key.scalar_type() == value.scalar_type(),
      "flagos cambricon: fused attention requires matching dtypes");

  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t seq_q = query.size(2);
  const int64_t seq_kv = key.size(2);
  const int64_t head_dim = query.size(3);
  TORCH_CHECK(
      key.size(0) == batch && value.size(0) == batch &&
          key.size(1) == heads && value.size(1) == heads,
      "flagos cambricon: fused attention requires matching batch and heads");
  TORCH_CHECK(
      value.size(2) == seq_kv && value.size(3) == head_dim,
      "flagos cambricon: fused attention requires matching key/value shapes");

  CambriconDeviceGuard guard(query);
  const cnnlHandle_t handle =
      HandleForDevice(static_cast<int>(query.device().index()));

  // Everything the descriptor describes has to be dense; the packed transposes
  // above produce strided views otherwise.
  at::Tensor q = ToPacked(query).contiguous();
  at::Tensor k = ToPacked(key).contiguous();
  at::Tensor v = ToPacked(value).contiguous();

  at::Tensor out_packed =
      at::empty({batch * seq_q, heads, head_dim}, query.options());
  // LSE carries heads on dim 0 -- the API rejects anything else, with the
  // message "softmax_lse's head_num(dims[0]) must be equal to query's
  // head_num(dims[1])".
  at::Tensor lse =
      at::empty({heads, batch * seq_q}, query.options().dtype(at::kFloat));

  // cu_seqlens, cumulative sequence lengths, [0, S, 2S, ...]. The API requires
  // it even when there is one sequence per batch (is_pack_mode without varlen);
  // a null descriptor is rejected.
  at::Tensor cu_seqlens = at::arange(
      0, (batch + 1) * seq_q, seq_q,
      query.options().dtype(at::kInt));

  cnnlFlashAttentionDescriptor_t fa_desc = nullptr;
  CheckCnnl(cnnlCreateFlashAttentionDescriptor(&fa_desc), "cnnlCreateFlashAttentionDescriptor");
  // No sliding window; dilation 1 is what torch_mlu passes as a reserved value.
  CheckCnnl(
      cnnlSetFlashAttentionSlidingWindowSize(fa_desc, -1, -1, 1),
      "cnnlSetFlashAttentionSlidingWindowSize");
  CheckCnnl(
      cnnlSetFlashAttentionDescriptor_v2(
          fa_desc,
          CNNL_DTYPE_FLOAT,  // compute dtype: accumulate the scores in fp32
          CNNL_COMPUTATION_HIGH_PRECISION,
          is_causal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE,
          CNNL_ATTN_TENSOR_LAYOUT_PACKED,
          /*is_pack_mode=*/true,
          /*is_out_zero=*/false,
          /*return_softmax=*/false,
          static_cast<int>(seq_q),
          static_cast<int>(seq_kv),
          /*p_dropout=*/0.0f,
          static_cast<float>(scale)),
      "cnnlSetFlashAttentionDescriptor_v2");

  TensorDesc q_desc, k_desc, v_desc, out_desc, lse_desc, cusq_desc;
  q_desc.Set(q);
  k_desc.Set(k);
  v_desc.Set(v);
  out_desc.Set(out_packed);
  lse_desc.Set(lse);
  cusq_desc.Set(cu_seqlens);

  size_t workspace_size = 0;
  const cnnlStatus_t size_status = cnnlGetFlashAttentionForwardWorkspaceSize_v2(
      handle, fa_desc,
      q_desc.get(), k_desc.get(), v_desc.get(),
      cusq_desc.get(), cusq_desc.get(),
      /*seqused_k_desc=*/nullptr,
      /*attn_mask_desc=*/nullptr,
      /*alibi_slopes_desc=*/nullptr,
      lse_desc.get(),
      &workspace_size);
  if (size_status != CNNL_STATUS_SUCCESS) {
    cnnlDestroyFlashAttentionDescriptor(fa_desc);
    CheckCnnl(size_status, "cnnlGetFlashAttentionForwardWorkspaceSize_v2");
  }

  at::Tensor workspace;
  if (workspace_size > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        query.options().dtype(at::kByte));
  }

  const cnnlStatus_t status = cnnlFlashAttentionForward_v2(
      handle, fa_desc,
      q_desc.get(), q.const_data_ptr(),
      k_desc.get(), k.const_data_ptr(),
      v_desc.get(), v.const_data_ptr(),
      cusq_desc.get(), cu_seqlens.const_data_ptr(),
      cusq_desc.get(), cu_seqlens.const_data_ptr(),
      /*seqused_k_desc=*/nullptr, /*seqused_k=*/nullptr,
      /*attn_mask_desc=*/nullptr, /*attn_mask=*/nullptr,
      /*alibi_slopes_desc=*/nullptr, /*alibi_slopes=*/nullptr,
      /*rng_state=*/nullptr,
      workspace_size > 0 ? workspace.mutable_data_ptr() : nullptr,
      workspace_size,
      /*dropout_mask_desc=*/nullptr, /*dropout_mask=*/nullptr,
      lse_desc.get(), lse.mutable_data_ptr(),
      out_desc.get(), out_packed.mutable_data_ptr());

  cnnlDestroyFlashAttentionDescriptor(fa_desc);
  CheckCnnl(status, "cnnlFlashAttentionForward_v2");

  return FromPacked(out_packed, batch, heads, seq_q, head_dim);
}

} // namespace at::native::flagos::cambricon

#endif // FLAGOS_CAMBRICON_CNNL_EXTRA
#endif // USE_CAMBRICON