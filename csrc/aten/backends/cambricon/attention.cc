// Copyright (c) 2026, BAAI. All rights reserved.
//
// Fused scaled-dot-product attention on Cambricon, through CNNL's flash
// attention.
//
// Why this is a vendor kernel and not a FlagGems route. The decomposition every
// other path here takes is `q @ k^T -> softmax -> @ v`, which materialises the
// B*H*S*S score matrix in HBM twice per layer -- 256 MB per layer at batch 2,
// 4 heads and seq 4096 -- and it reads it back for the second matmul. The fused
// kernel tiles over the sequence and never writes it out. On the backward side
// the same difference applies to the stored probabilities: this one keeps only
// the log-sum-exp, per (head, token).
//
// FlagGems does ship a cambricon attention under
// runtime/backend/_cambricon/ops/attention.py, but it cannot load on this
// platform: importing it pulls in fused/FLA, which assumes torch_mlu's device
// management and takes the process down with it. That is the same root cause as
// flagos-ai/FlagGems#8448 -- the backend is written against torch_mlu, and this
// install is torch+cpu plus a PrivateUse1 shim.
//
// The CNNL entry points are cnnlFlashAttentionForward_v2 and
// cnnlFlashAttentionBackward_v2 from libcnnl_extra. Their only prior art is
// torch_mlu's own csrc/aten/operators/cnnl/internal/attention_internal.cpp; the
// argument shapes below (the descriptor order, the mandatory cu_seqlens even in
// the packed non-varlen case, and softmax_lse carrying heads on dim 0) were
// taken from there and then confirmed against a host reference before this file
// was written. Both directions were probed standalone first: forward exact to
// 3.8e-05, backward dq/dk to 5e-07 and dv to 2e-05.

#include "aten/backends/cambricon/cambricon_common.h"
#include "aten/backends/cambricon/attention.h"

#ifdef USE_CAMBRICON
#ifdef FLAGOS_CAMBRICON_CNNL_EXTRA

#include <ATen/ops/empty.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cnnl_extra.h>

#include <array>
#include <cmath>

namespace at::native::flagos::cambricon {

namespace {

// The layout the CNNL attention describes is [total_tokens, heads, D] with
// heads packed into the token axis (PACKED + is_pack_mode), which is what
// torch_mlu uses for every non-varlen call. aten hands over
// [batch, heads, seq, D], so operands are transposed into the packed form and
// results are transposed back.
at::Tensor ToPacked(const at::Tensor& src) {
  // (B, H, S, D) -> (B*S, H, D)
  return src.permute({0, 2, 1, 3}).reshape(
      {src.size(0) * src.size(2), src.size(1), src.size(3)});
}

at::Tensor FromPacked(const at::Tensor& src, int64_t batch, int64_t heads,
                      int64_t seq, int64_t head_dim) {
  // (B*S, H, D) -> (B, H, S, D)
  return src.reshape({batch, seq, heads, head_dim}).permute({0, 2, 1, 3});
}

// Owns the flash-attention descriptor so both directions can fail out of any
// point between the create and the call without leaking it.
class Descriptor {
 public:
  Descriptor() {
    CheckCnnl(
        cnnlCreateFlashAttentionDescriptor(&desc_),
        "cnnlCreateFlashAttentionDescriptor");
  }
  ~Descriptor() {
    if (desc_ != nullptr) {
      cnnlDestroyFlashAttentionDescriptor(desc_);
    }
  }

  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;

  cnnlFlashAttentionDescriptor_t get() const { return desc_; }

 private:
  cnnlFlashAttentionDescriptor_t desc_ = nullptr;
};

// cu_seqlens, cumulative sequence lengths, [0, S, 2S, ...]. Both calls require
// it even when there is one sequence per batch (is_pack_mode without varlen); a
// null descriptor is rejected with a bare status 3 that does not say which
// argument was wrong.
at::Tensor MakeCuSeqlens(int64_t batch, int64_t seq, const at::Tensor& like) {
  return at::arange(0, (batch + 1) * seq, seq, like.options().dtype(at::kInt));
}

void CreateForwardDescriptor(Descriptor& fa, int64_t seq_q, int64_t seq_kv,
                             bool is_causal, double scale) {
  CheckCnnl(
      cnnlSetFlashAttentionSlidingWindowSize(fa.get(), -1, -1, 1),
      "cnnlSetFlashAttentionSlidingWindowSize");
  CheckCnnl(
      cnnlSetFlashAttentionDescriptor_v2(
          fa.get(),
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
}

void CreateBackwardDescriptor(Descriptor& fa, int64_t seq_q, int64_t seq_kv,
                              bool is_causal, double scale) {
  CheckCnnl(
      cnnlSetFlashAttentionSlidingWindowSize(fa.get(), -1, -1, 1),
      "cnnlSetFlashAttentionSlidingWindowSize");
  CheckCnnl(
      cnnlSetFlashAttentionBackwardDescriptor_v2(
          fa.get(),
          CNNL_DTYPE_FLOAT,
          CNNL_COMPUTATION_HIGH_PRECISION,
          is_causal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE,
          CNNL_ATTN_TENSOR_LAYOUT_PACKED,
          /*is_pack_mode=*/true,
          /*is_out_zero=*/false,
          /*is_store_softmax_d=*/false,
          static_cast<int>(seq_q),
          static_cast<int>(seq_kv),
          /*p_dropout=*/0.0f,
          static_cast<float>(scale)),
      "cnnlSetFlashAttentionBackwardDescriptor_v2");
}

void CheckShapes(const at::Tensor& query, const at::Tensor& key,
                 const at::Tensor& value) {
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
  TORCH_CHECK(
      key.size(0) == query.size(0) && value.size(0) == query.size(0) &&
          key.size(1) == query.size(1) && value.size(1) == query.size(1),
      "flagos cambricon: fused attention requires matching batch and heads");
  TORCH_CHECK(
      value.size(2) == key.size(2) && value.size(3) == query.size(3),
      "flagos cambricon: fused attention requires matching key/value shapes");
}

} // namespace

FusedAttentionOutput Forward(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double scale, bool is_causal) {
  CheckShapes(query, key, value);

  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t seq_q = query.size(2);
  const int64_t seq_kv = key.size(2);
  const int64_t head_dim = query.size(3);

  CambriconDeviceGuard guard(query);
  const cnnlHandle_t handle =
      HandleForDevice(static_cast<int>(query.device().index()));

  // Everything a descriptor describes has to be dense; the packed transposes
  // above produce strided views otherwise.
  at::Tensor q = ToPacked(query).contiguous();
  at::Tensor k = ToPacked(key).contiguous();
  at::Tensor v = ToPacked(value).contiguous();

  FusedAttentionOutput result;
  at::Tensor out_packed =
      at::empty({batch * seq_q, heads, head_dim}, query.options());
  // LSE carries heads on dim 0 -- the API rejects anything else, with the
  // message "softmax_lse's head_num(dims[0]) must be equal to query's
  // head_num(dims[1])".
  result.lse = at::empty(
      {heads, batch * seq_q}, query.options().dtype(at::kFloat));
  at::Tensor cu_seqlens = MakeCuSeqlens(batch, seq_q, query);

  Descriptor fa;
  CreateForwardDescriptor(fa, seq_q, seq_kv, is_causal, scale);

  TensorDesc q_desc, k_desc, v_desc, out_desc, lse_desc, cu_desc;
  q_desc.Set(q);
  k_desc.Set(k);
  v_desc.Set(v);
  out_desc.Set(out_packed);
  lse_desc.Set(result.lse);
  cu_desc.Set(cu_seqlens);

  size_t workspace_size = 0;
  CheckCnnl(
      cnnlGetFlashAttentionForwardWorkspaceSize_v2(
          handle, fa.get(),
          q_desc.get(), k_desc.get(), v_desc.get(),
          cu_desc.get(), cu_desc.get(),
          /*seqused_k_desc=*/nullptr,
          /*attn_mask_desc=*/nullptr,
          /*alibi_slopes_desc=*/nullptr,
          lse_desc.get(),
          &workspace_size),
      "cnnlGetFlashAttentionForwardWorkspaceSize_v2");

  at::Tensor workspace;
  if (workspace_size > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        query.options().dtype(at::kByte));
  }

  CheckCnnl(
      cnnlFlashAttentionForward_v2(
          handle, fa.get(),
          q_desc.get(), q.const_data_ptr(),
          k_desc.get(), k.const_data_ptr(),
          v_desc.get(), v.const_data_ptr(),
          cu_desc.get(), cu_seqlens.const_data_ptr(),
          cu_desc.get(), cu_seqlens.const_data_ptr(),
          /*seqused_k_desc=*/nullptr, /*seqused_k=*/nullptr,
          /*attn_mask_desc=*/nullptr, /*attn_mask=*/nullptr,
          /*alibi_slopes_desc=*/nullptr, /*alibi_slopes=*/nullptr,
          /*rng_state=*/nullptr,
          workspace_size > 0 ? workspace.mutable_data_ptr() : nullptr,
          workspace_size,
          /*dropout_mask_desc=*/nullptr, /*dropout_mask=*/nullptr,
          lse_desc.get(), result.lse.mutable_data_ptr(),
          out_desc.get(), out_packed.mutable_data_ptr()),
      "cnnlFlashAttentionForward_v2");

  result.out = FromPacked(out_packed, batch, heads, seq_q, head_dim);
  return result;
}

std::array<at::Tensor, 3> Backward(
    const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& out, const at::Tensor& lse,
    double scale, bool is_causal) {
  CheckShapes(query, key, value);

  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t seq_q = query.size(2);
  const int64_t seq_kv = key.size(2);
  const int64_t head_dim = query.size(3);

  CambriconDeviceGuard guard(query);
  const cnnlHandle_t handle =
      HandleForDevice(static_cast<int>(query.device().index()));

  at::Tensor q = ToPacked(query).contiguous();
  at::Tensor k = ToPacked(key).contiguous();
  at::Tensor v = ToPacked(value).contiguous();
  at::Tensor o = ToPacked(out).contiguous();
  at::Tensor do_ = ToPacked(grad_out).contiguous();
  at::Tensor cu_seqlens = MakeCuSeqlens(batch, seq_q, query);

  at::Tensor dq =
      at::empty({batch * seq_q, heads, head_dim}, query.options());
  at::Tensor dk =
      at::empty({batch * seq_q, heads, head_dim}, query.options());
  at::Tensor dv =
      at::empty({batch * seq_q, heads, head_dim}, query.options());

  Descriptor fa;
  CreateBackwardDescriptor(fa, seq_q, seq_kv, is_causal, scale);

  TensorDesc q_desc, k_desc, v_desc, o_desc, do_desc, lse_desc, cu_desc;
  TensorDesc dq_desc, dk_desc, dv_desc;
  q_desc.Set(q);
  k_desc.Set(k);
  v_desc.Set(v);
  o_desc.Set(o);
  do_desc.Set(do_);
  lse_desc.Set(lse);
  cu_desc.Set(cu_seqlens);
  dq_desc.Set(dq);
  dk_desc.Set(dk);
  dv_desc.Set(dv);

  size_t workspace_size = 0;
  CheckCnnl(
      cnnlGetFlashAttentionBackwardWorkspaceSize_v2(
          handle, fa.get(), q_desc.get(), k_desc.get(), v_desc.get(),
          cu_desc.get(), &workspace_size),
      "cnnlGetFlashAttentionBackwardWorkspaceSize_v2");

  at::Tensor workspace;
  if (workspace_size > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        query.options().dtype(at::kByte));
  }

  CheckCnnl(
      cnnlFlashAttentionBackward_v2(
          handle, fa.get(),
          do_desc.get(), do_.const_data_ptr(),
          q_desc.get(), q.const_data_ptr(),
          k_desc.get(), k.const_data_ptr(),
          v_desc.get(), v.const_data_ptr(),
          o_desc.get(), o.const_data_ptr(),
          lse_desc.get(), lse.const_data_ptr(),
          cu_desc.get(), cu_seqlens.const_data_ptr(),
          cu_desc.get(), cu_seqlens.const_data_ptr(),
          /*attn_mask_desc=*/nullptr, /*attn_mask=*/nullptr,
          /*alibi_slopes_desc=*/nullptr, /*alibi_slopes=*/nullptr,
          /*rng_state=*/nullptr,
          workspace_size > 0 ? workspace.mutable_data_ptr() : nullptr,
          workspace_size,
          dq_desc.get(), dq.mutable_data_ptr(),
          dk_desc.get(), dk.mutable_data_ptr(),
          dv_desc.get(), dv.mutable_data_ptr(),
          /*dropout_mask_desc=*/nullptr, /*dropout_mask=*/nullptr,
          /*softmax_d_desc=*/nullptr, /*softmax_d=*/nullptr),
      "cnnlFlashAttentionBackward_v2");

  // `.contiguous()` because the packed->aten conversion is a permute: a view
  // handed back to the autograd engine as a gradient is read for its storage,
  // and a strided result is both slower to consume and, on this backend, does
  // not survive the trip back to the owning device.
  return {
      FromPacked(dq, batch, heads, seq_q, head_dim).contiguous(),
      FromPacked(dk, batch, heads, seq_kv, head_dim).contiguous(),
      FromPacked(dv, batch, heads, seq_kv, head_dim).contiguous(),
  };
}

// autograd.Function, in C++.
//
// The forward keeps q/k/v, its output and the log-sum-exp. That is the point of
// a fused kernel on this side too: the backward recomputes the softmax from LSE
// instead of receiving a stored B*H*S*S probability matrix, which is what the
// decomposition has to keep alive between the two passes.
class ScaledDotProductAttentionFunction
    : public torch::autograd::Function<ScaledDotProductAttentionFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx, const at::Tensor& query,
      const at::Tensor& key, const at::Tensor& value, double scale,
      bool is_causal) {
    FusedAttentionOutput result = Forward(query, key, value, scale, is_causal);
    ctx->save_for_backward({query, key, value, result.out, result.lse});
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["is_causal"] = is_causal;
    return result.out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const double scale = ctx->saved_data["scale"].toDouble();
    const bool is_causal = ctx->saved_data["is_causal"].toBool();

    auto grads = Backward(grad_outputs[0], saved[0], saved[1], saved[2],
                          saved[3], saved[4], scale, is_causal);
    // One entry per forward argument. query, key and value take gradients;
    // `scale` and `is_causal` are not tensors and take none. The arity has to
    // match -- a short list is rejected, and spelling the absent slots as
    // `at::Tensor()` puts an uninitialised tensor there, which the engine reads
    // as "a gradient on the wrong device".
    torch::autograd::variable_list result(5);
    result[0] = grads[0];
    result[1] = grads[1];
    result[2] = grads[2];
    return result;
  }
};

at::Tensor ScaledDotProductAttention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    double scale, bool is_causal) {
  return ScaledDotProductAttentionFunction::apply(
      query, key, value, scale, is_causal);
}

} // namespace at::native::flagos::cambricon

#endif // FLAGOS_CAMBRICON_CNNL_EXTRA
#endif // USE_CAMBRICON