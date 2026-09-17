// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include "op_preparation.h"
#include "op_api_common.h"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/SDPBackend.h>
#include <limits>

// _fused_sdp_choice stub for PrivateUse1. PyTorch's scaled_dot_product_attention
// queries _fused_sdp_choice_stub via is_device_supported(PrivateUse1) to pick a
// fused backend. Registering this stub makes the device "supported" and returns
// efficient_attention (2), routing to _scaled_dot_product_efficient_attention
// (our aclnnFlashAttentionScore kernel) instead of the math decomposition.
// Must live in at::native for REGISTER_PRIVATEUSE1_DISPATCH (the macro references
// the unqualified stub symbol and declares its registrar in the enclosing ns).
namespace at::native {
static int64_t fused_sdp_choice_ascend(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  return static_cast<int64_t>(at::SDPBackend::efficient_attention);
}

REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &fused_sdp_choice_ascend);
} // namespace at::native

namespace at::native::flagos::ascend {

// Forward: _scaled_dot_product_efficient_attention(query, key, value, attn_bias,
//          compute_log_sumexp, dropout_p, is_causal, scale?)
// Returns: (output, log_sumexp, philox_seed, philox_offset)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrivScaledDotProductEfficientAttentionKernelAscend(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {

  namespace ascend = at::native::flagos::ascend;

  // Input validation
  TORCH_CHECK(query.dim() == 4, "query must be 4D [B, N, S, D]");
  TORCH_CHECK(key.dim() == 4 && value.dim() == 4, "key/value must be 4D [B, N, S, D]");
  TORCH_CHECK(query.is_privateuseone(), "SDPA Ascend: inputs must be on NPU");
  TORCH_CHECK(dropout_p == 0.0, "SDPA Ascend: dropout not yet supported (aclnn requires explicit mask handling)");

  int64_t B = query.size(0);
  int64_t N = query.size(1);  // num query heads
  int64_t S = query.size(2);  // seq_len
  int64_t D = query.size(3);  // head_dim

  // Grouped-query / multi-query attention: key and value may carry fewer heads
  // than the query (Qwen3 uses num_kv_heads < num_attention_heads). PyTorch's
  // SDPA repeats the kv heads (repeat_kv) before the math path; the aclnn flash
  // kernel expects q/k/v with matching head counts, so replicate each kv head
  // N/N_kv times along dim 1 to match the query. Contiguous so the aclnn tensor
  // wrapper sees a dense [B, N, S, D] buffer.
  // key/value carry their own seq_len (S_kv), which differs from the query's S
  // during incremental decode (query S==1, kv S==full context). Expand only the
  // head dim, preserving each tensor's own seq_len.
  at::Tensor key_eff = key;
  at::Tensor value_eff = value;
  int64_t N_kv = key.size(1);
  int64_t S_kv = key.size(2);
  if (N_kv != N) {
    TORCH_CHECK(N_kv > 0 && N % N_kv == 0,
        "SDPA Ascend GQA: query heads (", N, ") must be a multiple of kv heads (", N_kv, ")");
    int64_t repeat = N / N_kv;
    // [B, N_kv, S_kv, D] -> [B, N_kv, repeat, S_kv, D] -> [B, N, S_kv, D]
    key_eff = key.unsqueeze(2).expand({B, N_kv, repeat, S_kv, D}).reshape({B, N, S_kv, D}).contiguous();
    value_eff = value.unsqueeze(2).expand({B, N_kv, repeat, S_kv, D}).reshape({B, N, S_kv, D}).contiguous();
  }
  TORCH_CHECK(key_eff.size(1) == N && value_eff.size(1) == N,
      "GQA expand failed to match query head count");

  // Compute scale (default: 1/sqrt(D))
  double scale_value = scale.value_or(1.0 / std::sqrt(static_cast<double>(D)));

  // Dropout parameters
  double keep_prob = 1.0 - dropout_p;
  at::Tensor drop_mask;  // Leave undefined - aclnn will generate internally based on keep_prob

  // Sparse mode and token range for causal vs full attention
  int64_t sparse_mode;
  int64_t pre_tokens;
  int64_t next_tokens;
  at::Tensor atten_mask;

  if (is_causal) {
    sparse_mode = 0;  // disable sparse, use explicit mask
    pre_tokens = 65536;
    next_tokens = 65536;
    // Generate causal mask on CPU then move to device
    // (ones/triu not registered for Ascend backend)
    auto mask_2d = at::triu(
        at::ones({S, S_kv}, at::TensorOptions().dtype(at::kBool)), 1);
    atten_mask = mask_2d.unsqueeze(0).unsqueeze(0).to(query.device());
  } else {
    sparse_mode = 0;  // full bidirectional
    pre_tokens = 65536;
    next_tokens = 65536;
    atten_mask = at::Tensor();  // null
  }

  // PyTorch boolean masks use True for allowed positions, while ACLNN consumes
  // an exclusion mask. Additive attention bias is not a mask; keep rejecting it
  // until the CANN realShift contract can be supported without semantic loss.
  if (attn_bias.has_value() && attn_bias->defined()) {
    if (attn_bias->scalar_type() == at::kBool) {
      auto bias_mask = at::logical_not(*attn_bias);
      if (atten_mask.defined()) {
        atten_mask = at::logical_or(atten_mask, bias_mask);
      } else {
        atten_mask = bias_mask;
      }
    } else {
      // PyTorch lowers a public boolean allow-mask to an additive 0/-inf bias
      // before calling this efficient-attention overload. Accept that exact
      // representation, but do not reinterpret finite additive values as a
      // mask because doing so changes softmax semantics.
      auto is_zero = at::eq(*attn_bias, 0);
      auto is_neg_inf = at::eq(
          *attn_bias, -std::numeric_limits<double>::infinity());
      auto is_mask_value = at::logical_or(is_zero, is_neg_inf);
      TORCH_CHECK(is_mask_value.all().item<bool>(),
                  "SDPA Ascend: finite additive attention bias is not yet "
                  "supported");
      if (atten_mask.defined()) {
        atten_mask = at::logical_or(atten_mask, is_neg_inf);
      } else {
        atten_mask = is_neg_inf;
      }
    }
  }

  // Allocate output tensors
  auto output = ascend::OpPreparation::apply_tensor_without_format(
      query.sizes().vec(), query.options());

  // softmaxMax and softmaxSum: [B, N, S, 8] float32
  auto softmax_max = ascend::OpPreparation::apply_tensor_without_format(
      {B, N, S, 8}, query.options().dtype(at::kFloat));
  auto softmax_sum = ascend::OpPreparation::apply_tensor_without_format(
      {B, N, S, 8}, query.options().dtype(at::kFloat));

  // Prepare aclnn arguments
  AclTensorWrapper q_wrap(query);
  AclTensorWrapper k_wrap(key_eff);
  AclTensorWrapper v_wrap(value_eff);
  AclTensorWrapper mask_wrap(atten_mask);
  AclTensorWrapper drop_mask_wrap(drop_mask);
  AclTensorWrapper softmax_max_wrap(softmax_max);
  AclTensorWrapper softmax_sum_wrap(softmax_sum);
  AclTensorWrapper output_wrap(output);

  char input_layout[] = "BNSD";

  // Call aclnnFlashAttentionScore
  EXEC_ASCEND_CMD(
      aclnnFlashAttentionScore,
      q_wrap.get(),
      k_wrap.get(),
      v_wrap.get(),
      nullptr,  // realShiftOptional
      drop_mask_wrap.get(),  // dropMaskOptional
      nullptr,  // paddingMaskOptional
      mask_wrap.get(),  // attenMaskOptional
      nullptr,  // prefixOptional
      scale_value,
      keep_prob,
      pre_tokens,
      next_tokens,
      N,  // headNum
      input_layout,
      1,  // innerPrecise (1 for fp16/bf16)
      sparse_mode,
      softmax_max_wrap.get(),
      softmax_sum_wrap.get(),
      nullptr,  // softmaxOutOut (not needed)
      output_wrap.get()
  );

  // Construct log_sumexp from softmaxMax and softmaxSum
  // logsumexp = log(softmaxSum) + softmaxMax
  // Shape: [B, N, S, 8] -> reduce to [B, N, S] by taking first element
  // NOTE: This is a simplification; the full [B,N,S,8] carries tiling state
  // that might be needed for exact backward. For now, take [:,:,:,0].
  at::Tensor log_sumexp;
  if (compute_log_sumexp) {
    // Move to CPU, extract [:,:,:,0], compute logsumexp, move back
    // (slice/narrow/select all unregistered on ascend backend)
    auto softmax_sum_cpu = softmax_sum.cpu();
    auto softmax_max_cpu = softmax_max.cpu();

    auto softmax_sum_0 = softmax_sum_cpu.select(3, 0);  // [B, N, S]
    auto softmax_max_0 = softmax_max_cpu.select(3, 0);  // [B, N, S]

    auto log_sumexp_cpu = at::log(softmax_sum_0) + softmax_max_0;
    log_sumexp = log_sumexp_cpu.to(query.device());
  } else {
    log_sumexp = at::empty({0}, query.options());
  }

  // philox_seed and philox_offset (for dropout RNG state)
  // Since we don't support dropout yet, return dummy tensors
  // Create on CPU first to avoid unregistered fill_ on NPU
  auto philox_seed = at::scalar_tensor(0, at::dtype(at::kLong)).to(query.device());
  auto philox_offset = at::scalar_tensor(0, at::dtype(at::kLong)).to(query.device());

  // Store softmaxMax and softmaxSum in output for backward retrieval
  // HACK: We can't modify PyTorch's autograd ctx from here, so we'll need
  // a custom autograd Function wrapper in Python or store these globally.
  // For now, return them as-is and document the limitation.
  // TODO: Implement proper autograd Function wrapper that saves both tensors.

  return std::make_tuple(output, log_sumexp, philox_seed, philox_offset);
}

// Backward: _scaled_dot_product_efficient_attention_backward(
//   grad_out, query, key, value, attn_bias, output, logsumexp,
//   philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal, scale?)
// Returns: (grad_query, grad_key, grad_value, grad_attn_bias)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrivScaledDotProductEfficientAttentionBackwardKernelAscend(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    const at::Tensor& output,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool is_causal,
    std::optional<double> scale) {

  namespace ascend = at::native::flagos::ascend;

  // Input validation
  TORCH_CHECK(query.dim() == 4, "query must be 4D [B, N, S, D]");
  TORCH_CHECK(query.is_privateuseone(), "SDPA Ascend backward: inputs must be on NPU");
  TORCH_CHECK(dropout_p == 0.0, "SDPA Ascend backward: dropout not yet supported");

  int64_t B = query.size(0);
  int64_t N = query.size(1);
  int64_t S = query.size(2);
  int64_t D = query.size(3);

  double scale_value = scale.value_or(1.0 / std::sqrt(static_cast<double>(D)));
  double keep_prob = 1.0 - dropout_p;

  // Dropout mask: let aclnn generate internally based on keep_prob
  // (In production, should retrieve the exact forward mask from autograd ctx)
  at::Tensor drop_mask;  // undefined -> nullptr to aclnn

  // Reconstruct causal mask and parameters (must match forward)
  int64_t sparse_mode = 0;
  int64_t pre_tokens = 65536;
  int64_t next_tokens = 65536;
  at::Tensor atten_mask;
  if (is_causal) {
    // Generate causal mask on CPU then move to device
    auto mask_2d = at::triu(at::ones({S, S}, at::TensorOptions().dtype(at::kBool)), 1);
    atten_mask = mask_2d.unsqueeze(0).unsqueeze(0).to(query.device());
  } else {
    atten_mask = at::Tensor();
  }

  // RECOMPUTATION: Run forward again to get softmaxMax and softmaxSum.
  // This trades compute for memory (activation checkpointing strategy).
  // PyTorch's autograd only gives us logsumexp, but aclnn backward needs the
  // full [B,N,S,8] softmaxMax and softmaxSum tensors.
  auto softmax_max = ascend::OpPreparation::apply_tensor_without_format(
      {B, N, S, 8}, query.options().dtype(at::kFloat));
  auto softmax_sum = ascend::OpPreparation::apply_tensor_without_format(
      {B, N, S, 8}, query.options().dtype(at::kFloat));
  auto output_recompute = ascend::OpPreparation::apply_tensor_without_format(
      query.sizes().vec(), query.options());

  AclTensorWrapper q_wrap(query);
  AclTensorWrapper k_wrap(key);
  AclTensorWrapper v_wrap(value);
  AclTensorWrapper mask_wrap(is_causal ? atten_mask : at::Tensor());
  AclTensorWrapper drop_mask_wrap(drop_mask);
  AclTensorWrapper softmax_max_wrap(softmax_max);
  AclTensorWrapper softmax_sum_wrap(softmax_sum);
  AclTensorWrapper output_recompute_wrap(output_recompute);

  char input_layout[] = "BNSD";

  EXEC_ASCEND_CMD(
      aclnnFlashAttentionScore,
      q_wrap.get(),
      k_wrap.get(),
      v_wrap.get(),
      nullptr,  // realShiftOptional
      drop_mask_wrap.get(),  // dropMaskOptional
      nullptr,  // paddingMaskOptional
      mask_wrap.get(),
      nullptr,  // prefixOptional
      scale_value,
      keep_prob,
      pre_tokens,
      next_tokens,
      N,
      input_layout,
      1,  // innerPrecise
      sparse_mode,
      softmax_max_wrap.get(),
      softmax_sum_wrap.get(),
      nullptr,  // softmaxOutOut
      output_recompute_wrap.get()
  );

  // Allocate gradient outputs
  auto grad_query = ascend::OpPreparation::apply_tensor_without_format(
      query.sizes().vec(), query.options());
  auto grad_key = ascend::OpPreparation::apply_tensor_without_format(
      key.sizes().vec(), key.options());
  auto grad_value = ascend::OpPreparation::apply_tensor_without_format(
      value.sizes().vec(), value.options());

  AclTensorWrapper grad_out_wrap(grad_out);
  AclTensorWrapper grad_query_wrap(grad_query);
  AclTensorWrapper grad_key_wrap(grad_key);
  AclTensorWrapper grad_value_wrap(grad_value);

  // Call aclnnFlashAttentionScoreGrad
  EXEC_ASCEND_CMD(
      aclnnFlashAttentionScoreGrad,
      q_wrap.get(),
      k_wrap.get(),
      v_wrap.get(),
      grad_out_wrap.get(),  // dy
      nullptr,  // pseShiftOptional
      drop_mask_wrap.get(),  // dropMaskOptional
      nullptr,  // paddingMaskOptional
      mask_wrap.get(),  // attenMaskOptional
      softmax_max_wrap.get(),  // from recomputation
      softmax_sum_wrap.get(),  // from recomputation
      nullptr,  // softmaxInOptional
      output_recompute_wrap.get(),  // attentionInOptional (use recomputed)
      nullptr,  // prefixOptional
      scale_value,
      keep_prob,
      pre_tokens,
      next_tokens,
      N,
      input_layout,
      1,  // innerPrecise
      sparse_mode,
      grad_query_wrap.get(),
      grad_key_wrap.get(),
      grad_value_wrap.get(),
      nullptr  // dpseOut
  );

  auto grad_attn_bias = at::empty({0}, query.options());
  return std::make_tuple(grad_query, grad_key, grad_value, grad_attn_bias);
}

// Register to dispatcher
REGISTER_IMPL_TO_DISPATCHER(
    PrivScaledDotProductEfficientAttentionFn,
    priv_scaled_dot_product_efficient_attention_dispatcher,
    Backend::kAscend,
    PrivScaledDotProductEfficientAttentionKernelAscend)

REGISTER_IMPL_TO_DISPATCHER(
    PrivScaledDotProductEfficientAttentionBackwardFn,
    priv_scaled_dot_product_efficient_attention_backward_dispatcher,
    Backend::kAscend,
    PrivScaledDotProductEfficientAttentionBackwardKernelAscend)

} // namespace at::native::flagos::ascend
