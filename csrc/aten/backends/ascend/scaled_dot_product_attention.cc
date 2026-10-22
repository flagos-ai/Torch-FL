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
//
// The aclnn fused kernel has no dropout plumbing: dropout_p is passed through as
// keep_prob but no drop_mask / seed / offset is ever supplied, so a fused call
// with dropout_p > 0 would silently run *without* dropout -- a correctness hole,
// not just an unimplemented feature. The forward and backward kernels therefore
// reject dropout_p != 0. Since the composite asks this selector which backend to
// use before it reaches that check, a fused choice here turns every
// dropout-bearing training forward (e.g. a default BertConfig, whose
// attention_probs_dropout_prob is 0.1) into a hard RuntimeError.
//
// Decline the fused backend instead when dropout is requested: returning math
// sends the call to ATen's composite math decomposition
// (_scaled_dot_product_attention_math), which applies dropout correctly. This
// mirrors how a backend declines an unsupported case rather than hard-failing.
// dropout_p == 0.0 (inference, and the eval-mode BERT suite) still takes the
// fused aclnn path unchanged.
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
  if (dropout_p != 0.0) {
    return static_cast<int64_t>(at::SDPBackend::math);
  }
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
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(query));

  namespace ascend = at::native::flagos::ascend;

  // Input validation
  TORCH_CHECK(query.dim() == 4, "query must be 4D [B, N, S, D]");
  TORCH_CHECK(key.dim() == 4 && value.dim() == 4, "key/value must be 4D [B, N, S, D]");
  TORCH_CHECK(query.is_privateuseone(), "SDPA Ascend: inputs must be on NPU");
  // The aclnn fused kernel does not plumb dropout (no drop_mask/seed/offset), so
  // fused_sdp_choice_ascend declines this backend whenever dropout_p != 0 and
  // the composite takes its math decomposition instead. Reaching this line with
  // dropout therefore means the fused kernel was called directly, bypassing the
  // selector -- keep it as a guard rather than silently ignoring the dropout.
  TORCH_CHECK(dropout_p == 0.0, "SDPA Ascend: the fused aclnn path does not support dropout; dropout_p > 0 must go through the math backend (see fused_sdp_choice_ascend)");

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

  // PyTorch funnels both attention masking and attention bias through the one
  // attn_bias argument, while ACLNN models them as two inputs, and choosing the
  // wrong one silently changes the softmax:
  //   * a boolean tensor is an allow-mask (True = attend), and ACLNN's
  //     attenMask is an exclusion mask (True = drop), so it is negated;
  //   * an additive float tensor holding only 0 and -inf is exactly a mask too,
  //     and stays on the attenMask path where it composes with the causal mask
  //     for free;
  //   * any other additive tensor carries finite values. Those shift the
  //     logits rather than select them, so they go to ACLNN's realShift input.
  //     ACLNN applies that input to the *unscaled* scores -- measured: the
  //     kernel computes (QK^T + realShift) * scaleValue, while PyTorch adds
  //     attn_bias to the scores *after* scaling. The bias is therefore
  //     pre-divided by scaleValue so the two agree at any scale.
  at::Tensor pse_shift;  // ACLNN realShiftOptional
  if (attn_bias.has_value() && attn_bias->defined()) {
    TORCH_CHECK(attn_bias->dim() >= 2 && attn_bias->dim() <= 4,
                "SDPA Ascend: attn_bias must broadcast to [B, N, S, S_kv], got ",
                attn_bias->dim(), " dimensions");
    if (attn_bias->scalar_type() == at::kBool) {
      auto bias_mask = at::logical_not(*attn_bias);
      if (atten_mask.defined()) {
        atten_mask = at::logical_or(atten_mask, bias_mask);
      } else {
        atten_mask = bias_mask;
      }
    } else {
      TORCH_CHECK(attn_bias->scalar_type() == at::kHalf ||
                      attn_bias->scalar_type() == at::kBFloat16 ||
                      attn_bias->scalar_type() == at::kFloat,
                  "SDPA Ascend: attn_bias must be bool, float16, bfloat16 or "
                  "float32, got ", attn_bias->scalar_type());
      // PyTorch lowers a public boolean allow-mask to an additive 0/-inf bias
      // before calling this efficient-attention overload. That exact
      // representation is still a mask, and only pure masks take the cheap
      // path: reinterpreting an arbitrary finite value as an exclusion bit
      // would change softmax semantics instead of shifting the logits.
      auto is_neg_inf = at::eq(
          *attn_bias, -std::numeric_limits<double>::infinity());
      auto is_finite = at::logical_and(
          at::logical_not(is_neg_inf), at::ne(*attn_bias, 0));
      if (!is_finite.any().item<bool>()) {
        if (atten_mask.defined()) {
          atten_mask = at::logical_or(atten_mask, is_neg_inf);
        } else {
          atten_mask = is_neg_inf;
        }
      } else {
        TORCH_CHECK(scale_value != 0.0,
                    "SDPA Ascend: scale=0 cannot carry an additive attention "
                    "bias through the realShift input");
        // A real bias. Its -inf entries are still an exclusion mask -- letting
        // an infinity reach the shift add only risks a NaN in the online
        // softmax -- while the finite rest goes to realShift.
        if (is_neg_inf.any().item<bool>()) {
          if (atten_mask.defined()) {
            atten_mask = at::logical_or(atten_mask, is_neg_inf);
          } else {
            atten_mask = is_neg_inf;
          }
        }
        // Rescale while the tensor is still small: dividing after the expand
        // would allocate a second full [B, N, S, S_kv] buffer. Done in fp32
        // because 1/sqrt(D) is not exact in fp16/bf16.
        auto bias_f32 =
            at::where(is_neg_inf, at::zeros_like(*attn_bias), *attn_bias)
                .to(at::kFloat) /
            scale_value;
        // Right-align the broadcast dims, matching how PyTorch broadcasts a
        // mask against [B, N, S, S_kv], then materialize the shape ACLNN wants.
        while (bias_f32.dim() < 4) {
          bias_f32 = bias_f32.unsqueeze(0);
        }
        // .to() before .contiguous(), not after: a dtype change already copies
        // into a fresh buffer, so together they allocate once. Reversed, a fp32
        // query with a fp32 bias would take .to() as the no-op it is, leaving
        // .contiguous() to materialize the stride-0 expand on its own -- and
        // .contiguous() alone would leave a dtype change to copy a second time.
        pse_shift = bias_f32.expand({B, N, S, S_kv})
                        .to(query.scalar_type())
                        .contiguous();
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
  AclTensorWrapper pse_wrap(pse_shift);
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
      pse_wrap.get(),  // realShiftOptional
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
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(grad_out));

  namespace ascend = at::native::flagos::ascend;

  // Input validation
  TORCH_CHECK(query.dim() == 4, "query must be 4D [B, N, S, D]");
  TORCH_CHECK(query.is_privateuseone(), "SDPA Ascend backward: inputs must be on NPU");
  // Same invariant as the forward: fused_sdp_choice_ascend declines the fused
  // backend for dropout_p != 0, so this is only reachable on a direct call.
  TORCH_CHECK(dropout_p == 0.0, "SDPA Ascend backward: the fused aclnn path does not support dropout; dropout_p > 0 must go through the math backend (see fused_sdp_choice_ascend)");

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
