// Copyright (c) 2026, BAAI. All rights reserved.
//
// Fused scaled-dot-product attention for Enflame GCU (topsaten).
//
// Qwen-Image-2512's transformer issues 60 `F.scaled_dot_product_attention`
// calls per forward pass, and PyTorch serves every one of them with the math
// decomposition unless a backend registers the `_fused_sdp_choice` DispatchStub
// slot for PrivateUse1. The decomposition runs on device, so this is not a CPU
// fallback -- it is simply expensive: measured on an S60 at the model's own
// shape, 90.71 ms per call against 1.39 ms for the vendor flash op, and 11.637 s
// of a transformer step against 7.240 s. That is 37.8 % of the step, the single
// largest term left in it.
//
// FlagGems' own SDPA is not the answer here: measured at 133.84 ms per call it
// is slower than the math path it would replace.
//
// This is a hand-written translation unit because `codegen_gcu.py` cannot
// express two things this op needs (see
// docs/vendors/gcu/scaled-dot-product-attention.md for the full record):
//
//   1. a `REGISTER_PRIVATEUSE1_DISPATCH` registrar into an ATen DispatchStub --
//      every generator template registers into the flagos dispatcher instead;
//   2. a kernel body returning a 4-tuple -- every template writes one `at::Tensor`.
//
// The rest of the integration is not hand-written: the `m.impl` registration and
// the `= gcu` route are produced by listing the op in the generator's
// `HANDWRITTEN_OPS`, which is also how `rng.cc` and `native_dropout` are wired.

#include "topsaten_common.h"

#include <ATen/ATen.h>
#include <ATen/SDPBackend.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/ops/_scaled_dot_product_attention_math.h>
#include <ATen/ops/empty.h>
#include <c10/core/GradMode.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "../../generated/ops.h"

namespace at::native::flagos::gcu {
namespace {

// Is `tensor` dense under *some* permutation of its non-singleton dimensions?
//
// This is the layout boundary the vendor op was measured on. It accepts both
// spellings the plugin can produce -- a plain contiguous buffer and the
// transposed view of one -- and rejects everything else, which is the honest
// limit: only those two were measured, and an arbitrary strided layout is a
// separate question.
//
// The model's own query is the second case: a dense `(1, 4114, 24, 128)` buffer
// read as `(1, 24, 4114, 128)`, strides `(12638208, 128, 3072, 1)`. The vendor op
// honours those strides -- no staging copy, no transpose -- and agrees with an
// fp64 reference to 3.1e-3 relative, which is the bf16 noise floor. Note that
// /opt/tops/include/gcu/topsaten/topsaten_ops.h documents this operand as
// `(batch, q_seq_len, q_heads, head_size)` above the "4.0" overload and as
// `[batch, q_head_num, q_seq_len, head_size]` below the "3.0" one; the
// measurement, not the header, is what settles it.
//
// Size-1 dimensions are skipped because their stride carries no information --
// a stride of 0 on a singleton dim is a broadcast view, not a layout change.
bool DenseUnderSomePermutation(const at::Tensor& tensor) {
  std::vector<std::pair<int64_t, int64_t>> dims;  // (stride, size), ascending
  for (int64_t d = 0; d < tensor.dim(); ++d) {
    if (tensor.size(d) > 1) {
      dims.emplace_back(tensor.stride(d), tensor.size(d));
    }
  }
  std::sort(dims.begin(), dims.end());
  int64_t expected = 1;
  for (const auto& [stride, size] : dims) {
    if (stride != expected) {
      return false;
    }
    expected *= size;
  }
  return true;
}

} // namespace

// The one predicate behind both the stub's answer and the kernel's decision to
// run, so the backend the composite is told about is always a backend that can
// actually serve the call. Everything it declines stays on today's math path --
// which is correct, not a fallback.
bool FusedSdpEligible(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool enable_gqa) {
  if (attn_bias.has_value() && attn_bias->defined()) {
    return false;
  }
  if (dropout_p != 0.0 || is_causal || enable_gqa) {
    return false;
  }
  if (!query.defined() || !key.defined() || !value.defined()) {
    return false;
  }
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return false;
  }
  if (!query.device().is_privateuseone()) {
    return false;
  }
  if (key.device() != query.device() || value.device() != query.device()) {
    return false;
  }
  // bf16 is the only dtype the vendor op was measured on.
  if (query.scalar_type() != at::kBFloat16 ||
      key.scalar_type() != at::kBFloat16 ||
      value.scalar_type() != at::kBFloat16) {
    return false;
  }
  if (query.numel() == 0 || key.numel() == 0 || value.numel() == 0) {
    return false;
  }
  // k and v must agree, and both must match q on batch, head count and head
  // size. GQA (`k`/`v` carrying fewer heads) is declined rather than expanded:
  // the composite handles `enable_gqa` itself on the math path, and the vendor's
  // own head-repeat behaviour was never measured.
  if (key.sizes() != value.sizes()) {
    return false;
  }
  if (key.size(0) != query.size(0) || key.size(1) != query.size(1) ||
      key.size(3) != query.size(3)) {
    return false;
  }
  // Equal query and key sequence lengths only. The vendor op takes no mask, and
  // its convention for a `q_len != kv_len` window is unmeasured; the model is a
  // joint self-attention over one 4114-token sequence, so it never asks.
  if (key.size(2) != query.size(2)) {
    return false;
  }
  return DenseUnderSomePermutation(query) && DenseUnderSomePermutation(key) &&
      DenseUnderSomePermutation(value);
}

} // namespace at::native::flagos::gcu

// _fused_sdp_choice stub for PrivateUse1. PyTorch's scaled_dot_product_attention
// calls this stub to decide whether the device has a fused backend and which one
// to use. Registering it makes the device "supported" and lets this return
// efficient_attention (2), routing to _scaled_dot_product_efficient_attention --
// the kernel below -- instead of the math decomposition.
//
// The stub's return value *is* the backend selection, so a call it cannot serve
// returns math (0) and gets today's behaviour with no host round trip and no CPU
// fallback. That also covers the one thing the stub cannot see: the leaf's
// `compute_log_sumexp` argument, which the kernel checks itself.
//
// This must live in at::native, not in the anonymous namespace and not in the
// flagos one: REGISTER_PRIVATEUSE1_DISPATCH declares its registrar in the
// enclosing namespace and references the unqualified stub symbol.
namespace at::native {
static int64_t fused_sdp_choice_gcu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  // The leaf carries Autograd=True, so under grad a call would reach the
  // *backward* leaf -- on GCU that is a FlagGems kernel, and pairing it with a
  // vendor forward is unvalidated. Returning math here keeps training exactly as
  // it is today. (The diffusion loop runs under torch.no_grad().)
  if (c10::GradMode::is_enabled()) {
    return static_cast<int64_t>(at::SDPBackend::math);
  }
  const bool eligible = at::native::flagos::gcu::FusedSdpEligible(
      query, key, value, attn_mask, dropout_p, is_causal, enable_gqa);
  return static_cast<int64_t>(
      eligible ? at::SDPBackend::efficient_attention : at::SDPBackend::math);
}

REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &fused_sdp_choice_gcu);
} // namespace at::native

namespace at::native::flagos::gcu {

// Forward: _scaled_dot_product_efficient_attention(query, key, value, attn_bias,
//          compute_log_sumexp, dropout_p, is_causal, scale?)
// Returns: (output, log_sumexp, philox_seed, philox_offset)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrivScaledDotProductEfficientAttentionKernelGcu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  TORCH_CHECK(
      !c10::GradMode::is_enabled(),
      "SDPA GCU: the vendor flash-attention kernel is inference-only. Under "
      "grad the composite must be left on the math path -- see "
      "fused_sdp_choice_gcu, which returns math for exactly this reason.");
  TORCH_CHECK(
      FusedSdpEligible(query, key, value, attn_bias, dropout_p, is_causal,
                       /*enable_gqa=*/false),
      "SDPA GCU: call does not satisfy the vendor kernel's eligibility "
      "predicate; the composite should have selected the math backend for it.");

  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t seq = query.size(2);
  const int64_t head_dim = query.size(3);
  const double scale_value =
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));

  // The one argument the shared predicate cannot see, because the
  // `_fused_sdp_choice` stub is not given it. The vendor op does fill its
  // logsumexp slot, but that value has never been validated against anything,
  // and this is the only branch where the caller reads it (`compute_log_sumexp`
  // is set for the backward, which this kernel does not serve). So delegate to
  // the math op instead and hand back its second output verbatim -- the same
  // value the composite's math branch would have produced. Measured on the
  // model's envelope, that value is the `(B, H, S, S)` bf16 attention matrix
  // rather than a `(B, H, S)` fp32 logsumexp; that is ATen's own convention for
  // this op, and not something to invent a third answer to.
  //
  // The math op is a pure composite -- no kernel on AutogradPrivateUse1,
  // PrivateUse1, Autograd or CPU -- so the delegation stays on the device.
  if (compute_log_sumexp) {
    auto [out, logsumexp] = at::_scaled_dot_product_attention_math(
        query, key, value, /*attn_mask=*/std::nullopt, dropout_p, is_causal,
        /*dropout_mask=*/std::nullopt, scale, /*enable_gqa=*/false);
    auto unused = at::empty({0}, query.options());
    return std::make_tuple(out, logsumexp, unused, unused);
  }

  // Contiguous [B, H, S, D], which is what the math path returns too. The
  // vendor op was measured reading the model's strided view of q/k/v and writing
  // a contiguous output; there is no need to reproduce the input's layout.
  auto out = at::empty({batch, heads, seq, head_dim}, query.options());
  // logsumexp is not part of the vendor's "3.0" contract but the "4.0" overload
  // fills it, and it costs one allocation to receive it rather than alias a
  // scratch buffer.
  auto logsumexp = at::empty({batch, heads, seq}, query.options().dtype(at::kFloat));
  // The remaining output slots -- cum_seq_q, cum_seq_k, philox_seed,
  // philox_offset and debug_attn_mask -- are never written for this call:
  // dropout_p is 0.0 and return_debug_mask is false, and the eligibility
  // predicate has already rejected dropout. They still have to be described, and
  // a shared 1-element scratch is what the vendor op was measured accepting
  // (measured: it does not validate the shape of a slot it does not write).
  auto scratch = at::empty({1}, query.options());

  TopsatenTensorWrapper t_q(query), t_k(key), t_v(value);
  TopsatenTensorWrapper t_out(out), t_lse(logsumexp), t_scratch(scratch);

  int64_t max_q = 0;
  int64_t max_k = 0;
  topsatenScalar_t scale_arg = ToTopsatenScalar(scale_value, at::kFloat);
  topsatenPhiloxState_t philox{};
  philox.seed.val = 0;
  philox.offset.val = 0;

  // The "4.0" overload -- the one taking a topsatenPhiloxState_t. Its sibling
  // (no philox argument) is a different overload, not a wrapper.
  std::tuple<
      topsatenTensor&, topsatenTensor&, topsatenTensor&, topsatenTensor&,
      int64_t&, int64_t&, topsatenTensor&, topsatenTensor&, topsatenTensor&>
      outs{
          t_out.get(),     t_lse.get(),     t_scratch.get(), t_scratch.get(),
          max_q,           max_k,           t_scratch.get(), t_scratch.get(),
          t_scratch.get()};

  EXEC_TOPSATEN_CMD(
      topsatenScaledDotProductFlashAttention,
      out,
      outs,
      t_q.get(),
      t_k.get(),
      t_v.get(),
      /*dropout_p=*/0.0,
      /*is_causal=*/false,
      /*return_debug_mask=*/false,
      scale_arg,
      philox);

  // philox_seed / philox_offset are only consumed by dropout, which the
  // predicate forbids, so an empty tensor is the honest answer rather than a
  // fabricated seed.
  auto unused = at::empty({0}, query.options());
  return std::make_tuple(out, logsumexp, unused, unused);
}

// Registered for the generated dispatcher the wrapper in gcu_register.inc calls.
// No backward kernel: _scaled_dot_product_efficient_attention_backward is
// already on PrivateUse1 via gcu_flaggems_register.inc and routed to FlagGems,
// and the stub above keeps grad-mode calls off this forward entirely.
REGISTER_IMPL_TO_DISPATCHER(
    PrivScaledDotProductEfficientAttentionFn,
    priv_scaled_dot_product_efficient_attention_dispatcher,
    Backend::kGcu,
    PrivScaledDotProductEfficientAttentionKernelGcu)

} // namespace at::native::flagos::gcu
