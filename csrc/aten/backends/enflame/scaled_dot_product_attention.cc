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
// FlagGems' own SDPA is not the answer here, and it is priced rather than
// assumed. Mask-free it measures 133.84 ms per call -- slower than the math path
// it would replace -- and on a masked call it does not complete at all: the
// kernel aborts (exit 134), because the zero-valued strides its `expand` produces
// for the mask are rejected by Triton-GCU. Routing masked attention through
// FlagGems on this target would therefore need a change on the FlagGems side
// (a constant-folded mask stride, or an in-kernel mask path that does not rely on
// the expanded view) before it could be considered at all; `csrc/aten/
// sdp_choice_stub.cc`'s `FlagGemsEligible`/`RouteMask` is the shape such a route
// would take on the platforms where it does work.
//
// Qwen-Image-2.1 asks a harder question. Its transformer issues 98
// `F.scaled_dot_product_attention` calls per denoise step, and every one of them
// carries an `attn_mask`. One full image makes 171 calls in four groups:
//
//   text encoder   (72)  (1,32,40,128) / (1,8,40,128)  no mask, causal, GQA
//   prefix segments(33)  `(1,1,26,26)` bool            q_len == kv_len == 26
//   target image   (65)  `(1,1,1,4122)` bool           q_len 4096 != kv_len 4122
//   VAE             (1)  no mask                       already served here
//
// The vendor flash op takes no mask at all, so all 98 stayed on math -- and at
// the target-image shape math costs 147.14 ms per call against 13.81 ms for the
// same call through the masked entry point below, with a 4.785 GiB peak against
// 0.031 GiB. That peak is why the flagos leg needs a second card for a model the
// vendor's own torch runs on one: the fp32 score matrix alone is
// 32 * 4096 * 4122 * 4 = 2.01 GiB, and the bmm intermediate that feeds it is
// refused on the 40 GB card where the vendor leg finished with 0.03 GiB to
// spare.
//
// `libtopsaten` exports a second attention entry point that does take a mask,
// `topsatenScaledDotProductEfficientAttention`, and it is the one the vendor's
// own `libtorch_gcu.so` calls (`_fused_sdp_choice_gcu` ->
// `scaled_dot_product_attention_gcu`). It returns exactly the 4-tuple ATen's
// `_scaled_dot_product_efficient_attention` returns -- the leaf this file already
// registers. Measured on an S60 at the model's own shapes and its own dense
// (B,S,H,D)-read-as-(B,H,S,D) views (full record in
// docs/vendors/gcu/scaled-dot-product-attention.md):
//
//   q=4096 kv=4122  mask (1,1,1,4122) bf16   Efficient 13.81 ms   math 147.14 ms
//   q=4096 kv=4122  no mask                  Flash      5.46 ms
//   q=26   kv=26    mask (1,1,26,26) bool    Efficient  0.022 ms   math    1.779 ms
//
// The first two rows are one probe on one shape, so they are a like-for-like
// pair; the two mask rows differ only in dtype and are the same allow-mask, which
// makes the predicate the only thing that separates them. The third row is
// measured on a separate mask probe and is quoted for its order of magnitude.
//
// Both vendor ops honour a `q_len != kv_len` window -- each query attends to
// every key -- and the Efficient op's bool mask has ATen's polarity: at
// (q=64, kv=64) with a single False entry, the same-polarity reference agrees to
// 1.29e-03 (the bf16 noise floor) while the inverted one is wrong by 1.27e+00.
// So the mask is handed over unconverted, and the leaf is chosen by whether
// there is a mask to hand over.
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
// Widening the eligibility predicate and switching the leaf is a body change
// only -- `_scaled_dot_product_efficient_attention` is already in that list.

#include "topsaten_common.h"

#include <ATen/ATen.h>
#include <ATen/SDPBackend.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/ops/_scaled_dot_product_attention_math.h>
#include <ATen/ops/any.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/ne.h>
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

// Does the vendor op accept `mask` as the attention mask for this call?
//
// `topsatenScaledDotProductEfficientAttention` documents three mask shapes,
// right-aligned on `(batch, q_head_num, q_seq_len, kv_seq_len)`:
//
//   (batch, q_head_num, q_seq_len, kv_seq_len)   as given
//   (q_seq_len, kv_seq_len)                      treated as (1, 1, q, kv)
//   (kv_seq_len)                                 treated as (1, 1, 1, kv)
//
// The model asks with the last, and with the rank-4 spelling of it: its
// `joint_key_valid[:, None, None, :]` is `(1, 1, 1, 4122)` and its per-segment
// `seg_mask & seg_key_valid` is `(1, 1, 26, 26)`. Rank 3 is declined because the
// documentation does not describe how it would be aligned -- right-aligned, a
// rank-3 `(batch, q, kv)` would put `batch` on the query axis.
//
// The helper has to admit two spellings of the *same* call, because the mask is
// not the one the caller wrote by the time the kernel sees it. ATen's
// `scaled_dot_product_attention` composite asks the `_fused_sdp_choice` stub
// below with the caller's mask, then converts a bool mask to an *additive* one
// at the query's dtype -- `zeros_like(...).masked_fill(~mask, -inf)` -- and
// hands the leaf that. Measured on S60, interception below the composite for a
// bool `(1, 1, 1, 4122)` caller mask: the leaf receives bf16
// `(1, 32, 4096, 4122)` with strides `(4128, 0, 0, 1)` holding `0` and `-inf`,
// i.e. `expand`ed onto the two broadcast axes over a row padded to a multiple of
// 8. A predicate that accepted only the caller's bool mask therefore made the
// stub answer `efficient_attention` and the leaf refuse the same call, which is
// a hard error rather than a fallback. Both spellings are admitted here.
//
// That the stub is asked *before* the conversion is measured, not inferred from
// reading ATen. Two caller spellings pass the shape test and fail only on the
// innermost stride, and in both the conversion would have replaced that stride
// with 1:
//
//   caller (4096, 4122) bool, `.t()`: innermost stride 4096   -> math
//   caller 1 x 1 x 64 x 80 bool, innermost axis expanded      -> math
//
// An admitted conversion would have sent both to the leaf, so the stub must be
// reading the caller's tensor. The two sites therefore see *different* tensors,
// which is the reason this predicate has to be dtype-symmetric: a bool-only rule
// admits at the stub what it refuses at the leaf.
//
// The two can still only disagree in the direction "stub admits, leaf refuses",
// and the conversion cannot produce a spelling the leaf refuses -- it always
// yields the caller's shape right-aligned on (B, H, q, kv) with a size-1-or-full
// shape and an innermost stride of 1 (`where` materialises, the innermost axis is
// padded to a multiple of 8 and sliced back). Every one of the sixteen caller
// spellings in the verification matrix reaches a correct result: admitted ones on
// the leaf, declined ones on math, none raising.
//
// The additive reading is measured, not assumed (`/tmp/probe_effattn` on S60,
// relative error against an fp32 reference, so the op *adds* the row it is given
// -- ATen's float-mask semantics -- rather than reading it as a boolean):
//
//   0 / -inf row, stride-0 broadcast, q=4096 kv=4122   additive 3.3e-3
//                                                      bool reading 1.33e+00
//   finite random row, stride-0 broadcast              additive 4.1e-3
//                                                      bool reading 4.4e-01
//   finite random row, dense (B, H, q, kv)             additive 4.1e-3
//                                                      bool reading 4.4e-01
//
// The op leaves the logsumexp slot untouched, which is why the kernel hands it a
// 1-element scratch.
//
// dtypes other than the query's own are declined: an fp32 row is *silently*
// wrong. `topsatenScaledDotProductEfficientAttention` returns SUCCESS for it and
// writes an output that matches no reading of the mask at all (probe cell
// `addrnd32_bc`: 1.21 absolute against the additive reference, best match
// `nomask` at 1.19) -- it reads the row as if it were bf16 bytes. Since the
// composite converts to the query's dtype, and the query is required to be bf16
// above, this cannot be reached through `F.scaled_dot_product_attention`; it is
// checked because a 1.2 absolute error that arrives with a SUCCESS status is
// exactly the failure mode a predicate should not let through.
//
// The layout was measured the same way, and it is the *strides* that had to be
// settled, because the composite does not normalise the mask it converts: it
// mirrors the caller's shape, so a caller mask with a real query axis converts
// into a mask with a real query stride. Interception below the composite shows
//
//   caller (1, 1, 26, 26) bool  -> leaf bf16 (1, 32, 26, 26)  strides (832, 0, 32, 1)
//   caller (2, 1,  1, 80) bool  -> leaf bf16 (2,  3, 64, 80)  strides ( 80, 0,  0, 1)
//   caller (2, 1, 64, 80) bool  -> leaf bf16 (2,  3, 64, 80)  strides (5120, 0, 80, 1)
//
// -- the caller's strides kept where the axis was non-singleton, zero where it
// was expanded, and the innermost dimension padded to a multiple of 8 and sliced
// back. So the mask is a genuinely strided view, and the op is a strided reader:
//
//   bf16 0 / -inf row, 8-padded query axis (832, 0, 32, 1)   additive 4.3e-3
//   bf16 finite random rows, real query axis (q*80, 0, 80, 1) additive 4.1e-3
//   bool half-kept row, 8-padded query axis (832, 0, 32, 1)  keep     4.3e-3
//   bool half-kept row, stride-0 broadcast (80, 0, 0, 1)     keep     3.8e-3
//
// The two padded cells are the discriminating ones: with a row stride of 32 for
// a 26-wide row, an implementation that assumed a dense (q, kv) read would take
// row q from offset 26*q and answer with a different pattern, and the random-row
// plane cell would answer with row 0's bias for every query.
//
// That leaves shape and the innermost stride as the whole test, for both dtypes.
// The op broadcasts a size-1 dimension regardless of its stride -- the model's
// own `(1, 1, 1, 4122)` caller spellings have real strides on those axes and are
// read as a broadcast -- so only a non-singleton innermost dimension constrains
// the stride, and a non-unit one (a transposed or column-sliced mask) is
// unmeasured and declined. The cost of that clause is a lost fusion rather than a
// wrong answer: such a caller mask falls to the math path, which is correct and
// ~8x slower at the target shape. Nothing in either model spells a mask that way
// -- the three caller spellings above are what the 2.1 transformer asks -- so the
// clause is a bound on what has been measured, not a path the model takes.
bool SupportedVendorMask(
    const at::Tensor& mask, const at::Tensor& query, const at::Tensor& key) {
  if (!mask.defined() || mask.device() != query.device() || mask.numel() == 0) {
    return false;
  }
  const bool is_bool = mask.scalar_type() == at::kBool;
  const bool is_additive = mask.scalar_type() == at::kBFloat16 &&
      query.scalar_type() == at::kBFloat16;
  if (!is_bool && !is_additive) {
    return false;
  }
  const int64_t mask_rank = mask.dim();
  if (mask_rank != 1 && mask_rank != 2 && mask_rank != 4) {
    return false;
  }
  // Right-aligned against (batch, q_head_num, q_seq_len, kv_seq_len). A mask
  // dimension is either a singleton -- which the vendor broadcasts, per the
  // `(1, 1, 1, kv)` spelling -- or the full size.
  const int64_t full[4] = {
      query.size(0), query.size(1), query.size(2), key.size(2)};
  for (int64_t d = 0; d < mask_rank; ++d) {
    const int64_t size = mask.size(d);
    if (size != 1 && size != full[4 - mask_rank + d]) {
      return false;
    }
  }
  const int64_t last = mask_rank - 1;
  return mask.size(last) == 1 || mask.stride(last) == 1;
}

// Whether an additive mask asks for nothing, tested cheaply enough to be worth
// asking.
//
// Zero is the additive identity, so a mask whose every stored value is zero is a
// request to apply no mask at all -- and on the model the mask that arrives here
// is exactly that. On the target-image geometry the composite hands this kernel
// `(1, 32, 4096, 4122)` bf16 at strides `(4128, 0, 0, 1)`: one row of 4122
// stored values, none of them non-zero, expanded over 540M logical ones. Reading
// it is not free. Measured in-model on the model's own q/k/v at that geometry,
// with query, key and value held fixed and only the mask respelled: 19.3 ms per
// call with the mask against 5.2 ms without it, and 617.3 ms against 166.3 ms
// over the 32 calls of one forward.
//
// Dropping it is not an approximation. Against the vendor op's own masked entry
// point, on the model's geometry, the maskless spelling agrees bitwise on 32 of
// 32 calls (`max|d| 0.000e+00`). Off-model, on a nine-geometry grid spanning
// shapes the model does not ask for -- a 512-long query, a 128-long query
// against kv 555, batch 2 with 333 queries against kv 777, head_dim 64 and 256,
// a single query token -- the two spellings agree bitwise at every one, on both
// this build and the vendor's own torch. A control row filled with `-1000.0`
// does *not* agree at any of them, so the grid is measuring the mask rather than
// passing vacuously.
//
// The test is exact and conservative. Narrowing a stride-0 axis to index 0 is a
// view, not a copy -- every index along such an axis addresses the same element,
// so the slice holds the same values under a smaller shape -- which is what lets
// the check read 4122 stored values instead of 540M logical ones, and any
// non-zero value anywhere among them keeps the mask. The innermost axis is
// deliberately left alone: `SupportedVendorMask` admits it on `size == 1 ||
// stride == 1`, and a stride-0 innermost axis is a spelling nothing has
// measured. A dense mask is declined outright, because testing it would cost a
// full-size read and it could save at most that same read inside the op.
//
// A bool mask is never inert however it reads: its zero is the opposite of a
// no-op (`False` means "mask this key out"), and the vendor's bool entry point
// was measured one bf16 ulp away from the maskless one rather than bitwise equal
// to it, so the additive-identity argument does not carry over to it at all.
bool AdditiveMaskIsAllZero(const at::Tensor& mask) {
  if (mask.scalar_type() != at::kBFloat16) {
    return false;
  }
  at::Tensor stored = mask;
  for (int64_t d = 0; d + 1 < mask.dim(); ++d) {
    if (mask.size(d) > 1 && mask.stride(d) == 0) {
      stored = stored.narrow(d, 0, 1);
    }
  }
  if (stored.numel() == mask.numel()) {
    return false;
  }
  return !at::ne(stored, 0).any().item<bool>();
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
  // `q_len != kv_len` is admitted. The earlier version of this comment refused it
  // because the vendor's convention for a `q_len != kv_len` window was
  // unmeasured; it is measured now, on both entry points, at the shape the model
  // actually asks for (q=4096, kv=4122, the 4096 target-image tokens attending
  // all 4122 joint keys): each query attends to every key, and the result agrees
  // with an fp32 reference to 3.7e-3 relative -- the bf16 noise floor, and the
  // same figure the equal-length case gives.
  //
  // The vendor op that takes a mask is a different entry point from the one
  // without, so this is where the mask stops being a reason to decline.
  if (attn_bias.has_value() && attn_bias->defined() &&
      !SupportedVendorMask(*attn_bias, query, key)) {
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
// This is the answer the composite actually uses, and it is the only one of the
// two `_fused_sdp_choice` registrations in the tree that is mask-aware.
// `csrc/aten/register.cc` registers `aten::_fused_sdp_choice` on the same
// PrivateUse1 dispatch key with `WrapperFusedSdpChoice`, which returns
// efficient_attention unconditionally; that one owns the *dispatcher* slot, so it
// is what `torch._fused_sdp_choice(...)` reports and what a TorchDispatchMode
// observes. The composite does not call the dispatcher op -- it consults this
// DispatchStub, through `is_device_supported` -- which is why the two can answer
// differently without contradiction. Measured on S60: `torch._fused_sdp_choice`
// answers `efficient` for every mask spelling including ones this stub declines,
// while a real `F.scaled_dot_product_attention` call with the same spelling
// follows the answer below (a masked call reads `fused` with grad off and `math`
// with grad on, which is the grad branch a few lines down). Anything that reads
// the dispatcher op to predict this TU's behaviour will get it wrong.
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
  // `_fused_sdp_choice` stub is not given it. The vendor op does not fill its
  // logsumexp slot when `compute_log_sumexp` is false -- measured, 2048 of 2048
  // elements still at their sentinel value after a successful call, on both the
  // Flash and the Efficient entry point -- and this is the branch where the
  // caller reads that slot (`compute_log_sumexp` is set for the backward, which
  // this kernel does not serve). So delegate to the math op instead and hand back
  // its second output verbatim -- the same value the composite's math branch
  // would have produced. Measured on the model's envelope, that value is the
  // `(B, H, S, S)` bf16 attention matrix rather than a `(B, H, S)` fp32
  // logsumexp; that is ATen's own convention for this op, and not something to
  // invent a third answer to.
  //
  // `attn_bias` is forwarded rather than dropped. It cannot be dropped: the
  // predicate admits masked calls now, so a `compute_log_sumexp` call can carry a
  // mask that decides the result.
  //
  // The math op is a pure composite -- no kernel on AutogradPrivateUse1,
  // PrivateUse1, Autograd or CPU -- so the delegation stays on the device.
  if (compute_log_sumexp) {
    auto [out, logsumexp] = at::_scaled_dot_product_attention_math(
        query, key, value, attn_bias, dropout_p, is_causal,
        /*dropout_mask=*/std::nullopt, scale, /*enable_gqa=*/false);
    auto unused = at::empty({0}, query.options());
    return std::make_tuple(out, logsumexp, unused, unused);
  }

  // Contiguous [B, H, S, D], which is what the math path returns too. The
  // vendor op was measured reading the model's strided view of q/k/v and writing
  // a contiguous output; there is no need to reproduce the input's layout.
  // `S` is the query's sequence length, not the key's: with a `q_len != kv_len`
  // window the output follows the query.
  auto out = at::empty({batch, heads, seq, head_dim}, query.options());
  // ATen's own meta registration for this leaf is the contract to match, and it
  // says `logsumexp` is `(B, H, 0)` -- an empty tensor -- whenever
  // `compute_log_sumexp` is false, which is the only configuration that reaches
  // this point. The vendor op leaves the slot untouched in that configuration, so
  // a `(B, H, S)` fp32 allocation would be handed back holding whatever the
  // caching allocator last left there. An empty tensor is both the contract and
  // the honest answer.
  auto logsumexp =
      at::empty({batch, heads, 0}, query.options().dtype(at::kFloat));
  // The output slots the vendor op does not write -- logsumexp (it is left
  // untouched in the `compute_log_sumexp=false` configuration that reaches here),
  // cum_seq_q, cum_seq_k, philox_seed, philox_offset and debug_attn_mask on the
  // Flash entry point, seed and offset on the Efficient one -- still have to be
  // described, and a shared 1-element scratch is what the vendor op was measured
  // accepting (measured: it does not validate the shape of a slot it does not
  // write, and a `(1,)` fp32 scratch in the logsumexp position changes neither
  // the status nor the output of either entry point).
  auto scratch = at::empty({1}, query.options());

  TopsatenTensorWrapper t_q(query), t_k(key), t_v(value);
  TopsatenTensorWrapper t_out(out), t_scratch(scratch);

  topsatenScalar_t scale_arg = ToTopsatenScalar(scale_value, at::kFloat);
  topsatenPhiloxState_t philox{};
  philox.seed.val = 0;
  philox.offset.val = 0;

  // A mask that is not provably a no-op is spelled exactly as it is today; an
  // all-zero additive one is dropped, because a mask the op reads 4122 stored
  // values out of to apply nothing is the most expensive way to say "no mask".
  const bool masked = attn_bias.has_value() && attn_bias->defined() &&
      !AdditiveMaskIsAllZero(*attn_bias);

  if (masked) {
    // The mask-taking entry point, and the one the vendor's own torch reaches.
    // Its output tuple is the 4-tuple ATen declares for this leaf, in ATen's
    // order: out, logsumexp, seed, offset. The "4.0" overload takes the tuple by
    // value as a tuple of references -- its sibling, which takes
    // `std::tuple<topsatenTensor, ...>&` and has no philox argument, is a
    // different overload and not a wrapper, so the argument list below is what
    // picks this one.
    TopsatenTensorWrapper t_mask(*attn_bias);
    std::tuple<
        topsatenTensor&, topsatenTensor&, topsatenTensor&, topsatenTensor&>
        outs{
            t_out.get(), t_scratch.get(), t_scratch.get(), t_scratch.get()};
    EXEC_TOPSATEN_CMD(
        topsatenScaledDotProductEfficientAttention,
        out,
        outs,
        t_q.get(),
        t_k.get(),
        t_v.get(),
        t_mask.get(),
        /*compute_log_sumexp=*/false,
        /*dropout_p=*/0.0,
        /*is_causal=*/false,
        scale_arg,
        philox);
  } else {
    // The maskless entry point, reached either because the caller passed no mask
    // or because it passed one that is provably a no-op.
    //
    // The "4.0" overload -- the one taking a topsatenPhiloxState_t. Its sibling
    // (no philox argument) is a different overload, not a wrapper. `max_q` and
    // `max_k` are written only for a variable-length call, which this is not;
    // they are passed by reference and are not read back.
    int64_t max_q = 0;
    int64_t max_k = 0;
    std::tuple<
        topsatenTensor&, topsatenTensor&, topsatenTensor&, topsatenTensor&,
        int64_t&, int64_t&, topsatenTensor&, topsatenTensor&, topsatenTensor&>
        outs{
            t_out.get(),     t_scratch.get(), t_scratch.get(), t_scratch.get(),
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
  }

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
