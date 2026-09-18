// Copyright (c) 2026, BAAI. All rights reserved.
//
// scaled_dot_product_attention composite override for CUDA-boxing platforms
// (DCU via DTK libtorch_hip, MetaX/maca, PPU/tsingmicro, and any CUDA-
// compatible backend that reuses the per-op CUDA boxing path).
//
// WHY a composite override is needed (measured root cause,
// docs/bench_qwen3_dcu_route_perf.md): aten::scaled_dot_product_attention is a
// *composite* op (ATen/native/transformers/attention.cpp). Its fused-backend
// selection runs inside that composite and, once a backend is chosen, branches
// on the query's device type: only CUDA/XPU tensors are sent to
// at::_scaled_dot_product_flash_attention; every other device (including
// PrivateUse1) is sent to the CPU kernel _scaled_dot_product_flash_attention_for_cpu.
// So on the boxing route, routing the attention leaf ops per-op (conf
// "*_attention = cuda") can never reach the vendor fused kernel: the composite
// picks the CPU flash leaf *before* dispatch ever sees the flagos tensors. The
// earlier _fused_sdp_choice DispatchStub PrivateUse1 slot that this file
// registered had the same flaw -- the composite consults the stub only to choose
// a *backend*, then still routes the chosen backend's leaf by
// query_.device().type().
//
// The fix: intercept the composite itself at PrivateUse1, box q/k/v(+mask) to
// CUDA (one DeviceBoxingGuard round-trip for the whole attention), and call the
// native composite on the CUDA tensors. The composite then sees device == CUDA,
// runs the vendor's own _fused_sdp_choice selector (flash for Qwen3 decode
// shapes), and returns the exact tensor the vendor route returns -- numerics
// match by construction, and the decomposed math path (_safe_softmax + bmm) is
// never taken. Nothing about the composite is hardcoded here: shapes the vendor
// selector refuses fall back to its own math path exactly as on the CUDA route.
//
// That boxed route is what every conf gets by default. A platform whose conf
// routes scaled_dot_product_attention to flaggems takes FlagGems' Triton fused
// attention instead, for the calls that kernel implements -- see
// FlagGemsEligible() for the bounds and METAX_COMPOSITE_FLAGGEMS in
// scripts/codegen/gen_vendor_confs.py for the MetaX measurement behind the one
// conf that asks for it. Everything else keeps the boxed route: the calls
// outside those bounds, the other boxing confs (which name the op `cuda`), and
// the confs that do not name it at all (backends_cuda.conf, TsingMicro).
// HasBackendForOp() is what keeps that last case out of the route, because an
// unlisted op reads as flaggems -- the right default for the generated leaf
// kernels, the wrong reading of silence for a route that has to be measured on
// each platform. Turning it off is FLAGOS_OP_scaled_dot_product_attention=cuda
// at runtime or one line of the generator's measured set at build time.
//
// Like every other hand-written wrapper in this tree (register.cc's
// WrapperMatmul), this one reads the conf and not FLAGOS_FORCE_BACKEND: that
// variable is the Dispatcher's cohort switch, and a wrapper that never dispatches
// is not covered by it.
//
// Scope: inference only. Under grad mode with a grad-requiring input the wrapper
// falls through to the composite on the flagos tensors (the pre-change math
// decomposition over autograd-aware leaf kernels), because a boxed CUDA forward
// would leave backward with PrivateUse1 tensors where the autograd graph
// recorded CUDA. See WrapperScaledDotProductAttention below.
//
// Registered only on CUDA-boxing builds. Ascend (USE_ASCEND) keeps its own
// PrivateUse1 _fused_sdp_choice stub (returns efficient_attention for its
// aclnn kernel) in backends/ascend/scaled_dot_product_attention.cc; MUSA, GCU
// and BPU do not run CUDA-compatible kernels and must keep their own paths.
// Without this guard the PrivateUse1 stub registration below would collide with
// Ascend's registration of the same DispatchStub slot.

#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <torch/library.h>

#include <optional>

#include "common.h"
#include "device_boxing.h"

#ifdef FLAGOS_FLAGGEMS_PYTHON
#include "backends/flagos/python_op_caller.h"
#endif

#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_MUSA) && \
    !defined(USE_BPU)

namespace at::flagos {
namespace {

#ifdef FLAGOS_FLAGGEMS_PYTHON
// The route below exists only when the FlagGems Python path is compiled in, so
// the guard goes around the helpers as well: on a build without it there is no
// route for them to describe, and an unguarded definition would be an unused
// function rather than dead code the compiler can see through.
//
// Whether a call is inside the class the FlagGems route was measured on. The two
// groups are kept apart because they fail differently: outside the requirements
// the kernel returns the wrong answer or raises, outside the envelope it is
// merely unmeasured -- and an unmeasured call keeps the boxing path rather than
// acquiring a new one.
//
// Requirements. scaled_dot_product_attention_forward reads stride(0..3) of all
// three operands, so they must be 4-D; it walks one kv length, so key and value
// must have the same shape; it asserts equal head dims, equal q/kv head counts
// and dropout_p == 0; and it takes a mask to be 4-D when one is present, so an
// ATen-legal broadcast 2-D mask would index past the end.
//
// Envelope. The class .sdpaend.py patched in Python to take the MetaX
// measurement this route exists for: bf16, head_dim 128, query seq >= 1024,
// non-causal, no mask, no gqa, no dropout, no explicit scale -- Qwen-Image-2512's
// joint attention (q(1, 24, 4114, 128), 120 calls/step), which is what
// METAX_COMPOSITE_FLAGGEMS records. The clauses are the probe's predicate one for
// one, so the two can be diffed; nothing else about the call is asserted, and
// widening any clause is a measurement rather than an edit. head_dim in
// particular: an _attn_fwd tile's shared-memory block grows with BLOCK_DMODEL,
// and the autotune set keep() admits (flag_gems/ops/attention.py:173) is capped
// at BLOCK_N <= 32, so the tile cannot be traded down to fit. At head_dim 128
// some admitted configurations already exceed this part's 65536 B limit -- one
// asks for 98304 B, which is the number the compiler's OutOfResources names --
// while a legal tile remains, so the op runs there; at the VAE's head_dim 512 the
// same code path reports 294912 B and nothing in the set fits, so the kernel has
// no configuration to compile on this part and the call keeps the boxing route.
// The clause is drawn where the route was measured, not where the kernel could be
// argued to fit.
//
// No grad clause, unlike the probe's predicate: the fall-through above already
// sends grad-requiring calls to the composite, and under grad mode with no input
// requiring grad both routes return the same tensor, because autograd builds no
// node for either.
constexpr int64_t kRoutedHeadDim = 128;
constexpr int64_t kRoutedMinQuerySeq = 1024;

bool FlagGemsEligible(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  // Requirements.
  if (query.dim() != 4 || key.sizes() != value.sizes()) {
    return false;
  }
  if (query.size(3) != key.size(3) || query.size(1) != key.size(1)) {
    return false;
  }
  if (attn_mask.has_value() || dropout_p != 0.0 || enable_gqa) {
    return false;
  }
  // Envelope.
  if (query.size(3) != kRoutedHeadDim || is_causal || scale.has_value() ||
      query.size(2) < kRoutedMinQuerySeq) {
    return false;
  }
  if (query.scalar_type() != at::kBFloat16 ||
      key.scalar_type() != at::kBFloat16 ||
      value.scalar_type() != at::kBFloat16) {
    return false;
  }
  return true;
}
#endif // FLAGOS_FLAGGEMS_PYTHON

at::Tensor WrapperScaledDotProductAttention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  // Training path: when a grad-requiring input is present the composite's flash
  // forward records its autograd node on the CUDA-boxed tensors, but the guard
  // below unboxes them on scope exit -- so backward would later see PrivateUse1
  // tensors where the graph recorded CUDA (ToCopyBackward0 device mismatch; see
  // perf/sdpa_train_smoke.py). Fall through to the composite on the flagos
  // tensors instead: it then takes its math decomposition over leaf ops that
  // dispatch through the generated, autograd-aware PrivateUse1 kernels --
  // exactly the semantics this override replaced. This call is the native
  // composite body, not a dispatch, so it cannot re-enter this impl (no
  // recursion). Inference (no grad required) takes the fast boxed-CUDA path.
  if (at::GradMode::is_enabled() &&
      (query.requires_grad() || key.requires_grad() || value.requires_grad())) {
    return at::native::scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  }
#ifdef FLAGOS_FLAGGEMS_PYTHON
  // FlagGems route, when the conf asks for it (see the file header). Both halves
  // are load-bearing: HasBackendForOp() keeps a conf that never names this op on
  // the boxing path -- an unlisted op reads as kFlagGems, which is the right
  // default for the generated leaf kernels and would be the wrong reading of
  // silence here -- and FlagGemsEligible() keeps the calls the Triton kernel was
  // not measured on there as well.
  if (at::native::flagos::HasBackendForOp("scaled_dot_product_attention") &&
      at::native::flagos::GetBackendForOp("scaled_dot_product_attention") ==
          at::native::flagos::Backend::kFlagGems &&
      FlagGemsEligible(query, key, value, attn_mask, dropout_p, is_causal,
                       scale, enable_gqa)) {
    // Pass every argument rather than only those that differ from FlagGems'
    // defaults: the guard above is this route's contract, so relaxing it later
    // must not leave the old constants being forwarded silently.
    at::Tensor result = at::native::flagos::CallPythonOp_GenericKw(
        "flag_gems.scaled_dot_product_attention",
        {query, key, value},
        {
            at::native::flagos::PyKwarg{"attn_mask",
                                        c10::IValue(),
                                        /*is_dtype=*/false,
                                        /*is_none=*/true},
            at::native::flagos::PyKwarg{"dropout_p", c10::IValue(dropout_p)},
            at::native::flagos::PyKwarg{"is_causal", c10::IValue(is_causal)},
            at::native::flagos::PyKwarg{
                "scale",
                scale.has_value() ? c10::IValue(*scale) : c10::IValue(),
                /*is_dtype=*/false,
                /*is_none=*/!scale.has_value()},
            at::native::flagos::PyKwarg{"enable_gqa", c10::IValue(enable_gqa)},
        });
    // Same contract as the generated FlagGems kernels
    // (csrc/aten/generated/flaggems_python_kernels.cc): a tensor crossing back
    // from Python is unboxed explicitly, whatever device it claims.
    at::native::flagos::UnboxToFlagos(result);
    return result;
  }
#endif
  // DeviceBoxingGuard records raw TensorImpl* and must only be handed named
  // lvalues that outlive the guard, so the optional mask is bound to a local
  // first. An undefined mask Tensor is left untouched by the guard (matching the
  // CUDA path where attn_mask is None).
  at::Tensor mask_t = attn_mask.has_value() ? *attn_mask : at::Tensor();
  at::native::flagos::DeviceBoxingGuard guard(query, key, value, mask_t);
  // Call the native composite directly (register.cc's WrapperMatmul does the
  // same for aten::matmul): inside the guard the tensors are CUDA, so dispatch
  // cannot re-enter this PrivateUse1 impl (no recursion), and the composite's
  // backend decision runs on the same device type the vendor route sees.
  at::Tensor output = at::native::scaled_dot_product_attention(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  // The output was produced by CUDA kernels, so it is a fresh CUDA tensor (not
  // one of the boxed inputs) and must be explicitly unboxed back to flagos.
  at::native::flagos::UnboxToFlagos(output);
  return output;
}

} // namespace
} // namespace at::flagos

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(
      "scaled_dot_product_attention",
      TORCH_FN(at::flagos::WrapperScaledDotProductAttention));
}

#endif // CUDA-boxing builds
