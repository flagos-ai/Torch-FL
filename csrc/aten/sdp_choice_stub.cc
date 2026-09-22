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
// FlagGemsEligible() for the bounds and METAX_COMPOSITE_FLAGGEMS /
// DCU_COMPOSITE_FLAGGEMS in scripts/codegen/gen_vendor_confs.py for the
// measurements behind the confs that ask for it. Those calls include the ones a
// caller passes an
// explicit key-valid mask to, which this file converts twice: FlagGems reads a
// bool mask with the opposite polarity to torch and indexes the mask with
// unbound strides, so the bool row becomes an fp32 additive with stride 0 on its
// broadcast axes before the call (RouteMask). Everything else keeps the boxed
// route: the calls outside those bounds, the other boxing confs (which name the
// op `cuda`), and the confs that do not name it at all (backends_cuda.conf,
// TsingMicro).
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
// The wrapper is registered on two dispatch keys: PrivateUse1 for the calls that
// arrive without an autograd key, and AutogradPrivateUse1 for the ones that carry
// a graph. A Python TorchDispatchMode is what forces the second registration --
// it is measured and explained at the registrations at the end of this file.
//
// Registered only on CUDA-boxing builds. Ascend (USE_ASCEND) keeps its own
// PrivateUse1 _fused_sdp_choice stub (returns efficient_attention for its
// aclnn kernel) in backends/ascend/scaled_dot_product_attention.cc; MUSA, GCU
// and BPU do not run CUDA-compatible kernels and must keep their own paths.
// Without this guard the PrivateUse1 stub registration below would collide with
// Ascend's registration of the same DispatchStub slot.

#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros_like.h>
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
// non-causal, no gqa, no dropout, no explicit scale -- Qwen-Image-2512's
// joint attention (q(1, 24, 4114, 128), 120 calls/step), which is what
// METAX_COMPOSITE_FLAGGEMS records. DCU measured the same class on
// Qwen-Image-2.1's joint attention (q(1, 32, 4096, 128) x kv(1, 32, 4122, 128),
// 32 calls/forward), which is what DCU_COMPOSITE_FLAGGEMS records and which
// arrives carrying the mask the clause below is for. The clauses are the probes'
// predicates one for one, so the two can be diffed; nothing else about the call
// is asserted, and
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
//
// The mask clause (added 2026-09-19) is what brings Qwen-Image-2.1's joint
// attention onto this route. 2.1's prefill processor builds one call per prefix
// segment and passes the joint key-valid row -- (1,1,1,KV) bool, True for the
// keys attention may read -- as an explicit mask on every one of them
// (transformer_qwenimage21.py:478-560), so a route that only knows "no mask"
// leaves the whole 4096-query attention on the boxing path: measured in the
// pipeline at 40.239 ms/call boxed against 11.730 ms/call through this route.
// The clause is deliberately narrow: 4-D bool with one entry per key, i.e.
// (1,1,1,KV) once the size-1 axes are pinned. A materialized (B,H,Q,KV) mask is
// 1 GiB at this shape and is not what any caller here passes.
//
// DCU passes the same mask from the same model, and its conf asks for this route
// for that call. The clause is shared rather than duplicated: the mask is read by
// the backend's own copy of the kernel there
// (flag_gems/runtime/backend/_hygon/ops/attention.py), which indexes it the same
// unguarded way -- see RouteMask -- so what makes the call safe is the same
// stride contract, not a per-platform exception.
constexpr int64_t kRoutedHeadDim = 128;
constexpr int64_t kRoutedMinQuerySeq = 1024;

// The additive the mask route writes on the keys the caller dropped, replacing
// the bool the kernel cannot read as-is. Finite on purpose: a fully dropped
// block would take -inf as its running max and then divide 0 by 0, while -1e6
// underflows to a zero weight after exp2 against any logit this model produces
// and stays comparable as a maximum.
constexpr double kRoutedMaskFill = -1.0e6;

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
  // The mask requirement. size(3) == KV together with numel == KV is what pins
  // the shape to (1,1,1,KV), which is the only mask shape the route's
  // reshape({1,1,1,-1}).expand(...) can build from a view (see RouteMask).
  if (attn_mask.has_value()) {
    const at::Tensor& mask = *attn_mask;
    if (mask.dim() != 4 || mask.scalar_type() != at::kBool ||
        mask.size(3) != key.size(2) || mask.numel() != key.size(2)) {
      return false;
    }
  }
  if (dropout_p != 0.0 || enable_gqa) {
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

// The `attn_mask` kwarg this route hands FlagGems, built from the caller's row.
//
// The kernel cannot read a bool mask as-is: flag_gems converts one itself at
// flag_gems/ops/attention.py:928-929 (the backend copy DCU reads spells the same
// two lines at _hygon/ops/attention.py:816-817) with
// `attn_mask.to(query.dtype) * -1.0e6`, which maps True
// to -1e6 -- the inverse of torch's convention, where True is the key that
// attends -- and that conversion is skipped for a float mask, so supplying the
// additive is this route's job. Measured on this part against
// F.scaled_dot_product_attention on a materialized (1,2,1024,1024) bool mask:
// `as-is bool` and `-1e6 where True` both land at max|d| 5.840e-01, while
// `0 where True` lands at 1.953e-03 -- the bf16 floor the unfused reference
// itself sits at (4.883e-04 against an fp64 CPU reference over the same rows).
//
// The spelling below is the one that measurement checked byte-for-byte against
// the reference row, fp32 rather than the query's bf16 because the kernel adds
// this onto an fp32 accumulator. Three FlagGems routes in backends_metax.conf
// (zeros_like, full_like, where.self), so nothing here falls back to the vendor:
// measured on this part at KV 4122, 0.024 + 0.037 + 0.105 ms of submit time per
// call, 0.010 ms more for the reshape and expand, so 0.18 ms against the 28.5 ms
// per call the route saves on the shape it was measured for. All three land on
// flagos_python in a FLAGOS_LOG=dispatch census of a masked call; none of them
// reports a fallback.
//
// The stride contract the kernel's indexing needs. It forms the block pointer as
// batch_id*stride(0) + head_id*stride(1) + offs_m*stride(2) + offs_n*stride(3)
// with nothing bounding the first three (flag_gems/ops/attention.py:262-272 in
// the shared copy MetaX reads, _hygon/ops/attention.py:251-259 in the backend
// copy DCU reads), so what has
// to hold is that the last index each of them can reach contributes no offset:
// stride(i) * (size(i) - 1) == 0. expand() gives that for free -- an axis that
// grows from 1 takes stride 0, and an axis that stays at 1 is only ever indexed
// at 0 -- which is why all three are not required to be 0. Measured on a DCU
// bw1000, a (1,1,1,KV) mask leaves expand as (KV, 0, 0, 1) at batch 1, the batch
// stride surviving because that axis was never expanded, and as (0, 0, 0, 1)
// above it; both satisfy the condition. where() has already materialised a
// contiguous tensor, so the reshape is a shape no-op on every mask this clause
// admits; it is kept as the statement of the shape the expand relies on.
at::Tensor RouteMask(const at::Tensor& mask, const at::Tensor& query,
                     const at::Tensor& key) {
  const auto f32 = mask.options().dtype(at::kFloat);
  at::Tensor additive = at::where(
      mask, at::zeros_like(mask, f32), at::full_like(mask, kRoutedMaskFill, f32));
  return additive.reshape({1, 1, 1, -1}).expand(
      {query.size(0), query.size(1), query.size(2), key.size(2)});
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
    // must not leave the old constants being forwarded silently. The mask is the
    // one argument that is converted rather than forwarded -- FlagGems' own
    // reading of a bool mask is the inverse of torch's, and the kernel indexes
    // the mask with unbound strides (RouteMask).
    at::Tensor routed_mask =
        attn_mask.has_value() ? RouteMask(*attn_mask, query, key) : at::Tensor();
    at::Tensor result = at::native::flagos::CallPythonOp_GenericKw(
        "flag_gems.scaled_dot_product_attention",
        {query, key, value},
        {
            at::native::flagos::PyKwarg{"attn_mask",
                                        c10::IValue(routed_mask),
                                        /*is_dtype=*/false,
                                        /*is_none=*/!attn_mask.has_value()},
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

// The same body one key higher, for the calls that carry an autograd graph.
//
// A TorchDispatchMode is claimed at the Python key, which sits below every
// autograd key and above every backend key, and the region a mode redispatches
// into (`func(*args, **kwargs)`) is below autograd: the Python key kernel
// installs the autograd exclusion for the call it hands back. So a mode takes
// over before a kernel on the plain device key runs, and every op dispatched from
// inside that kernel then loses the autograd key -- including the composite this
// body re-runs for the grad case, which builds no graph at all. The output comes
// back detached (requires_grad=False, grad_fn=None) and its backward raises
// "element 0 of tensors does not require grad and does not have a grad_fn". No
// mode, no exclusion: the composite's leaf ops dispatch with the autograd key
// available and the graph is the one this override replaced.
//
// Measured with one synthetic op per dispatch key (torch.library.Library, the
// same at::mm body in each), grad mode on, a grad-requiring input and the call
// made twice, inside the mode and without it, so the two rows of a pair differ
// only in registration key. The table reproduces on CPU, so this is upstream
// dispatch semantics rather than a flagos autograd defect:
//
//   device | kernel key           | at::mm inside a mode       | backward
//   -------+----------------------+----------------------------+-------------------
//   cpu    | CPU    (device)      | rg=True, empty CppFunction | ok, no grad to in
//   cpu    | AutogradCPU          | MmBackward0                | ok
//   flagos | PrivateUse1 (device) | rg=False, grad_fn=None     | RuntimeError
//   flagos | AutogradPrivateUse1  | MmBackward0                | ok
//
// What is flagos-specific is that this op's grad path lives inside a device-key
// kernel body; every other op the same sweep covered has its formula at the
// autograd key and is immune. Claiming the op at AutogradPrivateUse1 moves the
// body above the mode, which is the fix this tree already applies to the same
// class of bug: contiguous and narrow (csrc/aten/register.cc:534-562) and
// non-Ascend matmul (:564-572). It is also worse here than on CPU, where the
// device-key body still produces a node -- an empty CppFunction whose backward
// reaches no input and reports no error, so a training step silently loses the
// attention gradient instead of failing: AutogradPrivateUse1 is a fallthrough
// (csrc/aten/register.cc:517-518) and this op has no Autograd[alias] kernel to
// fall through to, so nothing above the writer attaches a node at all.
//
// Both registrations are load-bearing, and the body is unchanged, so no route
// moves. Which one runs is decided by the key set, and either one then picks the
// branch itself. Measured key sets for a flagos tensor, plain, requires_grad_ and
// inside torch.no_grad() alike: DispatchKeySet(PrivateUse1, ADInplaceOrView,
// AutogradPrivateUse1, AutocastPrivateUse1) -- grad mode is a TLS flag, not a
// key-set bit -- so the AutogradPrivateUse1 kernel runs for inference as well,
// and its own GradMode::is_enabled() && requires_grad test sends inference down
// the same boxing/FlagGems branch it took before. The exception is a tensor
// created inside torch.inference_mode(), which loses the key
// (DispatchKeySet(PrivateUse1, AutocastPrivateUse1)); that call is served by the
// PrivateUse1 registration, visible from a mode, which sits below the autograd
// keys: it observes aten::scaled_dot_product_attention inside inference_mode and
// in no other grad mode. The grad branch still calls
// at::native::scaled_dot_product_attention -- the native composite body, not a
// dispatch -- so no registration can re-enter this wrapper.
//
// One consequence is intended and matches the other AutogradPrivateUse1 wrappers
// in this tree: wherever the autograd registration serves the call, a mode no
// longer observes aten::scaled_dot_product_attention itself, it observes the leaf
// ops of the branch that ran -- the composite's for the grad case, which is where
// the graph it has to keep now lives.
//
// No AutoDispatchBelowADInplaceOrView guard and no after_autograd_keyset
// redispatch, unlike the Ascend matmul wrapper
// (csrc/aten/generated/variable_type.cc:96-97): that body redispatches down to a
// fused backend kernel, this one calls the composite directly, so the view and
// inplace metadata still comes from the composite's own leaf ops as it did
// before (tests/integration/ops/test_matmul_backward_dispatch.py covers the
// no-mode graph; tests/integration/ops/test_sdpa_dispatch_mode.py covers this).
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl(
      "scaled_dot_product_attention",
      TORCH_FN(at::flagos::WrapperScaledDotProductAttention));
}

#endif // CUDA-boxing builds
