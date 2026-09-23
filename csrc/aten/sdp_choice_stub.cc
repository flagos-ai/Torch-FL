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
// and dropout_p == 0; it allocates its output with
// torch.empty_like(query, dtype=value.dtype) (attention.py:897) and asserts
// nothing about the three dtypes themselves, so they have to agree here or a
// mixed call would quietly come back as value.dtype -- measured on the C550:
// q and k bf16 with v fp16 at (1,2,512,128) called directly returns float16,
// while the same operands through this stub raise ATen's own "Expected query,
// key, and value to have the same dtype" from the boxing route; and it takes a
// mask to be
// 4-D when one is present, so an ATen-legal broadcast 2-D mask would index past
// the end.
//
// Envelope. The class the route has been measured on: 16-bit, 4-D, head_dim 16
// to 128, non-causal, no explicit scale, no gqa, any query length without a mask
// and >= 1024 with one, and at most the (1,1,1,KV) bool key row the mask clause
// below admits. It began as one
// shape -- Qwen-Image-2512's joint attention (q(1, 24, 4114, 128), 120
// calls/step, bf16, head_dim 128, query seq >= 1024), which is what
// METAX_COMPOSITE_FLAGGEMS records; DCU measured the same class on
// Qwen-Image-2.1's joint attention (q(1, 32, 4096, 128) x kv(1, 32, 4122, 128),
// 32 calls/forward), which is what DCU_COMPOSITE_FLAGGEMS records and which
// arrives carrying the mask the clause below is for. head_dim, the dtype and
// the query-length floor were widened on the C550 for issue #394, each on its
// own measurement; the query-length floor was removed for unmasked calls only,
// and why is its own paragraph below. The clauses are the probes' predicates one
// for one, so the
// two can be diffed, nothing else about the call is asserted, and widening any
// clause is a measurement rather than an edit.
//
// head_dim. The upper bound is the kernel's, and it is a bound on the tile the
// kernel compiles rather than on the width the caller passes: _attn_fwd takes
// HEAD_DIM as a constexpr and the call site hands it
// triton.next_power_of_2(head_dim) (attention.py:911, the padding lanes retired
// by hd_mask at :77), so 65 through 128 all compile at the same HEAD_DIM 128 and
// 129 is the first that rounds up to 256. The autotune set keep() admits
// (attention.py:173) is the 24 SMALL_HEAD_DIM_CONFIGS at BLOCK_N 16 and 32 plus
// the always-kept (128,32,4) and (128,128,8), 28 candidates; at HEAD_DIM 128
// every one of them compiles on this part -- measured one candidate at a time
// through the tuner -- and at HEAD_DIM 256 none of them does, all 28 reporting
// OutOfResources at Required: 163840 B against a 65536 B limit. So 128 is the
// largest head_dim that runs at all, and the VAE's head_dim 512 reports 294912
// B. The lower bound is the same assert read the other way: _attn_fwd carries
// tl.static_assert(BLOCK_N <= HEAD_DIM)
// (attention.py:219) and the smallest BLOCK_N in the tuner set is 16 -- the
// small-head_dim configurations exist for exactly that reason, "for small
// head_dim, we need to generate more configs" (attention.py:156) -- so head_dim
// 16 is the smallest measured case that compiles, head_dim 8 raises
// CompileTimeAssertionFailure, and 16, 32, 64, 96 and 128 were each measured on
// (1,2,1024,hd) bf16 against the same call with the conf switched to cuda:
// 113.6, 111.0, 113.2, 200.6 and 202.9 us on the route against 205.6, 218.9,
// 247.8, 253.2 and 258.9 us boxed. That is 0.46x to 0.79x, the route ahead at
// every one of them; the step from 64 to 96 is the tile width and not the
// route, since 96 pays for the 128-wide tile with 32 lanes dead.
//
// dtype. fp16 is the same kernel on the same path and lands at the fp16 floor
// rather than the bf16 one: 1.3e-04 to 1.8e-04 against an fp64 CPU reference on
// (1,2,512,128) at head_dim 16/64/128, against 1.5e-03 to 1.8e-03 for the bf16
// rows beside them. It also costs the route the same: (1,8,512,128) measures
// 111.0 us routed against 268.5 us boxed, against 111.9 and 268.1 us for the
// bf16 call at that shape. fp32 is refused because it does not run: it is the
// same kernel with a doubled element, and at head_dim 128 all 28 candidates are
// over the limit where the 16-bit ones all fit, each asking OutOfResources for
// Required: 196608 B. The head_dim ceiling and the dtype refusal are the same
// measurement taken twice.
//
// Query length. The >= 1024 floor was the joint attention's own length, not a
// property of the kernel: the grid is cdiv(query_seq, BLOCK_M) (attention.py:921)
// and the tile masks its query rows (q_load_mask, attention.py:240), so a short
// call is a small grid rather than a refused one. Measured on (1,2,Q,128) bf16,
// one process per arm, on a CUDA event pair around 30 launches with the median
// of seven such batches, against the same call with the conf switched to cuda:
//
//   Q        1      16     32     64     128    256    512    1000   1024
//   route    107.3  112.3  111.8  108.4  110.2  109.3  110.7  190.6  202.9
//   boxed    148.3  155.2  156.3  152.8  151.3  155.5  191.5  261.1  257.5
//
// Every length is faster on the route than off it, by 1.27x at the narrowest
// (Q 1024) to 1.72x at the widest (Q 512). The route's own time is flat to Q
// 512 because it is the keys the kernel walks and not the queries, and it grows
// only once the kv length does; the boxing route's math decomposition grows with
// the query length from the first row.
// Removing the floor is what closes issue #394: outside this
// envelope the MetaX boxing route cannot fuse at all -- MACA's fused SDPA is not
// inside ATen but behind its own Python patch of
// F.scaled_dot_product_attention -- so the same bf16 (1,8,512,128) call cost
// 282.2 us boxed against 53.8 us on the vendor's own entry point (5.24x).
//
// Masked query length. The floor that removes is still there for a call that
// carries a mask, at the length it was, 1024. A masked call pays RouteMask on
// every call -- a where() over zeros_like/full_like operands and the
// reshape/expand, measured at 0.024 + 0.037 + 0.105 + 0.010 ms of submit time at
// KV 4122 (see RouteMask) -- and that cost does not shrink with the sequence, so
// the length at which the route overtakes the boxing route moves out with it.
// Below the floor the shipped clause refuses the very calls a comparison would
// be about, so the two arms below are not the same measurement and the
// difference is stated rather than left in the table: the boxed arm is the
// shipped route switched off, and the masked arm is the kernel called with the
// same fp32 additive RouteMask builds, rebuilt once per call. That proxy is the
// route at the one length where the floor admits both -- at 1024 the stub's own
// routed arm measures 402.4 us and the proxy arm 404.6 -- and the additive's
// build is 230 to 243 us of it, flat across the four lengths, which is why the
// masked route is flat where the boxed one climbs. Measured on the C550 at
// head_dim 128 bf16 on (2,2,Q,128) carrying the key-valid row:
//
//   Q          256     512     768     1024
//   route      383.2   396.9   399.1   404.6  us
//   boxed      193.9   252.5   337.8   468.2  us
//
// The route is flat because the mask's own cost dominates it, so below 1024 the
// masked route is the slower one (1.98x at 256, 1.57x at 512, 1.18x at 768) and
// at 1024 it is the faster one (1.16x), which is where the floor is. Above the
// floor both arms are through the stub and the gap widens: 2048 reads 458.6
// against 1239.5 (2.70x) and 4096 reads 1680.5 against 4343.4 (2.58x).
// The other shapes this widening newly admits to the masked class agree at the
// length the floor admits them, all through the stub, all at 1024, all slower
// boxed and all flat against each other:
//
//   hd 128 bf16   402.4 vs 469.2    hd 128 fp16   402.1 vs 467.1
//   hd  64 bf16   409.6 vs 435.8    hd  64 fp16   398.7 vs 433.8
//
// So the floor is a floor on the masked class alone, and the widened
// head_dim and dtype bounds apply to it as well.
//
// The one cost the removed floor carries, measured rather than assumed: the
// autotuner is keyed on (KV_CTX, HEAD_DIM) (attention.py:174), so every new
// sequence length pays one tuning pass before it pays the route. A steady shape
// is unaffected; a model that walks many lengths pays it once per length, and
// both are bounded by the call count the route is for (120 calls per
// Qwen-Image step at one shape).
//
// Still outside, deliberately. is_causal, an explicit scale and enable_gqa were
// each measured working through this kernel on the C550 (0.54x, 0.73x and
// 0.50-0.58x) and are left out because each would widen a class that DCU's own
// copy of the kernel (flag_gems/runtime/backend/_hygon/ops/attention.py) starts
// serving too, with no DCU available to re-measure it on: the widening lands one
// measured class at a time. Two of them would also carry a contract of their
// own -- a causal clause would have to refuse a mask, because the composite this
// wrapper replaces refuses that pair outright (measured: "RuntimeError:
// _scaled_dot_product_attention: Explicit attn_mask should not be set when
// is_causal=True"), and an enable_gqa clause would have to replace the
// equal-head-counts requirement above with a divisibility one.
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
constexpr int64_t kRoutedMinHeadDim = 16;
constexpr int64_t kRoutedMaxHeadDim = 128;

// The query-length floor, which the widening below removed for unmasked calls
// and kept for masked ones. It is not a property of the kernel: it is the
// sequence length at which the route starts to pay for the additive the mask
// costs it on every call (see the header comment).
constexpr int64_t kRoutedMaskedMinQuerySeq = 1024;

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
  // The kernel asserts the three head dims are equal and reads one kv length,
  // but it sizes its output from value and asserts nothing about the dtypes, so
  // a mixed call would be silently answered in value's dtype.
  if (query.scalar_type() != key.scalar_type() ||
      query.scalar_type() != value.scalar_type()) {
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
  if (query.size(3) < kRoutedMinHeadDim || query.size(3) > kRoutedMaxHeadDim ||
      is_causal || scale.has_value()) {
    return false;
  }
  const auto dtype = query.scalar_type();
  if (dtype != at::kBFloat16 && dtype != at::kHalf) {
    return false;
  }
  // Masked calls keep the floor the widening removed elsewhere. The route builds
  // RouteMask's additive on every masked call, which the unmasked route does not
  // pay, so it needs a longer sequence before it beats the boxing route -- see
  // the header comment for the measurement.
  if (attn_mask.has_value() && query.size(2) < kRoutedMaskedMinQuerySeq) {
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
