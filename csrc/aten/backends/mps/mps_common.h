// Copyright (c) 2026, BAAI. All rights reserved.
//
// Marshalling helpers for the Apple Metal (MPSGraph) operator backend.
//
// The design (docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md,
// section 6) is MPSGraph-first: an op becomes an MPSGraph built over
// placeholders and run on a Metal command queue, with hand-written MSL only
// where MPSGraph has no operation for the semantics aten needs. Nothing here
// re-tags a tensor onto DispatchKey::MPS or CUDA; the tensors stay on
// PrivateUse1 and the runtime's shared MTLBuffers are described to MPSGraph
// directly.
//
// Two facts about the Apple memory model make the marshalling short:
//
//   * Every device allocation is an MTLBuffer with MTLResourceStorageModeShared,
//     so its `contents` bytes are both the GPU's and the host's. A tensor is
//     therefore already a GPU buffer, and describing it to MPSGraph is a lookup
//     rather than a copy.
//
//   * MPSGraphTensorData has no notion of an offset into a buffer, but
//     MPSNDArray does (`initWithBuffer:offset:descriptor:`), and
//     MPSGraphTensorData wraps an MPSNDArray. That path is what makes an
//     interior pointer -- which is the common case, because the caching
//     allocator subdivides one MTLBuffer into many blocks -- describe the right
//     bytes. The NSData route (initWithDevice:data:shape:dataType:) has no
//     offset either, so it is not used.
//
//     That initializer is macOS 15 API, and it is *not* declared by the
//     MetalPerformanceShaders framework in the Xcode 15.4 SDK that the hosted
//     macos-14 image pins -- measured on the first CI run of the MPS lane, where
//     the compiler resolved the same selector to the MPSMatrix initializer and
//     rejected the argument ("cannot initialize a parameter of type
//     'MPSMatrixDescriptor *' with an lvalue of type 'MPSNDArrayDescriptor *'").
//     MpsTensorData compiles the offset path only when the SDK in force can
//     express it and the runtime reports macOS 15.
//
//     Under an older toolchain an interior pointer cannot be *described*, but it
//     is still served natively: RunGraphImpl stages the operand into a dedicated
//     buffer at offset 0 and copies results back out, so an op whose operands the
//     caching allocator split off a larger MTLBuffer keeps the whole computation
//     on the GPU. It has to, because the alternative is not neutral. Declining
//     sends the op to cpu_fallback, which copies the operand to the host, runs a
//     torch kernel in torch's own threading, and copies the result back -- and
//     for the table reads this backend exists for (a 151936x1024 float32
//     embedding table is 622 MB) that is what turned a 0.7 ms node into the
//     69-149 ms the MPS cost classes measured on the hosted runner. One memcpy
//     between two shared allocations is the smaller cost by two orders of
//     magnitude, and on unified memory it is the same device's copy. The staging
//     buffers are pooled by size rather than allocated per call; see the
//     StagingPool comment in mps_common.mm for the reuse argument.
//
//   * An MPSNDArray is not a dense view of a buffer once it has more than one
//     dimension: it rounds the stride between rows up to a 16-byte boundary. A
//     (4, 5) float32 array is therefore laid out as 4 rows of 32 bytes, not of
//     the 20 bytes torch writes, and describing a torch-contiguous buffer with
//     that descriptor reads every row after the first from the wrong offset --
//     measured on device, row 1 of a (4, 5) operand compares as
//     [0, 0, 0, 0.4681, 0.1577] against a torch row of
//     [-1.5551, -0.3414, 1.853, 0.4681, -0.1577]. The same arithmetic is what
//     the initializer's "buffer is not large enough. Must be 128 bytes" message
//     reports for that shape (4 rows x 32 bytes). `rowBytes` is not a public
//     property of MPSNDArrayDescriptor and reads back 0 through KVC, so the
//     pitch cannot be set to 20 from here. Everything a kernel describes to
//     MPSGraph is therefore a 1-D array of `numel` elements -- one row, nothing
//     to pad -- and the aten shape is restored inside the graph, where a reshape
//     is a metadata operation. The helpers below are shaped around that rule.
//
// This header is Objective-C++: it imports the Metal and MPSGraph frameworks
// and names their types. Only the generated kernels and mps_common.mm include
// it, and only for a USE_MPS build.

#pragma once

#ifdef USE_MPS

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace at::native::flagos::mps {

// c10 scalar type -> MPSGraph element type. Returns false, leaving *out
// untouched, for a dtype MPSGraph has no element type for (float64 and the
// two complex types today). The caller takes the CPU round-trip for those
// rather than guessing at a widening, because a wrong element type is a wrong
// answer, not a failure.
bool MpsDataTypeOf(at::ScalarType dtype, MPSDataType* out);

// Whether the MPSGraph unary path computes this dtype. It is MpsDataTypeOf
// minus bool: MPSGraph has an element type for bool, but aten's `abs` does not
// define a bool kernel on any backend (`abs_cpu` raises NotImplementedError),
// so a bool operand takes the CPU round-trip and gets the same error rather
// than a silent identity. Unsupported dtypes (float64, the complex pair, the
// quantised types) fail the same test and round-trip too.
bool MpsSupportsUnaryDtype(at::ScalarType dtype);

// Whether an elementwise op over two operands computes exactly on the MPSGraph
// path. Both operands must carry the same dtype, and that dtype must be one
// MpsDataTypeOf maps: MPSGraph's binary arithmetic has no implicit promotion,
// whereas aten promotes (int32 + int64 -> int64, half + float -> float), so a
// mixed pair takes the CPU round-trip and keeps the reference's result dtype.
bool MpsSupportsBinaryDtype(at::ScalarType self_dtype, at::ScalarType other_dtype);

// Whether a `where.self` computes exactly on the MPSGraph path. The two branch
// widths take MpsSupportsBinaryDtype's rule for that predicate's reason -- the
// select builds one node in one element type and promotes nothing, while the
// reference promotes the branches to a common dtype -- and the condition width
// is checked against the reference's own narrowing rather than the graph's:
// bool and the deprecated uint8 are accepted, anything else is a decline so the
// reference raises its own message.
bool MpsSupportsWhereDtype(
    at::ScalarType self_dtype,
    at::ScalarType other_dtype,
    at::ScalarType condition_dtype);

// Whether a `cat` over a list of `dtype` computes exactly on the MPSGraph path.
// Both parameters are that one datatype: the concat node requires a single
// datatype across the list where the reference promotes a mixed list, which the
// body checks separately, so this only decides whether the uniform width is one
// the graph can carry.
bool MpsSupportsCatDtype(at::ScalarType dtype, at::ScalarType other_dtype);

// Whether an elementwise op over a tensor and a host scalar computes exactly on
// the MPSGraph path. The scalar becomes an MPSGraph constant of the tensor's
// dtype, so an integer tensor accepts an integer scalar only -- a float scalar
// promotes the result off the tensor's dtype, which the constant path cannot
// express -- and a complex scalar is declined outright.
bool MpsSupportsScalarDtype(at::ScalarType self_dtype, const at::Scalar& other);

// Whether a reduction computes exactly on the MPSGraph path: the float widths
// only, and only with no `dtype=` that moves the result off the operand's
// dtype. aten promotes an integer sum to int64 for every input width, and
// `mean` of an integer tensor has no result dtype at all without an explicit
// one, so both take the CPU round-trip rather than a graph that would answer in
// the wrong dtype.
bool MpsSupportsReduceDtype(
    at::ScalarType dtype,
    const std::optional<at::ScalarType>& out_dtype);

// Whether an aten `_softmax` over a tensor of `self_dtype` computes exactly on
// the MPSGraph path, and the dtype the result comes back in.
//
// `_softmax(self, dim, half_to_float)` returns the operand's own dtype, or
// float when `half_to_float` is set: that flag is the CUDA spelling of
// "accumulate in float and hand back float", and it is the one host argument
// here that decides the result dtype rather than the tensor. A body that takes
// it therefore widens the operand to the dtype it computed in, which is what
// the reference does as well.
//
// The widths are the ones the graph was measured on (Apple M5, four rows of
// 4096 values spread over [-6, 6], against a reference softmax that accumulates
// in float and rounds once):
//
//   float32   within 29 ulp, 1.9e-6 relative
//   float16   bit-identical on every element above 1e-4, and the rest differ by
//             at most one subnormal (4.7e-41) -- MPSGraph accumulates a half
//             reduction in float, the way the reference does, so half does not
//             need the round-trip that would otherwise be the safe answer
//   bfloat16  declined. A bfloat16 graph returned a denormal for every element
//             instead of the reference's value on this SDK, so the graph's
//             bfloat16 path is not one this backend can vouch for; the
//             round-trip keeps it correct until it is.
//
// float64 has no MPSGraph element type at all, and an integral or bool operand
// is a NotImplementedError in the reference rather than a result, so both take
// the round-trip and get it.
bool MpsSupportsSoftmaxDtype(
    at::ScalarType self_dtype,
    bool half_to_float,
    at::ScalarType* out_dtype);

// Whether `pow(self, exponent)` computes exactly on the MPSGraph path, on the
// same measurement as the softmax gate above: `x ** 2` was exact and `x ** 2.5`
// within 1 ulp in float32, and both matched the reference bit for bit in
// float16 -- including the NaN a negative base to a fractional exponent
// produces, which is where a naive `exp(log(x) * e)` composition would not be.
//
// An integral operand is declined rather than measured: aten raises on a
// negative exponent there ("Integers to negative integer powers are not
// allowed") and promotes to float on a fractional one, and MPSGraph's power has
// no integral form to match either answer with. The exponent itself is the
// constant MpsConstant carries, so a complex one is declined by the same rule
// every other scalar operand is.
bool MpsSupportsPowDtype(at::ScalarType self_dtype, const at::Scalar& exponent);

// The rank `dim` names once negative positions are resolved against `self`'s
// rank. False when `dim` is out of range for the tensor, which is the reference's
// error to report: the wrappers decline that case without touching the input.
bool MpsSoftmaxDim(const at::Tensor& self, int64_t dim, int64_t* out);

// The answer `_softmax(self, dim, half_to_float)` has when the graph is not the
// one producing it, computed on the host and moved back to `self`'s device.
//
// The flag cannot be handed to the CPU kernel: it is a CUDA spelling the host
// build rejects outright, so a declined width would raise instead of answering.
// This widens the operand and asks for the flag-clear result instead, which is
// the same "compute in float, hand back float" the flag stands for -- the rule
// the graph path follows by widening to its own `out_dtype`. Every decline in
// the softmax bodies routes through here so the two paths cannot drift apart.
at::Tensor MpsSoftmaxOnHost(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float);

// Whether `silu` computes exactly on the MPSGraph path, composed as
// `x * sigmoid(x)`.
//
// This is the first op in the layer MPSGraph has no selector for, so the body
// composes it -- and a composition is only allowed in where it reproduces the
// reference rather than approximating it. aten's CPU `silu` is its own kernel
// and does *not* evaluate `x * sigmoid(x)`: evaluated on the host the two
// readings differ by 7.8e-03 on float16 and 3.1e-02 on bfloat16, which would be
// a wrong answer and not a rounding difference. So this gate is set from a
// measurement rather than from the identity. Measured on an M5 against at::silu
// in the operand's own dtype, over operands spanning [-40, 40] (plus the
// saturated tails, denormals, and a rank-0 and rank-3 shape):
//
//   float32   9.5e-07 max abs, 2.7e-07 relative. Not bit-identical, and the
//             right reading of that is "both are correct": against an exact
//             float64 evaluation the graph and the CPU reference sit
//             1.2353e-06 away from it to the last digit -- the same number --
//             so the two round differently rather than one being wrong. The
//             difference is what a `silu` near -40 looks like, where the value
//             is ~1.7e-16 and every ulp of sigmoid moves it.
//   float16   bit-identical, including the inf, the -0.0 the tail produces and
//             the NaN an `inf * 0` gives.
//   bfloat16  bit-identical, same cases.
//
// All three are served, which is worth spelling out because it is the same
// answer the conv gate reached for a different reason: a bfloat16 softmax came
// back as denormals on this SDK and a bfloat16 pow was declined with it, so the
// widths a category serves are per-op here rather than a list this file keeps.
//
// float64 has no MPSGraph element type at all, and an integral or bool operand
// is the reference's own error -- aten's `silu` has no integral kernel, so
// `silu(int)` raises rather than promoting -- and both take the CPU round-trip
// and get exactly that.
bool MpsSupportsSiluDtype(at::ScalarType dtype);

// Whether `sigmoid` computes exactly on the MPSGraph path, and the increment
// that declined a width the sibling above serves -- which is the point of it.
//
// `sigmoid` is the activation `silu` is composed from, and it is the op a model
// reaches for on its own as well: a hand-written attention mask, a gating
// projection, a `torch.sigmoid` in a loss. MPSGraph has a real selector for it
// (`sigmoidWithTensor:`), so the body is one node -- but the selector is the
// same one the `silu` composition already builds, and measuring *it* through
// `silu` shows that the exactness `silu` measured is a property of the
// composition's tail rather than of the node.
//
// Measured on an M5 with the graph's sigmoid read out of the routed `silu`
// (silu(x)/x, which is the node the composition builds) against `at::sigmoid`
// evaluated in the operand's own dtype over [-40, 40] with the saturated tails,
// the denormals and the exact 0 included:
//
//   float32   1.788e-07 max abs, 3.091e-07 relative at x = 1.33 over the
//             sampled range. Serving it is therefore a rounding decision and not
//             an exactness claim, and it carries one more measured caveat: the
//             same subnormal flush that costs float16 its place also reaches
//             float32, but only where the *result* falls below the smallest
//             normal float. Measured, `sigmoid(-87.0)` is 1.6458e-38 on both
//             sides and `sigmoid(-88.0)` is 6.0546e-39 on the reference and
//             exactly 0 on the graph; below about -88.5 the reference is
//             subnormal too and the two agree on 0 again. So the divergence is
//             confined to the single-input band where the reference's answer is
//             subnormal-but-nonzero, and it is the same platform policy the
//             `div` body already records for a subnormal *result*. It is
//             recorded here rather than gated away because a sigmoid is never
//             evaluated there by a model and the round-trip at these extremes
//             would be a per-call host decision on a value the graph cannot
//             express; the float16 decline below is what the same policy costs
//             when the band is a wide one rather than a point.
//   float16   3.088e-02 max abs against the float16 reference, which is 6.2e-02
//             relative -- a wrong answer rather than a rounding difference, and
//             it is the *platform's* subnormal policy rather than the node's
//             arithmetic. 151465 of 400001 samples come back below float16's
//             smallest normal while the reference keeps them: this SDK flushes
//             a subnormal float16 *result* to zero, exactly as the `div` body
//             recorded, and the left tail of a sigmoid is almost entirely
//             subnormal at that width. So float16 is **declined** here, where
//             `silu` serves it -- because a silu's tail is multiplied by its
//             own x, which is likewise subnormal and flushes to the same zero
//             the reference produces, while a bare sigmoid's tail has no such
//             partner and the flush is visible.
//   bfloat16  1.061e-02 relative, largest at x = 4.58. bfloat16's exponent
//            range is float32's, so nothing here is subnormal and the
//            difference is rounding at a width that holds about three decimal
//            digits -- 8.289e-03 absolute against a reference whose own values
//            are of order 1e-2 to 1. The graph's sigmoid is not bit-identical
//            to aten's at this width, so it is declined by the same rule that
//            keeps `softmax` off bfloat16: a measured difference that is a
//            rounding rather than a bug is still a difference this backend
//            declines to claim.
//
// float64 has no MPSGraph element type, so it takes the round-trip.
//
// An integral or bool operand is *promoted* rather than rejected, and that was
// measured rather than assumed -- the first version of this note claimed aten's
// `sigmoid` has no integral kernel, and `at::sigmoid(torch.arange(-2, 3))`
// returns the float32 `[0.1192, 0.2689, 0.5, 0.7311, 0.8808]` on the CPU. The
// graph path produces the same promoted float32 result, because MpsRunGraph's
// `out` is allocated from the operand's options and the ceiling the flush above
// is about does not arise for a small integral input; the gate declines them
// anyway, since the round-trip is where the *promotion rule* lives and a gate
// that is right for the wrong reason is the failure this file keeps recording.
//
// The gate is deliberately the narrowest one in this file, and the contrast
// with `MpsSupportsSiluDtype` is the increment's finding: two ops that build the
// *same node* do not share a width list, because what the node's error means
// depends on what the surrounding graph does with it.
bool MpsSupportsSigmoidDtype(at::ScalarType dtype);

// Whether `exp` computes exactly on the MPSGraph path, and the increment that
// *widened* the served list past the sibling activations' -- which is the point
// of it.
//
// `exp` is the largest elementwise unary left on a small model's fallback
// profile and the node the rest of the cohort (`log`, `rsqrt`, a hand-written
// `tanh`) is written over, so it is served with a real selector
// (`exponentWithTensor:`) rather than a composition. Measured on an M5 against
// at::exp in the operand's own dtype, over each width's whole representable
// curve and over the saturated tails:
//
//   float32   1.897e-07 max relative, which is two ulp at the top of the range
//             (the largest argues are ~1e+38, so the ulp there is ~1e+31 and the
//             absolute difference is large while the relative one is tiny). Not
//             bit-identical, and the right reading is the same one `silu`'s
//             float32 recorded: measured against an exact float64 evaluation at
//             x = 20.0, the graph sits 2.06e+01 above it and the CPU reference
//             1.14e+01 below -- both are one ulp from exact, on opposite sides, so
//             the two round differently rather than one being wrong. 39.6% of a
//             linspace over [-87, 88] lands on a different ulp for the same
//             reason, and no sample is farther than two ulps apart.
//   float16   bit-identical, and this is the increment's finding: the entire
//             representable curve agrees, *including every subnormal*. Measured
//             over 400001 points spanning [-30, 11] -- which reaches the lowest
//             subnormal an `exp` can produce -- there are zero mismatches, so the
//             subnormal flush the `sigmoid` body recorded for this width is a
//             property of *that node* rather than of the platform: the same
//             exponential keeps subnormal results that the sigmoid node flushed.
//             Served, where the sibling activation is declined.
//   bfloat16  bit-identical over the normal range, measured over [-30, 88] with
//             zero mismatches. The subnormal band behaves like float32's below
//             (the graph flushes an `exp` whose reference answer is subnormal,
//             and the flush never reaches a normal answer), which is the platform
//             policy the `div` body records rather than a rounding difference.
//             Served, where the sibling `sigmoid` declined this width for a
//             rounding difference that an `exp` does not have: measured, the two
//             agree bit for bit here.
//
// The subnormal flush that does appear -- at float32 and bfloat16, on an input
// below about -87.3 where the reference's own `exp` is subnormal while the graph
// answers 0 -- is a *wrong answer* at those points, and it is accepted on the
// same policy the `sigmoid`, `div` and `silu` bodies accepted it: the flush is
// confined to the input band where the reference answer is itself subnormal, and
// measured, it never reaches one that is normal (the smallest normal float32 is
// 1.1755e-38, and the graph answers exactly down to -87.336, the largest input
// whose reference answer is normal). A model's `exp` operates on the logits of a
// softmax or the exponent of a power, where this band is not reached; the
// round trip at these extremes would be a per-call host decision on a value the
// graph cannot express.
//
// float64 has no MPSGraph element type and takes the round-trip. An integral or
// bool operand is *promoted* by the reference to float32 -- measured,
// `at::exp(torch.arange(-2, 3))` returns a float32 tensor, and the graph path
// reproduces it for a small input because MpsRunGraph allocates `out` from the
// operand's options and the width is widened by the promotion before the graph
// sees it -- but the gate declines them anyway, because the promotion rule is the
// reference's and a gate that is right for the wrong reason is the failure this
// file keeps recording. They keep the round-trip, which is where the rule lives.
//
// The gate is deliberately *wider* than `MpsSupportsSigmoidDtype`'s, and the
// contrast is the increment's finding read the other way round: two ops that
// share a node -- or here, two nodes of the same family -- do not share a width
// list, because what a flush or a rounding means depends on what the op computes
// rather than on the platform alone.
bool MpsSupportsExpDtype(at::ScalarType dtype);

// Whether `log` computes exactly on the MPSGraph path, and the increment whose
// honest divergence is a property of the *input* domain rather than of the
// magnitude -- which is the point of it.
//
// `log` is the inverse of the sibling `exp` above and the other half of every
// logit-space composition a model writes by hand (a log-softmax, a
// temperature-scaled score, a KL term), so it is served with a real selector
// (`logarithmWithTensor:`) rather than a composition. Measured on an M5 against
// at::log in the operand's own dtype, over 400001 samples per width and over
// the domain's edges:
//
//   float32   129358 of 400001 samples land on a different ulp, max relative
//             1.291e-07 and max absolute 7.629e-06 -- one ulp. Not
//             bit-identical, and the reading is the one `silu`'s and `exp`'s
//             float32 recorded: both roundings are correct and they simply
//             differ.
//   float16   3 of 400001 samples differ, max relative 6.184e-04 and max
//             absolute 3.906e-03 -- one float16 ulp at 6.3. The three are a
//             last-ulp rounding in which the *graph* is marginally closer to the
//             exact answer than the CPU reference is (measured at x = 552.5:
//             |graph - exact| = 1.953433e-03 against
//             |cpu - exact| = 1.952817e-03), so this is the same "both are
//             correct" reading rather than a graph error. Served, and worth
//             spelling out because `sigmoid` declined this width: the flush that
//             cost the sibling its place does not reach this node's arithmetic.
//   bfloat16  bit-identical, zero mismatches.
//
// The domain's edges are all reproduced exactly, which is what makes this
// increment's finding an *input* one: measured, log(0) = -inf, log(-0.0) = -inf,
// log(1) = 0.0, log(-1) = nan, log(-5) = nan, log(inf) = inf, log(-inf) = nan
// and log(nan) = nan, all matching the reference bit for bit.
//
// What does diverge at float32 is where the platform's subnormal policy does
// show: it flushes a subnormal *input* to zero before the node evaluates, and
// log(0) is -inf, so a positive input below the smallest normal comes back -inf
// where the reference answers a finite (and large) number. Measured, the
// boundary sits exactly at that normal: `1.1754944e-38` (the smallest normal
// float32) agrees with the reference while `1.17e-38` flushes to -inf. This is
// accepted on the same policy the `sigmoid`, `div`, `silu` and `exp` bodies
// accepted their subnormal divergences, and it has the same shape: the wrong
// answer is confined to the band where the reference's *input* is subnormal, and
// it never reaches a normal input -- measured, every input at or above the
// smallest normal reproduces the reference. A model's `log` operates on a
// probability or a logit, where that band is not reached; a per-call host
// decision there would be a check on a value the graph cannot express.
//
// float16 is the contrast that keeps this from being a blanket rule: measured,
// it does *not* flush subnormal inputs -- `log(5.96e-8) = -16.640625`, exact --
// so the boundary above is a float32 property of the platform rather than one
// this gate can generalise across widths.
//
// float64 has no MPSGraph element type and takes the round-trip. An integral or
// bool operand is *promoted* by the reference to float32, the same answer `exp`
// measured, and the gate declines them for the same reason: the promotion rule
// is the reference's, and the round-trip is where it lives.
//
// The gate's shape is therefore `exp`'s list with `exp`'s reasoning turned
// around: where the exponential's honest cost was a subnormal *result* it
// declines to express, the logarithm's is a subnormal *input* it flushes to
// zero -- and both stay inside the band where the reference's own answer is
// outside the range the graph can represent.
bool MpsSupportsLogDtype(at::ScalarType dtype);

// Whether `rsqrt` computes exactly on the MPSGraph path -- and the first gate
// here whose subject is a *composition* rather than a node.
//
// The body is `reciprocal(sqrt(x))`, two MPSGraph nodes rather than the fused
// `reciprocalSquareRootWithTensor:` MPSGraph names, for the reason recorded
// above the `T_RSQRT` template: the fused selector is macOS 15+ and the hosted
// MPS lane is not, so building it raises an `NSInvalidArgumentException` that
// aborts the process rather than declining. The composition is the same
// arithmetic on two nodes the SDK this backend is verified against does have.
//
// That the subject is a composition is what makes an independent predicate
// necessary rather than a reuse of a sibling's: two subnormal policies apply
// where `exp` and `log` had one each. The input half is the *sqrt's* -- a
// subnormal operand, if the platform flushes it, becomes zero before the node,
// and `rsqrt` of a value approaching zero is large rather than small, so the
// flush is a divergent magnitude and not a rounding. The result half is the
// *reciprocal's*, the same flush `sigmoid` and `div` measured: a reciprocal
// that is subnormal at the operand's width can come back zero where the
// reference answers a denormal.
//
// The domain edges were measured first, against the reference on the CPU, and
// every one of them has to be reproduced for the composition to be served at
// that width: `rsqrt(0) = inf`, `rsqrt(-0.0) = -inf`, `rsqrt(1) = 1`,
// `rsqrt(4) = 0.5`, `rsqrt(-1) = nan` (the sqrt of a negative is a nan and the
// reciprocal of a nan is a nan), `rsqrt(inf) = 0` and `rsqrt(nan) = nan`.
//
// The gate is measured per width; see the note in mps_common.mm for the
// numbers and for which half is the one that decides each width.
bool MpsSupportsRsqrtDtype(at::ScalarType dtype);

// Whether `sqrt` computes exactly on the MPSGraph path.
//
// A predicate of its own rather than `MpsSupportsRsqrtDtype` reused, and the
// reason is the same one that gave the sibling its own predicate read the other
// way: `rsqrt` here is the *composition* `reciprocal(sqrt(x))`, so its gate had
// to be right about two nodes at once -- the sqrt's input half and the
// reciprocal's result half -- while this body is the sqrt alone. A width the
// composition serves is therefore not automatically served here, and a width it
// declines is not automatically declined: the two gates ask about different
// computations and their answers come out different at the edges.
//
// What the body inherits from the sibling is the node itself.
// `squareRootWithTensor:` is the inner half of `T_RSQRT`, on both the SDK this
// backend is verified against and the one after it, so the selector is the same
// one the composition already builds rather than a new question. The domain
// edges are the ones a square root has rather than a reciprocal's:
// `sqrt(0) = 0` (and `sqrt(-0.0) = -0.0`, the signed zero the sibling's
// reciprocal turns into `-inf`), `sqrt(1) = 1`, `sqrt(4) = 2`,
// `sqrt(-1) = nan`, `sqrt(inf) = inf` and `sqrt(nan) = nan`.
//
// All three float widths are served, and the finding is *where* this differs
// from the sibling that shares the node: the same subnormal *input* flush that
// cost `log` and the `rsqrt` composition's input half shows up here at float32
// and bfloat16, but it reads differently -- the flushed operand comes back 0,
// a value inside the normal range, where the sibling's reciprocal turned the
// same flush into `inf`. float16 does not flush at all, as `log` and `rsqrt`
// both found.
//
// The gate is measured per width; see the note in mps_common.mm for the numbers,
// for the flush that decides the low-precision widths, and for the one edge
// whose failure is a wrong *kind* of answer rather than a wrong magnitude.
bool MpsSupportsSqrtDtype(at::ScalarType dtype);

// Whether `tanh` computes like the reference on the MPSGraph path.
//
// `tanh` is the activation a transformer block reaches for at both ends -- the
// tanh approximation inside a GELU and the one inside a recurrent or attention
// gate -- and MPSGraph names it directly as `tanhWithTensor:`. The body is
// therefore the plain one-node unary template plus one guard, and the guard is
// the first thing the node needs: measured at all three widths, the node
// *saturates* a NaN operand to +-1 rather than propagating it, so
// `isNaNWithTensor:` selects a NaN back over the node's answer and the body
// answers the reference's `nan` rather than the node's `1.0`.
//
// With the NaN guard in place the predicate is a pure node-fidelity question,
// and its answer is the first one on this backend that declines the *widest*
// float:
//
//   * float16 and bfloat16 are served. The node is the reference's own function
//     at these widths rather than an approximation of it -- measured, both have
//     the *reference's* max error against the exact tanh to the digit, and both
//     are *exactly* odd (max |g(x) + g(-x)| is 0.0 where the float32 node breaks
//     the same measurement by 1.788e-07). Two edges at these widths are the
//     host's rather than the body's -- the sign of a zero, and the bfloat16
//     subnormal band -- and both were measured both ways rather than pinned as
//     an invariant. See the note in mps_common.mm.
//   * float32 is *declined*. The node is a coarser approximation of a tanh at
//     this width, not a differently-rounded exact one, and the measurement says
//     so three ways: it breaks odd symmetry by 1.788e-07 where the reference is
//     exactly symmetric; its error against the exact answer is ~100x the
//     reference's at every scale of the curve (4.321e-08 against 4.328e-10 over
//     [0.001, 0.01), 1.069e-07 against 2.981e-08 over [0.5, 1)); and 165340 of
//     400001 samples over [-10, 10] come back a different float32 value, up to a
//     visibly different number inside the saturation band rather than a
//     neighbouring ulp. On a `torch.randn(1000000)` operand 545055 samples
//     differ and the graph is the less accurate of the two on 544303 of them.
//
// Serving that would replace the reference's answer with a coarser one on a
// majority of the values the most-called activation in a transformer block is
// handed, so the float32 caller keeps the `cpu_fallback` round-trip and only the
// two narrow widths -- where the node *is* the reference's function -- take the
// graph. This is the first gate here whose narrowness is about accuracy rather
// than a subnormal flush or a domain edge, and it is stated per width for that
// reason.
//
// float64 has no MPSGraph element type and an integral or bool operand is
// promoted by the reference to float32, so both keep the round-trip.
//
// The gate is measured per width; see the note in mps_common.mm for the numbers
// and for the round-trip cost the float32 decline leaves on the table.
bool MpsSupportsTanhDtype(at::ScalarType dtype);

// Whether `erf` computes like the reference on the MPSGraph path.
//
// `erf` is the exact-GELU kernel -- `0.5 * x * (1 + erf(x / sqrt(2)))` is the
// form a model writes when it wants the exact activation rather than the
// sibling `tanh`'s approximation -- and MPSGraph names it directly as
// `erfWithTensor:`, so the body is the plain one-node unary template.
//
// The finding here is the *opposite* of the sibling `tanh`'s, and that is the
// increment: where the tanh node was a coarser approximation of the reference
// at float32 and had to be declined there, the erf node is the reference's own
// function at every width. Measured on an M5 against `at::erf` in the operand's
// own dtype:
//
//   * float32 is served, and the node is *closer to the exact curve than the
//     reference is* rather than merely equal to it. Measured against the
//     float64 erf over a 400001-sample sweep of `[-3, 3]`, the node's max error
//     is 1.788e-07 where the reference's is 4.955e-07, and the node is the
//     closer of the two on 242217 of the 400001 samples. The reference's own
//     error is real rather than an artifact of the comparison: measured against
//     `erff` through float64, `at::erf` itself is off the correctly-rounded
//     float32 answer by up to 5.327e-07. So the two implementations round the
//     same curve differently at the last ulp, and neither is wrong -- which is
//     why this gate is not the sibling's question and why the width is served:
//     a `erf` that is a last ulp from the reference's rounding is a `erf`, and
//     a tanh that is 100x off the exact answer is not a tanh.
//   * both of those roundings land on the same *answer* wherever the curve is
//     flat enough to say so. Measured, the two agree bit for bit on 46.5% of the
//     sweep, and the disagreement is confined to the last ulp: 195438 of the
//     213871 differing samples are within 4 of the reference's own representable
//     step, and the large step counts are the subnormal tail near zero where a
//     single step is 1.4e-45 and the two agree on the answer to 1.8e-07 -- the
//     node and the reference both answer exactly 1.0 from `|x| >= 4`, where the
//     sweep's largest "step" is 876061900 and the two are the same number.
//   * float16 and bfloat16 are served on the sibling `tanh`'s evidence as well
//     as their own: measured over the same sweep, the node agrees bit for bit on
//     399878 of 400001 samples at float16 (max 3 steps) and on 399994 of 400001
//     at bfloat16. The narrow widths are where the reference is *also* only
//     approximate, so whatever rounding difference remains is below both
//     widths' ability to represent it.
//
// The domain's edges are reproduced exactly at all three widths, and unlike
// `tanh` there is no guard to add: measured, the bare node propagates a NaN at
// every width where `tanhWithTensor:` saturated it, answers `+-1` at `+-inf`
// (the bound the body's `1 - 2/(exp(2x)+1)` cousins get wrong), and keeps the
// sign of a zero -- `erf(-0.0)` is `-0.0` on both the node and the reference, at
// every width, on this host *and* the `macos-14` CI runner, so it is an
// invariant rather than the host-dependent reading `tanh`'s signed zero was.
//
// One band is host-dependent rather than deleted outright, and it is the same
// band the siblings recorded: the float16 and bfloat16 *subnormal operand*. The
// bfloat16 band is the tanh flush read at this node -- all 127 subnormal
// bfloat16 operands come back a zero like the reference's own rounding does --
// while float16 differs on 877 of its 1023 positive subnormals, and the
// reference itself answers zero for 3 of them. The band is worth recording
// because it is where a *step count* is meaningless, but it is not worth a
// test: the values involved are all below the width's smallest normal and a
// model's `erf` operand is a pre-activation, not a denormal. It is measured,
// recorded here and in the test module, and -- unlike the sibling's -- not
// asserted, because unlike the sibling's the two hosts agree on it.
//
// float64 has no MPSGraph element type and an integral or bool operand is
// promoted by the reference to float32, so both keep the round-trip; measured,
// `at::erf(torch.arange(1, 4))` returns a float32 tensor.
bool MpsSupportsErfDtype(at::ScalarType dtype);

// Whether a table read computes exactly on the MPSGraph path: one float width
// for the table, and an int or long index tensor.
//
// The table half mirrors the other gates -- float32, float16 and bfloat16 are
// served, float64 has no MPSGraph element type, and an integral or bool table is
// the reference's own error rather than a result.
//
// The index half is the narrower one, and it is the reference's list rather than
// this file's: `at::embedding` accepts Long and Int indices and rejects anything
// else with a `RuntimeError` naming exactly those two types, so a float index
// takes the round-trip and gets that message. The width matters to the graph too
// -- MPSGraph's gather takes an int32 or int64 index tensor and has no other
// integer input -- but the reason this gate is written from the reference is that
// the reference has an answer for a float index and a graph built from one would
// not.
//
// What this predicate does *not* cover is the index *range*, which is a property
// of the values rather than the types and is checked in the kernel body: see the
// out-of-range note in scripts/codegen/codegen_mps.py, where a measured
// first-row-returned silent wrong answer is why the range is validated on the
// host before any node is built.
//
// Nor does it cover `padding_idx`, because the forward op does not read it
// either: measured against the CPU reference, `aten::embedding` returns the same
// bytes for every `padding_idx` a caller can pass, including the -1 that the
// Python front end uses for "none" and values outside the table. The zeroing
// that name asks for belongs to `embedding_dense_backward`.
bool MpsSupportsEmbeddingDtype(
    at::ScalarType weight_dtype,
    at::ScalarType index_dtype);

// Whether `take` computes on the MPSGraph path.
//
// It is its own predicate rather than `MpsSupportsEmbeddingDtype` because the
// two ops do not accept the same index, and the difference is the reference's:
// measured, `aten::take` raises `RuntimeError: take(): Expected a long tensor for
// index, but got Int` for an int32 index, where `aten::index_select` accepts it.
// So the index half here is int64 alone, and an int32 index takes the round-trip
// for a `RuntimeError` the graph path could only answer by computing.
//
// The operand half is deliberately *wider* than the embedding gate rather than
// the same three float widths. `aten::take` passes its operand's dtype through
// (measured: float64 stays float64, int64 stays int64, bool stays bool), and it
// is a pure read with no arithmetic for a width to change under, so the widths
// MPSGraph has element types for are all servable. bool and the 64-bit widths
// are not among them and are declined here, which costs the round-trip and keeps
// the reference's own values.
//
// The index *range* is not covered, for the same reason it is not covered by the
// embedding gate: it is a property of the values. It is checked on the host in
// the kernel body, against the *flattened* operand -- `take` indexes
// `self.numel()` elements rather than a single axis -- and a negative index is
// in range there, unlike `index_select`'s.
bool MpsSupportsTakeDtype(
    at::ScalarType self_dtype,
    at::ScalarType index_dtype);

// Whether `self / other` computes on the MPSGraph path.
//
// The gate itself is deliberately the same test `mul` uses: two operands of one
// float width. What is worth recording is the one measured difference from the
// reference, because it is a boundary this branch accepts knowingly rather than
// a case that was missed.
//
// **MPSGraph flushes a subnormal *result* to zero.** Measured on an M5 through
// this backend, not on a standalone graph. The boundary is the result, at the
// smallest normal float: `1e-38 / 2` returns `5.0e-39` on the reference (a
// subnormal) and `0` from the graph, while `2.35e-38 / 2` returns the normal
// `1.175494e-38` from both. It holds for an operand array and for a folded
// constant alike -- `aten::div.Tensor(x, tensor(2.0))`,
// `aten::div.Tensor(x, 2.0)` and `aten::div.Scalar(x, 2.0)` all return `0` here
// -- so it is the graph's arithmetic rather than a property of one overload. A
// reciprocal overflows from the same boundary: `1 / 1.17e-38` is `8.5e37` on the
// reference and `inf` from the graph.
//
// This is not a division property, and the measurement says so plainly:
// `mul.Tensor` flushes the identical way and has been routed to this backend
// since stage 2 (`1e-38 * 0.5` -> `0`), as does `add.Tensor` (`1e-39 + 1e-39`
// -> `0`). So the choice here is "div behaves like the mul and add this branch
// already serves" or "div is the only arithmetic op on the device with a
// different subnormal policy from its three siblings", and the first is both the
// smaller surprise and the one already shipped. Nothing in an attention or
// normalization path evaluates a division whose result is subnormal -- those are
// O(1) by construction -- so this is a boundary the branch documents rather than
// one it meets. test_mps_div.py pins it as intentional beside the ``mul`` case
// it shares.
//
// The other case that separates the plausible readings -- `a / b` against
// `a * reciprocal(b)` -- does **not** show up here: measured against `at::div`
// over the zero and signed-zero denominators, the inf/nan pairs and the
// denormals, both readings are bit-identical to each other *and* to the
// reference. So the selector is a cost decision (one node against two) rather
// than a correctness one, and the two bodies that share it -- `div` and
// `reciprocal` -- are gated together rather than per op.
//
// Integral operands take the round-trip. `div(int, int)` is a float tensor on the
// reference rather than a truncating one, and a *fractional* scalar divisor
// promotes the same way, which one constant of the operand's own element type
// cannot express; float64 and the complex pair have no MPSGraph element type at
// all.
bool MpsSupportsDivDtype(at::ScalarType self_dtype, at::ScalarType other_dtype);

// The scalar-operand spelling of the same question: whether `self / scalar`
// computes on the MPSGraph path.
//
// This is *not* MpsSupportsScalarDtype, for the reason MpsSupportsDivDtype is not
// MpsSupportsBinaryDtype. That predicate is dtype-preserving by construction --
// it admits an integral operand with an integral divisor so the result stays in
// the operand's element type, which is what `mul`/`add`/`sub` need. Division is
// not dtype-preserving: measured on an M5 through this backend,
// `at::div.Scalar(tensor([-7]), 2)` is the float `-3.5` on the reference while
// the graph's integral division truncates and returns the int64 `-3`. So the
// operand has to be a float width, and the divisor is then free (an integral or
// fractional host scalar both land in the float result). The round-trip is what
// keeps the reference's dtype, which is the part a wrong gate gets wrong.
bool MpsSupportsDivScalarDtype(at::ScalarType self_dtype, const at::Scalar& other);

// Whether a matrix multiplication computes exactly on the MPSGraph path. Both
// operands must carry the same dtype -- aten's `mm` and `bmm` promote nothing,
// and a mismatched pair is a reference error rather than a result -- and that
// dtype must be one of the float widths the MPSGraph mul-add is defined for. The
// integral overloads take the CPU round-trip: aten accumulates an integral `mm`
// in the operand's own width and MPSGraph has no matmul for them, so a graph
// would be a guess rather than a measurement.
bool MpsSupportsGemmDtype(at::ScalarType self_dtype, at::ScalarType other_dtype);

// A gemm operand resolved into the layout the graph should read it in: the dense
// buffer to describe, and whether the graph has to swap that buffer's last two
// axes to recover the operand the caller passed.
//
// A transpose of the last two axes of a dense buffer is the one strided layout
// worth carrying into the graph instead of copying, and `nn.Linear` is why it is
// not a corner case: it hands a gemm `weight.t()`. Both halves of that are
// measured on an M5 for a (512, 1024) x (1024, 32768) f32 product: materialising
// the (1024, 32768) weight on the device costs 13 ms, because reading a
// 32768-column buffer down its columns touches a fresh cache line per element --
// more than the product itself -- while the transpose node is free (11.9 ms with
// the transpose in the graph against 13.1 ms with the operand already dense).
//
// `buffer` always holds the operand's `numel()` elements densely from its own
// offset, which is what MpsRunGraph describes: for a transposed operand it is
// the operand's own transpose view, which shares that memory, and otherwise the
// operand is made contiguous. Every other strided layout -- a slice, a
// broadcast -- takes that copy, because a buffer MpsRunGraph reads densely would
// otherwise be read as the wrong bytes.
//
// A body that branches on `transposed` has to name it in its graph key:
// MpsRunGraph folds in dtype and shape, and a transposed (n, n) operand and a
// dense (n, n) one are otherwise one key and one plan.
struct MpsGemmOperand {
  at::Tensor buffer;
  bool transposed;
};

MpsGemmOperand MpsGemmOperandOf(const at::Tensor& t);

// Whether a convolution computes exactly on the MPSGraph path: one float width
// shared by every operand, including the optional bias.
//
// This gate is deliberately wider than the softmax and pow one above, and the
// difference is measured rather than assumed. bfloat16 is *served* here: on an
// M5, a (2, 8, 17, 17) bfloat16 convolution against a 3x3 kernel compared as
// 6.52e-03 against an exact float64 reference -- and a bfloat16 *reference*
// convolution compared as 6.52e-03 against that same exact reference, the same
// number, because bfloat16 holds about three decimal digits and the two
// roundings land on the same value. float32 measured 3.31e-07 against the exact
// reference where the reference itself measured 6.86e-07, so the graph is if
// anything the more accurate of the two.
//
// That is the opposite of what a bfloat16 *softmax* did on the same SDK (a
// denormal for every element), which is why the two categories carry separate
// predicates instead of one dtype list: the measurement is per op, not per
// dtype.
//
// float64 has no MPSGraph element type at all, and an integral or bool
// convolution is a NotImplementedError in the reference rather than a result, so
// all three take the CPU round-trip and get exactly that.
bool MpsSupportsConvDtype(
    at::ScalarType input_dtype,
    at::ScalarType weight_dtype,
    at::ScalarType bias_dtype);

// A convolution's geometry, resolved on the host into what an MPSGraph
// convolution descriptor takes: the per-axis stride, dilation and the
// front/back padding pair each axis carries, the group count, and the (1, OC,
// 1, ...) shape the bias has to be reshaped to before it can be added to the
// result.
//
// aten hands `convolution_overrideable` one flat `padding` list and one flat
// `stride`/`dilation` list per spatial axis, and MPSGraph wants the same
// information split three ways. Both readings of the padding list exist in the
// wild -- symmetric pairs, and a front/back pair per axis -- and aten's is the
// second for an N-D call and the first for a 1-D one; `MpsConvResolve` takes the
// reading that matches the list it is given and reports an unreadable list as a
// decline rather than guessing.
struct MpsConvParams {
  // Per-axis stride and dilation, already known to have `spatial` entries.
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  // Per-axis (front, back), flattened the way the descriptor's
  // `paddingFronts`/`paddingBacks` arrays are filled.
  std::vector<int64_t> padding_front;
  std::vector<int64_t> padding_back;
  // The (1, OC, 1, ...) shape the bias is reshaped to. See T_CONV in
  // scripts/codegen/codegen_mps.py for why this shape rather than (1, ..., OC).
  std::vector<int64_t> bias_shape;
  int64_t groups = 1;
};

// Validates a forward convolution's arguments against the reference's contract
// and fills *params and *out_shape. False is the caller's cue to take the CPU
// round-trip: a shape or argument list the graph cannot be built from at all,
// including the rank, the argument lengths, the channel rules and the padding
// list -- every case where the reference's own error message is the right answer
// and a graph would be the wrong one.
//
// The output shape is the dilation-aware forward formula; a kernel larger than
// the padded input falls out of it as a non-positive extent, which is a decline
// here and the reference's own "Kernel size can't be greater than actual input
// size" there.
bool MpsConvResolve(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    MpsConvParams* params,
    std::vector<int64_t>* out_shape);

// The MPSGraph convolution descriptor for a resolved convolution: the class
// `spatial == 3 ? MPSGraphConvolution3DOpDescriptor
//              : MPSGraphConvolution2DOpDescriptor`, with NCHW/NCDHW data and
// OIHW/OIDHW weights, explicit (asymmetric) padding, and the per-axis stride,
// dilation and padding from `params`.
//
// The two classes are distinct types with distinct selectors, which is why this
// hands back a `MPSGraphObject*` the caller casts at the one place that knows
// which selector it is calling.
// +1, so the caller releases it; nil for a spatial rank this has no descriptor
// for, or a dtype MPSGraph has no element type for.
MPSGraphObject* MpsConvDescriptor(
    at::ScalarType dtype,
    int64_t spatial,
    const MpsConvParams& params);

// The axes `dims` covers in a tensor of `sizes`, written to *out: negative
// positions resolved against the rank, then sorted. A nullopt or EMPTY dims
// list means every axis, because that is what aten means by it -- `sum(x)` and
// `sum(x, [])` are both the full reduction. False for a position outside the
// rank or a repeat, the two inputs aten rejects, so the kernel can take the CPU
// round-trip and let the reference raise.
bool MpsReduceAxes(
    at::IntArrayRef sizes,
    at::OptionalIntArrayRef dims,
    std::vector<int64_t>* out);

// The shape of `sizes` reduced over `dims`, following aten's keepdim rule: with
// keepdim the rank is unchanged and every reduced axis becomes 1, without it
// the reduced axes are dropped. `dims` is normalised the way MpsReduceAxes
// does; the caller has already rejected what that function rejects.
std::vector<int64_t> MpsReduceOutShape(
    at::IntArrayRef sizes,
    at::OptionalIntArrayRef dims,
    bool keepdim);

// `axes` as the NSArray<NSNumber *> the MPSGraph reduction selectors take. An
// empty array is MPSGraph's no-op reduction, which is the right answer for a
// rank-0 operand -- there is nothing to reduce.
NSArray<NSNumber*>* MpsAxesArray(at::IntArrayRef axes);

// A constant holding `value` in its single element, in a dtype `MpsDataTypeOf`
// maps. One element is what MPSGraph broadcasts against an operand of any
// shape, so a constant never materialises a full tensor.
//
// nil for a value the constant path cannot carry exactly, which is the caller's
// signal to take the CPU round-trip: a fractional value in an integral dtype
// (which in aten would have moved the result off the operand's dtype anyway),
// and an int64 beyond +/-2^53, because the constant is handed to MPSGraph as a
// double.
MPSGraphTensor* MpsConstant(
    MPSGraph* graph,
    const at::Scalar& value,
    at::ScalarType dtype);

// Whether `value` is one MpsConstant can express in `dtype`: integral on an
// integral operand -- a fractional constant would promote the aten result off
// the operand's dtype -- within the range a double names exactly, and never
// complex. A body that has to decide *before* it has a graph -- a dtype
// predicate in codegen -- asks this instead of building the node and finding out,
// which is what keeps one predicate and one implementation of the rule.
bool MpsConstantExact(const at::Scalar& value, at::ScalarType dtype);

// `tensor * factor` as an MPSGraph node, with `factor` the host scalar an aten
// `alpha` (or a scalar operand) becomes. Returns `tensor` itself for a factor of
// 1 -- the common case -- so no node and no constant appear then. nil when the
// constant cannot be expressed: see MpsConstant.
MPSGraphTensor* MpsScaledByScalar(
    MPSGraph* graph,
    MPSGraphTensor* tensor,
    const at::Scalar& factor,
    at::ScalarType dtype);

// `tensor ** exponent` as an MPSGraph node, with `exponent` the host scalar an
// aten `pow.Tensor_Scalar` carries. Returns `tensor` itself for an exponent of
// 1 -- which is `x ** 1` for every x, NaN and -0.0 included -- so no node and
// no constant appear then, the same short-circuit MpsScaledByScalar takes.
// nil when the exponent cannot be expressed in `dtype`: see MpsConstant.
MPSGraphTensor* MpsPowByScalar(
    MPSGraph* graph,
    MPSGraphTensor* tensor,
    const at::Scalar& exponent,
    at::ScalarType dtype);

// One string naming a graph *body*: the op, plus every host value the body bakes
// in as a constant (an `alpha`, the axes a reduction covers). The operands' own
// signature -- dtype, shape and element count of every operand and of `out` --
// is folded into the cache key by MpsRunGraph itself, so a body key only has to
// name what the tensors cannot say.
//
// The split is one-sided and worth being blunt about. Folding the operand
// signature in MpsRunGraph makes it impossible for a body key to be too coarse
// *about a tensor* -- but it cannot rescue a body that reads a host value it
// never named: `add(alpha=1)` and `add(alpha=2)` over operands of the same shape
// and dtype would then share one key and the first plan would answer both. That
// is a wrong answer rather than a slow one, so every host value a graph builder
// reads has to appear in this key; KEY_PARTS in scripts/codegen/codegen_mps.py
// is where each op declares its own.
//
// A layout a builder decides for itself from an operand's strides is the same
// kind of host value and has to be named the same way; MpsGemmOperand is the one
// case of it today.
//
// The parts are appended in the order given, each with its own tag, so the key
// distinguishes an int 2 from a float 2.0 exactly where the graph does.
void MpsKeyAppend(std::string& key, const at::Scalar& value);
void MpsKeyAppend(std::string& key, at::IntArrayRef value);
void MpsKeyAppend(std::string& key, int64_t value);
// A bare host float, for the ops whose schema carries one directly rather than
// through a Scalar -- `native_layer_norm`'s `eps`. Its own overload rather than a
// Scalar: `MpsKeyAppend(key, eps)` with a `double` is ambiguous between the
// int64_t and Scalar overloads, and a `double` is what the schema hands over.
void MpsKeyAppend(std::string& key, double value);
void MpsKeyAppend(std::string& key, bool value);
void MpsKeyAppend(std::string& key, const char* value);

template <typename... Parts>
std::string MpsGraphKey(const char* op, const Parts&... parts) {
  std::string key(op);
  (MpsKeyAppend(key, parts), ...);
  return key;
}

// Runs `build` as an MPSGraph over buffer-backed operands and writes the result
// into `out`, reusing a graph already compiled for this `key` and this operand
// signature when there is one.
//
// Every operand is described as a 1-D buffer of its own `numel()` elements and
// reshaped to its aten shape inside the graph (see the row-pitch note above), so
// `build` receives the operands in order, already shaped, and must return a
// tensor with `out.numel()` elements; that result is flattened back for the
// result array. `out` must be a non-empty flagos tensor with the operands'
// dtype.
//
// `build` runs only when a plan has to be constructed -- the graph, its
// placeholders and the compiled executable are then cached under the key -- and
// must be a pure function of the operands' shapes and dtypes plus whatever host
// values `key` names, because a later call with an equal key reuses the graph it
// built without calling it again.
//
// The work is *scheduled* on the current stream's queue and this returns without
// waiting for it: building and compiling a graph per call cost milliseconds
// (measured ~4.5 ms for a one-node graph on an M5, of which ~3.5 ms was the
// build), and a per-call wait for one commit cost another ~370 us, while the
// same op scheduled and left to run costs ~20-75 us. Correctness rests on three
// things this file can rely on instead of a wait:
//
//   * Metal runs one queue in order, so every later op that reads `out` through
//     the same stream is already ordered behind the graph.
//   * Reads that leave the device -- the CPU round-trip a declined op or
//     `.cpu()` takes -- go through the runtime's copy path, which drains.
//   * A sentinel command buffer is committed after the graph
//     (FlagosMPSStreamSentinel), so StreamQuery and StreamSynchronize see this
//     work rather than only what the runtime itself committed. That is what the
//     caching allocator's block-reuse check reads.
//
// False when an operand or `out` is not backed by a buffer the runtime owns
// (MpsBufferOf), or when `build` returns nil to decline, or when the graph does
// not compile; all three are the caller's signal to take the CPU round-trip. A
// missing command queue is not that kind of condition: it means the host has no
// usable Metal device at all, so it is a TORCH_CHECK rather than a fallback.
bool MpsRunGraph(
    const std::string& key,
    at::TensorList inputs,
    const at::Tensor& out,
    MPSGraphTensor* (^build)(MPSGraph* graph, NSArray<MPSGraphTensor*>* args));

// The multi-result form of MpsRunGraph, for the ops that answer with more than
// one tensor -- `native_layer_norm` returns (out, mean, rstd) and
// `native_batch_norm` (out, save_mean, save_var). `outs` is every result in the
// order the aten schema returns them, and `build` returns that many tensors in
// the same order; each is flattened to its own `outs[i]` buffer.
//
// This is a second entry point rather than a widening of MpsRunGraph because the
// single-result contract is what almost every body here rests on: `build` has to
// answer one tensor and the caching allocator's block-reuse reasoning reads one
// `out`. Keeping that signature untouched means the 23 single-result kernels
// cannot be handed a plan that writes somewhere they did not describe.
//
// The result order is positional, and it is the caller's to get right: MPSGraph
// does not name the results, so the `i`th tensor `build` returns is flattened to
// the `i`th entry of `outs` and nothing reconciles the two. Every result must be
// a non-empty flagos tensor with the operands' dtype, and every result's numel is
// folded into the key.
//
// False on the same conditions as MpsRunGraph -- a nil `build`, a result the
// runtime does not own, a `build` that declines (nil or a short vector), or a
// graph that does not compile -- and the caller takes the CPU round-trip. An
// empty `outs` is false rather than a no-op, because there would be nothing to
// schedule and no way to tell that from a decline.
bool MpsRunGraphMulti(
    const std::string& key,
    at::TensorList inputs,
    at::TensorList outs,
    std::vector<MPSGraphTensor*> (^build)(
        MPSGraph* graph,
        NSArray<MPSGraphTensor*>* args));

// `sizes` as an MPSShape. Autoreleased; consume it inside the caller's
// autorelease pool.
MPSShape* MpsShapeOf(at::IntArrayRef sizes);

// MPSGraphTensorData describing the `count` elements of `dtype` that start at
// `byteOffset` inside `buffer`, which must be host-gpu shared. Returns a +1
// reference the caller releases, or nil for `count == 0` (MPSGraph rejects an
// empty tensor; the caller short-circuits those).
//
// The result is one-dimensional and the buffer is read densely from the offset,
// so a caller with a strided operand materialises it first. Shape does not
// belong here: see the row-pitch note above, and describe the shape with
// MpsReshape over a MpsFlatPlaceholder instead.
MPSGraphTensorData* MpsTensorData(
    id<MTLBuffer> buffer,
    size_t byteOffset,
    int64_t count,
    MPSDataType dtype);

// The 1-D placeholder matching a MpsTensorData over `count` elements. This is
// the only shape a buffer-backed tensor is given; `MpsReshape` turns it into the
// aten shape inside the graph.
MPSGraphTensor* MpsFlatPlaceholder(MPSGraph* graph, int64_t count, MPSDataType dtype);

// `tensor` viewed as `shape`, without moving data. Fails the graph run rather
// than silently truncating when the element counts disagree.
MPSGraphTensor* MpsReshape(MPSGraph* graph, MPSGraphTensor* tensor, at::IntArrayRef shape);

// `tensor` viewed as the 1-D `count` elements a MpsTensorData expects, which is
// what a graph result is flattened back to before it is handed to
// `runWithMTLCommandQueue:...resultsDictionary:`.
MPSGraphTensor* MpsFlatten(MPSGraph* graph, MPSGraphTensor* tensor, int64_t count);

// The MTLBuffer backing `t`'s storage and the byte offset of its first element
// inside it. False when the pointer is not inside a Metal allocation the
// runtime owns, or when the allocation is host memory. `t` must be defined and
// on the flagos device; the caller has already made it contiguous.
bool MpsBufferOf(const at::Tensor& t, id<MTLBuffer>* buffer, size_t* byteOffset);

// The MTLCommandQueue the current flagos stream runs commands on. Borrowed from
// the runtime; nil when the host has no usable Metal device.
id<MTLCommandQueue> MpsCurrentQueue();

// Release-and-nil for an object the helper above returned (+1). Kept here so the
// generated kernels do not each re-open the manual-retain/release question.
void MpsRelease(id object);

} // namespace at::native::flagos::mps

#endif // USE_MPS
