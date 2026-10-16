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
