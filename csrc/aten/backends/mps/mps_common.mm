// Copyright (c) 2026, BAAI. All rights reserved.
//
// Implementation of the Apple Metal (MPSGraph) marshalling helpers. See
// mps_common.h for what each one is for.
//
// This file is Objective-C++ without ARC, like the runtime under
// csrc/runtime/accelerator/mps/: objects this file creates are released
// explicitly, and the helpers hand back a +1 reference the caller releases.

#include "mps_common.h"

#ifdef USE_MPS

#include <algorithm>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/util/Exception.h>

// Pulls in the flagos C ABI (Stream_t, GetDevice, GetCurrentStreamForDevice)
// plus the two Metal side-door helpers the operator layer needs, the same way
// every other vendor backend includes runtime/accelerator/<vendor>/.
#include "runtime/accelerator/mps/mps.h"

namespace at::native::flagos::mps {

bool MpsDataTypeOf(at::ScalarType dtype, MPSDataType* out) {
  if (out == nullptr) {
    return false;
  }
  switch (dtype) {
    case at::kFloat:    *out = MPSDataTypeFloat32;  return true;
    case at::kHalf:     *out = MPSDataTypeFloat16;  return true;
    case at::kBFloat16: *out = MPSDataTypeBFloat16; return true;
    case at::kInt:      *out = MPSDataTypeInt32;    return true;
    case at::kLong:     *out = MPSDataTypeInt64;    return true;
    case at::kShort:    *out = MPSDataTypeInt16;    return true;
    case at::kChar:     *out = MPSDataTypeInt8;     return true;
    case at::kByte:     *out = MPSDataTypeUInt8;    return true;
    case at::kBool:     *out = MPSDataTypeBool;     return true;
    // float64 has no MPSGraph element type on this SDK, and the two complex
    // types have none anywhere. Everything else is either a quantised or an
    // unsigned-64 variant with no MPS counterpart either.
    default:            return false;
  }
}

MPSShape* MpsShapeOf(at::IntArrayRef sizes) {
  NSMutableArray<NSNumber*>* dims =
      [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(sizes.size())];
  for (const int64_t dim : sizes) {
    [dims addObject:@(dim)];
  }
  return dims;
}

bool MpsSupportsUnaryDtype(at::ScalarType dtype) {
  MPSDataType ignored;
  return dtype != at::kBool && MpsDataTypeOf(dtype, &ignored);
}

bool MpsSupportsBinaryDtype(at::ScalarType self_dtype, at::ScalarType other_dtype) {
  return self_dtype == other_dtype && MpsSupportsUnaryDtype(self_dtype);
}

bool MpsSupportsScalarDtype(at::ScalarType self_dtype, const at::Scalar& other) {
  if (!MpsSupportsUnaryDtype(self_dtype) || other.isComplex()) {
    return false;
  }
  if (c10::isIntegralType(self_dtype, /*includeBool=*/false)) {
    return c10::isIntegralType(other.type(), /*includeBool=*/false);
  }
  return true;
}

bool MpsSupportsReduceDtype(
    at::ScalarType dtype,
    const std::optional<at::ScalarType>& out_dtype) {
  if (out_dtype.has_value() && *out_dtype != dtype) {
    return false;
  }
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

bool MpsSupportsDivDtype(at::ScalarType self_dtype, at::ScalarType other_dtype) {
  // The float widths only, *not* the mul predicate, even though the two would
  // look interchangeable: `mul` is dtype-preserving on the reference and
  // division is not. Measured, because the difference is a wrong answer rather
  // than an exception -- `at::div(tensor([-7]), tensor([2]))` is the float
  // `-3.5` on the reference while the graph's integral division truncates in
  // place and returns the int64 `-3`. A gate that admitted the integral widths
  // would hand the caller the truncated tensor and a dtype that disagrees with
  // the CPU, so the pair takes the round-trip instead. See the note in
  // mps_common.h. `reciprocal` names this same predicate with both arguments
  // equal to the operand's dtype, so `x / 2` and `1 / x` are served on exactly
  // one width set.
  if (self_dtype != other_dtype) {
    return false;
  }
  return self_dtype == at::kFloat || self_dtype == at::kHalf ||
      self_dtype == at::kBFloat16;
}

bool MpsSupportsDivScalarDtype(
    at::ScalarType self_dtype,
    const at::Scalar& other) {
  // The float widths only, and the divisor is then unconstrained within them:
  // see the note in mps_common.h. A complex divisor has no MPSGraph element
  // type, so it is declined along with the complex operand widths.
  if (other.isComplex()) {
    return false;
  }
  return self_dtype == at::kFloat || self_dtype == at::kHalf ||
      self_dtype == at::kBFloat16;
}

bool MpsSupportsGemmDtype(at::ScalarType self_dtype, at::ScalarType other_dtype) {
  if (self_dtype != other_dtype) {
    return false;
  }
  return self_dtype == at::kFloat || self_dtype == at::kHalf ||
      self_dtype == at::kBFloat16;
}

bool MpsSupportsSoftmaxDtype(
    at::ScalarType self_dtype,
    bool half_to_float,
    at::ScalarType* out_dtype) {
  if (out_dtype == nullptr) {
    return false;
  }
  // The measured widths; see the header for what bfloat16 does instead.
  if (self_dtype != at::kFloat && self_dtype != at::kHalf) {
    return false;
  }
  *out_dtype = half_to_float ? at::kFloat : self_dtype;
  return true;
}

bool MpsSupportsPowDtype(at::ScalarType self_dtype, const at::Scalar& exponent) {
  if (self_dtype != at::kFloat && self_dtype != at::kHalf) {
    return false;
  }
  // A float constant carries any real the caller can hand aten, so the only
  // exponent this declines is a complex one -- which MpsConstant declines too.
  return MpsConstantExact(exponent, self_dtype);
}

bool MpsSupportsSiluDtype(at::ScalarType dtype) {
  // The three widths the composition was measured bit-identical to at::silu on.
  // See the note in mps_common.h: aten's `silu` is not `x * sigmoid(x)`, so this
  // list is a measurement rather than the identity it looks like.
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

bool MpsSupportsEmbeddingDtype(
    at::ScalarType weight_dtype,
    at::ScalarType index_dtype) {
  // The table widths, the same three the silu and conv gates serve.
  const bool table = weight_dtype == at::kFloat || weight_dtype == at::kHalf ||
                     weight_dtype == at::kBFloat16;
  // The reference's own list, not this file's: see the note in mps_common.h.
  const bool index = index_dtype == at::kLong || index_dtype == at::kInt;
  return table && index;
}

bool MpsSupportsTakeDtype(
    at::ScalarType self_dtype,
    at::ScalarType index_dtype) {
  // The widths MPSGraph has element types for. Wider than the embedding gate on
  // purpose: a read does not change its operand's width, so float64/int64/bool
  // are the only ones declined here and they are declined because the *graph*
  // has no element type for them, not because the reference does not.
  const bool readable = self_dtype == at::kFloat || self_dtype == at::kHalf ||
                        self_dtype == at::kBFloat16 || self_dtype == at::kInt ||
                        self_dtype == at::kShort || self_dtype == at::kChar ||
                        self_dtype == at::kByte;
  // int64 alone: measured, `aten::take` raises for int32 where
  // `aten::index_select` accepts it. See the note in mps_common.h.
  const bool index = index_dtype == at::kLong;
  return readable && index;
}

bool MpsSoftmaxDim(const at::Tensor& self, int64_t dim, int64_t* out) {
  if (out == nullptr) {
    return false;
  }
  const int64_t rank = self.dim();
  const int64_t resolved = dim < 0 ? dim + rank : dim;
  if (resolved < 0 || resolved >= rank) {
    return false;
  }
  *out = resolved;
  return true;
}

MpsGemmOperand MpsGemmOperandOf(const at::Tensor& t) {
  if (t.dim() >= 2) {
    // Not a `contiguous()` call: this is the layout test. The tensor's own last
    // two axes are swapped, so the axes in memory are the other way round and
    // the graph can put them back with a transpose node. `data_ptr()` and
    // `numel()` are the same for the view as for the tensor, so the buffer this
    // describes is the caller's own memory either way. See MpsGemmOperand.
    at::Tensor dense = t.transpose(-1, -2);
    if (dense.is_contiguous()) {
      return {dense, true};
    }
  }
  return {t.contiguous(), false};
}

bool MpsSupportsConvDtype(
    at::ScalarType input_dtype,
    at::ScalarType weight_dtype,
    at::ScalarType bias_dtype) {
  if (input_dtype != weight_dtype || weight_dtype != bias_dtype) {
    // The graph runs one element type for the whole node, while aten promotes
    // (a float tensor with a double weight is a float convolution, and a
    // mixed pair is otherwise a reference error), so a mixed call takes the
    // round-trip rather than a graph built at one width.
    return false;
  }
  return input_dtype == at::kFloat || input_dtype == at::kHalf ||
      input_dtype == at::kBFloat16;
}

bool MpsConvResolve(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    MpsConvParams* params,
    std::vector<int64_t>* out_shape) {
  if (params == nullptr || out_shape == nullptr) {
    return false;
  }
  const int64_t dim = input.dim() - 2;
  if (dim < 2 || dim > 3 || weight.dim() != input.dim()) {
    return false;
  }
  const int64_t spatial = dim;
  if (stride.size() != spatial || dilation.size() != spatial ||
      padding.size() != 2 * spatial || groups < 1) {
    return false;
  }
  const int64_t in_channels = input.size(1);
  const int64_t out_channels = weight.size(0);
  if (in_channels / groups != weight.size(1) || out_channels % groups != 0) {
    return false;
  }

  params->stride.assign(stride.begin(), stride.end());
  params->dilation.assign(dilation.begin(), dilation.end());
  params->padding_front.clear();
  params->padding_back.clear();
  // aten's flat `padding` is [front_0, front_1, ..., back_0, back_1, ...], and
  // MPSGraph takes the two halves as two arrays -- the same split, so the first
  // `spatial` entries are the fronts and the rest are the backs.
  for (int64_t axis = 0; axis < spatial; ++axis) {
    params->padding_front.push_back(padding[axis]);
    params->padding_back.push_back(padding[axis + spatial]);
  }
  params->groups = groups;

  // The dilation-aware forward formula, the same arithmetic the reference uses:
  // floor((in + front + back - dilation * (kernel - 1) - 1) / stride + 1).
  std::vector<int64_t> shape;
  shape.push_back(input.size(0));
  shape.push_back(out_channels);
  for (int64_t axis = 0; axis < spatial; ++axis) {
    const int64_t padded = input.size(axis + 2) + params->padding_front[axis] +
        params->padding_back[axis];
    const int64_t window =
        params->dilation[axis] * (weight.size(axis + 2) - 1) + 1;
    const int64_t extent = (padded - window) / params->stride[axis] + 1;
    if (extent <= 0) {
      // What the reference reports as "Kernel size can't be greater than actual
      // input size": there is no output extent to give the graph.
      return false;
    }
    shape.push_back(extent);
  }
  // A non-positive step is a zero-stride graph, and the reference rejects it in
  // its own words; the formula above already divided by it, so this can only
  // catch it before the descriptor does.
  for (int64_t axis = 0; axis < spatial; ++axis) {
    if (params->stride[axis] <= 0 || params->dilation[axis] <= 0) {
      return false;
    }
  }
  *out_shape = std::move(shape);

  // The bias broadcasts over the spatial axes and along the channel axis, so it
  // becomes (1, OC, 1, ...) -- see T_CONV for the two shapes that do not work.
  params->bias_shape.assign(static_cast<size_t>(dim), 1);
  params->bias_shape[1] = out_channels;
  return true;
}

MPSGraphObject* MpsConvDescriptor(
    at::ScalarType dtype,
    int64_t spatial,
    const MpsConvParams& params) {
  MPSDataType mps_dtype;
  if (!MpsDataTypeOf(dtype, &mps_dtype)) {
    // The descriptor carries no dtype -- the selector reads it off the source
    // and weights tensors -- but a dtype with no MPSGraph element type never
    // reaches a graph, so a descriptor for it is a decline.
    return nil;
  }
  const size_t rank = static_cast<size_t>(spatial);
  if (params.stride.size() != rank || params.dilation.size() != rank ||
      params.padding_front.size() != rank ||
      params.padding_back.size() != rank) {
    return nil;
  }
  // The per-axis argument order and the descriptor's X/Y/Z properties run in
  // opposite directions: aten's axis order is the fastest-moving one last, and
  // so is the descriptor's, which is what `p = rank - 1 - axis` maps between.
  // Reading it the other way round is a wrong number rather than a shape error
  // for any non-square kernel, which is why it is spelled out. Measured on the
  // 3-D probe, which transposes exactly the two spatial axes this doc string
  // compares against: only the (2, 3, 4) diag kernel agrees, the (2, 3, 4) and
  // the (3, 2) cases do not.
  const NSUInteger sx = static_cast<NSUInteger>(params.stride[rank - 1]);
  const NSUInteger sy = static_cast<NSUInteger>(params.stride[rank - 2]);
  const NSUInteger dx = static_cast<NSUInteger>(params.dilation[rank - 1]);
  const NSUInteger dy = static_cast<NSUInteger>(params.dilation[rank - 2]);
  const NSUInteger left =
      static_cast<NSUInteger>(params.padding_front[rank - 1]);
  const NSUInteger right =
      static_cast<NSUInteger>(params.padding_back[rank - 1]);
  const NSUInteger top =
      static_cast<NSUInteger>(params.padding_front[rank - 2]);
  const NSUInteger bottom =
      static_cast<NSUInteger>(params.padding_back[rank - 2]);
  const NSUInteger groups = static_cast<NSUInteger>(params.groups);
  if (spatial == 3) {
    const NSUInteger sz = static_cast<NSUInteger>(params.stride[0]);
    const NSUInteger dz = static_cast<NSUInteger>(params.dilation[0]);
    const NSUInteger front =
        static_cast<NSUInteger>(params.padding_front[0]);
    const NSUInteger back = static_cast<NSUInteger>(params.padding_back[0]);
    return [[MPSGraphConvolution3DOpDescriptor alloc]
        initWithStrideInX:sx
                strideInY:sy
                strideInZ:sz
          dilationRateInX:dx
          dilationRateInY:dy
          dilationRateInZ:dz
                   groups:groups
             paddingLeft:left
            paddingRight:right
              paddingTop:top
           paddingBottom:bottom
             paddingFront:front
              paddingBack:back
             paddingStyle:MPSGraphPaddingStyleExplicit
               dataLayout:MPSGraphTensorNamedDataLayoutNCDHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
  }
  if (spatial == 2) {
    return [[MPSGraphConvolution2DOpDescriptor alloc]
        initWithStrideInX:sx
                strideInY:sy
          dilationRateInX:dx
          dilationRateInY:dy
                   groups:groups
             paddingLeft:left
            paddingRight:right
              paddingTop:top
           paddingBottom:bottom
             paddingStyle:MPSGraphPaddingStyleExplicit
               dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
  }
  return nil;
}

bool MpsReduceAxes(
    at::IntArrayRef sizes,
    at::OptionalIntArrayRef dims,
    std::vector<int64_t>* out) {
  if (out == nullptr) {
    return false;
  }
  out->clear();
  const int64_t rank = static_cast<int64_t>(sizes.size());
  if (!dims.has_value() || dims->empty()) {
    for (int64_t axis = 0; axis < rank; ++axis) {
      out->push_back(axis);
    }
    return true;
  }
  for (const int64_t raw : *dims) {
    const int64_t axis = raw < 0 ? raw + rank : raw;
    if (axis < 0 || axis >= rank) {
      return false;
    }
    if (std::find(out->begin(), out->end(), axis) != out->end()) {
      return false;
    }
    out->push_back(axis);
  }
  std::sort(out->begin(), out->end());
  return true;
}

std::vector<int64_t> MpsReduceOutShape(
    at::IntArrayRef sizes,
    at::OptionalIntArrayRef dims,
    bool keepdim) {
  std::vector<int64_t> axes;
  if (!MpsReduceAxes(sizes, dims, &axes)) {
    // Unreachable from a kernel, which validates with MpsReduceAxes first and
    // takes the CPU round-trip when it says no. Keeping the unreduced shape
    // means the at::empty that follows is never wildly wrong.
    return sizes.vec();
  }
  std::vector<int64_t> shape;
  for (int64_t axis = 0; axis < static_cast<int64_t>(sizes.size()); ++axis) {
    const bool reduced =
        std::find(axes.begin(), axes.end(), axis) != axes.end();
    if (!reduced) {
      shape.push_back(sizes[axis]);
    } else if (keepdim) {
      shape.push_back(1);
    }
  }
  return shape;
}

NSArray<NSNumber*>* MpsAxesArray(at::IntArrayRef axes) {
  NSMutableArray<NSNumber*>* out = [NSMutableArray
      arrayWithCapacity:static_cast<NSUInteger>(axes.size())];
  for (const int64_t axis : axes) {
    [out addObject:@(axis)];
  }
  return out;
}

namespace {

// The one-dimensional shape a buffer-backed MPSNDArray is always given. See the
// row-pitch note in mps_common.h: any other shape would make MPSNDArray read the
// buffer with its own padded row stride.
MPSShape* FlatShape(int64_t count) {
  const int64_t dims[1] = {count};
  return MpsShapeOf(at::IntArrayRef(dims, 1));
}

} // namespace

MPSGraphTensorData* MpsTensorData(
    id<MTLBuffer> buffer,
    size_t byteOffset,
    int64_t count,
    MPSDataType dtype) {
  if (buffer == nil || count <= 0) {
    return nil;
  }
  MPSNDArray* array = nil;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  // The offset-carrying initializer is the reason the whole design works: the
  // caching allocator subdivides one MTLBuffer into many blocks, so an operand
  // is the common case and a buffer-relative pointer. It is declared by the
  // *newer* MetalPerformanceShaders framework and only from macOS 15, so this
  // branch is guarded twice -- on the SDK that parses the file (this build is
  // compiled against the SDK, not the host) and on the runtime the process
  // actually loaded.
  //
  // Measured against Xcode 15.4 (the SDK the hosted macos-14 image ships):
  // MPSNDArray.h there declares neither `initWithBuffer:offset:descriptor:` nor
  // `initWithBuffer:descriptor:`. With that SDK in force a later `if
  // (@available(macos 15.0))` does not help, because the selector does not
  // exist at *compile* time -- the compiler resolves the sent to the
  // MPSMatrix initializer of the same name and rejects the argument. Two
  // spellings fail and only one does not, so both the guard and the comment are
  // load-bearing.
  if (__builtin_available(macOS 15.0, *)) {
    MPSNDArrayDescriptor* descriptor =
        [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                               shape:FlatShape(count)];
    array = [[MPSNDArray alloc] initWithBuffer:buffer
                                        offset:static_cast<NSUInteger>(byteOffset)
                                    descriptor:descriptor];
  }
#endif
  if (array == nil) {
    // Older SDK: every MPSNDArray spans its buffer from the start, so an
    // interior pointer cannot be described and the operand is refused rather
    // than silently read from offset 0. MpsRunGraph turns a nil here into a
    // decline, which routes the op to cpu_fallback: correct, just not native.
    return nil;
  }
  MPSGraphTensorData* data =
      [[MPSGraphTensorData alloc] initWithMPSNDArray:array];
  [array release];
  return data;
}

MPSGraphTensor* MpsFlatPlaceholder(MPSGraph* graph, int64_t count, MPSDataType dtype) {
  return [graph placeholderWithShape:FlatShape(count) dataType:dtype name:nil];
}

MPSGraphTensor* MpsReshape(MPSGraph* graph, MPSGraphTensor* tensor, at::IntArrayRef shape) {
  return [graph reshapeTensor:tensor withShape:MpsShapeOf(shape) name:nil];
}

MPSGraphTensor* MpsFlatten(MPSGraph* graph, MPSGraphTensor* tensor, int64_t count) {
  return [graph reshapeTensor:tensor withShape:FlatShape(count) name:nil];
}

bool MpsConstantExact(const at::Scalar& value, at::ScalarType dtype) {
  MPSDataType ignored;
  if (!MpsDataTypeOf(dtype, &ignored)) {
    return false;
  }
  if (c10::isIntegralType(dtype, /*includeBool=*/false)) {
    // An integral constant has to be integral and exactly representable: a
    // fractional one promotes the aten result off the operand's dtype, and past
    // 2^53 a double no longer names the integer the scalar held.
    if (!value.isIntegral(/*includeBool=*/true)) {
      return false;
    }
    const int64_t integer = value.toLong();
    constexpr int64_t kLargestExact = int64_t{1} << 53;
    return integer <= kLargestExact && integer >= -kLargestExact;
  }
  return !value.isComplex();
}

MPSGraphTensor* MpsConstant(
    MPSGraph* graph,
    const at::Scalar& value,
    at::ScalarType dtype) {
  if (!MpsConstantExact(value, dtype) || graph == nil) {
    return nil;
  }
  MPSDataType mps_dtype;
  if (!MpsDataTypeOf(dtype, &mps_dtype)) {
    return nil;
  }
  if (c10::isIntegralType(dtype, /*includeBool=*/false)) {
    // MpsConstantExact has already established that the value names an integer
    // this dtype can hold exactly.
    return [graph constantWithScalar:static_cast<double>(value.toLong())
                               shape:FlatShape(1)
                            dataType:mps_dtype];
  }
  return [graph constantWithScalar:value.toDouble()
                             shape:FlatShape(1)
                          dataType:mps_dtype];
}

MPSGraphTensor* MpsScaledByScalar(
    MPSGraph* graph,
    MPSGraphTensor* tensor,
    const at::Scalar& factor,
    at::ScalarType dtype) {
  if (tensor == nil) {
    return nil;
  }
  if (factor.equal(1.0)) {
    // Every plain `+` and `-` carries alpha == 1, so the common case adds no
    // node: the operand is the factor-1 product.
    return tensor;
  }
  MPSGraphTensor* constant = MpsConstant(graph, factor, dtype);
  if (constant == nil) {
    return nil;
  }
  return [graph multiplicationWithPrimaryTensor:tensor
                                secondaryTensor:constant
                                           name:nil];
}

MPSGraphTensor* MpsPowByScalar(
    MPSGraph* graph,
    MPSGraphTensor* tensor,
    const at::Scalar& exponent,
    at::ScalarType dtype) {
  if (tensor == nil) {
    return nil;
  }
  if (exponent.equal(1.0)) {
    return tensor;
  }
  // One MPSGraph selector, `powerWithPrimaryTensor:`, takes any real exponent
  // as a broadcast one-element constant. It is not a composition of exp and
  // log: those would have to agree with the reference on the sign of a
  // negative base raised to a fractional power, which is where the measured
  // NaN comes from.
  MPSGraphTensor* constant = MpsConstant(graph, exponent, dtype);
  if (constant == nil) {
    return nil;
  }
  return [graph powerWithPrimaryTensor:tensor
                       secondaryTensor:constant
                                  name:nil];
}

bool MpsBufferOf(const at::Tensor& t, id<MTLBuffer>* buffer, size_t* byteOffset) {
  if (buffer == nullptr || byteOffset == nullptr || !t.defined()) {
    return false;
  }
  void* const ptr = t.data_ptr();
  if (ptr == nullptr) {
    return false;
  }
  size_t offset = 0;
  void* const raw = ::FlagosMPSBufferForPointer(ptr, &offset);
  if (raw == nullptr) {
    return false;
  }
  // Borrowed from the runtime's registry: the allocation owns it, and it
  // outlives this call because the tensor is alive across it.
  *buffer = (id<MTLBuffer>)raw;
  *byteOffset = offset;
  return true;
}

// The current stream, which is the queue MpsCurrentQueue() hands back. Internal:
// flagos.h's Stream_t is a runtime type the header does not name.
static Stream_t MpsCurrentStream() {
  int device = 0;
  if (::GetDevice(&device) != ::Success) {
    return nullptr;
  }
  return ::GetCurrentStreamForDevice(device);
}

id<MTLCommandQueue> MpsCurrentQueue() {
  Stream_t stream = MpsCurrentStream();
  if (stream == nullptr) {
    return nil;
  }
  return (id<MTLCommandQueue>)::FlagosMPSCommandQueue(stream);
}

// ---------------------------------------------------------------------------
// Graph key -> compiled plan
//
// The measured reason this cache exists: on an M5 a one-node graph built and run
// per call cost ~4.5 ms (128 elements: 4751 us for `abs`, 6458 us for `mul`), and
// a bare Metal probe attributes ~3.5-4.5 ms of that to `[[MPSGraph alloc] init]`
// plus node construction and only ~370 us to running an already-compiled graph.
// A per-call wait is the second half of the problem: the same executable left
// asynchronous costs ~18 us a call. So a plan is built once per key and the op
// is scheduled, not awaited.
// ---------------------------------------------------------------------------

namespace {

struct MpsPlan {
  MPSGraph* graph;                                   // +1
  // +1 each, the flattened targets, in the order the caller returns them. One
  // entry for every single-result body (the MpsRunGraph path) and one per
  // result for the multi-result ones (MpsRunGraphMulti). A vector rather than a
  // lone pointer so both paths share one plan cache and one executable: the
  // caching allocator reads `outs` on the host either way, and MPSGraph's
  // resultsArray is already an array.
  std::vector<MPSGraphTensor*> results;
  MPSGraphExecutable* executable;                    // +1
  MPSGraphExecutableExecutionDescriptor* descriptor; // +1, waitUntilCompleted == NO
  // For each slot of the executable's inputs array, the operand whose tensor
  // data goes there. MPSGraph documents that array as "the same order as
  // arguments of main function"; reading the permutation out of `feedTensors`
  // instead of trusting creation order means an executable that reports its
  // feeds differently still gets the right operand. Stored in the direction the
  // array is filled, so a run appends operands without a placeholder.
  std::vector<NSUInteger> input_order;
};

// Leaked deliberately, like the runtime's default stream: these own Objective-C
// objects whose release at static-destruction time is not ours to order, and the
// process is exiting anyway. The map is bounded, so a long run cannot grow it
// without limit.
std::unordered_map<std::string, MpsPlan*>& PlanCache() {
  static auto* cache = new std::unordered_map<std::string, MpsPlan*>();
  return *cache;
}

// Insertion order, for evicting the oldest plan once the cache is full.
std::deque<std::string>& PlanOrder() {
  static auto* order = new std::deque<std::string>();
  return *order;
}

// Enough for the handful of (dtype, shape) signatures a model actually presents,
// with room for the shapes a growing sequence length adds, and small enough that
// the compiled graphs cannot pile up. Each plan holds one graph plus one
// executable; a decode loop that presents a new sequence length per token is the
// case this bounds.
constexpr size_t kMaxPlans = 64;

// MPSGraph is not documented as thread-safe, and a plan is shared mutable state,
// so the whole of MpsRunGraph -- lookup, build, schedule -- holds this.
std::mutex& PlansMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

void ReleasePlan(MpsPlan* plan) {
  MpsRelease(plan->descriptor);
  MpsRelease(plan->executable);
  for (MPSGraphTensor* result : plan->results) {
    MpsRelease(result);
  }
  MpsRelease(plan->graph);
  delete plan;
}

// The operands' own signature, folded into every key: dtype and shape of each
// operand and of `out`. It is what lets a body key name only the host constants
// a graph bakes in (see mps_common.h).
void MpsKeyAppendTensor(std::string& key, const at::Tensor& tensor) {
  key.push_back('|');
  key += std::to_string(static_cast<int>(tensor.scalar_type()));
  key.push_back('[');
  for (const int64_t dim : tensor.sizes()) {
    key += std::to_string(dim);
    key.push_back(',');
  }
  key.push_back(']');
}

// Builds one plan: placeholders, the graph `build` returns, and the executable
// MPSGraph compiles from it. nil (and nothing retained) when `build` declines or
// the graph does not compile -- both are the caller's cue for the CPU
// round-trip.
//
// `build` returns one tensor per entry of `outs`, in that order. A nil tensor in
// the vector, a vector of the wrong length, and a nil `build` are all the same
// decline.
MpsPlan* BuildPlan(
    const std::string& key,
    at::TensorList inputs,
    at::TensorList outs,
    MPSGraphDevice* device,
    std::vector<MPSGraphTensor*> (^build)(
        MPSGraph* graph,
        NSArray<MPSGraphTensor*>* args)) {
  if (outs.empty()) {
    return nullptr;
  }
  // Every result is flattened back to a 1-D buffer of its own numel. Only the
  // first result's dtype is consulted: all results share the operands' dtype,
  // which MpsRunGraphMulti's caller has already established.
  MPSDataType out_dtype;
  if (!MpsDataTypeOf(outs[0].scalar_type(), &out_dtype)) {
    return nullptr;
  }
  MPSGraph* graph = [[MPSGraph alloc] init];
  if (graph == nil) {
    return nullptr;
  }
  std::vector<MPSGraphTensor*> results;
  MPSGraphExecutable* executable = nil;
  NSArray<MPSGraphTensor*>* placeholders = nil;
  @autoreleasepool {
    NSMutableArray<MPSGraphTensor*>* args =
        [NSMutableArray arrayWithCapacity:inputs.size()];
    NSMutableArray<MPSGraphTensor*>* buffers =
        [NSMutableArray arrayWithCapacity:inputs.size()];
    NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
        [NSMutableDictionary dictionaryWithCapacity:inputs.size()];
    for (size_t i = 0; i < inputs.size(); ++i) {
      MPSDataType dtype;
      if (!MpsDataTypeOf(inputs[i].scalar_type(), &dtype)) {
        break;
      }
      const int64_t count = inputs[i].numel();
      // The placeholder is the 1-D buffer view and the operand's aten shape is
      // reapplied over it, because an MPSNDArray over N-D memory reads with its
      // own row pitch (see the row-pitch note in mps_common.h).
      MPSGraphTensor* buffer = MpsFlatPlaceholder(graph, count, dtype);
      const int64_t dims[1] = {count};
      MPSGraphShapedType* type = [[MPSGraphShapedType alloc]
          initWithShape:MpsShapeOf(at::IntArrayRef(dims, 1))
               dataType:dtype];
      [feeds setObject:type forKey:buffer];
      MpsRelease(type);
      [buffers addObject:buffer];
      [args addObject:MpsReshape(graph, buffer, inputs[i].sizes())];
    }
    if (static_cast<size_t>([buffers count]) == inputs.size()) {
      std::vector<MPSGraphTensor*> built = build == nil
          ? std::vector<MPSGraphTensor*>()
          : build(graph, args);
      bool complete = built.size() == outs.size();
      for (size_t i = 0; complete && i < built.size(); ++i) {
        complete = built[i] != nil;
      }
      if (complete) {
        // One flatten per result, and each is a compile target in its own right:
        // MPSGraph returns them positionally, in the order they are listed here,
        // which is the order `outs` is written back in.
        NSMutableArray<MPSGraphTensor*>* targets =
            [NSMutableArray arrayWithCapacity:outs.size()];
        for (size_t i = 0; i < built.size(); ++i) {
          MPSGraphTensor* flattened =
              [MpsFlatten(graph, built[i], outs[i].numel()) retain];
          results.push_back(flattened);
          [targets addObject:flattened];
        }
        executable = [[graph compileWithDevice:device
                                        feeds:feeds
                                targetTensors:targets
                             targetOperations:nil
                        compilationDescriptor:nil] retain];
      }
    }
    placeholders = [buffers retain];
  }
  if (executable == nil) {
    MpsRelease(placeholders);
    for (MPSGraphTensor* result : results) {
      MpsRelease(result);
    }
    MpsRelease(graph);
    return nullptr;
  }
  auto* plan = new MpsPlan();
  plan->graph = graph;
  plan->results = std::move(results);
  plan->executable = executable;
  plan->descriptor = [[MPSGraphExecutableExecutionDescriptor alloc] init];
  plan->descriptor.waitUntilCompleted = NO;
  // Sentinel: a slot no operand has claimed yet.
  plan->input_order.assign(inputs.size(), inputs.size());
  NSArray<MPSGraphTensor*>* reported = executable.feedTensors;
  bool resolved = (reported != nil && reported.count == inputs.size());
  for (size_t i = 0; resolved && i < inputs.size(); ++i) {
    const NSUInteger position = [reported indexOfObject:placeholders[i]];
    if (position == NSNotFound || position >= inputs.size() ||
        plan->input_order[position] != inputs.size()) {
      resolved = false;
      break;
    }
    plan->input_order[position] = i;
  }
  if (!resolved) {
    // Fall back to the order MPSGraph documents ("the same order as arguments
    // of main function") rather than failing. This is not a guess between
    // equally likely readings: an operand whose type does not match the feed it
    // lands on does not raise, it aborts the process, which is a far worse way
    // to learn that `feedTensors` went away than a rebuilt-on-every-call plan
    // would be.
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
      plan->input_order[slot] = slot;
    }
  }
  MpsRelease(placeholders);
  if (PlanCache().size() >= kMaxPlans) {
    auto* oldest = PlanCache()[PlanOrder().front()];
    PlanCache().erase(PlanOrder().front());
    PlanOrder().pop_front();
    ReleasePlan(oldest);
  }
  PlanCache()[key] = plan;
  PlanOrder().push_back(key);
  return plan;
}

} // namespace

namespace {

void AppendDouble(std::string& key, double value) {
  // 17 significant digits round-trip every double, so two scalars that would
  // build different constants always produce different keys.
  char text[32];
  std::snprintf(text, sizeof(text), "%.17g", value);
  key += text;
}

} // namespace

void MpsKeyAppend(std::string& key, const at::Scalar& value) {
  key += "|s";
  if (value.isComplex()) {
    const c10::complex<double> complex = value.toComplexDouble();
    key.push_back('c');
    AppendDouble(key, complex.real());
    key.push_back(',');
    AppendDouble(key, complex.imag());
    return;
  }
  if (value.isSymbolic()) {
    key += "y";
    return;
  }
  if (value.isIntegral(/*includeBool=*/true)) {
    key.push_back('i');
    key += std::to_string(value.toLong());
    return;
  }
  key.push_back('f');
  AppendDouble(key, value.toDouble());
}

void MpsKeyAppend(std::string& key, at::IntArrayRef value) {
  key += "|a[";
  for (const int64_t element : value) {
    key += std::to_string(element);
    key.push_back(',');
  }
  key.push_back(']');
}

void MpsKeyAppend(std::string& key, int64_t value) {
  key += "|n";
  key += std::to_string(value);
}

void MpsKeyAppend(std::string& key, double value) {
  key += "|d";
  AppendDouble(key, value);
}

void MpsKeyAppend(std::string& key, bool value) {
  key += value ? "|b1" : "|b0";
}

void MpsKeyAppend(std::string& key, const char* value) {
  key += "|t";
  key += (value == nullptr ? "" : value);
}

namespace {

// The shared body of MpsRunGraph and MpsRunGraphMulti. `outs` is that function's
// result list -- one entry for the single-result entry point. Everything the two
// have in common lives here: buffer resolution, the key, the plan cache and the
// asynchronous schedule.
bool RunGraphImpl(
    const std::string& key,
    at::TensorList inputs,
    at::TensorList outs,
    std::vector<MPSGraphTensor*> (^build)(
        MPSGraph* graph,
        NSArray<MPSGraphTensor*>* args)) {
  if (build == nil || outs.empty()) {
    return false;
  }
  // `defined` and a non-empty numel are the two things a body's `out` has always
  // had to satisfy, now checked per result. An empty result is a decline rather
  // than an error, exactly as before: MPSGraph has no empty tensor and the
  // caller's body has already computed the right answer for that case.
  std::vector<MPSDataType> out_dtypes(outs.size(), MPSDataTypeFloat32);
  for (size_t i = 0; i < outs.size(); ++i) {
    if (!outs[i].defined() || outs[i].numel() <= 0) {
      return false;
    }
    TORCH_CHECK(
        MpsDataTypeOf(outs[i].scalar_type(), &out_dtypes[i]),
        "mps: no MPSGraph element type for ",
        outs[i].scalar_type());
  }
  // Resolve every buffer before the graph is built, so a pointer the Metal
  // allocator does not own costs the caller a fallback rather than half a run.
  std::vector<id<MTLBuffer>> buffers(inputs.size(), nil);
  std::vector<size_t> offsets(inputs.size(), 0);
  std::vector<MPSDataType> dtypes(inputs.size(), MPSDataTypeFloat32);
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!MpsBufferOf(inputs[i], &buffers[i], &offsets[i])) {
      return false;
    }
    TORCH_CHECK(
        MpsDataTypeOf(inputs[i].scalar_type(), &dtypes[i]),
        "mps: no MPSGraph element type for ",
        inputs[i].scalar_type());
  }
  std::vector<id<MTLBuffer>> out_buffers(outs.size(), nil);
  std::vector<size_t> out_offsets(outs.size(), 0);
  for (size_t i = 0; i < outs.size(); ++i) {
    if (!MpsBufferOf(outs[i], &out_buffers[i], &out_offsets[i])) {
      return false;
    }
  }
  Stream_t stream = MpsCurrentStream();
  id<MTLCommandQueue> queue = MpsCurrentQueue();
  TORCH_CHECK(
      queue != nil, "mps: no Metal command queue for the flagos device");
  // The caller's key names the body and its host constants; the operand
  // signature is added here, which is what keeps a body key from being too coarse
  // about a tensor. It cannot keep a body from reading a host value the caller
  // never named -- see the key contract in mps_common.h.
  std::string full_key = key;
  for (const at::Tensor& input : inputs) {
    MpsKeyAppendTensor(full_key, input);
  }
  // Every result's signature is folded in, not just the first: a multi-result
  // body can be called with the same operand shapes and a different (out, mean,
  // rstd) split -- `native_layer_norm` with a different `normalized_shape` has
  // the same input and a different mean. The result count is named too, so a
  // single-result and a multi-result body cannot collide on the same key.
  full_key += "|results";
  full_key += std::to_string(outs.size());
  for (const at::Tensor& out : outs) {
    MpsKeyAppendTensor(full_key, out);
  }

  std::lock_guard<std::mutex> lock(PlansMutex());
  MpsPlan* plan = nullptr;
  const auto found = PlanCache().find(full_key);
  if (found != PlanCache().end()) {
    plan = found->second;
  } else {
    @autoreleasepool {
      MPSGraphDevice* device =
          [MPSGraphDevice deviceWithMTLDevice:queue.device];
      plan = BuildPlan(full_key, inputs, outs, device, build);
    }
    if (plan == nullptr) {
      return false;
    }
  }
  bool ran = false;
  @autoreleasepool {
    std::vector<MPSGraphTensorData*> operands(inputs.size(), nil);
    for (size_t i = 0; i < inputs.size(); ++i) {
      operands[i] =
          MpsTensorData(buffers[i], offsets[i], inputs[i].numel(), dtypes[i]);
      // nil is either a zero-element tensor -- which the caller already
      // short-circuits for `out` and which a graph could not express anyway --
      // or an operand whose byte offset the SDK in force cannot describe. Both
      // are declines, not errors: the header contract is that a nil MPSGraph
      // description sends the op to the CPU round-trip, which is correct.
      if (operands[i] == nil) {
        for (MPSGraphTensorData* operand : operands) {
          MpsRelease(operand);
        }
        return false;
      }
    }
    // One MPSGraphTensorData per result, in the same positional order the
    // executable's target tensors were listed in, which is the order `outs` is
    // written back in. A nil here is the same decline as a nil operand.
    NSMutableArray<MPSGraphTensorData*>* result_data =
        [NSMutableArray arrayWithCapacity:outs.size()];
    for (size_t i = 0; i < outs.size(); ++i) {
      MPSGraphTensorData* data =
          MpsTensorData(out_buffers[i], out_offsets[i], outs[i].numel(),
                        out_dtypes[i]);
      if (data == nil) {
        for (MPSGraphTensorData* operand : operands) {
          MpsRelease(operand);
        }
        return false;
      }
      [result_data addObject:data];
    }
    // Held by the array (its own +1) and released again below, once the run has
    // been scheduled: the executable reads them during that run, so they have to
    // outlive the call only as long as the call.
    NSMutableArray<MPSGraphTensorData*>* array =
        [NSMutableArray arrayWithCapacity:inputs.size()];
    for (const NSUInteger operand : plan->input_order) {
      [array addObject:operands[operand]];
    }
    // Asynchronous on purpose: see the contract in mps_common.h. Measured on an
    // M5, the same one-node executable costs ~18 us scheduled this way against
    // ~370 us when awaited inline, because the wait is a round trip to the GPU
    // per op rather than the work itself.
    [plan->executable runAsyncWithMTLCommandQueue:queue
                                      inputsArray:array
                                     resultsArray:result_data
                              executionDescriptor:plan->descriptor];
    ran = true;
    for (MPSGraphTensorData* data : result_data) {
      MpsRelease(data);
    }
    for (MPSGraphTensorData* operand : operands) {
      MpsRelease(operand);
    }
  }
  if (ran) {
    // Puts the graph's work behind a buffer the stream layer can see, without
    // waiting for it: a later StreamQuery, StreamSynchronize or DeviceSynchronize
    // -- what allocator block reuse and a host read ask -- now covers this op.
    ::FlagosMPSStreamSentinel(stream);
  }
  return ran;
}

} // namespace

bool MpsRunGraph(
    const std::string& key,
    at::TensorList inputs,
    const at::Tensor& out,
    MPSGraphTensor* (^build)(MPSGraph* graph, NSArray<MPSGraphTensor*>* args)) {
  if (build == nil) {
    return false;
  }
  // The single-result block lifted into the vector form RunGraphImpl expects.
  // `out` is captured by value (it is a Tensor, a cheap refcount bump) so the
  // wrapper's lifetime is independent of the caller's reference.
  auto multi = ^std::vector<MPSGraphTensor*>(
      MPSGraph* graph, NSArray<MPSGraphTensor*>* args) {
    std::vector<MPSGraphTensor*> one;
    MPSGraphTensor* built = build(graph, args);
    if (built != nil) {
      one.push_back(built);
    }
    return one;
  };
  return RunGraphImpl(
      key, inputs, at::TensorList(&out, 1), multi);
}

bool MpsRunGraphMulti(
    const std::string& key,
    at::TensorList inputs,
    at::TensorList outs,
    std::vector<MPSGraphTensor*> (^build)(
        MPSGraph* graph,
        NSArray<MPSGraphTensor*>* args)) {
  return RunGraphImpl(key, inputs, outs, build);
}

void MpsRelease(id object) {
  [object release];
}

} // namespace at::native::flagos::mps

#endif // USE_MPS
