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
  MPSNDArrayDescriptor* descriptor =
      [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                             shape:FlatShape(count)];
  MPSNDArray* array = [[MPSNDArray alloc] initWithBuffer:buffer
                                                  offset:static_cast<NSUInteger>(byteOffset)
                                              descriptor:descriptor];
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

MPSGraphTensor* MpsConstant(
    MPSGraph* graph,
    const at::Scalar& value,
    at::ScalarType dtype) {
  MPSDataType mps_dtype;
  if (graph == nil || !MpsDataTypeOf(dtype, &mps_dtype)) {
    return nil;
  }
  if (c10::isIntegralType(dtype, /*includeBool=*/false)) {
    // An integral operand takes an integral constant only. A fractional one
    // would promote the aten result off the operand's dtype -- `int + 0.5` is a
    // float tensor -- which an integral element type cannot express, so the
    // constant is declined and the op takes the CPU round-trip.
    if (!value.isIntegral(/*includeBool=*/true)) {
      return nil;
    }
    const int64_t integer = value.toLong();
    constexpr int64_t kLargestExact = int64_t{1} << 53;
    if (integer > kLargestExact || integer < -kLargestExact) {
      return nil;
    }
    return [graph constantWithScalar:static_cast<double>(integer)
                               shape:FlatShape(1)
                            dataType:mps_dtype];
  }
  if (value.isComplex()) {
    return nil;
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

id<MTLCommandQueue> MpsCurrentQueue() {
  int device = 0;
  if (::GetDevice(&device) != ::Success) {
    return nil;
  }
  Stream_t stream = ::GetCurrentStreamForDevice(device);
  return (id<MTLCommandQueue>)::FlagosMPSCommandQueue(stream);
}

bool MpsRunGraph(
    at::TensorList inputs,
    const at::Tensor& out,
    MPSGraphTensor* (^build)(MPSGraph* graph, NSArray<MPSGraphTensor*>* args)) {
  if (build == nil || !out.defined() || out.numel() <= 0) {
    return false;
  }
  MPSDataType out_dtype;
  TORCH_CHECK(
      MpsDataTypeOf(out.scalar_type(), &out_dtype),
      "mps: no MPSGraph element type for ",
      out.scalar_type());
  // Resolve every buffer before the graph is built, so a pointer the Metal
  // allocator does not own costs the caller a fallback rather than half a run.
  std::vector<id<MTLBuffer>> buffers(inputs.size(), nil);
  std::vector<size_t> offsets(inputs.size(), 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!MpsBufferOf(inputs[i], &buffers[i], &offsets[i])) {
      return false;
    }
  }
  id<MTLBuffer> out_buffer = nil;
  size_t out_offset = 0;
  if (!MpsBufferOf(out, &out_buffer, &out_offset)) {
    return false;
  }
  id<MTLCommandQueue> queue = MpsCurrentQueue();
  TORCH_CHECK(
      queue != nil, "mps: no Metal command queue for the flagos device");
  bool ran = false;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [NSMutableDictionary dictionaryWithCapacity:inputs.size()];
    NSMutableArray<MPSGraphTensor*>* args =
        [NSMutableArray arrayWithCapacity:inputs.size()];
    std::vector<MPSGraphTensorData*> operands;
    operands.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      MPSDataType dtype;
      TORCH_CHECK(
          MpsDataTypeOf(inputs[i].scalar_type(), &dtype),
          "mps: no MPSGraph element type for ",
          inputs[i].scalar_type());
      MPSGraphTensorData* operand =
          MpsTensorData(buffers[i], offsets[i], inputs[i].numel(), dtype);
      TORCH_CHECK(
          operand != nil,
          "mps: cannot describe a zero-element tensor to MPSGraph");
      MPSGraphTensor* buffer =
          MpsFlatPlaceholder(graph, inputs[i].numel(), dtype);
      [feeds setObject:operand forKey:buffer];
      [args addObject:MpsReshape(graph, buffer, inputs[i].sizes())];
      operands.push_back(operand);
    }
    MPSGraphTensorData* result_data =
        MpsTensorData(out_buffer, out_offset, out.numel(), out_dtype);
    TORCH_CHECK(
        result_data != nil,
        "mps: cannot describe a zero-element tensor to MPSGraph");
    MPSGraphTensor* result = build(graph, args);
    if (result != nil) {
      MPSGraphTensor* flattened = MpsFlatten(graph, result, out.numel());
      // Synchronous: runWithMTLCommandQueue: commits to the queue and returns
      // once the results are written, so `out` is ready when this returns and
      // no explicit stream drain is needed. The command buffer lands on the
      // same queue the runtime's copies use, so it stays ordered behind any op
      // that produced an operand.
      [graph runWithMTLCommandQueue:queue
                              feeds:feeds
                   targetOperations:nil
                  resultsDictionary:@{flattened: result_data}];
      ran = true;
    }
    MpsRelease(graph);
    MpsRelease(result_data);
    for (MPSGraphTensorData* operand : operands) {
      MpsRelease(operand);
    }
  }
  return ran;
}

void MpsRelease(id object) {
  [object release];
}

} // namespace at::native::flagos::mps

#endif // USE_MPS
