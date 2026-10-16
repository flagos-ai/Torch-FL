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
  MPSGraphTensor* result;                            // +1, the flattened target
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
  MpsRelease(plan->result);
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
MpsPlan* BuildPlan(
    const std::string& key,
    at::TensorList inputs,
    const at::Tensor& out,
    MPSGraphDevice* device,
    MPSGraphTensor* (^build)(MPSGraph* graph, NSArray<MPSGraphTensor*>* args)) {
  const int64_t out_numel = out.numel();
  MPSDataType out_dtype;
  if (!MpsDataTypeOf(out.scalar_type(), &out_dtype)) {
    return nullptr;
  }
  MPSGraph* graph = [[MPSGraph alloc] init];
  if (graph == nil) {
    return nullptr;
  }
  MPSGraphTensor* result = nil;
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
      MPSGraphTensor* built = build(graph, args);
      if (built != nil) {
        result = [MpsFlatten(graph, built, out_numel) retain];
        executable = [[graph compileWithDevice:device
                                        feeds:feeds
                                targetTensors:@[result]
                             targetOperations:nil
                        compilationDescriptor:nil] retain];
      }
    }
    placeholders = [buffers retain];
  }
  if (executable == nil) {
    MpsRelease(placeholders);
    MpsRelease(result);
    MpsRelease(graph);
    return nullptr;
  }
  auto* plan = new MpsPlan();
  plan->graph = graph;
  plan->result = result;
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

void MpsKeyAppend(std::string& key, bool value) {
  key += value ? "|b1" : "|b0";
}

void MpsKeyAppend(std::string& key, const char* value) {
  key += "|t";
  key += (value == nullptr ? "" : value);
}

bool MpsRunGraph(
    const std::string& key,
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
  id<MTLBuffer> out_buffer = nil;
  size_t out_offset = 0;
  if (!MpsBufferOf(out, &out_buffer, &out_offset)) {
    return false;
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
  MpsKeyAppendTensor(full_key, out);

  std::lock_guard<std::mutex> lock(PlansMutex());
  MpsPlan* plan = nullptr;
  const auto found = PlanCache().find(full_key);
  if (found != PlanCache().end()) {
    plan = found->second;
  } else {
    @autoreleasepool {
      MPSGraphDevice* device =
          [MPSGraphDevice deviceWithMTLDevice:queue.device];
      plan = BuildPlan(full_key, inputs, out, device, build);
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
      TORCH_CHECK(
          operands[i] != nil,
          "mps: cannot describe a zero-element tensor to MPSGraph");
    }
    MPSGraphTensorData* result_data =
        MpsTensorData(out_buffer, out_offset, out.numel(), out_dtype);
    TORCH_CHECK(
        result_data != nil,
        "mps: cannot describe a zero-element tensor to MPSGraph");
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
                                     resultsArray:@[result_data]
                              executionDescriptor:plan->descriptor];
    ran = true;
    MpsRelease(result_data);
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

void MpsRelease(id object) {
  [object release];
}

} // namespace at::native::flagos::mps

#endif // USE_MPS
