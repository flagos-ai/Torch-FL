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
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/util/Exception.h>

// The host-round-trip spelling the softmax declines ask for. Included here
// rather than left to the generated kernel file so the helper and its only
// callers agree on `at::_softmax` resolving to the composite-free operator.
#include <ATen/ops/_softmax.h>

// Pulls in the flagos C ABI (Stream_t, GetDevice, GetCurrentStreamForDevice)
// plus the two Metal side-door helpers the operator layer needs, the same way
// every other vendor backend includes runtime/accelerator/<vendor>/.
#include "runtime/accelerator/mps/mps.h"
// and the C++ runtime surface: DrainAll, the host-read barrier the staging
// copies live behind.
#include "runtime/accelerator/mps/mps_runtime.h"
// and the allocator, whose record_stream is what keeps a block a queued graph
// is still reading out of the free pool until that work completes.
#include "runtime/allocator/caching_device_allocator.h"

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

bool MpsSupportsWhereDtype(
    at::ScalarType self_dtype,
    at::ScalarType other_dtype,
    at::ScalarType condition_dtype) {
  // The two branches take the binary pair's rule: the select builds one node in
  // one element type and promotes nothing, where the reference promotes
  // `self`/`other` to a common dtype (`int32` against `int64` is `int64`, half
  // against float is float), so a promoted pair takes the round-trip. See
  // T_WHERE.
  if (!MpsSupportsBinaryDtype(self_dtype, other_dtype)) {
    return false;
  }
  // The condition is the reference's own narrowing, not the graph's: `where`
  // accepts a bool condition and, with a deprecation warning, a uint8 one, and
  // raises for every other width (`where expected condition to be a boolean
  // tensor, but got a tensor with dtype Int`). Bool is the node's predicate
  // type; uint8 is widened by a comparison against zero in the body. A width
  // outside the pair is declined so the reference raises its own message rather
  // than the body answering a call the reference rejects.
  return condition_dtype == at::kBool || condition_dtype == at::kByte;
}

bool MpsSupportsCatDtype(at::ScalarType dtype, at::ScalarType other_dtype) {
  // Both arguments are the list's one datatype: the body checks the list is
  // uniform and this checks that width is carryable. `cat` promotes a mixed
  // list on the reference, which is the body's decline, so nothing here does the
  // promotion. The second parameter exists so this predicate reads like the
  // other two-operand gates; the body passes the same dtype twice.
  return dtype == other_dtype && MpsSupportsUnaryDtype(dtype);
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

bool MpsSupportsSigmoidDtype(at::ScalarType dtype) {
  // float32 alone, and the narrowness is the increment: see the note in
  // mps_common.h. float16 is declined because this SDK flushes a subnormal
  // float16 *result* to zero and the left tail of a sigmoid is subnormal at that
  // width, so the graph and the reference part by 3.1e-02; bfloat16 is declined
  // because the graph rounds differently from at::sigmoid at a width that holds
  // about three decimal digits. Both keep the round-trip, which stays correct.
  return dtype == at::kFloat;
}

bool MpsSupportsExpDtype(at::ScalarType dtype) {
  // The three widths measured bit-identical to at::exp in the operand's own
  // dtype, where a "difference" would mean a wrong answer: float16 over its whole
  // representable curve including every subnormal, and bfloat16 over the normal
  // range. float32 is served on the same reading `silu`'s float32 is -- see the
  // note in mps_common.h -- and float64 and the integral/bool widths keep the
  // round-trip, float64 because MPSGraph has no element type for it and the
  // others because the promotion rule lives in the reference.
  return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16;
}

bool MpsSupportsLogDtype(at::ScalarType dtype) {
  // The three widths measured to agree with at::log in the operand's own dtype:
  // float16 and bfloat16 bit-for-bit or to a last-ulp rounding, and float32 to one
  // ulp. See the note in mps_common.h. float64 has no MPSGraph element type and
  // the integral/bool widths are promoted by the reference, so both keep the
  // round-trip.
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

at::Tensor MpsSoftmaxOnHost(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float) {
  if (!half_to_float) {
    return at::_softmax(self.cpu(), dim, false).to(self.device());
  }
  // The flag means "compute in float, hand back float", and that is exactly
  // what the host can be asked for without the flag: widen first, then run the
  // flag-clear kernel. Forwarding `half_to_float=true` to the CPU instead asks
  // for a kernel that does not exist ("softmax with half to float conversion is
  // not supported on CPU"), which would turn every declined width into a raise.
  //
  // An integral or bool operand still raises here, which is the reference's
  // answer, and `at::_softmax` is the spelling that raises it: resolving to
  // `at::softmax` would reach the composite and never report the flag's
  // mismatch. A rank-0 operand has no `dim` in range, so its raise comes from
  // the dim check rather than from the width.
  return at::_softmax(self.cpu().to(at::kFloat), dim, false).to(self.device());
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

namespace {

// Whether this build/host can describe a buffer at a byte offset to MPSGraph.
//
// Two independent gates have to agree, and the second is the one that fires on
// the hosted macos-14 runner. The *SDK* gate is compile-time: the initializer is
// only declared from macOS 15, so an Xcode 15.4 SDK (14.5, what the runner
// ships) cannot name it at all. The *runtime* gate is separate: a process
// running on macOS 14 may or may not have loaded a framework that implements a
// macOS 15 selector. `__builtin_available` answers the runtime half, and the
// preprocessor answer below is constant per build, so this whole function folds
// away to a constant on either kind of toolchain.
bool OffsetTensorDataAvailable() {
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  return __builtin_available(macOS 15.0, *);
#else
  return false;
#endif
}

// A dedicated MTLBuffer holding one operand's bytes densely from offset 0, for
// the toolchains above that cannot describe an interior pointer.
//
// This is the difference between "declined" and "native" on macOS 14, and it is
// worth being explicit about why a copy is the right answer here rather than
// leaving the op on cpu_fallback. The fallback is not one copy: it is a
// device-to-host copy of the whole operand, a host kernel in torch's own
// threading, and a host-to-device copy of the result back -- and for the table
// reads this backend exists for (a 151936x1024 float32 embedding table is
// 622 MB) that is what turned a 0.7 ms node into the 69-149 ms the cost classes
// measured on the runner. One device-side copy into a staging buffer is a
// memcpy, and on a unified-memory Apple GPU it is a plain `memcpy` between two
// shared allocations rather than a PCIe round trip.
//
// The allocation is cached by size and reused across calls. These operands come
// in a handful of sizes (an attention weight, a token embedding table, a batch
// of activations) and a fresh MTLBuffer per call would put an allocation on the
// hot path of every op the runner otherwise serves natively. Bounded like the
// plan cache is, and for the same reason: a long run must not grow it without
// limit.
constexpr size_t kMaxStagingBuffers = 16;

struct StagingBuffer {
  id<MTLBuffer> buffer;  // +1
  size_t capacity;
};

std::unordered_map<size_t, StagingBuffer>& StagingPool() {
  // Leaked deliberately, like the plan cache: these own MTLBuffers whose
  // release at static-destruction time is not ours to order.
  static auto* pool = new std::unordered_map<size_t, StagingBuffer>();
  return *pool;
}

// The exact byte count the staging buffer has to hold, rounded up so an
// odd-sized operand does not land in a class of its own.
size_t StagingCapacity(size_t bytes) {
  constexpr size_t kGranularity = 1u << 20;  // 1 MiB
  return ((bytes + kGranularity - 1) / kGranularity) * kGranularity;
}

// The staging buffers one RunGraphImpl call has taken, released on the way out.
//
// A buffer the pool holds is left alone: it belongs to the pool and outlives the
// call. One the pool did not take was allocated because every pooled buffer of
// its class was already spoken for, so nothing else will ever hand it out and
// this is its one owner -- and since RunGraphImpl has half a dozen exits between
// taking a buffer and running the graph (a nil operand description, a body that
// declines, a queue that is not there), the release belongs in a destructor
// rather than at the end. Ownership is recorded when the buffer is taken rather
// than looked up at release time, which keeps the release from having to search
// the pool and lets the two live in either order.
//
// A private buffer is only ever created after a DrainAll, and the copy-back
// drain covers the results, so nothing read by a queued graph outlives the
// scope by more than the queue's own ordering allows.
struct ClaimedStaging {
  struct Entry {
    id<MTLBuffer> buffer;
    bool from_pool;
  };
  std::vector<Entry> buffers;

  id<MTLBuffer> Take(id<MTLBuffer> buffer, bool from_pool) {
    buffers.push_back(Entry{buffer, from_pool});
    return buffer;
  }

  bool Contains(id<MTLBuffer> buffer) const {
    for (const Entry& entry : buffers) {
      if (entry.buffer == buffer) {
        return true;
      }
    }
    return false;
  }

  bool AnyPrivate() const {
    for (const Entry& entry : buffers) {
      if (!entry.from_pool) {
        return true;
      }
    }
    return false;
  }

  ~ClaimedStaging() {
    for (const Entry& entry : buffers) {
      if (!entry.from_pool) {
        [entry.buffer release];
      }
    }
  }
};

// A buffer of at least `bytes`, owned by the pool and *claimed for the current
// call*. Borrowed, like MpsBufferOf's: it lives until the pool is torn down
// (never, in practice) and the caller must not release it. Returns nil when
// Metal cannot satisfy the request, which the caller turns into the same
// decline a nil operand always was.
//
// `claimed` is what keeps two operands of one call from landing on the same
// buffer, and it is load-bearing rather than tidy. The pools are keyed by
// rounded capacity, and a call's operands and results routinely round to the
// same class -- measured, `gather((4, 5), 0, (3, 1))` stages a 4-element input
// and a 3-element result, 16 and 24 bytes, both of which round to 1 MiB. Handing
// both the same buffer does not fail loudly: the result's write is ordered after
// the input's copy-in on the same queue, so the graph reads a zeroed operand and
// the op returns the right shape holding the wrong values (measured: `gather`
// answered `[0, 0, 0]` against `[0, 10, 15]`). A buffer already claimed by this
// call is therefore skipped, and a fresh one allocated outside the pool.
id<MTLBuffer> StagingBufferFor(size_t bytes, ClaimedStaging* claimed) {
  if (bytes == 0) {
    return nil;
  }
  const size_t capacity = StagingCapacity(bytes);
  auto& pool = StagingPool();
  const auto found = pool.find(capacity);
  if (found != pool.end()) {
    if (!claimed->Contains(found->second.buffer)) {
      return claimed->Take(found->second.buffer, /*from_pool=*/true);
    }
  }
  id<MTLDevice> device = torch_fl_mps::MetalDevice();
  id<MTLBuffer> buffer =
      [device newBufferWithLength:capacity
                          options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nil;
  }
  // Only a buffer the pool has never seen is worth keeping: the pool holds one
  // buffer per capacity class, so a second buffer of a class already present is
  // this call's own escape hatch, not a candidate for every later call.
  const bool pooled = found == pool.end() && pool.size() < kMaxStagingBuffers;
  if (pooled) {
    pool.emplace(capacity, StagingBuffer{buffer, capacity});
  }
  return claimed->Take(buffer, pooled);
}

// The buffer to describe `t` with, and the byte offset to describe it from.
//
// Offset 0 with the tensor's own MTLBuffer is the case the whole design is
// built around and the one this returns whenever the toolchain can express an
// offset. When it cannot -- an SDK older than macOS 15, or a host older than
// macOS 15 -- an interior operand cannot be described at all, and it is staged
// into a dedicated buffer instead of declined.
//
// The staging copy mirrors MpsGemmOperandOf's rule in one respect worth naming:
// what the graph reads is `t.numel()` elements densely from the offset, and
// this copies exactly those bytes, so a caller with a strided tensor has already
// materialised it (every body either calls `contiguous()` or has shown the
// layout is dense before it gets here).
//
// Returns false when the pointer is not inside a Metal allocation this runtime
// owns (a host tensor, a foreign or stale pointer) -- the same decline as
// before. That case is not a toolchain limitation and staging would not fix it.
bool ResolveOperand(
    const at::Tensor& t,
    id<MTLBuffer>* buffer,
    size_t* byteOffset,
    ClaimedStaging* claimed) {
  if (!MpsBufferOf(t, buffer, byteOffset)) {
    return false;
  }
  if (*byteOffset == 0 || OffsetTensorDataAvailable()) {
    return true;
  }
  // The copy is ordered behind any producer still writing `t`: Metal executes a
  // queue in order, but `t` may have been written by a kernel on a *different*
  // queue, so the host read is drained first. This is the same reasoning
  // MemoryManager::memcpy_ uses for its own copies.
  ::torch_fl_mps::DrainAll();
  const size_t bytes =
      static_cast<size_t>(t.numel()) *
      static_cast<size_t>(t.element_size());
  id<MTLBuffer> staging = StagingBufferFor(bytes, claimed);
  if (staging == nil) {
    return false;
  }
  // `t.data_ptr()` is the buffer's contents plus `byteOffset` -- that is what
  // MpsBufferOf established -- so the copy source is spelled the same way, and
  // the destination is the staging buffer's own base because it is described
  // from offset 0.
  std::memcpy(
      staging.contents,
      static_cast<const char*>((*buffer).contents) + *byteOffset,
      bytes);
  *buffer = staging;
  *byteOffset = 0;
  return true;
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
    // No MPSNDArray initializer here, so the offset-carrying description is
    // unavailable. That does not make the buffer undescribable: the *only* other
    // way MPSGraphTensorData accepts a buffer is
    // `initWithMTLBuffer:shape:dataType:`, which spans it from byte 0 -- the
    // same limitation, seen from the other side. It is also exactly the shape a
    // staged operand has, because ResolveOperand copies the operand to byte 0 of
    // a dedicated buffer and hands back offset 0.
    //
    // The offset is refused rather than ignored: a nonzero offset through this
    // initializer would silently read the wrong bytes (row 0 of the buffer
    // instead of the operand), which is the one failure mode worth more than the
    // fallback. An offset-0 operand reads correctly, and an interior one never
    // reaches here -- RunGraphImpl stages it first.
    if (byteOffset != 0) {
      return nil;
    }
    MPSGraphTensorData* direct =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                shape:FlatShape(count)
                                             dataType:dtype];
    if (direct == nil) {
      return nil;
    }
    return direct;
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
    // The input types the graph is specialized for, in the order `buffers`
    // collects them. Specialization is what turns the compiled graph into the
    // GPU program `runAsyncWithMTLCommandQueue:` executes, and it is the reason
    // this list is kept: it is handed to `specializeWithDevice:` below.
    NSMutableArray<MPSGraphShapedType*>* input_types =
        [NSMutableArray arrayWithCapacity:inputs.size()];
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
      MPSGraphShapedType* type = [[MPSGraphShapedType alloc]
          initWithShape:FlatShape(count)
               dataType:dtype];
      [feeds setObject:type forKey:buffer];
      [input_types addObject:type];
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
        // Specialize here rather than leaving it to the first run. The SDK
        // documents that without this call specialization "occurs at encode
        // time automatically", and on macOS 14 that encode-time path is what
        // dies: a SIGSEGV inside MPSGraph's own
        // convertMPSGraphShapesToMLIRTypes for a graph whose placeholders are
        // flat buffers, with every operand resolved and a compiled executable
        // already in hand -- i.e. past every point torch_fl can decline at.
        //
        // Calling it explicitly moves the same work to where a failure is
        // observable: `specializeWithDevice:` returns void, so the only signal
        // is whether the executable is usable afterwards. A graph that cannot
        // be specialized there is one this build cannot run, and the honest
        // answer is the CPU round-trip the caller already has, not a crash on
        // the encode that would have followed.
        [executable specializeWithDevice:device
                              inputTypes:input_types
                   compilationDescriptor:nil];
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
  // The graph path is only open on a device MPSGraph can encode for. This is
  // checked before anything is allocated or built, because the failure it
  // guards against is a SIGSEGV inside MPSGraph's own encode -- a fatal signal,
  // so a check placed after the executable exists would be a check placed after
  // the process is already committed to the crash. See MetalGraphUsable.
  //
  // Declining is safe by the same contract every other decline uses: false
  // sends the op to the CPU round-trip its body already implements, which is
  // the reference result -- slow on a device with no working graph compiler,
  // and correct.
  if (!torch_fl_mps::MetalGraphUsable()) {
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
  // The operands that had to be staged, and the buffer each one came from, so
  // the operand's own bytes are what the graph's inputs were copied *from* and
  // nothing has to be copied back for an input.
  //
  // `claimed` is the set of staging buffers this call has taken, threaded
  // through both loops below so the result of one operand cannot be handed the
  // buffer a sibling operand's input was staged into.
  ClaimedStaging claimed;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!ResolveOperand(inputs[i], &buffers[i], &offsets[i], &claimed)) {
      return false;
    }
    TORCH_CHECK(
        MpsDataTypeOf(inputs[i].scalar_type(), &dtypes[i]),
        "mps: no MPSGraph element type for ",
        inputs[i].scalar_type());
  }
  std::vector<id<MTLBuffer>> out_buffers(outs.size(), nil);
  std::vector<size_t> out_offsets(outs.size(), 0);
  // A result is the other direction: the graph writes the staging buffer, and
  // the bytes have to reach the caller's tensor afterwards. `nullptr` means the
  // result was described in place.
  std::vector<id<MTLBuffer>> out_originals(outs.size(), nil);
  std::vector<size_t> out_original_offsets(outs.size(), 0);
  for (size_t i = 0; i < outs.size(); ++i) {
    if (!MpsBufferOf(outs[i], &out_originals[i], &out_original_offsets[i])) {
      return false;
    }
    if (out_original_offsets[i] == 0 || OffsetTensorDataAvailable()) {
      out_buffers[i] = out_originals[i];
      out_offsets[i] = out_original_offsets[i];
      out_originals[i] = nil;
      continue;
    }
    const size_t bytes = static_cast<size_t>(outs[i].numel()) *
        static_cast<size_t>(outs[i].element_size());
    id<MTLBuffer> staging = StagingBufferFor(bytes, &claimed);
    if (staging == nil) {
      return false;
    }
    out_buffers[i] = staging;
    out_offsets[i] = 0;
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

    // The sentinel above makes the *stream* report the work, but that is not
    // enough on its own, because the caching allocator's block-reuse check
    // (find_free_block) compares stream pointers and never queries the stream.
    // A block returns to its pool the moment the tensor over it is dropped
    // (free_block, event_count == 0), and the next allocation of the same size
    // on the same stream picks it straight back up -- while this queued graph is
    // still reading the operand or writing the result that lived there. That is
    // a use-after-free with a benign-looking trigger: it does not need a free()
    // at all, just a dropped temporary, which is exactly what `torch.mm(a, b)`
    // with an out that is not held, or a reshaped operand, produces.
    //
    // record_stream closes that gap the way the CUDA path does: it attaches an
    // event on this stream to each block and bumps the block's event_count, so
    // `free_block` defers the return to the pool until the event completes
    // (process_events), and a block with a pending event cannot be merged away
    // (try_merge_blocks) or released (release_cached_blocks) either. The event is
    // the same stream the graph runs on, so it is ordered behind the run without
    // any extra synchronization of its own.
    //
    // Operands and results are both in flight: the graph reads every input and
    // writes every output, so both sets of blocks need the record. A tensor the
    // runtime does not own is a no-op in record_stream (the deleter check), which
    // is the right answer -- nothing in the block pool can alias it.
    if (c10::flagos::CachingDeviceAllocator::is_enabled()) {
      if (auto* alloc = c10::flagos::GetCachingAllocator()) {
        for (const at::Tensor& input : inputs) {
          alloc->record_stream(input.storage().data_ptr(), stream);
        }
        for (const at::Tensor& out : outs) {
          alloc->record_stream(out.storage().data_ptr(), stream);
        }
      }
    }
  }
  // The run is asynchronous, so anything the host has to do with a staging
  // buffer -- a result's copy-back, or a private buffer's release -- has to be
  // ordered behind it. That is one drain, covering both.
  //
  // A staged result is the one output the run does not finish by itself: the
  // graph wrote the staging buffer, and the caller is still holding a tensor
  // whose bytes are the old ones. A staged *input* needs no copy back, but its
  // buffer must not be released while the queued graph is still reading it,
  // which is what the drain below also buys for ClaimedStaging's destructor.
  //
  // This drain is also what makes the staging *pool* safe to reuse. A staging
  // buffer is handed to a later call, and the only way that later call can reach
  // a buffer a queued graph is still reading is by writing into it without
  // draining first: every writer here either drains (this block) or drains
  // before its copy *in* (ResolveOperand), and that drain covers every live
  // queue. So no copy into a staging buffer can overtake a queued reader of the
  // same buffer.
  bool any_staged = false;
  for (id<MTLBuffer> original : out_originals) {
    any_staged = any_staged || original != nil;
  }
  if (ran && (any_staged || claimed.AnyPrivate())) {
    ::torch_fl_mps::DrainAll();
  }
  if (ran && any_staged) {
    for (size_t i = 0; i < outs.size(); ++i) {
      if (out_originals[i] == nil) {
        continue;
      }
      std::memcpy(
          static_cast<char*>(out_originals[i].contents) + out_original_offsets[i],
          out_buffers[i].contents,
          static_cast<size_t>(outs[i].numel()) *
              static_cast<size_t>(outs[i].element_size()));
    }
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
