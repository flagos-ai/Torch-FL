// Copyright (c) 2026, BAAI. All rights reserved.
//
// Marshalling helpers for the Enflame GCU (topsaten) operator backend.
//
// Compared to Ascend's aclnn, the topsaten call shape is a single direct call:
// there is no workspace query / executor phase. A kernel therefore only needs
// to wrap its aten tensors into `topsatenTensor`, pick the stream, call the op
// and check the status.

#pragma once

#ifdef USE_GCU

#include "runtime/accelerator/gcu/tops_stream.h"

#include <ATen/ATen.h>
#include <ATen/ops/full.h>
#include <c10/core/Scalar.h>

#include <topsaten/topsaten.h>

#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace at::native::flagos::gcu {

inline const char* TopsatenStatusName(topsatenStatus_t status) {
  switch (status) {
    case TOPSATEN_STATUS_SUCCESS:        return "SUCCESS";
    case TOPSATEN_STATUS_ALLOC_FAILED:   return "ALLOC_FAILED";
    case TOPSATEN_STATUS_BAD_PARAM:      return "BAD_PARAM";
    case TOPSATEN_STATUS_NOT_SUPPORT:    return "NOT_SUPPORT";
    case TOPSATEN_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
    case TOPSATEN_STATUS_RUNTIME_ERROR:  return "RUNTIME_ERROR";
    case TOPSATEN_STATUS_EXECUTE_ERROR:  return "EXECUTE_ERROR";
    default:                             return "UNKNOWN";
  }
}

// topsaten requires a one-time global init before any op call.
inline void EnsureTopsatenInit() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    // Some ops (e.g. the tensor-with-scalar overloads) allocate a temporary
    // device buffer internally; point those allocations at the tops runtime.
    // Must be registered before init so the internal pools pick them up.
    topsatenMallocFuncRegister(
        [](void** p, size_t n) { return topsMalloc(p, n); });
    topsatenFreeFuncRegister([](void* p) { return topsFree(p); });
    topsatenMallocAsyncFuncRegister(
        [](void** p, size_t n, topsStream_t, uint64_t) {
          return topsMalloc(p, n);
        });
    topsatenFreeAsyncFuncRegister(
        [](void* p, topsStream_t) { return topsFree(p); });
    topsatenStatus_t status = topsatenInit();
    TORCH_CHECK(
        status == TOPSATEN_STATUS_SUCCESS,
        "topsatenInit failed: ", TopsatenStatusName(status));
  });
}

inline topsatenDataType_t ToTopsatenDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:    return TOPSATEN_DATA_FP32;
    case at::kDouble:   return TOPSATEN_DATA_F64;
    case at::kHalf:     return TOPSATEN_DATA_FP16;
    case at::kBFloat16: return TOPSATEN_DATA_BF16;
    case at::kLong:     return TOPSATEN_DATA_I64;
    case at::kInt:      return TOPSATEN_DATA_I32;
    case at::kShort:    return TOPSATEN_DATA_I16;
    case at::kChar:     return TOPSATEN_DATA_I8;
    case at::kByte:     return TOPSATEN_DATA_U8;
    case at::kBool:     return TOPSATEN_DATA_PRED;
    default:
      TORCH_CHECK(false, "Unsupported dtype for topsaten: ", type);
  }
}

// topsaten has no int64, float64 or complex kernels: every op returns NOT_SUPPORT
// for an I64 operand (verified across add/mul/eq/abs/sum), and an F64 operand
// fails the same way (measured on S60 across add/abs/reciprocal, vendor log
// "datatype not support yet"). Complex is the same story one layer earlier --
// `ToTopsatenDataType` has no mapping for it at all, so an op on a complex
// operand raises before it reaches the vendor. The first caller is the complex
// rotary embedding of Qwen-Image: the freqs live on the device, `cat` on them
// raised `Unsupported dtype for topsaten: ComplexFloat`, and the transformer
// step could not start.
//
// Callers check this and run the op on CPU instead, which keeps int64 tensors
// (indices, masks, counters), float64 tensors and complex tensors working
// instead of raising. GradScaler depends on the float64 path: it computes the
// inverse scale as `scale.double().reciprocal().float()`.
inline bool TopsatenSupportsDtype(at::ScalarType type) {
  return type != at::kLong && type != at::kDouble && !c10::isComplexType(type);
}

// Which dtypes topsatenArange may be handed.
//
// Narrower than TopsatenSupportsDtype on purpose. `arange` is the one op whose
// vendor entry point takes no size: it fills an output the caller has already
// sized, so a dtype the kernel declines leaves the buffer unwritten instead of
// raising, and the whole point of the op is the values in that buffer. The set
// is measured on S60 rather than inferred from ToTopsatenDataType, which only
// says the dtype is *representable*.
//
// Bool is deliberately absent even though the vendor takes PRED: ATen has no
// arange kernel for it at all (`arange_cpu not implemented for 'Bool'`, and a
// GCU caller gets the same on a hosted tensor), so declining here is what keeps
// the host path's NotImplementedError instead of inventing a result CPU and
// CUDA both refuse to produce.
inline bool TopsatenArangeDtype(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
    case at::kInt:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kHalf:
    case at::kBFloat16:
      return true;
    default:
      return false;
  }
}

// Which dtypes topsatenIndexSelect accepts an *index* operand as.
//
// An index tensor is the one place an integer operand has to be braced for
// specially rather than declined: the coordinate data is consumed as integers,
// not as element data, so the "no int64 kernels" rule does not apply to it. Both
// dtypes are measured, not inferred -- index_select at dim 0/1/2 and with a
// 0-dim, empty or non-contiguous index returns the host result for an i32 index
// and for an i64 one. index_fill is the opposite case: the vendor rejects an i64
// index and only ever takes an i32 one, so it narrows its index through
// `TopsatenIndexFillIndex` below instead of using this.
inline bool TopsatenIndexDtype(at::ScalarType type) {
  return type == at::kInt || type == at::kLong;
}

// The two operand limits topsatenIndexSelect documents that its dtype table
// does not express: every dim of `in` and `out` must hold fewer than 2^24
// elements, and every operand buffer must stay under 4.0 GB (topsaten_ops.h).
// Neither is a performance threshold -- past either the kernel returns an
// error -- so a call that trips one is served by the host path rather than
// raising where ATen's own kernel would have produced a result.
inline bool TopsatenIndexSelectFits(at::TensorList operands) {
  constexpr int64_t kMaxDim = int64_t(1) << 24;
  constexpr int64_t kMaxBufferBytes = int64_t(4) << 30;
  for (const at::Tensor& t : operands) {
    if (t.numel() * t.element_size() >= kMaxBufferBytes) {
      return false;
    }
    for (const int64_t size : t.sizes()) {
      if (size >= kMaxDim) {
        return false;
      }
    }
  }
  return true;
}

// The i32 index tensor topsatenIndexFill requires, built from ATen's own index.
//
// index_fill is the one index op whose operand types the two sides disagree on
// outright: ATen takes an int64 index and nothing else (`index_fill_(): Expected
// dtype int64 for index.`), and the vendor takes an int32 one and nothing else
// (BAD_PARAM for an i64 index at every dim, rank, self dtype and scalar tag
// measured on S60). The vendor's own bounds check is also not safe to lean on:
// an index equal to the dim size satisfies it and writes one element past the
// end of the tensor, and a larger index aborts the process outright
// (`Index out-of-bounds! bound=2, index=5`). ATen, meanwhile, accepts a negative
// index in this op and wraps it (measured: -1 in dim 0 of a (2,3,4) fills row 1,
// while -3 raises). So the index is read back once and wrapped and narrowed in
// the same pass. A value still out of range afterwards -- or one that cannot be
// represented as int32, which a truncating cast would silently fold back into
// range -- returns an undefined tensor, and the caller sends that call to the
// host path, where ATen raises its own message for it.
//
// The transfer is O(num_indices) and `self` stays on the device, which is the
// whole point: the host path this replaces copies the *tensor being filled* to
// the host and back.
inline at::Tensor TopsatenIndexFillIndex(const at::Tensor& index, int64_t size) {
  auto host = index.to(at::kCPU).contiguous().reshape({-1});
  auto narrowed = at::empty(host.sizes(), host.options().dtype(at::kInt));
  const int64_t* src = host.const_data_ptr<int64_t>();
  int32_t* dst = narrowed.data_ptr<int32_t>();
  const int64_t numel = host.numel();
  for (int64_t i = 0; i < numel; ++i) {
    int64_t value = src[i];
    if (value < 0) {
      value += size;
    }
    if (value < 0 || value >= size) {
      return at::Tensor();
    }
    dst[i] = static_cast<int32_t>(value);
  }
  return narrowed.to(index.device());
}

// The i32 index tensor topsatenEmbeddingDenseBackward requires, built from
// ATen's own index.
//
// The vendor's parameter table says int64 and its runtime disagrees: every int64
// index set comes back BAD_PARAM with `topsatenEmbeddingBackward error: index
// tensor must be of int32 data type`, and the same call with an int32 index
// succeeds and matches ATen's own kernel row for row -- measured on S60 through
// the vendor symbol itself, with padding_idx set, with scale_grad_by_freq set,
// and for a 1-D index as well as the [n, 1] the table asks for. ATen meanwhile
// hands this op an int64 index and nothing else, so the narrowing is the op's
// job.
//
// It cannot be a plain cast. ATen does not raise on an index outside
// [0, num_weights): it *ignores* that element of grad (measured -- 9, -2 and
// 2**32 + 1, each with num_weights 6, all leave the table holding only the
// in-range contributions). A truncating cast folds 2**32 + 1 back to 1, and the
// vendor then accumulates grad's row for it into row 1 where ATen accumulated
// nothing -- measured, and a silently wrong gradient rather than an ignored
// element. So the index is read back once and checked in the same pass: an entry
// outside [0, num_weights) returns an undefined tensor and the caller sends that
// call to the host path, where ATen's own kernel ignores it. A num_weights that
// does not fit in int32 takes the host path for the same reason -- a table that
// large cannot be addressed by the vendor at all.
//
// The transfer is O(num_indices) and `grad` -- num_indices * embed_dim
// elements, larger by the embedding width -- stays on the device, which is the
// whole point: the host path this replaces copies grad to the host and back.
inline at::Tensor TopsatenEmbeddingIndex(const at::Tensor& indices,
                                         int64_t num_weights,
                                         const at::Device& device) {
  if (num_weights <= 0 || num_weights > std::numeric_limits<int32_t>::max()) {
    return at::Tensor();
  }
  // int32 indices are legal for ATen's own kernel, so they are widened rather
  // than declined; both dtypes then take the same validation below.
  auto host = indices.to(at::kCPU).to(at::kLong).contiguous().reshape({-1});
  auto narrowed = at::empty(host.sizes(), host.options().dtype(at::kInt));
  const int64_t* src = host.const_data_ptr<int64_t>();
  int32_t* dst = narrowed.data_ptr<int32_t>();
  const int64_t numel = host.numel();
  for (int64_t i = 0; i < numel; ++i) {
    const int64_t value = src[i];
    if (value < 0 || value >= num_weights) {
      return at::Tensor();
    }
    dst[i] = static_cast<int32_t>(value);
  }
  return narrowed.to(device);
}

// Which dtypes topsatenEmbeddingDenseBackward may be handed. Its parameter
// table lists result/grad as float, fp16, bf16, i32, u32, i16, u16, i8, u8 --
// everything TopsatenSupportsDtype allows except PRED, which the vendor omits
// even though its arange and index kernels do take a bool. A bool embedding
// weight is legal in ATen, so declining here is what keeps the host path's
// result instead of a vendor status error.
inline bool TopsatenEmbeddingDenseBackwardDtype(at::ScalarType type) {
  return TopsatenSupportsDtype(type) && type != at::kBool;
}

// A tops device pointer resolves only against the *current* device, so an op
// on flagos:1 must run with device 1 selected. Restores the previous device.
class TopsDeviceGuard {
 public:
  explicit TopsDeviceGuard(const at::Tensor& tensor) {
    if (!tensor.defined() || !tensor.device().is_privateuseone()) {
      return;
    }
    const int target = static_cast<int>(tensor.device().index());
    if (target < 0) {
      return;
    }
    int current = -1;
    if (topsGetDevice(&current) != topsSuccess || current == target) {
      return;
    }
    if (topsSetDevice(target) != topsSuccess) {
      return;
    }
    prev_device_ = current;
  }

  ~TopsDeviceGuard() {
    if (prev_device_ >= 0) {
      topsSetDevice(prev_device_);
    }
  }

  TopsDeviceGuard(const TopsDeviceGuard&) = delete;
  TopsDeviceGuard& operator=(const TopsDeviceGuard&) = delete;

 private:
  int prev_device_ = -1;
};

// `topsatenSize_t` only holds a `const int64_t*`, so the shape/stride arrays
// must outlive the topsatenTensor. Copy them into the wrapper.
class TopsatenTensorWrapper {
 public:
  explicit TopsatenTensorWrapper(const at::Tensor& tensor)
      : sizes_(tensor.sizes().vec()), strides_(tensor.strides().vec()) {
    TORCH_CHECK(tensor.defined(), "topsaten: undefined tensor");
    // topsaten rejects a rank-0 shape ("dims/strides length is invalid"), so a
    // scalar tensor (e.g. the result of a full reduction) is described as the
    // equivalent 1-element vector.
    if (sizes_.empty()) {
      sizes_.assign(1, 1);
      strides_.assign(1, 1);
    }
    // data_ptr() already accounts for storage_offset, so no SetOffset here.
    tops_tensor_ = topsatenTensor(
        topsatenSize_t(sizes_.data(), static_cast<int64_t>(sizes_.size())),
        topsatenSize_t(strides_.data(), static_cast<int64_t>(strides_.size())),
        ToTopsatenDataType(tensor.scalar_type()),
        const_cast<void*>(tensor.const_data_ptr()));
  }

  // Non-const: topsaten output parameters are `topsatenTensor&`.
  topsatenTensor& get() {
    return tops_tensor_;
  }

  TopsatenTensorWrapper(const TopsatenTensorWrapper&) = delete;
  TopsatenTensorWrapper& operator=(const TopsatenTensorWrapper&) = delete;

 private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  topsatenTensor tops_tensor_;
};

// Holds the int64 array backing a `topsatenSize_t` argument (dims lists etc).
class TopsatenSizeWrapper {
 public:
  explicit TopsatenSizeWrapper(at::IntArrayRef values)
      : values_(values.vec()) {}

  explicit TopsatenSizeWrapper(std::vector<int64_t> values)
      : values_(std::move(values)) {}

  topsatenSize_t get() const {
    return topsatenSize_t(values_.data(), static_cast<int64_t>(values_.size()));
  }

  TopsatenSizeWrapper(const TopsatenSizeWrapper&) = delete;
  TopsatenSizeWrapper& operator=(const TopsatenSizeWrapper&) = delete;

 private:
  std::vector<int64_t> values_;
};

// Materializes a Scalar as a device tensor of `sizes`.
//
// topsaten's add/sub/mul/div tensor-with-scalar overloads stage the scalar into
// a host buffer that the driver refuses inside this process ("Cannot create
// memory object for kernel parameter 2"), so those ops go through the
// tensor-with-tensor overload instead. topsaten does not broadcast, hence the
// full-size tensor. Built on the host and copied once.
inline at::Tensor ScalarToDeviceTensor(
    const at::Scalar& scalar,
    at::IntArrayRef sizes,
    const at::TensorOptions& options) {
  return at::full(sizes, scalar, options.device(at::kCPU)).to(options.device());
}

// `topsatenScalar_t` is a plain {dtype, union{double fval; int64_t ival;}}, so
// the union member has to match the dtype the op will read it back as.
inline topsatenScalar_t ToTopsatenScalar(
    const at::Scalar& scalar,
    at::ScalarType type) {
  topsatenScalar_t out{};
  out.dtype = ToTopsatenDataType(type);
  if (at::isFloatingType(type)) {
    out.fval = scalar.to<double>();
  } else if (type == at::kBool) {
    out.ival = scalar.to<bool>() ? 1 : 0;
  } else {
    out.ival = scalar.to<int64_t>();
  }
  return out;
}

// Marshals an at::TensorList into the std::vector<topsatenTensor> the foreach
// ops take, owning one wrapper per tensor so every sizes/strides array stays
// alive for the duration of the call.
//
// The foreach kernels write through the strides they are given and accept an
// output that aliases an input (verified on hardware), so an in-place foreach
// can hand the same list as both operand and destination -- but only when each
// tensor is contiguous, since a non-contiguous destination cannot be written
// back through a temporary. Callers gate on IsForeachEligible() for that.
class TopsatenTensorList {
 public:
  explicit TopsatenTensorList(at::TensorList tensors) {
    wrappers_.reserve(tensors.size());
    tops_.reserve(tensors.size());
    for (const at::Tensor& t : tensors) {
      wrappers_.push_back(std::make_unique<TopsatenTensorWrapper>(t));
      tops_.push_back(wrappers_.back()->get());
    }
  }

  std::vector<topsatenTensor>& get() {
    return tops_;
  }

  TopsatenTensorList(const TopsatenTensorList&) = delete;
  TopsatenTensorList& operator=(const TopsatenTensorList&) = delete;

 private:
  std::vector<std::unique_ptr<TopsatenTensorWrapper>> wrappers_;
  std::vector<topsatenTensor> tops_;
};

// A foreach list goes to topsaten only if every tensor is a contiguous,
// zero-offset, topsaten-representable tensor on the same device. Anything else
// (an int64 list, a sliced parameter view, a mixed-device list) takes the
// generic path, which is still correct, just slower.
inline bool IsForeachEligible(at::TensorList tensors) {
  if (tensors.empty()) {
    return false;
  }
  auto device = tensors[0].device();
  if (device.type() != c10::kPrivateUse1) {
    return false;
  }
  for (const at::Tensor& t : tensors) {
    if (!t.defined() || !TopsatenSupportsDtype(t.scalar_type()) ||
        !t.is_contiguous() || t.storage_offset() != 0 || t.numel() == 0 ||
        t.device() != device) {
      return false;
    }
  }
  return true;
}

// Bounds used to fill in an absent clamp limit: clamping against the dtype's
// own extreme leaves that side untouched. Floating types use -inf/+inf so that
// NaN handling matches an unbounded clamp.
inline at::Scalar DtypeLowest(at::ScalarType type) {
  if (at::isFloatingType(type)) {
    return at::Scalar(-std::numeric_limits<double>::infinity());
  }
  if (type == at::kBool) {
    return at::Scalar(false);
  }
  return at::Scalar(std::numeric_limits<int64_t>::lowest());
}

inline at::Scalar DtypeHighest(at::ScalarType type) {
  if (at::isFloatingType(type)) {
    return at::Scalar(std::numeric_limits<double>::infinity());
  }
  if (type == at::kBool) {
    return at::Scalar(true);
  }
  return at::Scalar(std::numeric_limits<int64_t>::max());
}

} // namespace at::native::flagos::gcu

// Issues a topsaten op on the shared stream of `guard_tensor`'s device and
// waits for it, mirroring EXEC_ASCEND_CMD's synchronous contract.
//
// "Synchronous" has to mean both directions. The trailing synchronise waits for
// the op just issued; it does not wait for work another producer already queued,
// and on this backend there is another producer. `mean.dim`, `pow.Scalar` and
// the other reduction/pointwise ops that carry `= flaggems` in
// configs/backends_gcu.conf run through triton, and topsaten submits to the same
// tops stream triton launches on -- but an op that is merely *submitted* behind
// a queued kernel does not read that kernel's output on this hardware. Measured
// on an S60 at `transformer_blocks.0.attn.norm_q` -- the RMSNorm every attention
// block of the Qwen-Image transformer runs -- with the producer
// `variance = hidden_states.to(float32).pow(2).mean(-1, keepdim=True)`
// (FlagGems) and the consumer `rsqrt(variance + eps)` (topsaten), compared
// against a float64 reference on the real 1024x1024 activation: 40 of 40 rounds
// wrong at 2.371e-02 with no barrier, 0 of 40 wrong at 6.941e-09 when the stream
// was drained before the consumer was issued. A stale `variance` is not a
// small numerical difference: `variance + eps` can come out negative, `rsqrt`
// then returns NaN, and the instrumented run recorded this module's output NaN
// in exactly 131072 places -- 1024 rows of 128 lanes -- with finite values as
// large as 5.66e7 beside them. That is one poisoned row per attention head, in
// every block, on every step, and the image the pipeline decoded after 50 such
// steps was black (mean=0.0000 std=0.0000, against mean=0.3334 std=0.2581 for
// the vendor leg from the same initial latents).
//
// The pre-op drain is close to free in the steady state, because the post-op
// drain below has already emptied the stream: only work issued since the last
// topsaten op can be in flight, which is exactly the FlagGems work that needs
// waiting for.
#define EXEC_TOPSATEN_CMD(op, guard_tensor, ...)                              \
  do {                                                                        \
    at::native::flagos::gcu::EnsureTopsatenInit();                            \
    at::native::flagos::gcu::TopsDeviceGuard _tops_guard((guard_tensor));     \
    topsStream_t _tops_stream =                                               \
        at::native::flagos::gcu::GetCurrentTopsStream();                      \
    topsError_t _tops_pre = topsStreamSynchronize(_tops_stream);              \
    TORCH_CHECK(                                                              \
        _tops_pre == topsSuccess,                                             \
        #op, " pre-stream sync failed: ", topsGetErrorString(_tops_pre));     \
    topsatenStatus_t _tops_status = topsaten::op(__VA_ARGS__, _tops_stream);   \
    TORCH_CHECK(                                                              \
        _tops_status == TOPSATEN_STATUS_SUCCESS,                              \
        #op, " failed: ",                                                     \
        at::native::flagos::gcu::TopsatenStatusName(_tops_status));           \
    topsError_t _tops_sync = topsStreamSynchronize(_tops_stream);             \
    TORCH_CHECK(                                                              \
        _tops_sync == topsSuccess,                                            \
        #op, " stream sync failed: ", topsGetErrorString(_tops_sync));        \
  } while (0)

#endif // USE_GCU
