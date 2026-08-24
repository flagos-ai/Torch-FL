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

// topsaten has no int64 or float64 kernels: every op returns NOT_SUPPORT for an
// I64 operand (verified across add/mul/eq/abs/sum), and an F64 operand fails the
// same way (measured on S60 across add/abs/reciprocal, vendor log "datatype not
// support yet"). Callers check this and run the op on CPU instead, which keeps
// int64 tensors (indices, masks, counters) and float64 tensors working instead
// of raising. GradScaler depends on the float64 path: it computes the inverse
// scale as `scale.double().reciprocal().float()`.
inline bool TopsatenSupportsDtype(at::ScalarType type) {
  return type != at::kLong && type != at::kDouble;
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
#define EXEC_TOPSATEN_CMD(op, guard_tensor, ...)                              \
  do {                                                                        \
    at::native::flagos::gcu::EnsureTopsatenInit();                            \
    at::native::flagos::gcu::TopsDeviceGuard _tops_guard((guard_tensor));     \
    topsStream_t _tops_stream =                                               \
        at::native::flagos::gcu::GetCurrentTopsStream();                      \
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
