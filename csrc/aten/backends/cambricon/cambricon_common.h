// Copyright (c) 2026, BAAI. All rights reserved.
//
// Marshalling helpers for the Cambricon MLU (CNNL) operator backend.
//
// CNNL is a descriptor-based API: a kernel builds a `cnnlTensorDescriptor_t`
// for each operand, an operation descriptor for the entry point, and passes
// raw device pointers. There is no workspace-query phase for the elementwise
// ops used here -- `cnnlOpTensor` accepts a null workspace -- so a kernel is:
// open a device guard, describe the operands, and enqueue on the runtime stream.
//
// One handle per device per host thread. `cnnlCreate` binds the handle to whichever device is
// current, so a handle is not portable across devices and must not be shared
// between them; `HandleForDevice` therefore keys its cache by device ordinal.
// A process (or a test) may switch devices between calls.
//
// The handle is created lazily and never destroyed. A cnnl handle owns the
// library's allocator state; tearing it down at unload while tensors allocated
// through it are still reachable is worse than letting process exit reclaim it,
// which is what every other vendor backend here does with its library context.

#pragma once

#ifdef USE_CAMBRICON

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include "runtime/guard.h"

#include <cnnl.h>
#include <cnrt.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::native::flagos::cambricon {

// cnrt status as an exception, with the call that failed. cnnlStatus_t and
// cnrtRet_t are distinct enums with overlapping success values (0), so they get
// separate wrappers rather than one template that would accept either.
inline void CheckCnrt(cnrtRet_t status, const char* what) {
  TORCH_CHECK(
      status == cnrtSuccess,
      "flagos cambricon: ",
      what,
      " failed with cnrt status ",
      static_cast<int>(status));
}

inline void CheckCnnl(cnnlStatus_t status, const char* what) {
  TORCH_CHECK(
      status == CNNL_STATUS_SUCCESS,
      "flagos cambricon: ",
      what,
      " failed with cnnl status ",
      static_cast<int>(status));
}

// Selects the device an op's tensors live on, and restores the previous one.
//
// A cnrt pointer resolves only against the *current* device, so an op on
// flagos:1 has to run with device 1 selected. Same shape as kunlunxin's guard
// and seq for the same reason; the difference is that this one is also what
// makes `cnnlCreate` bind to the right device in `HandleForDevice` below.
class CambriconDeviceGuard {
 public:
  explicit CambriconDeviceGuard(const at::Tensor& tensor) {
    const int target = static_cast<int>(tensor.device().index());
    int current = -1;
    if (cnrtGetDevice(&current) != cnrtSuccess || current == target) {
      return;
    }
    if (cnrtSetDevice(target) != cnrtSuccess) {
      return;
    }
    previous_ = current;
  }

  ~CambriconDeviceGuard() {
    if (previous_ >= 0) {
      cnrtSetDevice(previous_);
    }
  }

  CambriconDeviceGuard(const CambriconDeviceGuard&) = delete;
  CambriconDeviceGuard& operator=(const CambriconDeviceGuard&) = delete;

 private:
  int previous_ = -1;
};

// The CNNL handle for the current device, created on first use.
//
// `cnnlCreate` binds to the current device, so the map is keyed by ordinal. The
// device guard is the caller's (`at::cuda`-style guard is not used here because
// this backend has its own `DeviceGuard` in runtime/accelerator/cambricon/).
inline cnnlHandle_t HandleForDevice(int device_index) {
  // CNNL handles carry mutable queue state. Do not share a handle between
  // threads that may select different streams on the same device.
  static thread_local std::unordered_map<int, cnnlHandle_t> handles;
  auto it = handles.find(device_index);
  if (it != handles.end()) {
    CheckCnnl(
        cnnlSetQueue(it->second, reinterpret_cast<cnrtQueue_t>(
            GetCurrentStreamForDevice(device_index))),
        "cnnlSetQueue");
    return it->second;
  }

  // cnnlCreate binds to whatever is current, so move the runtime to the device
  // this handle is for and leave it there -- flagos already keeps the current
  // device pinned to the op's device for the duration of a kernel.
  int previous = 0;
  if (cnrtGetDevice(&previous) == cnrtSuccess && previous != device_index) {
    CheckCnrt(cnrtSetDevice(device_index), "cnrtSetDevice");
  }

  cnnlHandle_t handle = nullptr;
  CheckCnnl(cnnlCreate(&handle), "cnnlCreate");
  // Use the same queue as runtime copies and FlagGems kernels.
  CheckCnnl(
      cnnlSetQueue(handle, reinterpret_cast<cnrtQueue_t>(
          GetCurrentStreamForDevice(device_index))),
      "cnnlSetQueue");
  handles.emplace(device_index, handle);
  return handle;
}

// cnnlDataType_t for an aten dtype. Only the types a kernel declares support
// for are mapped; everything else is a programming error in that kernel's
// dtype gate, not a fallback the marshalling layer should invent.
inline cnnlDataType_t ToCnnlDtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return CNNL_DTYPE_FLOAT;
    case at::kHalf:
      return CNNL_DTYPE_HALF;
    case at::kBFloat16:
      return CNNL_DTYPE_BFLOAT16;
    case at::kDouble:
      return CNNL_DTYPE_DOUBLE;
    case at::kInt:
      return CNNL_DTYPE_INT32;
    case at::kLong:
      return CNNL_DTYPE_INT64;
    case at::kShort:
      return CNNL_DTYPE_INT16;
    case at::kChar:
      return CNNL_DTYPE_INT8;
    case at::kBool:
      return CNNL_DTYPE_BOOL;
    default:
      TORCH_CHECK(
          false,
          "flagos cambricon: no cnnl data type for ",
          at::toString(dtype));
  }
}

// RAII for the two descriptor kinds a kernel builds. Destructors are nothrow in
// practice (the destroy calls return a status), and a destructor must not
// throw during unwinding, so those returns are ignored deliberately.
class TensorDesc {
 public:
  TensorDesc() { CheckCnnl(cnnlCreateTensorDescriptor(&desc_), "cnnlCreateTensorDescriptor"); }
  ~TensorDesc() { cnnlDestroyTensorDescriptor(desc_); }

  TensorDesc(const TensorDesc&) = delete;
  TensorDesc& operator=(const TensorDesc&) = delete;

  // Describes `tensor` as CNNL sees it: same dims, same dtype, ARRAY layout.
  //
  // Requires a contiguous tensor. cnnlSetTensorDescriptorEx takes explicit
  // strides, and the elementwise ops used here accept them, but a strided
  // operand has not been measured against the alternative (materialise a
  // contiguous copy first), so kernels gate on contiguity and let the caller
  // decide rather than silently paying for an unmeasured layout.
  void Set(const at::Tensor& tensor) {
    const auto sizes = tensor.sizes();
    std::vector<int> dims(sizes.begin(), sizes.end());
    TORCH_CHECK(
        !dims.empty(),
        "flagos cambricon: a rank-0 tensor reached a CNNL descriptor (",
        tensor.scalar_type(), ")");
    CheckCnnl(
        cnnlSetTensorDescriptor(
            desc_,
            CNNL_LAYOUT_ARRAY,
            ToCnnlDtype(tensor.scalar_type()),
            static_cast<int>(dims.size()),
            dims.empty() ? nullptr : dims.data()),
        "cnnlSetTensorDescriptor");
  }

  // Describes `tensor` with its own strides.
  //
  // CNNL requires the lowest-dimension stride to be 1, so this only serves
  // views that already satisfy it. A transposed matrix does not -- w.t() of a
  // (K, N) weight carries strides (1, K) -- and is handled by describing the
  // underlying storage and setting the matmul transpose flag instead, which
  // keeps the weight from being copied (copying it made the multiply 242 times
  // slower than the same product on a contiguous operand).
  void SetStrided(const at::Tensor& tensor) {
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    std::vector<int> dims(sizes.begin(), sizes.end());
    std::vector<int> steps(strides.begin(), strides.end());
    CheckCnnl(
        cnnlSetTensorDescriptorEx(
            desc_,
            CNNL_LAYOUT_ARRAY,
            ToCnnlDtype(tensor.scalar_type()),
            static_cast<int>(dims.size()),
            dims.empty() ? nullptr : dims.data(),
            dims.empty() ? nullptr : steps.data()),
        "cnnlSetTensorDescriptorEx");
  }

  cnnlTensorDescriptor_t get() const { return desc_; }

 private:
  cnnlTensorDescriptor_t desc_ = nullptr;
};

class OpTensorDesc {
 public:
  OpTensorDesc(cnnlOpTensorDesc_t op, at::ScalarType compute) {
    CheckCnnl(cnnlCreateOpTensorDescriptor(&desc_), "cnnlCreateOpTensorDescriptor");
    CheckCnnl(
        cnnlSetOpTensorDescriptor(
            desc_, op, ToCnnlDtype(compute), CNNL_NOT_PROPAGATE_NAN),
        "cnnlSetOpTensorDescriptor");
  }
  ~OpTensorDesc() { cnnlDestroyOpTensorDescriptor(desc_); }

  OpTensorDesc(const OpTensorDesc&) = delete;
  OpTensorDesc& operator=(const OpTensorDesc&) = delete;

  cnnlOpTensorDescriptor_t get() const { return desc_; }

 private:
  cnnlOpTensorDescriptor_t desc_ = nullptr;
};

} // namespace at::native::flagos::cambricon

#endif // USE_CAMBRICON