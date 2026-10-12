// Copyright (c) 2026, BAAI. All rights reserved.
//
// Marshalling helpers for the Kunlunxin XPU (xdnn_pytorch) operator backend.
//
// The call shape is one direct library call, like topsaten on GCU and unlike
// Ascend's aclnn: there is no workspace-query / executor-preparation phase. A
// kernel opens a device guard, wraps its aten tensors into
// `xdnn_pytorch::Tensor`, calls the vendor entry point and drains the stream.
//
// Two things about this library are worth knowing before reading on:
//
//   * Its entry points take an `xpukernel::Context*` first and return an `int`
//     status, not a Tensor. The output is an out-parameter, so the caller
//     allocates it and describes its dtype -- which is how `cast` works at all,
//     since it takes no dtype argument (`cast` reads the target type off the
//     output descriptor).
//
//   * `xdnn_pytorch` and `xpytorch::xpu::api` are both top-level namespaces, and
//     this file lives in `at::native::flagos::kunlunxin`, so every name from
//     either library has to be qualified: an unqualified `Tensor` inside this
//     namespace is `at::Tensor`, and an unqualified `ScalarType` is
//     `at::ScalarType`. Getting that wrong compiles to the wrong type rather
//     than failing, so the prefixes below are load-bearing.

#pragma once

#ifdef USE_KUNLUNXIN

#include "runtime/accelerator/kunlunxin/xpu_stream.h"

#include <ATen/ATen.h>
#include <c10/core/Scalar.h>

#include <xpu/defs.h>
#include <xpu/refactor/context/xpukernel_newcontext.h>
#include <xpu/runtime.h>
#include <xdnn_pytorch/xdnn_pytorch.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::native::flagos::kunlunxin {

// The vendor context, created once per device. `create_context()` binds to
// whatever device is current when it is called, so the cache is keyed by device
// and creation switches the current device around the call. A context is never
// destroyed: `destroy_context()` would have to run from a static destructor,
// whose ordering against libxpurt's own teardown is unknown, and a process that
// exits without it reclaims the same memory one instruction later. GCU's
// `EnsureTopsatenInit` makes the same trade for the same reason.
inline int XdnnDeviceIndex(const at::Tensor& tensor) {
  if (!tensor.defined() || !tensor.device().is_privateuseone()) {
    return 0;
  }
  const int device = static_cast<int>(tensor.device().index());
  return device < 0 ? 0 : device;
}

inline xpytorch::xpu::api::Context* GetKunlunxinContext(const at::Tensor& tensor) {
  const int device = XdnnDeviceIndex(tensor);
  static std::mutex mutex;
  static std::unordered_map<int, xpytorch::xpu::api::Context*> contexts;
  std::lock_guard<std::mutex> lock(mutex);
  const auto it = contexts.find(device);
  if (it != contexts.end()) {
    return it->second;
  }

  int previous = 0;
  const bool switch_back =
      xpu_current_device(&previous) == 0 && previous != device;
  if (switch_back && xpu_set_device(device) != 0) {
    TORCH_CHECK(false, "kunlunxin: cannot select XPU device ", device);
  }
  xpytorch::xpu::api::Context* context = xpytorch::xpu::api::create_context();
  if (switch_back) {
    xpu_set_device(previous);
  }
  TORCH_CHECK(
      context != nullptr,
      "kunlunxin: create_context() returned null on device ",
      device);
  contexts.emplace(device, context);
  return context;
}

// The xdnn wrappers report failure as a plain int. The codes are the XPU
// runtime's errno space (`XPU_ERRNO_BASE` is 700) surfaced one layer up, which
// is what `xpu_strerror` names; the wrapper's own codes fall outside that space
// and come back as the runtime's unknown-code string. The numeric code is
// always part of the message so an unnamed one is still reportable.
inline std::string KunlunxinStatusName(int status) {
  std::string name = xpu_strerror(status);
  return name + " (" + std::to_string(status) + ")";
}

// `xpu_wait(nullptr)` is the only synchronisation primitive the XPU SDK exports:
// there is no `xpu_device_synchronize`, `xpu_stream_synchronize` or
// `xpu_stream_query`. A null stream waits for the device's default stream, which
// is where every kernel here lands -- `GetKunlunxinContext` leaves the context's
// `xpu_stream` at its default, and the runtime layer's stream bookkeeping
// (csrc/runtime/accelerator/kunlunxin/stream.cc) records the caller's stream
// without ever handing it to the SDK's launch path. So this drains exactly the
// work this op queued.
//
// Called on both sides of every vendor call. The trailing drain is what makes
// the op synchronous, which the dispatcher's contract requires. The leading one
// is the discipline measured on GCU: a vendor kernel that is merely *submitted*
// behind already-queued work does not necessarily read that work's output, and
// on this backend the other producer is the FlagGems/Triton path, which shares
// the same default stream (an op listed `= flaggems` in
// configs/backends_kunlunxin.conf launches through triton into the same queue).
// It is close to free in the steady state because the post-op drain has already
// emptied the stream.
inline void DrainXpuStream() {
  xpu_wait(nullptr);
}

// Exactly the aten dtypes `ToXdnnScalarType` below maps. Kernels test this
// first and take the CPU round-trip for anything outside it, which in practice
// means complex and the quantised dtypes: the op stays correct, and -- unlike
// the CUDA-boxing path this backend replaces -- nothing CUDA-backed is left
// behind to answer for it.
inline bool KunlunxinSupportsDtype(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
    case at::kDouble:
    case at::kHalf:
    case at::kBFloat16:
    case at::kInt:
    case at::kLong:
    case at::kShort:
    case at::kChar:
    case at::kByte:
    case at::kBool:
      return true;
    default:
      return false;
  }
}

// Which aten dtypes have a vendor scalar type. An unmapped dtype raises rather
// than being silently reinterpreted. Complex is the notable absence: the vendor
// enum has kcomplex64/kcomplex128 members, but no op in this backend is
// exercised on them, so they are deliberately not mapped -- a raise is the
// honest answer where an untested reinterpretation would be a silent one.
// `KunlunxinSupportsDtype` above is the same list, as a predicate.
inline xdnn_pytorch::ScalarType ToXdnnScalarType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:    return xdnn_pytorch::ScalarType::kfloat32;
    case at::kDouble:   return xdnn_pytorch::ScalarType::kfloat64;
    case at::kHalf:     return xdnn_pytorch::ScalarType::kfloat16;
    case at::kBFloat16: return xdnn_pytorch::ScalarType::kbfloat16;
    case at::kInt:      return xdnn_pytorch::ScalarType::kint32;
    case at::kLong:     return xdnn_pytorch::ScalarType::kint64;
    case at::kShort:    return xdnn_pytorch::ScalarType::kint16;
    case at::kChar:     return xdnn_pytorch::ScalarType::kint8;
    case at::kByte:     return xdnn_pytorch::ScalarType::kuint8;
    case at::kBool:     return xdnn_pytorch::ScalarType::kbool;
    default:
      TORCH_CHECK(
          false, "kunlunxin: unsupported dtype for the XPU backend: ", type);
  }
}

// The scalar operands (`alpha` on add/sub, the exponent-style scalars on the
// composing ops) are declared float32 and the libraries' kernels compute in
// float32, so the value is narrowed to float here rather than being passed at
// at::Scalar's own precision, which the vendor descriptor has no field for.
// `Scalar` has no converting constructor -- only `Scalar::create<T>(T)` -- so
// this is the only way to build one.
inline xdnn_pytorch::Scalar ToXdnnScalar(const at::Scalar& value) {
  return xdnn_pytorch::Scalar::create<float>(value.to<float>());
}

inline xdnn_pytorch::optional<xdnn_pytorch::ScalarType> ToXdnnOptionalDtype(
    ::std::optional<at::ScalarType> dtype) {
  if (!dtype.has_value()) {
    return xdnn_pytorch::optional<xdnn_pytorch::ScalarType>();
  }
  return xdnn_pytorch::optional<xdnn_pytorch::ScalarType>(
      ToXdnnScalarType(dtype.value()));
}

// A device guard on the same pattern as the GCU backend's TopsDeviceGuard: a
// vendor pointer resolves only against the *current* device, so an op on
// flagos:1 has to run with device 1 selected. Restores the previous device.
class KunlunxinDeviceGuard {
 public:
  explicit KunlunxinDeviceGuard(const at::Tensor& tensor) {
    const int target = XdnnDeviceIndex(tensor);
    int current = -1;
    if (xpu_current_device(&current) != 0 || current == target) {
      return;
    }
    if (xpu_set_device(target) != 0) {
      return;
    }
    previous_ = current;
  }

  ~KunlunxinDeviceGuard() {
    if (previous_ >= 0) {
      xpu_set_device(previous_);
    }
  }

  KunlunxinDeviceGuard(const KunlunxinDeviceGuard&) = delete;
  KunlunxinDeviceGuard& operator=(const KunlunxinDeviceGuard&) = delete;

 private:
  int previous_ = -1;
};

// An `xdnn_pytorch::Tensor` describing an aten tensor.
//
// The vendor descriptor owns copies of the shape and strides (they are
// `fixed_vector`s, not views), so nothing here borrows from the aten tensor and
// the wrapper is safe to move into a `std::vector` -- which is what `cat`
// needs. `data_ptr()` already accounts for the storage offset, so the
// descriptor's own offset field is left at 0.
//
// A rank-0 tensor is described as the equivalent 1-element vector: the vendor
// reshape path rejects a zero-length dims list, which is the same reason the
// GCU wrapper rewrites scalars this way.
class XdnnTensorWrapper {
 public:
  explicit XdnnTensorWrapper(const at::Tensor& tensor)
      : sizes_(tensor.defined() ? tensor.sizes().vec() : std::vector<int64_t>()),
        strides_(tensor.defined() ? tensor.strides().vec()
                                  : std::vector<int64_t>()),
        storage_(
            tensor.defined() ? const_cast<void*>(tensor.const_data_ptr())
                             : nullptr) {
    TORCH_CHECK(tensor.defined(), "kunlunxin: undefined tensor");
    if (sizes_.empty()) {
      sizes_.assign(1, 1);
      strides_.assign(1, 1);
    }
    tensor_ = xdnn_pytorch::Tensor(
        xdnn_pytorch::Shape(sizes_),
        xdnn_pytorch::Stride(strides_),
        ToXdnnScalarType(tensor.scalar_type()),
        storage_,
        xdnn_pytorch::MemoryFormat::Default,
        0,
        /*is_cpu=*/false);
  }

  // Inputs are `const Tensor&` in the vendor signatures, outputs `Tensor&`.
  const xdnn_pytorch::Tensor& get() const { return tensor_; }
  xdnn_pytorch::Tensor& get() { return tensor_; }

  XdnnTensorWrapper(const XdnnTensorWrapper&) = delete;
  XdnnTensorWrapper& operator=(const XdnnTensorWrapper&) = delete;

 private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  void* storage_;
  xdnn_pytorch::Tensor tensor_;
};

// `sum.dim_IntList` and `cat` take their dims as a `std::vector<int64_t>`,
// which is the type the header declares, so there is nothing to convert; this
// keeps the call sites readable where the source is an `IntArrayRef` or an
// `OptionalIntArrayRef`.
inline std::vector<int64_t> ToXdnnDims(at::IntArrayRef dims) {
  return dims.vec();
}

inline std::vector<int64_t> ToXdnnDims(at::OptionalIntArrayRef dims) {
  return dims.has_value() ? dims->vec() : std::vector<int64_t>();
}

} // namespace at::native::flagos::kunlunxin

// Issues one xdnn call and waits for it: pre-op drain, call, status check,
// post-op drain. See DrainXpuStream for why both drains exist.
//
// The device guard is *not* part of this macro. It has to be opened before the
// output tensor is allocated (at::empty resolves the device from the current
// one, and the allocator would otherwise hand back memory on the wrong card),
// so kernels open it as their first statement and call this only for the call
// itself.
#define KUNLUNXIN_EXEC(op, ...)                                                \
  do {                                                                         \
    at::native::flagos::kunlunxin::DrainXpuStream();                           \
    const int _kunlunxin_rc = (__VA_ARGS__);                                   \
    TORCH_CHECK(                                                               \
        _kunlunxin_rc == 0,                                                    \
        #op,                                                                   \
        " failed on the Kunlunxin XPU backend: ",                              \
        at::native::flagos::kunlunxin::KunlunxinStatusName(_kunlunxin_rc));     \
    at::native::flagos::kunlunxin::DrainXpuStream();                           \
  } while (0)

#endif // USE_KUNLUNXIN
