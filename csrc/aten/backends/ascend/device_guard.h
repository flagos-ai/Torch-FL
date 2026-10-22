// Copyright (c) 2026, BAAI. All rights reserved.
//
// Issue #326: the device context an Ascend kernel runs in.
//
// Every layer below a kernel boundary reads the *ambient* current device rather
// than the device of its tensor arguments: the aclnn workspace is allocated
// with an index-less `at::kPrivateUse1` TensorOptions (which resolves to
// whatever device is current when `at::empty` runs), `EXEC_ASCEND_CMD` takes
// its launch stream from `GetCurrentAclStream()`, and `ExecAscendCached` keys
// the executor cache and allocates its workspace from `::GetDevice`. The
// generated output allocation compounded it: `OpPreparation` forced the device
// *type* to PrivateUse1 while dropping the index. A kernel whose inputs were
// all on flagos:0 but whose ambient device was flagos:1 therefore returned its
// result on flagos:1, and -- worse on a two-device box -- enqueued work on the
// wrong device's stream.
//
// `OpDeviceGuard` closes that at the operator boundary: each kernel selects the
// device of its primary tensor argument on entry and restores the caller's
// device on exit, so every ambient-device read downstream becomes correct
// without each of those layers having to learn about tensors. Every Ascend
// kernel takes one, generated and handwritten alike; `DeviceOf` below is the
// per-signature way to name the primary argument.
//
// This header is deliberately small (flagos.h + ATen/core/Tensor.h) so the
// kernels that only delegate to CPU can take the guard without pulling in the
// aclnn marshalling in op_api_common.h.

#pragma once

#include <flagos.h>
#include <ATen/core/Tensor.h>

#include <optional>

namespace at::native::flagos::ascend {

// Cost on the dispatch path is one thread-local read (`::GetDevice` returns
// device.cc's cached `gCurrentDevice`, not a runtime call). `::SetDevice` is
// only reached when the device actually differs, because it is not cheap -- it
// round-trips through `aclrtGetDeviceCount` before `aclrtSetDevice`.
class OpDeviceGuard {
 public:
  explicit OpDeviceGuard(const c10::Device& target) {
    if (!target.is_privateuseone() || target.index() < 0) {
      return;
    }
    int current = 0;
    ::GetDevice(&current);
    if (current == target.index()) {
      return;
    }
    previous_ = current;
    ::SetDevice(target.index());
    active_ = true;
  }

  ~OpDeviceGuard() {
    if (active_) {
      ::SetDevice(previous_);
    }
  }

  OpDeviceGuard(const OpDeviceGuard&) = delete;
  OpDeviceGuard& operator=(const OpDeviceGuard&) = delete;

 private:
  int previous_ = -1;
  bool active_ = false;
};

// The device to guard with, taken from a kernel's primary tensor argument.
// Returns an index-less PrivateUse1 device (which `OpDeviceGuard` reads as "no
// guard") when there is nothing to derive one from, so one call expression
// covers every kernel shape without a per-kernel null branch.
inline c10::Device DeviceOf(const at::Tensor& tensor) {
  return tensor.defined() ? tensor.device()
                          : c10::Device(c10::DeviceType::PrivateUse1, -1);
}

inline c10::Device DeviceOf(at::TensorList tensors) {
  return tensors.empty() ? c10::Device(c10::DeviceType::PrivateUse1, -1)
                         : tensors[0].device();
}

inline c10::Device DeviceOf(const at::ITensorListRef& tensors) {
  return tensors.size() == 0 ? c10::Device(c10::DeviceType::PrivateUse1, -1)
                             : (*tensors.begin()).device();
}

// Factories (`zeros`, `linspace`, `scalar_tensor`, ...) have no tensor argument
// to derive a device from, but do carry an explicit one. An absent device stays
// index-less, which reads as "no guard".
inline c10::Device DeviceOf(const ::std::optional<at::Device>& device) {
  return device.has_value() ? *device
                            : c10::Device(c10::DeviceType::PrivateUse1, -1);
}

} // namespace at::native::flagos::ascend
