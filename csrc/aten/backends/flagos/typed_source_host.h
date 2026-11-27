// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <atomic>
#include <string>
#include <tuple>

#include "source_host.h"

namespace at::native::flagos {

class TypedSourceGuard {
 public:
  bool Check(const at::Tensor& value) {
    if (!value.defined()) return true;
    if (value.key_set().has(c10::DispatchKey::Python) ||
        value.key_set().has(c10::DispatchKey::Functionalize))
      return false;
    if (!device_ && value.device().is_privateuseone()) device_ = value.device();
    return true;
  }
  bool Check(const c10::Device& value) {
    if (value.is_privateuseone()) device_ = value;
    return true;
  }
  template <class T>
  bool Check(const T&) {
    return true;
  }
  template <class T>
  bool Check(const std::optional<T>& value) {
    return !value || Check(*value);
  }
  template <class T>
  bool Check(const c10::List<T>& values) {
    for (const auto& value : values)
      if (!Check(static_cast<T>(value))) return false;
    return true;
  }
  template <class T>
  bool Check(at::ArrayRef<T> values) {
    for (const auto& value : values)
      if (!Check(value)) return false;
    return true;
  }
  bool Check(const at::ITensorListRef& values) {
    for (const auto& value : values)
      if (!Check(value)) return false;
    return true;
  }
  void Enter() {
    if (device_) {
      c10::impl::VirtualGuardImpl implementation(device_->type());
      if (implementation.getDevice() != *device_) guard_.reset_device(*device_);
    }
  }

 private:
  std::optional<c10::Device> device_;
  c10::OptionalDeviceGuard guard_;
};

c10::Device TypedSourceFactoryDevice(std::optional<c10::Device> device,
                                     const at::Tensor* like = nullptr);

// The generated core owns each slot. Registration validates its immutable ABI
// before publishing a pointer; warm calls only load that pointer, never a map.
class TypedSourceSlot {
 public:
  TypedSourceSlot(const char* op, const char* host, const char* abi, const char* catalog);
  bool Available() const {
    return address.load(std::memory_order_acquire) != 0 && CanRun();
  }
  template <class R, class F, class... A>
  R Call(F&& fallback, const A&... args) const {
    const auto target = address.load(std::memory_order_acquire);
    if (target && CanRun()) {
      TypedSourceGuard guard;
      if ((guard.Check(args) && ...)) {
        guard.Enter();
        R result;
        using Entry = bool (*)(R&, const A&...);
        if (reinterpret_cast<Entry>(target)(result, args...)) {
          hits.fetch_add(1, std::memory_order_relaxed);
          return result;
        }
      }
    }
    return fallback();
  }

  const char* const op;
  const char* const host;
  const char* const abi;
  const char* const catalog;
  std::atomic<uintptr_t> address{0};
  mutable std::atomic<int64_t> hits{0};

 private:
  bool CanRun() const {
    // ATen's mixed-scalar where creates a scalar tensor inside a composite
    // with grad mode enabled. That factory has no differentiable inputs.
    return scalar_factory_ || NativeSourceCanRun();
  }
  const bool scalar_factory_;
};

}  // namespace at::native::flagos
