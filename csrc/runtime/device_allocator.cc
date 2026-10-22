// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.cpp
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "device_allocator.h"
#include "allocator/caching_device_allocator.h"

namespace c10::flagos {

// Use caching allocator by default, fall back to passthrough if disabled.
static at::Allocator* get_device_allocator() {
  if (CachingDeviceAllocator::is_enabled()) {
    auto* caching = GetCachingAllocator();
    if (caching) {
      return caching;
    }
  }
  // Fallback: direct passthrough allocator (no caching).
  static DeviceAllocator passthrough_alloc;
  return &passthrough_alloc;
}

// We cannot use REGISTER_ALLOCATOR with a function call directly,
// so we use a wrapper struct that delegates.
struct AllocatorProxy final : at::Allocator {
  at::DataPtr allocate(size_t nbytes) override {
    return get_device_allocator()->allocate(nbytes);
  }
  at::DeleterFnPtr raw_deleter() const override {
    return get_device_allocator()->raw_deleter();
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    get_device_allocator()->copy_data(dest, src, count);
  }
};

static AllocatorProxy global_alloc_proxy;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_alloc_proxy);

} // namespace c10::flagos
