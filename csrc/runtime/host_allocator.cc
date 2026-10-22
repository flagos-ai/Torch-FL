// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <c10/core/Allocator.h>
#include <ATen/core/TensorBase.h>

#include <flagos.h>

namespace c10::flagos {

namespace {

struct HostAllocator final : c10::Allocator {
  HostAllocator() = default;

  static void ReportAndDelete(void* ptr) {
    if (ptr) {
      ::FreeHost(ptr);
    }
  }

  c10::DataPtr allocate(size_t size) override {
    void* ptr = nullptr;
    if (size > 0) {
      ::MallocHost(&ptr, size);
    }
    return {ptr, ptr, &ReportAndDelete, c10::DeviceType::CPU};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    ::Memcpy(dest, src, count, MemcpyHostToHost);
  }
};

static HostAllocator host_alloc;

} // namespace

c10::Allocator* GetHostAllocator() {
  return &host_alloc;
}

} // namespace c10::flagos
