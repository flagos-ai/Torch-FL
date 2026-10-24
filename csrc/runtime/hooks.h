// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU) && !defined(USE_MPS)
#include <cuda_runtime.h>
#endif
#if defined(USE_MUSA)
// MUSA has no cudart shim (unlike metax/dcu), so the pinned-pointer probe below
// uses the musa* spelling of the same API.
#include <musa_runtime.h>
#endif
#include <flagos.h>

#include "generator.h"
#include "host_allocator.h"

namespace c10::flagos {

#if defined(USE_DCU)
// On DCU, flagos.h's PointerGetAttributes has no symbol in the built library:
// DTK's <cuda_runtime.h> turns the two-argument name into a template, so the
// definition in runtime/accelerator/nvidia/memory.cc is a different one from
// the flat function this header calls. Declared here, beside its only caller,
// for the reason guard.h gives; both declarations name the same definition.
__attribute__((visibility("default"))) Error_t
DcuPointerGetAttributes(PointerAttributes* attributes, const void* ptr);
#endif

struct HooksInterface : public at::PrivateUse1HooksInterface {
  HooksInterface() {};
  ~HooksInterface() override = default;

  bool isAvailable() const override {
    int count = 0;
    ::GetDeviceCount(&count);
    return count > 0;
  }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return true;
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return GetHostAllocator();
  }

  bool isPinnedPtr(const void* data) const override {
    // First check flagos's own registry
    PointerAttributes fo_attr{};
#if defined(USE_DCU)
    DcuPointerGetAttributes(&fo_attr, data);
#else
    ::PointerGetAttributes(&fo_attr, data);
#endif
    if (fo_attr.type == MemoryTypeHost) {
      return true;
    }

#if defined(USE_MUSA)
    // Same fallback as the CUDA branch below, in musa* spelling: torch_musa's
    // pinned allocator uses musaMallocHost, so those pointers are absent from
    // flagos's registry yet still valid pinned memory for DDP.
    musaPointerAttributes musa_attr{};
    musaError_t merr = musaPointerGetAttributes(&musa_attr, data);
    if (merr == musaSuccess && musa_attr.type == musaMemoryTypeHost) {
      return true;
    }
    if (merr != musaSuccess) {
      musaGetLastError();
    }
#elif !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_BPU) && !defined(USE_MPS)
    // Fallback: check if it's CUDA pinned memory
    // This is needed because when CUDA is present, PyTorch's pinned memory
    // allocator defaults to CUDA's cudaMallocHost, which won't be in flagos's
    // registry but is still valid pinned memory for DDP operations.
    cudaPointerAttributes cuda_attr{};
    cudaError_t err = cudaPointerGetAttributes(&cuda_attr, data);
    if (err == cudaSuccess && cuda_attr.type == cudaMemoryTypeHost) {
      return true;
    }
    // Clear any CUDA error
    if (err != cudaSuccess) {
      cudaGetLastError();
    }
#endif

    return false;
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    return c10::flagos::GetDefaultGenerator(device_index);
  }

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return at::make_generator<GeneratorImpl>(device_index);
  }

  // Required by FSDP for storage resize (parameter sharding)
  void resizePrivateUse1Bytes(
      const c10::Storage& storage,
      size_t newsize) const override {
    auto* storage_impl = storage.unsafeGetStorageImpl();
    size_t old_nbytes = storage_impl->nbytes();

    if (newsize == old_nbytes) {
      return;
    }

    auto* allocator = storage_impl->allocator();
    TORCH_CHECK(allocator != nullptr,
                "resizePrivateUse1Bytes: allocator is null");

    if (newsize == 0) {
      // Free the old storage and set to empty
      storage_impl->set_data_ptr_noswap(
          at::DataPtr(nullptr, nullptr,
                      allocator->raw_deleter(),
                      storage_impl->device()));
      storage_impl->set_nbytes(0);
      return;
    }

    // Allocate new memory
    at::DataPtr new_data = allocator->allocate(newsize);
    TORCH_CHECK(new_data.get() != nullptr || newsize == 0,
                "Failed to allocate ", newsize, " bytes for resize");

    // Copy existing data if any
    if (old_nbytes > 0 && storage_impl->data()) {
      size_t copy_size = std::min(old_nbytes, newsize);
      ::Memcpy(new_data.get(), storage_impl->data(),
               copy_size, MemcpyDeviceToDevice);
    }

    // Replace storage data
    storage_impl->set_data_ptr_noswap(std::move(new_data));
    storage_impl->set_nbytes(newsize);
  }
};

} // namespace c10::flagos
