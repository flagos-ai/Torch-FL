// Copyright (c) 2026, BAAI. All rights reserved.

// Iluvatar corex implementation of DeviceMemoryInterface.
//
// Same C API as the CUDA backend -- corex ships the CUDA runtime under its own
// name (cuda_runtime.h at <corex>/include, libcudart.so.10.2) -- but with the
// one part of cuda_memory.h that is genuinely CUDA-and-torch-only removed: the
// caching-allocator delegation to c10::cuda::CUDACachingAllocator. That
// delegation exists on the CUDA platform because flagos storage there *is* torch
// CUDA storage (boxing only relabels the device), so both allocators have to be
// one pool. Iluvatar is not a boxing platform: nothing in its build enters the
// CUDA dispatch key, so there is no torch CUDA allocator to share with, and the
// symbols needed to ask for one live in libc10_cuda.so -- a library the CPU torch
// wheel does not ship and that no iluvatar build may link (see
// runtime/accelerator/iluvatar/stream.cc).
//
// provides_caching() therefore keeps the interface default (false) and the
// caching allocator in front of this (runtime/allocator/caching_device_
// allocator.cc) caches the corex blocks itself.
//
// Same shape as backends/dcu_memory.h, which drops the same delegation for the
// same kind of reason (DTK's wheel exports c10::hip, not c10::cuda).

#pragma once

#include "../device_memory_interface.h"

#include <cuda_runtime.h>
#include "runtime/accelerator/iluvatar/iluvatar_context.h"

namespace c10::flagos {

// corex implementation of DeviceMemoryInterface. Directly calls the CUDA-shaped
// runtime APIs for raw memory operations.
class IluvatarDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    return Success;
  }

  Error_t device_free(void* ptr) override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t get_device_index(int* device) override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaError_t err = cudaGetDevice(device);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t set_device(int device) override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaError_t err = cudaSetDevice(device);
    return (err == cudaSuccess) ? Success : ErrorInvalidDevice;
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaError_t err = cudaMemGetInfo(free, total);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_create(Event_t* event) override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaEvent_t cuda_event;
    cudaError_t err =
        cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      return ErrorUnknown;
    }
    *event = reinterpret_cast<Event_t>(cuda_event);
    return Success;
  }

  Error_t event_destroy(Event_t event) override {
    cudaError_t err = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaError_t err = cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(event),
        reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_query(Event_t event) override {
    cudaError_t err = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
    if (err == cudaSuccess) {
      return Success;
    } else if (err == cudaErrorNotReady) {
      return ErrorNotReady;
    }
    return ErrorUnknown;
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind)
      override {
    c10::flagos::iluvatar::EnsureIluvatarContext();
    cudaMemcpyKind cuda_kind;
    switch (kind) {
      case MemcpyHostToHost:
        cuda_kind = cudaMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
      case MemcpyDeviceToHost:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
      case MemcpyDeviceToDevice:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
      default:
        return ErrorUnknown;
    }
    cudaError_t err = cudaMemcpy(dst, src, count, cuda_kind);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }
};

} // namespace c10::flagos
