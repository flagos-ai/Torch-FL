// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <cuda_runtime.h>

namespace c10::flagos {

// CUDA implementation of DeviceMemoryInterface.
// Directly calls CUDA runtime APIs for raw memory operations.
class CudaDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    return Success;
  }

  Error_t device_free(void* ptr) override {
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t get_device_index(int* device) override {
#if defined(FLAGOS_ACCEL_CUDA) && defined(FLAGOS_NATIVE_HOSTS)
    return ::GetDevice(device);
#else
    cudaError_t err = cudaGetDevice(device);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
#endif
  }

  Error_t set_device(int device) override {
    cudaError_t err = cudaSetDevice(device);
    return (err == cudaSuccess) ? Success : ErrorInvalidDevice;
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    cudaError_t err = cudaMemGetInfo(free, total);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_create(Event_t* event) override {
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

  // --- Caching-allocator delegation to c10::cuda::CUDACachingAllocator ---
  //
  // flagos and CUDA share the same physical GPU memory (boxing only relabels
  // the device, no copy). By routing flagos `at::empty` through the SAME CUDA
  // caching allocator that boxed kernels use for their outputs, all device
  // memory lives in one pool -> stats/empty_cache/OOM-retry are unified, and
  // torch.randn (a compute-factory that allocates on the CUDA device) is now
  // reflected in flagos memory stats.
  // No c10::cuda in the process (see the include guard above), so there is
  // nothing to delegate to. Returning false leaves CachingDeviceAllocator on
  // its own block pool over this backend's cudaMalloc/cudaFree, which are plain
  // runtime calls the vendor toolkit backs.
  bool provides_caching() const override { return false; }
};

} // namespace c10::flagos
