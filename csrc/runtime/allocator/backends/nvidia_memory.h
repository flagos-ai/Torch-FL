// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <cuda_runtime.h>

// The caching-allocator delegation below reaches into c10::cuda, which only
// exists when torch was built with CUDA and that build's libtorch_cuda.so is in
// the process. The default FlagGems-primary wheel (FLAGOS_BUILD_BOXING=OFF)
// ships neither -- it links the CPU-only torch wheel and drives the vendor
// runtime's cudaMalloc/cudaFree through the block pool instead -- so the
// includes and the whole delegation block are gated on the boxing build.
#ifdef FLAGOS_BOXING_BUILD
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <mutex>
#endif

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
    cudaError_t err = cudaGetDevice(device);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
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
#ifdef FLAGOS_BOXING_BUILD
  bool provides_caching() const override { return true; }

  void* caching_alloc(size_t nbytes, Stream_t stream) override {
    ensure_initialized();
    if (stream != nullptr) {
      return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
          nbytes, reinterpret_cast<cudaStream_t>(stream));
    }
    return c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);
  }

  void caching_free(void* ptr) override {
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
  }

  void caching_empty_cache() override {
    ensure_initialized();
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

  void caching_record_stream(void* ptr, Stream_t stream) override {
    if (ptr == nullptr || stream == nullptr) {
      return;
    }
    int device = 0;
    cudaGetDevice(&device);
    // Non-owning DataPtr: the real deleter lives with the owning tensor; here
    // we only need a handle for recordStream to associate the block.
    at::DataPtr dp(
        ptr, ptr, [](void*) {},
        c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
    auto cuda_stream = c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(stream),
        static_cast<c10::DeviceIndex>(device));
    c10::cuda::CUDACachingAllocator::recordStream(dp, cuda_stream);
    // dp holds a no-op deleter, so its destruction here frees nothing; the
    // owning tensor's DataPtr remains the sole owner of the block.
  }

  bool caching_get_stats(int device, AllocatorStats* out) override {
    ensure_initialized();
    auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(
        static_cast<c10::DeviceIndex>(device));
    constexpr auto kAgg =
        static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    out->bytes_allocated =
        static_cast<size_t>(st.allocated_bytes[kAgg].current);
    out->bytes_reserved = static_cast<size_t>(st.reserved_bytes[kAgg].current);
    out->peak_allocated = static_cast<size_t>(st.allocated_bytes[kAgg].peak);
    out->peak_reserved = static_cast<size_t>(st.reserved_bytes[kAgg].peak);
    out->num_alloc_calls = static_cast<size_t>(st.allocation[kAgg].allocated);
    out->num_free_calls = static_cast<size_t>(st.allocation[kAgg].freed);
    out->num_device_malloc = static_cast<size_t>(st.num_device_alloc);
    out->num_device_free = static_cast<size_t>(st.num_device_free);
    out->num_alloc_retries = static_cast<size_t>(st.num_alloc_retries);
    return true;
  }

  void caching_reset_peak_stats(int device) override {
    ensure_initialized();
    c10::cuda::CUDACachingAllocator::resetPeakStats(
        static_cast<c10::DeviceIndex>(device));
  }

 private:
  // The CUDA caching allocator asserts unless its per-device tables are sized
  // by init(device_count). PyTorch normally does this in torch.cuda lazy init;
  // in the flagos external-libtorch scheme we may allocate before that runs, so
  // initialize it ourselves (C++-only, no Python lazy-init needed).
  void ensure_initialized() {
    static std::once_flag flag;
    std::call_once(flag, []() {
      int n = 0;
      cudaGetDeviceCount(&n);
      c10::cuda::CUDACachingAllocator::init(n);
    });
  }
#else
  // No c10::cuda in the process (see the include guard above), so there is
  // nothing to delegate to. Returning false leaves CachingDeviceAllocator on
  // its own block pool over this backend's cudaMalloc/cudaFree, which are plain
  // runtime calls the vendor toolkit backs.
  bool provides_caching() const override { return false; }
#endif
};

} // namespace c10::flagos
