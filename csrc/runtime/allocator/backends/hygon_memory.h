// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <ATen/Context.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <optional>

#include <cuda_runtime.h>

namespace c10::flagos {

// DCU (Hygon, DTK) implementation of DeviceMemoryInterface.
//
// DTK ships a CUDA compatibility toolkit ($ROCM_PATH/cuda/cuda-12) whose
// libcudart.so.12 is a thin shim over libgalaxyhip.so -- the very same runtime
// libtorch_hip.so is built against. So the plain CUDA runtime calls below reach
// the same driver state the vendor's HIP kernels use, and no hipify is needed
// for raw memory ops.
//
// Caching is delegated to torch's own device allocator, exactly as on NVIDIA --
// but reached through the device-type-generic registry rather than the
// c10::cuda:: namespace. That indirection matters here: the DCU wheel exports
// c10::hip::HIPCachingAllocator and has zero c10::cuda symbols, and we cannot
// call the HIP API directly because cuda_runtime.h and hip/hip_runtime.h cannot
// coexist in one translation unit (dim3, uchar1..char4 etc. collide).
// c10::getDeviceAllocator(kCUDA) sidesteps both problems: it is a header-only
// lookup in c10's allocator registry, where the hipified build registers its
// HIPCachingAllocator under DeviceType::CUDA (the same reason the boxing
// kernels work at all).
//
// Net effect: flagos tensors and the boxed kernels' outputs share ONE pool, so
// memory_allocated()/memory_reserved()/empty_cache() report and act on real
// usage -- matching backends/cuda_memory.h.
class DcuDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    // Ensure the allocation lands on the current process device (matters under
    // DDP, where each rank sets its own device).
    int device = 0;
    cudaGetDevice(&device);
    cudaError_t set_err = cudaSetDevice(device);
    if (set_err != cudaSuccess) {
      *ptr = nullptr;
      return ErrorInvalidDevice;
    }
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

  // --- Caching delegation to torch's own (HIP) device allocator ---

  bool provides_caching() const override { return true; }

  void* caching_alloc(size_t nbytes, Stream_t stream) override {
#ifndef FLAGOS_BOXING_BUILD
    const auto physical = physical_stream(stream);
    c10::impl::VirtualGuardImpl implementation(c10::DeviceType::HIP);
    std::optional<c10::StreamGuard> guard;
    if (implementation.getStream(physical.device()) != physical) guard.emplace(physical);
#else
    // CachingDeviceAllocator only ever asks for the current stream (it passes
    // nullptr), and the generic DeviceAllocator has no alloc-with-stream entry
    // point, so there is nothing to honor here. raw_allocate() associates the
    // block with the current stream, which is the one the boxed kernels run on.
    (void)stream;
#endif
    return allocator()->raw_allocate(nbytes);
  }

  void caching_free(void* ptr) override { allocator()->raw_deallocate(ptr); }

  void caching_empty_cache() override { allocator()->emptyCache(); }

  void caching_record_stream(void* ptr, Stream_t stream) override {
#ifndef FLAGOS_BOXING_BUILD
    if (!ptr) return;
    const auto physical = physical_stream(stream);
    // Preserve the allocator's expected deleter identity while borrowing the
    // pointer solely for dependency tracking. Ownership stays with the tensor.
    at::DataPtr borrowed(ptr, ptr, allocator()->raw_deleter(), physical.device());
    try {
      allocator()->recordStream(borrowed, physical);
    } catch (...) {
      borrowed.release_context();
      throw;
    }
    borrowed.release_context();
#else
    // No-op. DeviceAllocator::recordStream needs a c10::Stream on the CUDA
    // device type, and the only way to build one from a raw stream handle is
    // c10::cuda::getStreamFromExternal -- a symbol this wheel does not export.
    // This is not a regression: flagos stream ids are raw device stream
    // pointers, so before delegation the block lookup for a torch-pool pointer
    // failed and record_stream returned without doing anything either.
#endif
  }

  bool caching_get_stats(int device, AllocatorStats* out) override {
    auto st = allocator()->getDeviceStats(static_cast<c10::DeviceIndex>(device));
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
    allocator()->resetPeakStats(static_cast<c10::DeviceIndex>(device));
  }

 private:
#ifndef FLAGOS_BOXING_BUILD
  static c10::Stream physical_stream(Stream_t stream) {
    int device = 0;
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cannot read HCU allocation device");
    if (!stream) stream = ::GetCurrentStreamForDevice(device);
    // DTK's generic device allocator accepts HIP external-stream descriptors.
    // Their IDs carry aligned raw handles. No TensorImpl or dispatch key changes.
    return c10::Stream(c10::Stream::UNSAFE,
        c10::Device(c10::DeviceType::HIP, static_cast<c10::DeviceIndex>(device)),
        static_cast<c10::StreamId>(reinterpret_cast<uintptr_t>(stream)));
  }
#endif
  // The HIP caching allocator asserts unless its per-device tables are sized by
  // init(device_count). PyTorch normally does that during torch.cuda lazy init;
  // in the flagos external-libtorch scheme we may allocate before that runs, so
  // drive it ourselves. lazyInitDevice is the device-generic entry point, so
  // this needs no c10::cuda / c10::hip symbol.
  static c10::DeviceAllocator* allocator() {
    static c10::DeviceAllocator* alloc = []() {
      at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
      return c10::getDeviceAllocator(c10::DeviceType::CUDA);
    }();
    return alloc;
  }
};

} // namespace c10::flagos
