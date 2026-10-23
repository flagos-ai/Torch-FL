// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "../device_memory_interface.h"

#include <musa_runtime.h>
#include <cstdio>

namespace c10::flagos {

// Moore Threads MUSA implementation of DeviceMemoryInterface.
// Calls the musa* runtime API directly for raw memory operations.
//
// Note on caching: unlike backends/cuda_memory.h, this does NOT delegate to the
// vendor caching allocator (c10::musa::MUSACachingAllocator). That allocator
// installs itself as *the* PrivateUse1 allocator via
// at::SetAllocator(c10::kPrivateUse1, ...) -- the same slot flagos claims with
// REGISTER_ALLOCATOR(PrivateUse1) in runtime/device_allocator.cc. Since MUSA is
// a PrivateUse1 backend there is no second key to separate them (contrast CUDA,
// where the CUDA key holds the vendor pool and PrivateUse1 holds ours), so
// routing through it would mean two owners of one allocator slot. flagos keeps
// its own caching layer on top of raw musaMalloc instead.
class MusaDeviceMemory final : public DeviceMemoryInterface {
 public:
  Error_t device_malloc(void** ptr, size_t size) override {
    musaError_t err = musaMalloc(ptr, size);
    if (err != musaSuccess) {
      fprintf(stderr, "[flagos-musa] musaMalloc(%zu bytes) failed: %s\n",
              size, musaGetErrorString(err));
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    return Success;
  }

  Error_t device_free(void* ptr) override {
    musaError_t err = musaFree(ptr);
    if (err != musaSuccess) {
      fprintf(stderr, "[flagos-musa] musaFree(%p) failed: %s\n",
              ptr, musaGetErrorString(err));
      return ErrorUnknown;
    }
    return Success;
  }

  Error_t get_device_index(int* device) override {
    musaError_t err = musaGetDevice(device);
    return (err == musaSuccess) ? Success : ErrorUnknown;
  }

  Error_t set_device(int device) override {
    musaError_t err = musaSetDevice(device);
    return (err == musaSuccess) ? Success : ErrorInvalidDevice;
  }

  Error_t get_memory_info(size_t* free, size_t* total) override {
    musaError_t err = musaMemGetInfo(free, total);
    return (err == musaSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_create(Event_t* event) override {
    musaEvent_t musa_event;
    musaError_t err =
        musaEventCreateWithFlags(&musa_event, musaEventDisableTiming);
    if (err != musaSuccess) {
      return ErrorUnknown;
    }
    *event = reinterpret_cast<Event_t>(musa_event);
    return Success;
  }

  Error_t event_destroy(Event_t event) override {
    musaError_t err = musaEventDestroy(reinterpret_cast<musaEvent_t>(event));
    return (err == musaSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_record(Event_t event, Stream_t stream) override {
    musaError_t err = musaEventRecord(
        reinterpret_cast<musaEvent_t>(event),
        reinterpret_cast<musaStream_t>(stream));
    return (err == musaSuccess) ? Success : ErrorUnknown;
  }

  Error_t event_query(Event_t event) override {
    musaError_t err = musaEventQuery(reinterpret_cast<musaEvent_t>(event));
    if (err == musaSuccess) {
      return Success;
    } else if (err == musaErrorNotReady) {
      return ErrorNotReady;
    }
    return ErrorUnknown;
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind)
      override {
    musaMemcpyKind musa_kind;
    switch (kind) {
      case MemcpyHostToHost:
        musa_kind = musaMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        musa_kind = musaMemcpyHostToDevice;
        break;
      case MemcpyDeviceToHost:
        musa_kind = musaMemcpyDeviceToHost;
        break;
      case MemcpyDeviceToDevice:
        musa_kind = musaMemcpyDeviceToDevice;
        break;
      default:
        return ErrorUnknown;
    }
    musaError_t err = musaMemcpy(dst, src, count, musa_kind);
    return (err == musaSuccess) ? Success : ErrorUnknown;
  }
};

} // namespace c10::flagos
