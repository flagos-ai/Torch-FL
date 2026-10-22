// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstddef>

#include <flagos.h>

#include "allocator_stats.h"

namespace c10::flagos {

// Abstract interface for device memory operations.
// Each hardware backend (CUDA, Metax, Ascend) implements this interface
// to provide raw memory allocation and synchronization primitives.
struct DeviceMemoryInterface {
  virtual ~DeviceMemoryInterface() = default;

  // Raw device memory allocation (e.g., cudaMalloc)
  virtual Error_t device_malloc(void** ptr, size_t size) = 0;
  // Raw device memory free (e.g., cudaFree)
  virtual Error_t device_free(void* ptr) = 0;

  // Device management
  virtual Error_t get_device_index(int* device) = 0;
  virtual Error_t set_device(int device) = 0;

  // Memory info query (free / total bytes on current device)
  virtual Error_t get_memory_info(size_t* free, size_t* total) = 0;

  // Event-based synchronization for stream safety
  virtual Error_t event_create(Event_t* event) = 0;
  virtual Error_t event_destroy(Event_t event) = 0;
  virtual Error_t event_record(Event_t event, Stream_t stream) = 0;
  // Returns Success if event has completed, ErrorNotReady otherwise
  virtual Error_t event_query(Event_t event) = 0;

  // Memcpy for block data migration
  virtual Error_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind) = 0;

  // --- Optional caching-allocator delegation ---
  //
  // A backend whose platform already ships a mature caching allocator (e.g.
  // CUDA's c10::cuda::CUDACachingAllocator) can return true from
  // provides_caching(). CachingDeviceAllocator then delegates ALL allocation,
  // caching, stats and empty_cache to it, bypassing its own block pool. This
  // keeps flagos `empty` and boxed-kernel outputs in a single shared pool so
  // memory stats reflect the true device footprint.
  //
  // Backends without such a facility (default) return false and get the
  // built-in block-pool caching. Ascend/others keep the self-managed path.
  virtual bool provides_caching() const { return false; }

  // Allocate/free through the platform caching allocator. stream may be null
  // for the default stream.
  virtual void* caching_alloc(size_t /*nbytes*/, Stream_t /*stream*/) {
    return nullptr;
  }
  virtual void caching_free(void* /*ptr*/) {}

  // Release cached-but-unused memory back to the device.
  virtual void caching_empty_cache() {}

  // Mark that a pointer is used on the given stream (deferred-free safety).
  virtual void caching_record_stream(void* /*ptr*/, Stream_t /*stream*/) {}

  // Fill *out with the platform allocator's per-device stats. Returns false if
  // unsupported (caller then falls back to its own stats).
  virtual bool caching_get_stats(int /*device*/, AllocatorStats* /*out*/) {
    return false;
  }
  virtual void caching_reset_peak_stats(int /*device*/) {}
};

} // namespace c10::flagos
