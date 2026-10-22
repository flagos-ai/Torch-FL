// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstddef>

namespace c10::flagos {

// Statistics for memory usage on a single device.
struct AllocatorStats {
  size_t bytes_allocated = 0;    // currently allocated by user
  size_t bytes_reserved = 0;     // total held by allocator (allocated + cached)
  size_t peak_allocated = 0;
  size_t peak_reserved = 0;
  size_t num_alloc_calls = 0;
  size_t num_free_calls = 0;
  size_t num_device_malloc = 0;  // actual calls to device_malloc
  size_t num_device_free = 0;    // actual calls to device_free
  size_t num_alloc_retries = 0;  // OOM retries
};

} // namespace c10::flagos
