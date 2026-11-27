// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.cpp
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "device_allocator.h"
#include "allocator/caching_device_allocator.h"

#include <c10/core/CachingDeviceAllocator.h>
#include <torch/version.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>

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
struct AllocatorProxy final : c10::DeviceAllocator {
  at::DataPtr allocate(size_t nbytes) override {
    auto result = get_device_allocator()->allocate(nbytes);
    initialized_.store(true, std::memory_order_relaxed);
    return result;
  }
  at::DeleterFnPtr raw_deleter() const override {
    return get_device_allocator()->raw_deleter();
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    get_device_allocator()->copy_data(dest, src, count);
  }

  bool initialized() override {
    return initialized_.load(std::memory_order_relaxed);
  }

  void emptyCache(c10::MempoolId_t pool) override {
    TORCH_CHECK_NOT_IMPLEMENTED(
        pool == c10::MempoolId_t{}, "FlagOS does not support private allocator pools");
    if (CachingDeviceAllocator::is_enabled()) {
      GetCachingAllocator()->empty_cache();
    }
  }

  void recordStream(const at::DataPtr& ptr, c10::Stream stream) override {
    if (CachingDeviceAllocator::is_enabled()) {
      GetCachingAllocator()->record_stream(ptr, reinterpret_cast<Stream_t>(stream.id()));
    }
  }

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    c10::CachingDeviceAllocator::DeviceStats result;
    if (!CachingDeviceAllocator::is_enabled()) {
      return result;
    }
    std::lock_guard<std::mutex> lock(stats_mutex_);
    const auto stats = GetCachingAllocator()->get_stats(device);
    const auto baseline = stats_baselines_.find(device);
    const auto since = baseline == stats_baselines_.end() ? AllocatorStats{} : baseline->second;

    // The FlagOS interface reports aggregate usage across both block pools.
    constexpr auto aggregate =
        static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
    result.allocated_bytes[aggregate].current = stats.bytes_allocated;
    result.allocated_bytes[aggregate].peak = stats.peak_allocated;
    result.reserved_bytes[aggregate].current = stats.bytes_reserved;
    result.reserved_bytes[aggregate].peak = stats.peak_reserved;
    result.allocation[aggregate].current =
        stats.num_alloc_calls - std::min(stats.num_alloc_calls, stats.num_free_calls);
    result.allocation[aggregate].allocated = stats.num_alloc_calls - since.num_alloc_calls;
    result.allocation[aggregate].freed = stats.num_free_calls - since.num_free_calls;
    result.num_device_alloc = stats.num_device_malloc - since.num_device_malloc;
    result.num_device_free = stats.num_device_free - since.num_device_free;
    result.num_alloc_retries = stats.num_alloc_retries - since.num_alloc_retries;
    return result;
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    if (CachingDeviceAllocator::is_enabled()) {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      stats_baselines_[device] = GetCachingAllocator()->get_stats(device);
    }
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    if (CachingDeviceAllocator::is_enabled()) {
      GetCachingAllocator()->reset_peak_stats(device);
    }
  }

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override {
    return GetCachingAllocator()->get_memory_info(device);
  }
#endif

 private:
  std::atomic<bool> initialized_{false};
  std::mutex stats_mutex_;
  std::unordered_map<c10::DeviceIndex, AllocatorStats> stats_baselines_;
};

static AllocatorProxy global_alloc_proxy;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_alloc_proxy);

} // namespace c10::flagos
