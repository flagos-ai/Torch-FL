// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <macros.h>

#include "allocator_stats.h"
#include "block.h"
#include "device_memory_interface.h"

#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace c10::flagos {

// AllocatorStats is defined in allocator_stats.h (shared with
// DeviceMemoryInterface's caching-delegation API without a circular include).

// The caching device allocator for PrivateUse1 devices.
// Maintains per-device free block pools and reuses memory to avoid
// repeated expensive device malloc/free calls.
class FLAGOS_EXPORT CachingDeviceAllocator final : public at::Allocator {
 public:
  explicit CachingDeviceAllocator(
      std::unique_ptr<DeviceMemoryInterface> backend);
  ~CachingDeviceAllocator() override;

  // at::Allocator interface
  at::DataPtr allocate(size_t nbytes) override;
  at::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  // --- Caching-specific public API ---

  // Release all unoccupied cached memory back to the device.
  void empty_cache();

  // Record that a block's memory is used on the given stream.
  // This ensures the block won't be reused until the stream completes.
  void record_stream(const at::DataPtr& ptr, Stream_t stream);

  // Get statistics for a device.
  AllocatorStats get_stats(int device);

  // Reset the peak watermarks for a device, keeping the live totals: the next
  // peak read is a maximum over what is live from here on, not an empty-struct
  // count. Matches the platform allocator's resetPeakStats, which is what the
  // delegating path uses, and torch.cuda.reset_peak_memory_stats.
  void reset_peak_stats(int device);

  // Whether caching is enabled (controlled by env var).
  static bool is_enabled();

 private:
  // Per-device allocator state, each with its own lock.
  struct DeviceState {
    std::recursive_mutex mutex;
    BlockPool large_blocks{false};
    BlockPool small_blocks{true};
    std::unordered_set<Block*> active_blocks;
    // Events waiting to be processed: (event, block)
    std::deque<std::pair<Event_t, Block*>> outstanding_events;
    AllocatorStats stats;

    DeviceState() = default;
  };

  // Allocate a block from the cache or device.
  Block* alloc_block(size_t size, Stream_t stream, int device);

  // Free a block back to the cache.
  void free_block(Block* block);

  // Try to find a suitable cached block.
  Block* find_free_block(
      size_t size,
      Stream_t stream,
      BlockPool& pool,
      int device);

  // Allocate a new segment from the device and create a block.
  bool alloc_from_device(size_t size, Stream_t stream, int device, Block** out);

  // Try to split a block if it's significantly larger than needed.
  void try_split_block(Block* block, size_t size);

  // Merge a freed block with adjacent free blocks. Returns the surviving block.
  Block* try_merge_blocks(Block* block, DeviceState& state);

  // Release all cached blocks for a device (used during OOM retry).
  bool release_cached_blocks(DeviceState& state);

  // Process completed events and return blocks to the pool.
  void process_events(DeviceState& state);

  // Static deleter function for DataPtr. Receives the Block* directly as the
  // DataPtr context (set at allocation), so no ptr->block lookup is needed.
  static void block_deleter(void* ctx);

  // Static deleter for the delegation path (frees via backend caching allocator).
  static void delegated_deleter(void* ptr);

  // Get or create per-device state.
  DeviceState& get_device_state(int device);

  std::unique_ptr<DeviceMemoryInterface> backend_;
  std::vector<std::unique_ptr<DeviceState>> device_states_;
  std::recursive_mutex device_states_mutex_;

  // Global pointer to the singleton, used by the static deleter.
  static CachingDeviceAllocator* instance_;
};

// Get the global caching allocator instance (creates on first call).
FLAGOS_EXPORT CachingDeviceAllocator* GetCachingAllocator();

} // namespace c10::flagos
