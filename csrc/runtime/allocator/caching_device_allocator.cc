// Copyright (c) 2026, BAAI. All rights reserved.

#include "caching_device_allocator.h"

#if defined(USE_ASCEND)
#include "backends/ascend_memory.h"
#endif
#if defined(USE_DCU)
#include "backends/dcu_memory.h"
#endif
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_DCU) && \
    !defined(USE_GCU) && !defined(USE_MUSA) && !defined(USE_BPU)
#include "backends/cuda_memory.h"
#endif
#if defined(USE_TSINGMICRO)
#include "backends/tsingmicro_memory.h"
#endif
#if defined(USE_GCU)
#include "backends/gcu_memory.h"
#endif
#if defined(USE_MUSA)
#include "backends/musa_memory.h"
#endif
#if defined(USE_BPU)
#include "backends/bpu_memory.h"
#endif

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdlib>

namespace c10::flagos {

// Static singleton pointer for the deleter callback.
CachingDeviceAllocator* CachingDeviceAllocator::instance_ = nullptr;

// Check env var to determine if caching is enabled.
bool CachingDeviceAllocator::is_enabled() {
  static bool enabled = []() {
    const char* env = std::getenv("FLAGOS_USE_CACHING_ALLOCATOR");
    if (env && std::string(env) == "0") {
      return false;
    }
    return true;
  }();
  return enabled;
}

CachingDeviceAllocator::CachingDeviceAllocator(
    std::unique_ptr<DeviceMemoryInterface> backend)
    : backend_(std::move(backend)) {
  instance_ = this;
}

CachingDeviceAllocator::~CachingDeviceAllocator() {
#if !defined(USE_TSINGMICRO) && !defined(USE_BPU) && !defined(USE_GCU)
  // Release all cached memory on destruction.
  // On TsingMicro, skip this — the TX runtime may already be shut down
  // at process exit, causing segfaults in txFree.
  // Same on BPU: this allocator is a function-local static, so its destructor
  // runs from __run_exit_handlers, by which point libhbucp's own FINI_ARRAY
  // teardown may already have released the heap the UCP blocks live in, and
  // hbUCPFree aborts with "double free or corruption (fasttop)". Calling
  // torch_fl._C._empty_cache() before exit frees the same blocks cleanly, which
  // confirms the free path itself is fine and only the exit ordering is not.
  // Leaking at process exit is harmless: the kernel reclaims the ION/UCP
  // carveout when the fd closes.
  //
  // GCU has the same exit-ordering problem, and it only shows up for blocks on
  // a non-zero device: `torch.empty(8, device='flagos:1')` then exit aborts in
  // topsFree with "corrupted double-linked list", while the identical call on
  // flagos:0 exits cleanly. MALLOC_PERTURB_=42 makes it deterministic, which is
  // why CI (glibc, no tcache reuse to mask it) hit it while a bare local run
  // did not. `torch_fl._C._empty_cache()` before exit is clean here too, so as
  // above this is teardown ordering rather than a bad free.
  for (auto& state_ptr : device_states_) {
    if (state_ptr) {
      release_cached_blocks(*state_ptr);
    }
  }
#endif
  instance_ = nullptr;
}

CachingDeviceAllocator::DeviceState& CachingDeviceAllocator::get_device_state(
    int device) {
  std::lock_guard<std::recursive_mutex> lock(device_states_mutex_);
  if (device >= static_cast<int>(device_states_.size())) {
    device_states_.resize(device + 1);
  }
  if (!device_states_[device]) {
    device_states_[device] = std::make_unique<DeviceState>();
  }
  return *device_states_[device];
}

at::DataPtr CachingDeviceAllocator::allocate(size_t nbytes) {
  int device = -1;
  backend_->get_device_index(&device);

  if (nbytes == 0) {
    // No memory to hand out, but the DataPtr's device still has to be the
    // current one: it is what the resulting tensor reports as its device, and
    // the composite factories (arange, eye, ...) build their output by
    // `at::empty({0}, options)` and then filling it with an `_out` variant.
    // Pinning index 0 here made `torch.arange(..., device="flagos:1")` come
    // back as flagos:0. `empty_memory_format` has already installed a
    // DeviceGuard for the requested device, so the current index is right.
    //
    // A zero-byte request must not need a live runtime, so a failed query
    // degrades to index 0 rather than raising.
    return {nullptr, nullptr, &block_deleter,
            c10::Device(c10::DeviceType::PrivateUse1,
                        static_cast<c10::DeviceIndex>(device < 0 ? 0 : device))};
  }

  TORCH_CHECK(device >= 0, "CachingDeviceAllocator: invalid device index");

  // Delegation path: the backend ships its own caching allocator (e.g. CUDA).
  // Route the allocation through it so flagos `empty` and boxed-kernel outputs
  // share one pool. We skip the built-in block pool entirely.
  if (backend_->provides_caching()) {
    void* ptr = backend_->caching_alloc(nbytes, /*stream=*/nullptr);
    TORCH_CHECK(
        ptr != nullptr,
        "CachingDeviceAllocator (delegated): failed to allocate ",
        nbytes,
        " bytes on device ",
        device);
    return {ptr, ptr, &delegated_deleter,
            c10::Device(c10::DeviceType::PrivateUse1,
                        static_cast<c10::DeviceIndex>(device))};
  }

  // Use nullptr as stream for default stream allocations.
  // In a more complete implementation, we would get the current stream.
  Stream_t stream = nullptr;

  Block* block = alloc_block(nbytes, stream, device);
  TORCH_CHECK(
      block != nullptr,
      "CachingDeviceAllocator: failed to allocate ",
      nbytes,
      " bytes on device ",
      device);

  auto curr_device =
      c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(device));
  // Stash the Block* as the DataPtr context so the deleter can recover it in
  // O(1) with no side map / lock. The data pointer and context differ (data =
  // device memory, context = Block metadata), which DataPtr supports directly.
  return {block->ptr, block, &block_deleter, curr_device};
}

at::DeleterFnPtr CachingDeviceAllocator::raw_deleter() const {
  return &block_deleter;
}

void CachingDeviceAllocator::copy_data(
    void* dest,
    const void* src,
    std::size_t count) const {
  backend_->memcpy(dest, src, count, MemcpyDeviceToDevice);
}

// --- Core allocation logic ---

Block* CachingDeviceAllocator::alloc_block(
    size_t size,
    Stream_t stream,
    int device) {
  size_t alloc_size = round_size(size);
  auto& state = get_device_state(device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);

  // Process any completed events first to reclaim blocks.
  process_events(state);

  // Select pool based on size.
  BlockPool& pool =
      (alloc_size <= kSmallSize) ? state.small_blocks : state.large_blocks;

  // Try to find a free block in the pool.
  Block* block = find_free_block(alloc_size, stream, pool, device);

  if (!block) {
    // No suitable cached block found, allocate from device.
    // First attempt:
    if (!alloc_from_device(alloc_size, stream, device, &block)) {
      // OOM - try to free cached blocks and retry.
      process_events(state);
      release_cached_blocks(state);
      state.stats.num_alloc_retries++;

      if (!alloc_from_device(alloc_size, stream, device, &block)) {
        // Still OOM after releasing cache.
        return nullptr;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(block != nullptr && block->ptr != nullptr);

  // Split block if significantly larger than needed.
  try_split_block(block, alloc_size);

  block->allocated = true;
  block->requested_size = size;
  state.active_blocks.insert(block);

  // Update stats.
  state.stats.bytes_allocated += block->size;
  state.stats.peak_allocated =
      std::max(state.stats.peak_allocated, state.stats.bytes_allocated);
  state.stats.num_alloc_calls++;

  return block;
}

void CachingDeviceAllocator::free_block(Block* block) {
  auto& state = get_device_state(block->device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);

  block->allocated = false;
  state.active_blocks.erase(block);

  // Update stats.
  state.stats.bytes_allocated -= block->size;
  state.stats.num_free_calls++;

  // If there are outstanding events on other streams, defer the free.
  if (block->event_count > 0) {
    // Block will be returned to pool when events complete.
    return;
  }

  // Try to merge with adjacent free blocks.
  Block* merged = try_merge_blocks(block, state);

  // Return block to its original pool.
  // Use the pool pointer stored on the block (set at allocation time)
  // rather than re-classifying by size. This ensures that a small-pool
  // block that grows after merging remains findable from the small pool.
  TORCH_INTERNAL_ASSERT(merged->pool != nullptr);
  merged->pool->blocks.insert(merged);
}

Block* CachingDeviceAllocator::find_free_block(
    size_t size,
    Stream_t stream,
    BlockPool& pool,
    int device) {
  // Create a search key.
  Block search_key(device, stream, size);
  auto it = pool.blocks.lower_bound(&search_key);

  if (it == pool.blocks.end()) {
    return nullptr;
  }

  Block* block = *it;
  // Must be on the same stream (or we'd need synchronization).
  if (block->stream != stream) {
    return nullptr;
  }

  pool.blocks.erase(it);
  return block;
}

bool CachingDeviceAllocator::alloc_from_device(
    size_t size,
    Stream_t stream,
    int device,
    Block** out) {
  size_t alloc_size = get_allocation_size(size);

  // Ensure device is set correctly.
  backend_->set_device(device);

  void* ptr = nullptr;
  Error_t err = backend_->device_malloc(&ptr, alloc_size);
  if (err != Success || ptr == nullptr) {
    *out = nullptr;
    return false;
  }

  auto& state = get_device_state(device);
  state.stats.num_device_malloc++;
  state.stats.bytes_reserved += alloc_size;
  state.stats.peak_reserved =
      std::max(state.stats.peak_reserved, state.stats.bytes_reserved);

  BlockPool& pool =
      (size <= kSmallSize) ? state.small_blocks : state.large_blocks;
  Block* block = new Block(device, stream, alloc_size, &pool, ptr);
  *out = block;
  return true;
}

void CachingDeviceAllocator::try_split_block(Block* block, size_t size) {
  if (block->size <= size) {
    return;
  }

  size_t remaining = block->size - size;
  if (remaining < kMinBlockSize) {
    return;
  }

  // Create a new block for the remainder.
  Block* remaining_block = new Block(
      block->device,
      block->stream,
      remaining,
      block->pool,
      static_cast<char*>(block->ptr) + size);

  remaining_block->prev = block;
  remaining_block->next = block->next;
  if (block->next) {
    block->next->prev = remaining_block;
  }
  block->next = remaining_block;
  block->size = size;

  // Insert remainder into the pool.
  block->pool->blocks.insert(remaining_block);
}

Block* CachingDeviceAllocator::try_merge_blocks(
    Block* block,
    DeviceState& state) {
  // Merge with next block if it's free.
  if (block->next && !block->next->allocated &&
      block->next->event_count == 0) {
    Block* next = block->next;

    // Remove next from its pool (use pool pointer, not size-based lookup).
    TORCH_INTERNAL_ASSERT(next->pool != nullptr);
    next->pool->blocks.erase(next);

    block->size += next->size;
    block->next = next->next;
    if (next->next) {
      next->next->prev = block;
    }
    delete next;
  }

  // Merge with prev block if it's free.
  if (block->prev && !block->prev->allocated &&
      block->prev->event_count == 0) {
    Block* prev = block->prev;

    // Remove prev from its pool (use pool pointer, not size-based lookup).
    TORCH_INTERNAL_ASSERT(prev->pool != nullptr);
    prev->pool->blocks.erase(prev);

    prev->size += block->size;
    prev->next = block->next;
    if (block->next) {
      block->next->prev = prev;
    }

    // block is now absorbed into prev.
    delete block;
    return prev;
  }

  return block;
}

bool CachingDeviceAllocator::release_cached_blocks(DeviceState& state) {
  bool released = false;

  auto release_pool = [&](BlockPool& pool) {
    std::vector<Block*> to_release;
    std::vector<Block*> to_keep;

    for (auto* block : pool.blocks) {
      if (!block->is_split()) {
        to_release.push_back(block);
      } else {
        to_keep.push_back(block);
      }
    }
    pool.blocks.clear();

    for (auto* block : to_keep) {
      pool.blocks.insert(block);
    }

    for (auto* block : to_release) {
      backend_->device_free(block->ptr);
      state.stats.bytes_reserved -= block->size;
      state.stats.num_device_free++;
      delete block;
      released = true;
    }
  };

  release_pool(state.small_blocks);
  release_pool(state.large_blocks);

  return released;
}

void CachingDeviceAllocator::process_events(DeviceState& state) {
  while (!state.outstanding_events.empty()) {
    auto& [event, block] = state.outstanding_events.front();

    Error_t status = backend_->event_query(event);
    if (status == ErrorNotReady) {
      // Events are ordered; if this one isn't ready, none after it are.
      break;
    }

    // Event completed - destroy it and decrement block's event count.
    backend_->event_destroy(event);
    block->event_count--;

    if (block->event_count == 0 && !block->allocated) {
      // Block is fully released from all streams, return to pool.
      Block* merged = try_merge_blocks(block, state);
      TORCH_INTERNAL_ASSERT(merged->pool != nullptr);
      merged->pool->blocks.insert(merged);
    }

    state.outstanding_events.pop_front();
  }
}

// --- Public API ---

void CachingDeviceAllocator::empty_cache() {
  if (backend_->provides_caching()) {
    backend_->caching_empty_cache();
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(device_states_mutex_);
  for (auto& state_ptr : device_states_) {
    if (state_ptr) {
      std::lock_guard<std::recursive_mutex> dev_lock(state_ptr->mutex);
      process_events(*state_ptr);
      release_cached_blocks(*state_ptr);
    }
  }
}

void CachingDeviceAllocator::record_stream(
    const at::DataPtr& ptr,
    Stream_t stream) {
  if (!ptr.get()) {
    return;
  }

  if (backend_->provides_caching()) {
    backend_->caching_record_stream(ptr.get(), stream);
    return;
  }

  // The Block* is stored as the DataPtr context by allocate(). Only trust it
  // when the deleter matches (i.e. this DataPtr came from our block pool, not
  // the delegation path or a foreign allocator).
  if (ptr.get_deleter() != &block_deleter) {
    return;
  }
  Block* block = static_cast<Block*>(ptr.get_context());
  if (!block) {
    return;
  }

  auto& state = get_device_state(block->device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);

  // If the block is already on this stream, nothing to do.
  if (block->stream == stream) {
    return;
  }

  // Record an event on the given stream. When the event completes,
  // we know the stream is done using this block.
  Event_t event = nullptr;
  Error_t err = backend_->event_create(&event);
  TORCH_CHECK(err == Success, "Failed to create event for record_stream");

  err = backend_->event_record(event, stream);
  TORCH_CHECK(err == Success, "Failed to record event for record_stream");

  block->event_count++;
  state.outstanding_events.push_back({event, block});
}

AllocatorStats CachingDeviceAllocator::get_stats(int device) {
  if (backend_->provides_caching()) {
    AllocatorStats stats;
    if (backend_->caching_get_stats(device, &stats)) {
      return stats;
    }
    return stats;  // empty stats if backend reports unsupported
  }
  auto& state = get_device_state(device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);
  return state.stats;
}

void CachingDeviceAllocator::reset_stats(int device) {
  if (backend_->provides_caching()) {
    backend_->caching_reset_peak_stats(device);
    return;
  }
  auto& state = get_device_state(device);
  std::lock_guard<std::recursive_mutex> lock(state.mutex);
  state.stats = AllocatorStats{};
}

// Static deleter invoked by DataPtr when a tensor is freed. The context is the
// Block* stashed by allocate(), so freeing is O(1) with no map lookup or lock.
void CachingDeviceAllocator::block_deleter(void* ctx) {
  if (!ctx || !instance_) {
    return;
  }
  instance_->free_block(static_cast<Block*>(ctx));
}

// Deleter for the delegation path: free straight back to the backend's caching
// allocator (no flagos block pool involved).
void CachingDeviceAllocator::delegated_deleter(void* ptr) {
  if (!ptr || !instance_) {
    return;
  }
  instance_->backend_->caching_free(ptr);
}

// --- Global accessor ---

CachingDeviceAllocator* GetCachingAllocator() {
  static std::unique_ptr<CachingDeviceAllocator> alloc;
  static std::once_flag flag;
  std::call_once(flag, []() {
    // Select backend based on build configuration.
#if defined(USE_ASCEND)
    auto backend = std::make_unique<AscendDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#elif defined(USE_TSINGMICRO)
    auto backend = std::make_unique<TsingMicroDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#elif defined(USE_DCU)
    // DCU: DTK's CUDA compatibility layer supplies the runtime API, and caching
    // delegates to torch's own allocator via the device-generic registry --
    // c10::hip under the hood, no c10::cuda symbols (see backends/dcu_memory.h).
    auto backend = std::make_unique<DcuDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#elif defined(USE_GCU)
    auto backend = std::make_unique<GcuDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#elif defined(USE_MUSA)
    // MUSA: raw musa* runtime with flagos's own caching on top. The vendor
    // MUSACachingAllocator claims the same PrivateUse1 allocator slot flagos
    // registers, so it is deliberately not delegated to (see musa_memory.h).
    auto backend = std::make_unique<MusaDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#elif defined(USE_BPU)
    // BPU: memory comes from the UCP allocator via the accelerator/bpu
    // contract functions, which also own the virtual->physical map the
    // zero-copy inference path needs.
    auto backend = std::make_unique<BPUDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#else
    // CUDA (and Metax, which uses CUDA-compatible API)
    auto backend = std::make_unique<CudaDeviceMemory>();
    alloc = std::make_unique<CachingDeviceAllocator>(std::move(backend));
#endif
  });
  return alloc.get();
}

} // namespace c10::flagos
