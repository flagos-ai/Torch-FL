// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstddef>
#include <set>

#include <flagos.h>

namespace c10::flagos {

struct BlockPool;

// A memory block managed by the caching allocator.
// Blocks form a doubly-linked list within a segment (contiguous device
// allocation). When freed, blocks are returned to their BlockPool for reuse.
struct Block {
  int device;            // device index
  Stream_t stream;       // stream on which the block was allocated
  size_t size;           // actual block size (may be larger than requested)
  size_t requested_size; // size originally requested by user
  BlockPool* pool;       // owning pool (small or large)
  void* ptr;             // device memory pointer
  bool allocated;        // whether this block is currently in use

  Block* prev;           // previous block in segment (for merging)
  Block* next;           // next block in segment (for merging)

  int event_count;       // number of outstanding events (stream safety)

  Block(int device, Stream_t stream, size_t size, BlockPool* pool, void* ptr)
      : device(device),
        stream(stream),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr),
        allocated(false),
        prev(nullptr),
        next(nullptr),
        event_count(0) {}

  // Constructor for search key (used in set lookups)
  Block(int device, Stream_t stream, size_t size)
      : device(device),
        stream(stream),
        size(size),
        requested_size(0),
        pool(nullptr),
        ptr(nullptr),
        allocated(false),
        prev(nullptr),
        next(nullptr),
        event_count(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

// Comparator for ordering blocks in a BlockPool.
// Sorts by: (stream, size, ptr) to enable best-fit search per stream.
struct BlockComparator {
  bool operator()(const Block* a, const Block* b) const {
    if (a->stream != b->stream)
      return reinterpret_cast<uintptr_t>(a->stream) <
             reinterpret_cast<uintptr_t>(b->stream);
    if (a->size != b->size)
      return a->size < b->size;
    return reinterpret_cast<uintptr_t>(a->ptr) <
           reinterpret_cast<uintptr_t>(b->ptr);
  }
};

// A pool of free blocks, ordered for efficient best-fit lookup.
struct BlockPool {
  std::set<Block*, BlockComparator> blocks;
  bool is_small; // true for small pool (<=1MB), false for large pool

  explicit BlockPool(bool small) : is_small(small) {}
};

// Allocation strategy constants (matching PyTorch defaults)
constexpr size_t kMinBlockSize = 512;           // all allocations rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;          // 1 MB - threshold between small/large pools
constexpr size_t kSmallBuffer = 2097152;        // 2 MB - small pool segment size
constexpr size_t kLargeBuffer = 20971520;       // 20 MB - pre-allocation for 1-10MB requests
constexpr size_t kMinLargeAlloc = 10485760;     // 10 MB - round up large allocs to 2MB multiples
constexpr size_t kRoundLarge = 2097152;         // 2 MB - rounding for large allocations

// Round up size to the nearest allocation granularity
inline size_t round_size(size_t size) {
  if (size < kMinBlockSize) {
    return kMinBlockSize;
  }
  if (size <= kSmallSize) {
    // Round up to nearest 512 bytes
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
  }
  // Round up to nearest 2MB
  return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}

// Determine how much to allocate from the device for a given request
inline size_t get_allocation_size(size_t size) {
  if (size <= kSmallSize) {
    return kSmallBuffer;  // always allocate 2MB segments for small pool
  }
  if (size < kMinLargeAlloc) {
    return kLargeBuffer;  // allocate 20MB for medium requests
  }
  // For large requests, round up to 2MB boundary
  return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}

} // namespace c10::flagos
