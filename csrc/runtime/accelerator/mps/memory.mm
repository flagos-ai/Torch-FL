// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Apple Metal memory layer.
//
// Apple GPUs share memory with the CPU, so a device tensor is backed by an
// MTLBuffer created with MTLResourceStorageModeShared: its `contents` pointer is
// host-dereferenceable and the GPU reads the same bytes, which makes Memcpy a
// plain memcpy and needs no bounce buffer.
//
// That is exactly why Memcpy drains the queue first. Ops (the stage-1 Metal
// kernels) encode work into MTLCommandBuffers committed on a command queue; a
// host read of `contents` before that work finishes sees stale bytes. The drain
// is a no-op when nothing is pending, so paying it on every copy is cheap
// insurance against a nondeterministic, kernel-looking wrong answer.
//
// Which pointers are device memory lives in a side table keyed by base pointer,
// because the flagos ABI has no vendor query that answers it. PointerGetAttributes
// reads that table, and copy_ops dispatches on the result -- a wrong answer here
// shows up later as a wrong `.cpu()`, not as a runtime error.

#include "mps.h"
#include "mps_runtime.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace {

struct Block {
  MemoryType type = MemoryTypeUnmanaged;
  int device = -1;
  // +1 for device allocations, nil for host ones. Held until Free.
  id<MTLBuffer> buffer = nil;
  size_t size = 0;
};

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  Error_t allocate(void** ptr, size_t size, MemoryType type) {
    if (!ptr) {
      return ErrorUnknown;
    }

    int device = 0;
    GetDevice(&device);

    if (type == MemoryType::MemoryTypeHost) {
      // Pinned host memory is a plain malloc on Apple: unified memory removes the
      // page-locking distinction, and the GPU never reads a host buffer directly
      // (kernels bind MTLBuffers). The registry entry is what keeps
      // PointerGetAttributes -- and DDP's isPinnedPtr -- truthful about it.
      void* host = std::malloc(size ? size : 1);
      if (host == nullptr) {
        *ptr = nullptr;
        return ErrorMemoryAllocation;
      }
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        Block block;
        block.type = type;
        block.device = device;
        block.size = size;
        m_registry[host] = block;
      }
      *ptr = host;
      return Success;
    }

    id<MTLDevice> metalDevice = torch_fl_mps::MetalDevice();
    if (metalDevice == nil) {
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    // MTLBuffer requires a non-zero length; a zero-byte request still needs a
    // unique, freeable pointer. One byte is enough to hand back and never read.
    const NSUInteger length = size ? static_cast<NSUInteger>(size) : 1;
    id<MTLBuffer> buffer =
        [metalDevice newBufferWithLength:length
                                 options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    void* base = buffer.contents;
    if (base == nullptr) {
      [buffer release];
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      Block block;
      block.type = MemoryType::MemoryTypeDevice;
      block.device = device;
      block.buffer = buffer;
      block.size = size;
      m_registry[base] = block;
    }
    *ptr = base;
    return Success;
  }

  Error_t free_(void* ptr) {
    if (!ptr) {
      return Success;
    }

    Block block;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_registry.find(ptr);
      if (it == m_registry.end()) {
        return ErrorUnknown;
      }
      block = it->second;
      m_registry.erase(it);
    }

    if (block.buffer != nil) {
      // Python objects may be released in an exit hook after this runtime's
      // own drain. Drain before the final buffer goes away so no queued graph
      // still names it.
      torch_fl_mps::DrainAll();
      [block.buffer release];
    } else {
      std::free(ptr);
    }
    return Success;
  }

  Error_t memcpy_(void* dst, const void* src, size_t count, MemcpyKind /*kind*/) {
    if (count == 0) {
      return Success;
    }
    if (!dst || !src) {
      return ErrorUnknown;
    }
    // Every buffer this runtime hands out is shared, so the copy direction does
    // not change the mechanics -- only the ordering does. Drain first: `src` may
    // be a device buffer a queued kernel is still writing.
    torch_fl_mps::DrainAll();
    std::memcpy(dst, src, count);
    return Success;
  }

  Error_t memset_(void* devPtr, int value, size_t count) {
    if (!devPtr || count == 0) {
      return ErrorUnknown;
    }
    torch_fl_mps::DrainAll();
    std::memset(devPtr, value, count);
    return Success;
  }

  Error_t getPointerAttributes(PointerAttributes* attributes, const void* ptr) {
    if (!attributes || !ptr) {
      return ErrorUnknown;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    const Block* info = getBlockInfoNoLock(ptr);
    if (info == nullptr) {
      attributes->type = MemoryType::MemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = const_cast<void*>(info->buffer != nil
          ? info->buffer.contents
          : ptr);
    }
    return Success;
  }

  // The MTLBuffer backing `ptr` and how far into it `ptr` starts. Returns
  // nullptr for a host allocation, a pointer this runtime never handed out, or
  // one whose allocation has since been freed -- the operator layer treats all
  // three the same way (fall back to a CPU round-trip) rather than guessing.
  //
  // Non-const only because the registry scan shares getBlockInfoNoLock, which
  // reads m_registry under m_mutex; the lookup itself mutates nothing.
  void* bufferForPointer(const void* ptr, size_t* byteOffset) {
    if (!ptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    const Block* info = getBlockInfoNoLock(ptr);
    if (info == nullptr || info->buffer == nil) {
      return nullptr;
    }
    if (byteOffset != nullptr) {
      const char* p = static_cast<const char*>(ptr);
      const char* base = static_cast<const char*>(info->buffer.contents);
      *byteOffset = static_cast<size_t>(p - base);
    }
    return (void*)info->buffer;
  }

 private:
  MemoryManager() = default;

  ~MemoryManager() {
    // The allocation registry outlives the C++ stream module. Its holder is
    // released after that module's own atexit hook, so encode the wait against
    // the queue directly here, without looking up any C++ static state.
    id<MTLCommandQueue> queue =
        (id<MTLCommandQueue>)FlagosMPSDefaultCommandQueue();
    if (queue != nil) {
      id<MTLCommandBuffer> buffer = [queue commandBuffer];
      if (buffer != nil) {
        [buffer commit];
        [buffer waitUntilCompleted];
      }
    }
    for (auto& [ptr, block] : m_registry) {
      if (block.type == MemoryType::MemoryTypeHost) {
        std::free(ptr);
      } else if (block.buffer != nil) {
        [block.buffer release];
      }
    }
  }

  // The block whose allocation contains `ptr`. A pointer may be offset into a
  // live allocation (a tensor view's storage plus an offset), so look at the
  // nearest base below it rather than requiring an exact key.
  const Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it == m_registry.begin()) {
      return nullptr;
    }
    --it;
    const char* p = static_cast<const char*>(ptr);
    const char* base = static_cast<const char*>(it->first);
    if (p >= base && p < base + it->second.size) {
      return &it->second;
    }
    return nullptr;
  }

  std::map<void*, Block> m_registry;
  std::mutex m_mutex;
};

} // namespace

Error_t Malloc(void** devPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      devPtr, size, MemoryType::MemoryTypeDevice);
}

Error_t Free(void* devPtr) {
  return MemoryManager::getInstance().free_(devPtr);
}

Error_t MallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      hostPtr, size, MemoryType::MemoryTypeHost);
}

Error_t FreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free_(hostPtr);
}

Error_t Memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
  return MemoryManager::getInstance().memcpy_(dst, src, count, kind);
}

Error_t MemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    MemcpyKind kind,
    Stream_t /*stream*/) {
  // There is no separate copy engine to hand this to, and a shared buffer needs
  // a host memcpy either way. The synchronous path is a valid implementation of
  // the async contract (the BPU precedent), and it drains first so the copy is
  // ordered behind any queued writer.
  return MemoryManager::getInstance().memcpy_(dst, src, count, kind);
}

Error_t PointerGetAttributes(PointerAttributes* attributes, const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

// Non-contract helper for the operator layer. Declared in mps.h.
extern "C" void* FlagosMPSBufferForPointer(const void* ptr, size_t* byteOffset) {
  return MemoryManager::getInstance().bufferForPointer(ptr, byteOffset);
}

Error_t Memset(void* devPtr, int value, size_t count) {
  return MemoryManager::getInstance().memset_(devPtr, value, count);
}

Error_t MemsetAsync(
    void* devPtr,
    int value,
    size_t count,
    Stream_t /*stream*/) {
  return MemoryManager::getInstance().memset_(devPtr, value, count);
}

// Non-contract helper for the allocator's get_memory_info. Declared in mps.h.
extern "C" void FlagosMPSDeviceMemoryInfo(
    size_t* freeBytes,
    size_t* totalBytes) {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  id<MTLDevice> device = torch_fl_mps::MetalDevice();
  if (device != nil) {
    // recommendedMaxWorkingSetSize is the ceiling Metal suggests staying under,
    // which is the closest thing to "total usable GPU memory"; currentAllocatedSize
    // is what this process has already created on the device. The difference is a
    // conservative free, not a driver-reported one -- Metal offers no exact query.
    total_bytes = static_cast<size_t>(device.recommendedMaxWorkingSetSize);
    const size_t allocated = static_cast<size_t>(device.currentAllocatedSize);
    free_bytes = total_bytes > allocated ? total_bytes - allocated : 0;
  }
  if (freeBytes) {
    *freeBytes = free_bytes;
  }
  if (totalBytes) {
    *totalBytes = total_bytes;
  }
}
