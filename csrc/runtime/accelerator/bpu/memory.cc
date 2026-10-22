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

// D-Robotics RDK BPU (Horizon "nash-p" BPU) memory layer.
//
// The UCP allocator returns memory that is mapped into the calling process as
// well as addressable by the BPU: hbUCPSysMem carries both a host `virAddr` and
// a `phyAddr`. That is what makes a real device allocator possible here --
// device pointers are host-dereferenceable, so Memcpy is a plain memcpy and no
// bounce buffer is needed.
//
// `MallocCached` is used rather than `hbUCPMalloc` because the host writes
// tensor data through the mapping; the cache has to be flushed before the BPU
// reads it (see the Memcpy paths below), which is cheaper than uncached host
// stores.

#include <flagos.h>

extern "C" {
#include <hb_ucp_sys.h>
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace {

struct Block {
  MemoryType type = MemoryType::MemoryTypeUnmanaged;
  int device = -1;
  hbUCPSysMem mem{};  // phyAddr/virAddr/memSize as returned by the UCP allocator
  size_t size = 0;
};

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  Error_t allocate(void** ptr, size_t size, MemoryType type) {
    if (!ptr || size == 0) {
      return ErrorUnknown;
    }

    int device = 0;
    if (type == MemoryType::MemoryTypeDevice) {
      GetDevice(&device);
    }

    hbUCPSysMem mem{};
    // Host pinned memory has no separate UCP API; the device mapping is already
    // host-visible, so both kinds come from the same allocator. The distinction
    // is kept in the registry so PointerGetAttributes stays truthful.
    int32_t rc = hbUCPMallocCached(&mem, static_cast<uint64_t>(size), device);
    if (rc != 0 || mem.virAddr == nullptr) {
      fprintf(
          stderr,
          "[flagos-bpu] hbUCPMallocCached(%zu bytes, device %d) failed: rc=%d\n",
          size,
          device,
          static_cast<int>(rc));
      *ptr = nullptr;
      return ErrorMemoryAllocation;
    }

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_registry[mem.virAddr] = Block{type, device, mem, size};
    }
    *ptr = mem.virAddr;
    return Success;
  }

  Error_t free(void* ptr) {
    if (!ptr) {
      return Success;
    }

    hbUCPSysMem mem{};
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_registry.find(ptr);
      if (it == m_registry.end()) {
        return ErrorUnknown;
      }
      mem = it->second.mem;
      m_registry.erase(it);
    }

    return (hbUCPFree(&mem) == 0) ? Success : ErrorUnknown;
  }

  // Flush the CPU cache for whichever managed block contains [ptr, ptr+count).
  // Cache maintenance is per-allocation in the UCP API, so a partial write
  // still flushes its whole block; that is correct, just conservative.
  Error_t flush(const void* ptr, int flag) {
    std::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);
    if (!info) {
      // Not ours (e.g. a plain host buffer) -- nothing to maintain.
      return Success;
    }
    return (hbUCPMemFlush(&info->mem, flag) == 0) ? Success : ErrorUnknown;
  }

  Error_t memcpy_(void* dst, const void* src, size_t count, MemcpyKind kind) {
    if (!dst || !src || count == 0) {
      return ErrorUnknown;
    }

    // Device memory is host-mapped, so every direction is a host memcpy. What
    // differs is the cache maintenance around it: invalidate before reading
    // what the BPU wrote, clean after writing what the BPU will read.
    switch (kind) {
      case MemcpyDeviceToHost:
      case MemcpyDeviceToDevice:
        flush(src, HB_SYS_MEM_CACHE_INVALIDATE);
        break;
      default:
        break;
    }

    std::memcpy(dst, src, count);

    switch (kind) {
      case MemcpyHostToDevice:
      case MemcpyDeviceToDevice:
        flush(dst, HB_SYS_MEM_CACHE_CLEAN);
        break;
      default:
        break;
    }
    return Success;
  }

  Error_t getPointerAttributes(PointerAttributes* attributes, const void* ptr) {
    if (!attributes || !ptr) {
      return ErrorUnknown;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);
    if (!info) {
      attributes->type = MemoryType::MemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = info->mem.virAddr;
    }
    return Success;
  }

  Error_t memset_(void* devPtr, int value, size_t count) {
    if (!devPtr || count == 0) {
      return ErrorUnknown;
    }
    std::memset(devPtr, value, count);
    flush(devPtr, HB_SYS_MEM_CACHE_CLEAN);
    return Success;
  }

  // Physical address of a managed pointer, for building hbDNNTensor without a
  // copy. Returns 0 when the pointer is not device memory we allocated.
  uint64_t physicalAddress(const void* ptr) {
    std::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);
    if (!info) {
      return 0;
    }
    const auto offset = static_cast<const char*>(ptr) -
        static_cast<const char*>(info->mem.virAddr);
    return info->mem.phyAddr + static_cast<uint64_t>(offset);
  }

 private:
  MemoryManager() = default;

  Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it != m_registry.begin()) {
      --it;
      const char* p = static_cast<const char*>(ptr);
      const char* base = static_cast<const char*>(it->first);
      if (p >= base && p < (base + it->second.size)) {
        return &it->second;
      }
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
  return MemoryManager::getInstance().free(devPtr);
}

Error_t MallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(
      hostPtr, size, MemoryType::MemoryTypeHost);
}

Error_t FreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free(hostPtr);
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
  // The BPU submission path is synchronous (hbUCPSubmitTask + WaitTaskDone), so
  // there is no asynchronous copy engine to target. A synchronous copy is a
  // valid implementation of the async contract.
  return MemoryManager::getInstance().memcpy_(dst, src, count, kind);
}

Error_t PointerGetAttributes(PointerAttributes* attributes, const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
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

// Non-contract helper used by the Python runtime to hand a tensor's storage
// straight to hbDNNInferV2. Declared in bpu.h.
extern "C" uint64_t FlagosBPUPhysicalAddress(const void* ptr) {
  return MemoryManager::getInstance().physicalAddress(ptr);
}
