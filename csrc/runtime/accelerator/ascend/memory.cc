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

#include <flagos.h>
#include <acl/acl_rt.h>
#include "acl_stream.h"

#include <cstdio>
#include <map>
#include <mutex>

namespace {

struct Block {
  MemoryType type = MemoryType::MemoryTypeUnmanaged;
  int device = -1;
  void* pointer = nullptr;
  size_t size = 0;
};

class MemoryManager {
 public:
  static MemoryManager& getInstance() {
    static MemoryManager instance;
    return instance;
  }

  Error_t allocate(void** ptr, size_t size, MemoryType type) {
    if (!ptr || size == 0)
      return ErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    void* mem = nullptr;
    int current_device = -1;

    if (type == MemoryType::MemoryTypeDevice) {
      GetDevice(&current_device);

      aclError set_err = aclrtSetDevice(current_device);
      if (set_err != ACL_SUCCESS) {
        fprintf(stderr, "[flagos-ascend] aclrtSetDevice(%d) failed: %d\n",
                current_device, static_cast<int>(set_err));
        return ErrorMemoryAllocation;
      }

      aclError err = aclrtMalloc(&mem, size, ACL_MEM_MALLOC_HUGE_FIRST);
      if (err != ACL_SUCCESS || mem == nullptr) {
        fprintf(stderr, "[flagos-ascend] aclrtMalloc(%zu bytes) on device %d failed: %d\n",
                size, current_device, static_cast<int>(err));
        return ErrorMemoryAllocation;
      }
    } else {
      aclError err = aclrtMallocHost(&mem, size);
      if (err != ACL_SUCCESS || mem == nullptr)
        return ErrorMemoryAllocation;
    }

    m_registry[mem] = {type, current_device, mem, size};
    *ptr = mem;
    return Success;
  }

  Error_t free(void* ptr) {
    if (!ptr)
      return Success;

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_registry.find(ptr);
    if (it == m_registry.end())
      return ErrorUnknown;

    const auto& info = it->second;
    aclError err;
    if (info.type == MemoryType::MemoryTypeDevice) {
      err = aclrtFree(info.pointer);
    } else {
      err = aclrtFreeHost(info.pointer);
    }

    m_registry.erase(it);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
    if (!dst || !src || count == 0)
      return ErrorUnknown;

    aclrtMemcpyKind acl_kind;
    switch (kind) {
      case MemcpyHostToHost:     acl_kind = ACL_MEMCPY_HOST_TO_HOST; break;
      case MemcpyHostToDevice:   acl_kind = ACL_MEMCPY_HOST_TO_DEVICE; break;
      case MemcpyDeviceToHost:   acl_kind = ACL_MEMCPY_DEVICE_TO_HOST; break;
      case MemcpyDeviceToDevice: acl_kind = ACL_MEMCPY_DEVICE_TO_DEVICE; break;
      default: return ErrorUnknown;
    }

    // aclnn ops enqueue asynchronously on their device's default stream (per-op
    // sync was removed from EXEC_ASCEND_CMD). This blocking aclrtMemcpy runs
    // outside that stream's ordering, so any transfer touching device memory
    // must first drain the default streams: a D2H read would otherwise observe
    // stale data, and an H2D/D2D write could race a pending consumer/producer.
    // The caller does not pass the owning device, so every default stream that
    // has been created is drained (see acl_stream.h). Host-to-host transfers
    // touch no device memory and need no barrier.
    if (kind != MemcpyHostToHost) {
      at::native::flagos::ascend::DrainDefaultAclStreams();
    }

    aclError err = aclrtMemcpy(dst, count, src, count, acl_kind);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t memcpyAsync(void* dst, const void* src, size_t count,
                      MemcpyKind kind, Stream_t stream) {
    if (!dst || !src || count == 0)
      return ErrorUnknown;

    aclrtMemcpyKind acl_kind;
    switch (kind) {
      case MemcpyHostToHost:     acl_kind = ACL_MEMCPY_HOST_TO_HOST; break;
      case MemcpyHostToDevice:   acl_kind = ACL_MEMCPY_HOST_TO_DEVICE; break;
      case MemcpyDeviceToHost:   acl_kind = ACL_MEMCPY_DEVICE_TO_HOST; break;
      case MemcpyDeviceToDevice: acl_kind = ACL_MEMCPY_DEVICE_TO_DEVICE; break;
      default: return ErrorUnknown;
    }

    aclError err = aclrtMemcpyAsync(dst, count, src, count, acl_kind, (aclrtStream)stream);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t getPointerAttributes(PointerAttributes* attributes, const void* ptr) {
    if (!attributes || !ptr)
      return ErrorUnknown;

    std::lock_guard<std::mutex> lock(m_mutex);
    Block* info = getBlockInfoNoLock(ptr);

    if (!info) {
      attributes->type = MemoryType::MemoryTypeUnmanaged;
      attributes->device = -1;
      attributes->pointer = const_cast<void*>(ptr);
    } else {
      attributes->type = info->type;
      attributes->device = info->device;
      attributes->pointer = info->pointer;
    }

    return Success;
  }

  Error_t memset(void* devPtr, int value, size_t count) {
    aclError err = aclrtMemset(devPtr, count, value, count);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t memsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
    aclError err = aclrtMemsetAsync(devPtr, count, value, count, (aclrtStream)stream);
    return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
  }

 private:
  MemoryManager() = default;

  Block* getBlockInfoNoLock(const void* ptr) {
    auto it = m_registry.upper_bound(const_cast<void*>(ptr));
    if (it != m_registry.begin()) {
      --it;
      const char* p_char = static_cast<const char*>(ptr);
      const char* base_char = static_cast<const char*>(it->first);
      if (p_char >= base_char && p_char < (base_char + it->second.size)) {
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
  return MemoryManager::getInstance().allocate(devPtr, size, MemoryType::MemoryTypeDevice);
}

Error_t Free(void* devPtr) {
  return MemoryManager::getInstance().free(devPtr);
}

Error_t MallocHost(void** hostPtr, size_t size) {
  return MemoryManager::getInstance().allocate(hostPtr, size, MemoryType::MemoryTypeHost);
}

Error_t FreeHost(void* hostPtr) {
  return MemoryManager::getInstance().free(hostPtr);
}

Error_t Memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
  return MemoryManager::getInstance().memcpy(dst, src, count, kind);
}

Error_t MemcpyAsync(void* dst, const void* src, size_t count,
                    MemcpyKind kind, Stream_t stream) {
  return MemoryManager::getInstance().memcpyAsync(dst, src, count, kind, stream);
}

Error_t PointerGetAttributes(PointerAttributes* attributes, const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

Error_t Memset(void* devPtr, int value, size_t count) {
  return MemoryManager::getInstance().memset(devPtr, value, count);
}

Error_t MemsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
  return MemoryManager::getInstance().memsetAsync(devPtr, value, count, stream);
}
