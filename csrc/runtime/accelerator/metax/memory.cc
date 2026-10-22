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
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
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
    if (!ptr || size == 0) {
      return ErrorUnknown;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    void* mem = nullptr;
    int current_device = -1;

    if (type == MemoryType::MemoryTypeDevice) {
      GetDevice(&current_device);
      const cudaError_t set_err = cudaSetDevice(current_device);
      if (set_err != cudaSuccess) {
        fprintf(
            stderr,
            "[flagos-metax] cudaSetDevice(%d) failed: %s\n",
            current_device,
            cudaGetErrorString(set_err));
        return ErrorMemoryAllocation;
      }
      const cudaError_t err = cudaMalloc(&mem, size);
      if (err != cudaSuccess || mem == nullptr) {
        fprintf(
            stderr,
            "[flagos-metax] cudaMalloc(%zu) on device %d failed: %s\n",
            size,
            current_device,
            cudaGetErrorString(err));
        return ErrorMemoryAllocation;
      }
    } else {
      const cudaError_t err = cudaMallocHost(&mem, size);
      if (err != cudaSuccess || mem == nullptr) {
        return ErrorMemoryAllocation;
      }
    }

    m_registry[mem] = {type, current_device, mem, size};
    *ptr = mem;
    return Success;
  }

  Error_t free(void* ptr) {
    if (!ptr) {
      return Success;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_registry.find(ptr);
    if (it == m_registry.end()) {
      return ErrorUnknown;
    }

    const auto& info = it->second;
    cudaError_t err;
    if (info.type == MemoryType::MemoryTypeDevice) {
      err = cudaFree(info.pointer);
    } else {
      err = cudaFreeHost(info.pointer);
    }

    m_registry.erase(it);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
    if (!dst || !src || count == 0) {
      return ErrorUnknown;
    }

    cudaMemcpyKind cuda_kind;
    switch (kind) {
      case MemcpyHostToHost:
        cuda_kind = cudaMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
      case MemcpyDeviceToHost:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
      case MemcpyDeviceToDevice:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
      default:
        return ErrorUnknown;
    }

    const cudaError_t err = cudaMemcpy(dst, src, count, cuda_kind);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind,
      Stream_t stream) {
    if (!dst || !src || count == 0) {
      return ErrorUnknown;
    }

    cudaMemcpyKind cuda_kind;
    switch (kind) {
      case MemcpyHostToHost:
        cuda_kind = cudaMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
      case MemcpyDeviceToHost:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
      case MemcpyDeviceToDevice:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
      default:
        return ErrorUnknown;
    }

    const cudaError_t err = cudaMemcpyAsync(
        dst, src, count, cuda_kind, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? Success : ErrorUnknown;
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
      attributes->pointer = info->pointer;
    }

    return Success;
  }

  Error_t memset(void* devPtr, int value, size_t count) {
    const cudaError_t err = cudaMemset(devPtr, value, count);
    return (err == cudaSuccess) ? Success : ErrorUnknown;
  }

  Error_t memsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
    const cudaError_t err =
        cudaMemsetAsync(devPtr, value, count, reinterpret_cast<cudaStream_t>(stream));
    return (err == cudaSuccess) ? Success : ErrorUnknown;
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
  return MemoryManager::getInstance().memcpy(dst, src, count, kind);
}

Error_t MemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    MemcpyKind kind,
    Stream_t stream) {
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
