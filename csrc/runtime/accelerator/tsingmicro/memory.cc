#include <flagos.h>
#include <tx_runtime.h>

#include <map>
#include <mutex>
#include <cstring>
#include <cstdio>

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

      // Ensure TX device is set correctly before allocation
      txError_t set_err = txSetDevice(static_cast<uint32_t>(current_device));
      if (set_err != TX_SUCCESS) {
        fprintf(stderr, "[flagos-tsingmicro] txSetDevice(%d) failed: %s\n",
                current_device, txGetErrorString(set_err));
        return ErrorMemoryAllocation;
      }

      txError_t err = txMalloc(&mem, static_cast<uint64_t>(size));
      if (err != TX_SUCCESS || mem == nullptr) {
        fprintf(stderr, "[flagos-tsingmicro] txMalloc(%zu bytes) on device %d failed: %s\n",
                size, current_device, txGetErrorString(err));
        return ErrorMemoryAllocation;
      }
    } else {
      txError_t err = txMallocHost(&mem, static_cast<uint64_t>(size));
      if (err != TX_SUCCESS || mem == nullptr)
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
    txError_t err;
    if (info.type == MemoryType::MemoryTypeDevice) {
      err = txFree(info.pointer);
    } else {
      err = txFreeHost(info.pointer);
    }

    m_registry.erase(it);
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t memcpy(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind) {
    if (!dst || !src || count == 0)
      return ErrorUnknown;

    // txMemcpyKind enum values match MemcpyKind directly
    // (0=H2H, 1=H2D, 2=D2H, 3=D2D)
    txMemcpyKind tx_kind;
    switch (kind) {
      case MemcpyHostToHost:
        tx_kind = txMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        tx_kind = txMemcpyHostToDevice;
        break;
      case MemcpyDeviceToHost:
        tx_kind = txMemcpyDeviceToHost;
        break;
      case MemcpyDeviceToDevice:
        tx_kind = txMemcpyDeviceToDevice;
        break;
      default:
        return ErrorUnknown;
    }

    txError_t err = txMemcpy(dst, src, static_cast<uint64_t>(count), tx_kind);
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      MemcpyKind kind,
      Stream_t stream) {
    if (!dst || !src || count == 0)
      return ErrorUnknown;

    txMemcpyKind tx_kind;
    switch (kind) {
      case MemcpyHostToHost:
        tx_kind = txMemcpyHostToHost;
        break;
      case MemcpyHostToDevice:
        tx_kind = txMemcpyHostToDevice;
        break;
      case MemcpyDeviceToHost:
        tx_kind = txMemcpyDeviceToHost;
        break;
      case MemcpyDeviceToDevice:
        tx_kind = txMemcpyDeviceToDevice;
        break;
      default:
        return ErrorUnknown;
    }

    txError_t err = txMemcpyAsync(
        dst, src, static_cast<uint64_t>(count), tx_kind,
        reinterpret_cast<txStream_t>(stream));
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t getPointerAttributes(
      PointerAttributes* attributes,
      const void* ptr) {
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
    txError_t err = txMemset(devPtr, value, static_cast<uint64_t>(count));
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
  }

  Error_t memsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
    txError_t err = txMemsetAsync(
        devPtr, value, static_cast<uint64_t>(count),
        reinterpret_cast<txStream_t>(stream));
    return (err == TX_SUCCESS) ? Success : ErrorUnknown;
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

Error_t Memcpy(
    void* dst,
    const void* src,
    size_t count,
    MemcpyKind kind) {
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

Error_t PointerGetAttributes(
    PointerAttributes* attributes,
    const void* ptr) {
  return MemoryManager::getInstance().getPointerAttributes(attributes, ptr);
}

Error_t Memset(void* devPtr, int value, size_t count) {
  return MemoryManager::getInstance().memset(devPtr, value, count);
}

Error_t MemsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
  return MemoryManager::getInstance().memsetAsync(devPtr, value, count, stream);
}