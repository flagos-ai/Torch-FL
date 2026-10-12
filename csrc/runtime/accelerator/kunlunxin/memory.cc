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

// Kunlunxin P800 XPU memory layer, ported from the CUDA runtime sources onto the
// native xpu_* API (see runtime/accelerator/kunlunxin and the top-level
// CMakeLists kunlunxin arm). Three places differ from the CUDA original because
// the XPU SDK offers less than the CUDA runtime does:
//
//   * PointerGetAttributes cannot ask the driver. `xpu_pointer_get_attrs` and
//     `xpu_host_pointer_get_attrs` are declared in <xpu/runtime.h> but are NOT
//     exported by libxpurt.so.2 -- linking against either fails. The registry
//     below therefore answers from what this process allocated, exactly as the
//     CUDA original already did for the block bookkeeping; the driver fallback
//     that would have covered foreign pointers is unavailable.
//   * There is no xpu_memset / xpu_memset_async. Memset is emulated by writing a
//     process-lifetime staging buffer and copying it to the device, then waiting
//     for the copy to land (see memset_).
//   * XPUMemcpyKind has no host-to-host member (values 0/1/2 are D2H/H2D/D2D),
//     so MemcpyHostToHost is served by std::memcpy rather than by the runtime.

#include <flagos.h>
#include <xpu/runtime.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

#include "xpu_stream.h"

namespace {

// Size of the host staging buffer used by the memset emulation. Small enough to
// stay in cache and to make the per-call host allocation negligible, large
// enough that a typical tensor fill is a handful of copies.
constexpr size_t kMemsetChunkBytes = 1u << 20;

struct Block {
  MemoryType type = MemoryType::MemoryTypeUnmanaged;
  int device = -1;
  void* pointer = nullptr;
  size_t size = 0;
};

// Maps a flagos MemcpyKind onto XPUMemcpyKind. Returns false for
// MemcpyHostToHost, which the runtime cannot express -- the caller handles that
// one itself. Mapping explicitly (rather than casting) means a future enum
// change surfaces here instead of silently copying in the wrong direction.
bool toXpuMemcpyKind(MemcpyKind kind, XPUMemcpyKind* out) {
  switch (kind) {
    case MemcpyHostToDevice:
      *out = XPU_HOST_TO_DEVICE;
      return true;
    case MemcpyDeviceToHost:
      *out = XPU_DEVICE_TO_HOST;
      return true;
    case MemcpyDeviceToDevice:
      *out = XPU_DEVICE_TO_DEVICE;
      return true;
    default:
      return false;
  }
}

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

      // xpu_malloc allocates on the *current* device; make sure that is the one
      // the caller's tensor lives on. This matters in multi-process setups such
      // as DDP, where the device is selected per rank.
      if (current_device >= 0) {
        const int set_err = xpu_set_device(current_device);
        if (set_err != 0) {
          fprintf(
              stderr,
              "[flagos-kunlunxin] xpu_set_device(%d) failed: %s\n",
              current_device,
              xpu_strerror(set_err));
          return ErrorMemoryAllocation;
        }
      }

      int err = xpu_malloc(&mem, size, XPU_MEM_MAIN);
      if (err != 0 || mem == nullptr) {
        fprintf(
            stderr,
            "[flagos-kunlunxin] xpu_malloc(%zu bytes) on device %d failed: %s\n",
            size,
            current_device,
            xpu_strerror(err));
        return ErrorMemoryAllocation;
      }
    } else {
      int err = xpu_host_alloc(&mem, size, 0);
      if (err != 0 || mem == nullptr) {
        fprintf(
            stderr,
            "[flagos-kunlunxin] xpu_host_alloc(%zu bytes) failed: %s\n",
            size,
            xpu_strerror(err));
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

    const Block& info = it->second;
    int err = (info.type == MemoryType::MemoryTypeDevice)
        ? xpu_free(info.pointer)
        : xpu_host_free(info.pointer);

    m_registry.erase(it);
    return (err == 0) ? Success : ErrorUnknown;
  }

  Error_t memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
    if (!dst || !src || count == 0) {
      return ErrorUnknown;
    }

    if (kind == MemcpyHostToHost) {
      std::memcpy(dst, src, count);
      return Success;
    }

    XPUMemcpyKind xpu_kind;
    if (!toXpuMemcpyKind(kind, &xpu_kind)) {
      return ErrorUnknown;
    }

    // The vendor op entry points hand back no completion handle, so the op
    // layer drains its stream after every dispatch and this blocking copy
    // normally has nothing to wait for. Draining anyway keeps the ordering
    // guarantee independent of that call-site discipline -- see xpu_stream.h.
    at::native::flagos::kunlunxin::DrainDefaultXpuStreams();

    return (xpu_memcpy(dst, src, count, xpu_kind) == 0) ? Success : ErrorUnknown;
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

    if (kind == MemcpyHostToHost) {
      // No stream-ordered host-to-host copy in the runtime; the copy is already
      // synchronous on the host, so ordering is trivially satisfied.
      std::memcpy(dst, src, count);
      return Success;
    }

    XPUMemcpyKind xpu_kind;
    if (!toXpuMemcpyKind(kind, &xpu_kind)) {
      return ErrorUnknown;
    }

    return (xpu_memcpy_async(
                dst,
                src,
                count,
                xpu_kind,
                reinterpret_cast<XPUStream>(stream)) == 0)
        ? Success
        : ErrorUnknown;
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

  Error_t memset_(void* devPtr, int value, size_t count, Stream_t stream) {
    if (!devPtr || count == 0) {
      return ErrorUnknown;
    }

    // No xpu_memset exists, so the fill goes through a host staging buffer:
    // fill it once, then copy it over the destination in chunks. The tail chunk
    // copies whatever remains.
    //
    // The staging buffer is allocated once and kept for the process lifetime
    // rather than per call. Because it is shared, the fill -> copy -> wait
    // sequence is a critical section: a second memset must not overwrite the
    // buffer while an earlier copy is still reading it. m_memset_mutex
    // serialises the sequence, and the wait below drains the copies before the
    // buffer is handed to the next caller.
    std::lock_guard<std::mutex> lock(m_memset_mutex);
    if (m_memset_staging == nullptr) {
      const int alloc_err = xpu_host_alloc(&m_memset_staging, kMemsetChunkBytes, 0);
      if (alloc_err != 0 || m_memset_staging == nullptr) {
        fprintf(
            stderr,
            "[flagos-kunlunxin] memset staging xpu_host_alloc(%zu bytes) failed: %s\n",
            kMemsetChunkBytes,
            xpu_strerror(alloc_err));
        m_memset_staging = nullptr;
        return ErrorMemoryAllocation;
      }
    }

    // Only the bytes that can actually be copied need filling; the largest
    // single copy is min(count, kMemsetChunkBytes).
    const size_t fill =
        count < kMemsetChunkBytes ? count : kMemsetChunkBytes;
    std::memset(m_memset_staging, value, fill);

    auto* dst = static_cast<unsigned char*>(devPtr);
    Error_t result = Success;
    for (size_t offset = 0; offset < count; offset += kMemsetChunkBytes) {
      const size_t bytes = (count - offset < kMemsetChunkBytes)
          ? (count - offset)
          : kMemsetChunkBytes;
      const int copy_err = xpu_memcpy_async(
          dst + offset,
          m_memset_staging,
          bytes,
          XPU_HOST_TO_DEVICE,
          reinterpret_cast<XPUStream>(stream));
      if (copy_err != 0) {
        result = ErrorUnknown;
        break;
      }
    }

    // The copies above read m_memset_staging asynchronously and the buffer is
    // reused by the next memset, so they must have landed before this call
    // returns -- otherwise the next fill would overwrite host memory the DMA
    // engine is still reading and the device would latch a half-written
    // pattern. The wait also gives the synchronous entry point (Memset, which
    // passes a null stream) its blocking contract: a null XPUStream means the
    // current device's default stream, so the same call drains both paths.
    xpu_wait(reinterpret_cast<XPUStream>(stream));
    return result;
  }

 private:
  MemoryManager() = default;

  // Process-lifetime host staging buffer for the memset emulation (see
  // memset_). It is deliberately never freed: it is a single fixed 1 MiB block
  // that lives as long as the process, and releasing it from the singleton's
  // static destructor would race driver teardown.
  void* m_memset_staging = nullptr;

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
  // Guards the fill/copy/wait sequence of memset_ over the shared staging
  // buffer. Separate from m_mutex, which guards m_registry.
  std::mutex m_memset_mutex;
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
  return MemoryManager::getInstance().memset_(devPtr, value, count, nullptr);
}

Error_t MemsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
  return MemoryManager::getInstance().memset_(devPtr, value, count, stream);
}
