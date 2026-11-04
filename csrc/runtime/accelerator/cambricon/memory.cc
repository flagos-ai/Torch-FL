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

// Cambricon MLU memory layer, over libcnrt.
//
// Compared with the CUDA original there is one structural difference: cnrt's
// memcpy takes a direction tag (cnrtMemTransDir_t) instead of inferring it from
// the two pointers, and host-to-host is a separate tag rather than something
// the runtime does implicitly. The mapping below is explicit for that reason --
// a cast between the two enums would copy in the wrong direction once either
// side renumbers.
//
// PointerGetAttributes asks the driver (cnrtPointerGetAttributes) instead of
// consulting an allocation registry. Unlike the Kunlunxin XPU SDK, whose
// pointer-attribute calls are declared but not exported, cnrt does export this
// one, and answering from the driver is also what covers foreign pointers (a
// tensor handed over by another process or by a C++ extension).

#include <flagos.h>

#include <cnrt.h>

#include <cstdio>
#include <cstring>

namespace {

// Maps a flagos MemcpyKind onto cnrtMemTransDir_t. Returns false for
// MemcpyHostToHost, which cnrt refuses as a device copy direction -- the caller
// serves it with std::memcpy.
bool toCnrtDir(MemcpyKind kind, cnrtMemTransDir_t* out) {
  switch (kind) {
    case MemcpyHostToDevice:
      *out = cnrtMemcpyHostToDev;
      return true;
    case MemcpyDeviceToHost:
      *out = cnrtMemcpyDevToHost;
      return true;
    case MemcpyDeviceToDevice:
      *out = cnrtMemcpyDevToDev;
      return true;
    default:
      return false;
  }
}

Error_t toError(cnrtRet_t status, const char* what) {
  if (status == cnrtSuccess) {
    return Success;
  }
  fprintf(stderr, "[flagos-cambricon] %s failed: %d\n", what, static_cast<int>(status));
  return ErrorUnknown;
}

} // namespace

Error_t Malloc(void** devPtr, size_t size) {
  if (!devPtr || size == 0) {
    return ErrorUnknown;
  }

  // cnrtMalloc allocates on the *current* device, so in a multi-process setup
  // (one rank per device) the caller's device must already be current. The
  // device guard around every op makes that true; this is the same assumption
  // the CUDA arm makes.
  const cnrtRet_t status = cnrtMalloc(devPtr, size);
  if (status != cnrtSuccess) {
    fprintf(
        stderr,
        "[flagos-cambricon] cnrtMalloc(%zu bytes) failed: %d\n",
        size,
        static_cast<int>(status));
    return ErrorMemoryAllocation;
  }
  return Success;
}

Error_t Free(void* devPtr) {
  if (!devPtr) {
    return Success;
  }
  return toError(cnrtFree(devPtr), "cnrtFree");
}

Error_t MallocHost(void** hostPtr, size_t size) {
  if (!hostPtr || size == 0) {
    return ErrorUnknown;
  }

  // cnrtHostMalloc is pinned (page-locked) memory, the counterpart of
  // cudaMallocHost in the CUDA arm.
  const cnrtRet_t status = cnrtHostMalloc(hostPtr, size);
  if (status != cnrtSuccess) {
    fprintf(
        stderr,
        "[flagos-cambricon] cnrtHostMalloc(%zu bytes) failed: %d\n",
        size,
        static_cast<int>(status));
    return ErrorMemoryAllocation;
  }
  return Success;
}

Error_t FreeHost(void* hostPtr) {
  if (!hostPtr) {
    return Success;
  }
  return toError(cnrtFreeHost(hostPtr), "cnrtFreeHost");
}

Error_t Memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
  if (!dst || !src) {
    return ErrorUnknown;
  }
  if (count == 0) {
    return Success;
  }

  if (kind == MemcpyHostToHost) {
    std::memcpy(dst, src, count);
    return Success;
  }

  cnrtMemTransDir_t dir;
  if (!toCnrtDir(kind, &dir)) {
    return ErrorUnknown;
  }

  // A copy onto the same buffer is a no-op, and cnrt does not accept it: it
  // returns CN_MEMORY_ERROR_INVALID_ADDRESS (632015) for `cnrtMemcpy` with
  // dst == src, where cudaMemcpy and the other vendor APIs treat it as a
  // successful copy of zero effect. PyTorch issues these -- `copy_data` in the
  // device allocator is handed identical pointers when a tensor is constructed
  // from storage it already owns -- and on a Qwen3 backward that was thousands of
  // rejected copies per step, each one logged and each one leaving the buffer
  // alone. Nothing inspected the return value, so the run looked healthy.
  if (dst == src) {
    return Success;
  }

  // cnrtMemcpy is the blocking form; it synchronises internally, so it needs no
  // queue argument once the default queue exists on the target device.
  return toError(
      cnrtMemcpy(dst, const_cast<void*>(src), count, dir), "cnrtMemcpy");
}

Error_t MemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    MemcpyKind kind,
    Stream_t stream) {
  if (!dst || !src) {
    return ErrorUnknown;
  }
  if (count == 0) {
    return Success;
  }

  if (kind == MemcpyHostToHost) {
    // No stream-ordered host-to-host copy in cnrt; the copy is host-synchronous
    // already, so ordering against the device queue is trivially satisfied.
    std::memcpy(dst, src, count);
    return Success;
  }

  cnrtMemTransDir_t dir;
  if (!toCnrtDir(kind, &dir)) {
    return ErrorUnknown;
  }

  // Same reasoning as `Memcpy`: cnrt rejects a copy onto the same buffer, and a
  // no-op is what the caller asked for.
  if (dst == src) {
    return Success;
  }

  // A null queue means "the default queue of the current device", which is also
  // how cnrt interprets it.
  return toError(
      cnrtMemcpyAsync(
          dst,
          const_cast<void*>(src),
          count,
          reinterpret_cast<cnrtQueue_t>(stream),
          dir),
      "cnrtMemcpyAsync");
}

Error_t PointerGetAttributes(PointerAttributes* attributes, const void* ptr) {
  if (!attributes || !ptr) {
    return ErrorUnknown;
  }

  cnrtPointerAttributes_t raw{};
  const cnrtRet_t status = cnrtPointerGetAttributes(&raw, ptr);
  if (status != cnrtSuccess) {
    // A pointer the driver does not recognise is not necessarily an error worth
    // failing the call over; report it the way the CUDA arm reports host memory.
    attributes->type = MemoryType::MemoryTypeUnmanaged;
    attributes->device = -1;
    attributes->pointer = const_cast<void*>(ptr);
    return Success;
  }

  switch (raw.type) {
    case cnrtMemTypeDevice:
      attributes->type = MemoryType::MemoryTypeDevice;
      break;
    case cnrtMemTypeHost:
      attributes->type = MemoryType::MemoryTypeHost;
      break;
    default:
      attributes->type = MemoryType::MemoryTypeUnmanaged;
      break;
  }
  attributes->device = raw.device;
  attributes->pointer = raw.devicePointer ? raw.devicePointer
                                          : const_cast<void*>(ptr);
  return Success;
}

Error_t Memset(void* devPtr, int value, size_t count) {
  if (!devPtr) {
    return ErrorUnknown;
  }
  if (count == 0) {
    return Success;
  }
  return toError(cnrtMemset(devPtr, value, count), "cnrtMemset");
}

Error_t MemsetAsync(void* devPtr, int value, size_t count, Stream_t stream) {
  if (!devPtr) {
    return ErrorUnknown;
  }
  if (count == 0) {
    return Success;
  }
  return toError(
      cnrtMemsetAsync(
          devPtr, value, count, reinterpret_cast<cnrtQueue_t>(stream)),
      "cnrtMemsetAsync");
}