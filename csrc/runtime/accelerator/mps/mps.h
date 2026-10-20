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

// Apple Metal runtime extensions beyond the flagos C ABI.
//
// The ABI in csrc/include/flagos.h has no capacity query, but Metal does
// (MTLDevice.recommendedMaxWorkingSetSize / currentAllocatedSize), so the
// allocator's get_memory_info reports real numbers instead of the zeros a
// backend with no query at all has to settle for (see bpu_memory.h). Declaring
// it here rather than widening the flagos ABI keeps the cross-platform contract
// unchanged; only the allocator (csrc/) and this runtime ever name it.

#pragma once

#include <stddef.h>

#include <flagos.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bytes the Metal device can still hand out and its recommended working-set
// ceiling. Either pointer may be NULL, in which case that value is skipped.
// Reports zeros when no MPS-capable device is present.
FLAGOS_EXPORT void FlagosMPSDeviceMemoryInfo(size_t* freeBytes, size_t* totalBytes);

// The MTLBuffer whose `contents` cover `ptr`, and the byte offset of `ptr`
// inside it, for the operator layer (csrc/aten/backends/mps/).
//
// This is what lets a kernel describe an aten tensor to MPSGraph without a
// copy: the tensor's `data_ptr` is the buffer's contents plus its storage
// offset, and the caching allocator subdivides one MTLBuffer into many blocks,
// so the pointer is usually interior. Returns NULL -- and leaves *byteOffset
// untouched -- when `ptr` is not inside a Metal allocation this runtime owns
// (a host tensor, a foreign pointer, a stale one). The returned buffer is a
// borrowed reference: it lives until the owning allocation is freed, and the
// caller must not release it.
FLAGOS_EXPORT void* FlagosMPSBufferForPointer(const void* ptr, size_t* byteOffset);

// The MTLCommandQueue a flagos stream runs on, as an opaque pointer. A null
// `stream` resolves to the default stream, matching the rest of this runtime.
// Returns NULL when no Metal device is present. Borrowed like the buffer above.
FLAGOS_EXPORT void* FlagosMPSCommandQueue(Stream_t stream);

// The runtime-owned queue behind the default stream, without creating it.
FLAGOS_EXPORT void* FlagosMPSDefaultCommandQueue(void);

// Records a completion sentinel for a producer that schedules work on the
// stream's queue without handing back the MTLCommandBuffer it used, which is
// what MPSGraph's asynchronous runs do.
//
// The stream layer tracks "the newest committed buffer" and answers
// StreamQuery/StreamSynchronize/DeviceSynchronize from it, on the guarantee that
// Metal executes a queue in order. A producer that keeps its own command buffer
// private would leave that pointer naming older work, so a query could report a
// stream idle while the producer's kernel is still writing the buffer a tensor
// points at -- and the caching allocator reads StreamQuery to decide whether a
// freed block may be reused. This commits one empty command buffer to the
// stream's queue and records it as the newest, which puts the producer's work
// behind the sentinel without the sentinel waiting for anything itself.
//
// A no-op when `stream` has no queue (no Metal device), like the accessor above.
FLAGOS_EXPORT void FlagosMPSStreamSentinel(Stream_t stream);

// Drain queued work and invoke the caching allocator's exit hook. Called both
// by the Python atexit path and as the final safety net for a process that
// imported torch_fl but never drained a queue itself. Idempotent.
FLAGOS_EXPORT void FlagosMPSDrainAtExit(void);

// Drop the allocator's exit callback before the owning allocator is destroyed.
FLAGOS_EXPORT void FlagosMPSClearAllocatorShutdown(void);

#ifdef __cplusplus
} // extern "C"
#endif
