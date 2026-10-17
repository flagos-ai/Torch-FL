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

// Internal state for the Metal runtime floor. Every translation unit under
// csrc/runtime/accelerator/mps/ is Objective-C++ (.mm) because Metal is an
// Objective-C API; the shared objects live here so device.mm, memory.mm and
// stream.mm agree on one MTLDevice, one set of live command queues and one
// drain path.
//
// Only this directory includes the header. It is deliberately not part of the
// public flagos ABI (csrc/include/flagos.h); the C ABI the rest of torch-fl
// links against is unchanged.

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <flagos.h>

#include <chrono>
#include <cstdint>

namespace torch_fl_mps {

// The process-wide Metal device, or nil when the host has no MPS-capable GPU.
// Created once on first use: MTLCreateSystemDefaultDevice() is not free, and a
// Mac that cannot run MPS must keep answering "no device" cheaply.
id<MTLDevice> MetalDevice();

// Whether a usable Apple GPU is present. A discrete AMD GPU on an Intel Mac
// answers MTLCreateSystemDefaultDevice() too but does not back MPS, so unified
// memory is the discriminator rather than a non-nil device.
bool MetalAvailable();

// Whether this host's Metal device is one MPSGraph can actually encode on.
//
// MetalAvailable answers "is there an Apple GPU with unified memory", which is
// the right question for allocation and memcpy -- those go straight to MTLDevice
// and a paravirtualized device serves them fine. It is the wrong question for
// the graph path: MPSGraph compiles through MLIR and re-specializes at encode
// time, and on a paravirtualized device (`AppleParavirtGPUMetalIOGPUFamily`,
// what the hosted macOS runners expose) that encode faults inside MPSGraph's own
// `runAsyncWithMTLCommandQueue:` with a SIGSEGV at a small address -- past every
// point the operator layer can decline at, because the graph is built and the
// handler is committed before the dying call is reached.
//
// A fatal signal cannot be caught and unwound from, so there is nothing to
// recover with at the call site: the only place the answer can be honoured is
// before the graph is built at all. Hence a capability query the operator layer
// consults, and a device-name test for the one family known to fault.
//
// False whenever no device is present, so a caller can treat
// `MetalAvailable() && MetalGraphUsable()` as the single "the native graph path
// is open" predicate.
bool MetalGraphUsable();

// One command queue plus the last command buffer committed to it. Tracking the
// latest buffer is enough to drain the queue: Metal executes a queue in order,
// so waiting on the newest committed buffer waits on everything before it.
struct StreamState {
  id<MTLCommandQueue> queue = nil;
  id<MTLCommandBuffer> last_committed = nil;
  // Monotonic signal value, incremented per event recorded on this queue. Every
  // MTLSharedEvent carries one counter, so a value taken from it uniquely names
  // one record point and can never be confused with an earlier one.
  uint64_t signal_counter = 0;
};

// One shared event and the record point it currently describes.
struct EventState {
  id<MTLSharedEvent> shared = nil;
  // The signal value EventRecord encoded, and the command buffer carrying it.
  // The buffer is kept (not just the value) so EventElapsedTime can read the
  // GPUStartTime/GPUEndTime Metal stamped on it.
  uint64_t value = 0;
  id<MTLCommandBuffer> stamp = nil;
  bool recorded = false;
  bool enable_timing = false;
  // Host-clock stamps, the fallback when the GPU timestamps are unavailable
  // (e.g. a queue that never reached the GPU because there was no work to run).
  std::chrono::steady_clock::time_point host_stamp{};
};

// The device's default queue, created lazily. Apple exposes one GPU, so the
// default is global rather than per-index.
StreamState* DefaultStream();

// Interpret a flagos Stream_t. A null handle means "the current stream", which
// is what csrc/runtime/guard.h passes on this backend (its synthetic stream id
// is 0, i.e. null) and what the caching allocator passes for the default stream.
StreamState* AsStream(Stream_t stream);

// Commit a command buffer to a queue and remember it as the one to drain.
void Commit(StreamState* stream, id<MTLCommandBuffer> buffer);

// Wait for every command buffer committed so far on this queue.
void Drain(StreamState* stream);

// Wait on every live queue. Memcpy and DeviceSynchronize use this: a host read
// of a shared buffer must not race a kernel that is still encoding into it, and
// the copy sites cannot always name the producing stream.
void DrainAll();

// Live-queue registry, used by DrainAll(). StreamDestroy removes its entry.
void RegisterStream(StreamState* stream);
void UnregisterStream(StreamState* stream);

// EventState cast helpers.
EventState* AsEvent(Event_t event);

} // namespace torch_fl_mps
