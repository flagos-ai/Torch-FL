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

// Shared Metal state for the flagos device runtime.
//
// device.mm, memory.mm and stream.mm all reach the GPU through one MTLDevice,
// one lazily built default command queue, and one registry of live queues, so
// the pieces live here rather than in any single translation unit. There is no
// Objective-C ARC: this file builds with the compiler default (manual retain /
// release), which is what lets the plain C++ side tables elsewhere hold
// MTLBuffer and MTLCommandBuffer references without ownership qualifiers. Every
// +1 object stored in those tables is retained on the way in and released on the
// way out.

#include "mps_runtime.h"

#include <mutex>
#include <set>
#include <vector>

namespace torch_fl_mps {
namespace {

std::mutex gDeviceMutex;
id<MTLDevice> gDevice = nil;
bool gDeviceResolved = false;

std::mutex gDefaultStreamMutex;
StreamState* gDefaultStream = nullptr;

// Every queue DrainAll() must wait on. The default queue is one entry; each
// StreamCreate() adds another and StreamDestroy() takes it out.
std::mutex gRegistryMutex;
std::set<StreamState*> gLiveStreams;

} // namespace

id<MTLDevice> MetalDevice() {
  std::lock_guard<std::mutex> lock(gDeviceMutex);
  if (!gDeviceResolved) {
    // +1 (Create rule). Held for the life of the process; a Mac has one GPU and
    // re-querying here would only rebuild the same object.
    gDevice = MTLCreateSystemDefaultDevice();
    gDeviceResolved = true;
  }
  return gDevice;
}

bool MetalAvailable() {
  id<MTLDevice> device = MetalDevice();
  // A non-nil device is not enough: an Intel Mac with a discrete AMD GPU also
  // answers MTLCreateSystemDefaultDevice(), but MPS (and the shared-memory
  // assumption this runtime is built on) needs unified memory.
  return device != nil && device.hasUnifiedMemory;
}

StreamState* DefaultStream() {
  std::lock_guard<std::mutex> lock(gDefaultStreamMutex);
  if (gDefaultStream == nullptr) {
    id<MTLDevice> device = MetalDevice();
    if (device == nil) {
      return nullptr;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return nullptr;
    }
    auto* state = new StreamState();
    state->queue = queue; // +1, owned by the state for the process lifetime
    gDefaultStream = state;
    RegisterStream(state);
  }
  return gDefaultStream;
}

StreamState* AsStream(Stream_t stream) {
  // A null handle is the default stream. csrc/runtime/guard.h synthesizes
  // c10::Stream(UNSAFE, device, 0) on this backend, so id 0 -- a null Stream_t
  // -- is the stream every aten call and every allocator event arrives on.
  if (stream == nullptr) {
    return DefaultStream();
  }
  return reinterpret_cast<StreamState*>(stream);
}

void Commit(StreamState* stream, id<MTLCommandBuffer> buffer) {
  if (stream == nullptr || buffer == nil) {
    return;
  }
  // The buffer usually comes from -[MTLCommandQueue commandBuffer], i.e. +0,
  // and the caller's autorelease pool may drain before the next wait. Take a
  // reference so "the newest committed buffer" stays valid for Drain().
  [buffer retain];
  [buffer commit];
  id<MTLCommandBuffer> previous = stream->last_committed;
  stream->last_committed = buffer;
  if (previous != nil) {
    [previous release];
  }
}

void Drain(StreamState* stream) {
  if (stream == nullptr) {
    return;
  }
  id<MTLCommandBuffer> buffer = stream->last_committed;
  if (buffer == nil) {
    return;
  }
  // Metal executes a queue strictly in order, so waiting on the newest buffer
  // waits on everything before it. waitUntilCompleted on an already-finished
  // buffer returns immediately; on one that was never committed it would raise,
  // which is why last_committed only ever names a committed buffer.
  const MTLCommandBufferStatus status = buffer.status;
  if (status == MTLCommandBufferStatusCommitted ||
      status == MTLCommandBufferStatusScheduled) {
    [buffer waitUntilCompleted];
  }
}

void DrainAll() {
  // Snapshot under the lock, then wait outside it: draining can block for the
  // duration of a kernel, and RegisterStream on another thread must not stall
  // behind that.
  std::vector<StreamState*> streams;
  {
    std::lock_guard<std::mutex> lock(gRegistryMutex);
    streams.assign(gLiveStreams.begin(), gLiveStreams.end());
  }
  for (StreamState* stream : streams) {
    Drain(stream);
  }
}

void RegisterStream(StreamState* stream) {
  if (stream == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(gRegistryMutex);
  gLiveStreams.insert(stream);
}

void UnregisterStream(StreamState* stream) {
  if (stream == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(gRegistryMutex);
  gLiveStreams.erase(stream);
}

EventState* AsEvent(Event_t event) {
  return reinterpret_cast<EventState*>(event);
}

} // namespace torch_fl_mps
