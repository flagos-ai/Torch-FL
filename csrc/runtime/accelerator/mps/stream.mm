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

// Apple Metal stream and event layer.
//
// A flagos stream is one MTLCommandQueue; a flagos event is one MTLSharedEvent
// plus the signal value EventRecord encoded into it. Metal orders a queue in
// order, so a stream's "pending work" is its newest committed command buffer,
// and an event is complete once its shared event's signaledValue has reached the
// value that was encoded.
//
// Two mappings are load-bearing:
//
//   * StreamQuery and EventQuery return ErrorNotReady (not a generic error)
//     while work is outstanding. The caching allocator's block-reuse path reads
//     ErrorNotReady as "keep the block", and anything else as a hard failure.
//   * A null Stream_t is the default stream. csrc/runtime/guard.h synthesizes
//     c10::Stream(UNSAFE, device, 0) on this backend and passes null to
//     EventRecord/StreamWaitEvent, so null must resolve to the process default
//     queue rather than be rejected.

#include "mps_runtime.h"

#include <chrono>
#include <cstdint>

namespace {

// The current stream is a thread-local, matching CUDA semantics; Apple has one
// GPU, so a single slot is enough (GetCurrentStreamForDevice still validates the
// index it is handed).
thread_local Stream_t gCurrentStream = nullptr;

bool ValidDevice(int device) {
  int count = 0;
  ::GetDeviceCount(&count);
  return device >= 0 && device < count;
}

} // namespace

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int priority) {
  // MTLCommandQueue has no flags or priority; a fresh queue is the whole API.
  (void)flags;
  (void)priority;
  if (!stream) {
    return ErrorUnknown;
  }
  id<MTLDevice> device = torch_fl_mps::MetalDevice();
  if (device == nil) {
    *stream = nullptr;
    return ErrorInvalidDevice;
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (queue == nil) {
    *stream = nullptr;
    return ErrorUnknown;
  }
  auto* state = new torch_fl_mps::StreamState();
  state->queue = queue; // +1, released by StreamDestroy
  torch_fl_mps::RegisterStream(state);
  *stream = reinterpret_cast<Stream_t>(state);
  return Success;
}

Error_t StreamCreate(Stream_t* stream) {
  return StreamCreateWithPriority(stream, 0, 0);
}

Error_t StreamGetPriority(Stream_t /*stream*/, int* priority) {
  if (!priority) {
    return ErrorUnknown;
  }
  *priority = 0;
  return Success;
}

Stream_t GetDefaultStreamForDevice(int device) {
  if (!ValidDevice(device)) {
    return nullptr;
  }
  return reinterpret_cast<Stream_t>(torch_fl_mps::DefaultStream());
}

Stream_t GetCurrentStreamForDevice(int device) {
  if (!ValidDevice(device)) {
    return nullptr;
  }
  if (gCurrentStream == nullptr) {
    gCurrentStream = GetDefaultStreamForDevice(device);
  }
  return gCurrentStream;
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  if (!ValidDevice(device)) {
    return ErrorInvalidDevice;
  }
  // A null handle resets to the default stream, mirroring the null-means-default
  // convention the rest of this file follows.
  gCurrentStream =
      (stream == nullptr) ? GetDefaultStreamForDevice(device) : stream;
  return Success;
}

Error_t StreamDestroy(Stream_t stream) {
  if (stream == nullptr) {
    // Nothing was allocated for the default stream; it lives for the process.
    return Success;
  }
  auto* state = reinterpret_cast<torch_fl_mps::StreamState*>(stream);
  if (state == torch_fl_mps::DefaultStream()) {
    return Success;
  }
  if (gCurrentStream == stream) {
    gCurrentStream = nullptr;
  }
  torch_fl_mps::UnregisterStream(state);
  if (state->last_committed != nil) {
    [state->last_committed release];
    state->last_committed = nil;
  }
  if (state->queue != nil) {
    [state->queue release];
    state->queue = nil;
  }
  delete state;
  return Success;
}

Error_t StreamQuery(Stream_t stream) {
  torch_fl_mps::StreamState* state = torch_fl_mps::AsStream(stream);
  if (state == nullptr || state->last_committed == nil) {
    return Success;
  }
  const MTLCommandBufferStatus status = state->last_committed.status;
  return (status == MTLCommandBufferStatusCompleted ||
          status == MTLCommandBufferStatusError)
      ? Success
      : ErrorNotReady;
}

Error_t StreamSynchronize(Stream_t stream) {
  torch_fl_mps::Drain(torch_fl_mps::AsStream(stream));
  return Success;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  (void)flags;
  auto* state = torch_fl_mps::AsStream(stream);
  auto* event_state = torch_fl_mps::AsEvent(event);
  if (state == nullptr || state->queue == nil || event_state == nullptr ||
      event_state->shared == nil || !event_state->recorded) {
    // No event recorded yet, so there is nothing to order behind.
    return Success;
  }
  id<MTLCommandBuffer> buffer = [state->queue commandBuffer];
  if (buffer == nil) {
    return ErrorUnknown;
  }
  [buffer encodeWaitForEvent:event_state->shared value:event_state->value];
  torch_fl_mps::Commit(state, buffer);
  return Success;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  if (!event) {
    return ErrorUnknown;
  }
  id<MTLDevice> device = torch_fl_mps::MetalDevice();
  if (device == nil) {
    *event = nullptr;
    return ErrorInvalidDevice;
  }
  id<MTLSharedEvent> shared = [device newSharedEvent];
  if (shared == nil) {
    *event = nullptr;
    return ErrorUnknown;
  }
  auto* state = new torch_fl_mps::EventState();
  state->shared = shared; // +1, released by EventDestroy
  state->enable_timing = (flags & EventEnableTiming) != 0;
  *event = reinterpret_cast<Event_t>(state);
  return Success;
}

Error_t EventCreate(Event_t* event) {
  return EventCreateWithFlags(event, 0);
}

Error_t EventDestroy(Event_t event) {
  auto* state = torch_fl_mps::AsEvent(event);
  if (state == nullptr) {
    return Success;
  }
  if (state->stamp != nil) {
    [state->stamp release];
    state->stamp = nil;
  }
  if (state->shared != nil) {
    [state->shared release];
    state->shared = nil;
  }
  delete state;
  return Success;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  auto* state = torch_fl_mps::AsEvent(event);
  if (state == nullptr) {
    return ErrorUnknown;
  }
  auto* stream_state = torch_fl_mps::AsStream(stream);
  if (stream_state == nullptr || stream_state->queue == nil) {
    // No queue to encode into (no device): a host-side record is still a valid
    // answer, and EventElapsedTime falls back to the host clock for it.
    state->host_stamp = std::chrono::steady_clock::now();
    state->recorded = true;
    return Success;
  }

  id<MTLCommandBuffer> buffer = [stream_state->queue commandBuffer];
  if (buffer == nil) {
    return ErrorUnknown;
  }
  const uint64_t value = ++stream_state->signal_counter;
  // Retain before committing: Commit() also retains, but the local reference
  // keeps the buffer alive across the two stores below.
  [buffer retain];
  [buffer encodeSignalEvent:state->shared value:value];
  torch_fl_mps::Commit(stream_state, buffer);

  id<MTLCommandBuffer> previous = state->stamp;
  state->stamp = buffer;
  if (previous != nil) {
    [previous release];
  }
  state->value = value;
  state->recorded = true;
  state->host_stamp = std::chrono::steady_clock::now();
  return Success;
}

Error_t EventSynchronize(Event_t event) {
  auto* state = torch_fl_mps::AsEvent(event);
  if (state == nullptr) {
    return ErrorUnknown;
  }
  if (!state->recorded || state->shared == nil) {
    return Success;
  }
  while ([state->shared signaledValue] < state->value) {
    if ([state->shared waitUntilSignaledValue:state->value
                                   timeoutMS:1000]) {
      break;
    }
  }
  return Success;
}

Error_t EventQuery(Event_t event) {
  auto* state = torch_fl_mps::AsEvent(event);
  if (state == nullptr) {
    return ErrorUnknown;
  }
  if (!state->recorded || state->shared == nil) {
    return ErrorNotReady;
  }
  return [state->shared signaledValue] >= state->value ? Success : ErrorNotReady;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  auto* a = torch_fl_mps::AsEvent(start);
  auto* b = torch_fl_mps::AsEvent(end);
  if (!ms || a == nullptr || b == nullptr) {
    return ErrorUnknown;
  }
  if (!a->recorded || !b->recorded) {
    return ErrorNotReady;
  }

  // Metal has no elapsed-time query. GPUStartTime/GPUEndTime on the command
  // buffer carrying each signal are the closest thing: the interval between the
  // start buffer's execution start and the end buffer's completion. They are
  // only populated once a buffer has finished, so an unfinished record falls
  // back to the host clock captured at EventRecord.
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  if (a->stamp != nil &&
      a->stamp.status >= MTLCommandBufferStatusCompleted) {
    start_seconds = a->stamp.GPUStartTime;
  }
  if (b->stamp != nil &&
      b->stamp.status >= MTLCommandBufferStatusCompleted) {
    end_seconds = b->stamp.GPUEndTime;
  }

  if (start_seconds > 0.0 && end_seconds >= start_seconds) {
    *ms = static_cast<float>((end_seconds - start_seconds) * 1000.0);
  } else {
    const std::chrono::duration<float, std::milli> delta =
        b->host_stamp - a->host_stamp;
    *ms = delta.count();
  }
  return Success;
}
