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

// BPU stream and event layer.
//
// The BPU has no stream abstraction. Work is submitted as a task
// (hbUCPSubmitTask) and waited on (hbUCPWaitTaskDone) by the runtime before it
// returns, so every operation is already complete by the time torch could
// observe it. Streams therefore reduce to opaque non-null tokens and events to
// host-side timestamps -- which keeps torch's stream/event machinery working
// (c10::Stream requires a valid handle) without pretending to an ordering
// guarantee the hardware does not expose.
//
// EventElapsedTime is genuinely useful despite this: since submission is
// synchronous, wall-clock between two records is the real device time.

#include <flagos.h>

#include <chrono>
#include <cstdint>

namespace {

using Clock = std::chrono::steady_clock;

struct EventImpl {
  Clock::time_point stamp{};
  bool recorded = false;
};

// A single sentinel token stands in for the (only, implicit) stream. Handing out
// a non-null pointer matters: c10 treats a null stream handle as invalid.
// (not constexpr: a reinterpret_cast is not a constant expression)
const auto kDefaultStream = reinterpret_cast<Stream_t>(static_cast<uintptr_t>(0x1));

} // namespace

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int /*flags*/,
    int /*priority*/) {
  if (!stream) {
    return ErrorUnknown;
  }
  *stream = kDefaultStream;
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

Error_t StreamDestroy(Stream_t /*stream*/) {
  // Nothing was allocated in StreamCreate, so there is nothing to release.
  return Success;
}

Error_t StreamQuery(Stream_t /*stream*/) {
  // Submission is synchronous: a stream is never busy when observed.
  return Success;
}

Error_t StreamSynchronize(Stream_t /*stream*/) {
  return Success;
}

Error_t StreamWaitEvent(
    Stream_t /*stream*/,
    Event_t /*event*/,
    unsigned int /*flags*/) {
  // Any recorded event has already completed, so the wait is satisfied.
  return Success;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int /*flags*/) {
  if (!event) {
    return ErrorUnknown;
  }
  *event = reinterpret_cast<Event_t>(new EventImpl());
  return Success;
}

Error_t EventCreate(Event_t* event) {
  return EventCreateWithFlags(event, 0);
}

Error_t EventDestroy(Event_t event) {
  delete reinterpret_cast<EventImpl*>(event);
  return Success;
}

Error_t EventRecord(Event_t event, Stream_t /*stream*/) {
  auto* impl = reinterpret_cast<EventImpl*>(event);
  if (!impl) {
    return ErrorUnknown;
  }
  impl->stamp = Clock::now();
  impl->recorded = true;
  return Success;
}

Error_t EventSynchronize(Event_t /*event*/) {
  return Success;
}

Error_t EventQuery(Event_t event) {
  auto* impl = reinterpret_cast<EventImpl*>(event);
  if (!impl) {
    return ErrorUnknown;
  }
  return impl->recorded ? Success : ErrorNotReady;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  auto* a = reinterpret_cast<EventImpl*>(start);
  auto* b = reinterpret_cast<EventImpl*>(end);
  if (!ms || !a || !b) {
    return ErrorUnknown;
  }
  if (!a->recorded || !b->recorded) {
    return ErrorNotReady;
  }
  const std::chrono::duration<float, std::milli> delta = b->stamp - a->stamp;
  *ms = delta.count();
  return Success;
}
