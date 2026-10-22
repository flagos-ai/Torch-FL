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
#include <xpu/runtime.h>

#include <mutex>
#include <unordered_map>
#include <vector>

#include "xpu_stream.h"

namespace {

std::mutex stream_mutex;
std::unordered_map<int, XPUStream> default_streams;
thread_local std::unordered_map<int, XPUStream> current_streams;

XPUStream get_default_stream(int device) {
  std::lock_guard<std::mutex> lock(stream_mutex);
  auto it = default_streams.find(device);
  if (it != default_streams.end()) {
    return it->second;
  }

  // An XPUStream is bound to the device that was current when it was created,
  // so the creating call has to run on `device` even when a different one is
  // current -- which is the normal case, since the registry is queried for a
  // device other than the caller's on every multi-device run.
  int previous_device = 0;
  if (xpu_current_device(&previous_device) != 0) {
    return nullptr;
  }
  if (previous_device != device && xpu_set_device(device) != 0) {
    return nullptr;
  }

  XPUStream stream = nullptr;
  const int create_err = xpu_stream_create(&stream);

  if (previous_device != device) {
    xpu_set_device(previous_device);
  }

  if (create_err != 0) {
    // Do NOT cache the failure. Caching a null here would pin a null default
    // stream for this device for the rest of the process -- every later
    // GetDefaultStreamForDevice / GetCurrentStreamForDevice would hit the map
    // and return null, even after a transient create failure clears, with no
    // retry. Returning without inserting lets the next lookup try again.
    return nullptr;
  }

  default_streams.emplace(device, stream);
  return stream;
}

} // namespace

Stream_t GetDefaultStreamForDevice(int device) {
  return reinterpret_cast<Stream_t>(get_default_stream(device));
}

Stream_t GetCurrentStreamForDevice(int device) {
  auto it = current_streams.find(device);
  if (it != current_streams.end()) {
    return reinterpret_cast<Stream_t>(it->second);
  }
  return GetDefaultStreamForDevice(device);
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  current_streams[device] = reinterpret_cast<XPUStream>(stream);
  return Success;
}

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int /*flags*/,
    int /*priority*/) {
  // The XPU SDK has no stream priority: xpu_stream_create takes neither a
  // priority nor a flags argument, so there is nothing to honour here. The
  // priority is dropped and an ordinary stream is returned; the caller pairs
  // this with DeviceGetStreamPriorityRange, which reports a single level.
  if (!stream) {
    return ErrorUnknown;
  }
  return (xpu_stream_create(reinterpret_cast<XPUStream*>(stream)) == 0)
      ? Success
      : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  if (!stream) {
    return ErrorUnknown;
  }
  return (xpu_stream_create(reinterpret_cast<XPUStream*>(stream)) == 0)
      ? Success
      : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t /*stream*/, int* priority) {
  // No priority query in the XPU SDK; see StreamCreateWithPriority.
  *priority = 0;
  return Success;
}

Error_t StreamDestroy(Stream_t stream) {
  return (xpu_stream_destroy(reinterpret_cast<XPUStream>(stream)) == 0)
      ? Success
      : ErrorUnknown;
}

Error_t StreamQuery(Stream_t /*stream*/) {
  // The XPU SDK ships no stream-status query (there is no xpu_stream_query
  // among the runtime exports), so the runtime cannot answer whether work on
  // this stream has completed. This is distinct from ErrorUnknown: the runtime
  // is healthy and the handle is valid, it simply has no answer -- the same
  // distinction the tops runtime makes when its StreamQuery arm answers
  // ErrorNotSupported. Callers that need a completion signal use an event.
  return ErrorNotSupported;
}

Error_t StreamSynchronize(Stream_t stream) {
  return (xpu_wait(reinterpret_cast<XPUStream>(stream)) == 0) ? Success
                                                              : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int /*flags*/) {
  // xpu_stream_wait_event takes no flags argument.
  return (xpu_stream_wait_event(
              reinterpret_cast<XPUStream>(stream),
              reinterpret_cast<XPUEvent>(event)) == 0)
      ? Success
      : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int /*flags*/) {
  // xpu_event_create takes no flags argument and offers no timing switch, so
  // EventDisableTiming -- the only flag the contract defines -- needs no
  // translation. (Timestamps are unavailable either way; see EventElapsedTime.)
  if (!event) {
    return ErrorUnknown;
  }
  return (xpu_event_create(reinterpret_cast<XPUEvent*>(event)) == 0) ? Success
                                                                     : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  if (!event) {
    return ErrorUnknown;
  }
  return (xpu_event_create(reinterpret_cast<XPUEvent*>(event)) == 0) ? Success
                                                                     : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  return (xpu_event_destroy(reinterpret_cast<XPUEvent>(event)) == 0) ? Success
                                                                     : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  return (xpu_event_record(
              reinterpret_cast<XPUEvent>(event),
              reinterpret_cast<XPUStream>(stream)) == 0)
      ? Success
      : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  return (xpu_event_wait(reinterpret_cast<XPUEvent>(event)) == 0) ? Success
                                                                  : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  const int err = xpu_event_query(reinterpret_cast<XPUEvent>(event));
  if (err == 0) {
    return Success;
  }
  // Any non-zero result means the recorded work has not completed as far as
  // the runtime will tell us (a finished event reports 0/XPU_SUCCESS), so it is
  // read as "not ready" rather than as a fatal error.
  //
  // Do not match a single magic not-ready code here. -600 (CUDA's
  // cudaErrorNotReady / CUDA_ERROR_NOT_READY, surfaced through the CUDA-shaped
  // shim the SDK ships next to the native header) is one value xpu_event_query
  // returns while work is outstanding, but the native XPUError_t space starts
  // at XPU_ERRNO_BASE (700) upwards, so a build that answers with a native
  // not-ready code instead would slip past a literal `== -600` test and fall
  // through to ErrorUnknown. That matters because the caching allocator's
  // process_events treats ErrorNotReady as "still running, stop scanning" and
  // EVERY other status, including ErrorUnknown, as "completed": it then
  // destroys the event, drops the block's reference, and can hand a block an
  // outstanding op is still writing back to the pool -- silent corruption.
  // Reporting ErrorNotReady for any unresolved non-zero result is the safe
  // reading; the worst case for a genuine hard error is an event that is never
  // reclaimed (a leak) instead of a recycled live block.
  return ErrorNotReady;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  // No xpu_event_elapsed_time exists in the XPU runtime, and xpu_event_create
  // takes no timing flag, so there is no way to obtain a GPU timestamp. This is
  // the honest answer; the Python layer falls back to host timing.
  (void)ms;
  (void)start;
  (void)end;
  return ErrorNotSupported;
}

namespace at::native::flagos::kunlunxin {

FLAGOS_EXPORT void DrainDefaultXpuStreams() {
  std::vector<XPUStream> streams;
  {
    std::lock_guard<std::mutex> lock(stream_mutex);
    streams.reserve(default_streams.size());
    for (const auto& entry : default_streams) {
      streams.push_back(entry.second);
    }
  }
  for (XPUStream stream : streams) {
    if (stream) {
      xpu_wait(stream);
    }
  }
}

} // namespace at::native::flagos::kunlunxin
