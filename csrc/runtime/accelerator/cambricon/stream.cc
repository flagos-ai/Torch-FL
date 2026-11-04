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

// Cambricon MLU stream and event layer, over libcnrt.
//
// Three naming differences from the CUDA original drive the whole file:
//
//   * Streams are cnrtQueue_t and events are cnrtNotifier_t. A notifier is
//     recorded with cnrtPlaceNotifier and waited on with cnrtWaitNotifier --
//     there is no "wait this queue's current point" with a timeout, so
//     cnrtQueueWaitNotifier covers the cross-queue dependency.
//   * A "not ready" sweep is cnrtErrorNotReady, which is why StreamQuery and
//     EventQuery map that one code to ErrorNotReady rather than flattening it
//     into ErrorUnknown: callers poll on it.
//   * cnrt has no stream priority band to query, so priorities are passed
//     through and DeviceGetStreamPriorityRange reports the neutral range.

#include <flagos.h>

#include <cnrt.h>

#include <cstdio>
#include <map>
#include <mutex>

namespace {

// Per-device current stream, the registry runtime/guard.h's stream arms read.
//
// A cnrt queue belongs to the device that was current when it was created, so
// the default queue has to be create-on-demand per device rather than one
// process-wide handle: an op on flagos:1 must not inherit flagos:0's queue.
//
// The registry is out of line here rather than a header-level inline with a
// function-local static, for the reason module musa records: an inline
// function-local static is per shared object under -fvisibility=hidden, so
// libtorch_fl.so and libtorch_bindings.so would each build their own map and
// the process would end up with two "default" queues. One definition, in the
// one library every other object links against.
std::map<int, cnrtQueue_t> default_queues;
std::map<int, cnrtQueue_t> current_queues;
std::mutex queue_mutex;

cnrtQueue_t get_default_queue(int device) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  auto it = default_queues.find(device);
  if (it != default_queues.end()) {
    return it->second;
  }

  int previous_device = 0;
  if (cnrtGetDevice(&previous_device) != cnrtSuccess) {
    return nullptr;
  }
  if (previous_device != device && cnrtSetDevice(device) != cnrtSuccess) {
    return nullptr;
  }

  cnrtQueue_t queue = nullptr;
  const cnrtRet_t create_err = cnrtQueueCreate(&queue);

  if (previous_device != device) {
    cnrtSetDevice(previous_device);
  }

  if (create_err != cnrtSuccess) {
    // Do NOT cache the failure: a cached null would pin a null default queue for
    // this device for the rest of the process, even after a transient create
    // failure clears, with no retry.
    return nullptr;
  }

  default_queues.emplace(device, queue);
  return queue;
}

Error_t toError(cnrtRet_t status) {
  switch (status) {
    case cnrtSuccess:
      return Success;
    case cnrtErrorNotReady:
      return ErrorNotReady;
    default:
      return ErrorUnknown;
  }
}

} // namespace

Stream_t GetDefaultStreamForDevice(int device) {
  return reinterpret_cast<Stream_t>(get_default_queue(device));
}

Stream_t GetCurrentStreamForDevice(int device) {
  // Look up under the lock, then fall through to the default queue *after*
  // releasing it: get_default_queue takes the same non-recursive mutex.
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    auto it = current_queues.find(device);
    if (it != current_queues.end()) {
      return reinterpret_cast<Stream_t>(it->second);
    }
  }
  return reinterpret_cast<Stream_t>(get_default_queue(device));
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  current_queues[device] = reinterpret_cast<cnrtQueue_t>(stream);
  return Success;
}

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int priority) {
  if (!stream) {
    return ErrorUnknown;
  }

  cnrtQueue_t queue = nullptr;
  const cnrtRet_t status =
      cnrtQueueCreateWithPriority(&queue, flags, priority);
  if (status != cnrtSuccess) {
    fprintf(
        stderr,
        "[flagos-cambricon] cnrtQueueCreateWithPriority(priority=%d) failed: %d\n",
        priority,
        static_cast<int>(status));
    return toError(status);
  }

  *stream = reinterpret_cast<Stream_t>(queue);
  return Success;
}

Error_t StreamCreate(Stream_t* stream) {
  if (!stream) {
    return ErrorUnknown;
  }

  cnrtQueue_t queue = nullptr;
  const cnrtRet_t status = cnrtQueueCreate(&queue);
  if (status != cnrtSuccess) {
    return toError(status);
  }

  *stream = reinterpret_cast<Stream_t>(queue);
  return Success;
}

Error_t StreamGetPriority(Stream_t stream, int* priority) {
  // cnrt exposes no getter for a queue's priority. Reporting the same neutral
  // value DeviceGetStreamPriorityRange reports keeps the pair consistent: a
  // caller that reads one and the other sees one priority level throughout.
  if (!priority) {
    return ErrorUnknown;
  }
  *priority = 0;
  (void)stream;
  return Success;
}

Error_t StreamDestroy(Stream_t stream) {
  if (!stream) {
    return Success;
  }
  return toError(cnrtQueueDestroy(reinterpret_cast<cnrtQueue_t>(stream)));
}

Error_t StreamQuery(Stream_t stream) {
  return toError(cnrtQueueQuery(reinterpret_cast<cnrtQueue_t>(stream)));
}

Error_t StreamSynchronize(Stream_t stream) {
  // A null queue means the current device's default queue, which cnrt accepts
  // for the sync and destroy calls the same way CUDA does for its null stream.
  return toError(cnrtQueueSync(reinterpret_cast<cnrtQueue_t>(stream)));
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  if (!event) {
    return ErrorUnknown;
  }
  return toError(
      cnrtQueueWaitNotifier(
          reinterpret_cast<cnrtNotifier_t>(event),
          reinterpret_cast<cnrtQueue_t>(stream),
          flags));
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  if (!event) {
    return ErrorUnknown;
  }

  cnrtNotifier_t notifier = nullptr;
  const cnrtRet_t status = cnrtNotifierCreateWithFlags(&notifier, flags);
  if (status != cnrtSuccess) {
    return toError(status);
  }

  *event = reinterpret_cast<Event_t>(notifier);
  return Success;
}

Error_t EventCreate(Event_t* event) {
  if (!event) {
    return ErrorUnknown;
  }

  cnrtNotifier_t notifier = nullptr;
  const cnrtRet_t status = cnrtNotifierCreate(&notifier);
  if (status != cnrtSuccess) {
    return toError(status);
  }

  *event = reinterpret_cast<Event_t>(notifier);
  return Success;
}

Error_t EventDestroy(Event_t event) {
  if (!event) {
    return Success;
  }
  return toError(cnrtNotifierDestroy(reinterpret_cast<cnrtNotifier_t>(event)));
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  if (!event) {
    return ErrorUnknown;
  }
  return toError(
      cnrtPlaceNotifier(
          reinterpret_cast<cnrtNotifier_t>(event),
          reinterpret_cast<cnrtQueue_t>(stream)));
}

Error_t EventSynchronize(Event_t event) {
  if (!event) {
    return ErrorUnknown;
  }
  return toError(cnrtWaitNotifier(reinterpret_cast<cnrtNotifier_t>(event)));
}

Error_t EventQuery(Event_t event) {
  if (!event) {
    return ErrorUnknown;
  }
  return toError(cnrtQueryNotifier(reinterpret_cast<cnrtNotifier_t>(event)));
}
Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  if (!ms || !start || !end) {
    return ErrorUnknown;
  }
  // Notifier-based timing needs the timing flag set at creation; callers that do
  // not ask for it (the caching allocator's events) pass EventDisableTiming and
  // get cnrt's own error back rather than a fabricated zero.
  return toError(
      cnrtNotifierElapsedTime(
          reinterpret_cast<cnrtNotifier_t>(start),
          reinterpret_cast<cnrtNotifier_t>(end),
          ms));
}
