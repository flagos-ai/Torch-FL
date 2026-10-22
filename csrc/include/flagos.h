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

#pragma once

#include <cstddef>

#ifndef FLAGOS_EXPORT
#ifdef _WIN32
#define FLAGOS_EXPORT __declspec(dllexport)
#else
#define FLAGOS_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum Error_t {
  Success = 0,
  ErrorUnknown = 1,
  ErrorNotReady = 2,
  ErrorInvalidDevice = 3,
  ErrorMemoryAllocation = 4,
  // The call was well-formed and the runtime is healthy, but the vendor runtime
  // implements no counterpart for it. Distinct from ErrorUnknown on purpose: a
  // query that returns this reports nothing outstanding, because the runtime has
  // no way to report anything else (see the tops StreamQuery/EventQuery arms).
  ErrorNotSupported = 5,
} Error_t;

typedef enum MemcpyKind {
  MemcpyHostToHost = 0,
  MemcpyHostToDevice = 1,
  MemcpyDeviceToHost = 2,
  MemcpyDeviceToDevice = 3
} MemcpyKind;

typedef enum MemoryType {
  MemoryTypeUnmanaged = 0,
  MemoryTypeHost = 1,
  MemoryTypeDevice = 2
} MemoryType;

struct PointerAttributes {
  MemoryType type;
  int device;
  void* pointer;
};

typedef enum EventFlags {
  EventDisableTiming = 0x0,
  EventEnableTiming = 0x1,
} EventFlags;

struct Stream;
struct Event;
typedef struct Stream* Stream_t;
typedef struct Event* Event_t;

// Memory
FLAGOS_EXPORT Error_t Malloc(void** devPtr, size_t size);
FLAGOS_EXPORT Error_t Free(void* devPtr);
FLAGOS_EXPORT Error_t MallocHost(void** hostPtr, size_t size);
FLAGOS_EXPORT Error_t FreeHost(void* hostPtr);
FLAGOS_EXPORT Error_t
Memcpy(void* dst, const void* src, size_t count, MemcpyKind kind);
FLAGOS_EXPORT Error_t MemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    MemcpyKind kind,
    Stream_t stream);
FLAGOS_EXPORT Error_t
PointerGetAttributes(PointerAttributes* attributes, const void* ptr);
FLAGOS_EXPORT Error_t Memset(void* devPtr, int value, size_t count);
FLAGOS_EXPORT Error_t MemsetAsync(void* devPtr, int value, size_t count, Stream_t stream);

// Device
FLAGOS_EXPORT Error_t GetDeviceCount(int* count);
FLAGOS_EXPORT Error_t SetDevice(int device);
FLAGOS_EXPORT Error_t GetDevice(int* device);
FLAGOS_EXPORT Error_t
DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
FLAGOS_EXPORT Error_t DeviceSynchronize(void);

// Stream
FLAGOS_EXPORT Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int priority);
FLAGOS_EXPORT Error_t StreamCreate(Stream_t* stream);
FLAGOS_EXPORT Error_t StreamGetPriority(Stream_t stream, int* priority);
FLAGOS_EXPORT Stream_t GetDefaultStreamForDevice(int device);
FLAGOS_EXPORT Stream_t GetCurrentStreamForDevice(int device);
FLAGOS_EXPORT Error_t SetCurrentStreamForDevice(int device, Stream_t stream);
FLAGOS_EXPORT Error_t StreamDestroy(Stream_t stream);
FLAGOS_EXPORT Error_t StreamQuery(Stream_t stream);
FLAGOS_EXPORT Error_t StreamSynchronize(Stream_t stream);
FLAGOS_EXPORT Error_t
StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags);

// Event
FLAGOS_EXPORT Error_t
EventCreateWithFlags(Event_t* event, unsigned int flags);
FLAGOS_EXPORT Error_t EventCreate(Event_t* event);
FLAGOS_EXPORT Error_t EventDestroy(Event_t event);
FLAGOS_EXPORT Error_t EventRecord(Event_t event, Stream_t stream);
FLAGOS_EXPORT Error_t EventSynchronize(Event_t event);
FLAGOS_EXPORT Error_t EventQuery(Event_t event);
FLAGOS_EXPORT Error_t
EventElapsedTime(float* ms, Event_t start, Event_t end);

#ifdef __cplusplus
} // extern "C"
#endif
