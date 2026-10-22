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
#include <musa_runtime.h>

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int priority) {
  musaError_t err = musaStreamCreateWithPriority(
      (musaStream_t*)stream, flags, priority);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  musaError_t err = musaStreamCreate((musaStream_t*)stream);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t stream, int* priority) {
  musaError_t err = musaStreamGetPriority((musaStream_t)stream, priority);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamDestroy(Stream_t stream) {
  musaError_t err = musaStreamDestroy((musaStream_t)stream);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  musaError_t err = musaStreamQuery((musaStream_t)stream);
  if (err == musaSuccess) {
    return Success;
  } else if (err == musaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t StreamSynchronize(Stream_t stream) {
  musaError_t err = musaStreamSynchronize((musaStream_t)stream);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  musaError_t err = musaStreamWaitEvent((musaStream_t)stream, (musaEvent_t)event, flags);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  musaError_t err = musaEventCreateWithFlags((musaEvent_t*)event, flags);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  musaError_t err = musaEventCreate((musaEvent_t*)event);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  musaError_t err = musaEventDestroy((musaEvent_t)event);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  musaError_t err = musaEventRecord((musaEvent_t)event, (musaStream_t)stream);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  musaError_t err = musaEventSynchronize((musaEvent_t)event);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  musaError_t err = musaEventQuery((musaEvent_t)event);
  if (err == musaSuccess) {
    return Success;
  } else if (err == musaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  musaError_t err = musaEventElapsedTime(ms, (musaEvent_t)start, (musaEvent_t)end);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}
