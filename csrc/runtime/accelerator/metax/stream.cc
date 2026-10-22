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
#include <cuda_runtime.h>

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int priority) {
  const cudaError_t err = cudaStreamCreateWithPriority(
      reinterpret_cast<cudaStream_t*>(stream), flags, priority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  const cudaError_t err = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(stream));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t stream, int* priority) {
  const cudaError_t err =
      cudaStreamGetPriority(reinterpret_cast<cudaStream_t>(stream), priority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamDestroy(Stream_t stream) {
  const cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  const cudaError_t err = cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream));
  if (err == cudaSuccess) {
    return Success;
  }
  if (err == cudaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t StreamSynchronize(Stream_t stream) {
  const cudaError_t err =
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  const cudaError_t err = cudaStreamWaitEvent(
      reinterpret_cast<cudaStream_t>(stream),
      reinterpret_cast<cudaEvent_t>(event),
      flags);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  const cudaError_t err =
      cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t*>(event), flags);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  const cudaError_t err = cudaEventCreate(reinterpret_cast<cudaEvent_t*>(event));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  const cudaError_t err = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  const cudaError_t err = cudaEventRecord(
      reinterpret_cast<cudaEvent_t>(event),
      reinterpret_cast<cudaStream_t>(stream));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  const cudaError_t err =
      cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  const cudaError_t err = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (err == cudaSuccess) {
    return Success;
  }
  if (err == cudaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  const cudaError_t err = cudaEventElapsedTime(
      ms,
      reinterpret_cast<cudaEvent_t>(start),
      reinterpret_cast<cudaEvent_t>(end));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}
