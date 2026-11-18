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

#include <mutex>
#include <unordered_map>

namespace {

// The non-boxing runtime owns these queues. Keep the registry out of headers
// so the allocator, device guard and prebuilt launcher share one instance.
std::mutex stream_mutex;
std::unordered_map<int, cudaStream_t> default_streams;
thread_local std::unordered_map<int, cudaStream_t> current_streams;

cudaStream_t DefaultStream(int device) {
  std::lock_guard<std::mutex> lock(stream_mutex);
  const auto found = default_streams.find(device);
  if (found != default_streams.end()) return found->second;
  int previous = 0;
  if (cudaGetDevice(&previous) != cudaSuccess) return nullptr;
  if (previous != device && cudaSetDevice(device) != cudaSuccess) return nullptr;
  cudaStream_t stream = nullptr;
  const auto status = cudaStreamCreate(&stream);
  if (previous != device) cudaSetDevice(previous);
  if (status != cudaSuccess) return nullptr;
  default_streams.emplace(device, stream);
  return stream;
}

}  // namespace

Stream_t GetDefaultStreamForDevice(int device) {
  return reinterpret_cast<Stream_t>(DefaultStream(device));
}

Stream_t GetCurrentStreamForDevice(int device) {
  const auto found = current_streams.find(device);
  if (found != current_streams.end()) return reinterpret_cast<Stream_t>(found->second);
  auto stream = GetDefaultStreamForDevice(device);
  if (stream) current_streams.emplace(device, reinterpret_cast<cudaStream_t>(stream));
  return stream;
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  current_streams[device] = reinterpret_cast<cudaStream_t>(stream);
  return Success;
}

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
