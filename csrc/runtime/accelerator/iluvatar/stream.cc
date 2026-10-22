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

// The Iluvatar corex stream registry.
//
// Everything else in this file talks to corex's CUDA-shaped C runtime
// (cuda_runtime.h, linked as libcudart.so.10.2) and to nothing else. That is the
// point of the file existing: the backend used to take its current stream from
// torch (`c10::cuda::getCurrentCUDAStream`), which made every iluvatar build link
// libtorch_cuda.so and libc10_cuda.so -- neither of which a CPU torch wheel
// ships. A build that links them cannot be loaded by an interpreter running
// torch 2.x+cpu, so the platform had no compliant test environment at all.
//
// The shape is Kunlunxin's (runtime/accelerator/kunlunxin/stream.cc), because the
// problem is the same one: a vendor runtime with no torch behind it still has to
// answer "which stream is current on this device". Two registries do that:
//
//   * a per-device default stream, created once and shared by every caller, so
//     two ops dispatched back to back are ordered without an explicit barrier;
//   * a thread-local "current" override per device, which is what
//     setCurrentStream / exchangeStream in runtime/guard.h move.
//
// Both are process-wide state and therefore defined here, out of line in
// libflagos.so. An `inline` function-local static would give each shared object
// its own copy under -fvisibility=hidden; musa_stream.h records the measured
// consequences of exactly that mistake (two "default" streams, Triton and the
// vendor kernels never ordered against each other).

#include <flagos.h>
#include <cuda_runtime.h>
#include "iluvatar_context.h"

#include <mutex>
#include <unordered_map>

namespace {

std::mutex stream_mutex;
std::unordered_map<int, cudaStream_t> default_streams;
thread_local std::unordered_map<int, cudaStream_t> current_streams;

// A stream is bound to the device that was current when it was created, so the
// creating call has to run on `device` even when another one is current --
// which is the normal case, since the registry is queried for a device other
// than the caller's on every multi-device run.
cudaStream_t get_default_stream(int device) {
  std::lock_guard<std::mutex> lock(stream_mutex);
  auto it = default_streams.find(device);
  if (it != default_streams.end()) {
    return it->second;
  }

  int previous_device = 0;
  if (cudaGetDevice(&previous_device) != cudaSuccess) {
    return nullptr;
  }
  if (previous_device != device && cudaSetDevice(device) != cudaSuccess) {
    return nullptr;
  }

  cudaStream_t stream = nullptr;
  const cudaError_t create_err = cudaStreamCreate(&stream);

  if (previous_device != device) {
    cudaSetDevice(previous_device);
  }

  if (create_err != cudaSuccess) {
    // Do NOT cache the failure: a cached null would pin an unusable default
    // stream for this device for the rest of the process, with no retry after a
    // transient create failure clears. Returning without inserting lets the next
    // lookup try again.
    return nullptr;
  }

  default_streams.emplace(device, stream);
  return stream;
}

} // namespace

Stream_t GetDefaultStreamForDevice(int device) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  return reinterpret_cast<Stream_t>(get_default_stream(device));
}

Stream_t GetCurrentStreamForDevice(int device) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  auto it = current_streams.find(device);
  if (it != current_streams.end()) {
    return reinterpret_cast<Stream_t>(it->second);
  }
  return GetDefaultStreamForDevice(device);
}

Error_t SetCurrentStreamForDevice(int device, Stream_t stream) {
  current_streams[device] = reinterpret_cast<cudaStream_t>(stream);
  return Success;
}

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int priority) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaStreamCreateWithPriority(
      reinterpret_cast<cudaStream_t*>(stream), flags, priority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(stream));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t stream, int* priority) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err =
      cudaStreamGetPriority(reinterpret_cast<cudaStream_t>(stream), priority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamDestroy(Stream_t stream) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream));
  if (err == cudaSuccess) {
    return Success;
  } else if (err == cudaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t StreamSynchronize(Stream_t stream) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err =
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  cudaError_t err = cudaStreamWaitEvent(
      reinterpret_cast<cudaStream_t>(stream),
      reinterpret_cast<cudaEvent_t>(event),
      flags);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}