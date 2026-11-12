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
#include <cstdint>

#include <mutex>
#include <unordered_map>

// The CUDA stream registry.
//
// Everything else in this file talks to CUDA's C runtime and to nothing else, and the
// registry keeps it that way. Without it the backend answers "which stream is current on
// this device" by asking torch (c10::cuda::getCurrentCUDAStream), which makes every nvidia
// build link libtorch_cuda.so and libc10_cuda.so -- neither of which a CPU torch wheel
// ships. A build that links them cannot be loaded by an interpreter running torch+cpu, so
// the platform would have no compliant test environment at all. Iluvatar, Kunlunxin and
// Cambricon answer this the same way (runtime/accelerator/iluvatar/stream.cc records the
// reasoning at length); this is the CUDA arm of it.
//
// Two registries do the work:
//
//   * a per-device default stream, created once and shared by every caller, so two ops
//     dispatched back to back are ordered without an explicit barrier;
//   * a thread-local "current" override per device, which is what setCurrentStream /
//     exchangeStream in runtime/guard.h move.
//
// Both are process-wide state and therefore defined here, out of line in libflagos.so. An
// `inline` function-local static would give each shared object its own copy under
// -fvisibility=hidden, and two "default" streams means Triton and the operator kernels are
// never ordered against each other.
namespace {

std::mutex stream_mutex;
std::unordered_map<int, cudaStream_t> default_streams;
thread_local std::unordered_map<int, cudaStream_t> current_streams;

// A stream is bound to the device that was current when it was created, so the creating
// call has to run on `device` even when another one is current -- which is the normal case,
// since the registry is queried for a device other than the caller's on every multi-device
// run.
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
    // Do NOT cache the failure: a cached null would pin an unusable default stream for this
    // device for the rest of the process, with no retry after a transient create failure
    // clears. Returning without inserting lets the next lookup try again.
    return nullptr;
  }

  default_streams.emplace(device, stream);
  return stream;
}

}  // namespace

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
  current_streams[device] = reinterpret_cast<cudaStream_t>(stream);
  return Success;
}

Error_t StreamCreateWithPriority(
    Stream_t* stream,
    unsigned int flags,
    int priority) {
  cudaError_t err = cudaStreamCreateWithPriority(
      (cudaStream_t*)stream, flags, priority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamCreate(Stream_t* stream) {
  cudaError_t err = cudaStreamCreate((cudaStream_t*)stream);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamGetPriority(Stream_t stream, int* priority) {
  cudaError_t err = cudaStreamGetPriority((cudaStream_t)stream, priority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamDestroy(Stream_t stream) {
  cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamQuery(Stream_t stream) {
  cudaError_t err = cudaStreamQuery((cudaStream_t)stream);
  if (err == cudaSuccess) {
    return Success;
  } else if (err == cudaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t StreamSynchronize(Stream_t stream) {
  cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t StreamWaitEvent(Stream_t stream, Event_t event, unsigned int flags) {
  cudaError_t err = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, flags);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreateWithFlags(Event_t* event, unsigned int flags) {
  cudaError_t err = cudaEventCreateWithFlags((cudaEvent_t*)event, flags);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventCreate(Event_t* event) {
  cudaError_t err = cudaEventCreate((cudaEvent_t*)event);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventDestroy(Event_t event) {
  cudaError_t err = cudaEventDestroy((cudaEvent_t)event);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventRecord(Event_t event, Stream_t stream) {
  cudaError_t err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventSynchronize(Event_t event) {
  cudaError_t err = cudaEventSynchronize((cudaEvent_t)event);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t EventQuery(Event_t event) {
  cudaError_t err = cudaEventQuery((cudaEvent_t)event);
  if (err == cudaSuccess) {
    return Success;
  } else if (err == cudaErrorNotReady) {
    return ErrorNotReady;
  }
  return ErrorUnknown;
}

Error_t EventElapsedTime(float* ms, Event_t start, Event_t end) {
  cudaError_t err = cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

#if defined(USE_DCU)
// The flagos stream handle for the stream csrc/ synchronizes, encoded in
// ``c10::Stream::id()`` exactly as the Kunlunxin and Iluvatar arms encode theirs.
// It lives in c10::flagos because that is where guard.h declares it, next to the
// interface that calls it; the namespace is spelled out instead of entering it
// through <c10/core/Stream.h>, since this library keeps c10 out of its link set.
namespace c10::flagos {
__attribute__((visibility("default"))) Stream_t DcuStreamHandleOf(uintptr_t id) {
  return reinterpret_cast<Stream_t>(id);
}
} // namespace c10::flagos
#endif
