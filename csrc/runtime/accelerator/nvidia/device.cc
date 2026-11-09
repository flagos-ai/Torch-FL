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

namespace {
// The stream currently being captured by flagos.CUDAGraph, or nullptr when not
// capturing. Set via SetStreamCapturing from the Python graph wrapper; read back
// by GetStreamCapturing (declared in flagos.h) both here and in the torch_fl
// dispatch layer's index bounds check.
Stream_t g_capture_stream = nullptr;
} // namespace

void SetStreamCapturing(Stream_t stream) {
  g_capture_stream = stream;
}

Stream_t GetStreamCapturing(void) {
  return g_capture_stream;
}

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  int cuda_count = 0;
  cudaError_t err = cudaGetDeviceCount(&cuda_count);
  if (err != cudaSuccess) {
    *count = 0;
    return ErrorUnknown;
  }

  *count = cuda_count;
  return Success;
}

// Ask the driver rather than caching the index ourselves. flagos aliases the
// same physical device as torch's CUDA backend, and cudaSetDevice's current
// device is already thread-local, so a shadow copy here can only diverge from
// it: torch.cuda.set_device() moves the driver's index without going through
// our SetDevice, leaving the cached value stale. Any DeviceGuard that then
// reads the stale value as "previous" restores the process to a device that was
// never current -- observed as a flagos allocation resetting
// torch.cuda.current_device() back to 0.
Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  cudaError_t err = cudaGetDevice(device);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t SetDevice(int device) {
  int count = 0;
  GetDeviceCount(&count);

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    return ErrorUnknown;
  }

  return Success;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  cudaError_t err = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t DeviceSynchronize(void) {
  // A device-wide synchronize invalidates HIP stream capture. During capture the
  // flagos backend has no per-stream synchronize to fall back on, so skip it:
  // the capture stream's own ordering already provides the sequencing the caller
  // needs, and replay will enforce it.
  if (GetStreamCapturing() != nullptr) {
    return Success;
  }
  cudaError_t err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}
