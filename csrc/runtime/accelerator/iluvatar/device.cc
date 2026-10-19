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

// Iluvatar corex device API. This is a direct build of the CUDA runtime
// sources against corex's CUDA-shaped C runtime (cuda_runtime.h at
// <corex>/include, linked as libcudart.so.10.2) -- unlike the CUDA platform it
// is compiled against no torch at all, so nothing here may reach c10::cuda.
// See stream.cc for why that matters.

#include <flagos.h>
#include <cuda_runtime.h>
#include "iluvatar_context.h"

Error_t GetDeviceCount(int* count) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
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

// Ask the driver rather than caching the index ourselves. cudaSetDevice's
// current device is already thread-local, so a shadow copy here can only
// diverge from it: anything that moves the corex driver's index without going
// through our SetDevice leaves the cached value stale, and a DeviceGuard that
// then reads it as "previous" restores the process to a device that was never
// current. (On the CUDA platform the thing that does this is
// torch.cuda.set_device(); iluvatar has no such in-process caller, but the
// registry is also reachable from C++ that sets the device directly, and a
// second source of truth is wrong with or without torch.)
Error_t GetDevice(int* device) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
  if (!device) {
    return ErrorUnknown;
  }

  cudaError_t err = cudaGetDevice(device);
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}

Error_t SetDevice(int device) {
  c10::flagos::iluvatar::EnsureIluvatarContext();
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
  cudaError_t err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}
