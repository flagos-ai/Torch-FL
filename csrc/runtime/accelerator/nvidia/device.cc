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

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  // CUDA fixes the visible device set when the process initializes the runtime.
  // With exactly one visible device, even a CUDA caller outside FlagOS can
  // only select ordinal zero. Keep reading the driver in multi-device jobs.
  static const bool single_device = [] {
    int count = 0;
    return GetDeviceCount(&count) == Success && count == 1;
  }();
  if (single_device) {
    *device = 0;
    return Success;
  }

  // torch.cuda.set_device() can change the thread's current device without
  // going through FlagOS. A multi-device guard must read the driver's value.
  const auto err = cudaGetDevice(device);
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
  cudaError_t err = cudaDeviceSynchronize();
  return (err == cudaSuccess) ? Success : ErrorUnknown;
}
