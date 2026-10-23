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

// Moore Threads MUSA runtime. Unlike MetaX/DCU there is no libcudart shim in
// the MUSA toolkit, so this is a direct port of the CUDA runtime sources onto
// the native musa* API (musa_runtime_api.h), which mirrors CUDA 1:1.

#include <flagos.h>
#include <musa_runtime.h>

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  int musa_count = 0;
  musaError_t err = musaGetDeviceCount(&musa_count);
  if (err != musaSuccess) {
    *count = 0;
    return ErrorUnknown;
  }

  *count = musa_count;
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  musaError_t err = musaGetDevice(device);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t SetDevice(int device) {
  int count = 0;
  GetDeviceCount(&count);

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  musaError_t err = musaSetDevice(device);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  musaError_t err =
      musaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
  return (err == musaSuccess) ? Success : ErrorUnknown;
}

Error_t DeviceSynchronize(void) {
  musaError_t err = musaDeviceSynchronize();
  return (err == musaSuccess) ? Success : ErrorUnknown;
}
