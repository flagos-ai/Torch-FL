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
#include <tops_runtime_api.h>

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  int tops_count = 0;
  topsError_t err = topsGetDeviceCount(&tops_count);
  if (err != topsSuccess) {
    *count = 0;
    return ErrorUnknown;
  }

  *count = tops_count;
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  topsError_t err = topsGetDevice(device);
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t SetDevice(int device) {
  int count = 0;
  GetDeviceCount(&count);

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  topsError_t err = topsSetDevice(device);
  return (err == topsSuccess) ? Success : ErrorUnknown;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  // The tops runtime exposes no stream priority API.
  *leastPriority = 0;
  *greatestPriority = 0;
  return Success;
}

Error_t DeviceSynchronize(void) {
  topsError_t err = topsDeviceSynchronize();
  return (err == topsSuccess) ? Success : ErrorUnknown;
}
