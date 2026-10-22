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
#include <xpu/runtime.h>

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  int xpu_count = 0;
  if (xpu_device_count(&xpu_count) != 0) {
    *count = 0;
    return ErrorUnknown;
  }

  *count = xpu_count;
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  if (xpu_current_device(device) != 0) {
    return ErrorUnknown;
  }
  return Success;
}

Error_t SetDevice(int device) {
  int count = 0;
  GetDeviceCount(&count);

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  return (xpu_set_device(device) == 0) ? Success : ErrorUnknown;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  // The XPU SDK exposes neither stream priorities nor a query for the range.
  // Reporting the neutral single-priority answer is what the caller acts on:
  // there is one priority level, so nothing to choose.
  *leastPriority = 0;
  *greatestPriority = 0;
  return Success;
}

Error_t DeviceSynchronize(void) {
  // NULL means "the default stream of the current device" in the XPU runtime.
  return (xpu_wait(NULL) == 0) ? Success : ErrorUnknown;
}
