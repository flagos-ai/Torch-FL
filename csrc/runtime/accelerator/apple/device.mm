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

// Apple Metal device layer.
//
// An Apple Silicon Mac has exactly one GPU, so this presents a single device 0
// rather than inventing indices that would imply separate memory spaces. The
// count is 0 on a Mac with no MPS-capable device, which is what torch should
// see -- reporting a device that fails on first allocation is worse than
// reporting none.

#include "mps_runtime.h"

namespace {

// Kept for API symmetry with the other backends; only index 0 is ever valid.
thread_local int gCurrentDevice = 0;

} // namespace

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }
  *count = torch_fl_mps::MetalAvailable() ? 1 : 0;
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }
  *device = gCurrentDevice;
  return Success;
}

Error_t SetDevice(int device) {
  int count = 0;
  GetDeviceCount(&count);
  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }
  gCurrentDevice = device;
  return Success;
}

Error_t DeviceGetStreamPriorityRange(
    int* leastPriority,
    int* greatestPriority) {
  // MTLCommandQueue has no priority knob (unlike CUDA's stream priority), so
  // report a range of zero rather than a value no queue would honour.
  if (leastPriority) {
    *leastPriority = 0;
  }
  if (greatestPriority) {
    *greatestPriority = 0;
  }
  return Success;
}

Error_t DeviceSynchronize(void) {
  // Ops encode into MTLCommandBuffers that are committed but not necessarily
  // finished, so a device-wide wait means draining every live queue.
  torch_fl_mps::DrainAll();
  return Success;
}
