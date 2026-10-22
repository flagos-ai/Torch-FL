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

// BPU device layer.
//
// The board has four BPU cores (/dev/bpu_core0-3) but they are not independent
// devices: hb_bpu_core_open() takes a core *mask* and a scheduling policy, so
// core selection is the driver's job, not the caller's. Presenting one device
// keeps torch's device indices honest -- claiming four would imply four separate
// memory spaces, which is wrong, since UCP memory is allocated for the SoC.

#include <flagos.h>

extern "C" {
#include <hb_bpu.h>
}

namespace {

// Kept for API symmetry with the other backends; only index 0 is ever valid.
thread_local int gCurrentDevice = 0;

} // namespace

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }
  // Report a device only when the BPU is actually usable. hb_bpu_core_num()
  // returns 0 when the driver is absent, which is what torch should see rather
  // than a device that fails on first allocation.
  *count = (hb_bpu_core_num() > 0) ? 1 : 0;
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
  // The BPU driver does have task priorities (hb_bpu_task_set_prio), but they
  // apply per submitted task, not per stream. Report no range.
  if (leastPriority) {
    *leastPriority = 0;
  }
  if (greatestPriority) {
    *greatestPriority = 0;
  }
  return Success;
}

Error_t DeviceSynchronize(void) {
  // Task submission goes through hbUCPSubmitTask/hbUCPWaitTaskDone, which the
  // runtime already waits on before returning, so there is never outstanding
  // work to drain at this level.
  return Success;
}
