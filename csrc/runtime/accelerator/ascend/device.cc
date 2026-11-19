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
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstdio>

namespace {

thread_local int gCurrentDevice = 0;
thread_local bool gDeviceBound = false;

bool ensureAclInit() {
  static bool initialized = []() {
    aclError err = aclInit(nullptr);
    if (err != ACL_SUCCESS && err != ACL_ERROR_REPEAT_INITIALIZE) {
      fprintf(stderr, "[flagos-ascend] aclInit failed: %d\n", static_cast<int>(err));
      return false;
    }
    return true;
  }();
  return initialized;
}

} // namespace

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  if (!ensureAclInit()) {
    *count = 0;
    return ErrorUnknown;
  }

  uint32_t ascend_count = 0;
  aclError err = aclrtGetDeviceCount(&ascend_count);
  if (err != ACL_SUCCESS) {
    *count = 0;
    return ErrorUnknown;
  }

  *count = static_cast<int>(ascend_count);
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  // ACL's current context belongs to the calling thread. Bind each thread
  // before publishing its cached device, including threads first using device 0.
  if (!gDeviceBound) {
    const Error_t err = SetDevice(gCurrentDevice);
    if (err != Success) {
      return err;
    }
  }

  *device = gCurrentDevice;
  return Success;
}

Error_t SetDevice(int device) {
  // Allocations and guard restoration repeatedly select the current device.
  // A bound thread already owns that context; avoid two driver calls per probe.
  if (gDeviceBound && gCurrentDevice == device) {
    return Success;
  }
  if (!ensureAclInit()) {
    return ErrorUnknown;
  }

  int count = 0;
  if (GetDeviceCount(&count) != Success) {
    return ErrorUnknown;
  }

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  aclError err = aclrtSetDevice(device);
  if (err != ACL_SUCCESS) {
    return ErrorUnknown;
  }

  gCurrentDevice = device;
  gDeviceBound = true;
  return Success;
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  *leastPriority = 0;
  *greatestPriority = 0;
  return Success;
}

Error_t DeviceSynchronize(void) {
  aclError err = aclrtSynchronizeDevice();
  return (err == ACL_SUCCESS) ? Success : ErrorUnknown;
}
