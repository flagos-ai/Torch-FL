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

// Cambricon MLU device layer, over libcnrt.
//
// torch_fl owns the device rather than borrowing torch_mlu's: the wheel runs on
// the stock torch+cpu and the vendor plugin is deliberately not installed, so
// there is nothing else in the process that could initialise the runtime on
// our behalf. Everything here is the thin cnrt counterpart of the CUDA arm.
//
// Device selection is process-global in cnrt (cnrtSetDevice), the same model the
// CUDA arm has (cudaSetDevice). Torch's guard machinery serialises around it.

#include <flagos.h>

#include <cnrt.h>

#include <cstdio>

namespace {

// Maps a cnrt return code onto the flagos error enum. cnrt reports "not ready"
// separately for queues and notifiers, and that is the one distinction callers
// act on (StreamQuery/EventQuery), so it keeps its own arm.
Error_t toError(cnrtRet_t status) {
  switch (status) {
    case cnrtSuccess:
      return Success;
    default:
      return ErrorUnknown;
  }
}

} // namespace

Error_t GetDeviceCount(int* count) {
  if (!count) {
    return ErrorUnknown;
  }

  unsigned int mlu_count = 0;
  const cnrtRet_t status = cnrtGetDeviceCount(&mlu_count);
  if (status != cnrtSuccess) {
    *count = 0;
    fprintf(
        stderr,
        "[flagos-cambricon] cnrtGetDeviceCount failed: %d\n",
        static_cast<int>(status));
    return toError(status);
  }

  *count = static_cast<int>(mlu_count);
  return Success;
}

Error_t GetDevice(int* device) {
  if (!device) {
    return ErrorUnknown;
  }

  int ordinal = -1;
  const cnrtRet_t status = cnrtGetDevice(&ordinal);
  if (status != cnrtSuccess) {
    return toError(status);
  }

  *device = ordinal;
  return Success;
}

Error_t SetDevice(int device) {
  int count = 0;
  if (GetDeviceCount(&count) != Success) {
    return ErrorUnknown;
  }

  if (device < 0 || device >= count) {
    return ErrorInvalidDevice;
  }

  const cnrtRet_t status = cnrtSetDevice(device);
  if (status != cnrtSuccess) {
    fprintf(
        stderr,
        "[flagos-cambricon] cnrtSetDevice(%d) failed: %d\n",
        device,
        static_cast<int>(status));
  }
  return toError(status);
}

Error_t DeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  // cnrt clamps an out-of-range priority into the meaningful band and offers no
  // API that reports the band, so there is nothing to query. The neutral answer
  // -- one priority level -- is what a caller can act on: it will not try to
  // order work by priority.
  if (!leastPriority || !greatestPriority) {
    return ErrorUnknown;
  }
  *leastPriority = 0;
  *greatestPriority = 0;
  return Success;
}

Error_t DeviceSynchronize(void) {
  return toError(cnrtSyncDevice());
}