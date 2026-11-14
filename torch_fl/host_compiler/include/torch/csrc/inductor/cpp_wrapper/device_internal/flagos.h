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

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/core/DeviceGuard.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
extern "C" void* GetCurrentStreamForDevice(int device_index);
static inline AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** stream) {
  *stream = GetCurrentStreamForDevice(device_index);
  return AOTI_TORCH_SUCCESS;
}
struct AOTIFlagosDeviceGuard {
  c10::DeviceGuard guard;
  explicit AOTIFlagosDeviceGuard(int32_t device_index)
      : guard(c10::Device(c10::kPrivateUse1, device_index)) {}
};
#define AOTICudaGuard AOTIFlagosDeviceGuard
