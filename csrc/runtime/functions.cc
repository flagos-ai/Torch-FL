// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <flagos.h>

#include "exception.h"
#include "functions.h"

#include <c10/util/Exception.h>
#include <limits>

namespace c10::flagos {

Error_t GetDeviceCount(int* dev_count) {
  return ::GetDeviceCount(dev_count);
}

Error_t GetDevice(DeviceIndex* device) {
  int tmp_device = -1;
  auto err = ::GetDevice(&tmp_device);
  *device = static_cast<DeviceIndex>(tmp_device);
  return err;
}

Error_t SetDevice(DeviceIndex device) {
  int cur_device = -1;
  ::GetDevice(&cur_device);
  if (device == cur_device) {
    return Success;
  }
  return ::SetDevice(device);
}

int DeviceCountImpl() {
  int count = 0;
  GetDeviceCount(&count);
  return count;
}

FLAGOS_EXPORT DeviceIndex DeviceCount() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = DeviceCountImpl();
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      TORCH_WARN("Device initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

FLAGOS_EXPORT DeviceIndex CurrentDevice() {
  DeviceIndex cur_device = -1;
  GetDevice(&cur_device);
  return cur_device;
}

FLAGOS_EXPORT void SetCurrentDevice(DeviceIndex device) {
  SetDevice(device);
}

FLAGOS_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device) {
  int current_dev = -1;
  ::GetDevice(&current_dev);

  if (device != current_dev) {
    ::SetDevice(device);
  }

  return static_cast<DeviceIndex>(current_dev);
}

} // namespace c10::flagos
