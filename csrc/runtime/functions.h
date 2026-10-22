// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <flagos.h>
#include <macros.h>
#include <c10/core/Device.h>
#include <cstdint>

namespace c10::flagos {

using DeviceIndex = int8_t;

Error_t GetDeviceCount(int* dev_count);
Error_t GetDevice(DeviceIndex* device);
Error_t SetDevice(DeviceIndex device);

FLAGOS_EXPORT DeviceIndex DeviceCount() noexcept;
FLAGOS_EXPORT DeviceIndex CurrentDevice();
FLAGOS_EXPORT void SetCurrentDevice(DeviceIndex device);
FLAGOS_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device);

} // namespace c10::flagos
