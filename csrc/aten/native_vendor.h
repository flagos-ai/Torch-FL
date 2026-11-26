// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace at::native::flagos {

void RegisterNativeVendorKernel(const std::string& vendor, const std::string& op,
                                const std::string& abi, uintptr_t address);

namespace vendor {
void ActivateNvidiaVendor();
}

}  // namespace at::native::flagos
