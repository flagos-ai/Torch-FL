// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

namespace flagos::bindings {

at::Tensor FlagosToCudaView(const at::Tensor& source);
bool TryRegisterNvidiaFlashAttention3();

} // namespace flagos::bindings
