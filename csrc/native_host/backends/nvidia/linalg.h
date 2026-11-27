// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

namespace flagos::nvidia {

at::Tensor TryMatrixProduct(const at::Tensor& left, const at::Tensor& right,
                            at::Tensor* out = nullptr);

}  // namespace flagos::nvidia
