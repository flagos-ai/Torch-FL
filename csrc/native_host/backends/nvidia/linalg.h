// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

namespace flagos::nvidia {

at::Tensor TryMatrixProduct(const at::Tensor& left, const at::Tensor& right,
                            at::Tensor* out = nullptr, const at::Tensor* bias = nullptr,
                            float beta = 0, float alpha = 1);

}  // namespace flagos::nvidia
