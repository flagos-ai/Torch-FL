// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

#include <optional>

namespace flagos::nvidia {

at::Tensor TryCuDnnAttention(const at::Tensor& query, const at::Tensor& key,
                             const at::Tensor& value, bool causal,
                             std::optional<double> scale, bool enable_gqa);

}  // namespace flagos::nvidia
