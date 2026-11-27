// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace at { class Tensor; }

namespace flagos::nvidia {

enum class PointwiseOp { PowScalar, Reciprocal, Rsqrt, Silu };

at::Tensor TryPointwise(PointwiseOp op, const at::Tensor& input,
                       float scalar = 0);
at::Tensor TryDivide(const at::Tensor& left, const at::Tensor& right);

struct DivisionLayout {
  static constexpr int max_dimensions = 8;
  int dimensions;
  int64_t sizes[max_dimensions];
  int64_t strides[3][max_dimensions];
};

}  // namespace flagos::nvidia
