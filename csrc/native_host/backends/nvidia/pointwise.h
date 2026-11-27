// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

namespace at { class Tensor; }
namespace c10 { template <typename T> class ArrayRef; }

namespace flagos::nvidia {

enum class PointwiseOp { PowScalar, Reciprocal, Rsqrt, Silu, Sqrt };

at::Tensor TryPointwise(PointwiseOp op, const at::Tensor& input,
                       float scalar = 0);
at::Tensor TryDivide(const at::Tensor& left, const at::Tensor& right);
at::Tensor TryMultiply(const at::Tensor& left, const at::Tensor& right);
at::Tensor TryMeanLastDim(const at::Tensor& input, bool keepdim);
at::Tensor TryChannelNorm(const at::Tensor& input, bool keepdim);
std::tuple<at::Tensor, at::Tensor, at::Tensor> TryLayerNorm(
    const at::Tensor& input, c10::ArrayRef<int64_t> normalized_shape,
    const at::Tensor& weight, const at::Tensor& bias, double epsilon);

struct PointwiseLayout {
  static constexpr int max_dimensions = 8;
  int dimensions;
  int64_t sizes[max_dimensions];
  int64_t strides[3][max_dimensions];
};

}  // namespace flagos::nvidia
