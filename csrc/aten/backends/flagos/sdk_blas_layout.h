// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace at::native::flagos::sdk_blas {

inline int Dimension(int64_t value) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              "native SDK BLAS dimension exceeds its integer ABI");
  return static_cast<int>(value);
}

struct Operand {
  at::Tensor tensor;
  bool transpose;
  int leading;
};

inline Operand Describe(at::Tensor value) {
  const auto rows = value.size(-2), columns = value.size(-1);
  if (value.stride(-1) == 1 && value.stride(-2) >= std::max<int64_t>(columns, 1))
    return {value, false, Dimension(value.stride(-2))};
  if (value.stride(-2) == 1 && value.stride(-1) >= std::max<int64_t>(rows, 1))
    return {value, true, Dimension(value.stride(-1))};
  return {value.contiguous(), false, Dimension(std::max<int64_t>(columns, 1))};
}

inline std::pair<Operand, Operand> ColumnMajorOperands(
    const at::Tensor& left, const at::Tensor& right, int m, int n, int k) {
  // C(row-major)^T = B^T A^T in a column-major API.
  auto a = Describe(right), b = Describe(left);
  if (!a.transpose && k <= 1) a.leading = std::max(n, 1);
  if (a.transpose && n <= 1) a.leading = std::max(k, 1);
  if (!b.transpose && m <= 1) b.leading = std::max(k, 1);
  if (b.transpose && k <= 1) b.leading = std::max(m, 1);
  return {std::move(a), std::move(b)};
}

}  // namespace at::native::flagos::sdk_blas
