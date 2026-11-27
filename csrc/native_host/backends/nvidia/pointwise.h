// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace at { class Tensor; }

namespace flagos::nvidia {

enum class PointwiseOp { PowScalar, Reciprocal, Rsqrt, Silu };

at::Tensor TryPointwise(PointwiseOp op, const at::Tensor& input,
                       float scalar = 0);

}  // namespace flagos::nvidia
