// Copyright (c) 2026, BAAI. All rights reserved.
#pragma once

#include <ATen/core/Tensor.h>
#include <array>
#include <tuple>

namespace at::native::flagos {
std::tuple<at::Tensor, at::Tensor> MatmulBackwardKernelAscend(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    std::array<bool, 2> mask);
} // namespace at::native::flagos
