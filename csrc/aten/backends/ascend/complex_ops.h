// Copyright (c) 2026, BAAI. All rights reserved.
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>

namespace at::native::flagos::ascend {
// Implementations are emitted by scripts/codegen/codegen_ascend.py.
at::Tensor ComplexIndex(const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices);
at::Tensor ComplexMatmul(const at::Tensor& left, const at::Tensor& right);
at::Tensor RealMatmulDouble(const at::Tensor& left, const at::Tensor& right);
} // namespace at::native::flagos::ascend
