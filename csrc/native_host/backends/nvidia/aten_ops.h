// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

#include <optional>

namespace at::native::flagos::vendor {

at::Tensor NvidiaConvolution(const at::Tensor& input, const at::Tensor& weight,
                             const std::optional<at::Tensor>& bias,
                             at::IntArrayRef stride, at::IntArrayRef padding,
                             at::IntArrayRef dilation, int64_t groups);

at::Tensor NvidiaMul(const at::Tensor& self, const at::Tensor& other);

at::Tensor NvidiaDiv(const at::Tensor& self, const at::Tensor& divisor);
at::Tensor NvidiaDiv(const at::Tensor& self, const at::Scalar& divisor);
at::Tensor& NvidiaDivInplace(at::Tensor& self, const at::Tensor& divisor);
at::Tensor& NvidiaDivInplace(at::Tensor& self, const at::Scalar& divisor);

at::Tensor& NvidiaMmOut(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out);

at::Tensor NvidiaVectorNorm(const at::Tensor& input, const at::Scalar& order,
                            at::OptionalIntArrayRef dim, bool keepdim,
                            std::optional<at::ScalarType> dtype);

at::Tensor NvidiaScatterAdd(const at::Tensor& self, int64_t dim,
                            const at::Tensor& index, const at::Tensor& source);

at::Tensor& NvidiaScatterAddInplace(at::Tensor& self, int64_t dim,
                                   const at::Tensor& index, const at::Tensor& source);

at::Tensor& NvidiaIndexPut(at::Tensor& self,
                          const c10::List<std::optional<at::Tensor>>& indices,
                          const at::Tensor& values, bool accumulate, bool unsafe);

}  // namespace at::native::flagos::vendor
