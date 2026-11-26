// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "precision_runtime.h"

#include <ATen/native/BinaryOps.h>
#include <ATen/native/Copy.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/constant_pad_nd_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/pow_native.h>
#include <ATen/ops/sub_native.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace at::native::flagos::vendor {

inline at::Tensor Add(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha) {
  VendorScope scope(self.is_privateuseone() ? self : other);
  at::Tensor output;
  auto iterator = at::TensorIterator::binary_op(output, self, other);
  at::native::alpha_check(iterator.dtype(), alpha);
  DescribeCudaMemory(iterator);
  at::native::add_stub(at::kCUDA, iterator, alpha);
  return iterator.output();
}

inline at::Tensor Mul(const at::Tensor& self, const at::Tensor& other) {
  VendorScope scope(self.is_privateuseone() ? self : other);
  StructuredOutput<at::native::structured_mul_out> kernel;
  kernel.meta(self, other);
  DescribeCudaMemory(kernel);
  at::native::mul_stub(at::kCUDA, kernel);
  return kernel.output;
}

inline at::Tensor Sub(const at::Tensor& self, const at::Tensor& other,
                       const at::Scalar& alpha) {
  VendorScope scope(self.is_privateuseone() ? self : other);
  StructuredOutput<at::native::structured_sub_out> kernel;
  kernel.meta(self, other, alpha);
  DescribeCudaMemory(kernel);
  at::native::add_stub(at::kCUDA, kernel, -alpha);
  return kernel.output;
}

inline at::Tensor PowScalar(const at::Scalar& base, const at::Tensor& exponent) {
  VendorScope scope(exponent);
  StructuredOutput<at::native::structured_pow_Scalar_out> kernel;
  kernel.meta(base, exponent);
  kernel.impl(base, exponent, kernel.output);
  return kernel.output;
}

inline at::Tensor Cat(const at::ITensorListRef& inputs, int64_t dim) {
  TORCH_CHECK(!inputs.empty(), "torch.cat expects a nonempty tensor list");
  const auto& first = inputs.front();
  for (const auto& input : inputs) {
    TORCH_CHECK(input.device() == first.device(),
                "torch.cat expects inputs on the same device");
  }
  VendorScope scope(first);
  StructuredOutput<at::native::structured_cat_out_cuda> kernel;
  const auto meta = kernel.meta(inputs, dim);
  kernel.impl(inputs, meta.dim, meta.valid, meta.all_contiguous,
              meta.all_same_dtype, meta.all_same_sizes_and_stride,
              meta.memory_format, kernel.output);
  return std::move(kernel.output);
}

inline at::Tensor ConstantPadNd(const at::Tensor& input, at::IntArrayRef pad,
                               const at::Scalar& value) {
  const auto zero = !value.isComplex() && value.toDouble() == 0.0 &&
      !std::signbit(value.toDouble());
  if (!zero || input.layout() != at::kStrided || input.numel() == 0 ||
      input.is_conj() || input.is_neg() || input.has_names() ||
      pad.empty() || pad.size() % 2 != 0 || pad.size() / 2 > input.dim() ||
      std::any_of(pad.begin(), pad.end(), [](auto width) { return width < 0; }) ||
      std::none_of(pad.begin(), pad.end(), [](auto width) { return width > 0; })) {
    return at::native::constant_pad_nd(input, pad, value);
  }

  auto sizes = input.sizes().vec();
  for (size_t index = 0; index < pad.size(); ++index) {
    auto& extent = sizes[sizes.size() - 1 - index / 2];
    TORCH_CHECK(pad[index] <= std::numeric_limits<int64_t>::max() - extent,
                "padding would overflow the output extent");
    extent += pad[index];
  }
  VendorScope scope(input);
  auto output = at::empty(sizes, input.options(), input.suggest_memory_format());
  auto destination = output;
  for (size_t index = 0; index < pad.size(); index += 2) {
    const auto axis = input.dim() - 1 - static_cast<int64_t>(index / 2);
    if (pad[index] || pad[index + 1]) {
      destination = destination.narrow(axis, pad[index], input.size(axis));
    }
  }
  auto copy = at::TensorIteratorConfig()
      .add_output(destination)
      .add_const_input(input)
      .resize_outputs(false)
      .build();
  DescribeCudaMemory(copy);
  const auto queue = reinterpret_cast<cudaStream_t>(
      GetCurrentStreamForDevice(input.device().index()));
  C10_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.nbytes(), queue));
  at::native::copy_stub(at::kCUDA, copy, false);
  return output;
}

}  // namespace at::native::flagos::vendor
