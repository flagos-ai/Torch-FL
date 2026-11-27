// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "pointwise.h"

#include <ATen/TensorIterator.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/DeviceGuard.h>
#include <cuda_runtime_api.h>
#include <flagos.h>

#include <limits>

namespace flagos::nvidia {

cudaError_t LaunchPointwise(PointwiseOp op, bool bf16, const void* input, void* output,
                            const int64_t size, float scalar, cudaStream_t stream);
cudaError_t LaunchDivide(const float* left, const float* right, float* out,
                         int64_t size, DivisionLayout layout, cudaStream_t stream);
cudaError_t LaunchMeanLastDim(const float* input, float* output, int rows, int width,
                              float scale, cudaStream_t stream);
cudaError_t LaunchChannelNorm(const void* input, void* output, int batches,
                              int channels, int cells, cudaStream_t stream);

at::Tensor TryPointwise(PointwiseOp op, const at::Tensor& input, float scalar) {
  const auto dtype = input.scalar_type();
  const bool bf16 = dtype == at::kBFloat16;
  if (!input.device().is_privateuseone() || input.layout() != at::kStrided ||
      input.is_neg() || input.has_names() || !input.is_non_overlapping_and_dense() ||
      (dtype != at::kFloat && !(bf16 && op == PointwiseOp::Silu))) {
    return {};
  }
  const c10::OptionalDeviceGuard guard(input.device());
  auto output = at::empty_like(input, {}, c10::MemoryFormat::Preserve);
  if (input.numel() == 0) return output;
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(input.device().index()));
  const auto status = LaunchPointwise(op, bf16, input.const_data_ptr(), output.mutable_data_ptr(),
                                     input.numel(), scalar, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK pointwise launch failed: ",
              cudaGetErrorString(status));
  return output;
}

at::Tensor TryDivide(const at::Tensor& left, const at::Tensor& right) {
  if (left.scalar_type() != at::kFloat || right.scalar_type() != at::kFloat ||
      !left.device().is_privateuseone() || left.device() != right.device() ||
      left.layout() != at::kStrided || right.layout() != at::kStrided ||
      left.is_neg() || right.is_neg() || left.has_names() || right.has_names()) {
    return {};
  }
  at::Tensor output;
  auto iter = at::TensorIterator::binary_float_op(output, left, right);
  if (iter.numel() == 0) return iter.output();
  if (iter.ndim() > DivisionLayout::max_dimensions) return {};
  DivisionLayout layout{};
  if (!iter.is_contiguous()) {
    layout.dimensions = iter.ndim();
    for (int dim = 0; dim < iter.ndim(); ++dim) {
      layout.sizes[dim] = iter.shape()[dim];
      for (int arg = 0; arg < 3; ++arg)
        layout.strides[arg][dim] = iter.strides(arg)[dim] / sizeof(float);
    }
  }
  const c10::OptionalDeviceGuard guard(left.device());
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(left.device().index()));
  const auto status = LaunchDivide(
      static_cast<const float*>(iter.data_ptr(1)), static_cast<const float*>(iter.data_ptr(2)),
      static_cast<float*>(iter.data_ptr(0)), iter.numel(), layout, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK divide launch failed: ",
              cudaGetErrorString(status));
  return iter.output();
}

at::Tensor TryMeanLastDim(const at::Tensor& input, bool keepdim) {
  if (input.scalar_type() != at::kFloat || !input.device().is_privateuseone() ||
      input.dim() < 2 || !input.is_contiguous() || input.is_neg() || input.has_names())
    return {};
  const auto width = input.size(-1);
  if (width < 128 || width > 8160 || width % 4 ||
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16)
    return {};
  const auto rows = input.numel() / width;
  if (rows < 16 || rows > std::numeric_limits<int>::max() - 15) return {};
  auto shape = input.sizes().vec();
  if (keepdim)
    shape.back() = 1;
  else
    shape.pop_back();
  const c10::OptionalDeviceGuard guard(input.device());
  auto result = at::empty(shape, input.options());
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(input.device().index()));
  const auto scale = static_cast<float>(rows) / static_cast<float>(input.numel());
  const auto status = LaunchMeanLastDim(input.const_data_ptr<float>(), result.mutable_data_ptr<float>(),
                                       rows, width, scale, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK mean launch failed: ",
              cudaGetErrorString(status));
  return result;
}

at::Tensor TryChannelNorm(const at::Tensor& input, bool keepdim) {
  if (input.scalar_type() != at::kBFloat16 || !input.device().is_privateuseone() ||
      (input.dim() != 4 && input.dim() != 5) || !input.is_contiguous() ||
      input.is_neg() || input.has_names()) return {};
  const auto batches = input.size(0), channels = input.size(1);
  if (batches < 1 || batches > 65535 || channels < 64 || channels > 1020) return {};
  const auto cells = input.numel() / (batches * channels);
  if (cells < 128 || cells > std::numeric_limits<int>::max() - 31 || cells % 4 ||
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 8) return {};
  auto shape = input.sizes().vec();
  if (keepdim)
    shape[1] = 1;
  else
    shape.erase(shape.begin() + 1);
  const c10::OptionalDeviceGuard guard(input.device());
  auto result = at::empty(shape, input.options());
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(input.device().index()));
  const auto status = LaunchChannelNorm(input.const_data_ptr(), result.mutable_data_ptr(),
                                        batches, channels, cells, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK channel norm launch failed: ",
              cudaGetErrorString(status));
  return result;
}

}  // namespace flagos::nvidia
