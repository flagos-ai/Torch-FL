// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "pointwise.h"

#include <ATen/TensorIterator.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/layer_norm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/llvmMathExtras.h>
#include <cuda_runtime_api.h>
#include <flagos.h>

#include <algorithm>
#include <limits>

namespace flagos::nvidia {

cudaError_t LaunchPointwise(PointwiseOp op, bool bf16, const void* input, void* output,
                            const int64_t size, float scalar, cudaStream_t stream);
cudaError_t LaunchDivide(const float* left, const float* right, float* out,
                         int64_t size, PointwiseLayout layout, cudaStream_t stream);
cudaError_t LaunchMultiply(const void* left, const void* right, void* output, int64_t count,
                            bool bf16, PointwiseLayout layout, cudaStream_t stream);
cudaError_t LaunchMeanLastDim(const float* input, float* output, int rows, int width,
                              int lanes, int rows_per_block, float scale, cudaStream_t stream);
cudaError_t LaunchLayerNorm(const void* input, const void* weight, const void* bias,
                            void* output, float* mean, float* rstd, int rows,
                            int width, float epsilon, cudaStream_t stream);
cudaError_t LaunchChannelNorm(const void* input, void* output, int groups,
                              int channels, int cells, float* partials, int partitions,
                              cudaStream_t stream);

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
  if (iter.ndim() > PointwiseLayout::max_dimensions) return {};
  PointwiseLayout layout{};
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

at::Tensor TryMultiply(const at::Tensor& left, const at::Tensor& right) {
  const auto dtype = left.scalar_type();
  const auto device = left.device();
  if ((dtype != at::kFloat && dtype != at::kBFloat16) ||
      right.scalar_type() != dtype || !device.is_privateuseone() ||
      right.device() != device || left.layout() != at::kStrided ||
      right.layout() != at::kStrided || left.is_neg() || right.is_neg() ||
      left.has_names() || right.has_names()) {
    return {};
  }
  const c10::OptionalDeviceGuard guard(device);
  at::Tensor output;
  PointwiseLayout layout{};
  const void* a = left.const_data_ptr();
  const void* b = right.const_data_ptr();
  if (left.sizes().equals(right.sizes()) && left.strides().equals(right.strides()) &&
      left.is_non_overlapping_and_dense()) {
    output = at::empty_strided(left.sizes(), left.strides(), left.options());
  } else {
    auto iterator = at::TensorIterator::binary_op(output, left, right);
    if (!iterator.is_contiguous()) {
      // Leave large strided products to the published tiled FlagGems kernels.
      if (iterator.ndim() > PointwiseLayout::max_dimensions ||
          iterator.numel() > (1 << 20) || !iterator.can_use_32bit_indexing()) {
        return {};
      }
      layout.dimensions = iterator.ndim();
      for (int dim = 0; dim < layout.dimensions; ++dim) {
        layout.sizes[dim] = iterator.shape()[dim];
        for (int argument = 0; argument < 3; ++argument)
          layout.strides[argument][dim] = iterator.strides(argument)[dim] / left.element_size();
      }
    }
    output = iterator.output();
    a = iterator.data_ptr(1);
    b = iterator.data_ptr(2);
  }
  if (!output.numel()) return output;
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device.index()));
  const auto status = LaunchMultiply(a, b, output.mutable_data_ptr(), output.numel(),
                                    dtype == at::kBFloat16, layout, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK multiply failed: ",
              cudaGetErrorString(status));
  return output;
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
  if (rows < 1 || rows > std::numeric_limits<int>::max() - 15) return {};
  const auto rows_per_block = c10::llvm::PowerOf2Floor(std::min<int64_t>(rows, 16));
  const auto lanes = c10::llvm::PowerOf2Floor(std::min<int64_t>(width / 4, 512 / rows_per_block));
  if (rows_per_block > 1 && (width + lanes - 1) / lanes >= rows_per_block * 16) return {};
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
                                       rows, width, lanes, rows_per_block, scale, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK mean launch failed: ",
              cudaGetErrorString(status));
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> TryLayerNorm(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const at::Tensor& weight, const at::Tensor& bias, double epsilon) {
  const auto device = input.device();
  if (!device.is_privateuseone()) return {};
  for (const auto* tensor : {&input, &weight, &bias}) {
    if (!tensor->defined() || tensor->device() != device ||
        tensor->scalar_type() != at::kBFloat16 || tensor->layout() != at::kStrided ||
        !tensor->is_contiguous() ||
        !tensor->numel() || tensor->is_neg() || tensor->has_names() ||
        reinterpret_cast<uintptr_t>(tensor->const_data_ptr()) % 8) {
      return {};
    }
  }
  if (at::GradMode::is_enabled() &&
      (input.requires_grad() || weight.requires_grad() || bias.requires_grad())) {
    return {};
  }
  const auto [rows, width] =
      at::native::_check_layer_norm_inputs(input, normalized_shape, weight, bias);
  if (width % 4 || width > (1 << std::numeric_limits<float>::digits) ||
      rows > std::numeric_limits<int>::max()) {
    return {};
  }
  auto stats_shape = input.sizes().vec();
  std::fill(stats_shape.end() - normalized_shape.size(), stats_shape.end(), 1);
  const c10::OptionalDeviceGuard guard(device);
  auto output = at::empty(input.sizes(), input.options());
  auto mean = at::empty(stats_shape, input.options().dtype(at::kFloat));
  auto rstd = at::empty(stats_shape, input.options().dtype(at::kFloat));
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device.index()));
  const auto status = LaunchLayerNorm(
      input.const_data_ptr(), weight.const_data_ptr(), bias.const_data_ptr(),
      output.mutable_data_ptr(), mean.mutable_data_ptr<float>(), rstd.mutable_data_ptr<float>(),
      rows, width, epsilon, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK layer norm failed: ",
              cudaGetErrorString(status));
  return {output, mean, rstd};
}

at::Tensor TryChannelNorm(const at::Tensor& input, bool keepdim) {
  if (input.scalar_type() != at::kBFloat16 || !input.device().is_privateuseone() ||
      (input.dim() != 4 && input.dim() != 5) || !input.is_non_overlapping_and_dense() ||
      input.is_neg() || input.has_names()) return {};
  const auto channels = input.size(1), cells = input.stride(1);
  if (cells < 128 || cells > std::numeric_limits<int>::max() - 31 || cells % 4 ||
      channels < 64 || input.numel() > std::numeric_limits<int>::max() / 2 ||
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 8) return {};
  const auto groups = input.numel() / (channels * cells);
  if (groups < 1 || groups > 65535) return {};
  // Removing the channel axis must leave the result physically contiguous.
  for (int64_t dim = input.dim() - 1, expected = 1; dim >= 0; --dim) {
    if (dim == 1 || input.size(dim) <= 1) continue;
    const auto stride = input.stride(dim);
    if ((stride > cells ? stride / channels : stride) != expected) return {};
    expected *= input.size(dim);
  }
  auto shape = input.sizes().vec();
  if (keepdim)
    shape[1] = 1;
  else
    shape.erase(shape.begin() + 1);
  const c10::OptionalDeviceGuard guard(input.device());
  int64_t partitions = 1;
  const auto values_per_lane = (channels + 3) / 4;
  if (values_per_lane >= 256) {
    // The CUDA reduction splits wide channel planes across blocks when
    // their output tiles alone would leave multiprocessors underoccupied.
    const auto output_tiles = (groups * cells + 127) / 128;
    const auto device = input.device().index();
    int multiprocessors = 0, threads_per_mp = 0;
    if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&threads_per_mp, cudaDevAttrMaxThreadsPerMultiProcessor, device) != cudaSuccess)
      return {};
    const int64_t target_tiles = int64_t(multiprocessors) * (threads_per_mp / 128);
    if (output_tiles <= target_tiles) {
      partitions = std::max(std::min((target_tiles + output_tiles - 1) / output_tiles,
                                     (values_per_lane + 15) / 16),
                            (values_per_lane + 255) / 256);
      if (partitions > 65535) return {};
    }
  }
  auto result = at::empty(shape, input.options());
  at::Tensor storage;
  float* partials = nullptr;
  if (partitions > 1) {
    storage = at::empty({partitions, result.numel()}, input.options().dtype(at::kFloat));
    partials = storage.mutable_data_ptr<float>();
  }
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(input.device().index()));
  const auto status = LaunchChannelNorm(input.const_data_ptr(), result.mutable_data_ptr(),
                                        groups, channels, cells, partials, partitions, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK channel norm launch failed: ",
              cudaGetErrorString(status));
  return result;
}

}  // namespace flagos::nvidia
