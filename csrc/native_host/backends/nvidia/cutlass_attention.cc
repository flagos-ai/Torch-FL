// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "cutlass_attention.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>

#include <cmath>
#include <limits>

namespace flagos::nvidia {
namespace {

bool IsAlignedAttentionInput(const at::Tensor& input, c10::Device device) {
  constexpr int64_t alignment = 8;
  constexpr auto max_stride = std::numeric_limits<int>::max();
  return input.dim() == 4 && input.layout() == at::kStrided &&
      input.device() == device && input.scalar_type() == at::kBFloat16 &&
      !input.is_nested() && !input.is_neg() && !input.has_names() &&
      input.stride(3) == 1 && input.stride(1) <= max_stride &&
      input.stride(2) <= max_stride && input.stride(0) % alignment == 0 &&
      input.stride(1) % alignment == 0 && input.stride(2) % alignment == 0 &&
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % (alignment * 2) == 0;
}

AttentionInput Describe(const at::Tensor& input) {
  return {input.const_data_ptr(), input.stride(0), static_cast<int>(input.stride(1)),
          static_cast<int>(input.stride(2))};
}

}  // namespace

at::Tensor TryCutlassAttention(const at::Tensor& query, const at::Tensor& key,
                              const at::Tensor& value, std::optional<double> scale) {
  if (!query.is_privateuseone() || !at::globalContext().userEnabledMemEfficientSDP() ||
      !IsAlignedAttentionInput(query, query.device()) ||
      !IsAlignedAttentionInput(key, query.device()) ||
      !IsAlignedAttentionInput(value, query.device())) {
    return {};
  }
  const auto batches = query.size(0), heads = query.size(1);
  const auto queries = query.size(2), keys = key.size(2);
  const auto qk_width = query.size(3), value_width = value.size(3);
  if (batches < 1 || batches > 65535 || heads < 1 || heads > 65535 ||
      queries < 1 || queries > std::numeric_limits<int>::max() ||
      keys < 1 || keys > std::numeric_limits<int>::max() ||
      qk_width <= 256 || qk_width > 1024 || qk_width % 8 ||
      value_width < 1 || value_width > 1024 || value_width % 8 ||
      key.size(3) != qk_width || value.size(2) != keys ||
      key.size(0) != batches || value.size(0) != batches ||
      key.size(1) != heads || value.size(1) != heads) {
    return {};
  }
  const auto device = query.device().index();
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      major < 8) return {};

  const c10::OptionalDeviceGuard guard(query.device());
  const auto shape = {batches, queries, heads, value_width};
  auto output = at::empty(shape, query.options());
  auto accumulator = at::empty(shape, query.options().dtype(at::kFloat));
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device));
  const auto resolved_scale = scale.value_or(1.0 / std::sqrt(static_cast<double>(qk_width)));
  const auto status = LaunchCutlassAttention(
      Describe(query), Describe(key), Describe(value), batches, heads, queries,
      keys, qk_width, value_width, resolved_scale, output.mutable_data_ptr(),
      accumulator.mutable_data_ptr(), queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA CUTLASS attention launch failed: ",
              cudaGetErrorString(status));
  return output.transpose(1, 2);
}

}  // namespace flagos::nvidia
