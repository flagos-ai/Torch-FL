// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "precision_runtime.h"

#include <ATen/Context.h>
#include <ATen/ops/_cudnn_attention_forward_native.h>
#include <c10/util/env.h>

namespace at::native::flagos::vendor {

inline at::Tensor CudnnSdpa(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& mask, double dropout, bool causal,
    std::optional<double> scale, bool enable_gqa) {
  auto& context = at::globalContext();
  if (!context.userEnabledCuDNNSDP() ||
      (context.deterministicAlgorithms() && !context.deterministicAlgorithmsWarnOnly()) ||
      mask || dropout != 0.0 || !query.is_privateuseone() ||
      query.scalar_type() != at::kBFloat16 || query.dim() != 4 || query.is_nested()) {
    return {};
  }
  const auto width = query.size(3);
  if (width < 32 || width > 128 || width % 8 != 0 ||
      query.size(0) == 0 || query.size(1) == 0 || query.size(2) == 0) {
    return {};
  }
  for (const auto* input : {&query, &key, &value}) {
    if (input->device() != query.device() || input->scalar_type() != at::kBFloat16 ||
        input->layout() != at::kStrided || input->dim() != 4 || input->is_nested() ||
        input->requires_grad() || input->is_conj() || input->is_neg() ||
        input->has_names() || input->size(0) != query.size(0) ||
        input->size(3) != width || input->stride(3) != 1) {
      return {};
    }
  }
  if (key.size(1) == 0 || key.size(2) == 0 || key.size(1) != value.size(1) ||
      key.size(2) != value.size(2) || (causal && query.size(2) != key.size(2)) ||
      (query.size(1) != key.size(1) &&
       (!enable_gqa || query.size(1) % key.size(1) != 0))) {
    return {};
  }
  const auto* properties = at::cuda::getDeviceProperties(query.device().index());
  // The pinned CUDA release prefers this cuDNN implementation on Hopper.
  static const bool prefer_cudnn =
      c10::utils::check_env("TORCH_CUDNN_SDPA_DEPRIORITIZED") != true &&
      at::detail::getCUDAHooks().versionRuntimeCuDNN() > 91500;
  if (!prefer_cudnn || properties->major != 9 || properties->minor != 0) return {};

  using at::SDPBackend;
  constexpr auto initial_order = std::array{
      SDPBackend::flash_attention, SDPBackend::efficient_attention, SDPBackend::math,
      SDPBackend::cudnn_attention, SDPBackend::overrideable};
  const auto order = context.sDPPriorityOrder();
  if (order != initial_order && order.front() != SDPBackend::cudnn_attention) return {};

  VendorScope scope(query);
  auto result = std::get<0>(at::native::_cudnn_attention_forward(
      query, key, value, std::nullopt, std::nullopt, std::nullopt,
      query.size(2), key.size(2), false, dropout, causal, false, scale));
  TORCH_INTERNAL_ASSERT(result.device() == query.device());
  return result;
}

} // namespace at::native::flagos::vendor
