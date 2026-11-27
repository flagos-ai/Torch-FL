// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ops/div.h>
#include <ATen/ops/embedding.h>
#include <ATen/ops/empty.h>

namespace at::native::flagos {

// Count each index by accumulating unit gradients through the backend's
// unscaled embedding reduction. Gather the counts before scaling token gradients.
template <typename Accumulate>
at::Tensor ScaleEmbeddingGradByFrequency(
    const at::Tensor& grad, const at::Tensor& indices, Accumulate&& accumulate) {
  auto shape = indices.sizes().vec();
  shape.push_back(1);
  auto ones = at::empty(shape, grad.options()).fill_(1);
  auto counts = accumulate(ones);
  return at::div(grad, at::embedding(counts, indices));
}

}  // namespace at::native::flagos
