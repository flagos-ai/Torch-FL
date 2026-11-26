// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>

#include <unordered_map>

namespace at::native::flagos {

inline void BumpForeachVersions(at::TensorList tensors) {
  for (const auto& tensor : tensors) tensor.unsafeGetTensorImpl()->bump_version();
}

// Batched in-place kernels cannot reorder writes to overlapping views or
// tensors shared by two entries. The ATen sequential path handles those lists.
inline bool CanBatchForeach(
    at::TensorList result,
    at::TensorList first,
    at::TensorList second,
    int64_t max_elements) {
  if (result.empty() || !result.front().defined() ||
      !result.front().is_privateuseone() ||
      (!first.empty() && first.size() != result.size()) ||
      (!second.empty() && second.size() != result.size())) {
    return false;
  }

  const auto device = result.front().device();
  const auto dtype = result.front().scalar_type();
  const bool check_grad = at::GradMode::is_enabled();
  auto eligible = [&](const at::Tensor& tensor) {
    return tensor.defined() && tensor.device() == device &&
        tensor.scalar_type() == dtype && tensor.layout() == at::kStrided &&
        tensor.is_contiguous() && !tensor.is_neg() && !tensor.is_conj() &&
        (!check_grad || !tensor.requires_grad());
  };

  std::unordered_map<const c10::StorageImpl*, size_t> writes;
  writes.reserve(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    const auto& tensor = result[i];
    if (!eligible(tensor) ||
        (!first.empty() &&
         (!eligible(first[i]) || tensor.sizes() != first[i].sizes())) ||
        (!second.empty() &&
         (!eligible(second[i]) || tensor.sizes() != second[i].sizes()))) {
      return false;
    }
    if (tensor.numel() &&
        (tensor.numel() > max_elements ||
         !writes.emplace(tensor.storage().unsafeGetStorageImpl(), i).second)) {
      return false;
    }
  }

  auto independent = [&](const at::Tensor& tensor, size_t index) {
    if (!tensor.numel()) return true;
    const auto found = writes.find(tensor.storage().unsafeGetStorageImpl());
    return found == writes.end() ||
        (found->second == index &&
         tensor.storage_offset() == result[index].storage_offset());
  };
  for (size_t i = 0; i < result.size(); ++i) {
    if ((!first.empty() && !independent(first[i], i)) ||
        (!second.empty() && !independent(second[i], i))) {
      return false;
    }
  }
  return true;
}

}  // namespace at::native::flagos
