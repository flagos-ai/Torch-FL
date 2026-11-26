// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/accumulate.h>

namespace at::native::flagos::thead_vendor {

inline bool OrdinaryTensor(const at::Tensor& tensor) {
  return tensor.defined() && tensor.device().is_privateuseone() &&
      tensor.layout() == at::kStrided && !tensor.has_names() &&
      !tensor.is_conj() && !tensor.is_neg() &&
      !tensor.key_set().has(c10::DispatchKey::Python) &&
      !tensor.key_set().has(c10::DispatchKey::Functionalize) &&
      !tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
}

inline bool CanRun(const at::Tensor& lhs, const at::Tensor& rhs) {
  return lhs.device() == rhs.device() && lhs.scalar_type() == rhs.scalar_type() &&
      (lhs.scalar_type() == at::kBFloat16 || lhs.scalar_type() == at::kFloat);
}

inline bool Concatenation(const at::ITensorListRef& values, int64_t dim) {
  if (values.size() != 2) return false;
  auto item = values.begin();
  const auto& first = *item;
  ++item;
  const auto& second = *item;
  if (!OrdinaryTensor(first) || !OrdinaryTensor(second) || !CanRun(first, second) ||
      first.numel() == 0 || second.numel() == 0 || first.dim() != second.dim()) {
    return false;
  }
  const auto rank = first.dim();
  if (dim < 0) dim += rank;
  if (rank < 3 || rank > 4 || dim < 0 || dim >= rank) return false;
  const bool inference = c10::InferenceMode::is_enabled();
  if (dim == rank - 1) {
    const auto limit = inference ? 4096 : 262144;
    return first.numel() <= limit && second.numel() <= limit;
  }
  return inference && rank == 4 && dim == rank - 2 &&
      first.size(0) == 1 && second.size(0) == 1 && second.size(dim) == 1 &&
      first.stride(rank - 1) == 1 && second.stride(rank - 1) == 1;
}

inline bool Multiplication(const at::Tensor& lhs, const at::Tensor& rhs) {
  const bool inference = c10::InferenceMode::is_enabled();
  const auto limit = inference ? 4096 : 1 << 20;
  // The public ATen autograd wrapper remains above the accelerated backend.
  if ((c10::GradMode::is_enabled() &&
       !c10::impl::tls_is_dispatch_keyset_excluded(c10::autograd_dispatch_keyset)) ||
      !OrdinaryTensor(lhs) || !rhs.defined() ||
      (lhs.scalar_type() != at::kBFloat16 && lhs.scalar_type() != at::kFloat) ||
      lhs.dim() == 0 || lhs.dim() > 4 || lhs.numel() == 0 || lhs.numel() > limit) {
    return false;
  }
  if (!inference && rhs.device().is_cpu() && rhs.dim() == 0 &&
      rhs.unsafeGetTensorImpl()->is_wrapped_number() && c10::isFloatingType(rhs.scalar_type())) {
    return true;
  }
  if (!OrdinaryTensor(rhs) || !CanRun(lhs, rhs) || rhs.dim() == 0 || rhs.dim() > 4 ||
      rhs.numel() == 0 || rhs.numel() > limit ||
      (inference && (!lhs.is_contiguous() || !rhs.is_contiguous()))) {
    return false;
  }
  const auto sizes = at::infer_size_dimvector(lhs.sizes(), rhs.sizes());
  return c10::multiply_integers(sizes) <= limit;
}

}  // namespace at::native::flagos::thead_vendor
