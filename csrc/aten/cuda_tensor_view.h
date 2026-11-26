// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/Tensor.h>

#include <algorithm>

namespace at::native::flagos {

// Keep the FlagOS allocation owned by its original tensor for the duration of
// the CUDA call. The caller orders any asynchronous kernels with that storage.
inline at::Tensor FlagosToCudaView(const at::Tensor& self) {
  TORCH_CHECK(self.is_privateuseone(), "Expected flagos tensor, got ", self.device());
  const auto device = c10::Device(c10::kCUDA, std::max<int>(self.device().index(), 0));
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(), self.storage().nbytes(),
      c10::DataPtr(self.storage().data_ptr().get(), new at::Tensor(self),
                   [](void* owner) { delete static_cast<at::Tensor*>(owner); }, device),
      nullptr, false);
  auto view = at::detail::make_tensor_base<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CUDA), self.dtype());
  auto* metadata = view.unsafeGetTensorImpl();
  metadata->set_storage_offset(self.storage_offset());
  metadata->set_sizes_and_strides(self.sizes(), self.strides());
  if (self.is_conj()) view._set_conj(true);
  if (self.is_neg()) view._set_neg(true);
  return view;
}

}  // namespace at::native::flagos
