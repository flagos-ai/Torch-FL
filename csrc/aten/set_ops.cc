// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "set_ops.h"

#include <ATen/native/Resize.h>

namespace at::native::flagos {

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  // Directly manipulate TensorImpl as the storage spellings below do.
  // at::native::set_tensor_ re-enters the dispatcher through set_, whose
  // source_Tensor case reaches this registration again. Module._apply calls
  // this path for every parameter, so the recursion ends in a stack overflow.
  auto* self_impl = self.unsafeGetTensorImpl();
  const auto* source_impl = source.unsafeGetTensorImpl();
  TORCH_CHECK(
      self_impl->dtype() == source_impl->dtype(),
      "set_ must not be called on a Tensor with different types");
  self_impl->set_storage_keep_dtype(source_impl->storage());
  self_impl->set_storage_offset(source_impl->storage_offset());
  self_impl->set_sizes_and_strides(source_impl->sizes(), source_impl->strides());
  return self;
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  // Directly manipulate TensorImpl to avoid infinite recursion.
  // at::native::set_ internally calls self.set_(source, 0, ...) which
  // dispatches back to set_.source_Storage_storage_offset, and that
  // implementation used to call result.set_(storage) which dispatches
  // back here, causing infinite recursion and stack overflow.
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  int64_t stride_val = 1;
  self.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(source));
  self.unsafeGetTensorImpl()->set_storage_offset(0);
  self.unsafeGetTensorImpl()->set_sizes_and_strides(
      c10::IntArrayRef(&new_size, 1), c10::IntArrayRef(&stride_val, 1));
  return self;
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  // Directly manipulate TensorImpl to avoid infinite recursion.
  // Previously called result.set_(storage) which dispatches through the
  // dispatcher back to set_.source_Storage, creating an infinite loop.
  result.unsafeGetTensorImpl()->set_storage_keep_dtype(std::move(storage));
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  result.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return result;
}

} // namespace at::native::flagos
