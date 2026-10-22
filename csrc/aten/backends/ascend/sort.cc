// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

std::tuple<at::Tensor, at::Tensor> SortImpl(
    const at::Tensor& self, bool stable, int64_t dim, bool descending) {
  namespace ascend = at::native::flagos::ascend;

  int64_t d = dim < 0 ? dim + self.dim() : dim;
  // sort preserves the full shape; values keep dtype, indices are int64.
  auto values = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      self.sizes(), self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_values(values);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD(aclnnSort, acl_self.get(), stable, d, descending,
                  acl_values.get(), acl_indices.get());
  return std::make_tuple(values, indices);
}

} // namespace

// sort(Tensor self, int dim=-1, bool descending=False) -> (values, indices)
std::tuple<at::Tensor, at::Tensor> SortKernelAscend(
    const at::Tensor& self, int64_t dim, bool descending) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return SortImpl(self, /*stable=*/false, dim, descending);
}

// sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False)
//   -> (values, indices). transformers' TopPLogitsWarper calls torch.sort(),
//   which resolves to this overload on recent torch.
std::tuple<at::Tensor, at::Tensor> SortStableKernelAscend(
    const at::Tensor& self, ::std::optional<bool> stable, int64_t dim,
    bool descending) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return SortImpl(self, stable.value_or(false), dim, descending);
}

REGISTER_IMPL_TO_DISPATCHER(SortFn, sort_dispatcher, Backend::kAscend, SortKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(SortStableFn, sort_stable_dispatcher, Backend::kAscend, SortStableKernelAscend)

} // namespace at::native::flagos
