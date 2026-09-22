// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

// topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True)
//   -> (Tensor values, Tensor indices)
//
// aclnnTopk(self, k, dim, largest, sorted, valuesOut, indicesOut). Used by
// transformers' TopKLogitsWarper during sampling. Output shape equals the
// input with the reduced dim resized to k; indices are int64.
std::tuple<at::Tensor, at::Tensor> TopkKernelAscend(
    const at::Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;

  int64_t d = dim < 0 ? dim + self.dim() : dim;
  auto out_shape = self.sizes().vec();
  out_shape[d] = k;

  auto values = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options());
  auto indices = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_values(values);
  ascend::AclTensorWrapper acl_indices(indices);

  EXEC_ASCEND_CMD(aclnnTopk, acl_self.get(), k, d, largest, sorted,
                  acl_values.get(), acl_indices.get());
  return std::make_tuple(values, indices);
}

REGISTER_IMPL_TO_DISPATCHER(TopkFn, topk_dispatcher, Backend::kAscend, TopkKernelAscend)

} // namespace at::native::flagos
