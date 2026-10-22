// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

// argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
//
// aclnnArgMax requires a concrete reduction dim, so when dim is nullopt we
// flatten to 1-D and reduce over axis 0 (matching torch's global-argmax
// semantics). The output is int64 (torch always returns Long indices).
at::Tensor ArgmaxKernelAscend(const at::Tensor& self,
                              ::std::optional<int64_t> dim, bool keepdim) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;

  at::Tensor input;
  int64_t reduce_dim;
  if (dim.has_value()) {
    input = self;
    reduce_dim = dim.value();
  } else {
    // Global argmax: flatten, reduce dim 0, keepdim is ignored by torch here
    // (result is a 0-d scalar unless keepdim was requested on the flat view).
    input = self.reshape({-1});
    reduce_dim = 0;
  }

  // Compute output shape: drop (or keep as size-1) the reduced dim.
  std::vector<int64_t> out_sizes;
  int64_t ndim = input.dim();
  int64_t d = reduce_dim < 0 ? reduce_dim + ndim : reduce_dim;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == d) {
      if (keepdim) out_sizes.push_back(1);
    } else {
      out_sizes.push_back(input.size(i));
    }
  }

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_sizes, input.options().dtype(at::kLong));

  ascend::AclTensorWrapper acl_self(input);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnArgMax, acl_self.get(), d, keepdim, acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(ArgmaxFn, argmax_dispatcher, Backend::kAscend, ArgmaxKernelAscend)

} // namespace at::native::flagos
