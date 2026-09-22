// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

// scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
//
// Out-of-place scatter: out = self.clone(); out.scatter_(dim, index, src).
// aclnnScatter(self, dim, index, src, reduce, out) with reduce=0 (replace).
// transformers' TopPLogitsWarper uses scatter() to unsort the removal mask.
at::Tensor ScatterSrcKernelAscend(const at::Tensor& self, int64_t dim,
                                  const at::Tensor& index, const at::Tensor& src) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;

  int64_t d = dim < 0 ? dim + self.dim() : dim;
  // aclnnScatter writes the full result to `out`; seed it with self so entries
  // not covered by index retain their original values.
  auto out = self.clone();

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_index(index);
  ascend::AclTensorWrapper acl_src(src);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnScatter, acl_self.get(), d, acl_index.get(),
                  acl_src.get(), static_cast<int64_t>(0), acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(ScatterSrcFn, scatter_src_dispatcher, Backend::kAscend, ScatterSrcKernelAscend)

} // namespace at::native::flagos
