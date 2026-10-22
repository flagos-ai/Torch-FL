// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <ATen/ATen.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

// masked_select(Tensor self, Tensor mask) -> Tensor
//
// Returns a 1-D tensor of the elements of `self` where `mask` is true, in
// row-major order. The output length is data-dependent (the number of true
// entries), so it cannot be expressed by the shape-formula codegen and lives
// here as a bespoke kernel.
//
// self and mask broadcast against each other (PyTorch semantics). aclnn's
// aclnnMaskedSelect requires a pre-sized output buffer, so we first materialise
// the broadcast mask, count its true entries on host (one device->host sync via
// .item()), allocate the 1-D output, then run the kernel.
at::Tensor MaskedSelectKernelAscend(const at::Tensor& self, const at::Tensor& mask) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;

  // Broadcast self and mask to a common shape (aclnn wants matching, contiguous
  // buffers). infer_size gives the broadcasted shape.
  auto bshape = at::infer_size(self.sizes(), mask.sizes());
  auto self_b = self.expand(bshape).contiguous();
  auto mask_b = mask.expand(bshape).contiguous();

  // Count true elements: sum the bool mask (promoted to int64) and read to host.
  int64_t count = mask_b.to(at::kLong).sum().item<int64_t>();

  // aclnnMaskedSelect requires the output buffer pre-sized to the full number of
  // broadcast elements (it writes `count` entries then reports the used length
  // via workspace metadata). Allocate numel, run, then narrow to `count`.
  int64_t numel = self_b.numel();
  auto out_full = ascend::OpPreparation::apply_tensor_without_format(
      {numel}, self.options());

  ascend::AclTensorWrapper acl_self(self_b);
  ascend::AclTensorWrapper acl_mask(mask_b);
  ascend::AclTensorWrapper acl_out(out_full);

  EXEC_ASCEND_CMD(aclnnMaskedSelect, acl_self.get(), acl_mask.get(), acl_out.get());
  return out_full.narrow(0, 0, count);
}

REGISTER_IMPL_TO_DISPATCHER(MaskedSelectFn, masked_select_dispatcher, Backend::kAscend, MaskedSelectKernelAscend)

} // namespace at::native::flagos
