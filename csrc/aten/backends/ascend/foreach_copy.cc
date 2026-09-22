// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include "device_guard.h"
#include <ATen/core/Tensor.h>

namespace at::native::flagos {

// _foreach_copy_(Tensor(a!)[] self, Tensor[] src, bool non_blocking=False) -> ()
//
// FSDP2 uses this to copy gathered parameter shards back into place. CANN has
// no aclnnForeachCopy, so decompose to individual copy_ calls (which route to
// aclnnInplaceCopy, eliminating the CPU round-trip that made contiguous slow).

void ForeachCopyKernelAscend(
    at::TensorList self,
    at::TensorList src,
    bool non_blocking) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  TORCH_CHECK(
      self.size() == src.size(),
      "_foreach_copy_: self and src must have the same length, got ",
      self.size(),
      " vs ",
      src.size());

  for (size_t i = 0; i < self.size(); ++i) {
    self[i].copy_(src[i], non_blocking);
  }
}

REGISTER_IMPL_TO_DISPATCHER(
    ForeachCopyInplaceFn,
    foreach_copy_inplace_dispatcher,
    Backend::kAscend,
    ForeachCopyKernelAscend)

} // namespace at::native::flagos
