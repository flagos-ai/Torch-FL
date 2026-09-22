// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include "device_guard.h"
#include <ATen/core/Tensor.h>

namespace at::native::flagos {

// lift_fresh(Tensor(a) self) -> Tensor(a)
//
// A functionalization primitive: it marks a freshly-created tensor (typically
// from torch.tensor(scalar, device=...)) as safe to alias without a defensive
// copy. Semantically it is the identity -- CUDA/CPU both return `self`
// unchanged. transformers' generate() calls it via
// torch.tensor(bos_token_id, device='flagos') in _prepare_special_tokens, so
// the Ascend backend needs it registered even though there is no aclnn kernel.
at::Tensor LiftFreshKernelAscend(const at::Tensor& self) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  return self;
}

REGISTER_IMPL_TO_DISPATCHER(LiftFreshFn, lift_fresh_dispatcher, Backend::kAscend, LiftFreshKernelAscend)

} // namespace at::native::flagos
