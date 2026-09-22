// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "runtime/generator.h"
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

// multinomial(Tensor self, int num_samples, bool replacement=False, *,
//             Generator? generator=None) -> Tensor
//
// aclnnMultinomial(self, numsamples, replacement, seed, offset, out). Output is
// int64 sampled indices with the sample dim replaced by num_samples ([N] input
// -> [num_samples]; [B, N] -> [B, num_samples]). transformers' _sample() calls
// this to draw the next token. We pull a 64-bit seed from the default CPU
// generator so successive calls differ; offset is left at 0.
at::Tensor MultinomialKernelAscend(const at::Tensor& self, int64_t num_samples,
                                   bool replacement,
                                   ::std::optional<at::Generator> generator) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;

  auto out_shape = self.sizes().vec();
  out_shape.back() = num_samples;
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kLong));

  auto seed = c10::flagos::ReserveSeed(
      generator, static_cast<c10::DeviceIndex>(self.device().index()));
  int64_t offset = 0;

  ascend::AclTensorWrapper acl_self(self);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnMultinomial, acl_self.get(), num_samples, replacement,
                  seed, offset, acl_out.get());
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(MultinomialFn, multinomial_dispatcher, Backend::kAscend, MultinomialKernelAscend)

} // namespace at::native::flagos
