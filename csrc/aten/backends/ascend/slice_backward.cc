// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor SliceBackwardKernelAscend(const at::Tensor& grad_output, at::IntArrayRef input_sizes,
                                     int64_t dim, int64_t start, int64_t end, int64_t step) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(grad_output));
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input_sizes, grad_output.options());
  grad_input.zero_();

  auto slice = grad_input.slice(dim, start, end, step);
  slice.copy_(grad_output);

  return grad_input;
}

REGISTER_IMPL_TO_DISPATCHER(SliceBackwardFn, slice_backward_dispatcher, Backend::kAscend, SliceBackwardKernelAscend)

} // namespace at::native::flagos
