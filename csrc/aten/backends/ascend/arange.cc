// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <cmath>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

// Compute the number of elements in arange(start, end, step), matching
// PyTorch's reference (aten/src/ATen/native/RangeFactories.cpp): ceil for
// integral dtypes, and a fudge-factor guard for floating point.
int64_t ArangeSize(const at::Scalar& start, const at::Scalar& end,
                   const at::Scalar& step, at::ScalarType dtype) {
  if (c10::isIntegralType(dtype, /*includeBool=*/false)) {
    int64_t s = start.toLong(), e = end.toLong(), st = step.toLong();
    TORCH_CHECK(st != 0, "arange: step must be nonzero");
    if ((st > 0 && e < s) || (st < 0 && e > s)) return 0;
    // ceil division that also works for negative step.
    return (e - s + st - (st > 0 ? 1 : -1)) / st;
  }
  double s = start.toDouble(), e = end.toDouble(), st = step.toDouble();
  TORCH_CHECK(st != 0, "arange: step must be nonzero");
  double n = std::ceil((e - s) / st);
  return n < 0 ? 0 : static_cast<int64_t>(n);
}

} // namespace

// arange.start_step(Scalar start, Scalar end, Scalar step, ScalarType?, ...) -> Tensor
at::Tensor ArangeStartStepKernelAscend(
    const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  namespace ascend = at::native::flagos::ascend;

  // Default dtype: long if all args integral, else the default float type
  // (mirrors torch's arange type-promotion for the common cases used by
  // transformers' cache_position = arange(...)).
  at::ScalarType out_dtype = dtype.value_or(
      (start.isIntegral(false) && end.isIntegral(false) && step.isIntegral(false))
          ? at::kLong
          : at::typeMetaToScalarType(c10::get_default_dtype()));

  auto options = at::TensorOptions()
      .dtype(out_dtype)
      .layout(layout.value_or(at::kStrided))
      .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
      .pinned_memory(pin_memory.value_or(false));

  int64_t n = ArangeSize(start, end, step, out_dtype);
  auto out = ascend::OpPreparation::apply_tensor_without_format({n}, options);
  if (n == 0) return out;

  ascend::AclScalarWrapper acl_start(start, out_dtype);
  ascend::AclScalarWrapper acl_end(end, out_dtype);
  ascend::AclScalarWrapper acl_step(step, out_dtype);
  ascend::AclTensorWrapper acl_out(out);

  EXEC_ASCEND_CMD(aclnnArange, acl_start.get(), acl_end.get(), acl_step.get(),
                  acl_out.get());
  return out;
}

// arange.start(Scalar start, Scalar end, ...) -> step defaults to 1.
at::Tensor ArangeStartKernelAscend(
    const at::Scalar& start, const at::Scalar& end,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  return ArangeStartStepKernelAscend(start, end, at::Scalar(1), dtype, layout,
                                     device, pin_memory);
}

// arange(Scalar end, ...) -> start defaults to 0, step to 1.
at::Tensor ArangeKernelAscend(
    const at::Scalar& end,
    ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(device));
  return ArangeStartStepKernelAscend(at::Scalar(0), end, at::Scalar(1), dtype,
                                     layout, device, pin_memory);
}

REGISTER_IMPL_TO_DISPATCHER(ArangeStartStepFn, arange_start_step_dispatcher, Backend::kAscend, ArangeStartStepKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(ArangeStartFn, arange_start_dispatcher, Backend::kAscend, ArangeStartKernelAscend)
REGISTER_IMPL_TO_DISPATCHER(ArangeFn, arange_dispatcher, Backend::kAscend, ArangeKernelAscend)

} // namespace at::native::flagos
