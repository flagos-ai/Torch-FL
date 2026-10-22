// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include "device_guard.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/isin.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/any.h>
#include <ATen/ops/zeros.h>

namespace at::native::flagos {

// isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
//
// CANN has no aclnnIsIn kernel. The previous implementation computed on CPU by
// copying both inputs D2H and the result H2D -- three transfers per call, each
// forcing a stream sync. That is catastrophically slow when HF generate() calls
// isin on a VOCAB-SIZED elements tensor every decode step (measured 4.5 ms/call
// on a (151936,) input: ~2.4 MB copied per step, ~14 ms/token, the single
// largest cost in the generate loop, dwarfing all model.forward ops).
//
// Instead compute entirely on-device: isin(elements, test) is
// (elements.unsqueeze(-1) == test_elements).any(-1). We special-case the common
// tiny test_elements by OR-ing per-value equalities to avoid materializing the
// (numel x |test|) broadcast for a large elements tensor. All ops (eq, any,
// logical_or) are registered aclnn kernels, so no host round-trip occurs.
at::Tensor IsinTensorTensorKernelAscend(const at::Tensor& elements,
                                        const at::Tensor& test_elements,
                                        bool assume_unique, bool invert) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(elements));
  (void)assume_unique;  // no fast-path distinction on device
  const int64_t n_test = test_elements.numel();

  at::Tensor result;
  if (n_test == 0) {
    // Nothing to match: all-false (or all-true when inverted).
    result = at::zeros(elements.sizes(), elements.options().dtype(at::kBool));
  } else {
    // OR together elements == test_elements[i] for each test value, all on
    // device (test_flat[i] is a 0-dim device tensor -> eq.Tensor). The test set
    // is tiny in practice (eos/pad ids), so this stays cheap and avoids
    // materializing a (elements.numel() x n_test) broadcast intermediate.
    auto test_flat = test_elements.reshape({n_test});
    result = elements == test_flat[0];
    for (int64_t i = 1; i < n_test; ++i) {
      result = at::logical_or(result, elements == test_flat[i]);
    }
  }

  if (invert) {
    result = result.logical_not();
  }
  return result;
}

REGISTER_IMPL_TO_DISPATCHER(IsinTensorTensorFn, isin_tensor_tensor_dispatcher, Backend::kAscend, IsinTensorTensorKernelAscend)

} // namespace at::native::flagos
