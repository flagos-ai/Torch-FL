// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#if defined(FLAGOS_NATIVE_THEAD_VENDOR)
#include "../../generated/ops.h"
#include "../../../native_host/backends/thead/adamw_precision.h"

#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>
#include <torch/library.h>

#include <atomic>
#include <limits>

namespace at::native::flagos {
namespace {

std::atomic<int64_t> addcmul_hits{0};
std::atomic<int64_t> addcdiv_hits{0};

bool DenseBF16(const at::Tensor& tensor) {
  return tensor.defined() && tensor.device().is_privateuseone() &&
      tensor.scalar_type() == at::kBFloat16 && tensor.is_contiguous() && !tensor.is_neg();
}

bool Compatible(const at::Tensor& result, const at::Tensor& operand) {
  if (!DenseBF16(operand) || result.device() != operand.device() ||
      !result.sizes().equals(operand.sizes())) {
    return false;
  }
  const auto overlap = at::get_overlap_status(result, operand);
  return overlap == at::MemOverlapStatus::No || overlap == at::MemOverlapStatus::Full;
}

template <auto Launch>
at::Tensor& Update(at::Tensor& result, const at::Tensor& first, const at::Tensor& second,
                   const at::Scalar& alpha, const Dispatcher<AddcmulInplaceFn>& dispatcher,
                   std::atomic<int64_t>& hits) {
  if (!DenseBF16(result) || result.numel() == 0 ||
      result.numel() > std::numeric_limits<int>::max() || alpha.isComplex() ||
      !Compatible(result, first) || !Compatible(result, second)) {
    return dispatcher.DispatchBackend(Backend::kFlagGems, result, first, second, alpha);
  }
  const c10::OptionalDeviceGuard guard(result.device());
  const auto stream = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(result.device().index()));
  const auto status = Launch(static_cast<uint16_t*>(result.mutable_data_ptr()),
      static_cast<const uint16_t*>(first.const_data_ptr()),
      static_cast<const uint16_t*>(second.const_data_ptr()), result.numel(), alpha.toFloat(), stream);
  TORCH_CHECK(status == cudaSuccess, "T-Head AdamW precision kernel: ", cudaGetErrorString(status));
  hits.fetch_add(1, std::memory_order_relaxed);
  return result;
}

at::Tensor& Addcmul(at::Tensor& result, const at::Tensor& first, const at::Tensor& second,
                   const at::Scalar& alpha) {
  return Update<thead_adamw::Addcmul>(result, first, second, alpha, addcmul_inplace_dispatcher, addcmul_hits);
}

at::Tensor& Addcdiv(at::Tensor& result, const at::Tensor& first, const at::Tensor& second,
                   const at::Scalar& alpha) {
  return Update<thead_adamw::Addcdiv>(result, first, second, alpha, addcdiv_inplace_dispatcher, addcdiv_hits);
}

REGISTER_IMPL_TO_DISPATCHER(AddcmulInplaceFn, addcmul_inplace_dispatcher, Backend::kThead, Addcmul)
REGISTER_IMPL_TO_DISPATCHER(AddcdivInplaceFn, addcdiv_inplace_dispatcher, Backend::kThead, Addcdiv)

}  // namespace

TORCH_LIBRARY_FRAGMENT(flagos, m) {
  m.def("native_thead_adamw_hits() -> Dict(str, int)", []() {
    c10::Dict<std::string, int64_t> result;
    result.insert("addcmul_", addcmul_hits.load(std::memory_order_relaxed));
    result.insert("addcdiv_", addcdiv_hits.load(std::memory_order_relaxed));
    return result;
  });
}

}  // namespace at::native::flagos
#endif
