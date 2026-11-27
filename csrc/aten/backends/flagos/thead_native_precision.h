// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sdk_blas_layout.h"

#include <acblas_v2.h>
#include <c10/core/DeviceGuard.h>
#include <dlfcn.h>
#include <flagos.h>

#include <map>
#include <memory>

namespace at::native::flagos::thead_precision {

inline void Check(acblasStatus_t status, const char* operation) {
  TORCH_CHECK(status == ACBLAS_STATUS_SUCCESS,
              "native T-Head ", operation, " failed: ", static_cast<int>(status));
}

inline c10::Dict<std::string, std::string> Libraries() {
  Dl_info info{};
  TORCH_CHECK(dladdr(reinterpret_cast<void*>(&acblasGemmEx), &info) && info.dli_fname,
              "cannot identify native T-Head SDK library");
  c10::Dict<std::string, std::string> result;
  result.insert("acblas", info.dli_fname);
  return result;
}

struct BlasState {
  acblasHandle_t handle = nullptr;
  acblasMath_t mode = ACBLAS_DEFAULT_MATH;

  BlasState() { Check(acblasCreate(&handle), "acblasCreate"); }
  ~BlasState() { acblasDestroy(handle); }
};

inline acblasHandle_t Handle(c10::DeviceIndex device) {
  const auto reduction = at::globalContext().allowBF16ReductionCuBLAS();
  TORCH_CHECK(reduction != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
              "native T-Head SDK BLAS cannot enforce the requested split-K prohibition");
  thread_local std::map<c10::DeviceIndex, std::unique_ptr<BlasState>> states;
  auto& state = states[device];
  if (!state) state = std::make_unique<BlasState>();
  const auto mode = static_cast<acblasMath_t>(
      reduction == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK
          ? ACBLAS_DEFAULT_MATH : ACBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  if (state->mode != mode) {
    Check(acblasSetMathMode(state->handle, mode), "acblasSetMathMode");
    state->mode = mode;
  }
  Check(acblasSetStream(state->handle, reinterpret_cast<hggcStream_t>(
      GetCurrentStreamForDevice(device))), "acblasSetStream");
  return state->handle;
}

template <int Rank>
at::Tensor Matrix(const at::Tensor& left, const at::Tensor& right) {
  TORCH_CHECK(left.dim() == Rank && right.dim() == Rank,
              "native T-Head BLAS received the wrong matrix rank");
  TORCH_CHECK(left.device().is_privateuseone() && left.device() == right.device(),
              "native T-Head BLAS requires matching flagos devices");
  TORCH_CHECK(left.scalar_type() == at::kBFloat16 && right.scalar_type() == at::kBFloat16,
              "unmeasured native T-Head BLAS dtype");
  TORCH_CHECK(left.size(-1) == right.size(-2), "native T-Head BLAS contracted dimensions differ");
  if constexpr (Rank == 3)
    TORCH_CHECK(left.size(0) == right.size(0), "native T-Head BLAS batch dimensions differ");
  const c10::OptionalDeviceGuard guard(left.device());
  const int m = sdk_blas::Dimension(left.size(-2)), n = sdk_blas::Dimension(right.size(-1));
  const int k = sdk_blas::Dimension(left.size(-1));
  auto sizes = left.sizes().vec();
  sizes.back() = n;
  auto output = at::empty(sizes, left.options());
  if (!output.numel()) return output;
  if (!k) return output.zero_();
  const auto [a, b] = sdk_blas::ColumnMajorOperands(left, right, m, n, k);
  const auto trans_a = a.transpose ? ACBLAS_OP_T : ACBLAS_OP_N;
  const auto trans_b = b.transpose ? ACBLAS_OP_T : ACBLAS_OP_N;
  const float alpha = 1.0f, beta = 0.0f;
  const auto handle = Handle(left.device().index());
  if constexpr (Rank == 2) {
    Check(acblasGemmEx(handle, trans_a, trans_b, n, m, k,
          &alpha, a.tensor.const_data_ptr(), HGGC_R_16BF, a.leading,
          b.tensor.const_data_ptr(), HGGC_R_16BF, b.leading, &beta,
          output.mutable_data_ptr(), HGGC_R_16BF, n,
          ACBLAS_COMPUTE_32F, ACBLAS_GEMM_DEFAULT_TENSOR_OP), "acblasGemmEx");
  } else {
    Check(acblasGemmStridedBatchedEx(handle, trans_a, trans_b, n, m, k,
          &alpha, a.tensor.const_data_ptr(), HGGC_R_16BF, a.leading, a.tensor.stride(0),
          b.tensor.const_data_ptr(), HGGC_R_16BF, b.leading, b.tensor.stride(0), &beta,
          output.mutable_data_ptr(), HGGC_R_16BF, n, output.stride(0),
          sdk_blas::Dimension(left.size(0)), ACBLAS_COMPUTE_32F,
          ACBLAS_GEMM_DEFAULT_TENSOR_OP), "acblasGemmStridedBatchedEx");
  }
  return output;
}

}  // namespace at::native::flagos::thead_precision
