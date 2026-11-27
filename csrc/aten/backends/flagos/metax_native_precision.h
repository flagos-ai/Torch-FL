// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Host metadata and SDK calls only; numerical kernels belong to MACA BLAS.
#pragma once

#include "sdk_blas_layout.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>
#include <mcblas/mcblas.h>
#include <mcr/mc_runtime.h>
#include <dlfcn.h>

#include <map>
#include <memory>

namespace at::native::flagos::metax_precision {

inline void Check(mcblasStatus_t status, const char* operation) {
  TORCH_CHECK(status == MCBLAS_STATUS_SUCCESS,
              "native MetaX ", operation, " failed: ", static_cast<int>(status));
}

inline std::string LibraryOf(void* address) {
  Dl_info info{};
  TORCH_CHECK(address && dladdr(address, &info) && info.dli_fname,
              "cannot identify native MetaX SDK library");
  return info.dli_fname;
}

inline c10::Dict<std::string, std::string> Libraries() {
  c10::Dict<std::string, std::string> result;
  result.insert("mcblas", LibraryOf(dlsym(RTLD_DEFAULT, "mcblasGemmEx")));
  result.insert("mcblas_lt", LibraryOf(dlsym(RTLD_DEFAULT, "mcblasLtCreate")));
  result.insert("mcruntime", LibraryOf(reinterpret_cast<void*>(&mcGetDevice)));
  return result;
}

struct BlasState {
  mcblasHandle_t handle = nullptr;
  mcblasMath_t mode = MCBLAS_DEFAULT_MATH;

  explicit BlasState(uint64_t stream) {
    Check(mcblasCreate(&handle), "mcblasCreate");
    Check(mcblasSetStream(handle, reinterpret_cast<mcStream_t>(stream)), "mcblasSetStream");
    Check(mcblasSetPointerMode(handle, MCBLAS_POINTER_MODE_HOST), "mcblasSetPointerMode");
  }

  ~BlasState() { if (handle) mcblasDestroy(handle); }
};

inline BlasState& State(const at::Tensor& input) {
  const auto device = input.device().index();
  const auto stream = reinterpret_cast<uint64_t>(GetCurrentStreamForDevice(device));
  thread_local std::map<std::pair<int, uint64_t>, std::unique_ptr<BlasState>> states;
  auto& state = states[{device, stream}];
  if (!state) state = std::make_unique<BlasState>(stream);
  const auto reduction = at::globalContext().allowBF16ReductionCuBLAS();
  TORCH_CHECK(reduction != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
              "native MetaX SDK BLAS cannot enforce the requested split-K prohibition");
  const auto mode = static_cast<mcblasMath_t>(
      reduction == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK
          ? MCBLAS_DEFAULT_MATH : MCBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  if (state->mode != mode) {
    Check(mcblasSetMathMode(state->handle, mode), "mcblasSetMathMode");
    state->mode = mode;
  }
  return *state;
}

template <int Rank>
at::Tensor Matrix(const at::Tensor& left, const at::Tensor& right) {
  TORCH_CHECK(left.dim() == Rank && right.dim() == Rank,
              "native MetaX BLAS received the wrong matrix rank");
  TORCH_CHECK(left.device().is_privateuseone() && left.device() == right.device(),
              "native MetaX BLAS requires matching flagos devices");
  TORCH_CHECK(left.scalar_type() == at::kBFloat16 && right.scalar_type() == at::kBFloat16,
              "unmeasured native MetaX BLAS dtype");
  TORCH_CHECK(left.size(-1) == right.size(-2), "native MetaX BLAS contracted dimensions differ");
  if constexpr (Rank == 3)
    TORCH_CHECK(left.size(0) == right.size(0), "native MetaX BLAS batch dimensions differ");
  c10::OptionalDeviceGuard guard(left.device());
  const int m = sdk_blas::Dimension(left.size(-2)), n = sdk_blas::Dimension(right.size(-1));
  const int k = sdk_blas::Dimension(left.size(-1));
  auto sizes = left.sizes().vec();
  sizes.back() = n;
  auto output = at::empty(sizes, left.options());
  if (!output.numel()) return output;
  if (!k) return output.zero_();
  const auto [a, b] = sdk_blas::ColumnMajorOperands(left, right, m, n, k);
  const auto trans_a = a.transpose ? MCBLAS_OP_T : MCBLAS_OP_N;
  const auto trans_b = b.transpose ? MCBLAS_OP_T : MCBLAS_OP_N;
  const int leading_output = std::max(n, 1);
  const float alpha = 1.0f, beta = 0.0f;
  auto& state = State(left);
  if constexpr (Rank == 2) {
    Check(mcblasGemmEx(state.handle, trans_a, trans_b, n, m, k,
          &alpha, a.tensor.data_ptr(), MACA_R_16BF, a.leading,
          b.tensor.data_ptr(), MACA_R_16BF, b.leading, &beta,
          output.data_ptr(), MACA_R_16BF, leading_output,
          MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT_TENSOR_OP), "mcblasGemmEx");
  } else {
    Check(mcblasGemmStridedBatchedEx(state.handle, trans_a, trans_b, n, m, k,
          &alpha, a.tensor.data_ptr(), MACA_R_16BF, a.leading, a.tensor.stride(0),
          b.tensor.data_ptr(), MACA_R_16BF, b.leading, b.tensor.stride(0), &beta,
          output.data_ptr(), MACA_R_16BF, leading_output, output.stride(0),
          sdk_blas::Dimension(left.size(0)), MCBLAS_COMPUTE_32F,
          MCBLAS_GEMM_DEFAULT_TENSOR_OP), "mcblasGemmStridedBatchedEx");
  }
  return output;
}

}  // namespace at::native::flagos::metax_precision
