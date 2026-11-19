// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Host metadata and SDK calls only; numerical kernels belong to MACA BLAS.
#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>
#include <mcblas/mcblas.h>
#include <mcr/mc_runtime.h>
#include <dlfcn.h>

#include <limits>
#include <map>
#include <memory>

namespace at::native::flagos::metax_precision {

inline void Check(mcblasStatus_t status, const char* operation) {
  TORCH_CHECK(status == MCBLAS_STATUS_SUCCESS,
              "native MetaX ", operation, " failed: ", static_cast<int>(status));
}

inline int Dimension(int64_t value) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              "native MetaX BLAS dimension exceeds its integer ABI");
  return static_cast<int>(value);
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

struct Operand {
  at::Tensor tensor;
  mcblasOperation_t transpose;
  int leading;
};

inline Operand Describe(at::Tensor value) {
  const auto rows = value.size(-2), columns = value.size(-1);
  if (value.stride(-1) == 1 && value.stride(-2) >= std::max<int64_t>(columns, 1))
    return {value, MCBLAS_OP_N, Dimension(value.stride(-2))};
  if (value.stride(-2) == 1 && value.stride(-1) >= std::max<int64_t>(rows, 1))
    return {value, MCBLAS_OP_T, Dimension(value.stride(-1))};
  value = value.contiguous();
  return {value, MCBLAS_OP_N, Dimension(std::max<int64_t>(columns, 1))};
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
  const int m = Dimension(left.size(-2)), n = Dimension(right.size(-1));
  const int k = Dimension(left.size(-1));
  auto sizes = left.sizes().vec();
  sizes.back() = n;
  auto output = at::empty(sizes, left.options());
  if (!output.numel()) return output;
  if (!k) return output.zero_();
  auto a = Describe(right), b = Describe(left);
  // C(row-major)^T = B^T A^T in the column-major SDK API. Preserve actual
  // transpose and leading dimensions, including the vendor's degenerate cases.
  if (a.transpose == MCBLAS_OP_N && k <= 1) a.leading = std::max(n, 1);
  if (a.transpose == MCBLAS_OP_T && n <= 1) a.leading = std::max(k, 1);
  if (b.transpose == MCBLAS_OP_N && m <= 1) b.leading = std::max(k, 1);
  if (b.transpose == MCBLAS_OP_T && k <= 1) b.leading = std::max(m, 1);
  const int leading_output = std::max(n, 1);
  const float alpha = 1.0f, beta = 0.0f;
  auto& state = State(left);
  if constexpr (Rank == 2) {
    Check(mcblasGemmEx(state.handle, a.transpose, b.transpose, n, m, k,
          &alpha, a.tensor.data_ptr(), MACA_R_16BF, a.leading,
          b.tensor.data_ptr(), MACA_R_16BF, b.leading, &beta,
          output.data_ptr(), MACA_R_16BF, leading_output,
          MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT_TENSOR_OP), "mcblasGemmEx");
  } else {
    Check(mcblasGemmStridedBatchedEx(state.handle, a.transpose, b.transpose, n, m, k,
          &alpha, a.tensor.data_ptr(), MACA_R_16BF, a.leading, a.tensor.stride(0),
          b.tensor.data_ptr(), MACA_R_16BF, b.leading, b.tensor.stride(0), &beta,
          output.data_ptr(), MACA_R_16BF, leading_output, output.stride(0),
          Dimension(left.size(0)), MCBLAS_COMPUTE_32F,
          MCBLAS_GEMM_DEFAULT_TENSOR_OP), "mcblasGemmStridedBatchedEx");
  }
  return output;
}

}  // namespace at::native::flagos::metax_precision
