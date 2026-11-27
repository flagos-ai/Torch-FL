// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "linalg.h"

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/_linalg_solve_ex_meta.h>
#include <aten/backends/flagos/sdk_blas_layout.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/env.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <flagos.h>
#include <torch/library.h>

#include <array>
#include <charconv>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <tuple>

namespace flagos::nvidia {
namespace {

namespace sdk_blas = at::native::flagos::sdk_blas;

void* OpenSdk(const char* library, int major) {
  const auto soname = std::string(library) + ".so." + std::to_string(major);
  auto* handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(handle, "Torch-FL requires the NVIDIA SDK library ", soname,
              ": ", dlerror());
  return handle;
}

void* BlasLibrary() {
  static auto* handle = OpenSdk("libcublas", CUBLAS_VER_MAJOR);
  return handle;
}

void* SolverLibrary() {
  static auto* handle = OpenSdk("libcusolver", CUSOLVER_VER_MAJOR);
  return handle;
}

template <typename Fn>
Fn LoadSymbol(void* library, const char* name) {
  auto* symbol = dlsym(library, name);
  TORCH_CHECK(symbol, "NVIDIA linear algebra SDK is missing ", name);
  return reinterpret_cast<Fn>(symbol);
}

#define FLAGOS_SDK_FUNCTION(library, name) LoadSymbol<decltype(&name)>(library, #name)

void Check(cublasStatus_t status, const char* routine) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, routine, " failed with cuBLAS status ",
              static_cast<int>(status));
}

void Check(cusolverStatus_t status, const char* routine) {
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, routine, " failed with cuSOLVER status ",
              static_cast<int>(status));
}

struct SolverHandles {
  SolverHandles() {
    auto create_solver = FLAGOS_SDK_FUNCTION(SolverLibrary(), cusolverDnCreate);
    auto create_blas = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasCreate_v2);
    destroy_solver = FLAGOS_SDK_FUNCTION(SolverLibrary(), cusolverDnDestroy);
    destroy_blas = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasDestroy_v2);
    Check(create_solver(&solver), "cusolverDnCreate");
    const auto status = create_blas(&blas);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_solver(solver);
      Check(status, "cublasCreate");
    }
  }

  ~SolverHandles() {
    destroy_blas(blas);
    destroy_solver(solver);
  }

  void SetStream(cudaStream_t stream) {
    static const auto set_solver_stream = FLAGOS_SDK_FUNCTION(SolverLibrary(), cusolverDnSetStream);
    static const auto set_blas_stream = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasSetStream_v2);
    Check(set_solver_stream(solver, stream), "cusolverDnSetStream");
    Check(set_blas_stream(blas, stream), "cublasSetStream");
  }

  cusolverDnHandle_t solver = nullptr;
  cublasHandle_t blas = nullptr;
  decltype(&cusolverDnDestroy) destroy_solver;
  decltype(&cublasDestroy_v2) destroy_blas;
};

template <typename T>
auto Functions();

#define FLAGOS_SOLVER_FUNCTIONS(T, letter)                                      \
  template <>                                                                \
  auto Functions<T>() {                                                      \
    return std::make_tuple(                                                   \
        FLAGOS_SDK_FUNCTION(SolverLibrary(), cusolverDn##letter##getrf_bufferSize), \
        FLAGOS_SDK_FUNCTION(SolverLibrary(), cusolverDn##letter##getrf),         \
        FLAGOS_SDK_FUNCTION(SolverLibrary(), cusolverDn##letter##getrs),         \
        FLAGOS_SDK_FUNCTION(BlasLibrary(), cublas##letter##getrsBatched));       \
  }

FLAGOS_SOLVER_FUNCTIONS(float, S)
FLAGOS_SOLVER_FUNCTIONS(double, D)
FLAGOS_SOLVER_FUNCTIONS(cuComplex, C)
FLAGOS_SOLVER_FUNCTIONS(cuDoubleComplex, Z)
#undef FLAGOS_SOLVER_FUNCTIONS

size_t MatrixWorkspaceBytes(int device) {
  static const auto workspace_config = c10::utils::get_env("CUBLAS_WORKSPACE_CONFIG");
  if (workspace_config) {
    size_t bytes = 0;
    auto remaining = std::string_view(*workspace_config);
    while (remaining.size() > 1 && remaining.front() == ':') {
      size_t kib = 0, count = 0;
      const auto end = remaining.data() + remaining.size();
      const auto size = std::from_chars(remaining.data() + 1, end, kib);
      if (size.ec != std::errc{} || size.ptr == end || *size.ptr != ':') break;
      const auto chunks = std::from_chars(size.ptr + 1, end, count);
      const auto max_kib = (std::numeric_limits<int64_t>::max() - bytes) / 1024;
      if (chunks.ec != std::errc{} || (count && kib > max_kib / count)) break;
      bytes += kib * count * 1024;
      if (chunks.ptr == end) return bytes;
      remaining.remove_prefix(chunks.ptr - remaining.data());
    }
  }
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  return major == 9 && minor == 0 ? 32 * 1024 * 1024 : (4096 * 2 + 16 * 8) * 1024;
}

struct MatrixHandle {
  MatrixHandle(const at::Tensor& like, cudaStream_t queue) : device(like.device().index()) {
    static const auto create = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasCreate_v2);
    static const auto set_stream = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasSetStream_v2);
    static const auto set_workspace = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasSetWorkspace_v2);
    destroy = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasDestroy_v2);
    workspace = at::empty({static_cast<int64_t>(MatrixWorkspaceBytes(device))},
                          like.options().dtype(at::kByte));
    Check(create(&handle), "cublasCreate");
    try {
      Check(set_stream(handle, queue), "cublasSetStream");
      Check(set_workspace(handle, workspace.mutable_data_ptr(), workspace.numel()),
            "cublasSetWorkspace");
    } catch (...) {
      destroy(handle);
      throw;
    }
  }

  ~MatrixHandle() {
    int before = device;
    cudaGetDevice(&before);
    cudaSetDevice(device);
    destroy(handle);
    cudaSetDevice(before);
  }

  cublasHandle_t handle = nullptr;
  cublasMath_t math = CUBLAS_DEFAULT_MATH;
  int device;
  decltype(&cublasDestroy_v2) destroy;
  at::Tensor workspace;
};

using GemmEx = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int,
    const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
using GemmStridedBatchedEx = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType_t, int, long long int,
    const void*, cudaDataType_t, int, long long int,
    const void*, void*, cudaDataType_t, int, long long int, int,
    cublasComputeType_t, cublasGemmAlgo_t);

void MatrixProduct(const at::Tensor& left, const at::Tensor& right, at::Tensor& out) {
  const auto device = left.device().index();
  const auto stream = GetCurrentStreamForDevice(device);
  const c10::DeviceGuard guard(left.device());
  static thread_local std::map<std::pair<int, uintptr_t>, MatrixHandle> handles;
  auto& state = handles.try_emplace(
      std::make_pair(device, reinterpret_cast<uintptr_t>(stream)), left,
      reinterpret_cast<cudaStream_t>(stream)).first->second;

  const auto m = sdk_blas::Dimension(left.size(-2));
  const auto n = sdk_blas::Dimension(right.size(-1));
  const auto k = sdk_blas::Dimension(left.size(-1));
  const auto batches = left.dim() == 2 ? 1 : sdk_blas::Dimension(left.size(0));
  const auto [a, b] = sdk_blas::ColumnMajorOperands(left, right, m, n, k);
  const auto trans_a = a.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto trans_b = b.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  const bool bf16 = left.scalar_type() == at::kBFloat16;
  const auto& context = at::globalContext();
  cublasMath_t mode = CUBLAS_DEFAULT_MATH;
  if (bf16 && batches == 1 && context.allowBF16ReductionCuBLAS() !=
      at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
    mode = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
  } else if (!bf16 && context.allowTF32CuBLAS() && !at::NoTF32Guard::should_disable_tf32()) {
    mode = CUBLAS_TF32_TENSOR_OP_MATH;
  }
  if (mode != state.math) {
    static const auto set_math = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasSetMathMode);
    Check(set_math(state.handle, mode), "cublasSetMathMode");
    state.math = mode;
  }

  const float alpha = 1.0f, beta = 0.0f;
  const auto handle = state.handle;
  if (bf16 && batches == 1) {
    static const auto gemm = LoadSymbol<GemmEx>(BlasLibrary(), "cublasGemmEx");
    Check(gemm(handle, trans_a, trans_b, n, m, k, &alpha,
               a.tensor.const_data_ptr(), CUDA_R_16BF, a.leading,
               b.tensor.const_data_ptr(), CUDA_R_16BF, b.leading, &beta,
               out.mutable_data_ptr(), CUDA_R_16BF, n,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx");
  } else if (bf16) {
    static const auto gemm = LoadSymbol<GemmStridedBatchedEx>(BlasLibrary(), "cublasGemmStridedBatchedEx");
    Check(gemm(handle, trans_a, trans_b, n, m, k, &alpha,
               a.tensor.const_data_ptr(), CUDA_R_16BF, a.leading, a.tensor.stride(0),
               b.tensor.const_data_ptr(), CUDA_R_16BF, b.leading, b.tensor.stride(0), &beta,
               out.mutable_data_ptr(), CUDA_R_16BF, n, out.stride(0), batches,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmStridedBatchedEx");
  } else if (batches == 1) {
    static const auto gemm = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasSgemm_v2);
    Check(gemm(handle, trans_a, trans_b, n, m, k, &alpha,
               a.tensor.const_data_ptr<float>(), a.leading,
               b.tensor.const_data_ptr<float>(), b.leading, &beta,
               out.mutable_data_ptr<float>(), n), "cublasSgemm");
  } else {
    static const auto gemm = FLAGOS_SDK_FUNCTION(BlasLibrary(), cublasSgemmStridedBatched);
    Check(gemm(handle, trans_a, trans_b, n, m, k, &alpha,
               a.tensor.const_data_ptr<float>(), a.leading, a.tensor.stride(0),
               b.tensor.const_data_ptr<float>(), b.leading, b.tensor.stride(0), &beta,
               out.mutable_data_ptr<float>(), n, out.stride(0), batches),
          "cublasSgemmStridedBatched");
  }
}

#undef FLAGOS_SDK_FUNCTION

struct SolveMetadata : at::meta::structured__linalg_solve_ex {
  std::array<at::Tensor, 4> outputs;

  const at::Tensor& maybe_get_output(int64_t index) override { return outputs[index]; }

  void set_output_strided(int64_t index, at::IntArrayRef sizes, at::IntArrayRef strides,
                          at::TensorOptions options, at::DimnameList) override {
    outputs[index] = strides.empty() ? at::empty(sizes, options)
                                    : at::empty_strided(sizes, strides, options);
  }

  void set_output_raw_strided(int64_t index, at::IntArrayRef sizes, at::IntArrayRef strides,
                              at::TensorOptions options, at::DimnameList names) override {
    set_output_strided(index, sizes, strides, options, names);
  }
};

int SdkDimension(int64_t value) {
  TORCH_CHECK(value <= std::numeric_limits<int>::max(), "linalg.solve dimension exceeds the NVIDIA SDK range");
  return static_cast<int>(value);
}

at::Tensor MatrixPointers(const at::Tensor& value, int batch, int64_t matrix_elements) {
  const auto address = reinterpret_cast<int64_t>(value.const_data_ptr());
  const auto stride_bytes = matrix_elements * value.element_size();
  return at::arange(batch, value.options().dtype(at::kLong)).mul(stride_bytes).add(address);
}

template <typename T>
void Solve(SolverHandles& handles, at::Tensor& lu, at::Tensor& pivots, at::Tensor& info,
           at::Tensor& rhs, cublasOperation_t operation, bool check_errors) {
  static const auto api = Functions<T>();
  const auto& [buffer_size, factor, solve, solve_batched] = api;
  const auto n = SdkDimension(lu.size(-1));
  const auto columns = SdkDimension(rhs.size(-1));
  const auto lu_batch = info.numel();
  const auto matrix_elements = static_cast<int64_t>(n) * n;
  auto* lu_data = reinterpret_cast<T*>(lu.mutable_data_ptr());
  auto* pivot_data = pivots.mutable_data_ptr<int>();
  auto* info_data = info.mutable_data_ptr<int>();
  if (lu.numel()) {
    int workspace_size = 0;
    Check(buffer_size(handles.solver, n, n, lu_data, n, &workspace_size), "getrf_bufferSize");
    auto workspace = at::empty({workspace_size}, lu.options());
    auto* buffer = reinterpret_cast<T*>(workspace.mutable_data_ptr());
    for (int64_t batch = 0; batch < lu_batch; ++batch) {
      Check(factor(handles.solver, n, n, lu_data + batch * matrix_elements, n, buffer,
                   pivot_data + batch * n, info_data + batch), "getrf");
    }
  } else {
    info.zero_();
  }
  if (check_errors) at::native::_linalg_check_errors(info, "torch.linalg.solve_ex", lu.dim() == 2);
  if (!rhs.numel()) return;

  const auto rhs_elements = static_cast<int64_t>(n) * columns;
  const auto count = SdkDimension(rhs.numel() / rhs_elements);
  auto batch_lu = lu;
  auto batch_pivots = pivots;
  if (count != lu_batch) {
    auto shape = rhs.sizes().vec();
    shape.back() = n;
    batch_lu = lu.expand(shape).mT().contiguous().mT();
    shape.pop_back();
    batch_pivots = pivots.expand(shape).contiguous();
  }
  if (n <= 8 && columns < 256) {
    // The batched cuBLAS solve preserves the small-matrix CUDA rounding path.
    auto factors = MatrixPointers(batch_lu, count, matrix_elements);
    auto solutions = MatrixPointers(rhs, count, rhs_elements);
    int argument_error = 0;
    Check(solve_batched(handles.blas, operation, n, columns,
          reinterpret_cast<const T* const*>(factors.const_data_ptr()), n,
          batch_pivots.const_data_ptr<int>(),
          reinterpret_cast<T* const*>(solutions.const_data_ptr()), n, &argument_error, count),
          "getrsBatched");
    TORCH_CHECK(argument_error == 0, "getrsBatched rejected argument ", -argument_error);
  } else {
    auto status = at::empty({1}, info.options());
    auto* factors = reinterpret_cast<T*>(batch_lu.mutable_data_ptr());
    auto* solutions = reinterpret_cast<T*>(rhs.mutable_data_ptr());
    const auto* indices = batch_pivots.const_data_ptr<int>();
    for (int batch = 0; batch < count; ++batch) {
      Check(solve(handles.solver, operation, n, columns, factors + batch * matrix_elements,
                  n, indices + batch * n, solutions + batch * rhs_elements, n,
                  status.mutable_data_ptr<int>()), "getrs");
    }
  }
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> SolveOut(
    const at::Tensor& a, const at::Tensor& b, bool left, bool check_errors,
    at::Tensor& result, at::Tensor& lu, at::Tensor& pivots, at::Tensor& info) {
  TORCH_CHECK(a.device() == b.device(), "linalg.solve inputs must share a device");
  SolveMetadata prepared;
  prepared.meta(a, b, left, check_errors);
  auto& [x, factors, indices, status] = prepared.outputs;
  const auto vector = at::native::linalg_solve_is_vector_rhs(a, b);
  auto result_view = vector ? x.unsqueeze(-1) : (left ? x : x.mT());
  // A batched vector result can have its batch and row dimensions interleaved.
  // The SDK expects each column-major matrix to occupy a separate range.
  auto rhs = result_view.mT().is_contiguous() ? result_view : at::empty_strided(
      result_view.sizes(), at::native::batched_matrix_contiguous_strides(result_view.sizes(), true),
      result_view.options());
  rhs.copy_(vector ? b.unsqueeze(-1) : (left ? b : b.mT()));
  const bool factor_transpose = a.is_contiguous() && !a.is_complex();
  factors.copy_(factor_transpose ? a.mT() : a);

  const c10::DeviceGuard guard(a.device());
  static thread_local std::map<int, SolverHandles> handles_per_device;
  auto& handles = handles_per_device.try_emplace(a.device().index()).first->second;
  handles.SetStream(reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(a.device().index())));
  const auto operation = factor_transpose == left ? CUBLAS_OP_T : CUBLAS_OP_N;
  switch (a.scalar_type()) {
    case at::kFloat: Solve<float>(handles, factors, indices, status, rhs, operation, check_errors); break;
    case at::kDouble: Solve<double>(handles, factors, indices, status, rhs, operation, check_errors); break;
    case at::kComplexFloat: Solve<cuComplex>(handles, factors, indices, status, rhs, operation, check_errors); break;
    case at::kComplexDouble: Solve<cuDoubleComplex>(handles, factors, indices, status, rhs, operation, check_errors); break;
    default: TORCH_CHECK(false, "linalg.solve does not support ", a.scalar_type());
  }
  if (!rhs.is_same(result_view)) result_view.copy_(rhs);

  std::array<at::Tensor*, 4> outputs{&result, &lu, &pivots, &info};
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& out = *outputs[i];
    const auto& value = prepared.outputs[i];
    TORCH_CHECK(out.device() == value.device() && out.scalar_type() == value.scalar_type(),
                "linalg.solve output must match the expected dtype and device");
    at::native::resize_output(out, value.sizes());
    at::assert_no_internal_overlap(out);
    out.copy_(value);
  }
  return std::tie(result, lu, pivots, info);
}

}  // namespace

at::Tensor TryMatrixProduct(const at::Tensor& left, const at::Tensor& right,
                            at::Tensor* out) {
  const auto rank = left.dim();
  const auto dtype = left.scalar_type();
  if ((rank != 2 && rank != 3) || right.dim() != rank ||
      !left.device().is_privateuseone() || left.device() != right.device() ||
      left.layout() != at::kStrided || right.layout() != at::kStrided ||
      (dtype != at::kFloat && dtype != at::kBFloat16) || right.scalar_type() != dtype ||
      left.is_neg() || right.is_neg() || left.has_names() || right.has_names() ||
      left.size(-1) != right.size(-2) || (rank == 3 && left.size(0) != right.size(0)) ||
      at::globalContext().blasPreferredBackend() != at::BlasBackend::Cublas ||
      (dtype == at::kBFloat16 && at::globalContext().allowBF16ReductionCuBLAS() ==
          at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK)) {
    return {};
  }
  auto shape = left.sizes().vec();
  shape.back() = right.size(-1);
  at::Tensor result;
  if (out) {
    if (out->device() != left.device() || out->scalar_type() != dtype ||
        !out->is_contiguous() || out->is_neg() || out->has_names() ||
        out->is_alias_of(left) || out->is_alias_of(right)) return {};
    at::native::resize_output(*out, shape);
    result = *out;
  } else {
    result = at::empty(shape, left.options());
  }
  if (result.numel() == 0) return result;
  if (left.size(-1) == 0) return result.zero_();
  MatrixProduct(left, right, result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_linalg_solve_ex.result", SolveOut);
}

}  // namespace flagos::nvidia
