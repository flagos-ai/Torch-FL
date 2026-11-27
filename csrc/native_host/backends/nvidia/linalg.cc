// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "linalg.h"

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/_linalg_solve_ex_meta.h>
#include <aten/backends/flagos/sdk_blas_layout.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/env.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <flagos.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

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

void* BlasLtLibrary() {
  static auto* handle = OpenSdk("libcublasLt", CUBLAS_VER_MAJOR);
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
    if (lt) destroy_lt(lt);
    destroy(handle);
    cudaSetDevice(before);
  }

  cublasHandle_t handle = nullptr;
  cublasLtHandle_t lt = nullptr;
  cublasMath_t math = CUBLAS_DEFAULT_MATH;
  int device;
  decltype(&cublasDestroy_v2) destroy;
  decltype(&cublasLtDestroy) destroy_lt = nullptr;
  at::Tensor workspace;
  at::Tensor lt_workspace;
};

MatrixHandle& MatrixState(const at::Tensor& input, cudaStream_t queue) {
  static thread_local std::map<std::pair<int, uintptr_t>, MatrixHandle> handles;
  return handles.try_emplace(
      std::make_pair(input.device().index(), reinterpret_cast<uintptr_t>(queue)),
      input, queue).first->second;
}

size_t LtWorkspaceBytes() {
  static const auto bytes = [] {
    constexpr size_t default_bytes = 1024 * 1024;
    const auto setting = c10::utils::get_env("CUBLASLT_WORKSPACE_SIZE");
    if (!setting) return default_bytes;
    size_t kib = 0;
    const auto end = setting->data() + setting->size();
    const auto parsed = std::from_chars(setting->data(), end, kib);
    if (parsed.ec != std::errc{} || parsed.ptr != end ||
        kib > std::numeric_limits<int64_t>::max() / 1024) return default_bytes;
    return kib * 1024;
  }();
  return bytes;
}

template <typename Handle, typename Create, typename Destroy, typename... Args>
auto LtDescriptor(Create create, Destroy destroy, Args... args) {
  Handle handle = nullptr;
  Check(create(&handle, args...), "cuBLASLt descriptor creation");
  return std::unique_ptr<std::remove_pointer_t<Handle>, Destroy>(handle, destroy);
}

uint32_t LtAlignment(const void* ptr) {
  uint32_t alignment = 256;
  while (reinterpret_cast<uintptr_t>(ptr) % alignment) alignment /= 2;
  return alignment;
}

bool FusedBFloat16Bias(const at::Tensor& left, const at::Tensor& right,
                       const at::Tensor& bias, at::Tensor& output, float alpha) {
  static const auto disabled = c10::utils::get_env("DISABLE_ADDMM_CUDA_LT") == "1";
  if (disabled || bias.dim() != 1 || !bias.is_contiguous() ||
      bias.size(0) != right.size(1) || left.size(0) <= 1 ||
      left.size(1) <= 1 || right.size(1) <= 1 ||
      at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) return false;

  static const auto create = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtCreate);
  static const auto destroy = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtDestroy);
  static const auto make_operation = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmulDescCreate);
  static const auto drop_operation = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmulDescDestroy);
  static const auto set_operation = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmulDescSetAttribute);
  static const auto make_layout = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatrixLayoutCreate);
  static const auto drop_layout = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatrixLayoutDestroy);
  static const auto make_preference = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmulPreferenceCreate);
  static const auto drop_preference = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmulPreferenceDestroy);
  static const auto set_preference = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmulPreferenceSetAttribute);
  static const auto select = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmulAlgoGetHeuristic);
  static const auto matmul = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatmul);

  const c10::DeviceGuard guard(left.device());
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(left.device().index()));
  auto& state = MatrixState(left, queue);
  if (!state.lt) {
    Check(create(&state.lt), "cublasLtCreate");
    state.destroy_lt = destroy;
  }
  static const auto unified = c10::utils::check_env("TORCH_CUBLASLT_UNIFIED_WORKSPACE") == true;
  size_t workspace_size = LtWorkspaceBytes();
  if (!unified && !state.lt_workspace.defined()) {
    state.lt_workspace = at::empty({static_cast<int64_t>(workspace_size)},
                                   left.options().dtype(at::kByte));
  }
  auto& workspace = unified ? state.workspace : state.lt_workspace;
  workspace_size = std::min(workspace_size, static_cast<size_t>(workspace.numel()));

  const auto m = sdk_blas::Dimension(left.size(0));
  const auto n = sdk_blas::Dimension(right.size(1));
  const auto k = sdk_blas::Dimension(left.size(1));
  const auto [a, b] = sdk_blas::ColumnMajorOperands(left, right, m, n, k);
  const auto trans_a = a.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto trans_b = b.transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto epilogue = CUBLASLT_EPILOGUE_BIAS;
  const auto bias_data = bias.const_data_ptr();
  auto op = LtDescriptor<cublasLtMatmulDesc_t>(make_operation, drop_operation,
                                               CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const auto set = [&](cublasLtMatmulDescAttributes_t attribute, const auto& value) {
    Check(set_operation(op.get(), attribute, &value, sizeof(value)), "cublasLtMatmulDescSetAttribute");
  };
  set(CUBLASLT_MATMUL_DESC_TRANSA, trans_a);
  set(CUBLASLT_MATMUL_DESC_TRANSB, trans_b);
  set(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
  set(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_data);

  const auto a_layout = LtDescriptor<cublasLtMatrixLayout_t>(make_layout, drop_layout,
      CUDA_R_16BF, a.transpose ? k : n, a.transpose ? n : k, a.leading);
  const auto b_layout = LtDescriptor<cublasLtMatrixLayout_t>(make_layout, drop_layout,
      CUDA_R_16BF, b.transpose ? m : k, b.transpose ? k : m, b.leading);
  const auto c_layout = LtDescriptor<cublasLtMatrixLayout_t>(make_layout, drop_layout,
      CUDA_R_16BF, n, m, n);
  const auto preference = LtDescriptor<cublasLtMatmulPreference_t>(make_preference, drop_preference);
  const auto prefer = [&](cublasLtMatmulPreferenceAttributes_t attribute, const auto& value) {
    Check(set_preference(preference.get(), attribute, &value, sizeof(value)), "cublasLtMatmulPreferenceSetAttribute");
  };
  const auto reduction = at::globalContext().allowBF16ReductionCuBLAS();
  if (reduction != at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
    const uint32_t mask = reduction == at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
        ? CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_NONE
        : CUBLASLT_REDUCTION_SCHEME_NONE;
    prefer(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
  }
  prefer(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size);
  prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, LtAlignment(a.tensor.const_data_ptr()));
  prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, LtAlignment(b.tensor.const_data_ptr()));
  prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, LtAlignment(output.const_data_ptr()));
  prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, LtAlignment(bias_data));

  cublasLtMatmulHeuristicResult_t config{};
  int returned = 0;
  const auto status = select(state.lt, op.get(), a_layout.get(), b_layout.get(),
                             c_layout.get(), c_layout.get(), preference.get(), 1, &config, &returned);
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || (status == CUBLAS_STATUS_SUCCESS && !returned)) return false;
  Check(status, "cublasLtMatmulAlgoGetHeuristic");
  constexpr float beta = 0;
  auto* out_data = output.mutable_data_ptr();
  return matmul(state.lt, op.get(), &alpha, a.tensor.const_data_ptr(), a_layout.get(),
                b.tensor.const_data_ptr(), b_layout.get(), &beta, out_data, c_layout.get(),
                out_data, c_layout.get(), &config.algo, workspace.mutable_data_ptr(),
                workspace_size, queue) == CUBLAS_STATUS_SUCCESS;
}

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

void MatrixProduct(const at::Tensor& left, const at::Tensor& right, at::Tensor& out,
                   float beta, float alpha) {
  const auto device = left.device().index();
  const c10::DeviceGuard guard(left.device());
  auto& state = MatrixState(left, reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device)));

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

at::Tensor Scale(const at::Tensor& input, float alpha) {
  static const auto create = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtCreate);
  static const auto destroy = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtDestroy);
  static const auto init_transform =
      FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatrixTransformDescInit);
  static const auto init_layout =
      FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatrixLayoutInit);
  static const auto transform = FLAGOS_SDK_FUNCTION(BlasLtLibrary(), cublasLtMatrixTransform);
  using Handle = decltype(LtDescriptor<cublasLtHandle_t>(create, destroy));
  static thread_local std::map<int, Handle> handles;
  const auto device = input.device();
  const c10::DeviceGuard guard(device);
  auto handle = handles.find(device.index());
  if (handle == handles.end()) {
    handle = handles.emplace(device.index(), LtDescriptor<cublasLtHandle_t>(create, destroy)).first;
  }
  const auto dtype = input.scalar_type() == at::kBFloat16 ? CUDA_R_16BF : CUDA_R_32F;
  const auto size = input.numel();
  cublasLtMatrixTransformDescOpaque_t descriptor;
  cublasLtMatrixLayoutOpaque_t layout;
  Check(init_transform(&descriptor, CUDA_R_32F), "cublasLtMatrixTransformDescInit");
  Check(init_layout(&layout, dtype, size, 1, size), "cublasLtMatrixLayoutInit");
  auto output = at::empty_strided(input.sizes(), input.strides(), input.options());
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device.index()));
  constexpr float zero = 0;
  const auto status = transform(handle->second.get(), &descriptor, &alpha,
      input.const_data_ptr(), &layout, &zero, nullptr, nullptr,
      output.mutable_data_ptr(), &layout, queue);
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH)
    return {};
  Check(status, "cublasLtMatrixTransform");
  return output;
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

at::Tensor TryScale(const at::Tensor& input, float alpha) {
  const auto dtype = input.scalar_type();
  if ((dtype != at::kBFloat16 && dtype != at::kFloat) ||
      !input.device().is_privateuseone() || input.layout() != at::kStrided ||
      input.is_neg() || input.has_names() || !input.is_non_overlapping_and_dense() ||
      input.numel() == 0 || input.numel() > std::numeric_limits<int>::max() ||
      !std::isnormal(alpha)) {
    return {};
  }
  return Scale(input, alpha);
}

at::Tensor TryMatrixProduct(const at::Tensor& left, const at::Tensor& right,
                            at::Tensor* out, const at::Tensor* bias, float beta,
                            float alpha) {
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
  if (bias && (rank != 2 || (dtype == at::kFloat && left.size(0) != 1) ||
                 bias->scalar_type() != dtype || bias->device() != left.device() ||
                 bias->layout() != at::kStrided || bias->is_neg() || bias->has_names() ||
                 !at::is_expandable_to(bias->sizes(), shape))) return {};
  at::Tensor result;
  if (out) {
    if (out->device() != left.device() || out->scalar_type() != dtype ||
        !out->is_contiguous() || out->is_neg() || out->has_names() ||
        out->is_alias_of(left) || out->is_alias_of(right) ||
        (bias && out->is_alias_of(*bias))) return {};
    at::native::resize_output(*out, shape);
    result = *out;
  } else {
    result = at::empty(shape, left.options());
  }
  if (result.numel() == 0) return result;
  if (bias && dtype == at::kBFloat16 && beta == 1 &&
      FusedBFloat16Bias(left, right, *bias, result, alpha)) return result;
  if (bias && beta != 0) result.copy_(bias->expand(shape));
  if (left.size(-1) == 0) return beta == 0 ? result.zero_() : result.mul_(beta);
  MatrixProduct(left, right, result, beta, alpha);
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_linalg_solve_ex.result", SolveOut);
}

}  // namespace flagos::nvidia
