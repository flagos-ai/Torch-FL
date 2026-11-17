// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Host adapters for published NVIDIA numerical libraries. No tensor is retagged
// and no CUDA tensor or ATen CUDA redispatch is created here.
#pragma once

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cuda/cuBlasCommonArgs.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <flagos.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>

namespace at::native { void rsqrt_kernel_cuda(at::TensorIteratorBase&); }

namespace at::native::flagos::vendor {
namespace py = pybind11;

inline std::string LibraryOf(void* address) {
  Dl_info info{};
  TORCH_CHECK(address && dladdr(address, &info) && info.dli_fname,
              "cannot identify a loaded native vendor library");
  return info.dli_fname;
}

inline py::dict Libraries() {
  py::dict result;
  result["torch_cuda"] = LibraryOf(reinterpret_cast<void*>(&at::native::rsqrt_kernel_cuda));
  result["c10_cuda"] = LibraryOf(reinterpret_cast<void*>(&c10::cuda::getCurrentCUDAStream));
  result["cublas"] = LibraryOf(reinterpret_cast<void*>(&cublasCreate));
  result["cublas_lt"] = LibraryOf(dlsym(RTLD_DEFAULT, "cublasLtCreate"));
  return result;
}

inline c10::DeviceIndex VendorDevice(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_privateuseone(),
              "native vendor entry requires a flagos tensor");
  return input.device().index();
}

inline c10::cuda::CUDAStream VendorStream(const at::Tensor& input) {
  const auto device = VendorDevice(input);
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());
  });
  return c10::cuda::getStreamFromExternal(
      reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device)), device);
}

struct VendorScope {
  c10::cuda::CUDAStreamGuard guard;
  explicit VendorScope(const at::Tensor& input) : guard(VendorStream(input)) {}
};

struct VendorDeviceScope {
  // BLAS receives the FlagOS stream explicitly through its owned handle. It
  // needs the correct device, but does not consult ATen's CUDA stream registry.
  c10::cuda::CUDAGuard guard;
  explicit VendorDeviceScope(const at::Tensor& input) : guard(VendorDevice(input)) {}
};

inline void DescribeCudaMemory(at::TensorIteratorBase& iterator) {
  // OperandInfo is the iterator's pointer descriptor, not TensorImpl. NVIDIA
  // memory is physically CUDA memory; input/output devices and key sets remain
  // PrivateUse1. The original vendor kernel checks these descriptors.
  for (int i = 0; i < iterator.ntensors(); ++i) {
    const auto device = iterator.tensor(i).device();
    TORCH_CHECK(device.is_privateuseone(), "unexpected native vendor operand device");
    iterator.operand(i).device = c10::Device(c10::DeviceType::CUDA, device.index());
  }
}

template <class Impl>
struct StructuredOutput : Impl {
  at::Tensor output;

  const at::Tensor& maybe_get_output(int64_t index) override {
    TORCH_CHECK(index == 0, "native vendor entry requires one output");
    return output;
  }

  void set_output_raw_strided(int64_t index, at::IntArrayRef sizes,
                             at::IntArrayRef strides, at::TensorOptions options,
                             at::DimnameList names) override {
    TORCH_CHECK(index == 0 && options.device().is_privateuseone(),
                "native vendor meta must allocate a flagos output");
    TORCH_CHECK(options.layout() == at::kStrided && !options.pinned_memory(),
                "native vendor meta requires unpinned strided device storage");
    // VendorScope already selected the actual device. Use the same tensor
    // constructors and allocator as FlagOS's empty kernels, without a second
    // dispatcher traversal solely to allocate the structured result.
    auto* allocator = at::GetAllocator(at::kPrivateUse1);
    constexpr c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
    const auto dtype = c10::typeMetaToScalarType(options.dtype());
    output = strides.empty()
        ? at::detail::empty_generic(sizes, allocator, keys, dtype, options.memory_format_opt())
        : at::detail::empty_strided_generic(sizes, strides, allocator, keys, dtype);
    TORCH_INTERNAL_ASSERT(output.device() == options.device());
    if (!names.empty()) at::namedinference::propagate_names(output, names);
    if constexpr (std::is_base_of_v<at::TensorIteratorBase, Impl>) {
      Impl::set_output_raw_strided(index, sizes, strides, options, names);
    }
  }

  void set_output_strided(int64_t index, at::IntArrayRef sizes,
                         at::IntArrayRef strides, at::TensorOptions options,
                         at::DimnameList names) override {
    set_output_raw_strided(index, sizes, strides, options, names);
  }
};

template <class Impl, class... Args>
at::Tensor StructuredCall(const at::Tensor& input, const Args&... args) {
  StructuredOutput<Impl> kernel;
  kernel.meta(input, args...);
  if constexpr (std::is_base_of_v<at::TensorIteratorBase, Impl>) {
    DescribeCudaMemory(kernel);
  }
  kernel.impl(input, args..., kernel.output);
  TORCH_INTERNAL_ASSERT(kernel.output.device().is_privateuseone());
  return kernel.output;
}

template <class Impl, class Stub>
at::Tensor Reduction(Stub& stub, const at::Tensor& input,
                     at::OptionalIntArrayRef dims, bool keepdim,
                     std::optional<at::ScalarType> dtype) {
  StructuredOutput<Impl> kernel;
  kernel.meta(input, dims, keepdim, dtype);
  if (input.numel() == 0) {
    // The policy admits the mean family, whose empty identity is NaN. This
    // fill uses the existing FlagGems route; no reduction kernel accepts N=0.
    kernel.output.fill_(std::numeric_limits<double>::quiet_NaN());
    return kernel.output;
  }
  auto iterator = at::native::make_reduction(
      "native vendor reduction", kernel.output, input, dims, keepdim,
      kernel.output.scalar_type());
  DescribeCudaMemory(iterator);
  stub(at::kCUDA, iterator);
  return kernel.output;
}

inline void CheckBlas(cublasStatus_t status, const char* call) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "native NVIDIA ", call, " failed: ", static_cast<int>(status));
}

struct BlasState {
  cublasHandle_t handle = nullptr;
  at::Tensor workspace;
  int device;
  cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;

  BlasState(int device, cudaStream_t stream) : device(device) {
    CheckBlas(cublasCreate(&handle), "cublasCreate");
    CheckBlas(cublasSetStream(handle, stream), "cublasSetStream");
    CheckBlas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
              "cublasSetPointerMode");
  }
  ~BlasState() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cublasDestroy(handle);
    if (previous >= 0) cudaSetDevice(previous);
  }
};

inline BlasState& BlasHandle(const at::Tensor& input) {
  const int device = input.device().index();
  const auto stream = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device));
  using Key = std::pair<int, uintptr_t>;
  static thread_local std::map<Key, std::unique_ptr<BlasState>> states;
  auto& state = states[{device, reinterpret_cast<uintptr_t>(stream)}];
  if (!state) state = std::make_unique<BlasState>(device, stream);
  // Ask the vendor for its workspace policy, including the current device and
  // CUBLAS_WORKSPACE_CONFIG. These are resource caches, never model plans.
  const auto bytes = at::cuda::getChosenWorkspaceSize();
  if (!state->workspace.defined() ||
      static_cast<size_t>(state->workspace.numel()) != bytes) {
    state->workspace = at::empty({static_cast<int64_t>(bytes)},
                                input.options().dtype(at::kByte));
    CheckBlas(cublasSetWorkspace(state->handle, state->workspace.mutable_data_ptr(), bytes),
              "cublasSetWorkspace");
  }
  return *state;
}

struct BatchOperand {
  at::Tensor tensor;
  cublasOperation_t operation;
  int64_t leading;
  int64_t batch_stride;

  BatchOperand(const at::Tensor& input, bool transpose_result, int64_t m, int64_t n) {
    const int fast = transpose_result ? 2 : 1;
    const int slow = transpose_result ? 1 : 2;
    tensor = input;
    if (input.stride(fast) == 1 && input.stride(slow) >= std::max<int64_t>(1, m)) {
      operation = CUBLAS_OP_N;
      leading = input.stride(slow);
    } else if (input.stride(slow) == 1 && input.stride(fast) >= std::max<int64_t>(1, n)) {
      operation = CUBLAS_OP_T;
      leading = input.stride(fast);
    } else {
      if (!input.is_contiguous() || input.stride(1) == 0 || input.stride(2) == 0)
        tensor = input.clone(at::MemoryFormat::Contiguous);
      operation = transpose_result ? CUBLAS_OP_N : CUBLAS_OP_T;
      leading = tensor.stride(1);
    }
    batch_stride = tensor.stride(0);
  }
};

inline void AdjustLeading(cublasOperation_t opa, cublasOperation_t opb,
                           int64_t m, int64_t n, int64_t k,
                           int64_t& lda, int64_t& ldb, int64_t& ldc) {
  // Match the vendor BLAS argument convention for one-element dimensions.
  if (n <= 1) ldc = std::max<int64_t>(m, 1);
  if (opa == CUBLAS_OP_N) {
    if (k <= 1) lda = std::max<int64_t>(m, 1);
  } else if (m <= 1) lda = std::max<int64_t>(k, 1);
  if (opb == CUBLAS_OP_N) {
    if (n <= 1) ldb = std::max<int64_t>(k, 1);
  } else if (k <= 1) ldb = std::max<int64_t>(n, 1);
}

inline void BlasCall(const at::Tensor& a, const at::Tensor& b, const at::Tensor& out,
                     cublasOperation_t opa, cublasOperation_t opb,
                     int64_t m, int64_t n, int64_t k,
                     int64_t lda, int64_t ldb, int64_t ldc, int64_t batches,
                     int64_t stride_a = 0, int64_t stride_b = 0, int64_t stride_c = 0) {
  AdjustLeading(opa, opb, m, n, k, lda, ldb, ldc);
  for (const auto size : {m, n, k, lda, ldb, ldc, batches}) {
    TORCH_CHECK(size > 0 && size <= std::numeric_limits<int>::max(),
                "native vendor matrix dimension exceeds cuBLAS limits");
  }
  auto& state = BlasHandle(a);
  const auto handle = state.handle;
  cublasMath_t math = CUBLAS_DEFAULT_MATH;
  if (batches == 1) {
    const auto reduction = at::globalContext().allowBF16ReductionCuBLAS();
    TORCH_CHECK(reduction != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
                "disabling split-K requires the cuBLASLt backend");
    if (reduction != at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK)
      math = static_cast<cublasMath_t>(math | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }
  if (state.math_mode != math) {
    CheckBlas(cublasSetMathMode(handle, math), "cublasSetMathMode");
    state.math_mode = math;
  }
  const float alpha = 1.0f, beta = 0.0f;
  if (batches == 1) {
    CheckBlas(cublasGemmEx(handle, opa, opb, m, n, k, &alpha,
                          a.const_data_ptr(), CUDA_R_16BF, lda,
                          b.const_data_ptr(), CUDA_R_16BF, ldb, &beta,
                          out.mutable_data_ptr(), CUDA_R_16BF, ldc,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx");
  } else {
    CheckBlas(cublasGemmStridedBatchedEx(
                  handle, opa, opb, m, n, k, &alpha,
                  a.const_data_ptr(), CUDA_R_16BF, lda, stride_a,
                  b.const_data_ptr(), CUDA_R_16BF, ldb, stride_b, &beta,
                  out.mutable_data_ptr(), CUDA_R_16BF, ldc, stride_c,
                  batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "cublasGemmStridedBatchedEx");
  }
}

template <class Impl, int Rank>
at::Tensor Matrix(const at::Tensor& input, const at::Tensor& other) {
  TORCH_CHECK(input.dim() == Rank && other.dim() == Rank,
              "native vendor matrix rank mismatch");
  TORCH_CHECK(input.device() == other.device() && input.layout() == at::kStrided &&
              other.layout() == at::kStrided, "native vendor matrix device/layout mismatch");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && other.scalar_type() == at::kBFloat16,
              "native vendor matrix precision entry supports BF16 inputs");
  TORCH_CHECK(at::globalContext().blasPreferredBackend() == at::BlasBackend::Cublas,
              "native vendor matrix precision entry requires the validated cuBLAS backend");
  StructuredOutput<Impl> kernel;
  kernel.meta(input, other);
  if (kernel.output.numel() == 0) return kernel.output;
  if (input.size(-1) == 0) return kernel.output.zero_();
  if constexpr (Rank == 2) {
    at::native::cublasCommonArgs args(input, other, kernel.output);
    BlasCall(*args.mata, *args.matb, *args.result,
             args.transa == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T,
             args.transb == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T,
             args.m, args.n, args.k, args.lda, args.ldb, args.result_ld, 1);
    if (!args.result->is_same(kernel.output)) kernel.output.copy_(*args.result);
  } else {
    TORCH_INTERNAL_ASSERT(kernel.output.is_contiguous());
    const bool transpose = kernel.output.stride(1) != 1;
    const int leading = transpose ? 1 : 2;
    const int64_t m = kernel.output.size(transpose ? 2 : 1);
    const int64_t n = kernel.output.size(leading), k = input.size(2);
    BatchOperand a(transpose ? other : input, transpose, m, k);
    BatchOperand b(transpose ? input : other, transpose, k, n);
    BlasCall(a.tensor, b.tensor, kernel.output, a.operation, b.operation,
             m, n, k, a.leading, b.leading, kernel.output.stride(leading), input.size(0),
             a.batch_stride, b.batch_stride, kernel.output.stride(0));
  }
  return kernel.output;
}
}  // namespace at::native::flagos::vendor
