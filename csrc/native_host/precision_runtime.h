// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Host adapters for published vendor numerical libraries. No tensor is retagged
// and no CUDA tensor or ATen CUDA redispatch is created here.
#pragma once

#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#if defined(FLAGOS_NATIVE_HYGON)
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/native/cuda/cuBlasCommonArgs.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#endif
#include <ATen/record_function.h>
#include <dlfcn.h>
#include <flagos.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
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
#if defined(FLAGOS_NATIVE_HYGON)
  result["torch_hip"] = LibraryOf(reinterpret_cast<void*>(&at::native::rsqrt_kernel_cuda));
  result["c10_hip"] = LibraryOf(reinterpret_cast<void*>(&c10::hip::getCurrentHIPStream));
  for (const auto* name : {"hipblasCreate", "hipblasLtCreate", "rocblas_gemm_ex"})
    result[name] = LibraryOf(dlsym(RTLD_DEFAULT, name));
#else
  result["torch_cuda"] = LibraryOf(reinterpret_cast<void*>(&at::native::rsqrt_kernel_cuda));
  result["c10_cuda"] = LibraryOf(reinterpret_cast<void*>(&c10::cuda::getCurrentCUDAStream));
#if defined(FLAGOS_NATIVE_THEAD)
  result["cudart"] = LibraryOf(reinterpret_cast<void*>(&cudaGetDevice));
#else
  result["cublas"] = LibraryOf(reinterpret_cast<void*>(&cublasCreate));
  result["cublas_lt"] = LibraryOf(dlsym(RTLD_DEFAULT, "cublasLtCreate"));
#endif
#endif
  return result;
}

inline c10::DeviceIndex VendorDevice(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_privateuseone(),
              "native vendor entry requires a flagos tensor");
  return input.device().index();
}

#if defined(FLAGOS_NATIVE_HYGON)
inline c10::hip::HIPStream VendorStream(c10::DeviceIndex device) {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    c10::hip::HIPCachingAllocator::init(c10::hip::device_count());
  });
  return c10::hip::getStreamFromExternal(
      reinterpret_cast<hipStream_t>(GetCurrentStreamForDevice(device)), device);
}
inline c10::hip::HIPStream VendorStream(const at::Tensor& input) {
  return VendorStream(VendorDevice(input));
}

struct VendorScope {
  std::optional<c10::hip::HIPStreamGuard> guard;
  explicit VendorScope(c10::DeviceIndex device) {
    const auto stream = VendorStream(device);
    // Default and external-null stream IDs can denote the same HIP handle.
    // Avoid exchanging vendor TLS when both the device and handle match.
    if (c10::hip::current_device() != stream.device_index() ||
        c10::hip::getCurrentHIPStream(stream.device_index()).stream() != stream.stream())
      guard.emplace(stream);
  }
  explicit VendorScope(const at::Tensor& input) : VendorScope(VendorDevice(input)) {}
};

class LibraryStreamScope {
 public:
  explicit LibraryStreamScope(int device) : device_(device) {}
  LibraryStreamScope& Enter() {
    TORCH_CHECK(!active_, "native stream scope is already active");
    TORCH_CHECK(device_ >= 0 && device_ < c10::hip::device_count(), "invalid HCU device index");
    active_.emplace(static_cast<c10::DeviceIndex>(device_));
    return *this;
  }
  bool Exit(py::object, py::object, py::object) {
    active_.reset();
    return false;
  }
 private:
  int device_;
  std::optional<VendorScope> active_;
};

inline void BindStreamScope(py::module_& module) {
  py::class_<LibraryStreamScope>(module, "StreamScope")
      .def(py::init<int>())
      .def("__enter__", &LibraryStreamScope::Enter, py::return_value_policy::reference_internal)
      .def("__exit__", &LibraryStreamScope::Exit);
  module.def("_stream_state", [](int device) {
    return py::make_tuple(c10::hip::current_device(),
        reinterpret_cast<uintptr_t>(c10::hip::getCurrentHIPStream(device).stream()));
  });
}

#else
inline c10::cuda::CUDAStream VendorStream(const at::Tensor& input) {
  const auto device = VendorDevice(input);
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());
  });
  const auto stream = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device));
  if (stream == cudaStreamLegacy) return c10::cuda::getDefaultCUDAStream(device);
  return c10::cuda::getStreamFromExternal(stream, device);
}

struct VendorScope {
  std::optional<c10::cuda::CUDAStreamGuard> guard;
  explicit VendorScope(const at::Tensor& input) {
    const auto stream = VendorStream(input);
    int current = -1;
    if (::GetDevice(&current) != Success || current != stream.device_index() ||
        c10::cuda::getCurrentCUDAStream(stream.device_index()) != stream) {
      guard.emplace(stream);
    }
  }
};

struct VendorDeviceScope {
  // BLAS receives the FlagOS stream explicitly through its owned handle. It
  // needs the correct device, but does not consult ATen's CUDA stream registry.
  std::optional<c10::cuda::CUDAGuard> guard;
  explicit VendorDeviceScope(const at::Tensor& input) {
    const auto device = VendorDevice(input);
    int current = -1;
    if (::GetDevice(&current) != Success || current != device) guard.emplace(device);
  }
};

#endif

template <at::ScalarType Dtype, class... Lists>
bool CanBatchForeach(at::TensorList self, Lists... lists) {
  if (self.empty() || !self.front().defined() || !self.front().is_privateuseone())
    return false;

  const auto device = self.front().device();
  auto eligible = [device](at::TensorList values) {
    return std::all_of(values.begin(), values.end(), [device](const at::Tensor& tensor) {
      return tensor.defined() && tensor.device() == device &&
          tensor.layout() == at::kStrided && tensor.scalar_type() == Dtype &&
          tensor.is_contiguous() && !tensor.is_neg() && !tensor.is_conj();
    });
  };
  return eligible(self) && (eligible(lists) && ...);
}

inline void DescribeCudaMemory(at::TensorIteratorBase& iterator,
                               bool allow_cpu_scalars = false) {
  // OperandInfo is the iterator's pointer descriptor, not TensorImpl. NVIDIA
  // memory is physically CUDA memory; input/output devices and key sets remain
  // PrivateUse1. The original vendor kernel checks these descriptors.
  for (int i = 0; i < iterator.ntensors(); ++i) {
    const auto device = iterator.tensor(i).device();
    if (allow_cpu_scalars && iterator.is_cpu_scalar(i)) continue;
    TORCH_CHECK(device.is_privateuseone(), "unexpected native vendor operand device");
    iterator.operand(i).device = c10::Device(c10::DeviceType::CUDA, device.index());
  }
}

template <class Impl>
struct StructuredOutput : Impl {
  at::Tensor output;
  const at::Tensor* destination = nullptr;

  explicit StructuredOutput(const at::Tensor* out = nullptr) : destination(out) {}

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
    const auto dtype = c10::typeMetaToScalarType(options.dtype());
    if (destination && destination->device() == options.device() &&
        destination->layout() == at::kStrided && destination->scalar_type() == dtype &&
        destination->sizes() == sizes && names.empty() && !destination->has_names() &&
        !destination->is_neg() && !destination->is_conj() &&
        (strides.empty()
            ? destination->is_contiguous(options.memory_format_opt().value_or(
                  at::MemoryFormat::Contiguous))
            : destination->strides() == strides)) {
      output = *destination;
    } else {
      // VendorScope already selected the device. Use FlagOS's ordinary
      // storage allocator directly for the structured result.
      auto* allocator = at::GetAllocator(at::kPrivateUse1);
      constexpr c10::DispatchKeySet keys(c10::DispatchKey::PrivateUse1);
      output = strides.empty()
          ? at::detail::empty_generic(sizes, allocator, keys, dtype, options.memory_format_opt())
          : at::detail::empty_strided_generic(sizes, strides, allocator, keys, dtype);
    }
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

template <class Impl>
at::Tensor Concatenate(const at::ITensorListRef& tensors, int64_t dim) {
  const auto device = tensors.front().device();
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device, "cat inputs must share a device");
  }
  StructuredOutput<Impl> kernel;
  const auto layout = kernel.meta(tensors, dim);
  kernel.impl(tensors, layout.dim, layout.valid, layout.all_contiguous,
              layout.all_same_dtype, layout.all_same_sizes_and_stride,
              layout.memory_format, kernel.output);
  return kernel.output;
}

template <class Meta, class Stub>
at::Tensor BinaryPointwise(Stub& stub, const at::Tensor& lhs, const at::Tensor& rhs) {
  StructuredOutput<Meta> kernel;
  kernel.meta(lhs, rhs);
  DescribeCudaMemory(kernel, /*allow_cpu_scalars=*/true);
  stub(at::kCUDA, kernel);
  return kernel.output;
}

template <class Impl>
at::Tensor& SoftmaxBackwardOut(const at::Tensor& grad, const at::Tensor& output,
                              int64_t dim, at::ScalarType dtype, at::Tensor& out) {
  TORCH_CHECK(grad.device() == output.device() && out.device() == grad.device(),
              "softmax backward requires all tensors on the same device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, grad);
  at::assert_no_overlap(out, output);
  StructuredOutput<Impl> kernel(&out);
  kernel.meta(grad, output, dim, dtype);
  TORCH_CHECK(kernel.output.scalar_type() == out.scalar_type(),
              "softmax backward output dtype must match the result");
  if constexpr (std::is_base_of_v<at::TensorIteratorBase, Impl>) {
    DescribeCudaMemory(kernel);
  }
  kernel.impl(grad, output, dim, dtype, kernel.output);
  if (kernel.output.is_same(out)) return out;
  at::native::resize_output(out, kernel.output.sizes());
  return out.copy_(kernel.output);
}

template <class Impl, class Stub>
at::Tensor Reduction(Stub& stub, const at::Tensor& input,
                     at::OptionalIntArrayRef dims, bool keepdim,
                     std::optional<at::ScalarType> dtype,
                     double empty_value = std::numeric_limits<double>::quiet_NaN()) {
  StructuredOutput<Impl> kernel;
  kernel.meta(input, dims, keepdim, dtype);
  if (input.numel() == 0) {
    // Fill the operation's empty result through FlagGems. The reduction
    // kernels require a nonempty input.
#if defined(FLAGOS_NATIVE_HYGON)
    return at::full(kernel.output.sizes(), empty_value, kernel.output.options());
#else
    kernel.output.fill_(empty_value);
    return kernel.output;
#endif
  }
  auto iterator = at::native::make_reduction(
      "native vendor reduction", kernel.output, input, dims, keepdim,
      kernel.output.scalar_type());
  DescribeCudaMemory(iterator);
  stub(at::kCUDA, iterator);
  return kernel.output;
}

#if !defined(FLAGOS_NATIVE_HYGON)
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
#endif
}  // namespace at::native::flagos::vendor
