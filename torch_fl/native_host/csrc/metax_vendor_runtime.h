// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// CPU host descriptors for an original published vendor GPU stub. The included
// vendor header supplies its original environment getters, never flag stubs.
#pragma once

#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/mean_native.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <flagos.h>
#include <ATen/native/cuda/maca_kernels/Context_.h>
#include <dlfcn.h>
#include <atomic>
#include <limits>

namespace at::cuda { at::Tensor silu(const at::Tensor&); }

namespace at::native::flagos::metax_vendor {
namespace py = pybind11;

inline std::string LibraryOf(void* address) {
  Dl_info info{};
  TORCH_CHECK(address && dladdr(address, &info) && info.dli_fname,
              "cannot identify a MetaX vendor numerical library");
  return info.dli_fname;
}

inline py::dict Libraries() {
  py::dict result;
  result["torch_cuda"] = LibraryOf(reinterpret_cast<void*>(&at::cuda::silu));
  result["c10_cuda"] = LibraryOf(reinterpret_cast<void*>(&c10::cuda::getCurrentCUDAStream));
  return result;
}

inline c10::cuda::CUDAStream StreamFor(const at::Tensor& input) {
  TORCH_CHECK(input.device().is_privateuseone() && input.layout() == at::kStrided,
              "MetaX vendor entry requires strided FlagOS device storage");
  const auto device = input.device().index();
  return c10::cuda::getStreamFromExternal(
      reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device)), device);
}

template <class Impl>
struct StructuredOutput : Impl {
  at::Tensor output;
  const at::Tensor& maybe_get_output(int64_t index) override {
    TORCH_CHECK(index == 0, "MetaX vendor entry requires one output");
    return output;
  }
  void set_output_raw_strided(int64_t index, at::IntArrayRef sizes,
                             at::IntArrayRef strides, at::TensorOptions options,
                             at::DimnameList names) override {
    TORCH_CHECK(index == 0 && options.device().is_privateuseone(),
                "MetaX vendor meta requires FlagOS output storage");
    output = strides.empty() ? at::empty(sizes, options)
                            : at::empty_strided(sizes, strides, options);
    if (!names.empty()) at::namedinference::propagate_names(output, names);
  }
  void set_output_strided(int64_t index, at::IntArrayRef sizes,
                         at::IntArrayRef strides, at::TensorOptions options,
                         at::DimnameList names) override {
    set_output_raw_strided(index, sizes, strides, options, names);
  }
};

template <class Impl, class... Args>
at::Tensor Structured(const at::Tensor& input, const Args&... args) {
  c10::cuda::CUDAStreamGuard guard(StreamFor(input));
  StructuredOutput<Impl> kernel;
  kernel.meta(input, args...);
  kernel.impl(input, args..., kernel.output);
  TORCH_CHECK(kernel.output.device() == input.device(),
              "MetaX vendor result changed its device");
  return kernel.output;
}

template <class Stub>
at::Tensor Unary(Stub& stub, const at::Tensor& input) {
  TORCH_CHECK(input.device().is_privateuseone() && input.layout() == at::kStrided,
              "MetaX vendor entry requires strided FlagOS device storage");
  auto output = at::empty_like(input);
  if (!input.numel()) return output;
  const auto device = input.device().index();
  c10::cuda::CUDAStreamGuard guard(StreamFor(input));
  auto iterator = at::TensorIterator::unary_op(output, input);
  for (int index = 0; index < iterator.ntensors(); ++index) {
    TORCH_CHECK(iterator.tensor(index).device() == input.device(),
                "unexpected MetaX vendor iterator operand");
    iterator.operand(index).device = c10::Device(c10::DeviceType::CUDA, device);
  }
  stub(at::kCUDA, iterator);
  return output;
}

template <class Impl, class Stub>
at::Tensor Reduction(Stub& stub, const at::Tensor& input,
                     at::OptionalIntArrayRef dims, bool keepdim,
                     std::optional<at::ScalarType> dtype) {
  c10::cuda::CUDAStreamGuard guard(StreamFor(input));
  StructuredOutput<Impl> kernel;
  kernel.meta(input, dims, keepdim, dtype);
  if (!input.numel()) {
    kernel.output.fill_(std::numeric_limits<float>::quiet_NaN());
    return kernel.output;
  }
  auto iterator = at::native::make_reduction(
      "native MetaX mean", kernel.output, input, dims, keepdim,
      kernel.output.scalar_type());
  for (int index = 0; index < iterator.ntensors(); ++index)
    iterator.operand(index).device = c10::Device(c10::DeviceType::CUDA, input.device().index());
  stub(at::kCUDA, iterator);
  return kernel.output;
}
}  // namespace at::native::flagos::metax_vendor
