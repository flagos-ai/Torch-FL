// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// CPU host descriptors for an original published vendor GPU stub. The included
// vendor header supplies its original environment getters, never flag stubs.
#pragma once

#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <ATen/record_function.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <flagos.h>
#include <ATen/native/cuda/maca_kernels/Context_.h>
#include <dlfcn.h>
#include <atomic>

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

template <class Stub>
at::Tensor Unary(Stub& stub, const at::Tensor& input) {
  TORCH_CHECK(input.device().is_privateuseone() && input.layout() == at::kStrided,
              "MetaX vendor entry requires strided FlagOS device storage");
  auto output = at::empty_like(input);
  if (!input.numel()) return output;
  const auto device = input.device().index();
  const auto stream = c10::cuda::getStreamFromExternal(
      reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device)), device);
  c10::cuda::CUDAStreamGuard guard(stream);
  auto iterator = at::TensorIterator::unary_op(output, input);
  for (int index = 0; index < iterator.ntensors(); ++index) {
    TORCH_CHECK(iterator.tensor(index).device() == input.device(),
                "unexpected MetaX vendor iterator operand");
    iterator.operand(index).device = c10::Device(c10::DeviceType::CUDA, device);
  }
  stub(at::kCUDA, iterator);
  return output;
}
}  // namespace at::native::flagos::metax_vendor
