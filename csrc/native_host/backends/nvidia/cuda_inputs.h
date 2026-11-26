// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "aten/cuda_tensor_view.h"
#include "precision_runtime.h"
#include "runtime/allocator/caching_device_allocator.h"

#include <c10/cuda/CUDAException.h>
#include <c10/util/SmallVector.h>

namespace at::native::flagos::vendor {

// Use the CUDA API for vendor operations that must see CUDA tensor metadata.
// Each view retains the original allocation and never changes its TensorImpl.
template <typename Call>
decltype(auto) WithCudaInputs(at::TensorList inputs, Call&& call) {
  const auto device = inputs.front().device();
  VendorScope scope(inputs.front());
  const auto stream = GetCurrentStreamForDevice(device.index());
  const auto queue = reinterpret_cast<cudaStream_t>(stream);
  auto* allocator = c10::flagos::GetCachingAllocator();
  bool ordered = allocator && c10::flagos::CachingDeviceAllocator::is_enabled();
  c10::SmallVector<at::Tensor, 4> views;
  for (const auto& input : inputs) {
    if (!input.defined() || input.is_cpu()) {
      views.push_back(input);
      continue;
    }
    TORCH_CHECK(input.device() == device && input.layout() == at::kStrided,
                "native NVIDIA operation requires strided inputs on one device");
    if (ordered) ordered = allocator->record_stream(input.storage().data_ptr(), stream);
    views.push_back(FlagosToCudaView(input));
  }
  const auto run = [&]() -> decltype(auto) {
    try {
      return call(at::TensorList(views));
    } catch (...) {
      if (!ordered) C10_CUDA_CHECK(cudaStreamSynchronize(queue));
      throw;
    }
  };
  decltype(auto) result = run();
  if (!ordered) C10_CUDA_CHECK(cudaStreamSynchronize(queue));
  return result;
}

inline at::Tensor& CopyCudaResult(at::Tensor& output, const at::Tensor& result) {
  if (result.numel()) FlagosToCudaView(output).copy_(result);
  return output;
}

}  // namespace at::native::flagos::vendor
