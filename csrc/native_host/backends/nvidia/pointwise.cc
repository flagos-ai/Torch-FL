// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "pointwise.h"

#include <ATen/ops/empty_like.h>
#include <c10/core/DeviceGuard.h>
#include <cuda_runtime_api.h>
#include <flagos.h>

namespace flagos::nvidia {

cudaError_t LaunchPointwise(PointwiseOp op, bool bf16, const void* input, void* output,
                            const int64_t size, float scalar, cudaStream_t stream);

at::Tensor TryPointwise(PointwiseOp op, const at::Tensor& input, float scalar) {
  const auto dtype = input.scalar_type();
  const bool bf16 = dtype == at::kBFloat16;
  if (!input.device().is_privateuseone() || input.layout() != at::kStrided ||
      input.is_neg() || input.has_names() || !input.is_non_overlapping_and_dense() ||
      (dtype != at::kFloat && !(bf16 && op == PointwiseOp::Silu))) {
    return {};
  }
  const c10::OptionalDeviceGuard guard(input.device());
  auto output = at::empty_like(input, {}, c10::MemoryFormat::Preserve);
  if (input.numel() == 0) return output;
  const auto queue = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(input.device().index()));
  const auto status = LaunchPointwise(op, bf16, input.const_data_ptr(), output.mutable_data_ptr(),
                                     input.numel(), scalar, queue);
  TORCH_CHECK(status == cudaSuccess, "NVIDIA SDK pointwise launch failed: ",
              cudaGetErrorString(status));
  return output;
}

}  // namespace flagos::nvidia
