// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>

namespace at::native::flagos {

namespace {

at::Tensor MakeNearestExactIndex(int64_t input_size,
                                 int64_t output_size,
                                 const ::std::optional<double>& scale,
                                 const at::Device& device) {
  TORCH_CHECK(input_size > 0 && output_size > 0,
              "_upsample_nearest_exact2d: spatial sizes must be positive");
  const double inverse_scale =
      scale.has_value() && scale.value() > 0.0
      ? 1.0 / scale.value()
      : static_cast<double>(input_size) / static_cast<double>(output_size);

  auto cpu_index = at::empty(
      {output_size}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  auto* index_data = cpu_index.data_ptr<int64_t>();
  for (int64_t i = 0; i < output_size; ++i) {
    index_data[i] = std::min(
        static_cast<int64_t>(std::floor((static_cast<double>(i) + 0.5) *
                                        inverse_scale)),
        input_size - 1);
  }
  return cpu_index.to(device);
}

} // namespace

at::Tensor PrivUpsampleNearestExact2dKernelAscend(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    ::std::optional<double> scales_h,
    ::std::optional<double> scales_w) {
  TORCH_CHECK(self.dim() == 4,
              "_upsample_nearest_exact2d: expected a 4D input, got ",
              self.dim(), "D");
  TORCH_CHECK(output_size.size() == 2,
              "_upsample_nearest_exact2d: expected two output dimensions");

  auto height_index = MakeNearestExactIndex(
      self.size(2), output_size[0], scales_h, self.device());
  auto width_index = MakeNearestExactIndex(
      self.size(3), output_size[1], scales_w, self.device());

  // Index selection reproduces PyTorch's nearest-exact source-coordinate
  // formula while keeping the input and output data entirely on the device.
  auto height_resized = at::index_select(self, 2, height_index);
  return at::index_select(height_resized, 3, width_index);
}

REGISTER_IMPL_TO_DISPATCHER(
    PrivUpsampleNearestExact2dFn,
    priv_upsample_nearest_exact2d_dispatcher,
    Backend::kAscend,
    PrivUpsampleNearestExact2dKernelAscend)

} // namespace at::native::flagos
