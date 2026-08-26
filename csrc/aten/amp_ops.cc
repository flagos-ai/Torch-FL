// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "amp_ops.h"

#include "copy_ops.h"
#include <ATen/ops/isfinite.h>

namespace at::native::flagos {
namespace {

at::Tensor to_cpu_sync(const at::Tensor& tensor) {
  return tensor.to(at::kCPU);
}

void copy_cpu_to_device(const at::Tensor& source, at::Tensor& target) {
  at::native::flagos::_copy_from(source, target, false);
}

}  // namespace

void amp_foreach_non_finite_check_and_unscale(
    at::TensorList self,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  TORCH_CHECK(
      found_inf.is_privateuseone() && inv_scale.is_privateuseone(),
      "flagos AMP non-finite check expects device tensors for found_inf and "
      "inv_scale");

  auto found_inf_cpu = to_cpu_sync(found_inf);
  auto inv_scale_cpu = to_cpu_sync(inv_scale);
  bool any_non_finite = false;

  for (const auto& tensor : self) {
    TORCH_CHECK(
        tensor.is_privateuseone(),
        "flagos AMP non-finite check expects device gradients");
    auto tensor_cpu = to_cpu_sync(tensor);
    auto finite = at::isfinite(tensor_cpu);
    any_non_finite = any_non_finite || !finite.all().item<bool>();
    tensor_cpu.mul_(inv_scale_cpu);
    auto mutable_tensor = const_cast<at::Tensor&>(tensor);
    copy_cpu_to_device(tensor_cpu, mutable_tensor);
  }

  if (any_non_finite) {
    found_inf_cpu.fill_(1);
  } else {
    found_inf_cpu.zero_();
  }
  copy_cpu_to_device(found_inf_cpu, found_inf);
}

void amp_foreach_non_finite_check_and_unscale_out(
    at::TensorList self,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale,
    at::TensorList out) {
  TORCH_CHECK(
      self.size() == out.size(),
      "flagos AMP non-finite check output list must match input list size");
  for (size_t i = 0; i < self.size(); ++i) {
    auto& output = const_cast<at::Tensor&>(out[i]);
    output.copy_(self[i]);
  }
  amp_foreach_non_finite_check_and_unscale(out, found_inf, inv_scale);
}

}  // namespace at::native::flagos
