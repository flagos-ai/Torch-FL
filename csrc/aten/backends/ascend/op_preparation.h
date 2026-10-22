// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>

namespace at::native::flagos::ascend {

class OpPreparation {
 public:
  static at::Tensor apply_tensor_without_format(
      at::IntArrayRef sizes,
      const at::TensorOptions& options) {
    return at::empty(sizes, RetargetToPrivateUse1(options));
  }

  static at::Tensor apply_tensor_with_format(
      at::IntArrayRef sizes,
      const at::TensorOptions& options,
      int64_t format = 2 /* ACL_FORMAT_ND */) {
    return at::empty(sizes, RetargetToPrivateUse1(options));
  }

  static int8_t get_cube_math_type(bool allow_hf32 = false) {
    return allow_hf32 ? 1 : 0;
  }

  static void check_tensor(
      std::initializer_list<at::Tensor> inputs,
      at::Tensor& output,
      at::ScalarType dtype,
      at::IntArrayRef sizes) {
    if (output.sizes() != sizes) {
      output.resize_(sizes);
    }
  }

 private:
  // Force the device *type* to PrivateUse1 while keeping the caller's device
  // index. Dropping the index (the previous `options.device(DeviceType)`) built
  // a `c10::Device(PrivateUse1, -1)`, which `at::empty` resolves against the
  // ambient current device -- so an op whose inputs were all on flagos:0 still
  // returned its result on whatever device the caller last made current. A
  // CPU-sourced options carries index -1, which stays index-less here and keeps
  // the intended "ambient device" behavior on the fallback paths.
  static at::TensorOptions RetargetToPrivateUse1(
      const at::TensorOptions& options) {
    const auto device = options.device();
    return options.device(c10::Device(
        c10::DeviceType::PrivateUse1, device.has_index() ? device.index() : -1));
  }
};

} // namespace at::native::flagos::ascend
