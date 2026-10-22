// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include "common.h"

namespace at::native::flagos {

at::Tensor contiguous(
    const at::Tensor& self,
    c10::MemoryFormat memory_format);

at::Tensor materialize_math_bits(
    const at::Tensor& self,
    c10::MemoryFormat memory_format = c10::MemoryFormat::Preserve);

at::Tensor clone(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format);

} // namespace at::native::flagos
