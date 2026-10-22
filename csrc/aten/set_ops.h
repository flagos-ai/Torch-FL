// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include "common.h"

namespace at::native::flagos {

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source);

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source);

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride);

} // namespace at::native::flagos
