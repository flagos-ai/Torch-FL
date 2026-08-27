// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include "common.h"

namespace at::native::flagos {

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset);

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<at::MemoryFormat> memory_format);

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride);

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size);

at::Tensor expand(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit);

at::Tensor narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length);
at::Tensor alias(const at::Tensor& self);

at::Tensor transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1);

at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims);

at::Tensor select_int(const at::Tensor& self, int64_t dim, int64_t index);

at::Tensor slice_tensor(const at::Tensor& self, int64_t dim, ::std::optional<int64_t> start, ::std::optional<int64_t> end, int64_t step);

at::Tensor squeeze(const at::Tensor& self);

at::Tensor squeeze_dim(const at::Tensor& self, int64_t dim);

at::Tensor unsqueeze(const at::Tensor& self, int64_t dim);

at::Tensor unsafe_view(const at::Tensor& self, at::IntArrayRef size);

at::Tensor t(const at::Tensor& self);

::std::vector<at::Tensor> unbind_int(const at::Tensor& self, int64_t dim);

at::Tensor unfold(const at::Tensor& self, int64_t dimension, int64_t size, int64_t step);

at::Tensor _conj(const at::Tensor& self);

at::Tensor _neg_view(const at::Tensor& self);

} // namespace at::native::flagos
