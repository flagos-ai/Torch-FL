// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include "common.h"

namespace at::native::flagos {

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking);

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst);

// A `Memcpy` that first binds the device the transfer touches and drains that
// device's queue. A bare Memcpy is blocking only against whichever device is
// current when it is issued, so a copy of a tensor living elsewhere can read
// that buffer's pre-kernel contents (issue #281, MUSA).
//
// `on_device` is the accelerator-side operand the copy touches -- the source for
// a D2H or D2D copy, the destination for an H2D one -- and must outlive the
// copy itself.
void BlockingMemcpy(
    void* dst,
    const void* src,
    size_t nbytes,
    MemcpyKind kind,
    const at::Tensor& on_device);

at::Scalar _local_scalar_dense(const at::Tensor& self);

at::Tensor _to_copy(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format);

} // namespace at::native::flagos
