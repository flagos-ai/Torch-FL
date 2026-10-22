// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <optional>

#include "dispatcher.h"

namespace at::native::flagos {

using LocalScalarDenseFn = at::Scalar (*)(const at::Tensor&);
DECLARE_DISPATCHER(LocalScalarDenseFn, local_scalar_dense_dispatcher)

using ToCopyFn = at::Tensor (*)(
    const at::Tensor&,
    std::optional<c10::ScalarType>,
    std::optional<c10::Layout>,
    std::optional<c10::Device>,
    std::optional<bool>,
    bool,
    std::optional<c10::MemoryFormat>);
DECLARE_DISPATCHER(ToCopyFn, to_copy_dispatcher)

}  // namespace at::native::flagos
