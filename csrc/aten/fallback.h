// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include "common.h"

namespace at::native::flagos {

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

} // namespace at::native::flagos
