// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "fallback.h"

#include <ATen/native/CPUFallback.h>
#include <flagos_env.h>

namespace at::native::flagos {

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  static const bool log_enabled = LogEnabled("fallback");
  if (log_enabled) {
    fprintf(stderr, "[flagos cpu_fallback] %s\n",
            op.schema().name().c_str());
  }
  at::native::cpu_fallback(op, stack);
}

} // namespace at::native::flagos
