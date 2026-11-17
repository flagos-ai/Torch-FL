// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "fallback.h"

#include <ATen/native/CPUFallback.h>
#include <flagos_env.h>
#include <torch/library.h>
#include <atomic>

namespace at::native::flagos {

namespace {
std::atomic<bool> strict_host_routes{false};
}

TORCH_LIBRARY_FRAGMENT(flagos, m) {
  m.def("strict_host_routes(bool enabled) -> ()", [](bool enabled) {
    strict_host_routes.store(enabled, std::memory_order_relaxed);
  });
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(!strict_host_routes.load(std::memory_order_relaxed),
              "native FlagGems hosts do not permit CPU fallback: ", op.schema());
  static const bool log_enabled = LogEnabled("fallback");
  if (log_enabled) {
    const auto& schema = op.schema();
    const auto name = schema.overload_name().empty()
        ? schema.name()
        : schema.name() + "." + schema.overload_name();
    fprintf(stderr, "[flagos cpu_fallback] %s\n", name.c_str());
  }
  at::native::cpu_fallback(op, stack);
}

} // namespace at::native::flagos
