// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/aten/native/Common.h
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>

#include <flagos.h>

#include <string>

namespace at::native::flagos {

// Backend selector for unified op wrappers.
// Determines which physical backend impl() dispatches to.
// kUncached is a sentinel used by Dispatcher's per-op backend cache; it is
// never stored in the BackendTable and never returned by GetBackendForOp.
// Keep it last so the real backends stay contiguous.
enum class Backend {
  kCuda,
  kFlagOs,
  kFlagOsPython,
  kAscend,
  kMusa,
  kMetax,
  kTsingMicro,
  kGcu,
  kTileOps,
  kUncached
};

// Returns the backend for a given op name, loaded once from config file at startup.
// Config file path: $FLAGOS_BACKEND_CONFIG or torch_fl/configs/backends.conf
// Format: "op_name = backend"  (backend: "flagos" | "flaggems" | "cuda" | "metax" | "tileops")
// Default when op is not listed: FlagOS.
Backend GetBackendForOp(const std::string& op_name);

// Memory guard to ensure proper synchronization when accessing device memory
class MemoryGuard {
 public:
  template <typename... Tensors>
  explicit MemoryGuard(const Tensors&... tensors) {
    (acquire(tensors), ...);
  }

  ~MemoryGuard() {
    for (void* ptr : acquired_ptrs_) {
      // No explicit release needed for CUDA-backed memory
    }
  }

 private:
  void acquire(const at::Tensor& tensor) {
    if (tensor.defined() && tensor.is_privateuseone()) {
      void* ptr = tensor.data_ptr();
      if (ptr) {
        acquired_ptrs_.push_back(ptr);
      }
    }
  }

  std::vector<void*> acquired_ptrs_;
};

} // namespace at::native::flagos
