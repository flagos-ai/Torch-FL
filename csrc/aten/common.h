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
  // Hygon DCU operators implemented directly against the DTK SDK (rocBLAS &c)
  // by libdcu_aten_ops.so, which torch_fl loads at import when
  // FLAGOS_DCU_SDK_OPS=1. Unlike kCuda -- which on DCU boxes PrivateUse1 to the
  // CUDA key and lands in DTK's forked libtorch_hip.so -- this slot needs no
  // vendor torch build at all: the plugin links only the official libc10 /
  // libtorch_cpu plus the SDK. Its kernels are installed at runtime through the
  // registration ABI in backends/dcu_sdk/dcu_sdk_abi.h, so the slot is empty
  // until that plugin is loaded (see GetFn()'s kDcuSdk diagnostics).
  kDcuSdk,
  kUncached
};

// Returns the backend for a given op name, loaded once from config file at startup.
// Config file path: $FLAGOS_BACKEND_CONFIG or torch_fl/configs/backends.conf
// Format: "op_name = backend"  (backend: "flagos" | "flaggems" | "cuda" |
// "metax" | "tileops" | "dcu_sdk")
// Default when op is not listed: FlagOS.
Backend GetBackendForOp(const std::string& op_name);

// The config/logging spelling of a backend, e.g. Backend::kDcuSdk -> "dcu_sdk".
const char* BackendName(Backend backend);

// Raises when the routing config selects a backend whose kernel slot is empty.
// Out-of-line on purpose: this is the cold path of every Dispatcher<> template
// instantiation, and the message text should exist once rather than per op.
[[noreturn]] void ReportMissingKernel(
    const std::string& op_name,
    Backend backend);

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
