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
  kFlagGemsCpp,    // FlagGems C++ runtime (liboperators.so, conf key: flaggems_cpp)
  kFlagGems,       // FlagGems Python/Triton path (conf key: flaggems)
  kAscend,
  kMusa,
  kMetax,
  kTsingMicro,
  kGcu,
  kTileOps,
  // No accelerated implementation on this platform. The op is deliberately not
  // registered on PrivateUse1, so it reaches the boxed cpu_fallback instead of
  // arriving here with an empty kernel slot. Recorded explicitly (rather than
  // by omission) so a vendor conf states coverage for every op and support can
  // be counted from the file.
  kNone,
  kUncached
};

// Returns the backend for a given op name, loaded once from config file at startup.
// Config file path: $FLAGOS_BACKEND_CONFIG or torch_fl/configs/backends.conf
// Format: "op_name = backend"
//   backend: "flaggems_cpp" -- FlagGems C++ path (liboperators.so)
//            "flaggems"     -- FlagGems Python (Triton) path
//            "tileops"      -- TileOps Triton shims
//            "<vendor>"     -- vendor-native kernel (cuda | ascend | musa |
//                              metax | gcu | tsingmicro)
//            "none"         -- no accelerated impl; reaches cpu_fallback
// Default when op is not listed: kFlagGems.
Backend GetBackendForOp(const std::string& op_name);

// Dtypes this build's FlagGems (Triton) route cannot serve, whatever op is
// asking. Consulted by Dispatcher so a `flaggems` route falls back to the
// vendor kernel for those dtypes instead of failing inside the compiler.
//
// A vendor conf is a per-op routing table, so it cannot express "FlagGems,
// except for dtype X" -- and on Ascend that exception is real and broad.
// BiShengHIR rejects the float64 instantiation of nearly every kernel
// FlagGems' pointwise codegen produces: on Ascend910 with CANN 9.0.0 and
// FlagTree 0.6.2a1+ascend3.5, add/sub/div/neg/abs/exp/log/sqrt/reciprocal/
// where/clamp/fill_/zeros_like/ones_like/ones/full/arange over float64
// all raise MLIRCompilationError from triton's spec/ascend compiler
// ("'hivm.hir.vbrc' op failed to verify that operand at idx 0 should have
// element type ...", or "ub overflow" on the larger shapes), while mul, cat
// and the comparison ops -- which lower through a different path -- do work.
// The same calls reach aclnn kernels through the vendor slot and return the
// right float64 answer, so the fallback is a gain rather than a loss.
//
// Deliberately a predicate on the dtype alone and not on (op, dtype): the
// per-op form is exactly the NATIVE_TRITON_GAPS entry the conf generator
// already has, and it would have to name every pointwise op in the file.
bool FlagGemsRejectsDtype(at::ScalarType dtype);

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
