// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "common.h"
#include <c10/util/Exception.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace at::native::flagos {

// Lightweight op dispatcher replacing PyTorch's DispatchStub.
//
// A dispatcher owns an op_name (e.g. "mm") and a set of per-backend kernel
// pointers. Multiple ops can share one dispatcher (e.g. "mm" and "mm.out"
// share the same kernels). The op name used for config lookup and
// logging is passed at the call site via DispatchAs(), or defaults
// to the dispatcher's op_name via operator().
//
// Usage:
//   // header:
//   using MmFn = void(*)(const Tensor&, const Tensor&, Tensor&);
//   DECLARE_DISPATCHER(MmFn, mm_dispatcher)
//
//   // cpp:
//   ADD_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, "mm")
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagOs, my_flagos_mm)
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kCuda,   my_cuda_mm)
//
//   // call (uses op_name "mm" for config lookup):
//   mm_dispatcher(self, mat2, out);
//
//   // call with op name override (uses "mm.out" for config lookup):
//   mm_dispatcher.DispatchAs("mm.out", self, mat2, out);

template <typename FnPtr>
class Dispatcher {
 public:
  // constexpr constructor ensures constant initialization (placed in .bss/.data
  // at load time), which is guaranteed to complete before any dynamic
  // initialization (DispatchRegistrar constructors). This eliminates the
  // static initialization order fiasco when DEFINE and REGISTER are in
  // different translation units.
  constexpr explicit Dispatcher(const char* op_name)
      : op_name_(op_name) {}

  void RegisterKernel(Backend device, FnPtr fn) {
    switch (device) {
      case Backend::kCuda:          cuda_fn_ = fn;           break;
      case Backend::kFlagOs:        flagos_fn_ = fn;         break;
      case Backend::kFlagOsPython:  flagos_python_fn_ = fn;  break;
      case Backend::kAscend:        ascend_fn_ = fn;         break;
      case Backend::kMusa:          musa_fn_ = fn;           break;
      case Backend::kMetax:         metax_fn_ = fn;          break;
      case Backend::kTsingMicro:   tsingmicro_fn_ = fn;    break;
      case Backend::kGcu:           gcu_fn_ = fn;            break;
      case Backend::kTileOps:       tileops_fn_ = fn;        break;
    }
  }

  template <typename... Args>
  decltype(auto) operator()(Args&&... args) const {
    // Hot path: the op name is fixed (op_name_) and the backend routing is
    // immutable once the config is loaded, so resolve it once and cache. This
    // avoids constructing a std::string from op_name_ and hashing it in the
    // BackendTable on EVERY op call — measured as a significant per-op cost in
    // the Ascend eager decode loop (thousands of ops/token).
    Backend backend = cached_backend_;
    if (__builtin_expect(backend == Backend::kUncached, 0)) {
      backend = GetBackendForOp(op_name_);
      cached_backend_ = backend;
    }
    LogDispatch(op_name_, backend);
    auto fn = GetFn(backend);
    TORCH_CHECK(fn, op_name_, ": backend not registered");
    return fn(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) DispatchAs(const std::string& op_name, Args&&... args) const {
    auto backend = GetBackendForOp(op_name);
    LogDispatch(op_name, backend);
    auto fn = GetFn(backend);
    TORCH_CHECK(fn, op_name, ": backend not registered");
    return fn(std::forward<Args>(args)...);
  }

 private:
  FnPtr GetFn(Backend device) const {
    switch (device) {
      case Backend::kCuda:          return cuda_fn_;
      case Backend::kFlagOs:        return flagos_fn_;
      case Backend::kFlagOsPython:  return flagos_python_fn_;
      case Backend::kAscend:        return ascend_fn_;
      case Backend::kMusa:          return musa_fn_;
      case Backend::kMetax:         return metax_fn_;
      case Backend::kTsingMicro:   return tsingmicro_fn_;
      case Backend::kGcu:           return gcu_fn_;
      // TileOPs kernels live in Python and are bound via torch.library on
      // PrivateUse1, which intercepts before this dispatcher is reached. A conf
      // entry of "tileops" therefore only lands here when that registration did
      // not happen (TileOPs missing, non-SM90 host, or the op was filtered out),
      // so fall back instead of hard-failing on an empty slot.
      case Backend::kTileOps:
        if (tileops_fn_) return tileops_fn_;
        return cuda_fn_ ? cuda_fn_ : flagos_fn_;
    }
    return nullptr;
  }

  static void LogDispatch(const std::string& op_name, Backend backend) {
    static const bool enabled = []() {
      const char* v = std::getenv("FLAGOS_LOG_DISPATCH");
      return v && std::string(v) == "1";
    }();
    if (!enabled) return;
    const char* name;
    switch (backend) {
      case Backend::kCuda:          name = "cuda"; break;
      case Backend::kFlagOs:        name = "flagos"; break;
      case Backend::kFlagOsPython:  name = "flagos_python"; break;
      case Backend::kAscend:        name = "ascend"; break;
      case Backend::kMusa:          name = "musa"; break;
      case Backend::kMetax:         name = "metax"; break;
      case Backend::kTsingMicro:   name = "tsingmicro"; break;
      case Backend::kGcu:           name = "gcu"; break;
      case Backend::kTileOps:       name = "tileops"; break;
      default:                           name = "unknown"; break;
    }
    fprintf(stderr, "[flagos dispatch] %s -> %s\n", op_name.c_str(), name);
  }

  const char* op_name_ = nullptr;
  // Per-op backend cache for the hot operator() path (see comment there).
  // mutable: operator() is const but memoizes on first call. Benign data race
  // under concurrent first-use — all threads compute the same immutable value.
  mutable Backend cached_backend_ = Backend::kUncached;
  FnPtr cuda_fn_           = nullptr;
  FnPtr flagos_fn_         = nullptr;
  FnPtr flagos_python_fn_  = nullptr;
  FnPtr ascend_fn_         = nullptr;
  FnPtr musa_fn_           = nullptr;
  FnPtr metax_fn_          = nullptr;
  FnPtr tsingmicro_fn_    = nullptr;
  FnPtr gcu_fn_            = nullptr;
  FnPtr tileops_fn_        = nullptr;
};

namespace detail {
template <typename FnPtr>
struct DispatchRegistrar {
  DispatchRegistrar(Dispatcher<FnPtr>& dispatcher, Backend device, FnPtr fn) {
    dispatcher.RegisterKernel(device, fn);
  }
};
} // namespace detail

} // namespace at::native::flagos

#define DECLARE_DISPATCHER(fn_type, name) \
  extern ::at::native::flagos::Dispatcher<fn_type> name;

#define ADD_IMPL_TO_DISPATCHER(fn_type, name, op_name) \
  ::at::native::flagos::Dispatcher<fn_type> name(op_name);

#define REGISTER_IMPL_TO_DISPATCHER_UID2(fn_type, name, device, fn, uid) \
  __attribute__((used)) static ::at::native::flagos::detail::DispatchRegistrar<fn_type>    \
      name##_registrar_##uid(name, device, fn);
#define REGISTER_IMPL_TO_DISPATCHER_UID(fn_type, name, device, fn, uid) \
  REGISTER_IMPL_TO_DISPATCHER_UID2(fn_type, name, device, fn, uid)
#define REGISTER_IMPL_TO_DISPATCHER(fn_type, name, device, fn) \
  REGISTER_IMPL_TO_DISPATCHER_UID(fn_type, name, device, fn, __COUNTER__)
