// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "common.h"
#include <c10/util/Exception.h>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

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
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagGemsCpp, my_flaggems_cpp_mm)
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kCuda,   my_cuda_mm)
//
//   // call (uses op_name "mm" for config lookup):
//   mm_dispatcher(self, mat2, out);
//
//   // call with op name override (uses "mm.out" for config lookup):
//   mm_dispatcher.DispatchAs("mm.out", self, mat2, out);

namespace detail {

// Does this argument alone force the call off the FlagGems route?
//
// Overloads cover only the argument types that carry a dtype -- a Tensor, an
// optional Tensor, the Tensor-list forms, and an explicit ScalarType. Anything
// else (Scalar, bool, IntArrayRef, Generator, ...) cannot answer the question
// on its own, so the catch-all reports "no". A Scalar is deliberately not
// included: it is not a tensor dtype, and the MUSA case where a Python-float
// operand decides the outcome is a per-op gap the conf generator records.
//
// Overload resolution rather than SFINAE: a Tensor argument matches the
// non-template overload exactly, which wins over the catch-all template.
inline bool FlagGemsRejectsArg(const at::Tensor& t) {
  return FlagGemsRejectsDtype(t.scalar_type());
}

inline bool FlagGemsRejectsArg(const ::std::optional<at::Tensor>& t) {
  return t.has_value() && FlagGemsRejectsDtype(t->scalar_type());
}

inline bool FlagGemsRejectsArg(const at::ArrayRef<at::Tensor>& l) {
  for (const auto& t : l) {
    if (FlagGemsRejectsDtype(t.scalar_type())) return true;
  }
  return false;
}

inline bool FlagGemsRejectsArg(const at::ITensorListRef& l) {
  for (const auto& t : l) {
    if (FlagGemsRejectsDtype(t.scalar_type())) return true;
  }
  return false;
}

// `c10::List<optional<Tensor>>` (the foreach_* schema shape) is deliberately
// absent: its ListElementReference has neither has_value() nor ->, so the
// overload would need an IValue unwrap for a case no Ascend op reaches.

inline bool FlagGemsRejectsArg(const ::std::optional<at::ScalarType>& d) {
  return d.has_value() && FlagGemsRejectsDtype(*d);
}

// Covers the factory ops, whose dtype arrives as a bare argument rather than
// through a tensor: torch.ones(4, dtype=torch.float64) has no operand to
// inspect.
inline bool FlagGemsRejectsArg(at::ScalarType d) {
  return FlagGemsRejectsDtype(d);
}

template <typename T>
inline bool FlagGemsRejectsArg(const T&) {
  return false;
}

template <typename... Args>
inline bool FlagGemsRejectsArgs(const Args&... args) {
  return (... || FlagGemsRejectsArg(args));
}

} // namespace detail

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
      case Backend::kCuda:          cuda_fn_ = fn;              break;
      case Backend::kFlagGemsCpp:   flaggems_cpp_fn_ = fn;     break;
      case Backend::kFlagGems:      flaggems_fn_ = fn;         break;
      case Backend::kAscend:        ascend_fn_ = fn;           break;
      case Backend::kMusa:          musa_fn_ = fn;             break;
      case Backend::kMetax:         metax_fn_ = fn;            break;
      case Backend::kTsingMicro:    tsingmicro_fn_ = fn;       break;
      case Backend::kGcu:           gcu_fn_ = fn;              break;
      case Backend::kTileOps:       tileops_fn_ = fn;          break;
      // Neither names a kernel set, so there is no slot to fill. kNone is the
      // conf's "no accelerated impl here" verdict; kUncached is the sentinel
      // operator() uses before it has resolved the op. Listed explicitly rather
      // than via default: so adding a real backend still trips -Wswitch here.
      case Backend::kNone:
      case Backend::kUncached:
        break;
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
    auto fn = ResolveFn(backend, args...);
    LogDispatch(op_name_, backend);

    // Strict mode: ALL_USE_FLAGGEMS / ALL_USE_VENDOR require impl to exist
    if (!fn) {
      static const bool strict_flaggems = []() {
        const char* v = std::getenv("ALL_USE_FLAGGEMS");
        return v && std::string(v) != "0" && std::string(v) != "";
      }();
      static const bool strict_vendor = []() {
        const char* v = std::getenv("ALL_USE_VENDOR");
        return v && std::string(v) != "0" && std::string(v) != "";
      }();

      if (strict_flaggems && (backend == Backend::kFlagGemsCpp || backend == Backend::kFlagGems)) {
        std::string msg = std::string(op_name_) +
                         ": ALL_USE_FLAGGEMS=1 but FlagGems impl not compiled (set FLAGGEMS_KERNEL=1 or FLAGGEMS_PYTHON=1)";
        throw std::runtime_error(msg);
      }
      if (strict_vendor && backend != Backend::kFlagGemsCpp && backend != Backend::kFlagGems &&
          backend != Backend::kNone && backend != Backend::kTileOps) {
        std::string msg = std::string(op_name_) +
                         ": ALL_USE_VENDOR=1 but vendor kernel not registered";
        throw std::runtime_error(msg);
      }
    }

    TORCH_CHECK(fn, op_name_, DispatchFailureMessage(backend));
    return fn(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) DispatchBackend(Backend backend, Args&&... args) const {
    auto fn = GetFn(backend);
    TORCH_CHECK(fn, op_name_, DispatchFailureMessage(backend));
    return fn(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) DispatchAs(const std::string& op_name, Args&&... args) const {
    auto backend = GetBackendForOp(op_name);
    auto fn = ResolveFn(backend, args...);
    LogDispatch(op_name, backend);

    // Strict mode: ALL_USE_FLAGGEMS / ALL_USE_VENDOR require impl to exist
    if (!fn) {
      static const bool strict_flaggems = []() {
        const char* v = std::getenv("ALL_USE_FLAGGEMS");
        return v && std::string(v) != "0" && std::string(v) != "";
      }();
      static const bool strict_vendor = []() {
        const char* v = std::getenv("ALL_USE_VENDOR");
        return v && std::string(v) != "0" && std::string(v) != "";
      }();

      if (strict_flaggems && (backend == Backend::kFlagGemsCpp || backend == Backend::kFlagGems)) {
        std::string msg = op_name +
                         ": ALL_USE_FLAGGEMS=1 but FlagGems impl not compiled (set FLAGGEMS_KERNEL=1 or FLAGGEMS_PYTHON=1)";
        throw std::runtime_error(msg);
      }
      if (strict_vendor && backend != Backend::kFlagGemsCpp && backend != Backend::kFlagGems &&
          backend != Backend::kNone && backend != Backend::kTileOps) {
        std::string msg = op_name +
                         ": ALL_USE_VENDOR=1 but vendor kernel not registered";
        throw std::runtime_error(msg);
      }
    }

    TORCH_CHECK(fn, op_name, DispatchFailureMessage(backend));
    return fn(std::forward<Args>(args)...);
  }

 private:
  // The build's own native kernel slot, with the backend name that selects it.
  // Exactly one is ever populated: a build registers its native kernels into
  // its own slot (kAscend on Ascend, kMusa on MUSA, ...) and, on the
  // CUDA-compatible platforms, boxing kernels into kCuda. Ordered
  // most-specific-first so a build that somehow populated two still prefers
  // the vendor's own kernels.
  std::pair<FnPtr, Backend> VendorSlot() const {
    if (ascend_fn_)      return {ascend_fn_,      Backend::kAscend};
    if (musa_fn_)        return {musa_fn_,        Backend::kMusa};
    if (metax_fn_)       return {metax_fn_,       Backend::kMetax};
    if (gcu_fn_)         return {gcu_fn_,         Backend::kGcu};
    if (tsingmicro_fn_)  return {tsingmicro_fn_,  Backend::kTsingMicro};
    if (cuda_fn_)        return {cuda_fn_,        Backend::kCuda};
    return {nullptr, Backend::kNone};
  }

  // GetFn, plus the dtype escape from the FlagGems route.
  //
  // A conf routes per op, so it cannot say "FlagGems, except for float64" --
  // but on Ascend FlagGems' pointwise codegen simply does not compile for
  // float64, and the vendor kernels do (see FlagGemsRejectsDtype). The
  // substitution has to happen where the arguments are visible, which is here.
  // `backend` is rewritten alongside, so the dispatch log and the
  // "backend not registered" message name what actually ran rather than what
  // the conf asked for.
  //
  // Only the Python route is guarded: the C++ one (kFlagGemsCpp) exists only
  // on the vendors that measured it, so its conf entries are already a per-op
  // decision made from a measurement.
  template <typename... Args>
  FnPtr ResolveFn(Backend& backend, const Args&... args) const {
    FnPtr fn = GetFn(backend);
    if (backend != Backend::kFlagGems || fn == nullptr) return fn;
    if (!detail::FlagGemsRejectsArgs(args...)) return fn;

    auto [vendor_fn, vendor_backend] = VendorSlot();
    if (vendor_fn == nullptr) {
      // FlagGems is the only implementation this build has for the op, so
      // there is nothing to fall back to. Let the call fail where it would
      // have failed anyway, rather than reporting a routing problem.
      return fn;
    }
    backend = vendor_backend;
    return vendor_fn;
  }

  FnPtr GetFn(Backend device) const {
    switch (device) {
      case Backend::kCuda:          return cuda_fn_;
      // FlagGems C++ runtime is only compiled in for a FLAGGEMS_KERNEL=ON
      // build (flaggems_cpp_kernels.cc, behind FLAGOS_FLAGGEMS_CPP), which needs
      // liboperators.so built for the vendor. A platform ships ONE conf, so the
      // conf cannot know whether that opt-in build is the one running: MetaX's
      // conf routes ops here because a MACA-built FlagGems is worth using when
      // present, and the same file has to stay correct when it is absent.
      // Degrade to the boxing kernel rather than hard-failing on the empty slot.
      // Boxing is preferred over the Python FlagGems path because these ops
      // reach the C++ set precisely where the platform's triton backend is the
      // weak link (metax rejects the SPLIT_K kwarg gems' bmm passes), so the
      // Python path is not a safe substitute. Mirrors the kTileOps case below.
      case Backend::kFlagGemsCpp:
        if (flaggems_cpp_fn_) return flaggems_cpp_fn_;
        return cuda_fn_ ? cuda_fn_ : flaggems_fn_;
      // FlagGems Python path is only compiled in for a FLAGGEMS_PYTHON=ON build
      // (flaggems_python_kernels.cc, behind FLAGOS_FLAGGEMS_PYTHON). When uncompiled,
      // degrade to the boxing kernel rather than raising "backend not registered".
      // This keeps one conf correct for both builds (Python FlagGems ON/OFF).
      case Backend::kFlagGems:
        if (flaggems_fn_) return flaggems_fn_;
        return cuda_fn_ ? cuda_fn_ : flaggems_cpp_fn_;
      case Backend::kAscend:        return ascend_fn_;
      case Backend::kMusa:          return musa_fn_;
      case Backend::kMetax:         return metax_fn_;
      case Backend::kTsingMicro:    return tsingmicro_fn_;
      case Backend::kGcu:           return gcu_fn_;
      // TileOPs kernels live in Python and are bound via torch.library on
      // PrivateUse1, which intercepts before this dispatcher is reached. A conf
      // entry of "tileops" therefore only lands here when that registration did
      // not happen (TileOPs missing, non-SM90 host, or the op was filtered out),
      // so fall back instead of hard-failing on an empty slot.
      case Backend::kTileOps:
        if (tileops_fn_) return tileops_fn_;
        return cuda_fn_ ? cuda_fn_ : flaggems_cpp_fn_;
      // "none" means the platform has no accelerated impl for this op. Codegen
      // is expected to skip m.impl() for it so the call reaches cpu_fallback
      // and never arrives here. Landing here means the registration and the
      // conf disagree, so return nullptr and let the caller raise -- with a
      // message that names the real cause (see DispatchFailureMessage).
      case Backend::kNone:
        return nullptr;
      case Backend::kUncached:
        return nullptr;
    }
    return nullptr;
  }

  // Distinguishes "conf says none but the op was registered anyway" from a
  // genuinely missing kernel. The first is a codegen/conf mismatch and the
  // operator-support docs are the place to fix it; the second is a build gap.
  static std::string DispatchFailureMessage(Backend backend) {
    if (backend == Backend::kNone) {
      return ": routed to 'none' (no accelerated impl on this platform) but the "
             "op is registered on PrivateUse1 -- regenerate the vendor conf so "
             "registration and routing agree";
    }
    return ": backend not registered";
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
      case Backend::kFlagGemsCpp:   name = "flagos"; break;  // keep old name for test compat
      case Backend::kFlagGems:      name = "flagos_python"; break;  // keep old name for test compat
      case Backend::kAscend:        name = "ascend"; break;
      case Backend::kMusa:          name = "musa"; break;
      case Backend::kMetax:         name = "metax"; break;
      case Backend::kTsingMicro:    name = "tsingmicro"; break;
      case Backend::kGcu:           name = "gcu"; break;
      case Backend::kTileOps:       name = "tileops"; break;
      case Backend::kNone:          name = "none"; break;
      default:                      name = "unknown"; break;
    }
    fprintf(stderr, "[flagos dispatch] %s -> %s\n", op_name.c_str(), name);
  }

  const char* op_name_ = nullptr;
  // Per-op backend cache for the hot operator() path (see comment there).
  // mutable: operator() is const but memoizes on first call. Benign data race
  // under concurrent first-use — all threads compute the same immutable value.
  mutable Backend cached_backend_ = Backend::kUncached;
  FnPtr cuda_fn_           = nullptr;
  FnPtr flaggems_cpp_fn_   = nullptr;
  FnPtr flaggems_fn_       = nullptr;
  FnPtr ascend_fn_         = nullptr;
  FnPtr musa_fn_           = nullptr;
  FnPtr metax_fn_          = nullptr;
  FnPtr tsingmicro_fn_     = nullptr;
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
