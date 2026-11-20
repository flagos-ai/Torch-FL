// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "common.h"
#include <c10/util/Exception.h>
#include <atomic>
#include <cstdio>
#include <flagos_env.h>
#include <optional>
#include <stdexcept>
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
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kBoxing,   my_cuda_mm)
//
//   // call (uses op_name "mm" for config lookup):
//   mm_dispatcher(self, mat2, out);
//
//   // call with op name override (uses "mm.out" for config lookup):
//   mm_dispatcher.DispatchAs("mm.out", self, mat2, out);

namespace detail {

inline bool ConfiguredDtypeMatchesArg(uint64_t mask, const at::Tensor& value) {
  if (!value.defined()) return false;
  const auto dtype = static_cast<unsigned>(value.scalar_type());
  return dtype < 64 && (mask & (uint64_t{1} << dtype));
}

inline bool ConfiguredDtypeMatchesArg(uint64_t mask, const std::optional<at::Tensor>& value) {
  return value.has_value() && ConfiguredDtypeMatchesArg(mask, *value);
}

inline bool ConfiguredDtypeMatchesArg(uint64_t mask, const at::TensorList& values) {
  for (const auto& value : values) if (ConfiguredDtypeMatchesArg(mask, value)) return true;
  return false;
}

inline bool ConfiguredDtypeMatchesArg(uint64_t mask, const at::ITensorListRef& values) {
  for (const auto& value : values) if (ConfiguredDtypeMatchesArg(mask, value)) return true;
  return false;
}

template <typename T>
inline bool ConfiguredDtypeMatchesArg(uint64_t, const T&) {
  // ScalarType arguments can describe an output (mean.dim's dtype), so input
  // dtype clauses inspect tensors only, including optional/list operands.
  return false;
}

template <typename... Args>
inline bool ConfiguredDtypeMatchesArgs(uint64_t mask, const Args&... args) {
  return (... || ConfiguredDtypeMatchesArg(mask, args));
}

// Does this argument, for this op, force the call off the FlagGems route?
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
//
// The op name is threaded through because a gap can be the intersection of one
// op and one dtype, which neither the conf nor the dtype-wide predicate can
// state (see FlagGemsRejectsOpDtype). Every overload forwards it unchanged, so
// a new dtype-carrying argument type is one overload, not a second parameter
// list.
inline bool FlagGemsRejectsArg(const char* op_name, const at::Tensor& t) {
  return FlagGemsRejectsOpDtype(op_name, t.scalar_type());
}

inline bool FlagGemsRejectsArg(const char* op_name,
                               const ::std::optional<at::Tensor>& t) {
  return t.has_value() && FlagGemsRejectsOpDtype(op_name, t->scalar_type());
}

inline bool FlagGemsRejectsArg(const char* op_name,
                               const at::ArrayRef<at::Tensor>& l) {
  for (const auto& t : l) {
    if (FlagGemsRejectsOpDtype(op_name, t.scalar_type())) return true;
  }
  return false;
}

inline bool FlagGemsRejectsArg(const char* op_name,
                               const at::ITensorListRef& l) {
  for (const auto& t : l) {
    if (FlagGemsRejectsOpDtype(op_name, t.scalar_type())) return true;
  }
  return false;
}

// `c10::List<optional<Tensor>>` (the foreach_* schema shape) is deliberately
// absent: its ListElementReference has neither has_value() nor ->, so the
// overload would need an IValue unwrap for a case no Ascend op reaches.

inline bool FlagGemsRejectsArg(const char* op_name,
                               const ::std::optional<at::ScalarType>& d) {
  return d.has_value() && FlagGemsRejectsOpDtype(op_name, *d);
}

// Covers the factory ops, whose dtype arrives as a bare argument rather than
// through a tensor: torch.ones(4, dtype=torch.float64) has no operand to
// inspect.
inline bool FlagGemsRejectsArg(const char* op_name, at::ScalarType d) {
  return FlagGemsRejectsOpDtype(op_name, d);
}

template <typename T>
inline bool FlagGemsRejectsArg(const char* op_name, const T&) {
  (void)op_name;
  return false;
}

template <typename... Args>
inline bool FlagGemsRejectsArgs(const char* op_name, const Args&... args) {
  return (... || FlagGemsRejectsArg(op_name, args));
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
      case Backend::kBoxing:          cuda_fn_ = fn;              break;
      case Backend::kFlagGemsCpp:   flaggems_cpp_fn_ = fn;     break;
      case Backend::kFlagGems:      flaggems_fn_ = fn;         break;
      case Backend::kAscend:        ascend_fn_ = fn;           break;
      case Backend::kMthreads:      mthreads_fn_ = fn;        break;
      case Backend::kMetax:
        metax_fn_.store(fn, std::memory_order_release);
        break;
      case Backend::kTsingMicro:    tsingmicro_fn_ = fn;       break;
      case Backend::kEnflame:       enflame_fn_ = fn;         break;
      case Backend::kKunlunxin:     kunlunxin_fn_ = fn;        break;
      case Backend::kIluvatar:      iluvatar_fn_ = fn;         break;
      case Backend::kCambricon:     cambricon_fn_ = fn;        break;
      case Backend::kTileOps:       tileops_fn_ = fn;          break;
      case Backend::kApple:         apple_fn_ = fn;           break;
      case Backend::kHygon:
        hygon_fn_.store(fn, std::memory_order_release);
        break;
      case Backend::kNvidia:
        nvidia_fn_.store(fn, std::memory_order_release);
        break;
      // Neither names a kernel set, so there is no slot to fill. kNone is the
      // conf's "no accelerated impl here" verdict; kUncached is the sentinel
      // operator() uses before it has resolved the op. Listed explicitly rather
      // than via default: so adding a real backend still trips -Wswitch here.
      case Backend::kNone:
      case Backend::kUncached:
        break;
    }
  }

  // The kernel `device` resolves to, including GetFn's own degradation, or
  // nullptr. Lets a kernel set claim an empty slot without displacing the set
  // the conf's key actually names (soft_lowp/dispatch.cc).
  FnPtr KernelFor(Backend device) const {
    return GetFn(device);
  }

  bool HasKernel(Backend device) const {
    return KernelFor(device) != nullptr;
  }

  template <typename... Args>
  decltype(auto) operator()(Args&&... args) const {
    // Hot path: the op name is fixed (op_name_) and the backend routing is
    // immutable once the config is loaded, so resolve it once and cache. This
    // avoids constructing a std::string from op_name_ and hashing it in the
    // BackendTable on EVERY op call — measured as a significant per-op cost in
    // the Ascend eager decode loop (thousands of ops/token).
    const auto* configured = cached_config_.load(std::memory_order_acquire);
    if (__builtin_expect(configured == nullptr, 0)) {
      configured = &GetBackendConfigForOp(op_name_);
      cached_config_.store(configured, std::memory_order_release);
    }
    Backend backend = configured->backend;
    auto fn = ResolveFn(op_name_, *configured, backend, args...);
    LogDispatch(op_name_, backend);

    if (!fn) ThrowIfForcedBackendMissing(op_name_, backend);

    TORCH_CHECK(fn, op_name_, DispatchFailureMessage(backend));
    return fn(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) DispatchBackend(Backend backend, Args&&... args) const {
    auto fn = GetFn(backend);
    TORCH_CHECK(fn, op_name_, DispatchFailureMessage(backend));
    return fn(std::forward<Args>(args)...);
  }

  // Route exactly the way operator() would, without the generated wrapper's
  // dtype prelude in front of it. The software low-precision matrix path needs
  // this for the variants that cannot re-enter: its wrapper takes every
  // FLAGOS_SOFT_LOWP build's floating-point GEMMs as well as the FP8/FP4 ones
  // (the generated .inc cannot tell whether this build compiles the software
  // path), and the software path has to send the ordinary ones to whatever the
  // conf names -- the vendor slot, boxing -- rather than to one hard-coded
  // slot.
  //
  // A decoded FP8 pair asks first (RouteAcceptsDecoded) and then goes through
  // DecodedKernel: the two names can disagree, and the route a decoded pair
  // takes is not reachable with an FP8 call's operands.
  template <typename... Args>
  decltype(auto) DispatchConfigured(Args&&... args) const {
    return operator()(std::forward<Args>(args)...);
  }

  // The backend the conf names for `route_key`, and what it resolves to.
  // `KernelFor` is the accessor soft_lowp/ops.cc needs to ask whether the
  // answer is that file's own fallback wrapper; `ConfiguredBackend` is
  // kFlagGems when the conf does not hold the key.
  Backend ConfiguredBackend(const char* route_key) const {
    return GetBackendForOp(route_key);
  }

  // Can a decoded ordinary-dtype pair be handed to this route?
  //
  // Three noes. The dispatcher's own routing key must have a kernel at all --
  // `none` means the platform has no accelerated impl for the op, and the
  // generated registration was expected to stay away from it. The route must
  // leave FlagGems out of it: that route runs Python, which resolves the
  // operands' dtype against torch's own dtype objects on the way in, and the
  // spine of this path is that the pair is decoded to an ordinary one instead
  // of being spelled in a low-precision dtype. The other arms do not; and the
  // kernel the route lands on must exist. Where it does not -- an empty
  // flaggems_cpp slot on a build without liboperators -- GetFn degrades to
  // whatever else is registered, and that is not the route the conf named.
  // And the name must be the conf's: an op the table does not hold answers
  // kFlagGems, which the second test already refuses.
  bool RouteAcceptsDecoded(const char* route_key) const {
    if (!HasBackendForOp(route_key)) {
      return false;
    }
    const auto configured = GetBackendForOp(route_key);
    if (configured == Backend::kNone || configured == Backend::kFlagGems) {
      return false;
    }
    return GetFn(configured) != nullptr;
  }

  // The kernel a decoded ordinary-dtype pair goes to: the conf's route for
  // `route_key`, or the slot FLAGOS_SOFT_LOWP_STAY_ON_ROUTE names, or a
  // runtime_error saying why neither. The only place that raises
  // runtime_error rather than TORCH_CHECK; the software path's callers expect
  // that.
  //
  // DispatchAs resolves the key but its kernel comes out of GetFn, which
  // degrades an empty slot to whatever else is registered: a conf naming
  // flaggems for a key whose C++ slot is empty -- Medusa GPU, no liboperators --
  // lands on this file's own fallback wrappers, which answers with the pair
  // again. That degradation is the wrong answer here, so the fall-through is
  // opt-in by naming a slot: FLAGOS_SOFT_LOWP_STAY_ON_ROUTE=boxing (or
  // flaggems_cpp) hands a pair the named route cannot run to that slot instead
  // of raising. A slot is not a conf key -- no conf names `boxing` -- so the
  // flag is read here, where the difference is known.
  //
  // The kernel is returned rather than called here: the call belongs at the
  // call site, made on the kernel's own type (soft_lowp/ops.cc).
  FnPtr DecodedKernel(const std::string& route_key) const {
    auto backend = GetBackendForOp(route_key);
    if (!HasBackendForOp(route_key) || backend == Backend::kNone ||
        backend == Backend::kFlagGems) {
      throw std::runtime_error(DecodedRefusalFor(route_key, backend));
    }
    auto fn = GetFn(backend);
    if (!fn) {
      const std::string stay =
          flagos_env::EnvValue("FLAGOS_SOFT_LOWP_STAY_ON_ROUTE");
      if (stay.empty()) {
        throw std::runtime_error(DecodedRefusalFor(route_key, backend));
      }
      fn = DecodedFallbackFn(stay, backend);
      if (!fn) throw std::runtime_error(DecodedRefusalFor(route_key, backend));
    }
    LogDispatch(route_key, backend);
    return fn;
  }


  // The key this dispatcher routes by -- op_name_ -- for a caller that has to
  // name it before it can call: DecodedKernel takes the key, and the kernel it
  // returns then takes it as its first argument.
  const char* OpKey() const { return op_name_; }

  // Route exactly as operator() would, but by `op`'s conf key instead of this
  // dispatcher's own ("mm.out", "addmm.dtype", ...): one dispatcher answers
  // several schemas, and the conf routes each by its own name. `op` is only the
  // key; the kernel is called with `args` alone.
  template <typename... Args>
  decltype(auto) DispatchAs(const char* op, Args&&... args) const {
    const std::string op_name(op);
    const auto& configured = GetBackendConfigForOp(op_name);
    auto backend = configured.backend;
    auto fn = ResolveFn(op_name.c_str(), configured, backend, args...);
    LogDispatch(op_name, backend);

    if (!fn) ThrowIfForcedBackendMissing(op_name, backend);

    TORCH_CHECK(fn, op_name, DispatchFailureMessage(backend));
    return fn(std::forward<Args>(args)...);
  }

 private:
  // GetFn, plus the dtype escape from the FlagGems route.
  //
  // Existing platform capability guards predate configured dtype clauses.
  // On Ascend FlagGems' pointwise codegen simply does not compile for
  // float64, and the vendor kernels do. The same is true of gaps that are the
  // intersection of one op and one dtype (bool `neg`), which is why the op
  // name is part of the question rather than a filter on it. See
  // FlagGemsRejectsDtype and FlagGemsRejectsOpDtype. The substitution has to
  // happen where both the arguments and the name are visible, which is here.
  // `backend` is rewritten alongside, so the dispatch log and the
  // "backend not registered" message name what actually ran rather than what
  // the conf asked for.
  //
  // Only the Python route is guarded: the C++ one (kFlagGemsCpp) exists only
  // on the vendors that measured it, so its conf entries are already a per-op
  // decision made from a measurement.
  // The build's own native kernel slot, with the backend name that selects it.
  // Exactly one is ever populated: a build registers its native kernels into
  // its own slot (kAscend on Ascend, kMthreads on MUSA, ...) and, on the
  // CUDA-compatible platforms, boxing kernels into kBoxing -- which this does
  // not offer. The boxing kernels are not this build's own (a CPU-torch shim or
  // another vendor's runtime can host them), and the conf's own degradation
  // path already reads a FlagGems route with no FlagGems kernel as
  // `cuda_fn_ ? cuda_fn_ : flaggems_fn_`. Offering the slot when both exist
  // made FlagGems a second runtime route: NVIDIA's conf leaves index_select off
  // FlagGems for a broken kernel, and a FlagGems route could still land on the
  // boxing kernel. Ordered most-specific-first so a build that somehow
  // populated two still prefers the vendor's own kernels.
  std::pair<FnPtr, Backend> VendorSlot() const {
    if (ascend_fn_)      return {ascend_fn_,      Backend::kAscend};
    if (mthreads_fn_)    return {mthreads_fn_,    Backend::kMthreads};
    if (auto fn = metax_fn_.load(std::memory_order_acquire)) return {fn, Backend::kMetax};
    if (enflame_fn_)     return {enflame_fn_,     Backend::kEnflame};
    if (kunlunxin_fn_)   return {kunlunxin_fn_,   Backend::kKunlunxin};
    if (iluvatar_fn_)    return {iluvatar_fn_,    Backend::kIluvatar};
    if (tsingmicro_fn_)  return {tsingmicro_fn_,  Backend::kTsingMicro};
    if (apple_fn_)       return {apple_fn_,       Backend::kApple};
    return {nullptr, Backend::kNone};
  }

  template <typename... Args>
  FnPtr ResolveFn(const char* op_name, const BackendConfigEntry& configured, Backend& backend,
                  const Args&... args) const {
    for (const auto& candidate : configured.dtypes) {
      if (detail::ConfiguredDtypeMatchesArgs(candidate.dtype_mask, args...)) {
        if (auto fn = GetFn(candidate.backend)) {
          backend = candidate.backend;
          return fn;
        }
      }
    }
    FnPtr fn = GetFn(backend);
    if (backend != Backend::kFlagGems || fn == nullptr) return fn;
    if (!detail::FlagGemsRejectsArgs(op_name, args...)) return fn;

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
      // The boxing kernel set is the opt-in (FLAGOS_BUILD_BOXING=ON); the
      // default CUDA wheel is FlagGems-primary and compiles none of it. A
      // platform ships ONE conf, so -- exactly as for the three sets below --
      // the same file has to stay correct for both builds. Degrade to the
      // FlagGems Python path rather than hard-failing on the empty slot: an op
      // whose boxing kernel is absent is otherwise reachable only through the
      // registration FlagGems installs itself, and returning nullptr there
      // turns "this build does not box" into a dispatch error for ops that ran
      // before the route moved. Same shape as the kFlagGems arm below.
      case Backend::kBoxing:
        return cuda_fn_ ? cuda_fn_ : flaggems_fn_;
      // FlagGems C++ runtime is only compiled in for a FLAGOS_BUILD_FLAGGEMS_CPP=ON
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
      // FlagGems Python path is only compiled in for a FLAGOS_BUILD_FLAGGEMS=ON build
      // (flaggems_python_kernels.cc, behind FLAGOS_FLAGGEMS_PYTHON). When uncompiled,
      // degrade to the boxing kernel rather than raising "backend not registered".
      // This keeps one conf correct for both builds (Python FlagGems ON/OFF).
      case Backend::kFlagGems:
        if (flaggems_fn_) return flaggems_fn_;
        return cuda_fn_ ? cuda_fn_ : flaggems_cpp_fn_;
      case Backend::kAscend:        return ascend_fn_;
      case Backend::kMthreads:      return mthreads_fn_;
      case Backend::kMetax:         return metax_fn_.load(std::memory_order_acquire);
      case Backend::kTsingMicro:    return tsingmicro_fn_;
      case Backend::kEnflame:       return enflame_fn_;
      case Backend::kKunlunxin:     return kunlunxin_fn_;
      case Backend::kIluvatar:      return iluvatar_fn_;
      case Backend::kCambricon:     return cambricon_fn_;
      case Backend::kApple:         return apple_fn_;
      case Backend::kNvidia:        return nvidia_fn_.load(std::memory_order_acquire);
      case Backend::kHygon:         return hygon_fn_.load(std::memory_order_acquire);
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

  // Decoded pairs only accept raw kernels from these slots.
  FnPtr DecodedFallbackFn(const std::string& name, Backend& backend) const {
    if (ParseBackendName(name, &backend)) {
      switch (backend) {
        case Backend::kFlagGemsCpp: return flaggems_cpp_fn_;
        case Backend::kBoxing: return cuda_fn_;
        case Backend::kNvidia: return nvidia_fn_.load(std::memory_order_acquire);
        case Backend::kHygon: return hygon_fn_.load(std::memory_order_acquire);
        default: break;
      }
    }
    TORCH_CHECK(false,
                "FLAGOS_SOFT_LOWP_STAY_ON_ROUTE=\"",
                name,
                "\" names no live slot for a decoded pair; use boxing or "
                "flaggems_cpp");
  }

  // Why a decoded pair will not be handed to a route: the conf's answer for
  // the key, and what was registered for it. An unstated key is its own
  // answer: GetBackendForOp answers FlagGems for a key the table does not hold,
  // and says in its own comment that this means the running conf and the
  // compiled registration disagree, so the message says that rather than claim
  // the conf asked for FlagGems.
  static std::string DecodedRefusalFor(const std::string& op_name,
                                       Backend backend) {
    if (!HasBackendForOp(op_name)) {
      return op_name +
             ": the backend config does not name this op, so the loaded conf "
             "and the compiled registration disagree and there is no route to "
             "hand a decoded pair to";
    }
    if (backend == Backend::kNone) {
      return op_name +
             ": the backend config routes this op to 'none', so there is no "
             "accelerated implementation for a decoded pair to go to";
    }
    if (backend == Backend::kFlagGems) {
      return op_name +
             ": the backend config routes this op to FlagGems, which resolves "
             "the operand dtype against torch's own dtype objects before any "
             "kernel runs and cannot take a pair decoded in place of an "
             "FP8/FP4 one";
    }
    return op_name +
           ": the backend config routes this op to a backend no kernel was "
           "registered for";
  }

  static void LogDispatch(const std::string& op_name, Backend backend) {
    static const bool enabled = LogEnabled("dispatch");
    if (!enabled) return;
    fprintf(stderr, "[flagos dispatch] %s -> %s\n", op_name.c_str(),
            BackendLogName(backend));
  }

  const char* op_name_ = nullptr;
  // Per-op backend cache for the hot operator() path (see comment there).
  // Publish the immutable entry as one pointer, including its dtype clauses.
  mutable std::atomic<const BackendConfigEntry*> cached_config_{nullptr};
  FnPtr cuda_fn_           = nullptr;
  FnPtr flaggems_cpp_fn_   = nullptr;
  FnPtr flaggems_fn_       = nullptr;
  std::atomic<FnPtr> nvidia_fn_{nullptr};
  std::atomic<FnPtr> hygon_fn_{nullptr};
  FnPtr ascend_fn_         = nullptr;
  FnPtr mthreads_fn_       = nullptr;
  std::atomic<FnPtr> metax_fn_{nullptr};
  FnPtr tsingmicro_fn_     = nullptr;
  FnPtr enflame_fn_        = nullptr;
  FnPtr kunlunxin_fn_      = nullptr;
  FnPtr iluvatar_fn_       = nullptr;
  FnPtr cambricon_fn_      = nullptr;
  FnPtr tileops_fn_        = nullptr;
  FnPtr apple_fn_          = nullptr;
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
