// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include "common.h"
#include <c10/util/Exception.h>
#include <cstdio>
#include <exception>
#include <flagos_env.h>
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
//   REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kBoxing,   my_cuda_mm)
//
//   // call (uses op_name "mm" for config lookup):
//   mm_dispatcher(self, mat2, out);
//
//   // call with op name override (uses "mm.out" for config lookup):
//   mm_dispatcher.DispatchAs("mm.out", self, mat2, out);

namespace detail {

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

namespace soft_lowp {

// The software low-precision path's view of the route it hands an
// ordinary-dtype matrix pair back to. Defined in soft_lowp/dispatch.cc, so only
// a build that compiles that file calls these.
//
// A route can refuse the operands it is given without being broken.
// KernelFor() answers "is something registered here", not "can that something
// run", and for the FlagGems Python route those are different questions: the
// callable is looked up on first call, so a FlagGems kernel that raises at
// compile time -- `flag_gems.bmm` on MetaX wants K >= 16, a 4x4 test operand
// gives K = 4 -- is still "a kernel" here. It is a real kernel, though: the
// conf routed the op there and it declined these operands.

// The backend whose kernel raised `error`, or Backend::kUncached when the
// exception did not come from a FlagGems route at all.
Backend RefusingRoute(const std::exception& error);

// Whether any Python kernel has run in this process yet. A route whose callable
// this build has not looked up cannot have been called.
bool PythonRouteResolved();

} // namespace soft_lowp

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
      case Backend::kMetax:         metax_fn_ = fn;            break;
      case Backend::kTsingMicro:    tsingmicro_fn_ = fn;       break;
      case Backend::kEnflame:       enflame_fn_ = fn;         break;
      case Backend::kKunlunxin:     kunlunxin_fn_ = fn;        break;
      case Backend::kIluvatar:      iluvatar_fn_ = fn;         break;
      case Backend::kTileOps:       tileops_fn_ = fn;          break;
      case Backend::kApple:         apple_fn_ = fn;           break;
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
    Backend backend = cached_backend_;
    if (__builtin_expect(backend == Backend::kUncached, 0)) {
      backend = GetBackendForOp(op_name_);
      cached_backend_ = backend;
    }
    auto fn = ResolveFn(op_name_, backend, args...);
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
  // path), and the software path has to hand the ordinary ones back to whatever
  // the conf names -- FlagGems on MetaX, boxing elsewhere -- rather than to one
  // hard-coded slot.
  template <typename... Args>
  decltype(auto) DispatchConfigured(Args&&... args) const {
    return operator()(std::forward<Args>(args)...);
  }

  // The same for the one caller where re-entry is a real risk: the software
  // low-precision path handing an ordinary-dtype matrix pair back to the conf.
  //
  // That caller's route can BE this dispatcher's own kernel set.
  // `soft_lowp/dispatch.cc` registers its mm/bmm on Backend::kFlagGemsCpp when
  // no FlagGems C++ kernel owns the slot, so a `flaggems_cpp` route hands the
  // pair to this dispatcher, which hands it straight back -- the recursion that
  // ran the stack out on MetaX (SIGSEGV, `torch.bmm` float32, CI run
  // 40242938356). Its signature is ordinary-dtype only, so the infinite variant
  // is a kernel that raises, and the two are told apart by whether the kernel
  // this frame called was resolved before the call:
  //
  //   * `error` came from a FlagGems route whose callable had not been resolved
  //     yet: the route ran for the first time and refused these operands, which
  //     is ordinary here -- an FP8/FP4 pair reaches the wrapper, the software
  //     path decodes it and hands the bf16 pair back, and the route may decline
  //     the shape (FlagGems' MetaX bmm compiles only K >= 16). The refusal is
  //     reported and the call continues on what the conf's own degradation
  //     reads: this build's native kernels, else boxing.
  //   * anything else -- a resolved kernel still raising, an exception from
  //     this code or from the dispatcher -- is rethrown. A loaded kernel that
  //     raises twice would recurse just the same, and a defect must not be
  //     laundered into a fallback.
  //
  // Without the route refusing, this is operator(): same route, same log line,
  // same message.
  template <typename... Args>
  decltype(auto) DispatchConfiguredForSoftLowp(Args&&... args) const {
#if defined(FLAGOS_SOFT_LOWP)
    Backend backend = cached_backend_;
    if (__builtin_expect(backend == Backend::kUncached, 0)) {
      backend = GetBackendForOp(op_name_);
      cached_backend_ = backend;
    }
    auto fn = ResolveFn(op_name_, backend, args...);
    LogDispatch(op_name_, backend);
    if (!fn) ThrowIfForcedBackendMissing(op_name_, backend);
    TORCH_CHECK(fn, op_name_, DispatchFailureMessage(backend));

    const bool was_resolved = soft_lowp::PythonRouteResolved();
    try {
      return fn(std::forward<Args>(args)...);
    } catch (const std::exception& error) {
      if (!was_resolved) throw;
      const auto refused_by = soft_lowp::RefusingRoute(error);
      if (refused_by == Backend::kUncached) throw;

      FnPtr next_fn = nullptr;
      const char* refused_name = nullptr;
      const char* next_name = nullptr;
      if (!ResolveBelow(refused_by, next_fn, refused_name, next_name)) {
        // Nothing below the route: the refusal is the answer, exactly as it is
        // for any call on a build that routes here directly.
        throw;
      }
      LogDispatchRefused(op_name_, refused_name, next_name);
      return next_fn(std::forward<Args>(args)...);
    }
#else
    return operator()(std::forward<Args>(args)...);
#endif
  }

  template <typename... Args>
  decltype(auto) DispatchAs(const std::string& op_name, Args&&... args) const {
    auto backend = GetBackendForOp(op_name);
    auto fn = ResolveFn(op_name.c_str(), backend, args...);
    LogDispatch(op_name, backend);

    if (!fn) ThrowIfForcedBackendMissing(op_name, backend);

    TORCH_CHECK(fn, op_name, DispatchFailureMessage(backend));
    return fn(std::forward<Args>(args)...);
  }

 private:
  // FLAGOS_FORCE_BACKEND asks for one backend family across the whole routing
  // table, so a miss for the family it named is a hard error rather than a
  // silent fall-through. Only `flaggems` and `vendor` get here: `tileops` is a
  // repin of ops the conf already annotates `# tileops`, so a miss there means
  // the op was never a TileOPs candidate and nothing was asked of it.
  void ThrowIfForcedBackendMissing(const std::string& op_name,
                                   Backend backend) const {
    const std::string& forced = ForcedBackendMode();
    if (forced.empty()) return;

    if (forced == "flaggems" &&
        (backend == Backend::kFlagGemsCpp || backend == Backend::kFlagGems)) {
      throw std::runtime_error(
          op_name +
          ": FLAGOS_FORCE_BACKEND=flaggems but no FlagGems implementation is "
          "compiled into this wheel (rebuild with FLAGOS_BUILD_FLAGGEMS_CPP=1 "
          "or FLAGOS_BUILD_FLAGGEMS=1)");
    }
    if (forced == "vendor" && backend != Backend::kFlagGemsCpp &&
        backend != Backend::kFlagGems && backend != Backend::kNone &&
        backend != Backend::kTileOps) {
      throw std::runtime_error(
          op_name +
          ": FLAGOS_FORCE_BACKEND=vendor but no vendor kernel is registered "
          "for this op");
    }
  }

  // Where a route that refused its operands sends the call: this build's own
  // native kernels if it has any, else the boxing set. The same two places the
  // conf's own degradation reads (GetFn's kFlagGems and kFlagGemsCpp arms),
  // and nowhere else.
  bool ResolveBelow(Backend refused, FnPtr& fn, const char*& refused_name,
                    const char*& next_name) const {
    auto [vendor_fn, vendor_backend] = VendorSlot();
    if (vendor_fn != nullptr) {
      fn = vendor_fn;
      refused_name = NameOf(refused);
      next_name = NameOf(vendor_backend);
      return true;
    }
    if (cuda_fn_ != nullptr) {
      fn = cuda_fn_;
      refused_name = NameOf(refused);
      next_name = NameOf(Backend::kBoxing);
      return true;
    }
    return false;
  }

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
  // boxing kernel. Ordered
  // most-specific-first so a build that somehow populated two still prefers
  // the vendor's own kernels.
  std::pair<FnPtr, Backend> VendorSlot() const {
    if (ascend_fn_)      return {ascend_fn_,      Backend::kAscend};
    if (mthreads_fn_)    return {mthreads_fn_,    Backend::kMthreads};
    if (metax_fn_)       return {metax_fn_,       Backend::kMetax};
    if (enflame_fn_)     return {enflame_fn_,     Backend::kEnflame};
    if (kunlunxin_fn_)   return {kunlunxin_fn_,   Backend::kKunlunxin};
    if (iluvatar_fn_)    return {iluvatar_fn_,    Backend::kIluvatar};
    if (tsingmicro_fn_)  return {tsingmicro_fn_,  Backend::kTsingMicro};
    if (apple_fn_)       return {apple_fn_,       Backend::kApple};
    return {nullptr, Backend::kNone};
  }

  // GetFn, plus the dtype escape from the FlagGems route.
  //
  // A conf routes per op, so it cannot say "FlagGems, except for float64" --
  // but on Ascend FlagGems' pointwise codegen simply does not compile for
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
  template <typename... Args>
  FnPtr ResolveFn(const char* op_name, Backend& backend,
                  const Args&... args) const {
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
      case Backend::kMetax:         return metax_fn_;
      case Backend::kTsingMicro:    return tsingmicro_fn_;
      case Backend::kEnflame:       return enflame_fn_;
      case Backend::kKunlunxin:     return kunlunxin_fn_;
      case Backend::kIluvatar:      return iluvatar_fn_;
      case Backend::kApple:         return apple_fn_;
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
  //
  // Every message keeps the phrase "backend not registered": it is what callers
  // and tests match on to tell "this wheel cannot do that" from "the op is
  // broken". The build set, when the wheel records one, is appended as the
  // reason rather than replacing it.
  static std::string DispatchFailureMessage(Backend backend) {
    if (backend == Backend::kNone) {
      return ": routed to 'none' (no accelerated impl on this platform) but the "
             "op is registered on PrivateUse1 -- regenerate the vendor conf so "
             "registration and routing agree";
    }
    std::string message = ": backend not registered";
#if defined(FLAGOS_BUILTIN_KERNELS)
    // The compile definition setup.py derives from the same kernel switches as
    // the build record, so the wheel can name the missing set instead of
    // leaving the user to guess whether it is a build gap or a broken install.
    const char* kernel_set = KernelSetOf(backend);
    if (kernel_set != nullptr &&
        !flagos_env::ListedIn(FLAGOS_BUILTIN_KERNELS, kernel_set)) {
      message += std::string(" (the '") + kernel_set +
                 "' kernel set was not compiled into this wheel)";
    }
#endif
    return message;
  }

  // Which kernel set provides a backend, named the way setup.py::KERNEL_SET_NAME
  // and the KERNELS tuple in torch_fl/_build_config.py name it. nullptr for a
  // backend that no kernel set owns.
  static const char* KernelSetOf(Backend backend) {
    switch (backend) {
      case Backend::kBoxing:        return "boxing";
      case Backend::kFlagGemsCpp: return "flaggems_cpp";
      case Backend::kFlagGems:    return "flaggems";
      case Backend::kTileOps:     return "tileops";
      // Every remaining backend is the accelerator's own native kernel library
      // (ACLNN, mudnn, topsaten, ...), which the single FLAGOS_BUILD_VENDOR switch
      // compiles. There is one such library per build, never two.
      case Backend::kAscend:
      case Backend::kMthreads:
      case Backend::kMetax:
      case Backend::kTsingMicro:
      case Backend::kEnflame:     return "vendor";
      case Backend::kKunlunxin:   return "vendor";
      case Backend::kIluvatar:    return "vendor";
      case Backend::kApple:       return "vendor";
      case Backend::kNone:
      case Backend::kUncached:    return nullptr;
    }
    return nullptr;
  }

  // The names the dispatch log prints. Several are what the SDK-era conf spelled
  // the same kernel (`musa`, `gcu`, `mps`, `flagos`, `flagos_python`) and are kept
  // for the tests and logs that read them. tests/unit/test_ops_conf_log_names.py
  // holds this list against the shipped confs: a conf key the log cannot print
  // has no dispatch line to assert against.
  static void LogDispatch(const std::string& op_name, Backend backend) {
    static const bool enabled = LogEnabled("dispatch");
    if (!enabled) return;
    fprintf(stderr, "[flagos dispatch] %s -> %s\n", op_name.c_str(),
            NameOf(backend));
  }

  // The name the dispatch log prints for `backend`.
  static const char* NameOf(Backend backend) {
    const char* name;
    switch (backend) {
      case Backend::kBoxing:        name = "boxing"; break;
      case Backend::kFlagGemsCpp:   name = "flagos"; break;  // keep old name for test compat
      case Backend::kFlagGems:      name = "flagos_python"; break;  // keep old name for test compat
      case Backend::kAscend:        name = "ascend"; break;
      case Backend::kMthreads:      name = "musa"; break;
      case Backend::kMetax:         name = "metax"; break;
      case Backend::kTsingMicro:    name = "tsingmicro"; break;
      case Backend::kEnflame:       name = "gcu"; break;
      case Backend::kKunlunxin:     name = "kunlunxin"; break;
      case Backend::kIluvatar:      name = "iluvatar"; break;
      case Backend::kTileOps:       name = "tileops"; break;
      case Backend::kApple:         name = "mps"; break;
      case Backend::kNone:          name = "none"; break;
      default:                      name = "unknown"; break;
    }
    return name;
  }

  // The dispatch log's line for a route that refused its operands and the one
  // that took the call instead. Kept apart from LogDispatch so a call still
  // writes exactly one `op -> backend` line, with the move-on beside it.
  static void LogDispatchRefused(const char* op_name, const char* refused,
                                 const char* next) {
    static const bool enabled = LogEnabled("dispatch");
    if (!enabled) return;
    fprintf(stderr,
            "[flagos dispatch] %s: %s refused these operands, -> %s\n", op_name,
            refused, next);
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
  FnPtr mthreads_fn_       = nullptr;
  FnPtr metax_fn_          = nullptr;
  FnPtr tsingmicro_fn_     = nullptr;
  FnPtr enflame_fn_        = nullptr;
  FnPtr kunlunxin_fn_      = nullptr;
  FnPtr iluvatar_fn_       = nullptr;
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
