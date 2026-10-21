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
  kKunlunxin,
  kIluvatar,
  kTileOps,
  kMps,            // Apple Metal / MPSGraph (conf key: mps)
  // No accelerated implementation on this platform. The op is deliberately not
  // registered on PrivateUse1, so it reaches the boxed cpu_fallback instead of
  // arriving here with an empty kernel slot. Recorded explicitly (rather than
  // by omission) so a vendor conf states coverage for every op and support can
  // be counted from the file.
  kNone,
  kUncached
};

// Returns the backend for a given op name, loaded once from config file at startup.
// Config file path, in order of precedence:
//   1. SetBackendConfigPath(), called by torch_fl._select_backend_config()
//   2. $FLAGOS_BACKEND_CONFIG
//   3. torch_fl/configs/backends_cuda.conf, located from this library's own path
// Format: "op_name = backend"
//   backend: "flaggems_cpp" -- FlagGems C++ path (liboperators.so)
//            "flaggems"     -- FlagGems Python (Triton) path
//            "tileops"      -- TileOps Triton shims
//            "<vendor>"     -- vendor-native kernel (cuda | ascend | musa |
//                              metax | gcu | tsingmicro | kunlunxin | iluvatar)
//            "none"         -- no accelerated impl; reaches cpu_fallback
// Default when op is not listed: kFlagGems.
Backend GetBackendForOp(const std::string& op_name);

// True when the conf names `op_name` at all, i.e. whether the value above is
// the file's routing decision or the unlisted default. Most callers do not need
// this: for a generated leaf kernel the two are equivalent, because the
// default and the absence of an override lead to the same kernel. It matters
// for an op whose FlagGems support is per-platform and opt-in by measurement --
// a conf that predates that route, or a third party's, has not asked for it, so
// "unlisted" has to keep the pre-existing path instead of silently acquiring a
// new one. A caller that reads false must therefore do what the op did before
// the FlagGems route existed, not pick a default of its own; see
// csrc/aten/sdp_choice_stub.cc.
bool HasBackendForOp(const std::string& op_name);

// Record the conf path Python resolved at import time. Must be called before
// the first op dispatch, which is when the table is built. It is not written to
// os.environ: the wheel's own selection has to stay distinguishable from a
// user's FLAGOS_BACKEND_CONFIG, and an environment write makes them identical
// for the rest of the process.
//
// FLAGOS_EXPORT because of who calls it: the binding is
// torch_fl._C._set_backend_config_path, and torch_fl/csrc/module.cc is linked
// into libtorch_bindings.so, not into libtorch_fl.so. CMakeLists.txt sets
// CMAKE_CXX_VISIBILITY_PRESET hidden, so an unannotated definition here is
// local to libtorch_fl.so and the import fails with an undefined symbol rather
// than a link error -- shared-library linking leaves undefined symbols alone.
FLAGOS_EXPORT void SetBackendConfigPath(const std::string& path);

// The value of FLAGOS_FORCE_BACKEND -- "flaggems", "vendor" or "tileops" -- or an
// empty string when it is unset (or was unparseable, which warns and reads as
// unset). Resolved once per process, so Dispatcher can consult it on the
// dispatch-miss path without re-reading the environment per op.
const std::string& ForcedBackendMode();

// True when FLAGOS_LOG -- a comma-separated list -- names `item`, one of
// "dispatch", "fallback" or "op_cache". Three separate booleans used to gate
// these, and none could be discovered by reading the environment: you had to
// know the name first. One list puts the whole menu in one place, in the
// variable's own value.
//
// An item that names nothing is reported once per process, because a typo would
// otherwise turn a diagnostic off silently -- which is exactly what those
// booleans did not do.
bool LogEnabled(const char* item);

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
// On MetaX the broad gap is complex64/complex128. Its Triton backend has no
// complex pointer type, so a FlagGems kernel instantiated from either dtype
// fails during compilation with ``KeyError: 'complex64'`` (or complex128).
// The CUDA-ABI boxing kernels reach MetaX torch's native complex support, so
// the vendor slot is both available and the correct fallback for these calls.
// DCU is the same gap and the same fallback, measured separately on its own
// shipped wheel (see common.cc); the difference is that DCU reached it through
// a silent dtype loss first, because its dtype converter had no complex arm
// either (issue #540), so the symptom there was a complex request quietly
// returning float32 rather than a KeyError. The converter is fixed alongside
// the escape, and the diversion then carries the factory family (`zeros`,
// `ones`, `randn`, `abs`) to the vendor kernels with dtype and device intact.
// See the measurement in common.cc.
//
// On GCU the same shape of gap exists for int64, and it fails in two
// different places on the way to the same non-existent kernel. A kernel whose
// operand is i64 makes flag_gems' pointwise codegen request a pass option the
// installed toolkit does not declare -- flag_gems 5.3.2 and the enflame 3.6
// triton backend append `enable_i64=true` to `--convert-gpu-to-gcu`
// (runtime/backend/_enflame/gcu300/utils/pointwise_dynamic.py, and
// triton/backends/enflame/compiler.py), while
// /opt/triton_gcu/bin/gcu-compiler-opt (2026-05-21, LLVM 21.0.0git) exposes
// only `--chipset` and `--vector-bit-width` for that pass:
//
//   Exception: <unknown>:0: error: <Pass-Options-Parser>: no such option
//   enable_i64
//
// raised from triton/backends/enflame/toolkit.py's `_run_command`. A kernel
// that gets past that raises the compiler's own verdict instead,
//
//   RuntimeError: Pipeline run failed: PassManager execution failed
//
// from `error: 64-bit data type not supported on GCU300!`. Measured on the
// shipped `backends_gcu.conf` (torch 2.10.0+cpu, S60, card 2): `clamp`,
// `clamp_min`, `clamp_max`, `clamp_`, `fmod.Tensor` and `rsub.Scalar` over
// int64 hit one or the other, and `remainder.Tensor` over int64 is worse
// than either -- it returns a plausible int32 tensor, so nothing upstream
// can tell it went wrong.
//
// The blast radius was measured rather than assumed, because the escape
// cannot help an op the platform has no vendor kernel for: `ResolveFn` is
// only consulted on the FlagGems route and returns that route unchanged when
// the op's vendor slot is empty, so a route can only move for an op the conf
// left on FlagGems *and* the GCU codegen registered a kernel for. That
// intersection is exactly seven ops on the shipped conf -- clamp,
// fmod.Tensor, gelu, mean, mean.dim, remainder.Tensor, silu -- and the vendor
// kernels that take them over are int64-safe by construction: every generated
// GCU kernel that calls topsaten is gated on TopsatenSupportsDtype, which
// excludes int64, so int64 there is the template's CPU round-trip.
//
// float64 is deliberately absent. It fails with the same two signatures
// (measured on the same conf: clamp/clamp_min/clamp_max/fmod.Tensor/
// rsub.Scalar/gelu/silu/mean over float64 all raise), but one FlagGems f64
// route inside that same seven -- remainder.Tensor -- is correct today, so a
// 64-bit-wide rule would trade a working FlagGems kernel for the vendor
// template's CPU round-trip in order to fix nothing, and no cohort here
// measures f64. That half wants its own decision with its own evidence.
//
// Ascend and GCU are separate `#if` branches because neither is a subset of
// the other: Ascend serves int64 on FlagGems and rejects float64, GCU serves
// float64 there and rejects int64.
//
// Deliberately a predicate on the dtype alone and not on (op, dtype): every op
// that reaches this build's FlagGems route with one of the dtypes above is
// affected, so naming the op would carry no information. A gap that is
// specific to one op cannot be expressed here and does not belong here --
// see FlagGemsRejectsOpDtype below.
bool FlagGemsRejectsDtype(at::ScalarType dtype);

// The (op, dtype) form of the same escape, for the gaps where neither half
// decides on its own: a FlagGems kernel that is unguarded about the element
// type it code-generates, called with a dtype the reference implementation of
// that op rejects outright, or with one the platform's triton dot intrinsic
// will not accept. Two shapes are in the table today: `neg` over bool, and
// `mm` / `mm.out` over int64/int32/int16/uint8 -- integral matmul, where the
// op must stay on FlagGems for float. The first is shared by every platform;
// the second is an Ascend measurement (see common.cc for both).
//
// Both of the other mechanisms were measured against this gap and neither can
// state it. A conf entry routes a whole op, and the dtype-wide predicate above
// cannot distinguish matmul from the other integral operators.
//
// `op_name` is the routed name -- the conf key, so it carries a ".out" suffix
// only for the calls dispatched under one -- not the ATen schema name.
bool FlagGemsRejectsOpDtype(const char* op_name, at::ScalarType dtype);

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
