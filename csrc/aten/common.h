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
  // The CUDA-ABI boxing kernel: re-dispatch PrivateUse1 to CUDA into a vendor's
  // own libtorch (DeviceBoxingGuard) instead of launching a kernel of our own.
  // Named for what it does, not for the runtime it borrows -- MetaX, Hygon DCU
  // and the T-Head PPU all route through this one slot. Conf key: boxing
  // ("cuda" is the legacy spelling, still parsed).
  kBoxing,
  kFlagGemsCpp,    // FlagGems C++ runtime (liboperators.so, conf key: flaggems_cpp)
  kFlagGems,       // FlagGems Python/Triton path (conf key: flaggems)
  // One value per vendor, spelled the way its conf key is (see AGENTS.md,
  // "Platform names"). These three used to be named after the SDK rather than
  // the vendor.
  kAscend,
  kMthreads,       // conf key: mthreads
  kMetax,
  kTsingMicro,
  kEnflame,        // conf key: enflame
  kKunlunxin,
  kIluvatar,
  kTileOps,
  kApple,          // Apple Metal / MPSGraph (conf key: apple)
  // D-Robotics RDK BPU. The kernels submit compiled operator artifacts from
  // the operator library named by FLAGOS_BPU_OPLIB; without it every kernel
  // takes the CPU round-trip. Conf key: drobotics.
  kDrobotics,
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
//   3. torch_fl/configs/backends_nvidia.conf, located from this library's own path
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
// Ascend complex dtypes also bypass FlagGems: its pointwise compiler cannot
// represent complex pointers. Generated vendor kernels use CANN complex
// factorizations or device-resident real components, retaining their precision.
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
// shipped `backends_enflame.conf` (torch 2.10.0+cpu, S60, card 2): `clamp`,
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
// will not accept. Examples include `neg` over bool, integral matmul, and
// int64 `where` with scalar conditions. The floating matmul route depends on
// its precision policy; see common.cc for the measured cases.
//
// Both of the other mechanisms were measured against this gap and neither can
// state it. A conf entry routes a whole op: `neg` over
// fp16/bf16/fp32/fp64/int8/int16/int32/int64/uint8 is correct on the FlagGems
// route, so a NATIVE_TRITON_GAPS entry would move all of them off it and onto
// the Ascend template's CPU round-trip -- which IsUnaryDtypeSupported sends
// every integral through, costing 7-30x on integral `neg` to gain 10x on
// fp32. The dtype-wide predicate above cannot state it either: bool is not a
// dtype FlagGems fails for in general (add/sub/abs/... all take a bool operand
// on this build), so a dtype-wide rule would take those down with it.
//
// `op_name` is the routed name -- the conf key, so it carries a ".out" suffix
// only for the calls dispatched under one -- not the ATen schema name.
bool FlagGemsRejectsOpDtype(const char* op_name, at::ScalarType dtype);

// The smallest contracted extent (the K of an M x K by K x N matmul) this
// platform's triton `tl.dot` will compile, or 0 where it has no such floor.
//
// Third shape of the same escape as the two predicates above, and the one
// neither of them can state: this is not a dtype question and not an op
// question, it is a *shape* floor, and it is the platform's triton backend
// that imposes it rather than any kernel's own guard.
//
// A caller below the floor must widen K itself; zero-filling the contracted
// axis is exact for a matmul, because the added lanes contribute 0 * 0 to
// every sum. Only K: `min_dot_size` bounds M and N at 1 everywhere, and
// padding those would change the output shape, which the `.out` forms cannot
// have.
//
// This belongs to the build, not to a chip: a wheel is built against exactly
// one FlagTree (.github/version-pins.env), so the floor its triton imposes is
// fixed for the life of the wheel. It also fails safe -- a FlagTree that
// lowers the floor turns the padding into a small wasted copy, never a wrong
// answer.
int64_t TritonMinDotContractExtent();

// The contracted axis of a matmul operand: the columns of the left side, the
// rows of the right one, at either rank.
int64_t MatmulContractDim(const at::Tensor& operand, bool is_rhs);

// Widen the contracted axis to `target` with zeros.
//
// Exact for a matmul: the added lanes are 0 on both sides and contribute
// 0 * 0 = 0 to every element of the sum, so the product is unchanged. Only the
// contracted axis is touched, so M and N -- and therefore the output shape --
// are the same, which is what lets the `.out` forms take this unchanged.
//
// Built with empty + zero_ + narrow().copy_() rather than at::constant_pad_nd:
// that op is itself routed to FlagGems on the platforms that have a floor, and
// FlagGems' triton is the thing being worked around here. empty and copy_
// carry no conf key, so this cannot bounce off a route.
at::Tensor PadMatmulContractDim(const at::Tensor& operand, bool is_rhs,
                                int64_t target);

// Widen both operands together when the platform's dot intrinsic will not
// compile their shared contracted extent. Returns whether it padded, so a
// caller that has to narrow the result back knows to do it.
//
// Lives here rather than in the soft-lowp decode that first needed it because
// the floor is a property of the platform's triton, not of low-precision
// arithmetic: an ordinary fp32 `mm` routed to FlagGems hits the same assertion
// with the same small K. See TritonMinDotContractExtent above.
bool PadMatmulPairForDot(at::Tensor& lhs, at::Tensor& rhs);

// Whether this platform's FlagGems matmul must not be handed `dtype`, with the
// host round-trip as the only alternative.
//
// Fourth shape of the escape the three predicates above describe, and the one
// none of them can state. It is a (platform, op-family, dtype) gap where the
// platform has no vendor kernel to divert to:
//
//   * FlagGemsRejectsOpDtype states exactly this intersection, but it works by
//     *substituting* VendorSlot()'s kernel (see ResolveFn in dispatcher.h), and
//     returns the FlagGems pointer unchanged when that slot is empty. MetaX
//     registers no vendor matmul, so an entry there compiles and does nothing.
//   * FlagGemsRejectsDtype is dtype-wide. float64 is not a dtype MetaX's
//     FlagGems fails for in general -- the pointwise family serves it -- so a
//     dtype-wide rule would move ops that work today.
//   * A conf entry routes a whole op. `mm` over fp16/bf16/fp32 is correct and
//     more accurate than boxing on this route (see METAX_FLAGGEMS_MEASURED),
//     so moving the op would give up the float case to fix the float64 one.
//
// Measured on a MetaX C550 (torch 2.10.0+metax3.8.1.0, flag_gems 5.4.0):
// FlagGems' tiled `mm` accumulator is declared from `dot_out_dtype` but the
// `tl.dot` inside the K loop re-assigns it at the operand's own width, which
// this triton refuses to compile:
//
//   triton.compiler.errors.CompilationError: at 75:4:
//     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
//     for k in range(0, tl.cdiv(K, BLOCK_K)):
//   AssertionError("Loop-carried variable acc has initial type
//   <['16', '16'], fp32> but is re-assigned to <['16', '16'], fp64> in loop!")
//
// On the development host the same call does not raise -- one autotune config
// reports `PassManager::run failed`, the tuner silently picks another, and the
// product comes back wrong by rel 0.97 to 1.77. A wrong answer with no
// exception is the worse of the two outcomes, so this is not a CI workaround.
//
// Answers false everywhere else, so a platform without the gap pays one enum
// compare. This is the same remedy Ascend applies for a dtype aclnn will not
// take (`MatmulKernelAscend` runs `at::matmul` on a CPU copy, see
// csrc/aten/backends/ascend/matmul.cc); the difference is only that MetaX has
// no vendor kernel to host it in, so the generated FlagGems kernel carries it.
bool FlagGemsMatmulNeedsHost(at::ScalarType dtype);

// Run `mm` on the host and bring the answer back to `reference`'s device, for
// the dtype FlagGemsMatmulNeedsHost names.
//
// Exact rather than a degradation: the host computes the matmul at the full
// requested width, which is what the dtype asked for and what the device route
// fails to deliver. Same shape as ascend::MoveCpuResultToDevice.
at::Tensor MatmulOnHost(const at::Tensor& lhs, const at::Tensor& rhs);

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
