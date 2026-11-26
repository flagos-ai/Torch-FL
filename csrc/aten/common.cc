// Copyright (c) 2026, BAAI. All rights reserved.

#include "common.h"

#include <ATen/Context.h>

#include <atomic>
#include <cstring>

namespace at::native::flagos {

namespace {
std::atomic<bool> strict_host_routes{false};
} // namespace

bool FlagGemsRejectsDtype(at::ScalarType dtype) {
#if defined(FLAGOS_ACCEL_METAX)
  // Triton's MetaX backend cannot lower complex pointers. FlagGems kernels
  // therefore fail before launch for every complex-valued op measured here
  // (the first FlagQuantum failure is flip):
  //
  //   KeyError: 'complex64'
  //
  // MetaX is CUDA ABI compatible, and boxing the same calls into the vendor
  // torch kernels preserves both the complex dtype and device residency.
  return dtype == at::kComplexFloat || dtype == at::kComplexDouble;
#elif defined(FLAGOS_ACCEL_DCU)
  // Same gap, same shape as MetaX, and measured separately in the shipped DCU
  // environment (torch 2.10.0+dtk2604, HIP 6.3.26113, DTK, `backends_hygon.conf`,
  // boxing ON). The released wheel predates this change, so it reproduces the
  // two defects below; the fix was measured on a libtorch_fl.so rebuilt from
  // this source and copied into that same environment.
  //
  // Without this arm a complex factory never failed loudly: it silently lost
  // its dtype before FlagGems was reached, because the wheel's dtype converter
  // had no complex arm (issue #540). The visible result was
  //
  //   torch.zeros(4, dtype=torch.complex64, device="flagos:0").dtype
  //   # -> torch.float32
  //
  // FlagGems' HCU Triton backend has no complex pointer type either, so simply
  // preserving the dtype in the converter is not enough -- the call must leave
  // the FlagGems route. With this escape the complex call reroutes to the DCU
  // CUDA boxing kernels, measured on the rebuilt build (FLAGOS_LOG=dispatch,
  // and values checked against the CPU reference):
  //
  //   zeros  -> cuda   complex64/128 preserved (was silent float32)
  //   ones   -> cuda   complex64/128 preserved (was silent float32)
  //   randn  -> cuda   complex64/128 preserved (was silent float32)
  //   abs    -> cuda   correct real result    (was KeyError: 'complex64')
  //   fill_  -> cuda   correct               (composed through the same escape)
  //
  // the last one because OnesKernelCuda and ZeroKernelCuda are compositional
  // (`at::empty(...).fill_(1)` / `.zero_()`), and the inner in-place call is
  // itself diverted -- `fill_` and `zero_` are registered on the FlagGems
  // route and the DCU codegen has a kBoxing kernel for each. The escape is
  // dtype-scoped, so the same op over float is untouched: measured
  // `ones(float32)` still logs `ones -> flagos_python`.
  //
  // ResolveFn returns the FlagGems route unchanged when no vendor kernel
  // exists, so an op outside this intersection keeps its own outcome (loud
  // failure) rather than changing route.
  return dtype == at::kComplexFloat || dtype == at::kComplexDouble;
#elif defined(USE_ILUVATAR)
  // FlagGems' complex elementwise path is wrong here: `add` returns zeros for
  // complex128, and `div` raises for complex64 (flag_gems/ops/add.py decomposes
  // both operands with `view_as_real` and then runs its own kernel). The
  // in-place spellings are handled by the conf, which routes them straight to
  // the native kernels; the out-of-place `add.Tensor`/`add.Scalar` are routed
  // to FlagGems because that is what the platform's callers need -- 545 calls
  // per forward of issue #667's model, measured 2757.9 -> 2369.4 ms when the
  // route moved -- so their complex operands have to leave the route at run
  // time instead.
  //
  // ResolveFn returns the FlagGems route unchanged when the dispatcher has no
  // kIluvatar slot, which today is every op except the add pair: register.cc's
  // USE_ILUVATAR arm publishes an out-of-place host round trip for exactly
  // those two. Nothing else changes route.
  return c10::isComplexType(dtype);
#elif defined(USE_ASCEND)
  // FlagTree cannot instantiate general complex kernels. The generated
  // native route decomposes supported complex operations into real kernels.
  return dtype == at::kDouble || c10::isComplexType(dtype);
#elif defined(USE_GCU)
  // See the declaration in common.h for the measurement. int64 is the dtype
  // that matters: it fails inside the compiler for every flag_gems pointwise
  // kernel that takes one, and for remainder.Tensor it does not fail at all --
  // it returns int32. float64 fails the same way but is deliberately left
  // alone, because one FlagGems float64 route in the same seven-op
  // intersection (remainder.Tensor) is correct today.
  return dtype == at::kLong;
#else
  // No vendor route is known to be dtype-limited in this way. MUSA's
  // FlagGems gaps are per-op (a Python-float operand against a bf16 tensor)
  // and are already recorded in NATIVE_TRITON_GAPS, so they are not reached
  // from here.
  (void)dtype;
  return false;
#endif
}

bool FlagGemsRejectsOpDtype(const char* op_name, at::ScalarType dtype) {
  if (FlagGemsRejectsDtype(dtype)) return true;
#if defined(USE_ASCEND)
  // The intersection this table exists for: a FlagGems kernel that does not
  // guard the element type it code-generates, for a dtype the reference
  // implementation of that op rejects outright. See the declaration in
  // common.h for why neither a conf entry nor the dtype-wide predicate can
  // state it.
  //
  // `neg` over bool. flag_gems/ops/neg.py is a bare pointwise `-x`, so a bool
  // operand is code-generated like any other element type and lowers to
  // `hivm.hir.vadd` over `i1`, which BiShengIR refuses to verify:
  //
  //   MLIRCompilationError: 'hivm.hir.vadd' op failed to verify that operand
  //   at idx 0 and 1 should have element type 16-bit signless integer or
  //   32-bit signless integer or 16-bit float or 32-bit float or 64-bit
  //   signless integer
  //
  // (Ascend910, CANN 9.0.0, FlagTree 0.6.2a1+ascend3.5.) Reached through the
  // vendor slot instead, the aclnn kernel falls back to at::neg on a CPU copy
  // for a dtype IsUnaryDtypeSupported does not cover -- which is where the
  // reference behaviour
  //
  //   RuntimeError: Negation, the `-` operator, on a bool tensor is not
  //   supported. If you are trying to invert a mask, use the `~` or
  //   `logical_not()` operator instead.
  //
  // is raised. tests/integration/test_dtype_coverage.py pins that contract.
  //
  // `neg_` is deliberately absent: its vendor template (T_INPLACE_UNARY) has
  // no dtype guard at all, so routing bool there reaches aclnnInplaceNeg and
  // fails with "aclnnInplaceNegGetWorkspaceSize failed, ret=161002" rather
  // than with the reference message. That is a separate defect in the
  // in-place codegen and is left visible rather than papered over here.
  //
  // `mm` and `mm.out` over the integral dtypes are the second entry, and the
  // only one where the op stays on the FlagGems route while part of its dtype
  // set leaves it. The FlagGems matmul kernel builds its tl.dot from the
  // operand element type, and the Ascend dot intrinsic takes floating point
  // only, so int64/int32/int16/uint8 operands die inside the compiler:
  //
  //   triton.compiler.errors.CompilationError: at 56:15:
  //     acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
  //   AssertionError('Unsupported lhs dtype int64')
  //
  // (Ascend910C, CANN 9.0.0, shipped backends_ascend.conf.) int8 is the
  // exception and is left alone: tl.dot accepts it and the FlagGems route is
  // correct for it. The aclnn kernel behind the vendor slot serves all four --
  // integer matmul is exact there, so this is not a precision trade. Measured
  // on the same conf: int64/int32/int16/uint8 `mm` on the vendor route agree
  // with the CPU reference exactly.
  //
  // A route cannot state this split, which is why the table has to. `mm` was
  // moved off the vendor route because the aclnn cube path downcasts fp32 to
  // HF32 across the whole float family -- 4.7e-3 against a CPU reference, where
  // tests/integration/runtime asserts 1e-3 and the FlagGems kernel answers
  // 8.6e-6 against that same reference (see NATIVE_TRITON_GAPS in
  // scripts/codegen/gen_vendor_confs.py). Leaving mm on the vendor route keeps
  // the float failure; leaving every dtype of it on FlagGems turns working
  // integral matmul into a compile error.
  struct OpDtypeGap {
    at::ScalarType dtype;
    const char* op_name;
  };
  static constexpr OpDtypeGap kGaps[] = {
      {at::kBool, "neg"},
      // The Ascend where override cannot lower an int64 select for a scalar
      // condition and a one-dimensional result (circuit basis indexing).
      {at::kLong, "where.self"},
      {at::kLong, "where.self_out"},
      // Integral matmul: the four dtypes the Ascend dot intrinsic refuses.
      // "mm.out" is its own conf key, so it needs its own entries.
      {at::kLong, "mm"},
      {at::kInt, "mm"},
      {at::kShort, "mm"},
      {at::kByte, "mm"},
      {at::kLong, "mm.out"},
      {at::kInt, "mm.out"},
      {at::kShort, "mm.out"},
      {at::kByte, "mm.out"},
  };

  for (const OpDtypeGap& gap : kGaps) {
    // dtype first, so the common path costs one enum compare per entry before
    // any strcmp -- this runs on every FlagGems dispatch.
    if (dtype == gap.dtype && std::strcmp(op_name, gap.op_name) == 0) {
      return true;
    }
  }
  return false;
#else
  // No vendor route is known to have a gap that is specific to one op and
  // dtype in this way. See the dtype-wide case above for what MUSA records
  // instead.
  (void)op_name;
  return false;
#endif
}

int64_t TritonMinDotContractExtent() {
#if defined(FLAGOS_ACCEL_METAX)
  // 16, measured. FlagTree's adaptive block size (AABS) shrinks a block that
  // is wider than the tensor -- `if bs > ts: updated_bs = next_power_of_2(ts)`
  // in triton/runtime/adjust_kernel_param.py -- so FlagGems' BLOCK_K of 32
  // becomes next_power_of_2(K) for a small K. Its `metax` arm then restores M
  // and N to >= 16 and leaves K where the shrink put it, while the `iluvatar`
  // and `sunrise` arms next to it restore all three. The dot intrinsic then
  // refuses the tile:
  //
  //   triton/spec/metax/triton/language/semantic.py: in dot
  //     assert lhs.shape[-2].value >= min_dot_size[0] and ...
  //   AssertionError: Input shapes should have M >= 1, N >= 1 and K >= 16
  //
  // Measured on a MetaX C550 (torch 2.10.0+metax3.8.1.0, flag_gems 5.4.0),
  // calling FlagGems' mm directly with a cold FLAGGEMS_CACHE_DIR:
  //
  //   (4, 4) x (4, 4)    CompilationError, the assertion above
  //   (4, 8) x (8, 4)    CompilationError, the assertion above
  //   (4, 16) x (16, 4)  OK, max_abs 4.4e-07 against a CPU fp64 reference
  //   (4, 32) x (32, 4)  OK, max_abs 1.8e-06
  //
  // The floor is 16 and not `next_power_of_2(K) < 16`: AABS only shrinks when
  // the block exceeds the extent, so a config that already carries a small
  // BLOCK_K trips the same assertion at a K the power-of-two rule would pass
  // (the thead measurement in scripts/codegen/gen_vendor_confs.py records
  // K = 4/8/8 failing while 8/8/15 passes -- which config the tuner draws
  // decides it). Padding every K below 16 costs a copy of a tensor with fewer
  // than 16 columns and removes the dependence on that draw.
  //
  // This is a shim for a vendor-wheel gap, and it has an expiry: FlagGems pads
  // K itself from 5.6, and FlagTree's metax arm should restore K the way its
  // iluvatar arm does. When either lands in the pinned cohort, delete this
  // value and the padding that reads it.
  return 16;
#else
  // No other platform that compiles a soft-lowp decode is known to impose a
  // contracted-extent floor. Hygon DCU's FlagTree arm also restores only M and
  // N, but its HIP dot intrinsic has not been measured to refuse a small K --
  // if its lane ever raises the assertion above, that is this same shim with
  // its own measurement, not a reason to widen this one on a guess.
  return 0;
#endif
}

int64_t MatmulContractDim(const at::Tensor& operand, bool is_rhs) {
  return is_rhs ? operand.size(-2) : operand.size(-1);
}

at::Tensor PadMatmulContractDim(const at::Tensor& operand, bool is_rhs,
                                int64_t target) {
  auto shape = operand.sizes().vec();
  const auto axis = is_rhs ? shape.size() - 2 : shape.size() - 1;
  const auto extent = shape[axis];
  shape[axis] = target;
  auto padded = at::empty(shape, operand.options());
  padded.zero_();
  padded.narrow(static_cast<int64_t>(axis), 0, extent).copy_(operand);
  return padded;
}

bool PadMatmulPairForDot(at::Tensor& lhs, at::Tensor& rhs) {
  const auto floor = TritonMinDotContractExtent();
  if (floor <= 0) {
    return false;
  }
  // Early out on the path that matters: a real GEMM has a contracted extent in
  // the hundreds or thousands and pays one integer compare here.
  const auto extent = MatmulContractDim(lhs, false);
  if (extent >= floor || extent == 0) {
    return false;
  }
  if (MatmulContractDim(rhs, true) != extent) {
    // Mismatched operands are the caller's error, and the kernel below reports
    // it with the shapes in the message. Padding them to a common target here
    // would turn that into a wrong answer.
    return false;
  }
  lhs = PadMatmulContractDim(lhs, false, floor);
  rhs = PadMatmulContractDim(rhs, true, floor);
  return true;
}

bool TryEmptyMatmul(const at::Tensor& lhs, const at::Tensor& rhs,
                    int64_t rank, at::Tensor& result) {
  if (lhs.numel() != 0 && rhs.numel() != 0) return false;
  TORCH_CHECK(lhs.dim() == rank && rhs.dim() == rank,
              "expected two rank-", rank, " matmul operands");
  TORCH_CHECK(lhs.device() == rhs.device(), "matmul operands must use the same device");
  TORCH_CHECK(lhs.scalar_type() == rhs.scalar_type(),
              "matmul operands must have the same dtype");
  TORCH_CHECK(lhs.size(-1) == rhs.size(-2),
              "matmul contraction dimensions do not match");
  TORCH_CHECK(rank == 2 || lhs.size(0) == rhs.size(0),
              "bmm batch dimensions do not match");

  auto shape = lhs.sizes().vec();
  shape.back() = rhs.size(-1);
  result = at::empty(shape, lhs.options());
  if (result.numel() > 0) result.zero_();
  return true;
}

bool TryNativeHostBmmOuterProduct(const at::Tensor& lhs, const at::Tensor& rhs,
                                  at::Tensor& result) {
#if defined(FLAGOS_NATIVE_HOSTS) && defined(FLAGOS_ACCEL_METAX)
  if (lhs.dim() != 3 || rhs.dim() != 3 || lhs.size(0) != rhs.size(0) ||
      lhs.size(2) != 1 || rhs.size(1) != 1 || lhs.device() != rhs.device() ||
      lhs.scalar_type() != at::kFloat || rhs.scalar_type() != at::kFloat) {
    return false;
  }
  auto operand = [](const at::Tensor& input) {
    if (!at::globalContext().allowTF32CuBLAS()) return input;
    auto bits = input.contiguous().view(at::kInt);
    auto options = input.options().dtype(at::kInt);
    auto shift = at::full(at::IntArrayRef{}, 13, options);
    auto bias = at::full(at::IntArrayRef{}, 4095, options);
    auto low_bit = at::bitwise_and(at::bitwise_right_shift(bits, shift), 1);
    auto rounded = at::bitwise_and(at::add(at::add(bits, bias), low_bit), -8192).view(at::kFloat);
    auto special = at::ge(at::bitwise_and(bits, 0x7fffffff), 0x7f800000);
    return at::where(special, input, rounded);
  };
  result = at::mul(operand(lhs), operand(rhs));
  return true;
#else
  (void)lhs;
  (void)rhs;
  (void)result;
  return false;
#endif
}

void SetStrictHostRoutes(bool enabled) {
  strict_host_routes.store(enabled, std::memory_order_relaxed);
}

bool CpuFallbackAllowed() {
  return !strict_host_routes.load(std::memory_order_relaxed);
}

bool FlagGemsMatmulNeedsHost(at::ScalarType dtype) {
#if defined(FLAGOS_ACCEL_METAX)
  // See the declaration in common.h for the compiler assertion, the silent
  // wrong answer the same call produces on a development host, and why neither
  // FlagGemsRejectsOpDtype nor FlagGemsRejectsDtype nor a conf entry can state
  // this gap. float32 and below are correct on this route and stay on it.
  return dtype == at::kDouble;
#else
  (void)dtype;
  return false;
#endif
}

at::Tensor MatmulOnHost(const at::Tensor& lhs, const at::Tensor& rhs) {
  TORCH_CHECK(CpuFallbackAllowed(),
              "native FlagGems hosts do not permit CPU fallback: aten::mm (host matmul escape)");
  return at::mm(lhs.cpu(), rhs.cpu()).to(lhs.device());
}

} // namespace at::native::flagos
