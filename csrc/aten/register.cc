// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "empty.h"
#include "strided_ops.h"
#include "copy_ops.h"
#include "copy_dispatcher.h"
#include "set_ops.h"
#include "contiguous_ops.h"
#include "fallback.h"

// Generated dispatcher headers
#include "generated/ops.h"

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/_conj_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/Functions.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/div_native.h>
#include <ATen/SDPBackend.h>
#include <torch/library.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include "common.h"
#include "index_bounds.h"
#include "backends/soft_lowp/format.h"
#include "backends/soft_lowp/ops.h"
// TsingMicro's RNG kernels are claimed directly below rather than through a
// generated list, so their declarations have to be visible here.
#if defined(USE_TSINGMICRO)
#include "backends/tsingmicro/rng.h"
#endif
#include "runtime/allocator/caching_device_allocator.h"
#include "dispatcher.h"

#include <c10/core/impl/LocalDispatchKeySet.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

// True when a convolution cannot be served by either a native kernel or a
// FlagGems route, so it has to take the CPU crossing from
// FlagosWrapperConvolutionOverrideable below.
//
// Stated here, next to the wrapper, because the wrapper is defined well before
// the registration arm and a macro that arm defined afterwards would not reach
// it. It is derived from the accelerator discriminator alone, which CMake sets
// for every platform (csrc/CMakeLists.txt), so it needs no other build state.
//
// The per-platform reasoning, all measured on the CUDA-ABI lanes:
//   * cuda, dcu  -- their backends_<platform>.conf routes conv to `cuda`, and a
//     FlagGems-primary build compiles no boxed kernels, so the route resolves
//     to nothing and `convolution_overrideable` falls to ATen's raising
//     CompositeExplicitAutograd stub. FlagGems has no conv kernel to route to on
//     either platform (register_flaggems_cuda.inc / _dcu.inc name none).
//   * metax, ppu -- the conf routes conv to FlagGems, so the conf-derived list
//     does claim `convolution_overrideable` and only the forward is served.
//     Nothing claims `convolution_backward_overrideable` that way, so the
//     backward still needs the crossing (the same shape as the Iluvatar
//     exception).
//
// BPU, Kunlunxin, TsingMicro and Iluvatar select the wrapper through their own
// USE_* macros in the guard below, so they are deliberately not listed again.
#if defined(FLAGOS_ACCEL_CUDA) || defined(FLAGOS_ACCEL_DCU) || \
    defined(FLAGOS_ACCEL_METAX) || defined(FLAGOS_ACCEL_PPU)
#define FLAGOS_CONV_NEEDS_CPU_CROSSING 1
#endif

// Forward declarations for the Ascend matmul kernels (csrc/aten/backends/ascend/matmul.cc).
// That file is only compiled when USE_ASCEND is on, so BOTH declarations must be
// guarded the same way: a .so links fine with an undefined symbol and only fails
// at dlopen, so an unguarded reference here builds a CUDA wheel that dies on
// `import torch_fl` with "undefined symbol: ...MatmulKernelAscend...".
#if defined(USE_ASCEND)
namespace at::native::flagos {
  at::Tensor MatmulKernelAscend(const at::Tensor& self, const at::Tensor& other);
  std::tuple<at::Tensor, at::Tensor> MatmulBackwardKernelAscend(
      const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
      ::std::array<bool, 2> mask);
}
#endif

namespace at::flagos {

namespace {

at::Tensor WrapperEmptyMemoryFormat(
    c10::IntArrayRef size,
    ::std::optional<at::ScalarType> dtype_opt,
    ::std::optional<at::Layout> layout_opt,
    ::std::optional<at::Device> device_opt,
    ::std::optional<bool> pin_memory_opt,
    ::std::optional<at::MemoryFormat> memory_format_opt) {
  return at::native::flagos::empty_memory_format(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

at::Tensor WrapperEmptyStrided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    ::std::optional<at::ScalarType> dtype_opt,
    ::std::optional<at::Layout> layout_opt,
    ::std::optional<at::Device> device_opt,
    ::std::optional<bool> pin_memory_opt) {
  return at::native::flagos::empty_strided(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor WrapperAsStrided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    ::std::optional<c10::SymInt> storage_offset) {
  return at::native::flagos::as_strided(self, size, stride, storage_offset);
}

const at::Tensor& WrapperResize(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::flagos::resize_(self, size, memory_format);
}

at::Tensor WrapperReshapeAlias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::flagos::_reshape_alias(self, size, stride);
}

at::Tensor WrapperCopyFrom(
    const at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  return at::native::flagos::_copy_from(self, src, non_blocking);
}

at::Tensor WrapperCopyFromAndResize(
    const at::Tensor& self, const at::Tensor& dst) {
  return at::native::flagos::_copy_from_and_resize(self, dst);
}

at::Tensor& WrapperCopy_(
    at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  at::native::flagos::_copy_from(src, self, non_blocking);
  return self;
}

at::Scalar WrapperLocalScalarDense(const at::Tensor& self) {
  return at::native::flagos::_local_scalar_dense(self);
}

at::Tensor& WrapperSetSourceTensor(
    at::Tensor& self, const at::Tensor& source) {
  return at::native::flagos::set_source_Tensor_(self, source);
}

at::Tensor& WrapperSetSourceStorage(at::Tensor& self, at::Storage source) {
  return at::native::flagos::set_source_Storage_(self, source);
}

at::Tensor& WrapperSetSourceStorageOffset(
    at::Tensor& self,
    at::Storage source,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::native::flagos::set_source_Storage_storage_offset_(
      self, source, storage_offset, size, stride);
}

at::Tensor WrapperView(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::flagos::view(self, size);
}

at::Tensor WrapperExpand(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit) {
  return at::native::flagos::expand(self, size, implicit);
}

at::Tensor WrapperNarrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length) {
  return at::native::flagos::narrow(self, dim, start, length);
}

at::Tensor WrapperUnfold(const at::Tensor& self, int64_t dimension, int64_t size, int64_t step) {
  return at::native::flagos::unfold(self, dimension, size, step);
}

at::Tensor WrapperContiguous(
    const at::Tensor& self, at::MemoryFormat memory_format) {
  return at::native::flagos::contiguous(self, memory_format);
}

at::Tensor WrapperClone(
    const at::Tensor& self,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::flagos::clone(self, memory_format);
}

at::Tensor WrapperToCopy(
    const at::Tensor& self,
    ::std::optional<at::ScalarType> dtype,
    ::std::optional<at::Layout> layout,
    ::std::optional<at::Device> device,
    ::std::optional<bool> pin_memory,
    bool non_blocking,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::flagos::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

// index_put_ / _index_put_impl_: scatter on the CPU and copy back.
//
// This is the last-resort path, not the intended one. `Tensor?[]` has no
// IValueToPython conversion, so neither op can be code-generated, and they are
// in the codegen's MANUAL_REGISTERED_OPS. What covers them is
// `_patch_flaggems_index_put` in torch_fl/flagos/__init__.py, which claims
// index_put_ on PrivateUse1 through torch.library and runs FlagGems' Triton
// kernel in place of this round-trip -- see the measurements in its docstring
// (68.6 ms -> 0.29 ms on one wide call) and docs/reference/operator-support.md.
//
// Two ways to land here anyway, and both are worth knowing about when a
// profile shows host traffic on an indexed write:
//   * no flag_gems, or a flag_gems without `index_put_` (the patch is
//     best-effort and returns early), or
//   * a FlagGems kernel call that raised, which the patch falls back to this
//     function through `_flagos_index_put_roundtrip`.
// The fallback is deliberate: both produce the same tensor, so the round-trip
// costs time and nothing else, and a missing kernel must not be an error.
at::Tensor& WrapperIndexPut_(
    at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate) {
  at::Tensor self_cpu = self.cpu();
  at::Tensor values_cpu = values.cpu();
  c10::List<::std::optional<at::Tensor>> indices_cpu;
  for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
    auto opt = indices.get(i);
    if (opt.has_value() && opt->defined()) {
      indices_cpu.push_back(opt->cpu());
    } else {
      indices_cpu.push_back(std::nullopt);
    }
  }
  at::index_put_(self_cpu, indices_cpu, values_cpu, accumulate);
  self.copy_(self_cpu);
  return self;
}

at::Tensor& WrapperIndexPutImpl_(
    at::Tensor& self,
    const c10::List<::std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    bool accumulate,
    bool /*unsafe*/) {
  return WrapperIndexPut_(self, indices, values, accumulate);
}

void WrapperRecordStream(at::Tensor& self, at::Stream s) {
  if (!c10::flagos::CachingDeviceAllocator::is_enabled()) {
    // No-op when caching allocator is disabled.
    return;
  }
  auto* alloc = c10::flagos::GetCachingAllocator();
  if (!alloc) {
    // No-op when caching allocator is not available on this platform.
    return;
  }
  // Convert at::Stream to flagos Stream_t.
  // The stream id encodes the underlying device stream pointer.
  Stream_t stream = reinterpret_cast<Stream_t>(s.id());
  alloc->record_stream(self.storage().data_ptr(), stream);
}

// _fused_sdp_choice: tells PyTorch's scaled_dot_product_attention which fused
// backend to use for PrivateUse1 tensors. Returning efficient_attention (2)
// routes to _scaled_dot_product_efficient_attention, which has an Ascend
// aclnnFlashAttentionScore kernel. Without this, SDPA falls back to the math
// decomposition path (_safe_softmax etc.), which is slower and needs many more
// ops registered.
int64_t WrapperFusedSdpChoice(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  return static_cast<int64_t>(at::SDPBackend::efficient_attention);
}

// Convolution on a device whose vendor SDK exposes no conv operator.
//
// aten::convolution dispatches PrivateUse1 to convolution_overrideable, and the
// only other kernel registered for that op is a CompositeExplicitAutograd stub
// that raises NotImplementedError. So the boxed cpu_fallback cannot help here:
// it moves the arguments to CPU and redispatches the *same* op, which lands back
// on the stub. These wrappers cross to CPU and then call at::convolution -- a
// different op, and the one that actually has a CPU kernel -- and move the
// result back.
//
// Shared by every platform in that position: BPU (no per-op kernels at all),
// Kunlunxin (the xdnn op library covers only the runtime-conformance set) and
// TsingMicro (FlagGems-primary -- its registration is the conf-derived FlagGems
// list, which names no conv). The three had byte-identical copies of this; one
// definition is the same code with one less place to drift. A vendor that later
// gains a native conv should register that instead, not fork this.
//
// Iluvatar joins for the backward only. Its `convolution_overrideable` is
// FlagGems-claimed, its forward reaches ixdnn, and nothing claims
// `convolution_backward_overrideable` -- no native backward exists in
// codegen_iluvatar.py -- so the backward of every conv hit
// "NotImplementedError" (measured 2026-10-20 on the CI lane,
// test_conv1d_dispatch.py::TestConv1dBackward). The USE_ILUVATAR arm registers
// the backward wrapper below and deliberately not the forward one.
//
// The CUDA-ABI platforms reach this through FLAGOS_ACCEL_*, not a USE_* macro:
// their USE_* macro is empty (they re-tag a vendor CUDA build rather than own
// one), and since CUDA boxing became opt-in (#520) their default wheel is
// FlagGems-primary, which compiles the conf-derived FlagGems list. On cuda and
// dcu that list names no conv, so the forward needs this wrapper too; metax and
// ppu do route conv to FlagGems and only need it when that route is absent.
// FLAGOS_CONV_NEEDS_CPU_CROSSING below is the single place that decides.
#if defined(USE_BPU) || defined(USE_KUNLUNXIN) || defined(USE_TSINGMICRO) || \
    defined(USE_ILUVATAR) || defined(FLAGOS_CONV_NEEDS_CPU_CROSSING)
at::Tensor FlagosWrapperConvolutionOverrideable(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ::std::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups) {
  auto out = at::convolution_symint(
      input.cpu(),
      weight.cpu(),
      bias.has_value() && bias->defined()
          ? ::std::optional<at::Tensor>(bias->cpu())
          : ::std::nullopt,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups);
  return out.to(input.device());
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
FlagosWrapperConvolutionBackwardOverrideable(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    bool transposed,
    c10::SymIntArrayRef output_padding,
    c10::SymInt groups,
    ::std::array<bool, 3> output_mask) {
  // convolution_backward wants the forward bias *sizes*, not the bias, and only
  // to shape grad_bias -- which is a plain sum over the non-channel dims, so the
  // channel count is all it needs.
  ::std::optional<c10::SymIntArrayRef> bias_sizes = ::std::nullopt;
  c10::SymInt out_channels =
      transposed ? weight.sym_size(1) * groups : weight.sym_size(0);
  ::std::vector<c10::SymInt> bias_shape{out_channels};
  if (output_mask[2]) {
    bias_sizes = c10::SymIntArrayRef(bias_shape);
  }

  auto out = at::convolution_backward_symint(
      grad_output.cpu(),
      input.cpu(),
      weight.cpu(),
      bias_sizes,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output_mask);

  auto to_dev = [&](const at::Tensor& t) {
    return t.defined() ? t.to(input.device()) : t;
  };
  return ::std::make_tuple(
      to_dev(::std::get<0>(out)),
      to_dev(::std::get<1>(out)),
      to_dev(::std::get<2>(out)));
}
#endif // USE_BPU || USE_KUNLUNXIN || USE_TSINGMICRO || USE_ILUVATAR

// ============================================================
// Generated wrappers for 71 CUDA operators
// ============================================================
#define FLAGOS_GEN_WRAPPERS
#include "generated/register.inc"
#undef FLAGOS_GEN_WRAPPERS

// matmul: intercept aten::matmul at PrivateUse1 for the Ascend backend so it
// matmul: intercept aten::matmul at PrivateUse1 for the Ascend backend so it
// routes to aclnnMatmul directly instead of decomposing via
// CompositeImplicitAutograd into mm + bmm + view. Non-Ascend backends (MetaX
// etc.) call at::native::matmul directly so PyTorch's composite decomposition
// runs and mm/bmm reach the appropriate backend kernels.
static at::Tensor WrapperMatmul(
    const at::Tensor& self, const at::Tensor& other) {
#if defined(USE_ASCEND)
  // aten::matmul is CompositeImplicitAutograd: normally it decomposes into
  // mm/bmm/view, and autograd records the backward through those sub-ops. Taking
  // the fused aclnnMatmul kernel stops that decomposition, so autograd binds the
  // op's real derivative, aten::matmul_backward -- which is registered for
  // PrivateUse1 below (WrapperMatmulBackward). The generated
  // AutogradPrivateUse1 kernel (csrc/aten/generated/variable_type.cc) builds the
  // MatmulBackward0 node and redispatches here, so the fused path is used for
  // training as well as inference.
  //
  // The runtime GetBackendForOp check still matters on an Ascend build: the conf
  // can route matmul elsewhere. But it must sit INSIDE the #if -- a runtime
  // branch does not remove the link-time reference, and MatmulKernelAscend is
  // only compiled when USE_ASCEND is on.
  if (at::native::flagos::GetBackendForOp("matmul") ==
      at::native::flagos::Backend::kAscend) {
    return at::native::flagos::MatmulKernelAscend(self, other);
  }
#endif
  // Fall through to the composite decomposition (mm/bmm/view) by calling the
  // CompositeImplicitAutograd implementation directly. This avoids re-entering
  // WrapperMatmul (no recursion) while letting the decomposed sub-ops dispatch
  // normally to their PrivateUse1 kernels, which have working autograd. Used by
  // non-Ascend backends, which have no fused matmul kernel.
  return at::native::matmul(self, other);
}

// matmul_backward: the derivative aten::matmul binds to once a backend owns the
// forward op (see WrapperMatmul). Only Ascend has a fused kernel; other backends
// never reach here, since without a concrete matmul kernel autograd keeps
// recording the mm/bmm decomposition instead.
#if defined(USE_ASCEND)
static std::tuple<at::Tensor, at::Tensor> WrapperMatmulBackward(
    const at::Tensor& grad, const at::Tensor& self, const at::Tensor& other,
    ::std::array<bool, 2> mask) {
  return at::native::flagos::MatmulBackwardKernelAscend(
      grad, self, other, mask);
}
#endif

#if defined(USE_ILUVATAR)
// _conj is the lazy-view half of aten::conj: an alias of self carrying the
// Conjugate bit, storage untouched. csrc/aten/strided_ops.cc implements it
// through the GENERATED FlagGems dispatcher, which only answers a `flaggems`
// route; iluvatar's conf routes `_conj` to `iluvatar` (FlagGems' `_conj` is an
// eager kernel), so that slot is unpopulated and the call raised "backend not
// registered". Answer it here, the way the conf says it runs.
//
// There is no cpu_fallback alternative: `_conj` is a view, and ATen's ViewOp
// fallback refuses every view ("View operators don't support falling back to
// run on the CPU, since the tensor's storage cannot be shared across devices").
// Measured on u180, before these registrations existed: `torch.conj` on a
// PrivateUse1 tensor raised, which took down every path that resolves a
// Conjugate bit (copy_, clone, contiguous, resolve_conj, arithmetic) -- the
// tests/integration/test_math_bits_contract.py group.
//
// `_conj_copy.out` is the materializing half of the same pair and keeps the
// `none` route with the other out= copies: `at::conj_physical` writes the same
// bytes with the bit cleared. It is registered here rather than left to the
// shared cpu_fallback so that the view and its materialization are answered by
// one platform, so a fallback contract change cannot split them. Both stay out
// of the FLAGOS_ILUVATAR_KERNEL-gated native .inc because a FlagGems-only
// build needs this answer too.
//
// Both are ordinary (non-CatchAll, non-Autograd) PrivateUse1 kernels. The
// registration makes the op a genuine PrivateUse1 op, so a CatchAll entry would
// simply never be reached: `torch.conj(z)` on PrivateUse1 resolves its autograd
// group first, and the boxed `cpu_fallback` object -- which every unclaimed op
// gets from the CatchAll arm near the bottom of this file -- would carry the
// inner `_conj` to CPU with a device mismatch.
at::Tensor FlagosWrapperPrivConj(const at::Tensor& self) {
  return at::native::_conj(self);
}

at::Tensor& FlagosWrapperPrivConjCopyOut(
    const at::Tensor& self, at::Tensor& out) {
  out.copy_(at::conj_physical(self));
  return out;
}

// view_as_real is the other view FlagGems cannot supply here, and it is the
// op flag_gems' complex elementwise path decomposes both operands with
// (flag_gems/ops/add.py). Its route was `none`, which a view cannot take: the
// ViewOp key refuses every view, so the call warned and returned zero-filled
// storage. Measured on u180, `torch.view_as_real` read back exactly once
// answered from ATen here, for f64, f32, complex64 and complex128 sources --
// while `complex128 + 0` still returned zeros, which is flag_gems' complex add
// path and not this view (see the view_as_real entry in NATIVE_TRITON_GAPS).
// Same contract as _conj above: metadata only, no kernel, dtype or device.
at::Tensor FlagosWrapperPrivViewAsReal(const at::Tensor& self) {
  return at::native::view_as_real(self);
}

// add/div are the consumer ops the Conjugate contract measures in
// tests/integration/test_math_bits_contract.py. FlagGems' complex path
// decomposes both operands with view_as_real and then runs its own kernel,
// and on iluvatar that kernel returns zeros for complex128 in `add` and
// raises for complex64 in `div`, so the conf routes these overloads here
// instead (see the entries in NATIVE_TRITON_GAPS).
//
// The conf names these three in-place schemas, not add.Tensor, for a
// registration reason: an m.impl on the PrivateUse1 key belongs to the schema's
// own C++ signature, so a four-argument out= kernel under add.Tensor aborts at
// load with "Mismatch in kernel C++ signatures" (measured on u180). ATen's
// `add` lowers the out-of-place spelling to add_.Tensor, so `z + t`, `z + 1`
// and the conjugated `z + 0` reach these kernels through that lowering.
//
// They return Tensor& because ATen's add_.Tensor family (CPU, meta) does, and
// the registrar matches signatures exactly. add_ calls the operator
// (at::add_out), not a lower-level kernel: ATen registers the complex out=
// kernel through its structured meta/decomposition path, which only a call
// through at:: picks up. `at::native::add_out` does not exist in this build
// (measured: "not a member of at::native").
at::Tensor& FlagosWrapperPrivAddTensorInplace(
    at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return at::add_out(self, self, other, alpha);
}

at::Tensor& FlagosWrapperPrivAddScalarInplace(
    at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  return at::add_out(self, self, other, alpha);
}

at::Tensor& FlagosWrapperPrivDivInplace(
    at::Tensor& self, const at::Tensor& other) {
  // ATen's complex div kernel writes its destination before it reads its
  // sources, so the destination must not alias a source. `self` is a source,
  // and the spelling `x.div_(y)` hands it the operand itself: measured on
  // u180, the quotient of complex128 by a complex128 divisor came back as the
  // unchanged operand, and complex64 came back as zeros. Divide on the host
  // and copy the result back.
  //
  // The host round trip is the cpu_fallback shape, but the fallback writes to
  // a host copy and never to `self` -- which is the defect this route exists
  // to fix. `at::reciprocal` would sidestep the kernel, but FlagGems has no
  // complex route for it, and computing the reciprocal in Triton for a
  // 2-element divide is the worse trade.
  at::Tensor source = self;
  if (self.is_conj()) {
    source = self.resolve_conj();
  }
  const auto cpu_self = source.to(c10::kCPU);
  const auto cpu_other = other.to(c10::kCPU);
  const auto cpu_out = at::div(cpu_self, cpu_other);
  self.copy_(cpu_out);
  return self;
}
#endif

bool HasCompatibleShallowCopyType(
    const at::Tensor& self, const at::Tensor& from) {
  const auto self_keys = self.key_set();
  const auto from_keys = from.key_set();
  const auto is_dense = [](c10::DispatchKeySet keys) {
    return keys.has(c10::DispatchKey::CPU) ||
        keys.has(c10::DispatchKey::PrivateUse1);
  };
  return self_keys == from_keys || (is_dense(self_keys) && is_dense(from_keys));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, CatchAll, m) {
  m.impl(
      "_has_compatible_shallow_copy_type",
      TORCH_FN(HasCompatibleShallowCopyType));
}

// Register basic operators for PrivateUse1 dispatch key
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", WrapperEmptyMemoryFormat);
  m.impl("empty_strided", WrapperEmptyStrided);
  m.impl("as_strided", WrapperAsStrided);
  m.impl("resize_", WrapperResize);
  m.impl("_reshape_alias", WrapperReshapeAlias);
  m.impl("_copy_from", WrapperCopyFrom);
  m.impl("_copy_from_and_resize", WrapperCopyFromAndResize);
  m.impl("copy_", WrapperCopy_);
  m.impl("_local_scalar_dense", WrapperLocalScalarDense);
  m.impl("set_.source_Tensor", WrapperSetSourceTensor);
  m.impl("set_.source_Storage", WrapperSetSourceStorage);
  m.impl(
      "set_.source_Storage_storage_offset", WrapperSetSourceStorageOffset);
  m.impl("view", WrapperView);
  m.impl("expand", WrapperExpand);
  m.impl("narrow", WrapperNarrow);
  m.impl("unfold", WrapperUnfold);
  m.impl("contiguous", WrapperContiguous);
  m.impl("clone", WrapperClone);
  m.impl("_to_copy", WrapperToCopy);
  m.impl("index_put_", WrapperIndexPut_);
  m.impl("_index_put_impl_", WrapperIndexPutImpl_);
  m.impl("record_stream", WrapperRecordStream);
  m.impl("_fused_sdp_choice", WrapperFusedSdpChoice);
  // matmul: on Ascend, claim the fused aclnnMatmul kernel here on plain
  // PrivateUse1. The generated AutogradPrivateUse1 kernel
  // (csrc/aten/generated/variable_type.cc) intercepts above this, builds the
  // MatmulBackward0 node and redispatches down to us, so training and inference
  // both take the fused path. Its derivative, matmul_backward, is a normal
  // backend op and is registered right below.
#if defined(USE_ASCEND)
  m.impl("matmul", WrapperMatmul);
  m.impl("matmul_backward", WrapperMatmulBackward);
#endif

  // ============================================================
  // Generated m.impl registrations for the generated operators
  // ============================================================
  // GCU registers only its topsaten coverage set. Claiming an op on PrivateUse1
  // without a kernel behind it turns into the dispatcher's "backend not
  // registered" error, whereas leaving it unregistered reaches the cpu_fallback
  // below. FlagGems Python kernels are compiled alongside GCU, but only wrappers
  // in this coverage set may select their kFlagGems dispatcher slot.
  //
  // MUSA is the same story: musa_register.inc lists exactly the ops with a mudnn
  // kernel behind them, and no CUDA boxing kernels are compiled in
  // (musa_runtime, not cudart), so uncovered ops must fall through rather than
  // be claimed. The two convolution `*_overrideable` ops are the exception that
  // proves the rule -- ATen's default for them raises instead of being boxable,
  // so they get real kernels in backends/musa/mudnn_conv.cc.
  //
  // Ascend joined them: it used to fall through to the full register.inc below,
  // claiming every generated op while owning aclnn kernels for a few hundred.
  // That made `none` in backends_ascend.conf unreachable as a fallback -- the op
  // was registered, so the call reached the dispatcher and raised on the empty
  // slot instead of boxing to CPU. ascend_register.inc lists exactly the aclnn
  // coverage set, generated and handwritten.
  #if defined(USE_GCU)
    #if defined(FLAGOS_GCU_KERNEL)
    #include "backends/gcu/generated/gcu_register.inc"
    #endif
    // Registered whenever the FlagGems Python path is compiled in, not only
    // under FLAGOS_GCU_KERNEL: backends_gcu.conf routes ops to FlagGems by
    // default, so these kFlagGems dispatcher slots must exist or those routes
    // raise "backend not registered". Same contract as the MUSA include below.
    #if defined(FLAGOS_FLAGGEMS_PYTHON)
    #include "backends/gcu/generated/gcu_flaggems_register.inc"
    #endif
  #elif defined(USE_KUNLUNXIN)
    // Kunlunxin owns its ops natively: it registers the xdnn_pytorch coverage
    // set into the kKunlunxin slot rather than claiming every generated op.
    // Claiming an op on PrivateUse1 with no kernel behind it raises "backend
    // not registered" instead of boxing to CPU, so the list has to be exactly
    // the set of ops backends/kunlunxin/generated/kunlunxin_kernels.cc
    // implements -- which is also what gen_vendor_confs.py routes to
    // "kunlunxin"; everything else in the conf says "none" and reaches
    // cpu_fallback.
    #if defined(FLAGOS_KUNLUNXIN_KERNEL)
    #include "backends/kunlunxin/generated/kunlunxin_register.inc"
    #endif
    // The vendor SDK exposes no conv operator, so the *_overrideable ops cannot
    // be generated (see the wrapper block above) -- they are the one pair of
    // ops claimed here without an xdnn kernel, and they take the CPU route.
    m.impl("convolution_overrideable", FlagosWrapperConvolutionOverrideable);
    m.impl("convolution_backward_overrideable",
        FlagosWrapperConvolutionBackwardOverrideable);
  #elif defined(USE_ASCEND)
    #include "backends/ascend/generated/ascend_register.inc"
  #elif defined(USE_MUSA)
    #if defined(FLAGOS_MUSA_KERNEL)
    #include "backends/musa/generated/musa_register.inc"
    #endif
    // Registered unconditionally when FlagGems is compiled in: backends_musa.conf
    // routes ops to FlagGems by default, so these kFlagGems dispatcher slots must
    // exist or those routes raise "backend not registered". Gating them on an
    // opt-in env var was correct only while the default conf was mudnn-only.
    #if defined(FLAGOS_FLAGGEMS_PYTHON)
    #include "backends/musa/generated/musa_flaggems_register.inc"
    #endif
  #elif defined(USE_ILUVATAR)
    // Iluvatar is FlagGems-primary, with a vendor-native slot alongside it: the
    // corex ixdnn/ixblas libraries (the CUDA-named C API) back the ops the conf
    // routes to "iluvatar", and FlagGems stays the default for everything else.
    // The native set is claimed only under FLAGOS_ILUVATAR_KERNEL so a
    // FlagGems-only build still compiles; the two .inc files are disjoint, as
    // both are included into the same PrivateUse1 TORCH_LIBRARY_IMPL body and a
    // duplicate m.impl for one op is a compile error.
    //
    // This arm must exist even when it claimed nothing else -- without it
    // iluvatar would fall to the generic #else below, which includes the full
    // generated register.inc and claims every generated op, making `none` in
    // backends_iluvatar.conf a "backend not registered" raise instead of a
    // cpu_fallback. That is the same defect the Ascend comment above records.
    #if defined(FLAGOS_ILUVATAR_KERNEL)
    #include "backends/iluvatar/generated/iluvatar_register.inc"
    #endif
    // Registered whenever the FlagGems Python path is compiled in, not only
    // under FLAGOS_ILUVATAR_KERNEL: backends_iluvatar.conf routes ops to
    // FlagGems by default, so these kFlagGems dispatcher slots must exist or
    // those routes raise "backend not registered". Same contract as MUSA.
    #if defined(FLAGOS_FLAGGEMS_PYTHON)
    #include "backends/iluvatar/generated/iluvatar_flaggems_register.inc"
    #endif
    // No convolution backward exists on this platform and the boxed fallback
    // cannot reach one (see the shared wrapper above), so every conv backward
    // takes the CPU crossing. Unconditional: the wrapper needs no vendor or
    // FlagGems kernel. Only the backward -- the forward keeps its FlagGems /
    // ixdnn routes.
    m.impl("convolution_backward_overrideable",
        FlagosWrapperConvolutionBackwardOverrideable);
    // See FlagosWrapperPrivConj above. Unconditional like the conv pair
    // above: the metadata kernel needs neither a vendor nor a FlagGems slot, and both
    // op names carry no CPU fallback to reach instead.
    m.impl("_conj", FlagosWrapperPrivConj);
    m.impl("_conj_copy.out", FlagosWrapperPrivConjCopyOut);
    m.impl("view_as_real", FlagosWrapperPrivViewAsReal);
    // See FlagosWrapperPrivAddTensorInplace above: FlagGems' complex kernel is
    // wrong for add (complex128) and raises for div (complex64) here, so these
    // run ATen's kernels instead.
    m.impl("add_.Tensor", FlagosWrapperPrivAddTensorInplace);
    m.impl("add_.Scalar", FlagosWrapperPrivAddScalarInplace);
    m.impl("div_.Tensor", FlagosWrapperPrivDivInplace);
  #elif defined(USE_BPU)
    // BPU registers no compute ops. The BPU's unit of execution is a whole
    // compiled graph (a .hbm produced by hbdk4), so there is no per-operator
    // kernel to claim -- and claiming an op on PrivateUse1 without a kernel
    // behind it raises "backend not registered" instead of falling back.
    // Leaving the list out routes every op to cpu_fallback below; acceleration
    // comes from the torch.compile backend in torch_fl/backends/bpu/.
    //
    // The *_overrideable ops are the exception, and the generic fallback cannot
    // serve them (see the wrapper block above).
    m.impl("convolution_overrideable", FlagosWrapperConvolutionOverrideable);
    m.impl("convolution_backward_overrideable",
        FlagosWrapperConvolutionBackwardOverrideable);
  #elif defined(USE_MPS)
    // Apple Metal. Per the design doc
    // (docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md,
    // section 6) the backend is MPSGraph-first: kernels live under
    // csrc/aten/backends/mps/, and each op is claimed only once its kernel
    // exists. Claiming an op without a kernel behind it raises "backend not
    // registered" instead of falling back, so this include is exactly the
    // coverage set the operator layer implements -- currently the `unary`
    // category (stage 1). Every other op stays unregistered and reaches
    // cpu_fallback.
    //
    // This is a native backend: it does NOT re-tag onto DispatchKey::MPS or
    // CUDA and does not build the CUDA boxing kernels (FLAGOS_BUILD_BOXING is
    // pinned off in cmake/flagos_platforms.json).
    #include "backends/mps/generated/mps_register.inc"
  #else
  // CUDA-ABI boxing platforms: cuda, metax, dcu, ppu and tsingmicro all share
  // this arm. With FLAGOS_BUILD_BOXING=ON the wheel carries the generated CUDA
  // boxing kernels (generated/cuda_kernels.cc), so every generated op may be
  // claimed here and a `cuda` route lands on a real kernel.
  //
  // Boxing is off by default for all five now (cmake/flagos_platforms.json).
  // Those wheels link no CUDA kernels and no vendor torch, so claiming a
  // `cuda`-routed op would raise "backend not registered" instead of reaching
  // the cpu_fallback arm below. Register only the ops the platform's conf routes
  // to a FlagGems backend, from the generated per-platform list; every other op
  // stays unregistered and falls through to cpu_fallback. Boxing remains an
  // explicit opt-in (-DFLAGOS_BUILD_BOXING=ON / FLAGOS_BUILD_BOXING=1).
  #if defined(FLAGOS_BOXING_BUILD)
  #define FLAGOS_GEN_IMPLS
  #include "generated/register.inc"
  #undef FLAGOS_GEN_IMPLS
  #elif defined(FLAGOS_FLAGGEMS_PYTHON)
  #if defined(FLAGOS_ACCEL_CUDA)
  #include "generated/register_flaggems_cuda.inc"
  #elif defined(FLAGOS_ACCEL_METAX)
  #include "generated/register_flaggems_metax.inc"
  #elif defined(FLAGOS_ACCEL_DCU)
  #include "generated/register_flaggems_dcu.inc"
  #elif defined(FLAGOS_ACCEL_PPU)
  #include "generated/register_flaggems_ppu.inc"
  #elif defined(FLAGOS_ACCEL_TSINGMICRO)
  #include "generated/register_flaggems_tsingmicro.inc"
  #endif
  // The *_overrideable conv ops are the exception the generic cpu_fallback
  // cannot serve (see the wrapper comment above): ATen's only other kernel for
  // them raises, and a fallback re-dispatches the *same* op, so it lands back on
  // that raise. The condition differs per platform because what the generated
  // list above claims differs:
  //
  //   * TsingMicro: FlagGems-primary with no vendor op library, so the list is
  //     its whole claim -- and it names no conv, because the conf it derives
  //     from routes none.
  //   * cuda, dcu: their conf routes conv to `cuda`, and this arm only runs with
  //     boxing off (the FLAGOS_BOXING_BUILD arm above claims everything
  //     already), so that route resolves to nothing. The conf-derived list names
  //     no conv for either, so both ops need the crossing.
  //   * metax, ppu: the conf routes conv to FlagGems and the list does claim
  //     convolution_overrideable, so only the backward needs it. Claiming the
  //     forward too would duplicate the m.impl the list already emitted into
  //     this same body, which is a compile error.
  //
  // Same exception BPU and Kunlunxin make, and the same shared wrapper serves
  // all of them.
  #if defined(FLAGOS_ACCEL_TSINGMICRO) || defined(FLAGOS_ACCEL_CUDA) || \
      defined(FLAGOS_ACCEL_DCU)
  m.impl("convolution_overrideable", FlagosWrapperConvolutionOverrideable);
  m.impl("convolution_backward_overrideable",
      FlagosWrapperConvolutionBackwardOverrideable);
  #elif defined(FLAGOS_ACCEL_METAX) || defined(FLAGOS_ACCEL_PPU)
  // metax/ppu already claim the forward through the generated list above.
  m.impl("convolution_backward_overrideable",
      FlagosWrapperConvolutionBackwardOverrideable);
  #endif
  // RNG: the same exception as the convolution wrappers above. TsingMicro has
  // no random-number unit and its conf names no RNG op, so these reached the
  // generic cpu_fallback -- which redispatchs randn.generator on the CPU and
  // trips at::native's CPU-generator assertion against a flagos generator
  // ("Expected a 'cpu' device type for generator but found 'flagos'"). The
  // kernels live in backends/tsingmicro/rng.cc (host draw, device copy) and
  // consume c10::flagos::ReserveSeed, so torch.flagos.manual_seed and the
  // get/set_rng_state entry points own the stream these draw from.
  //
  // Claimed directly rather than through the conf, because the codegen that
  // builds register_flaggems_tsingmicro.inc only emits ops its conf routes to a
  // FlagGems backend, and a `tsingmicro` route is not one -- it would drop these
  // silently. The REGISTER_IMPL_TO_DISPATCHER slots in rng.cc are filled for
  // symmetry with the sibling vendors and for a future conf route; the claims
  // below are what actually keep these ops off the fallback.
  #if defined(USE_TSINGMICRO) && !defined(FLAGOS_BOXING_BUILD)
    m.impl("rand", at::native::flagos::RandKernelTsingMicro);
    m.impl("rand.generator", at::native::flagos::RandGeneratorKernelTsingMicro);
    m.impl("rand.out", at::native::flagos::RandOutKernelTsingMicro);
    m.impl("rand.names_out", at::native::flagos::RandNamesOutKernelTsingMicro);
    m.impl("rand_like", at::native::flagos::RandLikeKernelTsingMicro);
    m.impl("rand_like.generator", at::native::flagos::RandLikeGeneratorKernelTsingMicro);
    m.impl("rand_like.out", at::native::flagos::RandLikeOutKernelTsingMicro);
    m.impl("uniform_", at::native::flagos::UniformInplaceKernelTsingMicro);
    m.impl("normal_", at::native::flagos::NormalInplaceKernelTsingMicro);
    m.impl("normal.float_float", at::native::flagos::NormalFloatFloatKernelTsingMicro);
    m.impl("normal.Tensor_float", at::native::flagos::NormalTensorFloatKernelTsingMicro);
    m.impl("normal.Tensor_Tensor", at::native::flagos::NormalTensorTensorKernelTsingMicro);
    m.impl("randint", at::native::flagos::RandintKernelTsingMicro);
    m.impl("randint.low", at::native::flagos::RandintLowKernelTsingMicro);
    m.impl("randint.out", at::native::flagos::RandintOutKernelTsingMicro);
    m.impl("randint.low_out", at::native::flagos::RandintLowOutKernelTsingMicro);
    m.impl("randint_like", at::native::flagos::RandintLikeKernelTsingMicro);
    m.impl("randint_like.low_dtype", at::native::flagos::RandintLikeLowDtypeKernelTsingMicro);
    m.impl("randint_like.out", at::native::flagos::RandintLikeOutKernelTsingMicro);
    m.impl("randint_like.low_dtype_out", at::native::flagos::RandintLikeLowDtypeOutKernelTsingMicro);
    m.impl("randperm", at::native::flagos::RandpermKernelTsingMicro);
    m.impl("randperm.out", at::native::flagos::RandpermOutKernelTsingMicro);
    m.impl("randn_like", at::native::flagos::RandnLikeKernelTsingMicro);
    m.impl("randn_like.out", at::native::flagos::RandnLikeOutKernelTsingMicro);
    m.impl("randn.names_out", at::native::flagos::RandnNamesOutKernelTsingMicro);
    m.impl("random_.from", at::native::flagos::RandomInplaceFromKernelTsingMicro);
    m.impl("native_dropout", at::native::flagos::NativeDropoutKernelTsingMicro);
    m.impl("native_dropout_backward", at::native::flagos::NativeDropoutBackwardKernelTsingMicro);
    m.impl("bernoulli_.Tensor", at::native::flagos::BernoulliInplaceTensorKernelTsingMicro);
    m.impl("binomial", at::native::flagos::BinomialKernelTsingMicro);
    m.impl("_standard_gamma", at::native::flagos::PrivStandardGammaKernelTsingMicro);
    m.impl("_sample_dirichlet", at::native::flagos::PrivSampleDirichletKernelTsingMicro);
    m.impl("randn", at::native::flagos::RandnKernelTsingMicro);
    m.impl("randn.generator", at::native::flagos::RandnGeneratorKernelTsingMicro);
    m.impl("randn_like.generator", at::native::flagos::RandnLikeGeneratorKernelTsingMicro);
    m.impl("randn_like.generator_out", at::native::flagos::RandnLikeGeneratorOutKernelTsingMicro);
    m.impl("randint.generator", at::native::flagos::RandintGeneratorKernelTsingMicro);
    m.impl("randint.low_generator", at::native::flagos::RandintLowGeneratorKernelTsingMicro);
    m.impl("randperm.generator", at::native::flagos::RandpermGeneratorKernelTsingMicro);
    m.impl("random_", at::native::flagos::RandomInplaceKernelTsingMicro);
    m.impl("random_.to", at::native::flagos::RandomInplaceToKernelTsingMicro);
    m.impl("bernoulli", at::native::flagos::BernoulliKernelTsingMicro);
    m.impl("bernoulli_.float", at::native::flagos::BernoulliInplaceFloatKernelTsingMicro);
    m.impl("exponential", at::native::flagos::ExponentialKernelTsingMicro);
    m.impl("exponential_", at::native::flagos::ExponentialInplaceKernelTsingMicro);
    m.impl("multinomial", at::native::flagos::MultinomialKernelTsingMicro);
    m.impl("poisson", at::native::flagos::PoissonKernelTsingMicro);
  #endif
  // FLAGOS_BUILD_FLAGGEMS=OFF leaves the block empty: no accelerated route is
  // compiled in, so every op is meant to reach cpu_fallback.
  #endif
  #endif
}

// Register fallback for all unimplemented operators
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&at::native::flagos::cpu_fallback>());
}

// The AutocastPrivateUse1 policies and fallback live in aten/autocast.cc.
// A dispatch key accepts only one backend fallback, so they must not be
// registered here as well.

// Register AutogradPrivateUse1 fallback to dispatch to PrivateUse1
// This ensures operators like where.ScalarSelf work correctly through autograd dispatch
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

// Register autograd-aware contiguous for PrivateUse1 tensors.
//
// Problem: contiguous registered on PrivateUse1 bypasses autograd recording
// (AutogradPrivateUse1 is fallthrough), causing grad_fn=None on the output
// and breaking gradient propagation (e.g., in attention layers that use
// transpose().contiguous()). On CUDA, contiguous() returns a tensor with
// CloneBackward0 grad_fn; on flagos it returned grad_fn=None.
//
// Solution: Register contiguous on AutogradPrivateUse1 so it intercepts
// the call before fallthrough. When the tensor actually needs copying
// (is non-contiguous), we use clone(memory_format) which properly records
// autograd operations. clone dispatches to PrivateUse1::clone which
// handles the actual data copy.
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl("contiguous", [](const at::Tensor& self, c10::MemoryFormat memory_format) -> at::Tensor {
    if (self.is_contiguous(memory_format)) {
      return self;
    }
    // clone(memory_format) creates a contiguous copy with autograd tracking.
    // This dispatches to PrivateUse1::clone (which uses empty + copy_),
    // and autograd records CloneBackward0 for gradient propagation.
    return self.clone(memory_format);
  });

  // narrow: same shape of problem as contiguous. aten::narrow is
  // CompositeImplicitAutograd upstream, and the dispatcher only falls back to
  // that math kernel for an autograd key when the matching backend slot is
  // empty. Claiming `narrow` on PrivateUse1 above fills that slot, so
  // AutogradPrivateUse1 stopped resolving to the composite and landed on
  // torchgen's Autograd[alias] stub -- "derivative for aten::narrow is not
  // implemented" -- as soon as the input required grad. The forward view still
  // worked, so only backward regressed (flagos-ai/Torch-FL#205).
  //
  // The same WrapperNarrow body run one key higher fixes it: narrow_symint
  // normalizes the bounds and calls at::slice_symint, a *dispatched* call that
  // re-enters autograd from here and records SliceBackward0 -- exactly the graph
  // PyTorch's composite builds, and the reason `x[:, 1:4]` never had this
  // problem. slice_backward is itself differentiable, so double backward works.
  //
  // No AutoDispatchBelowADInplaceOrView guard here: the inner slice must keep
  // its ADInplaceOrView kernel so the result carries proper view metadata.
  m.impl("narrow", WrapperNarrow);

  // matmul: on Ascend the fused aclnnMatmul kernel is claimed on plain
  // PrivateUse1 (above), and the generated AutogradPrivateUse1 kernel
  // (csrc/aten/generated/variable_type.cc) sits in front of it to build the
  // autograd graph. Other backends have no fused kernel, so they keep PyTorch's
  // composite decomposition; intercepting here lets them reach it without the
  // autograd key binding aten::matmul_backward, which they cannot implement.
#if !defined(USE_ASCEND)
  m.impl("matmul", WrapperMatmul);
#endif
}

} // namespace at::flagos
