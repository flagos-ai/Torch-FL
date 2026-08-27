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
#include <ATen/SDPBackend.h>
#include <torch/library.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include "common.h"
#include "backends/soft_lowp/format.h"
#include "backends/soft_lowp/ops.h"
#include "runtime/allocator/caching_device_allocator.h"
#include "dispatcher.h"

#include <c10/core/impl/LocalDispatchKeySet.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

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

#if defined(USE_BPU)
// Convolution on a device with no per-op kernels.
//
// aten::convolution dispatches PrivateUse1 to convolution_overrideable, and the
// only other kernel registered for that op is a CompositeExplicitAutograd stub
// that raises NotImplementedError. So the boxed cpu_fallback cannot help here:
// it moves the arguments to CPU and redispatches the same op, which lands back
// on the stub. These wrappers cross to CPU and then call at::convolution --
// a different op, and the one that actually has a CPU kernel.
at::Tensor BPUWrapperConvolutionOverrideable(
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
BPUWrapperConvolutionBackwardOverrideable(
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
#endif // USE_BPU

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

#if defined(USE_MUSA) && defined(FLAGOS_FLAGGEMS_PYTHON)
bool MusaFlagGemsEnabled() {
  const char* value = std::getenv("FLAGOS_USE_FLAGGEMS");
  if (value != nullptr) {
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (normalized != "" && normalized != "0" && normalized != "off" &&
        normalized != "false") {
      return true;
    }
  }

  // FLAGOS_BACKEND_CONFIG is an advanced override that wins over the convenience
  // switch in Python. Register the narrow hybrid schema set when that config is
  // selected explicitly too, otherwise its flagos_python routes have no kernel.
  const char* config = std::getenv("FLAGOS_BACKEND_CONFIG");
  if (config == nullptr) return false;
  std::string path(config);
  auto slash = path.find_last_of("/\\");
  auto filename = slash == std::string::npos ? path : path.substr(slash + 1);
  return filename == "backends_musa_flagos_py.conf";
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
  // in this coverage set may select their kFlagOsPython dispatcher slot.
  //
  // MUSA is the same story: musa_register.inc lists exactly the ops with a mudnn
  // kernel behind them, and no CUDA boxing kernels are compiled in
  // (musa_runtime, not cudart), so uncovered ops must fall through rather than
  // be claimed. The two convolution `*_overrideable` ops are the exception that
  // proves the rule -- ATen's default for them raises instead of being boxable,
  // so they get real kernels in backends/musa/mudnn_conv.cc.
  #if defined(USE_GCU)
    #if defined(FLAGOS_GCU_KERNEL)
    #include "backends/gcu/generated/gcu_register.inc"
    #endif
  #elif defined(USE_MUSA)
    #if defined(FLAGOS_MUSA_KERNEL)
    #include "backends/musa/generated/musa_register.inc"
    #endif
    #if defined(FLAGOS_FLAGGEMS_PYTHON)
    if (MusaFlagGemsEnabled()) {
    #include "backends/musa/generated/musa_flaggems_register.inc"
    }
    #endif
  #elif defined(USE_BPU)
    // BPU registers no compute ops. The BPU's unit of execution is a whole
    // compiled graph (a .hbm produced by hbdk4), so there is no per-operator
    // kernel to claim -- and claiming an op on PrivateUse1 without a kernel
    // behind it raises "backend not registered" instead of falling back.
    // Leaving the list out routes every op to cpu_fallback below; acceleration
    // comes from the torch.compile backend in torch_fl/backends/bpu/.
    //
    // The *_overrideable ops are the exception, and the generic fallback cannot
    // serve them. aten::convolution routes PrivateUse1 to
    // convolution_overrideable, whose only other kernel is a
    // CompositeExplicitAutograd stub that raises NotImplementedError -- so
    // cpu_fallback, which moves the arguments to CPU and redispatches the *same*
    // op, lands right back on that stub. These wrappers instead call
    // at::convolution on the CPU tensors, which is the op that actually has a
    // CPU kernel.
    m.impl("convolution_overrideable", BPUWrapperConvolutionOverrideable);
    m.impl("convolution_backward_overrideable",
        BPUWrapperConvolutionBackwardOverrideable);
  #else
  #define FLAGOS_GEN_IMPLS
  #include "generated/register.inc"
  #undef FLAGOS_GEN_IMPLS
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
