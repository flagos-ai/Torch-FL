// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contiguous_ops.h"
#include "copy_ops.h"

#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/copy_native.h>
#include <flagos.h>
#include "device_boxing.h"
// ascend_copy.h and gcu_copy.h are included unconditionally: each supplies an
// inline no-op fallback for the platforms it does not implement, so the other
// #else branches below (TsingMicro, MUSA-without-mudnn) still compile. The
// Kunlunxin header is native like them but has no such fallback, so it is
// guarded and only pulled in for the platform that defines USE_KUNLUNXIN.
#include "backends/ascend/ascend_copy.h"
#include "backends/enflame/gcu_copy.h"
#if defined(USE_KUNLUNXIN)
#include "backends/kunlunxin/kunlunxin_copy.h"
#endif
// Cambricon is native like Kunlunxin but its declaration is header-only, so it
// is safe to include unconditionally: the namespace exists in every build and
// StridedCopy is only called from the USE_CAMBRICON arm below.
#include "backends/cambricon/cambricon_copy.h"
#if defined(USE_MPS)
#include "backends/apple/mps_copy.h"
#endif

#if defined(FLAGOS_MUSA_KERNEL)
#include "backends/mthreads/mudnn_common.h"
#endif

// Use ATen's strided copy when the linked PyTorch provides its CUDA kernel.
// The CPU PyTorch build uses the fallback copy paths below.
#if defined(FLAGOS_TORCH_LINKS_LIBTORCH_CUDA) && \
    !defined(USE_ASCEND) && \
    !defined(USE_TSINGMICRO) && !defined(USE_GCU) && !defined(USE_MUSA) && \
    !defined(USE_BPU) && !defined(USE_KUNLUNXIN) && !defined(USE_ILUVATAR) && \
    !defined(USE_MPS)
#define FLAGOS_CONTIGUOUS_CUDA_BOXING 1
#else
#define FLAGOS_CONTIGUOUS_CUDA_BOXING 0
#endif

namespace at::native::flagos {

namespace {

// Temporarily expose the storage representation without lazy math bits so the
// existing backend-specific clone path can copy it. The guard restores the
// original metadata before returning to the caller; all operations performed
// while it is active are synchronous with respect to tensor metadata.
class MathBitsGuard {
 public:
  explicit MathBitsGuard(const at::Tensor& tensor)
      : tensor_(tensor), conj_(tensor.is_conj()), neg_(tensor.is_neg()) {
    if (conj_) tensor_._set_conj(false);
    if (neg_) tensor_._set_neg(false);
  }

  ~MathBitsGuard() {
    if (conj_) tensor_._set_conj(true);
    if (neg_) tensor_._set_neg(true);
  }

  bool active() const { return conj_ || neg_; }
  bool conj() const { return conj_; }
  bool neg() const { return neg_; }

 private:
  at::Tensor tensor_;
  bool conj_;
  bool neg_;
};

// Materialize lazy math bits using the active backend's ordinary operators.
// This deliberately avoids CUDA boxing: GCU, MUSA, Ascend, DCU and other
// PrivateUse1 backends may not provide a CUDA runtime at all.
at::Tensor materialize_math_bits_impl(
    const at::Tensor& self,
    c10::MemoryFormat memory_format) {
  MathBitsGuard bits(self);
  if (!bits.active()) return self;

  auto result = self.clone(memory_format);
  if (bits.conj()) result = at::_conj_physical(result);
  if (bits.neg()) result = at::neg(result);
  return result;
}

// Copy into a fresh dense allocation with the requested result strides.
// Both legs go through BlockingMemcpy, which binds the device of the tensor each
// transfer touches and drains that device's queue first: a bare Memcpy is only
// ordered against whichever device happens to be current when it is issued, so
// gathering a flagos:1 source while flagos:0 is current would read what that
// buffer held before its kernels ran (issue #281, MUSA -- the same hazard, and
// the same two-precondition barrier, that copy_ops uses for every other copy).
#if !FLAGOS_CONTIGUOUS_CUDA_BOXING
void copy_strided_via_host(
    const at::Tensor& self,
    const at::Tensor& result) {
  if (TryGemsDeviceCopy(self, result)) return;
  TORCH_CHECK(CpuFallbackAllowed(),
              "native FlagGems hosts do not permit CPU fallback: device contiguous/clone ",
              self.scalar_type(), " sizes=", self.sizes(),
              " strides=", self.strides(), " destination strides=", result.strides());
  size_t storage_size = self.storage().nbytes();
  at::Tensor storage_cpu = at::empty(
      {static_cast<int64_t>(storage_size)},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  if (storage_size > 0) {
    BlockingMemcpy(
        storage_cpu.data_ptr(),
        self.storage().data(),
        storage_size,
        MemcpyDeviceToHost,
        self);
  }

  at::Tensor cpu_view = at::empty({0}, self.options().device(at::kCPU));
  cpu_view.set_(
      storage_cpu.storage(),
      self.storage_offset(),
      self.sizes(),
      self.strides());

  auto cpu_result = at::empty_strided(
      result.sizes(), result.strides(), result.options().device(at::kCPU));
  cpu_result.copy_(cpu_view);

  size_t nbytes = cpu_result.numel() * cpu_result.element_size();
  if (nbytes > 0) {
    BlockingMemcpy(
        result.data_ptr(),
        cpu_result.data_ptr(),
        nbytes,
        MemcpyHostToDevice,
        result);
  }
}
#endif

} // namespace

at::Tensor materialize_math_bits(
    const at::Tensor& self,
    c10::MemoryFormat memory_format) {
  return materialize_math_bits_impl(self, memory_format);
}

at::Tensor contiguous(
    const at::Tensor& self,
    c10::MemoryFormat memory_format) {
  if ((self.is_conj() || self.is_neg()) &&
      !self.is_contiguous(memory_format)) {
    return materialize_math_bits_impl(self, memory_format);
  }
  if (self.is_contiguous(memory_format)) {
    return self;
  }

  auto result = at::empty(self.sizes(), self.options().memory_format(memory_format));

  if (self.is_privateuseone()) {
    int64_t numel = self.numel();
    if (numel > 0) {
#if defined(FLAGOS_MUSA_KERNEL)
      // MUSA: mudnn's Unary::IDENTITY does the strided copy on device -- a mudnn
      // Tensor carries strides on both operands, so the gather needs no
      // intermediate buffer and no CPU round-trip.
      musa_ops::MudnnCopy(self, result);
#elif !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU) && !defined(USE_KUNLUNXIN) && \
    !defined(USE_ILUVATAR) && !defined(USE_MPS) && !defined(USE_CAMBRICON)
      // The CUDA gather: hand it to the native TensorIterator strided copy
      // kernel. `DeviceBoxingGuard` only rewrites device metadata to CUDA --
      // the kernel behind it is a CUDA DispatchStub, so the arm is chosen by
      // whether this build's torch has one (see FLAGOS_CONTIGUOUS_CUDA_BOXING
      // above), not by whether the boxing kernels were compiled. Where they are
      // both there it is worth a lot of cudaMemcpy: `_to_copy` hits this arm for
      // every non-contiguous source before it even reaches its dtype cast.
#if FLAGOS_CONTIGUOUS_CUDA_BOXING
      DeviceBoxingGuard guard(self, result);
      at::native::copy_(result, self, false);
#else
      // No native CUDA copy kernel in this wheel's torch: gather through host
      // memory. The vendor arms below do the same when their on-device path
      // declines.
      copy_strided_via_host(self, result);
#endif
#elif defined(USE_MPS)
      if (!mps::CopyStridedToDense(self, result)) {
        copy_strided_via_host(self, result);
      }
#else
      // Ascend copies the strided source into the contiguous result on-device
      // via aclnnInplaceCopy, GCU via topsatenCopy, Kunlunxin via the XPU
      // runtime's own strided copy. Falls back to a CPU round-trip only if that
      // path is unavailable.
#if defined(USE_GCU)
      const bool copied_on_device = gcu::StridedCopy(result, self);
#elif defined(USE_KUNLUNXIN)
      const bool copied_on_device = kunlunxin::StridedCopy(result, self);
#elif defined(USE_CAMBRICON)
      // cnnlCopy_v2 describes the source with cnnlSetTensorDescriptorEx, so a
      // strided gather runs on the device. Before this arm the CPU round-trip
      // below served it, and that was the most expensive call in a training
      // step: 16228 us for the (2,64,16,512) view RoPE produces, against 18 us
      // through cnnlCopy_v2.
      const bool copied_on_device = cambricon::StridedCopy(result, self);
#else
      const bool copied_on_device = ascend::StridedCopy(result, self);
#endif
      if (!copied_on_device) {
        copy_strided_via_host(self, result);
      }
#endif
    }

    return result;
  }

  result.copy_(self);
  return result;
}

at::Tensor empty_for_copy(
    const at::Tensor& self,
    const at::TensorOptions& options,
    c10::MemoryFormat memory_format) {
  at::Tensor result;
  if (memory_format != c10::MemoryFormat::Preserve) {
    result = at::empty(self.sizes(), options.memory_format(memory_format));
  } else if (self.is_non_overlapping_and_dense()) {
    result = at::empty_strided(self.sizes(), self.strides(), options);
  } else {
    result = at::empty_strided(
        self.sizes(), at::infer_dense_strides(self.sizes(), self.strides()), options);
  }
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor clone(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Preserve);
  if (self.is_conj() || self.is_neg()) {
    return materialize_math_bits_impl(self, memory_format);
  }

  auto result = empty_for_copy(self, self.options(), memory_format);
  if (self._is_zerotensor()) {
    return result.zero_();
  }
  if (self.numel() == 0) {
    return result;
  }
  if (!self.is_privateuseone()) {
    return result.copy_(self);
  }
  if (self.is_non_overlapping_and_dense() && self.strides().equals(result.strides())) {
    CopySameDevice(self, result);
    return result;
  }

#if defined(USE_MPS)
  if (!mps::CopyStridedToDense(self, result)) {
    copy_strided_via_host(self, result);
  }
#else
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU) && !defined(USE_KUNLUNXIN) && \
    !defined(USE_ILUVATAR) && !defined(USE_MPS) && !defined(USE_CAMBRICON)
  // Use DeviceBoxingGuard to leverage CUDA's native strided copy kernel
  // instead of an expensive CPU round-trip -- but only in a wheel whose torch
  // actually ships that kernel. See the note above
  // FLAGOS_CONTIGUOUS_CUDA_BOXING: the guard only rewrites device metadata,
  // the kernel behind it is a CUDA DispatchStub.
#if FLAGOS_CONTIGUOUS_CUDA_BOXING
  DeviceBoxingGuard guard(self, result);
  at::native::copy_(result, self, false);
#else
  copy_strided_via_host(self, result);
#endif
#else
  // MUSA joins the non-boxing group: copy_ reaches _copy_from, which
  // torch_musa implements for strided/dtype-casting copies on device.
  // On-device strided copy -- aclnnInplaceCopy on Ascend, topsatenCopy on
  // GCU, the XPU runtime's own copy on Kunlunxin -- instead of
  // result.copy_(self), which would bounce through a CPU round-trip. This is
  // the Qwen3 GQA repeat_kv hotspot (~59% of inference time before this
  // change), and on GCU it is the `.expand(...).clone()` / `.contiguous()`
  // pattern in Qwen-Image's attention: 975.6 ms per call at the
  // (1, 24, 4114, 4114) fp32 shape against 4.6 ms for the vendor call.
#if defined(USE_GCU)
  const bool copied_on_device = gcu::StridedCopy(result, self);
#elif defined(USE_KUNLUNXIN)
  const bool copied_on_device = kunlunxin::StridedCopy(result, self);
#elif defined(USE_CAMBRICON)
  const bool copied_on_device = cambricon::StridedCopy(result, self);
#else
  const bool copied_on_device = ascend::StridedCopy(result, self);
#endif
  if (!copied_on_device) {
    result.copy_(self);
  }
#endif
#endif
  return result;
}

} // namespace at::native::flagos
