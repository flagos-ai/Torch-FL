// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contiguous_ops.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/copy_native.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>
#include "device_boxing.h"
// ascend_copy.h and gcu_copy.h are included unconditionally: each supplies an
// inline no-op fallback for the platforms it does not implement, so the other
// #else branches below (TsingMicro, MUSA-without-mudnn) still compile. The
// Kunlunxin header is native like them but has no such fallback, so it is
// guarded and only pulled in for the platform that defines USE_KUNLUNXIN.
#include "backends/ascend/ascend_copy.h"
#include "backends/gcu/gcu_copy.h"
#if defined(USE_KUNLUNXIN)
#include "backends/kunlunxin/kunlunxin_copy.h"
#endif

#if defined(FLAGOS_MUSA_KERNEL)
#include "backends/musa/mudnn_common.h"
#endif

// Whether the CUDA-platform arms of contiguous()/clone() below may hand the
// strided gather to ATen's native CUDA copy kernel (DeviceBoxingGuard +
// at::native::copy_). Those arms are a pure "do it on device" shortcut, and the
// kernel they dispatch to -- plus the c10::cuda stream bookkeeping around it --
// lives in libtorch_cuda.so. The default FlagGems-primary wheel does not ship
// that library (it links the CPU-only pip torch), so on it the gather goes
// through copy_strided_via_host instead, which is also what the de-boxed
// backends (Ascend, GCU, Kunlunxin, MUSA, ...) always use. csrc/CMakeLists.txt
// defines FLAGOS_BOXING_BUILD only for a wheel built with FLAGOS_BUILD_BOXING=ON
// -- the same switch register.cc reads before claiming the generated CUDA
// kernels -- and the USE_* exclusions match the ones on the arms.
#if defined(FLAGOS_BOXING_BUILD) && !defined(USE_ASCEND) && \
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

// Copy the strided flagos tensor `self` into `result` -- an already-allocated
// flagos tensor with `self`'s sizes, laid out as `memory_format` -- through host
// memory: D2H the source's whole storage, rebuild its strides as a CPU view, let
// the host copy kernel produce the contiguous result, then H2D it.
//
// This is the de-boxed strided gather: the only implementation the de-boxed
// backends have, and what the CUDA platform uses when it has no native CUDA copy
// kernel to dispatch to (see FLAGOS_CONTIGUOUS_CUDA_BOXING above, which is the
// predicate for whether the DeviceBoxingGuard arms below exist).
#if !FLAGOS_CONTIGUOUS_CUDA_BOXING
void copy_strided_via_host(
    const at::Tensor& self,
    const at::Tensor& result,
    c10::MemoryFormat memory_format) {
  // A blocking cudaMemcpy orders against whichever device is current when it is
  // issued and drains that device's queue, so the source's device has to be
  // made current first -- reading a flagos:1 buffer while flagos:0 is current
  // otherwise returns that buffer's pre-kernel contents. This is the pair of
  // reasons copy_ops' BlockingCopyGuard exists, and it takes the same device-wide
  // sync here (rather than a per-stream wait) because a wheel without the boxing
  // kernels also has no libtorch_cuda.so to ask which stream is current.
  c10::DeviceGuard on_source(self.device());
  ::DeviceSynchronize();

  size_t storage_size = self.storage().nbytes();
  at::Tensor storage_cpu = at::empty(
      {static_cast<int64_t>(storage_size)},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  if (storage_size > 0) {
    Memcpy(
        storage_cpu.data_ptr(),
        self.storage().data(),
        storage_size,
        MemcpyDeviceToHost);
  }

  at::Tensor cpu_view = at::empty({0}, self.options().device(at::kCPU));
  cpu_view.set_(
      storage_cpu.storage(),
      self.storage_offset(),
      self.sizes(),
      self.strides());

  auto cpu_contig = at::empty(
      self.sizes(),
      self.options().device(at::kCPU).memory_format(memory_format));
  cpu_contig.copy_(cpu_view);

  size_t nbytes = cpu_contig.numel() * cpu_contig.element_size();
  if (nbytes > 0) {
    Memcpy(
        result.data_ptr(), cpu_contig.data_ptr(), nbytes, MemcpyHostToDevice);
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
    !defined(USE_ILUVATAR) && !defined(USE_MPS)
#if FLAGOS_CONTIGUOUS_CUDA_BOXING
      // CUDA platform: use DeviceBoxingGuard to invoke native CUDA strided copy
      // kernel on-device, avoiding expensive CPU round-trip.
      DeviceBoxingGuard guard(self, result);
      at::native::copy_(result, self, false);
#else
      // CUDA platform without the boxing kernels: there is no native CUDA copy
      // kernel in this wheel, so gather through host memory.
      copy_strided_via_host(self, result, memory_format);
#endif
#else
      // Ascend copies the strided source into the contiguous result on-device
      // via aclnnInplaceCopy, GCU via topsatenCopy, Kunlunxin via the XPU
      // runtime's own strided copy. Falls back to a CPU round-trip only if that
      // path is unavailable.
#if defined(USE_GCU)
      const bool copied_on_device = gcu::StridedCopy(result, self);
#elif defined(USE_KUNLUNXIN)
      const bool copied_on_device = kunlunxin::StridedCopy(result, self);
#else
      const bool copied_on_device = ascend::StridedCopy(result, self);
#endif
      if (!copied_on_device) {
        copy_strided_via_host(self, result, memory_format);
      }
#endif
    }

    return result;
  }

  result.copy_(self);
  return result;
}

at::Tensor clone(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Preserve);
  if (self.is_conj() || self.is_neg()) {
    return materialize_math_bits_impl(self, memory_format);
  }

  if (memory_format == c10::MemoryFormat::Preserve) {
    if (self.is_contiguous()) {
      auto result = at::empty_like(self);
      size_t nbytes = self.numel() * self.element_size();
      if (nbytes > 0 && self.is_privateuseone()) {
        Memcpy(result.data_ptr(), self.data_ptr(), nbytes, MemcpyDeviceToDevice);
      } else if (nbytes > 0) {
        result.copy_(self);
      }
      return result;
    }
    // For non-contiguous with Preserve, fall through to create contiguous copy.
    // NOTE: Do NOT call contiguous() here to avoid infinite recursion when
    // contiguous is unregistered from PrivateUse1 (CompositeImplicitAutograd
    // contiguous calls clone, which would call contiguous again).
    memory_format = c10::MemoryFormat::Contiguous;
  }

  // Non-contiguous clone.
  if (self.is_privateuseone()) {
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU) && !defined(USE_KUNLUNXIN) && \
    !defined(USE_ILUVATAR) && !defined(USE_MPS)
    auto result = at::empty(
        self.sizes(), self.options().memory_format(memory_format));
#if FLAGOS_CONTIGUOUS_CUDA_BOXING
    // Use DeviceBoxingGuard to leverage CUDA's native strided copy kernel
    // instead of an expensive CPU round-trip.
    DeviceBoxingGuard guard(self, result);
    at::native::copy_(result, self, false);
#else
    // Without the boxing kernels there is no native CUDA copy kernel to
    // dispatch to; gather through host memory instead.
    copy_strided_via_host(self, result, memory_format);
#endif
    return result;
#else
    // MUSA joins the non-boxing group: copy_ reaches _copy_from, which
    // torch_musa implements for strided/dtype-casting copies on device.
    auto result = at::empty(
        self.sizes(), self.options().memory_format(memory_format));
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
#else
    const bool copied_on_device = ascend::StridedCopy(result, self);
#endif
    if (!copied_on_device) {
      result.copy_(self);
    }
    return result;
#endif
  }

  auto result = at::empty(self.sizes(), self.options().memory_format(memory_format));
  result.copy_(self);
  return result;
}

} // namespace at::native::flagos
