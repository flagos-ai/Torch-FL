// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "contiguous_ops.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/copy_native.h>
#include <flagos.h>
#include "device_boxing.h"
// Included unconditionally: the #else branches below cover TsingMicro, GCU and
// MUSA-without-mudnn as well as Ascend, and this header supplies inline no-op
// fallbacks for those platforms.
#include "backends/ascend/ascend_copy.h"

#if defined(FLAGOS_MUSA_KERNEL)
#include "backends/musa/mudnn_common.h"
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
    !defined(USE_MUSA) && !defined(USE_BPU)
      // CUDA platform: use DeviceBoxingGuard to invoke native CUDA strided copy
      // kernel on-device, avoiding expensive CPU round-trip.
      DeviceBoxingGuard guard(self, result);
      at::native::copy_(result, self, false);
#else
      // Ascend: copy the strided source into the contiguous result on-device
      // via aclnnInplaceCopy. Falls back to a CPU round-trip only if that path
      // is unavailable.
      if (!ascend::StridedCopy(result, self)) {
        size_t storage_size = self.storage().nbytes();
        at::Tensor storage_cpu = at::empty(
            {static_cast<int64_t>(storage_size)},
            at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        Memcpy(storage_cpu.data_ptr(), self.storage().data(), storage_size, MemcpyDeviceToHost);

        at::Tensor cpu_view = at::empty({0}, self.options().device(at::kCPU));
        cpu_view.set_(
            storage_cpu.storage(),
            self.storage_offset(),
            self.sizes(),
            self.strides());

        auto cpu_contig = at::empty(self.sizes(), self.options().device(at::kCPU).memory_format(memory_format));
        cpu_contig.copy_(cpu_view);

        size_t nbytes = cpu_contig.numel() * cpu_contig.element_size();
        Memcpy(result.data_ptr(), cpu_contig.data_ptr(), nbytes, MemcpyHostToDevice);
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

  // Non-contiguous clone: use DeviceBoxingGuard to leverage CUDA's native
  // strided copy kernel instead of expensive CPU round-trip.
  if (self.is_privateuseone()) {
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU)
    auto result = at::empty(
        self.sizes(), self.options().memory_format(memory_format));
    DeviceBoxingGuard guard(self, result);
    at::native::copy_(result, self, false);
    return result;
#else
    // MUSA joins the non-boxing group: copy_ reaches _copy_from, which
    // torch_musa implements for strided/dtype-casting copies on device.
    auto result = at::empty(
        self.sizes(), self.options().memory_format(memory_format));
    // On-device strided copy (aclnnInplaceCopy) instead of result.copy_(self),
    // which would bounce through a CPU round-trip. This is the Qwen3 GQA
    // repeat_kv hotspot (~59% of inference time before this change).
    if (!ascend::StridedCopy(result, self)) {
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
