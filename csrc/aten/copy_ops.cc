// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "copy_ops.h"
#include "contiguous_ops.h"
#include "copy_dispatcher.h"

#include <ATen/native/Resize.h>
#include <ATen/ops/_pin_memory.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/_to_copy_ops.h>
#include <ATen/ops/copy_native.h>
#include <c10/core/DeviceGuard.h>
#include <flagos.h>

#include <optional>
#include "device_boxing.h"
// Included unconditionally: the #else branches below cover TsingMicro, GCU and
// MUSA-without-mudnn as well as Ascend, and this header supplies inline no-op
// fallbacks for those platforms.
#include "backends/ascend/ascend_copy.h"

#if defined(FLAGOS_MUSA_KERNEL)
#include "backends/musa/mudnn_common.h"
#elif defined(USE_MUSA)
#include "runtime/accelerator/musa/musa_stream.h"
#endif

// On the CUDA-family backends (including MetaX boxing) the flagos device shares
// the vendor's CUDA streams, so the current stream is readable from c10::cuda.
//
// USE_DCU is excluded for the same reason runtime/guard.h:22 excludes it: the
// DCU wheel is hipified, so <c10/cuda/CUDAStream.h> resolves through DTK's
// CUDA-compat shim against USE_ROCM-hipified torch headers and expands to hip*
// symbols the shim never declares (`'hipStreamCaptureStatus' was not declared`),
// and c10::cuda::getCurrentCUDAStream would not link there anyway -- DTK exports
// c10::hip with zero c10::cuda symbols. DCU still shares the vendor's streams,
// so it needs *some* barrier; see BlockingCopyGuard below.
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_DCU) && !defined(USE_BPU)
#define FLAGOS_COPY_HAS_CUDA_STREAM 1
#include <c10/cuda/CUDAStream.h>
#endif

namespace at::native::flagos {

namespace {

// Every `Memcpy` in this file is the *synchronous* one, and a synchronous
// memcpy is a blocking call with two preconditions, both about which device is
// current when it is issued.
//
// 1. The device the copy touches has to *be* the current one. A blocking
//    memcpy orders against the device selected at issue time, not against the
//    device that owns the buffer. MUSA makes the consequence blunt: reading
//    back a tensor on flagos:1 while flagos:0 was current returned the
//    buffer's *pre-kernel* contents -- 18 of 20 host reads of an addmm output
//    came back as the zeros the output was allocated with, and 18 of 20 became
//    0 of 20 once the read bound flagos:1 first (issue #281). Model-parallel
//    workloads hit this constantly rather than rarely: with
//    device_map="auto" the ambient device is whichever one ran last, which is
//    usually not the one holding the tensor being read back.
//
// 2. That device's queue has to be drained. `Memcpy` only orders against the
//    legacy default stream, and PyTorch creates its side streams with
//    cudaStreamNonBlocking, so work enqueued on one is NOT awaited by a plain
//    cudaMemcpy: the copy can read a buffer before the kernel producing it has
//    run. That is a silent wrong-data bug, not a crash -- FSDP2 CPU offload hit
//    it because its gradient D2H happens inside the reduce-scatter stream, and
//    the gradient reached the CPU as zeros (or as a previous tensor's
//    contents).
//
// Doing 1 first is also what makes 2 target the right queue: the CUDA-family
// stream lookup below resolves against whatever device is current, so draining
// without the guard drains the ambient device rather than the tensor's.
//
// Both are no-ops on the common path -- the inner device guard is only emplaced
// for a tensor that really lives elsewhere, and a drain of the default stream is
// already ordered -- so single-device callers pay a GetDevice and the stream
// probe this file already paid.
class BlockingCopyGuard {
 public:
  // `on_device` is the accelerator-side operand the copy touches: the source
  // for a D2H or D2D copy (the buffer being read), the destination for an H2D
  // one. It must outlive the guard, which is to say the whole copy.
  explicit BlockingCopyGuard(const at::Tensor& on_device) {
    const auto device = on_device.device();
    if (device.is_privateuseone() && device.has_index()) {
      int current = -1;
      if (::GetDevice(&current) == Success && current != device.index()) {
        // Constructing a c10::DeviceGuard unconditionally costs ~2.8us/call,
        // since its ctor and dtor both route through the guard registry
        // (empty.cc measures the same trade-off); skipping it when the device
        // already matches removes that from the single-device path.
        guard_.emplace(device);
      }
    }
    DrainCurrentQueue();
  }

  BlockingCopyGuard(const BlockingCopyGuard&) = delete;
  BlockingCopyGuard& operator=(const BlockingCopyGuard&) = delete;

 private:
  void DrainCurrentQueue() {
#if defined(FLAGOS_COPY_HAS_CUDA_STREAM)
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (stream.stream() != nullptr) {
      stream.synchronize();
    }
#elif defined(USE_DCU)
    // DCU shares the vendor's streams, so it needs this barrier just as much as
    // the other CUDA-family backends -- but it cannot ask which stream is
    // current (no c10::cuda symbols in a hipified wheel; see the guard above).
    // Fall back to a device-wide sync: a superset of the per-stream wait, so
    // still correct, just coarser. Only reached on the blocking-copy paths.
    ::DeviceSynchronize();
#elif defined(USE_MUSA)
    // No c10::cuda to ask here either, but the shared per-device stream is
    // reachable directly. Every producer submits to it -- mudnn through
    // EXEC_MUDNN_CMD, FlagGems and Triton through flagtree_shim -- so draining
    // it is the barrier the CUDA arm gets from stream.synchronize(). The guard
    // above is what makes GetDefaultMusaStream() resolve to *this tensor's*
    // device; it keys off the current device, like the mudnn handle.
    musaStreamSynchronize(at::native::flagos::musa::GetDefaultMusaStream());
#endif
  }

  std::optional<c10::DeviceGuard> guard_;
};

// The guard and the copy it protects always travel together: the guard has to
// be alive while the memcpy is issued, which is one statement's worth of scope.
// `on_device` is the accelerator-side operand the copy touches -- the source for
// a D2H or D2D copy, the destination for an H2D one. Callers that hold the raw
// pointers already, and need the guard to span more than one memcpy, use
// BlockingCopyGuard directly.
void BlockingMemcpy(
    void* dst,
    const void* src,
    size_t nbytes,
    MemcpyKind kind,
    const at::Tensor& on_device) {
  BlockingCopyGuard barrier(on_device);
  Memcpy(dst, src, nbytes, kind);
}

// A blocking `Memcpy` is blocking only against the device that is current when
// it is issued: the transfer is submitted to that device's queue, and nothing
// orders it against the *peer* device's. A same-device copy never notices --
// BlockingCopyGuard already drained the one queue involved -- but a copy that
// crosses devices can be read by the destination before it lands.
//
// Measured on MTT S5000 / mudnn v3300, on the 13x7 `input_ids != pad` mask cast
// from flagos:0 to int32 on flagos:1 -- modeling_layers.py:167, the expression
// that fails test_model_parallelism under device_map="auto": the result came
// back as uninitialized memory at 1 call in 2000 in one process and 1-2 in
// 20000 in another, and draining the *default stream* of either device first
// changed nothing -- only a device-wide sync did, 0 failures in 20000 in every
// process. So the transfer is not ordered against the destination's default
// stream either, which is what rules out the per-stream wait BlockingCopyGuard
// uses everywhere else. The end-to-end measurement agrees: on two builds that
// differ only in this barrier, Qwen3ModelTest::test_model_parallelism failed 8
// of 9 fresh processes without it -- 10 to 20 of 26 elements wrong, max|d|
// 0.20 to 0.46 -- and 0 of 9 with it. Issue #281.
//
// Device-wide, therefore, and skipped unless the copy really crosses devices, so
// the single-device path pays one device comparison and nothing else.
void SyncPeerCopyVisibility(const at::Tensor& src, const at::Tensor& dst) {
  if (src.device() == dst.device()) {
    return;
  }
#if defined(USE_MUSA)
  // The only platform with a measurement of this hazard. The CUDA-family
  // backends get the same guarantee from cudaMemcpy's own semantics, and a
  // device-wide sync there would be a real cost on every peer copy.
  ::DeviceSynchronize();
#endif
}

// The device-to-device copy whose two ends are both known, which is what makes
// the peer-visibility sync above possible. It has to be issued inside the
// guard's scope: BlockingCopyGuard restores the previous device on teardown, so
// a `BlockingMemcpy` followed by a separate device sync would drain whatever
// device happened to be ambient before rather than the one that owns the
// transfer.
void CrossDeviceMemcpy(
    void* dst_ptr,
    const void* src_ptr,
    size_t nbytes,
    const at::Tensor& src,
    const at::Tensor& dst) {
  BlockingCopyGuard barrier(src);
  Memcpy(dst_ptr, src_ptr, nbytes, MemcpyDeviceToDevice);
  SyncPeerCopyVisibility(src, dst);
}

// The maca CUDA copy kernel (reached via DeviceBoxingGuard + at::native::copy_)
// has no UInt16/UInt32/UInt64 case in its dtype dispatch, so any dtype *cast*
// to or from those types raises `"copy_" not implemented for 'UInt32'`. The
// host copy kernel supports the full set, so route those casts through a CPU
// round-trip instead. Same-dtype copies are unaffected: they take the direct
// memcpy path and never reach a dtype dispatch.
bool cast_needs_cpu_roundtrip(c10::ScalarType src, c10::ScalarType dst) {
  if (src == dst) {
    return false;
  }
  const auto is_unsigned = [](c10::ScalarType t) {
    return t == c10::kUInt16 || t == c10::kUInt32 || t == c10::kUInt64;
  };
  return is_unsigned(src) || is_unsigned(dst);
}

// Copy `src` (a flagos device tensor) into `dst` (a flagos device tensor),
// casting dtype and honoring both tensors' strides, via a host round-trip.
// This is the fallback for dtype casts the native CUDA copy kernel cannot do
// (see cast_needs_cpu_roundtrip). Each blocking Memcpy binds the device of the
// buffer it touches and drains that device's queue first -- src and dst need not
// live on the same device, so the guard is per-copy rather than per-function.
void strided_copy_cast_via_cpu(const at::Tensor& src, const at::Tensor& dst) {
  at::Tensor src_contig = src.is_contiguous()
      ? src
      : at::native::flagos::contiguous(src, c10::MemoryFormat::Contiguous);
  size_t nbytes = src_contig.numel() * src_contig.element_size();
  at::Tensor cpu_src =
      at::empty(src_contig.sizes(), src_contig.options().device(at::kCPU));
  if (nbytes > 0) {
    BlockingMemcpy(
        cpu_src.data_ptr(),
        src_contig.data_ptr(),
        nbytes,
        MemcpyDeviceToHost,
        src_contig);
  }

  // Map dst's whole storage (not just its view) into CPU byte memory so a
  // strided in-place copy preserves the parts of the storage dst's view does
  // not cover. The CPU copy then writes only dst's logical elements.
  size_t dst_storage_nbytes = dst.storage().nbytes();
  at::Tensor cpu_dst_storage = at::empty(
      {static_cast<int64_t>(dst_storage_nbytes)},
      dst.options().device(at::kCPU).dtype(at::kByte));
  int64_t dst_storage_offset_bytes =
      dst.storage_offset() * static_cast<int64_t>(dst.element_size());
  char* dst_storage_base =
      static_cast<char*>(dst.data_ptr()) - dst_storage_offset_bytes;
  if (dst_storage_nbytes > 0) {
    BlockingMemcpy(
        cpu_dst_storage.data_ptr(),
        dst_storage_base,
        dst_storage_nbytes,
        MemcpyDeviceToHost,
        dst);
  }

  at::Tensor cpu_dst = at::empty({0}, dst.options().device(at::kCPU));
  cpu_dst.set_(
      cpu_dst_storage.storage(),
      dst.storage_offset(),
      dst.sizes(),
      dst.strides());
  at::native::copy_(cpu_dst, cpu_src, false);

  if (dst_storage_nbytes > 0) {
    BlockingMemcpy(
        dst_storage_base,
        cpu_dst_storage.data_ptr(),
        dst_storage_nbytes,
        MemcpyHostToDevice,
        dst);
  }
}

} // namespace

ADD_IMPL_TO_DISPATCHER(
    LocalScalarDenseFn, local_scalar_dense_dispatcher, "_local_scalar_dense")
ADD_IMPL_TO_DISPATCHER(ToCopyFn, to_copy_dispatcher, "_to_copy")

namespace {

at::Scalar LocalScalarDenseKernel(const at::Tensor& self) {
  return ::at::native::flagos::_local_scalar_dense(self);
}

at::Tensor ToCopyKernel(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format) {
  return ::at::native::flagos::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

}  // namespace

REGISTER_IMPL_TO_DISPATCHER(
    LocalScalarDenseFn,
    local_scalar_dense_dispatcher,
    Backend::kFlagGemsCpp,
    LocalScalarDenseKernel);
REGISTER_IMPL_TO_DISPATCHER(
    ToCopyFn, to_copy_dispatcher, Backend::kFlagGemsCpp, ToCopyKernel);

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  // Raw memcpy copies storage bytes and therefore bypasses lazy Conjugate and
  // Negative math bits. Materialize those bits once before entering any of the
  // device-copy fast paths, which otherwise intentionally operate on raw data.
  if (self.is_conj() || self.is_neg()) {
    return at::native::flagos::_copy_from(
        at::native::flagos::materialize_math_bits(
            self, c10::MemoryFormat::Preserve),
        dst,
        non_blocking);
  }

  // Both flagos tensors: copy on-device.
  if (self.is_privateuseone() && dst.is_privateuseone()) {
    if (self.is_contiguous() && dst.is_contiguous() &&
        self.sizes().equals(dst.sizes()) &&
        self.scalar_type() == dst.scalar_type()) {
      // Fast path: both contiguous, same shape and dtype → direct memcpy.
      // Both operands are flagos tensors but not necessarily on the same one, so
      // this goes through the peer-aware form.
      size_t nbytes = self.numel() * self.element_size();
      if (nbytes > 0) {
        CrossDeviceMemcpy(
            dst.data_ptr(), self.data_ptr(), nbytes, self, dst);
      }
    } else {
#if defined(FLAGOS_MUSA_KERNEL)
      // MUSA: mudnn handles strides and dtype casts on device in one pass
      // (IDENTITY or CAST over stride-carrying Tensors). Without this,
      // at::native::copy_ would route to the CUDA DispatchStub and fail with
      // "missing kernel for cuda", since nothing ever fills the CUDA slot on
      // this platform.
      musa_ops::MudnnCopy(self, const_cast<at::Tensor&>(dst));
#elif !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU)
      // CUDA platform: use DeviceBoxingGuard to dispatch to native CUDA
      // strided copy kernel (handles strides, dtype casts on-device). The
      // native copy kernel has no UInt16/UInt32/UInt64 case, so those casts
      // take the host round-trip instead.
      if (cast_needs_cpu_roundtrip(self.scalar_type(), dst.scalar_type())) {
        strided_copy_cast_via_cpu(self, dst);
      } else {
        DeviceBoxingGuard guard(self, dst);
        at::native::copy_(const_cast<at::Tensor&>(dst), self, false);
      }
#else
      // Ascend: copy on-device via aclnnInplaceCopy, which honors both src and
      // dst strides/offset and casts dtype. Avoids the CPU round-trip below.
      if (!ascend::StridedCopy(dst, self)) {
        // Fallback: CPU round-trip (device->host, strided copy on CPU, host->device).
        at::Tensor self_contig = self.is_contiguous()
            ? self
            : at::native::flagos::contiguous(self, c10::MemoryFormat::Contiguous);
        size_t nbytes = self_contig.numel() * self_contig.element_size();
        at::Tensor cpu_src =
            at::empty(self_contig.sizes(), self_contig.options().device(at::kCPU));
        if (nbytes > 0) {
          BlockingMemcpy(
              cpu_src.data_ptr(),
              self_contig.data_ptr(),
              nbytes,
              MemcpyDeviceToHost,
              self_contig);
        }
        size_t dst_storage_nbytes = dst.storage().nbytes();
        at::Tensor cpu_dst_storage = at::empty(
            {static_cast<int64_t>(dst_storage_nbytes)},
            dst.options().device(at::kCPU).dtype(at::kByte));
        int64_t dst_storage_offset_bytes =
            dst.storage_offset() * static_cast<int64_t>(dst.element_size());
        char* dst_storage_base =
            static_cast<char*>(dst.data_ptr()) - dst_storage_offset_bytes;
        if (dst_storage_nbytes > 0) {
          BlockingMemcpy(
              cpu_dst_storage.data_ptr(),
              dst_storage_base,
              dst_storage_nbytes,
              MemcpyDeviceToHost,
              dst);
        }

        at::Tensor cpu_dst = at::empty({0}, dst.options().device(at::kCPU));
        cpu_dst.set_(
            cpu_dst_storage.storage(),
            dst.storage_offset(),
            dst.sizes(),
            dst.strides());
        at::native::copy_(cpu_dst, cpu_src, false);
        if (dst_storage_nbytes > 0) {
          BlockingMemcpy(
              dst_storage_base,
              cpu_dst_storage.data_ptr(),
              dst_storage_nbytes,
              MemcpyHostToDevice,
              dst);
        }
      }
#endif
    }
    return dst;
  }

  // Cross-device copies: ensure contiguous src, then memcpy.
  // For non-contiguous dst, copy into a contiguous temp on dst's device,
  // then use the boxing path to scatter into dst with proper strides.
  at::Tensor self_contig = self.is_contiguous() ? self
      : (self.is_privateuseone()
             ? at::native::flagos::contiguous(self, c10::MemoryFormat::Contiguous)
             : self.contiguous());

  size_t nbytes = self_contig.numel() * self_contig.element_size();

  // Every branch below issues a blocking `Memcpy`. The source may have been
  // produced by kernels on a non-default stream (and `contiguous()` above may
  // itself have just enqueued one there), which a synchronous cudaMemcpy does
  // not wait for. Drain the current stream first.
  //
  // The guard binds the accelerator-side operand whichever side it is on -- the
  // source when the source is the flagos tensor, the destination otherwise --
  // because exactly one of the four branches below has a flagos source. On a
  // CUDA-platform backend a flagos tensor's index is the CUDA index of the same
  // physical device, so a "CUDA" destination binds correctly too.
  BlockingCopyGuard barrier(self_contig.is_privateuseone() ? self_contig : dst);

  if (self.is_cpu() && dst.is_privateuseone()) {
    if (dst.is_contiguous()) {
      Memcpy(dst.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyHostToDevice);
    } else {
      auto tmp = at::empty(self_contig.sizes(), dst.options());
      Memcpy(tmp.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyHostToDevice);
#if defined(USE_ASCEND) || defined(USE_TSINGMICRO) || defined(USE_GCU) || \
    defined(USE_MUSA) || defined(USE_BPU)
      at::native::flagos::_copy_from(tmp, dst, false);
#else
      DeviceBoxingGuard guard(tmp, dst);
      at::native::copy_(const_cast<at::Tensor&>(dst), tmp, false);
#endif
    }
  } else if (self.is_privateuseone() && dst.is_cpu()) {
    if (dst.is_contiguous()) {
      Memcpy(dst.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToHost);
    } else {
      auto tmp = at::empty(self_contig.sizes(), dst.options());
      Memcpy(tmp.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToHost);
      at::native::copy_(const_cast<at::Tensor&>(dst), tmp, false);
    }
  } else if (self.is_privateuseone() && dst.is_cuda()) {
    if (dst.is_contiguous()) {
      Memcpy(dst.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
    } else {
      auto tmp = at::empty(self_contig.sizes(), dst.options());
      Memcpy(tmp.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
      at::native::copy_(const_cast<at::Tensor&>(dst), tmp, false);
    }
  } else if (self.is_cuda() && dst.is_privateuseone()) {
    if (dst.is_contiguous()) {
      Memcpy(dst.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
    } else {
      auto tmp = at::empty(self_contig.sizes(), dst.options());
      Memcpy(tmp.data_ptr(), self_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
#if defined(USE_ASCEND) || defined(USE_TSINGMICRO) || defined(USE_GCU) || \
    defined(USE_MUSA) || defined(USE_BPU)
      at::native::flagos::_copy_from(tmp, dst, false);
#else
      DeviceBoxingGuard guard(tmp, dst);
      at::native::copy_(const_cast<at::Tensor&>(dst), tmp, false);
#endif
    }
  } else {
    TORCH_CHECK(
        false,
        "Unsupported device combination for copy: ",
        self.device(),
        " -> ",
        dst.device());
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::flagos::_copy_from(self, dst, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  TORCH_CHECK(
      self.numel() == 1,
      "_local_scalar_dense expects a tensor with 1 element");
  at::Tensor cpu_tensor = at::empty({1}, self.options().device(at::kCPU));
  // `.item()` on a value just computed on a side stream must see that kernel's
  // result, and a blocking cudaMemcpy does not wait for a non-blocking stream.
  // It also has to be issued while *self's* device is current -- `.item()` is the
  // most common host read in a model (loss printing, `if loss > x`, schedulers)
  // and is usually called on a tensor whose device is not the ambient one.
  BlockingMemcpy(
      cpu_tensor.data_ptr(),
      self.data_ptr(),
      self.element_size(),
      MemcpyDeviceToHost,
      self);
  return cpu_tensor.item();
}

at::Tensor _to_copy(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(
      !layout_opt.has_value() || self.layout() == layout_opt.value(),
      "to(options) doesn't support converting to a different layout, "
      "but got self.layout being ",
      self.layout(),
      " and options.layout set as ",
      layout_opt.value());
  TORCH_CHECK(
      self.layout() == c10::kStrided,
      "flagos _to_copy only supports strided tensors, but got ",
      self.layout());
  TORCH_CHECK(
      !self.is_quantized(),
      "flagos _to_copy does not support quantized tensors yet");
  const bool want_pinned = pin_memory_opt.value_or(false);
  auto device = device_opt.value_or(self.device());
  auto dtype = dtype_opt.value_or(self.scalar_type());
  auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Preserve);

  // pin_memory is a host-memory concept: only a CPU destination can be pinned.
  // (Pinning is applied to the result CPU tensor below via _pin_memory, using
  // the flagos host allocator = cudaMallocHost on MetaX.)
  TORCH_CHECK(
      !want_pinned || device.is_cpu(),
      "flagos _to_copy: pin_memory=True is only valid for a CPU destination, "
      "but got destination device ",
      device);

  if ((device.is_privateuseone() || device.is_cuda()) && device.index() < 0) {
    const auto self_device = self.device();
    const auto device_index = self_device.type() == device.type() &&
            self_device.index() >= 0
        ? self_device.index()
        : 0;
    device = c10::Device(device.type(), device_index);
  }

  if (device == self.device() && dtype == self.scalar_type()) {
    if (memory_format == c10::MemoryFormat::Preserve) {
      return self.clone();
    }
    return self.clone().contiguous(memory_format);
  }

  bool src_is_flagos = self.is_privateuseone();
  bool dst_is_flagos = device.is_privateuseone();
  bool dst_is_cuda = device.is_cuda();
  bool dst_is_cpu = device.is_cpu();

  at::Tensor result;

  // Branches that end in a blocking `Memcpy` drain the current stream first --
  // after their `contiguous()` call, which may itself enqueue a kernel there.
  // `tensor.to("cpu")` inside `with flagos.stream(s)` returned stale data
  // before this. Branches that redispatch to a native CUDA kernel instead need
  // no drain: those kernels are stream-ordered on their own.

  if (src_is_flagos && dst_is_cuda) {
    int device_index =
        device.index() >= 0 ? device.index()
                            : (self.device().index() >= 0 ? self.device().index() : 0);
#if defined(USE_ASCEND) || defined(USE_TSINGMICRO) || defined(USE_GCU) || \
    defined(USE_MUSA)
    // No CUDA runtime on these platforms: keep the explicit contiguous + memcpy
    // path (a "CUDA" destination here is only reachable via shared-memory tricks
    // that don't exist without cudart).
    at::Tensor self_contig = self.contiguous();
    at::Tensor temp = at::empty(
        self_contig.sizes(),
        self_contig.options().device(c10::Device(c10::kCUDA, device_index)));
    size_t nbytes = self_contig.numel() * self_contig.element_size();
    if (nbytes > 0) {
      BlockingMemcpy(
          temp.data_ptr(),
          self_contig.data_ptr(),
          nbytes,
          MemcpyDeviceToDevice,
          self_contig);
    }
    result = (dtype != self.scalar_type()) ? temp.to(dtype) : temp;
#else
    // CUDA platform fast path: flagos (PrivateUse1) and CUDA share the same GPU
    // memory, so box self to CUDA and redispatch straight to the native CUDA
    // _to_copy with an explicit CUDA DispatchKeySet. That skips re-entering the
    // dispatcher from the top (no risk of routing back to PrivateUse1) and lets
    // the native kernel read self's strides + cast dtype in one on-device pass,
    // allocating the result directly on CUDA -- no intermediate contiguous copy
    // or manual Memcpy. self is unboxed back to PrivateUse1 on guard teardown.
    DeviceBoxingGuard guard(self);
    result = at::_ops::_to_copy::redispatch(
        c10::DispatchKeySet(c10::DispatchKey::CUDA),
        self,
        dtype,
        /*layout=*/std::nullopt,
        c10::Device(c10::kCUDA, device_index),
        /*pin_memory=*/std::nullopt,
        non_blocking,
        memory_format);
#endif
  } else if (src_is_flagos && dst_is_flagos) {
    int device_index = device.index() >= 0 ? device.index() : 0;
    at::Tensor self_contig = self.contiguous();
    if (dtype != self.scalar_type()) {
#if defined(FLAGOS_MUSA_KERNEL)
      // MUSA: mudnn's Unary::CAST converts dtype on device, so no CPU
      // round-trip for a plain `.to(dtype)`.
      result = at::empty(self_contig.sizes(), self_contig.options()
          .dtype(dtype).device(c10::Device(c10::kPrivateUse1, device_index)));
      musa_ops::MudnnCopy(self_contig, result);
#elif defined(USE_ASCEND) || defined(USE_TSINGMICRO) || defined(USE_GCU) || \
    defined(USE_MUSA) || defined(USE_BPU)
      // No CUDA runtime on these backends, so the CUDA TensorIterator cast
      // below is unavailable.
#ifdef USE_ASCEND
      // Ascend casts on-device via aclnnCast, avoiding the D2H->CPU->H2D
      // round-trip that dominated HF RMSNorm (two fp16<->fp32 casts per layer).
      result = ascend::DtypeCast(self_contig, dtype);
#endif
      if (!result.defined()) {
        // Fallback: CPU round-trip when no on-device cast is available
        // (TsingMicro / GCU / MUSA / BPU, or an Ascend dtype pair aclnnCast
        // rejects).
        size_t nbytes = self_contig.numel() * self_contig.element_size();
        at::Tensor cpu_tensor =
            at::empty(self_contig.sizes(), self_contig.options().device(at::kCPU));
        if (nbytes > 0) {
          BlockingMemcpy(
              cpu_tensor.data_ptr(),
              self_contig.data_ptr(),
              nbytes,
              MemcpyDeviceToHost,
              self_contig);
        }
        cpu_tensor = cpu_tensor.to(dtype);
        result = at::empty(
            cpu_tensor.sizes(),
            cpu_tensor.options().device(c10::Device(c10::kPrivateUse1, device_index)));
        size_t result_nbytes = cpu_tensor.numel() * cpu_tensor.element_size();
        if (result_nbytes > 0) {
          BlockingMemcpy(
              result.data_ptr(),
              cpu_tensor.data_ptr(),
              result_nbytes,
              MemcpyHostToDevice,
              result);
        }
      }
#else
      // CUDA platform: use DeviceBoxingGuard + CUDA TensorIterator copy kernel
      // for dtype cast on-device, avoiding costly CPU round-trip. The native
      // copy kernel has no UInt16/UInt32/UInt64 case, so those casts take the
      // host round-trip instead.
      if (cast_needs_cpu_roundtrip(self_contig.scalar_type(), dtype)) {
        result = at::empty(
            self_contig.sizes(),
            self_contig.options()
                .dtype(dtype)
                .device(c10::Device(c10::kPrivateUse1, device_index)));
        strided_copy_cast_via_cpu(self_contig, result);
      } else {
        result = at::empty(
            self_contig.sizes(),
            self_contig.options()
                .dtype(dtype)
                .device(c10::Device(c10::kPrivateUse1, device_index)));
        DeviceBoxingGuard guard(self_contig, result);
        at::native::copy_(result, self_contig, false);
      }
#endif
    } else {
      result = at::empty(
          self_contig.sizes(),
          self_contig.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t nbytes = self_contig.numel() * self_contig.element_size();
      if (nbytes > 0) {
        // `device` may name a different flagos device than self's, which is the
        // cross-device case `device_map="auto"` hits on every forward.
        CrossDeviceMemcpy(
            result.data_ptr(), self_contig.data_ptr(), nbytes, self_contig, result);
      }
    }
  } else if (src_is_flagos && dst_is_cpu) {
    at::Tensor self_contig = self.contiguous();
    at::Tensor temp =
        at::empty(self_contig.sizes(), self_contig.options().device(at::kCPU));
    size_t nbytes = self_contig.numel() * self_contig.element_size();
    if (nbytes > 0) {
      BlockingMemcpy(
          temp.data_ptr(),
          self_contig.data_ptr(),
          nbytes,
          MemcpyDeviceToHost,
          self_contig);
    }
    result = (dtype != self.scalar_type()) ? temp.to(dtype) : temp;
  } else if (!src_is_flagos && dst_is_flagos) {
    int device_index = device.index() >= 0 ? device.index() : 0;
    at::Tensor src_contig = self.contiguous();
    if (dtype != self.scalar_type()) {
      src_contig = src_contig.to(dtype);
    }
    result = at::empty(
        src_contig.sizes(),
        src_contig.options().device(c10::Device(c10::kPrivateUse1, device_index)));
    size_t nbytes = src_contig.numel() * src_contig.element_size();
    if (nbytes > 0) {
      BlockingCopyGuard barrier(result);
      if (self.is_cpu()) {
        Memcpy(result.data_ptr(), src_contig.data_ptr(), nbytes, MemcpyHostToDevice);
      } else if (self.is_cuda()) {
        Memcpy(result.data_ptr(), src_contig.data_ptr(), nbytes, MemcpyDeviceToDevice);
      } else {
        TORCH_CHECK(false, "_to_copy: unsupported source device ", self.device());
      }
    }
  } else {
    at::Tensor cpu_tensor = self.to(at::kCPU).to(dtype);
    if (dst_is_flagos) {
      int device_index = device.index() >= 0 ? device.index() : 0;
      result = at::empty(
          cpu_tensor.sizes(),
          cpu_tensor.options().device(c10::Device(c10::kPrivateUse1, device_index)));
      size_t nbytes = cpu_tensor.numel() * cpu_tensor.element_size();
      if (nbytes > 0) {
        BlockingMemcpy(
            result.data_ptr(),
            cpu_tensor.data_ptr(),
            nbytes,
            MemcpyHostToDevice,
            result);
      }
    } else {
      result = cpu_tensor.to(device);
    }
  }

  if (memory_format != c10::MemoryFormat::Preserve) {
    result = result.contiguous(memory_format);
  }

  // Copy the result (CPU) into pinned host memory when requested. Guarded above
  // so this only runs for a CPU destination; _pin_memory routes to the flagos
  // host allocator (cudaMallocHost on MetaX) registered via the hooks.
  if (want_pinned) {
    result = at::_pin_memory(result, std::nullopt);
  }

  return result;
}

} // namespace at::native::flagos
