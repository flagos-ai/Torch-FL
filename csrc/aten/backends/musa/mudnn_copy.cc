// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device copy for the MUSA backend: the replacement for torch_musa's
// `at::musa::_copy_from`.
//
// mudnn's Unary ops read strides on both operands, which was verified directly
// against the library: a transposed source gathers correctly into a contiguous
// destination, a strided destination is scattered into without touching the gaps,
// stride-0 (broadcast) sources replicate, and CAST does any of those while
// converting dtype. So every copy shape this backend needs is one Unary::Run --
// no temporary buffer, no CPU round-trip. The one shape it cannot serve itself
// is a source and destination on *different* devices; see the staging helper.

#include "aten/backends/musa/mudnn_common.h"

#ifdef USE_MUSA

namespace at::native::flagos::musa_ops {

namespace {

// Returns a contiguous copy of `src` on `dst`'s device, at src's own dtype.
//
// Only used when the two operands disagree on device. A mudnn op runs entirely
// on one device -- the handle, the stream and every operand address resolve
// against whatever device is current (MusaDeviceGuard, mudnn_common.h) -- so
// handing it one operand per device is not a copy it can perform. Measured on
// MTT S5000 / mudnn v3300: it dereferences the foreign pointer, the sync in
// EXEC_MUDNN_CMD reports `an illegal memory access was encountered`, and from
// there the whole context is poisoned (`[flagos-musa] musaFree(...) failed: an
// illegal memory access was encountered` on the next teardown) so every later
// op in the process fails too. Issue #250 is one such run: transformers with
// device_map="auto" places input_ids and logits on different devices, and
// modeling_layers.py then casts between them.
at::Tensor StageSourceOnDstDevice(const at::Tensor& src, const at::Tensor& dst) {
  TORCH_CHECK(
      src.is_privateuseone(),
      "MudnnCopy: source must be a flagos tensor to stage across devices, got ",
      src.device());

  // Both allocations below carry an explicit device, and empty.cc installs a
  // DeviceGuard when that device is not the current one, so neither depends on
  // what is selected here. `.contiguous()` reaches back into this file through
  // contiguous_ops.cc, but with both operands on src's device, so that inner
  // call takes the plain single-device path below.
  at::Tensor src_contig = src.is_contiguous() ? src : src.contiguous();
  at::Tensor staged =
      at::empty(src_contig.sizes(), src_contig.options().device(dst.device()));

  const size_t nbytes = src_contig.numel() * src_contig.element_size();
  if (nbytes > 0) {
    // A blocking musaMemcpy only orders against the device that is current when
    // it is issued, and every producer on src's device -- mudnn, FlagGems,
    // Triton -- writes to that device's default stream. Drain that queue before
    // reading through it. (copy_ops.cc has the same barrier in
    // SyncCurrentStreamBeforeBlockingCopy, but it compiles out on MUSA: the
    // FLAGOS_COPY_HAS_CUDA_STREAM guard excludes USE_MUSA, since there is no
    // c10::cuda here to ask for the current stream.)
    MusaDeviceGuard src_guard(src);
    musaStreamSynchronize(at::native::flagos::musa::GetDefaultMusaStream());
    musaMemcpy(
        staged.data_ptr(),
        src_contig.const_data_ptr(),
        nbytes,
        musaMemcpyDeviceToDevice);
  }
  return staged;
}

} // namespace

void MudnnCopy(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(
      src.defined() && dst.defined(), "MudnnCopy: undefined tensor");
  if (dst.numel() == 0) {
    return;
  }
  TORCH_CHECK(
      MudnnSupportsDtype(src.scalar_type()) &&
          MudnnSupportsDtype(dst.scalar_type()),
      "MudnnCopy: unsupported dtype ", src.scalar_type(), " -> ",
      dst.scalar_type());

  // Both operands of the Unary below have to live on the same device; bring the
  // source over when they do not.
  const at::Tensor src_staged =
      src.device() == dst.device() ? src : StageSourceOnDstDevice(src, dst);

  // Broadcast to the destination shape when needed. expand() only adjusts
  // sizes/strides (introducing 0 strides), which mudnn handles, so this stays a
  // view -- no allocation.
  at::Tensor src_view = src_staged;
  if (!src_staged.sizes().equals(dst.sizes())) {
    src_view = src_staged.expand(dst.sizes());
  }

  MudnnTensorWrapper t_src(src_view);
  MudnnTensorWrapper t_dst(dst);

  mudnn::Unary op;
  const bool needs_cast = src.scalar_type() != dst.scalar_type();
  op.SetMode(
      needs_cast ? mudnn::Unary::Mode::CAST : mudnn::Unary::Mode::IDENTITY);

  EXEC_MUDNN_CMD(
      needs_cast ? "mudnn copy (CAST)" : "mudnn copy (IDENTITY)",
      dst,
      op.Run(_mudnn_h, t_dst.get(), t_src.get()));
}

} // namespace at::native::flagos::musa_ops

#endif // USE_MUSA
