// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "strided_ops.h"
#include "generated/ops.h"

#include <ATen/EmptyTensor.h>
#include <ATen/native/Resize.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/t_native.h>
#include <ATen/ops/unbind_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/_conj_native.h>
#include <ATen/ops/_neg_view_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_as_complex_native.h>

namespace at::native::flagos {

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  auto int_size = C10_AS_INTARRAYREF_SLOW(size);
  auto int_stride = C10_AS_INTARRAYREF_SLOW(stride);
  std::optional<int64_t> int_offset = storage_offset.has_value()
      ? std::optional<int64_t>(storage_offset->expect_int())
      : std::nullopt;
  return at::native::as_strided_tensorimpl(self, int_size, int_stride, int_offset);
}

namespace {

// Grow `self`'s storage to hold `size` contiguous elements, the way
// at::native::maybe_resize_storage_cpu does -- except for how the old contents
// are moved.
//
// Reaching at::native::resize_ with a flagos tensor lands in
// maybe_resize_storage_cpu -> resize_bytes_cpu. That helper allocates the new
// block through the storage's own allocator, which is correct here (a flagos
// storage carries the PrivateUse1 allocator), but then moves the existing
// bytes with a **host** memcpy: `libtorch_cpu.so+0x2164723: call memcpy@plt`,
// called from resize_bytes_cpu+0x168 (objdump, plus the SIGSEGV register dump
// in /tmp/bt_segv2.py). The bytes of a flagos storage live in HBM, so the
// first load of that copy faults. It is why growing a tensor that already has
// content segfaults while growing one from empty does not -- with an empty new
// shape the helper returns before it allocates or copies anything, and a
// shrink or an unchanged shape never gets that far.
//
// CUDA does not have this problem because it has its own Resize.h with
// maybe_resize_storage_cuda / resize_bytes_cuda, which performs that copy with
// cudaMemcpy. This is the same two steps -- allocate, then move the bytes
// through the backend -- written with the allocator's copy_data, which is the
// device memcpy (CachingDeviceAllocator::copy_data -> MemcpyDeviceToDevice).
//
// Called before at::native::resize_ so that the shape update ATen performs
// afterwards finds the storage already large enough and never enters
// resize_bytes_cpu.
void maybe_resize_storage_flagos(const at::Tensor& self, c10::IntArrayRef size) {
  auto* self_ = self.unsafeGetTensorImpl();
  const size_t new_size_bytes = at::detail::computeStorageNbytesContiguous(
      size,
      self_->dtype().itemsize(),
      static_cast<size_t>(self_->storage_offset()));
  // A zero-element requirement needs no storage, and computing one from a
  // non-zero storage_offset would be wrong for it (ATen bails on the same
  // condition -- on the new numel -- in maybe_resize_storage_cpu).
  if (new_size_bytes == 0) {
    return;
  }
  const c10::Storage& storage = self_->unsafe_storage();
  TORCH_CHECK(storage, "Cannot resize a flagos tensor that has no storage");
  if (new_size_bytes <= storage.nbytes()) {
    return;
  }
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();
  TORCH_CHECK(
      storage_impl->resizable(), "Trying to resize storage that is not resizable");
  c10::Allocator* allocator = storage_impl->allocator();
  TORCH_CHECK(
      allocator != nullptr, "Trying to resize storage without an allocator");
  // The block pool resolves the device to allocate from at call time, from the
  // current one, and allocate() installs no guard of its own -- empty_strided
  // pays for the same guard. Growth is rare enough that the guard costs
  // nothing measurable here.
  const c10::DeviceGuard device_guard(self.device());
  c10::DataPtr new_data = allocator->allocate(new_size_bytes);
  if (storage.nbytes() > 0) {
    allocator->copy_data(new_data.get(), storage.data(), storage.nbytes());
  }
  storage_impl->set_data_ptr_noswap(std::move(new_data));
  storage_impl->set_nbytes(new_size_bytes);
}

} // namespace

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  auto int_size = C10_AS_INTARRAYREF_SLOW(size);
  // Grow on the device first: at::native::resize_ is the CPU implementation,
  // and growing from inside it host-memcpys the device contents (see above).
  maybe_resize_storage_flagos(self, int_size);
  return at::native::resize_(self, int_size, memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor expand(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit) {
  return at::native::expand(self, C10_AS_INTARRAYREF_SLOW(size), implicit);
}

// NOTE: all view ops call at::native:: directly (not the tensor member method).
// The member methods re-dispatch through PrivateUse1, which routes back here and
// causes infinite recursion -> stack overflow. at::native:: are the raw stride
// implementations that operate on metadata without re-dispatching.
at::Tensor narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length) {
  // at::native::narrow_symint computes the slice bounds and calls at::slice_symint,
  // which re-dispatches to the registered flagos slice_tensor (pure metadata, no
  // recursion). Calling self.narrow(...) here would re-enter this same kernel via
  // PrivateUse1 -> infinite recursion -> stack overflow (SIGSEGV).
  return at::native::narrow_symint(self, dim, start, length);
}

at::Tensor transpose_int(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  return at::native::transpose(self, dim0, dim1);
}

at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims) {
  return at::native::permute(self, dims);
}

at::Tensor select_int(const at::Tensor& self, int64_t dim, int64_t index) {
  return at::native::select_symint(self, dim, index);
}

at::Tensor slice_tensor(const at::Tensor& self, int64_t dim, ::std::optional<int64_t> start, ::std::optional<int64_t> end, int64_t step) {
  return at::native::slice(self, dim, start, end, step);
}

at::Tensor squeeze(const at::Tensor& self) {
  return at::native::squeeze(self);
}

at::Tensor squeeze_dim(const at::Tensor& self, int64_t dim) {
  return at::native::squeeze(self, dim);
}

at::Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  return at::native::unsqueeze(self, dim);
}

at::Tensor unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  return at::native::_unsafe_view(self, size);
}

// unfold constructs a strided view (extracting all sliding windows of `size`
// along `dimension` with the given `step`); it is pure metadata, no GPU kernel.
// at::native::unfold computes the new size/stride and calls as_strided, which
// re-dispatches to the registered flagos as_strided. Without this registration
// PyTorch has no PrivateUse1 unfold impl, and since view ops can't fall back to
// CPU (storage can't be shared across devices) it returns an uninitialized
// tensor -- callers like Tensor.repeat() then read garbage (SIGSEGV downstream).
at::Tensor unfold(const at::Tensor& self, int64_t dimension, int64_t size, int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

at::Tensor detach(const at::Tensor& self) {
  return at::native::detach(self);
}

// alias() returns a view sharing self's storage (pure metadata). at::native::
// alias avoids re-dispatching through PrivateUse1 back into this kernel.
at::Tensor alias(const at::Tensor& self) {
  return at::native::alias(self);
}

// t() is the 2-D (or <=2-D) transpose used by nn.Linear (F.linear does
// input.matmul(weight.t())). Pure metadata, like transpose_int.
at::Tensor t(const at::Tensor& self) {
  return at::native::t(self);
}

// unbind returns views along dim; at::native::unbind builds them via select,
// which we route to at::native::select above (no re-dispatch recursion).
::std::vector<at::Tensor> unbind_int(const at::Tensor& self, int64_t dim) {
  return at::native::unbind(self, dim);
}

// _conj / _neg_view are the lazy math-bit views: an alias of self carrying the
// Conjugate / Negative bit, with storage untouched. Like alias() they are pure
// metadata and need no vendor kernel, but a view op cannot reach cpu_fallback --
// storage is not shareable across devices -- so without a backend registration
// the dispatcher raises "_conj: backend not registered", and every path that
// resolves a math bit (copy_, clone, contiguous, resolve_conj / resolve_neg)
// fails with it. at::native:: avoids re-dispatching back through PrivateUse1.
at::Tensor _conj(const at::Tensor& self) {
  return at::native::_conj(self);
}

at::Tensor _neg_view(const at::Tensor& self) {
  return at::native::_neg_view(self);
}

// view_as_real / view_as_complex reinterpret the last dimension as a complex
// scalar (or the reverse) without touching storage: sizes, strides and dtype are
// recomputed in at::native's ComplexHelper.h and the result is built over self's
// own storage. There is no topsaten entry point for either and there cannot be
// one -- the op moves no data -- so scripts/codegen/codegen_gcu.py cannot
// generate a kernel for them (every category template ends in a topsaten::
// call) and lists them in METADATA_OPS instead, which is what puts them in
// gcu_register.inc and routes them here.
//
// Registering them matters beyond coverage: a view op that reaches cpu_fallback
// is *copied*, so the result silently stops aliasing the input, and the Qwen-
// Image RoPE path calls each of these 488 times per denoise step (19.3 ms and
// 18.5 ms per call at (1, 4096, 24, 128) -- the two largest remaining host
// round trips on the S60). Like the ops above, at::native:: avoids
// re-dispatching back through PrivateUse1 into this same kernel.
//
// No dtype guard: the complex reinterpretation is metadata, so it is correct for
// every real dtype the dispatcher accepts, and the tensor stays on flagos.
at::Tensor view_as_real(const at::Tensor& self) {
  return at::native::view_as_real(self);
}

at::Tensor view_as_complex(const at::Tensor& self) {
  return at::native::view_as_complex(self);
}

// View ops are pure metadata (stride) operations; they route through the
// generated dispatchers but need a backend kernel registered.
//
// The list is emitted once per accelerator whose conf routes these ops to it.
// Ascend has done so since the native backend landed. TsingMicro joins it
// because it has no accelerator kernel for any of them: its non-FlagGems
// implementations *are* the CUDA boxing kernels, and those need a vendor CUDA
// runtime the TX8110 wheel does not link. A view op the conf leaves unlisted
// therefore falls through the FlagGems default to Backend::kFlagGems, whose
// slot has no kernel -- Dispatcher::GetFn degrades that to cuda_fn_, and the
// call dies in torch/cuda with
//
//   NotImplementedError: Could not run 'aten::as_strided' with arguments
//   from the 'CUDA' backend
//
// which is what every slicing index, `mask[:, 0] = 1` and `select` reduce to.
// Routing them to kTsingMicro reaches the same at::native:: metadata helpers
// the Ascend build uses, so the tensor stays on flagos.
//
// NOTE (reachability): REGISTER_IMPL_TO_DISPATCHER only fills this dispatcher's
// backend slot -- it is NOT an m.impl() claim on PrivateUse1. The op still has
// to be claimed by the build's generated registration for the dispatcher to be
// consulted at all, and the two platforms get there differently:
//
//   * Ascend ships its own private registration set,
//     csrc/aten/backends/ascend/generated/ascend_register.inc (374 m.impl, incl.
//     `select.int` and `permute`), included at csrc/aten/register.cc:568 under
//     `#elif defined(USE_ASCEND)`. That claim is what makes the kAscend slots
//     here reachable.
//   * MPS claims only the eight views in
//     csrc/aten/backends/apple/generated/mps_register.inc; the remaining kMps
//     slots stay unreachable because their routes remain `none`.
//   * TsingMicro has NO equivalent .inc. It takes register.cc's `#else` arm:
//     with FLAGOS_BUILD_BOXING=ON the generated/register.inc claims these ops
//     and the kTsingMicro slots here are reachable; with boxing OFF (the
//     in-tree default) only generated/register_flaggems_tsingmicro.inc is
//     included, and it lists the ops the conf routes to `flaggems` -- the
//     `tsingmicro`-routed view ops are not in it. On that build they are
//     unclaimed, reach cpu_fallback, and the handlers below are dead code.
//
// So this registration is correct, and necessary for the boxing-ON TX8110 build
// the fleet measured; it is inert (not harmful) on a boxing-OFF wheel.
#define FLAGOS_REGISTER_METADATA_VIEW_OPS(device)                             \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      TransposeIntFn, transpose_int_dispatcher, device, transpose_int)        \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      PermuteFn, permute_dispatcher, device, permute)                         \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      SelectIntFn, select_int_dispatcher, device, select_int)                 \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      SliceTensorFn, slice_tensor_dispatcher, device, slice_tensor)           \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      SqueezeFn, squeeze_dispatcher, device, squeeze)                         \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      SqueezeDimFn, squeeze_dim_dispatcher, device, squeeze_dim)              \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      UnsqueezeFn, unsqueeze_dispatcher, device, unsqueeze)                   \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      PrivUnsafeViewFn, priv_unsafe_view_dispatcher, device, unsafe_view)     \
  REGISTER_IMPL_TO_DISPATCHER(DetachFn, detach_dispatcher, device, detach)    \
  REGISTER_IMPL_TO_DISPATCHER(TFn, t_dispatcher, device, t)                   \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      UnbindIntFn, unbind_int_dispatcher, device, unbind_int)                 \
  REGISTER_IMPL_TO_DISPATCHER(AliasFn, alias_dispatcher, device, alias)       \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      PrivConjFn, priv_conj_dispatcher, device, _conj)                        \
  REGISTER_IMPL_TO_DISPATCHER(                                                \
      PrivNegViewFn, priv_neg_view_dispatcher, device, _neg_view)

FLAGOS_REGISTER_METADATA_VIEW_OPS(Backend::kAscend)
FLAGOS_REGISTER_METADATA_VIEW_OPS(Backend::kTsingMicro)
#if defined(USE_MPS)
// Qwen3.5 repeatedly reshapes and slices attention tensors. These operations
// only update sizes, strides and storage offsets, so their MPS route returns a
// view of the shared Metal buffer rather than a CPU copy. The MPS registration
// file claims the same metadata-only subset below.
FLAGOS_REGISTER_METADATA_VIEW_OPS(Backend::kApple)
#endif

#undef FLAGOS_REGISTER_METADATA_VIEW_OPS

// The two complex views are registered for GCU only. Ascend has the identical
// gap in its conf (both ops route to `none`), but that platform was not
// revalidated for this change, so its routing is left as it stands rather than
// widened from a measurement taken on the S60.
REGISTER_IMPL_TO_DISPATCHER(
    ViewAsRealFn,
    view_as_real_dispatcher,
    Backend::kEnflame,
    view_as_real)

REGISTER_IMPL_TO_DISPATCHER(
    ViewAsComplexFn,
    view_as_complex_dispatcher,
    Backend::kEnflame,
    view_as_complex)

} // namespace at::native::flagos
