// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "strided_ops.h"
#include "generated/ops.h"

#include <ATen/native/Resize.h>
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

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
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

// View ops are pure metadata (stride) operations; they route through the
// generated dispatchers but need a backend kernel registered. Register them
// for the Ascend backend so the generated wrappers in register.inc resolve.
REGISTER_IMPL_TO_DISPATCHER(
    TransposeIntFn,
    transpose_int_dispatcher,
    Backend::kAscend,
    transpose_int)

REGISTER_IMPL_TO_DISPATCHER(
    PermuteFn,
    permute_dispatcher,
    Backend::kAscend,
    permute)

REGISTER_IMPL_TO_DISPATCHER(
    SelectIntFn,
    select_int_dispatcher,
    Backend::kAscend,
    select_int)

REGISTER_IMPL_TO_DISPATCHER(
    SliceTensorFn,
    slice_tensor_dispatcher,
    Backend::kAscend,
    slice_tensor)

REGISTER_IMPL_TO_DISPATCHER(
    SqueezeFn,
    squeeze_dispatcher,
    Backend::kAscend,
    squeeze)

REGISTER_IMPL_TO_DISPATCHER(
    SqueezeDimFn,
    squeeze_dim_dispatcher,
    Backend::kAscend,
    squeeze_dim)

REGISTER_IMPL_TO_DISPATCHER(
    UnsqueezeFn,
    unsqueeze_dispatcher,
    Backend::kAscend,
    unsqueeze)

REGISTER_IMPL_TO_DISPATCHER(
    PrivUnsafeViewFn,
    priv_unsafe_view_dispatcher,
    Backend::kAscend,
    unsafe_view)

REGISTER_IMPL_TO_DISPATCHER(
    DetachFn,
    detach_dispatcher,
    Backend::kAscend,
    detach)

REGISTER_IMPL_TO_DISPATCHER(
    TFn,
    t_dispatcher,
    Backend::kAscend,
    t)

REGISTER_IMPL_TO_DISPATCHER(
    UnbindIntFn,
    unbind_int_dispatcher,
    Backend::kAscend,
    unbind_int)

REGISTER_IMPL_TO_DISPATCHER(
    AliasFn,
    alias_dispatcher,
    Backend::kAscend,
    alias)

REGISTER_IMPL_TO_DISPATCHER(
    PrivConjFn,
    priv_conj_dispatcher,
    Backend::kAscend,
    _conj)

REGISTER_IMPL_TO_DISPATCHER(
    PrivNegViewFn,
    priv_neg_view_dispatcher,
    Backend::kAscend,
    _neg_view)

} // namespace at::native::flagos
