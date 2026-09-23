// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

// Sparse COO support on the SparsePrivateUse1 ("Sparseflagos") dispatch key.
//
// WHY THIS KEY NEEDS ITS OWN REGISTRATION
//
// c10::DispatchKey::SparsePrivateUse1 is the sparse functionality key for the
// PrivateUse1 backend, so a COO tensor built on `device="flagos:0"` carries it
// and nothing else in this plugin serves it.  BackendSelect derives the key from
// the TensorOptions through c10::computeDispatchKey(dtype, layout, device), so
// `torch.sparse_coo_tensor(..., device="flagos:0")` arrives here directly -- the
// dense PrivateUse1 registrations in csrc/aten/register.cc are never consulted,
// even though PrivateUse1 is this key's backend component.  The boxed
// PrivateUse1 cpu_fallback registered there is unreachable from here as well:
// OperatorEntry::computeDispatchTableEntryWithDebug resolves a backend fallback
// by looking up the fallback slot of the exact key it is resolving and does not
// walk the key set down to a lower key's fallback.
//
// Without a kernel behind this key construction fails with
//   Could not run 'aten::_sparse_coo_tensor_with_dims_and_tensors' with
//   arguments from the 'Sparseflagos' backend. ... it is only available for
//   these backends: [PrivateUse1, Meta, SparseCPU, SparseCUDA, ...]
// which is flagos-ai/Torch-FL#294: a COO tensor could not be constructed on
// flagos at all, even though the same call works on CPU and on raw CUDA/DCU.
//
// WHY THE ATEN IMPLEMENTATIONS ARE REGISTERED DIRECTLY
//
// None of these ops needs a vendor kernel.  Construction is metadata work --
// resize_/resize_and_clear_ only set sizes and fill indices_/values_ with
// at::empty(), and alias_into_sparse() shallow-copies the caller's tensors.
// The accessors below return the SparseTensorImpl's own indices_/values_, the
// flag ops flip a bool, and clone/copy move the two dense buffers with the
// dense copy kernels the plugin already claims.  Every dense primitive they
// reach (empty.memory_format, empty_like, resize_, copy_) is already claimed on
// PrivateUse1 for dense flagos tensors, so the ATen implementations run
// unchanged.
//
// The set registered here is drawn from the COO structure surface that upstream
// itself implements with a single device-agnostic symbol shared by SparseCPU
// and SparseCUDA (torchgen's native_functions.yaml lists both keys on one
// dispatch line), plus the two constructors; the handful of members of that
// surface which turn out to compute on values are listed at the end of this
// comment and left out.  Registering these symbols for this key introduces no
// new kernel and no new host/device assumption -- it is the same treatment
// upstream gives SparseMeta, which is likewise a sparse functionality key with
// no device of its own.
//
// The `device` option of the constructors is forwarded untouched rather than
// routed through CUDA and unboxed, which is what the generated kernels do for
// dense results.  That path cannot work for a sparse tensor: the
// DenseTensorImpl-only SetTensorImplDevice in csrc/aten/device_boxing.h
// dereferences the DataPtr of a Storage that a SparseTensorImpl never has
// (it is built by TensorImpl(key_set, data_type, device_opt) with a
// default-constructed Storage, so storage_impl_ is null).
//
// WHAT IS DELIBERATELY NOT REGISTERED
//
// The boundary is the same one that separates the two halves of this file's
// registration: every op registered below reads or moves sizes, the coalesced
// flag, indices and values, and none of them computes on values.  Everything
// outside that boundary is a value kernel and needs its own numerical
// validation rather than a structural registration.
//
// * `_coalesce` has genuinely separate SparseCPU and SparseCUDA implementations
//   and the CPU one is not device-agnostic: _coalesce_sparse_cpu walks
//   `values.data_ptr<scalar_t>()` with at::native::cpublas::axpy/copy, i.e. host
//   code on device memory.  It stays unregistered; `coalesce()` is only served
//   for the cases that short-circuit before dispatch (already coalesced, or
//   nnz < 2).
// * `_to_dense` is counted in that same device-agnostic set, but its ATen
//   implementation is `at::zeros(sizes, options.layout(kStrided)).add_(self)`,
//   so it is a wrapper over the *sparse* `add_` -- the dense-plus-sparse value
//   kernel -- rather than structure work.  Registering it would not make
//   `to_dense()` succeed: it would only move the NotImplementedError from
//   `aten::_to_dense` down to `aten::add_.Tensor`, where the operation that is
//   actually missing is harder to see.  `to_dense()` therefore still reports
//   `aten::_to_dense` as unavailable, and the sparse add family is what has to
//   land for it.  `x.to("cpu")` and `x.clone()` do not need it: they move the
//   two buffers through the dense copy kernels.
// * The sparse arithmetic family (add/mul/div/abs/... , addmm, bmm, softmax,
//   sparse_mask, index_select, sparse-sparse matmul) is out of scope for this
//   change.  Those are value kernels, several of which have real per-backend
//   CUDA implementations, and they need their own numerical validation.
//
// See flagos-ai/Torch-FL#294 for the measured before/after behaviour of every
// operation in this table.

#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/_dimI_native.h>
#include <ATen/ops/_dimV_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/copy_sparse_to_sparse_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/indices_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/values_native.h>
#include <torch/library.h>

namespace at::native::flagos {
namespace {

// Construction.  These are the ops the composite forms
// (`_sparse_coo_tensor_with_dims.out`,
// `_sparse_coo_tensor_with_dims_and_tensors.out`, `sparse_coo_tensor.size`,
// `_sparse_coo_tensor_unsafe`, `empty_sparse`) dispatch into, so they are the
// only constructor kernels this key needs: everything else in the construction
// family carries a CompositeExplicitAutograd kernel, which the dispatcher uses
// as the default backend kernel for every key including this one.
//
// new_with_dims_sparse() and new_with_dims_and_tensor_sparse_symint() live in
// ATen/native/sparse/SparseTensor.cpp and are exported TORCH_API.  They pick the
// dispatch key from the device in the TensorOptions, and
// C10_FORALL_BACKEND_DEVICE_TYPES includes PrivateUse1, so `device=flagos:0`
// resolves to SparsePrivateUse1 and builds a SparseTensorImpl keyed for flagos.
//
// Delegate to these rather than to at::_sparse_coo_tensor_* / at::empty_sparse:
// those re-enter the dispatcher on the key being registered here.

TORCH_LIBRARY_IMPL(aten, SparsePrivateUse1, m) {
  m.impl(
      "_sparse_coo_tensor_with_dims", at::native::new_with_dims_sparse);
  m.impl(
      "_sparse_coo_tensor_with_dims_and_tensors",
      at::native::new_with_dims_and_tensor_sparse_symint);

  // Structure queries: these read the SparseTensorImpl's own metadata
  // (sparse_dim, dense_dim, nnz, the coalesced flag) and never touch a buffer.
  m.impl("sparse_dim", at::native::sparse_dim_sparse);
  m.impl("dense_dim", at::native::dense_dim_sparse);
  m.impl("_dimI", at::native::sparse_dim_sparse);
  m.impl("_dimV", at::native::dense_dim_sparse);
  m.impl("_nnz", at::native::_nnz_sparse);
  m.impl("is_coalesced", at::native::is_coalesced_sparse);
  m.impl("_coalesced_", at::native::_coalesced_sparse_);

  // Index/value access.  `indices()`/`values()` are the checked forms the
  // Tensor API exposes (they require a coalesced tensor); `_indices()` and
  // `_values()` are the unchecked ones the rest of ATen builds on.
  m.impl("_indices", at::native::_indices_sparse);
  m.impl("_values", at::native::_values_sparse);
  m.impl("indices", at::native::indices_sparse);
  m.impl("values", at::native::values_sparse);

  // Buffer movement.  All three move the indices_/values_ tensors with the dense
  // kernels, so they are correct on whatever device the sparse tensor is tagged
  // for; none of them computes on values.
  m.impl("clone", at::native::clone_sparse);
  m.impl("copy_", at::native::copy_sparse_wrapper_);
  m.impl("copy_sparse_to_sparse_", at::native::copy_sparse_);
}

} // namespace
} // namespace at::native::flagos
