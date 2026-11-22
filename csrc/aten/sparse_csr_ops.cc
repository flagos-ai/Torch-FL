// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

// Sparse compressed (CSR/CSC/BSR/BSC) support on the SparseCsrPrivateUse1
// ("SparseCsrflagos") dispatch key.
//
// WHY THIS KEY NEEDS ITS OWN REGISTRATION
//
// c10::DispatchKey::SparseCsrPrivateUse1 is the sparse compressed functionality
// key for the PrivateUse1 backend.  BackendSelect derives the key from the
// TensorOptions through c10::computeDispatchKey(dtype, layout, device), so
// `torch.sparse_csr_tensor(..., device="flagos:0")` carries it and nothing else
// in this plugin serves it: the dense PrivateUse1 registrations in
// csrc/aten/register.cc are never consulted, even though PrivateUse1 is this
// key's backend component.  The boxed PrivateUse1 cpu_fallback registered there
// is unreachable from here as well --
// OperatorEntry::computeDispatchTableEntryWithDebug resolves a backend fallback
// by looking up the fallback slot of the exact key it is resolving and does not
// walk the key set down to a lower key's fallback.  Unlike the COO key handled
// in csrc/aten/sparse_ops.cc, most of this surface does already have a
// CompositeExplicitAutograd kernel, and that default is what the caller reaches
// instead of a kernel: `crow_indices_default` raises
//   crow_indices expected sparse row compressed tensor layout but got SparseCsr
// and `empty.memory_format` has no default at all, so allocation fails with
//   Could not run 'aten::empty.memory_format' with arguments from the
//   'SparseCsrflagos' backend.
// which is flagos-ai/Torch-FL#293: compressed sparse construction succeeds on
// flagos but every accessor, allocation and SpMM on top of it fails, while the
// same calls work on CPU and on raw CUDA/DCU.
//
// WHY THE ATEN IMPLEMENTATIONS ARE REGISTERED DIRECTLY
//
// The set registered here is drawn from the compressed sparse surface that
// upstream itself implements with a single device-agnostic symbol shared by
// SparseCsrCPU and SparseCsrCUDA (torchgen's native_functions.yaml lists both
// keys on one dispatch line, and often SparseCsrMeta as a third).
//
// * Accessors (crow_indices/col_indices/ccol_indices/row_indices/values,
//   sparse_dim/dense_dim/_nnz) return the SparseCsrTensorImpl's own member
//   tensors and sizes.  They read metadata and never compute on values.
// * Allocation and shape (empty.memory_format, empty_like, clone, copy_,
//   resize_, resize_as_sparse_, zero_) allocate the three member tensors with
//   at::empty and move them with the dense kernels the plugin already claims.
// * Conversions (_to_sparse_csr/_to_sparse_csc/_to_sparse_bsr/_to_sparse_bsc/
//   _to_sparse, and _to_dense) rebuild the member tensors and go through
//   `_sparse_compressed_tensor_unsafe`, whose device option is forwarded
//   untouched, so they stay on flagos.
//
// Conversions that flip the compressed dimension require a PrivateUse1
// kernel for ATen's flatten_indices_stub. Torch-FL does not supply one.
//
// WHAT IS DELIBERATELY NOT REGISTERED
//
// The boundary is again "structure work, not value work".  Everything outside
// it is a value kernel and needs its own numerical validation:
//
// * `_sparse_csr_sum.dim_dtype_out` / `_sparse_csr_prod.dim_dtype_out` have
//   genuinely separate SparseCsrCPU and SparseCsrCUDA implementations and no
//   SparseCsrMeta one, i.e. they are real per-device kernels rather than a
//   shared device-agnostic symbol.
// * `torch.sparse.mm(..., reduce="sum")` reaches
//   `_sparse_mm_reduce_impl_sparse_csr_cpu`, a CPU-only kernel in v2.10; it is
//   out of scope for this change and is tracked separately.
// * The sparse arithmetic family (add/mul/div/abs/..., addmm with a sparse
//   self, bmm, softmax, sparse_mask, index_select, sparse-sampled matmul,
//   sparse-sparse matmul into a sparse result) is not touched.  Several of
//   those have real per-backend CUDA implementations and need their own
//   numerical validation.
// * `_convert_indices_from_csr_to_coo` is reached from `_to_dense` on dense
//   flagos index tensors and is served by the PrivateUse1 fallback like any
//   other dense op, so it needs no registration here.
//
// See flagos-ai/Torch-FL#293 for the measured before/after behaviour of every
// operation in this table.

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/_to_sparse_bsc_native.h>
#include <ATen/ops/_to_sparse_bsr_native.h>
#include <ATen/ops/_to_sparse_csc_native.h>
#include <ATen/ops/_to_sparse_csr_native.h>
#include <ATen/ops/_to_sparse_native.h>
#include <ATen/ops/ccol_indices_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/col_indices_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/crow_indices_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_indices_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/zero_native.h>
#include <torch/library.h>


namespace at::native::flagos {
namespace {

// `_to_sparse` and `_to_sparse.sparse_dim` are one ATen implementation each
// (sparse_compressed_to_sparse has two overloads, dense_to_sparse and
// sparse_coo_to_sparse three apiece), so the overload to register has to be
// named explicitly.
using ToSparseFn = Tensor (*)(
    const Tensor&,
    std::optional<Layout>,
    at::OptionalIntArrayRef,
    std::optional<int64_t>);
using ToSparseSparseDimFn = Tensor (*)(const Tensor&, int64_t);

} // namespace

// Everything down to here is ATen native code that reaches no vendor library,
// so it is registered on every platform: the accessors read the sparse impl's
// own metadata, and allocation, buffer movement and the layout conversions
// rebuild the member tensors with at::empty and the dense kernels, on whatever
// device the sparse tensor is tagged for.
//
TORCH_LIBRARY_IMPL(aten, SparseCsrPrivateUse1, m) {
  // Structure queries.  These read the SparseCsrTensorImpl's own metadata
  // (sparse_dim is 2 for every compressed layout, dense_dim and nnz come from
  // the member tensors' shapes) and never touch a buffer.
  m.impl("sparse_dim", at::native::sparse_dim_sparse_csr);
  m.impl("dense_dim", at::native::dense_dim_sparse_csr);
  m.impl("_nnz", at::native::_nnz_sparse_csr);

  // Index/value access.  crow/col are the row-compressed pair (CSR, BSR) and
  // ccol/row the column-compressed pair (CSC, BSC); the ATen implementations
  // pick the member tensor for the layout and raise the layout error the
  // CompositeExplicitAutograd defaults used to raise for every layout.
  m.impl("crow_indices", at::native::crow_indices_sparse_csr);
  m.impl("col_indices", at::native::col_indices_sparse_csr);
  m.impl("ccol_indices", at::native::ccol_indices_sparse_csr);
  m.impl("row_indices", at::native::row_indices_sparse_csr);
  m.impl("values", at::native::values_sparse_csr);

  // Allocation and buffer movement.  All of these build or move the three
  // member tensors with at::empty and the dense kernels, so they are correct on
  // whatever device the sparse tensor is tagged for; none of them computes on
  // values.
  m.impl("empty.memory_format", at::native::empty_sparse_compressed);
  m.impl("empty_like", at::native::empty_like_sparse_csr);
  m.impl("clone", at::native::clone_sparse_compressed);
  m.impl("copy_", at::native::copy_sparse_compressed_);
  m.impl("resize_", at::native::resize_sparse_csr_);
  m.impl("resize_as_sparse_", at::native::resize_as_sparse_compressed_);
  m.impl("zero_", at::native::zero_sparse_csr_);

  // Layout conversions.  Each rebuilds the member tensors for the target layout
  // and forwards the tensor's own options to
  // `_sparse_compressed_tensor_unsafe`, whose CompositeImplicitAutograd kernel
  // picks the dispatch key from the device option, so the result stays a flagos
  // sparse tensor.
  m.impl("_to_sparse_csr", at::native::sparse_compressed_to_sparse_csr);
  m.impl("_to_sparse_csc", at::native::sparse_compressed_to_sparse_csc);
  m.impl("_to_sparse_bsr", at::native::sparse_compressed_to_sparse_bsr);
  m.impl("_to_sparse_bsc", at::native::sparse_compressed_to_sparse_bsc);
  m.impl(
      "_to_sparse",
      static_cast<ToSparseFn>(at::native::sparse_compressed_to_sparse));
  m.impl(
      "_to_sparse.sparse_dim",
      static_cast<ToSparseSparseDimFn>(
          at::native::sparse_compressed_to_sparse));

  // Densification.  sparse_compressed_to_dense lays the values into a dense
  // zeros tensor through `_convert_indices_from_csr_to_coo` and `index_add_`,
  // all of which act on the dense operand and are already served for flagos.
  m.impl("_to_dense", at::native::sparse_compressed_to_dense);

}

} // namespace at::native::flagos
