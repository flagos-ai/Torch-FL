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
// The one op that is not structure work is the sparse matrix multiply:
// `mm`/`mm.out` and `addmm`/`addmm.out`.  Those are handled in the second block
// below by boxing the operands into the CUDA key frame and delegating to the
// cuSPARSE implementations, in the same way the generated dense kernels box.
//
// The CSC<->CSR conversion also needs one DispatchStub slot that is set on the
// stub object rather than through the dispatcher; see the block on
// flatten_indices_stub below.
//
// This is also why a sparse tensor is boxed differently from a dense one.
// DeviceBoxingGuard rewrites a tensor's Storage DataPtr, and a
// SparseCsrTensorImpl is built without a Storage at all -- upstream constructs
// it as TensorImpl(key_set, data_type, device) and only ever holds three dense
// member tensors -- so the sparse tensor has no DataPtr of its own to rewrite
// and the device the cuSPARSE path reads is the one on its own impl.  The
// storage half of the rewrite therefore has to be skipped for it; see
// SetTensorImplDevice in csrc/aten/device_boxing.h.
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

#include <ATen/native/DispatchStub.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/_to_sparse_bsc_native.h>
#include <ATen/ops/_to_sparse_bsr_native.h>
#include <ATen/ops/_to_sparse_csc_native.h>
#include <ATen/ops/_to_sparse_csr_native.h>
#include <ATen/ops/_to_sparse_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/ccol_indices_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/col_indices_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/crow_indices_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_indices_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/zero_native.h>
#include <torch/library.h>

#include "device_boxing.h"

// flatten_indices_stub, the one ATen entry point on this path that is not
// reachable through the dispatcher.
//
// `_to_sparse_csr`/`_to_sparse_csc` and their BSR/BSC counterparts all go
// through sparse_compressed_to_flipped, which for a 2-D compressed tensor
// rebuilds the flipped layout's index tensors and hashes them with
// at::sparse::flatten_indices on a *dense* flagos int64 tensor.  That helper
// forwards to at::native::flatten_indices_stub -- an instruction-set dispatch
// stub with a CPU kernel and a CUDA kernel and no PrivateUse1 entry -- and
// DispatchStubImpl::get_call_ptr internal-asserts on a device that has no
// kernel rather than falling back to another device's, so the conversion dies
// with
//   DispatchStub: missing kernel for flagos
// even though every op below it on the path is registered and works.
//
// ATen reserves a PrivateUse1 slot on every DispatchStub for exactly this
// situation (set_privateuse1_dispatch_ptr, reachable from the
// REGISTER_PRIVATEUSE1_DISPATCH macro), so the slot registered below is what
// turns the assert into a call.  Filling the slot needs the stub's *type*
// only, which is why the stub is redeclared here: upstream declares it in
// ATen/native/sparse/SparseStubs.h, and the torch wheel ships no
// torch/include/ATen/native/sparse/ at all.  The redeclaration below is
// upstream's own DECLARE_DISPATCH line and binds to the single exported
// `at::native::flatten_indices_stub` object in libtorch_cpu.so.
//
// The slot's kernel then re-enters *the same helper* with the index tensor
// boxed into the CUDA key frame, so the stub is stepped on by libtorch_cpu's
// own inlined copy rather than by this translation unit, and the CUDA kernel
// (FlattenIndicesCommon.h: an arange plus a hash over the index tensor's raw
// pointers, device-agnostic) runs as it does for raw CUDA/DCU.  Calling the
// stub from here instead would mean naming DispatchStub::operator(), which is
// an inline that passes a build-internal set of CPU-capability arguments to
// DispatchStubImpl::get_call_ptr -- that header is not self-contained across
// the wheel boundary and would leave an undefined symbol at import time.  The
// result is allocated inside the boxed frame and is unboxed before it is
// returned, because sparse_compressed_to_flipped goes straight on to `.sort()`
// and `index_select` on flagos tensors and would find a CUDA-tagged operand
// there.
namespace at::native {

using flatten_indices_fn = Tensor (*)(const Tensor& indices, IntArrayRef size);
DECLARE_DISPATCH(flatten_indices_fn, flatten_indices_stub)

} // namespace at::native

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
// The two registrations below are the exception.  addmm.out with a
// compressed-sparse operand computes the product with cuSPARSE, and the CUDA
// implementation is the only one this plugin can reach; flatten_indices_stub's
// PrivateUse1 slot delegates to the stub's CUDA kernel.  Both therefore need
// the CUDA-compatible libtorch (libtorch_hip.so / libtorch_cuda.so, i.e. the
// cuSPARSE symbols and the stub's CUDA slot) that the CUDA-boxing platforms
// link and the others do not: Ascend, Enflame GCU, MUSA and BPU.  On those the
// guards keep an undefined cuSPARSE symbol out of the shared library, and a
// PrivateUse1 slot that could only assert on its way to CUDA out of the stub,
// so the ops stay unimplemented there -- the same
// "DispatchStub: missing kernel" / "Could not run" errors the stock wheel
// raises -- instead of half-registered.  This is the same predicate
// csrc/aten/sdp_choice_stub.cc uses, for the same reason.
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_MUSA) && \
    !defined(USE_BPU)

namespace {

// A DispatchStub slot is set on the stub object, not through the dispatcher, so
// this kernel and its registrar sit at namespace scope rather than among the
// m.impl calls below.  The macro resolves the unqualified stub and the
// `flatten_indices_stub_DECLARE_DISPATCH_type` it names out of the enclosing
// at::native, which is where the redeclaration above put them.
Tensor flatten_indices_flagos(const Tensor& indices, IntArrayRef size) {
  DeviceBoxingGuard guard(indices);
  auto flattened = at::sparse::flatten_indices(indices, size);
  UnboxToFlagos(flattened);
  return flattened;
}

// Sparse matrix multiply.  addmm.out is the only kernel here that has to leave
// the key it is registered on.
//
// addmm.out with a compressed-sparse operand is not structure work: the
// SparseCsrCPU/CUDA kernels compute the product with a library SpMM, and the
// CUDA one is the only implementation available on this platform.  It begins
// with sparse::impl::_check_is_cuda on self, mat1, mat2 and result, and builds
// cuSPARSE descriptors from the sparse operand's member tensors, so every
// operand has to read as CUDA for the duration of the call.
//
// DeviceBoxingGuard does that for the dense operands (these are storage-backed,
// so the ordinary in-place rewrite applies) and, after the storage skip in
// SetTensorImplDevice, for the compressed-sparse operand too.  Which of
// mat1/mat2 is the sparse one depends on the call, and a sparse-sparse product
// produces a sparse result, so all four operands are handed to the guard and
// each is retagged according to what it actually is.
//
// The result's own device metadata matters here as well: the CUDA kernel
// resizes it with resize_output/resize_as_sparse_ when it does not match, and
// both of those run under the CUDA key for the duration of the call.
Tensor& addmm_out_sparse_compressed_flagos(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  DeviceBoxingGuard guard(self, mat1, mat2, result);
  return at::native::addmm_out_sparse_compressed_cuda(
      self, mat1, mat2, beta, alpha, result);
}

} // namespace

REGISTER_PRIVATEUSE1_DISPATCH(flatten_indices_stub, &flatten_indices_flagos)

#endif // CUDA-boxing builds

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

  // Sparse matrix multiply.  `mm`/`addmm` are structured delegates of their
  // `.out` forms, and on this key the structured meta function is unreachable
  // in the form upstream implements it -- it allocates its output through the
  // very `empty.memory_format` this key had no kernel for, and it runs before
  // any backend kernel would.  Registering the op itself resolves it on the key
  // directly, the same way upstream serves SparseCsrCPU/CUDA/Meta.
  //
  // The four travel with addmm.out and not without it: upstream implements mm,
  // _sparse_csr_mm, _sparse_csr_mm_out and addmm_sparse_compressed_dense as
  // delegation to at::addmm_out plus a result allocation, so registering them
  // on a platform where addmm.out is not compiled would only move the failure
  // one frame, and the CSC inputs _sparse_csr_mm accepts would reach
  // _to_sparse_csr first anyway.
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_MUSA) && \
    !defined(USE_BPU)
  m.impl("mm", at::native::_sparse_csr_mm);
  m.impl("mm.out", at::native::_sparse_csr_mm_out);
  m.impl("addmm", at::native::addmm_sparse_compressed_dense);
  m.impl("addmm.out", addmm_out_sparse_compressed_flagos);
#endif // CUDA-boxing builds
}

} // namespace at::native::flagos
