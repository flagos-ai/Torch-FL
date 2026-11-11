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
// * The sparse arithmetic family (add/mul/div/abs/... , addmm, softmax,
//   sparse_mask, index_select, sparse-sparse matmul) is out of scope for this
//   change.  Those are value kernels, several of which have real per-backend
//   CUDA implementations, and they need their own numerical validation.
//
// One member of that family has since been added, in the second block below:
// `bmm` with a sparse COO left operand (flagos-ai/Torch-FL#793).  It is a value
// kernel, so it is registered where a value kernel can actually run -- by
// boxing the operands onto the vendor's CUDA kernels, the same way
// csrc/aten/sparse_csr_ops.cc serves the compressed-sparse SpMM -- and not on
// the structure-surface reasoning above.  Nothing else of the family moved.
//
// See flagos-ai/Torch-FL#294 for the measured before/after behaviour of every
// operation in this table.

#include "device_boxing.h"
#include "fallback.h"

#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/_dimI_native.h>
#include <ATen/ops/_dimV_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/bmm.h>
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

// Same predicate as csrc/aten/sparse_csr_ops.cc's SpMM block: the CUDA-ABI
// platforms whose wheel links a libtorch with CUDA kernels, and only when this
// build is a boxing one.  The excluded platforms never put a sparse tensor on
// the CUDA key (Ascend, TsingMicro, GCU, MUSA, BPU, Kunlunxin, Iluvatar, MPS,
// Cambricon), and a FlagGems-primary build has no bmm_sparse_cuda to bind, so
// registering it there would only move the failure from the dispatch table to
// the dlopen.
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU) && !defined(USE_KUNLUNXIN) && \
    !defined(USE_ILUVATAR) && !defined(USE_MPS) && !defined(USE_CAMBRICON) && \
    defined(FLAGOS_BOXING_BUILD)

namespace {

// Serve `aten::bmm(sparse-left, dense-right)` by boxing onto the vendor's CUDA
// kernels. Called only for the served form; everything else is handed to
// cpu_fallback.
void bmm_sparse_flagos(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  at::Tensor& self = torch::jit::peek(stack, 0, 2).toTensor();
  at::Tensor& mat2 = torch::jit::peek(stack, 1, 2).toTensor();
  // `bmm(sparse, dense)` is the whole contract: upstream's
  // bmm_sparse_cpu/_cuda begin with "bmm_sparse: Tensor 'mat2' must be dense",
  // so answering a dense-left/sparse-right call here would invent behaviour
  // the vendor backend does not have.
  //
  // Autograd is declined on `requires_grad`, not on grad mode. The boxed CUDA
  // call would record its backward against the CUDA-keyed operands the guard
  // restores, leaving a caller-visible grad_fn pointing at tensors nobody
  // holds -- so a call that would actually record a graph takes cpu_fallback,
  // exactly as csrc/aten/sdp_choice_stub.cc keeps its boxed route to
  // inference. Grad mode alone is not the test: it is on by default, and the
  // reported case (and most inference under eager) runs with it on and no
  // operand requiring a gradient, where nothing is recorded and the boxed call
  // is safe.
  const bool autograd_needed =
      self.requires_grad() || mat2.requires_grad();
  const bool serves = self.is_sparse() && !mat2.is_sparse() &&
      !mat2.is_sparse_csr() && !autograd_needed;
  if (!serves) {
    at::native::flagos::cpu_fallback(op, stack);
    return;
  }

  at::Tensor result;
  {
    // DeviceBoxingGuard records raw TensorImpl* and must only be handed lvalues
    // that outlive it, which self/mat2 (the caller's own tensors) are. It
    // rewrites both operands' impl key sets -- SparsePrivateUse1 becomes
    // SparseCUDA -- and puts them back on the way out, so the caller's tensors
    // come back exactly as they went in. The sparse operand has no Storage of
    // its own, which SetTensorImplDevice already accounts for.
    at::native::flagos::DeviceBoxingGuard guard(self, mat2);
    // Inside the guard both operands are CUDA, so this re-enters the
    // dispatcher on SparseCUDA and reaches bmm_sparse_cuda; it cannot recurse
    // back here.
    result = at::bmm(self, mat2);
  }
  // The kernel produced a fresh CUDA tensor rather than one of the boxed
  // operands, so the guard does not unbox it.
  at::native::flagos::UnboxToFlagos(result);
  torch::jit::drop(stack, 2);
  torch::jit::push(stack, result);
}

} // namespace

#endif // CUDA-boxing builds

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

  // Sparse COO bmm (#793). A value kernel, so it is boxed onto the vendor's
  // CUDA implementation rather than registered on an ATen symbol the way the
  // structure surface above is; the kernel is defined above this body.
#if !defined(USE_ASCEND) && !defined(USE_TSINGMICRO) && !defined(USE_GCU) && \
    !defined(USE_MUSA) && !defined(USE_BPU) && !defined(USE_KUNLUNXIN) && \
    !defined(USE_ILUVATAR) && !defined(USE_MPS) && !defined(USE_CAMBRICON) && \
    defined(FLAGOS_BOXING_BUILD)
  m.impl(
      "bmm", torch::CppFunction::makeFromBoxedFunction<&bmm_sparse_flagos>());
#endif // CUDA-boxing builds
}

} // namespace

} // namespace at::native::flagos
