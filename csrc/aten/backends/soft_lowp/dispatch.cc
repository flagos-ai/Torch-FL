// Copyright 2026 FlagOS Contributors
//
// Dispatch registrations for software-emulated low-precision matrix kernels.

#include "ops.h"

#include "../../generated/ops.h"
#include "../../dispatcher.h"

namespace at::native::flagos::soft_lowp {
namespace {

at::Tensor MmKernel(const at::Tensor& self, const at::Tensor& mat2) {
  return Mm(self, mat2);
}

at::Tensor MmDtypeKernel(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  return MmDtype(self, mat2, out_dtype);
}

at::Tensor& MmDtypeOutKernel(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out) {
  return MmDtypeOut(self, mat2, out_dtype, out);
}

at::Tensor& MmOutKernel(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  return MmOut(self, mat2, out);
}

at::Tensor BmmKernel(const at::Tensor& self, const at::Tensor& mat2) {
  return Bmm(self, mat2);
}

at::Tensor BmmDtypeKernel(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  return BmmDtype(self, mat2, out_dtype);
}

at::Tensor& BmmDtypeOutKernel(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out) {
  return BmmDtypeOut(self, mat2, out_dtype, out);
}

at::Tensor& BmmOutKernel(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  return BmmOut(self, mat2, out);
}

at::Tensor AddmmKernel(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return Addmm(self, mat1, mat2, beta, alpha);
}

at::Tensor AddmmDtypeKernel(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return AddmmDtype(self, mat1, mat2, out_dtype, beta, alpha);
}

at::Tensor& AddmmDtypeOutKernel(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  return AddmmDtypeOut(self, mat1, mat2, out_dtype, beta, alpha, out);
}

at::Tensor& AddmmOutKernel(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  return AddmmOut(self, mat1, mat2, beta, alpha, out);
}

at::Tensor& AddmmInplaceKernel(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return AddmmInplace(self, mat1, mat2, beta, alpha);
}

} // namespace

ADD_IMPL_TO_DISPATCHER(MmFn, mm_fallback_dispatcher, "mm")
ADD_IMPL_TO_DISPATCHER(BmmFn, bmm_fallback_dispatcher, "bmm")

namespace {

at::Tensor OrdinaryMmKernel(const at::Tensor& self, const at::Tensor& mat2) {
  return OrdinaryMm(self, mat2);
}

at::Tensor OrdinaryBmmKernel(const at::Tensor& self, const at::Tensor& mat2) {
  return OrdinaryBmm(self, mat2);
}

// Register mm/bmm/addmm/.out on Backend::kFlagGemsCpp unless another kernel set
// already owns the slot.
//
// Why the software path registers mm/bmm at all: an ordinary-dtype pair does
// not stay in MmImpl/BmmImpl, it is handed back to the conf ("the route that
// names this op's accelerated kernel"). A FLAGOS_SOFT_LOWP build without FlagGems
// C++ kernels (DCU, a CPU-torch build) has no other kernel for mm/bmm, so the
// hand-back needs one: with these wrappers the slot resolves and they compute
// the pair the conventional way; without them it is "backend not registered".
// (An FP8/FP4 call never comes through the dispatcher -- the generated wrapper's
// IsLowPrecision prelude calls soft_lowp::Mm/Bmm directly.)
//
// A FLAGOS_SOFT_LOWP build that also compiles flaggems_cpp_kernels.cc (MetaX's
// MACA liboperators) does own the slot: those registrations write the same
// dispatcher/backend field and must survive. When this file overwrote them,
// `flaggems_cpp` -- which backends_metax.conf names for bmm -- resolved to these
// wrappers, so the hand-back returned to the kernel that made it and recursed
// until the stack ran out: SIGSEGV in torch.bmm on float32 (4, 64, 128) x
// (4, 128, 32), CI run 40242938356. Both sides use REGISTER_IMPL_TO_DISPATCHER,
// so which one registered first was link order, and link order is not a way to
// decide routing.
//
// The fallback dispatchers below are registered in both cases: they are how
// ops.cc recognises that the slot is this file (equal kernel) rather than a
// FlagGems C++ kernel, and on a file-owned slot they remain a route to the
// same computation that cannot come back here.
void RegisterGemmKernels() {
  MmFn mm_fallback_fn = OrdinaryMmKernel;
  BmmFn bmm_fallback_fn = OrdinaryBmmKernel;
  // The file-local *Kernel wrappers above (the public Mm/Bmm/Addmm are
  // RegisteredImpl<T> -- fine to register, they forward here -- but these are
  // direct).
  MmFn mm_fn = MmKernel;
  BmmFn bmm_fn = BmmKernel;
  AddmmFn addmm_fn = AddmmKernel;
  MmOutFn mm_out_fn = MmOutKernel;
  BmmOutFn bmm_out_fn = BmmOutKernel;
  mm_fallback_dispatcher.RegisterKernel(Backend::kFlagGemsCpp, mm_fallback_fn);
  bmm_fallback_dispatcher.RegisterKernel(Backend::kFlagGemsCpp, bmm_fallback_fn);
  if (mm_dispatcher.HasKernel(Backend::kFlagGemsCpp) ||
      bmm_dispatcher.HasKernel(Backend::kFlagGemsCpp) ||
      addmm_dispatcher.HasKernel(Backend::kFlagGemsCpp) ||
      mm_out_dispatcher.HasKernel(Backend::kFlagGemsCpp) ||
      bmm_out_dispatcher.HasKernel(Backend::kFlagGemsCpp)) {
    return;
  }
  mm_dispatcher.RegisterKernel(Backend::kFlagGemsCpp, mm_fn);
  bmm_dispatcher.RegisterKernel(Backend::kFlagGemsCpp, bmm_fn);
  addmm_dispatcher.RegisterKernel(Backend::kFlagGemsCpp, addmm_fn);
  mm_out_dispatcher.RegisterKernel(Backend::kFlagGemsCpp, mm_out_fn);
  bmm_out_dispatcher.RegisterKernel(Backend::kFlagGemsCpp, bmm_out_fn);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(MmDtypeFn, mm_dtype_dispatcher, Backend::kFlagGemsCpp,
                            MmDtypeKernel)
REGISTER_IMPL_TO_DISPATCHER(
    MmDtypeOutFn, mm_dtype_out_dispatcher, Backend::kFlagGemsCpp, MmDtypeOutKernel)

REGISTER_IMPL_TO_DISPATCHER(BmmDtypeFn, bmm_dtype_dispatcher, Backend::kFlagGemsCpp,
                            BmmDtypeKernel)
REGISTER_IMPL_TO_DISPATCHER(
    BmmDtypeOutFn, bmm_dtype_out_dispatcher, Backend::kFlagGemsCpp, BmmDtypeOutKernel)

REGISTER_IMPL_TO_DISPATCHER(
    AddmmDtypeFn, addmm_dtype_dispatcher, Backend::kFlagGemsCpp, AddmmDtypeKernel)
REGISTER_IMPL_TO_DISPATCHER(
    AddmmDtypeOutFn,
    addmm_dtype_out_dispatcher,
    Backend::kFlagGemsCpp,
    AddmmDtypeOutKernel)
REGISTER_IMPL_TO_DISPATCHER(
    AddmmOutFn, addmm_out_dispatcher, Backend::kFlagGemsCpp, AddmmOutKernel)
REGISTER_IMPL_TO_DISPATCHER(
    AddmmInplaceFn, addmm_inplace_dispatcher, Backend::kFlagGemsCpp, AddmmInplaceKernel)

} // namespace at::native::flagos::soft_lowp
