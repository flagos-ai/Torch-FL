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

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagOs, MmKernel)
REGISTER_IMPL_TO_DISPATCHER(MmDtypeFn, mm_dtype_dispatcher, Backend::kFlagOs,
                            MmDtypeKernel)
REGISTER_IMPL_TO_DISPATCHER(
    MmDtypeOutFn, mm_dtype_out_dispatcher, Backend::kFlagOs, MmDtypeOutKernel)
REGISTER_IMPL_TO_DISPATCHER(MmOutFn, mm_out_dispatcher, Backend::kFlagOs, MmOutKernel)

REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kFlagOs, BmmKernel)
REGISTER_IMPL_TO_DISPATCHER(BmmDtypeFn, bmm_dtype_dispatcher, Backend::kFlagOs,
                            BmmDtypeKernel)
REGISTER_IMPL_TO_DISPATCHER(
    BmmDtypeOutFn, bmm_dtype_out_dispatcher, Backend::kFlagOs, BmmDtypeOutKernel)
REGISTER_IMPL_TO_DISPATCHER(BmmOutFn, bmm_out_dispatcher, Backend::kFlagOs, BmmOutKernel)

REGISTER_IMPL_TO_DISPATCHER(
    AddmmFn, addmm_dispatcher, Backend::kFlagOs, AddmmKernel)
REGISTER_IMPL_TO_DISPATCHER(
    AddmmDtypeFn, addmm_dtype_dispatcher, Backend::kFlagOs, AddmmDtypeKernel)
REGISTER_IMPL_TO_DISPATCHER(
    AddmmDtypeOutFn,
    addmm_dtype_out_dispatcher,
    Backend::kFlagOs,
    AddmmDtypeOutKernel)
REGISTER_IMPL_TO_DISPATCHER(
    AddmmOutFn, addmm_out_dispatcher, Backend::kFlagOs, AddmmOutKernel)
REGISTER_IMPL_TO_DISPATCHER(
    AddmmInplaceFn, addmm_inplace_dispatcher, Backend::kFlagOs, AddmmInplaceKernel)

} // namespace at::native::flagos::soft_lowp
