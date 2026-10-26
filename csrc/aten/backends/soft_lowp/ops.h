// Copyright 2026 FlagOS Contributors
//
// Device-side software emulation for low-precision matrix operations.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

#include "../../dispatcher.h"
// Defines the MmFn/BmmFn types the fallback dispatcher declarations below use.
#include "../../generated/ops.h"

namespace at::native::flagos::soft_lowp {

// Suppress the low-precision wrapper while a decoded operation invokes its
// original ATen composite. This preserves the single fused addmm operation
// without recursively re-entering the software low-precision implementation.
bool IsDispatchSuppressed();

class DispatchSuppressionGuard {
 public:
  DispatchSuppressionGuard();
  ~DispatchSuppressionGuard();

  DispatchSuppressionGuard(const DispatchSuppressionGuard&) = delete;
  DispatchSuppressionGuard& operator=(const DispatchSuppressionGuard&) = delete;

 private:
  bool previous_;
};

at::Tensor Mm(const at::Tensor& self, const at::Tensor& mat2);
// The ordinary-dtype mm/bmm computation, with the conf out of the loop:
// dispatch.cc binds it to the mm_fallback/bmm_fallback dispatchers, which is
// how Dispatcher::DispatchConfiguredForSoftLowp recognises a FlagGems C++ slot
// this file filled itself.
at::Tensor OrdinaryMm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor OrdinaryBmm(const at::Tensor& self, const at::Tensor& mat2);

// Fallback GEMM dispatchers: the unboxed, unregistered counterpart of mm/bmm.
// dispatch.cc registers OrdinaryMm/OrdinaryBmm on Backend::kFlagGemsCpp here
// when (and only when) it owns that slot of mm_dispatcher/bmm_dispatcher; a
// caller then goes through these to the same computation through the same
// PrivateUse1 dispatch path, instead of recursing.
DECLARE_DISPATCHER(MmFn, mm_fallback_dispatcher)
DECLARE_DISPATCHER(BmmFn, bmm_fallback_dispatcher)
at::Tensor MmDtype(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype);
at::Tensor& MmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out);
at::Tensor& MmOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out);

at::Tensor Bmm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor BmmDtype(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype);
at::Tensor& BmmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out);
at::Tensor& BmmOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out);

at::Tensor Addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha);
at::Tensor AddmmDtype(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha);
at::Tensor& AddmmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out);
at::Tensor& AddmmOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out);

at::Tensor& AddmmInplace(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha);

} // namespace at::native::flagos::soft_lowp
