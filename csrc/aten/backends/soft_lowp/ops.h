// Copyright 2026 FlagOS Contributors
//
// Device-side software emulation for low-precision matrix operations.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

namespace at::native::flagos::soft_lowp {

at::Tensor Mm(const at::Tensor& self, const at::Tensor& mat2);
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
