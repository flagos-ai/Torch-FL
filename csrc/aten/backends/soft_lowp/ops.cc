// Copyright 2026 FlagOS Contributors
//
// Device-side software emulation for low-precision matrix operations.

#include "ops.h"

#include "convert.h"
#include "format.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/where.h>
#include <ATen/native/Resize.h>
#include <c10/util/Exception.h>
#include <vector>

#include "../../device_boxing.h"
#include "../../../include/flagos.h"

#if defined(USE_DCU)
#include <cuda_runtime.h>
#endif

#if defined(USE_DCU)
namespace {
using ::Stream_t;
}
#endif

namespace at::native::flagos::soft_lowp {
namespace {

at::Tensor DecodeFp4Nibbles(const at::Tensor& raw) {
  // Decode FP4 nibbles using arithmetic operations.
  // FP4 format: seem (sign, 2-bit exponent, 1-bit mantissa)
  auto decode = [](const at::Tensor& nibble) {
    // Extract fields as bfloat16 for computation
    auto exponent = at::bitwise_and(
                        at::bitwise_right_shift(nibble, at::Scalar(1)),
                        at::Scalar(3))
                        .to(c10::kBFloat16);
    auto mantissa = at::bitwise_and(nibble, at::Scalar(1)).to(c10::kBFloat16);

    // Normal values: 2^(e-1) * (1 + m*0.5)
    auto normal = at::exp2(exponent - 1) *
                  (1 + mantissa * static_cast<double>(0.5));
    // Subnormal (e==0): m*0.5
    auto subnormal = mantissa * static_cast<double>(0.5);
    auto magnitude = at::where(
        at::eq(exponent, static_cast<double>(0)), subnormal, normal);

    // Apply sign bit (bit 3)
    auto negative = at::ne(at::bitwise_and(nibble, at::Scalar(8)),
                           static_cast<int64_t>(0));
    return at::where(negative, -magnitude, magnitude);
  };

  // Extract low and high nibbles
  auto low = at::bitwise_and(raw, at::Scalar(0x0F));
  auto high = at::bitwise_right_shift(raw, at::Scalar(4));

  // Decode both and stack along last dimension
  auto decoded_low = decode(low);
  auto decoded_high = decode(high);
  return at::stack({decoded_low, decoded_high}, -1);
}

at::Tensor DecodeMatrix(const at::Tensor& input, bool packed_dim0) {
  if (input.scalar_type() == c10::kFloat4_e2m1fn_x2) {
    auto raw = input.view(c10::kByte);
    auto unpacked = DecodeFp4Nibbles(raw);
    if (packed_dim0) {
      // [K / 2, N] -> [K, N]
      auto shape = input.sizes().vec();
      shape[0] *= 2;
      return unpacked.reshape(shape);
    }
    // [M, K / 2] -> [M, K]
    auto shape = input.sizes().vec();
    shape.back() *= 2;
    return unpacked.reshape(shape);
  }

  // FP8 handled via boxing
  auto result = at::empty(input.sizes(), input.options().dtype(c10::kBFloat16));
  if (IsFp8(input.scalar_type())) {
    at::Tensor boxed = input;
    at::Tensor boxed_result = result;
    SetTensorDevice(boxed, c10::DeviceType::CUDA);
    SetTensorDevice(boxed_result, c10::DeviceType::CUDA);
    auto decoded = at::native::copy_(boxed_result, boxed, false);
    UnboxToFlagos(decoded);
    UnboxToFlagos(boxed);
    return decoded;
  }

  return result.copy_(input);
}

at::Tensor DecodeBatchMatrix(const at::Tensor& input, bool packed_dim1) {
  if (input.scalar_type() == c10::kFloat4_e2m1fn_x2) {
    auto raw = input.view(c10::kByte);
    auto unpacked = DecodeFp4Nibbles(raw);
    auto shape = input.sizes().vec();
    if (packed_dim1) {
      // [B, K / 2, N] -> [B, K, N]
      shape[1] *= 2;
      return unpacked.reshape(shape);
    }
    // [B, M, K / 2] -> [B, M, K]
    shape.back() *= 2;
    return unpacked.reshape(shape);
  }

  // FP8 handled via boxing
  auto result = at::empty(input.sizes(), input.options().dtype(c10::kBFloat16));
  if (IsFp8(input.scalar_type())) {
    at::Tensor boxed = input;
    at::Tensor boxed_result = result;
    SetTensorDevice(boxed, c10::DeviceType::CUDA);
    SetTensorDevice(boxed_result, c10::DeviceType::CUDA);
    auto decoded = at::native::copy_(boxed_result, boxed, false);
    UnboxToFlagos(decoded);
    UnboxToFlagos(boxed);
    return decoded;
  }

  return result.copy_(input);
}

bool HasLowpInput(const at::Tensor& tensor) {
  return tensor.defined() && IsLowPrecision(tensor.scalar_type());
}

void CheckDevice(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.device().is_privateuseone(),
      "soft-lowp ",
      name,
      " requires PrivateUse1 tensors, got ",
      tensor.device());
}

void CheckMatrixInputs(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(HasLowpInput(self) || HasLowpInput(mat2),
              "soft-lowp matrix kernel requires a low-precision input");
  CheckDevice(self, "matrix operation");
  CheckDevice(mat2, "matrix operation");
  TORCH_CHECK(self.is_contiguous() && mat2.is_contiguous(),
              "soft-lowp matrix kernel requires contiguous inputs");
}

at::Tensor DecodeIfNeeded(const at::Tensor& input, bool packed_dim0 = false) {
  if (!IsLowPrecision(input.scalar_type())) {
    return input;
  }
  TORCH_CHECK(input.dim() == 2,
              "soft-lowp matrix decode supports only rank-2 tensors");
  if (input.numel() == 0) {
    return at::empty(input.sizes(), input.options().dtype(c10::kBFloat16));
  }
  return DecodeMatrix(input, packed_dim0);
}

at::Tensor DecodeBatchIfNeeded(const at::Tensor& input, bool packed_dim1 = false) {
  if (!IsLowPrecision(input.scalar_type())) {
    return input;
  }
  TORCH_CHECK(input.dim() == 3,
              "soft-lowp batch matrix decode supports only rank-3 tensors");
  if (input.numel() == 0) {
    return at::empty(input.sizes(), input.options().dtype(c10::kBFloat16));
  }
  return DecodeBatchMatrix(input, packed_dim1);
}

c10::ScalarType DefaultOutputDtype(const at::Tensor& self, c10::ScalarType requested) {
  if (requested != c10::ScalarType::Undefined) {
    return requested;
  }
  return IsLowPrecision(self.scalar_type()) ? c10::kBFloat16 : self.scalar_type();
}

} // namespace

namespace {

at::Tensor MmImpl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  CheckMatrixInputs(self, mat2);
  auto lhs = DecodeIfNeeded(self, false);
  auto rhs = DecodeIfNeeded(mat2, true);
  auto result = at::mm(lhs, rhs);
  const auto output_dtype = DefaultOutputDtype(self, out_dtype);
  return output_dtype == result.scalar_type() ? result : result.to(output_dtype);
}

at::Tensor BmmImpl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  CheckMatrixInputs(self, mat2);
  auto lhs = DecodeBatchIfNeeded(self, false);
  auto rhs = DecodeBatchIfNeeded(mat2, true);
  auto result = at::bmm(lhs, rhs);
  const auto output_dtype = DefaultOutputDtype(self, out_dtype);
  return output_dtype == result.scalar_type() ? result : result.to(output_dtype);
}

at::Tensor AddmmImpl(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    c10::ScalarType out_dtype) {
  CheckMatrixInputs(mat1, mat2);
  auto bias = self;
  CheckDevice(bias, "addmm");
  if (IsLowPrecision(bias.scalar_type())) {
    bias = DecodeIfNeeded(bias, false);
  }
  auto lhs = DecodeIfNeeded(mat1, false);
  auto rhs = DecodeIfNeeded(mat2, true);
  auto result = at::addmm(bias, lhs, rhs, beta, alpha);
  const auto output_dtype = DefaultOutputDtype(mat1, out_dtype);
  return output_dtype == result.scalar_type() ? result : result.to(output_dtype);
}

} // namespace

at::Tensor Mm(const at::Tensor& self, const at::Tensor& mat2) {
  return MmImpl(self, mat2, c10::ScalarType::Undefined);
}

at::Tensor MmDtype(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  return MmImpl(self, mat2, out_dtype);
}

at::Tensor& MmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out) {
  auto result = MmImpl(self, mat2, out_dtype);
  TORCH_CHECK(out.sizes() == result.sizes(), "soft-lowp mm.dtype_out shape mismatch");
  out.copy_(result);
  return out;
}

at::Tensor& MmOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  auto result = MmImpl(self, mat2, out.scalar_type());
  TORCH_CHECK(out.sizes() == result.sizes(), "soft-lowp mm.out shape mismatch");
  out.copy_(result);
  return out;
}

at::Tensor Bmm(const at::Tensor& self, const at::Tensor& mat2) {
  return BmmImpl(self, mat2, c10::ScalarType::Undefined);
}

at::Tensor BmmDtype(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  return BmmImpl(self, mat2, out_dtype);
}

at::Tensor& BmmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out) {
  auto result = BmmImpl(self, mat2, out_dtype);
  TORCH_CHECK(out.sizes() == result.sizes(), "soft-lowp bmm.dtype_out shape mismatch");
  out.copy_(result);
  return out;
}

at::Tensor& BmmOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  auto result = BmmImpl(self, mat2, out.scalar_type());
  TORCH_CHECK(out.sizes() == result.sizes(), "soft-lowp bmm.out shape mismatch");
  out.copy_(result);
  return out;
}

at::Tensor Addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return AddmmImpl(self, mat1, mat2, beta, alpha, c10::ScalarType::Undefined);
}

at::Tensor AddmmDtype(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return AddmmImpl(self, mat1, mat2, beta, alpha, out_dtype);
}

at::Tensor& AddmmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto result = AddmmImpl(self, mat1, mat2, beta, alpha, out_dtype);
  TORCH_CHECK(out.sizes() == result.sizes(), "soft-lowp addmm.dtype_out shape mismatch");
  out.copy_(result);
  return out;
}

at::Tensor& AddmmOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto result = AddmmImpl(self, mat1, mat2, beta, alpha, out.scalar_type());
  TORCH_CHECK(out.sizes() == result.sizes(), "soft-lowp addmm.out shape mismatch");
  out.copy_(result);
  return out;
}

at::Tensor& AddmmInplace(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto result = AddmmImpl(self, mat1, mat2, beta, alpha, self.scalar_type());
  self.copy_(result);
  return self;
}

} // namespace at::native::flagos::soft_lowp
