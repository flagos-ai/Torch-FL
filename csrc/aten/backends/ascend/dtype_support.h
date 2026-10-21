// Copyright (c) 2026, BAAI. All rights reserved.
//
// Dtype capability helpers for Ascend aclnn kernels.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <c10/core/ScalarType.h>
#include <string_view>

#include "runtime/accelerator/ascend/acl_stream.h"

namespace at::native::flagos::ascend {

// True for the SoCs whose CANN build implements aclnn matmul in fp16 only.
//
// The 310B line (Orange Pi AIpro 20T and friends) is the one measured case: on
// an Ascend310B1, fp32 aclnnMatmul / aclnnMm / aclnnBatchMatMul all fail at
// GetWorkspaceSize, while the same shapes in fp16 succeed, so every fp32 matmul
// has to fall back to the CPU. bf16 fails there too, which the same predicate
// already excludes.
//
// Matched on the "310B" prefix rather than an exhaustive list of exact SoC
// names, because CANN reports board-specific suffixes (Ascend310B1..) that
// change with the product line. An empty name (ACL unavailable) is treated as
// full-featured so a host we cannot classify keeps its existing behaviour.
inline bool IsFp16OnlyMatmulSoc(std::string_view soc_name) {
  // "Ascend310B1", "Ascend310B4", ... -> the "310B" family.
  constexpr std::string_view kPrefix = "Ascend310B";
  return soc_name.substr(0, kPrefix.size()) == kPrefix;
}

// aclnn matmul accepts floating-point inputs up to float32 -- but only on the
// SoCs whose CANN build implements those instantiations. The 310B line is
// fp16-only: aclnnMatmulGetWorkspaceSize rejects float32 outright (161002
// ACLNN_ERR_PARAM_INVALID for cubeMathType 0/3, 561103 ACLNN_ERR_INNER_NULLPTR
// for the HF32 variants), measured on an Ascend310B1 with CANN 9.1.1. Sending
// float32 anyway fails the op instead of falling back, so the capability has to
// be asked of the hardware rather than assumed from the CANN release.
//
// Storage and elementwise kernels can still support float64, so matmul callers
// must use the CPU fallback to preserve PyTorch semantics.
inline bool IsMatmulDtypeSupported(c10::ScalarType dtype) {
  const bool fp16_only =
      IsFp16OnlyMatmulSoc(::at::native::flagos::ascend::GetSocName());
  if (fp16_only) {
    return dtype == c10::kHalf;
  }
  return dtype == c10::kHalf || dtype == c10::kBFloat16 || dtype == c10::kFloat;
}

// CANN's generated unary kernels have inconsistent integral contracts: some
// reject an integral dtype, some preserve it when PyTorch promotes to float,
// and others accept only a subset of integer widths. Use native ACLNN only for
// real floating-point inputs and preserve PyTorch semantics through the CPU
// fallback for every integral and bool input.
inline bool IsUnaryDtypeSupported(c10::ScalarType dtype) {
  return dtype == c10::kHalf || dtype == c10::kBFloat16 ||
      dtype == c10::kFloat || dtype == c10::kDouble;
}

inline c10::ScalarType BinaryResultType(
    std::string_view op,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto dtype = at::result_type(self, other);
  if (op == "aclnnDiv" && c10::isIntegralType(dtype, true)) {
    return at::get_default_dtype_as_scalartype();
  }
  return dtype;
}

// CPU fallback for an Ascend operation whose native aclnn implementation does
// not accept the requested dtype. The caller supplies the already-dispatched
// CPU result and this helper only moves it back to the original device.
inline at::Tensor MoveCpuResultToDevice(
    at::Tensor cpu_result,
    const at::Tensor& reference) {
  return std::move(cpu_result).to(reference.device());
}

} // namespace at::native::flagos::ascend
