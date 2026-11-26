// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device dtype cast for the Iluvatar backend.
//
// `_to_copy` is a hand-written composite (csrc/aten/copy_ops.cc), not a
// generated op, so the conf cannot route it: its same-device dtype branch calls
// a per-platform `DtypeCast` hook, and Ascend (aclnnCast), GCU
// (topsatenToCopy), Kunlunxin and MUSA (mudnn) each supply one. Iluvatar did
// not, so every `.to(dtype)` on a flagos tensor took the D2H -> CPU cast -> H2D
// round trip.
//
// There is no vendor cast to borrow here: the Iluvatar torch wheel is built with
// `_GLIBCXX_USE_CXX11_ABI=0` and cannot be boxed into the CPU wheel, and CoreX
// links no elementwise library. The device kernel therefore comes from FlagGems
// through the existing C++ -> Python bridge, which is also what routes every
// other elementwise op on this platform.
//
// Measured on a BI-V150 (CoreX 4.5.0, FlagGems 5.4.0): a bf16 <-> fp32 cast of
// `(1, 267, 2048)` drops from 0.598 ms to 0.145 ms, and `(1968, 2048)` from
// 3.468 ms to 0.163 ms, because the round trip pays a PCIe copy each way while
// the kernel does not scale with the copies. End to end, issue #667's model
// (12 samples, bf16) runs 3086.01 ms -> 2799.90 ms mean.

#pragma once

#include <ATen/core/Tensor.h>

#include <cstdlib>

#if defined(FLAGOS_FLAGGEMS_PYTHON)
#include "../flagos/python_op_caller.h"
#endif

namespace at::native::flagos::iluvatar {

namespace dtype_cast_detail {

// Which (source, target) pairs FlagGems' `to_copy` was measured exact for on a
// BI-V150 (CoreX 4.5.0, FlagGems 5.4.0), element for element against
// `x.cpu().to(dtype)`. The whitelist is the measurement, not a dtype taxonomy:
// several pairs in the neighbourhood of these are silently wrong, and the ones
// listed below are the pairs this platform's callers actually ask for.
//
// Accepted:
//   * float16 / bfloat16 / float32 in every direction (the hot path: HF does a
//     bf16 <-> fp32 cast around every RMSNorm);
//   * the integer types and bool as a *source* to those three floats;
//   * those three floats as a source to int8/int16/int32/int64/uint8.
//
// Declined, each for a measured reason:
//   * any float64 source: the kernel reads the low 32 bits of each element
//     (f64 -> f32 differed by 2.87 on randn input);
//   * any float source to uint16/uint32/uint64: wrong values;
//   * any float source to bool: truncates towards zero first, so 0.5 -> false
//     where ATen's contract (nonzero -> true) says true;
//   * float8 and quantized types in either direction: FlagGems has no `to_copy`
//     kernel for them and the Triton compilation raises.
//
// A declined pair returns an undefined tensor and copy_ops.cc takes the CPU
// round trip it already implements, so declining costs speed, never
// correctness.
inline bool DtypeCastSupported(at::ScalarType src, at::ScalarType dst) {
  const bool src_float = src == at::kHalf || src == at::kBFloat16 ||
      src == at::kFloat;
  const bool dst_float = dst == at::kHalf || dst == at::kBFloat16 ||
      dst == at::kFloat;
  if (src_float && dst_float) {
    return true;
  }
  if (src == at::kBool || src == at::kByte || src == at::kChar ||
      src == at::kShort || src == at::kInt || src == at::kLong) {
    return dst_float;
  }
  if (src_float) {
    return dst == at::kByte || dst == at::kChar || dst == at::kShort ||
        dst == at::kInt || dst == at::kLong;
  }
  return false;
}

} // namespace dtype_cast_detail

// `FLAGOS_ILUVATAR_DEVICE_CAST=0` puts every cast back on the CPU round trip.
// It is the rollback switch for a host where the FlagGems kernel turns out to
// be slower than the copies, and the same knob an A/B measurement needs.
inline bool DtypeCastEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("FLAGOS_ILUVATAR_DEVICE_CAST");
    return value == nullptr || value[0] != '0' || value[1] != '\0';
  }();
  return enabled;
}

// Cast a contiguous flagos tensor to `dtype` on the device. Returns an
// undefined tensor when the pair is outside the measured set above or this
// build has no FlagGems Python path; the caller then falls back to its CPU
// round trip.
inline at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype) {
#if defined(FLAGOS_FLAGGEMS_PYTHON)
  if (!src.defined() || src.scalar_type() == dtype) {
    return at::Tensor();
  }
  if (!DtypeCastEnabled()) {
    return at::Tensor();
  }
  if (!dtype_cast_detail::DtypeCastSupported(src.scalar_type(), dtype)) {
    return at::Tensor();
  }
  // The dotted qualname is resolved through flag_gems' package top level first
  // (see GetFunc in python_op_caller.cc), so a vendor-tuned `to_copy` wins over
  // the generic one without this call site knowing which backend is active.
  return CallPythonOp_TD("flag_gems.ops.to.to_copy", src, dtype);
#else
  (void)src;
  (void)dtype;
  return at::Tensor();
#endif
}

} // namespace at::native::flagos::iluvatar