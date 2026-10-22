// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device strided copy for the Ascend backend. Lets the platform-neutral
// copy_/clone/contiguous paths avoid the CPU round-trip (device->host strided
// copy->device) that dominates GQA repeat_kv clones in Qwen3 inference.

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::flagos::ascend {

#if defined(USE_ASCEND)

// Copy `src` into `dst` entirely on the NPU via aclnnInplaceCopy, which handles
// differing strides/offsets and dtype casts on-device. `dst` must be an
// allocated PrivateUse1 tensor with matching sizes; `src` may be non-contiguous.
// Returns true on success. Callers use the return value to fall back to the CPU
// round-trip if the on-device path is unavailable.
bool StridedCopy(const at::Tensor& dst, const at::Tensor& src);

// Cast `src` (a contiguous PrivateUse1 tensor) to `dtype` entirely on the NPU
// via aclnnCast, returning a freshly-allocated contiguous PrivateUse1 tensor.
// Replaces the D2H -> CPU cast -> H2D round-trip in _to_copy's Ascend dtype
// path, which dominated HF RMSNorm (two fp16<->fp32 casts per layer). Returns
// an undefined tensor if the on-device path is unavailable (caller falls back).
at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype);

#else

// Non-Ascend builds: the shared copy_/clone/contiguous paths in copy_ops.cc and
// contiguous_ops.cc reach these from an #else branch that covers TsingMicro,
// GCU and MUSA-without-mudnn as well as Ascend. Those platforms have no aclnn,
// so provide inline no-ops that report "unavailable" and let the caller take
// the CPU round-trip it already implements as the fallback.
//
// These must be defined (not just declared): a .so links with undefined symbols
// and only fails at dlopen, so a bare declaration would produce a wheel that
// imports fine on Ascend and dies with "undefined symbol" everywhere else.
inline bool StridedCopy(const at::Tensor&, const at::Tensor&) {
  return false;
}

inline at::Tensor DtypeCast(const at::Tensor&, at::ScalarType) {
  return at::Tensor();
}

#endif

} // namespace at::native::flagos::ascend
