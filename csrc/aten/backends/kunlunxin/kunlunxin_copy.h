// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device copy primitives for the Kunlunxin XPU backend.
//
// The XPU SDK's ATen layer has no kernel for a strided copy or for a dtype
// cast, and the "no on-device path" arm of csrc/aten/copy_ops.cc and
// csrc/aten/contiguous_ops.cc is a full host round-trip: D2H the operand, do
// the copy on the CPU, H2D the result. That is the `repeat_kv` hotspot in
// Qwen3's GQA attention and the `.expand(...).clone()` pattern in Qwen-Image,
// where the same host round-trip on GCU measured 975.6 ms per call at the
// (1, 24, 4114, 4114) fp32 shape against 4.6 ms for the vendor call.
//
// Both operations are one call away from the vendor: `xdnn_pytorch::contiguous`
// materialises a strided operand through its own strides, and
// `xdnn_pytorch::cast` converts between dtypes. `cast` takes no dtype argument
// -- it reads the target type off the output descriptor -- which is why
// DtypeCast below allocates the destination first and describes it at the
// target dtype rather than passing the dtype to the call.
//
// This is copy infrastructure calling the vendor C ABI, not a per-operator
// kernel, which is the same shape as Ascend's ascend_copy.h/.cc and GCU's
// gcu_copy.h/.cc; the codegen rule in CLAUDE.md covers operator integration,
// and _to_copy/copy_/contiguous/clone are hand-written ATen composite plumbing
// on every backend here.
//
// Every caller keeps its existing host round-trip as the fallback: these
// functions decline (false / undefined tensor) rather than raise whenever the
// call is one the vendor cannot serve.

#pragma once

#include <ATen/core/Tensor.h>

// The definitions live in kunlunxin_copy.cc, which CMake excludes from the
// build whenever the Kunlunxin native kernels are not being built
// (FLAGOS_BUILD_VENDOR=OFF drops all of backends/kunlunxin/). Gating the
// declarations on the same FLAGOS_KUNLUNXIN_KERNEL switch register.cc uses
// keeps a declaration from outliving its definition, which would be an
// undefined symbol at link rather than a compile error.
#if defined(USE_KUNLUNXIN) && defined(FLAGOS_KUNLUNXIN_KERNEL)

namespace at::native::flagos::kunlunxin {

// Copies `src` into `dst` on the device, honoring sizes, strides and
// storage_offset on both sides; `src` may be broadcast against `dst`. The two
// dtypes must match. Returns false when the call is one this path declines, in
// which case the caller must fall back to its host round-trip.
bool StridedCopy(const at::Tensor& dst, const at::Tensor& src);

// Returns a contiguous on-device copy of `src` converted to `dtype`, or an
// undefined tensor when the cast is one this path declines.
at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype);

} // namespace at::native::flagos::kunlunxin

#else

namespace at::native::flagos::kunlunxin {

// Without the Kunlunxin kernels the definitions above are not compiled, and a
// bare declaration would build a wheel that imports fine and then dies with
// "undefined symbol" at the first cast or strided copy. Inline no-op fallbacks
// keep every caller total, so the shared `#else` arms in copy_ops.cc /
// contiguous_ops.cc can name these unconditionally, exactly as they already do
// for ascend_copy.h. Callers use the return value / definedness to keep their
// existing host round-trip.
inline bool StridedCopy(const at::Tensor&, const at::Tensor&) {
  return false;
}

inline at::Tensor DtypeCast(const at::Tensor&, at::ScalarType) {
  return at::Tensor();
}

} // namespace at::native::flagos::kunlunxin

#endif // USE_KUNLUNXIN && FLAGOS_KUNLUNXIN_KERNEL
