// Copyright (c) 2026, BAAI. All rights reserved.
//
// On-device copy primitives for the Enflame GCU backend.
//
// GCU has no ATen kernel for a strided copy or for a dtype cast, and the
// "no on-device path" arm of csrc/aten/copy_ops.cc and
// csrc/aten/contiguous_ops.cc is a full host round-trip: D2H the operand, do
// the copy on the CPU, H2D the result. At the shape that dominates Qwen-Image's
// forward -- the (1, 24, 4114, 4114) fp32 attention matrix, 1.5 GiB -- measured
// on S60 through the regular dispatcher:
//
//   aten `_to_copy` f32 -> bf16          437.9 ms   (vendor plugin: 5.8 ms)
//   `small.expand(full).contiguous()`    975.6 ms   (same 1.5 GiB as a clone:
//                                                    14.2 ms)
//
// Both are one call away from the vendor: `topsatenCopy` takes a strided source
// and a broadcastable destination, and `topsatenToCopy` takes an explicit output
// dtype. Issued directly, the same two operations cost 5.6 ms and 4.6 ms
// (measured, /tmp/topsaten_copy_speed.cc). These two functions are that route.
//
// This is copy infrastructure calling the vendor C ABI, not a per-operator
// kernel, which is the same shape as Ascend's ascend_copy.h/.cc; the codegen
// rule in CLAUDE.md covers operator integration, and _to_copy/copy_/contiguous/
// clone are hand-written ATen composite plumbing on every backend here.
//
// Every caller keeps its existing host round-trip as the fallback: these
// functions decline (false / undefined tensor) rather than raise whenever the
// call is one the vendor cannot serve.

#pragma once

#include <ATen/core/Tensor.h>

// The definitions live in gcu_copy.cc, which CMake excludes from the build
// whenever the GCU native kernels are not being built (FLAGOS_BUILD_VENDOR=OFF
// drops all of backends/gcu/). Gating the declarations on the same
// FLAGOS_GCU_KERNEL switch register.cc uses keeps a declaration from outliving
// its definition, which would be an undefined symbol at link rather than a
// compile error.
#if defined(USE_GCU) && defined(FLAGOS_GCU_KERNEL)

namespace at::native::flagos::gcu {

// Copies `src` into `dst` on the device, honoring sizes, strides and
// storage_offset on both sides; `src` may be broadcast against `dst`. The two
// dtypes must match. Returns false when the call is one this path declines, in
// which case the caller must fall back to its host round-trip.
bool StridedCopy(const at::Tensor& dst, const at::Tensor& src);

// Returns a contiguous on-device copy of `src` converted to `dtype`, or an
// undefined tensor when the cast is one this path declines.
at::Tensor DtypeCast(const at::Tensor& src, at::ScalarType dtype);

} // namespace at::native::flagos::gcu

#else

namespace at::native::flagos::gcu {

// Without the GCU kernels the definitions above are not compiled, and a bare
// declaration would build a wheel that imports fine and then dies with
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

} // namespace at::native::flagos::gcu

#endif // USE_GCU && FLAGOS_GCU_KERNEL
