// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/ATen.h>

#if defined(USE_CAMBRICON)

namespace at::native::flagos::cambricon {

// Copies `src` into the contiguous `dst` on the device, honouring `src`'s
// strides.
//
// contiguous_ops.cc sends a strided PrivateUse1 source through a host
// round-trip when this returns false, and that round-trip is the single most
// expensive thing a training step does: `x.contiguous()` on the
// (2,64,16,512) stride (524288,1,32768,64) view RoPE hands it measured 16228 us
// against 18 us for the same gather through cnnlCopy_v2. The multiply inside
// every binary kernel gathers its operands this way, so the cost was paid
// several times a layer.
//
// Returns false for anything this path declines -- a non-contiguous or
// mismatched destination, a source out of the dtype range -- in which case the
// caller keeps its existing fallback.
bool StridedCopy(const at::Tensor& dst, const at::Tensor& src);

} // namespace at::native::flagos::cambricon

#else

namespace at::native::flagos::cambricon {

// Same shape every other vendor header takes in a build without its kernels:
// an inline no-op so contiguous_ops.cc can name this unconditionally, and the
// caller keeps its host round-trip because the answer is "not copied".
inline bool StridedCopy(const at::Tensor&, const at::Tensor&) {
  return false;
}

} // namespace at::native::flagos::cambricon

#endif
