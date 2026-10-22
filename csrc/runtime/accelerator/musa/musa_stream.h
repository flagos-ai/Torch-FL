// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized musa stream management for the Moore Threads MUSA backend.
// All MUSA ops share one default stream per device so that consecutive kernels
// keep their implicit ordering without explicit cross-stream synchronization.
//
// The definition lives out of line in musa_stream.cc, which the build places in
// libflagos.so -- the one MUSA runtime library every other shared object in the
// process links against. That placement is load-bearing, not cosmetic: this used
// to be an `inline` function with a function-local static, and an inline
// function-local static is *per shared object*. torch_fl._C
// (libtorch_bindings.so) and the mudnn kernels (libtorch_fl.so) therefore each
// built their own cache, each calling musaStreamCreate once, and the process
// ended up with two "default" streams:
//
//   [mudnn] dev=0 set(0x5559b8cec2b0)=0 get=0x5559b8cec2b0   <- libtorch_fl.so
//   python S = 0x5559b8cdc550                                <- libtorch_bindings.so
//
// Triton takes its launch stream from the Python binding
// (flagtree_shim.get_musa_current_raw_stream -> _C._get_musa_current_raw_stream),
// while every mudnn op ran on the other one, so a Triton kernel and the mudnn op
// consuming its output were never ordered against each other. Measured on MTT
// S5000 / mudnn v3300: `fill_c[(1,)](o, 9.0); torch.add(a, o)` repeatedly read
// the buffer's pre-fill contents. Both observations that appeared to contradict
// each other are explained by this one defect -- a host-side
// musaStreamSynchronize(S) before the mudnn op *did* fix it (a host sync orders
// against every stream), while adding the same synchronize inside the mudnn
// entry macro did not (it drained the empty second stream). One out-of-line
// definition restores the single shared queue this header always claimed.

#pragma once

#ifdef USE_MUSA

#include <musa_runtime.h>

// The symbol has to cross a shared-object boundary, and the build sets
// CMAKE_CXX_VISIBILITY_PRESET=hidden, so it needs an explicit default. The same
// attribute has to appear on the declaration and the definition: GCC takes the
// visibility from the first declaration it sees and ignores a later, different
// one, which would silently leave the definition hidden again.
#if defined(__GNUC__) || defined(__clang__)
#define FLAGOS_MUSA_STREAM_API __attribute__((visibility("default")))
#else
#define FLAGOS_MUSA_STREAM_API
#endif

namespace at::native::flagos::musa {

// One stream per device: a musa stream belongs to the device that was current
// when it was created, so a single global stream would be wrong as soon as an
// op runs on flagos:1.
FLAGOS_MUSA_STREAM_API musaStream_t GetDefaultMusaStream();

} // namespace at::native::flagos::musa

#endif // USE_MUSA
