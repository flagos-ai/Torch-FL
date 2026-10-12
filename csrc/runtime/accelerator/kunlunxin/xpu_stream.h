// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized XPU stream management for the Kunlunxin backend.
// Every Kunlunxin op (native xdnn_pytorch kernels and any FlagGems kernel that
// runs through the same dispatcher) shares one default stream per device, so
// two ops dispatched back to back are ordered without explicit synchronization.

#pragma once

#ifdef USE_KUNLUNXIN

#include <flagos.h>
#include <xpu/runtime.h>

namespace at::native::flagos::kunlunxin {

// Returns the default XPU stream of the current device. An XPUStream belongs to
// the device that was current when it was created, so there is one per device
// rather than a single process-wide handle; the registry lives in stream.cc and
// these are thin accessors onto it.
inline XPUStream GetDefaultXpuStream() {
  int device = 0;
  if (xpu_current_device(&device) != 0) {
    return nullptr;
  }
  return reinterpret_cast<XPUStream>(GetDefaultStreamForDevice(device));
}

inline XPUStream GetCurrentXpuStream() {
  int device = 0;
  if (xpu_current_device(&device) != 0) {
    return nullptr;
  }
  return reinterpret_cast<XPUStream>(GetCurrentStreamForDevice(device));
}

// Wait for every per-device default stream created so far.
//
// Called before a blocking (non-stream-ordered) transfer that touches device
// memory. The caller does not always know which device owns the pointer, so all
// of them are drained; only devices that actually ran an op have a stream here.
//
// Unlike the Ascend port this is defence in depth rather than a measured
// requirement: the Kunlunxin op layer waits on its stream after every dispatch
// (the vendor entry points give no completion handle of their own), so no
// transfer was observed to race a pending kernel on this hardware. It stays
// because it is cheap and because the op layer may later learn to dispatch
// without the per-op wait, at which point this becomes load-bearing. Keep it a
// single exterior-linkage definition in stream.cc -- an `inline` function-local
// static would give each shared object its own copy under
// -fvisibility=hidden.
FLAGOS_EXPORT void DrainDefaultXpuStreams();

} // namespace at::native::flagos::kunlunxin

#endif // USE_KUNLUNXIN
