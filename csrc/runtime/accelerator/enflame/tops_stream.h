// Copyright (c) 2026, BAAI. All rights reserved.
//
// Centralized tops stream management for the Enflame GCU backend.

#pragma once

#ifdef USE_GCU

#include <flagos.h>
#include <tops_runtime_api.h>

namespace at::native::flagos::gcu {

inline topsStream_t GetDefaultTopsStream() {
  int device = 0;
  if (topsGetDevice(&device) != topsSuccess) {
    return nullptr;
  }
  return reinterpret_cast<topsStream_t>(GetDefaultStreamForDevice(device));
}

inline topsStream_t GetCurrentTopsStream() {
  int device = 0;
  if (topsGetDevice(&device) != topsSuccess) {
    return nullptr;
  }
  return reinterpret_cast<topsStream_t>(GetCurrentStreamForDevice(device));
}

} // namespace at::native::flagos::gcu

#endif // USE_GCU
