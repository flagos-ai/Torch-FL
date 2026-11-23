// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <musa_runtime.h>

namespace at::native::flagos::musa {

// The registry in libflagos owns both the per-device defaults and pooled streams.
__attribute__((visibility("default"))) musaStream_t GetCurrentMusaStream();
__attribute__((visibility("default"))) musaStream_t GetMusaStreamFromPool(int device, int priority);

}  // namespace at::native::flagos::musa
