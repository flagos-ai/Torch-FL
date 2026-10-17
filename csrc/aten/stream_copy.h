// Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <flagos.h>
#include "runtime/dcu_stream.h"

namespace at::native::flagos {

// Callers bind the accelerator-side operand's device for the whole transfer.
// Keep the existing runtime call unchanged outside the optional DCU provider.
inline Error_t BlockingRuntimeMemcpy(
    void* dst, const void* src, size_t count, MemcpyKind kind) {
#ifdef FLAGOS_DCU_VENDOR_STREAMS
  c10::flagos::dcu::CopyBytesOnCurrentStream(dst, src, count, kind);
  return Success;
#else
  return ::Memcpy(dst, src, count, kind);
#endif
}

} // namespace at::native::flagos
