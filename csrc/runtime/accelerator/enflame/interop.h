// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <macros.h>
#include <cstdint>

// Producer 1 is topsaten, producer 2 is the prebuilt Triton launcher.
// ABI 1 drains at producer transitions; ABI 2 records/waits a device event.
// Begin/End bracket submission while holding the interop lock. Normal builds
// retain the conservative pre/post drain until a tracked launcher opts in.
extern "C" {
FLAGOS_EXPORT int FlagosEnflameEnableInterop(uint32_t abi);
FLAGOS_EXPORT int FlagosEnflameBeginLaunch(void* stream, uint32_t producer);
FLAGOS_EXPORT int FlagosEnflameEndLaunch(void* stream);
FLAGOS_EXPORT void FlagosEnflameForgetStream(void* stream);
}
