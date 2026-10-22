// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
// with namespace c10::openreg renamed to c10::flagos and OpenRegGuardImpl renamed to GuardImpl.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "guard.h"

namespace c10::flagos {

// Register the device guard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardImpl);

} // namespace c10::flagos
