// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
// with namespace c10::openreg renamed to c10::flagos and OpenRegGuardImpl renamed to GuardImpl.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "guard.h"
#include "functions.h"

namespace c10::flagos {

// Register the device guard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, GuardImpl);

bool HasOwnedStreams() noexcept {
  return FLAGOS_GUARD_HAS_OWNED_STREAM;
}

uintptr_t OwnedStreamHandle(int64_t device, bool use_default) {
  TORCH_CHECK(HasOwnedStreams(), "This build has no owned stream registry");
  TORCH_CHECK(device >= -1 && device < DeviceCount(), "Invalid stream device index: ", device);
  const c10::Device target(GuardImpl::static_type, static_cast<DeviceIndex>(device));
  const GuardImpl guard;
  const auto stream = use_default ? guard.getDefaultStream(target) : guard.getStream(target);
  // The runtime-backed guard stores the device-library handle in the ID.
  TORCH_CHECK(stream.id() != 0, "Native stream is unavailable for device ", device);
  return static_cast<uintptr_t>(stream.id());
}

} // namespace c10::flagos
