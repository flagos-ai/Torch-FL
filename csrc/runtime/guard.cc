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

bool HasGuardedStreams() noexcept {
#if defined(USE_ASCEND) || defined(USE_KUNLUNXIN) || FLAGOS_GUARD_HAS_PLATFORM_STREAM
  return true;
#else
  return false;
#endif
}

namespace {

uintptr_t StreamHandle(int64_t device, bool use_default) {
  TORCH_CHECK(device >= -1 && device < DeviceCount(), "Invalid stream device index: ", device);
  const c10::Device target(GuardImpl::static_type, static_cast<DeviceIndex>(device));
  const GuardImpl guard;
  const auto stream = use_default ? guard.getDefaultStream(target) : guard.getStream(target);
  return static_cast<uintptr_t>(stream.id());
}

} // namespace

uintptr_t CurrentStreamHandle(int64_t device) {
  TORCH_CHECK(HasGuardedStreams(), "This build has no guard-backed stream registry");
  return StreamHandle(device, false);
}

uintptr_t OwnedStreamHandle(int64_t device, bool use_default) {
  TORCH_CHECK(HasOwnedStreams(), "This build has no owned stream registry");
  auto handle = StreamHandle(device, use_default);
  TORCH_CHECK(handle != 0, "Native stream is unavailable for device ", device);
  return handle;
}

} // namespace c10::flagos
