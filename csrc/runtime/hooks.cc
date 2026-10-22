// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegHooks.cpp
// with namespace c10::openreg renamed to c10::flagos and OpenRegHooksInterface renamed to HooksInterface.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "hooks.h"

namespace c10::flagos {

static bool register_hook_flag [[maybe_unused]] = []() {
  if (!at::isPrivateUse1HooksRegistered()) {
    at::RegisterPrivateUse1HooksInterface(new HooksInterface());
  }
  return true;
}();

} // namespace c10::flagos
