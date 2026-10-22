// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.h
// with OpenRegGeneratorImpl renamed to GeneratorImpl and getDefaultOpenRegGenerator renamed to getDefaultGenerator.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include <c10/core/Device.h>

#include <cstdint>
#include <optional>

#include "functions.h"

namespace c10::flagos {

// This PrivateUse1 generator is the authoritative RNG state for native MUSA
// kernels and the FlagGems Python path. Both paths reserve a seed from this
// generator, so manual_seed/get_rng_state/set_rng_state cover mixed workloads.
class GeneratorImpl : public at::CPUGeneratorImpl {
 public:
  GeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~GeneratorImpl() override = default;
};

const at::Generator& GetDefaultGenerator(
    c10::DeviceIndex device_index = -1);

// Reserve one independent stochastic operation from the selected generator.
// The returned seed is suitable for initializing a device-side Philox engine;
// callers must not maintain a second persistent generator state.
FLAGOS_EXPORT uint64_t ReserveSeed(
    const std::optional<at::Generator>& generator,
    c10::DeviceIndex device_index = -1);

} // namespace c10::flagos
