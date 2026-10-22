// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGenerator.cpp
// with namespace c10::openreg renamed to c10::flagos, OpenRegGeneratorImpl renamed to GeneratorImpl,
// and getDefaultOpenRegGenerator renamed to getDefaultGenerator.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "generator.h"

#include <mutex>

// Default, global generators, one per device.
static std::vector<at::Generator> default_generators;

namespace c10::flagos {

const at::Generator& GetDefaultGenerator(c10::DeviceIndex device_index) {
  static bool flag [[maybe_unused]] = []() {
    auto device_nums = DeviceCount();
    default_generators.resize(device_nums);
    for (auto i = 0; i < device_nums; i++) {
      default_generators[i] = at::make_generator<GeneratorImpl>(i);
      default_generators[i].seed();
    }
    return true;
  }();

  c10::DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = CurrentDevice();
  } else {
    TORCH_CHECK(idx >= 0 && idx < DeviceCount());
  }
  return default_generators[idx];
}

uint64_t ReserveSeed(
    const std::optional<at::Generator>& generator,
    c10::DeviceIndex device_index) {
  at::Generator gen = generator.has_value() && generator->defined()
      ? *generator
      : GetDefaultGenerator(device_index);
  TORCH_CHECK(
      gen.device().type() == c10::DeviceType::PrivateUse1,
      "Expected a flagos generator, but found ",
      gen.device());
  if (device_index >= 0 && gen.device().has_index()) {
    TORCH_CHECK(
        gen.device().index() == device_index,
        "Expected a generator for flagos:",
        device_index,
        ", but found ",
        gen.device());
  }

  // A device-less flagos Generator follows the current device, matching
  // torch.Generator(device="flagos") semantics.
  std::lock_guard<std::mutex> lock(gen.mutex());
  return gen.get<at::CPUGeneratorImpl>()->random64();
}

} // namespace c10::flagos
