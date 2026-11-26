// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "typed_source_host.h"

#include <torch/library.h>

#include <map>
#include <mutex>
#include <string_view>

namespace at::native::flagos {
namespace {
struct Sources {
  std::mutex mutex;
  std::map<std::string, TypedSourceSlot*> slots;
  int64_t registered = 0;
};
Sources& Registry() {
  static auto* registry = new Sources;
  return *registry;
}
}  // namespace

TypedSourceSlot::TypedSourceSlot(const char* op, const char* host, const char* abi,
                                 const char* catalog)
    : op(op), host(host), abi(abi), catalog(catalog),
      scalar_factory_(std::string_view(op) == "aten::scalar_tensor") {
  auto& registry = Registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  TORCH_CHECK(registry.slots.emplace(op, this).second, "duplicate typed source slot: ", op);
}

c10::Device TypedSourceFactoryDevice(std::optional<c10::Device> device, const at::Tensor* like) {
  if (device && device->has_index()) return *device;
  if (like && like->defined() && like->device().is_privateuseone() && like->device().has_index())
    return like->device();
  c10::impl::VirtualGuardImpl implementation(c10::DeviceType::PrivateUse1);
  return implementation.getDevice();
}

TORCH_LIBRARY_FRAGMENT(flagos, m) {
  m.def("native_source_typed_catalog() -> str", []() {
    auto& registry = Registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    std::string result = "[";
    for (const auto& entry : registry.slots) {
      if (result.size() > 1) result += ',';
      result += entry.second->catalog;
    }
    return result + ']';
  });
  m.def(
      "native_source_register_typed(str op, str host, str abi, int address) -> ()",
      [](const std::string& op, const std::string& host, const std::string& abi, int64_t address) {
        auto& registry = Registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.slots.find(op);
        TORCH_CHECK(found != registry.slots.end(), "unknown typed source slot: ", op);
        auto* slot = found->second;
        TORCH_CHECK(address != 0 && host == slot->host && abi == slot->abi,
                    "typed source ABI mismatch: ", op);
        const auto previous = slot->address.load(std::memory_order_acquire);
        TORCH_CHECK(previous == 0 || previous == static_cast<uintptr_t>(address),
                    "typed source already registered: ", op);
        if (!previous) ++registry.registered;
        slot->address.store(static_cast<uintptr_t>(address), std::memory_order_release);
      });
  m.def("native_source_typed_stats() -> int[]", []() {
    auto& registry = Registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    int64_t hits = 0;
    for (const auto& entry : registry.slots) hits += entry.second->hits.load();
    return std::vector<int64_t>{registry.registered, hits};
  });
  m.def("native_source_typed_hits(str op) -> int", [](const std::string& op) {
    auto& registry = Registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.slots.find(op);
    TORCH_CHECK(found != registry.slots.end(), "unknown typed source slot: ", op);
    return found->second->hits.load();
  });
}
}  // namespace at::native::flagos
