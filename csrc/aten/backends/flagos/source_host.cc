// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "source_host.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/library.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>

namespace at::native::flagos {
namespace {
struct Source {
  int64_t address;
  int64_t outputs;
  int64_t abi;
};
std::atomic<bool> populated{false};
std::atomic<bool> tensor_sources{false};
std::atomic<int64_t> hits{0};
std::shared_mutex mutex;
std::deque<std::string> source_names;
std::unordered_map<std::string_view, Source> sources;

std::string_view Name(const char* name) {
  std::string_view value(name);
  if (value.compare(0, 10, "flag_gems.") == 0) return value.substr(10);
  return value.find('.') == std::string_view::npos ? value : std::string_view();
}

bool Device(const c10::IValue& value, std::optional<c10::Device>& device) {
  if (value.isTensor()) {
    const auto& tensor = value.toTensor();
    if (!tensor.defined()) return true;
    if (tensor.key_set().has(c10::DispatchKey::Python) ||
        tensor.key_set().has(c10::DispatchKey::Functionalize))
      return false;
    if (!device && tensor.device().is_privateuseone()) device = tensor.device();
  } else if (value.isList()) {
    for (const auto& child : value.toList())
      if (!Device(child, device)) return false;
  } else if (value.isTuple()) {
    for (const auto& child : value.toTupleRef().elements())
      if (!Device(child, device)) return false;
  }
  return true;
}

bool FindSource(const char* name, int64_t outputs, Source& source) {
  if (!populated.load(std::memory_order_acquire) || c10::GradMode::is_enabled()) return false;
  std::shared_lock<std::shared_mutex> lock(mutex);
  auto found = sources.find(Name(name));
  if (found == sources.end() || found->second.outputs != outputs) return false;
  source = found->second;
  return true;
}

bool GuardSource(c10::ArrayRef<c10::IValue> args, c10::ArrayRef<PyKwarg> kwargs,
                 c10::OptionalDeviceGuard& guard) {
  std::optional<c10::Device> device;
  for (const auto& value : args)
    if (!Device(value, device)) return false;
  for (const auto& value : kwargs)
    if (!Device(value.value, device)) return false;
  if (device) {
    c10::impl::VirtualGuardImpl implementation(device->type());
    if (implementation.getDevice() != *device) guard.reset_device(*device);
  }
  return true;
}
}  // namespace

bool TrySourceHostTensor(const char* name, c10::ArrayRef<c10::IValue> args,
                         c10::ArrayRef<PyKwarg> kwargs, at::Tensor& result) {
  if (!tensor_sources.load(std::memory_order_acquire)) return false;
  Source source{};
  if (!FindSource(name, 1, source) || source.abi != 2) return false;
  c10::OptionalDeviceGuard guard;
  if (!GuardSource(args, kwargs, guard)) return false;
  if (!reinterpret_cast<SourceHostTensorEntry>(source.address)(args, kwargs, result)) return false;
  hits.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool TrySourceHost(const char* name, const std::vector<c10::IValue>& args,
                   const std::vector<PyKwarg>& kwargs, int64_t outputs,
                   std::vector<at::Tensor>& result) {
  Source source{};
  if (!FindSource(name, outputs, source)) return false;
  c10::OptionalDeviceGuard guard;
  if (!GuardSource(args, kwargs, guard)) return false;
  if (source.abi == 2) {
    at::Tensor value;
    if (!reinterpret_cast<SourceHostTensorEntry>(source.address)(args, kwargs, value)) return false;
    result = {std::move(value)};
  } else {
    if (!reinterpret_cast<SourceHostEntry>(source.address)(args, kwargs, result)) return false;
  }
  TORCH_CHECK(static_cast<int64_t>(result.size()) == outputs, "native source output arity changed");
  hits.fetch_add(1, std::memory_order_relaxed);
  return true;
}

TORCH_LIBRARY_FRAGMENT(flagos, m) {
  m.def("native_source_register(str name, int address, int outputs, int abi) -> ()",
        [](const std::string& name, int64_t address, int64_t outputs, int64_t abi) {
          TORCH_CHECK((abi == 1 || (abi == 2 && outputs == 1)) && address != 0 && outputs > 0,
                      "invalid native source ABI");
          auto key = Name(name.c_str());
          TORCH_CHECK(!key.empty(), "native source must name a FlagGems function");
          std::unique_lock<std::shared_mutex> lock(mutex);
          auto found = sources.find(key);
          TORCH_CHECK(found == sources.end() ||
                          (found->second.address == address && found->second.abi == abi),
                      "native source is already registered: ", key);
          if (found == sources.end()) {
            source_names.emplace_back(key);
            sources.emplace(source_names.back(), Source{address, outputs, abi});
          } else {
            found->second = Source{address, outputs, abi};
          }
          populated.store(true, std::memory_order_release);
          if (abi == 2) tensor_sources.store(true, std::memory_order_release);
        });
  m.def("native_source_stats() -> int[]", []() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return std::vector<int64_t>{static_cast<int64_t>(sources.size()), hits.load()};
  });
}
}  // namespace at::native::flagos
