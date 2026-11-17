// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Parameterized, source-preserving host programs. No tensor or stream is retained.
#include "host_program.h"

#include <ATen/ATen.h>
#include <c10/core/GradMode.h>
#include <torch/library.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace at::native::flagos {
namespace {
using Guard = bool (*)(const at::Tensor*, size_t);
using Entry = void (*)(AtenTensorHandle*, AtenTensorHandle*);
struct Program {
  void* module;
  Guard guard;
  Entry entry;
  int64_t outputs;
  int guard_version;
};
std::atomic<bool> enabled{false};
std::atomic<int64_t> hits{0};
std::mutex mutex;
std::unordered_map<std::string, std::vector<Program>> programs;
std::unordered_set<std::string> loaded;
size_t resident_count = 0;
int64_t resident_bytes = 0;

// Scalars are specialization constants. Lists retain structure, tensor metadata
// belongs to the generated guard, and tensor contents never enter the key.
bool Append(const c10::IValue& value, std::ostream& key,
            std::vector<at::Tensor>* tensors) {
  if (value.isTensor()) {
    const auto& t = value.toTensor();
    if (!t.defined()) { key << "N;"; return true; }
    if (t.device().is_cpu() && t.unsafeGetTensorImpl()->is_wrapped_number()) {
      return Append(c10::IValue(t.item()), key, tensors);
    }
    if (!t.device().is_privateuseone()) return false;
    key << "T;";
    if (tensors) tensors->push_back(t);
  } else if (value.isNone()) {
    key << "N;";
  } else if (value.isBool()) {
    key << 'B' << value.toBool() << ';';
  } else if (value.isInt()) {
    key << 'I' << value.toInt() << ';';
  } else if (value.isDouble()) {
    double number = value.toDouble();
    uint64_t bits;
    std::memcpy(&bits, &number, sizeof(bits));
    key << 'D' << bits << ';';
  } else if (value.isString()) {
    auto text = value.toStringRef();
    key << 'S' << text.size() << ':' << text << ';';
  } else if (value.isList()) {
    auto list = value.toList();
    key << 'L' << list.size() << '[';
    for (const auto& item : list) if (!Append(item, key, tensors)) return false;
    key << ']';
  } else if (value.isIntList()) {
    auto list = value.toIntVector();
    key << 'L' << list.size() << '[';
    for (auto item : list) if (!Append(item, key, tensors)) return false;
    key << ']';
  } else if (value.isTensorList()) {
    auto list = value.toTensorVector();
    key << 'L' << list.size() << '[';
    for (const auto& item : list) if (!Append(item, key, tensors)) return false;
    key << ']';
  } else {
    return false;
  }
  return true;
}

std::string Key(const char* name, const std::vector<c10::IValue>& args,
                const std::vector<PyKwarg>& kwargs, int64_t outputs,
                std::vector<at::Tensor>* tensors) {
  if ((std::strchr(name, '.') && std::strncmp(name, "flag_gems.", 10) != 0) ||
      c10::GradMode::is_enabled()) return {};
  std::ostringstream key;
  key << name << '/' << outputs << '(';
  for (const auto& arg : args) if (!Append(arg, key, tensors)) return {};
  key << "){";
  for (const auto& kw : kwargs) {
    key << kw.name << ':' << kw.is_dtype << ':' << kw.is_none << ':';
    if (!Append(kw.is_none ? c10::IValue() : kw.value, key, tensors)) return {};
  }
  key << '}';
  return key.str();
}

void Register(const std::string& key, const std::string& path,
              int64_t outputs, int64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex);
  const auto identity = key + '\n' + path;
  if (loaded.count(identity)) return;
  TORCH_CHECK(outputs > 0 && outputs <= 16 && !key.empty(), "invalid host program");
  TORCH_CHECK(bytes > 0 && resident_count < 256 &&
              resident_bytes + bytes <= 512 * 1024 * 1024,
              "host program resident budget exhausted");
  auto& variants = programs[key];
  void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(module, "host program load failed: ", dlerror());
  auto guard = reinterpret_cast<Guard>(dlsym(module, "flagos_host_guard"));
  auto entry = reinterpret_cast<Entry>(dlsym(module, "flagos_host_entry"));
  auto version = reinterpret_cast<int (*)()>(dlsym(module, "flagos_host_guard_version"));
  if (!guard || !entry) {
    dlclose(module);
    TORCH_CHECK(false, "missing host program ABI");
  }
  // Modules remain resident until process exit. Disabling lookup cannot unload
  // code while another thread or an asynchronous device launch still uses it.
  variants.push_back({module, guard, entry, outputs, version ? version() : 0});
  loaded.insert(identity);
  resident_count++;
  resident_bytes += bytes;
}
} // namespace

bool HostProgramsEnabled() { return enabled.load(std::memory_order_relaxed); }
std::string HostProgramKey(const char* name, const std::vector<c10::IValue>& args,
                           const std::vector<PyKwarg>& kwargs, int64_t outputs) {
  return Key(name, args, kwargs, outputs, nullptr);
}

bool RunHostProgram(const std::string& key, const std::vector<at::Tensor>& tensors,
                    std::vector<at::Tensor>& result) {
  bool needs_safe_ranges = false;
  for (const auto& tensor : tensors) {
    if (!tensor.defined() || tensor.layout() != at::kStrided || tensor.is_nested()) return false;
    needs_safe_ranges |= tensor.numel() == 0 ||
        std::any_of(tensor.strides().begin(), tensor.strides().end(), [](int64_t s) { return s == 0; });
  }
  Program selected{};
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = programs.find(key);
    if (it == programs.end()) return false;
    for (const auto& program : it->second) {
      if (needs_safe_ranges && program.guard_version < 2) continue;
      if (program.guard(tensors.data(), tensors.size())) { selected = program; break; }
    }
  }
  if (!selected.entry) return false;
  std::vector<AtenTensorHandle> inputs, out(selected.outputs, nullptr);
  result.reserve(selected.outputs);
  for (const auto& tensor : tensors)
    inputs.push_back(torch::aot_inductor::new_tensor_handle(at::Tensor(tensor)));
  // Generated entry takes ownership of all inputs. Never retry after submission.
  try {
    selected.entry(inputs.data(), out.data());
  } catch (...) {
    for (auto handle : out)
      if (handle) delete torch::aot_inductor::tensor_handle_to_tensor_pointer(handle);
    throw;
  }
  for (auto handle : out) {
    auto tensor = torch::aot_inductor::tensor_handle_to_tensor_pointer(handle);
    result.push_back(std::move(*tensor));
    delete tensor;
  }
  hits.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool TryHostProgram(const char* name, const std::vector<c10::IValue>& args,
                    const std::vector<PyKwarg>& kwargs, int64_t outputs,
                    std::vector<at::Tensor>& result) {
  if (!HostProgramsEnabled()) return false;
  std::vector<at::Tensor> tensors;
  auto key = Key(name, args, kwargs, outputs, &tensors);
  if (key.empty()) return false;
  return RunHostProgram(key, tensors, result);
}

TORCH_LIBRARY_FRAGMENT(flagos, m) {
  m.def("host_program_enable(bool value) -> ()", [](bool value) { enabled.store(value); });
  m.def("host_program_register(str key, str path, int outputs, int bytes) -> ()", Register);
  m.def("host_program_run(str key, Tensor[] inputs) -> Tensor[]",
        [](const std::string& key, const std::vector<at::Tensor>& inputs) {
    TORCH_CHECK(!c10::GradMode::is_enabled(), "host program validation requires no_grad");
    std::vector<at::Tensor> result;
    TORCH_CHECK(RunHostProgram(key, inputs, result), "call outside compiled guard domain");
    return result;
  });
  m.def("host_program_stats() -> int[]", []() {
    std::lock_guard<std::mutex> lock(mutex);
    return std::vector<int64_t>{static_cast<int64_t>(resident_count), resident_bytes, hits.load()};
  });
}
} // namespace at::native::flagos
