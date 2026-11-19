// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "../../generated/ops.h"

#include <torch/library.h>

#include <map>
#include <mutex>
#include <string>

namespace at::native::flagos {
namespace {
struct VendorEntry {
  const char* vendor;
  const char* op;
  const char* abi;
  const char* catalog;
  void (*install)(uintptr_t);
  uintptr_t address = 0;
};

#define NATIVE_VENDOR_ENTRY(vendor, op, abi, catalog, type, dispatcher) \
  {vendor, op, abi, catalog, [](uintptr_t address) {                         \
     dispatcher.RegisterKernel(std::string(vendor) == "hygon" ? Backend::kHygon : Backend::kNvidia, reinterpret_cast<type>(address)); \
   }},
VendorEntry entries[] = {
#include "../../generated/native_vendor_entries.inc"
#include "../../generated/native_vendor_hygon_entries.inc"
};
#undef NATIVE_VENDOR_ENTRY

std::mutex& RegistryMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(flagos, m) {
  m.def("native_vendor_catalog(str vendor=\"nvidia\") -> str", [](const std::string& vendor) {
    std::string result = "[";
    for (const auto& entry : entries) {
      if (vendor != entry.vendor) continue;
      if (result.size() > 1) result += ',';
      result += entry.catalog;
    }
    return result + ']';
  });
  m.def("native_vendor_register(str vendor, str op, str abi, int address) -> ()",
        [](const std::string& vendor, const std::string& op,
           const std::string& abi, int64_t address) {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    for (auto& entry : entries) {
      if (vendor != entry.vendor || op != entry.op) continue;
      TORCH_CHECK(address != 0 && abi == entry.abi,
                  "native vendor ABI mismatch: ", vendor, ":", op);
      const auto pointer = static_cast<uintptr_t>(address);
      TORCH_CHECK(entry.address == 0 || entry.address == pointer,
                  "native vendor entry already registered: ", op);
      if (!entry.address) {
        entry.install(pointer);
        entry.address = pointer;
      }
      return;
    }
    TORCH_CHECK(false, "unknown native vendor entry: ", vendor, ":", op);
  });
}
}  // namespace at::native::flagos
