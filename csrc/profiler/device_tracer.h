// Copyright 2026 FlagOS Contributors
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

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace c10 {
namespace flagos {
namespace profiler {

enum class EventKind {
  Kernel,
  Memcpy,
  Memset,
  Runtime,
};

struct DeviceEvent {
  EventKind kind;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint32_t correlation_id = 0;  // Links runtime↔kernel

  // Torch's correlation id for this activity, resolved by the vendor layer from
  // its external-correlation records. This is a DIFFERENT numbering from
  // correlation_id above: kineto's getLinkedActivity() callback keys on this
  // one, while correlation_id pairs a runtime event with its device event.
  // Empty when the vendor layer had no external-correlation record for this
  // activity -- deliberately optional rather than 0, since 0 is a valid torch
  // correlation id and conflating the two silently mis-attributes device time.
  std::optional<int32_t> external_correlation_id;

  uint32_t device = 0;
  uint32_t stream = 0;
  uint32_t thread_id = 0;  // Runtime events run on CPU thread
  std::string name;  // Already demangled
  std::map<std::string, std::string> metadata;  // "grid"→"[8,16,5]", etc.
};

class DeviceTracer {
 public:
  virtual ~DeviceTracer() = default;

  virtual bool available() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::vector<DeviceEvent> drain() = 0;
  virtual void pushCorrelation(uint64_t id) {}
  virtual void popCorrelation() {}
  virtual int deviceCount() const = 0;
};

// Factory: returns CUPTI tracer on NVIDIA, CANN tracer on Ascend, etc.
// Vendor selection is compile-time (based on FLAGOS_ACCELERATOR cmake var).
std::unique_ptr<DeviceTracer> MakeDeviceTracer();

}  // namespace profiler
}  // namespace flagos
}  // namespace c10
