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
//
// Ascend device tracer. CANN's public MSPTI activity API is used instead of
// torch_npu or the file-oriented msprof pipeline. Memcpy and memset
// interception requires the caller to preload libmspti before the Ascend
// runtime; ordinary torch_fl imports do not install that interposer.

#include "device_tracer.h"

#if defined(FLAGOS_HAVE_MSPTI)

#include <mspti.h>
#include <flagos_env.h>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace c10 {
namespace flagos {
namespace profiler {
namespace {

constexpr size_t kBufferAlignment = 8;
constexpr size_t kBufferSize = 8 * 1024 * 1024;
constexpr uint64_t kMaxPlausibleDurationNs = 3600ull * 1000 * 1000 * 1000;

bool debug_enabled() {
  // Was "the variable exists", which made the debug switch's =0 turn the
  // logging ON. The shared truth table reads 1/true/on/yes as on.
  static const bool enabled = flagos_env::EnvFlag("FLAGOS_TRACE");
  return enabled;
}

#define FLAGOS_MSPTI_LOG(expr) \
  do { \
    if (debug_enabled()) { \
      std::cerr << expr; \
    } \
  } while (false)

bool timestamps_plausible(uint64_t start, uint64_t end) {
  return start != 0 && end >= start && end - start <= kMaxPlausibleDurationNs;
}

std::string copy_name(const char* value, const char* fallback) {
  return value && *value ? std::string(value) : std::string(fallback);
}

std::vector<uint32_t> visible_devices() {
  const char* value = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  if (!value || !*value) {
    return {};
  }
  std::vector<uint32_t> result;
  const char* begin = value;
  while (*begin) {
    char* end = nullptr;
    unsigned long id = std::strtoul(begin, &end, 10);
    if (end == begin) {
      ++begin;
      continue;
    }
    if (id <= std::numeric_limits<uint32_t>::max()) {
      result.push_back(static_cast<uint32_t>(id));
    }
    begin = end;
    while (*begin == ',' || *begin == ' ') {
      ++begin;
    }
  }
  return result;
}

uint32_t logical_device(uint32_t physical) {
  static const std::vector<uint32_t> visible = visible_devices();
  for (size_t i = 0; i < visible.size(); ++i) {
    if (visible[i] == physical) {
      return static_cast<uint32_t>(i);
    }
  }
  return physical;
}

struct MspTiApi {
  using Subscribe = msptiResult (*)(msptiSubscriberHandle*, msptiCallbackFunc, void*);
  using Unsubscribe = msptiResult (*)(msptiSubscriberHandle);
  using RegisterCallbacks = msptiResult (*)(
      msptiBuffersCallbackRequestFunc, msptiBuffersCallbackCompleteFunc);
  using ActivityEnable = msptiResult (*)(msptiActivityKind);
  using ActivityDisable = msptiResult (*)(msptiActivityKind);
  using ActivityFlushAll = msptiResult (*)(uint32_t);
  using ActivityGetNextRecord = msptiResult (*)(uint8_t*, size_t, msptiActivity**);
  using PushExternal = msptiResult (*)(msptiExternalCorrelationKind, uint64_t);
  using PopExternal = msptiResult (*)(msptiExternalCorrelationKind, uint64_t*);

  void* handle = nullptr;
  Subscribe subscribe = nullptr;
  Unsubscribe unsubscribe = nullptr;
  RegisterCallbacks register_callbacks = nullptr;
  ActivityEnable activity_enable = nullptr;
  ActivityDisable activity_disable = nullptr;
  ActivityFlushAll activity_flush_all = nullptr;
  ActivityGetNextRecord activity_get_next_record = nullptr;
  PushExternal push_external = nullptr;
  PopExternal pop_external = nullptr;

  bool load() {
    if (handle) {
      return complete();
    }
    const char* candidates[] = {
        "libmspti.so",
        "/usr/local/Ascend/ascend-toolkit/latest/tools/mspti/lib64/libmspti.so",
        "/usr/local/Ascend/cann-9.0.0/tools/mspti/lib64/libmspti.so",
    };
    for (const char* candidate : candidates) {
      handle = dlopen(candidate, RTLD_NOW | RTLD_GLOBAL);
      if (handle) {
        break;
      }
    }
    if (!handle) {
      return false;
    }
    // Function-pointer member names intentionally match the public symbols
    // after the leading `mspti` has been removed.
    subscribe = reinterpret_cast<Subscribe>(dlsym(handle, "msptiSubscribe"));
    unsubscribe = reinterpret_cast<Unsubscribe>(dlsym(handle, "msptiUnsubscribe"));
    register_callbacks = reinterpret_cast<RegisterCallbacks>(
        dlsym(handle, "msptiActivityRegisterCallbacks"));
    activity_enable = reinterpret_cast<ActivityEnable>(dlsym(handle, "msptiActivityEnable"));
    activity_disable = reinterpret_cast<ActivityDisable>(dlsym(handle, "msptiActivityDisable"));
    activity_flush_all = reinterpret_cast<ActivityFlushAll>(dlsym(handle, "msptiActivityFlushAll"));
    activity_get_next_record = reinterpret_cast<ActivityGetNextRecord>(
        dlsym(handle, "msptiActivityGetNextRecord"));
    push_external = reinterpret_cast<PushExternal>(
        dlsym(handle, "msptiActivityPushExternalCorrelationId"));
    pop_external = reinterpret_cast<PopExternal>(
        dlsym(handle, "msptiActivityPopExternalCorrelationId"));
    return complete();
  }

  bool complete() const {
    return subscribe && unsubscribe && register_callbacks && activity_enable &&
        activity_disable && activity_flush_all && activity_get_next_record &&
        push_external && pop_external;
  }
};

struct PendingEvent {
  EventKind kind;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint64_t vendor_correlation = 0;
  uint32_t physical_device = 0;
  uint32_t stream = 0;
  uint32_t thread_id = 0;
  std::string name;
  std::map<std::string, std::string> metadata;
};

class CannDeviceTracer;
std::atomic<CannDeviceTracer*> g_active_tracer{nullptr};

void buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records);
void buffer_completed(uint8_t* buffer, size_t size, size_t valid_size);

class CannDeviceTracer final : public DeviceTracer {
 public:
  // Availability is a build-time capability here. Resolve libmspti lazily in
  // start() so importing torch_fl does not install the CANN interposer in
  // ordinary, non-profiler processes.
  CannDeviceTracer() : available_(true) {}

  ~CannDeviceTracer() override {
    if (active_) {
      stop();
    }
    // Do not dlclose libmspti. CANN may retain interposed ACL function
    // pointers after a session, and unloading the library during Python
    // interpreter teardown can crash otherwise unrelated Ascend work.
  }

  bool available() const override { return available_; }

  void start() override {
    if (!available_ || active_) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      return;
    }
    pending_.clear();
    external_correlation_.clear();
    correlation_ids_.clear();
    next_correlation_id_ = 1;
    session_failed_ = false;

    // libmspti is resolved here, not at construction: an Ascend process that
    // never profiles must not load CANN's interposer at all.
    if (!api_.load()) {
      session_failed_ = true;
      FLAGOS_MSPTI_LOG("[flagos] libmspti is unavailable\n");
      return;
    }

    msptiSubscriberHandle subscriber = nullptr;
    msptiResult result = api_.subscribe(&subscriber, nullptr, nullptr);
    if (result != MSPTI_SUCCESS) {
      session_failed_ = true;
      FLAGOS_MSPTI_LOG("[flagos] msptiSubscribe failed: " << result << "\n");
      return;
    }
    subscriber_ = subscriber;

    result = api_.register_callbacks(buffer_requested, buffer_completed);
    if (result != MSPTI_SUCCESS) {
      session_failed_ = true;
      api_.unsubscribe(subscriber_);
      subscriber_ = nullptr;
      FLAGOS_MSPTI_LOG("[flagos] msptiActivityRegisterCallbacks failed: " << result << "\n");
      return;
    }

    const msptiActivityKind required[] = {
        MSPTI_ACTIVITY_KIND_KERNEL,
        MSPTI_ACTIVITY_KIND_API,
        MSPTI_ACTIVITY_KIND_ACL_API,
        MSPTI_ACTIVITY_KIND_NODE_API,
        MSPTI_ACTIVITY_KIND_RUNTIME_API,
        MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    };
    for (msptiActivityKind kind : required) {
      msptiResult enable_result = api_.activity_enable(kind);
      if (enable_result == MSPTI_SUCCESS) {
        enabled_kinds_.push_back(kind);
      } else {
        // NODE_API is not available in some CANN releases and is not required
        // for the runtime/kernel correlation path.
        if (kind != MSPTI_ACTIVITY_KIND_NODE_API) {
          session_failed_ = true;
        }
        FLAGOS_MSPTI_LOG("[flagos] msptiActivityEnable(" << kind
                        << ") returned " << enable_result << "\n");
      }
    }
    // Memory activity is not emitted as a Kineto event, but enabling it is
    // harmless and keeps the collector ready for future allocator metadata.
    enable_optional(MSPTI_ACTIVITY_KIND_MEMORY);
    enable_optional(MSPTI_ACTIVITY_KIND_MEMCPY);
    enable_optional(MSPTI_ACTIVITY_KIND_MEMSET);

    if (session_failed_) {
      cleanup_session(subscriber, false);
      return;
    }
    active_ = true;
    g_active_tracer.store(this, std::memory_order_release);
    FLAGOS_MSPTI_LOG("[flagos] MSPTI session started\n");
  }

  void stop() override {
    if (!active_) {
      return;
    }

    // MSPTI's documented ordering is important: unsubscribe first, then flush.
    // The completion callback may run synchronously during flush, so keep the
    // global pointer installed until flush has returned.
    msptiSubscriberHandle subscriber = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      subscriber = subscriber_;
      subscriber_ = nullptr;
      active_ = false;
    }
    cleanup_session(subscriber, true);
    FLAGOS_MSPTI_LOG("[flagos] MSPTI session stopped\n");
  }

  std::vector<DeviceEvent> drain() override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_set<uint64_t> device_correlations;
    std::unordered_set<uint64_t> runtime_correlations;
    for (const auto& event : pending_) {
      if (event.vendor_correlation == 0) {
        continue;
      }
      if (event.kind == EventKind::Runtime) {
        runtime_correlations.insert(event.vendor_correlation);
      } else {
        device_correlations.insert(event.vendor_correlation);
      }
    }

    std::unordered_map<uint64_t, const PendingEvent*> selected_runtime;
    for (const auto& event : pending_) {
      if (event.kind != EventKind::Runtime ||
          !device_correlations.count(event.vendor_correlation)) {
        continue;
      }
      auto it = selected_runtime.find(event.vendor_correlation);
      if (it == selected_runtime.end() ||
          runtime_score(event.name, pending_kind(event.vendor_correlation)) >
              runtime_score(it->second->name, pending_kind(event.vendor_correlation))) {
        selected_runtime[event.vendor_correlation] = &event;
      }
    }

    std::vector<DeviceEvent> result;
    result.reserve(device_correlations.size() + selected_runtime.size());
    for (const auto& event : pending_) {
      if (event.kind == EventKind::Runtime) {
        auto it = selected_runtime.find(event.vendor_correlation);
        if (it == selected_runtime.end() || it->second != &event) {
          continue;
        }
      } else if (!runtime_correlations.count(event.vendor_correlation)) {
        // Kineto uses the vendor correlation to draw an ac2g arrow. Do not
        // export a device event whose launch/runtime half was not observed.
        continue;
      }
      if (event.vendor_correlation == 0 || !timestamps_plausible(event.start_ns, event.end_ns)) {
        continue;
      }
      DeviceEvent output;
      output.kind = event.kind;
      output.start_ns = event.start_ns;
      output.end_ns = event.end_ns;
      output.correlation_id = correlation_id(event.vendor_correlation);
      output.device = logical_device(event.physical_device);
      output.stream = event.stream;
      output.thread_id = event.thread_id;
      output.name = event.name;
      output.metadata = event.metadata;
      output.metadata["correlation"] = std::to_string(event.vendor_correlation);
      if (event.kind != EventKind::Runtime) {
        output.metadata["physical device"] = std::to_string(event.physical_device);
      }
      auto external = external_correlation_.find(event.vendor_correlation);
      if (external != external_correlation_.end()) {
        output.external_correlation_id = static_cast<int32_t>(external->second);
      }
      result.push_back(std::move(output));
    }

    pending_.clear();
    external_correlation_.clear();
    correlation_ids_.clear();
    return result;
  }

  void pushCorrelation(uint64_t id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      api_.push_external(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
    }
  }

  void popCorrelation() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      api_.pop_external(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, nullptr);
    }
  }

  int deviceCount() const override {
    const auto visible = visible_devices();
    return visible.empty() ? 1 : static_cast<int>(visible.size());
  }

  void process_buffer(uint8_t* buffer, size_t valid_size) {
    msptiActivity* record = nullptr;
    while (api_.activity_get_next_record(buffer, valid_size, &record) == MSPTI_SUCCESS) {
      process_record(record);
    }
  }

 private:
  void enable_optional(msptiActivityKind kind) {
    if (api_.activity_enable(kind) == MSPTI_SUCCESS) {
      enabled_kinds_.push_back(kind);
      return;
    }
    if (kind == MSPTI_ACTIVITY_KIND_MEMCPY) {
      FLAGOS_MSPTI_LOG("[flagos] MSPTI memcpy activity unavailable\n");
    } else if (kind == MSPTI_ACTIVITY_KIND_MEMSET) {
      FLAGOS_MSPTI_LOG("[flagos] MSPTI memset activity unavailable\n");
    }
  }

  void cleanup_session(msptiSubscriberHandle subscriber, bool flush) {
    // MSPTI requires unsubscribe before flushing. Keep the global pointer alive
    // through the flush because the completion callback may run synchronously.
    if (subscriber) {
      api_.unsubscribe(subscriber);
    }
    if (flush) {
      api_.activity_flush_all(1);
    }
    for (msptiActivityKind kind : enabled_kinds_) {
      api_.activity_disable(kind);
    }
    enabled_kinds_.clear();
    subscriber_ = nullptr;
    g_active_tracer.store(nullptr, std::memory_order_release);
  }

  static int runtime_score(const std::string& name, EventKind kind) {
    const bool launch = name.find("LaunchKernel") != std::string::npos ||
        name.find("ExecuteKernel") != std::string::npos;
    const bool memcpy = name.find("Memcpy") != std::string::npos ||
        name.find("MemCopy") != std::string::npos;
    const bool memset = name.find("Memset") != std::string::npos ||
        name.find("MemSet") != std::string::npos;
    if (kind == EventKind::Kernel) {
      return launch ? 100 : 10;
    }
    if (kind == EventKind::Memcpy) {
      return memcpy ? 100 : 10;
    }
    if (kind == EventKind::Memset) {
      return memset ? 100 : 10;
    }
    return 1;
  }

  EventKind pending_kind(uint64_t correlation) const {
    for (const auto& event : pending_) {
      if (event.vendor_correlation == correlation && event.kind != EventKind::Runtime) {
        return event.kind;
      }
    }
    return EventKind::Kernel;
  }

  uint32_t correlation_id(uint64_t vendor) {
    auto it = correlation_ids_.find(vendor);
    if (it != correlation_ids_.end()) {
      return it->second;
    }
    uint32_t value = next_correlation_id_++;
    if (value == 0) {
      value = next_correlation_id_++;
    }
    correlation_ids_[vendor] = value;
    return value;
  }

  void process_record(const msptiActivity* base) {
    if (!base) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    switch (base->kind) {
      case MSPTI_ACTIVITY_KIND_KERNEL: {
        const auto* record = reinterpret_cast<const msptiActivityKernel*>(base);
        if (!timestamps_plausible(record->start, record->end) || record->correlationId == 0) {
          return;
        }
        PendingEvent event;
        event.kind = EventKind::Kernel;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.vendor_correlation = record->correlationId;
        event.physical_device = record->ds.deviceId;
        event.stream = record->ds.streamId;
        event.name = copy_name(record->name, "AscendKernel");
        event.metadata["kernel type"] = copy_name(record->type, "unknown");
        event.metadata["device"] = std::to_string(record->ds.deviceId);
        event.metadata["stream"] = std::to_string(record->ds.streamId);
        pending_.push_back(std::move(event));
        return;
      }
      case MSPTI_ACTIVITY_KIND_MEMCPY: {
        const auto* record = reinterpret_cast<const msptiActivityMemcpy*>(base);
        if (!timestamps_plausible(record->start, record->end) || record->correlationId == 0) {
          return;
        }
        PendingEvent event;
        event.kind = EventKind::Memcpy;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.vendor_correlation = record->correlationId;
        event.physical_device = record->deviceId;
        event.stream = record->streamId;
        event.name = "Memcpy";
        event.metadata["bytes"] = std::to_string(record->bytes);
        event.metadata["copy kind"] = std::to_string(static_cast<int>(record->copyKind));
        event.metadata["async"] = std::to_string(record->isAsync);
        pending_.push_back(std::move(event));
        return;
      }
      case MSPTI_ACTIVITY_KIND_MEMSET: {
        const auto* record = reinterpret_cast<const msptiActivityMemset*>(base);
        if (!timestamps_plausible(record->start, record->end) || record->correlationId == 0) {
          return;
        }
        PendingEvent event;
        event.kind = EventKind::Memset;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.vendor_correlation = record->correlationId;
        event.physical_device = record->deviceId;
        event.stream = record->streamId;
        event.name = "Memset";
        event.metadata["bytes"] = std::to_string(record->bytes);
        event.metadata["value"] = std::to_string(record->value);
        event.metadata["async"] = std::to_string(record->isAsync);
        pending_.push_back(std::move(event));
        return;
      }
      case MSPTI_ACTIVITY_KIND_API:
      case MSPTI_ACTIVITY_KIND_ACL_API:
      case MSPTI_ACTIVITY_KIND_NODE_API:
      case MSPTI_ACTIVITY_KIND_RUNTIME_API: {
        const auto* record = reinterpret_cast<const msptiActivityApi*>(base);
        if (!timestamps_plausible(record->start, record->end) || record->correlationId == 0) {
          return;
        }
        PendingEvent event;
        event.kind = EventKind::Runtime;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.vendor_correlation = record->correlationId;
        event.thread_id = record->pt.threadId;
        event.name = copy_name(record->name, "AscendRuntime");
        event.metadata["thread"] = std::to_string(record->pt.threadId);
        pending_.push_back(std::move(event));
        return;
      }
      case MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
        const auto* record = reinterpret_cast<const msptiActivityExternalCorrelation*>(base);
        if (record->externalKind == MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 &&
            record->correlationId != 0) {
          external_correlation_[record->correlationId] = record->externalId;
        }
        return;
      }
      default:
        return;
    }
  }

  MspTiApi api_;
  bool available_ = false;
  bool active_ = false;
  bool session_failed_ = false;
  msptiSubscriberHandle subscriber_ = nullptr;
  std::vector<msptiActivityKind> enabled_kinds_;
  std::mutex mutex_;
  std::vector<PendingEvent> pending_;
  std::unordered_map<uint64_t, uint64_t> external_correlation_;
  std::unordered_map<uint64_t, uint32_t> correlation_ids_;
  uint32_t next_correlation_id_ = 1;
};

void buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
  *buffer = static_cast<uint8_t*>(aligned_alloc(kBufferAlignment, kBufferSize));
  if (!*buffer) {
    *size = 0;
    *max_num_records = 0;
    return;
  }
  *size = kBufferSize;
  *max_num_records = 0;
}

void buffer_completed(uint8_t* buffer, size_t size, size_t valid_size) {
  CannDeviceTracer* tracer = g_active_tracer.load(std::memory_order_acquire);
  if (tracer && buffer) {
    tracer->process_buffer(buffer, valid_size);
  }
  free(buffer);
}

}  // namespace

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<CannDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10

#else

namespace c10 {
namespace flagos {
namespace profiler {

class CannDeviceTracer final : public DeviceTracer {
 public:
  bool available() const override { return false; }
  void start() override {}
  void stop() override {}
  std::vector<DeviceEvent> drain() override { return {}; }
  int deviceCount() const override { return 0; }
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<CannDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10

#endif
