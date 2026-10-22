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
// Enflame GCU device tracer. TOPSPTI activity records are translated into the
// vendor-neutral DeviceEvent contract consumed by the Kineto adaptor.

#include "device_tracer.h"

#if defined(USE_GCU)

#include "topspti_shim.h"
#include <flagos_env.h>

#include <cxxabi.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(FLAGOS_HAVE_TOPSPTI)
#include <topspti_activity.h>
#include <topspti_callbacks.h>
#endif

namespace c10::flagos::profiler {
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

#define FLAGOS_TOPSPTI_LOG(expr) \
  do { \
    if (debug_enabled()) { \
      std::cerr << expr; \
    } \
  } while (false)

bool timestamps_plausible(uint64_t start, uint64_t end) {
  return start != 0 && end >= start && end - start <= kMaxPlausibleDurationNs;
}

uint64_t realtime_ns() {
  timespec ts{};
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

struct ClockSample {
  uint64_t vendor_ns = 0;
  uint64_t realtime_ns = 0;
  bool valid = false;
};

struct PendingEvent {
  EventKind kind;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint32_t correlation_id = 0;
  uint32_t device = 0;
  uint32_t stream = 0;
  uint32_t thread_id = 0;
  std::string name;
  std::map<std::string, std::string> metadata;
};

class GcuTopsptiDeviceTracer;
std::atomic<GcuTopsptiDeviceTracer*> g_active_tracer{nullptr};
thread_local uint64_t g_current_correlation = 0;

#if defined(FLAGOS_HAVE_TOPSPTI)
void buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records);
void buffer_completed(uint8_t* buffer, size_t size, size_t valid_size);

void topspti_callback(
    void* userdata,
    Topspti_CallbackDomain domain,
    Topspti_CallbackId cbid,
    const void* callback_data);
#endif

class GcuTopsptiDeviceTracer final : public DeviceTracer {
 public:
  bool available() const override {
#if defined(FLAGOS_HAVE_TOPSPTI)
    return TopsptiShim::get().load() && TopsptiShim::get().available();
#else
    return false;
#endif
  }

  void start() override {
#if defined(FLAGOS_HAVE_TOPSPTI)
    auto& api = TopsptiShim::get();
    if (!api.load() || active_) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      events_.clear();
      external_correlation_.clear();
      start_sample_ = sample_clock(api);
    }

    if (api.activity_register_callbacks(buffer_requested, buffer_completed) != 0) {
      FLAGOS_TOPSPTI_LOG("[flagos] topsptiActivityRegisterCallbacks failed\n");
      return;
    }

    const Topspti_ActivityKind kinds[] = {
        TOPSPTI_ACTIVITY_KIND_KERNEL,
        TOPSPTI_ACTIVITY_KIND_MEMCPY,
        TOPSPTI_ACTIVITY_KIND_MEMSET,
        TOPSPTI_ACTIVITY_KIND_RUNTIME,
        TOPSPTI_ACTIVITY_KIND_DRIVER,
    };
    for (Topspti_ActivityKind kind : kinds) {
      if (api.activity_enable(kind) != 0) {
        FLAGOS_TOPSPTI_LOG("[flagos] topsptiActivityEnable(" << kind
                           << ") failed\n");
      }
    }

    if (api.subscribe && api.enable_all_domains) {
      if (api.subscribe(&subscriber_, topspti_callback, this) == 0) {
        api.enable_all_domains(1, subscriber_);
      }
    }

    active_ = true;
    g_active_tracer.store(this, std::memory_order_release);
    api.activity_flush_all(1);
    FLAGOS_TOPSPTI_LOG("[flagos] TOPSPTI session started\n");
#endif
  }

  void stop() override {
#if defined(FLAGOS_HAVE_TOPSPTI)
    auto& api = TopsptiShim::get();
    if (!api.load() || !active_) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      end_sample_ = sample_clock(api);
    }
    api.activity_flush_all(1);
    for (Topspti_ActivityKind kind : {
             TOPSPTI_ACTIVITY_KIND_KERNEL,
             TOPSPTI_ACTIVITY_KIND_MEMCPY,
             TOPSPTI_ACTIVITY_KIND_MEMSET,
             TOPSPTI_ACTIVITY_KIND_RUNTIME,
             TOPSPTI_ACTIVITY_KIND_DRIVER}) {
      if (api.activity_disable) {
        api.activity_disable(kind);
      }
    }
    if (api.unsubscribe && subscriber_) {
      api.unsubscribe(subscriber_);
      subscriber_ = nullptr;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_ = false;
      g_active_tracer.store(nullptr, std::memory_order_release);
    }
    FLAGOS_TOPSPTI_LOG("[flagos] TOPSPTI session stopped with " << events_.size()
                       << " events\n");
#endif
  }

  std::vector<DeviceEvent> drain() override {
#if defined(FLAGOS_HAVE_TOPSPTI)
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DeviceEvent> result;
    result.reserve(events_.size());
    for (const PendingEvent& pending : events_) {
      DeviceEvent event;
      event.kind = pending.kind;
      event.start_ns = to_realtime(pending.start_ns);
      event.end_ns = to_realtime(pending.end_ns);
      event.correlation_id = pending.correlation_id;
      event.device = pending.device;
      event.stream = pending.stream;
      event.thread_id = pending.thread_id;
      event.name = pending.name;
      event.metadata = pending.metadata;
      const auto external = external_correlation_.find(pending.correlation_id);
      if (external != external_correlation_.end()) {
        event.external_correlation_id =
            static_cast<int32_t>(external->second);
      }
      result.push_back(std::move(event));
    }
    events_.clear();
    external_correlation_.clear();
    return result;
#else
    return {};
#endif
  }

  void pushCorrelation(uint64_t id) override {
    g_current_correlation = id;
  }

  void popCorrelation() override {
    g_current_correlation = 0;
  }

#if defined(FLAGOS_HAVE_TOPSPTI)
  void record_external_correlation(uint32_t correlation_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (g_current_correlation != 0) {
      external_correlation_[correlation_id] = g_current_correlation;
    }
  }
#endif

  int deviceCount() const override {
    using GetDeviceCount = int (*)(int*);
    auto fn = reinterpret_cast<GetDeviceCount>(
        dlsym(RTLD_DEFAULT, "topsGetDeviceCount"));
    int count = 0;
    if (fn && fn(&count) == 0 && count > 0) {
      return count;
    }
    return 1;
  }

#if defined(FLAGOS_HAVE_TOPSPTI)
  void process_buffer(uint8_t* buffer, size_t valid_size) {
    auto& api = TopsptiShim::get();
    Topspti_Activity* record = nullptr;
    while (api.activity_get_next_record(buffer, valid_size, &record) == 0 &&
           record != nullptr) {
      process_record(record);
    }
  }
#endif

 private:
#if defined(FLAGOS_HAVE_TOPSPTI)
  static ClockSample sample_clock(TopsptiShim& api) {
    ClockSample sample;
    if (!api.get_timestamp) {
      return sample;
    }
    const uint64_t before = realtime_ns();
    uint64_t vendor = 0;
    const TopsptiResult result = api.get_timestamp(&vendor);
    const uint64_t after = realtime_ns();
    sample.vendor_ns = vendor;
    sample.realtime_ns = before + (after - before) / 2;
    sample.valid = result == 0 && vendor != 0;
    return sample;
  }

  uint64_t to_realtime(uint64_t vendor_ns) const {
    if (!start_sample_.valid) {
      return vendor_ns;
    }
    if (!end_sample_.valid || end_sample_.vendor_ns <= start_sample_.vendor_ns) {
      return start_sample_.realtime_ns + (vendor_ns - start_sample_.vendor_ns);
    }
    const long double vendor_delta = static_cast<long double>(
        end_sample_.vendor_ns - start_sample_.vendor_ns);
    const long double realtime_delta = static_cast<long double>(
        end_sample_.realtime_ns - start_sample_.realtime_ns);
    const long double elapsed = static_cast<long double>(vendor_ns) -
        static_cast<long double>(start_sample_.vendor_ns);
    return static_cast<uint64_t>(static_cast<long double>(
        start_sample_.realtime_ns) + elapsed * realtime_delta / vendor_delta);
  }

  void append(PendingEvent event) {
    if (!timestamps_plausible(event.start_ns, event.end_ns) ||
        event.correlation_id == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      events_.push_back(std::move(event));
    }
  }

  void process_record(const Topspti_Activity* base) {
    if (!base) {
      return;
    }
    switch (base->kind) {
      case TOPSPTI_ACTIVITY_KIND_KERNEL: {
        const auto* record = reinterpret_cast<const Topspti_ActivityKernel*>(base);
        PendingEvent event;
        event.kind = EventKind::Kernel;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.correlation_id = record->correlationId;
        event.device = record->deviceId;
        event.stream = record->streamId;
        event.name = record->name && *record->name ? record->name : "GCU kernel";
        event.metadata["grid"] = "[" + std::to_string(record->gridX) + "," +
            std::to_string(record->gridY) + "," + std::to_string(record->gridZ) + "]";
        event.metadata["block"] = "[" + std::to_string(record->blockX) + "," +
            std::to_string(record->blockY) + "," + std::to_string(record->blockZ) + "]";
        event.metadata["device"] = std::to_string(record->deviceId);
        event.metadata["context"] = std::to_string(record->contextId);
        event.metadata["stream"] = std::to_string(record->streamId);
        event.metadata["correlation"] = std::to_string(record->correlationId);
        append(std::move(event));
        return;
      }
      case TOPSPTI_ACTIVITY_KIND_MEMCPY: {
        const auto* record = reinterpret_cast<const Topspti_ActivityMemcpy*>(base);
        PendingEvent event;
        event.kind = EventKind::Memcpy;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.correlation_id = record->correlationId;
        event.device = record->deviceId;
        event.stream = record->streamId;
        event.name = "Memcpy";
        event.metadata["bytes"] = std::to_string(record->bytes);
        event.metadata["copy kind"] = std::to_string(record->copyKind);
        event.metadata["device"] = std::to_string(record->deviceId);
        event.metadata["context"] = std::to_string(record->contextId);
        event.metadata["stream"] = std::to_string(record->streamId);
        event.metadata["correlation"] = std::to_string(record->correlationId);
        append(std::move(event));
        return;
      }
      case TOPSPTI_ACTIVITY_KIND_MEMSET: {
        const auto* record = reinterpret_cast<const Topspti_ActivityMemset*>(base);
        PendingEvent event;
        event.kind = EventKind::Memset;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.correlation_id = record->correlationId;
        event.device = record->deviceId;
        event.stream = record->streamId;
        event.name = "Memset";
        event.metadata["bytes"] = std::to_string(record->bytes);
        event.metadata["value"] = std::to_string(record->value);
        event.metadata["device"] = std::to_string(record->deviceId);
        event.metadata["context"] = std::to_string(record->contextId);
        event.metadata["stream"] = std::to_string(record->streamId);
        event.metadata["correlation"] = std::to_string(record->correlationId);
        append(std::move(event));
        return;
      }
      case TOPSPTI_ACTIVITY_KIND_RUNTIME:
      case TOPSPTI_ACTIVITY_KIND_DRIVER: {
        const auto* record = reinterpret_cast<const Topspti_ActivityAPI*>(base);
        PendingEvent event;
        event.kind = EventKind::Runtime;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.correlation_id = record->correlationId;
        event.thread_id = record->threadId;
        event.name = callback_name(base->kind, record->cbid);
        event.metadata["cbid"] = std::to_string(record->cbid);
        event.metadata["correlation"] = std::to_string(record->correlationId);
        event.metadata["thread"] = std::to_string(record->threadId);
        append(std::move(event));
        return;
      }
      default:
        return;
    }
  }

  std::string callback_name(Topspti_ActivityKind kind, uint32_t cbid) const {
    const char* name = nullptr;
    auto& api = TopsptiShim::get();
    using GetCallbackName = TopsptiResult (*)(
        Topspti_CallbackDomain, uint32_t, const char**);
    auto get_name = reinterpret_cast<GetCallbackName>(
        dlsym(api.handle, "topsptiGetCallbackName"));
    if (get_name && get_name(
            kind == TOPSPTI_ACTIVITY_KIND_DRIVER
                ? TOPSPTI_CB_DOMAIN_DRIVER_API
                : TOPSPTI_CB_DOMAIN_RUNTIME_API,
            cbid, &name) == 0 && name && *name) {
      return name;
    }
    return kind == TOPSPTI_ACTIVITY_KIND_DRIVER ? "topsDriver" : "topsRuntime";
  }

  std::mutex mutex_;
  bool active_ = false;
  uint64_t current_correlation_ = 0;
  std::vector<PendingEvent> events_;
  std::unordered_map<uint32_t, uint64_t> external_correlation_;
  ClockSample start_sample_;
  ClockSample end_sample_;
  Topspti_SubscriberHandle subscriber_ = nullptr;
#endif
};

#if defined(FLAGOS_HAVE_TOPSPTI)
void buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
  *buffer = static_cast<uint8_t*>(std::aligned_alloc(kBufferAlignment, kBufferSize));
  if (!*buffer) {
    *size = 0;
    *max_num_records = 0;
    return;
  }
  *size = kBufferSize;
  *max_num_records = 0;
}

void buffer_completed(uint8_t* buffer, size_t size, size_t valid_size) {
  (void)size;
  auto* tracer = g_active_tracer.load(std::memory_order_acquire);
  if (tracer && buffer) {
    tracer->process_buffer(buffer, valid_size);
  }
  std::free(buffer);
}

void topspti_callback(
    void* userdata,
    Topspti_CallbackDomain domain,
    Topspti_CallbackId cbid,
    const void* callback_data) {
  (void)domain;
  (void)cbid;
  auto* tracer = static_cast<GcuTopsptiDeviceTracer*>(userdata);
  const auto* data = static_cast<const Topspti_CallbackData*>(callback_data);
  if (!tracer || !data) {
    return;
  }
  // TOPSPTI has no push/pop external-correlation API. The API-enter callback is
  // the only place where torch's correlation (set by the kineto session on this
  // thread) and the vendor correlation id are both known, so the mapping is
  // recorded here and resolved in drain() once device records arrive.
  if (data->callbackSite == TOPSPTI_API_ENTER) {
    tracer->record_external_correlation(data->correlationId);
  }
}
#endif

}  // namespace

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<GcuTopsptiDeviceTracer>();
}

}  // namespace c10::flagos::profiler

#else

namespace c10::flagos::profiler {
class GcuTopsptiDeviceTracer final : public DeviceTracer {
 public:
  bool available() const override { return false; }
  void start() override {}
  void stop() override {}
  std::vector<DeviceEvent> drain() override { return {}; }
  int deviceCount() const override { return 0; }
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<GcuTopsptiDeviceTracer>();
}
}  // namespace c10::flagos::profiler

#endif
