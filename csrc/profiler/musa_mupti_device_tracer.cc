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
// Moore Threads MUSA device tracer. MUPTI activity records are translated into
// the vendor-neutral DeviceEvent contract consumed by the Kineto adaptor.

#include "device_tracer.h"

#if defined(USE_MUSA)

#include "mupti_shim.h"
#include <flagos_env.h>

#include <cxxabi.h>
#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace c10::flagos::profiler {
namespace {

constexpr size_t kBufferSize = 8 * 1024 * 1024;
constexpr size_t kBufferAlignment = 8;
constexpr uint64_t kMaxPlausibleDurationNs = 3600ull * 1000 * 1000 * 1000;

bool debug_enabled() {
  // Was "the variable exists", which made the debug switch's =0 turn the
  // logging ON. The shared truth table reads 1/true/on/yes as on.
  static const bool enabled = flagos_env::EnvFlag("FLAGOS_TRACE");
  return enabled;
}

#define FLAGOS_MUPTI_LOG(expr) \
  do { \
    if (debug_enabled()) { \
      std::cerr << expr; \
    } \
  } while (false)

bool timestamps_plausible(uint64_t start, uint64_t end) {
  return start != 0 && end >= start && end - start <= kMaxPlausibleDurationNs;
}

std::string demangle(const char* name) {
  if (!name || !*name) {
    return "MUSA kernel";
  }
  int status = 0;
  std::unique_ptr<char, decltype(&std::free)> result(
      abi::__cxa_demangle(name, nullptr, nullptr, &status), &std::free);
  return status == 0 && result ? std::string(result.get()) : std::string(name);
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

class MusaMuptiDeviceTracer;
std::atomic<MusaMuptiDeviceTracer*> g_active_tracer{nullptr};

void buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records);
void buffer_completed(
    MUcontext context,
    uint32_t stream_id,
    uint8_t* buffer,
    size_t size,
    size_t valid_size);

class MusaMuptiDeviceTracer final : public DeviceTracer {
 public:
  bool available() const override {
#if defined(FLAGOS_HAVE_MUPTI)
    return MuptiShim::get().available();
#else
    return false;
#endif
  }

  void start() override {
#if defined(FLAGOS_HAVE_MUPTI)
    auto& api = MuptiShim::get();
    if (!api.load() || active_) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_) {
        return;
      }
      events_.clear();
      external_correlation_.clear();
      start_sample_ = sample_clock(api);
    }

    const MUptiResult callback_result =
        api.ActivityRegisterCallbacks(buffer_requested, buffer_completed);
    if (callback_result != MUPTI_SUCCESS) {
      FLAGOS_MUPTI_LOG("[flagos] muptiActivityRegisterCallbacks failed: "
                       << callback_result << "\n");
      return;
    }

    const MUpti_ActivityKind kinds[] = {
        MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        MUPTI_ACTIVITY_KIND_KERNEL,
        MUPTI_ACTIVITY_KIND_MEMCPY,
        MUPTI_ACTIVITY_KIND_MEMSET,
        MUPTI_ACTIVITY_KIND_RUNTIME,
        MUPTI_ACTIVITY_KIND_DRIVER,
        MUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    };
    for (MUpti_ActivityKind kind : kinds) {
      const MUptiResult result = api.ActivityEnable(kind);
      if (result != MUPTI_SUCCESS) {
        FLAGOS_MUPTI_LOG("[flagos] muptiActivityEnable(" << kind
                         << ") failed: " << result << "\n");
      }
    }

    active_ = true;
    g_active_tracer.store(this, std::memory_order_release);
    // Flush once after arming to make any records queued during setup visible.
    api.ActivityFlushAll(MUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    FLAGOS_MUPTI_LOG("[flagos] MUPTI session started\n");
#endif
  }

  void stop() override {
#if defined(FLAGOS_HAVE_MUPTI)
    auto& api = MuptiShim::get();
    if (!api.load()) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!active_) {
        return;
      }
      end_sample_ = sample_clock(api);
    }

    // The completion callback can run synchronously during flush. Keep the
    // active pointer installed until the forced flush has returned.
    api.ActivityFlushAll(MUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    for (MUpti_ActivityKind kind : {
             MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
             MUPTI_ACTIVITY_KIND_KERNEL,
             MUPTI_ACTIVITY_KIND_MEMCPY,
             MUPTI_ACTIVITY_KIND_MEMSET,
             MUPTI_ACTIVITY_KIND_RUNTIME,
             MUPTI_ACTIVITY_KIND_DRIVER,
             MUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION}) {
      api.ActivityDisable(kind);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      active_ = false;
      g_active_tracer.store(nullptr, std::memory_order_release);
    }
    FLAGOS_MUPTI_LOG("[flagos] MUPTI session stopped with " << events_.size()
                     << " events\n");
#endif
  }

  std::vector<DeviceEvent> drain() override {
#if defined(FLAGOS_HAVE_MUPTI)
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
#if defined(FLAGOS_HAVE_MUPTI)
    auto& api = MuptiShim::get();
    if (active_ && api.ActivityPushExternalCorrelationId) {
      api.ActivityPushExternalCorrelationId(
          MUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
    }
#else
    (void)id;
#endif
  }

  void popCorrelation() override {
#if defined(FLAGOS_HAVE_MUPTI)
    auto& api = MuptiShim::get();
    if (active_ && api.ActivityPopExternalCorrelationId) {
      uint64_t ignored = 0;
      api.ActivityPopExternalCorrelationId(
          MUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &ignored);
    }
#endif
  }

  int deviceCount() const override {
    using GetDeviceCount = int (*)(int*);
    auto fn = reinterpret_cast<GetDeviceCount>(dlsym(RTLD_DEFAULT, "musaGetDeviceCount"));
    int count = 0;
    if (fn && fn(&count) == 0 && count > 0) {
      return count;
    }
    return 1;
  }

#if defined(FLAGOS_HAVE_MUPTI)
  void process_buffer(uint8_t* buffer, size_t valid_size) {
    auto& api = MuptiShim::get();
    MUpti_Activity* record = nullptr;
    while (api.ActivityGetNextRecord(buffer, valid_size, &record) == MUPTI_SUCCESS &&
           record != nullptr) {
      process_record(record);
    }
  }
#endif

 private:
#if defined(FLAGOS_HAVE_MUPTI)
  static ClockSample sample_clock(MuptiShim& api) {
    ClockSample sample;
    if (!api.GetTimestamp) {
      return sample;
    }
    const uint64_t before = realtime_ns();
    uint64_t vendor = 0;
    const MUptiResult result = api.GetTimestamp(&vendor);
    const uint64_t after = realtime_ns();
    sample.vendor_ns = vendor;
    sample.realtime_ns = before + (after - before) / 2;
    sample.valid = result == MUPTI_SUCCESS && vendor != 0;
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
    return static_cast<uint64_t>(static_cast<long double>(start_sample_.realtime_ns) +
        elapsed * realtime_delta / vendor_delta);
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

  void process_record(const MUpti_Activity* base) {
    if (!base) {
      return;
    }
    switch (base->kind) {
      case MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      case MUPTI_ACTIVITY_KIND_KERNEL: {
        const auto* record = reinterpret_cast<const MUpti_ActivityKernel6*>(base);
        PendingEvent event;
        event.kind = EventKind::Kernel;
        event.start_ns = record->start;
        event.end_ns = record->end;
        event.correlation_id = record->correlationId;
        event.device = record->deviceId;
        event.stream = record->streamId;
        event.name = demangle(record->name);
        event.metadata["grid"] = "[" + std::to_string(record->gridX) + "," +
            std::to_string(record->gridY) + "," + std::to_string(record->gridZ) + "]";
        event.metadata["block"] = "[" + std::to_string(record->blockX) + ","+
            std::to_string(record->blockY) + "," + std::to_string(record->blockZ) + "]";
        event.metadata["registers per thread"] =
            std::to_string(record->registersPerThread);
        event.metadata["shared memory"] = std::to_string(
            static_cast<int64_t>(record->staticSharedMemory) +
            record->dynamicSharedMemory);
        event.metadata["device"] = std::to_string(record->deviceId);
        event.metadata["context"] = std::to_string(record->contextId);
        event.metadata["stream"] = std::to_string(record->streamId);
        event.metadata["correlation"] = std::to_string(record->correlationId);
        event.metadata["queued"] = timestamps_plausible(record->queued, record->start)
            ? std::to_string(record->queued) : "0";
        append(std::move(event));
        return;
      }
      case MUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto* record = reinterpret_cast<const MUpti_ActivityMemcpy4*>(base);
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
      case MUPTI_ACTIVITY_KIND_MEMSET: {
        const auto* record = reinterpret_cast<const MUpti_ActivityMemset3*>(base);
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
      case MUPTI_ACTIVITY_KIND_RUNTIME:
      case MUPTI_ACTIVITY_KIND_DRIVER: {
        const auto* record = reinterpret_cast<const MUpti_ActivityAPI*>(base);
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
      case MUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
        const auto* record = reinterpret_cast<const MUpti_ActivityExternalCorrelation*>(base);
        if (record->externalKind == MUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 &&
            record->correlationId != 0) {
          std::lock_guard<std::mutex> lock(mutex_);
          external_correlation_[record->correlationId] = record->externalId;
        }
        return;
      }
      default:
        return;
    }
  }

  std::string callback_name(MUpti_ActivityKind kind, uint32_t cbid) const {
    const char* name = nullptr;
    if (MuptiShim::get().GetCallbackName &&
        MuptiShim::get().GetCallbackName(
            kind == MUPTI_ACTIVITY_KIND_DRIVER ? MUPTI_CB_DOMAIN_DRIVER_API
                                                : MUPTI_CB_DOMAIN_RUNTIME_API,
            cbid, &name) == MUPTI_SUCCESS &&
        name && *name) {
      return name;
    }
    return kind == MUPTI_ACTIVITY_KIND_DRIVER ? "musaDriver" : "musaRuntime";
  }

  std::mutex mutex_;
  bool active_ = false;
  std::vector<PendingEvent> events_;
  std::unordered_map<uint32_t, uint64_t> external_correlation_;
  ClockSample start_sample_;
  ClockSample end_sample_;
#endif
};

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

void buffer_completed(
    MUcontext context,
    uint32_t stream_id,
    uint8_t* buffer,
    size_t size,
    size_t valid_size) {
  (void)context;
  (void)stream_id;
  (void)size;
  MusaMuptiDeviceTracer* tracer = g_active_tracer.load(std::memory_order_acquire);
#if defined(FLAGOS_HAVE_MUPTI)
  if (tracer && buffer) {
    tracer->process_buffer(buffer, valid_size);
  }
#endif
  std::free(buffer);
}

}  // namespace

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<MusaMuptiDeviceTracer>();
}

}  // namespace c10::flagos::profiler

#else

namespace c10::flagos::profiler {
class MusaMuptiDeviceTracer final : public DeviceTracer {
 public:
  bool available() const override { return false; }
  void start() override {}
  void stop() override {}
  std::vector<DeviceEvent> drain() override { return {}; }
  int deviceCount() const override { return 0; }
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<MusaMuptiDeviceTracer>();
}
}  // namespace c10::flagos::profiler

#endif
