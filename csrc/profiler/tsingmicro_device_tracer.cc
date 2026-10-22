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
// DeviceTracer for the TsingMicro TX8 NPU.
//
// The collection itself lives in a separate library, libtorch_fl_profiler.so
// (csrc/profiler/tsingmicro_profiler_tool.cc), which links only
// rocprofiler-sdk. It has to be separate: rocprofiler dlopens its tools, and
// libtorch_fl.so depends on libflagos.so -> libhpgr.so, so keeping the tool in
// the main library would make rocprofiler drag the runtime in while it was
// still initializing and the tool would never get a client id.
//
// This side reaches the tool through the C API in tsingmicro_profiler_api.h,
// resolved on first use with dlopen/dlsym.
//
// Lifecycle: Kineto drives one session per profiling run (configure -> start
// -> processTrace -> stop), while the tool's tracing services are configured
// once when rocprofiler initializes it.
//
// Requires TSM_PROFILER_EN=1, the tsmprofiler libraries on LD_PRELOAD, and
// ROCP_TOOL_LIBRARIES pointing at the tool; see
// docs/vendors/tsingmicro/installation.md. Without them the tool never
// registers, start() finds no library, and the trace simply has no device
// timeline.

#include "device_tracer.h"

#if defined(FLAGOS_HAVE_TSMPROFILER)

#include "profiler/tsingmicro_profiler_api.h"

#include <flagos.h>
#include <flagos_env.h>

// Only for the kernel dispatch record layout; the symbols come from
// libtorch_fl_profiler.so at runtime, not from a link-time dependency.
#include <rocprofiler-sdk/buffer_tracing.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace c10 {
namespace flagos {
namespace profiler {

namespace {

using kernel_record_t = rocprofiler_buffer_tracing_tsm_kernel_dispatch_record_t;

inline bool tsm_debug() {
  static const bool on = flagos_env::EnvFlag("FLAGOS_TRACE");
  return on;
}
#define FLAGOS_TSM_LOG(expr) \
  do { if (tsm_debug()) { std::cerr << expr; } } while (0)

// Lane of the default stream, and the first lane handed out. The default
// stream never owns lane 0: kineto writes the device row's process metadata on
// tid 0 (output_json.cpp: handleDeviceInfo), so a lane there is swallowed by
// the device row. Pinning the default stream also gives kernels that cannot be
// attributed to any op a stable lane to fall back to.
constexpr int64_t kDefaultStreamLane = 1;

// ---------------------------------------------------------------------------
// The tool's C API, resolved once from libtorch_fl_profiler.so.
// ---------------------------------------------------------------------------
struct ToolApi {
  int (*start)();
  int (*stop)();
  int64_t (*clock_offset_ns)();
  void (*push_correlation)(uint64_t);
  void (*pop_correlation)();
  int (*take_records)(torch_fl_profiler_records_t*);
  void (*release_records)(torch_fl_profiler_records_t*);
  int (*lookup_launch)(uint64_t, uint64_t*, uint64_t*);
  int64_t (*push_count)();
  int64_t (*correlation_cb_count)();
  int64_t (*api_cb_count)();
};

template <typename T>
bool resolve(void* handle, const char* name, T* out) {
  *out = reinterpret_cast<T>(dlsym(handle, name));
  return *out != nullptr;
}

const ToolApi* tool_api() {
  static const ToolApi* api = []() -> const ToolApi* {
    // FLAGOS_TRACER_LIBRARY overrides the library to dlopen, matching the
    // CUPTI/MUPTI/TOPSPTI shims; the default is the tool next to
    // libtorch_fl.so. When rocprofiler already loaded it through
    // ROCP_TOOL_LIBRARIES, dlopen returns that same handle by SONAME, so the
    // tool's context is not created a second time.
    void* handle = nullptr;
    const std::string override_library =
        flagos_env::EnvValue("FLAGOS_TRACER_LIBRARY");
    if (!override_library.empty()) {
      handle = dlopen(override_library.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) {
        FLAGOS_TSM_LOG("[flagos] FLAGOS_TRACER_LIBRARY=" << override_library
                       << " could not be loaded: " << dlerror() << "\n");
      }
    }
    if (handle == nullptr) {
      handle = dlopen("libtorch_fl_profiler.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == nullptr) {
      FLAGOS_TSM_LOG("[flagos] cannot load libtorch_fl_profiler.so: "
                     << static_cast<const char*>(dlerror()) << "\n");
      return nullptr;
    }
    static ToolApi resolved{};
    const bool ok =
        resolve(handle, "torch_fl_profiler_start", &resolved.start) &&
        resolve(handle, "torch_fl_profiler_stop", &resolved.stop) &&
        resolve(handle, "torch_fl_profiler_clock_offset_ns",
                &resolved.clock_offset_ns) &&
        resolve(handle, "torch_fl_profiler_push_correlation",
                &resolved.push_correlation) &&
        resolve(handle, "torch_fl_profiler_pop_correlation",
                &resolved.pop_correlation) &&
        resolve(handle, "torch_fl_profiler_take_records",
                &resolved.take_records) &&
        resolve(handle, "torch_fl_profiler_release_records",
                &resolved.release_records) &&
        resolve(handle, "torch_fl_profiler_lookup_launch",
                &resolved.lookup_launch) &&
        resolve(handle, "torch_fl_profiler_push_count", &resolved.push_count) &&
        resolve(handle, "torch_fl_profiler_correlation_cb_count",
                &resolved.correlation_cb_count) &&
        resolve(handle, "torch_fl_profiler_api_cb_count",
                &resolved.api_cb_count);
    if (!ok) {
      FLAGOS_TSM_LOG(
          "[flagos] libtorch_fl_profiler.so is missing API symbols\n");
      return nullptr;
    }
    return &resolved;
  }();
  return api;
}

// rocprofiler's clock -> Unix epoch ns. Kineto's capture window, and every CPU
// op in the trace, are stamped with epoch ns.
uint64_t toEpochNs(uint64_t timestamp, int64_t offset) {
  return static_cast<uint64_t>(static_cast<int64_t>(timestamp) + offset);
}

std::string format_dim3(const rocprofiler_dim3_t& dims) {
  return "[" + std::to_string(dims.x) + ", " + std::to_string(dims.y) + ", " +
         std::to_string(dims.z) + "]";
}

}  // namespace

/**
 * TsingMicroDeviceTracer: TX8 implementation of the DeviceTracer interface.
 *
 * Owns the op -> stream bookkeeping and the translation of rocprofiler
 * dispatch records into vendor-neutral DeviceEvents. The rocprofiler tool
 * itself is a separate library; see the note at the top of this file.
 */
class TsingMicroDeviceTracer : public DeviceTracer {
 public:
  bool available() const override {
    // The build found the rocprofiler headers, so the tracer is compiled and
    // kineto registration is meaningful. Whether the tool is actually loaded
    // depends on the run-time environment (TSM_PROFILER_EN, LD_PRELOAD,
    // ROCP_TOOL_LIBRARIES); when it is missing, start() logs it and the
    // session simply produces no device timeline, exactly like a CUDA run
    // without CUPTI.
    return true;
  }

  void start() override {
    // Sample the device once, here: every event has to agree on it, and the
    // current device can change under us while the window is open.
    int device = 0;
    device_id_ = (::GetDevice(&device) == Success) ? device : 0;
    {
      std::lock_guard<std::mutex> lock(stream_mutex_);
      stream_by_op_.clear();
      stream_ordinal_.clear();
      // Seed the default stream so a kernel with no op to attribute it to
      // lands on a stable lane even when the window's first op ran elsewhere.
      stream_ordinal_[0] = kDefaultStreamLane;
      next_stream_lane_ = kDefaultStreamLane + 1;
    }
    const ToolApi* api = tool_api();
    if (api == nullptr) {
      // Unconditional, once: a trace with no device timeline is otherwise a
      // completely silent failure, and this warning is the breadcrumb a
      // "why is my NPU timeline empty" report needs. Same rationale as the
      // empty-resolver warning in flagos_kineto_profiler.cc.
      static std::once_flag warn_once;
      std::call_once(warn_once, [] {
        std::cerr
            << "[flagos] TsingMicro profiler tool (libtorch_fl_profiler.so) is "
               "not loaded.\n"
            << "[flagos]   The trace will have no NPU device timeline. Set "
               "TSM_PROFILER_EN=1, put the tsmprofiler\n"
            << "[flagos]   libraries on LD_PRELOAD, and name the tool in "
               "ROCP_TOOL_LIBRARIES; see\n"
            << "[flagos]   docs/vendors/tsingmicro/installation.md.\n";
      });
      return;
    }
    api->start();
  }

  void stop() override {
    const ToolApi* api = tool_api();
    if (api != nullptr) {
      api->stop();
    }
  }

  std::vector<DeviceEvent> drain() override {
    std::vector<DeviceEvent> events;
    const ToolApi* api = tool_api();
    if (api == nullptr) {
      return events;
    }

    torch_fl_profiler_records_t records{};
    if (api->take_records(&records) != 0) {
      return events;
    }
    // Hand the buffer back on every exit path.
    struct RecordsGuard {
      const ToolApi* api;
      torch_fl_profiler_records_t* records;
      ~RecordsGuard() { api->release_records(records); }
    } guard{api, &records};

    const int64_t offset = api->clock_offset_ns();
    FLAGOS_TSM_LOG("[flagos] tx8 drain: " << records.count
                   << " record(s), profiler_push_count=" << api->push_count()
                   << ", correlation callbacks="
                   << api->correlation_cb_count()
                   << ", runtime api callbacks=" << api->api_cb_count()
                   << "\n");

    const auto* kernels = static_cast<const kernel_record_t*>(records.data);
    std::lock_guard<std::mutex> lock(stream_mutex_);
    events.reserve(records.count * 2);
    // DeviceEvent::correlation_id is 32-bit and rocprofiler's internal ids are
    // 64-bit, so pair each launch with its kernel through a dense id of our
    // own rather than truncating the vendor's.
    uint32_t next_correlation = 1;
    for (size_t i = 0; i < records.count; ++i) {
      const auto& record = kernels[i];
      const auto& info = record.dispatch_info;
      // kernel_name is a fixed 256-byte field and is not guaranteed to be
      // NUL-terminated, so bound the read explicitly.
      const std::string kernel_name(
          info.kernel_name,
          strnlen(info.kernel_name, sizeof(info.kernel_name)));

      // The AP window is the interval the viewer gets; the kernel's own
      // start/end is the fallback when the runtime reports no AP interval.
      uint64_t start_ns = record.ap_start_timestamp;
      uint64_t end_ns = record.ap_end_timestamp;
      if (end_ns <= start_ns) {
        start_ns = record.start_timestamp;
        end_ns = record.end_timestamp;
      }
      if (end_ns <= start_ns) {
        continue;
      }

      // Correlate the kernel back to the aten op that launched it. The op id
      // pushed in pushCorrelationId is the thread-local external correlation
      // id, and the runtime copies it onto the kernel dispatch record, so this
      // is an exact join -- no timestamp guessing and no lookup table.
      const uint64_t op_id = record.correlation_id.external.value;
      const uint32_t correlation = next_correlation++;

      // Which stream row this kernel goes on, from the op that launched it.
      int64_t stream = kDefaultStreamLane;
      const auto op_it = stream_by_op_.find(op_id);
      if (op_it != stream_by_op_.end()) {
        stream = op_it->second;
      }

      // The CPU -> kernel arrow starts at the launch API call. Emit it as a
      // runtime activity on the op's own track, the way CUDA's
      // cudaLaunchKernel record does, so the flow pair has a real event at
      // both ends.
      uint64_t launch_start = 0;
      uint64_t launch_end = 0;
      if (api->lookup_launch(record.correlation_id.internal, &launch_start,
                             &launch_end) == 0 &&
          launch_end > launch_start) {
        DeviceEvent runtime;
        runtime.kind = EventKind::Runtime;
        runtime.start_ns = toEpochNs(launch_start, offset);
        runtime.end_ns = toEpochNs(launch_end, offset);
        runtime.correlation_id = correlation;
        runtime.device = static_cast<uint32_t>(device_id_);
        runtime.stream = static_cast<uint32_t>(stream);
        runtime.thread_id = static_cast<uint32_t>(record.thread_id);
        runtime.name = kernel_name;
        if (op_id != 0) {
          runtime.external_correlation_id = static_cast<int32_t>(op_id);
        }
        events.push_back(std::move(runtime));
      }

      DeviceEvent kernel;
      kernel.kind = EventKind::Kernel;
      kernel.start_ns = toEpochNs(start_ns, offset);
      kernel.end_ns = toEpochNs(end_ns, offset);
      kernel.correlation_id = correlation;
      kernel.device = static_cast<uint32_t>(device_id_);
      kernel.stream = static_cast<uint32_t>(stream);
      kernel.thread_id = static_cast<uint32_t>(record.thread_id);
      kernel.name = kernel_name;
      // No external id means the runtime did not stamp this record (the op
      // ran before the window opened, or TSM_PROFILER_EN is unset); leave the
      // optional empty so the adaptor does not try to link id 0.
      if (op_id != 0) {
        kernel.external_correlation_id = static_cast<int32_t>(op_id);
      }
      // Launch geometry, under the keys the CUDA and ROCm profilers use
      // (MetadataFieldCatalog.h: kGrid/kBlock), so the viewer shows it the
      // same way as on the GPU.
      kernel.metadata["grid"] = format_dim3(info.grid_size);
      kernel.metadata["block"] = format_dim3(info.block_size);
      events.push_back(std::move(kernel));
    }
    return events;
  }

  void pushCorrelation(uint64_t id) override {
    // Capture the stream here, not in drain(): this is the only point where
    // the current stream is still the one the op is about to launch on.
    int device = 0;
    if (::GetDevice(&device) != Success) {
      device = 0;
    }
    const int64_t raw_stream =
        static_cast<int64_t>(reinterpret_cast<intptr_t>(
            ::GetCurrentStreamForDevice(device)));
    {
      std::lock_guard<std::mutex> lock(stream_mutex_);
      auto it = stream_ordinal_.find(raw_stream);
      if (it == stream_ordinal_.end()) {
        it = stream_ordinal_.emplace(raw_stream, next_stream_lane_++).first;
      }
      stream_by_op_[id] = it->second;
    }
    const ToolApi* api = tool_api();
    if (api != nullptr) {
      api->push_correlation(id);
    }
  }

  void popCorrelation() override {
    const ToolApi* api = tool_api();
    if (api != nullptr) {
      api->pop_correlation();
    }
  }

  int deviceCount() const override {
    int count = 0;
    if (::GetDeviceCount(&count) != Success) {
      return 0;
    }
    return count;
  }

 private:
  // op id -> the lane of the stream current when that op entered.
  // stream_ordinal_ is the raw stream handle -> lane map behind it, and
  // next_stream_lane_ the ordinal to hand out next; both are reset in start().
  std::mutex stream_mutex_;
  std::unordered_map<uint64_t, int64_t> stream_by_op_;
  std::unordered_map<int64_t, int64_t> stream_ordinal_;
  int64_t next_stream_lane_{kDefaultStreamLane + 1};
  // Device the window was recorded on, sampled in start().
  int32_t device_id_{0};
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<TsingMicroDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10

#else  // !defined(FLAGOS_HAVE_TSMPROFILER)

namespace c10 {
namespace flagos {
namespace profiler {

class TsingMicroDeviceTracer final : public DeviceTracer {
 public:
  bool available() const override { return false; }
  void start() override {}
  void stop() override {}
  std::vector<DeviceEvent> drain() override { return {}; }
  int deviceCount() const override { return 0; }
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<TsingMicroDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10

#endif
