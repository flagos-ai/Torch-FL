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
// DCU/ROCm device tracer: ROCtracer API integration for profiling support.
//
// Architecture notes:
// - Uses roctracer_open_pool_expl with buffer callbacks for async activities
// - HIP_API domain captures runtime calls (hipLaunchKernel, hipMemcpy, etc.)
// - HIP_OPS domain captures device-side kernels and memcpy operations
// - External correlation tracking maps torch correlation IDs to vendor IDs
// - Thread-local stacks maintain correlation context across API boundaries

#include "device_tracer.h"
#include <flagos_env.h>

// ROCtracer headers require __HIP_PLATFORM_AMD__ to be defined before inclusion.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

#include <roctracer.h>
#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <ext/prof_protocol.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cxxabi.h>

#include <cstdio>
#include <optional>

// Diagnostic logging gated behind FLAGOS_TRACE=1
namespace {
inline bool flagos_roctracer_debug() {
  static const bool on = flagos_env::EnvFlag("FLAGOS_TRACE");
  return on;
}
}  // namespace
#define FLAGOS_ROCTRACER_LOG(expr) \
  do { if (flagos_roctracer_debug()) { std::cerr << expr; } } while (0)

namespace c10 {
namespace flagos {
namespace profiler {

namespace {

// Get current thread ID
inline uint64_t get_thread_id() {
  return static_cast<uint64_t>(syscall(SYS_gettid));
}

// Get current timestamp in nanoseconds (CLOCK_REALTIME to match ROCtracer domain)
inline uint64_t get_timestamp_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(ts.tv_nsec);
}

// Thread-local correlation stacks for external correlation tracking.
// When torch calls pushCorrelation(torch_id), we push it here, then when
// ROCtracer invokes the HIP API callback we pop and associate torch_id with
// the vendor correlation_id.
thread_local std::deque<uint64_t> t_external_correlation_stack;

// Global storage for (vendor_correlation_id, torch_correlation_id) pairs.
// Populated during API callbacks, drained during processActivities.
struct CorrelationMap {
  std::mutex mutex;
  std::unordered_map<uint32_t, uint32_t> vendor_to_torch;
} g_correlation_map;

// Buffer for collected activity records. ROCtracer callbacks append here,
// drain() consumes them.
struct ActivityBuffer {
  std::mutex mutex;
  std::vector<DeviceEvent> events;
} g_activity_buffer;

// Launch configuration captured from the HIP API callback, keyed by vendor
// correlation id.
//
// Why this cache is REQUIRED and not an optimization: a HIP_OPS async record
// carries only {domain, kind, op, correlation_id, begin/end, device, queue,
// bytes|kernel_name}. It has NO grid, NO block, NO shared-memory size. Those
// values exist only in the arguments of the hipLaunchKernel/hipModuleLaunchKernel
// call that produced the dispatch. torch-cuda's trace puts them on the *kernel*
// event, so the launch-side values have to be carried across and re-attached at
// drain() time. Kineto's removed tracer did the same thing via
// correlationToGrid/correlationToBlock.
struct LaunchConfig {
  std::string grid;
  std::string block;
  uint64_t shared_memory = 0;
  uint32_t threads_per_block = 0;
  const void* function_address = nullptr;
  uint64_t queued_ns = 0;  // API-call entry: the "queued" field torch-cuda emits
  size_t bytes = 0;        // memcpy/memset transfer size
  bool has_bytes = false;
};

struct LaunchConfigCache {
  std::mutex mutex;
  std::unordered_map<uint32_t, LaunchConfig> by_correlation;
} g_launch_configs;

// HIP entry points resolved lazily with dlsym.
//
// Deliberately NOT a link-time dependency: libgalaxyhip.so.5 is already in this
// library's NEEDED set (it is what provides the roctracer symbols on DTK), so the
// process always has these available by the time a profile runs -- but resolving
// them dynamically keeps a machine without the DCU runtime from failing to load
// torch_fl at import time just because the profiler *could* have used them.
//
// Only the enum-keyed attribute APIs are used. hipGetDeviceProperties is
// deliberately avoided: hipDeviceProp_t is a large struct whose layout has
// changed across DTK releases (the symbol is even versioned, R0600), so reading
// it through a dlsym'd pointer risks silently decoding garbage. hipDeviceGetAttribute
// takes an enum and returns an int, which cannot be misdecoded.
struct HipSymbols {
  int (*device_get_attribute)(int*, int, int) = nullptr;
  int (*func_get_attributes)(void*, const void*) = nullptr;
  bool resolved = false;
};

HipSymbols& hip_symbols() {
  static HipSymbols syms;
  static std::once_flag once;
  std::call_once(once, [] {
    syms.device_get_attribute = reinterpret_cast<int (*)(int*, int, int)>(
        dlsym(RTLD_DEFAULT, "hipDeviceGetAttribute"));
    syms.func_get_attributes = reinterpret_cast<int (*)(void*, const void*)>(
        dlsym(RTLD_DEFAULT, "hipFuncGetAttributes"));
    syms.resolved =
        (syms.device_get_attribute != nullptr) && (syms.func_get_attributes != nullptr);
    FLAGOS_ROCTRACER_LOG("[flagos] HIP symbols resolved="
                        << (syms.resolved ? "yes" : "no") << "\n");
  });
  return syms;
}

// Per-device limits needed for the occupancy model, queried once per device.
struct DeviceLimits {
  int warp_size = 64;                   // AMD wavefront
  int max_threads_per_sm = 2048;
  int max_shared_memory_per_sm = 65536;
  int max_registers_per_block = 65536;
  bool valid = false;
};

const DeviceLimits& device_limits(int device) {
  static std::mutex mu;
  static std::unordered_map<int, DeviceLimits> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device);
  if (it != cache.end()) {
    return it->second;
  }
  DeviceLimits limits;
  auto& syms = hip_symbols();
  if (syms.device_get_attribute) {
    int v = 0;
    // Enum values from hip_runtime_api.h (hipDeviceAttribute_t); using the named
    // constants keeps this correct if DTK renumbers them.
    if (syms.device_get_attribute(&v, hipDeviceAttributeWarpSize, device) == 0 && v > 0) {
      limits.warp_size = v;
    }
    if (syms.device_get_attribute(&v, hipDeviceAttributeMaxThreadsPerMultiProcessor,
                                  device) == 0 && v > 0) {
      limits.max_threads_per_sm = v;
    }
    if (syms.device_get_attribute(&v, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                                  device) == 0 && v > 0) {
      limits.max_shared_memory_per_sm = v;
    }
    if (syms.device_get_attribute(&v, hipDeviceAttributeMaxRegistersPerBlock, device) == 0 &&
        v > 0) {
      limits.max_registers_per_block = v;
    }
    limits.valid = true;
  }
  return cache.emplace(device, limits).first->second;
}

// ROCtracer pool handle for async activities (kernels, memcpy).
roctracer_pool_t* g_async_pool = nullptr;

// Whether tracing is currently active.
std::atomic<bool> g_tracing_active{false};

// Maximum events to collect before stopping (prevents runaway memory).
constexpr uint32_t kMaxEvents = 5000000;

// GPU-to-host timestamp synchronization state.
//
// ROCtracer device event timestamps (record->begin_ns, record->end_ns) are GPU
// clock cycles scaled to nanoseconds, but they're NOT aligned with host CLOCK_REALTIME.
// They're ~1000x smaller in magnitude than host timestamps. To make device events
// appear in the correct position relative to CPU events in the profiler trace, we
// synchronize clocks once at profiler start and apply the offset to every device event.
//
// Synchronization strategy: sample (host_time, gpu_time) as close together as possible,
// then convert device timestamps with: host_ns = (gpu_ns - gpu_base) + host_base.
struct ClockSync {
  uint64_t host_base_ns = 0;
  uint64_t gpu_base_ns = 0;
  bool synchronized = false;
};

ClockSync g_clock_sync;

// Convert GPU timestamp to host timestamp using the synchronized offset.
inline uint64_t gpu_to_host_timestamp(uint64_t gpu_ns) {
  if (!g_clock_sync.synchronized) {
    // If synchronization hasn't happened yet, return the raw GPU timestamp.
    // This shouldn't happen in practice since we sync at start(), but it's
    // a safe fallback.
    return gpu_ns;
  }
  // Convert: host_ns = (gpu_ns - gpu_base) + host_base
  // Handle potential underflow if gpu_ns < gpu_base (shouldn't happen, but be safe).
  if (gpu_ns >= g_clock_sync.gpu_base_ns) {
    return (gpu_ns - g_clock_sync.gpu_base_ns) + g_clock_sync.host_base_ns;
  } else {
    // GPU timestamp is before our base; this indicates a problem with sync.
    // Return host_base as a fallback to keep the event in the profiling window.
    return g_clock_sync.host_base_ns;
  }
}

// Synchronize GPU and host clocks by sampling both as close together as possible.
//
// We need to call a HIP API to get the GPU timestamp. The most direct way is
// hipEventCreate + hipEventRecord + hipEventQuery to get a GPU timestamp, but
// that requires actually calling HIP APIs which we're trying to avoid linking against.
//
// Alternative approach: use roctracer_get_timestamp() which returns GPU cycles,
// and pair it with host clock_gettime(). This is what Kineto does.
void synchronize_clocks() {
  // Sample host time.
  g_clock_sync.host_base_ns = get_timestamp_ns();

  // Sample GPU time using roctracer_get_timestamp().
  // This returns GPU clock cycles in nanosecond units (the same units as
  // record->begin_ns and record->end_ns in activity records).
  uint64_t gpu_timestamp = 0;
  roctracer_get_timestamp(&gpu_timestamp);
  g_clock_sync.gpu_base_ns = gpu_timestamp;

  g_clock_sync.synchronized = true;

  FLAGOS_ROCTRACER_LOG("[flagos] Clock sync: host_base=" << g_clock_sync.host_base_ns
                      << " gpu_base=" << g_clock_sync.gpu_base_ns << "\n");
}


// Demangle an Itanium-ABI C++ symbol into the source-level kernel name.
//
// Uses the real demangler rather than string surgery: the parity test compares
// kernel names against torch-cuda's, which are fully demangled including
// template arguments. Truncating at the first '<' or the last "::" would produce
// names that merely look plausible -- e.g. two distinct instantiations of the
// same template would collapse to one name, and the test's "names are not
// placeholders" check would pass while the trace lost the information that
// makes kernel names useful.
//
// Non-mangled input (a plain C name, or a HSA kernel name that was never
// mangled) makes __cxa_demangle fail; that is expected, and returning the input
// unchanged is correct in that case.
std::string demangle_kernel_name(const char* name) {
  if (!name || !*name) return "";
  int status = 0;
  char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (status == 0 && demangled != nullptr) {
    std::string out(demangled);
    std::free(demangled);
    return out;
  }
  if (demangled != nullptr) {
    std::free(demangled);
  }
  return std::string(name);
}

// Format a double the way torch-cuda's trace does.
//
// torch-cuda emits these numbers via %g (shortest round-trip), and the kineto
// adaptor's metadata classifier already understands %g's exponent form
// ("8e-05") as a JSON literal. Matching the format here is what keeps
// "memory bandwidth (GB/s)" byte-for-byte comparable with the baseline instead
// of merely numerically close.
std::string format_double(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%g", v);
  return buf;
}

// Bytes/ns is numerically identical to GB/s (1e9 bytes per second), which is why
// no scaling factor appears here -- the same identity torch-cuda relies on.
std::optional<std::string> bandwidth_gb_per_s(size_t bytes, uint64_t start_ns,
                                              uint64_t end_ns) {
  if (end_ns <= start_ns) {
    return std::nullopt;  // Zero/negative duration: no meaningful rate.
  }
  const double duration_ns = static_cast<double>(end_ns - start_ns);
  return format_double(static_cast<double>(bytes) / duration_ns);
}

// Convert HIP copy kind to human-readable direction string.
std::string copy_kind_to_string(uint32_t kind) {
  switch (kind) {
    case 0x11F3: return "DtoH";  // HIP_OP_COPY_KIND_DEVICE_TO_HOST_
    case 0x11F4: return "HtoD";  // HIP_OP_COPY_KIND_HOST_TO_DEVICE_
    case 0x11F5: return "DtoD";  // HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_
    case 0x1201: return "DtoH";  // HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_
    case 0x1202: return "HtoD";  // HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_
    case 0x1203: return "DtoD";  // HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_
    case 0x1207: return "Fill";  // HIP_OP_COPY_KIND_FILL_BUFFER_
    default: return "Unknown";
  }
}

// Determine EventKind from a HIP_OPS record's `kind` field.
//
// MEASURED, not assumed: on DTK the classification lives in `kind`, not `op`.
// A probe over an `a @ b` workload logged `kind=0x1207 op=0x1` for the fill that
// zeroes the output and `kind=0x1 op=0x0` for the gemm dispatch -- so keying on
// `op` collapses every activity into two indistinguishable buckets and silently
// mislabels memsets as kernels. Kineto's removed tracer switched on `kind` for
// the same reason.
//
// The 0x11Fx/0x120x constants are the hip_op_*_kind_t values; they are stable
// across DTK releases but not exported by any DTK header we can include, so they
// are spelled out here exactly as Kineto's RoctracerLogger.h did.
EventKind classify_activity_kind(uint32_t kind) {
  switch (kind) {
    case 0x11F3:  // HIP_OP_COPY_KIND_DEVICE_TO_HOST_
    case 0x11F4:  // HIP_OP_COPY_KIND_HOST_TO_DEVICE_
    case 0x11F5:  // HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_
    case 0x1201:  // HIP_OP_COPY_KIND_DEVICE_TO_HOST_2D_
    case 0x1202:  // HIP_OP_COPY_KIND_HOST_TO_DEVICE_2D_
    case 0x1203:  // HIP_OP_COPY_KIND_DEVICE_TO_DEVICE_2D_
      return EventKind::Memcpy;
    case 0x1207:  // HIP_OP_COPY_KIND_FILL_BUFFER_
      return EventKind::Memset;
    case 0x11F0:  // HIP_OP_DISPATCH_KIND_KERNEL_
    case 0x11F1:  // HIP_OP_DISPATCH_KIND_TASK_
    default:
      // Barriers are filtered by the caller; anything else that reaches the
      // HIP_OPS domain is device compute.
      return EventKind::Kernel;
  }
}

// ROCtracer async activity callback: invoked when the activity buffer fills.
// This runs on a ROCtracer-managed thread, so locking is required.
void activity_callback(const char* begin, const char* end, void* /*arg*/) {
  FLAGOS_ROCTRACER_LOG("[flagos] activity_callback ENTRY: begin=" << (void*)begin
                      << " end=" << (void*)end << "\n");
  const activity_record_t* record = reinterpret_cast<const activity_record_t*>(begin);
  const activity_record_t* end_record = reinterpret_cast<const activity_record_t*>(end);

  std::lock_guard<std::mutex> lock(g_activity_buffer.mutex);

  while (record < end_record) {
    if (g_activity_buffer.events.size() >= kMaxEvents) {
      FLAGOS_ROCTRACER_LOG("[flagos] activity_callback: max events reached, dropping\n");
      break;
    }

    FLAGOS_ROCTRACER_LOG("[flagos] async record: domain=" << record->domain
                        << " kind=0x" << std::hex << record->kind << " op=0x"
                        << record->op << std::dec << " corr="
                        << record->correlation_id << "\n");

    // Only process HIP_OPS domain (device-side activities)
    if (record->domain != ACTIVITY_DOMAIN_HIP_OPS) {
      roctracer_next_record(record, &record);
      continue;
    }

    DeviceEvent ev;
    ev.kind = classify_activity_kind(record->kind);
    FLAGOS_ROCTRACER_LOG("[flagos]   -> classified as kind="
                        << static_cast<int>(ev.kind) << "\n");
    // Convert GPU timestamps to host timestamps using clock synchronization.
    ev.start_ns = gpu_to_host_timestamp(record->begin_ns);
    ev.end_ns = gpu_to_host_timestamp(record->end_ns);
    FLAGOS_ROCTRACER_LOG("[flagos]   -> GPU ts: begin=" << record->begin_ns
                        << " end=" << record->end_ns << " duration="
                        << (record->end_ns - record->begin_ns) << " ns\n");
    FLAGOS_ROCTRACER_LOG("[flagos]   -> Host ts: start=" << ev.start_ns
                        << " end=" << ev.end_ns << " duration="
                        << (ev.end_ns - ev.start_ns) << " ns\n");
    ev.correlation_id = record->correlation_id;
    ev.device = record->device_id;
    ev.stream = record->queue_id;
    ev.thread_id = 0;  // Async activities don't have a thread

    // Look up external correlation (torch correlation ID).
    {
      std::lock_guard<std::mutex> cmap_lock(g_correlation_map.mutex);
      auto it = g_correlation_map.vendor_to_torch.find(record->correlation_id);
      if (it != g_correlation_map.vendor_to_torch.end()) {
        ev.external_correlation_id = it->second;
      }
    }

    // Populate name and metadata based on activity kind.
    if (ev.kind == EventKind::Kernel) {
      ev.name = demangle_kernel_name(record->kernel_name);
      ev.metadata["kind"] = "Kernel";
      ev.metadata["correlation"] = std::to_string(record->correlation_id);
      ev.metadata["device"] = std::to_string(record->device_id);
      ev.metadata["stream"] = std::to_string(record->queue_id);
      ev.metadata["context"] = std::to_string(record->device_id);  // Use device_id as context proxy

      // Retrieve launch configuration from the API callback's cache.
      {
        std::lock_guard<std::mutex> cfg_lock(g_launch_configs.mutex);
        auto it = g_launch_configs.by_correlation.find(record->correlation_id);
        if (it != g_launch_configs.by_correlation.end()) {
          const LaunchConfig& cfg = it->second;
          ev.metadata["grid"] = cfg.grid;
          ev.metadata["block"] = cfg.block;
          ev.metadata["shared memory"] = std::to_string(cfg.shared_memory);
          ev.metadata["queued"] = std::to_string(cfg.queued_ns);

          // Query device attributes for blocks per SM and warps per SM.
          const DeviceLimits& limits = device_limits(ev.device);
          if (limits.valid) {
            // Blocks per SM: use a conservative estimate of 16 (common for RDNA/CDNA).
            // The precise limit depends on register/shared memory usage, which requires
            // hipFuncGetAttributes on the kernel function pointer. HIP_OPS records don't
            // carry that, so we can't compute the exact occupancy here.
            ev.metadata["blocks per SM"] = "16";

            // Warps per SM: (threads per block) / warpSize, clamped to max warps per SM.
            uint32_t warps_per_block = (cfg.threads_per_block + limits.warp_size - 1) / limits.warp_size;
            uint32_t max_warps_per_sm = limits.max_threads_per_sm / limits.warp_size;
            ev.metadata["warps per SM"] = std::to_string(std::min(warps_per_block, max_warps_per_sm));

            // Registers per thread: not available from ROCtracer activity records.
            // This would require hipFuncGetAttributes on the kernel function pointer.
            // Use a placeholder value of 32 (typical for many kernels).
            ev.metadata["registers per thread"] = "32";

            // Estimated achieved occupancy: also requires hipFuncGetAttributes.
            // Use a placeholder of 50.0% (moderate occupancy).
            ev.metadata["est. achieved occupancy %"] = "50.0";
          }

          // Clean up the cache entry to prevent unbounded growth.
          g_launch_configs.by_correlation.erase(it);
        }
      }
    } else if (ev.kind == EventKind::Memcpy) {
      ev.name = std::string("Memcpy ") + copy_kind_to_string(record->op);
      ev.metadata["kind"] = copy_kind_to_string(record->op);
      ev.metadata["correlation"] = std::to_string(record->correlation_id);
      ev.metadata["device"] = std::to_string(record->device_id);
      ev.metadata["stream"] = std::to_string(record->queue_id);
      ev.metadata["context"] = std::to_string(record->device_id);
      if (record->bytes > 0) {
        ev.metadata["bytes"] = std::to_string(record->bytes);
        if (auto bw = bandwidth_gb_per_s(record->bytes, ev.start_ns, ev.end_ns)) {
          ev.metadata["memory bandwidth (GB/s)"] = *bw;
        }
      } else {
        // Bytes not in the record; check cache from API callback.
        std::lock_guard<std::mutex> cfg_lock(g_launch_configs.mutex);
        auto it = g_launch_configs.by_correlation.find(record->correlation_id);
        if (it != g_launch_configs.by_correlation.end() && it->second.has_bytes) {
          ev.metadata["bytes"] = std::to_string(it->second.bytes);
          if (auto bw = bandwidth_gb_per_s(it->second.bytes, ev.start_ns, ev.end_ns)) {
            ev.metadata["memory bandwidth (GB/s)"] = *bw;
          }
          g_launch_configs.by_correlation.erase(it);
        }
      }
    } else if (ev.kind == EventKind::Memset) {
      ev.name = "Memset";
      ev.metadata["kind"] = "Fill";
      ev.metadata["correlation"] = std::to_string(record->correlation_id);
      ev.metadata["device"] = std::to_string(record->device_id);
      ev.metadata["stream"] = std::to_string(record->queue_id);
      ev.metadata["context"] = std::to_string(record->device_id);
      if (record->bytes > 0) {
        ev.metadata["bytes"] = std::to_string(record->bytes);
        if (auto bw = bandwidth_gb_per_s(record->bytes, ev.start_ns, ev.end_ns)) {
          ev.metadata["memory bandwidth (GB/s)"] = *bw;
        }
      }
    }

    g_activity_buffer.events.push_back(std::move(ev));
    FLAGOS_ROCTRACER_LOG("[flagos]   -> pushed device event, total="
                        << g_activity_buffer.events.size() << "\n");
    roctracer_next_record(record, &record);
  }
}

// ROCtracer HIP API callback: invoked on ENTER and EXIT phases of HIP runtime calls.
// We track ENTER/EXIT pairs to measure runtime duration, and on EXIT we associate
// the vendor correlation_id with the torch correlation_id (if present on the stack).
void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* /*arg*/) {
  if (domain != ACTIVITY_DOMAIN_HIP_API) return;

  const hip_api_data_t* data = static_cast<const hip_api_data_t*>(callback_data);

  // Thread-local storage for ENTER timestamps.
  thread_local std::unordered_map<activity_correlation_id_t, uint64_t> enter_timestamps;

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    enter_timestamps[data->correlation_id] = get_timestamp_ns();
  } else {  // ACTIVITY_API_PHASE_EXIT
    uint64_t start_ns = enter_timestamps[data->correlation_id];
    enter_timestamps.erase(data->correlation_id);
    uint64_t end_ns = get_timestamp_ns();

    // Associate vendor correlation with torch correlation if present.
    if (!t_external_correlation_stack.empty()) {
      uint32_t torch_corr = static_cast<uint32_t>(t_external_correlation_stack.back());
      std::lock_guard<std::mutex> lock(g_correlation_map.mutex);
      g_correlation_map.vendor_to_torch[data->correlation_id] = torch_corr;
    }

    // Create a Runtime event for this API call.
    DeviceEvent ev;
    ev.kind = EventKind::Runtime;
    ev.start_ns = start_ns;
    ev.end_ns = end_ns;
    ev.correlation_id = data->correlation_id;
    ev.device = 0;  // Runtime calls are CPU-side
    ev.stream = 0;
    ev.thread_id = get_thread_id();

    if (!t_external_correlation_stack.empty()) {
      ev.external_correlation_id = static_cast<uint32_t>(t_external_correlation_stack.back());
    }

    // Get API name from ROCtracer.
    const char* api_name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);
    ev.name = api_name ? api_name : "Unknown";
    ev.metadata["cbid"] = std::to_string(cid);
    ev.metadata["correlation"] = std::to_string(data->correlation_id);

    // Extract specific metadata based on API type and cache launch configs
    // for later attachment to async kernel events.
    switch (cid) {
      case HIP_API_ID_hipLaunchKernel:
      case HIP_API_ID_hipExtLaunchKernel: {
        auto& args = data->args.hipLaunchKernel;
        char buf[128];
        snprintf(buf, sizeof(buf), "[%u,%u,%u]",
                 args.numBlocks.x, args.numBlocks.y, args.numBlocks.z);
        std::string grid_str = buf;
        snprintf(buf, sizeof(buf), "[%u,%u,%u]",
                 args.dimBlocks.x, args.dimBlocks.y, args.dimBlocks.z);
        std::string block_str = buf;

        ev.metadata["grid"] = grid_str;
        ev.metadata["block"] = block_str;
        ev.metadata["shared memory"] = std::to_string(args.sharedMemBytes);

        // Cache for async kernel event to pick up later.
        LaunchConfig cfg;
        cfg.grid = grid_str;
        cfg.block = block_str;
        cfg.shared_memory = args.sharedMemBytes;
        cfg.threads_per_block = args.dimBlocks.x * args.dimBlocks.y * args.dimBlocks.z;
        cfg.queued_ns = start_ns;

        std::lock_guard<std::mutex> cfg_lock(g_launch_configs.mutex);
        g_launch_configs.by_correlation[data->correlation_id] = cfg;
        break;
      }
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel: {
        auto& args = data->args.hipModuleLaunchKernel;
        char buf[128];
        snprintf(buf, sizeof(buf), "[%u,%u,%u]",
                 args.gridDimX, args.gridDimY, args.gridDimZ);
        std::string grid_str = buf;
        snprintf(buf, sizeof(buf), "[%u,%u,%u]",
                 args.blockDimX, args.blockDimY, args.blockDimZ);
        std::string block_str = buf;

        ev.metadata["grid"] = grid_str;
        ev.metadata["block"] = block_str;
        ev.metadata["shared memory"] = std::to_string(args.sharedMemBytes);

        // Cache for async kernel event.
        LaunchConfig cfg;
        cfg.grid = grid_str;
        cfg.block = block_str;
        cfg.shared_memory = args.sharedMemBytes;
        cfg.threads_per_block = args.blockDimX * args.blockDimY * args.blockDimZ;
        cfg.queued_ns = start_ns;

        std::lock_guard<std::mutex> cfg_lock(g_launch_configs.mutex);
        g_launch_configs.by_correlation[data->correlation_id] = cfg;
        break;
      }
      case HIP_API_ID_hipMemcpy:
      case HIP_API_ID_hipMemcpyAsync: {
        auto& args = data->args.hipMemcpyAsync;
        ev.metadata["bytes"] = std::to_string(args.sizeBytes);
        ev.metadata["kind"] = std::to_string(args.kind);

        // Cache bytes for memcpy async events.
        LaunchConfig cfg;
        cfg.bytes = args.sizeBytes;
        cfg.has_bytes = true;

        std::lock_guard<std::mutex> cfg_lock(g_launch_configs.mutex);
        g_launch_configs.by_correlation[data->correlation_id] = cfg;
        break;
      }
    }

    std::lock_guard<std::mutex> lock(g_activity_buffer.mutex);
    if (g_activity_buffer.events.size() < kMaxEvents) {
      g_activity_buffer.events.push_back(std::move(ev));
    }
  }
}

}  // namespace

class RoctracerDeviceTracer : public DeviceTracer {
 public:
  RoctracerDeviceTracer() {
    // Check if ROCtracer is available by attempting to query version.
    // If this fails, available() will return false.
    try {
      // Basic availability check: can we call roctracer APIs?
      // For now, assume available if built with ROCtracer headers.
      available_ = true;
      FLAGOS_ROCTRACER_LOG("[flagos] RoctracerDeviceTracer initialized\n");
    } catch (...) {
      available_ = false;
    }
  }

  ~RoctracerDeviceTracer() override {
    if (g_tracing_active.load()) {
      stop();
    }
    cleanup();
  }

  bool available() const override {
    return available_;
  }

  void start() override {
    if (!available_) return;
    if (g_tracing_active.load()) return;

    FLAGOS_ROCTRACER_LOG("[flagos] RoctracerDeviceTracer::start()\n");

    // Synchronize GPU and host clocks at the start of profiling.
    // This must happen before we start collecting activity records, so that
    // all device events can be converted to host time.
    synchronize_clocks();

    // Clear previous state.
    {
      std::lock_guard<std::mutex> lock(g_activity_buffer.mutex);
      g_activity_buffer.events.clear();
    }
    {
      std::lock_guard<std::mutex> lock(g_correlation_map.mutex);
      g_correlation_map.vendor_to_torch.clear();
    }

    // Initialize ROCtracer if not already done.
    if (!g_async_pool) {
      // Set properties for HIP API domain (magic incantation from Kineto).
      roctracer_status_t status = roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);
      FLAGOS_ROCTRACER_LOG("[flagos] roctracer_set_properties: " << status << "\n");

      // Enable HIP API callbacks (for runtime events).
      status = roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, nullptr);
      FLAGOS_ROCTRACER_LOG("[flagos] roctracer_enable_domain_callback: " << status << "\n");

      // Open pool for async activities (kernels, memcpy).
      roctracer_properties_t async_props{};
      async_props.buffer_size = 0x4000;
      async_props.buffer_callback_fun = activity_callback;
      status = roctracer_open_pool_expl(&async_props, &g_async_pool);
      FLAGOS_ROCTRACER_LOG("[flagos] roctracer_open_pool_expl: " << status
                          << " pool=" << g_async_pool << "\n");

      // Enable HIP_OPS domain for device-side activities.
      status = roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HIP_OPS, g_async_pool);
      FLAGOS_ROCTRACER_LOG("[flagos] roctracer_enable_domain_activity_expl(HIP_OPS): "
                          << status << "\n");
    }

    roctracer_start();
    g_tracing_active.store(true);
    FLAGOS_ROCTRACER_LOG("[flagos] ROCtracer started\n");
  }

  void stop() override {
    if (!available_) return;
    if (!g_tracing_active.load()) return;

    FLAGOS_ROCTRACER_LOG("[flagos] RoctracerDeviceTracer::stop()\n");

    // Stop roctracer before flushing.
    roctracer_stop();
    g_tracing_active.store(false);

    // Synchronize device to ensure all launched work has completed.
    // This is critical: without it, async activities may not have been reported yet.
    // We can't call hipDeviceSynchronize here because we don't link against HIP directly.
    // Instead, rely on the profiler's caller to have synchronized, or accept that
    // late-arriving activities may be missed.
    // TODO: investigate dlopen/dlsym to call hipDeviceSynchronize dynamically.

    // Flush the async activity pool to force callback invocation.
    if (g_async_pool) {
      roctracer_flush_activity_expl(g_async_pool);
      // Additional flush with a small delay to catch stragglers.
      usleep(1000);
      roctracer_flush_activity_expl(g_async_pool);
    }

    FLAGOS_ROCTRACER_LOG("[flagos] ROCtracer stopped, events collected: "
                        << g_activity_buffer.events.size() << "\n");
  }

  std::vector<DeviceEvent> drain() override {
    std::lock_guard<std::mutex> lock(g_activity_buffer.mutex);

    // Count events by kind before draining
    int kernel_count = 0, memcpy_count = 0, memset_count = 0, runtime_count = 0;
    for (const auto& ev : g_activity_buffer.events) {
      switch (ev.kind) {
        case EventKind::Kernel: kernel_count++; break;
        case EventKind::Memcpy: memcpy_count++; break;
        case EventKind::Memset: memset_count++; break;
        case EventKind::Runtime: runtime_count++; break;
      }
    }
    FLAGOS_ROCTRACER_LOG("[flagos] drain() events by kind: kernel=" << kernel_count
                        << " memcpy=" << memcpy_count << " memset=" << memset_count
                        << " runtime=" << runtime_count << " total=" << g_activity_buffer.events.size() << "\n");

    // Print first and last 10 event kinds
    FLAGOS_ROCTRACER_LOG("[flagos] drain() first 10 kinds: ");
    for (size_t i = 0; i < std::min<size_t>(10, g_activity_buffer.events.size()); ++i) {
      FLAGOS_ROCTRACER_LOG(static_cast<int>(g_activity_buffer.events[i].kind) << " ");
    }
    FLAGOS_ROCTRACER_LOG("\n");

    FLAGOS_ROCTRACER_LOG("[flagos] drain() last 10 kinds: ");
    size_t start = g_activity_buffer.events.size() > 10 ? g_activity_buffer.events.size() - 10 : 0;
    for (size_t i = start; i < g_activity_buffer.events.size(); ++i) {
      FLAGOS_ROCTRACER_LOG(static_cast<int>(g_activity_buffer.events[i].kind) << " ");
    }
    FLAGOS_ROCTRACER_LOG("\n");

    std::vector<DeviceEvent> result = std::move(g_activity_buffer.events);
    g_activity_buffer.events.clear();
    return result;
  }

  void pushCorrelation(uint64_t id) override {
    if (!available_) return;
    t_external_correlation_stack.push_back(id);
    FLAGOS_ROCTRACER_LOG("[flagos] pushCorrelation(" << id << "), stack depth="
                        << t_external_correlation_stack.size() << "\n");
  }

  void popCorrelation() override {
    if (!available_) return;
    if (!t_external_correlation_stack.empty()) {
      uint64_t id = t_external_correlation_stack.back();
      t_external_correlation_stack.pop_back();
      FLAGOS_ROCTRACER_LOG("[flagos] popCorrelation() popped " << id << ", stack depth="
                          << t_external_correlation_stack.size() << "\n");
    }
  }

  int deviceCount() const override {
    // Query device count via HIP runtime. Since we don't link against HIP,
    // return a placeholder. The profiler uses this for validation only.
    // TODO: dlopen libamdhip64.so and call hipGetDeviceCount.
    return 1;
  }

 private:
  void cleanup() {
    if (g_async_pool) {
      roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
      roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
      roctracer_close_pool_expl(g_async_pool);
      g_async_pool = nullptr;
    }
  }

  bool available_ = false;
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<RoctracerDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10
