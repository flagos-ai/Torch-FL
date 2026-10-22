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
// Standalone rocprofiler-sdk tool for the TsingMicro TX8 NPU.
//
// Built as libtorch_fl_profiler.so and loaded by rocprofiler itself (through
// ROCP_TOOL_LIBRARIES). It links nothing but rocprofiler-sdk on purpose:
// libtorch_fl.so depends on libflagos.so -> libhpgr.so and the tsm profiler
// libraries, and dlopening that chain while rocprofiler was still initializing
// would leave the tool without a client id.
//
// The Kineto side (csrc/profiler/tsingmicro_device_tracer.cc, in
// libtorch_fl.so) drives this through the C API in tsingmicro_profiler_api.h.
//
// Logging here uses fprintf rather than torch_fl's logger on purpose. The
// logger is defined in libtorch_fl.so, which this library must not depend on --
// it is not in the DT_NEEDED list, so the symbol would have to resolve from the
// global scope at dlopen time, and Python loads extension modules with
// RTLD_LOCAL. If it fails to resolve, rocprofiler's dlopen fails, the tool
// never registers, and profiling is silently dead.

#include "profiler/tsingmicro_profiler_api.h"

#include <flagos_env.h>

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using kernel_record_t = rocprofiler_buffer_tracing_tsm_kernel_dispatch_record_t;

// Same switch as the Kineto side: ordinary profiling runs stay quiet, and
// FLAGOS_TRACE=1 turns on the record dumps used when the timeline is wrong.
bool debug_enabled() {
  static const bool on = flagos_env::EnvFlag("FLAGOS_TRACE");
  return on;
}

struct ToolState {
  rocprofiler_context_id_t context{};
  rocprofiler_buffer_id_t buffer{};
  bool context_ready{false};

  std::mutex mutex;
  std::vector<kernel_record_t> records;

  // Timestamps of the launch API call, keyed by internal correlation id -- the
  // runtime tags the launch call and the kernel it dispatches with the same
  // one. Kineto draws the CPU -> kernel arrow from a flow whose start sits on
  // a real event; CUDA uses the cudaLaunchKernel record for that, so the
  // equivalent here is the txLaunchKernel window.
  std::mutex launch_mutex;
  std::unordered_map<uint64_t, uint64_t> launch_begin;
  std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> launch_window;

  // rocprofiler timestamps -> Unix epoch ns, sampled when the window opens.
  std::atomic<int64_t> clock_offset_ns{0};

  std::atomic<int64_t> push_count{0};
  std::atomic<int64_t> correlation_cb_count{0};
  std::atomic<int64_t> api_cb_count{0};
};

ToolState& state() {
  static ToolState s;
  return s;
}

// ---------------------------------------------------------------------------
// rocprofiler callbacks
// ---------------------------------------------------------------------------
void tool_buffer_callback(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                          rocprofiler_record_header_t** headers,
                          size_t num_headers, void* user_data,
                          uint64_t /*drop_count*/) {
  auto* s = static_cast<ToolState*>(user_data);
  std::lock_guard<std::mutex> lock(s->mutex);
  for (size_t i = 0; i < num_headers; ++i) {
    auto* header = headers[i];
    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) {
      continue;
    }
    if (header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
      continue;
    }
    s->records.push_back(*static_cast<kernel_record_t*>(header->payload));
  }
}

// rocprofiler asks the tool for an external correlation id when it stamps a
// record. Declining (returning non-zero) is deliberate: per the SDK contract
// that makes rocprofiler fall back to the thread-local value pushed by
// torch_fl_profiler_push_correlation, which is the op id Kineto handed us. That
// value is what the runtime copies into correlation_id.external on both the
// TSM runtime API callback records and the kernel dispatch records, so the
// kernel -> op join needs no lookup table here.
int tool_external_correlation_cb(
    rocprofiler_thread_id_t /*thread_id*/,
    rocprofiler_context_id_t /*context_id*/,
    rocprofiler_external_correlation_id_request_kind_t /*kind*/,
    rocprofiler_tracing_operation_t /*operation*/,
    uint64_t /*internal_corr_id_value*/,
    rocprofiler_user_data_t* /*external_corr_id_value*/,
    void* /*data*/) {
  state().correlation_cb_count.fetch_add(1);
  return 1;
}

// The TSM runtime API callback is only used for the launch window now: the
// callback record carries no timestamps, so bracket the call ourselves. The
// runtime tags the launch API call and the kernel it dispatches with the same
// internal id, so this window can be looked up from the kernel record. The
// kernel -> op join itself needs nothing from this path: the external id is
// already on the kernel dispatch record.
//
// This only fires when the runtime is profiling, i.e. when TSM_PROFILER_EN is
// set: without it hpgr never calls rocprofiler_set_api_table and none of the
// tx* calls are intercepted.
void tool_api_tracing_cb(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /*user_data*/,
                         void* /*callback_data*/) {
  auto& s = state();
  s.api_cb_count.fetch_add(1);
  const uint64_t internal = record.correlation_id.internal;
  if (internal == 0) {
    return;
  }
  // The callback record carries no timestamps, so bracket the call ourselves.
  rocprofiler_timestamp_t ts = 0;
  rocprofiler_get_timestamp(&ts);
  std::lock_guard<std::mutex> lock(s.launch_mutex);
  if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
    s.launch_begin[internal] = ts;
    return;
  }
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT) {
    return;
  }
  const auto begin = s.launch_begin.find(internal);
  if (begin != s.launch_begin.end()) {
    s.launch_window[internal] = {begin->second, ts};
    s.launch_begin.erase(begin);
  }
}

// ---------------------------------------------------------------------------
// rocprofiler tool
// ---------------------------------------------------------------------------
int tool_init(rocprofiler_client_finalize_t /*finalize*/, void* user_data) {
  auto* s = static_cast<ToolState*>(user_data);
  // rocprofiler can reach us twice -- once through the ROCP_TOOL_LIBRARIES
  // scan and once through rocprofiler_force_configure(). A second context
  // would capture every kernel twice, so only the first init counts.
  if (s->context_ready) {
    return 0;
  }
  if (rocprofiler_create_context(&s->context) != ROCPROFILER_STATUS_SUCCESS) {
    return -1;
  }
  if (rocprofiler_create_buffer(s->context, 8192, 7936,
                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                tool_buffer_callback, s,
                                &s->buffer) != ROCPROFILER_STATUS_SUCCESS) {
    return -1;
  }
  if (rocprofiler_configure_buffer_tracing_service(
          s->context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
          s->buffer) != ROCPROFILER_STATUS_SUCCESS) {
    return -1;
  }
  // Buffer records are delivered on a rocprofiler-owned callback thread; a
  // buffer with no assigned thread never delivers, so this is not optional.
  rocprofiler_callback_thread_t callback_thread{};
  if (rocprofiler_create_callback_thread(&callback_thread) !=
      ROCPROFILER_STATUS_SUCCESS) {
    return -1;
  }
  if (rocprofiler_assign_callback_thread(s->buffer, callback_thread) !=
      ROCPROFILER_STATUS_SUCCESS) {
    return -1;
  }
  // A null kinds array means every tracing operation asks us for an external
  // correlation id.
  if (rocprofiler_configure_external_correlation_id_request_service(
          s->context, nullptr, 0, tool_external_correlation_cb,
          nullptr) != ROCPROFILER_STATUS_SUCCESS) {
    return -1;
  }
  // Diagnostic only: never fail the tool over it.
  if (rocprofiler_configure_callback_tracing_service(
          s->context, ROCPROFILER_CALLBACK_TRACING_TSM_RUNTIME_API, nullptr, 0,
          tool_api_tracing_cb, nullptr) != ROCPROFILER_STATUS_SUCCESS) {
    if (debug_enabled()) {
      fprintf(stderr,
              "[torch_fl_profiler] TSM runtime API callback tracing "
              "unavailable\n");
    }
  }
  int valid = 0;
  rocprofiler_context_is_valid(s->context, &valid);
  s->context_ready = (valid != 0);
  if (debug_enabled()) {
    fprintf(stderr, "[torch_fl_profiler] tool_init: context_ready=%d\n",
            static_cast<int>(s->context_ready));
  }
  return s->context_ready ? 0 : -1;
}

void tool_fini(void* /*user_data*/) {}

rocprofiler_tool_configure_result_t* tool_configure(
    uint32_t /*version*/, const char* /*runtime_version*/,
    uint32_t /*priority*/, rocprofiler_client_id_t* id) {
  id->name = "torch_fl_profiler";
  static auto config = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini,
      static_cast<void*>(&state())};
  return &config;
}

struct ToolRegistration {
  ToolRegistration() {
    int status = 0;
    if (rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS &&
        status == 0) {
      rocprofiler_force_configure(&tool_configure);
    }
  }
};
ToolRegistration tool_registration;

}  // namespace

// rocprofiler discovers tools by scanning ROCP_TOOL_LIBRARIES for this symbol,
// or via rocprofiler_force_configure(). Only the scan path works when this
// library is dlopen'd during rocprofiler's own initialization: at that moment
// rocprofiler is "currently being configured", so force_configure returns
// CONFIGURATION_LOCKED and is a no-op. Export the symbol with C linkage at
// global scope so the scan finds it.
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t version, const char* runtime_version, uint32_t priority,
    rocprofiler_client_id_t* id) {
  return tool_configure(version, runtime_version, priority, id);
}

// ---------------------------------------------------------------------------
// C API used by the Kineto side
// ---------------------------------------------------------------------------
extern "C" int torch_fl_profiler_start(void) {
  auto& s = state();
  rocprofiler_timestamp_t ts = 0;
  if (rocprofiler_get_timestamp(&ts) == ROCPROFILER_STATUS_SUCCESS) {
    const auto epoch_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    s.clock_offset_ns.store(epoch_ns - static_cast<int64_t>(ts));
    if (debug_enabled()) {
      fprintf(stderr,
              "[torch_fl_profiler] clock offset: rocprofiler_ts=%lld "
              "epoch_ns=%lld offset=%lld\n",
              static_cast<long long>(ts), static_cast<long long>(epoch_ns),
              static_cast<long long>(epoch_ns - static_cast<int64_t>(ts)));
    }
  }
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.records.clear();
  }
  {
    std::lock_guard<std::mutex> lock(s.launch_mutex);
    s.launch_begin.clear();
    s.launch_window.clear();
  }
  s.push_count.store(0);
  s.correlation_cb_count.store(0);
  s.api_cb_count.store(0);
  if (s.context_ready) {
    rocprofiler_start_context(s.context);
  }
  return 0;
}

extern "C" int torch_fl_profiler_stop(void) {
  auto& s = state();
  if (s.context_ready) {
    rocprofiler_stop_context(s.context);
    rocprofiler_flush_buffer(s.buffer);
  }
  return 0;
}

extern "C" int64_t torch_fl_profiler_clock_offset_ns(void) {
  return state().clock_offset_ns.load();
}

extern "C" void torch_fl_profiler_push_correlation(uint64_t id) {
  auto& s = state();
  s.push_count.fetch_add(1);
  if (!s.context_ready) {
    return;
  }
  rocprofiler_thread_id_t tid = 0;
  rocprofiler_get_thread_id(&tid);
  rocprofiler_user_data_t user_data{};
  user_data.value = id;
  rocprofiler_push_external_correlation_id(s.context, tid, user_data);
}

extern "C" void torch_fl_profiler_pop_correlation(void) {
  auto& s = state();
  if (!s.context_ready) {
    return;
  }
  rocprofiler_thread_id_t tid = 0;
  rocprofiler_get_thread_id(&tid);
  rocprofiler_user_data_t user_data{};
  rocprofiler_pop_external_correlation_id(s.context, tid, &user_data);
}

extern "C" int torch_fl_profiler_take_records(
    torch_fl_profiler_records_t* out) {
  if (out == nullptr) {
    return -1;
  }
  out->data = nullptr;
  out->count = 0;
  out->stride = sizeof(kernel_record_t);

  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.records.empty()) {
    return 0;
  }
  const size_t bytes = s.records.size() * sizeof(kernel_record_t);
  out->data = std::malloc(bytes);
  if (out->data == nullptr) {
    return -1;
  }
  std::memcpy(out->data, s.records.data(), bytes);
  out->count = s.records.size();
  s.records.clear();
  return 0;
}

extern "C" void torch_fl_profiler_release_records(
    torch_fl_profiler_records_t* records) {
  if (records == nullptr) {
    return;
  }
  std::free(records->data);
  records->data = nullptr;
  records->count = 0;
  records->stride = 0;
}

extern "C" int torch_fl_profiler_lookup_launch(uint64_t internal_correlation_id,
                                               uint64_t* start_ns,
                                               uint64_t* end_ns) {
  if (start_ns == nullptr || end_ns == nullptr) {
    return -1;
  }
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.launch_mutex);
  const auto it = s.launch_window.find(internal_correlation_id);
  if (it == s.launch_window.end()) {
    return -1;
  }
  *start_ns = it->second.first;
  *end_ns = it->second.second;
  return 0;
}

extern "C" int64_t torch_fl_profiler_push_count(void) {
  return state().push_count.load();
}

extern "C" int64_t torch_fl_profiler_correlation_cb_count(void) {
  return state().correlation_cb_count.load();
}

extern "C" int64_t torch_fl_profiler_api_cb_count(void) {
  return state().api_cb_count.load();
}
