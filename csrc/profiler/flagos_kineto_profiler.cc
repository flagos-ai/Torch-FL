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
// Kineto adaptor layer. Everything vendor-specific lives behind DeviceTracer
// (csrc/profiler/device_tracer.h); this file only translates DeviceEvent into
// libkineto::GenericTraceActivity and owns the correlation/flow wiring.

// MUST precede every other include: kineto's GenericTraceActivity.h pulls in
// <fmt/format.h>, and once that is parsed without FMT_HEADER_ONLY the inline
// definitions are gone for the rest of the TU. GenericTraceActivity::addMetadata
// is a template calling fmt::format, and libtorch_cpu.so compiles fmt with
// hidden visibility (it exports no fmt::vformat), so a plugin instantiating
// addMetadata otherwise links against an undefined symbol -- observed as
// `undefined symbol: _ZN3fmt3v127vformat...` at import time. Header-only fmt
// resolves it locally; no other TU in this library includes fmt, so there is no
// ODR conflict.
#define FMT_HEADER_ONLY 1
#include <fmt/format.h>

#include "flagos_kineto_profiler.h"
#include "device_tracer.h"
#include <flagos_env.h>

#include <kineto/ActivityType.h>
#include <kineto/ILoggerObserver.h>
#include <kineto/libkineto.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Diagnostic logging is gated behind FLAGOS_TRACE=1 so that normal profiling
// runs stay quiet. One switch covers every tracer: the build compiles exactly
// one device tracer (csrc/CMakeLists.txt), so the name is unambiguous.
namespace {
inline bool flagos_kineto_debug() {
  static const bool on = flagos_env::EnvFlag("FLAGOS_TRACE");
  return on;
}
}  // namespace
#define FLAGOS_KINETO_LOG(expr) \
  do { if (flagos_kineto_debug()) { std::cerr << expr; } } while (0)

namespace c10 {
namespace flagos {

namespace detail {

// Instrumentation counters for correlation push/pop verification
std::atomic<uint64_t> g_correlation_push_count{0};
std::atomic<uint64_t> g_correlation_pop_count{0};

namespace {

// True when `v` may be written into the trace JSON *unquoted*.
//
// Why this exists: kineto's GenericTraceActivity::addMetadata stores every value
// with quoted=false, and metadataJson() then emits it as `"key": <raw>`. That is
// fine only while every value happens to be a JSON literal. The moment one bare
// identifier goes through -- a kernel name, a memory-kind label, an "N/A" -- the
// emitted object is `"key": N/A`, which is not JSON. And because kineto
// concatenates all activities into ONE document, that single event invalidates
// the ENTIRE trace file: json.load() fails on the whole thing, not just its own
// event, so the failure is wildly disproportionate to the mistake that caused it.
//
// DeviceEvent::metadata is std::map<string,string>, so by the time we get here
// even genuinely numeric values are text -- we cannot switch on a static type
// and must classify by textual form instead.
//
// Deliberately conservative: anything not provably a JSON literal is quoted. A
// spuriously quoted number renders as a string in a trace viewer (cosmetic); an
// unquoted non-literal breaks the file (fatal). This is NOT a JSON validator --
// it only needs to be sound in the "may I skip quoting?" direction.
bool metadataValueIsJsonLiteral(const std::string& v) {
  if (v.empty()) {
    return false;
  }
  if (v == "true" || v == "false" || v == "null") {
    return true;
  }
  // Bracketed lists: our "grid"/"block" values are built as "[8,16,5]". Trusting
  // the brackets is enough here because we are the only producer of them.
  if (v.front() == '[' && v.back() == ']') {
    return true;
  }
  // Optionally-signed number: -?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?
  //
  // The exponent branch is NOT hypothetical: "memory bandwidth (GB/s)" is
  // formatted with shortest-roundtrip %g to match torch-cuda byte-for-byte, and
  // %g switches to exponent form for small magnitudes (a 4-byte copy over a long
  // interval yields "8e-05"). Exponent notation is valid JSON, so rejecting it
  // here would quote a number that torch-cuda emits bare -- a silent parity
  // break rather than a crash, which is the harder kind to notice.
  size_t i = (v[0] == '-' || v[0] == '+') ? 1 : 0;
  size_t digits_before = 0;
  while (i < v.size() && v[i] >= '0' && v[i] <= '9') {
    ++i;
    ++digits_before;
  }
  if (digits_before == 0) {
    return false;
  }
  if (i < v.size() && v[i] == '.') {
    ++i;
    size_t digits_after = 0;
    while (i < v.size() && v[i] >= '0' && v[i] <= '9') {
      ++i;
      ++digits_after;
    }
    if (digits_after == 0) {
      return false;  // "1." is not valid JSON
    }
  }
  if (i < v.size() && (v[i] == 'e' || v[i] == 'E')) {
    ++i;
    if (i < v.size() && (v[i] == '+' || v[i] == '-')) {
      ++i;
    }
    size_t exp_digits = 0;
    while (i < v.size() && v[i] >= '0' && v[i] <= '9') {
      ++i;
      ++exp_digits;
    }
    if (exp_digits == 0) {
      return false;  // "1e" / "1e+" is not a number
    }
  }
  return i == v.size();
}

// Minimal escaping for the quoted path. addMetadataQuoted wraps the value in
// quotes but does NOT escape anything inside it, so an embedded `"` or `\` would
// terminate the string early and -- again -- corrupt the whole document. Only
// these two characters can do that; control characters would technically also be
// illegal JSON but no value we produce contains them, and a full escaper is more
// machinery than this needs.
std::string escapeForJsonString(const std::string& v) {
  std::string out;
  out.reserve(v.size());
  for (char c : v) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

libkineto::ActivityType toKinetoActivityType(profiler::EventKind kind) {
  switch (kind) {
    case profiler::EventKind::Kernel:
      return libkineto::ActivityType::CONCURRENT_KERNEL;
    case profiler::EventKind::Memcpy:
      return libkineto::ActivityType::GPU_MEMCPY;
    case profiler::EventKind::Memset:
      return libkineto::ActivityType::GPU_MEMSET;
    case profiler::EventKind::Runtime:
      return libkineto::ActivityType::PRIVATEUSE1_RUNTIME;
  }
  return libkineto::ActivityType::CONCURRENT_KERNEL;
}

// Builds the kineto activity for one device event. Correlation/flow wiring is
// applied by the caller, which is the only part that needs the kineto callback.
libkineto::GenericTraceActivity toActivity(const profiler::DeviceEvent& ev) {
  libkineto::GenericTraceActivity activity;
  activity.activityType = toKinetoActivityType(ev.kind);
  activity.activityName = ev.name;
  activity.startTime = static_cast<int64_t>(ev.start_ns);
  activity.endTime = static_cast<int64_t>(ev.end_ns);
  activity.id = static_cast<int32_t>(ev.correlation_id);

  if (ev.kind == profiler::EventKind::Runtime) {
    // Runtime calls happen on the CPU: device 0, and the "resource" lane is the
    // issuing thread rather than a device stream.
    activity.device = 0;
    activity.resource = static_cast<int32_t>(ev.thread_id);
    activity.threadId = static_cast<int32_t>(ev.thread_id);
  } else {
    activity.device = static_cast<int32_t>(ev.device);
    activity.resource = static_cast<int32_t>(ev.stream);
  }

  for (const auto& [key, value] : ev.metadata) {
    // addMetadata is kineto's UNQUOTED path (see metadataValueIsJsonLiteral).
    // Route anything that is not a bare JSON literal through the quoted one.
    if (metadataValueIsJsonLiteral(value)) {
      activity.addMetadata(key, value);
    } else {
      activity.addMetadataQuoted(key, escapeForJsonString(value));
    }
  }
  return activity;
}

}  // namespace
}  // namespace detail

// ===== FlagosKinetoProfilerSession Implementation =====

FlagosKinetoProfilerSession::FlagosKinetoProfilerSession()
    : tracer_(profiler::MakeDeviceTracer()) {}

void FlagosKinetoProfilerSession::start() {
  events_.clear();
  if (tracer_ && tracer_->available()) {
    tracer_->start();
  }
  status_ = libkineto::TraceStatus::RECORDING;
}

void FlagosKinetoProfilerSession::stop() {
  if (tracer_ && tracer_->available()) {
    tracer_->stop();
    // Drain here rather than in processTrace: stop() is where the tracer has
    // just flushed, and kineto may call either processTrace overload (or none)
    // afterwards.
    events_ = tracer_->drain();
  }
  status_ = libkineto::TraceStatus::PROCESSING;
  FLAGOS_KINETO_LOG("[flagos] session stop: drained " << events_.size()
                   << " device events\n");
}

void FlagosKinetoProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  FLAGOS_KINETO_LOG("[flagos] processTrace(1-arg) called with "
                   << events_.size() << " events\n");
  for (const auto& ev : events_) {
    detail::toActivity(ev).log(logger);
  }
}

void FlagosKinetoProfilerSession::processTrace(
    libkineto::ActivityLogger& logger,
    libkineto::getLinkedActivityCallback getLinkedActivity,
    int64_t startTime,
    int64_t endTime) {
  FLAGOS_KINETO_LOG("[flagos] processTrace(4-arg) called with "
                   << events_.size() << " events, window=[" << startTime
                   << "," << endTime << "]\n");

  // Count events by kind at the start of processTrace
  int kernel_count = 0, memcpy_count = 0, memset_count = 0, runtime_count = 0;
  for (const auto& ev : events_) {
    switch (ev.kind) {
      case profiler::EventKind::Kernel: kernel_count++; break;
      case profiler::EventKind::Memcpy: memcpy_count++; break;
      case profiler::EventKind::Memset: memset_count++; break;
      case profiler::EventKind::Runtime: runtime_count++; break;
    }
  }
  FLAGOS_KINETO_LOG("[flagos] processTrace events by kind: kernel=" << kernel_count
                   << " memcpy=" << memcpy_count << " memset=" << memset_count
                   << " runtime=" << runtime_count << "\n");

  // Print first and last event timestamps to diagnose window filtering
  if (!events_.empty()) {
    auto& first = events_[0];
    auto& last = events_[events_.size() - 1];
    FLAGOS_KINETO_LOG("[flagos] First event: kind=" << static_cast<int>(first.kind)
                     << " start=" << first.start_ns << " end=" << first.end_ns << "\n");
    FLAGOS_KINETO_LOG("[flagos] Last event: kind=" << static_cast<int>(last.kind)
                     << " start=" << last.start_ns << " end=" << last.end_ns << "\n");
  }

  // Capture-window predicate.
  //
  // UNITS (verified, not assumed): kineto's startTime/endTime are ns since the
  // UNIX epoch, and the tracer's event timestamps are the same domain --
  // cuptiGetTimestamp() was measured 2.7us away from CLOCK_REALTIME, i.e. both
  // are realtime-epoch ns, so these compare directly with no scaling.
  // (CLOCK_MONOTONIC/BOOTTIME are ~1.78e18 ns away, so a boot-relative reading
  // of either side would have been off by ~56 years and filtered everything.)
  auto in_window = [startTime, endTime](const profiler::DeviceEvent& ev) {
    return static_cast<int64_t>(ev.end_ns) >= startTime &&
           static_cast<int64_t>(ev.start_ns) <= endTime;
  };

  // Correlation ids that actually have a device-side activity *that we will
  // emit*. A flow needs both ends: emitting an 's' half for a runtime call that
  // launched nothing (e.g. cudaStreamSynchronize, cudaMalloc) -- or whose kernel
  // fell outside the window -- leaves a dangling arrow, which is not what
  // torch+cuda produces.
  std::set<uint32_t> device_correlations;
  std::set<uint32_t> runtime_correlations;
  int device_in_window = 0;
  int device_total = 0;
  for (const auto& ev : events_) {
    if (ev.kind != profiler::EventKind::Runtime) {
      ++device_total;
      if (device_total <= 3) {
        FLAGOS_KINETO_LOG("[flagos] Device event #" << device_total << ": kind=" << static_cast<int>(ev.kind)
                         << " start=" << ev.start_ns << " end=" << ev.end_ns
                         << " window=[" << startTime << "," << endTime << "]\n");
      }
      if (in_window(ev)) {
        device_correlations.insert(ev.correlation_id);
        ++device_in_window;
      }
    } else if (in_window(ev) && ev.correlation_id != 0) {
      runtime_correlations.insert(ev.correlation_id);
    }
  }
  FLAGOS_KINETO_LOG("[flagos] Found " << device_in_window << "/" << device_total << " device events in window\n");

  // A renderable flow requires one runtime and one device activity with the same
  // vendor correlation id. Device records without a runtime counterpart occur
  // for work submitted by an internal vendor thread and must not emit a lone
  // finish half.
  std::set<uint32_t> paired_correlations;
  std::set_intersection(
      device_correlations.begin(), device_correlations.end(),
      runtime_correlations.begin(), runtime_correlations.end(),
      std::inserter(paired_correlations, paired_correlations.end()));

  // Whether kineto actually handed us a resolver. An EMPTY callback is a total
  // loss of linking: every activity is then emitted with linked == nullptr,
  // which is precisely the ablation state Task 1 measured returning aten::mm's
  // self_device_time_total to 0. That degradation is otherwise completely
  // silent, so warn once, unconditionally -- this is the breadcrumb a
  // "device time is mysteriously 0" report needs, and it must not be gated
  // behind FLAGOS_TRACE for exactly that reason. Same rationale and
  // shape as the tracer's reportLayoutMismatch: rare by construction, and
  // catastrophic when it happens.
  const bool have_resolver = static_cast<bool>(getLinkedActivity);
  if (!have_resolver) {
    static std::once_flag warn_once;
    std::call_once(warn_once, [] {
      std::cerr
          << "[flagos] kineto called processTrace with an EMPTY "
             "getLinkedActivity callback.\n"
          << "[flagos]   Device activities cannot be linked to the CPU ops that"
             " issued them, so per-operator\n"
          << "[flagos]   device time (self_device_time_total) will be 0 and"
             " ac2g flow arrows will be absent.\n"
          << "[flagos]   The device timeline itself is unaffected. This"
             " indicates a libkineto that does not supply\n"
          << "[flagos]   the resolver to plugin profiler sessions.\n";
    });
  }

  size_t linked_count = 0;
  // Debug: print first 10 event kinds before processing
  FLAGOS_KINETO_LOG("[flagos] First 10 events_ kinds: ");
  for (size_t i = 0; i < std::min<size_t>(10, events_.size()); ++i) {
    FLAGOS_KINETO_LOG(static_cast<int>(events_[i].kind) << " ");
  }
  FLAGOS_KINETO_LOG("\n");

  size_t link_candidates = 0;
  size_t dropped_out_of_window = 0;
  int dropped_kernel = 0, dropped_memcpy = 0, dropped_memset = 0, dropped_runtime = 0;
  for (const auto& ev : events_) {
    // Drop activities outside the capture window. The tracer keeps recording
    // into its buffers across the whole process lifetime (device activity kinds
    // are armed at import time), so without this the trace carries launches from
    // before profiling started -- measured at 66/258 runtime events ending
    // before the first cpu_op, including a 250ms entry against a 157ms span.
    if (!in_window(ev)) {
      ++dropped_out_of_window;
      // Debug: print all dropped device events
      if (ev.kind != profiler::EventKind::Runtime) {
        FLAGOS_KINETO_LOG("[flagos] Dropped device event: kind=" << static_cast<int>(ev.kind)
                         << " start=" << ev.start_ns << " end=" << ev.end_ns
                         << " (window=[" << startTime << "," << endTime << "])\n");
      }
      switch (ev.kind) {
        case profiler::EventKind::Kernel: ++dropped_kernel; break;
        case profiler::EventKind::Memcpy: ++dropped_memcpy; break;
        case profiler::EventKind::Memset: ++dropped_memset; break;
        case profiler::EventKind::Runtime: ++dropped_runtime; break;
      }
      continue;
    }

    libkineto::GenericTraceActivity activity = detail::toActivity(ev);
    FLAGOS_KINETO_LOG("[flagos] toActivity: kind=" << static_cast<int>(ev.kind)
                     << " -> activityType=" << static_cast<int>(activity.activityType)
                     << " name=" << activity.activityName << "\n");

    // Resolve the CPU op that produced this device/runtime activity. The
    // callback is keyed on the *torch* correlation id, which is what
    // pushCorrelationId() published to the tracer; the tracer resolved it into
    // external_correlation_id. An absent mapping means "no CPU op to link to" --
    // we must NOT fall back to id 0, which is a valid torch correlation id
    // (hence the optional; see device_tracer.h).
    if (ev.external_correlation_id.has_value()) {
      ++link_candidates;
      if (have_resolver) {
        if (const auto* cpu_activity =
                getLinkedActivity(*ev.external_correlation_id)) {
          // MEASURED (Task 1 ablation): setting `linked` is what makes torch's
          // _parse_kineto_results attribute device time to the CPU op. With this
          // line removed and everything else identical, aten::mm's
          // self_device_time_total drops from ~816us to 0.
          activity.linked = cpu_activity;
          ++linked_count;
        }
      }
    }

    // Flow arrow ("ac2g"): kineto's chrome writer emits one half per activity --
    // 's' when flowStart is set, 'f' otherwise -- and pairs them by flow id. So
    // BOTH halves have to come from us, and both must carry the SAME id, which
    // is the vendor correlation id (shared by a launch's runtime event and the
    // kernel/memcpy event it produced). Using the torch correlation id here
    // instead would emit only unpaired 'f' halves that no viewer renders.
    if (ev.correlation_id != 0 && paired_correlations.count(ev.correlation_id)) {
      activity.flow.id = ev.correlation_id;
      activity.flow.type = libkineto::kLinkAsyncCpuGpu;
      activity.flow.start = (ev.kind == profiler::EventKind::Runtime) ? 1 : 0;
    }
    activity.log(logger);
  }

  // Report candidates alongside linked, so the three failure modes are
  // distinguishable in a log: "0/0 candidates" means nothing was profiled or the
  // tracer resolved no external correlations, "0/N candidates" means the
  // resolver was present but rejected every id, and the empty-resolver case is
  // called out explicitly above. Previously a bare "linked 0/N" conflated all
  // three.
  FLAGOS_KINETO_LOG("[flagos] processTrace(4-arg) linked " << linked_count << "/"
                   << link_candidates << " link candidates (of "
                   << events_.size() << " events, resolver="
                   << (have_resolver ? "present" : "EMPTY") << "), dropped "
                   << dropped_out_of_window << " outside window (kernel="
                   << dropped_kernel << " memcpy=" << dropped_memcpy
                   << " memset=" << dropped_memset << " runtime=" << dropped_runtime << ")\n");
}

std::unique_ptr<libkineto::DeviceInfo> FlagosKinetoProfilerSession::getDeviceInfo() {
  return std::make_unique<libkineto::DeviceInfo>(
      /*id=*/0,
      /*sortIndex=*/0,
      /*name=*/"flagos:GPU",
      /*label=*/"GPU");
}

std::vector<libkineto::ResourceInfo> FlagosKinetoProfilerSession::getResourceInfos() {
  std::vector<libkineto::ResourceInfo> resources;
  // Report up to 32 streams (will only show streams that actually had activity)
  for (int i = 0; i < 32; ++i) {
    resources.emplace_back(
        /*deviceId=*/0,
        /*id=*/i,
        /*sortIndex=*/i,
        /*name=*/std::string("Stream ") + std::to_string(i));
  }
  return resources;
}

std::unique_ptr<libkineto::CpuTraceBuffer> FlagosKinetoProfilerSession::getTraceBuffer() {
  return nullptr;  // We use processTrace instead
}

void FlagosKinetoProfilerSession::pushCorrelationId(uint64_t id) {
  if (tracer_ && tracer_->available()) {
    tracer_->pushCorrelation(id);
    detail::g_correlation_push_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void FlagosKinetoProfilerSession::popCorrelationId() {
  if (tracer_ && tracer_->available()) {
    tracer_->popCorrelation();
    detail::g_correlation_pop_count.fetch_add(1, std::memory_order_relaxed);
  }
}

// ===== FlagosKinetoProfiler Implementation =====

const std::string& FlagosKinetoProfiler::name() const {
  static const std::string kName = "flagos";
  return kName;
}

const std::set<libkineto::ActivityType>& FlagosKinetoProfiler::availableActivities() const {
  static const std::set<libkineto::ActivityType> kActivities = {
      libkineto::ActivityType::CONCURRENT_KERNEL,
      libkineto::ActivityType::GPU_MEMCPY,
      libkineto::ActivityType::GPU_MEMSET,
      libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
  };
  return kActivities;
}

std::unique_ptr<libkineto::IActivityProfilerSession> FlagosKinetoProfiler::configure(
    const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  FLAGOS_KINETO_LOG("[flagos] FlagosKinetoProfiler::configure called with " << activityTypes.size() << " activity types\n");
  return std::make_unique<FlagosKinetoProfilerSession>();
}

std::unique_ptr<libkineto::IActivityProfilerSession> FlagosKinetoProfiler::configure(
    int64_t profileStartTime,
    int64_t profileDuration,
    const std::set<libkineto::ActivityType>& activityTypes,
    const libkineto::Config& config) {
  return std::make_unique<FlagosKinetoProfilerSession>();
}

// ===== Registration =====

void registerFlagosKinetoProfiler() {
  libkineto::api().registerProfilerFactory(
      []() { return std::make_unique<FlagosKinetoProfiler>(); });
}

// Static initialization: register the kineto profiler when a device tracer is
// actually usable. The tracer's own static init (cupti_device_tracer.cc) is what
// arms the vendor library before the first device context; this registrar only
// cares whether a tracer exists at all.
namespace {
struct KinetoProfilerRegistrar {
  KinetoProfilerRegistrar() {
#if defined(FLAGOS_HAVE_MSPTI) || defined(FLAGOS_HAVE_MUPTI) || \
    defined(FLAGOS_HAVE_TOPSPTI)
    // Do not construct a vendor profiler shim during shared-library
    // initialization. Its handle is intentionally session-scoped and ordinary
    // operator processes must not touch the vendor profiler at import time.
    registerFlagosKinetoProfiler();
    FLAGOS_KINETO_LOG("[flagos] FlagosKinetoProfiler registered with kineto\n");
#else
    auto tracer = profiler::MakeDeviceTracer();
    if (tracer && tracer->available()) {
      registerFlagosKinetoProfiler();
      FLAGOS_KINETO_LOG("[flagos] FlagosKinetoProfiler registered with kineto\n");
    } else {
      FLAGOS_KINETO_LOG("[flagos] no device tracer available, "
                       "FlagosKinetoProfiler not registered\n");
    }
#endif
  }
};
static KinetoProfilerRegistrar g_registrar;
}  // namespace

}  // namespace flagos
}  // namespace c10

// C API for testing correlation push/pop calls
extern "C" {
__attribute__((visibility("default")))
uint64_t flagos_kineto_get_correlation_push_count() {
  return c10::flagos::detail::g_correlation_push_count.load(std::memory_order_relaxed);
}

__attribute__((visibility("default")))
uint64_t flagos_kineto_get_correlation_pop_count() {
  return c10::flagos::detail::g_correlation_pop_count.load(std::memory_order_relaxed);
}

__attribute__((visibility("default")))
void flagos_kineto_reset_correlation_counters() {
  c10::flagos::detail::g_correlation_push_count.store(0, std::memory_order_relaxed);
  c10::flagos::detail::g_correlation_pop_count.store(0, std::memory_order_relaxed);
}
}
