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
// NVIDIA device tracer: the only file in the profiler that knows what CUPTI is.
// Everything above it consumes DeviceEvent through the DeviceTracer interface,
// so adding a vendor means adding a sibling of this file and nothing else.

#include "device_tracer.h"
#include "cupti_shim.h"
#include <flagos_env.h>

#include <cxxabi.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <time.h>
#include <unordered_map>
#include <utility>
#include <vector>

// CUDA's header-only occupancy calculator. Pure header: no linking, no CUDA
// context, no device query -- it takes device attributes and a launch config as
// plain integers and returns the theoretical max resident blocks per SM. That is
// what turns a grid size into torch-cuda's "est. achieved occupancy %".
// <climits> must precede it: the header uses INT_MAX without including it.
#if defined(FLAGOS_HAVE_CUDA_OCCUPANCY)
#include <cuda_occupancy.h>
#endif

// Diagnostic logging is gated behind FLAGOS_TRACE=1 so that normal
// profiling runs stay quiet (these callbacks fire once per buffer/session).
namespace {
inline bool flagos_cupti_debug() {
  static const bool on = flagos_env::EnvFlag("FLAGOS_TRACE");
  return on;
}
}  // namespace
#define FLAGOS_CUPTI_LOG(expr) \
  do { if (flagos_cupti_debug()) { std::cerr << expr; } } while (0)

// CUPTI activity record layouts. We mirror the cu12 runtime's
// cupti_activity.h EXACTLY (verified against nvidia-cuda-cupti-cu12's
// CUpti_ActivityKernel9 / CUpti_ActivityMemcpy6): the records are
// __attribute__((packed)) with no natural-alignment padding, `kind` is a
// 4-byte enum, and the kernel `name` is a `const char*` pointer field located
// deep in the struct -- NOT an inline string at a small offset. An earlier
// hand-guessed layout (uint8_t kind + pad[7], name at +56) decoded garbage
// (empty names, zero durations), so these must stay byte-accurate.
//
// CUpti_ActivityKind is a 4-byte enum on this platform.
using CUpti_ActivityKind_t = uint32_t;

#pragma pack(push, 1)
struct CUpti_Activity {
  CUpti_ActivityKind_t kind;
};

// Prefix of CUpti_ActivityKernel9 up to and including `name`. Field order and
// widths are copied verbatim from the cu12 header; the packed attribute makes
// offsets match the runtime records byte-for-byte.
struct CUpti_ActivityKernel9_Compat {
  CUpti_ActivityKind_t kind;
  uint8_t cacheConfig;            // union { uint8_t both; ... } cacheConfig
  uint8_t sharedMemoryConfig;
  uint16_t registersPerThread;
  uint32_t partitionedGlobalCacheRequested;  // enum, 4 bytes
  uint32_t partitionedGlobalCacheExecuted;   // enum, 4 bytes
  uint64_t start;
  uint64_t end;
  uint64_t completed;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
  int32_t blockX;
  int32_t blockY;
  int32_t blockZ;
  int32_t staticSharedMemory;
  int32_t dynamicSharedMemory;
  uint32_t localMemoryPerThread;
  uint32_t localMemoryTotal;
  uint32_t correlationId;
  int64_t gridId;
  const char* name;
  // Fields past `name`, needed for the `queued` timestamp that torch-cuda's
  // trace reports. Verified identical in BOTH the pip cu12 CUPTI header (the
  // copy actually bound at runtime, CUPTI_API_VERSION 26) and the system
  // CUDA-13.0 header, so extending this far is no more version-fragile than the
  // prefix above. We stop at `submitted`: everything past it is unused, and
  // every mirrored field is a field that can go wrong.
  void* reserved0;
  uint64_t queued;
  uint64_t submitted;
  // ... remaining fields omitted (we only read up to `submitted`).
};

// Prefix of CUpti_ActivityMemcpy6 up to correlationId.
struct CUpti_ActivityMemcpy_Compat {
  CUpti_ActivityKind_t kind;
  uint8_t copyKind;
  uint8_t srcKind;
  uint8_t dstKind;
  uint8_t flags;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  uint32_t correlationId;
};

// Prefix of CUpti_ActivityMemset4 up to correlationId. Same shape as the memcpy
// record except the leading discriminator is a 4-byte `value` rather than the
// four 1-byte copy/src/dst/flags fields -- so the payload fields land at the
// same offsets, which is why the same plausibility checks apply unchanged.
struct CUpti_ActivityMemset_Compat {
  CUpti_ActivityKind_t kind;
  uint32_t value;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  uint32_t correlationId;
};

// CUpti_ActivityAPI, used for CUPTI_ACTIVITY_KIND_RUNTIME (and DRIVER). Copied
// verbatim from the cu12 header; this record has been stable for many CUPTI
// releases (no versioned CUpti_ActivityAPI<N> exists), so it is the least
// layout-risky of the mirrors here.
struct CUpti_ActivityAPI_Compat {
  CUpti_ActivityKind_t kind;
  uint32_t cbid;  // CUpti_CallbackId
  uint64_t start;
  uint64_t end;
  uint32_t processId;
  uint32_t threadId;
  uint32_t correlationId;
  uint32_t returnValue;
};

// CUpti_ActivityExternalCorrelation: maps a CUPTI correlationId to the
// externalId we pushed via cuptiActivityPushExternalCorrelationId (i.e. torch's
// own correlation id). Without this we cannot translate CUPTI's numbering into
// the numbering kineto's getLinkedActivity callback expects.
struct CUpti_ActivityExternalCorrelation_Compat {
  CUpti_ActivityKind_t kind;
  uint32_t externalKind;
  uint64_t externalId;
  uint32_t correlationId;
  uint32_t reserved;
};
#pragma pack(pop)

// MetaX's versioned records are aligned to 8 bytes and include fields after the
// prefixes above. The parser only reads the common prefix, but the scanner must
// advance by the complete vendor record size or it will land in the middle of
// the next record.
constexpr size_t kMcptiKernel8RecordSize = 200;
constexpr size_t kMcptiMemcpy5RecordSize = 88;
constexpr size_t kMcptiMemset4RecordSize = 88;

namespace c10 {
namespace flagos {
namespace profiler {

namespace {

// --- record-layout self-check ------------------------------------------------
//
// The structs above mirror ONE CUPTI version's layout by hand. NVIDIA adds
// fields and publishes a new CUpti_ActivityKernel<N> every few releases, so on a
// different CUPTI the mirror can be wrong -- and the failure mode is silent: we
// decode whatever bytes land at those offsets and emit a trace full of empty
// names and zero durations (exactly the bug this code was fixed for once).
//
// So rather than trust the mirror, check the values it produces. A correct
// decode has properties that garbage almost never satisfies, and each check is
// version-independent -- it asserts something true of any sane kernel record,
// not something true of cu12 specifically:
//
//   * end >= start                  (a kernel cannot finish before it starts)
//   * duration is not absurd        (CUPTI timestamps are ns since boot; a
//                                    multi-hour single kernel means the offsets
//                                    are misaligned, not a slow kernel)
//   * start is a plausible boot-relative ns timestamp, i.e. non-zero
//   * name, if non-null, points at readable non-empty text
//
// On failure we drop the record and report once, naming the bound CUPTI version,
// so the user gets "your CUPTI layout is unsupported" instead of a mysteriously
// empty timeline.

// A single kernel longer than this means we are reading the wrong offsets. Real
// kernels run micro- to milliseconds; an hour is ~7 orders of magnitude beyond
// anything legitimate, so this rejects garbage without risking a false positive
// on a genuinely slow kernel.
constexpr uint64_t kMaxPlausibleDurationNs = 3600ull * 1000 * 1000 * 1000;

std::atomic<uint64_t> g_layout_reject_count{0};
std::once_flag g_layout_warn_once;

// Reports the first rejected record, then stays quiet: a bad layout rejects
// every record, and one diagnostic is informative where thousands are noise.
void reportLayoutMismatch(const char* what) {
  auto& shim = CuptiShim::get();
  std::call_once(g_layout_warn_once, [&] {
    std::cerr
        << "[flagos] CUPTI activity records failed the layout self-check ("
        << what << ").\n"
        << "[flagos]   bound CUPTI: "
        << (shim.library_path[0] ? shim.library_path : "<unknown>")
        << " (API version " << shim.api_version << ")\n"
        << "[flagos]   This build mirrors the CUpti_ActivityKernel9 /"
           " CUpti_ActivityMemcpy6 layouts by hand, so a CUPTI whose record\n"
        << "[flagos]   layout differs cannot be decoded. GPU kernel events will"
           " be missing from the trace; CPU-side profiling is unaffected.\n"
        << "[flagos]   Set FLAGOS_TRACER_LIBRARY to a matching libcupti, or"
           " report this CUPTI version so its layout can be added.\n";
  });
  g_layout_reject_count.fetch_add(1, std::memory_order_relaxed);
}

// True when a decoded (start, end) pair is self-consistent.
//
// NVIDIA CUPTI timestamps are boot-relative and normally start > 0. MetaX MCPTI
// timestamps are vendor-domain (hardware counter) and can legitimately be small
// values. Requiring start > 0 would reject all MCPTI records as layout mismatches.
// The core layout check is end >= start and reasonable duration; those properties
// hold in any timestamp domain.
bool timestampsPlausible(uint64_t start, uint64_t end) {
  if (end < start) {
    return false;
  }
  return (end - start) <= kMaxPlausibleDurationNs;
}

// True when a record actually carries timing, as opposed to declaring that it
// has none.
//
// Both CUPTI and MCPTI document `start == end == 0` as "timestamp information
// could not be collected for this activity" -- it is a sentinel, not a
// zero-length event at the origin of the clock. Such a record must be dropped
// before timestamp conversion rather than reported as a layout mismatch: the
// layout is fine, the vendor simply had nothing to put in those fields.
//
// Passing the sentinel through is not harmless. toRealtime() maps the vendor
// clock onto CLOCK_REALTIME by interpolating from a session start sample, so a
// 0 input becomes a large negative offset from the session start and lands
// nowhere near the capture window. Observed on MetaX as kernels timestamped
// 350743133379 against a window of ~1787048584694419530, which silently dropped
// every device event from the trace.
bool timestampsCollected(uint64_t start, uint64_t end) {
  return !(start == 0 && end == 0);
}

// CUPTI's documented "this timestamp was never captured" sentinel.
constexpr uint64_t kCuptiTimestampUnknown = 0xFFFFFFFFFFFFFFFFull;

// True when the `queued` timestamp decoded out of the extended Kernel9 mirror is
// trustworthy.
//
// This is the same idea as timestampsPlausible, applied to the one field we read
// past the region Task 1/3 validated empirically. `queued` is a CUPTI timestamp
// in the same epoch-ns domain as `start`, and a launch is by definition queued
// before it starts -- so `0 < queued <= start` is a property any correct decode
// satisfies and a misaligned one almost never does.
//
// Deliberately NOT routed through reportLayoutMismatch: this field is only
// populated when latency timestamps were enabled, and we never call
// cuptiActivityEnableLatencyTimestamps -- so failing here is the NORMAL case,
// not a layout bug. Burning the once-flag on it would suppress the real layout
// diagnostic.
//
// Both "unset" spellings are rejected. CUPTI documents CUPTI_TIMESTAMP_UNKNOWN
// (0xFFFF...), but the copy bound here was MEASURED writing a literal 0 into
// this field on every kernel record -- so checking only the documented sentinel
// would have let a bogus 0 through as if it were a real timestamp. Verified by
// instrumenting the branch and dumping the raw field; the offsets themselves
// were cross-checked against the installed CUPTI header
// (name=104, queued=120, submitted=128), so the 0 is what CUPTI stored, not a
// misread.
bool queuedTimestampPlausible(uint64_t queued, uint64_t start) {
  return queued != 0 && queued != kCuptiTimestampUnknown && queued <= start;
}

// --- occupancy ---------------------------------------------------------------
//
// torch-cuda's kernel args carry three derived fields that are not in the CUPTI
// record: "blocks per SM", "warps per SM" and "est. achieved occupancy %".
// Reproducing them needs (a) the launch geometry, which the record has, and
// (b) static device properties, which it does not.
//
// FORMULA (verified byte-exact against 20 distinct live torch-cuda kernel
// configurations on this A100 -- 60/60 across the three fields, compared as
// formatted strings, not as approximate numbers):
//
//   blocks per SM = gridBlocks / smCount                       (NOT clamped)
//   warps per SM  = blocksPerSM * blockSize / warpSize         (NOT clamped)
//   occupancy %   = min(blocksPerSM, theoreticalMaxBlocksPerSM)
//                   * blockSize / maxThreadsPerSM * 100
//
// Two things here are easy to get wrong and were settled by measurement:
//
//  1. Only the occupancy computation clamps. "blocks per SM" and "warps per SM"
//     are reported UNCAPPED -- a live 2048-block/256-thread kernel reports
//     warps per SM = 151.703705, far above the SM's 64-warp capacity.
//  2. warps per SM uses PLAIN division by warpSize, not a ceiling. A live
//     16-thread bitonic-sort kernel reports 0.004630 == (1/108)*16/32, whereas
//     a ceil(16/32)=1 form would give 0.009259 -- double. (An earlier draft of
//     this work specified ceil; the two forms agree for every block size that
//     is a multiple of 32, which is why the difference only shows on a kernel
//     like that one.)
//
// The clamp in (a) is what makes this correct rather than merely plausible: a
// naive warpsPerSM/maxWarps ratio scores a register-limited 6272-block kernel at
// 100% where the real answer is 25%.

// Static per-device properties needed by the occupancy calculator. Cached
// because the alternative is a handful of driver queries per KERNEL RECORD,
// inside the buffer-processing hot path.
struct DeviceOccupancyProps {
  bool valid = false;
  int sm_count = 0;
  int warp_size = 0;
  int max_threads_per_sm = 0;
#if defined(FLAGOS_HAVE_CUDA_OCCUPANCY)
  cudaOccDeviceProp occ_prop;
#endif
};

// cudaDeviceGetAttribute, resolved the same way deviceCount() resolves
// cudaGetDeviceCount: this build is CPU-torch + an external libtorch_cuda.so, so
// libcudart is in the process but not on our link line. Querying an attribute
// does not create a context.
int queryDeviceAttribute(int attr, int device, bool* ok) {
  using GetAttrFn = int (*)(int*, int, int);
#if defined(FLAGOS_METAX_MCPTI)
  static const GetAttrFn fn = reinterpret_cast<GetAttrFn>(
      dlsym(RTLD_DEFAULT, "wcudaDeviceGetAttribute"));
#else
  static const GetAttrFn fn = reinterpret_cast<GetAttrFn>(
      dlsym(RTLD_DEFAULT, "cudaDeviceGetAttribute"));
#endif
  int value = 0;
  if (fn == nullptr || fn(&value, attr, device) != 0) {
    *ok = false;
    return 0;
  }
  return value;
}

// cudaDeviceAttr values from driver_types.h (checked against the installed
// CUDA-13.0 header). These are a stable public ABI enum, append-only across
// releases.
enum : int {
  kAttrMaxThreadsPerBlock = 1,
  kAttrMaxSharedMemoryPerBlock = 8,
  kAttrWarpSize = 10,
  kAttrMaxRegistersPerBlock = 12,
  kAttrMultiProcessorCount = 16,
  kAttrMaxThreadsPerMultiProcessor = 39,
  kAttrComputeCapabilityMajor = 75,
  kAttrComputeCapabilityMinor = 76,
  kAttrMaxSharedMemoryPerMultiprocessor = 81,
  kAttrMaxRegistersPerMultiprocessor = 82,
};

DeviceOccupancyProps queryDeviceOccupancyProps(int device) {
  DeviceOccupancyProps props;
#if defined(FLAGOS_HAVE_CUDA_OCCUPANCY)
  bool ok = true;
  const int cc_major = queryDeviceAttribute(kAttrComputeCapabilityMajor, device, &ok);
  const int cc_minor = queryDeviceAttribute(kAttrComputeCapabilityMinor, device, &ok);
  const int max_threads_per_block =
      queryDeviceAttribute(kAttrMaxThreadsPerBlock, device, &ok);
  const int max_threads_per_sm =
      queryDeviceAttribute(kAttrMaxThreadsPerMultiProcessor, device, &ok);
  const int regs_per_block = queryDeviceAttribute(kAttrMaxRegistersPerBlock, device, &ok);
  const int regs_per_sm =
      queryDeviceAttribute(kAttrMaxRegistersPerMultiprocessor, device, &ok);
  const int warp_size = queryDeviceAttribute(kAttrWarpSize, device, &ok);
  const int shmem_per_block =
      queryDeviceAttribute(kAttrMaxSharedMemoryPerBlock, device, &ok);
  const int shmem_per_sm =
      queryDeviceAttribute(kAttrMaxSharedMemoryPerMultiprocessor, device, &ok);
  const int sm_count = queryDeviceAttribute(kAttrMultiProcessorCount, device, &ok);

  // Any failed query, or a zero in a divisor, makes every derived number
  // meaningless -- so mark the whole set invalid and omit the fields rather than
  // publish a 0.0 that is indistinguishable from a real measurement.
  if (!ok || sm_count <= 0 || warp_size <= 0 || max_threads_per_sm <= 0) {
    return props;
  }

  props.occ_prop.computeMajor = cc_major;
  props.occ_prop.computeMinor = cc_minor;
  props.occ_prop.maxThreadsPerBlock = max_threads_per_block;
  props.occ_prop.maxThreadsPerMultiprocessor = max_threads_per_sm;
  props.occ_prop.regsPerBlock = regs_per_block;
  props.occ_prop.regsPerMultiprocessor = regs_per_sm;
  props.occ_prop.warpSize = warp_size;
  props.occ_prop.sharedMemPerBlock = static_cast<size_t>(shmem_per_block);
  props.occ_prop.sharedMemPerMultiprocessor = static_cast<size_t>(shmem_per_sm);
  props.occ_prop.numSms = sm_count;
  // Left at the default-constructed 0: neither affects the result under
  // FUNC_SHMEM_LIMIT_DEFAULT, confirmed by a sweep over 2700 (blockSize,
  // registers, static/dynamic shared memory) combinations in which varying both
  // changed nothing.
  props.occ_prop.sharedMemPerBlockOptin = 0;
  props.occ_prop.reservedSharedMemPerBlock = 0;

  props.sm_count = sm_count;
  props.warp_size = warp_size;
  props.max_threads_per_sm = max_threads_per_sm;
  props.valid = true;
#else
  (void)device;
#endif
  return props;
}

// Cached device props. Guarded by g_tracer_mutex, which processBuffer already
// holds -- the buffer-completed callback is the only caller.
constexpr int kMaxCachedDevices = 16;
DeviceOccupancyProps g_device_props[kMaxCachedDevices];
bool g_device_props_cached[kMaxCachedDevices] = {};

const DeviceOccupancyProps* cachedDeviceOccupancyProps(uint32_t device) {
  if (device >= kMaxCachedDevices) {
    return nullptr;
  }
  if (!g_device_props_cached[device]) {
    g_device_props[device] = queryDeviceOccupancyProps(static_cast<int>(device));
    g_device_props_cached[device] = true;
  }
  return g_device_props[device].valid ? &g_device_props[device] : nullptr;
}

struct KernelOccupancy {
  bool valid = false;
  double blocks_per_sm = 0.0;
  double warps_per_sm = 0.0;
  int occupancy_pct = 0;
};

// Computes the three derived occupancy fields, or returns valid=false when the
// device attributes were unavailable (see the omit-vs-zero note above).
//
// Arithmetic is done in FLOAT, not double, deliberately: torch-cuda's own values
// are float-rounded, and only float reproduces them exactly. Live counter-
// example: 640 blocks / 108 SMs * 4 warps prints 23.703703 as a float and
// 23.703704 as a double -- and torch-cuda reports 23.703703.
KernelOccupancy computeKernelOccupancy(
    uint32_t device,
    int64_t grid_blocks,
    int32_t block_size,
    uint16_t registers_per_thread,
    int64_t shared_memory) {
  KernelOccupancy out;
#if defined(FLAGOS_HAVE_CUDA_OCCUPANCY)
  const DeviceOccupancyProps* props = cachedDeviceOccupancyProps(device);
  if (props == nullptr || grid_blocks <= 0 || block_size <= 0) {
    return out;
  }

  cudaOccFuncAttributes attr;
  attr.maxThreadsPerBlock = INT_MAX;  // the record does not carry a limit
  attr.numRegs = static_cast<int>(registers_per_thread);
  // Static and dynamic shared memory are summed into sharedSizeBytes with a zero
  // dynamic argument. The record reports them separately, but the calculator
  // treats the split as immaterial: verified over 2700 combinations, passing
  // (static+dynamic, 0) and (static, dynamic) never differed.
  attr.sharedSizeBytes = static_cast<size_t>(shared_memory < 0 ? 0 : shared_memory);
  attr.partitionedGCConfig = PARTITIONED_GC_OFF;
  attr.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
  attr.maxDynamicSharedSizeBytes = 0;

  cudaOccDeviceState state;
  cudaOccResult result;
  if (cudaOccMaxActiveBlocksPerMultiprocessor(
          &result, &props->occ_prop, &attr, &state, block_size, 0) !=
      CUDA_OCC_SUCCESS) {
    return out;
  }

  const float blocks_per_sm =
      static_cast<float>(grid_blocks) / static_cast<float>(props->sm_count);
  const float warps_per_sm =
      blocks_per_sm * static_cast<float>(block_size) /
      static_cast<float>(props->warp_size);
  const float occupancy =
      std::min(blocks_per_sm,
               static_cast<float>(result.activeBlocksPerMultiprocessor)) *
      static_cast<float>(block_size) /
      static_cast<float>(props->max_threads_per_sm) * 100.0f;

  out.blocks_per_sm = blocks_per_sm;
  out.warps_per_sm = warps_per_sm;
  out.occupancy_pct = static_cast<int>(std::lroundf(occupancy));
  out.valid = true;
#else
  (void)device;
  (void)grid_blocks;
  (void)block_size;
  (void)registers_per_thread;
  (void)shared_memory;
#endif
  return out;
}

// Formats a derived occupancy value the way torch-cuda's trace does: fixed six
// decimals of a float ("2.370370", "151.703705", "4.000000"). Matching the
// FORMAT as well as the value is what lets a parity test diff the two traces
// textually instead of with a float tolerance.
std::string formatOccupancyValue(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6f", v);
  return std::string(buf);
}

// Shortest decimal string that round-trips back to exactly `v`. This is what
// torch-cuda's "memory bandwidth (GB/s)" values look like -- full precision,
// no trailing zeros ("0.2962962962962963", "12.641975308641975") -- as opposed
// to the fixed-6-decimal form used for the occupancy fields. Verified by
// reproducing all 10 distinct bandwidth values from a live torch-cuda trace
// character-for-character.
//
// Note %g may emit exponent notation for small magnitudes ("8e-05"); that is
// valid JSON and the kineto adaptor's literal classifier accepts it.
std::string formatShortestRoundtrip(double v) {
  char buf[64];
  for (int precision = 1; precision <= 17; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    if (std::strtod(buf, nullptr) == v) {
      break;
    }
  }
  return std::string(buf);
}

// "memory bandwidth (GB/s)" for a transfer record. bytes-per-nanosecond IS
// GB/s (1e9 bytes / 1e9 ns), so no scaling is needed -- confirmed against live
// torch-cuda values, e.g. 512 bytes over 1728ns -> 0.2962962962962963.
//
// Guards end > start rather than reusing timestampsPlausible, which only gives
// end >= start: a zero-duration record would divide by zero and put "inf" in
// the trace, which is not valid JSON. Returns false to mean "omit the key".
bool computeMemoryBandwidth(uint64_t bytes, uint64_t start, uint64_t end,
                            std::string* out) {
  if (end <= start) {
    return false;
  }
  *out = formatShortestRoundtrip(static_cast<double>(bytes) /
                                 static_cast<double>(end - start));
  return true;
}

// True when `name` is either absent or points at readable non-empty text.
// A misaligned layout usually yields a wild pointer here; probing it with a
// short bounded read is far cheaper than the SIGSEGV that blind trust invites.
bool kernelNamePlausible(const char* name) {
  if (name == nullptr) {
    return true;  // absent is legal; we substitute a placeholder below
  }
  constexpr size_t kMaxNameLen = 4096;
  size_t len = strnlen(name, kMaxNameLen);
  return len > 0 && len < kMaxNameLen;
}

// CUPTI interns kernel names as the linker emitted them, i.e. mangled for any
// C++ template kernel (`_ZN2at6native...`). torch's own CUDA path demangles
// before logging, so we do too -- otherwise the flagos timeline is unreadable
// where the CUDA timeline is not. Names that are already plain (`ampere_sgemm_*`
// and other extern-"C"-style kernels) fail to demangle with status -2 and are
// passed through untouched.
std::string demangleName(const char* mangled) {
  if (mangled == nullptr || mangled[0] == '\0') {
    return "kernel";
  }
  int status = -1;
  char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  if (status == 0 && demangled != nullptr) {
    std::string result(demangled);
    free(demangled);
    return result;
  }
  if (demangled != nullptr) {
    free(demangled);
  }
  return std::string(mangled);
}

}  // namespace

class CuptiDeviceTracer;

namespace {
void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
void bufferCompleted(
    CUcontext context,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize);

// Global tracer pointer for the buffer callbacks (CUPTI callbacks are C-style
// and cannot capture context).
CuptiDeviceTracer* g_active_tracer = nullptr;
std::mutex g_tracer_mutex;

// Buffer pool for CUPTI activity records
constexpr size_t kBufferSize = 8 * 1024 * 1024;  // 8MB per buffer
constexpr size_t kBufferAlignment = 8;
}  // namespace

/**
 * CuptiDeviceTracer: NVIDIA implementation of the DeviceTracer interface.
 * Owns every CUPTI detail -- buffer callbacks, activity-record layouts, the
 * layout self-check, and external-correlation resolution -- and hands the layer
 * above a flat vector of vendor-neutral DeviceEvents.
 */
class CuptiDeviceTracer : public DeviceTracer {
 public:
  bool available() const override { return CuptiShim::get().ok; }

  void start() override {
    auto& shim = CuptiShim::get();
    if (!shim.ok) {
      FLAGOS_CUPTI_LOG("[flagos] CUPTI not available in start()\n");
      return;
    }

    // CUPTI activity timestamps are in the clock domain returned by
    // cuptiGetTimestamp. Kineto's capture window uses CLOCK_REALTIME. Sample both
    // domains at each end of the session: PPU's compatibility clock differs in
    // both epoch and observed rate under load, so a single offset is insufficient.
    start_clock_sample_ = sampleClocks(shim);

    // Publish this tracer and reset its buffers under the lock, then RELEASE the
    // lock before touching CUPTI. cuptiActivityFlushAll() invokes bufferCompleted
    // synchronously on this same thread, and bufferCompleted also locks
    // g_tracer_mutex -- holding it across the flush self-deadlocks on the
    // non-recursive mutex (observed as a hang in torch.profiler start_trace).
    {
      std::lock_guard<std::mutex> lock(g_tracer_mutex);
      g_active_tracer = this;
      events_.clear();
      external_correlation_.clear();
    }

    FLAGOS_CUPTI_LOG("[flagos] CuptiDeviceTracer::start() called\n");

    // PyTorch's kineto owns the process-global CUPTI activity callbacks on
    // CUDA-enabled builds, so reclaim them for the FlagOS session. Registering at
    // module initialization alone is insufficient: kineto may replace them after
    // torch_fl is imported, which makes HGPTI reject our buffer at flush as
    // "unknown" and leaves the PrivateUse1 trace empty.
    CUptiResult res_cb =
        shim.ActivityRegisterCallbacks(bufferRequested, bufferCompleted);

    // KERNEL and CONCURRENT_KERNEL are mutually exclusive in PPU's CUPTI
    // compatibility layer: whichever is enabled first succeeds and the second
    // returns HGPTI_ERROR_NOT_COMPATIBLE. Enable both so NVIDIA gets concurrent
    // records while PPU accepts the already-armed mode; the parser handles either
    // record kind.
    CUptiResult res1 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    CUptiResult res_k = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    CUptiResult res2 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    CUptiResult res_ms = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    CUptiResult res_rt = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    CUptiResult res_ec =
        shim.ActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
    FLAGOS_CUPTI_LOG("[flagos] Activity setup results: CALLBACKS=" << res_cb
              << ", CONCURRENT_KERNEL=" << res1
              << ", KERNEL=" << res_k << ", MEMCPY=" << res2
              << ", MEMSET=" << res_ms
              << ", RUNTIME=" << res_rt
              << ", EXTERNAL_CORRELATION=" << res_ec << "\n");

    // Force a flush to kickstart CUPTI activity collection (lock released above).
    CUptiResult res3 = shim.ActivityFlushAll(1);
    FLAGOS_CUPTI_LOG("[flagos] Initial ActivityFlushAll result: " << res3 << "\n");
  }

  void stop() override {
    auto& shim = CuptiShim::get();
    if (!shim.ok) {
      return;
    }

    FLAGOS_CUPTI_LOG("[flagos] CuptiDeviceTracer::stop() called\n");

    end_clock_sample_ = sampleClocks(shim);

    // Flush while activities and the active tracer are still live. PPU's HGPTI
    // backend follows the documented forced-termination ordering and may deliver
    // the completion callback synchronously from this call.
    FLAGOS_CUPTI_LOG("[flagos] Flushing CUPTI activities...\n");
    CUptiResult flush_res = shim.ActivityFlushAll(1);
    FLAGOS_CUPTI_LOG("[flagos] ActivityFlushAll result: " << flush_res << "\n");

    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);

    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    g_active_tracer = nullptr;

  }

  // Resolution of external correlations happens HERE, not at parse time: CUPTI
  // makes no ordering guarantee between an activity record and the
  // EXTERNAL_CORRELATION record that maps it, and they routinely land in
  // different buffers. Resolving per-record would silently lose every mapping
  // that arrived late -- and losing it is invisible (getLinkedActivity returns
  // nullptr, per-op device time reverts to 0, no diagnostic). drain() runs after
  // stop()'s blocking flush, so by then every record has been seen.
  std::vector<DeviceEvent> drain() override {
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
#if defined(FLAGOS_METAX_MCPTI)
    auto& shim = CuptiShim::get();
#endif
    for (auto& ev : events_) {
      ev.start_ns = toRealtime(ev.start_ns);
      ev.end_ns = toRealtime(ev.end_ns);
      ev.external_correlation_id = lookupExternalCorrelation(ev.correlation_id);
#if defined(FLAGOS_METAX_MCPTI)
      // mcptiActivityGetApiName is not reentrant from bufferCompleted, so resolve
      // runtime names only after stop() has flushed and disabled all activities.
      if (ev.kind == EventKind::Runtime && shim.ActivityGetApiName) {
        auto cbid = ev.metadata.find("cbid");
        if (cbid != ev.metadata.end()) {
          const char* api_name = nullptr;
          const auto value = static_cast<uint32_t>(std::stoul(cbid->second));
          if (shim.ActivityGetApiName(
                  CUPTI_ACTIVITY_KIND_RUNTIME, value, &api_name) == CUPTI_SUCCESS &&
              api_name != nullptr && api_name[0] != '\0') {
            ev.name = api_name;
          }
        }
      }
#endif
    }
    auto result = std::move(events_);
    return result;
  }

  void pushCorrelation(uint64_t id) override {
    auto& shim = CuptiShim::get();
    if (shim.ok && shim.ActivityPushExternalCorrelationId) {
      shim.ActivityPushExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
    }
  }

  void popCorrelation() override {
    auto& shim = CuptiShim::get();
    if (shim.ok && shim.ActivityPopExternalCorrelationId) {
      uint64_t id = 0;
      shim.ActivityPopExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id);
    }
  }

  // Queried through the already-loaded CUDA runtime rather than linked against:
  // this build is CPU-torch + an external libtorch_cuda.so, so libcudart is in
  // the process but not on our link line. cudaGetDeviceCount does not create a
  // context, so this is safe to call at any point.
  int deviceCount() const override {
    static const int count = [] {
      using GetDeviceCountFn = int (*)(int*);
      auto fn = reinterpret_cast<GetDeviceCountFn>(
          dlsym(RTLD_DEFAULT, "cudaGetDeviceCount"));
      int n = 0;
      if (fn != nullptr && fn(&n) == 0 && n > 0) {
        return n;
      }
      return 1;
    }();
    return count;
  }

  // Called from bufferCompleted with g_tracer_mutex held.
  void processBuffer(uint8_t* buffer, size_t validSize) {
    auto& shim = CuptiShim::get();

    CUpti_Activity* record = nullptr;
    CUpti_Activity* prev_record = nullptr;
    size_t record_count = 0;
    constexpr size_t kMaxRecordsPerBuffer = 100000;
#if defined(FLAGOS_METAX_MCPTI)
    size_t scan_offset = 0;
#endif
    while (record_count < kMaxRecordsPerBuffer) {
      CUptiResult status = CUPTI_SUCCESS;
#if defined(FLAGOS_METAX_MCPTI)
      record = nullptr;
      // MCPTI 3.8 returns a stale pointer after some external-correlation
      // records, so its iterator can stop in the middle of a valid buffer. Walk
      // the documented 8-byte-aligned record layouts ourselves and resync on
      // the next plausible activity header when a vendor record is malformed.
      size_t record_size = 0;
      while (scan_offset + sizeof(uint32_t) <= validSize) {
        auto* candidate = buffer + scan_offset;
        const auto kind = *reinterpret_cast<const uint32_t*>(candidate);
        if (kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION &&
            scan_offset + sizeof(CUpti_ActivityExternalCorrelation_Compat) <=
                validSize) {
          auto* ext = reinterpret_cast<
              CUpti_ActivityExternalCorrelation_Compat*>(candidate);
          if (ext->externalKind >= 1 && ext->externalKind <= 5 &&
              ext->correlationId != 0) {
            record_size = sizeof(CUpti_ActivityExternalCorrelation_Compat);
          }
        } else if (kind == CUPTI_ACTIVITY_KIND_RUNTIME &&
                   scan_offset + sizeof(CUpti_ActivityAPI_Compat) <= validSize) {
          auto* rt = reinterpret_cast<CUpti_ActivityAPI_Compat*>(candidate);
          if (rt->cbid < 1000 && rt->correlationId != 0 &&
              timestampsCollected(rt->start, rt->end) &&
              timestampsPlausible(rt->start, rt->end)) {
            record_size = sizeof(CUpti_ActivityAPI_Compat);
          }
        } else if ((kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                    kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) &&
                   scan_offset + kMcptiKernel8RecordSize <= validSize) {
          auto* kernel = reinterpret_cast<CUpti_ActivityKernel9_Compat*>(candidate);
          const bool geometry_plausible =
              kernel->gridX > 0 && kernel->gridX < (1 << 20) &&
              kernel->gridY > 0 && kernel->gridY < (1 << 20) &&
              kernel->gridZ > 0 && kernel->gridZ < (1 << 20) &&
              kernel->blockX > 0 && kernel->blockX < (1 << 20) &&
              kernel->blockY > 0 && kernel->blockY < (1 << 20) &&
              kernel->blockZ > 0 && kernel->blockZ < (1 << 20) &&
              kernel->correlationId != 0;
          if (geometry_plausible &&
              ((kernel->start == 0 && kernel->end == 0) ||
               (timestampsCollected(kernel->start, kernel->end) &&
                timestampsPlausible(kernel->start, kernel->end)))) {
            record_size = kMcptiKernel8RecordSize;
          }
        } else if ((kind == CUPTI_ACTIVITY_KIND_MEMCPY ||
                    kind == CUPTI_ACTIVITY_KIND_MEMSET) &&
                   scan_offset + kMcptiMemcpy5RecordSize <= validSize) {
          auto* copy = reinterpret_cast<CUpti_ActivityMemcpy_Compat*>(candidate);
          if (copy->bytes != 0 && copy->correlationId != 0 &&
              timestampsCollected(copy->start, copy->end) &&
              timestampsPlausible(copy->start, copy->end)) {
            record_size = (kind == CUPTI_ACTIVITY_KIND_MEMCPY)
                ? kMcptiMemcpy5RecordSize
                : kMcptiMemset4RecordSize;
          }
        }
        if (record_size != 0) {
          record = reinterpret_cast<CUpti_Activity*>(candidate);
          scan_offset += record_size;
          break;
        }
        scan_offset += 8;
      }
      if (record == nullptr || scan_offset > validSize) {
        break;
      }
#else
      status = shim.ActivityGetNextRecord(buffer, validSize, &record);
      // NVIDIA CUPTI: iteration ends when status != SUCCESS or record == null.
      if (status != CUPTI_SUCCESS || !record || record == prev_record) {
        break;
      }
#endif
      if (status != CUPTI_SUCCESS || !record || record == prev_record) {
        break;
      }
      prev_record = record;
      ++record_count;

      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
          record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        auto* kernel = reinterpret_cast<CUpti_ActivityKernel9_Compat*>(record);

        // Validate before trusting the mirrored layout (see the self-check notes
        // above). Emitting a record that failed these checks would put garbage
        // timestamps into the trace, which is worse than omitting it.
        if (!timestampsCollected(kernel->start, kernel->end)) {
          // Vendor reported "no timing available" for this kernel; not a layout
          // problem, so drop it quietly rather than burning the mismatch flag.
          FLAGOS_CUPTI_LOG("[flagos] skipping kernel record with no timing\n");
          continue;
        }
        if (!timestampsPlausible(kernel->start, kernel->end)) {
          reportLayoutMismatch("kernel timestamps implausible");
          continue;
        }
        if (!kernelNamePlausible(kernel->name)) {
          reportLayoutMismatch("kernel name pointer unreadable");
          continue;
        }

        DeviceEvent ev;
        ev.kind = EventKind::Kernel;
        ev.start_ns = kernel->start;
        ev.end_ns = kernel->end;
        ev.correlation_id = kernel->correlationId;
        ev.device = kernel->deviceId;
        ev.stream = kernel->streamId;
        // `name` is a const char* field in the record (a pointer into CUPTI's
        // own interned string table), valid only for the lifetime of the buffer
        // callback. Copy it into our std::string now.
        ev.name = demangleName(kernel->name);
        ev.metadata["grid"] = "[" + std::to_string(kernel->gridX) + "," +
                              std::to_string(kernel->gridY) + "," +
                              std::to_string(kernel->gridZ) + "]";
        ev.metadata["block"] = "[" + std::to_string(kernel->blockX) + "," +
                               std::to_string(kernel->blockY) + "," +
                               std::to_string(kernel->blockZ) + "]";
        ev.metadata["registers per thread"] =
            std::to_string(kernel->registersPerThread);
        ev.metadata["shared memory"] = std::to_string(
            static_cast<int64_t>(kernel->staticSharedMemory) +
            kernel->dynamicSharedMemory);
        ev.metadata["device"] = std::to_string(kernel->deviceId);
        ev.metadata["context"] = std::to_string(kernel->contextId);
        ev.metadata["stream"] = std::to_string(kernel->streamId);
        // The CUPTI correlation id, matching torch-cuda's "correlation" field.
        // NOT the torch/external id -- that one is emitted by kineto itself as
        // "External id" and drives device-time attribution.
        ev.metadata["correlation"] = std::to_string(kernel->correlationId);

        // `queued` comes from the part of the record layout past what the
        // original mirror covered, so it is validated separately. When CUPTI did
        // not capture it (the normal case -- we never enable latency timestamps,
        // and it then reads back as CUPTI_TIMESTAMP_UNKNOWN) we report 0, which
        // is exactly what torch-cuda's trace shows on this machine. Emitting the
        // raw sentinel would put 1.8e19 in the trace; omitting the key would
        // break arg-set parity with torch-cuda on every kernel.
        ev.metadata["queued"] =
            queuedTimestampPlausible(kernel->queued, kernel->start)
                ? std::to_string(kernel->queued)
                : "0";

        const int64_t grid_blocks = static_cast<int64_t>(kernel->gridX) *
                                    kernel->gridY * kernel->gridZ;
        const int64_t block_size = static_cast<int64_t>(kernel->blockX) *
                                   kernel->blockY * kernel->blockZ;
        const KernelOccupancy occ = computeKernelOccupancy(
            kernel->deviceId,
            grid_blocks,
            static_cast<int32_t>(block_size),
            kernel->registersPerThread,
            static_cast<int64_t>(kernel->staticSharedMemory) +
                kernel->dynamicSharedMemory);
        // Omit all three rather than emit zeros when the device attributes were
        // unavailable: a 0.0 occupancy is indistinguishable from a real
        // measurement of a badly-occupied kernel, whereas an absent key is
        // honest and a parity test will say so loudly.
        if (occ.valid) {
          ev.metadata["blocks per SM"] = formatOccupancyValue(occ.blocks_per_sm);
          ev.metadata["warps per SM"] = formatOccupancyValue(occ.warps_per_sm);
          ev.metadata["est. achieved occupancy %"] =
              std::to_string(occ.occupancy_pct);
        }
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
        auto* memcpy_rec = reinterpret_cast<CUpti_ActivityMemcpy_Compat*>(record);

        if (!timestampsCollected(memcpy_rec->start, memcpy_rec->end)) {
          FLAGOS_CUPTI_LOG("[flagos] skipping memcpy record with no timing\n");
          continue;
        }
        if (!timestampsPlausible(memcpy_rec->start, memcpy_rec->end)) {
          reportLayoutMismatch("memcpy timestamps implausible");
          continue;
        }

        DeviceEvent ev;
        ev.kind = EventKind::Memcpy;
        ev.start_ns = memcpy_rec->start;
        ev.end_ns = memcpy_rec->end;
        ev.correlation_id = memcpy_rec->correlationId;
        ev.device = memcpy_rec->deviceId;
        ev.stream = memcpy_rec->streamId;
        ev.name = "Memcpy";
        ev.metadata["bytes"] = std::to_string(memcpy_rec->bytes);
        ev.metadata["device"] = std::to_string(memcpy_rec->deviceId);
        ev.metadata["context"] = std::to_string(memcpy_rec->contextId);
        ev.metadata["stream"] = std::to_string(memcpy_rec->streamId);
        ev.metadata["correlation"] = std::to_string(memcpy_rec->correlationId);
        {
          std::string bandwidth;
          if (computeMemoryBandwidth(memcpy_rec->bytes, memcpy_rec->start,
                                     memcpy_rec->end, &bandwidth)) {
            ev.metadata["memory bandwidth (GB/s)"] = std::move(bandwidth);
          }
        }
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
        auto* memset_rec = reinterpret_cast<CUpti_ActivityMemset_Compat*>(record);

        if (!timestampsCollected(memset_rec->start, memset_rec->end)) {
          FLAGOS_CUPTI_LOG("[flagos] skipping memset record with no timing\n");
          continue;
        }
        if (!timestampsPlausible(memset_rec->start, memset_rec->end)) {
          reportLayoutMismatch("memset timestamps implausible");
          continue;
        }

        DeviceEvent ev;
        ev.kind = EventKind::Memset;
        ev.start_ns = memset_rec->start;
        ev.end_ns = memset_rec->end;
        ev.correlation_id = memset_rec->correlationId;
        ev.device = memset_rec->deviceId;
        ev.stream = memset_rec->streamId;
        ev.name = "Memset";
        ev.metadata["bytes"] = std::to_string(memset_rec->bytes);
        ev.metadata["device"] = std::to_string(memset_rec->deviceId);
        ev.metadata["context"] = std::to_string(memset_rec->contextId);
        ev.metadata["stream"] = std::to_string(memset_rec->streamId);
        ev.metadata["correlation"] = std::to_string(memset_rec->correlationId);
        {
          std::string bandwidth;
          if (computeMemoryBandwidth(memset_rec->bytes, memset_rec->start,
                                     memset_rec->end, &bandwidth)) {
            ev.metadata["memory bandwidth (GB/s)"] = std::move(bandwidth);
          }
        }
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
        // Runtime API calls (cudaLaunchKernel etc.) -- the correlation pivot that
        // lets kineto link a CPU op to the device kernel it launched.
        auto* rt = reinterpret_cast<CUpti_ActivityAPI_Compat*>(record);

        if (!timestampsCollected(rt->start, rt->end)) {
          FLAGOS_CUPTI_LOG("[flagos] skipping runtime record with no timing\n");
          continue;
        }
        if (!timestampsPlausible(rt->start, rt->end)) {
          reportLayoutMismatch("runtime timestamps implausible");
          continue;
        }

        DeviceEvent ev;
        ev.kind = EventKind::Runtime;
        ev.start_ns = rt->start;
        ev.end_ns = rt->end;
        ev.correlation_id = rt->correlationId;
        ev.thread_id = rt->threadId;
        // The name MUST come from the cbid. Hardcoding "cudaLaunchKernel" (as an
        // earlier revision did) mislabels every non-launch entry point -- most
        // visibly a 21ms blocking cudaStreamSynchronize showing up as a launch,
        // which reads as a pathological kernel-launch cost rather than a sync.
        //
        // MetaX upgrades this name in drain() using MCPTI's own resolver; the
        // static table below is the value that survives if that call is
        // unavailable. Resolving here would mean calling back into the profiler
        // library from inside its own bufferCompleted callback, which deadlocks.
#if defined(FLAGOS_METAX_MCPTI)
        ev.name = mcptiRuntimeCbidToName(rt->cbid);
#else
        ev.name = cuptiRuntimeCbidToName(rt->cbid);
#endif
        // torch-cuda's cuda_runtime args are exactly {External id, cbid,
        // correlation}; kineto supplies External id itself. The raw numeric cbid
        // is kept even though ev.name already carries its human spelling --
        // torch-cuda reports both, and the number is what identifies an entry
        // point our cbid table does not know.
        ev.metadata["cbid"] = std::to_string(rt->cbid);
        ev.metadata["correlation"] = std::to_string(rt->correlationId);
        events_.push_back(std::move(ev));

      } else if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
        // Not emitted as an event; recorded so drain() can translate CUPTI
        // correlation ids into torch correlation ids.
        auto* ext =
            reinterpret_cast<CUpti_ActivityExternalCorrelation_Compat*>(record);

        // This record has no timestamps to sanity-check, but it is the branch
        // whose silent misdecode is *least* visible: a wrong layout yields
        // garbage externalIds, getLinkedActivity then returns nullptr for
        // everything, and per-op device time silently reverts to 0 with no
        // diagnostic at all. externalKind is the one field we can verify --
        // pushCorrelation only ever pushes CUSTOM0, so anything else means we
        // are reading the wrong offsets.
        if (ext->externalKind != CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
          reportLayoutMismatch("external correlation kind unexpected");
          continue;
        }

        external_correlation_[ext->correlationId] = ext->externalId;
      }
    }
    if (record_count >= kMaxRecordsPerBuffer) {
      std::cerr << "[flagos] processBuffer exceeded max record count ("
                << kMaxRecordsPerBuffer << ") for buffer size " << validSize
                << " bytes; possible ActivityGetNextRecord infinite loop\n";
    }
  }

 private:
  // Returns the torch correlation id for a CUPTI correlation id, or nullopt when
  // no EXTERNAL_CORRELATION record mapped it (GPU work issued outside a profiled
  // RecordFunction: allocator traffic, warmup, pre-window kernels).
  //
  // This MUST NOT collapse "not found" onto 0: correlation id 0 is meaningful in
  // torch's scheme (autograd/profiler.py treats corr_id == 0 as a *frontend*
  // function event). Returning 0 for a miss would hand id 0 to getLinkedActivity
  // for every unmapped activity, and any kineto that resolves it would silently
  // attribute all of that device time to one wrong CPU op.
  std::optional<int32_t> lookupExternalCorrelation(
      uint32_t cupti_correlation_id) const {
    auto it = external_correlation_.find(cupti_correlation_id);
    if (it == external_correlation_.end()) {
      return std::nullopt;
    }
    return static_cast<int32_t>(it->second);
  }

  struct ClockSample {
    uint64_t vendor_ns = 0;
    uint64_t realtime_ns = 0;
    bool valid = false;
  };

  static ClockSample sampleClocks(CuptiShim& shim) {
    ClockSample sample;
    if (!shim.GetTimestamp) {
      return sample;
    }

    timespec before{};
    timespec after{};
    clock_gettime(CLOCK_REALTIME, &before);
    CUptiResult result = shim.GetTimestamp(&sample.vendor_ns);
    clock_gettime(CLOCK_REALTIME, &after);
    const uint64_t before_ns = static_cast<uint64_t>(before.tv_sec) * 1000000000ULL +
        static_cast<uint64_t>(before.tv_nsec);
    const uint64_t after_ns = static_cast<uint64_t>(after.tv_sec) * 1000000000ULL +
        static_cast<uint64_t>(after.tv_nsec);
    sample.realtime_ns = before_ns + (after_ns - before_ns) / 2;
    sample.valid = result == CUPTI_SUCCESS && sample.vendor_ns != 0;
    FLAGOS_CUPTI_LOG("[flagos] clock sample result=" << result
                     << " vendor=" << sample.vendor_ns
                     << " realtime=" << sample.realtime_ns << "\n");
    return sample;
  }

  uint64_t toRealtime(uint64_t vendor_ns) const {
    if (!start_clock_sample_.valid) {
      return vendor_ns;
    }
    if (!end_clock_sample_.valid ||
        end_clock_sample_.vendor_ns <= start_clock_sample_.vendor_ns) {
      return start_clock_sample_.realtime_ns +
          (vendor_ns - start_clock_sample_.vendor_ns);
    }

    const long double vendor_delta = static_cast<long double>(
        end_clock_sample_.vendor_ns - start_clock_sample_.vendor_ns);
    const long double realtime_delta = static_cast<long double>(
        end_clock_sample_.realtime_ns - start_clock_sample_.realtime_ns);
    const long double elapsed = static_cast<long double>(vendor_ns) -
        static_cast<long double>(start_clock_sample_.vendor_ns);
    return static_cast<uint64_t>(
        static_cast<long double>(start_clock_sample_.realtime_ns) +
        elapsed * realtime_delta / vendor_delta);
  }

  std::vector<DeviceEvent> events_;
  // cupti correlationId -> externalId (torch correlation id), populated from
  // EXTERNAL_CORRELATION records. Guarded by g_tracer_mutex (processBuffer
  // writes it; drain reads it after stop()).
  std::unordered_map<uint32_t, uint64_t> external_correlation_;
  ClockSample start_clock_sample_;
  ClockSample end_clock_sample_;
};

namespace {

void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  FLAGOS_CUPTI_LOG("[flagos] CUPTI bufferRequested callback invoked\n");
  *buffer = (uint8_t*)aligned_alloc(kBufferAlignment, kBufferSize);
  if (*buffer == nullptr) {
    // Under memory pressure an 8MB allocation can fail. Handing CUPTI a null
    // pointer while still claiming *size = 8MB invites it to write into address
    // 0; the documented way to decline a buffer is a null pointer with a zero
    // size. Degrade to dropped records (cuptiActivityGetNumDroppedRecords will
    // account for them) rather than crashing the profiled process.
    *size = 0;
    *maxNumRecords = 0;
    static std::once_flag warn_once;
    std::call_once(warn_once, [] {
      std::cerr << "[flagos] CUPTI activity buffer allocation failed ("
                << kBufferSize << " bytes); activity records will be dropped "
                   "while memory is short. The trace will be incomplete.\n";
    });
    return;
  }
  *size = kBufferSize;
  *maxNumRecords = 0;  // no limit
}

void bufferCompleted(
    CUcontext context,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  FLAGOS_CUPTI_LOG("[flagos] CUPTI bufferCompleted callback invoked, validSize=" << validSize << "\n");
  std::lock_guard<std::mutex> lock(g_tracer_mutex);
  if (!g_active_tracer || !buffer) {
    if (buffer) {
      free(buffer);
    }
    return;
  }

  auto& shim = CuptiShim::get();
  if (!shim.ok) {
    free(buffer);
    return;
  }

  g_active_tracer->processBuffer(buffer, validSize);
  free(buffer);
}

// Static initialization: arm CUPTI at module load.
struct CuptiTracerInit {
  CuptiTracerInit() {
    auto& shim = CuptiShim::get();
    if (!shim.ok) {
      FLAGOS_CUPTI_LOG("[flagos] CUPTI not available, tracer not armed\n");
      return;
    }

    // Register CUPTI callbacks ONCE at initialization - not per-session.
    // The empirically-verified working sequence (see memory:
    // cupti-must-arm-before-cuda-context) arms CUPTI by calling
    // RegisterCallbacks *and* ActivityEnable together at module-load time,
    // matching a manual ctypes probe that reliably captured kernels. Enabling
    // only in start() (long after libtorch_cuda has initialized its own CUPTI
    // state) captured nothing. We enable here and keep the enables in start()
    // as a harmless re-assertion.
    //
    // Device-side kinds only (KERNEL/MEMCPY/MEMSET). RUNTIME and
    // EXTERNAL_CORRELATION are armed in start() instead, because they are NOT
    // free for non-profiling users: RUNTIME instruments every CUDA runtime API
    // entry/exit on the CPU side, and arming it here cost a MEASURED ~+24% on a
    // launch-bound workload (10.6us -> 12.9us per op, 3 A/B pairs, far outside
    // the ~0.1ms run-to-run spread) for anyone who merely imports torch_fl. The
    // constraint the memory records is that *callback registration* precede the
    // first CUDA context -- which this still satisfies -- not that every
    // activity kind be armed at import.
    FLAGOS_CUPTI_LOG("[flagos] Registering CUPTI activity callbacks...\n");
    CUptiResult res =
        shim.ActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    FLAGOS_CUPTI_LOG("[flagos] ActivityRegisterCallbacks result: " << res << "\n");
    CUptiResult en_ck =
        shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    CUptiResult en_k = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    CUptiResult en_m = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    CUptiResult en_s = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    FLAGOS_CUPTI_LOG("[flagos] init ActivityEnable CONCURRENT_KERNEL=" << en_ck
              << " KERNEL=" << en_k << " MEMCPY=" << en_m
              << " MEMSET=" << en_s
              << " (RUNTIME/EXTERNAL_CORRELATION deferred to start())\n");
  }
};
static CuptiTracerInit g_tracer_init;

}  // namespace

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<CuptiDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10
