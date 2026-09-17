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

#pragma once

#include <dlfcn.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <flagos_env.h>

// Forward declarations to avoid including cupti headers directly.
// This keeps CUPTI include paths out of the main build and prevents
// header pollution. Function pointers are dlopen'd at runtime.

// Opaque CUDA types. MetaX's compatibility runtime uses MCcontext in its
// callback ABI; keep the callback typedef exact so -Werror builds do not need
// an unsafe function-pointer cast.
#if defined(FLAGOS_METAX_MCPTI)
typedef struct MCctx_st* CUcontext;
#else
typedef struct CUctx_st* CUcontext;
#endif

// CUPTI result type (matches cupti_result.h)
typedef enum {
  CUPTI_SUCCESS = 0,
  CUPTI_ERROR_INVALID_PARAMETER = 1,
  // PPU's CUPTI compatibility layer forwards HGPTI result values, whose
  // MAX_LIMIT_REACHED and INVALID_KIND differ from NVIDIA CUPTI. Avoid using
  // either vendor's numeric result here; ActivityGetNextRecord returns a null
  // record when iteration is complete, which is the portable termination signal.
  CUPTI_ERROR_NOT_INITIALIZED = 15
} CUptiResult;

// Activity kinds (subset). Values MUST match the cu12 runtime's
// cupti_activity.h (verified against nvidia-cuda-cupti-cu12): note
// CONCURRENT_KERNEL is 10 in cu12, not 9 as an earlier draft assumed. Using
// the wrong value silently enables/matches the wrong kind (0 kernels captured).
typedef enum {
  CUPTI_ACTIVITY_KIND_INVALID = 0,
  CUPTI_ACTIVITY_KIND_MEMCPY = 1,
  CUPTI_ACTIVITY_KIND_MEMSET = 2,
  CUPTI_ACTIVITY_KIND_KERNEL = 3,
  CUPTI_ACTIVITY_KIND_DRIVER = 4,
  CUPTI_ACTIVITY_KIND_RUNTIME = 5,
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,
  CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39
} CUpti_ActivityKind;

// External correlation kinds (minimal subset)
typedef enum {
  CUPTI_EXTERNAL_CORRELATION_KIND_INVALID = 0,
  CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN = 1,
  CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0 = 3
} CUpti_ExternalCorrelationKind;

// Runtime API callback id -> human name fallback for naming
// CUPTI_ACTIVITY_KIND_RUNTIME records. NVIDIA and MetaX use different callback
// id spaces, so this table is only used for the vendor selected at build time.
// MetaX normally resolves names through mcptiActivityGetApiName; the fallback
// below is retained for old MCPTI builds that do not export that helper.
//
// Getting one of these wrong is invisible at runtime: a plausible but wrong
// label is emitted rather than an error. Keep the static table deliberately
// small and use a generic label for ids that are not known.
inline const char* cuptiRuntimeCbidToName(uint32_t cbid) {
  switch (cbid) {
    // --- launches ---
    case 211: return "cudaLaunchKernel";                 // _v7000
    case 214: return "cudaLaunchKernel";                 // _ptsz_v7000
    case 269: return "cudaLaunchCooperativeKernel";      // _v9000
    case 270: return "cudaLaunchCooperativeKernel";      // _ptsz_v9000
    case 430: return "cudaLaunchKernelExC";              // _v11060
    case 431: return "cudaLaunchKernelExC";              // _ptsz_v11060
    // --- transfers ---
    case 31:  return "cudaMemcpy";                       // _v3020
    case 41:  return "cudaMemcpyAsync";                  // _v3020
    case 225: return "cudaMemcpyAsync";                  // _ptsz_v7000
    case 49:  return "cudaMemset";                       // _v3020
    case 51:  return "cudaMemsetAsync";                  // _v3020
    case 235: return "cudaMemsetAsync";                  // _ptsz_v7000
    // --- synchronization (the entries most often misread as launches) ---
    case 131: return "cudaStreamSynchronize";            // _v3020
    case 239: return "cudaStreamSynchronize";            // _ptsz_v7000
    case 165: return "cudaDeviceSynchronize";            // _v3020
    case 137: return "cudaEventSynchronize";             // _v3020
    case 135: return "cudaEventRecord";                  // _v3020
    case 242: return "cudaEventRecord";                  // _ptsz_v7000
    case 147: return "cudaStreamWaitEvent";              // _v3020
    case 247: return "cudaStreamWaitEvent";              // _ptsz_v7000
    // --- allocation ---
    case 20:  return "cudaMalloc";                       // _v3020
    case 22:  return "cudaFree";                         // _v3020
    // Unmapped ids keep a generic label rather than a guessed one: the record is
    // still a real runtime call worth showing, and mislabeling it is worse than
    // leaving it unnamed.
    default:  return "cudaRuntime";
  }
}

#if defined(FLAGOS_METAX_MCPTI)
inline const char* mcptiRuntimeCbidToName(uint32_t cbid) {
  switch (cbid) {
    // --- device and synchronization ---
    case 18: return "mcDeviceSynchronize";
    case 19: return "mcGetDevice";
    case 20: return "mcGetDeviceCount";
    case 22: return "mcGetDeviceProperties";
    case 23: return "mcSetDevice";
    case 46: return "mcStreamSynchronize";
    case 47: return "mcStreamWaitEvent";
    // --- launches ---
    case 56: return "mcLaunchKernel";
    case 60: return "mcModuleLaunchKernel";
    case 62: return "mcLaunchKernelExC";
    // --- events ---
    case 74: return "mcEventRecord";
    case 76: return "mcEventSynchronize";
    // --- allocation ---
    case 107: return "mcMalloc";
    case 108: return "mcFree";
    // --- transfers ---
    case 118: return "mcMemcpy";
    case 119: return "mcMemcpyAsync";
    case 146: return "mcMemset";
    case 147: return "mcMemsetAsync";
    default: return "mcRuntime";
  }
}
#endif

// Opaque activity record base
struct CUpti_Activity;

// Callback function types (match cupti_activity.h signatures exactly)
typedef void (*CUpti_BuffersCallbackRequestFunc)(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords);

typedef void (*CUpti_BuffersCallbackCompleteFunc)(
    CUcontext context,
    uint32_t streamId,
    uint8_t* buffer,
    size_t size,
    size_t validSize);

namespace c10 {
namespace flagos {

/**
 * CUPTI Activity API dlopen shim.
 * Dynamically loads CUPTI function pointers at runtime to avoid linking
 * against libcupti.so at build time. This allows the CPU-only build to
 * remain clean while still supporting CUPTI profiling when the runtime
 * environment has CUDA available.
 */
struct CuptiShim {
  bool ok = false;

  // CUPTI Activity API function pointers
  CUptiResult (*ActivityEnable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityDisable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityRegisterCallbacks)(
      CUpti_BuffersCallbackRequestFunc,
      CUpti_BuffersCallbackCompleteFunc) = nullptr;
  CUptiResult (*ActivityFlushAll)(uint32_t) = nullptr;
  CUptiResult (*ActivityGetNextRecord)(
      uint8_t*, size_t, CUpti_Activity**) = nullptr;
  CUptiResult (*ActivityGetNumDroppedRecords)(
      CUcontext, uint32_t, size_t*) = nullptr;
  CUptiResult (*ActivityPushExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t) = nullptr;
  CUptiResult (*ActivityPopExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t*) = nullptr;
#if defined(FLAGOS_METAX_MCPTI)
  // MCPTI's native resolver avoids applying NVIDIA cbid meanings to MetaX
  // runtime records. It is optional for compatibility with older SDKs.
  CUptiResult (*ActivityGetApiName)(
      CUpti_ActivityKind, uint32_t, const char**) = nullptr;
#endif
  CUptiResult (*GetTimestamp)(uint64_t*) = nullptr;
  // Optional: only used for diagnostics, so a failure to resolve it is not fatal.
  CUptiResult (*GetVersion)(uint32_t*) = nullptr;

  // CUPTI API version of the bound library (0 when unknown). Callers use this to
  // report *which* CUPTI they are talking to, so a record-layout mismatch is
  // attributable instead of showing up as an unexplained empty trace.
  uint32_t api_version = 0;
  // Path of the library the symbols actually came from ("" when unknown).
  const char* library_path = "";

  static CuptiShim& get() {
    static CuptiShim inst;
    return inst;
  }

  bool available() const { return ok; }

 private:
  CuptiShim() {
    // CRITICAL (see memory: cupti-must-arm-before-cuda-context): bind the
    // profiler library that belongs to the runtime actually running in this
    // process. If a different major-version library is selected, activity
    // callbacks may register successfully while capturing no records.
    // Prefer the copy already loaded in this process because it is the one the
    // runtime selected.
    // Escape hatch, checked FIRST so that an explicit path always wins. It has
    // to outrank the already-loaded copy below, because the situation that
    // motivates setting it -- a preloaded CUPTI whose record layout we cannot
    // decode -- is exactly the situation where a preloaded copy exists. An
    // override that only applied when nothing was loaded would be dead in the
    // one case it is advertised for (see reportLayoutMismatch's diagnostic).
    void* handle = nullptr;
    const std::string override_path = flagos_env::EnvValue("FLAGOS_TRACER_LIBRARY");
    if (!override_path.empty()) {
      handle = dlopen(override_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
      if (!handle) {
        fprintf(stderr,
                "[flagos-cupti-shim] FLAGOS_TRACER_LIBRARY=%s could not be "
                "loaded: %s\n",
                override_path.c_str(), dlerror());
      }
    }

    if (!handle) {
      handle = dlopen(nullptr, RTLD_LAZY | RTLD_GLOBAL);
      if (handle && !dlsym(handle, "cuptiActivityRegisterCallbacks")) {
        // Nothing CUPTI-shaped in the already-loaded set; fall through to
        // explicit dlopen of a versioned library.
        handle = nullptr;
      }
    }

    if (!handle) {
#if defined(FLAGOS_METAX_MCPTI)
      // MetaX's compatibility library has a different soname from NVIDIA's
      // CUPTI. Keep this branch build-specific so a CUDA build never binds an
      // unrelated MCPTI installation.
      const char* candidates[] = {
          "libmcpti.so",
      };
#else
      // Nothing preloaded: try sonames from newest to oldest, then the
      // unversioned name (a devel symlink). This list is a fallback ordering,
      // not a supported-version list.
      const char* candidates[] = {
          "libcupti.so.13",
          "libcupti.so.12",
          "libcupti.so",
      };
#endif
      for (const char* name : candidates) {
        if (handle) {
          break;
        }
        handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      }
    }

    if (!handle) {
      return; // CUPTI not available, ok remains false
    }

    // Load function pointers using dlsym
#define LOAD_SYM(field, sym) \
    field = reinterpret_cast<decltype(field)>(dlsym(handle, sym))

    LOAD_SYM(ActivityEnable, "cuptiActivityEnable");
    LOAD_SYM(ActivityDisable, "cuptiActivityDisable");
    LOAD_SYM(ActivityRegisterCallbacks, "cuptiActivityRegisterCallbacks");
    LOAD_SYM(ActivityFlushAll, "cuptiActivityFlushAll");
    LOAD_SYM(ActivityGetNextRecord, "cuptiActivityGetNextRecord");
    LOAD_SYM(ActivityGetNumDroppedRecords, "cuptiActivityGetNumDroppedRecords");
    LOAD_SYM(ActivityPushExternalCorrelationId,
             "cuptiActivityPushExternalCorrelationId");
    LOAD_SYM(ActivityPopExternalCorrelationId,
             "cuptiActivityPopExternalCorrelationId");
#if defined(FLAGOS_METAX_MCPTI)
    LOAD_SYM(ActivityGetApiName, "mcptiActivityGetApiName");
#endif
    LOAD_SYM(GetTimestamp, "cuptiGetTimestamp");
    LOAD_SYM(GetVersion, "cuptiGetVersion");

#undef LOAD_SYM

    // Record which library and which CUPTI API version we ended up bound to.
    // Both are diagnostics only -- nothing branches on the version, since the
    // whole point is to not hardcode version knowledge.
    if (GetVersion) {
      uint32_t v = 0;
      if (GetVersion(&v) == CUPTI_SUCCESS) {
        api_version = v;
      }
    }
    if (ActivityRegisterCallbacks) {
      Dl_info info;
      if (dladdr(reinterpret_cast<void*>(ActivityRegisterCallbacks), &info) &&
          info.dli_fname) {
        library_path = info.dli_fname;
      }
    }

    if (flagos_env::EnvFlag("FLAGOS_TRACE")) {
      if (ActivityRegisterCallbacks) {
        fprintf(stderr,
                "[flagos-cupti-shim] bound cuptiActivityRegisterCallbacks -> %s "
                "(CUPTI API version %u)\n",
                library_path[0] ? library_path : "<unknown>", api_version);
      } else {
        fprintf(stderr, "[flagos-cupti-shim] cuptiActivityRegisterCallbacks NOT resolved\n");
      }
    }

    // Mark as available if critical functions loaded successfully
    ok = ActivityEnable && ActivityRegisterCallbacks &&
         ActivityFlushAll && ActivityGetNextRecord;
  }
};

}  // namespace flagos
}  // namespace c10
