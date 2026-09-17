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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <flagos_env.h>

#include <musa.h>

// This shim intentionally includes only the public MUPTI activity declarations.
// MUPTI is optional at runtime, so the shared library is never linked directly.
#if defined(FLAGOS_HAVE_MUPTI)
#include <mupti_activity.h>
#else
using MUptiResult = int;
using MUpti_ActivityKind = int;
using MUpti_ExternalCorrelationKind = int;
using MUpti_CallbackDomain = int;
using MUpti_Activity = struct MUpti_Activity;
using MUcontext = struct MUctx_st*;
using MUpti_BuffersCallbackRequestFunc = void (*)(
    uint8_t**, size_t*, size_t*);
using MUpti_BuffersCallbackCompleteFunc = void (*)(
    MUcontext, uint32_t, uint8_t*, size_t, size_t);
#endif

namespace c10::flagos::profiler {

struct MuptiShim {
  bool ok = false;
  bool loaded = false;
  void* handle = nullptr;

  MUptiResult (*ActivityEnable)(MUpti_ActivityKind) = nullptr;
  MUptiResult (*ActivityDisable)(MUpti_ActivityKind) = nullptr;
  MUptiResult (*ActivityRegisterCallbacks)(
      MUpti_BuffersCallbackRequestFunc,
      MUpti_BuffersCallbackCompleteFunc) = nullptr;
  MUptiResult (*ActivityFlushAll)(uint32_t) = nullptr;
  MUptiResult (*ActivityGetNextRecord)(
      uint8_t*, size_t, MUpti_Activity**) = nullptr;
  MUptiResult (*ActivityGetNumDroppedRecords)(
      MUcontext, uint32_t, size_t*) = nullptr;
  MUptiResult (*ActivityPushExternalCorrelationId)(
      MUpti_ExternalCorrelationKind, uint64_t) = nullptr;
  MUptiResult (*ActivityPopExternalCorrelationId)(
      MUpti_ExternalCorrelationKind, uint64_t*) = nullptr;
  MUptiResult (*GetTimestamp)(uint64_t*) = nullptr;
  MUptiResult (*GetCallbackName)(
      MUpti_CallbackDomain, uint32_t, const char**) = nullptr;

  static MuptiShim& get() {
    static MuptiShim instance;
    return instance;
  }

  bool available() const {
#if defined(FLAGOS_HAVE_MUPTI)
    return true;
#else
    return false;
#endif
  }

  bool load() {
    if (loaded) {
      return ok;
    }
    loaded = true;
    // An explicitly set but empty FLAGOS_TRACER_LIBRARY means "unset", so the
    // built-in names below still get their turn.
    const std::string override_library = flagos_env::EnvValue("FLAGOS_TRACER_LIBRARY");
    const char* candidates[] = {
        override_library.c_str(),
        "libmupti.so",
        "libmupti.so.1.2",
        "/usr/local/musa/lib/libmupti.so",
        "/usr/local/musa-5.1.0/lib/libmupti.so",
    };

    for (const char* candidate : candidates) {
      if (!candidate || !*candidate) {
        continue;
      }
      handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
      if (handle) {
        break;
      }
    }
    if (!handle) {
      return false;
    }

#define FLAGOS_LOAD_MUPTI(field, symbol) \
    field = reinterpret_cast<decltype(field)>(dlsym(handle, symbol))
    FLAGOS_LOAD_MUPTI(ActivityEnable, "muptiActivityEnable");
    FLAGOS_LOAD_MUPTI(ActivityDisable, "muptiActivityDisable");
    FLAGOS_LOAD_MUPTI(ActivityRegisterCallbacks, "muptiActivityRegisterCallbacks");
    FLAGOS_LOAD_MUPTI(ActivityFlushAll, "muptiActivityFlushAll");
    FLAGOS_LOAD_MUPTI(ActivityGetNextRecord, "muptiActivityGetNextRecord");
    FLAGOS_LOAD_MUPTI(
        ActivityGetNumDroppedRecords, "muptiActivityGetNumDroppedRecords");
    FLAGOS_LOAD_MUPTI(
        ActivityPushExternalCorrelationId,
        "muptiActivityPushExternalCorrelationId");
    FLAGOS_LOAD_MUPTI(
        ActivityPopExternalCorrelationId,
        "muptiActivityPopExternalCorrelationId");
    FLAGOS_LOAD_MUPTI(GetTimestamp, "muptiGetTimestamp");
    FLAGOS_LOAD_MUPTI(GetCallbackName, "muptiGetCallbackName");
#undef FLAGOS_LOAD_MUPTI

    ok = ActivityEnable && ActivityDisable && ActivityRegisterCallbacks &&
        ActivityFlushAll && ActivityGetNextRecord &&
        ActivityPushExternalCorrelationId && ActivityPopExternalCorrelationId;
    return ok;
  }
};

}  // namespace c10::flagos::profiler
