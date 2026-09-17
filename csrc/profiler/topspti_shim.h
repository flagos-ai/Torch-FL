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

#if defined(FLAGOS_HAVE_TOPSPTI)
#include <topspti_activity.h>
#include <topspti_callbacks.h>
#else
using TopsptiResult = int;
using Topspti_CallbackId = uint32_t;
using Topspti_ActivityKind = int;
using Topspti_CallbackDomain = int;
using Topspti_SubscriberHandle = void*;
using Topspti_BuffersCallbackRequestFunc = void (*)(
    uint8_t**, size_t*, size_t*);
using Topspti_BuffersCallbackCompleteFunc = void (*)(
    uint8_t*, size_t, size_t);
using Topspti_CallbackFunc = void (*)(
    void*, Topspti_CallbackDomain, Topspti_CallbackId, const void*);
struct Topspti_Activity_st {
  Topspti_ActivityKind kind;
};
using Topspti_Activity = Topspti_Activity_st;
#endif

namespace c10::flagos::profiler {

struct TopsptiShim {
  using ActivityRegisterCallbacks = TopsptiResult (*) (
      Topspti_BuffersCallbackRequestFunc,
      Topspti_BuffersCallbackCompleteFunc);
  using ActivityEnable = TopsptiResult (*)(Topspti_ActivityKind);
  using ActivityDisable = TopsptiResult (*)(Topspti_ActivityKind);
  using ActivityFlushAll = TopsptiResult (*)(uint32_t);
  using ActivityGetNextRecord = TopsptiResult (*) (
      uint8_t*, size_t, Topspti_Activity**);
  using GetTimestamp = TopsptiResult (*)(uint64_t*);
  using Subscribe = TopsptiResult (*) (
      Topspti_SubscriberHandle*, Topspti_CallbackFunc, void*);
  using Unsubscribe = TopsptiResult (*)(Topspti_SubscriberHandle);
  using EnableAllDomains = TopsptiResult (*) (
      uint32_t, Topspti_SubscriberHandle);

  bool loaded = false;
  bool ok = false;
  void* handle = nullptr;
  ActivityRegisterCallbacks activity_register_callbacks = nullptr;
  ActivityEnable activity_enable = nullptr;
  ActivityDisable activity_disable = nullptr;
  ActivityFlushAll activity_flush_all = nullptr;
  ActivityGetNextRecord activity_get_next_record = nullptr;
  GetTimestamp get_timestamp = nullptr;
  Subscribe subscribe = nullptr;
  Unsubscribe unsubscribe = nullptr;
  EnableAllDomains enable_all_domains = nullptr;

  bool available() const {
#if defined(FLAGOS_HAVE_TOPSPTI)
    return ok;
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
        "libtopspti.so",
        "/opt/tops/extras/TOPSPTI/lib64/libtopspti.so",
        "/opt/tops/lib/libtopspti_prof.so",
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

#define FLAGOS_LOAD_TOPSPTI(field, symbol) \
  field = reinterpret_cast<decltype(field)>(dlsym(handle, symbol))
    FLAGOS_LOAD_TOPSPTI(
        activity_register_callbacks, "topsptiActivityRegisterCallbacks");
    FLAGOS_LOAD_TOPSPTI(activity_enable, "topsptiActivityEnable");
    FLAGOS_LOAD_TOPSPTI(activity_disable, "topsptiActivityDisable");
    FLAGOS_LOAD_TOPSPTI(activity_flush_all, "topsptiActivityFlushAll");
    FLAGOS_LOAD_TOPSPTI(
        activity_get_next_record, "topsptiActivityGetNextRecord");
    FLAGOS_LOAD_TOPSPTI(get_timestamp, "topsptiGetTimestamp");
    FLAGOS_LOAD_TOPSPTI(subscribe, "topsptiSubscribe");
    FLAGOS_LOAD_TOPSPTI(unsubscribe, "topsptiUnsubscribe");
    FLAGOS_LOAD_TOPSPTI(enable_all_domains, "topsptiEnableAllDomains");
#undef FLAGOS_LOAD_TOPSPTI

    ok = activity_register_callbacks && activity_enable && activity_disable &&
        activity_flush_all && activity_get_next_record && get_timestamp;
    return ok;
  }

  static TopsptiShim& get() {
    static TopsptiShim shim;
    return shim;
  }
};

}  // namespace c10::flagos::profiler
