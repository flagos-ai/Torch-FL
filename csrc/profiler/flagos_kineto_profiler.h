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

#include "device_tracer.h"

#include <kineto/IActivityProfiler.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace c10 {
namespace flagos {

/**
 * FlagosKinetoProfilerSession: IActivityProfilerSession implementation that
 * collects a device timeline through the vendor-neutral DeviceTracer interface.
 *
 * This class is the kineto adaptor: it knows about kineto activity types,
 * flow arrows and correlation linking, and nothing at all about vendor-specific
 * tracing APIs. All vendor specifics live behind DeviceTracer (see
 * cupti_device_tracer.cc for NVIDIA/CUDA, future *_device_tracer.cc for others).
 */
class FlagosKinetoProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  FlagosKinetoProfilerSession();

  void start() override;
  void stop() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  // 4-arg overload: kineto hands us a callback that resolves a *torch*
  // correlation id to the CPU op activity, plus the capture window. This is the
  // only hook through which a plugin session can set `linked` / flow ids, i.e.
  // the only way to get ac2g flow arrows and per-op device-time attribution.
  void processTrace(
      libkineto::ActivityLogger& logger,
      libkineto::getLinkedActivityCallback getLinkedActivity,
      int64_t startTime,
      int64_t endTime) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;
  std::vector<std::string> errors() override { return {}; }
  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;

 private:
  friend class FlagosKinetoProfiler;
  std::unique_ptr<profiler::DeviceTracer> tracer_;
  // Drained from the tracer in stop(); converted to kineto activities in
  // processTrace so both overloads see the same data.
  std::vector<profiler::DeviceEvent> events_;
};

/**
 * FlagosKinetoProfiler: IActivityProfiler implementation for flagos backend
 * that provides CONCURRENT_KERNEL, GPU_MEMCPY, GPU_MEMSET, and PRIVATEUSE1_RUNTIME
 * activities via a vendor-agnostic DeviceTracer interface.
 */
class FlagosKinetoProfiler : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;

  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t profileStartTime,
      int64_t profileDuration,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
};

/**
 * registerFlagosKinetoProfiler: Register the kineto profiler with libkineto.
 * Called automatically at static initialization time if a DeviceTracer is available.
 */
void registerFlagosKinetoProfiler();

}  // namespace flagos
}  // namespace c10

// C API for testing correlation push/pop calls
extern "C" {
__attribute__((visibility("default"))) uint64_t flagos_kineto_get_correlation_push_count();
__attribute__((visibility("default"))) uint64_t flagos_kineto_get_correlation_pop_count();
__attribute__((visibility("default"))) void flagos_kineto_reset_correlation_counters();
}
