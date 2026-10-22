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
// DeviceTracer for platforms with no supported device activity API (Ascend,
// GCU, MUSA, BPU, TsingMicro). Every vendor tracer defines the same
// MakeDeviceTracer() factory, so exactly one of them is compiled per build (see
// the selection in csrc/CMakeLists.txt); without this file those platforms
// would either fail to link or, worse, bind a tracer whose activity API their
// runtime does not implement.
//
// available() returning false is what the layer above keys on: the kineto
// adaptor then skips registration entirely rather than exporting a trace with
// an empty device timeline, which is far easier to interpret as "no device
// profiling on this platform" than a silently kernel-less trace.

#include "device_tracer.h"

#include <memory>
#include <vector>

namespace c10 {
namespace flagos {
namespace profiler {

namespace {

class UnavailableDeviceTracer : public DeviceTracer {
 public:
  bool available() const override { return false; }
  void start() override {}
  void stop() override {}
  std::vector<DeviceEvent> drain() override { return {}; }
  int deviceCount() const override { return 0; }
};

}  // namespace

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<UnavailableDeviceTracer>();
}

}  // namespace profiler
}  // namespace flagos
}  // namespace c10
