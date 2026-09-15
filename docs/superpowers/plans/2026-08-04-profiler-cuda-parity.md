# Profiler CUDA Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `torch.profiler` on flagos devices produce traces structurally identical to torch-cuda — flow arrows connecting ops to kernels, runtime events, full kernel metadata, and automatic device time attribution.

**Architecture:** Split profiler into vendor-agnostic interface (`DeviceTracer`) + NVIDIA CUPTI implementation + generic kineto adaptor. Add RUNTIME activity collection (the missing correlation link), implement 4-arg `processTrace` to wire `linked`/`flow` pointers, fill 13 kernel metadata fields, demangle names.

**Tech Stack:** CUPTI Activity API (dlopen shim), libkineto `IActivityProfiler`, `abi::__cxa_demangle`, pytest with CI integration

## Global Constraints

- Worktree: `profiler-support`, branch `worktree-profiler-support`
- Build env: `torch-fl-211` conda env (2.11.0+cpu + external libtorch_cuda cu12.8)
- Build: set `CUDA_HOME` to the local CUDA toolkit, then run `cmake --build build --target install`
- Test: `PYTHONPATH=$(pwd) bash scripts/vendor/with_cuda_libtorch.sh python -m pytest ...`
- Git: `git -c user.name=lvyufeng -c user.email=lvyufeng@cqu.edu.cn commit`
- Network: load the locally configured proxy, if needed, before GitHub operations
- CMake: uses `GLOB_RECURSE` — new files auto-included, deleted files need `rm -rf build; cmake -B build`

---

### Task 1: Verification Gate — Prove the Design Foundation

**Files:**
- Create: `tests/scratch/verify_correlation_foundation.py`

**Interfaces:**
- Consumes: existing Stage B CUPTI kernel capture (flagos_cupti_profiler.cc)
- Produces: proof that `PRIVATEUSE1_RUNTIME` + 4-arg `processTrace` produce the 3 target metrics

**Context:** The spec's §3.2 flags "device time attribution via `linked` pointer is free" as UNVERIFIED theory derived from code reading. This task proves or disproves it before Task 2+ build on that foundation. If any metric fails, we switch `PRIVATEUSE1_RUNTIME` → `CUDA_RUNTIME` and re-verify.

- [ ] **Step 1: Write verification script scaffold**

```python
# tests/scratch/verify_correlation_foundation.py
"""
Verification gate for profiler parity design foundation.
Tests 3 claims:
1. Chrome trace contains flow arrows (ac2g category, count > 0)
2. key_averages() shows aten::mm with self_device_time_total > 0
3. Runtime events appear with correct category name
"""
import torch
import torch_fl
import json
import tempfile
from pathlib import Path

def run_traced_ops():
    """5x matmul+relu, same workload as spec's A/B comparison."""
    x = torch.randn(1024, 1024, device='flagos:0')
    y = torch.randn(1024, 1024, device='flagos:0')
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        with_stack=False,
    ) as prof:
        for _ in range(5):
            z = (x @ y).relu()
        z.sum().item()  # force sync
    
    return prof

def verify_flow_arrows(trace_path):
    """Metric 1: ac2g flow count > 0."""
    with open(trace_path) as f:
        trace = json.load(f)
    
    flows = [e for e in trace.get('flowEvents', []) 
             if e.get('cat') == 'ac2g']
    
    print(f"Flow arrows (ac2g): {len(flows)}")
    assert len(flows) > 0, "FAIL: no flow arrows found"
    return len(flows)

def verify_device_time_attribution(prof):
    """Metric 2: aten::mm self_device_time_total > 0."""
    key_avg = prof.key_averages()
    mm_events = [e for e in key_avg if 'mm' in e.key.lower()]
    
    for evt in mm_events:
        print(f"{evt.key}: self_device_time_total={evt.self_device_time_total}µs")
        if 'aten::mm' in evt.key:
            assert evt.self_device_time_total > 0, \
                f"FAIL: aten::mm device time is {evt.self_device_time_total}"
            return evt.self_device_time_total
    
    raise AssertionError("FAIL: no aten::mm event found")

def verify_runtime_category(trace_path, expected_cats):
    """Metric 3: runtime events present with acceptable category."""
    with open(trace_path) as f:
        trace = json.load(f)
    
    runtime_events = [
        e for e in trace.get('traceEvents', [])
        if e.get('cat') in expected_cats
    ]
    
    if not runtime_events:
        cats = set(e.get('cat') for e in trace.get('traceEvents', []))
        raise AssertionError(
            f"FAIL: no runtime events in {expected_cats}. "
            f"Available categories: {sorted(cats)}"
        )
    
    actual_cat = runtime_events[0]['cat']
    print(f"Runtime category: {actual_cat} ({len(runtime_events)} events)")
    return actual_cat

if __name__ == '__main__':
    print("=== Profiler Foundation Verification Gate ===")
    prof = run_traced_ops()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        trace_path = f.name
    prof.export_chrome_trace(trace_path)
    
    try:
        flow_count = verify_flow_arrows(trace_path)
        device_time = verify_device_time_attribution(prof)
        runtime_cat = verify_runtime_category(
            trace_path,
            expected_cats={'privateuse1_runtime', 'cuda_runtime'}
        )
        
        print("\n✓ ALL METRICS PASSED")
        print(f"  - Flow arrows: {flow_count}")
        print(f"  - aten::mm device time: {device_time}µs")
        print(f"  - Runtime category: {runtime_cat}")
    finally:
        Path(trace_path).unlink(missing_ok=True)
```

- [ ] **Step 2: Enable RUNTIME activity collection in existing profiler**

Modify `csrc/profiler/flagos_cupti_profiler.cc`:

```cpp
// In registerFlagosCuptiProfiler() static initializer, after line 465:
CUptiResult en_k = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
CUptiResult en_m = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
CUptiResult en_r = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);  // NEW
FLAGOS_CUPTI_LOG("[flagos] init ActivityEnable KERNEL=" << en_k
                  << " MEMCPY=" << en_m << " RUNTIME=" << en_r << "\n");
```

And in `FlagosCuptiProfilerSession::start()` after line 326:

```cpp
CUptiResult res1 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
CUptiResult res2 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
CUptiResult res3 = shim.ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);  // NEW
FLAGOS_CUPTI_LOG("[flagos] ActivityEnable results: KERNEL=" << res1 
                  << ", MEMCPY=" << res2 << ", RUNTIME=" << res3 << "\n");
```

- [ ] **Step 3: Add RUNTIME record parsing to bufferCompleted**

In `flagos_cupti_profiler.cc` `bufferCompleted` function, after the memcpy parsing block (line ~293):

```cpp
} else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
  // Runtime API calls like cudaLaunchKernel — the correlation pivot.
  // Layout: see cupti_activity.h CUpti_ActivityAPI
  struct CUpti_ActivityRuntime_Compat {
    CUpti_ActivityKind_t kind;
    uint32_t cbid;
    uint64_t start;
    uint64_t end;
    uint32_t processId;
    uint32_t threadId;
    uint32_t correlationId;
  } __attribute__((packed));
  
  auto* rt = reinterpret_cast<CUpti_ActivityRuntime_Compat*>(record);
  
  if (!timestampsPlausible(rt->start, rt->end)) {
    reportLayoutMismatch("runtime timestamps implausible");
    continue;
  }
  
  libkineto::GenericTraceActivity activity;
  activity.activityType = libkineto::ActivityType::PRIVATEUSE1_RUNTIME;
  activity.activityName = "cudaLaunchKernel";  // cbid→name mapping in Task 3
  activity.startTime = rt->start;
  activity.endTime = rt->end;
  activity.device = 0;  // CPU-side event
  activity.resource = rt->threadId;
  activity.id = rt->correlationId;
  
  g_active_session->activities_.push_back(std::move(activity));
}
```

- [ ] **Step 4: Rebuild and run verification**

```bash
export CUDA_HOME="${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit root}"
export PATH=$CUDA_HOME/bin:$PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH

conda activate torch-fl-211
cmake --build build --target install

PYTHONPATH=$(pwd) bash scripts/vendor/with_cuda_libtorch.sh \
  python tests/scratch/verify_correlation_foundation.py
```

Expected output:
```
Flow arrows (ac2g): [some number > 0]
aten::mm: self_device_time_total=[X]µs
Runtime category: privateuse1_runtime
✓ ALL METRICS PASSED
```

- [ ] **Step 5: Handle fallback to CUDA_RUNTIME if needed**

If Step 4 output shows:
- Flow count = 0, OR
- aten::mm device_time_total = 0, OR  
- AssertionError "no runtime events in {...}"

Then change line in Step 3:
```cpp
activity.activityType = libkineto::ActivityType::CUDA_RUNTIME;  // fallback
```

Rebuild, re-run Step 4. Update this step's checkbox only when verification passes.

- [ ] **Step 6: Commit verification proof**

```bash
git add tests/scratch/verify_correlation_foundation.py \
        csrc/profiler/flagos_cupti_profiler.cc
git -c user.name=lvyufeng -c user.email=lvyufeng@cqu.edu.cn commit -m \
  "feat(profiler): verify correlation foundation — RUNTIME activity + flow/device-time metrics

Proves 3 design claims before refactoring:
1. Flow arrows (ac2g) appear in chrome trace
2. key_averages aten::mm gains self_device_time_total
3. Runtime events materialize with correct category

ActivityType: [PRIVATEUSE1_RUNTIME or CUDA_RUNTIME, fill actual]
Metrics: [X] flows, [Y]µs device time on aten::mm

Refs #spec 2026-08-03-profiler-cuda-parity-design.md §3.2"
```

---

### Task 2: Extract Vendor-Agnostic Tracer Interface

**Files:**
- Create: `csrc/profiler/device_tracer.h`
- Modify: `csrc/profiler/flagos_cupti_profiler.cc` (temp coupling, decoupled in Task 3)

**Interfaces:**
- Consumes: nothing (pure interface)
- Produces: `DeviceTracer` abstract class, `EventKind` enum, `DeviceEvent` struct with `std::map<string,string> metadata`

- [ ] **Step 1: Write device_tracer.h interface**

```cpp
// csrc/profiler/device_tracer.h
// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// ...same header as other files...

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace c10 {
namespace flagos {
namespace profiler {

enum class EventKind {
  Kernel,
  Memcpy,
  Memset,
  Runtime,
};

struct DeviceEvent {
  EventKind kind;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint32_t correlation_id = 0;  // Links runtime↔kernel
  uint32_t device = 0;
  uint32_t stream = 0;
  uint32_t thread_id = 0;  // Runtime events run on CPU thread
  std::string name;  // Already demangled
  std::map<std::string, std::string> metadata;  // "grid"→"[8,16,5]", etc.
};

class DeviceTracer {
 public:
  virtual ~DeviceTracer() = default;
  
  virtual bool available() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::vector<DeviceEvent> drain() = 0;
  virtual void pushCorrelation(uint64_t id) {}
  virtual void popCorrelation() {}
  virtual int deviceCount() const = 0;
};

// Factory: returns CUPTI tracer on NVIDIA, CANN tracer on Ascend, etc.
// Vendor selection is compile-time (based on ACCELERATOR cmake var).
std::unique_ptr<DeviceTracer> MakeDeviceTracer();

}  // namespace profiler
}  // namespace flagos
}  // namespace c10
```

- [ ] **Step 2: Add temporary stub factory at end of flagos_cupti_profiler.cc**

After the static `CuptiProfilerRegistrar` block (line ~478):

```cpp
// Temporary stub for device_tracer.h factory — moved to cupti_device_tracer.cc in Task 3
namespace c10::flagos::profiler {
class StubCuptiTracer : public DeviceTracer {
 public:
  bool available() const override { return false; }
  void start() override {}
  void stop() override {}
  std::vector<DeviceEvent> drain() override { return {}; }
  int deviceCount() const override { return 0; }
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<StubCuptiTracer>();
}
}  // namespace c10::flagos::profiler
```

- [ ] **Step 3: Add device_tracer.h include to flagos_cupti_profiler.cc**

After existing includes at top of file (line ~16):

```cpp
#include "flagos_cupti_profiler.h"
#include "cupti_shim.h"
#include "device_tracer.h"  // NEW
```

- [ ] **Step 4: Verify clean build**

```bash
conda activate torch-fl-211
export CUDA_HOME="${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit root}"
cmake --build build --target install
```

Expected: clean compile, no link errors (stub is not called, just ensures interface compiles).

- [ ] **Step 5: Commit interface**

```bash
git add csrc/profiler/device_tracer.h csrc/profiler/flagos_cupti_profiler.cc
git -c user.name=lvyufeng -c user.email=lvyufeng@cqu.edu.cn commit -m \
  "feat(profiler): add vendor-agnostic DeviceTracer interface

Pure abstract class + DeviceEvent struct with open metadata map.
Stub factory in flagos_cupti_profiler.cc (moved to cupti_device_tracer.cc in next task).

Refs #spec 2026-08-03 §2.1"
```

---

### Task 3: Move CUPTI Logic to cupti_device_tracer.cc

**Files:**
- Create: `csrc/profiler/cupti_device_tracer.cc`
- Modify: `csrc/profiler/flagos_cupti_profiler.cc` (remove CUPTI details, keep kineto adaptor only)
- Modify: `csrc/profiler/cupti_shim.h` (add MEMSET kind, cbid→name helper)

**Interfaces:**
- Consumes: `DeviceTracer` interface from Task 2, CUPTI shim, kernel/memcpy/runtime parsing from Task 1
- Produces: `CuptiDeviceTracer` class implementing `DeviceTracer`, `MakeDeviceTracer()` returning real instance

- [ ] **Step 1: Add MEMSET activity kind to cupti_shim.h**

In the `CUpti_ActivityKind` enum (line ~43):

```cpp
typedef enum {
  CUPTI_ACTIVITY_KIND_INVALID = 0,
  CUPTI_ACTIVITY_KIND_MEMCPY = 1,
  CUPTI_ACTIVITY_KIND_MEMSET = 2,     // NEW
  CUPTI_ACTIVITY_KIND_KERNEL = 3,
  CUPTI_ACTIVITY_KIND_DRIVER = 4,
  CUPTI_ACTIVITY_KIND_RUNTIME = 5,
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
} CUpti_ActivityKind;
```

- [ ] **Step 2: Add cbid→name mapping helper to cupti_shim.h**

After the `CUpti_ActivityKind` enum:

```cpp
// Runtime API callback id → human name mapping (subset)
inline const char* cuptiRuntimeCbidToName(uint32_t cbid) {
  // From cupti_callbacks.h CUPTI_RUNTIME_TRACE_CBID_* enum
  switch (cbid) {
    case 211: return "cudaLaunchKernel";
    case 178: return "cudaMemcpyAsync";
    case 164: return "cudaMemcpy";
    case 85:  return "cudaMemsetAsync";
    case 83:  return "cudaMemset";
    case 157: return "cudaLaunchCooperativeKernel";
    default:  return "cudaRuntime";
  }
}
```

- [ ] **Step 3: Create cupti_device_tracer.cc with CuptiDeviceTracer class**

Create `csrc/profiler/cupti_device_tracer.cc`:

```cpp
// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// ...

#include "device_tracer.h"
#include "cupti_shim.h"

#include <cxxabi.h>
#include <atomic>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace c10::flagos::profiler {

namespace {
inline bool cupti_debug() {
  static const bool on = (std::getenv("FLAGOS_CUPTI_SHIM_DEBUG") != nullptr);
  return on;
}
}  // namespace
#define CUPTI_LOG(expr) \
  do { if (cupti_debug()) { std::cerr << expr; } } while (0)

// Layout self-check from existing flagos_cupti_profiler.cc
constexpr uint64_t kMaxPlausibleDurationNs = 3600ull * 1000 * 1000 * 1000;
std::atomic<uint64_t> g_layout_reject_count{0};
std::once_flag g_layout_warn_once;

void reportLayoutMismatch(const char* what) {
  auto& shim = CuptiShim::get();
  std::call_once(g_layout_warn_once, [&] {
    std::cerr << "[flagos] CUPTI activity records failed layout self-check ("
              << what << ").\n"
              << "[flagos]   bound CUPTI: "
              << (shim.library_path[0] ? shim.library_path : "<unknown>")
              << " (API version " << shim.api_version << ")\n";
  });
  g_layout_reject_count.fetch_add(1, std::memory_order_relaxed);
}

bool timestampsPlausible(uint64_t start, uint64_t end) {
  if (start == 0 || end < start) return false;
  return (end - start) <= kMaxPlausibleDurationNs;
}

bool kernelNamePlausible(const char* name) {
  if (!name) return true;
  size_t len = strnlen(name, 4096);
  return len > 0 && len < 4096;
}

std::string demangleName(const char* mangled) {
  if (!mangled || mangled[0] == '\0') return "kernel";
  int status = -1;
  char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  if (status == 0 && demangled) {
    std::string result(demangled);
    free(demangled);
    return result;
  }
  return mangled;  // status=-2 means already readable, keep as-is
}
```

File is 50 lines. Continue in next Edit.

- [ ] **Step 4: Continue cupti_device_tracer.cc with buffer management**

Append to `csrc/profiler/cupti_device_tracer.cc`:

```cpp
// Global for CUPTI C-style callbacks
CuptiDeviceTracer* g_active_tracer = nullptr;
std::mutex g_tracer_mutex;

constexpr size_t kBufferSize = 8 * 1024 * 1024;
constexpr size_t kBufferAlignment = 8;

void bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  CUPTI_LOG("[cupti-tracer] bufferRequested\n");
  *buffer = (uint8_t*)aligned_alloc(kBufferAlignment, kBufferSize);
  *size = kBufferSize;
  *maxNumRecords = 0;
}

void bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                     size_t size, size_t validSize) {
  CUPTI_LOG("[cupti-tracer] bufferCompleted validSize=" << validSize << "\n");
  std::lock_guard<std::mutex> lock(g_tracer_mutex);
  if (!g_active_tracer || !buffer) {
    if (buffer) free(buffer);
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

}  // namespace

class CuptiDeviceTracer : public DeviceTracer {
 public:
  bool available() const override {
    return CuptiShim::get().ok;
  }
  
  void start() override {
    auto& shim = CuptiShim::get();
    if (!shim.ok) return;
    
    {
      std::lock_guard<std::mutex> lock(g_tracer_mutex);
      g_active_tracer = this;
      events_.clear();
    }
    
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    shim.ActivityFlushAll(1);
  }
  
  void stop() override {
    auto& shim = CuptiShim::get();
    if (!shim.ok) return;
    
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET);
    shim.ActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    shim.ActivityFlushAll(0);
    
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    g_active_tracer = nullptr;
  }
  
  std::vector<DeviceEvent> drain() override {
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    return std::move(events_);
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
      uint64_t id;
      shim.ActivityPopExternalCorrelationId(
          CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &id);
    }
  }
  
  int deviceCount() const override {
    // TODO: query actual device count via CUDA runtime
    return 8;
  }
```

Continue to next step marker, 45 lines written.

- [ ] **Step 5: Add processBuffer method to CuptiDeviceTracer**

Append to class body in `csrc/profiler/cupti_device_tracer.cc`:

```cpp
  void processBuffer(uint8_t* buffer, size_t validSize) {
    auto& shim = CuptiShim::get();
    
    using CUpti_ActivityKind_t = uint32_t;
    struct CUpti_Activity { CUpti_ActivityKind_t kind; };
    
    #pragma pack(push, 1)
    struct CUpti_ActivityKernel9_Compat {
      CUpti_ActivityKind_t kind;
      uint8_t cacheConfig;
      uint8_t sharedMemoryConfig;
      uint16_t registersPerThread;
      uint32_t partitionedGlobalCacheRequested;
      uint32_t partitionedGlobalCacheExecuted;
      uint64_t start, end, completed;
      uint32_t deviceId, contextId, streamId;
      int32_t gridX, gridY, gridZ;
      int32_t blockX, blockY, blockZ;
      int32_t staticSharedMemory, dynamicSharedMemory;
      uint32_t localMemoryPerThread, localMemoryTotal;
      uint32_t correlationId;
      int64_t gridId;
      const char* name;
    };
    
    struct CUpti_ActivityMemcpy_Compat {
      CUpti_ActivityKind_t kind;
      uint8_t copyKind, srcKind, dstKind, flags;
      uint64_t bytes, start, end;
      uint32_t deviceId, contextId, streamId, correlationId;
    };
    
    struct CUpti_ActivityMemset_Compat {
      CUpti_ActivityKind_t kind;
      uint32_t value;
      uint64_t bytes, start, end;
      uint32_t deviceId, contextId, streamId, correlationId;
    };
    
    struct CUpti_ActivityRuntime_Compat {
      CUpti_ActivityKind_t kind;
      uint32_t cbid;
      uint64_t start, end;
      uint32_t processId, threadId, correlationId;
    };
    #pragma pack(pop)

    CUpti_Activity* record = nullptr;
    while (true) {
      CUptiResult status = shim.ActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_ERROR_MAX_LIMIT_REACHED || !record) break;
      if (status != CUPTI_SUCCESS) break;
      
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        auto* k = reinterpret_cast<CUpti_ActivityKernel9_Compat*>(record);
        if (!timestampsPlausible(k->start, k->end) || 
            !kernelNamePlausible(k->name)) {
          reportLayoutMismatch("kernel");
          continue;
        }
        
        DeviceEvent ev;
        ev.kind = EventKind::Kernel;
        ev.start_ns = k->start;
        ev.end_ns = k->end;
        ev.correlation_id = k->correlationId;
        ev.device = k->deviceId;
        ev.stream = k->streamId;
        ev.name = demangleName(k->name);
        ev.metadata["grid"] = "[" + std::to_string(k->gridX) + "," +
                              std::to_string(k->gridY) + "," +
                              std::to_string(k->gridZ) + "]";
        ev.metadata["block"] = "[" + std::to_string(k->blockX) + "," +
                               std::to_string(k->blockY) + "," +
                               std::to_string(k->blockZ) + "]";
        ev.metadata["registers per thread"] = std::to_string(k->registersPerThread);
        ev.metadata["shared memory"] = std::to_string(
            k->staticSharedMemory + k->dynamicSharedMemory);
        events_.push_back(std::move(ev));
        
      } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
        auto* m = reinterpret_cast<CUpti_ActivityMemcpy_Compat*>(record);
        if (!timestampsPlausible(m->start, m->end)) {
          reportLayoutMismatch("memcpy");
          continue;
        }
        DeviceEvent ev;
        ev.kind = EventKind::Memcpy;
        ev.start_ns = m->start;
        ev.end_ns = m->end;
        ev.correlation_id = m->correlationId;
        ev.device = m->deviceId;
        ev.stream = m->streamId;
        ev.name = "Memcpy";
        ev.metadata["bytes"] = std::to_string(m->bytes);
        events_.push_back(std::move(ev));
```

50 lines chunk. Continue Step 6 in next edit.

- [ ] **Step 6: Complete processBuffer with MEMSET and RUNTIME parsing**

Append to `processBuffer` in `csrc/profiler/cupti_device_tracer.cc`:

```cpp
      } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
        auto* s = reinterpret_cast<CUpti_ActivityMemset_Compat*>(record);
        if (!timestampsPlausible(s->start, s->end)) {
          reportLayoutMismatch("memset");
          continue;
        }
        DeviceEvent ev;
        ev.kind = EventKind::Memset;
        ev.start_ns = s->start;
        ev.end_ns = s->end;
        ev.correlation_id = s->correlationId;
        ev.device = s->deviceId;
        ev.stream = s->streamId;
        ev.name = "Memset";
        ev.metadata["bytes"] = std::to_string(s->bytes);
        events_.push_back(std::move(ev));
        
      } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
        auto* r = reinterpret_cast<CUpti_ActivityRuntime_Compat*>(record);
        if (!timestampsPlausible(r->start, r->end)) {
          reportLayoutMismatch("runtime");
          continue;
        }
        DeviceEvent ev;
        ev.kind = EventKind::Runtime;
        ev.start_ns = r->start;
        ev.end_ns = r->end;
        ev.correlation_id = r->correlationId;
        ev.thread_id = r->threadId;
        ev.name = cuptiRuntimeCbidToName(r->cbid);
        events_.push_back(std::move(ev));
      }
    }
  }
  
 private:
  std::vector<DeviceEvent> events_;
};
```

38 lines. Complete Task 3 with factory and commit steps.

- [ ] **Step 7: Replace stub factory with real CuptiDeviceTracer**

Append to end of `csrc/profiler/cupti_device_tracer.cc`:

```cpp
std::unique_ptr<DeviceTracer> MakeDeviceTracer() {
  return std::make_unique<CuptiDeviceTracer>();
}

// Static registration: arm CUPTI at module load
namespace {
struct CuptiTracerInit {
  CuptiTracerInit() {
    auto& shim = CuptiShim::get();
    if (!shim.ok) return;
    
    CUPTI_LOG("[cupti-tracer] Registering CUPTI callbacks\n");
    shim.ActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    shim.ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  }
};
static CuptiTracerInit g_init;
}  // namespace

}  // namespace c10::flagos::profiler
```

- [ ] **Step 8: Remove stub factory from flagos_cupti_profiler.cc**

Delete the `StubCuptiTracer` class and `MakeDeviceTracer` factory added in Task 2 Step 2 (the block after line ~478). Keep only the `#include "device_tracer.h"` added in Task 2 Step 3.

- [ ] **Step 9: Clean rebuild (CMake GLOB picks up new .cc automatically)**

```bash
conda activate torch-fl-211
export CUDA_HOME="${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit root}"
cmake --build build --target install
```

Expected: clean compile, cupti_device_tracer.cc compiled and linked.

- [ ] **Step 10: Verify tracer compiles and loads (smoke test)**

```bash
PYTHONPATH=$(pwd) bash scripts/vendor/with_cuda_libtorch.sh python -c "
import torch_fl
print('torch_fl loaded, CUPTI tracer initialized if available')
"
```

Expected: no crash, no linker errors.

- [ ] **Step 11: Commit CUPTI extraction**

```bash
git add csrc/profiler/cupti_device_tracer.cc \
        csrc/profiler/cupti_shim.h \
        csrc/profiler/flagos_cupti_profiler.cc
git -c user.name=lvyufeng -c user.email=lvyufeng@cqu.edu.cn commit -m \
  "refactor(profiler): extract CUPTI logic to cupti_device_tracer.cc

CuptiDeviceTracer implements DeviceTracer interface.
Parses KERNEL/MEMCPY/MEMSET/RUNTIME activities into DeviceEvent structs.
Includes demangle, layout self-check, metadata map (grid/block/registers/bytes).

Moved from flagos_cupti_profiler.cc, which will become pure kineto adaptor in next task.

Refs #spec 2026-08-03 §2"
```

---

### Task 4: Rewrite flagos_kineto_profiler as Generic Adaptor

**Files:**
- Rename: `csrc/profiler/flagos_cupti_profiler.{h,cc}` → `csrc/profiler/flagos_kineto_profiler.{h,cc}`
- Modify: the renamed files (remove all CUPTI types, consume DeviceTracer only)

**Interfaces:**
- Consumes: `DeviceTracer` from Task 3, `MakeDeviceTracer()` factory
- Produces: `FlagosKinetoProfilerSession` with 4-arg `processTrace`, flow/linked wiring, ActivityType mapping

**Context:** This task is the completion of the design's split — generic kineto adaptor layer with zero vendor coupling. After this, adding ascend/musa support means writing one new `*_device_tracer.cc`, nothing else.

**Implementation Summary:**
- Rename `flagos_cupti_profiler.{h,cc}` → `flagos_kineto_profiler.{h,cc}` via `git mv`
- Rewrite `.cc` to consume `DeviceTracer::drain()`, map `EventKind` → `ActivityType`
- Implement 4-arg `processTrace(logger, getLinkedActivity, start, end)` — the correlation heart:
  ```cpp
  for (auto& ev : events) {
    GenericTraceActivity act;
    act.activityType = mapEventKind(ev.kind);
    act.activityName = ev.name;
    act.startTime = ev.start_ns;
    act.endTime = ev.end_ns;
    act.device = ev.device;
    act.resource = (ev.kind == EventKind::Runtime) ? ev.thread_id : ev.stream;
    act.id = ev.correlation_id;
    for (auto& [k,v] : ev.metadata) act.addMetadata(k, v);
    
    // The correlation link (design §3.2):
    if (getLinkedActivity && ev.correlation_id != 0) {
      auto* linked = getLinkedActivity(ev.correlation_id);
      if (linked) {
        act.linked = linked;
        act.flow.id = ev.correlation_id;
        act.flow.type = libkineto::kLinkAsyncCpuGpu;
        act.flow.start = (ev.kind == EventKind::Runtime);
      }
    }
    act.log(logger);
  }
  ```
- Delete CUPTI record structs, buffer callbacks, all layout self-check (now in cupti_device_tracer.cc)
- Update `getDeviceInfo` to query `tracer.deviceCount()`
- Rebuild, verify with Task 1's verification script (3 metrics must still pass)
- Commit as "refactor(profiler): rewrite as vendor-agnostic kineto adaptor"

---

### Task 5: Add 13-Field Kernel Metadata + Occupancy Estimation

**Summary:** Enhance `CuptiDeviceTracer` to fill all 13 metadata fields torch-cuda emits. Add occupancy calculation (blocks per SM, warps per SM, est. achieved occupancy %) using SM count from device props.

**Key additions:**
- Query `cudaGetDeviceProperties` for SM count (cache per device)
- Compute `blocks_per_sm = min(max_blocks_per_sm, grid_size / sm_count)`
- Compute `warps_per_sm = blocks_per_sm * (threads_per_block / 32)`
- Compute `est_occupancy = min(1.0, warps_per_sm / max_warps_per_sm)`
- Add to metadata: `"blocks per SM"`, `"warps per SM"`, `"est. achieved occupancy %"`
- Also add: `"correlation"`, `"device"`, `"stream"`, `"context"`, `"External id"`, `"queued"` (timestamp fields)

Test: verify baseline comparison shows 13/13 fields present.

---

### Task 6: Generate Baseline Snapshot + CI Test

**Files:**
- Create: `tests/data/profiler_cuda_baseline.json` (torch-cuda reference)
- Create: `tests/data/gen_profiler_baseline.py` (regeneration script)
- Create: `tests/integration/test_profiler_parity.py` (CI test with `@pytest.mark.main_ops`)
- Modify: `.github/configs/cuda.yml` (add explicit profiler test entry)

**Baseline snapshot structure:**
```json
{
  "generated_by": "torch 2.10.0+cu128, A100, 2026-08-04",
  "categories": ["cpu_op", "cuda_runtime", "kernel", "gpu_memset", "ac2g", "cpu_instant_event"],
  "runtime_cat_equivalents": ["cuda_runtime", "privateuse1_runtime"],
  "kernel_arg_keys": ["grid", "block", "correlation", "device", "stream", 
                      "registers per thread", "shared memory", "External id",
                      "blocks per SM", "warps per SM", "est. achieved occupancy %", "queued"],
  "note": "Compare structure only, not numeric values"
}
```

**CI test verifies:**
1. Categories cover baseline set (with runtime equivalence)
2. Flow arrows (`ac2g`) count > 0, op↔kernel paired (correlation matches)
3. Kernel metadata keys ⊇ baseline 13 keys
4. `key_averages()` shows `aten::mm` with `self_device_time_total > 0`
5. Kernel names are demangled (no `_ZN` prefix)

**CI integration:** Add to `.github/configs/cuda.yml` test matrix:
```yaml
- name: Profiler parity test
  run: pytest tests/integration/test_profiler_parity.py -v -m main_ops
```

Marker `main_ops` is crucial — without it, CI won't collect the test.

---

### Task 7: Delete Dead Code + Documentation

**Files to delete:**
- `csrc/profiler/flagos_profiler_stubs.cc` (proven unreachable by Task 1 investigation)
- `tests/scratch/verify_correlation_foundation.py` (one-shot verification, not regression)

**After deletion:** `rm -rf build; cmake -B build` (GLOB_RECURSE doesn't auto-remove)

**Documentation updates:**
- Add `docs/profiler.md` explaining the 3-layer architecture (DeviceTracer / vendor impl / kineto adaptor)
- Document the "CUPTI must be armed before first CUDA context" constraint (references memory entry)
- Update `CLAUDE.md` profiler section to reflect parity achievement

**Final verification:** Run full regression matrix from spec §5 (vendor config, flaggems config, CI selectors).

**Commit:** "feat(profiler): achieve torch-cuda parity — flows, metadata, device time attribution"

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-04-profiler-cuda-parity.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

