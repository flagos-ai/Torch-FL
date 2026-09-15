# PrivateUse1 profiler support — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to
> implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** give `torch.profiler` flagos (PrivateUse1) capabilities equivalent to torch-cuda:
operator-level device timing (Stage A) plus a CUPTI kernel-level timeline (Stage B).

**Architecture:** Stage A implements and registers a
`torch::profiler::impl::ProfilerStubs` subclass through `registerPrivateUse1Methods`, using the
existing flagos event ABI and fixing the guard to carry the real CUDA stream. Stage B implements a
`libkineto::IActivityProfiler` subclass, dlopens the system libcupti to collect GPU events, and
injects them through kineto's external-profiler interface. libtorch/libkineto are not rebuilt.

**Tech stack:** C++17, PyTorch 2.11 PrivateUse1, libkineto's external-profiler interface, CUPTI
Activity API (dlopen), CMake, pytest.

## Global constraints

- Build/test environment: conda env `torch-fl-211`, Python 3.12, torch `2.11.0+cpu` (CPU wheel;
  never install pip CUDA torch). Activate conda through its portable shell hook:
  `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate torch-fl-211`.
- Build command:
  `FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation`
  (g++ only; CUDA symbols resolve from the external .so at runtime).
- Every run/test goes through `bash scripts/vendor/with_cuda_libtorch.sh <cmd>` (`LD_PRELOAD` injects
  libtorch_cuda.so; direct pytest fails during device initialization).
- Backend config: `FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf`.
- Qwen3 tests need an offline Hugging Face cache supplied by the user:
  `HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; model
  `Qwen/Qwen3-0.6B`.
- External cu128 CUDA assets live in `.libtorch_cuda_assets/`. CMake needs
  `-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE` because the CPU wheel lacks
  `cuda_cmake_macros.h`.
- Hardware used for verification: 8× A100-SXM4-40GB, driver 580, host CUDA 13.0 toolkit with
  cu128 userspace.
- `csrc/CMakeLists.txt` uses `GLOB_RECURSE *.cc`, so adding `csrc/profiler/*.cc` includes them
  automatically; no `add_subdirectory` is needed.
- CUPTI: discover `cupti_activity.h` through `$CUDA_HOME` / `find_path`; runtime library is the
  system `libcupti.so.13` or pip's `nvidia/cuda_cupti/lib/libcupti.so.12`.
- Commit after every task; follow TDD, DRY, and YAGNI.
- Unless a command says otherwise, run it from the repository root.

---

## File structure

- `csrc/profiler/flagos_profiler_stubs.cc` (new) — Stage A:
  `FlagosProfilerStubs` plus static registration.
- `csrc/runtime/guard.h` (modify stream methods and record) — real CUDA streams.
- `csrc/profiler/flagos_cupti_profiler.h` (new) — Stage B class declarations.
- `csrc/profiler/flagos_cupti_profiler.cc` (new) — Stage B CUPTI dlopen,
  `IActivityProfiler` implementation, and kineto registration.
- `csrc/profiler/cupti_shim.h` (new) — CUPTI function-pointer dlopen wrapper, isolating CUPTI
  headers.
- `tests/unit/test_profiler_privateuse1.py` (new) — Stage A/B Python unit tests.
- `csrc/CMakeLists.txt` (modify) — add the CUPTI include path for Stage B only.

---

## Task 0: build `_C` in the worktree and get `import torch_fl` working

**Files:**
- Modify: no source changes; build environment only

**Interfaces:**
- Consumes: none
- Produces: a usable `torch_fl._C` extension on which all later verification depends

- [ ] **Step 1: confirm the external CUDA assets exist**

```bash
ls -la .libtorch_cuda_assets/
```

Expected: `libtorch_cuda.so`, `libc10_cuda.so`, and the other runtime assets. If a worktree does
not have this ignored directory, copy the assets into it from an environment-specific location;
do not commit a machine-local symlink.

- [ ] **Step 2: generate the operator code**

```bash
python scripts/codegen/codegen_ops.py
```

Expected: generate `csrc/aten/generated/*.cc` (about 1824 ops) without errors.

- [ ] **Step 3: build `_C`**

```bash
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON \
  pip install -e . --no-build-isolation
```

Expected: compilation succeeds. FlagGems is disabled to avoid the background
`libtriton_jit.so` undefined `c10::MessageLogger` symbol caused by a source-version mismatch
between pip flag_gems and liboperators.so.

- [ ] **Step 4: verify import**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -c \
  "import torch_fl, torch; x=torch.randn(8,8,device='flagos'); torch.flagos.synchronize(); print('OK', (x@x).sum().item())"
```

Expected: prints `OK <number>` with no ImportError or device-initialization error.

- [ ] **Step 5: record the baseline (no commit — environment only)**

If any build script needed a portable fix, stage and commit that file. Otherwise skip.

---

## Task 1: make guard.h carry the real CUDA stream

**Files:**
- Modify: `csrc/runtime/guard.h` (stream getters and record)
- Test: `tests/unit/test_profiler_privateuse1.py` (stream regression)

**Interfaces:**
- Consumes: external libtorch_cuda's `c10::cuda::getCurrentCUDAStream(DeviceIndex)` and
  `c10::cuda::CUDAStream`; flagos ABI `::EventRecord(Event_t, Stream_t)`
- Produces: guard `getStream/getDefaultStream/exchangeStream/getNewStream` results that carry a
  real CUDA StreamId; `record` records on the supplied stream rather than `nullptr`. Task 2's
  `record()` depends on this for correct stream attribution.

- [ ] **Step 1: write the failing multi-stream attribution test**

Append to `tests/unit/test_profiler_privateuse1.py`:

```python
import torch
import torch_fl  # noqa: F401


def test_guard_stream_is_real_not_synthetic():
    """The guard current stream must match torch.flagos, not a constant synthetic zero."""
    s = torch.flagos.current_stream()
    s2 = torch.flagos.Stream()
    with torch.flagos.stream(s2):
        cur = torch.flagos.current_stream()
    assert cur.stream_id == s2.stream_id
    assert s.stream_id != s2.stream_id or s2.stream_id != 0
```

- [ ] **Step 2: run and confirm it fails**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_guard_stream_is_real_not_synthetic -v
```

Expected: FAIL; the current guard returns the synthetic id=0 for both streams.

- [ ] **Step 3: modify the guard.h stream getters**

Add in the CUDA-vendor section at the top of `csrc/runtime/guard.h`, protected with the existing
vendor-macro style:

```cpp
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_TSINGMICRO)
#include <c10/cuda/CUDAStream.h>
#endif
```

Change `getStream` to:

```cpp
c10::Stream getStream(c10::Device d) const noexcept override {
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_TSINGMICRO)
  auto s = c10::cuda::getCurrentCUDAStream(d.index());
  return c10::Stream(c10::Stream::UNSAFE, d, s.id());
#else
  return c10::Stream(c10::Stream::UNSAFE, d, 0);
#endif
}
```

Apply the same pattern to `getDefaultStream` (use `getDefaultCUDAStream(d.index())`),
`getStreamFromGlobalPool` (use `getStreamFromPool(isHighPriority, d.index())`), `exchangeStream`
(use `setCurrentCUDAStream(CUDAStream(...))` and return the old stream), and `getNewStream` (use
`getStreamFromPool`). Non-CUDA vendor branches retain the synthetic stream.

- [ ] **Step 4: make guard.h record on the supplied stream**

```cpp
void record(void** event, const c10::Stream& stream,
            const c10::DeviceIndex device_index,
            const c10::EventFlag flag) const override {
  if (!*event) {
    ::EventCreate((Event_t*)event);
  }
#if !defined(USE_ASCEND) && !defined(USE_GCU) && !defined(USE_TSINGMICRO)
  auto cs = c10::cuda::CUDAStream(c10::cuda::CUDAStream::UNCHECKED,
                                  c10::Stream(c10::Stream::UNSAFE,
                                              stream.device(), stream.id()));
  ::EventRecord(*(Event_t*)event, (Stream_t)cs.stream());
#else
  ::EventRecord(*(Event_t*)event, nullptr);
#endif
}
```

`cs.stream()` returns a `cudaStream_t`; `Stream_t` is `struct Stream*`. They are converted under
the flagos ABI exactly as the existing `(cudaStream_t)stream` conversion in `cuda/stream.cc`.

- [ ] **Step 5: rebuild and run the test**

```bash
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON \
  pip install -e . --no-build-isolation
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_guard_stream_is_real_not_synthetic -v
```

Expected: PASS.

- [ ] **Step 6: guard regression — ordinary ops and copy remain correct**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/integration/ops/ -m "not flaggems and not flaggems_python" -q
```

Expected: matches the baseline (roughly 311 passed), with no new failure.

- [ ] **Step 7: commit**

```bash
git add csrc/runtime/guard.h tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): guard.h carries real CUDA stream for event attribution"
```

---

## Task 2: Stage A — FlagosProfilerStubs

**Files:**
- Create: `csrc/profiler/flagos_profiler_stubs.cc`
- Test: `tests/unit/test_profiler_privateuse1.py` (Stage A)

**Interfaces:**
- Consumes: `torch::profiler::impl::ProfilerStubs`
  (`torch/csrc/profiler/stubs/base.h`), `registerPrivateUse1Methods`; flagos ABI
  `EventCreateWithFlags/EventRecord/EventElapsedTime/EventDestroy/DeviceSynchronize/GetDeviceCount`
  (`csrc/include/flagos.h`); the guard record semantics from Task 1; and
  `torch::profiler::impl::getTime()`
- Produces: static initialization calls
  `registerPrivateUse1Methods(new FlagosProfilerStubs())`, allowing
  `profile(activities=[CPU, PrivateUse1])` to obtain operator-level device self-time via
  `KINETO_PRIVATEUSE1_FALLBACK`

- [ ] **Step 1: write the failing operator-level device timing test**

```python
from torch.profiler import profile, ProfilerActivity


def test_stage_a_privateuse1_device_time():
    x = torch.randn(1024, 1024, device="flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        for _ in range(5):
            y = x @ x
        torch.flagos.synchronize()
    ka = prof.key_averages()
    dev_times = [getattr(e, "self_device_time_total", 0) for e in ka]
    assert any(t > 0 for t in dev_times), f"no device time recorded: max={max(dev_times, default=0)}"
```

- [ ] **Step 2: run and confirm it fails**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_a_privateuse1_device_time -v
```

Expected: FAIL; no ProfilerStubs is registered, so every device time is zero.

- [ ] **Step 3: implement flagos_profiler_stubs.cc**

Create `csrc/profiler/flagos_profiler_stubs.cc`:

```cpp
// Copyright 2026 FlagOS Contributors. Apache-2.0.
#include <torch/csrc/profiler/stubs/base.h>
#include <c10/util/Exception.h>
#include <functional>
#include <memory>

#include "flagos.h"

namespace c10::flagos {
namespace {

using torch::profiler::impl::ProfilerStubs;
using torch::profiler::impl::ProfilerVoidEventStub;

struct FlagosProfilerStubs : public ProfilerStubs {
  void record(c10::DeviceIndex* device, ProfilerVoidEventStub* event,
              int64_t* cpu_ns) const override {
    if (device) {
      int d = 0;
      ::GetDevice(&d);
      *device = static_cast<c10::DeviceIndex>(d);
    }
    if (cpu_ns) {
      *cpu_ns = torch::profiler::impl::getTime();
    }
    Event_t ev = nullptr;
    ::EventCreateWithFlags(&ev, EventEnableTiming);
    ::EventRecord(ev, nullptr);  // current/default stream; multi-stream handling lives in guard
    *event = std::shared_ptr<void>(ev, [](void* p) {
      if (p) ::EventDestroy((Event_t)p);
    });
  }

  float elapsed(const ProfilerVoidEventStub* event,
                const ProfilerVoidEventStub* event2) const override {
    ::EventSynchronize((Event_t)event2->get());
    float ms = 0.0f;
    ::EventElapsedTime(&ms, (Event_t)event->get(), (Event_t)event2->get());
    return ms * 1000.0f;  // us
  }

  void mark(const char*) const override {}       // Stage A: no-op; add NVTX later
  void rangePush(const char*) const override {}
  void rangePop() const override {}
  bool enabled() const override { return true; }

  void onEachDevice(std::function<void(int)> op) const override {
    int count = 0;
    ::GetDeviceCount(&count);
    for (int i = 0; i < count; ++i) op(i);
  }

  void synchronize() const override { ::DeviceSynchronize(); }
  ~FlagosProfilerStubs() override = default;
};

struct RegisterFlagosStubs {
  RegisterFlagosStubs() {
    static FlagosProfilerStubs stubs;
    torch::profiler::impl::registerPrivateUse1Methods(&stubs);
  }
};
static RegisterFlagosStubs g_register_flagos_stubs;

}  // namespace
}  // namespace c10::flagos
```

The fallback path owns the flag/stream semantics passed to `record`; this implementation follows
`CUDAStubs` and records on the current stream. torch does not call `record` again when reusing a
non-empty `*event`, so this function always creates a new one.

- [ ] **Step 4: rebuild and run the test**

```bash
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON \
  pip install -e . --no-build-isolation
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_a_privateuse1_device_time -v
```

Expected: PASS with non-zero device self-time. If linking reports
`registerPrivateUse1Methods` undefined, confirm csrc links `torch_python_library` and uses the
matching torch headers.

- [ ] **Step 5: commit**

```bash
git add csrc/profiler/flagos_profiler_stubs.cc tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): Stage A FlagosProfilerStubs for op-level device timing"
```

---

## Task 3: verify Stage A on a real model

**Files:**
- Test: reuse `tests/integration/test_qwen3_infer.py` under a profiler, adding a separate
  assertion script rather than changing the original file

**Interfaces:**
- Consumes: Task 2's ProfilerStubs registration
- Produces: evidence that operator-level device timing works under a real model

- [ ] **Step 1: write the verification script**

Create `tests/integration/test_profiler_qwen3_infer.py`:

```python
import torch
import torch_fl  # noqa: F401
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"


def test_profiler_over_qwen3_infer():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("flagos")
    ids = tok("Hello", return_tensors="pt").input_ids.to("flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        with torch.no_grad():
            model(ids)
        torch.flagos.synchronize()
    ka = prof.key_averages()
    dev = [getattr(e, "self_device_time_total", 0) for e in ka]
    assert any(t > 0 for t in dev), "no device time in qwen3 infer profile"
    print(ka.table(sort_by="self_device_time_total", row_limit=10))
```

- [ ] **Step 2: run**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/integration/test_profiler_qwen3_infer.py -v -s
```

Expected: PASS and prints an operator table containing non-zero device time.

- [ ] **Step 3: commit**

```bash
git add tests/integration/test_profiler_qwen3_infer.py
git commit -m "test(profiler): Stage A verification over qwen3 infer"
```

---

## Task 4: Stage B — CUPTI dlopen shim

**Files:**
- Create: `csrc/profiler/cupti_shim.h`
- Test: `tests/unit/test_profiler_privateuse1.py` (CUPTI loadability probe)

**Interfaces:**
- Consumes: system `libcupti.so.13` / pip `libcupti.so.12`; CUPTI headers (types/enums only)
- Produces: the `c10::flagos::CuptiShim` singleton, exposing `bool available()` and function
  pointers for activityEnable/activityRegisterCallbacks/activityFlushAll/activityGetNextRecord/
  activityPushExternalCorrelationId/activityPopExternalCorrelationId; consumed by Task 5

- [ ] **Step 1: write the environment probe**

Append to `tests/unit/test_profiler_privateuse1.py`:

```python
import ctypes
import ctypes.util


def test_cupti_library_locatable():
    """Confirm that the Stage B runtime can locate and dlopen libcupti."""
    candidates = [ctypes.util.find_library("cupti"), "libcupti.so.13", "libcupti.so.12", "libcupti.so"]
    loaded = None
    for candidate in filter(None, candidates):
        try:
            loaded = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    assert loaded is not None, f"cannot dlopen libcupti from {candidates}"
```

- [ ] **Step 2: run it**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_cupti_library_locatable -v
```

Expected: PASS because the library exists. This is an environment guard, not a TDD red test.

- [ ] **Step 3: implement cupti_shim.h**

Create `csrc/profiler/cupti_shim.h`:

```cpp
// Copyright 2026 FlagOS Contributors. Apache-2.0.
#pragma once
#include <cupti_activity.h>  // types/enums only; runtime symbols are dlopened
#include <dlfcn.h>

namespace c10::flagos {

struct CuptiShim {
  bool ok = false;
  CUptiResult (*ActivityEnable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityDisable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*ActivityRegisterCallbacks)(
      CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc) = nullptr;
  CUptiResult (*ActivityFlushAll)(uint32_t) = nullptr;
  CUptiResult (*ActivityGetNextRecord)(uint8_t*, size_t, CUpti_Activity**) = nullptr;
  CUptiResult (*ActivityGetNumDroppedRecords)(CUcontext, uint32_t, size_t*) = nullptr;
  CUptiResult (*ActivityPushExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t) = nullptr;
  CUptiResult (*ActivityPopExternalCorrelationId)(
      CUpti_ExternalCorrelationKind, uint64_t*) = nullptr;

  static CuptiShim& get() {
    static CuptiShim inst;
    return inst;
  }

 private:
  CuptiShim() {
    const char* names[] = {"libcupti.so.13", "libcupti.so.12", "libcupti.so"};
    void* handle = nullptr;
    for (auto name : names) {
      handle = dlopen(name, RTLD_LAZY | RTLD_GLOBAL);
      if (handle) break;
    }
    if (!handle) return;
#define LOAD(field, sym) field = (decltype(field))dlsym(handle, sym)
    LOAD(ActivityEnable, "cuptiActivityEnable");
    LOAD(ActivityDisable, "cuptiActivityDisable");
    LOAD(ActivityRegisterCallbacks, "cuptiActivityRegisterCallbacks");
    LOAD(ActivityFlushAll, "cuptiActivityFlushAll");
    LOAD(ActivityGetNextRecord, "cuptiActivityGetNextRecord");
    LOAD(ActivityGetNumDroppedRecords, "cuptiActivityGetNumDroppedRecords");
    LOAD(ActivityPushExternalCorrelationId, "cuptiActivityPushExternalCorrelationId");
    LOAD(ActivityPopExternalCorrelationId, "cuptiActivityPopExternalCorrelationId");
#undef LOAD
    ok = ActivityEnable && ActivityRegisterCallbacks && ActivityFlushAll && ActivityGetNextRecord;
  }
};

}  // namespace c10::flagos
```

- [ ] **Step 4: add the CUPTI include path to CMake**

After `add_library(${LIBRARY_NAME} ...)` in `csrc/CMakeLists.txt`:

```cmake
# CUPTI headers for the profiler child. Runtime symbols are dlopened, so only
# the include path is needed, not the link library.
find_path(CUPTI_INCLUDE_DIR cupti_activity.h
  HINTS "$ENV{CUDA_HOME}/extras/CUPTI/include"
        "$ENV{CUDA_HOME}/targets/x86_64-linux/include")
if(CUPTI_INCLUDE_DIR)
  target_include_directories(${LIBRARY_NAME} PRIVATE ${CUPTI_INCLUDE_DIR})
  target_compile_definitions(${LIBRARY_NAME} PRIVATE FLAGOS_HAVE_CUPTI=1)
endif()
```

- [ ] **Step 5: rebuild and confirm the shim compiles**

```bash
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON \
  pip install -e . --no-build-isolation
```

Expected: compilation succeeds. The shim is header-only and Task 5 introduces the `.cc` that
uses it; proceed directly to Task 5 if the header is not compiled yet.

- [ ] **Step 6: commit**

```bash
git add csrc/profiler/cupti_shim.h csrc/CMakeLists.txt tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): Stage B CUPTI dlopen shim + cmake include path"
```

---

## Task 5: Stage B — FlagosCuptiProfiler and kineto registration

**Files:**
- Create: `csrc/profiler/flagos_cupti_profiler.h`,
  `csrc/profiler/flagos_cupti_profiler.cc`
- Test: `tests/unit/test_profiler_privateuse1.py` (Chrome trace contains kernel events)

**Interfaces:**
- Consumes: Task 4's `CuptiShim`; kineto's `libkineto::IActivityProfiler`,
  `IActivityProfilerSession`, `GenericTraceActivity`, and
  `libkineto::api().registerProfilerFactory` (`torch/include/kineto/*.h`)
- Produces: static initialization registers `FlagosCuptiProfiler` (name `"flagos_cupti"`,
  activities `CONCURRENT_KERNEL`/`GPU_MEMCPY`) through
  `libkineto::api().registerProfilerFactory(...)`; after profiling, `export_chrome_trace`
  contains GPU kernel events

- [ ] **Step 1: write the failing Chrome trace test**

```python
import json
import tempfile


def test_stage_b_chrome_trace_has_gpu_kernels():
    x = torch.randn(1024, 1024, device="flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        for _ in range(5):
            y = x @ x
        torch.flagos.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    kernel_like = [
        event for event in events
        if isinstance(event, dict)
        and (
            event.get("cat") in ("kernel", "Kernel", "gpu_op")
            or "kernel" in str(event.get("name", "")).lower()
        )
    ]
    assert kernel_like, "no GPU kernel events in chrome trace"
```

- [ ] **Step 2: run and confirm it fails**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_b_chrome_trace_has_gpu_kernels -v
```

Expected: FAIL; without a CUPTI child profiler the trace has no kernel events.

- [ ] **Step 3: declare flagos_cupti_profiler.h**

```cpp
// Copyright 2026 FlagOS Contributors. Apache-2.0.
#pragma once
#include <kineto/IActivityProfiler.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace c10::flagos {

class FlagosCuptiProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  void start() override;
  void stop() override;
  void processTrace(libkineto::ActivityLogger& logger) override;
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;
  std::vector<std::string> errors() override { return {}; }
  void pushCorrelationId(uint64_t id) override;
  void popCorrelationId() override;

 private:
  std::vector<libkineto::GenericTraceActivity> activities_;
};

class FlagosCuptiProfiler : public libkineto::IActivityProfiler {
 public:
  const std::string& name() const override;
  const std::set<libkineto::ActivityType>& availableActivities() const override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
  std::unique_ptr<libkineto::IActivityProfilerSession> configure(
      int64_t profileStartTime, int64_t profileDuration,
      const std::set<libkineto::ActivityType>& activityTypes,
      const libkineto::Config& config) override;
};

void registerFlagosCuptiProfiler();

}  // namespace c10::flagos
```

Every virtual signature must be checked against
`torch/include/kineto/IActivityProfiler.h` in the active environment before implementation.

- [ ] **Step 4: implement flagos_cupti_profiler.cc**

Create `csrc/profiler/flagos_cupti_profiler.cc` with this core logic:

1. `start()`: if `CuptiShim::get().ok`, call
   `ActivityRegisterCallbacks(bufferRequested, bufferCompleted)`, then enable
   `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` and `CUPTI_ACTIVITY_KIND_MEMCPY`.
2. Buffer callbacks: `bufferRequested` allocates an aligned buffer; `bufferCompleted` walks
   records with `ActivityGetNextRecord`, converts `CUpti_ActivityKernel*` /
   `CUpti_ActivityMemcpy` into `libkineto::GenericTraceActivity` (name, device,
   resource=streamId, start/end ns, id=correlationId, activityType), and pushes them into the
   active session's `activities_` through a global pointer.
3. `stop()`: call `ActivityFlushAll(1)`.
4. `processTrace(logger)`: call `logger.handleGenericActivity(a)` for each activity.
5. `getDeviceInfo`/`getResourceInfos`: return device and stream-resource descriptions named
   `"flagos:GPU"`.
6. `pushCorrelationId(id)`: if available, call
   `ActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id)`; pop in
   `popCorrelationId`.
7. Both `configure(...)` overloads return
   `std::make_unique<FlagosCuptiProfilerSession>()`.
8. `name()` returns static `"flagos_cupti"`; `availableActivities()` returns static
   `{CONCURRENT_KERNEL, GPU_MEMCPY}`.
9. `registerFlagosCuptiProfiler()` calls
   `libkineto::api().registerProfilerFactory([] { return std::make_unique<FlagosCuptiProfiler>(); });`.
10. A file-local static initializer registers only when `CuptiShim::get().ok`.

Use the field names from the active environment's `torch/include/kineto/output_base.h` and
`ITraceActivity.h`; verify them before writing the complete body.

- [ ] **Step 5: rebuild and run the test**

```bash
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON \
  pip install -e . --no-build-isolation
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_b_chrome_trace_has_gpu_kernels -v
```

Expected: PASS with GPU kernel events in the trace. If kineto never calls this profiler, confirm
`registerProfilerFactory` runs on `import torch_fl` (the static initializer runs when
libtorch_fl.so is loaded).

- [ ] **Step 6: commit**

```bash
git add csrc/profiler/flagos_cupti_profiler.h csrc/profiler/flagos_cupti_profiler.cc \
  tests/unit/test_profiler_privateuse1.py
git commit -m "feat(profiler): Stage B CUPTI child profiler injected via kineto"
```

---

## Task 6: Stage B correlation bridge (degradable acceptance)

**Files:**
- Modify: `csrc/profiler/flagos_cupti_profiler.cc` (Task 5 already forwards correlation; verify
  and harden it here)
- Test: `tests/unit/test_profiler_privateuse1.py` (correlation assertion with degradation)

**Interfaces:**
- Consumes: Task 5 session's `pushCorrelationId/popCorrelationId`
- Produces: GPU kernel events with correlationId mapped to CPU ops when possible; otherwise the
  acceptance bar degrades to "GPU track exists" (already guaranteed by Task 5), with correlation
  recorded as follow-up work

- [ ] **Step 1: write the correlation test**

```python
def test_stage_b_correlation_or_degrade():
    x = torch.randn(512, 512, device="flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        y = x @ x
        torch.flagos.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    flows = [e for e in events if isinstance(e, dict) and e.get("ph") in ("s", "t", "f")]
    kernels = [
        e for e in events
        if isinstance(e, dict) and "kernel" in str(e.get("name", "")).lower()
    ]
    if flows:
        print(f"correlation OK: {len(flows)} flow events")
    else:
        assert kernels, "neither correlation flows nor kernel track present"
        print("DEGRADED: kernel track present, no op<->kernel correlation (follow-up)")
```

- [ ] **Step 2: run**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/unit/test_profiler_privateuse1.py::test_stage_b_correlation_or_degrade -v -s
```

Expected: PASS and prints either `correlation OK` or `DEGRADED`. A degraded result is recorded as
follow-up but does not block this stage.

- [ ] **Step 3: commit**

```bash
git add tests/unit/test_profiler_privateuse1.py
git commit -m "test(profiler): Stage B correlation acceptance (degradable)"
```

---

## Task 7: verify Stage B on a real model and wrap up

**Files:**
- Test: extend `tests/integration/test_profiler_qwen3_infer.py` to assert the kernel track

**Interfaces:**
- Consumes: the Task 3 test and the Task 5 profiler
- Produces: final real-model evidence that the Chrome trace contains both operator device time
  and the GPU kernel timeline

- [ ] **Step 1: extend the qwen3 verification to export the trace and assert kernels**

```python
import json
import tempfile


def test_profiler_qwen3_chrome_trace_kernels():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("flagos")
    ids = tok("Hello world", return_tensors="pt").input_ids.to("flagos")
    torch.flagos.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
        with torch.no_grad():
            model(ids)
        torch.flagos.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    kernels = [e for e in events if "kernel" in str(e.get("name", "")).lower()]
    assert kernels, "no GPU kernels in qwen3 chrome trace"
    print(f"qwen3 trace: {len(kernels)} kernel events, saved {path}")
```

- [ ] **Step 2: run**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/integration/test_profiler_qwen3_infer.py -v -s
```

Expected: both tests PASS.

- [ ] **Step 3: full regression**

```bash
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -m pytest \
  tests/integration/ops/ -m "not flaggems and not flaggems_python" -q
```

Expected: matches the baseline, with no regression.

- [ ] **Step 4: commit**

```bash
git add tests/integration/test_profiler_qwen3_infer.py
git commit -m "test(profiler): Stage B verification over qwen3 infer (kernel timeline)"
```

---

## Self-review notes

- **Spec coverage**: architecture §2 → Tasks 1/2 (A) and Tasks 4/5 (B); guard §3.3 → Task 1;
  CUPTI wiring §4.2 → Task 4; kineto injection §4.3 → Task 5; degradable correlation §4.4 →
  Task 6; testing §5 → Tasks 3/7 for unit + real model; build prerequisite §5 → Task 0; CMake
  §5 → Task 4 Step 4. Full coverage.
- **Naming consistency**: `FlagosProfilerStubs` (Task 2), `CuptiShim` (Task 4), and
  `FlagosCuptiProfiler`/`FlagosCuptiProfilerSession` (Tasks 5/6/7) are consistent across tasks;
  flagos ABI names follow `csrc/include/flagos.h`.
- **Known uncertainties**: before Task 5, align kineto virtual signatures and
  `GenericTraceActivity` field names with the active environment's headers (the plan explicitly
  requires grepping them); Task 6 permits correlation to degrade.
