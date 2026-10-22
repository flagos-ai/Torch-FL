# torch_fl PrivateUse1 profiler support — design document

Date: 2026-07-31
Branch/worktree: profiler-support
Goal: make `torch.profiler` provide flagos devices with capabilities equivalent to torch-cuda —
both operator-level device timing and a CUPTI kernel-level timeline (Chrome trace / TensorBoard).

## 0. Background and verified facts

torch_fl is a PyTorch PrivateUse1 "flagos" backend. It currently has **no profiler / kineto /
record_function integration at all** (grepping `csrc/` and `torch_fl/` turns up nothing);
`tests/perf/` holds only application-level timing scripts built on `TorchDispatchMode`, which do
not go through `torch.profiler`.

Verified environment facts (conda env `torch-fl-211`, torch 2.11.0+cpu, 2× A100, driver 580):

1. **CPU-layer profiling is available for free**: `libtorch_cpu.so` contains 394 kineto symbols,
   `ProfilerActivity.CPU` works directly, and RecordFunction is independent of the dispatch key.
2. **The PrivateUse1 profiler hook points are all present**:
   `registerPrivateUse1Methods(ProfilerStubs*)`, `privateuse1Stubs()`, and
   `pushPRIVATEUSE1CallbacksStub` are all in the library; on the Python side
   `ProfilerActivity.PrivateUse1` + `ProfilerState::KINETO_PRIVATEUSE1_FALLBACK` are fully wired
   (`autograd/profiler.py` and `profiler/profiler.py` do
   `use_device = _get_privateuse1_backend_name()` → "flagos").
3. **The built-in CUPTI is an empty stub**: the CPU wheel is compiled with
   `-DLIBKINETO_NOCUPTI`. The `CuptiActivityApi::*` / `CuptiActivityProfiler::*` symbols do exist
   in libtorch_cpu (as T), but there is **no libcupti dlopen string and no real
   `cuptiActivityEnable`-style API string** — they are compile-time no-op replacements. The
   external `libtorch_cuda.so` has zero kineto symbols. → The built-in CUPTI path is unusable.
4. **kineto's external profiler registration interface is alive**: `libkineto::api()` and
   `ActivityProfilerProxy::addChildActivityProfiler(unique_ptr<IActivityProfiler>)` are symbols in
   libtorch_cpu.so (T), and the public headers
   `include/kineto/{IActivityProfiler,libkineto,ActivityProfilerInterface}.h` all ship. → We can
   write our own `IActivityProfiler` subclass and dlopen the system libcupti ourselves, without
   touching libtorch/libkineto.
5. **CUPTI is physically available**: `cupti_activity.h` is present under `$CUDA_HOME`, and
   both the system `libcupti.so.13` (2025.3.0) and pip's
   `nvidia/cuda_cupti/lib/libcupti.so.12` are available.
6. **The flagos event/stream ABI is ready**: `EventCreate` / `EventRecord` / `EventElapsedTime` /
   … in `csrc/include/flagos.h` are 1:1 wrappers around `cudaEvent*` for the CUDA vendor
   (`csrc/runtime/accelerator/cuda/stream.cc`). `GuardImpl` (`csrc/runtime/guard.h`) already wires
   these into `DeviceGuardImplInterface`'s event methods.
7. **The guard stream is a gap**: every stream accessor in `guard.h:58-77` returns a synthetic
   `Stream(UNSAFE,d,0)`, and `record` (:125) hardcodes recording on the `nullptr` null stream. →
   Operator timing on non-default streams would be scrambled; this must be fixed for parity with
   torch-cuda.

## 1. Decisions and scope

User decisions (2026-07-31):

- The goal is **full capability parity with torch-cuda**, including the **CUPTI kernel timeline**.
- **A + B in stages**: operator-level device timing first (Stage A), then the CUPTI child profiler
  on top (Stage B).
- CUPTI integration approach: **a kineto external profiler wired to the system libcupti
  ourselves** (no rebuilding of libtorch/libkineto).
- Guard stream implementation: **use the external libtorch_cuda's `c10::cuda::CUDAStream`
  directly**, guarded by `#if` for non-CUDA vendors, which fall back to the existing synthetic
  stream.
- `mark`/`rangePush`/`rangePop`: **no-ops in Stage A**; NVTX to be added later.
- Verification: **unit tests plus a real model (qwen3 inference)**.

Out of scope (YAGNI): no rebuilding of libkineto/libtorch; no changes to the existing
TorchDispatchMode scripts in `tests/perf/`; no refactoring unrelated to this goal.

## 2. Architecture overview

A new self-contained compilation unit, `csrc/profiler/`, with no changes to libtorch/libkineto.
Both stages share that directory.

- **Stage A (operator-level device timing)**:
  `FlagosProfilerStubs : torch::profiler::impl::ProfilerStubs`, statically initialized via
  `registerPrivateUse1Methods(&stubs)`. When the user calls
  `profile(activities=[CPU, PrivateUse1])`, torch enters `KINETO_PRIVATEUSE1_FALLBACK` and calls
  `record()`/`elapsed()` per op to obtain per-operator device self-time. The guard stream fix is a
  prerequisite.
- **Stage B (CUPTI kernel timeline)**:
  `FlagosCuptiProfiler : libkineto::IActivityProfiler` dlopens the system libcupti within the
  session, collects kernel/memcpy events through the CUPTI Activity API, and injects them into
  kineto via `registerProfilerFactory` / `addChildActivityProfiler`, producing a GPU timeline
  usable in Chrome trace / TensorBoard.

## 3. Stage A — components and data flow

### 3.1 New file `csrc/profiler/flagos_profiler_stubs.cc`

```cpp
struct FlagosProfilerStubs : torch::profiler::impl::ProfilerStubs {
  void record(c10::DeviceIndex* device, ProfilerVoidEventStub* event,
              int64_t* cpu_ns) const override;
  float elapsed(const ProfilerVoidEventStub* e1,
                const ProfilerVoidEventStub* e2) const override;
  void mark(const char*) const override {}         // Stage A: no-op
  void rangePush(const char*) const override {}     // NVTX added later
  void rangePop() const override {}
  bool enabled() const override { return true; }
  void onEachDevice(std::function<void(int)>) const override;
  void synchronize() const override;               // ::DeviceSynchronize()
};
```

Static initialization calls
`torch::profiler::impl::registerPrivateUse1Methods(new FlagosProfilerStubs())`.

### 3.2 Data flow

The torch profiler calls per op (semantics aligned with torch's `CUDAStubs` implementation):

- `record(device, event, cpu_ns)`: if `event->get()` is empty, call
  `EventCreateWithFlags(&ev, EventEnableTiming)`, wrap the `Event_t` in a `shared_ptr<void>`
  (whose deleter calls `EventDestroy`), and write it back into `*event`; call
  `EventRecord(ev, <current CUDA stream>)`; write back `*cpu_ns = getTime()` and `*device`.
- At aggregation time, call `elapsed(e1, e2)` for each event pair:
  `EventElapsedTime(&ms, e1, e2)`, returning `ms * 1000` (µs).
- `synchronize()` → `::DeviceSynchronize()`; `onEachDevice` iterates over `GetDeviceCount()`.

### 3.3 Prerequisite fix in `csrc/runtime/guard.h` (the core of parity)

- `getStream`/`getDefaultStream`/`exchangeStream`/`getNewStream`: on the CUDA vendor, return a
  `c10::Stream` carrying the real CUDA StreamId, obtained through the external libtorch_cuda's
  `c10::cuda::getCurrentCUDAStream(d)` / `CUDAStream`; non-CUDA vendors (`#if`) fall back to the
  existing `Stream(UNSAFE,d,0)`.
- `record(event, stream, ...)`: convert the incoming `c10::Stream` to a `cudaStream_t` and pass it
  to `EventRecord` instead of hardcoding `nullptr`.
- This change touches the guard that every op goes through, so it needs regression tests (ordinary
  ops / `copy_` / multiple streams) to confirm nothing breaks.

## 4. Stage B — the CUPTI child profiler

### 4.1 New files `csrc/profiler/flagos_cupti_profiler.{h,cc}`

```cpp
class FlagosCuptiProfilerSession : public libkineto::IActivityProfilerSession {
  void start() override;   // cuptiActivityEnable(KERNEL|MEMCPY|MEMSET|RUNTIME) + RegisterCallbacks
  void stop() override;    // cuptiActivityFlushAll
  void processTrace(libkineto::ActivityLogger&) override;   // CUPTI buffer -> kineto GenericTraceActivity
  std::unique_ptr<DeviceInfo> getDeviceInfo() override;
  std::vector<ResourceInfo> getResourceInfos() override;    // stream -> resource
  void pushCorrelationId(uint64_t) override;                // -> cuptiActivityPushExternalCorrelationId
  void popCorrelationId() override;
};

class FlagosCuptiProfiler : public libkineto::IActivityProfiler {
  const std::string& name() const override;                 // "flagos_cupti"
  const std::set<ActivityType>& availableActivities() const override; // CONCURRENT_KERNEL, GPU_MEMCPY...
  std::unique_ptr<IActivityProfilerSession> configure(...) override;
};
```

### 4.2 Wiring CUPTI ourselves (leaving libtorch's built-in stub alone)

- At runtime, `dlopen("libcupti.so.12"/".so.13")` and `dlsym` for
  `cuptiActivityEnable` / `RegisterCallbacks` / `FlushAll` / `GetNextRecord` /
  `ActivityPushExternalCorrelationId`, etc.
- At compile time, pull in only the types and enums from
  `$CUDA_HOME`'s `cupti_activity.h`; do not link against cupti.
- Double-buffered activity buffers (`bufferRequested`/`bufferCompleted` callbacks). `processTrace`
  walks `CUpti_ActivityKernel*` / `CUpti_ActivityMemcpy` records and converts them into kineto
  `GenericTraceActivity` entries carrying correlationId, device, stream, and start/end ns, feeding
  them to the logger.

### 4.3 Injecting into kineto

At static initialization or in `_C._init`:

```cpp
libkineto::api().registerProfilerFactory(
    []{ return std::make_unique<FlagosCuptiProfiler>(); });
```

When kineto generates a trace it calls back into this child profiler, merging the GPU kernel
timeline into the final Chrome trace.

### 4.4 Correlation bridging (an acceptance criterion that may be downgraded)

Under `KINETO_PRIVATEUSE1_FALLBACK`, torch pushes a correlation; the session's
`pushCorrelationId` forwards it to `cuptiActivityPushExternalCorrelationId` so kernels attach to
the corresponding op. Risk: whether CUPTI's external-correlation semantics line up exactly with
torch's PrivateUse1 fallback correlation is the least certain part of this stage. The
implementation should **first guarantee that the GPU kernel timeline appears at all (as an
independent GPU track)**; correlation attachment is an acceptance criterion, and if it is not met
it downgrades to "the GPU track exists but is not linked to ops", with correlation recorded as a
follow-up.

## 5. Testing and verification

### Stage A

- `tests/unit/test_profiler_privateuse1.py`: wrap matmul/add in
  `profile(activities=[CPU, PrivateUse1])` and assert that `key_averages()` contains entries with
  `device_type == PrivateUse1` and `self_device_time_total > 0`.
- Guard stream regression: ordinary ops, `copy_`, and multi-stream scenarios
  (`with torch.flagos.stream(s):`) all produce correct results; the id from
  `torch.flagos.current_stream()` matches what the guard recorded.
- C++ sanity: profile one op and confirm `record`/`elapsed` are called and return positive values.

### Stage B

- Unit test: after profiling, call `prof.export_chrome_trace(path)` and assert the exported JSON
  contains events with `cat == "kernel"` (or a GPU track) with count > 0.
- Correlation (may be downgraded): assert that some GPU kernel events carry a correlationId that
  maps to a CPU op; if not met, downgrade to passing on "the GPU track exists".

### Real model

- Wrap `tests/integration/test_qwen3_infer.py` in a profiler, export the trace, and confirm both
  op-level device time (A) and the kernel timeline (B).

### Build prerequisite (step 0 of the implementation plan)

- `torch_fl._C` is not yet built in this worktree, and FlagGems' `libtriton_jit.so` has an
  undefined `c10::MessageLogger` symbol (a version mismatch between pip flag_gems and
  liboperators.so). The first implementation step must be to build `_C` in this worktree and get
  `import torch_fl` working, or none of the verification can run.

### CMake

- Add `csrc/profiler/` to `csrc/CMakeLists.txt`. Stage A needs no new dependencies; Stage B needs
  the cupti include path (compile time only; the library is dlopened at runtime).

## 6. Delivery order

1. Step 0: build `_C` in the worktree and get `import torch_fl` working (resolve the triton symbol
   problem).
2. Stage A: the guard stream fix + `FlagosProfilerStubs` + unit tests (including the guard
   regression).
3. Stage A verification: wrap qwen3 inference in a profiler and confirm op-level device time.
4. Stage B: `FlagosCuptiProfiler` + the CUPTI wiring + kineto injection + unit tests.
5. Stage B verification: the Chrome trace has a GPU kernel track; correlation is an acceptance
   criterion that may be downgraded.
