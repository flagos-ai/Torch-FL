# torch_fl profiler parity with torch-cuda — design document

Date: 2026-08-03
Branch/worktree: profiler-support
Prior document: `2026-07-31-privateuse1-profiler-design.md` (the initial Stage A/B design)

Goal: make the traces `torch.profiler` produces for flagos devices **structurally** identical to
torch-cuda's — matching event categories, op↔kernel links, kernel metadata fields, and
operator-level device time attribution.

## 0. Current state and the measured gap

Stage B (the CUPTI kernel timeline) works: `torch.profiler` gets real GPU kernel events — 2261
named kernels under qwen3 inference, with correct names and durations. This document addresses
the **structural gap that remains** between that and torch-cuda.

Running the same code (5× `(x@y).relu()` + `sum`, 1024×1024) under `torch-cuda-210`
(2.10.0+cu128) and under flagos (2.11.0+cpu with an external libtorch_cuda) and comparing traces:

| Metric | torch-cuda | flagos |
|---|---|---|
| `ac2g` flow arrows | **59** | **0** |
| `cuda_runtime` events | **34** | **0** |
| kernel args field count | **13** | **0** |
| kernel name | `ampere_sgemm_128x64_nn` | `_ZN2at6native...` (not demangled) |
| `gpu_memset` | 10 | 0 |
| `key_averages` device time attribution | `aten::mm` = 821µs | attributed only to kernel names; all `aten::*` are 0 |

### Root cause: the whole correlation chain is broken

torch-cuda's chain (confirmed by measurement, not inference):

```
cpu_op (External id=2)
   |  same External id
cuda_runtime "cudaLaunchKernel" (correlation=13)   <- runs on a CPU thread (pid=2102704, tid=2102704)
   |  same correlation
kernel ampere_sgemm (correlation=13, External id=2) <- runs on a GPU stream (pid=0, tid=7)

flow: s@runtime(id=13) --ac2g--> f@kernel(id=13)
```

Measured verification:

- `cpu_op ∩ runtime` yields 15 External ids; `cpu_op ∩ kernel` yields the same 15.
- runtime and kernel share 15 correlation ids.
- The flow start ids are a subset of runtime's correlation set (True).

`cuda_runtime` is the **hub**: it holds both the `External id` (linking back to cpu_op) and the
`correlation` (linking forward to the kernel). We do not collect RUNTIME activities, which
accounts for 5 of the 6 rows of gap above.

### Two mechanism-level findings

1. **`IActivityProfilerSession::processTrace` has a four-argument overload**
   (`IActivityProfiler.h:104`) whose signature includes
   `getLinkedActivityCallback = std::function<const ITraceActivity*(int32_t)>` — kineto looks up
   the CPU-side activity by correlationId and hands it back to us. That is the official channel
   for populating `linked` / `flow`. **We only override the single-argument version**, so the
   callback is never invoked. This is the mechanical reason flow is 0.

2. **Stage A's `ProfilerStubs` is structurally dead code.**
   `autograd/profiler.py:330` is an if/else: `ProfilerActivity.PrivateUse1 in
   _supported_activities()` is True (confirmed by measurement) → it takes
   `kineto_activities.add(PrivateUse1)`, i.e. the ordinary kineto path. Only when that is False
   does it degrade to `KINETO_PRIVATEUSE1_FALLBACK` and call the stubs. Measured `profiler_kind` is
   `ProfilerState.KINETO`, and `privateuse1_elapsed_us()` is uniformly 0. torch-cuda itself also
   takes the kineto path.
   → The original test's "requires torch+cuda wheel" skip reason was wrong, but the conclusion
   that "this path should not be taken" was right.

## 1. Decisions (2026-08-03, confirmed by the user)

| Decision point | Choice |
|---|---|
| Acceptance bar | **Structural equality + an automated diff gate**; timing values are not compared |
| Baseline source | **A frozen snapshot committed to the repo**, with a regeneration script; CI needs no CUDA torch |
| Collector abstraction | **Layered**: a generic kineto layer plus a pluggable vendor collector |
| metadata representation | **A narrow core plus an open `map<string,string>`** |
| correlation approach | **Collect RUNTIME activities** (the same mechanism as torch-cuda) |
| Stage A stubs | **Delete** |
| activity kinds | KERNEL + MEMCPY + RUNTIME + MEMSET |
| ActivityType for runtime | **Try `PRIVATEUSE1_RUNTIME` first; fall back to `CUDA_RUNTIME` if it does not meet the bar** |

Out of scope (YAGNI): no rebuilding of libkineto/libtorch; no collection of
DRIVER/OVERHEAD/CUDA_SYNC; no attempt to reproduce kineto-internal bookkeeping events such as
`overhead` and `Activity Buffer Request`.

### Rejected approaches

- **Rebuilding libkineto with `LIBKINETO_NOCUPTI` turned off**: kineto is statically compiled into
  `libtorch_cpu.so`, so this amounts to rebuilding libtorch. It conflicts with the "CPU torch +
  external libtorch_cuda" architectural premise and would require a rebuild on every torch upgrade.
- **Reusing the external `libtorch_cuda.so`'s CUDA profiler path**: that .so measurably has zero
  kineto symbols; there is nothing to reuse.

## 2. Architecture

```
csrc/profiler/
  device_tracer.h            <- new. Vendor-agnostic interface, zero CUPTI types
  cupti_device_tracer.cc     <- new. The NVIDIA implementation (existing collection logic moves here)
  flagos_kineto_profiler.cc  <- the current flagos_cupti_profiler.cc, renamed and rewritten. The generic layer
  cupti_shim.h               <- kept (version decoupling already done; see §6)
  flagos_profiler_stubs.cc   <- deleted
```

### 2.1 The vendor-agnostic interface

```cpp
// device_tracer.h
namespace c10::flagos::profiler {

enum class EventKind { Kernel, Memcpy, Memset, Runtime };

struct DeviceEvent {
  EventKind kind;
  uint64_t start_ns = 0, end_ns = 0;
  uint32_t correlation_id = 0;   // the key linking runtime <-> kernel
  uint32_t device = 0, stream = 0;
  uint32_t thread_id = 0;        // Runtime events only: they land on a CPU thread
  std::string name;              // already demangled
  std::map<std::string, std::string> metadata;  // grid/block/registers/...
};

class DeviceTracer {
 public:
  virtual ~DeviceTracer() = default;
  virtual bool available() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::vector<DeviceEvent> drain() = 0;
  virtual void pushCorrelation(uint64_t) {}
  virtual void popCorrelation() {}
  virtual int deviceCount() const = 0;
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer();  // chosen by vendor at compile time
}
```

`metadata` uses `map<string,string>` rather than strongly typed members: the CUPTI collector fills
`metadata["grid"] = "[8,16,5]"`, and the generic layer blindly does
`for (auto& [k,v] : ev.metadata) act.addMetadata(k, v);`. Adding a vendor or a field touches no
interface; the cost is that field-name spelling is by convention and unchecked at compile time.

### 2.2 The generic layer's responsibilities

`flagos_kineto_profiler.cc` knows only `DeviceEvent` and contains **not one line of CUPTI code**:

- Implements `libkineto::IActivityProfiler` / `IActivityProfilerSession` and registers with kineto
- Establishes `linked` / `flow` inside the four-argument `processTrace` (§3)
- Pours `DeviceEvent::metadata` into `GenericTraceActivity::addMetadata`
- Produces `DeviceInfo` / `ResourceInfo`

Adding ascend later means writing only `cann_device_tracer.cc`; the generic layer is untouched.

## 3. The correlation chain (the core of the approach)

### 3.1 Implementation steps

1. **Collect RUNTIME activities** (adding `CUPTI_ACTIVITY_KIND_RUNTIME`) to obtain host-side events
   carrying a `correlationId`, landing on a CPU thread (the `thread_id` field).
2. **Implement the four-argument
   `processTrace(logger, getLinkedActivity, startTime, endTime)`.**
3. For each event, call `getLinkedActivity(correlation_id)` to retrieve the CPU-side activity and
   populate `activity.linked`.
4. Set flows in pairs:
   - runtime side: `flow.id = correlation_id`, `flow.type = kLinkAsyncCpuGpu` (=2),
     `flow.start = true`
   - kernel side: same id/type, `flow.start = false`

### 3.2 Device time attribution comes free

From torch's `autograd/profiler.py:710-735`:

```python
corr_id = kineto_event.linked_correlation_id()
if corr_id > 0:
    device_corr_map[corr_id].append(fe)
...
for fe in frontend_function_events:
    if fe.id in device_corr_map:
        for f_evt in device_corr_map[fe.id]:
            if f_evt.device_type in [DeviceType.CUDA, DeviceType.PrivateUse1, ...]:
                fe.append_kernel(f_evt.name, f_evt.device_index,
                                 f_evt.time_range.end - f_evt.time_range.start)
```

`GenericTraceActivity::correlationId()` returns the `id` field, and torch's
`linked_correlation_id()` reads exactly that link. **As long as `linked` is set correctly,
`aten::mm`'s `self_device_time_total` populates itself — no torch code needs touching.**

> To be verified: this conclusion is derived from the `device_corr_map` logic in
> `autograd/profiler.py:710-735` and the `getLinkedActivityCallback` signature at
> `IActivityProfiler.h:78`; the actual code was read in both places, but it has not been verified
> end to end. Verifying it is the first implementation step (§5 Step 1).

## 4. Collection scope and field population

### 4.1 activity kinds

| kind | Status | Category produced |
|---|---|---|
| `CONCURRENT_KERNEL` | existing | `kernel` |
| `MEMCPY` | existing | `gpu_memcpy` |
| `RUNTIME` | **new** | `privateuse1_runtime` / `cuda_runtime` |
| `MEMSET` | **new** | `gpu_memset` |

DRIVER / OVERHEAD / CUDA_SYNC are not collected.

### 4.2 The 13 kernel metadata fields

Matching torch-cuda's kernel `args`: `grid`, `block`, `registers per thread`, `shared memory`,
`stream`, `context`, `device`, `correlation`, `External id`, `queued`, `blocks per SM`,
`warps per SM`, `est. achieved occupancy %`.

Of these, grid / block / registers / shared memory are already decoded in the existing
`CUpti_ActivityKernel9_Compat` — they are simply not written into metadata. The three occupancy
fields must be derived from the SM count and the block count; that is this round's increment.

### 4.3 demangle

`abi::__cxa_demangle`, verified:

- mangled name → `void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<float, 4>(long, at::PhiloxCudaState)`
- an already-readable name (`ampere_sgemm_128x64_nn`) → status=-2, left as is

### 4.4 The ActivityType for runtime events (a decision with a verification gate)

The kineto enum offers two candidates: `CUDA_RUNTIME` (→ `cuda_runtime`) and
`PRIVATEUSE1_RUNTIME` (→ `privateuse1_runtime`, provided officially for custom backends). Both
strings exist in `libtorch_cpu.so`.

**Start with `PRIVATEUSE1_RUNTIME`** (semantically accurate and vendor-neutral). The first
implementation step measures three things:

1. flow arrows appear in the chrome trace
2. `aten::mm` in `key_averages` has `self_device_time_total > 0`
3. the category name is `privateuse1_runtime`

**If any of them fails → switch to `CUDA_RUNTIME` and re-measure.** The acceptance script passes
on either of `{"cuda_runtime", "privateuse1_runtime"}`.

Source of risk: torch's post-processing classifies via `kineto_event.device_type()`, and that
mapping lives on the C++ side. Whether `PRIVATEUSE1_RUNTIME` is recognized as a device runtime
that participates in correlation attribution must be measured.

## 5. The acceptance gate

### 5.1 A frozen baseline snapshot

`tests/data/profiler_cuda_baseline.json`, with a regeneration script (refreshed manually when
torch is upgraded):

```json
{
  "categories": ["cpu_op", "cuda_runtime", "kernel", "gpu_memset", "ac2g", "cpu_instant_event"],
  "kernel_arg_keys": ["grid", "block", "correlation", "device", "stream",
                      "registers per thread", "shared memory", "External id",
                      "queued", "context", "blocks per SM", "warps per SM",
                      "est. achieved occupancy %"],
  "flow_cat": "ac2g",
  "runtime_cat_equivalents": ["cuda_runtime", "privateuse1_runtime"],
  "generated_by": "torch 2.10.0+cu128 / A100",
  "note": "structure only; timing values are not compared"
}
```

`categories` records the **torch-cuda side verbatim** (which is necessarily `cuda_runtime`). When
comparing, the runtime entry is treated as an equivalence class per `runtime_cat_equivalents` —
flagos producing either `privateuse1_runtime` or `cuda_runtime` satisfies it (see §4.4). All other
category names must match exactly.

CI runs only the flagos side and asserts against the snapshot. **The baseline is generated in the
`torch-cuda-210` environment** (which exists and reports `torch.cuda.is_available()` as True).

### 5.2 Assertions

1. The category set covers the baseline (the runtime category name may be either variant)
2. Flow arrow count > 0, with op↔kernel pairing
3. All 13 kernel args fields present
4. Operators such as `aten::mm` in `key_averages()` have `self_device_time_total > 0`
5. Kernel names are demangled (no `_ZN` prefix)

### 5.3 CI integration (mandatory)

The existing profiler tests carry **no pytest marker at all**, and `grep profiler .github/` returns
nothing — the same class of problem as `test_rng_dispatch.py`'s "the file exists but CI never
selects it", only more complete.

The new tests must:

- carry the `main_ops` marker (the CI selector in `.github/configs/cuda.yml`)
- be wired explicitly into the cuda workflow
- be confirmed by CI logs to have actually run once landed (a local green run is not enough)

## 6. Error handling and degradation

| Situation | Behavior |
|---|---|
| CUPTI unavailable (no GPU / dlopen fails) | `MakeDeviceTracer()`'s tracer reports `available()` false and is not registered with kineto; CPU-side profiling is unaffected |
| Record layout mismatch | Caught by the implemented self-check: drop the record plus a one-time diagnostic naming the bound library and API version |
| `getLinkedActivity` returns null | That event gets no `linked`/`flow` but is still emitted as a standalone GPU event; no data is lost |
| A kineto version that never calls the four-argument `processTrace` | The base class forwards to the single-argument version by default, degrading to no flow without crashing |

### Version decoupling already completed (commit `a2296e0`)

CUPTI binding priority: `FLAGOS_CUPTI_LIBRARY` → an already-loaded in-process CUPTI → soname
fallback. The override must be checked **first**: the scenario that prompts a user to set it is
precisely "the preloaded CUPTI does not resolve", so checking it only when nothing is preloaded
would make it dead in the one situation it is recommended for.

Before decoding a record, a sanity self-check runs (`end >= start`, non-zero start, duration
< 1 hour, name readable and non-empty) — four properties any normal record satisfies on any CUPTI
version, encoding no cu12-specific knowledge.

Verified: both pip cu12 CUPTI (API 26) and system CUDA-13 (API 130000) pass 21/21 across the two
major versions; forcing the threshold to 1ns rejects everything, emits one diagnostic, and does
not crash.

## 7. Delivery order

1. **The verification gate**: collect RUNTIME with `PRIVATEUSE1_RUNTIME` and build flows via the
   four-argument `processTrace`, then measure the three metrics; fall back to `CUDA_RUNTIME` if
   the bar is not met. This step determines the foundation for everything after it.
2. Extract the `device_tracer.h` interface and move the existing CUPTI logic into
   `cupti_device_tracer.cc`.
3. Complete the generic layer: MEMSET collection, the 13 metadata fields, demangle, and
   DeviceInfo/ResourceInfo for multiple devices.
4. Delete `flagos_profiler_stubs.cc` and the tests it skipped.
5. Generate the baseline snapshot, the regeneration script, and the comparison test (with the
   `main_ops` marker).
6. Wire it into `.github/configs/cuda.yml` and confirm via CI logs that it actually runs.
7. qwen3 real-model regression: confirm the trace has complete op↔kernel links and operator-level
   device time.
