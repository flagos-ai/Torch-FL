# torch_fl profiler architecture: parity with torch-cuda

> Completed: 2026-08-04
> NVIDIA measurement: A100-SXM4-40GB; torch 2.11.0+cpu + external libtorch_cuda.so (cu128)
> MetaX measurement: C550; torch 2.10.0 MetaX wheel + MACA 3.8.0 MCPTI, boxing mode
> Reference baseline: torch 2.10.0+cu128, see `tests/data/profiler_cuda_baseline.json`
> Tests: `tests/integration/test_profiler_parity.py`, 7 structural assertions

The MetaX measurement passed all seven parity assertions. It is a local hardware
validation rather than a CI result; the compatibility matrix therefore records MetaX
profiler support as experimental until a vendor runner is available.

The Chrome trace that `torch.profiler.profile(activities=[CPU, PrivateUse1])` produces for
flagos devices is **structurally** equivalent to torch+cuda's:

- **Flow arrows** (`ac2g`) connect each CPU op to the device kernel it launched
- **Device time attribution**: `prof.key_averages()` reports a per-op `self_device_time_total`
- **Complete kernel metadata**: 13 fields, including grid/block, occupancy, shared memory,
  register count
- **Runtime events** carry real API names (decoded from cbid, not a hardcoded placeholder)
- **memcpy / memset** are collected alongside kernels

This document has two audiences. If you are bringing up the profiler on new hardware, read
§1 on the three-layer architecture. If you are touching correlation-related code, read §2 on
the two correlation-id schemes — **that is the most common bug in this codebase, and it
fails silently**.

---

## 1. Three-layer architecture

The entire Task 2–4 refactor had one goal: **adding a vendor should mean writing one file**.

```
csrc/profiler/
  device_tracer.h              ← vendor-agnostic interface (DeviceTracer / DeviceEvent / EventKind)
  cupti_device_tracer.cc       ← CUPTI/MCPTI implementation; vendor activity types live here
  cann_device_tracer.cc        ← Ascend implementation using public MSPTI activities
  musa_mupti_device_tracer.cc  ← MUSA implementation using MUPTI activities
  roctracer_device_tracer.cc   ← ROCtracer implementation for DCU
  unavailable_device_tracer.cc ← explicit no-device-activity fallback
  gcu_topspti_device_tracer.cc ← GCU implementation using TOPSPTI activities
  flagos_kineto_profiler.{h,cc}← generic kineto adaptor; zero vendor coupling
  cupti_shim.h                 ← dlopen binding for CUPTI-compatible activity libraries
  mupti_shim.h                 ← optional runtime binding for MUSA libmupti
  topspti_shim.h               ← optional runtime binding for GCU libtopspti
```

### 1.1 Vendor-agnostic interface — `device_tracer.h`

The complete contract every vendor must satisfy:

```cpp
enum class EventKind { Kernel, Memcpy, Memset, Runtime };

struct DeviceEvent {
  EventKind kind;
  uint64_t start_ns, end_ns;
  uint32_t correlation_id;                         // CUPTI id: pairs runtime↔kernel
  std::optional<int32_t> external_correlation_id;  // torch id: drives device time attribution
  uint32_t device, stream, thread_id;
  std::string name;                                // already demangled
  std::map<std::string, std::string> metadata;     // "grid" → "[8,16,5]", etc.
};

class DeviceTracer {
  virtual bool available() const = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::vector<DeviceEvent> drain() = 0;
  virtual void pushCorrelation(uint64_t id) {}
  virtual void popCorrelation() {}
  virtual int deviceCount() const = 0;
};

std::unique_ptr<DeviceTracer> MakeDeviceTracer();   // factory
```

`EventKind` / `DeviceEvent` / `DeviceTracer` are the **only** three types the kineto adaptor
knows about. Everything vendor-specific (CUPTI activity records, enum values, struct layout
mirrors) sits below this line.

Note that `external_correlation_id` is a `std::optional`, not a plain `int32_t`: **0 is a
valid torch correlation id**, so using 0 to mean "absent" would silently attribute device
time to the wrong operator.

### 1.2 NVIDIA implementation — `cupti_device_tracer.cc`

Every CUPTI/MCPTI type, callback-ID table, and activity-record layout mirror appears only in
this file:

- `CuptiTracerInit` (file-level static) — arms CUPTI at module load; see §3.1
- `bufferRequested` / `bufferCompleted` — CUPTI activity buffer callbacks
- `CuptiDeviceTracer::processBuffer()` — decodes CUPTI activity records into `DeviceEvent`s
- `cuptiActivityPushExternalCorrelationId` / `Pop...` — correlation push/pop
- Kernel name demangling (`abi::__cxa_demangle`)
- The 13 kernel metadata fields, matching torch-cuda exactly:
  `grid`, `block`, `registers per thread`, `shared memory`, `warps per SM`,
  `blocks per SM`, `est. achieved occupancy %`, `queued`, `context`, `stream`,
  `device`, `correlation`, `External id`
- The single definition of `MakeDeviceTracer()`, at the end of the file

### 1.3 Generic kineto adaptor — `flagos_kineto_profiler.{h,cc}`

**Zero vendor coupling**: this file includes no CUPTI header and mentions no vendor-specific
type. It does exactly two things — translate `DeviceEvent` into
`libkineto::GenericTraceActivity`, and wire up correlation and flows.

- `FlagosKinetoProfiler` / `FlagosKinetoProfilerSession` implement kineto's
  `IActivityProfiler` / `IActivityProfilerSession`
- **The four-argument `processTrace` must be overridden** (`ActivityLogger&`,
  `getLinkedActivityCallback`, `startTime`, `endTime`). Override only the single-argument
  form and kineto never hands you the resolver, so flows are permanently zero — which is
  exactly the state this project started in.
- **Capture-window filtering**: discards events outside `[startTime, endTime]`
  (Task 1 finding 5); see §3.2
- **Flow arrows**: `activity.flow.id = correlation_id`, with `flow.start = 1` on runtime
  events and `flow.start = 0` on device events. Both halves must carry the **same id** (the
  vendor correlation id) or no viewer can draw the arrow.
- **Device time linking**: `activity.linked = getLinkedActivity(*external_correlation_id)` —
  keyed on the **torch** correlation id, not the CUPTI one. Confusing the two: see §2.

### 1.4 What adding a new vendor takes

1. Write `csrc/profiler/{vendor}_device_tracer.cc`, implement `DeviceTracer`, and define
   `MakeDeviceTracer()` in it.
2. Add the tracer to the per-accelerator source selection in `csrc/CMakeLists.txt`. The
   build uses `GLOB_RECURSE`, so exactly one tracer factory must be compiled for every
   accelerator: CUPTI/MCPTI for CUDA, MetaX, and PPU; ROCtracer for DCU; and the
   unavailable tracer for platforms without a supported activity API.
3. Done. The kineto adaptor needs no vendor-specific changes.

---

## 2. The two correlation-id schemes (the important section)

**This is the most common profiler bug in this codebase, and it produces no error.**

The trace carries two entirely independent numbering schemes. They look alike, have the same
type, and both are called "correlation" — but they mean different things:

| | `correlation_id` | `external_correlation_id` |
|---|---|---|
| Whose id | **CUPTI**'s | **torch**'s |
| Source | carried on the CUPTI activity record | resolved from CUPTI `EXTERNAL_CORRELATION` records (kind 39) |
| Pairs what | a runtime call ↔ the device kernel it produced | a device/runtime activity ↔ the CPU op that issued it |
| Used for | drawing **flow arrows** | **device time attribution** |
| Trace field | `args["correlation"]` | `args["External id"]` |
| Where in code | `activity.flow.id` | the argument to `getLinkedActivity()` |

A measured pair, taken from a local trace:

```
runtime 'cudaLaunchKernel'         correlation=80  External id=3
kernel  'ampere_sgemm_128x64_nn'   correlation=80  External id=3
flow halves id=80: [('s', 'ac2g'), ('f', 'ac2g')]     ← paired by correlation
aten::mm's self_device_time_total  ← attributed by External id
```

### Why there have to be two

They count different things, and neither can be derived from the other. The CUPTI id
identifies "one runtime call and the device activity it produced"; the torch id identifies
"one `aten::*` operator". An operator typically issues many CUPTI-visible calls, so the
relationship is **one torch id to N CUPTI ids**: on one parity-workload trace, a single
`aten::mm`'s External id covered **69** distinct CUPTI correlation ids, and the fan-out
across the whole trace ranged from 1 to 69. CUPTI's `EXTERNAL_CORRELATION` record (kind 39)
is the only bridge between the two schemes.

### What happens when you pass the wrong one

**Passing the CUPTI id to `getLinkedActivity()` raises no error; it just silently zeroes
`self_device_time_total`.** Task 1 proved this by ablation: suppressing the single
`activity.linked` line and changing nothing else dropped `aten::mm`'s device time from 816µs
to 0µs, while the trace still generated normally and the kernel timeline stayed correct.

The reverse fails just as silently: keying flows on the torch id yields only unpaired `f`
halves, and no viewer draws a single arrow. Historically this produced 203 dangling `f`
halves — and the `count > 0` assertion of the day passed anyway.

So assertions 2 and 4 in `test_profiler_parity.py` pin down these two paths separately, and
neither is a "greater than zero" check: the flow assertion requires **every id to be
paired**, and the device-time assertion requires the value to **reconcile** with the sum of
device-event durations computed independently from the trace.

---

## 3. Implementation notes

### 3.1 The CUPTI arming constraint

`cuptiActivityRegisterCallbacks` **must be called before the first CUDA context is created**.
This is a measured conclusion (memory: `cupti-must-arm-before-cuda-context`): register after
a context already exists and the buffer callbacks are never invoked, so CUPTI collects zero
records. Registration therefore lives in the file-level static `CuptiTracerInit` in
`cupti_device_tracer.cc`, running when `import torch_fl` loads the shared library — before
any device operation.

But **which activity kinds get armed is split across two moments**:

| Armed at static init | Armed at session `start()` |
|---|---|
| `CONCURRENT_KERNEL`, `MEMCPY`, `MEMSET` | `RUNTIME`, `EXTERNAL_CORRELATION` |

This split is **a performance measurement, not a constraint**. `RUNTIME` instruments the
entry and exit of every CUDA runtime API call; arming it at import time measurably slowed a
launch-bound workload by **22%: 10.56 → 12.88 µs/op** (3 A/B pairs; the ~5ms delta against
run-to-run jitter of only ~0.1ms, and variance grew about 15× once armed). For users who
merely `import torch_fl` and never profile, that is a real regression. Deferring these two
kinds to `start()` returns it to **10.48 µs/op**, back inside the noise, with no loss of
collection coverage.

The constraint the memory records is that **callback registration** must precede the first
CUDA context — deferring kinds still satisfies it — not that every kind must be armed at
import.

**A trap worth remembering**: a GPU-bound workload (dominated by matmul rather than launches)
is entirely insensitive to this timing; the two approaches are indistinguishable there.
**Benchmark only a GPU-bound workload and you will wrongly conclude that import-time arming
is free.**

### 3.2 Capture-window filtering

The C++ predicate at `flagos_kineto_profiler.cc:266` is an **interval-overlap** test:

```cpp
auto in_window = [startTime, endTime](const profiler::DeviceEvent& ev) {
  return static_cast<int64_t>(ev.end_ns) >= startTime &&
         static_cast<int64_t>(ev.start_ns) <= endTime;
};
```

What happens without it: the tracer's device kinds are armed at import and keep recording for
the whole process lifetime, so activity from before the profiled region leaks into the trace
— measured, 66 of 258 runtime events ended before the first cpu_op, including one 250ms
record inside a 157ms span.

Parity assertion 7 (`test_capture_window_containment`) asserts something stronger — **strict
containment** (`ts >= lo && ts + dur <= hi`), the same semantics as libkineto's own
`outOfRange`. What justifies the stronger form: across 3 captures of 271 tracked events each,
zero violations, with the tightest margin about 1.0–1.7ms on the leading edge and about 30µs
on the trailing edge; that trailing 30µs is structural (the closing
`cudaDeviceSynchronize` necessarily precedes profiler stop).

If this ever does fail because of a small overrun at the **leading** edge, that is a
**finding** — activity genuinely in flight at profiler start — and should be reported. Do not
quietly weaken the assertion back to overlap semantics.

### 3.3 Timestamp clock domain

`cuptiGetTimestamp()` and kineto's `[startTime, endTime]` are **both UNIX epoch nanoseconds**
(measured: `cuptiGetTimestamp` differs from `CLOCK_REALTIME` by 2.7µs). Reading either side
with MONOTONIC or BOOTTIME instead puts them ~1.78e18 ns apart (about 56 years), and the
window filter discards everything. So the comparison in §3.2 is dimensionally sound: if a
containment check fails, suspect the logic or the window itself, not the clock domain.

### 3.4 JSON quoting for metadata

kineto's `GenericTraceActivity::addMetadata` always stores values with `quoted=false`, and
`metadataJson()` emits `"key": <raw>` directly. One bare identifier down that path (a kernel
name, a memory-kind label, `N/A`) produces `"key": N/A` — not valid JSON. And because kineto
concatenates every activity into **one document**, **a single event can make the entire trace
file fail `json.load()`**, a blast radius wildly out of proportion to the mistake.

`metadataValueIsJsonLiteral()` decides from the value's textual form whether it can be
emitted raw; everything else goes through `addMetadataQuoted`. The check is deliberately
conservative: over-quoting a number merely renders it as a string (cosmetic), while
under-quoting a non-literal destroys the whole file (fatal). Exponent notation must be
allowed through — `memory bandwidth (GB/s)` is formatted with `%g` to match torch-cuda
byte-for-byte, which emits `8e-05` at small magnitudes.

---

## 4. Debug environment variables

Both default to off. Note that they test for **being set**, not for a value, so
`FLAGOS_KINETO_SHIM_DEBUG=0` still enables logging; unset it to turn it off.

- **`FLAGOS_KINETO_SHIM_DEBUG`** — diagnostics for the kineto adaptor
  (`flagos_kineto_profiler.cc`): how many events were drained at session stop; the window
  `processTrace` received, linked/candidate counts, and how many entries the window
  discarded; profiler registration and `configure()` calls.

- **`FLAGOS_CUPTI_SHIM_DEBUG`** — diagnostics for the CUPTI tracer
  (`cupti_device_tracer.cc`) and the dlopen shim (`cupti_shim.h`): which `libcupti` was bound
  and its API version; callback registration and the `ActivityEnable` return value for each
  kind; buffer requests and completions, and per-record decoding.

**Two warnings are deliberately not gated** by either switch: an empty `getLinkedActivity`
callback, and an activity-record layout mismatch in the tracer. Both are rare but severe (the
first silently zeroes device time), and silence is precisely what makes them hard to find, so
they print unconditionally.

---

## 5. The parity test and its baseline

**Test**: `tests/integration/test_profiler_parity.py`, 7 assertions. All of them assert
**structure** only, never counts or durations — this is a shared GPU where both drift
run-to-run, so "exactly 18 arrows" is inevitably flaky while "every arrow is paired" is not.

| # | Assertion | What it checks |
|---|---|---|
| 1 | `test_category_coverage` | flagos emits every category torch-cuda does (with an equivalence mapping for the runtime category) |
| 2 | `test_flow_arrows_are_paired` | every `s` half has an `f` half with the same id (renderable, not merely present) |
| 3 | `test_arg_key_supersets` | each category's `args` keys are a superset of the baseline's (added fields are not a break; missing ones are) |
| 4 | `test_device_time_attribution` | `aten::mm`'s `self_device_time_total` equals the summed duration of the device events it owns |
| 5 | `test_kernel_names_are_demangled` | no bare C++ mangled symbols (`_ZN...`) |
| 6 | `test_runtime_names_come_from_cbid` | at least one runtime name goes beyond the generic fallback, proving the cbid table is in effect |
| 7 | `test_capture_window_containment` | no device/runtime event escapes the capture window; see §3.2 |

**Baseline**: `tests/data/profiler_cuda_baseline.json`, captured by
`tests/data/gen_profiler_baseline.py` on native torch+cuda (2.10.0+cu128). It holds only
categories, args keys, and notes on known gaps — **no counts and no durations**.

**Regenerating the baseline**:

```bash
conda activate torch-cuda-210      # must be a real-CUDA torch, not torch-fl-211
python tests/data/gen_profiler_baseline.py
```

The script takes no arguments and writes back to `tests/data/profiler_cuda_baseline.json`; it
refuses to run when `torch.cuda.is_available()` is false. **It cannot run under
`torch-fl-211`**: the two environments have incompatible libc10 ABIs, and importing torch_fl
there would rob the baseline of its meaning (the baseline must come from native torch, not
from the implementation under test).

**The baseline must be refreshed when torch is upgraded.** The generator's workload must also
stay in sync with the test's (`gen_profiler_baseline.py::run_traced_ops()` versus
`_run_traced_ops()` in the test), or the two traces are not comparable.

**How CI runs it** (7 cases, about 2 seconds on an A100; all 7 share one module-level fixture
so capture happens once):

```bash
PYTHONPATH=$(pwd) bash scripts/with_cuda_libtorch.sh \
    python -m pytest tests/integration/test_profiler_parity.py -v -m main_ops
```

---

## 6. Known gaps (recorded honestly, not papered over)

### 6.1 The `overhead` category is not collected

flagos never enables `CUPTI_ACTIVITY_KIND_OVERHEAD`, so CUPTI's self-reported profiling
overhead ("Activity Buffer Request", "Runtime Triggered Module Loading") never appears. These
measure the cost of profiling itself rather than the user's workload, so their absence does
not affect any measurement of that workload. Recorded together with the reason under
`categories_known_gap` in the baseline JSON, rather than quietly omitted.

### 6.2 MetaX MCPTI compatibility

The MetaX implementation uses the CUDA-compatible MCPTI activity API exposed by
MACA 3.8.0. MCPTI's runtime callback IDs are not NVIDIA CUPTI IDs, so the tracer
uses the MetaX callback namespace and defers `mcptiActivityGetApiName` calls until
after activity flushing. Calling that resolver from the MCPTI buffer callback can
deadlock the profiler.

MCPTI 3.8.0 can also return a stale iterator pointer after external-correlation
records. The MetaX path therefore scans the aligned vendor records independently,
advancing by the complete MCPTI record sizes and resynchronizing on malformed
candidates. Records with `start == end == 0` are skipped because MCPTI documents
that pair as the "timing unavailable" sentinel.

The validated workload passed all seven parity assertions on a C550 with the
MetaX 3.8.0 SDK. The test covered kernel, memcpy, memset, and runtime records,
flow pairing, external-correlation device-time attribution, kernel metadata,
callback-ID names, and capture-window filtering. The scanner has not yet been
validated across multiple MetaX SDK versions, devices, or non-default streams.

### 6.3 The flow-pairing assertion is stricter than torch-cuda itself

Parity assertion 2 requires **every** flow arrow to be paired. **torch-cuda does not satisfy
that** — same workload, 3 consecutive runs, fully reproducible:

```
torch-cuda: ac2g s=26  f=78  paired=False
flagos    : ac2g s=24  f=24  paired=True
```

torch-cuda's 52 surplus `f` halves hang off `cuda_runtime` events with no originating `s`
(`cudaDeviceGetAttribute`, `cudaMalloc`, `cudaOccupancyMaxActiveBlocks`, and similar). flagos
emits exactly one paired arrow per device event.

**So do not describe this assertion as "parity with torch-cuda"**: it is a flagos-specific
invariant, stronger than the baseline. It is kept because flagos genuinely satisfies it, and
regressing into dangling halves is precisely the bug it guards against.

---

## 7. Related material

- memory `cupti-must-arm-before-cuda-context` — how the callback-registration timing
  constraint was established
- memory `torch-fl-211-env` — environment setup and build for this branch (CPU torch +
  external libtorch_cuda.so)
- `docs/superpowers/specs/2026-08-03-profiler-cuda-parity-design.md` — the design document
  and the measured gap table from the start of the work


## Ascend CANN MSPTI integration

Ascend profiling is implemented in `csrc/profiler/cann_device_tracer.cc` using CANN's public MSPTI activity API. It does not import `torch_npu`, parse file-oriented `msprof` output, or add CANN types to the generic Kineto adapter. MSPTI supplies epoch-nanosecond kernel, runtime/API, memcpy, and external-correlation records through asynchronous activity buffers; the tracer also decodes memset records when a CANN release emits them.

`torch_fl` does not load `libmspti.so` during import. The C++ tracer resolves it lazily when a profiler session starts, so ordinary Ascend operator processes do not install the CANN interposer or pay its lifecycle cost. Kernel and runtime/API records work under that lazy load.

**Memcpy and memset are different**: CANN implements them by symbol interposition, so `libmspti.so` must already be in the ELF link map when `libascendcl.so` resolves those calls. A later `dlopen` cannot substitute, no matter how early it runs — measured with a standalone C program (no torch involved):

```text
no LD_PRELOAD, dlopen(mspti) then dlopen(ascendcl):  gpu_memcpy=0 gpu_memset=0
LD_PRELOAD=libmspti.so:                              gpu_memcpy=3 gpu_memset=3
```

This is a *process startup* requirement. `.github/scripts/set_env_ascend.sh` therefore discovers the library next to every other Ascend startup prerequisite, but stores the prepared preload under the inert `ASCEND_MSPTI_PRELOAD` variable. The common integration runner supports a per-test `environment` mapping and expands that value into `LD_PRELOAD` only for the profiler contract process. The contract is still invoked with the identical command on every platform:

```bash
python -m pytest tests/integration/test_profiler_contract.py -v -m profiler
```

This narrow scope is required for correctness, not merely tidiness. A CI run that exported MSPTI job-wide produced sporadic `SIGSEGV` (`-11`) exits in otherwise healthy add, embedding, le, mean, and mm subprocesses before the profiler test was reached. CANN 9.0's interposer is therefore not safe to impose on arbitrary operator processes. Attaching the prerequisite as structured per-test environment data preserves the uniform command without destabilizing unrelated tests.

Outside CI, the equivalent remains an environment export before starting the profiled Python process:

```bash
export LD_PRELOAD="$ASCEND_HOME/tools/mspti/lib64/libmspti.so${LD_PRELOAD:+:$LD_PRELOAD}"
python workload.py
```

The environment script skips the prepared preload when the file is absent, so a CANN image without the profiling tools does not get an `LD_PRELOAD` that `ld.so` can only warn about.

If the library is absent, another MSPTI subscriber already owns the process, or activation fails, CPU profiling continues normally. MSPTI shutdown follows the measured order: unsubscribe first, then flush pending activity buffers; the tracer deliberately keeps the process-level MSPTI library loaded until process exit because CANN may retain interposed ACL function pointers.

MSPTI correlation IDs are remapped to nonzero 32-bit IDs for Kineto flow arrows. External-correlation records retain torch's profiler correlation ID separately, which enables linked ATen device-time attribution. Runtime records are coalesced per device correlation, preferring launch APIs for kernels and the corresponding copy/set API for transfers. Physical CANN device IDs are mapped to logical ordinals through `ASCEND_RT_VISIBLE_DEVICES` when set.

The public CANN kernel record has no CUDA grid/block/occupancy fields; those fields are omitted rather than fabricated. CANN 9.0 declares `MSPTI_ACTIVITY_KIND_MEMSET` and emits real device records when the ACL memset symbols are resolved through the process-start MSPTI interposer. A direct global-symbol probe using `ctypes.CDLL(None)` produced positive-duration records for both `aclrtMemset` and `aclrtMemsetAsync`, including byte count, fill value, async flag, stream, device, and correlation metadata. A prior probe that called symbols through a separately loaded `libascendcl.so` handle bypassed the interposer and produced a false negative; it is not a valid capability test. Other CANN releases may remain unavailable until their MSPTI headers and runtime behavior are validated.

The shared cross-backend profiler contract (`tests/integration/test_profiler_contract.py` and `tests/integration/profiler_support.py`) exercises this on Ascend, but its memcpy and memset assertions are not symmetric. Both were measured on hardware against the same shared `profile_result()` workload (a small matmul+relu loop, a sort, and a host-to-device copy):

- `test_profiler_memcpy_events` is gated on `mspti_preload_active()`, which checks `/proc/self/maps` for the actually loaded `libmspti.so` rather than trusting the `LD_PRELOAD` string. When the environment script has made the library available before process start, the workload's host-to-device copy produces a real positive-duration `gpu_memcpy` record; a missing or invalid library causes the test to skip explicitly instead of claiming a capability it does not have. The link map is sampled once at `profiler_support` import, which is the only moment that answers the question: the module is loaded as a pytest plugin before anything starts a profiler session, so a hit there can only come from a process-start preload. Sampling later would also see the lazy `dlopen` in `CannDeviceTracer::start()`, which does *not* enable interposition — measured on 910, a late check reports "preloaded" and then the memcpy assertion fails on a workload that produced no records.
- `test_profiler_memset_events` stays disabled for Ascend regardless of preload. The direct `ctypes` probe above proves the CANN interception path itself works: `aclrtMemset`/`aclrtMemsetAsync` produce real `gpu_memset` records when called explicitly under preload. But nothing reachable from the shared workload calls that allocator path. `torch.zeros()` -- the only zeroing op the workload exercises -- routes to the `aclnnInplaceZero` kernel (`csrc/aten/backends/ascend/generated/ascend_kernels.cc`), not to the allocator's `aclrtMemset`/`aclrtMemsetAsync` calls in `csrc/runtime/accelerator/ascend/memory.cc`. Switching it only to make a profiler record appear would regress measured 910 latency from 12.7us to 133us at 1 MiB and from 17.4us to 9080us at 64 MiB (`aclrtMemsetAsync` is slower still). The high-performance kernel routing is therefore intentional, and the absent `gpu_memset` record is a correct-by-design capability difference.

Every CI backend runs this contract with the same command. `.github/configs/ascend.yml` carries no shell prefix; its structured `environment` field scopes the prepared preload to this process, and `.github/scripts/run_integration_tests.py` applies it without changing the command string.

`import torch_fl` deliberately does not re-exec the process to install the preload. Doing so would disturb file descriptors, multiprocessing, `torchrun`, and debuggers, and exporting the interposer job-wide demonstrably destabilizes unrelated CANN operator processes. Per-process integration environment data is the safe boundary at which to express this startup requirement.

For hardware support reporting, the Ascend memcpy capability is measured only when MSPTI was actually loaded. An `LD_PRELOAD` environment variable by itself is not evidence of that; `ld.so` warns and continues for nonexistent paths.


## MUSA MUPTI integration

MUSA uses `csrc/profiler/musa_mupti_device_tracer.cc` and the optional `mupti_shim.h`. The tracer
keeps MUPTI types and record decoding below the generic `DeviceTracer` boundary, and CMake
excludes CUPTI, ROCtracer, and MSPTI for `ACCELERATOR=musa`, leaving exactly one factory.
`muptiActivityRegisterCallbacks` and the activity kinds are armed at session start, not shared
library import, so ordinary MUSA execution does not load the profiler. The buffer callbacks decode
`MUpti_ActivityKernel6`, `MUpti_ActivityMemcpy4`, `MUpti_ActivityMemset3`, `MUpti_ActivityAPI`,
and `MUpti_ActivityExternalCorrelation`; invalid timestamps are dropped rather than emitted as
corrupt trace records. Kernel grid/block, shared-memory, register, context, stream, transfer-byte,
callback-name, and correlation metadata are copied before the MUPTI buffer is released.

MUPTI and Kineto use different timestamp clock domains on the validated host, so the tracer samples
`muptiGetTimestamp()` against `CLOCK_REALTIME` at session start and stop and maps activity times
with an affine conversion. The two correlation schemes remain independent: the MUPTI correlation
ID pairs runtime and device records for `ac2g` flows, while `CUSTOM0` external records map to the
Torch profiler ID used by `getLinkedActivity()`. `FLAGOS_MUPTI_LIBRARY` overrides library lookup,
and `FLAGOS_MUPTI_DEBUG=1` enables setup diagnostics. The MTT S5000 validation captured real
positive-duration kernel, runtime, and memcpy records and valid Chrome JSON; CPU-only Kineto
resolver behavior remains environment-dependent, so full profiler parity is not claimed.

## Enflame GCU TOPSPTI integration

GCU uses `csrc/profiler/gcu_topspti_device_tracer.cc` and the optional `topspti_shim.h`. The
Enflame TopsRider SDK ships a CUPTI-shaped tracing interface in
`/opt/tops/extras/TOPSPTI` (`libtopspti.so`), which delivers kernel, memcpy, memset, runtime,
and driver records through asynchronous activity buffers. The tracer keeps every TOPSPTI type
below the generic `DeviceTracer` boundary, and CMake excludes CUPTI, ROCtracer, MSPTI, and
MUPTI for `ACCELERATOR=gcu`, leaving exactly one `MakeDeviceTracer()` factory.
`topsptiActivityRegisterCallbacks` and the activity kinds are armed when a Kineto session
starts, not at shared-library import, so ordinary GCU operator processes never load the
vendor profiler. Records with implausible timestamps are dropped rather than emitted as
corrupt trace entries, and kernel grid/block, context, stream, byte-count, fill-value,
callback-name, and correlation metadata are copied before the buffer is released.

TOPSPTI reports its own device clock, so the tracer samples `topsptiGetTimestamp()` against
`CLOCK_REALTIME` at session start and stop and maps activity times with the same affine
conversion the MUPTI path uses. `FLAGOS_TOPSPTI_LIBRARY` overrides library lookup and
`FLAGOS_TOPSPTI_DEBUG=1` enables setup diagnostics. When the SDK headers are missing at build
time or the runtime library cannot be resolved, the GCU tracer compiles and reports as an
unavailable stub and CPU-only profiling continues normally.

**The GCU correlation path differs from every other vendor here, and §2 applies directly.**
TOPSPTI exposes no equivalent of `cuptiActivityPushExternalCorrelationId`, so there are no
external-correlation activity records to decode. The vendor `correlationId` still pairs a
runtime record with the device record it produced, which is what `ac2g` flow arrows need. To
recover the *torch* id that drives device-time attribution, the tracer records the Kineto
session's current correlation (a thread-local set by `pushCorrelation`/`popCorrelation`)
inside the TOPSPTI API-enter callback, keyed by vendor correlation id, and resolves the
mapping in `drain()`. Device timing therefore does not depend on that callback: if a TOPSPTI
release does not deliver runtime callbacks, kernels are still captured, and only
`External id` linkage is absent.

This integration is verified at the build level only: the tracer compiles against the
installed TOPSPTI headers, against a no-SDK GCU configuration, and in the non-GCU
configuration, and CMake detects `/opt/tops/extras/TOPSPTI/include`. It has **not** been run
on physical GCU hardware, so no captured-activity, flow-arrow, or device-time claim is made
for this platform; `tests/integration/test_profiler_gcu.py` is the gate that would establish
those, and it skips without a GCU device.
