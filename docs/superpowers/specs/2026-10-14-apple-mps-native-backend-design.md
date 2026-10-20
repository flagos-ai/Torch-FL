# Apple M-series (Metal/MPS) backend for torch-fl — design document

Date: 2026-10-14
Status: Accepted. Operator strategy is decided (§6): MPSGraph-first with an MSL
fallback, delivered stage by stage per §7.

## 1. Goal

Add Apple M-series GPUs (Apple Silicon: M1/M2/M3/M4) as a supported `flagos`
accelerator. A Mac with Apple Silicon should be able to run ordinary PyTorch
eager code on the `flagos` device, the way the CUDA, Ascend, GCU and MUSA
platforms do, by building a macOS arm64 wheel with `FLAGOS_ACCELERATOR=mps`.

The backend is **native**. torch-fl's own device runtime and its own operator
kernels talk to Apple's Metal stack directly. It reuses neither the CUDA runtime
sources nor the CUDA boxing machinery.

## 2. Non-goals

- **No boxing.** Boxing is explicitly ruled out and is on the path to removal
  from the tree. Concretely: no `DeviceBoxingGuard` /
  `SetTensorImplDevice` re-tag of `flagos` tensors onto another dispatch key, no
  re-dispatch onto `DispatchKey::MPS` or `DispatchKey::CUDA`, and no
  `cuda_kernels.cc` boxing backend for this platform. `FLAGOS_BUILD_BOXING` is
  pinned **off**.
- **No FlagGems / FlagTree / Triton.** There is no Metal backend in the Triton
  line torch-fl pins and no FlagTree build for Apple, so `FLAGOS_BUILD_FLAGGEMS`
  and `FLAGOS_BUILD_FLAGGEMS_CPP` are pinned off. Ops with no native kernel reach
  `cpu_fallback`.
- **No FlagCX / distributed and no profiler** in the first cut. Apple has no
  FlagCX build; the profiler needs a Metal/os_signpost tracer, which is a
  follow-up.
- **Apple GPUs only.** On a Mac without an MPS-capable GPU,
  `torch.flagos.device_count()` reports 0 rather than offering a device that
  fails on first use.
- **No `torch.compile`** initially.

## 3. Background

Two facts shape the design.

**The pattern already exists.** A non-CUDA native platform (`gcu`, `musa`,
`ascend`) is built from: a per-vendor runtime directory implementing the
28-function C ABI in `csrc/include/flagos.h` plus the 10 `DeviceMemoryInterface`
virtuals; a per-vendor aten kernel directory produced through the codegen
category system; a `Backend::k<Platform>` slot in `csrc/aten/dispatcher.h`; a
`torch_fl/configs/backends_<platform>.conf` routing table; and a
`writes_platform_marker` entry in `cmake/flagos_platforms.json`. Apple is an
instance of that pattern, not a new architecture. The bring-up order is the one
in `.claude/skills/runtime-bringup/SKILL.md`: get the floor green with zero
operators, then add kernels through the `native-op-backend` route.

**Apple exposes the GPU through Metal, and its operator libraries through MPS /
MPSGraph.** There is no C op library equivalent to ACLNN / topsaten / mudnn.
MPSGraph is the closest analogue — a graph API whose node set covers a large
fraction of elementwise, reduction, matmul, convolution and normalization ops —
and MPS provides standalone primitives (e.g. `MPSMatrixMultiplication`,
`MPSGraph` softmax). `libtorch`'s own MPS backend is built this way, which is
both the proof that the composition works and the reason re-implementing all of
it is a large project.

## 4. Why not the two cheaper-looking alternatives

### 4.1 Why not box to the MPS dispatch key

Re-tagging `flagos` tensors to `DispatchKey::MPS` and calling ATen's MPS kernels
is technically the least code, because it reuses the `mm`/`add`/`conv` kernels
`libtorch` already ships. It is also, precisely, boxing — the same device-metadata
rewrite the CUDA path uses, just aimed at a different dispatch key. It would
require a second box target threaded through `device_boxing.h` and
`scripts/codegen/codegen_ops.py`, which is the machinery being removed. It is
ruled out by §2.

### 4.2 Why not CPU fallback as the deliverable

`cpu_fallback` is the designated path for an *unrouted* op, and a platform whose
GPU never runs a kernel is a "Runtime only" entry in the matrix (the BPU
precedent), not accelerator support. CPU fallback with a working device runtime
is fine as an interim milestone (§7, stage 0) but not as the finished feature.

## 5. Architecture

### 5.1 Device runtime floor — `csrc/runtime/accelerator/mps/`

Same three-file split as every other backend (`device.cc` 5 functions,
`memory.cc` 9, `stream.cc` 14). Metal maps onto the contract as follows.

| `flagos.h` function | Metal implementation |
|---|---|
| `GetDeviceCount` | 1 when `MTLCreateSystemDefaultDevice()` returns an Apple GPU, else 0 |
| `GetDevice` / `SetDevice` | single index 0, `thread_local` (Apple has one GPU) |
| `DeviceGetStreamPriorityRange` | `MTLCommandQueue` has no priority knob; report 0/0 (BPU precedent) |
| `DeviceSynchronize` | drain the current command buffer(s) on the default queue and wait |
| `Malloc` / `Free` | `-[MTLDevice newBufferWithLength:options:]` retained in a side table; release on `Free` |
| `MallocHost` / `FreeHost` | `malloc` / `free` (unified memory; page-locking is not a separate concept) |
| `Memcpy` / `MemcpyAsync` | `memcpy` for shared buffers; a blit encoder on the stream's queue for private buffers |
| `Memset` / `MemsetAsync` | blit `fillBuffer` (or `memset` for shared buffers) |
| `PointerGetAttributes` | side table `ptr -> (size, device)`; `MemoryTypeDevice` |
| `StreamCreate*` | `-[MTLDevice newCommandQueue]` |
| `StreamGetPriority` | 0 |
| `StreamDestroy` / `StreamQuery` / `StreamSynchronize` | release / poll / `waitUntilCompleted` on the queue's committed buffers |
| `StreamWaitEvent` | encode a wait on a `MTLSharedEvent` |
| `EventCreate*` / `EventDestroy` / `EventRecord` / `EventSynchronize` / `EventQuery` | `MTLSharedEvent` with a monotonically increasing signal value |
| `EventElapsedTime` | Metal has no elapsed-time query; capture `MTLCommandBuffer.GPUStartTime`/`GPUEndTime` at record points and report their difference (the same technique the in-tree MPS work uses) |

Two behaviors from the skill are load-bearing and easy to get wrong:

- `StreamQuery` and `EventQuery` **must** return `ErrorNotReady` (not a generic
  error) while work is outstanding, or the caching allocator's block-reuse path
  misreads a live buffer as corrupt.
- `Memcpy` must **drain the current command buffer first** when ops dispatch
  asynchronously on the shared queue, or a `.cpu()` races the kernel that
  produces the data. Metal encodes work into a command buffer that is only
  committed at a synchronize point, so this is a real hazard here, not a
  theoretical one.

`MallocHost`/`FreeHost` and the allocator's `provides_caching()` follow the
defaults: return `false` from `provides_caching()` so the built-in block pool is
used. There is no mature Apple allocator to delegate to and no PrivateUse1 slot
collision to worry about.

### 5.2 Build and toolchain

- `cmake/flagos_platforms.json`: a new `mps` row with
  `project_languages: ["OBJCXX", "CXX", "C"]`, `runtime_dir: "mps"`,
  `use_macro: "USE_MPS"`, `writes_platform_marker: true`, `vendor_torch_libraries:
  false`, `c10_cuda_no_cmake_configure: true`, empty `wheel_local` and
  `pypi_lane` (no lane publishes an Apple wheel), and kernel defaults
  `vendor: true`, `boxing: false`, `flaggems: false`, `flaggems_cpp: false`,
  `tileops: false`.
- The runtime CMake glob is `*.cc` today; a `mps` arm must also pick up `*.mm`.
  Objective-C++ is required for the Metal/Foundation interop, so `OBJCXX` joins
  the `project()` languages for this platform only.
- Link Apple frameworks: `-framework Metal -framework Foundation` for the
  runtime; `-framework MetalPerformanceShaders -framework
  MetalPerformanceShadersGraph` for the operator layer (see §6).
- `setup.py`: a `mps` arm with no SDK discovery (there is no vendor SDK — the
  macOS torch wheel already carries MPS), no bundled libtorch, and no FlagTree
  pin. `python_requires` is not pinned to a vendor cp tag (no FlagTree to link).
- The wheel platform tag comes from the build host, so a build on macOS arm64
  produces a `macosx_*_arm64` wheel with no extra work.

### 5.3 Python wiring

- `torch_fl/_platform.py`: add `"mps": "mps"` to `_ACCELERATOR_TO_PLATFORM` and
  `"mps"` to `_CONF_PLATFORM_FRAGMENTS`.
- `torch_fl/accelerator/mps/` for the Python compat module (stream/event
  wrappers that `torch.flagos.Stream` dispatches to), mirroring the MUSA layout.
- The device name stays `flagos`; only the accelerator value and conf file are
  platform-specific.

## 6. Operator strategy — decided: MPSGraph-first, MSL fallback (Option A)

This is the one choice that changes the size of the project by an order of
magnitude. It is decided: **Option A**. The three options are recorded here so
the reasoning survives the choice.

**Option A — MPSGraph-first, MSL fallback (recommended).** Treat MPSGraph as the
"vendor operator library" the way Ascend treats ACLNN. `csrc/aten/backends/mps/`
wraps `MetalPerformanceShadersGraph` nodes; each op builds a one-node graph,
binds flagos-allocated `MTLBuffer`s through `MPSGraphTensorData`, runs it on an
`MPSGraphDevice`, and returns the flagos tensor. Ops MPSGraph cannot express
cheaply (odd dtypes, exotic layouts, small elementwise ops where graph build
cost dominates) are written as hand-authored MSL compute kernels compiled once at
startup with `newLibraryWithSource:` and cached. This reaches broad coverage
fastest and matches the repo's "reuse before reimplement" principle.

**Option B — pure MSL codegen.** No MPSGraph. Hand-write `.metal` kernels and
generate their C++ wrappers through a category system (`unary`, `binary`,
`reduce`, `gemm`, …) exactly like `codegen_ascend.py`. Highest control and best
small-op latency, but it is a re-implementation of a large slice of
`libtorch`'s MPS backend — the largest of the three options.

**Option C — runtime only, all ops on CPU.** Ship stage 0 alone: the device
runtime is real and `flagos` tensors allocate, copy and synchronize, but every op
reaches `cpu_fallback`. This is a genuinely useful first PR and a complete
"Runtime only" platform row, and it de-risks everything below it.

The recommended path is C first (it is stage 0 of both A and B and is verifiable
end to end on any Apple Silicon machine), then A, widening the op set by category
from `unary` outward and filling gaps with B.

**Marshalling contract (measured on an M5).** MPSNDArray is not a dense view of a
buffer once it has more than one dimension: it rounds the stride between rows up
to 16 bytes. A `(4, 5)` float32 operand, whose rows are 20 bytes, is therefore
read as four 32-byte rows and every row after the first comes from the wrong
offset — under the first implementation row 1 compared as
`[0, 0, 0, 0.4681, 0.1577]` against a torch row of
`[-1.5551, -0.3414, 1.853, 0.4681, -0.1577]`, and the same arithmetic is what
`initWithBuffer:offset:descriptor:` reports as "buffer is not large enough. Must
be 128 bytes" when a small buffer is passed. The pitch cannot be corrected from
the outside: `rowBytes` is not a public property of `MPSNDArrayDescriptor` and
reads back 0. Buffers are therefore always described to MPSGraph as a 1-D array
of `numel` elements and the aten shape is applied inside the graph, where a
reshape is a metadata operation (`MpsFlatPlaceholder`, `MpsReshape`,
`MpsFlatten` in `mps_common.h`). Stages 2 and 3 inherit the rule, and
`tests/integration/ops/test_mps_unary.py` pins it on the shapes -- and the
aligned controls -- that an aten-shaped descriptor would get wrong.

**Graph reuse, measured on an M5 (this is what stage 3 fixes).** The
marshalling contract above is only half of what the first implementation got
wrong. The nine stage-1/2 kernels ran per call at a cost that did not depend on
size: 4751 us for `abs` and 6458 us for `mul` at 128 elements, against CPU
fallbacks of 1.7-348 us, so the arithmetic was not what was being measured. A
bare-Metal probe separates the two halves: `[[MPSGraph alloc] init]` plus node
construction costs ~3.5-4.5 ms per call, and awaiting a command buffer that
already holds the compiled graph costs another ~270-370 us, while the same
graph scheduled with `runAsyncWithMTLCommandQueue:` and never awaited costs
~18 us. Only the `MPSGraphExecutable` plus
`runAsync…inputsArray:resultsArray:` path reaches that: reusing one `MPSGraph`
and its `runAsync…feeds:resultsDictionary:` measured ~159 us, and dynamic-shape
placeholders -- which would have collapsed the cache keys -- measured ~409 us.
`MpsRunGraph` therefore compiles a plan once per (body key, operand dtypes and
shapes), caches it up to 64 entries, schedules without waiting, and commits one
empty sentinel command buffer afterwards so that `StreamQuery` and
`StreamSynchronize` still cover the work (§5.1's allocator hazard). The same
ops then measure ~20-75 us per call, and a `.cpu()` read back immediately after
one still returns the right values.

## 7. Staged delivery

| Stage | Content | Exit criteria |
|---|---|---|
| 0 | Runtime floor only, `cpu_fallback` for all ops | The skill's operator-free smoke test passes: `device_count > 0`, `empty`/H2D/D2H, stream + timing event, allocator churn across two streams, `empty_cache` |
| 1 | `unary` category via MPSGraph/MSL, one op end to end | Unary op matches CPU within tolerance |
| 2 | `binary`, `binary_scalar`, `reduce` | Category suites pass |
| 3 | `gemm`/matmul, conv, softmax, the lanes a small model needs | Qwen3-0.6B (or a comparable small model) forward matches CPU |
| 4 | Per-op support record, matrix rows, vendor install doc | `docs/reference/operator-support.md` updated from measured results |

Each stage ends with a green build and a measured result; operator support is
recorded only from measurements, per CLAUDE.md.

Status: stage 0 landed with the `mps` platform row; stage 1 landed with `abs`
as the first `unary` op (`csrc/aten/backends/mps/`, `scripts/codegen/codegen_mps.py`,
`backends_mps.conf` routing `abs` to `mps`), verified on an M-series host with
`tests/integration/ops/test_mps_unary.py` (matches CPU across f32/f16/bf16 and
the integer dtypes at row widths that are not multiples of 16 bytes as well as
aligned ones, bool declined to the same `NotImplementedError` as CPU).
Stage 2 landed with the `binary`, `binary_scalar` and `reduce` categories --
`add`/`sub`/`mul` over two tensors and over a tensor and a host scalar, and
`sum.dim_IntList`/`mean.dim` -- verified with
`tests/integration/ops/test_mps_binary.py` and
`tests/integration/ops/test_mps_reduce.py` (CPU-equal across f32/f16/bf16 and
the integer dtypes at unaligned row widths, broadcasting inside the graph,
`alpha` folded in as a constant, and the declined cases -- mixed dtypes,
fractional scalars and alphas on integral operands, int64 beyond +/-2^53,
integer/bool/float64 reductions, empty operands, repeated or out-of-range axes
-- taking the annotated CPU round-trip). `max.dim`/`min.dim` stay on the
fallback: MPSGraph's argmax selector has no documented tie-break and the
reference has an answer.

Stage 3 opened with the `gemm` category -- `mm` and `bmm` -- the ops the
category exists for and the ones the stage's promise is measured against:
MPSGraph exposes a single selector,
`matrixMultiplicationWithPrimaryTensor:secondaryTensor:`, which broadcasts, so
one graph node serves both ops and the kernel is what enforces aten's shapes (a
non-2-D `mm` operand, a non-3-D or mismatched-batch `bmm` operand, and a
mismatched contracting dimension are declined to the CPU round-trip so the
reference raises its own error). The layout is the part worth recording:
`nn.Linear` hands over `weight.t()`, and a transposed last two axes over dense
memory is the one stride pattern the kernel recognises (`MpsGemmOperand` in
`mps_common.h`), carrying the transpose into the graph instead of copying the
operand. Measured on an M5 with a (512, 1024) x (1024, 32768) f32 product, the
graph transpose cost 11.9 ms against 13.1 ms for an already-dense operand and
2.96 ms for the transpose node on its own, so reading a weight down its columns
(70 ms) is what the copy was paying for. Because the layout is a host value the
kernel reads, it is part of the plan key: a `(32, 32)` dense operand and the
same buffer transposed present identical dtypes and shapes and would otherwise
be answered by one another's compiled graph. On a single Qwen3-0.6B layer --
eight `mm` and two `bmm` calls -- device dispatch time for the layer fell from
163.2 ms to 43.7-47.6 ms with the ten calls going from 121.5 ms (74% of it) to
0.42-0.51 ms (1%), and nine timed forwards went from a 239.0 ms median (212.6 ms
minimum) to 68.4 ms (61.9 ms); those are single-host numbers on a machine shared
with other load, so they are a same-session A/B rather than an absolute. The
stage-3 `gemm` increment is
verified with `tests/integration/ops/test_mps_gemm.py` (CPU-equal across
f32/f16/bf16, the graph-transpose and the copied slice case distinguished by
the allocator's peak, the declined shapes and integral/float64 round-trips, and
the dispatch log naming `mps`).

Stage 3 then added `softmax` and the two elementwise unaries a small model
reaches for beside it, `neg` and `pow.Tensor_Scalar`, each measured on the M5
against the CPU round-trip the previous commit's composite path paid -- a
device-to-host copy, the CPU kernel, and a host-to-device copy back, which is
what a device-resident model pays rather than a bare CPU kernel. A
(1, 16, 512, 512) attention-score softmax fell from 3.04 ms to 1.03 ms, the
(1, 512, 1024) `pow(x, 2)` of a LayerNorm variance from 1.35 ms to 0.38 ms, and
the (1, 16, 512, 64) `neg` a `rotate_half` is built from from 0.48 ms to
0.33 ms -- the last one the smallest win of the three because a 512x64 slice is
where the copy is cheap relative to the schedule. The largest single item on the
layer is still `embedding` on the fallback (44 ms of its dispatch time), which
this stage did not touch.

Two decisions narrow the surface and are worth recording. **`softmax.int` is
not registered because it cannot be.** On this PyTorch build it is
`CompositeImplicitAutograd`: the dispatcher decomposes it above the backend key,
so a kernel for it is never called and there is no wrapper in
`csrc/aten/generated/register.inc` to fill. What does reach `PrivateUse1` --
what `torch.softmax`, `F.softmax`, SDPA and `torch.softmax(dtype=...)` all lower
to -- is `aten._softmax(self, dim, half_to_float)`, so that one spelling is the
whole category. `softmax.int_out` joins the other `.out` forms on the fallback,
because serving one means writing into a tensor the caller owns. **`bfloat16`
is declined for both `softmax` and `pow`**, on a measurement rather than a
tolerance: a bfloat16 MPSGraph softmax returned a denormal for every element and
a bfloat16 `pow` returned values up to 36 from the reference on the SDK in use,
while float16 was bit-identical and float32 within 29 ulp. The gate takes
float32 and float16; everything else keeps the round-trip, which stays correct.

This increment is verified with `tests/integration/ops/test_mps_softmax.py`,
`test_mps_pow.py`, and the extended `test_mps_unary.py`: CPU-equal over every
axis including the negative spellings, f32/f16 accepted with bf16/f64/integral/
bool declined into the reference's own values and errors, an empty operand, a
rank-0 operand, a strided operand, plan reuse across axes, exponents and dtypes,
and the dispatch log naming `mps` for the public `torch.softmax` spelling rather
than only for `_softmax`. `conv` is the last lane of stage 3, and stage 4 is
open. There is still no CI job: the lane below needs a
GitHub-hosted `macos-14`/`macos-15` runner the container-based pipeline does not
yet use, so the operator suite is run by hand until that lands.

## 8. Conformance surface (files added or changed)

| File | Change |
|---|---|
| `cmake/flagos_platforms.json` | new `mps` row |
| `csrc/runtime/accelerator/mps/{device,memory,stream}.cc` (+ `.mm` as needed) | new |
| `csrc/runtime/accelerator/CMakeLists.txt` | `mps` arm: glob `*.mm`, link Metal/Foundation, `FATAL_ERROR` if the Metal SDK is absent |
| `csrc/CMakeLists.txt` | `USE_MPS` define; exclude `backends/mps/` from other platforms |
| `csrc/runtime/allocator/backends/mps_memory.h` | new `DeviceMemoryInterface` |
| `csrc/runtime/allocator/caching_device_allocator.cc` | include guard + `GetCachingAllocator()` arm |
| `csrc/aten/dispatcher.h` | `Backend::kMps` slot and `GetBackendForOp` arm (stage 1+) |
| `csrc/aten/register.cc` | `USE_MPS` arm that does not fall through to the generic `#else` |
| `csrc/aten/backends/mps/` | native kernels (+ `generated/`) |
| `scripts/codegen/codegen_mps.py` (+ `gen_vendor_confs.py` `VENDORS["mps"]`) | new generator / routing table |
| `torch_fl/configs/backends_mps.conf` | new |
| `torch_fl/_platform.py`, `torch_fl/accelerator/mps/` | detection + compat |
| `setup.py` | `mps` arm |
| `tests/integration/test_mps_runtime.py` | smoke test, `@pytest.mark.mps` |
| `tests/integration/ops/test_mps_unary.py` | stage-1 unary op suite, `@pytest.mark.mps` |
| `tests/integration/ops/test_mps_binary.py`, `test_mps_reduce.py` | stage-2 elementwise and reduction suites, `@pytest.mark.mps` |
| `tests/unit/test_platform_capability_matrix.py` inventory lines | add `mps` to the three inventories |
| `docs/reference/{compatibility,platform-capability-matrix,operator-support}.md`, `docs/vendors/mps/installation.md` | platform rows and guide |
| `.github/configs/mps.yml` + workflow job | CI on a GitHub-hosted `macos-14`/`macos-15` arm64 runner |

`tests/unit/test_platform_capability_matrix.py` machine-checks the three
directory inventories against the tree, so those doc lines and the new
directories must land together or unit tests fail.

## 9. Validation

- Stage 0 and each stage build with
  `FLAGOS_ACCELERATOR=mps pip install -e . --no-build-isolation` on Apple
  Silicon, which is the machine this design was written on — the floor is
  fully testable locally, unlike every other vendor backend.
- The operator-free smoke test from the skill is the stage-0 gate.
- Per-op: CPU comparison at the tolerances the skill records (unary ≤ 4.4e-5,
  binary ≤ 4.7e-6, comparisons exact), across shapes including empty and
  1-element, every claimed dtype, and non-contiguous inputs.
- Confirm tensors stay on `flagos` through a compute chain (no silent CPU
  fallback) and that `torch.__version__` still reports the expected build.
- CI: `macos-14`/`macos-15` GitHub runners are Apple Silicon and run MPS
  headless, so the smoke test and the operator suites can be CI-gated here more
  cheaply than on any vendor runner.

## 10. Risks and unknowns

- **MPSGraph from C++.** MPSGraph is an Objective-C API; the op layer must be
  Objective-C++, and every kernel boundary crosses the ObjC++/C++ seam. The
  extent of that seam is not yet measured.
- **Graph-build cost — resolved by the plan cache.** It measured far worse than
  expected (~4.5 ms per call, described in §6) and was worth fixing before
  adding ops: the fix is compiled-graph reuse keyed on the graph body and the
  operand signature, plus an asynchronous run with a sentinel commit, which
  takes the same ops to ~20-75 us. Whether the remaining per-op floor still
  argues for MSL kernels is a question for stage 3+ and for ops with real
  arithmetic, not for the elementwise family the cache now covers.
- **Unified memory and the allocator.** Apple GPUs use unified memory; whether
  the block pool should hand out `MTLResourceStorageModePrivate` buffers (fast,
  needs blit for host access) or `Shared` buffers (simple, host-visible) is a
  stage-0 measurement, not an assumption.
- **`MTLSharedEvent` timing.** `EventElapsedTime` has no direct Metal analogue;
  the `GPUStartTime`/`GPUEndTime` approach needs to be validated against a known
  duration.
- **Coexistence with `torch.backends.mps`.** Both the `flagos` PrivateUse1
  device and torch's own MPS backend will be live in the same process on a Mac.
  They own separate command queues and allocations, so dispatch should not
  conflict, but this needs an explicit test (import both, run on both).
- **The device copy is slower than it should be.** Materialising a strided
  operand on the device -- `w_t.contiguous()` -- measured 84.6 ms against the
  10.4 ms `torch.mps` needs for the same copy, so the fallback is 7-12x the
  reference on the one path the gemm kernels no longer need. Stage 3 took the
  copy off the hot path rather than fixing it; the copy itself is an open
  measurement on the runtime floor, not the op layer.
- **No CI lane.** The MPS suite is gated on an Apple Silicon host and run by
  hand. The pipeline is container-based and does not schedule a GitHub-hosted
  `macos-14`/`macos-15` runner, so until one is wired up a green MPS run is a
  local result rather than a checked one.
- **Scope.** Option B is a multi-week project; the staged plan exists so value
  lands at every stage instead of only at the end.
