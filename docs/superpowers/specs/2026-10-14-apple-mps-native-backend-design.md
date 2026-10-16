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
than only for `_softmax`.

Stage 3 closed with the `conv` category. Two facts decide its shape. **The
registered spelling is `convolution_overrideable`**, because `aten.convolution`
is `CompositeImplicitAutograd` on this build: it validates and decomposes above
the backend key, so a kernel for it is never called. It is `F.conv2d`,
`F.conv3d`, `nn.Conv2d` and `nn.ConvTranspose2d` that all reach
`aten.convolution_overrideable` on `PrivateUse1`, which the dispatch-log cases
hold. **The geometry lives on the host.** MPSGraph's convolution descriptors
(`MPSGraphConvolution2DOpDescriptor`/`...3DOpDescriptor`) carry the stride,
dilation, groups and an explicit front/back padding pair per axis -- aten's flat
`padding` is `[front_0, front_1, ..., back_0, back_1, ...]`, the same split the
descriptor wants -- and none of that is visible in the operand shapes, so every
one of those values is part of the plan key. Two mappings are easy to get wrong
and so are spelled out in `MpsConvDescriptor`: the per-axis order runs opposite
to aten's (`p = rank - 1 - axis` between them, a wrong number rather than a
shape error for a non-square kernel) and the bias has to be reshaped to
`(1, OC, 1, ...)` before the add, because `(1, 1, 1, OC)` crashes the MPSGraph
compiler and `broadcastTensor:toShape:` expands along the wrong axes. MPSGraph
has no 1-D selector, so `conv1d` takes the round-trip. The dtype gate is
deliberately wider than the softmax one: **bfloat16 is served here**, because a
bfloat16 convolution measured bit-identical to a bfloat16 reference convolution
on an M5 (both 6.52e-03 against an exact float64 reference -- bfloat16's own
precision) where a bfloat16 softmax came back denormal, which is why the
predicate is per category rather than per dtype.

This increment is verified with `tests/integration/ops/test_mps_conv.py`:
CPU-equal over the plain, strided, dilated, pointwise, depthwise, grouped,
asymmetric-padding and anisotropic-kernel cases in 2-D and 3-D, f32/f16/bf16
accepted with f64/integral/bool and every declined shape taking the reference's
own value or error, `conv1d` and `nn.ConvTranspose2d` round-tripping, plan reuse
across stride, padding, dilation, groups and the bias flag, and the dispatch log
naming `mps` for the public `F.conv2d`/`F.conv3d` spellings. That completes the
stage-3 lanes the design names (`gemm`, `conv`, `softmax`); stage 4 is open.

Stage 4 opened with the `silu` category, which is the first op here MPSGraph has
no selector for. The definition is `x * sigmoid(x)` and MPSGraph has
`sigmoidWithTensor:`, so the body is a two-node composition -- the first in this
layer -- and the interesting question is whether composing at all is legitimate.
It is not, automatically: **aten's CPU `silu` is its own kernel rather than the
definition**, and evaluated on the host the two readings differ by 7.8e-03 on
float16 and 3.1e-02 on bfloat16, which would be a wrong answer rather than a
rounding difference. So the gate is a measurement of the graph against
`at::silu` rather than a statement of the identity, taken on an M5 over operands
spanning [-40, 40] with the saturated tails, denormals and a rank-0 shape
included: bit-identical on float16 and bfloat16, and 2.7e-07 relative on
float32, where the graph and the reference sit 1.2353e-06 from an exact float64
evaluation to the last digit -- the same distance, so they round differently
rather than one being wrong. All three widths are therefore served, which is
wider than the `softmax` gate that declines bfloat16 and narrower than nothing:
**the widths a category serves are per op here, not a per-dtype list this
backend keeps.** The key has no host parts, since `silu` takes no argument
beyond its operand.

This increment is verified with `tests/integration/ops/test_mps_silu.py`:
CPU-equal on all three float widths, the SwiGLU `(batch, seq, 3072)` shape, the
saturated tail where a weaker `sigmoid` would show as a number, the sign of the
negative tail, the `inf`/`nan` pair, rank-0 and empty operands, strided operands,
plan reuse across interleaved dtypes and shapes, and the dispatch log naming
`mps` for both the `F.silu` and `aten.silu` spellings. There is a case there that
guards the *reference* rather than the kernel: it fails if a future PyTorch makes
`silu` the composition, because then the note this gate rests on would be stale.

Stage 4 continued with the `embedding` category, which section 7 recorded as
~47% of a Qwen3-0.6B layer's device dispatch time -- the largest single item
left. It is the first op here whose cost is a *read* of its operand rather than
a computation over it, and the reason it mattered was not the arithmetic: the op
was routed to `none`, so every call took `cpu_fallback`, which copies the whole
`(151936, 1024)` weight to the host. Measured on an M5, one `F.embedding` call
with a `(1, 512)` index tensor cost 55.3 ms, of which 55.8 ms was the host copy
of the table, against 0.697 ms for the same read performed entirely on the CPU.
The graph side is one node -- `gatherWithUpdatesTensor:indicesTensor:axis:0
batchDimensions:0` -- and it reproduces `at::embedding` bit for bit.

Two things about this kernel are worth recording, because each is a case where
the obvious reading of the reference is wrong. The first is the index range:
MPSGraph's gather has no range contract, and measured on an M5 with a 6-row
table and the index list `(5, 6, 2, 0)` it returns the table's *first row* for
the out-of-range 6 -- likewise for `-1` and for `100` -- where the reference
raises `IndexError: index out of range in self`. A plausible wrong answer is the
one outcome worse than a round-trip, so the body validates the range on the host
before any node exists and declines, handing the call to the reference for its
own error. The other gather spelling, `gatherAlongAxis:`, was measured first and
rejected: it requires the indices to carry the operand's rank, and its own
out-of-range behaviour is to insert a 0, which would look like a padding row.

The second is `padding_idx`, which the forward op **does not read**. Measured
against the CPU reference with a `(4, 3)` table and the index list
`(0, 1, 2, 3)`, the result is bit-identical for `padding_idx` `-1`, `-3`, `-4`
and `2`, and an out-of-range `padding_idx` (`7`, `-9`) is accepted rather than
rejected; the zeroing that name asks for is a property of
`embedding_dense_backward`, not of this op's output. That matters because the
first version of this body assumed otherwise and declined whenever
`padding_idx >= 0` -- and `F.embedding` passes `padding_idx=None`, which arrives
at the op as `-1` and was read as "count back from the end", so *every* ordinary
`F.embedding` call took the whole-table round-trip the increment existed to
remove. The kernel measured 77-99 ms with that bug and 0.25-0.29 ms without it.
`padding_idx` is not in the graph key either: two calls that differ only in it
produce the same bytes and are one plan.

This increment is verified with `tests/integration/ops/test_mps_embedding.py`:
bit-exact CPU equality on all three float widths and both integer index widths,
the index rank preserved from `(1,)` to `(2, 3, 4, 5)`, empty and non-contiguous
operands, every `padding_idx` spelling including the `None`-as-`-1` the front end
passes, the three out-of-range `IndexError` cases and the float-index
`RuntimeError`, plan reuse across dtypes and across `padding_idx`, the dispatch
log naming `mps` for both the `F.embedding` and `aten.embedding` spellings, a
declined float64 table that never reaches the graph, and a cost case that fails
if the whole-table round-trip comes back.

Stage 4 continued with `div`, `div.Scalar` and `reciprocal`, which close the
elementwise arithmetic set that `add`, `sub` and `mul` opened in stage 2. The
increment was chosen from a measurement rather than from the op list: on an M5
over a `(2, 512, 1024)` float32 tensor, `div.Scalar` cost 0.68-0.95 ms and
`div.Tensor` 1.36-2.66 ms against 0.24 ms for a device-to-host copy of the same
tensor, so the routed path was making three host passes where the kernel makes
one. All three ops are served by a single graph node,
`divisionWithPrimaryTensor:secondaryTensor:`, and the body is one node for
`reciprocal` too -- `reciprocalWithTensor:` was measured and rejected on cost,
since `a * reciprocal(b)` and `a / b` are bit-identical to each other and to
the reference over zero and signed-zero denominators, inf, nan and denormals,
which makes the reciprocal a two-node cost decision rather than a correctness
one. `div.out`, the `mode` spellings (`div.Tensor_mode`, `div.Scalar_mode`,
`div.out_mode`, `div.Scalar_out`, `div.Scalar_mode_out`), the in-place
`div_.Tensor` / `div_.Scalar`, `reciprocal.out` and `reciprocal_` are not
served: the `mode` forms need a floor or trunc composition after an exact
division, which is a different body.

The honest finding of this increment is a bug the first version shipped. The
gate reused `MpsSupportsBinaryDtype`, the predicate written for `mul`, and that
predicate admits integral widths -- which is correct for a multiplication and
wrong for a division, because `aten::div` promotes integers to float. Measured
through the backend, `aten::div.Tensor(tensor([-7]), tensor([2]))` returned the
truncated int64 `-3` where the reference returns the float32 `-3.5`: a silently
wrong value *and* a silently wrong dtype. The fix is a dedicated
`MpsSupportsDivDtype` that requires the two dtypes to match and both to be
float32, float16 or bfloat16, plus a second new predicate,
`MpsSupportsDivScalarDtype`, because `div.Scalar` reaches the same promotion
through `MpsSupportsScalarDtype` and had the same defect. `reciprocal` is gated
by the division predicate rather than by the wider unary one, since `x / 2` and
`1 / x` are one node family and have to agree on the widths they claim.

One behavioural difference from the reference is measured, recorded and shared:
MPSGraph flushes a subnormal *result* to zero. The boundary is the result, at
the smallest normal float -- `1e-38 / 2` returns `0` where the reference returns
`5.0e-39`, while `2.35e-38 / 2` returns the normal `1.175494e-38` untouched --
and it holds through both bodies, `aten::div.Tensor(x, tensor(2.0))` and
`aten::div.Scalar(x, 2.0)` alike; `reciprocal` overflows the same way, with
`1 / 1.17e-38` returning `inf` where the reference returns `8.5e37`. It is not a
division property: `mul.Tensor`, routed since stage 2, flushes identically
(`1e-38 * tensor(0.5)` returns `0`), as does `add.Tensor` (`1e-39 + 1e-39`
returns `0`). The subtlety worth recording is that the Python operators do *not*
reproduce it: `x / 2.0` and `x * 0.5` are resolved by PyTorch's own composite
wrapper before PrivateUse1 dispatch, so no graph node is built for the constant
and the result keeps its subnormal -- even though the dispatch log names
`div.Tensor` for that spelling, because that is the schema name and not the
kernel that ran. Only the opaque `aten::div.Scalar(t, 2.0)` and
`aten::div.Tensor(t, tensor([2.0]))` reach the graph. A future reader who
measures only the Python form would wrongly conclude the graph preserves
subnormals, so both spellings are pinned in the suite.

This increment is verified with `tests/integration/ops/test_mps_div.py`:
bit-exact CPU equality on all three float widths including the zero and
signed-zero sign bits, the inf and nan quadrants, exact quotients, the dtype
promotions the reference applies to integers and booleans, the subnormal
boundary through both bodies and in both the Python and the opaque spellings,
the declined integral, complex and mismatched-dtype pairs that never reach the
graph, plan reuse across shapes and dtypes, and the dispatch log naming `mps`
for the `a / b`, `a / 2.0`, `2.0 / a` and `torch.reciprocal` spellings.

Stage 4 continued with `index_select`, which was *not* the op this section had
recorded as next. The plan was `embedding_dense_backward`, and it was measured
and dropped: the only correct graph for it is `oneHot(N, V)ᵀ @ grad(N, H)`,
verified bit-exact, and its work is `O(V·N)` -- 18.8 ms at `N=1` (a win against
the fallback's 43 ms), 68.3 ms at `N=512` and 243 ms at `N=2048` (losses), with
chunking over `V` measured at 162 ms and 670 ms and no better. The
`scatterAlongAxis:withDataTensor:` node that would have made it linear is
unusable: the SDK enforces `indices.shape == updates.shape`, and with equal
shapes it writes updates as a rolling window rather than by row index (a
`(N, 1, 1)` sparse index is rejected outright, and MPSGraph exposes no
`broadcastTensor:withShape:` to build the equal-shape form). Every row *write*
primitive round-trips as well -- `index_copy_` 117 ms, `index_add_` 115 ms,
`out[idx] = grad` 66 ms -- all of them `aten::_to_copy` over the whole `(V, H)`
table, so the dense output has to be materialised either way. The measurement
pointed instead at the read family those ops all share, and that is this
increment.

`index_select` is the gather the `embedding` body above already builds, reached
through the op the caller names rather than through the embedding front end, and
it is served on the same evidence: measured on an M5 over a `(151936, 1024)`
float32 operand, the fallback cost 40.4 ms for a *single* row and the graph
costs 0.29 ms, with 0.29 ms at 512 rows and 0.78 ms at 4096 -- the fallback's
cost is the dimension of the operand (V=10000 → 2.4 ms, V=151936 → 41.7 ms, same
one row, profiled as `aten::_to_copy` over 622 MB) while the graph's cost is the
number of rows asked for. The body is one node,
`gatherWithUpdatesTensor:indicesTensor:axis:0 batchDimensions:0`, over the same
`(rows, row_width)` reshape the embedding body uses for the row-pitch reason
recorded above.

What is new here, and the whole of the increment's design, is that
`index_select` carries arguments where `embedding` carried only operands. The
body resolves `dim` against the operand's rank and **declines** unless it
resolves to axis 0 -- the only axis the one node reaches -- alongside an
out-of-range `dim`, an index whose dtype is not int32/int64, and an index that
falls outside the operand. Declining is not a degradation of the contract: the
round-trip *is* the reference's implementation and its own
`IndexError`/`RuntimeError`, so a declined call answers correctly and simply
costs what it costs today. The alternative -- building the axis-0 node and
returning its answer for a different axis -- is a plausible-looking wrong
answer, which is the failure class the `embedding` range check exists for.

The honest finding of this increment is a second member of that class, found by
the suite rather than by review. `index_select` takes a *vector* index and says
so (`IndexError: index_select(): Index is supposed to be a vector`), where
`embedding` reads any index shape as a batch dimension -- but the first version
of this body set `out_shape[0] = index.numel()` and flattened the index buffer
for the graph, so a `(2, 2)` index over a `(6, 4)` operand came back as a
`(4, 4)` tensor with a silent flatten where the reference raises. The fix is an
explicit `index.dim() != 1` decline, and it is held by a case that fails without
it.

This increment is verified with `tests/integration/ops/test_mps_index_select.py`:
bit-exact CPU equality across the three float widths and both index widths,
the index length becoming the leading dimension, rank-1 and rank-3 operands,
repeated and reordered indices, an empty index, non-contiguous operands and
non-contiguous indices, the `dim` cases (axis 1 and 2 declined, `dim=-3` over a
rank-3 operand served, an out-of-range `dim` reaching aten's error, and a square
operand read at `dim=0` and `dim=1` giving two different answers, which is what
`dim` being in the plan key is for), the range contract (out-of-range, negative,
and one bad index among good ones all raising the reference's `IndexError`, in
both index widths), the declined dtypes and the higher-rank index, plan reuse
across dtypes, index widths and index values, the dispatch log naming `mps`, and
a cost case over the `(151936, 1024)` shape that fails if the whole-operand
round-trip returns.

Stage 4 continued with `take`, the second reader of the same node and the one
that proved the node's *contract* is not the node's *behaviour*. It was picked
over `gather` on measurement: both route to `none` and both pay the same
whole-operand round-trip, and on a `(151936, 1024)` float32 operand the fallback
costs 45.6 ms for a single element, 46.3 ms for 512 and 42.0 ms for a
one-element `gather` at `dim=0` -- the cost is the operand, not the question,
which is the same profile `index_select` had. The node costs 0.29 ms.

Every argument `take` has, `index_select` does not: there is no `dim`, the
operand is read as a flat `numel()`-length sequence rather than along an axis,
the result's shape is the index's own shape rather than the index's length
prepended to the operand's trailing dimensions, the index may have any rank
rather than being a vector, and a negative index is in range rather than an
error. Each of those was measured against the CPU reference before the body was
written, and the last one cost a real bug. The first version of the body
range-checked the index on the host and handed the original through, which is
what the `index_select` body does and what the `embedding` range check is for.
Measured through that version, `take(t, [-1, 0, 19])` over a 20-element operand
returned `[0, 0, 19]` against the reference's `[19, 0, 19]`: MPSGraph's gather
answers a negative index with the *first* element, so the check had admitted the
index correctly and the graph had then read it wrongly. The fix is that the
range check and the wrap are one host pass -- a negative entry is normalised
against `numel()` and an entry outside `[-numel, numel)` takes the round-trip --
so the graph is only ever handed a non-negative index. That the wrong answer is
plausible is the point: it is the same failure class as the out-of-range index
the `embedding` check exists for, and it is why the case that holds it asks for a
single element rather than one negative index among positive ones.

The second finding is a graph limit worth recording because its cause is
unobvious. The body reshapes the index to its own shape from a buffer that
carries the operand's trailing 1 (the same row-pitch reshape the `embedding`
body uses), and a `(1, numel)` index is rejected by that reshape:
`error: 'mps.reshape' op the result shape is not compatible with the input shape
… (tensor<3x20xf32>) -> tensor<3xf32>`. The index's *runtime* shape overrides the
static placeholder shape the graph was compiled with, so the shapes the node
produces are the operand's `(N, 1)` against an index of `(K)` giving `(K, 1)`,
`(2, 2)` giving `(2, 2, 1)`, and so on -- read through `MPSGraphTensorData`'s
`mpsndarray` and `MPSNDArray readBytes:strideBytes:`, because a standalone probe
run with `runWithFeeds:` answers zeros and would have hidden it.

This increment is verified with `tests/integration/ops/test_mps_take.py`:
bit-exact CPU equality across the index ranks (rank-1, rank-2, rank-3, one row,
one column) and the operand ranks (rank-0 through rank-3, all read as flat), the
operand structure being discarded and the index shape being the result shape,
the `(2, 2)` index that separates the two siblings' shape rules, negative
indices including the two ends of the wrap, the int8-width operand the
normalisation reads `numel` from, the out-of-range cases in both signs reaching
the reference's own `IndexError`, the int64-only index contract, the empty
index, non-contiguous operands and indices, plan reuse across index values and
across a declined call, the dispatch log naming `mps` on both the served and the
declined path, and a cost case over the `(151936, 1024)` shape that fails if the
whole-operand round-trip returns.

Stage 4 continued with `gather`, the third reader of this node and the first
one whose node is a *different* node. `gatherWithUpdatesTensor:indicesTensor:`
has no axis, and the axis spelling `gatherAlongAxis:withUpdatesTensor:-
indicesTensor:name:` carries a shape precondition that `gather` does not ask
for: MPSGraph requires the updates tensor and the indices tensor to have the
same shape apart from `axis`, and it enforces that in the compiler rather than
at run time -- measured, a `(4, 5)` operand against a `(4, 2)` index does not
raise a deferrable error, it aborts the process with `failed assertion
'original module failed verification'`. So the body trims the operand to the
index's shape on the non-axis dimensions before it builds the node. The node is
permissive in the other direction -- it compiles a `(4, 5)` operand against a
`(4, 7)` index and answers `4x7` -- which is why the non-axis rule is checked on
the host: a non-axis dimension of the index may be smaller than the operand's
but never larger, and the reference is what says so (`Size does not match at
dimension 0 expected index [5, 5] to be no larger than self [4, 5] apart from
dimension 1`). The *axis* dimension is free to exceed the operand's in both
directions, which is the shape rule the body has to keep straight.

The honest finding of this increment is a shape rule with a view behind it, and
it was a real bug before it was a paragraph. `narrow` returns a *view*, and
`MpsRunGraph` describes every operand as a flat 1-D array of `numel` elements
read densely from that operand's buffer offset. The first version of the body
narrowed the operand and fed the view straight to the graph, so a
`gather((4, 5), 0, (3, 1))` over `arange(20)` -- whose reference answer is
`[0, 10, 15]` -- came back as `[0, 1, 2]`, which is `t.reshape(-1)[:3]`: the
operand's first three elements, not the narrowed block. The fix is a
`contiguous()` when the narrow changed the shape, and the measurement is
recorded above the body. The same class of finding, one layer down, is that the
trim must be a *host* `narrow` rather than the graph's `sliceTensor:`: a
`sliceTensor:` built from the same indices answers `[0, 10, 0]` where the
reference answers `[0, 10, 15]`.

Three more measured behaviours decide the body, and each is a decline rather
than a degradation, because a declined call takes the round-trip and therefore
the reference's own error. The **index rank must equal the operand's** in both
directions -- a `(5,)` index into a `(4, 5)` operand raises `Index tensor must
have the same number of dimensions as input tensor`, and so does a higher rank
-- so the rank is checked rather than left to the node, which accepted some
mismatched-rank inputs in the probes. The **index dtype** is int32 or int64,
where `take` is int64-only. An **out-of-range index** raises `index 99 is out
of bounds for dimension 0 with size 5` -- the bound is the operand's
`size(axis)`, not its `numel` -- and a **negative index raises**, the opposite
of `take`, so there is no wrap here and the check is a rejection. MPSGraph's
gather answers an out-of-range index with an inserted zero rather than failing,
which is the same plausible-answer failure class the `embedding` and `take`
range checks exist for.

The last measured finding belongs to the platform rather than to `gather`, and
it is recorded here so it is not re-derived as a `gather` regression. The trim
above costs what MPS charges for any strided copy whose parent is large:
measured on a `(151936, 1024)` float32 operand, a `dim=0` gather with a
full-width index costs 0.26 ms, the same gather with an `(8, 16)` index -- which
needs a `(151936, 16)` trim -- costs 52 ms, and the trim alone costs ~48 ms,
while the gather over an already-trimmed operand is back to 0.22 ms. A dense
`op[:8, :16].contiguous()`, a bare `view * 2` and `view.abs()` over the same
huge-parent view all cost ~42-47 ms, with the same 128-element view over an
8-row parent at 0.007 ms, so the cost is the parent allocation's size and not
the view's. A leading (`dim=0`) narrow stays a dense prefix and is cheap for
that reason, which is why the cost case in the suite is written on a `dim=1`
call whose non-axis narrow is the leading one.

This increment is verified with `tests/integration/ops/test_mps_gather.py` (65
cases): bit-exact CPU equality across the three float widths plus f16, bf16,
int32, int64, uint8 and bool, the index rank equalling the operand's, the
one-sided non-axis rule in both directions and on rank-2 and rank-3 operands,
the axis being free to exceed the operand's, the out-of-range and negative
indices reaching the reference's own `RuntimeError`, the int64-only sibling
contract that this op does *not* share, `dim` normalisation and an out-of-range
`dim`, non-contiguous operands and indices, the empty index, plan reuse, the
dispatch log naming `mps`, and the two cost cases above -- one asserting the
range pass does not move the operand, one asserting the trim costs what the
platform charges. The `[0, 10, 15]` value assertion that caught the view bug is
kept.

The remaining stage-4 work is `F.layer_norm` as a multi-op increment
(`native_batch_norm` plus `addcmul`, measured at 1.50-4.56 ms), and the
`native_batch_norm` it lowers to. `embedding_dense_backward` stays dropped for
the reason recorded above.
There is still no CI job: the lane below needs a
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
