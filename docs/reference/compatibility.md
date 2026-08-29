# Compatibility and Platform Support

## Status Definitions

| Status | Meaning |
|---|---|
| Stable | Critical paths are continuously tested and the supported version combination is documented. |
| Beta | The primary path is validated, but coverage, packaging, or release procedures are not yet stable. |
| Experimental | Validation exists for a specific setup, model, or hardware environment; interfaces or build procedures may change. |
| Runtime only | Device runtime support exists, but the platform is not a general eager operator backend. |

## Project Compatibility

| Component | Supported range | Notes |
|---|---|---|
| Python | 3.8 or later | From package metadata. Platform SDKs and available wheels may impose a narrower range. |
| PyTorch | 2.10.x (`>=2.10,<2.11`) | Generated ATen bindings are tied to this minor line. |
| FlagGems | Platform dependent | Installed from PyPI or a vendor-compatible build only where the platform route uses it. |
| Triton/compiler | Platform dependent | Use the compiler distribution required by the selected accelerator. |

## Platform Matrix

| Platform | Build selector | Execution path | Eager and autograd | `torch.compile` | Distributed | Profiler | FlagGems | Status |
|---|---|---|---|---|---|---|---|---|
| NVIDIA CUDA | `ACCELERATOR=cuda` (default) | CUDA boxing over an external `libtorch_cuda.so` | Stable | Experimental (inductor GPU device registered; no CI test step) | Beta (FlagCX + NCCL fallback, DDP live-verified) | Stable (CUPTI parity) | Beta (Python + C++ dispatch paths) | Stable |
| MetaX | `ACCELERATOR=metax` | CUDA-boxing reuse via `cu-bridge`/mxcc, or native MetaX kernels | Stable (FP16/BF16 autocast and GradScaler measured in boxing mode) | Experimental (vendor Triton and FlagTree MetaX measured on C550; vendor Triton CI-covered) | Experimental (NCCL-shaped `mccl` fallback; not CI-covered) | Experimental (MCPTI parity measured on C550; not CI-covered) | Experimental (Python dispatch; not CI-tested on MetaX) | Stable |
| Ascend | `ACCELERATOR=ascend` | Native ACLNN operator backend, optional FlagGems via triton-ascend | Stable (CI-covered ops, RNG suite) | Experimental (inductor via triton-ascend 3.2.0, measured on 910; three toolchain workarounds, serial compile, no CI step) | Experimental (HCCL fallback; architectural routing only, no collective-level CI) | Runtime only (device/runtime events not emitted; profiler parity suite excluded from CI) | Experimental (Python dispatch; not CI-tested on Ascend) | Beta |
| PPU | `ACCELERATOR=cuda` + `PPU_SDK`/`PPU_HOME` detection | Same CUDA-boxing path as NVIDIA CUDA, against the PPU's CUDA-13-compatible SDK | Experimental (FP16/BF16 autocast and GradScaler measured on PPU hardware, not in CI) | Not validated | Experimental (NCCL fallback via vendor-adapted `libnccl.so.2`; not CI-covered) | Not validated on this vendor's tracer | Experimental (vendor-index Triton required) | Experimental |
| Hygon DCU | `ACCELERATOR=dcu` | CUDA boxing over the hipified DTK torch build (HIP kernels under the CUDA dispatch key) | Beta (including FP16/BF16 autocast and GradScaler) | Experimental (FlagTree HCU validated on `gfx936`; not in CI) | Experimental (RCCL via DTK; all_reduce/DDP measured on 2 cards, not in CI) | Beta (parity suite runs in CI) | Beta (Python dispatch only) | Beta |
| Enflame GCU | `ACCELERATOR=gcu` | Native `libtopsaten.so` operator backend, with CPU fallback for unrouted/int64/float64 ops | Beta (operator, RNG, factory, and AMP suites CI-guarded on S60) | Not validated | Not validated | Runtime only (TOPSPTI collects activities; no device events on a CPU-only Kineto build) | Experimental (Python dispatch, requires vendor Triton) | Beta |
| Moore Threads MUSA | `ACCELERATOR=musa` | Native `mudnn` operator backend, with CPU fallback for unrouted ops | Experimental (including FP16/BF16 autocast and GradScaler measured on MTT S5000) | Experimental (MThreads FlagTree forward/backward measured on MTT S5000; vendor runtime required) | Not validated | Experimental (MUPTI device timeline measured on MTT S5000; CPU-Kineto linkage is environment-dependent) | Experimental (Python dispatch, requires vendor Triton) | Experimental |
| D-Robotics BPU | `ACCELERATOR=bpu` | No eager kernel sets are built; eager ops run on CPU | Runtime only (CPU fallback for eager) | Experimental (`torch.compile(backend="bpu")` graph path via hbdk4) | Not applicable | Not validated | Not applicable (no per-op kernel build) | Runtime only |
| TsingMicro | `ACCELERATOR=tsingmicro` | Runtime/build selector present; no per-op kernel set documented | Runtime only | Not validated | Not validated | Not validated | Not applicable | Runtime only |

## Platform Notes

### NVIDIA CUDA

The default and most exercised platform. [`.github/configs/cuda.yml`](../../.github/configs/cuda.yml)
(lines 69-95) runs vendor-backend and FlagGems-runtime operator suites, general/factory tests,
the profiler parity suite, and Qwen3-0.6B inference and training against a mounted model.
Distributed collectives and DDP gradient sync are live-verified on 2x/8x A100 through FlagCX
with an NCCL fallback (see
[`docs/architecture/distributed-flagcx.md`](../architecture/distributed-flagcx.md), lines 1-10
and §6). `torch.compile` registers flagos as a first-class inductor GPU device with integration
documented in
[`docs/architecture/torch-compile-integration.md`](../architecture/torch-compile-integration.md)
(lines 84-127), but [`.github/configs/cuda.yml`](../../.github/configs/cuda.yml) has no
`torch.compile` test step, so this path is not exercised in CI. The profiler achieves structural
parity with torch-cuda via CUPTI (see
[`docs/architecture/profiler.md`](../architecture/profiler.md), lines 1-21).

### MetaX

[`.github/configs/metax.yml`](../../.github/configs/metax.yml) runs representative
operator dispatch, factory/autograd, RNG, AMP, and `torch.compile` tests in CI. The operator
command explicitly excludes all FlagGems tests, so FlagGems on MetaX has no CI validation. The
shared `AutocastPrivateUse1` policy registrations dispatch FP16/BF16 lower-precision, FP32,
optional-dtype, and promote groups through the existing CUDA-boxing kernels. On a C550 with
MACA 3.8.0, the 25-case AMP suite passed, including nested autocast state, non-finite unscale,
finite scale growth, overflow backoff, and a forward/backward optimizer step. This evidence is
specific to MetaX boxing mode; the legacy handwritten MetaX kernels were not revalidated.

MetaX boxing has a separate software-emulation path for scalar FP8 and packed FP4 matrix
inputs. `mm`, `bmm`, and `addmm` (including `dtype`/`out` variants and in-place `addmm_`)
decode `float8_e4m3fn`, `float8_e5m2`, `float8_e4m3fnuz`, `float8_e5m2fnuz`,
`float8_e8m0fnu`, and `float4_e2m1fn_x2` to BF16 before invoking ordinary device GEMM.
The default result is BF16 and an explicit output dtype is honored. Block-scaled metadata
formats (MXFP4, NVFP4, and block FP4) and `_scaled_mm` families are not included. The
marked C550/MACA 3.8.0 suite passes for this checkout (`37 passed`), covering all five FP8
formats, packed FP4, matrix overloads, non-square packed layouts, and fail-closed scaled-mm.
This evidence applies to CUDA-boxing mode, not the legacy handwritten MetaX kernel mode.

**Evidence boundary:** this route change does not alter the generic FlagGems survey cohort;
that survey was not rerun because it does not exercise the shared software matrix wrappers.

`torch.compile(backend="flagos")` is experimentally validated in the same C550/MACA 3.8.0
boxing environment with both the installed vendor Triton and FlagTree main at revision
`140bd6ab1ad86c5df4b07b76d9c722e357a9166d` (Triton 3.6, MetaX backend). The complete compile
suite passes with outputs and gradients remaining on `flagos`, including forward, backward,
FP32/FP16, max-autotune, recompilation, FakeTensor tracing, and the FlagTree-specific result
check. MetaX CI continuously exercises the vendor Triton path; FlagTree remains a manual
measurement because it replaces the environment's installed `triton` distribution.

MetaX carries FSDP2 and Qwen3 training parity work (see repository history). Distributed
support routes through the same NCCL-shaped fallback as CUDA (`_VENDOR_PROFILES["metax"]` in
[`torch_fl/comm/process_group.py`](../../torch_fl/comm/process_group.py), line 93), but that is
architectural routing, not a CI-verified collective test on this platform. Profiler-tracer
parity was measured locally on a C550 with MACA 3.8.0 MCPTI: the seven-case
`tests/integration/test_profiler_parity.py` suite passed in MetaX boxing mode, covering kernel,
memcpy, memset, and runtime events, flow pairing, device-time attribution, kernel metadata,
runtime callback names, and capture-window filtering. This is not yet a CI result, and the
scanner remains unvalidated across multiple MetaX SDK versions, devices, and non-default
streams.

### Ascend

[`.github/configs/ascend.yml`](../../.github/configs/ascend.yml) (lines 78-97) runs the
Ascend-marked operator suite and the full RNG dispatch suite in CI. FlagGems-marked tests in
`test_rng_dispatch.py` are selected via `-m "main_ops"` (lines 90-94) but the FlagGems runtime
path is not enabled in CI, so those tests run against the native ACLNN backend. Model-level
(Qwen3) validation exists as a point measurement outside CI — training at 0.82x `torch_npu` on
real Ascend 910 hardware (`tests/perf/e2e_qwen3_train_ascend.py`) — but
[`.github/configs/ascend.yml`](../../.github/configs/ascend.yml) (line 110) explicitly defers
Qwen3 inference/training smoke pending a model mount on the CI runner. Profiler parity is
explicitly out of scope: the Ascend profiler path does not yet emit device-side event
categories, and `test_profiler_parity.py` is excluded from CI until it does (same file, lines
112-117). Distributed support recommends the FlagCX path (see
[`docs/architecture/distributed-flagcx.md`](../architecture/distributed-flagcx.md), lines
204-208); the native HCCL fallback and the `flagos→npu` zero-copy view are architectural
routing, not on-hardware-verified collectives (same file, lines 67-75).

`torch.compile(backend="flagos")` is experimentally validated on Ascend through
`triton-ascend` 3.2.0, not FlagTree. Measured on a real 910 (`Ascend910_9382`, CANN 9.0.0,
torch 2.10.0+cpu, Python 3.10): `tests/integration/test_compile.py` passes 30 of 32 with a
cold inductor cache, the two skips being the FlagTree-only and MetaX-only cases. FlagTree
itself is **not** validated on Ascend: its Ascend backend exists only on the 3.5 line, and
it routes its host runtime through `torch_npu`, which claims PrivateUse1 and so prevents
`torch_fl` from registering `flagos` at all. `torch_fl/compile/flagtree_ascend_policy.py`
registers a torch_npu-free backend policy to lift that block, but it is verified only up to
compiling the emitted C++ against ATen and resolving every strategy through FlagTree's real
registry — no FlagTree build carrying the Ascend backend is installed here, so end-to-end
kernel execution through FlagTree remains unvalidated
([upstream issue](https://github.com/flagos-ai/FlagTree/issues/1046)). Forward, backward, fused
elementwise and matmul+normalization graphs match eager. Three triton-ascend defects are
worked around rather than avoided — a miscompiled masked byte load, `ub overflow` raised as
a hard error, and a compile-worker/parent launch segfault — so the platform compiles
serially by default and has no C++ wrapper codegen; see
[`docs/architecture/torch-compile-integration.md`](../architecture/torch-compile-integration.md).
This is not in CI: the Ascend runner image does not carry `triton-ascend`, so the compile
suite must be run by hand on hardware that does.

### PPU

PPU is not a separate `ACCELERATOR` value. It is `ACCELERATOR=cuda` plus `PPU_SDK`/`PPU_HOME`
detection (see [`setup.py`](../../setup.py), lines 43-46 and 732-736), reusing the CUDA-boxing
build against the PPU's CUDA-13-compatible SDK. There is no CI manifest for this platform (no
`ppu.yml` under `.github/configs/`), so all capabilities here rest on the
[README](../../README.md)'s build-from-source instructions rather than automated tests.
The shared `AutocastPrivateUse1` policies and CUDA-boxing AMP routes expose
`torch.autocast("flagos")` and `torch.amp.GradScaler("flagos")`, with FP16 and
BF16 as the advertised target dtypes. The AMP contract is covered by
`tests/integration/test_amp_contract.py` (`-m amp`), but its runtime results are
only PPU evidence when run with `PPU_SDK` or `PPU_HOME` against a real PPU
device; ordinary NVIDIA CUDA and CPU runs do not validate this row. The current
PPU validation covered both FP16 and BF16 autocast, the mutable `found_inf`
unscale path, finite scale growth, overflow backoff, and an autocast training
step. The result remains experimental because it is not covered by CI.
FlagGems requires a vendor-index Triton build whose version string does not satisfy the
project's `triton>=3.5.1` pin (see [`setup.py`](../../setup.py), lines 790-807). Distributed
support is described as working via the NCCL fallback with a vendor-adapted `libnccl.so.2`, but
this is not covered by CI.

### Hygon DCU

[`.github/configs/dcu.yml`](../../.github/configs/dcu.yml) (lines 84-99) runs vendor-backend
and FlagGems-runtime operator suites, general tests, and the profiler parity suite in CI;
inference and training smoke are explicitly deferred pending a model mount and card-count
confirmation on the runner (same file, lines 57-61). DCU is a CUDA-boxing build over the
hipified DTK torch, so it reuses the generated CUDA boxing kernels with no hand-written kernels
of its own (see [`setup.py`](../../setup.py), lines 376-397). FlagGems C++ dispatch is not built
for DCU: [`CMakeLists.txt`](../../CMakeLists.txt) line 81 and [`setup.py`](../../setup.py) lines
376-393 force `FLAGGEMS_KERNEL=OFF` with the comment "FLAGGEMS_KERNEL needs liboperators.so,
which is not built for DTK," and [`.github/configs/dcu.yml`](../../.github/configs/dcu.yml) line
87 excludes C++ tests with `-m "not flaggems_cpp"`. Distributed collectives (all_reduce,
broadcast, all_gather, all_gather_into_tensor, reduce_scatter_tensor) and DDP are measured
working on 2 cards via RCCL (see
[`torch_fl/comm/process_group.py`](../../torch_fl/comm/process_group.py), lines 103-107), but
this measurement is not in CI. Mixed-precision training is validated on real DCU hardware with
FP16 and BF16 autocast policies, nested autocast state, finite and overflowing GradScaler steps,
and forward/backward optimizer execution. CPU-to-DCU copies and representative elementwise,
reduction, matrix multiplication, and backward operations preserve all tested PyTorch dtypes from
bool through complex128, including float64.

`torch.compile(backend="flagos")` is experimentally validated on Hygon with a
FlagTree 0.6.0 HCU build and PyTorch 2.10.0. The full compile integration suite
passes on the `gfx936` target, including forward and backward execution, FP32
and FP16 inputs, max-autotune, recompilation, FakeTensor tracing, and the
FlagTree-specific result/device checks. The FlagTree wheel must replace the
active `triton` installation before Python starts; the current DCU CI image
uses DTK Triton and does not exercise this path.

### Enflame GCU

[`.github/configs/gcu.yml`](../../.github/configs/gcu.yml) builds an isolated CPU-PyTorch wheel
and runs it against an S60 runner without importing the vendor `torch-gcu` plugin. GCU takes the
native operator route through `libtopsaten.so`, with ops lacking a topsaten kernel reaching
`cpu_fallback` instead of raising, and int64 operands always falling back to CPU because
topsaten has no int64 kernels (see [`setup.py`](../../setup.py), lines 414-439). FlagGems can
reach GCU only through the Python dispatch layer with Enflame's `triton_gcu` plugin, and
registration is skipped when it or the toolchain is missing (same file, lines 420-429); the CI
image does not install that stack, so the operator step excludes FlagGems markers.

The CI steps are the same contract suites the other platforms run, selected by marker rather than
by a file allowlist: the operator suite (568 passed, 33 skipped on S60), the full
`test_rng_dispatch.py` (111 passed), `test_factory_ops.py` (46 passed), and the shared AMP
contract `test_amp_contract.py` (measured as `test_amp.py`, 25 passed, before the suites were
unified). RNG is native: seeds are reserved from the flagos PrivateUse1 generator via
`c10::flagos::ReserveSeed`, so `torch.Generator(device="flagos")`, `manual_seed`,
`manual_seed_all`, `get_rng_state`, and `set_rng_state` all drive the same per-device stream, and
`random_` follows ATen's default integer bounds.

`test_profiler_contract.py` is excluded. The TOPSPTI tracer does collect activities on S60 (229
for a three-iteration matmul loop), but none surface as device events, because the CPU-only
PyTorch 2.10 wheel used here provides no PrivateUse1 Kineto resolver — the same environment
limitation recorded for MUSA rather than a GCU tracer defect. Distributed support is not
represented in tests or docs for this platform.

### Moore Threads MUSA

No CI manifest exists for this platform. MUSA takes the native operator route through `mudnn`,
with a documented CPU fallback for ops the vendor kernel table does not cover (see
[`setup.py`](../../setup.py), lines 440-457, and
`tests/integration/ops/test_musa_dispatch.py`, lines 15-24). No CUDA boxing kernels or FlagGems
C++ kernels are compiled for this platform; FlagGems Python dispatch is optional and needs the
vendor Triton backend. The shared `AutocastPrivateUse1` policies and native mudnn routes were
measured on an MTT S5000 with both FP16 and BF16. The 25-case AMP suite covers lower-precision
matmul, linear, and convolution; float32 and promote policies; nested autocast state; finite scale
growth; overflow backoff and optimizer-step skipping; and end-to-end GradScaler training. The
GradScaler unscale operation currently follows the correctness-oriented CPU fallback and copies
its mutated tensors and `found_inf` flag back to MUSA rather than using a native mudnn foreach
kernel. Distributed support remains unvalidated on hardware. The optional MUPTI tracer has been
measured on the available MTT S5000 host and emits real positive-duration kernel, runtime, and
memcpy activities with device, stream, name, and Chrome-trace metadata. `torch.compile` is
validated on this host against the vendor `flagtree-0.5.0+mthreads3.1` runtime and is not
established by stock Triton: the 22-case `tests/integration/test_compile.py` suite passed on the
MTT S5000, covering forward and backward execution, FP32/FP16 inputs, max-autotune,
recompilation, FakeTensor tracing, and output/gradient placement on `flagos`. FlagTree reaches the
device through `torch_fl.compile.flagtree_shim`, which rebinds the vendor MThreads driver's runtime
lookups onto `torch_fl`; the separate `torch_musa` plugin is not installed and its `__init__` is
never imported, since PrivateUse1 admits only one owner. The suite asserts that every rebound
driver callable resolves inside `torch_fl` rather than asserting the module name is absent, because
`torch_fl` publishes its own small compatibility surface under that name for FlagGems discovery.
Runtime coordinate-descent autotuning is disabled on MUSA because it benchmarks through CUDA
events the CPU PyTorch wheel does not provide. CPU-to-device linkage
depends on whether the installed PyTorch/Kineto build supplies the PrivateUse1 resolver; the
CPU-only PyTorch 2.10 wheel used for this measurement does not justify a general parity claim.

### D-Robotics BPU

BPU is **Runtime only** for eager execution: every kernel set is disabled at build time for
`ACCELERATOR=bpu` (see [`setup.py`](../../setup.py), lines 398-413), so eager compute falls
back to CPU. Acceleration is graph-level, through `torch.compile(backend="bpu")`, which
partitions a traced graph, quantizes it, and compiles it with hbdk4 into a `.hbm` artifact run
on the board's native runtime (see [`docs/vendors/bpu/integration.md`](../vendors/bpu/integration.md),
lines 1-19 and the "Compile pipeline" section). This graph path has measured results on
ResNet-18 and Qwen3-0.6B against vendor artifacts (same file, lines 296-352), which is why it is
marked Experimental rather than Runtime only in the `torch.compile` column, while eager
execution itself has no device kernel path at all. No CI manifest exists for this platform; the
evidence above is from on-board measurement, not continuous validation.

### TsingMicro

TsingMicro has a build selector (`ACCELERATOR=tsingmicro`) and links against the Kuiper SDK
(see [`CMakeLists.txt`](../../CMakeLists.txt), lines 196-216), but
[`setup.py`](../../setup.py) (lines 367-375) disables every kernel set for it, mirroring the
same "no per-op kernel to call" treatment documented for BPU (see
[`docs/vendors/bpu/integration.md`](../vendors/bpu/integration.md), lines 37-43). There is no
recoverable current install or test guide, and no CI manifest, for this platform. It is marked
**Runtime only**, and setup and operator validation should be treated as not currently
documented.

## Reading the Matrix

A "Stable" or "Beta" entry in the **Project Compatibility** table describes the project's
overall Python/PyTorch contract, not platform-wide validation. Each cell in the **Platform
Matrix** reflects only the evidence cited in that platform's notes above: a capability marked
"Not validated" may still work, but no test, CI manifest, or measurement in this repository
currently backs that claim. Distributed and profiler entries in particular distinguish
architectural routing (a table entry, a code path that exists) from on-hardware verification
(a CI job or a documented, reproducible measurement) — treat the latter as the bar for anything
described as validated.
