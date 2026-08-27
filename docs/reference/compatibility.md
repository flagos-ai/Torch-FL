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
| PyTorch | 2.9.x (`>=2.9,<2.10`) | Generated ATen bindings are tied to this minor line. |
| FlagGems | Platform dependent | Installed from PyPI or a vendor-compatible build only where the platform route uses it. |
| Triton/compiler | Platform dependent | Use the compiler distribution required by the selected accelerator. |

## Platform Matrix

| Platform | Build selector | Execution path | Eager and autograd | `torch.compile` | Distributed | Profiler | FlagGems | Status |
|---|---|---|---|---|---|---|---|---|
| NVIDIA CUDA | `ACCELERATOR=cuda` (default) | CUDA boxing over an external `libtorch_cuda.so` | Stable | Experimental (inductor GPU device registered; no CI test step) | Beta (FlagCX + NCCL fallback, DDP live-verified) | Stable (CUPTI parity) | Beta (Python + C++ dispatch paths) | Stable |
| MetaX | `ACCELERATOR=metax` | CUDA-boxing reuse via `cu-bridge`/mxcc, or native MetaX kernels | Stable | Not validated | Experimental (NCCL-shaped `mccl` fallback; not CI-covered) | Not validated on this vendor's tracer | Experimental (Python dispatch; not CI-tested on MetaX) | Stable |
| Ascend | `ACCELERATOR=ascend` | Native ACLNN operator backend, optional FlagGems via triton-ascend | Stable (CI-covered ops, RNG suite) | Not validated | Experimental (HCCL fallback; architectural routing only, no collective-level CI) | Runtime only (device/runtime events not emitted; profiler parity suite excluded from CI) | Experimental (Python dispatch; not CI-tested on Ascend) | Beta |
| PPU | `ACCELERATOR=cuda` + `PPU_SDK`/`PPU_HOME` detection | Same CUDA-boxing path as NVIDIA CUDA, against the PPU's CUDA-13-compatible SDK | Experimental | Not validated | Experimental (NCCL fallback via vendor-adapted `libnccl.so.2`; not CI-covered) | Not validated on this vendor's tracer | Experimental (vendor-index Triton required) | Experimental |
| Hygon DCU | `ACCELERATOR=dcu` | CUDA boxing over the hipified DTK torch build (HIP kernels under the CUDA dispatch key) | Beta (including FP16/BF16 autocast and GradScaler) | Not validated | Experimental (RCCL via DTK; all_reduce/DDP measured on 2 cards, not in CI) | Beta (parity suite runs in CI) | Beta (Python dispatch only) | Beta |
| Kunlun P800 | `ACCELERATOR=kunlun` | CUDA boxing over the XPU CUDA-compatibility runtime and vendor CUDA registrations | Experimental (runtime, copies, streams/events, `mm` smoke, and FP16/BF16 autocast with GradScaler measured; distributed AMP not validated) | Not validated | Not validated | Not validated | Experimental (5-route Python cohort; 4 STRICT, 1 BASIC_ONLY; remaining routes use boxing) | Experimental |
| Enflame GCU | `ACCELERATOR=gcu` | Native `libtopsaten.so` operator backend, with CPU fallback for unrouted/int64 ops | Experimental | Not validated | Not validated | Not validated | Experimental (Python dispatch, requires vendor Triton) | Experimental |
| Moore Threads MUSA | `ACCELERATOR=musa` | Native `mudnn` operator backend, with CPU fallback for unrouted ops | Experimental | Not validated | Not validated | Not validated | Experimental (Python dispatch, requires vendor Triton) | Experimental |
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

[`.github/configs/metax.yml`](../../.github/configs/metax.yml) (lines 99-125) runs representative
operator dispatch tests and general factory/autograd tests in CI, but its test command explicitly
excludes all FlagGems tests (`-m "not flaggems and not flaggems_python and not flaggems_cpp"` at
line 118), so FlagGems on MetaX has no CI validation. MetaX carries FSDP2 and Qwen3 training
parity work (see repository history). Distributed support routes through the same NCCL-shaped
fallback as CUDA (`_VENDOR_PROFILES["metax"]` in
[`torch_fl/comm/process_group.py`](../../torch_fl/comm/process_group.py), line 93), but that is
architectural routing, not a CI-verified collective test on this platform. `torch.compile` and
profiler-tracer parity are not represented in tests or docs for this platform.

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

### PPU

PPU is not a separate `ACCELERATOR` value. It is `ACCELERATOR=cuda` plus `PPU_SDK`/`PPU_HOME`
detection (see [`setup.py`](../../setup.py), lines 43-46 and 732-736), reusing the CUDA-boxing
build against the PPU's CUDA-13-compatible SDK. There is no CI manifest for this platform (no
`ppu.yml` under `.github/configs/`), so all capabilities here rest on the
[README](../../README.md)'s build-from-source instructions rather than automated tests.
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

### Enflame GCU

No CI manifest exists for this platform. GCU takes the native operator route through
`libtopsaten.so`, with ops lacking a topsaten kernel reaching `cpu_fallback` instead of raising,
and int64 operands always falling back to CPU because topsaten has no int64 kernels (see
[`setup.py`](../../setup.py), lines 414-439). FlagGems can reach GCU only through the Python
dispatch layer with Enflame's `triton_gcu` plugin, and registration is skipped when it or the
toolchain is missing (same file, lines 420-429). Distributed and profiler support are not
represented in tests or docs for this platform.

### Moore Threads MUSA

No CI manifest exists for this platform. MUSA takes the native operator route through `mudnn`,
with a documented CPU fallback for ops the vendor kernel table does not cover (see
[`setup.py`](../../setup.py), lines 440-457, and
`tests/integration/ops/test_musa_dispatch.py`, lines 15-24). No CUDA boxing kernels or FlagGems
C++ kernels are compiled for this platform; FlagGems Python dispatch is optional and needs the
vendor Triton backend. Distributed and profiler support are not represented in tests or docs for
this platform.

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
