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
| Python | One version per platform | Computed in `setup.py`: a FlagTree build exists for exactly one cp tag and the wheel links it, so the interpreter is fixed — 3.12 on CUDA/GCU/MetaX/PPU, 3.10 on DCU/MUSA, 3.11 on Ascend. `Requires-Python` says so, and the wheel filename repeats it. |
| PyTorch | 2.10.x (`>=2.10,<2.11`) | Generated ATen bindings are tied to this minor line. |
| FlagGems | The version in `.github/version-pins.env` | Declared as an exact requirement, not a range: the per-op routing tables were generated against one cohort. Published to the platform's vendor lane. |
| FlagTree | The platform's pin in `.github/version-pins.env` | Declared as an exact requirement and it *is* Triton — the package name carries the vendor's backend (`0.7.0+hcu3.6`, the trailing number being the Triton line). The pin tracks the newest build published to the platform's own lane. Platforms with no FlagTree build declare `triton>=3.5.1` instead. |
| FlagCX | The platform's pin in `.github/version-pins.env` | Declared only where the vendor runtime has a build (CUDA, DCU, GCU, MetaX, MUSA, Ascend); PPU has none yet, and on the rest the distributed path falls back to the NCCL-shaped route. |
| Triton/compiler | Comes from FlagTree | Use the compiler distribution required by the selected accelerator. |

The index requirements for installing those are in
[Installation](../getting-started/installation.md#runtime-dependencies-and-the-package-index).

## Wheel compatibility record

Every built wheel contains `torch_fl/compatibility.json`. It records the selected
platform and kernel sets, bundled libtorch location, build-time PyTorch version
and C++ ABI flag, the installed FlagTree/FlagGems/FlagCX versions observed during
the build, the explicitly declared vendor PyTorch version when one supplies
device libraries, and the wheel's Python requirements. These fields describe
the artifact; an observed build version is evidence of testing, not an implicit
pin on every downstream install. The package's version ranges remain the
installation contract.

An SDK version is recorded only when the builder sets `FLAGOS_SDK_VERSION` to a
verified value. An absent value means **unknown**, not compatible with all SDKs.
For a release wheel that declares vendor PyTorch libraries, set
`FLAGOS_VENDOR_TORCH_VERSION` to the version that supplied those libraries.
`--release` rejects a missing SDK version for every wheel, and a missing
vendor PyTorch version for any wheel that does declare a vendor libtorch. The
build-time C++ ABI flag describes the PyTorch interpreter used
for compilation; a vendor libtorch bundle may have a separate ABI provenance
that must be checked against the vendor SDK. This manifest does not prove host
driver compatibility.

Inspect a wheel before importing its native extension:

```bash
python scripts/tools/torch-fl-preflight --wheel dist/torch_fl-*.whl --platform cuda \
  --sdk-version 13.3 --check-installed
```

`--check-installed` checks declared dependency ranges, warns when an installed
optional package differs from the build environment, and warns when PyTorch is
within the declared range but not the exact build version. `--check-build-env`
requires exact build dependency versions and is used by every wheel CI job.
`--require-sdk` rejects wheels without a declared SDK version. Use
`scripts/tools/torch-fl-preflight` to inspect files before installation or run
`torch-fl-preflight --installed` to inspect the installed wheel. Both accept a
platform ID or its vendor name, such as `mps` or `apple`. The CLI reads
metadata without loading PyTorch or the native extension. SDK and driver
compatibility still requires platform-specific testing.
Generate a release table from the final wheel files with
`python scripts/tools/torch-fl-preflight --wheel dist/*.whl --release --markdown-table`;
the command fails if any artifact lacks its required provenance.

## Platform Matrix

| Platform | Build selector | Execution path | Eager and autograd | `torch.compile` | Distributed | Profiler | FlagGems | Status |
|---|---|---|---|---|---|---|---|---|
| NVIDIA CUDA | `FLAGOS_ACCELERATOR=cuda` (default) | FlagGems and measured native dtype routes | Stable | Experimental (inductor GPU device registered; no CI test step) | Beta (FlagCX + NCCL fallback, DDP live-verified) | Stable (CUPTI parity) | Beta (Python + C++ dispatch paths) | Stable |
| MetaX | `FLAGOS_ACCELERATOR=metax` | FlagGems, with optional AOT native hosts | CI-covered operator dispatch and AOT native hosts | Experimental (vendor Triton and FlagTree MetaX measured on C550; vendor Triton CI-covered) | Experimental (NCCL-shaped `mccl` fallback; not CI-covered) | Experimental (MCPTI parity measured on C550; not CI-covered) | FlagGems Python dispatch and AOT host checks in CI | Stable |
| Ascend | `FLAGOS_ACCELERATOR=ascend` | Native ACLNN operator backend, FlagGems via FlagTree (Triton 3.5) | Stable (CI-covered ops, RNG suite) | Experimental (inductor measured on 910 against triton-ascend 3.2.0 only; **not** revalidated on FlagTree, three toolchain workarounds, serial compile, no CI step) | Experimental (HCCL fallback; architectural routing only, no collective-level CI) | Beta (MSPTI kernel/runtime/flow/memcpy events plus device-time linkage, all CI-covered by the shared contract; parity suite excluded from CI) | Beta (Python dispatch; FlagGems-first conf, float64 and bool `neg` routes fall back to ACLNN) | Beta |
| PPU | `FLAGOS_ACCELERATOR=ppu` | FlagGems through the T-Head PPU runtime | Ordinary wheel operator tests in CI | Not validated | Experimental (NCCL fallback via vendor-adapted `libnccl.so.2`; not CI-covered) | Not validated on this vendor's tracer | Experimental (vendor-index Triton required) | Experimental |
| Hygon DCU | `FLAGOS_ACCELERATOR=dcu` | FlagGems and optional HCU native hosts, using the DTK runtime | Beta (ordinary wheel and native-host checks in CI) | Experimental (FlagTree HCU validated on `gfx936`; not in CI) | Experimental (RCCL via DTK; all_reduce/DDP measured on 2 cards, not in CI) | Beta (parity suite runs in CI) | Beta (Python dispatch only) | Beta |
| Kunlunxin (Kunlun P800) | `FLAGOS_ACCELERATOR=kunlunxin` | Native XPU backend over the vendor xdnn operator library, with 256 P800-measured FlagGems routes | Experimental (native device layer and vendor route not yet measured on P800 hardware; not in CI) | Not validated | Not validated | Not available (kunlunxin is not in the CUPTI tracer arm of the selection in `csrc/CMakeLists.txt`, so the build compiles the unavailable tracer) | FlagGems routes measured on the P800 (256 of a 618-op ceiling; ten profiles at 1e-4) | Experimental |
| Enflame GCU | `FLAGOS_ACCELERATOR=gcu` | Native `libtopsaten.so` operator backend, with CPU fallback for unrouted/int64/float64 ops | Beta (operator, RNG, factory, and AMP suites CI-guarded on S60) | Not validated | Not validated | Runtime only (TOPSPTI collects activities; no device events on a CPU-only Kineto build) | Experimental (Python dispatch, requires vendor Triton) | Beta |
| Moore Threads MUSA | `FLAGOS_ACCELERATOR=musa` | Native `mudnn` operator backend, with CPU fallback for unrouted ops | Experimental (including FP16/BF16 autocast and GradScaler measured on MTT S5000) | Experimental (MThreads FlagTree forward/backward measured on MTT S5000; vendor runtime required) | Not validated | Experimental (MUPTI device timeline measured on MTT S5000; CPU-Kineto linkage is environment-dependent) | Experimental (Python dispatch, requires vendor Triton) | Experimental |
| D-Robotics BPU | `FLAGOS_ACCELERATOR=bpu` | No eager kernel sets are built; eager ops run on CPU | Runtime only (CPU fallback for eager) | Experimental (`torch.compile(backend="bpu")` graph path via hbdk4) | Not applicable | Not validated | Not applicable (no per-op kernel build) | Runtime only |
| TsingMicro | `FLAGOS_ACCELERATOR=tsingmicro` | FlagGems-primary through the Kuiper TX runtime; the only native kernels are the host-draw RNG ops | Experimental (FlagGems routes, RNG and metadata-only views measured on TX8110; no CI) | Not validated | Not validated | Not validated | Experimental (Python dispatch via the `torch.txda` shim; no CI) | Experimental |

## Platform Notes

### NVIDIA CUDA

The NVIDIA CI job uses an ordinary FlagGems wheel. The integration manifest selects
operator and factory tests, profiler checks, and Qwen3-0.6B inference and training.
Distributed collectives and DDP gradient sync are live-verified on 2x/8x A100 through FlagCX
with an NCCL fallback (see
[`docs/architecture/distributed-flagcx.md`](../architecture/distributed-flagcx.md), lines 1-10
and §6). `torch.compile` registers flagos as a first-class inductor GPU device with integration
documented in
[`docs/architecture/torch-compile-integration.md`](../architecture/torch-compile-integration.md)
(lines 84-127). The integration manifest runs the compile suite with FlagTree.
The profiler achieves structural
parity with torch-cuda via CUPTI (see
[`docs/architecture/profiler.md`](../architecture/profiler.md), lines 1-21).

### MetaX

The MetaX CI job runs the operator, factory, RNG, AMP policy, and compile tests
selected in the [integration manifest](../../.github/integration-manifest.yml).
The same job builds and checks an opt-in native-host wheel. The
[MetaX installation guide](../vendors/metax/installation.md) describes the
two wheel builds and records the earlier C550 measurements.

The communicator uses the vendor's NCCL-shaped MCCL runtime. MetaX still
needs collective-level CI evidence and profiler measurements on the
current wheel across SDK versions and streams.

### Ascend

[`.github/configs/ascend.yml`](../../.github/configs/ascend.yml) (lines 78-97) runs the
Ascend-marked operator suite and the full RNG dispatch suite in CI. FlagGems-marked tests in
`test_rng_dispatch.py` are selected via `-m "main_ops"` (lines 90-94) but the FlagGems runtime
path is not enabled in CI, so those tests run against the native ACLNN backend. Model-level
(Qwen3) validation exists as a point measurement outside CI — training at 0.82x `torch_npu` on
real Ascend 910 hardware (`tests/perf/e2e_qwen3_train_ascend.py`) — but
[`.github/configs/ascend.yml`](../../.github/configs/ascend.yml) (line 110) explicitly defers
Qwen3 inference/training smoke pending a model mount on the CI runner. The
profiler parity suite compares against a torch-cuda baseline and is outside
the Ascend CI selection. The shared profiler contract runs on Ascend. MSPTI
emits named kernel, runtime/API, flow, and (under the process-start preload above) memcpy
activity, and the step was measured on 910 at `11 passed, 1 skipped`. The skip is
`gpu_memset`, a correct-by-design consequence of routing `torch.zeros()` through
`aclnnInplaceZero` rather than the allocator (see
[`docs/architecture/profiler.md`](../architecture/profiler.md)). The device-time
linkage case is one of the eleven that assert: MSPTI used to stamp every launch with
the id of the CPU op that started after it, which turned out to be this repository's
defect rather than the vendor's — `popCorrelation` passed a null out-parameter, which
CANN rejects without unwinding, so the external-correlation stack grew for the life of
the process (issue #425). With the pop fixed, the launching
operator owns its device events and the case asserts on this platform. Distributed support
recommends the FlagCX path (see
[`docs/architecture/distributed-flagcx.md`](../architecture/distributed-flagcx.md), lines
204-208); the native HCCL fallback and the `flagos→npu` zero-copy view are architectural
routing, not on-hardware-verified collectives (same file, lines 67-75).

The FlagGems route runs on FlagTree (`0.6.2a1+ascend3.5`, Triton 3.5), and the
route table is generated FlagGems-first with ACLNN as the fallback for everything
FlagGems cannot compile or run. Because a conf cannot express a per-dtype
exception, float64 calls that the conf routes to FlagGems are redirected to
ACLNN at runtime by `FlagGemsRejectsDtype` (`csrc/aten/common.cc`); see
[`docs/reference/operator-support.md`](operator-support.md) for the measured
route delta. The 910C CI runner is **not revalidated** — these results were
measured on a 910/910B host.

`torch.compile(backend="flagos")` is experimentally validated on Ascend through
`triton-ascend` 3.2.0. Measured on a real 910 (`Ascend910_9382`, CANN 9.0.0,
torch 2.10.0+cpu, Python 3.10): `tests/integration/test_compile.py` passes 30 of 32 with a
cold inductor cache, the two skips being the FlagTree-only and MetaX-only cases. That
measurement predates the move of Ascend's FlagGems route to FlagTree, and no CI step
or hand run has re-measured the compile path on the current toolchain, so treat these
numbers as unvalidated there. Forward, backward, fused
elementwise and matmul+normalization graphs match eager. Three triton-ascend defects are
worked around rather than avoided — a miscompiled masked byte load, `ub overflow` raised as
a hard error, and a compile-worker/parent launch segfault — so the platform compiles
serially by default and has no C++ wrapper codegen; see
[`docs/architecture/torch-compile-integration.md`](../architecture/torch-compile-integration.md).
Whether any of the three still applies to FlagTree is unmeasured.
This is not in CI: the Ascend runner image does not carry the Triton toolchain, so the
compile suite must be run by hand on hardware that does.

FlagTree's Ascend backend resolves its host runtime through `torch_npu`, which claims
PrivateUse1 and so would prevent `torch_fl` from registering `flagos` at all.
`torch_fl/compile/flagtree_ascend_policy.py` registers a torch_npu-free backend policy to
lift that block, and `torch_fl.flagos` installs it during device init so that eagerly
launched FlagGems kernels take it; the emitted C++ compiles against real ATen headers and
every strategy resolves through FlagTree's real registry
([upstream issue](https://github.com/flagos-ai/FlagTree/issues/1046)).

### PPU

T-Head builds the ordinary FlagGems wheel with the PPU SDK. The
[platform config](../../.github/configs/thead.yml) supplies the CI environment,
and the [integration manifest](../../.github/integration-manifest.yml)
selects the operator and runtime tests. Earlier FP16/BF16 autocast and
GradScaler measurements used the retired wheel.
FlagGems requires a vendor-index Triton build whose version string does not satisfy the
project's `triton>=3.5.1` pin (see [`setup.py`](../../setup.py), lines 790-807). Distributed
support is described as working via the NCCL fallback with a vendor-adapted `libnccl.so.2`, but
this is not covered by CI.

### Hygon DCU

The Hygon CI job runs ordinary FlagGems and general operator tests, then
builds and tests the opt-in HCU native hosts. The ordinary wheel uses the
DTK runtime with the upstream PyTorch core. FlagGems C++ dispatch requires
a DTK `liboperators.so`, which the platform does not ship. Multi-card
collectives and DDP were measured on two DCUs using RCCL (see
[`torch_fl/comm/process_group.py`](../../torch_fl/comm/process_group.py)).

The DCU wheel runs DTK's device libraries on the **official** PyTorch core rather
than DTK's forked one, so the installed `torch` must match the minor version the
bundle was built against (`2.10.0` for DTK 2604), and `torch_fl` must be imported
before `torch` in a fresh process. The 32 DTK-private ATen symbols the device
libraries and the plugin import are supplied by `libflagos_dtk_core_compat.so`,
whose coverage is re-proved from the binaries on every bundle; DTK's private fused
ops (`aten::native_fuse_rmsnorm` and friends) are therefore unreachable unless the
build and runtime both set `FLAGOS_DCU_VENDOR_CORE=1`. See
[DCU without DTK's core libraries](../vendors/dcu/vendor-free-core-libs.md).

`torch.compile(backend="flagos")` is experimentally validated on Hygon with a
FlagTree 0.6.0 HCU build and PyTorch 2.10.0. The full compile integration suite
passes on the `gfx936` target, including forward and backward execution, FP32
and FP16 inputs, max-autotune, recompilation, FakeTensor tracing, and the
FlagTree-specific result/device checks. The FlagTree wheel must replace the
active `triton` installation before Python starts; the current DCU CI image
uses DTK Triton and does not exercise this path.

### Kunlunxin (Kunlun P800)

Kunlunxin has its own `FLAGOS_ACCELERATOR=kunlunxin` value (see
[`cmake/flagos_platforms.json`](../../cmake/flagos_platforms.json)) and is a
**native PrivateUse1 backend** in the shape of MUSA / GCU / Ascend: the device
layer under `csrc/runtime/accelerator/kunlunxin/` calls the XPU SDK's own native
C runtime (`libxpurt`, headers under `<xpu/runtime.h>`) for device discovery,
allocation, host/device copies and streams, and the `kunlunxin` route key reaches
the vendor xdnn op library (`libxdnn_pytorch.so`, shipped inside the vendor torch
package). The XPU SDK's
CUDA-shaped surface (`cuda_runtime.h`, `libcudart.so.12`) is retained only as the
build input for the `csrc/runtime/hooks.h` `isPinnedPtr()` probe, which carries no
kernel and is unreachable from the routing layer. The build takes the host
compiler and does not need `nvcc`. FlagGems C++ stays off
(`FLAGOS_BUILD_FLAGGEMS_CPP=OFF`, since the XPU SDK ships no `liboperators.so`).

The default config,
[`torch_fl/configs/backends_kunlunxin.conf`](../../torch_fl/configs/backends_kunlunxin.conf),
is generated full-coverage from the `VENDORS` table: every covered op routes to
the native `kunlunxin` key or is left at `none`, with **no `cuda` route**.
Kunlunxin is deliberately absent from `FLAGGEMS_PYTHON_PLATFORMS` in
[`scripts/codegen/gen_vendor_confs.py`](../../scripts/codegen/gen_vendor_confs.py),
so no operator is rerouted to a FlagGems backend: the measured FlagGems allowlist
is a follow-up PR. There is no `.github/configs/kunlunxin.yml`, so no CI job
validates this platform; the checked-in harness to run on hardware is
`tests/manual/kunlunxin_runtime_smoke.py` (see
[`docs/vendors/kunlunxin/installation.md`](../vendors/kunlunxin/installation.md)).

Kunlunxin is not in the CUPTI tracer arm of the selection in
[`csrc/CMakeLists.txt`](../../csrc/CMakeLists.txt) (the `cuda` / `metax` / `ppu`
arm), so the build falls through to the final `else()` and compiles the
`unavailable_device_tracer`. Import is unaffected and `start()` is a no-op;
device profiling is not available on this platform.

**Runtime measurement boundary.** The native device layer and the vendor route
have **not** been measured on P800 hardware in this change — the de-box replaces
the runtime sources an earlier CUDA-boxing measurement exercised, so that result
does not transfer. Run `tests/manual/kunlunxin_runtime_smoke.py` (device count,
device index, allocation, host↔device round-trip, `Stream`, `synchronize`,
`empty_cache`, and `mm`) on a P800 before reading the route list as validated
operator support. The earlier boxing measurement was itself limited: the vendor
XPU runtime on that box is torch-2.9-bound, so eager ops that reach ATen's CUDA
dispatch raised `NotImplementedError` under torch 2.10 and nothing on the `cuda`
route executed.

### Enflame GCU

[`.github/configs/enflame.yml`](../../.github/configs/enflame.yml) builds an isolated CPU-PyTorch wheel
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

A CI manifest exists for this platform ([`.github/configs/mthreads.yml`](../../.github/configs/mthreads.yml)),
covering the isolated-environment preflight, MUSA dispatch, general factory ops, the unified AMP,
math-bits and profiler contracts, the native mudnn operator subset, and the RNG suite. Three groups
other platforms run are withheld with the reason recorded in that manifest: `torch.compile` and the
FlagGems hybrid route both need the vendor Triton stack the CI image does not bake, and profiler
parity compares against a torch-cuda baseline and is outside the MUSA CI selection.
MUSA takes the native operator route through `mudnn`,
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
memcpy activities with device, stream, name, and Chrome-trace metadata. The unified profiler
contract runs in CI for this platform and passed 11 of 12 cases on the MTT S5000 on 2026-08-29,
including device kernel and runtime events, flow pairing, CPU-to-device linkage, kernel-name
demangling and capture-window containment. Only `gpu_memset` is skipped, through the capability
table rather than a deselect, because MUPTI emits no memset records. `torch.compile` is
validated on this host against the vendor `flagtree-0.5.0+mthreads3.1` runtime and is not
established by stock Triton: `tests/integration/test_compile.py` passed on the MTT S5000 (25 passed
on 2026-08-29), covering forward and backward execution, FP32/FP16 inputs, max-autotune,
recompilation, FakeTensor tracing, and output/gradient placement on `flagos`. It is not yet a CI
group, because the CI image bakes no vendor Triton stack. FlagTree reaches the
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
`FLAGOS_ACCELERATOR=bpu` (see [`setup.py`](../../setup.py), lines 398-413), so eager compute falls
back to CPU. Acceleration is graph-level, through `torch.compile(backend="bpu")`, which
partitions a traced graph, quantizes it, and compiles it with hbdk4 into a `.hbm` artifact run
on the board's native runtime (see [`docs/vendors/bpu/integration.md`](../vendors/bpu/integration.md),
lines 1-19 and the "Compile pipeline" section). This graph path has measured results on
ResNet-18 and Qwen3-0.6B against vendor artifacts (same file, lines 296-352), which is why it is
marked Experimental rather than Runtime only in the `torch.compile` column, while eager
execution itself has no device kernel path at all. No CI manifest exists for this platform; the
evidence above is from on-board measurement, not continuous validation.

### TsingMicro

TsingMicro has a build selector (`FLAGOS_ACCELERATOR=tsingmicro`) and links against the Kuiper SDK
(see [`CMakeLists.txt`](../../CMakeLists.txt), lines 196-216). Unlike BPU, it does have an
eager operator path:
[`torch_fl/configs/backends_tsingmicro.conf`](../../torch_fl/configs/backends_tsingmicro.conf)
routes its ops to FlagGems, and the `#else` branch of
[`csrc/aten/register.cc`](../../csrc/aten/register.cc) claims them. There is no
vendor-native operator library to fall back to — TsingMicro's compute path is FlagGems
through a vendor Triton build (there is no FlagTree wheel for this platform), and a build
that cannot reach FlagGems has nothing behind it. The exception is the host-draw RNG family
in [`csrc/aten/backends/tsingmicro/rng.cc`](../../csrc/aten/backends/tsingmicro/rng.cc),
which `register.cc` claims directly because TX8 has no random-number unit.

The 2026-Q3 TX8 measurement below used the retired wheel, which compiled the full
generated `register.inc`. Because that list was claimed on
PrivateUse1 while FlagGems supplies no kernel for `empty_like`, the op resolved through the
dispatcher to the CUDA boxing kernel, which needs a CUDA runtime a TsingMicro CPU-torch
build does not have, so it raised `NotImplementedError: Could not run
'aten::empty_strided' with arguments from the 'CUDA' backend`. Every FlagGems kernel that
allocates its output through `torch.empty_like` (FlagGems' `pointwise_dynamic`, used by
`add`/`mul`/`silu`/`softmax` and the rest of the pointwise family) was therefore
unreachable, and `cat`/`rms_norm` failed through CUDA calls for the same reason. The
matmul family and the reductions that allocate differently (`mm`, `bmm`, `sum`, `mean`,
`randn`) did run and match the CPU reference. The TsingMicro-scoped registration subset
the measurement called for — a `register_flaggems_tsingmicro.inc`, as MUSA uses — is now
what the current build compiles
(`scripts/codegen/codegen_flaggems_register.py`), so `empty_like`, `cat` and
`rms_norm` fall through to `cpu_fallback` instead of resolving to CUDA. The default build
was re-measured on a TX8110 on 2026-10-21 (`torch 2.10.0+cpu`, `flag_gems 5.4.0`,
`triton 3.6.0`): `import torch_fl`, device enumeration, `randn`, `mean`, `unsqueeze`,
`erfc` and `cat` all succeed, and the RNG family is served by the host-draw kernels. That
run was a smoke test, not the per-overload survey the 2026-Q3 split above came from, so
that split remains the last per-operator correctness record. See the TsingMicro row in
[`docs/reference/operator-support.md`](operator-support.md).

This was measured on a TX8 host, not in CI. There is no CI manifest for this platform;
installation and verification steps are in
[the TsingMicro guide](../vendors/tsingmicro/installation.md).

## Reading the Matrix

A "Stable" or "Beta" entry in the **Project Compatibility** table describes the project's
overall Python/PyTorch contract, not platform-wide validation. Each cell in the **Platform
Matrix** reflects only the evidence cited in that platform's notes above: a capability marked
"Not validated" may still work, but no test, CI manifest, or measurement in this repository
currently backs that claim. Distributed and profiler entries in particular distinguish
architectural routing (a table entry, a code path that exists) from on-hardware verification
(a CI job or a documented, reproducible measurement) — treat the latter as the bar for anything
described as validated.
