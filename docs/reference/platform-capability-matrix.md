# Platform capability matrix

The twelve `FLAGOS_ACCELERATOR` values, what each is, and what it actually
supports. The tree layout made this "obvious" to maintainers but hid the
asymmetries — PPU is Python-only, TsingMicro has no Python compat, `soft_lowp`
is aten-only, and CI covers eight of the twelve — so this page is the
authoritative answer.

The directory inventory at the end is machine-checked by
`tests/unit/test_platform_capability_matrix.py`; update it with the directories
when a platform gains or loses one.

## Matrix

| `FLAGOS_ACCELERATOR` | Kind | runtime dir | aten backend | Python compat | conf routes | CI |
|---|---|---|---|---|---|---|
| `cuda` | native CUDA (FlagGems-primary; CUDA boxing opt-in) | `cuda` | — (C++ / FlagGems / CUDA) | `cuda` | 2034 | yes |
| `ppu` | CUDA-ABI (FlagGems-primary; PPU libtorch only in a boxing build) | `cuda` (+ `lib_ppu/` bundle) | — (FlagGems / boxing opt-in) | `ppu` | 2037 | yes |
| `metax` | CUDA-ABI (FlagGems-primary; native retired; boxing opt-in) | `metax` | — (FlagGems / boxing opt-in) | `metax` | 2037 | yes |
| `dcu` | CUDA-ABI (FlagGems-primary; DTK torch only in a boxing build) | `cuda` (+ `dcu` DTK-core shim) | — (FlagGems / boxing opt-in) | `dcu` | 2037 | yes |
| `kunlunxin` | vendor-native (xdnn op library) | `kunlunxin` | `kunlunxin` | — (no dir) | 2037 | yes |
| `ascend` | vendor-native (ACLNN) | `ascend` | `ascend` | `ascend` | 2037 | yes |
| `gcu` | vendor-native (topsaten) | `gcu` | `gcu` | `gcu` | 2037 | yes |
| `musa` | vendor-native (mudnn) | `musa` | `musa` | `musa` (stream/event only) | 2037 | yes |
| `tsingmicro` | CUDA-ABI (FlagGems-primary; boxing opt-in) | `tsingmicro` | — (FlagGems / boxing opt-in) | — (no dir) | 67 | no |
| `bpu` | no per-op kernels (graph compile) | `bpu` | — | `bpu` | 0 | no |
| `iluvatar` | FlagGems-primary + native vendor kernels (no boxing) | `cuda` | `iluvatar` (FlagGems register + ixdnn/ixblas native kernels) | — (no dir) | 2037 | no |
| `mps` | vendor-native (Metal / MPSGraph) | `mps` | `mps` (MPSGraph-first: `abs`, the elementwise `add`/`sub`/`mul` forms, `sum`/`mean`) | `mps` (stream/event only) | 2037 | no |

Column notes:

- **Kind** is the operator path the build selects (see
  `cmake/flagos_platforms.json` for the per-platform kernel-switch defaults).
  The CUDA-ABI platforms (`cuda`, `ppu`, `metax`, `dcu`, `tsingmicro`) are
  **FlagGems-primary by default**: `FLAGOS_BUILD_BOXING` is off, so the wheel
  compiles no CUDA boxing kernels, bundles no vendor libtorch, and routes the
  ops its conf names to FlagGems with everything else on `cpu_fallback`. Boxing
  is the explicit opt-in (`FLAGOS_BUILD_BOXING=1` / `-DFLAGOS_BUILD_BOXING=ON`)
  that restores the vendor-libtorch build.
- **runtime dir** is the `csrc/runtime/accelerator/<dir>` whose sources the
  device runtime is built from. `dcu` reuses `cuda` (DTK's CUDA compatibility
toolkit) and adds `dcu/` for its DTK-core ABI shim; `ppu` reuses `cuda` too.
  `kunlunxin` has its own `kunlunxin/` dir, built on the XPU SDK's native
  `libxpurt` runtime rather than the CUDA-shaped surface. `iluvatar` reuses
  `cuda` (the corex SDK presents a usable CUDA-shaped runtime, so the shared
  `cuda` sources are built as-is and there is no
  `csrc/runtime/accelerator/iluvatar/` directory).
- **aten backend** is the `csrc/aten/backends/<dir>` vendor kernel tree, when one
  exists. `flagos/` (the Python op caller) and `soft_lowp/` (the software
  low-precision matmul path shared by DCU/MetaX) are shared, not per-platform.
  `iluvatar` has a directory holding two registration `.inc` files — the
  generated `iluvatar_flaggems_register.inc` (the FlagGems Python/Triton routes,
  the platform default) and the native `iluvatar_register.inc` backed by
  `generated/iluvatar_kernels.cc` — whose kernels call corex's CUDA-named C ABI
  (`cudnn.h` → ixdnn, `cublas_v2.h` → ixblas) directly on the flagos
  `data_ptr()`, with no device retag and no CUDA dispatch key. The two sets are
  disjoint: FlagGems stays primary (an op in the FlagGems ceiling routes
  `flaggems`, and its native kernel is reachable only through an explicit
  `FLAGOS_FORCE_BACKEND=vendor` / `FLAGOS_OP_<name>=iluvatar` override — the
  runtime dtype escape has no `USE_ILUVATAR` arm), and the native kernels serve
  the ops FlagGems does not cover.
- **conf routes** is the number of `op = backend` entries in
  `torch_fl/configs/backends_<platform>.conf`. 2037 is the full-coverage set; 0
  and 67 are the by-design BPU/MPS and hand-written TsingMicro files.
- **CI** is whether `.github/configs/<platform>.yml` and a platform pipeline job
  exist.

## Shared kernel sets

`FLAGOS_BUILD_*` kernel switches (see `cmake/flagos_platforms.json`) decide which
sets a wheel compiles: `vendor` (the platform's native tree), `flaggems`
(FlagGems Python path), `flaggems_cpp` (FlagGems C++ path), `boxing` (generated
CUDA boxing kernels) and `tileops`. Which are on by default is a property of the
platform, not a per-build choice. `boxing` is off on every platform by default —
a CUDA-ABI wheel is FlagGems-primary and carries no vendor libtorch unless a
build opts into boxing.

## Known gaps

- **BPU** has no aten backend and an empty conf: its unit of execution is a whole
  compiled graph, so eager ops reach `cpu_fallback` and acceleration comes from
  the `torch.compile(backend="bpu")` path.
- **MPS** (Apple Silicon) has the device runtime floor — allocate, copy,
  synchronize and real Metal streams/events work — plus the first two stages of
  the MPSGraph operator layer: 9 ops run on the PrivateUse1 `mps` slot — `abs`,
  the elementwise `add`/`sub`/`mul` tensor and scalar forms (with aten's `alpha`),
  and the `sum`/`mean` reductions (with aten's axis and keepdim handling) — and
  everything else reaches `cpu_fallback`. The operator layer is MPSGraph-first
  with a hand-written MSL fallback (see
  `docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md`). It is
  native — no dispatch re-tag onto `DispatchKey::MPS`/`CUDA` and no CUDA boxing
  kernels (`FLAGOS_BUILD_BOXING` is pinned off). It has a
  `torch_fl/accelerator/mps/` directory holding only `mps_stream.py` (the
  stream/event wrappers `torch.flagos.Stream` dispatches to), so the Python
  compat column reads "stream/event only". There is no CI job: the pipeline is
  container-based on self-hosted runners, and a macOS arm64 runner has not been
  wired in.
- **TsingMicro** has a sparse hand-written conf (67 routes) and no CI job; its
  Python compat module does not exist.
- **MUSA** has no Python compat module and no compile-only CI job. It does have a
  `torch_fl/accelerator/musa/` directory, but it holds only `musa_stream.py` (the
  stream/event wrappers `torch.flagos.Stream` dispatches to) — not a
  `_musa_compat.py`, so the "Python compat" column below reads "stream/event only"
  rather than naming a module.
- **PPU** profiles through the CUPTI tracer but emits no `gpu_memset` activity, which is the
  capability `tests/integration/profiler_support.py` declares for it: a measured 512x512
  matmul trace carries the `kernel`, `gpu_memcpy` and `privateuse1_runtime` categories and no
  `gpu_memset`. That keeps the CUDA-baseline parity group off in `.github/configs/ppu.yml` —
  two of its tests need the memset category and one needs a `::`-qualified kernel name, which
  the vendor's `gemm_ktype0_*`/`sum_kernel_*` kernel set does not produce. It also ships no
  `lib/flagos_platform` marker (it uses the `lib_ppu/` bundle).
- **Kunlunxin** (Kunlun P800 XPU) is a native vendor backend: its
  `backends_kunlunxin.conf` routes ops to the `kunlunxin` key (the vendor xdnn
  op library) and leaves the rest at `none`, with **no `cuda` boxing route**. It
  has no `torch_fl/accelerator/` module. Its FlagGems cohort has
  not been revalidated on the P800, so no `flaggems` route is enabled yet and the
  platform is absent from `FLAGGEMS_PYTHON_PLATFORMS`; the allowlist is a
  follow-up.
- **Iluvatar** (BI-V150 / Tianshu, corex 4.5.0) is FlagGems-primary with a
  native, non-boxing vendor slot alongside it. Boxing with the CUDA kernels was
  explicitly ruled out; instead the ops FlagGems does not cover route to
  `csrc/aten/backends/iluvatar/`, whose kernels call corex's CUDA-named C ABI
  (ixdnn/ixblas) directly. Everything FlagGems covers still routes to the
  FlagGems Python/Triton path through FlagTree's iluvatar backend, and the rest
  falls back to `none` (`cpu_fallback`). A BI-V150 overload survey (2026-10-15,
  603-op cohort: the conf's own FlagGems routes) measured **STRICT 380 /
  BASIC_ONLY 66 / FAILED 21 / UNTESTED 136**; the 37 defect ops the previous
  639-op run measured failing are off the FlagGems route (to the native slot or
  to `cpu_fallback`), so the conf routes **603** `flaggems`, not the shared 640
  ceiling (`cat` measured separately in that earlier run, STRICT; in-band here).
  Most of the 37 are the
  `device_name="cuda"` vs `flagos` mismatch the DCU `patch_flaggems_device_name()`
  realignment addresses; adding iluvatar to `_ALIGNED_VENDORS` and re-measuring is
  the follow-up that would return them. It has no CI job and no runner lane — it
  builds locally only — and its wheel carries the local version segment
  `corex4.5.0`. The native kernel cohort is **not yet hardware-revalidated**: the
  ixdnn/ixblas path has a proven end-to-end conv2d result (bit-for-bit vs CPU on
  BI-V150) but the full cohort has not been re-measured.
- **CI covers 8 of 12**: TsingMicro, BPU, Iluvatar and MPS have no
  `.github/configs/` entry.

## Integration coverage

The eight platforms with a `.github/configs/<platform>.yml` run one integration
pipeline: a single ordered list of 22 groups in
`.github/integration-manifest.yml`, composed for each platform by
`.github/scripts/compose_integration_manifest.py` in the loader job. There is
no per-platform step list any more, so the eight run the same groups in the
same order and a config cannot add, drop or reword a step.

What a platform contributes to a step is a *value*, in one of two config keys:

- `environment_check` — the wheel, package and device expectations that
  `check_platform_environment.py` asserts before the pipeline starts. The
  manifest's `environment` group hands the whole block to that script as JSON
  through its `environment_from_config`, so the data a platform is checked
  against is the data it states.
- `step_environment` — a mapping of manifest step id to the environment
  variables that step alone needs on this platform: CUDA's
  `FLAGOS_USE_FLAGTREE=1` for `torch_compile`, Ascend's process-start
  `LD_PRELOAD` for `profiler_contract`. Keys are step ids rather than positions,
  so a step that moves in the manifest cannot silently carry another step's
  environment, and the composer refuses a key no manifest entry defines.

The model groups are uniform too: every config mounts the host checkpoint
`/data/models/Qwen/Qwen3-0.6B` at `/models/Qwen3-0.6B:ro`, sets
`MODEL_PATH=/models/Qwen3-0.6B` in `integration_environment`, and asserts the
mount through `environment_check.model_dir`, so `inference` and `training` are
plain manifest entries rather than a per-platform set.

The skips that remain are therefore pytest-level, and they are listed in
`tests/integration/platform_gates.py`: one entry per step whose target still
skips itself on some platform, with the measured reason and what would remove
it. The loader prints the entries that apply to a platform on every run, ahead
of the hardware job, and `tests/unit/test_platform_gates.py` holds the table to
the manifest, to the eight platforms and to the whitelist below.

| platform | steps whose target still skips |
|---|---|
| ascend | rng, musa_dispatch, profiler_parity, soft_lowp_matrix, native_comm_bridge, flaggems_conformance |
| cuda | rng, musa_dispatch, soft_lowp_matrix, native_comm_bridge, sdpa_lanes, flaggems_conformance |
| dcu | rng, musa_dispatch, sdpa_lanes |
| gcu | rng, musa_dispatch, profiler_parity, soft_lowp_matrix, native_comm_bridge, flaggems_conformance |
| kunlunxin | rng, vendor_ops, musa_dispatch, profiler_parity, soft_lowp_matrix, native_comm_bridge, sdpa_lanes, flaggems_conformance |
| metax | rng, musa_dispatch, profiler_parity, native_comm_bridge, sdpa_lanes |
| musa | profiler_parity, torch_compile, soft_lowp_matrix, native_comm_bridge, sdpa_lanes, flaggems_conformance |
| ppu | rng, musa_dispatch, profiler_parity, soft_lowp_matrix, native_comm_bridge, sdpa_lanes, flaggems_conformance |

The recorded skips, with the measured reason each is still there:

- `rng` — `tests/integration/ops/test_rng_dispatch.py`, the
  `TestRngSeedSource`, `TestRngMultiDevice`, `TestRngDistribution` and
  `TestRngDropout` classes, on every platform but MUSA. MUSA's vendor RNG
  library `murand` is broken on this runner: `murandGenerateUniform/Normal`
  fail with `LAUNCH_FAILURE` and `randperm` returns garbage (err 700), so the
  four classes that draw through it are gated inside the file. The
  reproducible-draw class still runs there. Removing the gate is what fixes it
  — `docs/vendors/musa/installation.md` records the same suite passing on an
  S5000 on 2026-08-17, so this is a driver/toolkit/image regression rather than
  a `torch_fl` defect.
- `vendor_ops` — `tests/integration/ops/test_flaggems_device_name.py` on
  Kunlunxin. Its flagtree XPU stack cannot import `flag_gems` at all: the XPU
  TLE backend raises `NotImplementedError` for `tle.device_mesh`, which
  `pytest.importorskip` does not catch, so the file skips itself. A flagtree
  XPU build that imports `flag_gems` removes the gate.
- `musa_dispatch` — `tests/integration/ops/test_musa_dispatch.py` on every
  platform but MUSA. Its cases carry `@pytest.mark.musa`, which the ops conftest
  skips elsewhere: MUSA compiles no CUDA boxing kernels, so the per-op
  `-> cuda` assertions cannot hold on the other platforms either. The step is
  kept universal so the MUSA dispatch coverage is one shared entry rather than
  a MUSA-only variant.
- `profiler_parity` — `tests/integration/test_profiler_parity.py` on Ascend,
  GCU, Kunlunxin, MetaX, MUSA and PPU. The comparison is against
  `tests/data/profiler_cuda_baseline.json`, so it is only meaningful where the
  tracer emits stock-CUDA-shaped records: MUSA's MUPTI reports no `gpu_memset`
  category and no occupancy-derived kernel args (2026-08-29: 5 passed, 2
  failed), and PPU's vendor kernels produce no `gpu_memset` record and no
  `::`-qualified kernel name (3 passed, 1 xpassed, 3 failed). Ascend, GCU and
  Kunlunxin gate it through the capability table in
  `tests/integration/profiler_support.py`. The structural contract those
  platforms do owe is the `profiler_contract` step.
- `torch_compile` — `tests/integration/test_compile.py` on MUSA.
  `torch.compile` needs a vendor Triton stack and this MUSA image ships none:
  `set_env_musa.sh` deliberately does not link
  `flagtree-0.5.0+mthreads3.1`, and stock PyPI `triton` is not a substitute —
  it carries no mthreads backend and cannot compile a MUSA kernel. The file
  gates itself on the active Triton lacking that backend rather than on the
  platform name, so provisioning the vendor flagtree removes the gate. The
  suite is green on an MTT S5000 with that runtime on `PYTHONPATH`
  (2026-08-29: 25 passed, 19 skipped).
- `soft_lowp_matrix` — `tests/integration/ops/test_soft_lowp_gate_dispatch.py`
  on Ascend, CUDA, GCU, Kunlunxin, MUSA and PPU. The software low-precision
  matmul path is compiled for DCU and MetaX only, and the ops conftest skips the
  `soft_lowp` marker elsewhere, so the command is identical on every platform
  and the selection is the capability. Removing the gate means compiling
  `csrc/aten/backends/soft_lowp` for the other six platforms.
- `native_comm_bridge` — `tests/integration/test_comm_native_bridge.py` on
  every platform but DCU. The native RCCL bridge is a DCU artifact: the file
  carries `pytestmark = pytest.mark.dcu` and skips itself everywhere else,
  because the groups it asserts are the ones the DCU image stages. The step is
  kept universal so a platform that grows an equivalent bridge has one shared
  entry to extend rather than a private one.
- `sdpa_lanes` — `tests/integration/vendor_lanes/test_sdpa_lanes.py` on CUDA,
  DCU, Kunlunxin, MetaX, MUSA and PPU. Only two confs reach
  `scaled_dot_product_attention` through a vendor lane of their own:
  `backends_ascend.conf` runs the ACLNN entry point and `backends_gcu.conf` the
  fused `_scaled_dot_product_efficient_attention` leaf. The suite reads the row
  for the conf the wheel resolved, so a platform without one skips rather than
  running a subset. DCU and MetaX route the operator to a FlagGems backend
  instead, which the `flaggems_conformance` step asserts; CUDA, Kunlunxin, MUSA
  and PPU keep the boxed composite, so there is no vendor lane to declare. The
  case set is one list in
  `tests/integration/vendor_lanes/sdpa_lane_caps.py` that every platform
  carrying a lane runs whole, and a chip that grows one adds a row there, which
  is what removes the gate.
- `flaggems_conformance` — its `sdpa` cases
  (`tests/integration/flaggems/test_flaggems_sdpa_route.py`) on Ascend, CUDA,
  GCU, Kunlunxin, MUSA and PPU. Only a conf that names
  `scaled_dot_product_attention` has a FlagGems SDPA route to assert:
  `backends_dcu.conf` and `backends_metax.conf` route it to `flaggems`, CUDA and
  PPU route it to `cuda`, and the other four leave it at `none`, so the wrapper
  in `csrc/aten/sdp_choice_stub.cc` keeps the boxed composite and the suite
  skips with the conf's own route as the reason. The case set is one list in
  `tests/integration/flaggems/sdpa_caps.py` that every platform carrying the
  route runs whole, and the per-chip numbers are a row in the same table, so a
  platform whose conf grows the route picks up every case by adding that one
  row.

## Directory inventory

Authoritative membership of the three per-platform trees. Machine-checked
against `os.listdir` by `tests/unit/test_platform_capability_matrix.py`.

- `csrc/runtime/accelerator/`: ascend, bpu, cuda, dcu, gcu, kunlunxin, metax, mps, musa, tsingmicro
- `csrc/aten/backends/`: ascend, flagos, gcu, iluvatar, kunlunxin, mps, musa, soft_lowp
- `torch_fl/accelerator/`: ascend, bpu, cuda, dcu, gcu, metax, mps, musa, ppu
