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

- **CUDA** is where the gap between the conf and the default wheel is widest.
  `backends_cuda.conf` routes 392 ops to `flaggems` and 1642 to `cuda`, and the
  default wheel compiles the generated FlagGems register list only, so those
  1642 ops are unregistered on PrivateUse1 and reach `cpu_fallback`. Of them,
  258 are FlagGems-capable on at least one of `metax`/`dcu`/`ppu` (unchanged --
  the same set -- when `tsingmicro` and `iluvatar` are counted) and are the
  cheap re-wire candidates; the other 1384 have no FlagGems route on any
  platform and need a native CUDA backend
  (cuBLAS/cuDNN/ATen CUDA) or an opt-in boxing build. The CUDA lane builds and
  tests the default wheel, so a case asserting a `-> cuda` route skips there.
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

## Directory inventory

Authoritative membership of the three per-platform trees. Machine-checked
against `os.listdir` by `tests/unit/test_platform_capability_matrix.py`.

- `csrc/runtime/accelerator/`: ascend, bpu, cuda, dcu, gcu, kunlunxin, metax, mps, musa, tsingmicro
- `csrc/aten/backends/`: ascend, flagos, gcu, iluvatar, kunlunxin, mps, musa, soft_lowp
- `torch_fl/accelerator/`: ascend, bpu, cuda, dcu, gcu, metax, mps, musa, ppu
