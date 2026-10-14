# Platform capability matrix

The eleven `FLAGOS_ACCELERATOR` values, what each is, and what it actually
supports. The tree layout made this "obvious" to maintainers but hid the
asymmetries — PPU is Python-only, TsingMicro has no Python compat, `soft_lowp`
is aten-only, and CI covers eight of the eleven — so this page is the
authoritative answer.

The directory inventory at the end is machine-checked by
`tests/unit/test_platform_capability_matrix.py`; update it with the directories
when a platform gains or loses one.

## Matrix

| `FLAGOS_ACCELERATOR` | Kind | runtime dir | aten backend | Python compat | conf routes | CI |
|---|---|---|---|---|---|---|
| `cuda` | native CUDA | `cuda` | — (C++ / FlagGems / CUDA) | `cuda` | 2034 | yes |
| `ppu` | CUDA-ABI boxing (bundled PPU libtorch) | `cuda` (+ `lib_ppu/` bundle) | — (boxing) | `ppu` | 2037 | yes |
| `metax` | CUDA-ABI boxing (bundled MACA libtorch) | `metax` | — (native retired) | `metax` | 2037 | yes |
| `dcu` | CUDA-ABI boxing (hipified DTK torch) | `cuda` (+ `dcu` DTK-core shim) | — (boxing) | `dcu` | 2037 | yes |
| `kunlunxin` | vendor-native (xdnn op library) | `kunlunxin` | `kunlunxin` | — (no dir) | 2037 | yes |
| `ascend` | vendor-native (ACLNN) | `ascend` | `ascend` | `ascend` | 2037 | yes |
| `gcu` | vendor-native (topsaten) | `gcu` | `gcu` | `gcu` | 2037 | yes |
| `musa` | vendor-native (mudnn) | `musa` | `musa` | `musa` (stream/event only) | 2037 | yes |
| `tsingmicro` | CUDA-ABI boxing (Kuiper) | `tsingmicro` | — | — (no dir) | 53 | no |
| `bpu` | no per-op kernels (graph compile) | `bpu` | — | `bpu` | 0 | no |
| `iluvatar` | FlagGems-primary (no vendor kernels, no boxing) | `cuda` | `iluvatar` (generated FlagGems register only) | — (no dir) | 2037 | no |

Column notes:

- **Kind** is the operator path the build selects (see
  `cmake/flagos_platforms.json` for the per-platform kernel-switch defaults).
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
  `iluvatar` has a directory but no vendor `.cc` kernel source: it holds only
  `generated/iluvatar_flaggems_register.inc`, the generated FlagGems registration
  for the Python/Triton routes.
- **conf routes** is the number of `op = backend` entries in
  `torch_fl/configs/backends_<platform>.conf`. 2037 is the full-coverage set; 0
  and 53 are the by-design BPU and hand-written TsingMicro files.
- **CI** is whether `.github/configs/<platform>.yml` and a platform pipeline job
  exist.

## Shared kernel sets

`FLAGOS_BUILD_*` kernel switches (see `cmake/flagos_platforms.json`) decide which
sets a wheel compiles: `vendor` (the platform's native tree), `flaggems`
(FlagGems Python path), `flaggems_cpp` (FlagGems C++ path), `boxing` (generated
CUDA boxing kernels) and `tileops`. Which are on by default is a property of the
platform, not a per-build choice.

## Known gaps

- **BPU** has no aten backend and an empty conf: its unit of execution is a whole
  compiled graph, so eager ops reach `cpu_fallback` and acceleration comes from
  the `torch.compile(backend="bpu")` path.
- **TsingMicro** has a sparse hand-written conf (53 routes) and no CI job; its
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
- **Iluvatar** (BI-V150 / Tianshu, corex 4.5.0) is FlagGems-primary: it has no
  vendor kernel library and no CUDA boxing path (boxing was explicitly ruled
  out), so all covered ops route to the FlagGems Python/Triton path through
  FlagTree's iluvatar backend and everything else falls back to `none`
  (`cpu_fallback`). Its 639 FlagGems routes are the *shared* FlagGems Python
  ceiling, not a measurement taken on BI-V150 hardware: no FlagGems overload
  survey has been run on a BI-V150 host yet, so the coverage is **provisional**
  until `tests/manual/flaggems_overload_survey.py` has been run there. It has no
  CI job and no runner lane — it builds locally only — and its wheel carries the
  local version segment `corex4.5.0`. It is also the first entry in
  `FLAGGEMS_PYTHON_PLATFORMS` with no native `.inc` in its
  `csrc/aten/backends/iluvatar/` tree.
- **CI covers 8 of 11**: TsingMicro, BPU and Iluvatar have no
  `.github/configs/` entry.

## Directory inventory

Authoritative membership of the three per-platform trees. Machine-checked
against `os.listdir` by `tests/unit/test_platform_capability_matrix.py`.

- `csrc/runtime/accelerator/`: ascend, bpu, cuda, dcu, gcu, kunlunxin, metax, musa, tsingmicro
- `csrc/aten/backends/`: ascend, flagos, gcu, iluvatar, kunlunxin, musa, soft_lowp
- `torch_fl/accelerator/`: ascend, bpu, cuda, dcu, gcu, metax, musa, ppu
