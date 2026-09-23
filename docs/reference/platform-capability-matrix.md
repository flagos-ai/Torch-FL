# Platform capability matrix

The nine `FLAGOS_ACCELERATOR` values, what each is, and what it actually
supports. The tree layout made this "obvious" to maintainers but hid the
asymmetries — PPU is Python-only, TsingMicro has no Python compat, `soft_lowp`
is aten-only, and CI covers seven of the nine — so this page is the authoritative
answer.

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
| `ascend` | vendor-native (ACLNN) | `ascend` | `ascend` | `ascend` | 2037 | yes |
| `gcu` | vendor-native (topsaten) | `gcu` | `gcu` | `gcu` | 2037 | yes |
| `musa` | vendor-native (mudnn) | `musa` | `musa` | — (no dir) | 2037 | yes |
| `tsingmicro` | CUDA-ABI boxing (Kuiper) | `tsingmicro` | — | — (no dir) | 53 | no |
| `bpu` | no per-op kernels (graph compile) | `bpu` | — | `bpu` | 0 | no |

Column notes:

- **Kind** is the operator path the build selects (see
  `cmake/flagos_platforms.json` for the per-platform kernel-switch defaults).
- **runtime dir** is the `csrc/runtime/accelerator/<dir>` whose sources the
  device runtime is built from. `dcu` reuses `cuda` (DTK's CUDA compatibility
  toolkit) and adds `dcu/` for its DTK-core ABI shim; `ppu` reuses `cuda` too.
- **aten backend** is the `csrc/aten/backends/<dir>` vendor kernel tree, when one
  exists. `flagos/` (the Python op caller) and `soft_lowp/` (the software
  low-precision matmul path shared by DCU/MetaX) are shared, not per-platform.
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
- **MUSA** has no Python compat module and no compile-only CI job.
- **PPU** withholds the profiler (its tracer is the unavailable implementation);
  it also ships no `lib/flagos_platform` marker (it uses the `lib_ppu/` bundle).
- **CI covers 7 of 9**: TsingMicro and BPU have no `.github/configs/` entry.

## Directory inventory

Authoritative membership of the three per-platform trees. Machine-checked
against `os.listdir` by `tests/unit/test_platform_capability_matrix.py`.

- `csrc/runtime/accelerator/`: ascend, bpu, cuda, dcu, gcu, metax, musa, tsingmicro
- `csrc/aten/backends/`: ascend, flagos, gcu, musa, soft_lowp
- `torch_fl/accelerator/`: ascend, bpu, cuda, dcu, gcu, metax, ppu
