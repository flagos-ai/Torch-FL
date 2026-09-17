# Environment Variables

torch_fl's environment surface has one namespace of its own — `FLAGOS_*` — and
reads a second set belonging to other projects. This document covers both, and
is the reference the code is checked against: every owned name in the tables
below is declared in `torch_fl/_env.py`'s `VARIABLES` registry, and
`tests/unit/test_env_registry.py` fails if the two ever disagree.

Names owned by someone else (torch, FlagGems, FlagCX, tilelang, the vendor SDKs)
are listed in [Interoperability variables](#interoperability-variables). They are
never renamed here — renaming one would be renaming it in the other project.

Nothing in this document is required to run a wheel. A wheel routes, compiles and
runs with an empty environment; the variables here select a different build, an
override for measurement, or a diagnostic.

## How a value is read

Every owned variable is read through `torch_fl/_env.py`, so the rules are stated
once instead of being re-invented per call site.

**Booleans.** `1`, `true`, `on` and `yes` are **on**; `0`, `false`, `off` and `no`
are **off**; anything else is not a boolean at all — torch_fl prints one
`[flagos]` line to stderr and uses the variable's default rather than treating
the value as truthy. Matching is case-insensitive.

| Value | Meaning |
|-------|---------|
| `1`, `true`, `on`, `yes` (any case) | on |
| `0`, `false`, `off`, `no` (any case) | off |
| unset, or empty (see below) | the default in the table below |
| anything else (`FLAGOS_ALIAS_CUDA=2`) | warns once, then the default |

**Empty means unset.** `FLAGOS_LOG=${EXTRA_LOG}` with `EXTRA_LOG` unset is the
same as not exporting `FLAGOS_LOG` at all, so shell idioms do not accidentally
override a default. An empty value is the default, not "off": a switch that
defaults on, such as `FLAGOS_ALIAS_CUDA`, stays on when set to `""`.

**Enums.** A switch naming a mode rather than a boolean (`FLAGOS_FORCE_BACKEND`)
reports the alternatives and uses the default when given a value outside them.

**Unknown names.** `import torch_fl` scans `os.environ` once and warns about any
`FLAGOS_*` name that is neither declared below nor part of the dynamic
`FLAGOS_OP_<op>` family — a misspelled `FLAGOS_LOG_DISPACH` would otherwise be
read by nobody and simply do nothing. Retired names are exempt: they are inert by
design, so a stale export stays silent.

## Owned variables

### Build selection

Which kernel sets are compiled into the wheel. They are inputs to `setup.py` and
the CMake build only; nothing at run time reads them. Which chip they apply to is
`FLAGOS_ACCELERATOR`'s job alone — there are no per-chip switches.

The wheel records what it was built with in `torch_fl/_build_config.py`
(`ACCELERATOR` and `KERNELS`), and that record is what every run-time reader
consults, so a stale export cannot make a wheel describe itself wrongly.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_ACCELERATOR` | Build | `cuda` | Hardware platform the wheel is built for: `cuda`, `ppu`, `metax`, `ascend`, `tsingmicro`, `dcu`, `gcu`, `musa`, or `bpu`. Read by `setup.py` alone — the wheel records it in `_build_config.py`, and that record, not a re-export, is what every run-time reader consults |
| `FLAGOS_BUILD_VENDOR` | Build | `ON`, `OFF` on `metax` | Compile the accelerator vendor's native kernels (a no-op where the vendor ships none: `cuda`, `dcu`, `ppu`, `tsingmicro`, `bpu`). MetaX defaults `OFF` because its native path is retired — the generated CUDA boxing kernels are what accelerate that platform. Pinned `ON` for `ascend` and `musa` |
| `FLAGOS_BUILD_FLAGGEMS` | Build | `ON`, `OFF` on `bpu` | Compile the FlagGems Python kernel wrappers (calls into Python, no C++ linking). Set `OFF` for a slim pure-boxing build |
| `FLAGOS_BUILD_BOXING` | Build | `ON`, `OFF` on `ascend`, `gcu` and `musa` | Compile the generated CUDA boxing kernels. `OFF` is a default for `ascend`, `gcu` and `musa`, which have no CUDA runtime to box onto |
| `FLAGOS_BUILD_FLAGGEMS_CPP` | Build | `ON` on `cuda` and `tsingmicro`, off elsewhere | Compile the FlagGems C++ wrapper, which links `liboperators.so`. Defaults `OFF` outside `cuda`/`tsingmicro` because that library has to be built for the vendor's own toolkit and pointed at with `FLAGGEMS_DIR`; a build with one may turn this `ON` explicitly (MetaX's MACA build is the case). Pinned `OFF` for `dcu`, `musa` and `bpu`, where no such library exists at all |
| `FLAGOS_BUILD_TILEOPS` | Build | `ON` on `cuda`, `OFF` elsewhere | Compile the TileOps kernel wrappers, which are TileLang on SM90 NVIDIA parts only |
| `FLAGOS_BUILD_JOBS` | Build | System CPU count | Parallel jobs for the CMake build. `MAX_JOBS` and `CMAKE_BUILD_PARALLEL_LEVEL` are honoured as lower-priority fallbacks |
| `FLAGOS_WHEEL_LOCAL` | Build | SDK-derived | Local version label for the wheel (e.g. `metax3.8.1`), for dev builds that must pin the exact SDK |
| `FLAGOS_SKIP_CUDA_ASSETS` | Build | `0` (off) | Do not bundle an external `libtorch_cuda.so` into the wheel, for in-tree builds. The build-time counterpart of `FLAGOS_DISABLE_CUDA_ASSETS` |
| `FLAGOS_CUDA_ASSETS_DIR` | Build | `.libtorch_cuda_assets` | Directory the external `libtorch_cuda.so` is copied from when bundling. A missing directory downgrades to a warning: the wheel then needs a runtime-supplied `libtorch_cuda.so` |
| `FLAGOS_PPU_MKL_DIR` | Build | `/usr/local/lib` | Directory the PPU libtorch bundling script takes MKL from |
| `FLAGOS_DCU_VENDOR_CORE` | Build & Runtime | `0` (off) | Use DTK's forked core libraries instead of the official PyTorch core. Must match at build and import time. See [DCU without DTK's core libraries](../vendors/dcu/vendor-free-core-libs.md) |

`setup.py` forces a per-accelerator value for the six `FLAGOS_BUILD_*` switches
in the rows above. An explicit environment value that contradicts a forced one is
rejected with an error naming both, rather than letting whichever `-D` CMake saw
last win.

### Operator routing

Which backend implementation (CUDA boxing, vendor native, FlagGems C++, FlagGems
Python, TileOps) each operator dispatches to. Routing is stated per op in a
single generated `backends_<platform>.conf`, so a wheel's default routing follows
from what was compiled in; the variables below override or widen that table.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_BACKEND_CONFIG` | Runtime | No default | Absolute path to a `backends_*.conf` file; overrides the conf torch_fl selects from the build record. For testing and debugging only — the wheel's own selection is not written here, and is reported by `torch_fl.backend_config_path()` |
| `FLAGOS_OP_<name>` | Runtime | No default | Per-operator backend override (e.g. `FLAGOS_OP_add__Tensor=cuda`); replace `.` with `__` in op names |
| `FLAGOS_FORCE_BACKEND` | Runtime | No default (off) | Repin every op onto one backend family for A/B measurement: `flaggems`, `vendor`, or `tileops`. An op the target does not implement is reported on stderr and left on its configured backend; an op it does implement but this wheel did not compile raises rather than falling back. The `tileops` mode repins the ops the conf annotates `# tileops` and additionally needs the `tileops` package, an SM90 device and a `FLAGOS_BUILD_TILEOPS=ON` build |
| `FLAGOS_DISABLE_FLAGGEMS_PY` | Runtime | `0` (off) | Leave the FlagGems Python layer unregistered (C++ stub-only mode) |

`FLAGOS_FORCE_BACKEND` is a single enum rather than the three switches it
replaced (`ALL_USE_FLAGGEMS`, `ALL_USE_VENDOR`, `FLAGOS_USE_TILEOPS`), so "two at
once" is unrepresentable instead of having to be detected and rejected at run
time. An unset value means "leave the conf's routing alone" — the default.

`FLAGOS_BACKEND_CONFIG` is the one documented escape hatch for testing. It holds
only what you export: the wheel never writes its own choice there, so reading the
variable answers "did the user override the conf?", and
`torch_fl.backend_config_path()` answers "which conf is in use?" in the order the
routing table reads them (override, then the build record's default).

### Runtime diagnostics

Logging and tracing. None of these changes routing.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_LOG` | Runtime | No default (all off) | Comma-separated stderr diagnostics: `dispatch` (backend chosen per operator), `fallback` (each `cpu_fallback` dispatch), `op_cache` (Ascend operator-cache hit/miss statistics). An entry naming none of the three is reported once per process rather than silently ignored |
| `FLAGOS_TRACE` | Runtime | `0` (off) | Verbose logging in the device profiler shim compiled into this build. One switch covers every accelerator: exactly one device tracer is compiled per build, so the name is never ambiguous |
| `FLAGOS_TRACER_LIBRARY` | Runtime | Auto-discovered | Override the tracer library the profiler shim `dlopen`s, when the default path does not match the installed driver |

### Vendor compatibility

Import-time shims that adapt a vendor's torch or driver to the flagos device.
None is needed on a stock CUDA box; each platform guide states which apply.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_ALIAS_CUDA` | Runtime | `1` (on) | Alias the `cuda` device string to `flagos` for drop-in compatibility. Set `0` to opt out |
| `FLAGOS_DISABLE_CUDA_SHIM` | Runtime | `0` (off) | Skip registering the `torch.cuda` compatibility shim for generic GPU operations |
| `FLAGOS_METAX_CUDART_SHIM` | Runtime | `0` (off) | Preload the libcudart version-tag shim before `import torch`. Required for MetaX with generic PyTorch wheels |
| `FLAGOS_METAX_COMPAT` | Runtime | `0` (off) | Patch FlagGems `torch.cuda` device queries for MetaX compatibility |
| `FLAGOS_DCU_HIP_VERSION` | Runtime | No default | Override HIP version detection for the DCU runtime |
| `FLAGOS_DCU_SKIP_RUNTIME_CHECK` | Runtime | `0` (off) | Skip the DCU post-import checks (torch/DTK version alignment and CUDA-key kernel presence), for deliberately testing a non-matching wheel pair |
| `FLAGOS_DCU_SDPA_FLASH` | Runtime | `0` (off) | Keep the fused SDPA backends enabled on DCU. The default disables flash/mem-efficient SDPA so `scaled_dot_product_attention`'s own choice agrees with the only kernel this stack can execute |
| `FLAGOS_DISABLE_APEX_COMPAT` | Runtime | `0` (off) | Disable the optional Apex multi-tensor compatibility layer; see the Apex note below |
| `FLAGOS_DIST_FORCE_NCCL` | Test | `0` (off) | In the manual MetaX distributed tests, skip FlagCX and use NCCL |
| `FLAGOS_DCU_SKIP_LEGACY_SMOKE` | Test | `0` (off) | In `.github/scripts/set_env_dcu.sh`, skip the legacy-mode smoke path (`FLAGOS_DCU_VENDOR_CORE=1`) after the decoupled gates have run |

**Apex compatibility.** On CUDA-ABI boxing vendors, torch_fl patches Apex's
common `MultiTensorApply` entry point when Apex is imported. The patch converts
flagos tensors to zero-copy CUDA views for direct `amp_C` calls and converts CUDA
results back to flagos views. It is optional and does not apply to native
non-CUDA backends. Set `FLAGOS_DISABLE_APEX_COMPAT=1` to disable it.

### Assets and libraries

How the external libtorch/CUDA runtime is found at build time and loaded at
import time.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_DISABLE_CUDA_ASSETS` | Runtime | `0` (off) | Skip preloading the bundled `libtorch_cuda.so` and CUDA libraries, for builds that use system libtorch |
| `FLAGOS_VENDOR_TORCH_LIB` | Build & Runtime | Auto-discovered | Path to the vendor torch's `lib` directory, used when no bundled `lib_maca`/`lib_dcu`/`lib_ppu` is present. Only the active accelerator's build reads it |
| `FLAGOS_USE_CACHING_ALLOCATOR` | Runtime | `1` (on) | Caching device allocator. Set `0` to hand every allocation straight to the vendor runtime |

### Compiler and feature backends

`torch.compile` integration and the specialized compilation paths.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_USE_FLAGTREE` | Runtime | `0` (off) | Assert that a FlagTree build is the active Triton. Required on Ascend when the compiler is FlagTree: the check fails loudly if the installed Triton is not FlagTree |
| `FLAGOS_COMPILE_FALLBACK_EAGER` | Runtime | `0` (off) | Fall back to eager mode when `torch.compile` encounters unsupported operations |
| `FLAGOS_TILEOPS_USE_L2` | Runtime | `0` (off) | Use the TileOps L2-cache tier |
| `FLAGOS_TILEOPS_CACHE_MAX` | Runtime | `512` | TileOps instance-cache capacity. Past the cap, results are rebuilt per call: slower but bounded |
| `FLAGOS_TILEOPS_DISABLE_ALL_CACHE` | Runtime | `0` (off) | Neutralize every TileLang cache. Correct but slow; must be set before `tileops` is imported. Sets `TILELANG_DISABLE_CACHE=1` |
| `FLAGOS_TILEOPS_FULL` | Test | `0` (off) | In the TileOps codegen tests, run the full manifest workload shape instead of the small one |

### Code generation

Inputs to `scripts/codegen/`. Never read by a built wheel.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_EXEC_CACHE` | Build (codegen) | `1` (on) | Cache Ascend operator-codegen execution results; `0` forces regeneration |
| `FLAGOS_CODEGEN_ALL` | Build (codegen) | `0` (off) | Generate routes for the full leaf-CUDA operator set rather than the supported subset |

### BPU compiler

The BPU path compiles through hbdk4 on an x86 host, so its variables describe
that host and its cache. See the
[BPU integration guide](../vendors/bpu/integration.md).

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_BPU_MARCH` | Runtime | `nash-p` | BPU micro-architecture. `nash-p` is the BPU, `nash-e` the S100 and `nash-m` the S100P |
| `FLAGOS_BPU_CACHE` | Runtime | `~/.cache/torch_fl_bpu` | Directory holding the BPU compiler cache |
| `FLAGOS_BPU_QUANTIZE` | Runtime | `1` (on) | Quantize BPU kernels. Without it hbdk4 keeps conv in float and lowers it to the CPU, so the BPU never runs the heavy work |
| `FLAGOS_BPU_ACT_SCALE` | Runtime | `0.05` | Fallback activation scale for tensors with no calibration entry |
| `FLAGOS_BPU_MLIR_LIBS` | Runtime | Unset | Directory of the BPU MLIR plugin libraries (`libhbtl.so`), preloaded by the x86 compile driver |
| `FLAGOS_BPU_X86_PYTHON` | Runtime | Unset | An x86_64 CPython with hbdk4 installed, run under an emulator: hbdk4 ships x86_64-only wheels |
| `FLAGOS_BPU_X86_EMULATOR` | Runtime | Unset | BPU x86_64 emulator binary. Useful because the distro `box64` is usually too old for hbdk4, or the user has one that is not in `PATH` |
| `FLAGOS_BPU_X86_STUBS` | Runtime | `<x86 python prefix>/../stubs` | Directory of import-only stand-ins for numba and torch, which hbdk4's ONNX entry point imports unconditionally |

## Interoperability variables

Names torch_fl reads or writes but does not own. Each is a contract with another
package; renaming one here would break it there, so they keep the vendor's or the
project's own spelling.

| Variable | Owner | Direction | Purpose |
|----------|-------|-----------|---------|
| `GEMS_VENDOR` | FlagGems | Set if unset | FlagGems' own vendor selector (`cuda`, `metax`, `ascend`, `musa`, …). torch_fl fills it from the detected hardware or the build record so FlagGems does not have to guess |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | PyTorch | Set if unset | torch's device-backend entry-point autoload. torch_fl sets it to `0` on MUSA builds so vendor plugins (e.g. `torch_musa`) do not claim `PrivateUse1` during `import torch` |
| `FLAGCX_TORCH_BACKEND` | FlagCX | Set if unset | FlagCX's torch plugin selector; torch_fl sets `flagos` |
| `TILELANG_DISABLE_CACHE` | tilelang | Set if unset | tilelang's kernel cache. `FLAGOS_TILEOPS_DISABLE_ALL_CACHE=1` sets it to `1` |
| `TRITON_ENABLE_TASKQUEUE` | FlagTree / torch_npu | Set if unset | The FlagTree Ascend Triton launch queue, on by default upstream. torch_fl turns it off so an unsupported async launch fails with a clear message instead of a silent override |
| `COMPILE_ARCH` | Enflame tops | Set if unset | The Enflame compiler's target architecture, derived from the installed GCU |
| `HB_DNN_USER_DEFINED_L2M_SIZES` | Horizon hbdk | Set if unset | The BPU runtime's L2 memspace sizing, set before first inference |
| `FLAGGEMS_DIR` | FlagGems | Read | FlagGems CMake config directory (`FlagGemsConfig.cmake`); auto-detected from the installed `flag_gems` when unset |
| `FLAGGEMS_SOURCE_DIR` | FlagGems | Read | Absolute path to the FlagGems source directory (Python Triton kernels); required when the FlagGems C++ runtime is active, and must match the revision `liboperators.so` was built against |
| `TORCHINDUCTOR_COMPILE_THREADS` | PyTorch | Read | torch's own compile-thread count; torch_fl honors it and uses it as the compile pool size |
| `CUDA_HOME` | NVIDIA / conda | Read | CUDA toolkit root for `FLAGOS_ACCELERATOR=cuda` and `ppu`. Defaults to system CUDA, else `$CONDA_PREFIX/targets/x86_64-linux` |
| `CONDA_PREFIX` | conda | Read | Conda environment prefix, the CUDA discovery fallback |
| `ASCEND_HOME` | Huawei CANN | Read | CANN toolkit path for Ascend NPU builds (`/usr/local/Ascend/ascend-toolkit/latest`) |
| `MUSA_HOME` | Moore Threads | Read | MUSA toolkit path (`/usr/local/musa`) |
| `MACA_PATH` (`MACA_HOME` fallback) | MetaX | Read | MetaX SDK path (`/opt/maca`) |
| `TOPS_HOME` | Enflame | Read | TopsRider SDK path for GCU builds (`/opt/tops`) |
| `ROCM_PATH` | Hygon DTK | Read | DTK path for DCU builds (`/opt/dtk`) |
| `PPU_SDK` | PPU | Read | PPU SDK path (`/usr/local/PPU_SDK`); its CUDA toolkit is `$PPU_SDK/CUDA_SDK` |
| `TOPSATEN_LIB` | Enflame | Read | Enflame topsaten library override; otherwise discovered under `$TOPS_HOME` |
| `MUDNN_LIB` | Moore Threads | Read | MUSA kernel-library override; otherwise discovered under `$MUSA_HOME/lib` |
| `TRITON_GCU_PATH` | Enflame Triton | Read | Vendor Triton/compiler root for GCU (`/opt/triton_gcu`) |

Each SDK name is the vendor's own — the one the vendor's `set_env` script writes.
There are no `FLAGOS_`/`METAX_`-style aliases; one name per vendor. Only the
active `FLAGOS_ACCELERATOR`'s entries apply, and CMake falls back to a built-in
default when the environment sets none.

The "set if unset" rows are filled by `_env.set_foreign()` and never override an
explicit export — including an empty one, which is how a user says "not this
vendor". torch_fl writes to the environment rather than passing a value down
because the consumer is another library that reads `os.environ` itself, and two
of these have to be in place before that library is imported.

## Worker count

`MAX_JOBS` and `CMAKE_BUILD_PARALLEL_LEVEL` are neither owned nor read by
torch_fl at run time; they are the conventional CMake/PyTorch build-parallelism
variables, honored only as the fallbacks behind `FLAGOS_BUILD_JOBS`.

## Retired names

These were read by earlier revisions and are not read by any code path now. An
exported value is inert — there is no alias, no deprecation window, and no
warning, since the name is listed as retired rather than unknown.

| Retired | Replaced by |
|---------|-------------|
| `ACCELERATOR` | `FLAGOS_ACCELERATOR`, and `_build_config.py` at run time |
| `VENDOR_KERNEL`, `FLAGGEMS_KERNEL`, `BOXING_KERNEL`, `FLAGGEMS_CPP`, `TILEOPS_KERNEL` | the `FLAGOS_BUILD_*` switches |
| `ALL_USE_FLAGGEMS`, `ALL_USE_VENDOR`, `FLAGOS_USE_TILEOPS` | `FLAGOS_FORCE_BACKEND` |
| `FLAGOS_LOG_DISPATCH`, `FLAGOS_LOG_FALLBACK`, `FLAGOS_CACHE_STATS` | `FLAGOS_LOG` |
| `FLAGOS_CUPTI_SHIM_DEBUG`, `FLAGOS_MUPTI_DEBUG`, `FLAGOS_MSPTI_DEBUG`, `FLAGOS_TOPSPTI_DEBUG`, `FLAGOS_ROCTRACER_DEBUG`, `FLAGOS_KINETO_SHIM_DEBUG` | `FLAGOS_TRACE` |
| `FLAGOS_CUPTI_LIBRARY`, `FLAGOS_MUPTI_LIBRARY`, `FLAGOS_TOPSPTI_LIBRARY` | `FLAGOS_TRACER_LIBRARY` |
| `FLAGOS_MACA_TORCH_LIB`, `FLAGOS_DCU_TORCH_LIB`, `FLAGOS_PPU_TORCH_LIB` | `FLAGOS_VENDOR_TORCH_LIB` |
| `FLAGOS_USE_FLAGGEMS`, `FLAGOS_USE_FLAGGEMS_CPP` | nothing — a conf is no longer selected by a variable, and whether the FlagGems C++ runtime exists is a property of the build record |
| `FLAGOS_USE_VENDOR_OPS` | nothing |

Dated measurement logs elsewhere in this repository quote these names because
they record the command that was actually run at the time; they are history, not
instructions.

## Platform-specific variables

Detailed setup and runtime variables for each accelerator backend are documented
in platform guides:

- [CUDA (NVIDIA)](../vendors/cuda/installation.md)
- [PPU (T-Head)](../vendors/ppu/installation.md)
- [MetaX](../vendors/metax/installation.md)
- [Ascend (Huawei)](../vendors/ascend/installation.md)
- [DCU (Hygon)](../vendors/dcu/installation.md)
- [GCU (Enflame)](../vendors/gcu/installation.md)
- [MUSA (Moore Threads)](../vendors/musa/installation.md)
- [BPU (Horizon Robotics)](../vendors/bpu/integration.md)

TsingMicro is supported as a build target (`FLAGOS_ACCELERATOR=tsingmicro`,
CUDA-compatible) but has no vendor guide yet; its SDK paths follow the CUDA
toolchain and are selected the same way the CUDA row above describes.

Platform guides document SDK paths, driver requirements, version compatibility,
and any additional environment setup (e.g. `LD_PRELOAD`, `LD_LIBRARY_PATH`).
