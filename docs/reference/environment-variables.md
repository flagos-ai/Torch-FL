# Environment Variables

This document lists configuration variables that control torch_fl's build, operator routing, and runtime behavior. Platform-specific setup variables are documented in vendor guides.

## Build Selection

These variables control which kernel sets are compiled into the wheel.
Which chip they apply to is `FLAGOS_ACCELERATOR`'s job alone -- there are no
per-chip switches. Defaults below are the CMake defaults; `setup.py` forces
per-accelerator values (see each branch) and any explicit environment value
wins over both via the generic pass-through.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_ACCELERATOR` | Build | `cuda` | Hardware platform: `cuda`, `ppu`, `metax`, `ascend`, `tsingmicro`, `dcu`, `gcu`, `musa`, or `bpu` |
| `FLAGOS_BUILD_VENDOR` | Build | `ON` | Build the `FLAGOS_ACCELERATOR` vendor's native kernels (no-op where the vendor ships none: `cuda`, `dcu`, `ppu`, `tsingmicro`, `bpu`; MetaX's native dir is retired and excluded). `setup.py` forces `OFF` for MetaX |
| `FLAGOS_BUILD_FLAGGEMS` | Build | `ON` | FlagGems integration: Python kernel wrappers (calls via Python, no C++ linking); set `OFF` for a slim pure-boxing build |
| `FLAGOS_BUILD_BOXING` | Build | `ON` | CUDA Boxing integration: generated boxing kernels for CUDA-ABI vendors (libtorch extracted from the vendor torch package); `setup.py` forces `OFF` for `ascend`/`gcu`/`musa`, which have no CUDA runtime |
| `FLAGOS_BUILD_FLAGGEMS_CPP` | Build | `ON` | Enable the FlagGems C++ wrapper (`cpp_wrapper`): links `liboperators.so`; `setup.py` forces `OFF` unless a vendor-built FlagGems is pointed at via `FLAGGEMS_DIR` |
| `FLAGOS_BUILD_TILEOPS` | Build | `ON` on CUDA, forced `OFF` elsewhere | TileOps kernel wrappers; `setup.py` forces `OFF` for non-CUDA builds |
| `FLAGOS_BUILD_JOBS` | Build | System CPU count | Parallel jobs for CMake build |

## SDK and Compiler Discovery

These variables locate platform SDKs and toolchains. Only the active
`FLAGOS_ACCELERATOR`'s entries apply; CMake falls back to a built-in default when the
environment sets none.

Each name is the vendor's own — the one the vendor's `set_env` script writes.
There are no `FLAGOS_`/`METAX_`-style aliases; one name per vendor.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `CUDA_HOME` | Build & runtime | Auto (system CUDA, else `$CONDA_PREFIX/targets/x86_64-linux`) | CUDA toolkit root for `FLAGOS_ACCELERATOR=cuda` and `ppu` |
| `ASCEND_HOME` | Build | `/usr/local/Ascend/ascend-toolkit/latest` | CANN toolkit path for Ascend NPU builds |
| `MUSA_HOME` | Build | `/usr/local/musa` | Moore Threads MUSA toolkit path |
| `TOPS_HOME` | Build | `/opt/tops` | Enflame TopsRider SDK path for GCU builds |
| `MACA_PATH` (`MACA_HOME` fallback) | Build | `/opt/maca` | MetaX SDK path |
| `ROCM_PATH` | Build | `/opt/dtk` | Hygon DTK path for DCU builds |
| `PPU_SDK` | Build | `/usr/local/PPU_SDK` | PPU SDK path; its CUDA toolkit is `$PPU_SDK/CUDA_SDK` |
| `CONDA_PREFIX` | Build & runtime | Auto-detected | Conda environment prefix (CUDA discovery fallback) |
| `TOPSATEN_LIB` | Build | Discovered under `$TOPS_HOME` | Enflame topsaten library override |
| `MUDNN_LIB` / `MURAND_LIB` | Build | Discovered under `$MUSA_HOME/lib` | MUSA kernel-library overrides |
| `TRITON_GCU_PATH` | Runtime | `/opt/triton_gcu` | Vendor Triton/compiler root for GCU |
| `FLAGGEMS_DIR` | Build | Auto-detected from the installed `flag_gems` | FlagGems CMake config directory (`FlagGemsConfig.cmake`) |

## Operator Routing

These variables control which backend implementation (CUDA boxing, vendor
native, FlagGems C++, FlagGems Python, TileOps) each operator dispatches to at
runtime. Routing is stated per op in a single `backends_<platform>.conf`, so a
wheel's default routing follows from what was compiled in; the variables below
override or widen that table.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_BACKEND_CONFIG` | Runtime | Derived from the build record (`_build_config.py` + `lib/flagos_platform`) | Absolute path to a `backends_*.conf` file; overrides auto-detection |
| `FLAGOS_OP_<name>` | Runtime | No default | Per-operator backend override (e.g., `FLAGOS_OP_add__Tensor=cuda`); replace `.` with `__` in op names |
| `FLAGOS_FORCE_BACKEND` | Runtime | No default (off) | Collapse the routing table onto one backend family for A/B measurement: `flaggems`, `vendor`, or `tileops`. An op only moves if that family actually implements it (known from the routed value plus its `# <backend>` annotation); the rest are reported on stderr and left on their configured backend, and for `flaggems`/`vendor` the dispatcher raises rather than silently falling back when the family was named yet nothing is compiled in. The `tileops` mode repins the ops the conf annotates `# tileops` and additionally needs the `tileops` package, an SM90 device, and a `FLAGOS_BUILD_TILEOPS=ON` build |
| `FLAGOS_DISABLE_FLAGGEMS_PY` | Runtime | `0` (off) | Leave the FlagGems Python layer unregistered (C++ stub-only mode) |
| `FLAGGEMS_SOURCE_DIR` | Runtime | Required when FlagGems is active | Absolute path to FlagGems source directory (Python Triton kernels); must match the version liboperators.so was built against |

`FLAGOS_FORCE_BACKEND` is a single enum rather than the three switches it
replaced (`ALL_USE_FLAGGEMS`, `ALL_USE_VENDOR`, `FLAGOS_USE_TILEOPS`), so
"two at once" is unrepresentable instead of having to be detected and rejected
at runtime. An unset value means "leave the conf's routing alone" — the default.

`FLAGOS_USE_FLAGGEMS` is retired. It named a conf when there were three; there is
now one conf per platform, so an exported value is a no-op.

The FlagGems C++ runtime (`kFlagOs`, no GIL) is chosen by the conf's
`flaggems_cpp` keys, not by a variable. `tests/integration/ops/conftest.py` uses
`FLAGOS_USE_FLAGGEMS_CPP=1` as a *test gate* for the `flaggems_cpp` mark; see
[Testing](../development/testing.md).

**Note on auto-detection**: `FLAGOS_BACKEND_CONFIG` is normally set by `torch_fl.__init__._select_backend_config()`, which reads the build record: the accelerator the wheel was built for (from `_build_config.py`) plus the `lib/flagos_platform` marker a native-kernel build writes. There is no mode variable — a wheel's routing follows from what was compiled in. Users should override `FLAGOS_BACKEND_CONFIG` only for testing or debugging.

## Runtime Diagnostics

Logging and tracing switches. All are off unless set, and none changes routing.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_LOG` | Runtime | unset (off) | Comma-separated stderr diagnostics, none of which changes routing: `dispatch` (the backend chosen for each operator), `fallback` (each `cpu_fallback` dispatch), `op_cache` (Ascend operator-cache hit/miss statistics). An entry that names none of the three is reported once per process rather than silently ignored |
| `FLAGOS_TRACE` | Runtime | `0` (off) | Verbose logging in the device profiler shim this wheel builds. One switch covers every accelerator: exactly one device tracer is compiled per build, so the name is never ambiguous |
| `FLAGOS_TRACER_LIBRARY` | Runtime | Auto-discovered | Override the tracer library the profiler shim `dlopen`s, when the default path does not match the installed driver |

## Vendor Compatibility

Import-time shims that adapt a vendor's torch or driver to the flagos device.
None is needed on a stock CUDA box; each platform guide states which apply.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_ALIAS_CUDA` | Runtime | `0` (off) | Alias `cuda` device string to `flagos` for drop-in compatibility |
| `FLAGOS_DISABLE_CUDA_SHIM` | Runtime | `0` (off) | Skip registering the `torch.cuda` compatibility shim for generic GPU operations |
| `FLAGOS_METAX_CUDART_SHIM` | Runtime | `0` (off) | Preload libcudart version-tag shim before `import torch` (MetaX-specific; required for generic PyTorch wheels) |
| `FLAGOS_METAX_COMPAT` | Runtime | `0` (off) | Patch FlagGems `torch.cuda` device queries for MetaX compatibility |
| `FLAGOS_DCU_HIP_VERSION` | Runtime | No default | Override HIP version detection for DCU runtime |
| `FLAGOS_DCU_VENDOR_CORE` | Build & Runtime | `0` (off) | Use DTK's forked core libraries instead of the official PyTorch core: bundles the full vendor core and symlinks it over the installed torch wheel. Must match at build and import time. See [DCU without DTK's core libraries](../vendors/dcu/vendor-free-core-libs.md) |
| `FLAGOS_DCU_SKIP_RUNTIME_CHECK` | Runtime | `0` (off) | Skip the DCU post-import checks (torch/DTK version alignment and CUDA-key kernel presence). For deliberately testing a non-matching wheel pair |
| `FLAGOS_DCU_SDPA_FLASH` | Runtime | `0` (off) | Keep the fused SDPA backends enabled on DCU. The default disables flash/mem-efficient SDPA so that `scaled_dot_product_attention`'s own choice agrees with the only kernel this stack can execute |
| `FLAGOS_DISABLE_APEX_COMPAT` | Runtime | `0` (off) | Disable the optional Apex multi-tensor compatibility layer; see the Apex note below |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Runtime | torch default | torch's own switch; torch_fl clears it on MUSA builds so vendor plugins (e.g., `torch_musa`) do not claim `PrivateUse1` during `import torch` |
| `GEMS_VENDOR` | Runtime | Auto-detected from hardware or build metadata | FlagGems' own vendor selector: `cuda` (default), `metax`, `ascend`, `musa`, `cambricon`; controls the distributed backend and device-specific Triton compilation |

**Apex compatibility**: On CUDA-ABI boxing vendors, Torch-FL automatically patches Apex's common `MultiTensorApply` entry point when Apex is imported. The patch converts flagos tensors to zero-copy CUDA views for direct `amp_C` calls and converts CUDA results back to flagos views. It is optional and does not apply to native non-CUDA backends. Set `FLAGOS_DISABLE_APEX_COMPAT=1` to disable it.

## Assets, Libraries, and Packaging

How the external libtorch/CUDA runtime is found at build time and loaded at
import time.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_DISABLE_CUDA_ASSETS` | Runtime | `0` (off) | Skip preloading the bundled `libtorch_cuda.so` and CUDA libraries (for builds that use system libtorch) |
| `FLAGOS_SKIP_CUDA_ASSETS` | Build | `0` (off) | Do not bundle an external `libtorch_cuda.so` into the wheel (for in-tree builds). The build-time counterpart of `FLAGOS_DISABLE_CUDA_ASSETS` |
| `FLAGOS_CUDA_ASSETS_DIR` | Build | `.libtorch_cuda_assets` | Directory the external `libtorch_cuda.so` is copied from when bundling. A missing directory downgrades to a warning: the wheel then needs a runtime-supplied `libtorch_cuda.so` |
| `FLAGOS_VENDOR_TORCH_LIB` | Build & Runtime | Auto-discovered | Path to the vendor torch's `lib` directory, used when no bundled `lib_maca/`/`lib_dcu/`/`lib_ppu/` is present. One name for MetaX, DCU and PPU, whose SDK layouts differ but whose `libtorch` role does not |
| `FLAGOS_USE_CACHING_ALLOCATOR` | Runtime | `1` (on) | Caching device allocator. Set `0` to hand every allocation straight to the vendor runtime |
| `FLAGOS_WHEEL_LOCAL` | Build | SDK-derived | Local version label for the wheel (e.g., `FLAGOS_WHEEL_LOCAL=metax3.8.1`), for dev builds that must pin the exact SDK |
| `FLAGCX_TORCH_BACKEND` | Build & Runtime | `flagos` | Select the Enflame FlagCX torch integration. `flagos` links `libflagos.so` and avoids the vendor `torch-gcu` package; an explicit value is preserved |

## Compiler and Feature Backends

These variables control torch.compile integration and specialized compilation paths.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_USE_FLAGTREE` | Runtime | `0` (off) | Assert that a FlagTree build is the active Triton. Required on Ascend when the compiler is FlagTree; the check fails loudly if the installed Triton is not FlagTree |
| `FLAGOS_COMPILE_FALLBACK_EAGER` | Runtime | `0` (off) | Fall back to eager mode when torch.compile encounters unsupported operations |
| `TORCHINDUCTOR_COMPILE_THREADS` | Runtime | torch default | torch's own compile-thread count; torch_fl honors it and uses it as the compile pool size |
| `FLAGOS_TILEOPS_USE_L2` | Runtime | `0` (off) | Use the TileOps L2-cache tier |
| `FLAGOS_TILEOPS_CACHE_MAX` | Runtime | `512` | TileOps instance-cache capacity. Past the cap, results are rebuilt per call: slower but bounded |
| `FLAGOS_TILEOPS_DISABLE_ALL_CACHE` | Runtime | `0` (off) | Neutralize every TileLang cache. Correct but slow; must be set before `tileops` is imported. Sets `TILELANG_DISABLE_CACHE=1` |
| `TILELANG_DISABLE_CACHE` | Runtime | TileLang default | TileLang's own cache switch, honored as-is |
| `FLAGOS_EXEC_CACHE` | Build (codegen) | `1` (on) | Cache Ascend operator-codegen execution results; `0` forces regeneration |

For BPU-specific compilation variables, see [BPU Integration Guide](../vendors/bpu/integration.md).

## Platform-specific Variables

Detailed setup and runtime variables for each accelerator backend are documented in platform guides:

- [CUDA (NVIDIA)](../vendors/cuda/installation.md)
- [MetaX](../vendors/metax/installation.md)
- [Ascend (Huawei)](../vendors/ascend/installation.md)
- [DCU (Hygon)](../vendors/dcu/installation.md)
- [GCU (Enflame)](../vendors/gcu/installation.md)
- [MUSA (Moore Threads)](../vendors/musa/installation.md)
- [BPU (Horizon Robotics)](../vendors/bpu/integration.md)

Platform guides document SDK paths, driver requirements, version compatibility, and any additional environment setup (e.g., `LD_PRELOAD`, `LD_LIBRARY_PATH`).
