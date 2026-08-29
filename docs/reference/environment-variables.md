# Environment Variables

This document lists configuration variables that control torch_fl's build, operator routing, and runtime behavior. Platform-specific setup variables are documented in vendor guides.

## Build Selection

These variables control which backends and kernels are compiled into the wheel.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `ACCELERATOR` | Build | `cuda` | Hardware platform: `cuda`, `metax`, `ascend`, `tsingmicro`, `dcu`, `gcu`, `musa`, or `bpu` |
| `FLAGOS_BUILD_JOBS` | Build | System CPU count | Parallel jobs for CMake build |
| `CUDA_KERNEL` | Build | `ON` | Enable CUDA boxing kernels; set `OFF` for pure vendor builds (Ascend/MUSA/GCU) |
| `ASCEND_KERNEL` | Build | `OFF` | Enable Ascend ACL kernels; set `ON` for Ascend builds |
| `GCU_KERNEL` | Build | Auto (enabled when `ACCELERATOR=gcu`) | Enable Enflame GCU topsaten kernels |
| `MUSA_KERNEL` | Build | Auto (enabled when `ACCELERATOR=musa`) | Enable Moore Threads MUSA mudnn kernels |
| `METAX_KERNEL` | Build | Auto (enabled when `ACCELERATOR=metax`) | Enable MetaX C++ kernel build |
| `FLAGGEMS_KERNEL` | Build | `ON` | Enable FlagGems C++ kernel wrappers (liboperators.so); set `OFF` for pure vendor builds |
| `FLAGGEMS_PYTHON` | Build | `OFF` | Enable FlagGems Python wrapper backend registration |
| `FLAGOS_METAX_BOXING` | Build | `OFF` | Enable MetaX boxing mode (reuse CUDA boxing kernels + optional FlagGems, no mxcc backend) |

## SDK and Compiler Discovery

These variables locate platform SDKs and toolchains. CMake searches common install paths when they are absent.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `CUDA_HOME` | Build & runtime | Auto-detected (system CUDA or conda prefix) | CUDA toolkit path for headers and libraries |
| `ASCEND_HOME` | Build | `/usr/local/Ascend/ascend-toolkit/latest` | CANN toolkit path for Ascend NPU builds |
| `MUSA_HOME` | Build | `/usr/local/musa` | Moore Threads MUSA toolkit path |
| `TOPS_HOME` | Build | `/opt/tops` | Enflame TopsRider SDK path for GCU builds |
| `METAX_PATH` | Build | `/opt/maca` or `$METAX_HOME` or `$MACA_PATH` or `$MACA_HOME` (first found) | MetaX SDK path |
| `METAX_ARCH` | Build | No global default | MetaX GPU architecture string (e.g., `mp_21`) |
| `METAX_MXCC` | Build | No global default | Path to mxcc/cucc compiler override |
| `DTK_ROOT` | Build | `$ROCM_PATH` or `/opt/dtk` | Hygon DTK path for DCU builds |
| `PPU_SDK` / `PPU_HOME` | Build | No global default | PPU SDK path for Tsingmicro builds |
| `CONDA_PREFIX` | Build & runtime | Auto-detected | Conda environment prefix (fallback for CUDA discovery) |

## Operator Routing

These variables control which backend implementation (CUDA boxing, vendor C++, FlagGems C++, or FlagGems Python) each operator dispatches to at runtime.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_BACKEND_CONFIG` | Runtime | Auto-selected by `torch_fl.__init__` based on hardware and switches below | Absolute path to a `backends_*.conf` file; overrides all auto-detection |
| `FLAGOS_USE_FLAGGEMS` | Runtime | `0` (off) | Enable FlagGems Triton operators via the Python backend (`flagos_python`); selects `backends_flaggems.conf` or platform-specific variant |
| `FLAGOS_USE_FLAGGEMS_CPP` | Runtime | `0` (off) | Enable FlagGems C++ operators (kFlagOs dispatch, no GIL); selects `backends_flaggems_cpp.conf`; requires wheel built with `FLAGGEMS_KERNEL=ON` |
| `FLAGOS_OP_<name>` | Runtime | No default | Per-operator backend override (e.g., `FLAGOS_OP_add__Tensor=cuda`); replace `.` with `__` in op names |
| `FLAGOS_LOG_DISPATCH` | Runtime | `0` (off) | Print backend selection to stderr for each operator dispatch |
| `FLAGOS_DISABLE_FLAGGEMS_PY` | Runtime | `0` (off) | Disable FlagGems Python-layer registration (C++ stub-only mode) |
| `FLAGGEMS_SOURCE_DIR` | Runtime | Required when FlagGems is active | Absolute path to FlagGems source directory (Python Triton kernels); must match the version liboperators.so was built against |
| `FLAGOS_DISABLE_APEX_COMPAT` | Runtime | `0` (off) | Disable optional Apex multi-tensor compatibility layer; see Apex compatibility section below |
| `GEMS_VENDOR` | Runtime | Auto-detected from hardware or build metadata | FlagGems vendor target: `cuda` (default), `metax`, `ascend`, `musa`, `cambricon`; controls distributed backend and device-specific Triton compilation |

**Apex compatibility**: On CUDA-ABI boxing vendors, Torch-FL automatically patches Apex's common `MultiTensorApply` entry point when Apex is imported. The patch converts flagos tensors to zero-copy CUDA views for direct `amp_C` calls and converts CUDA results back to flagos views. It is optional and does not apply to native non-CUDA backends. Set `FLAGOS_DISABLE_APEX_COMPAT=1` to disable it.

**Note on auto-detection**: `FLAGOS_BACKEND_CONFIG` is normally set by `torch_fl.__init__._select_backend_config()`, which detects the hardware platform (via `/dev/davinci*`, `/dev/mxcd`, or build-time `ACCELERATOR`) and applies the `FLAGOS_USE_FLAGGEMS` / `FLAGOS_USE_FLAGGEMS_CPP` / `FLAGOS_METAX_BOXING` switches to pick the correct config. Users should override `FLAGOS_BACKEND_CONFIG` only for testing or debugging.

## Runtime and Packaging

These variables control asset loading, bundled libtorch behavior, and import-time compatibility shims.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_DISABLE_CUDA_ASSETS` | Runtime | `0` (off) | Skip preloading bundled `libtorch_cuda.so` and CUDA libraries (for builds that use system libtorch) |
| `FLAGOS_DISABLE_CUDA_SHIM` | Runtime | `0` (off) | Skip registering the `torch.cuda` compatibility shim for generic GPU operations |
| `FLAGOS_ALIAS_CUDA` | Runtime | `0` (off) | Alias `cuda` device string to `flagos` for drop-in compatibility |
| `FLAGOS_METAX_CUDART_SHIM` | Runtime | `0` (off) | Preload libcudart version-tag shim before `import torch` (MetaX-specific; required for generic PyTorch wheels) |
| `FLAGOS_METAX_COMPAT` | Runtime | `0` (off) | Patch FlagGems `torch.cuda` device queries for MetaX compatibility |
| `FLAGOS_DCU_HIP_VERSION` | Runtime | No default | Override HIP version detection for DCU runtime |
| `FLAGOS_DCU_VENDOR_CORE` | Build & Runtime | `0` (off) | Use DTK's forked core libraries instead of the official PyTorch core: bundles the full vendor core and symlinks it over the installed torch wheel. Must match at build and import time. See [DCU without DTK's core libraries](../vendors/dcu/vendor-free-core-libs.md) |
| `FLAGOS_DCU_SKIP_RUNTIME_CHECK` | Runtime | `0` (off) | Skip the DCU post-import checks (torch/DTK version alignment and CUDA-key kernel presence). For deliberately testing a non-matching wheel pair |
| `FLAGOS_DCU_TORCH_LIB` | Build & Runtime | Auto-discovered | Path to DTK's `torch/lib`, used when no bundled `lib_dcu/` is present |
| `FLAGOS_SKIP_CUDA_ASSETS` | Build | `0` (off) | Skip bundling external `libtorch_cuda.so` into the wheel (for in-tree builds) |
| `FLAGOS_WHEEL_LOCAL` | Build | `0` (off) | Build a wheel with machine-local paths (non-portable; for dev testing only) |
| `FLAGCX_TORCH_BACKEND` | Build & Runtime | `torch_gcu` for Enflame FlagCX builds; torch-fl sets `flagos` by default | Select the Enflame FlagCX torch integration. `flagos` links `libflagos.so` and avoids the vendor `torch-gcu` package; an explicit value is preserved. |
| `FLAGOS_INSTALL_PATH` | Build | No default | FlagCX torch-fl compatibility install prefix containing `include/flagos.h` and `lib/libflagos.so` |
| `FLAGOS_INCLUDE_DIR` / `FLAGOS_LIBRARY_DIR` | Build | Derived from `FLAGOS_INSTALL_PATH` or `torch_fl` | Override the FlagOS C ABI header and library directories when building FlagCX |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Runtime | Set by torch_fl on MUSA builds | Stop vendor plugins (e.g., `torch_musa`) from claiming `PrivateUse1` during `import torch` |

## Compiler and Feature Backends

These variables control torch.compile integration and specialized compilation paths.

| Variable | Scope | Default | Purpose |
|----------|-------|---------|---------|
| `FLAGOS_COMPILE_FALLBACK_EAGER` | Runtime | `0` (off) | Fall back to eager mode when torch.compile encounters unsupported operations |
| `FLAGOS_USE_FLAGTREE` | Runtime | `0` (off) | Enable FlagTree compiler backend integration |

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
