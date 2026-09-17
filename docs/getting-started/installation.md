# Installation

## Choose a Platform

| Platform | Build selector | Execution path | Installation guide |
|---|---|---|---|
| NVIDIA CUDA | `FLAGOS_ACCELERATOR=cuda` (default) | CUDA boxing over an external `libtorch_cuda.so` | [CUDA Installation](../vendors/cuda/installation.md) |
| MetaX | `FLAGOS_ACCELERATOR=metax` | CUDA boxing via `cu-bridge` against the vendor libtorch | [MetaX Installation](../vendors/metax/installation.md) |
| Ascend | `FLAGOS_ACCELERATOR=ascend` | Native ACLNN operator backend, FlagGems via FlagTree (Triton 3.5) | [Ascend Installation](../vendors/ascend/installation.md) |
| PPU | `FLAGOS_ACCELERATOR=ppu` | Same CUDA-boxing path as NVIDIA CUDA, against the PPU's CUDA-13-compatible SDK, bundling its own libtorch into `lib_ppu/` | [PPU Installation](../vendors/ppu/installation.md) |
| Hygon DCU | `FLAGOS_ACCELERATOR=dcu` | CUDA boxing over the hipified DTK torch build (HIP kernels under the CUDA dispatch key) | [DCU Installation](../vendors/dcu/installation.md) |
| Enflame GCU | `FLAGOS_ACCELERATOR=gcu` | Native `libtopsaten.so` operator backend, with CPU fallback for unrouted/int64 ops | [GCU Installation](../vendors/gcu/installation.md) |
| Moore Threads MUSA | `FLAGOS_ACCELERATOR=musa` | FlagGems-first Triton kernels, native `mudnn` backend as fallback, with CPU fallback for unrouted ops | [MUSA Installation](../vendors/musa/installation.md) |
| D-Robotics BPU | `FLAGOS_ACCELERATOR=bpu` | No eager kernel sets are built; eager ops run on CPU | [BPU Installation](../vendors/bpu/installation.md) |
| TsingMicro | `FLAGOS_ACCELERATOR=tsingmicro` | Runtime/build selector present; no per-op kernel set documented | The runtime build branch exists, but a current end-to-end installation and validation procedure is not documented. |

## Common Requirements

All platforms require:

- **Python**: 3.8 or later
- **PyTorch**: 2.10.x (`>=2.10,<2.11`) — generated ATen bindings are tied to this minor line
- **CMake**: 3.18 or later
- **C++ build toolchain**: A working C++17 compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- **Platform SDK/runtime**: The vendor-specific SDK, compiler, and runtime libraries for your accelerator

Platform-specific requirements (CUDA toolkit version, vendor SDK paths, additional dependencies) are documented in each platform's installation guide.

## Source Installation Contract

Each platform installation guide defines:

1. The required `FLAGOS_ACCELERATOR` value for that platform
2. SDK/compiler environment variables (e.g., `CUDA_HOME`, `MACA_PATH`, `ASCEND_HOME`)
3. Any platform-specific build flags or dependencies

The general installation pattern is:

```bash
FLAGOS_ACCELERATOR=<platform> pip install --no-build-isolation -e .
```

The `--no-build-isolation` flag is required so that generated native bindings compile against the PyTorch installation visible in your current environment. Without it, pip creates an isolated build environment that may not see your platform SDK or the correct PyTorch installation.

## After Installation

- **First steps**: See the [Quick Start Guide](quickstart.md) for platform-independent usage examples.
- **Platform support**: Review the [Compatibility Matrix](../reference/compatibility.md) to understand what capabilities are validated for your platform.
- **Configuration**: Environment variables controlling runtime behavior are documented in the [Environment Variables Reference](../reference/environment-variables.md).
- **Testing**: Run the test suite following the [Testing Guide](../development/testing.md) to verify your installation.
