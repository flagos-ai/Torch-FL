# NVIDIA CUDA Installation Guide

## Overview

The CUDA backend reuses PyTorch's optimized CUDA kernels through compatibility-key boxing, eliminating the need for hand-written kernels. This route installs CPU-only PyTorch via pip and bundles a version-matched external `libtorch_cuda.so` that provides the full CUDA operator set. An optional FlagGems compiler path routes eligible operations to Triton kernels at runtime.

**Status:** Stable. CI validates vendor-backend and FlagGems operator suites, factory ops, profiler parity, and Qwen3-0.6B inference/training on 8-GPU A100 runners.

## Prerequisites

- NVIDIA GPU with compute capability 7.0 or later (tested on RTX 2080 Ti, A100)
- NVIDIA driver 470.0 or later (tested with 550.163.01)
- CUDA 12.x toolkit (headers and `nvcc` required for build; runtime dependencies are bundled)
- Python 3.8 or later
- PyTorch 2.10.0 (CPU wheel; installed automatically from the build script)
- `cmake >= 3.18`, `ninja`, `patchelf` (for build)

## Installation

### From Source (Standard Build)

Clone the repository and build with the CUDA accelerator:

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

# Install CPU PyTorch (if not already present)
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Build torch_fl with CUDA boxing kernels and bundled CUDA assets
ACCELERATOR=cuda \
  pip install --no-build-isolation -vvv -e .
```

This build:
- Generates CUDA boxing kernels from PyTorch's ATen schema (`csrc/aten/generated/cuda_kernels.cc`)
- Bundles `libtorch_cuda.so` and related CUDA dispatcher libraries into `torch_fl/lib/`
- Compiles FlagGems Python-dispatch integration by default (routing comes from `backends_cuda.conf`)
- Pins matching `nvidia-*-cu12` runtime dependencies for the bundled CUDA assets

### With FlagGems C++ Dispatch (Optional)

To enable the C++ fast path for FlagGems Triton kernels:

```bash
ACCELERATOR=cuda \
  FLAGGEMS_CPP=1 \
  FLAGGEMS_DIR=<path-to-FlagGems>/lib/cmake/FlagGems \
  pip install --no-build-isolation -vvv -e .
```

`FLAGGEMS_DIR` must point to the FlagGems CMake config directory containing `FlagGemsConfig.cmake` and `liboperators.so`. See [FlagGems integration](../flaggems/) for build details.

### Environment Variables

The following environment variables control runtime behavior:

- `FLAGOS_USE_FLAGGEMS_CPP=1`: Enable the `flaggems_cpp` test gate. Routing itself follows the conf's `flaggems_cpp` keys, so this variable only decides whether those tests are collected
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible to the process

## Verification

### Basic Device Check

```bash
python -c "
import torch_fl
import torch

print(f'PyTorch version: {torch.__version__}')
print(f'flagos devices: {torch.flagos.device_count()}')
print(f'flagos available: {torch.flagos.is_available()}')

# Basic computation
x = torch.randn(4, 4, device='flagos:0')
y = (x @ x).sum()
print(f'Sample result: {y.cpu().item():.4f}')
"
```

Expected output shows `flagos devices: N` (where N matches your GPU count) and a floating-point result.

### Operator Validation

Run the main operator test suite against the CUDA boxing backend:

```bash
pytest tests/integration/ops/ \
  -m "main_ops and not flaggems_python and not flaggems_cpp" \
  -v --tb=short
```

### FlagGems Validation

Test the FlagGems runtime path (FlagGems-first routing comes from
`torch_fl/configs/backends_cuda.conf`; no opt-in variable needed):

```bash
pytest tests/integration/ops/ \
  -m "flaggems and main_ops" \
  -v --tb=short
```

First run is slow — Triton compiles and autotunes every kernel. Subsequent runs reuse the cache.

### Factory Operations

Verify factory operations respect device placement:

```bash
pytest tests/integration/test_factory_ops.py -v --tb=short
```

### Distributed (Multi-GPU)

Verify collective communication and DDP gradient sync:

```bash
# Requires 2+ GPUs and FlagCX installed (optional; falls back to NCCL)
python tests/manual/test_flagos_dist_live.py --world-size 2
```

See [Distributed (FlagCX)](../../architecture/distributed-flagcx.md) for FlagCX installation and heterogeneous communication.

### Profiler

Verify profiler parity with torch.cuda:

```bash
pytest tests/integration/test_profiler_parity.py -v -m main_ops --tb=short
```

See [Profiler Architecture](../../architecture/profiler.md) for details on CUPTI integration and device event emission.

### Model Inference and Training

Run Qwen3-0.6B inference and training (requires model weights):

```bash
# Inference
pytest tests/integration/test_qwen3_infer.py \
  --model <model-path> -v --tb=short

# Training
pytest tests/integration/test_qwen3_train.py \
  --model <model-path> -v --tb=short
```

### torch.compile (Experimental)

The CUDA backend registers flagos as a first-class inductor GPU device. Basic verification:

```bash
python -c "
import torch
import torch_fl

@torch.compile
def f(x):
    return torch.nn.functional.gelu(x) + x

x = torch.randn(256, 256, device='flagos:0')
y = f(x)
print(f'Compiled result shape: {y.shape}')
"
```

See [torch.compile Integration](../../architecture/torch-compile-integration.md) for implementation details. This path is not exercised in CI.

## Troubleshooting

### Import Error: `Cannot initialize CUDA without ATen_cuda library`

**Cause:** `libtorch_cuda.so` was not preloaded before `import torch`, so PyTorch's CUDAHooks cached the stub implementation.

**Fix:** Ensure `import torch_fl` occurs before any torch import in your script. The torch_fl import hook preloads the bundled CUDA assets.

### Runtime Error: `CUDA error: invalid device ordinal`

**Cause:** `device="flagos:N"` index exceeds available GPU count, or `CUDA_VISIBLE_DEVICES` restricts visibility.

**Fix:** Check `torch.flagos.device_count()` and ensure the device index is in range.

### Symbol Version Mismatch

**Cause:** PyTorch CPU wheel version does not match the bundled `libtorch_cuda.so` version.

**Fix:** The build script pins torch 2.10.0+cpu and bundles matching CUDA assets. If you manually installed a different torch version, rebuild torch_fl or reinstall `torch==2.10.0+cpu`.

### FlagGems Import Error

**Cause:** `flag_gems` package not installed, or `FLAGGEMS_DIR` pointed to an incompatible build.

**Fix:** Install FlagGems from PyPI (`pip install flag_gems`) or build from source matching your CUDA version. See [FlagGems Integration](../flaggems/).

### nvidia-smi Shows GPUs, but `torch.flagos.device_count()` Returns 0

**Cause:** CUDA runtime initialization failed, often due to driver/runtime version skew.

**Fix:** Check `nvidia-smi` output for driver version and ensure CUDA runtime dependencies match. Reinstall `nvidia-*-cu12` packages if needed.

## Further Reading

- [External libtorch_cuda.so: Measured Validation](external-libtorch-cuda.md) — Deep dive into the CPU torch + external CUDA dispatcher architecture
- [torch.compile Integration](../../architecture/torch-compile-integration.md) — Inductor GPU device registration and compilation workflow
- [Distributed (FlagCX)](../../architecture/distributed-flagcx.md) — Multi-GPU collectives, DDP, and heterogeneous communication
- [Profiler Architecture](../../architecture/profiler.md) — CUPTI integration and device event emission
- [Environment Variables](../../reference/environment-variables.md) — Complete runtime configuration reference
- [Testing Guide](../../development/testing.md) — Running the full test suite and CI-equivalent validation
