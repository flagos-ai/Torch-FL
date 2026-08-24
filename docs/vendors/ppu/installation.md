# PPU Installation Guide

## Overview

PPU (`PPU_SDK`) presents itself as a **CUDA-compatible** device, reusing the CUDA build directly with no dedicated `ACCELERATOR=ppu` branch. The PPU `torch` wheel is a full CUDA-enabled build (`torch.version.cuda == '13.0'`, `torch.cuda.is_available() == True`), and `PPU_SDK/CUDA_SDK` is a complete CUDA 13 toolkit. This makes PPU the simplest compatibility-boxing case:

- **No stock `+cpu` wheel and no external `libtorch_cuda.so`** are required — the PPU torch wheel ships its own CUDA runtime
- PPU registers ops under the `CUDA` dispatch key (not `PrivateUse1`), so the generated CUDA boxing kernels are reused unchanged
- Build selector remains `ACCELERATOR=cuda`; `PPU_SDK` detection disables bundling external CUDA assets

**Status:** Experimental. No CI manifest exists for this platform; validation rests on the build-from-source instructions and setup-specific testing.

## Prerequisites

- PPU hardware with driver installed
- PPU SDK installed at `/usr/local/PPU_SDK` (or `$PPU_SDK`/`$PPU_HOME`)
- PPU torch wheel (locally built or vendor-provided, CUDA-enabled)
- Python 3.8 or later matching the PPU torch build
- `cmake >= 3.18`, `ninja`

## Installation

### Pure CUDA-Boxing Build (No FlagGems)

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

# CUDA_HOME points at the PPU CUDA_SDK; FLAGOS_SKIP_CUDA_ASSETS=1 skips
# bundling an external libtorch_cuda.so AND skips pinned nvidia-*-cu12 deps
# (PPU supplies CUDA 13 via PPU_SDK/CUDA_SDK).
ACCELERATOR=cuda \
  CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  CUDA_KERNEL=ON \
  FLAGGEMS_KERNEL=OFF \
  FLAGGEMS_PYTHON=OFF \
  FLAGOS_SKIP_CUDA_ASSETS=1 \
  pip install --no-build-isolation -vvv -e .
```

**Key differences from NVIDIA CUDA build:**
- `CUDA_HOME` points to `PPU_SDK/CUDA_SDK`, not `/usr/local/cuda`
- `FLAGOS_SKIP_CUDA_ASSETS=1` disables bundling external CUDA assets (PPU torch provides them)
- PPU torch already has `torch.cuda.is_available() == True`, so no CPU wheel is involved

### Runtime Configuration

Set `FLAGOS_DISABLE_CUDA_ASSETS=1` so the import-time preload of bundled `libtorch_cuda.so` is a no-op (there is none — PPU torch provides it):

```bash
export FLAGOS_DISABLE_CUDA_ASSETS=1
```

## Verification

### Basic Device Check

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 python -c "
import torch_fl
import torch

print(f'PyTorch version: {torch.__version__}')
print(f'torch.cuda available: {torch.cuda.is_available()}')
print(f'torch.version.cuda: {torch.version.cuda}')
print(f'flagos devices: {torch.flagos.device_count()}')

# Basic computation
x = torch.randn(4, 4, device='flagos')
y = x @ x
print(f'Sample matmul result shape: {y.shape}')
print(f'Sample result: {y.cpu()[0, 0].item():.4f}')
"
```

Expected output shows `torch.cuda available: True`, `torch.version.cuda: 13.0`, and a floating-point result.

### `torch.compile` with FlagTree

Build FlagTree from source with its PPU backend; the resulting `flagtree` wheel
installs a `triton` module, so it must replace or shadow the vendor Triton:

```bash
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
pip install -r python/requirements.txt
FLAGTREE_BACKEND=ppu MAX_JOBS=32 \
  pip install . --no-build-isolation -v
```

Require the active compiler to be FlagTree and run the compile contract:

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  FLAGOS_USE_FLAGTREE=1 \
  pytest tests/integration/test_compile.py -v --tb=short
```

PPU FlagTree currently initializes CUDA while selecting compiler hints in an
asynchronous Inductor worker, which can fail after the parent process has already
initialized the PPU context ([FlagTree #1031](https://github.com/flagos-ai/FlagTree/issues/1031)).
Torch-FL defaults PPU FlagTree to serial compilation as a compatibility measure.
Set `TORCHINDUCTOR_COMPILE_THREADS` explicitly only when testing an upstream fix
or deliberately selecting a different worker configuration.

### AMP and GradScaler

PPU uses the shared CUDA-boxing implementation, while the public AMP device name
is `flagos`. Autocast supports `torch.float16` and `torch.bfloat16`; the AMP
foreach kernels used by `GradScaler` are routed through the same PPU CUDA runtime:

```python
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401

model = torch.nn.Linear(8, 4, device="flagos")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("flagos")
inputs = torch.randn(2, 8, device="flagos")
targets = torch.randn(2, 4, device="flagos")

with torch.autocast("flagos", dtype=torch.bfloat16):
    loss = F.mse_loss(model(inputs), targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Run the PPU-only AMP contract after installing the PPU torch wheel and loading
the driver. `PPU_SDK` or `PPU_HOME` is required so the test cannot be mistaken
for validation on an ordinary NVIDIA CUDA build:

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  PPU_SDK=/usr/local/PPU_SDK \
  pytest tests/integration/test_amp_contract.py -m amp -v --tb=short
```

This test requires a real PPU device. A CPU-only environment or a stock NVIDIA
CUDA environment does not establish PPU AMP support. BF16 availability and
GradScaler behavior must be checked against the installed PPU hardware and
vendor torch wheel.

### Operator Validation (Pure Boxing)

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  pytest tests/unit tests/integration/ops \
    tests/integration/test_factory_ops.py \
  -q -m "not flaggems and not flaggems_python"
```

## Optional: FlagGems on PPU

Set `FLAGGEMS_PYTHON=ON` at build time (the default) and `FLAGOS_USE_FLAGGEMS=1` at runtime; `import torch_fl` then selects `backends_flaggems.conf` and routes discovered ops to FlagGems' Triton kernels.

PPU needs no compatibility shim beyond the generic CUDA one: `libcuda.so` is a real driver, so `is_nvidia_cuda_available()` succeeds, `GEMS_VENDOR=nvidia` is set automatically, and `triton.language.extra.cuda.libdevice` resolves.

### Installing PPU Triton

PPU's Triton comes from a vendor index, not PyPI, and its version string (`3.5.0+v0.2.0.ppu2.1.0`) does not satisfy the `triton>=3.5.1` pin. When `PPU_SDK` is detected, `setup.py` drops the `flag_gems`/`triton` requirements, so install them manually:

```bash
# Vendor index URL varies by PPU SDK release; consult your vendor documentation
pip install triton==3.5.0+v0.2.0.ppu2.1.0  # from vendor index
pip install flag_gems
```

**Troubleshooting: `Invalid cross-device link` installing PPU Triton**

The vendor `triton` sdist is a downloader shim that fetches the real wheel and `rename()`s it into pip's cache. If your pip cache and build directory are on different filesystems (e.g., cache on NFS, build in `/tmp`), that rename fails with `[Errno 18]`.

**Fix:** Fetch the wheel URL the shim reports (`Guessing wheel URL: ...`) with `curl` and `pip install` the file directly.

### FlagGems Build

```bash
ACCELERATOR=cuda \
  CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  CUDA_KERNEL=ON \
  FLAGGEMS_PYTHON=ON \
  FLAGOS_SKIP_CUDA_ASSETS=1 \
  pip install --no-build-isolation -vvv -e .
```

### FlagGems Verification

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  FLAGOS_USE_FLAGGEMS=1 \
  python -c "
import torch_fl, torch
x = torch.randn(256, 256, device='flagos')
result = torch.softmax(x, -1)
print(f'FlagGems softmax output shape: {result.shape}')
print(f'Row sum (expect 1.0): {result[0].sum().cpu().item():.6f}')
"
```

### FlagGems Operator Tests

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  FLAGOS_USE_FLAGGEMS=1 \
  pytest tests/integration/ops -q
```

First run is slow: Triton compiles and autotunes every kernel. Subsequent runs reuse the cache.

## Optional: Distributed with FlagCX

Distributed training works out of the box on the NCCL fallback — `PPU_SDK` ships a vendor-adapted `libnccl.so.2` and the PPU torch wheel is built with `USE_NCCL=1`, so `ProcessGroupNCCL` exists natively and `ProcessGroupFlagOS` uses it on the zero-copy CUDA view.

To use FlagCX (heterogeneous unified comm) instead, build it from source:

```bash
git clone https://github.com/FlagOpen/FlagCX.git
cd FlagCX

# --depth 1 skips submodules; without third-party/json the build fails
git submodule update --init --depth 1 third-party/json

make -j16 USE_PPU=1 \
  DEVICE_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  CCL_HOME=/usr/local/PPU_SDK/CUDA_SDK

# Build the torch plugin with the 'nvidia' adaptor (not auto-detected 'ppu').
# Both produce identical torch-side code (PPU is CUDA-ABI: same CUDAStreamGuard,
# CUDAEvent, devName="cuda"), but FlagCX only compiles its extended_api creator
# for NVIDIA and MetaX adaptors. The 'ppu' adaptor also fails to compile —
# PPU is missing from the plugin's per-adaptor #ifdef chains.
cd plugin/torch
FLAGCX_ADAPTOR=nvidia \
  FLAGCX_HOME=$(git rev-parse --show-toplevel) \
  python setup.py install
```

`import torch_fl` then prefers FlagCX automatically (`_try_build_flagcx` runs before the NCCL fallback); no env var is needed beyond putting `libflagcx.so` on the loader path:

```bash
LD_LIBRARY_PATH=<path-to-FlagCX>/build/lib:$LD_LIBRARY_PATH \
  python tests/manual/test_flagos_dist_live.py --world-size 4
```

## Run Tests

### Pure Boxing Path

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  pytest tests/unit tests/integration/ops \
    tests/integration/test_factory_ops.py \
  -q -m "not flaggems and not flaggems_python"
```

### FlagGems Path

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  FLAGOS_USE_FLAGGEMS=1 \
  pytest tests/integration/ops -q
```

### Distributed (Collectives + DDP)

```bash
# Works on NCCL; add LD_LIBRARY_PATH for FlagCX
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  python tests/manual/test_flagos_dist_live.py --world-size 4
```

## Troubleshooting

### `torch.cuda.is_available()` Returns False

**Cause:** PPU torch wheel is not correctly installed, or the PPU driver is not loaded.

**Fix:** Verify PPU torch with `python -c "import torch; print(torch.version.cuda)"` — expect `13.0`. Check PPU driver status with vendor-provided diagnostic tools.

### `FLAGOS_DISABLE_CUDA_ASSETS=1` Not Set

**Symptom:** Import fails with "libtorch_cuda.so not found" or similar.

**Fix:** Export `FLAGOS_DISABLE_CUDA_ASSETS=1` before running Python. This variable disables the import-time preload of bundled CUDA assets (which PPU builds do not ship).

### Build Error: `CUDA_HOME not set`

**Cause:** `CUDA_HOME` environment variable not pointing to `PPU_SDK/CUDA_SDK`.

**Fix:** Export `CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK` before building, or pass it explicitly in the build command.

### Triton Install: `Invalid cross-device link`

**Cause:** pip cache and build directory are on different filesystems; the vendor Triton sdist's `rename()` call fails.

**Fix:** Download the wheel URL directly from the sdist's output and install the file with `pip install <wheel-file>`.

### FlagCX: `extended_api creator not found`

**Cause:** Built the FlagCX torch plugin with auto-detected `ppu` adaptor, which does not export the extended_api creator.

**Fix:** Rebuild with `FLAGCX_ADAPTOR=nvidia` (see "Optional: Distributed with FlagCX" above).

## Further Reading

- [Environment Variables](../../reference/environment-variables.md) — Complete runtime configuration reference
- [Distributed (FlagCX)](../../architecture/distributed-flagcx.md) — Multi-GPU collectives and heterogeneous communication
- [Testing Guide](../../development/testing.md) — Running local validation
