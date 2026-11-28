# PPU Installation Guide

## Overview

PPU uses the CUDA-compatible toolchain in `PPU_SDK/CUDA_SDK` and selects
`FLAGOS_ACCELERATOR=ppu`. The default Torch-FL build uses official CPU Torch and
exposes the PPU through `torch.flagos`. The separate vendor Torch wheel is a
CUDA-enabled build with `torch.version.cuda == '13.0'`.

The T-Head wheel uses FlagGems on the PPU SDK runtime.
`backends_thead.conf` records the accelerated operators and CPU fallbacks.

## CI runtime environment

The CI config pins the official PPU SDK 2.1.0 runtime 2.3.0 by manifest digest:

```text
harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0@sha256:35fcbd0c3777f0af1f936fc725a91be76279ea77849d192ea11d6f90c12172bc
```

The [official runtime inventory](https://flagos-ai.github.io/release-info/zh-cn/runtime/thead-ppu2.1.0/)
contains Python 3.12, vendor Torch `2.10.0+v0.1.0.ppu2.1.1`, FlagTree
`0.7.0+ppu3.6`, FlagGems `5.4.0`, and NumPy `1.26.4`. The SDK is under
`/usr/local/PPU_SDK`; the image build history installs MKL `2021.1.1` into
`/usr/local/lib`. No new MKL
installation or vendor Torch build is performed by CI.

The platform hook creates a separate Python 3.12 venv and calls the same
`seed_runtime.py` used by the other runtime lanes. It copies only matching,
RECORD-owned distributions and regenerates console scripts for the target
interpreter. The image's vendor Torch, editable `.pth` files, and entire
`/flagos` or `/opt/flagtree` import paths are not copied. Python tools declared
in `.github/requirements-thead.txt` are checked after copying; pip installs only
missing or incompatible requirements. System tools and SDK libraries stay in
the image. FlagTree reuse requires matching metadata, an import origin inside
the venv and the PPU backend. The vendor's alternate `/opt/triton` compiler is
never activated. Integration-only model dependencies retain `transformers>=4.51,<5`,
sentencepiece, tiktoken and protobuf. SQLAlchemy uses the shared migrated-CI
range `>=2.0.55,<2.0.58`; compatible image packages are retained.

The CI lane uses published `torch==2.10.0+cpu` and compiles the ordinary
FlagGems wheel without a vendor Torch bundle. The image provides Python,
system build tools and the PPU SDK; a vendor Torch package is not required.
CI records the actual SDK
`2.1.0` separately from the wheel label selected by the shared release table.

`BASH_ENV=/dev/null`, empty `PYTHONPATH`, disabled user sites and disabled vendor
backend autoload persist into later Actions steps. The job uses the shared PPU
pip download cache and its ownership/write-probe fallback with a fresh venv.
Existing runner labels, explicit device mounts and shared-memory options are
retained. No privileged mode or host IPC/network options are added.

The official runtime lists host driver `2.0.0`, while the previous CI device tool
reported `1.3.2-d7f5a2`. Compatibility must be established by the new-image
hardware runs; local provisioning tests do not establish device support. The
hook and integration checks retain real device/runtime validation, and this
migration does not update host drivers or add test skips. Old-image green runs
are not acceptance evidence for this new runtime.

## Prerequisites

- PPU hardware with driver installed
- PPU SDK installed at `/usr/local/PPU_SDK` (or `$PPU_SDK`)
- Official PyTorch `2.10.0+cpu`
- Python 3.12 for the published PPU FlagTree wheel
- `cmake >= 3.18`, `ninja`

## Installation

### From Source (Default Build, FlagGems-primary)

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

# CUDA_HOME points at the PPU SDK's CUDA-compatible toolkit.
FLAGOS_ACCELERATOR=ppu \
  CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  FLAGOS_SKIP_CUDA_ASSETS=1 \
  pip install --no-build-isolation -vvv -e .
```

The wheel runs FlagGems kernels through the PPU runtime. Operators with a `none` route in `backends_thead.conf` use CPU fallback.

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

For the default wheel, `torch.version.cuda` is `None`. The FlagOS device count
should be nonzero and the matmul should return a floating-point result.

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

### AOT stream chain

Hardware CI also builds a `FLAGOS_BUILD_NATIVE_HOST=1` wheel using the
published PPU FlagTree wheel pinned in `.github/version-pins.env`. The AOT
build packages the published HGGC binary loader and submits kernels through
the PPU SDK on the current FlagOS stream. It supports software tensor
descriptors as provided by the published compiler.

CI preflights the AOT wheel and installs it in a separate import directory.
The common `test_compiled_stream_chain` exercises eager, Inductor and native
host submissions on one non-default stream from that installation.

### Native Host for eager execution

Build with the official PyTorch 2.10.0 CPU wheel and the pinned PPU FlagTree
and FlagGems wheels. Set `FLAGOS_BUILD_NATIVE_HOST=1` and
`FLAGOS_VENDOR_TORCH_LIB` to the `torch/lib` directory of the validated T-Head
PyTorch 2.10.0 distribution. The build verifies the vendor wheel's revision,
CUDA ABI and device libraries before packaging `libc10_cuda.so` and
`libtorch_cuda.so` under `native_host/_generated/thead_vendor`. The installed
wheel uses the official CPU PyTorch core.

Set `FLAGOS_NATIVE_HOST_DEVICE_BUNDLE` to the directory of prebuilt device
kernels at build time. Use `FLAGOS_NATIVE_HOST_REQUIRE_DEVICE_ARTIFACTS=1`
when running eager workloads to require those artifacts for FlagGems launches.

### AMP and GradScaler

Use `flagos` as the AMP device name. The supported autocast dtypes are
`torch.float16` and `torch.bfloat16`:

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

Run the PPU-only AMP contract after installing the default Torch-FL wheel and loading
the driver. `FLAGOS_ACCELERATOR=ppu` (with `PPU_SDK` present) is what the contract's
platform check reads, so the test cannot be mistaken
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

### Operator validation

```bash
python -m pytest tests/integration/ops/test_reference_ops.py -v --tb=short
python -m pytest tests/integration/ -m multi_device -v --tb=short
```

## FlagGems on PPU

`import torch_fl` loads `torch_fl/configs/backends_thead.conf`.

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
FLAGOS_ACCELERATOR=ppu \
  CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  FLAGOS_SKIP_CUDA_ASSETS=1 \
  pip install --no-build-isolation -vvv -e .
```

### FlagGems Verification

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
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

### FlagGems Path

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  pytest tests/integration/ops -q
```

### Distributed (Collectives + DDP)

```bash
# Works on NCCL; add LD_LIBRARY_PATH for FlagCX
FLAGOS_DISABLE_CUDA_ASSETS=1 \
  python tests/manual/test_flagos_dist_live.py --world-size 4
```

## Troubleshooting

### `torch.flagos.device_count()` Returns Zero

Check the PPU driver, SDK and device visibility with `ppu-smi`. With the
default wheel, `torch.version.cuda` is `None`; use `torch.flagos` to check the
PPU device.

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
