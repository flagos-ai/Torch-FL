# MetaX Installation Guide

## Overview

MetaX ships a **self-contained boxing wheel**: it reuses PyTorch's CUDA boxing kernels (compiled with host `g++`, no `mxcc`) and bundles the MetaX-forked libtorch C++ runtime inside the wheel. The target machine needs only the stock `torch==2.10.0+cpu` wheel, this `torch_fl` wheel, and the `/opt/maca` driver runtime.

**Status:** Stable. CI validates representative operator dispatch and factory/autograd tests on 8-GPU MetaX runners. FlagGems and model-level validation are not covered in CI.

## Prerequisites

### Build Host (Wheel Builder)

Requires the full MetaX SDK and a `torch+metax` wheel to extract the forked libtorch:

- MetaX MACA SDK (driver + cu-bridge), installed to `/opt/maca` or `$METAX_PATH`
- `torch+metax` wheel (`maca-pytorch`) from the MetaX developer portal (SoftNova)
- Python 3.8 or later matching the target deployment environment
- `patchelf` (`pip install patchelf`)

**Getting the MetaX SDK and torch+metax wheel:**  
Both are distributed through the MetaX developer portal: <https://developer.metax-tech.com/softnova>. Registration and login are required. Download the MACA SDK matching your card and driver version, and the `torch+metax` wheel built for the same MACA version and your Python version. Install the SDK to `/opt/maca` (or export `METAX_PATH` to the install location).

### Target Host (Deployment)

Requires only:

- Official `torch==2.10.0+cpu` from PyPI (no CUDA)
- The `torch_fl` wheel built below
- `/opt/maca` driver runtime (present on any machine with a MetaX card)

No separate `torch+metax` wheel, no manual `LD_LIBRARY_PATH` configuration.

## Installation

### Building the Wheel

#### Step 1: Build the Boxing Artifacts

On a machine with the MetaX SDK and `torch+metax` wheel available:

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

# Build the boxing artifacts (METAX_KERNEL forced OFF in boxing mode)
ACCELERATOR=metax \
  FLAGOS_METAX_BOXING=1 \
  FLAGOS_MACA_TORCH_LIB=<path-to-torch+metax>/torch/lib \
  FLAGOS_WHEEL_LOCAL=metax3.8.1 \
  python setup.py bdist_wheel
```

**Parameters:**
- `FLAGOS_MACA_TORCH_LIB`: Path to the `torch+metax` wheel's `torch/lib` directory (source of forked libtorch)
- `FLAGOS_WHEEL_LOCAL`: Local version tag (e.g., `metax3.8.1` → wheel version `0.1.0+metax3.8.1`), identifying the target MACA/driver version

#### Step 2: Bundle the Forked Libtorch

```bash
FLAGOS_MACA_TORCH_LIB=<path-to-torch+metax>/torch/lib \
  MACA_PATH=/opt/maca \
  bash scripts/bundle_maca_libtorch.sh
```

This script:
- Copies 8 forked libtorch `.so` files from `torch+metax/torch/lib` into `torch_fl/lib_maca/`
- Rewrites RPATH with `patchelf` so libraries find each other via `$ORIGIN` and locate the MACA runtime at `/opt/maca/lib`

#### Step 3: Repackage the Wheel

```bash
python setup.py build_py
cp build/lib.*/torch_fl/_C.*.so build/lib.*/torch_fl/
python setup.py bdist_wheel --skip-build --bdist-dir "$(mktemp -d)"
```

The result is `dist/torch_fl-0.1.0+metax3.8.1-cp312-cp312-linux_x86_64.whl` (~1.1 GB — it bundles the forked libtorch and exceeds PyPI's 100 MB limit; distribute via private index or direct transfer).

### Installation on Target Host

#### Install Dependencies

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_fl-0.1.0+metax3.8.1-cp312-cp312-linux_x86_64.whl
```

#### Runtime Configuration

Set the boxing mode environment variable:

```bash
export FLAGOS_METAX_BOXING=1
```

**Import order:** On MetaX, you **must** import `torch_fl` before `import torch`:

```python
import torch_fl  # Must import first
import torch
```

**Reason:** PyTorch's bundled CUDA 12.x runtime is ABI-incompatible with MetaX's cu-bridge (CUDA 11.6 compatibility layer). `torch_fl` preloads a shim library to provide the required symbol versions before torch initializes.

## Verification

### Basic Device Check

```bash
export FLAGOS_METAX_BOXING=1
python -c "
import torch_fl  # Import first
import torch

print(f'PyTorch version: {torch.__version__}')
print(f'torch.cuda devices (MetaX): {torch.cuda.device_count()}')
print(f'flagos devices: {torch.flagos.device_count()}')
print(f'flagos available: {torch.flagos.is_available()}')

# Basic computation
x = torch.randn(4, 4, device='flagos:0')
y = (x + x).sum()
print(f'Sample result: {y.cpu().item():.4f}')
"
```

Expected output shows `torch.cuda devices: N` and `flagos devices: N` (MetaX cards present as `torch.cuda` devices), and a floating-point result.

### Operator Validation

Run representative operator tests:

```bash
export FLAGOS_METAX_BOXING=1
pytest \
  tests/integration/ops/test_abs_dispatch.py \
  tests/integration/ops/test_add_dispatch.py \
  tests/integration/ops/test_bmm_dispatch.py \
  tests/integration/ops/test_mm_dispatch.py \
  tests/integration/ops/test_softmax_dispatch.py \
  -m "not flaggems and not flaggems_python and not flaggems_cpp" \
  -v --tb=short
```

### Factory and Autograd

```bash
export FLAGOS_METAX_BOXING=1
pytest tests/integration/test_factory_ops.py -v --tb=short
```

### Automatic Mixed Precision

MetaX boxing reuses the CUDA operator path while exposing PyTorch's device-generic
AMP API through `flagos`. Both FP16 and BF16 are supported autocast targets, and
`GradScaler` uses the boxed CUDA non-finite check and unscale kernels:

```python
import torch_fl  # Import first on MetaX.
import torch

model = torch.nn.Linear(8, 4).to("flagos")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("flagos")
x = torch.randn(2, 8, device="flagos")
target = torch.randn(2, 4, device="flagos")

with torch.autocast("flagos", dtype=torch.float16):
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Run the complete autocast and GradScaler contract on MetaX hardware:

```bash
export FLAGOS_METAX_BOXING=1
pytest tests/integration/test_amp_contract.py -m amp -v --tb=short
```

The measured C550/MACA 3.8.0 coverage includes FP16 and BF16 lower-precision,
FP32, optional-dtype, and promote autocast policies; nested state; non-finite
unscale; finite scale growth; overflow backoff; and a forward/backward optimizer
step. This validation applies to the boxing path, not the legacy handwritten
MetaX kernel mode.

### Low-Precision Matrix Operations

MetaX boxing also provides software emulation for scalar FP8 and packed FP4 matrix
inputs. The supported dtypes are `float8_e4m3fn`, `float8_e5m2`,
`float8_e4m3fnuz`, `float8_e5m2fnuz`, `float8_e8m0fnu`, and
`float4_e2m1fn_x2`. The shared device-side path decodes inputs to BF16 and uses
ordinary device GEMM for `mm`, `bmm`, and `addmm`, including `dtype` and `out`
variants and in-place `addmm_`. Without an explicit output dtype, results are BF16;
an explicit output dtype is preserved.

This support does not include block-scaled metadata formats such as MXFP4, NVFP4,
or block FP4, and `_scaled_mm`/`_scaled_grouped_mm` remain fail-closed because no
software kernel is provided for their scale metadata contract. Validate the path on
a C550 target with:

```bash
export FLAGOS_METAX_BOXING=1
pytest tests/integration/ops/test_soft_lowp_gate_dispatch.py -m soft_lowp -v -s --tb=short
```

On the C550/MACA 3.8.0 environment used for this checkout, the complete marked suite
passed (`37 passed`), including all five FP8 formats, packed FP4, `mm`/`bmm`/`addmm`,
`dtype`/`out` variants, non-square packed layouts, in-place `addmm_`, and the fail-closed
scaled-mm check. Re-run the command above after changing the build or vendor runtime;
the result applies to the CUDA-boxing path and not the legacy handwritten MetaX kernel mode.

### torch.compile and FlagTree

The boxing path supports `torch.compile(backend="flagos")` with either the MetaX
Triton distribution or a MetaX-enabled [FlagTree](https://github.com/flagos-ai/FlagTree)
build. FlagTree installs the module named `triton`; `FLAGOS_USE_FLAGTREE=1`
asserts that this replacement is active rather than switching it at runtime.

A Triton is required: the official `torch==2.10.0+cpu` wheel this path installs
against ships none, and Inductor raises `TritonMissing` without one. Install or
expose the MetaX Triton distribution (`triton-3.6.0+metax*`) alongside the CPU
wheel; linking just the `triton` package and its `.dist-info` is enough, and the
`.dist-info` is required because Triton discovers its hardware backends through
`importlib.metadata` entry points.

Inductor forwards the Triton backend name (`maca` on MetaX) to its benchmarker as
a torch device during autotuning. The vendor MetaX torch build patches that
in-tree; the official CPU wheel does not, so torch_fl maps `maca` back to `cuda`
itself (`torch_fl/compile/device_interface.py`). Both refer to the same physical
GPU, so no separate configuration is needed.
Build FlagTree main (Triton 3.6 for PyTorch 2.10) in a separate environment:

```bash
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
export FLAGTREE_BACKEND=metax
MAX_JOBS=64 python -m pip install . --no-build-isolation -v
```

Then run the compile contract with that environment's site-packages ahead of the
normal MetaX Triton installation:

```bash
PYTHONPATH=/path/to/flagtree-venv/lib/python3.12/site-packages \
  FLAGOS_USE_FLAGTREE=1 \
  FLAGOS_METAX_BOXING=1 \
  pytest tests/integration/test_compile.py -v --tb=short
```

On C550 with MACA 3.8.0, the full suite passed with FlagTree revision
`140bd6ab1ad86c5df4b07b76d9c722e357a9166d` (Triton 3.6, MetaX backend),
covering forward, backward, FP32/FP16, max-autotune, recompilation, FakeTensor
tracing, and output/gradient residency on `flagos`. The installed MetaX Triton
path passed the same applicable tests; only the FlagTree-identity test is skipped
outside a FlagTree environment.

## Optional: FlagGems on MetaX

The MetaX boxing wheel compiles FlagGems Python-dispatch kernels by default, so FlagGems is a runtime switch. Enabling it requires two additional target-side dependencies:

```bash
# On the target MetaX machine, in addition to torch+cpu and torch_fl:
pip install triton-metax flag_gems
```

`triton-metax` emits `mcfatbin` for MetaX GPUs; `flag_gems` provides the Triton kernel library.

### Runtime Configuration

```bash
export FLAGOS_METAX_BOXING=1
export FLAGOS_USE_FLAGGEMS=1  # Opt into FlagGems; unset = pure boxing
```

`import torch_fl` then auto-selects `backends_metax_flaggems.conf`, which routes most ops to FlagGems' Triton kernels and falls back to the CUDA boxing kernel for ops `triton-metax` cannot compile (`mm`/`bmm`/`mean.dim` — FlagGems uses a SPLIT_K kwarg or CUDA-context path `triton-metax` rejects).

### FlagGems Verification

```bash
export FLAGOS_METAX_BOXING=1
export FLAGOS_USE_FLAGGEMS=1
python -c "
import torch_fl, torch
x = torch.randn(1024, device='flagos:0')
result = torch.nn.functional.silu(x).sum()
print(f'FlagGems SILU result: {result.cpu().item():.4f}')
"
```

Without `triton-metax`/`flag_gems` installed, leave `FLAGOS_USE_FLAGGEMS` unset — the pure boxing path has no extra dependencies.

## Distributed (Experimental)

Multi-GPU distributed training routes through the NCCL-shaped `mccl` fallback. The routing exists architecturally (see `_VENDOR_PROFILES["metax"]` in `torch_fl/comm/process_group.py`), but collective-level validation is not covered in CI. Manual verification on MetaX hardware is required.

## Model Inference and Training (Manual)

MetaX carries FSDP2 and Qwen3 training parity work in repository history, but these tests are not part of the CI manifest. Model-level validation remains a manual exercise on MetaX hardware.

## Troubleshooting

### Import Error: `undefined symbol` from libtorch

**Cause:** Import order violation — `import torch` occurred before `import torch_fl`, so the cudart shim was not loaded.

**Fix:** Always `import torch_fl` first in your scripts.

### Runtime Error: `device count mismatch`

**Cause:** `torch.cuda.device_count()` and `torch.flagos.device_count()` differ, indicating the MACA runtime or driver is not correctly initialized.

**Fix:** Verify `/opt/maca` is present and accessible. Check `LD_LIBRARY_PATH` does not override MACA runtime paths.

### `FLAGOS_METAX_BOXING=1` not set

**Symptom:** `import torch_fl` fails with configuration errors or missing libraries.

**Fix:** Export `FLAGOS_METAX_BOXING=1` before running Python. This variable gates the MetaX-specific import-time setup.

### Wheel Too Large for PyPI

**Expected behavior.** The bundled forked libtorch causes the wheel to exceed PyPI's 100 MB limit (~1.1 GB). Distribute via a private package index or direct file transfer.

### FlagGems: `triton-metax` not found

**Cause:** `triton-metax` is not available on PyPI; it must come from a vendor-specific index.

**Fix:** Obtain `triton-metax` from the MetaX developer portal or your vendor contact. If unavailable, disable FlagGems (unset `FLAGOS_USE_FLAGGEMS`) and use the pure boxing path.

## Further Reading

- [Environment Variables](../../reference/environment-variables.md) — Complete runtime configuration reference
- [Testing Guide](../../development/testing.md) — Running local validation equivalent to CI
- [Distributed (FlagCX)](../../architecture/distributed-flagcx.md) — Multi-GPU communication architecture (NCCL-shaped fallback for MetaX)
