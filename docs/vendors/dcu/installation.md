# Hygon DCU Installation Guide

## Overview

Hygon DCU (DTK) reuses the **CUDA boxing route** with a dedicated `ACCELERATOR=dcu` branch. Two properties of the vendor stack enable this:

- The DCU `torch` wheel is a **hipified** build: it registers HIP kernels under the `CUDA` dispatch key and its tensors report `DeviceType::CUDA` (`torch.version.cuda is None`, `torch.version.hip == '6.3.x'`). Generated PrivateUse1 → CUDA boxing kernels dispatch into `libtorch_hip.so` unchanged.
- DTK ships a **CUDA compatibility toolkit** at `$DTK_ROOT/cuda/cuda-*` whose `libcudart.so.12` is a thin shim over `libgalaxyhip.so` — the same runtime `libtorch_hip.so` uses. Runtime sources compile as-is with plain host `g++`; no `nvcc`, no `hipcc`, no hipify pass.

The build is **pure boxing**: `CUDA_KERNEL`, `FLAGGEMS_KERNEL`, and `FLAGGEMS_PYTHON` are all forced off by default (DTK ships its own Triton, so the NVIDIA-targeted PyPI `triton` wheel is the wrong artifact).

**Status:** Beta. CI validates vendor-backend and FlagGems-runtime operator suites, general tests, and profiler parity on DCU runners. Inference and training smoke tests are deferred pending model mount and card-count confirmation.

## Prerequisites

- Hygon DCU hardware with driver installed
- DTK (Hygon Deep Learning Toolkit) installed at `/opt/dtk` (or `$DTK_ROOT`/`$ROCM_PATH`)
- DTK torch wheel (hipified, registers kernels under CUDA dispatch key) — its
  device libraries are bundled at build time; see [Vendor core libraries](#vendor-core-libraries)
- The **official** `torch` wheel of the same minor version installed in the build
  and runtime environment (the `+cpu` build is enough)
- Python 3.8 or later matching the DTK torch build
- `cmake >= 3.18`, `ninja`, `patchelf`

## Installation

### From Source

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

# Source DTK environment (exports ROCM_PATH and other variables)
source /opt/dtk/env.sh

# Build with ACCELERATOR=dcu (pure boxing, no FlagGems by default)
ACCELERATOR=dcu pip install --no-build-isolation -vvv -e .
```

`DTK_ROOT` resolves from `DTK_ROOT` → `ROCM_PATH` → `/opt/dtk`. Pass it explicitly if DTK lives elsewhere.

### Build Notes

- **`ACCELERATOR=dcu` forces boxing mode:** `CUDA_KERNEL=OFF`, `FLAGGEMS_KERNEL=OFF`, `FLAGGEMS_PYTHON=OFF` in `setup.py`. The generated PrivateUse1 → CUDA boxing kernels (`csrc/aten/generated/cuda_kernels.cc`) are the only kernel set compiled.
- **No `nvcc` or `hipcc` needed:** CUDA runtime sources compile with plain `g++` using DTK's CUDA compatibility toolkit headers.
- **MIOpen CMake config fix:** DTK's exported MIOpen config bakes in `/usr/lib/x86_64-linux-gnu/librt.so`, which no longer exists on glibc ≥ 2.34 (librt was folded into libc). The `ACCELERATOR=dcu` branch rewrites that dangling path to `-lrt`.

## Vendor core libraries

The DCU wheel bundles DTK's **device** libraries (`libtorch_hip.so`,
`libc10_hip.so`, `libmagma.so`) and runs them on the **official** PyTorch core.
`torch_fl` dlopens them before `import torch`; your torch installation is not
modified. Operator coverage is unchanged — the HIP kernels still register under
the `CUDA` dispatch key.

Two consequences worth knowing:

- The installed torch must match the minor version the bundle was built against.
  A mismatch is rejected at import with the required version named, because
  `dlopen` accepts it silently (the symbol names match) and the failure would
  otherwise appear as a wrong-looking result much later.
- Import order matters: `import torch_fl` **before** `torch` in a fresh process.
  PyTorch caches its CUDA hooks on first import, so a later preload registers
  kernels that device init cannot reach.

To build the bundle:

```bash
ACCELERATOR=dcu python setup.py build_ext --inplace   # builds the ABI shim
bash scripts/bundle_dcu_libtorch.sh                    # stages DTK's device libs
```

`FLAGOS_DCU_VENDOR_CORE=1` on both commands selects the legacy mode, which
bundles DTK's full core set and symlinks it over the installed torch wheel
(reversible via `torch/lib/_orig_backup/`). Use it if you need DTK's private
fused ops, which are unreachable in the default mode. The two bundle layouts are
not interchangeable: selecting a mode that does not match the bundle on disk
fails at import rather than half-wiring the process.

Full measurements and rationale: [vendor-free-core-libs.md](vendor-free-core-libs.md).

## Verification

### Basic Device Check

```bash
python -c "
import torch, torch_fl

print(f'PyTorch version: {torch.__version__}')
print(f'torch.version.hip: {torch.version.hip}')
print(f'flagos devices: {torch.flagos.device_count()}')

# Basic computation
x = torch.randn(512, 512, device='flagos')
result = torch.mm(x, x)
print(f'mm output shape: {result.shape}')
print(f'mm matches .cuda(): {torch.allclose(result.cpu(), torch.mm(x.cpu().cuda(), x.cpu().cuda()).cpu())}')
"
```

Expected output shows `torch.version.hip: 6.3.x`, `flagos devices: N`, and `mm matches .cuda(): True`.

### Automatic Mixed Precision

DCU supports PyTorch's device-generic autocast and gradient scaling APIs with FP16 and BF16:

```python
import torch
import torch_fl  # noqa: F401

model = torch.nn.Linear(8, 4).to("flagos")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("flagos")
x = torch.randn(2, 8, device="flagos")

with torch.autocast("flagos", dtype=torch.float16):
    loss = model(x).square().mean()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Run the DCU AMP coverage, including both autocast dtypes and finite/non-finite GradScaler paths:

```bash
pytest tests/integration/test_amp_contract.py -m amp -v
```

### `torch.compile` with FlagTree

DCU `torch.compile` is validated with FlagTree's HCU backend on the Hygon
`gfx936` target. FlagTree replaces the active `triton` module at installation
time, so build it in a separate environment or install location rather than
trying to import a `flagtree` module at runtime:

```bash
git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
export FLAGTREE_BACKEND=hcu
MAX_JOBS=16 python -m pip install . --no-build-isolation -v
```

Use the resulting FlagTree environment together with the PyTorch 2.10 and
torch-fl installation. These variables select the HCU backend and assert that
the active Triton is FlagTree:

```bash
export FLAGTREE_BACKEND=hcu
export FLAGOS_USE_FLAGTREE=1
export TRITON_BACKENDS_IN_TREE=1
export GEMS_VENDOR=hygon

python -c 'import triton; from triton._flagtree_backend import FLAGTREE_BACKEND; print(triton.__version__, FLAGTREE_BACKEND)'
pytest tests/integration/test_compile.py -v
```

The measured Hygon run used FlagTree 0.6.0, PyTorch 2.10.0, and produced
`GPUTarget(backend='hip', arch='gfx936', warp_size=64)`. All 15 compile tests
passed. Compiled outputs and backward gradients remained on `flagos`; eager
and compiled results matched. The DCU CI manifest does not include this check
because its pinned image currently supplies DTK Triton rather than a FlagTree
HCU build.

### Operator Tests (Vendor Backend)

Run the main operator suite against the DCU boxing backend:

```bash
pytest tests/unit tests/integration/test_allocator.py \
  tests/integration/test_factory_ops.py -q

pytest tests/integration/ops/ \
  -m "main_ops and not flaggems and not flaggems_python" \
  -v --tb=short
```

### Profiler Parity

Verify profiler parity with torch.cuda:

```bash
pytest tests/integration/test_profiler_parity.py \
  -v -m main_ops --tb=short
```

See [Profiler Architecture](../../architecture/profiler.md) for details on device event emission for DCU.

## Enabling FlagGems on DCU

DTK ships its own Triton (the `hcu` backend) and a FlagGems build whose `hygon` vendor declares `device_name="cuda"`, which is exactly what the boxing route expects. Both live in the DTK system interpreter, so point your environment at them rather than installing PyPI wheels:

### Step 1: Expose DTK's Triton and FlagGems

```bash
pip install pyyaml sqlalchemy  # flag_gems imports these

mkdir -p <gems-path> && cd <gems-path>
ln -s /usr/local/lib/python3.10/dist-packages/triton .
ln -s /usr/local/lib/python3.10/dist-packages/flag_gems .
```

Adjust the Python path (`python3.10`) to match your DTK system interpreter.

### Step 2: Build with FlagGems Python Dispatch

```bash
source /opt/dtk/env.sh

ACCELERATOR=dcu \
  FLAGGEMS_PYTHON=1 \
  pip install --no-build-isolation -e .
```

### Step 3: Runtime Configuration

```bash
export PYTHONPATH=<gems-path>
export TRITON_BACKENDS_IN_TREE=1  # DTK install has no dist-info;
                                  # entry-point backend discovery finds nothing
export FLAGOS_USE_FLAGGEMS=1
```

`GEMS_VENDOR=hygon` is set automatically on a DCU build, so you no longer need to export it manually. This matters beyond FlagGems: `GEMS_VENDOR` also selects the comm profile (see `torch_fl/comm/process_group.py`), and DCU is a CUDA-ABI vendor whose `ProcessGroupNCCL` is RCCL underneath.

A DCU build records `ACCELERATOR=dcu` in `torch_fl/_build_config.py`, so `FLAGOS_USE_FLAGGEMS=1` alone selects `backends_dcu_flaggems.conf` — no need to re-export `ACCELERATOR` at runtime.

### FlagGems Configuration

`backends_dcu_flaggems.conf` is `backends_flaggems.conf` with ops `hcu` Triton cannot compile or run routed back to the cuda boxing kernel:

- `silu_backward`: `tl.math.div_rn` has no `create_precise_divf` lowering
- `slice_backward`: output is correct standalone, but feeding that grad to MIOpen's `convolution_backward` triggers a hardware VMFault

Override per-op routing with `FLAGOS_OP_<name>=flagos_python|cuda`.

### FlagGems Verification

```bash
export PYTHONPATH=<gems-path>
export TRITON_BACKENDS_IN_TREE=1
export FLAGOS_USE_FLAGGEMS=1

pytest tests/integration/ops/ \
  -m "flaggems and main_ops" \
  -v --tb=short
```

No marker deselection needed — the FlagGems path is now enabled.

## Multi-Card on DCU

Multi-card works with FlagGems enabled, over FlagCX or the RCCL fallback. The automatic `GEMS_VENDOR=hygon` routes `ProcessGroupFlagOS` to the CUDA-ABI profile (zero-copy `_flagos_to_cuda_view` + `ProcessGroupNCCL`, which on DTK is RCCL: `dist.is_nccl_available() == True` and `torch.cuda.nccl.version()` reports `(2, 22, 3)`).

```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1  # RCCL warns when unset; affects
                                    # multi-card throughput and stability

# Collectives + DDP (FlagGems can be on or off)
python tests/manual/test_flagos_dist_live.py --world-size 2
```

**Prerequisite:** Factory ops honoring their `device` index is required. Every rank>0 worker builds tensors via a factory, and an output allocated on device 0 while the Triton kernel launches on device N is a cross-device write that faults the GPU. See `tests/integration/test_factory_device_index.py`.

## Implementation Notes

### Memory Pool

flagos tensors and boxed kernels' outputs share one pool. `dcu_memory.h` delegates caching to torch's own allocator through the device-generic registry (`c10::getDeviceAllocator(kCUDA)`) rather than the `c10::cuda::` namespace — the DCU wheel exports `c10::hip::HIPCachingAllocator` and has zero `c10::cuda` symbols, and `cuda_runtime.h` cannot share a translation unit with `hip/hip_runtime.h`. So `memory_allocated()` / `memory_reserved()` / `empty_cache()` report and act on real usage.

### `.cuda()` Autograd and torch_fl Cannot Share a Process

PyTorch's `register_privateuse1_backend` makes `at::getAccelerator()` return `PrivateUse1`, so the autograd engine finds no stream metadata for a pure-CUDA graph and asserts in `engine.cpp`. This is upstream PrivateUse1 behavior, identical on CUDA/MetaX/Ascend — flagos-device autograd is unaffected. Take `.cuda()` baselines in a separate process.

## Troubleshooting

### Import Error: `cannot open shared object file: libhydmi.so`

**Cause:** Hygon DMI device management library not found; `hyhal` driver stack not on `LD_LIBRARY_PATH`.

**Fix:** Ensure `/usr/local/hyhal/lib` (or `/opt/hyhal/lib` if symlinked) is on `LD_LIBRARY_PATH`. CI setup prepends it after probing the host driver mount.

### Build Error: `MIOpen: librt.so not found`

**Cause:** DTK's MIOpen CMake config references the no-longer-present `/usr/lib/x86_64-linux-gnu/librt.so`.

**Fix:** This is automatically rewritten by the `ACCELERATOR=dcu` branch in `CMakeLists.txt`. If you see this error, verify `ACCELERATOR=dcu` is set and you are using the latest repository code.

### Triton: `No backend registered for 'hcu'`

**Cause:** `TRITON_BACKENDS_IN_TREE=1` not set, so entry-point discovery finds nothing.

**Fix:** Export `TRITON_BACKENDS_IN_TREE=1` and ensure `PYTHONPATH` includes the directory with symlinked `triton` and `flag_gems`.

### FlagGems: `silu_backward` or `slice_backward` crashes

**Cause:** `hcu` Triton backend limitation or MIOpen interaction bug.

**Expected behavior.** `backends_dcu_flaggems.conf` already routes these ops to the cuda boxing kernel. If you see crashes, verify `FLAGOS_USE_FLAGGEMS=1` is set and the config is being selected.

### Multi-card: GPU VMFault or device-side hang

**Cause:** `HSA_FORCE_FINE_GRAIN_PCIE=1` not set, or factory ops not respecting device index.

**Fix:** Export `HSA_FORCE_FINE_GRAIN_PCIE=1` before running multi-card tests. Verify `tests/integration/test_factory_device_index.py` passes.

### `torch.flagos.device_count()` Returns 0

**Cause:** DCU driver not loaded, or `/dev/kfd` and `/dev/dri` device nodes not accessible.

**Fix:** Check vendor-provided driver diagnostics. If running in a container, ensure `--privileged` or device passthrough is configured.

## Further Reading

- [DCU without DTK's core libraries](vendor-free-core-libs.md) — measured symbol attribution, the 32-symbol ABI shim, and the preload order
- [Environment Variables](../../reference/environment-variables.md) — Complete runtime configuration reference
- [Profiler Architecture](../../architecture/profiler.md) — Device event emission for DCU
- [Distributed (FlagCX)](../../architecture/distributed-flagcx.md) — Multi-GPU collectives and RCCL fallback
- [Testing Guide](../../development/testing.md) — Running local validation equivalent to CI
