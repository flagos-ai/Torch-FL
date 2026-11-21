# MetaX Installation Guide

## Overview

The ordinary MetaX wheel uses FlagGems on the MACA runtime. The optional native-host wheel uses the same `backends_metax.conf` name with measured routes staged at build time.

**Status:** Stable. CI validates representative operator dispatch and factory/autograd tests on 8-GPU MetaX runners. FlagGems and model-level validation are not covered in CI.

## Prerequisites

### Build Host (Wheel Builder)

Requires the full MetaX SDK and a `torch+metax` wheel to extract the forked libtorch:

- MetaX MACA SDK (driver + cu-bridge), installed to `/opt/maca` or `$MACA_PATH`
- `torch+metax` wheel (`maca-pytorch`) from the MetaX developer portal (SoftNova)
- Python 3.8 or later matching the target deployment environment
- `patchelf` (`pip install patchelf`)

**Getting the MetaX SDK and torch+metax wheel:**  
Both are distributed through the MetaX developer portal: <https://developer.metax-tech.com/softnova>. Registration and login are required. Download the MACA SDK matching your card and driver version, and the `torch+metax` wheel built for the same MACA version and your Python version. Install the SDK to `/opt/maca` (or export `MACA_PATH` to the install location).

### Target Host (Deployment)

Requires only:

- Official `torch==2.10.0+cpu` from PyPI (no CUDA)
- The `torch_fl` wheel built below
- `/opt/maca` driver runtime (present on any machine with a MetaX card)

No separate `torch+metax` wheel, no manual `LD_LIBRARY_PATH` configuration.

## Installation

### Building the Wheel

#### Step 1: Build the FlagGems Artifacts

On a machine with the MetaX SDK and `torch+metax` wheel available:

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

# Build the ordinary MetaX wheel
FLAGOS_ACCELERATOR=metax \
  python setup.py bdist_wheel

```

**Parameters:**
- `FLAGOS_VENDOR_TORCH_LIB`: Path to the `torch+metax` wheel's `torch/lib` directory (source of forked libtorch)
- `FLAGOS_WHEEL_LOCAL`: Optional. The wheel's local version label defaults to the
  platform's SDK (`maca3.8.1.3` → `2.10.0+maca3.8.1.3`), which distinguishes it
  from the same platform built against another MACA — the metax lane carries
  both. Override it only for a dev build that must pin something else.

#### Step 2: Bundle the Forked Libtorch

```bash
FLAGOS_VENDOR_TORCH_LIB=<path-to-torch+metax>/torch/lib \
  MACA_PATH=/opt/maca \
  bash scripts/vendor/bundle_metax_libtorch.sh
```

This script:
- Copies 8 forked libtorch `.so` files from `torch+metax/torch/lib` into `torch_fl/lib_metax/`
- Rewrites RPATH with `patchelf` so libraries find each other via `$ORIGIN` and locate the MACA runtime at `/opt/maca/lib`

#### Step 3: Repackage the Wheel

```bash
python setup.py build_py
cp build/lib.*/torch_fl/_C.*.so build/lib.*/torch_fl/
python setup.py bdist_wheel --skip-build --bdist-dir "$(mktemp -d)"
```

The result is `dist/torch_fl-2.10.0+maca3.8.1.3-cp312-cp312-linux_x86_64.whl` (~1.1 GB — it bundles the forked libtorch and exceeds PyPI's 100 MB limit; distribute via private index or direct transfer).

### Installation on Target Host

#### Install Dependencies

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_fl-2.10.0+maca3.8.1.3-cp312-cp312-linux_x86_64.whl
```

#### Runtime Configuration

No environment variable is needed: a MetaX wheel records that it was built for
MetaX, and `import torch_fl` selects `backends_metax.conf` from that record. The
import-order rule below is what matters.

**Import order:** On MetaX, you **must** import `torch_fl` before `import torch`:

```python
import torch_fl  # Must import first
import torch
```

**Reason:** PyTorch's bundled CUDA 12.x runtime is ABI-incompatible with MetaX's cu-bridge (CUDA 11.6 compatibility layer). `torch_fl` preloads a shim library to provide the required symbol versions before torch initializes.

## Verification

### Basic Device Check

```bash
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
pytest tests/integration/test_factory_ops.py -v --tb=short
```

### Automatic Mixed Precision

Use the PyTorch AMP API with the `flagos` device name:

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
pytest tests/integration/test_amp_contract.py -m amp -v --tb=short
```

The measured C550/MACA 3.8.0 coverage includes FP16 and BF16 lower-precision,
FP32, optional-dtype, and promote autocast policies; nested state; non-finite
unscale; finite scale growth; overflow backoff; and a forward/backward optimizer
step. These measurements were taken on the former boxing wheel.

### Low-Precision Matrix Operations

The MetaX runtime provides software emulation for scalar FP8 and packed FP4 matrix
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
pytest tests/integration/ops/test_soft_lowp_gate_dispatch.py -m soft_lowp -v -s --tb=short
```

On the C550/MACA 3.8.0 environment used for this checkout, the complete marked suite
passed (`37 passed`), including all five FP8 formats, packed FP4, `mm`/`bmm`/`addmm`,
`dtype`/`out` variants, non-square packed layouts, in-place `addmm_`, and the fail-closed
scaled-mm check. Re-run the command above after changing the build or vendor runtime;
The recorded 37 passes came from the former boxing wheel.

The suite is measured with the tests' own 4x4 and `(2, 4, 4)` operands. A decoded BF16
pair is still handed to the route the conf names rather than computed in place, so an
operand that route cannot compile is reported instead of answered: FlagGems' MetaX `bmm`
requires `K >= 16`, and a `(2, 4, 4)` `bmm` reaches it whenever `bmm` is not routed to the
FlagGems C++ kernels (a wheel without `liboperators.so`, or `FLAGOS_BUILD_FLAGGEMS_CPP=0`).
The C++ kernel has the same minimum tile, so those operand sizes keep that limit either
way; they are outside what this path was measured on.

### torch.compile and FlagTree

The MetaX backend uses `torch.compile(backend="flagos")` with either the MetaX
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

A released MetaX FlagTree wheel is available from the FlagOS index
(`flagtree==0.6.1+metax3.6`; see *FlagGems on MetaX* below) — building from source
is what the revision-pinned evidence above used, and the two install the same
`triton` module.

Then run the compile contract with that environment's site-packages ahead of the
normal MetaX Triton installation:

```bash
PYTHONPATH=/path/to/flagtree-venv/lib/python3.12/site-packages \
  FLAGOS_USE_FLAGTREE=1 \
  pytest tests/integration/test_compile.py -v --tb=short
```

On C550 with MACA 3.8.0, the full suite passed with FlagTree revision
`140bd6ab1ad86c5df4b07b76d9c722e357a9166d` (Triton 3.6, MetaX backend),
covering forward, backward, FP32/FP16, max-autotune, recompilation, FakeTensor
tracing, and output/gradient residency on `flagos`. The installed MetaX Triton
path passed the same applicable tests; only the FlagTree-identity test is skipped
outside a FlagTree environment.

## FlagGems on MetaX

The ordinary wheel uses the FlagGems routes in `backends_metax.conf`.
An operator with no device route reaches `cpu_fallback`. The opt-in AOT wheel
stages its measured routes under the same config filename.

The `scaled_dot_product_attention` wrapper checks which inputs the FlagGems
kernel supports. Other inputs use the ATen composite on flagos tensors, and
the constituent operators follow their own routes.

FlagGems is a runtime dependency for the operators routed to it. A missing
FlagGems callable raises at dispatch.

Two target-side dependencies, neither of them on PyPI:

```bash
# On the target MetaX machine, in addition to torch+cpu and torch_fl:
# FlagTree's MetaX Triton build. It installs as the `triton` module and supplies
# the `metax` backend the generated kernels were measured against.
python3 -m pip uninstall -y triton  # repeat until fully uninstalled
pip install flagtree==0.6.1+metax3.6 \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple

# FlagGems, at the revision the checked-in kernels were generated from.
pip install "git+https://github.com/FlagOpen/FlagGems.git@5a58df410c551c4f4eb41d31887cd75fd596804a"
```

The FlagGems revision is load-bearing, not cosmetic. A generated kernel calls its operator by package-level name (`flag_gems.<name>`), resolved by `getattr` at dispatch time (`csrc/aten/backends/flagos/python_op_caller.cc:GetFunc`), so a cohort that does not define one of those names fails exactly the routes that use it. The revision above resolves all 667 names the checked-in `csrc/aten/generated/flaggems_python_kernels.cc` calls; `.github/scripts/set_env_metax.sh` measures that ratio before every integration job and refuses to run when it is not `0/667`.

That measurement has an import-order requirement of its own, because the setup venv runs the stock `torch+cpu` wheel: its `torch/lib` carries no `libtorch_cuda.so`, so `torch.cuda.is_available()` is `False` until `torch_fl` has selected the MetaX libtorch through a private facade. In that state the MetaX Triton backend reports itself inactive (`triton/backends/metax/driver.py:is_active`), and `flag_gems` reaches `triton.runtime.driver.active` while it is being imported (`flag_gems.fused` -> `pointwise_dynamic` -> `triton.runtime.jit.parse` -> the Triton hint manager's backend lookup), so a bare `import flag_gems` in the venv fails with `RuntimeError: 0 active drivers ([]). There should only be one.` — the FlagGems install is fine, the probe is simply running too early. The ratio is therefore taken in a process that imports `torch_fl` first, at the end of the setup script rather than beside the FlagGems install, since `torch_fl` is not importable until `setup.py build_ext` has run. The same rule applies to any ad-hoc check against the venv: `import torch_fl`
first, or the device surface is not there yet.

### Runtime Configuration

```bash
```

`FLAGOS_USE_FLAGGEMS` is not part of this: the retired switch selected a conf on no platform, and nothing reads it any more. Setting it does not change routing; `@pytest.mark.flaggems` cases run wherever they are collected.

### FlagGems Verification

```bash
FLAGOS_LOG=dispatch python -c "
import torch_fl, torch
x = torch.randn(1024, device='flagos:0')
result = torch.nn.functional.silu(x).sum()
print(f'FlagGems SILU result: {result.cpu().item():.4f}')
"
```

The dispatch log line the run prints for `silu` names the backend that served it (`[flagos dispatch] silu -> flagos_python`) — the conf's route, not an environment switch. `tests/integration/ops/backend_conf.py:routed_backend()` resolves that name from the conf so a test can assert it on any platform.

Install the pinned FlagTree and FlagGems wheels before using the MetaX operator routes.

## Distributed (Experimental)

Multi-GPU distributed training routes through the NCCL-shaped `mccl` fallback. The routing exists architecturally (see `_VENDOR_PROFILES["metax"]` in `torch_fl/comm/process_group.py`), but collective-level validation is not covered in CI. Manual verification on MetaX hardware is required.

## Model Inference and Training (Manual)

MetaX carries FSDP2 and Qwen3 training parity work in repository history, but these tests are not part of the CI manifest. Model-level validation remains a manual exercise on MetaX hardware.

## fp32 Matmul Precision (TF32)

An fp32 `mm`/`bmm` on MetaX computes in fp32 by default. The MetaX container
image used by this project's CI does not: it exports

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

ATen reads that variable while it initialises its TF32 state, so it is applied
before any user code runs and before the device is touched:

```python
torch.backends.cuda.matmul.allow_tf32      # True
torch.backends.cuda.matmul.fp32_precision  # 'tf32'
```

MACA's GEMM then takes its TF32 kernel for an fp32 product — profiling an fp32
`addmm` in that process names `mcblas__Mck_tf32gemm_tn_..._tf32_...` — which is
the documented meaning of the variable and is what PyTorch asked for. The vendor
kernel is faithful to that setting; `torch_fl` never changes it, and no routing
change recovers the bits, because the same setting is inherited by every other
vendor GEMM the process reaches (fp32 `scaled_dot_product_attention` on the
MetaX build is served from the vendor path and loses precision the same way).
This is a property of the deployment environment, not of the wheel.

The loss is ~900x, whatever the op and shape. Relative Frobenius error against an
fp64 CPU reference, 5 shapes × 20 seeds on a C550:

| computation | relative error |
| --- | --- |
| fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`) | 6.0e-08 .. 2.9e-07 |
| TF32 (image default) | 2.6e-04 .. 3.3e-04 |

Check it in one line:

```bash
python -c "
import torch_fl, torch
torch.manual_seed(0)
a, b = torch.randn(64, 128), torch.randn(128, 32)
got = torch.mm(a.to('flagos:0'), b.to('flagos:0')).cpu().double()
ref = torch.mm(a.double(), b.double())
print('relative error:', ((got - ref).norm() / ref.norm()).item())
# ~3e-07 is fp32, ~3e-04 is TF32
"
```

### Turning it off

Set the variable to `0` (or leave it unset) **before the process starts** — `0` is
the explicit-off spelling, and it is what cancels an inherited `1`. Unset and `0`
are numerically identical; a non-numeric value such as `false` or an empty string
raises a `UserWarning` and is treated as off.

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python your_script.py
```

For a process that cannot change its environment, the setting is reachable
in-process, and the legacy accessor is the one to use:

```python
torch.backends.cuda.matmul.allow_tf32 = False   # clean: no warning, both accessors stay readable
```

`torch.backends.cuda.matmul.fp32_precision = "ieee"` also takes effect, but it
moves the object to the new API alone, and reading `allow_tf32` afterwards then
raises `RuntimeError: ... mix of the legacy and new APIs to set the TF32 status
for cublas matmul`. Setting `allow_tf32` repairs that state, so it is the setter
to reach for if either API has already been touched.

`torch_fl`'s own CI pins `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` in
`.github/configs/metax.yml` (`integration_environment`), so the MetaX
integration job measures fp32 rather than TF32.
`tests/integration/test_dtype_coverage.py` holds the resulting precision to 1e-5
relative, which fails if that pin is dropped. MetaX is the platform those two
cases were measured on. Ascend is the one backend that cannot meet the bound and
does not claim to — its cube takes HF32 by request, the switch the
`MM_RTOL`/`MM_ATOL` comment in `tests/integration/test_ops.py` and issue #409 are
about — so they are xfailed there. No other platform was measured; what the
cases rest on for them is their own code, not a routing table: the FlagGems
`mm`/`bmm` pass `allow_tf32=False` to `tl.dot` explicitly, and MUSA's `mudnn`
handle is refreshed from `at::globalContext().allowTF32CuBLAS()` on every call.

## Troubleshooting

### Import Error: `undefined symbol` from libtorch

**Cause:** Import order violation — `import torch` occurred before `import torch_fl`, so the cudart shim was not loaded.

**Fix:** Always `import torch_fl` first in your scripts.

### Runtime Error: `device count mismatch`

**Cause:** `torch.cuda.device_count()` and `torch.flagos.device_count()` differ, indicating the MACA runtime or driver is not correctly initialized.

**Fix:** Verify `/opt/maca` is present and accessible. Check `LD_LIBRARY_PATH` does not override MACA runtime paths.

### `import torch_fl` fails with configuration errors or missing libraries

**Cause:** the MetaX-specific import-time setup (libtorch selection, `torch.cuda`
shim) did not run.

**Fix:** confirm the wheel was built for MetaX (`FLAGOS_ACCELERATOR=metax`, recorded at
build time) and that the vendor torch is reachable through
`FLAGOS_VENDOR_TORCH_LIB` or the bundled `lib_metax/`. No mode environment variable
is involved any more.

### Wheel Too Large for PyPI

Optional native precision builds can be larger when they include vendor
libraries. Use the MetaX package index for published wheels.

### FlagGems: `ModuleNotFoundError: No module named 'flag_gems'`

The FlagGems Python dispatcher resolves the selected callable from the installed
`flag_gems` package. Install the package and the MetaX Triton provider described
in [FlagGems on MetaX](#flaggems-on-metax).

## Further Reading

- [Environment Variables](../../reference/environment-variables.md) — Complete runtime configuration reference
- [Testing Guide](../../development/testing.md) — Running local validation equivalent to CI
- [Distributed (FlagCX)](../../architecture/distributed-flagcx.md) — Multi-GPU communication architecture (NCCL-shaped fallback for MetaX)
