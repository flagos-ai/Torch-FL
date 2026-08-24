# Ascend Installation Guide

Ascend NPU uses native CANN ACLNN operator kernels, not CUDA boxing. The platform supports two operator execution paths: a default pure ACLNN backend and an optional FlagGems route via triton-ascend.

## Prerequisites

- **CPU PyTorch 2.10.x**: `torch==2.10.0` from the upstream CPU index
- **CANN toolkit**: Ascend 910 with CANN 9.0.0 or compatible version
- **Python**: 3.8 or later
- **Operating System**: Linux (aarch64 verified in CI; x86_64 on real hardware)
- **Device node**: `/dev/davinci_manager` and `/dev/davinci*` devices must be accessible

## Installation

### 1. Install CPU PyTorch

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Source the CANN toolkit environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Verify that `ASCEND_HOME` points to a directory containing `lib64/` and `include/`:

```bash
ls $ASCEND_HOME/lib64/libascendcl.so
ls $ASCEND_HOME/include/aclnn_base.h
```

### 3. Build and install torch_fl

The default configuration uses native ACLNN kernels without FlagGems:

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

ACCELERATOR=ascend pip install --no-build-isolation -v -e .
```

Build flags:
- `ACCELERATOR=ascend`: selects the Ascend build path and native ACLNN kernel backend
- `ASCEND_KERNEL=1`: compiled automatically when `ACCELERATOR=ascend` (default ON)
- `CUDA_KERNEL=0`: automatically disabled for Ascend (no CUDA runtime exists)
- `--no-build-isolation`: ensures the build uses your installed CPU torch, not pip's overlay

The build runs `scripts/codegen_ascend.py` to generate operator kernels calling ACLNN APIs (`libopapi.so`) directly. Coverage is category-driven: unary, binary, reductions, and matmul families are generated; ops without an ACLNN mapping fall back to CPU.

## Verification

### Import order

**Critical**: always import `torch_fl` before other packages that might register device backends:

```python
import torch_fl  # Must come first
import torch
```

On an Ascend NPU box (detected via `/dev/davinci*`), `torch_fl` auto-selects `backends_ascend.conf` with no environment variable needed.

### Check device availability

```python
import torch_fl
import torch

print(f"flagos available: {torch_fl.flagos.is_available()}")
print(f"flagos devices: {torch_fl.flagos.device_count()}")

x = torch.randn(64, 64, device="flagos:0")
y = torch.abs(x)
print(f"abs matches CPU: {torch.allclose(y.cpu(), x.cpu().abs())}")
```

Expected output:
```
flagos available: True
flagos devices: 1  (or more, depending on your NPU count)
abs matches CPU: True
```

## Testing

Run the Ascend operator suite:

```bash
pytest tests/integration/ops/ -m "ascend" -v -s --tb=short
```

Run the RNG dispatch suite:

```bash
pytest tests/integration/ops/test_rng_dispatch.py -m "main_ops" -v -s --tb=short
```

Run general factory operator tests:

```bash
pytest tests/integration/test_factory_ops.py -v -s --tb=short
```

All three test groups are exercised in CI (see `.github/configs/ascend.yml` lines 78-97).

## Optional: FlagGems via triton-ascend

FlagGems provides Triton-compiled kernels as an alternative execution path. This is **optional** and **experimental** on Ascend; the native ACLNN backend is the default and recommended path.

### Why FlagGems requires a fork

The upstream FlagGems package links against `libtorch_npu.so` (from the `torch_npu` vendor package), which occupies the same `PrivateUse1` dispatch key as `torch_fl` and cannot coexist. Our fork replaces that dependency with a `FLAGOS` backend that obtains the ACL stream via `torch_fl`'s `GetCurrentStream` C API.

See [`docs/vendors/ascend/external-libtorch-npu.md`](external-libtorch-npu.md) for the technical analysis proving why `torch_npu` cannot act as a compatibility fallback.

### Install FlagGems (torch_fl branch)

```bash
git clone -b torch_fl https://github.com/Hchnr/FlagGems.git
cd FlagGems

source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install --no-build-isolation -e . \
  --config-settings=cmake.define.FLAGGEMS_BACKEND=FLAGOS \
  --config-settings=cmake.define.FLAGGEMS_BUILD_C_EXTENSIONS=OFF

cd ..
```

The `FLAGOS` backend skips the C++ extension build (`liboperators.so`) and provides only the Python dispatch path.

### Rebuild torch_fl with FlagGems Python wrappers

```bash
ACCELERATOR=ascend FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 \
  CUDA_KERNEL=0 ASCEND_KERNEL=1 \
  pip install --no-build-isolation -v -e .
```

Build flags:
- `FLAGGEMS_PYTHON=1`: enables Python-dispatch wrappers for FlagGems Triton kernels
- `FLAGGEMS_KERNEL=0`: disables C++ kernel wrappers (the FLAGOS backend does not build `liboperators.so`)
- `ASCEND_KERNEL=1`: keeps the native ACLNN backend for ops FlagGems cannot compile

### Patch triton-ascend

The stock `triton-ascend` package depends on `torch_npu`. Patch it to use the `flagos` device interface instead:

```bash
python scripts/patch_triton_ascend.py
```

The script is idempotent. After patching, clear stale kernel cache:

```bash
rm -rf ~/.triton/cache/
```

### Runtime libstdc++ compatibility

FlagGems pulls in `sqlalchemy`, which requires `CXXABI_1.3.15`. If your system `libstdc++.so.6` is older, preload conda's:

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

### Enable FlagGems at runtime

```bash
FLAGOS_USE_FLAGGEMS=1 python -c "
import torch_fl, flag_gems, torch
x = torch.randn(64, 64, device='flagos:0')
print('abs matches CPU:', torch.allclose(torch.abs(x).cpu(), x.cpu().abs()))
"
```

With `FLAGOS_USE_FLAGGEMS=1`, the runtime loads `backends_ascend_flagos_py.conf` instead of `backends_ascend.conf`. Ops that triton-ascend cannot compile are routed back to the ACLNN kernel (annotated per-op in the config).

Without `FLAGOS_USE_FLAGGEMS`, all ops use the pure ACLNN backend.

## Optional: torch.compile via triton-ascend

`torch.compile(backend="flagos")` compiles inductor's fused Triton kernels with
`triton-ascend`. It needs the same environment as the FlagGems route above — the
patched `triton-ascend` and its `libstdc++` preload — but not `FLAGOS_USE_FLAGGEMS`
and not `FLAGOS_USE_FLAGTREE`; the profile is picked from the `ACCELERATOR=ascend`
build. Eager ACLNN keeps working unchanged if `triton-ascend` is absent.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
import torch, torch_fl

def f(x):
    return torch.nn.functional.relu(x + 1.0) * 2.0

x = torch.randn(64, 64, device='flagos:0')
print('compile matches eager:', torch.allclose(torch.compile(f, backend='flagos')(x), f(x)))
"
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` matters if `torch_npu` is installed in the same
environment: `import torch` autoloads it, it claims PrivateUse1, and `torch_fl`
then refuses to register `flagos`.

Support is **experimental**. Measured on a real 910 (`Ascend910_9382`, CANN 9.0.0,
triton-ascend 3.2.0, torch 2.10.0+cpu, Python 3.10):
`tests/integration/test_compile.py` passes 29 of 30 with a cold inductor cache,
the remaining case being FlagTree-only. Run it with the cache cleared —

```bash
rm -rf /tmp/torchinductor_root
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest tests/integration/test_compile.py -v
```

— because a warm cache hides the compile-worker crash described below.

Three triton-ascend defects are worked around in
`torch_fl/compile/`, each documented with its measured evidence in
[`docs/architecture/torch-compile-integration.md`](../../architecture/torch-compile-integration.md):
a masked 2-D byte load that silently reads wrong data (this produced incorrect
`relu` gradients), `ub overflow` raised as a hard compile error rather than a
resource limit, and a segfault when the parent process launches a kernel built in
an inductor compile worker. The last one makes Ascend default to
`compile_threads=1`; an explicit `compile_threads` option or
`TORCHINDUCTOR_COMPILE_THREADS` still wins.

## Limitations

### No model mount in CI

`.github/configs/ascend.yml` (line 110) explicitly defers Qwen3 inference/training smoke tests pending a model mount on the CI runner. Point measurements outside CI show training at 0.82x `torch_npu` performance on real Ascend 910 hardware (`tests/perf/e2e_qwen3_train_ascend.py`).

### No device-side profiler parity yet

The Ascend profiler path does not yet emit device-side event categories (kernel, gpu_memcpy, gpu_memset, runtime, ac2g flows). The flagos trace has only `['Trace', 'cpu_op']` versus the torch-cuda baseline. `test_profiler_parity.py` is excluded from CI until device/runtime events are implemented (`.github/configs/ascend.yml` lines 112-117).

### torch.compile is not covered in CI

The Ascend CI runner image does not carry `triton-ascend`, so
`tests/integration/test_compile.py` has no CI step and the compile path is
validated only by hand on hardware that has the toolchain installed. The results
quoted above are point measurements, not a continuously enforced gate.

### Distributed support is architectural only

Distributed collectives route through FlagCX (recommended) or an HCCL fallback. The routing architecture is in place (see [`docs/architecture/distributed-flagcx.md`](../../architecture/distributed-flagcx.md) lines 204-208), but collective-level CI tests on Ascend hardware do not yet exist. The HCCL fallback and the `flagos→npu` zero-copy view are routing logic, not on-hardware-verified collectives (same document, lines 67-75).

## Reference Documentation

- [ACLNN operator codegen design](aclnn-codegen.md): category-driven code generation for native kernels
- [Ascend NPU integration plan](npu-plan.md): operator coverage strategy and acceptance criteria
- [External libtorch_npu analysis](external-libtorch-npu.md): why `torch_npu` cannot act as a boxing fallback
- [Compatibility matrix](../../reference/compatibility.md): platform status and capability claims
- [Testing guide](../../development/testing.md): running and interpreting test suites
- [Environment variables](../../reference/environment-variables.md): runtime environment variables and backend configs
