# Ascend Installation Guide

Ascend NPU uses native CANN ACLNN operator kernels, not CUDA boxing. The platform supports two operator execution paths: the FlagGems Python path (Triton kernels compiled by FlagTree) and the native ACLNN backend, which stays as the default fallback for every operator FlagGems does not cover.

## Prerequisites

- **CPU PyTorch 2.10.x**: `torch==2.10.0` from the upstream CPU index
- **CANN toolkit**: Ascend 910 with CANN 9.0.0 or compatible version
- **Python**: 3.11 (the FlagTree Ascend 3.5 wheel is cp311-only; Python 3.8+ works for an ACLNN-only build)
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

The default configuration enables the FlagGems Python route for measured operators,
with native ACLNN kernels as the fallback:

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

The build runs `scripts/codegen/codegen_ascend.py` to generate operator kernels calling ACLNN APIs (`libopapi.so`) directly. Coverage is category-driven: unary, binary, reductions, and matmul families are generated; ops without an ACLNN mapping fall back to CPU.

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

## FlagGems via FlagTree

FlagGems provides Triton-compiled kernels for the measured Ascend routes, with ACLNN remaining the native fallback. Ascend installations and CI enable this path by default; operators that FlagGems does not cover continue through ACLNN or CPU fallback.

The Triton build underneath is **FlagTree** — the FlagOS Triton fork — on its
Ascend 3.5 line (`flagtree 0.6.2a1+ascend3.5`, Triton 3.5). This replaced
`triton-ascend 3.2.x`, which lagged the Triton APIs current FlagGems uses, whose
task-queue launch path calls `at_npu::native::OpCommand` (a `torch_npu` symbol
this backend must not link), and whose removal from the environment is the whole
reason the route was re-measured. Nothing in this path imports or links
`torch_npu`.

### No torch_npu

The upstream FlagGems routing layer obtains the ACL stream through the device's
own interface, and FlagTree's Ascend backend resolves its host-side
implementation by importing `torch_npu` at discovery time. `torch_fl` handles
that without the real extension:

- It installs a lightweight stub under the `torch_npu` name, so the import
  succeeds instead of raising.
- It registers a third backend policy, `flagos`, on FlagTree's strategy registry
  (`torch_fl/compile/flagtree_ascend_policy.py`). That policy answers the same
  strategy names from torch_fl's own runtime: device and stream come from
  `torch.flagos` and the ACL stream registry, and the generated C++ uses plain
  ATen against PrivateUse1 (which under `torch_fl` *is* flagos) rather than
  `at_npu::`. It also forces `TRITON_ENABLE_TASKQUEUE=false`, because the task
  queue is torch_npu-only.

The real `torch_npu` extension cannot be used even as a fallback: it claims the
`PrivateUse1` dispatch key on import, the same key `flagos` needs, and the
coupling is at the C++ and link level rather than a Python import. See
[`docs/vendors/ascend/external-libtorch-npu.md`](external-libtorch-npu.md).

### Install FlagTree and FlagGems

Both come from the FlagOS index. FlagTree installs the module named `triton`, so
remove any stock or vendor Triton first — otherwise it is shadowed rather than
replaced. The FlagTree Ascend 3.5 wheel is published for cp311 only.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip uninstall -y triton
pip install --no-deps --index-url \
  https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  'flagtree===0.6.2a1+ascend3.5'

pip install --no-deps \
  'git+https://github.com/FlagOpen/FlagGems.git@d45285ba6423a3400019aa330daa6877908bf3cf'

pip install pybind11 packaging 'PyYAML==6.0.1' 'sqlalchemy==2.0.48' 'numpy<2'
```

FlagGems is pinned to a master commit validated against the FlagTree wheel above,
not to a release: an unpinned install is one upstream commit away from requiring
a Triton API the pin does not provide. `.github/scripts/set_env_ascend.sh` sets
both pins and is the authoritative recipe; `TORCH_FL_FLAGTREE_VERSION` and
`TORCH_FL_FLAGGEMS_REVISION` override them.

`numpy` stays below 2.x: 2.x breaks the stock `+cpu` torch C extensions at
import, and the Ascend test groups import both.

### Rebuild torch_fl

```bash
ACCELERATOR=ascend FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 \
  CUDA_KERNEL=0 ASCEND_KERNEL=1 \
  pip install --no-build-isolation -v -e .
```

Build flags:
- `FLAGGEMS_PYTHON=1` (the `ACCELERATOR=ascend` default): enables Python-dispatch wrappers for FlagGems Triton kernels
- `FLAGGEMS_KERNEL=0`: no C++ FlagGems kernels — nothing builds `liboperators.so` for this backend
- `ASCEND_KERNEL=1`: keeps the native ACLNN backend for ops FlagGems cannot compile or run

### Import order

`torch_fl` must be imported **before** `triton` or `flag_gems`. FlagTree's Ascend
backend picks its host-side implementation at discovery time, and only
torch_fl's stub plus backend policy make that import resolve without the real
`torch_npu`. Importing `flag_gems` first gets the torch_npu policy and the first
kernel dies on the stub.

```python
import torch_fl  # must precede triton / flag_gems
import torch
import flag_gems
```

To confirm the stack is correctly installed:

```bash
python -c "
import torch_fl, torch, triton, flag_gems
assert 'ascend' in triton.backends.backends
print(triton.__version__, flag_gems.__version__, flag_gems.vendor_name)
"
```

### Runtime libstdc++ compatibility

FlagGems pulls in `sqlalchemy`, which requires `CXXABI_1.3.15`. If your system `libstdc++.so.6` is older, preload conda's:

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

### Enable FlagGems at runtime

The generated Ascend configuration is FlagGems-first, so no runtime opt-in is
needed. Importing `torch_fl` initializes the default route; the following command
checks that the installed FlagGems package is available:

```bash
python -c "
import torch_fl, flag_gems, torch
x = torch.randn(64, 64, device='flagos:0')
print('abs matches CPU:', torch.allclose(torch.abs(x).cpu(), x.cpu().abs()))
"
```

There is no separate `*_flagos_py.conf`. An Ascend build is identified by its
`lib/flagos_platform` marker and always loads `backends_ascend.conf`, which is
generated FlagGems-first: every routable op is listed exactly once with its
resolved backend, ops FlagTree cannot compile or run sit on `ascend` (the ACLNN
kernel), and ops Ascend does not register at all are written `none` so they reach
`cpu_fallback`. Reading the file tells you the whole routing; the retired
`FLAGOS_USE_FLAGGEMS` switch is accepted nowhere any more, and the default
routes need no opt-in.

To measure the two backends against each other, collapse the table with
`ALL_USE_FLAGGEMS=1` or `ALL_USE_VENDOR=1` (mutually exclusive). Each only moves
an op when the target actually implements it; ops that cannot move are listed on
stderr and stay put, so `ALL_USE_VENDOR` is partial by nature.

### Runtime dtype fallback

A conf is a per-op routing table, so it cannot express "FlagGems, except for
float64". That exception is real on this compiler — BiShengHIR rejects the
float64 instantiation of nearly every pointwise kernel FlagGems emits — so it is
applied at runtime instead, by `FlagGemsRejectsDtype` in `csrc/aten/common.cc`
through `Dispatcher::ResolveFn`. A float64 call the conf routes to FlagGems, on an
op that also has an ACLNN kernel, lands on the native backend. Set
`FLAGOS_LOG_DISPATCH=1` to see which backend each call actually resolved to.
`tests/integration/ops/test_dtype_route_fallback.py` is the regression test.

## Optional: torch.compile via triton-ascend

`torch.compile(backend="flagos")` compiles inductor's fused Triton kernels with
the installed Triton build. The Ascend profile is picked from the
`ACCELERATOR=ascend` build and eager ACLNN keeps working unchanged.

**Status: not revalidated on FlagTree.** The measured result quoted below was
taken on `triton-ascend 3.2.0`; since Ascend's FlagGems route moved to FlagTree
`0.6.2a1+ascend3.5` no CI step or hand run has re-measured the compile path, so
treat it as unvalidated on the current toolchain. The path is also not covered by
CI at all — `tests/integration/test_compile.py` has no Ascend step in
`.github/configs/ascend.yml` — so these are point measurements either way.

The sections below still describe the triton-ascend environment, because that is
the toolchain the numbers came from. A hand-built `triton-ascend 3.2.x` is
installed the same way but from PyPI, and needs
`python scripts/vendor/patch_triton_ascend.py` to strip its `libtorch_npu`
linkage (see [`scripts/README.md`](../../../scripts/README.md)); the script is
kept for that path and is not used by the FlagTree route above.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "
import torch, torch_fl

def f(x):
    return torch.nn.functional.relu(x + 1.0) * 2.0

x = torch.randn(64, 64, device='flagos:0')
print('compile matches eager:', torch.allclose(torch.compile(f, backend='flagos')(x), f(x)))
"
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` is unnecessary in the FlagTree environment —
no `torch_npu` exists there — but matters if the real `torch_npu` is installed
alongside: `import torch` autoloads it, it claims PrivateUse1, and `torch_fl` then
refuses to register `flagos`.

Measured on a real 910 (`Ascend910_9382`, CANN 9.0.0,
triton-ascend 3.2.0, torch 2.10.0+cpu, Python 3.10):
`tests/integration/test_compile.py` passes 30 of 32 with a cold inductor cache,
the two remaining cases being FlagTree-only and MetaX-only. Run it with the cache
cleared —

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
`TORCHINDUCTOR_COMPILE_THREADS` still wins. Whether any of the three still apply
to FlagTree is unmeasured.

## Limitations

### No model mount in CI

`.github/configs/ascend.yml` (line 110) explicitly defers Qwen3 inference/training smoke tests pending a model mount on the CI runner. Point measurements outside CI show training at 0.82x `torch_npu` performance on real Ascend 910 hardware (`tests/perf/e2e_qwen3_train_ascend.py`).

### No device-side profiler parity yet

The Ascend profiler path does not yet emit device-side event categories (kernel, gpu_memcpy, gpu_memset, runtime, ac2g flows). The flagos trace has only `['Trace', 'cpu_op']` versus the torch-cuda baseline. `test_profiler_parity.py` is excluded from CI until device/runtime events are implemented (`.github/configs/ascend.yml` lines 112-117).

### torch.compile is not covered in CI

The Ascend CI runner image does not carry the Triton toolchain, so
`tests/integration/test_compile.py` has no CI step and the compile path is
validated only by hand on hardware that has it installed. The results
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
