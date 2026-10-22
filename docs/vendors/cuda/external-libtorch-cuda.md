# Measured: CPU torch + an external libtorch_cuda.so reuses CUDA operators

> Measured: 2026-07-16
> Machine: 2080ti (4× RTX 2080 Ti, driver 550.163.01)
> Verdict: **it works.** With only CPU-version torch installed via pip — no CUDA torch — loading
> a version-matched `libtorch_cuda.so` externally makes PyTorch's full set of registered CUDA
> kernels reusable, and the computed results are correct.

## Background and motivation

torch_fl (the `vm` / PrivateUse1 backend) wants to:

- Write **no kernels by hand** on NVIDIA, reusing PyTorch's already-optimized CUDA operators
  (via `device_boxing`: rewrite a `vm` tensor's metadata to CUDA, then call
  `structured_*_out_cuda`).
- Do so **without pip-installing CUDA torch** (large, drags in a pile of CUDA dependencies, and
  pins the Python environment to a specific CUDA version).

The core question: **can we pip-install CPU torch only, load a separate `libtorch_cuda.so` into
the process, and have the CUDA kernels register into the dispatcher for boxing to use?**

Earlier reasoning leaned toward "no", on the theory that the CPU wheel's `libtorch_cpu.so`
(built with `USE_CUDA=0`) might have CUDA-related symbols stripped and would fail to satisfy a
CUDA-built `libtorch_cuda.so`. **This measurement overturned that judgment.**

## Setting up the measurement

```bash
# 1. Clean conda env with CPU torch only
conda create -n libtorch_test python=3.12
pip install torch --index-url https://download.pytorch.org/whl/cpu
#   -> torch 2.13.0+cpu
#   torch/lib contains only libc10.so + libtorch_cpu.so; no CUDA .so at all
#   torch.cuda.is_available() == False

# 2. Download the exactly version-matched CUDA wheel (download only, do NOT install)
pip download torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126 -d /tmp/cuda_wheel --no-deps
#   unzip the wheel, extract libtorch_cuda.so (~1GB) + libc10_cuda.so

# 3. Install the CUDA runtime dependencies (standalone nvidia-* packages; they do not touch torch itself)
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
    nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
    nvidia-nvtx-cu12 nvidia-cuda-cupti-cu12 nvidia-cusparselt-cu12 \
    nvidia-nvjitlink-cu12 nvidia-cuda-cccl-cu12 nvidia-nvshmem-cu12
#   torch remains 2.13.0+cpu
```

> **Key point: the standalone libtorch release packages (download.pytorch.org/libtorch/...) 404
> on 2.13.0 and are unusable. Extracting the .so from a `pip download`ed CUDA wheel is more
> reliable** — it comes from the same pip build system as the CPU wheel, so ABI compatibility is
> highest and versions can be matched exactly.

## Results across four gates

Verification proceeded in dependency order through four gates, each of which passed:

| # | Gate | Result | Notes |
|---|---|---|---|
| 1 | **Symbol resolution** | ✅ no undefined symbols | `ctypes.CDLL(libtorch_cuda.so, RTLD_GLOBAL)` loads successfully. The CPU wheel's `libtorch_cpu.so`/`libc10.so` **fully satisfy** every symbol the CUDA-built `libtorch_cuda.so` needs — **no symbol stripping occurred** |
| 2 | **Kernel registration** | ✅ the CUDA key is fully populated after loading | Before loading, `mm/add/_softmax/bmm` all show `CUDA=False`; after, all `CUDA=True`. This shows the dispatcher is a global singleton in `libc10.so`: whoever loads the `.so`, the kernels register into that one table |
| 3 | **CUDAHooks / device init** | ✅ but **`LD_PRELOAD` is mandatory** | see "Hard constraints" below |
| 4 | **Real computation** | ✅ correct results | `mm max_err=9.5e-06` (normal fp32 GEMM precision), `add max_err=0.0`, `softmax rowsum=1.0`, genuinely executing on `cuda:0` |

### Verifying gates 1 and 2 (side-effect-free check)

```python
import ctypes, torch
from torch._C import _dispatch_dump_table  # simplified illustration
# before loading: aten::mm has CPU only
ctypes.CDLL(".../libc10_cuda.so", ctypes.RTLD_GLOBAL)
ctypes.CDLL(".../libtorch_cuda.so", ctypes.RTLD_GLOBAL)
# after loading: aten::mm / add / _softmax / bmm all show a CUDA implementation
```

Prerequisite: every `nvidia/*/lib` directory must be on `LD_LIBRARY_PATH` first (cudart, cublas,
cudnn, nvshmem, …), or loading fails for want of runtime libraries such as `libcudart.so.12` or
`libnvshmem_host.so.3`. (**Note: that class of failure is "missing CUDA runtime library", not a
symbol mismatch.**)

### Verifying gate 4 (real computation)

```python
# Key: libtorch_cuda.so must be loaded before import torch (see the constraints below)
a = torch.empty([N, K], device='cuda')  # factory path, constructed directly in C++; succeeds
# ... fill data, run mm/add/softmax ...
# mm  max_err = 9.5367431640625e-06
# add max_err = 0.0
# softmax rowsum mean = 1.0
```

## Hard constraints

### Constraint 1 (hard): libtorch_cuda.so must be loaded before `import torch`

- **Symptom**: calling `ctypes.CDLL(libtorch_cuda.so)` *after* `import torch` does register the
  kernels into the dispatcher (gate 2 passes), but device initialization raises
  `Cannot initialize CUDA without ATen_cuda library`.
- **Root cause**: PyTorch's **CUDAHooks mechanism**. `getCUDAHooks()` is first called during
  `import torch` and **caches** the stub hooks from `libtorch_cpu.so` (whose only job is to
  raise that error). The real hooks registered by a later-loaded `libtorch_cuda.so` cannot
  displace the cached stub.
- **Fix**: use `LD_PRELOAD` (or a build-time rpath, or `ctypes.CDLL` before `import torch`) so
  `libtorch_cuda.so` loads ahead of torch. Measured: with
  `LD_PRELOAD=".../libc10_cuda.so:.../libtorch_cuda.so"`, even the most demanding factory path,
  `torch.empty(device='cuda')`, returns a `cuda:0` tensor.

### Constraint 2 (no impact on torch_fl): the Python-level `torch.cuda._lazy_init` gate

- **Symptom**: high-level APIs such as `torch.randn(device='cuda')`, `a @ b`, `.to('cuda')`, and
  `.copy_()` explicitly call `torch.cuda._lazy_init()` and hit
  `AssertionError: Torch not compiled with CUDA enabled` in `torch/cuda/__init__.py`. This is a
  **Python-level compile-flag gate**, unrelated to whether the C++ dispatcher holds CUDA kernels.
- **No impact on torch_fl**: torch_fl's `vm` (PrivateUse1) + boxing path **never calls
  `torch.cuda.*`** — it allocates device memory through its own flagos allocator, and after
  boxing rewrites the metadata it calls `structured_*_out_cuda` directly in C++. The Python gate
  is therefore bypassed by construction.
- Reproducing this in pure Python (as done here) requires short-circuiting the gate to reach the
  real computation, but that is an artifact of the reproduction, not a real torch_fl constraint.

### Constraint 3: versions must match exactly

`libtorch_cuda.so` and the pip CPU torch must be **the same version** (e.g. `2.13.0` against
`2.13.0`; for nightlies, even the date must match). Mixing versions causes symbol or runtime
corruption from `at::Tensor`/ABI layout differences.

### Constraint 4: relies on CPU wheel symbol completeness (not officially guaranteed)

This approach depends on the property that the CPU wheel's `libtorch_cpu.so` provides every
symbol `libtorch_cuda.so` needs. **PyTorch makes no explicit guarantee of this.** It held on
2.13.0, but gates 1 and 2 should be re-run whenever the torch version is bumped.

## Integration notes for the `vm` backend (torch_fl)

1. **Preload timing**: `torch_fl/__init__.py` already preloads `libtorch.so` with
   `ctypes.CDLL(..., RTLD_GLOBAL)`. Add `libc10_cuda.so` + `libtorch_cuda.so` to that list and
   ensure it runs **before `import torch`** (or guarantee it via `LD_PRELOAD` / link-time rpath).
   This is the only hard constraint.
2. **The boxing path works**: `structured_*_out_cuda` and friends are registered and executable,
   so `device_boxing.h` (flagos-managed device memory + metadata rewrite + native kernel call)
   holds. **The boxing/structured reuse work from #15 needs no rollback.**
3. **CUDA runtime dependencies**: the `nvidia-*` pip packages must supply
   `libcudart/libcublas/libcudnn/libnvshmem` and friends, located via `LD_LIBRARY_PATH` or rpath.
4. **Do not touch the `torch.cuda` Python API**: keep boxing entirely in C++ so the `_lazy_init`
   gate is never triggered.

## What this buys, and what it costs

**Buys:**
- ✅ No pip CUDA torch; the Python side stays a clean `+cpu` environment
- ✅ Reuse of PyTorch's full set of optimized CUDA kernels — the `vm` backend needs **zero
  hand-written kernels and no cuBLAS/cuDNN glue**
- ✅ Keeps pace with the latest torch: bumping versions just means loading the matching
  `libtorch_cuda.so`

**Costs:**
- ⚠️ Depends on the unguaranteed "CPU wheel symbols are complete" property (re-test on upgrade;
  see constraint 4)
- ⚠️ `LD_PRELOAD` / preload timing is a hard constraint (constraint 1)
- ⚠️ `libtorch_cuda.so` is still in the process and still ABI-bound to that torch version (this
  is "can keep up with the latest", not "one binary across versions")

## One-line summary

> **Install CPU torch only via pip, load a version-matched `libtorch_cuda.so` externally before
> `import torch`, and the CUDA kernels register into the dispatcher and become reusable by the
> `vm`/boxing path — measured, with correct results.** The one hard constraint is load timing
> (the CUDAHooks caching problem, solved with `LD_PRELOAD`); the Python-level `torch.cuda` gate
> is irrelevant to torch_fl. This route gives the NVIDIA backend zero hand-written kernels with
> no dependency on pip CUDA torch.
