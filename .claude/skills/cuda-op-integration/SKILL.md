---
name: cuda-op-integration
description: >
  Adapt the torch_fl (PrivateUse1 "flagos") NVIDIA backend to a specific PyTorch
  version by generating all CUDA operators from schema instead of hand-writing
  kernels. Use this when porting the CUDA-boxing approach to a new torch version
  branch (e.g. 2.10/2.11/2.12), when `import torch_fl` crashes with "Mismatch in
  kernel C++ signatures", or when regenerating csrc/aten/generated/ after a torch
  bump. Covers: torchgen codegen, per-operator IListRef/ArrayRef signature
  matching, the external libtorch_cuda.so LD_PRELOAD scheme (CPU-only pip torch),
  and the build+test loop.
---

# CUDA Operator Integration (torch_fl flagos backend)

## What this achieves

The NVIDIA backend writes **zero hand-written CUDA kernels**. Instead it:

1. **Boxes** flagos (PrivateUse1) tensors to CUDA device metadata (no data copy —
   flagos and CUDA share GPU memory), calls PyTorch's own optimized CUDA kernel via
   the public `at::` API, then **unboxes** the result back to flagos.
2. **Generates** all boxing kernels + dispatcher glue + registration for every op in
   `torch_fl/backends_cuda.conf` from `native_functions.yaml` via `torchgen`.
3. Runs against an **external `libtorch_cuda.so`** that is `LD_PRELOAD`ed before
   `import torch` — so the pip environment stays CPU-only (`torch==X+cpu`), no pip
   CUDA torch, no nvcc at build time.

The whole backend compiles with **g++ only** (no nvcc) and links **only** against
`torch_cpu_library`. CUDA symbols (`at::add`, `at::cat`, …) resolve at runtime from
the preloaded `libtorch_cuda.so`.

## The three pillars (read these files first)

- `scripts/codegen/codegen_ops.py` — the generator. Reads the conf + torchgen's packaged
  `native_functions.yaml`, emits 4 files into `csrc/aten/generated/`:
  `ops.h` (typedefs + `DECLARE_DISPATCHER`), `ops.cc` (`ADD_IMPL_TO_DISPATCHER`),
  `cuda_kernels.cc` (boxing kernels + `REGISTER_IMPL_TO_DISPATCHER`), `register.inc`
  (wrapper fns + `m.impl()` lines, `#include`d twice by register.cc).
- `csrc/aten/device_boxing.h` — `DeviceBoxingGuard` / `TensorListBoxingGuard` /
  `MaterializeToTensorVec` / `Box`/`UnboxToFlagos`. The runtime mechanism.
- `scripts/vendor/with_cuda_libtorch.sh` — wraps any command with the LD_PRELOAD +
  LD_LIBRARY_PATH needed to inject the external CUDA libs. Test/run through this.
- `docs/vendors/cuda/external-libtorch-cuda.md` — full rationale + the 4 hard
  constraints. Read it once before adapting a new version.

## Procedure to adapt a new torch version (e.g. branch `2.12`)

Assume: CUDA fixed at **13.0** (cu130 wheels exist for 2.9–2.13); each version needs
its own conda env with `torch==<ver>+cpu` and a matching `libtorch_cuda.so`.

### Step 0 — Port codegen infra onto the target branch (if branched from `main`)

`main` has hand-written kernels and **no codegen infra**. Bring these over from
`2.13` (the reference implementation) with `git checkout 2.13 -- <paths>`:

```
scripts/codegen/codegen_ops.py
scripts/codegen/extract_name_map.py
csrc/aten/device_boxing.h
csrc/aten/dispatcher.h
csrc/aten/register.cc
scripts/vendor/with_cuda_libtorch.sh
docs/vendors/cuda/external-libtorch-cuda.md
```

Then **delete the hand-written kernels** that codegen replaces (they will collide at
registration): `csrc/aten/backends/cuda/*.{cu,cc}` and the per-op `csrc/aten/*.{cc,h}`
that have generated equivalents, plus `structured_ops.cc`. Keep the core runtime:
`empty.*`, `strided_ops.*`, `copy_*`, `set_ops.*`, `contiguous_ops.*`, `fallback.*`,
`common.*`, and everything under `runtime/`. Update `csrc/CMakeLists.txt` if needed
(the `generated/*.cc` are picked up by the existing `GLOB_RECURSE *.cc`).

Cross-check against `2.13`'s tree to see exactly which files survived:
`git diff --stat main 2.13 -- csrc/aten/`.

### Step 1 — Environment + external CUDA assets

```bash
# CPU-only torch of the target version
conda create -n libtorch_<ver> python=3.12 -y
conda activate libtorch_<ver>
pip install torch==<ver> --index-url https://download.pytorch.org/whl/cpu

# nvidia runtime libs (cu13 series) — provides libcudart/libcublas/libcudnn/…
pip install nvidia-cuda-runtime-cu13 nvidia-cublas-cu13 nvidia-cudnn-cu13 \
    nvidia-cuda-nvrtc-cu13 nvidia-cufft-cu13 nvidia-curand-cu13 \
    nvidia-cusolver-cu13 nvidia-cusparse-cu13 nvidia-nccl-cu13 \
    nvidia-nvtx-cu13 nvidia-cuda-cupti-cu13 nvidia-cusparselt-cu13 \
    nvidia-nvjitlink-cu13 nvidia-nvshmem-cu13

# version-matched libtorch_cuda.so (download wheel, DON'T install; extract .so)
pip download torch==<ver>+cu130 --index-url https://download.pytorch.org/whl/cu130 \
    -d /tmp/cuda_wheel_<ver> --no-deps
cd /tmp/cuda_wheel_<ver> && unzip -o torch-*.whl -d unpacked
mkdir -p <repo>/.libtorch_cuda_assets
cp unpacked/torch/lib/{libc10_cuda.so,libtorch_cuda.so,libtorch_cuda_linalg.so,\
libtorch_nvshmem.so,libcaffe2_nvrtc.so} <repo>/.libtorch_cuda_assets/
```

**Hard constraint (docs §约束1):** `libtorch_cuda.so` version must match `torch`
**bit-for-bit**. Mixed versions → ABI corruption / undefined symbols.

### Step 2 — Generate + build + test (the loop)

```bash
conda activate libtorch_<ver> && cd <repo>

# a) generate
python scripts/codegen/codegen_ops.py           # writes csrc/aten/generated/*

# b) build CPU-only (FlagGems OFF because it needs flag_gems; CUDA boxing ON)
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON \
  pip install -e . --no-build-isolation

# c) smoke test THROUGH the wrapper (LD_PRELOAD external libtorch_cuda.so)
FLAGOS_BACKEND_CONFIG=torch_fl/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh python -c "
import torch_fl, torch
a=torch.randn(4,4,device='flagos:0'); b=torch.randn(4,4,device='flagos:0')
print(torch.add(a,b).cpu()); print(torch.cat([a,b]).shape)
t=[torch.randn(3,device='flagos:0') for _ in range(2)]
torch._foreach_add_(t,[torch.ones(3,device='flagos:0')]*2); print('OK')"

# d) full op suite (deselect flaggems markers — that backend isn't built)
FLAGOS_BACKEND_CONFIG=torch_fl/backends_cuda.conf \
  bash scripts/vendor/with_cuda_libtorch.sh \
  pytest tests/integration/ops/ -q -m "not flaggems and not flaggems_python"
```

## Version-adaptation gotchas — THIS is the real work

Compilation succeeding means nothing; the failure mode is a **runtime crash at
`import torch_fl`** during kernel registration. The dispatcher verifies that the C++
signature you register matches what other dispatch keys registered for the same op.
Different torch versions disagree on TensorList spelling.

### Gotcha 1 — IListRef vs ArrayRef per operator (the big one)

PyTorch registers the *same* op under different dispatch keys with *different*
TensorList C++ types, and they are **inconsistent across operators**:

- `aten::cat` — the `Batched` key uses `c10::IListRef<Tensor>` → we must use IListRef.
- every `aten::_foreach_*` — the `CompositeExplicitAutograd` key uses
  `c10::ArrayRef<Tensor>` (`TensorList`) → we must use ArrayRef.

A single global `use_ilistref_for_tensor_lists` setting **cannot** satisfy both. The
generator therefore sets it **per operator** via `torchgen.local.parametrize(...)`
inside the op loop, driven by the `ARRAYREF_OPS` set at the top of
`scripts/codegen/codegen_ops.py`.

**When adapting a new version:** the *membership* of `ARRAYREF_OPS` can change. The
error message tells you exactly which way each op must go — read both kernels:

```
Mismatch in kernel C++ signatures
  operator: aten::_foreach_add_.Scalar(...)
  kernel 1: void (c10::ArrayRef<at::Tensor>, ...)   dispatch key: CompositeExplicitAutograd
  kernel 2: void (c10::IListRef<at::Tensor> const&, ...)  dispatch key: PrivateUse1  <- us
```

Here PyTorch wants **ArrayRef** (kernel 1) but we emitted IListRef (kernel 2) → add
that op to `ARRAYREF_OPS`. If it's the opposite (PyTorch wants IListRef, we gave
ArrayRef) → remove it. Regenerate, rebuild, re-import. Iterate until import is clean.
The crash is one-op-at-a-time, so loop: fix → regen → build → import → next op.

Confirm `torchgen.local.parametrize` still takes `use_ilistref_for_tensor_lists` on
the target version (it does on 2.9–2.13; verify on older):
`python -c "import inspect,torchgen.local as l; print(inspect.signature(l.parametrize))"`

### Gotcha 2 — factory ops that compute (arange) recurse infinitely

`arange` builds a tensor with `TensorOptions`. If the device defaults to PrivateUse1
and you call `at::arange(..., options)`, it **dispatches back into your own kernel →
infinite recursion → segfault** (not a signature error — a stack overflow at
runtime, e.g. in `test_cat_stack_position_ids`). Fix (already in `gen_factory`):
build on **CUDA** device so it hits the external kernel, then `UnboxToFlagos`. Pure
allocators (`zeros`/`scalar_tensor`/`new_ones`) avoid this by calling `at::empty` +
`.zero_()`/`.fill_()` — `at::empty` is our own real allocator, so no recursion.
Any *other* compute-factory added to the conf needs the same CUDA-device treatment.

### Gotcha 3 — LD_PRELOAD timing (CUDAHooks cache)

`libtorch_cuda.so` **must** load before `import torch`. Loading it after → kernels
register fine but device init throws "Cannot initialize CUDA without ATen_cuda
library" (torch caches stub CUDAHooks at first `import torch`). This is why testing
goes through `scripts/vendor/with_cuda_libtorch.sh` (LD_PRELOAD) and not a late
`ctypes.CDLL`. Never call `torch.cuda.*` from tests — the flagos boxing path stays in
C++ and sidesteps torch's Python `_lazy_init` gate.

### Gotcha 4 — test markers

Tests marked `@pytest.mark.flaggems` / `flaggems_python` route to the FlagGems
backend, which is **not built** in CPU-only mode → they'll error with "backend not
registered". Deselect with `-m "not flaggems and not flaggems_python"`. These are
environment-expected, not code bugs. (Also watch for mismarked tests — a `@cuda`
test that sets `FLAGOS_OP_*=flaggems` is a marker bug; fix the marker. `mm.out` /
`bmm.out` default to flagos/flaggems, so their `_out_flagos_default` dispatch-log
test should be `@pytest.mark.flaggems`, not `@pytest.mark.cuda`.)

### Gotcha 5 — CUDA caching allocator cold-start (external-libtorch only)

Under the external-libtorch scheme (CPU pip torch + preload), the FIRST CUDA op in a
fresh process may hit `Allocator not initialized for device` from
`CUDACachingAllocator.cpp` — because PyTorch normally primes the CUDA caching
allocator inside `torch.cuda._lazy_init()`, which this scheme never calls (that's the
whole point — stay out of the `torch.cuda` Python gate). It surfaces specifically on
**out-variant** ops (`mm.out`, `bmm.out`) forced to `cuda` as the very first CUDA op,
because they allocate into a caller-provided `out` before any functional op has warmed
the allocator. Non-out ops warm it as a side effect and then out-variants work.

Impact is narrow: only the `*_out_cuda_override` dispatch-log tests that spawn a fresh
subprocess doing *nothing but* the out-variant. In-process and all correctness tests
(including out-variant correctness) pass because something else already touched CUDA.
With **real** pip CUDA torch installed (as on the 2.13 reference env) `_lazy_init`
primes the allocator and these pass — so this is an environment artifact of the
external-libtorch approach, NOT a codegen defect. Options: (a) accept/xfail these 2
log-only tests under external-libtorch, or (b) prime once at import in the test
harness with a throwaway functional CUDA op. Do not try to hand-init PyTorch's CUDA
allocator — that reaches into `torch.cuda` internals the scheme deliberately avoids.

## Done criteria

- `python scripts/codegen/codegen_ops.py` emits 71 ops (or whatever the conf lists), no WARNINGs.
- Build succeeds with `FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON`.
- `import torch_fl` is clean (no signature mismatch, no segfault) through the wrapper.
- `pytest tests/integration/ops/ -m "not flaggems and not flaggems_python"` passes,
  modulo the 2 `*_out_cuda_override` log tests (Gotcha 5) under external-libtorch.
  Ascend tests skip; that's expected.
- `torch.__version__` stays `<ver>+cpu` throughout — no pip CUDA torch installed.

## Verified results

- **2.13** (reference, base env has real `2.13.0+cu130`): full suite green.
- **2.12.1** (external-libtorch, `torch==2.12.1+cpu` + cu130 `libtorch_cuda.so`):
  263 passed, 64 skipped (Ascend), 34 deselected (flaggems), 3 xpassed; the only
  2 failures are the Gotcha-5 allocator cold-start log tests. `ARRAYREF_OPS` needed
  no change from 2.13 → 2.12 (same dispatcher signature split). Reused base env's
  `nvidia/cu13/lib` runtime libs via symlink instead of reinstalling ~GBs of wheels.
