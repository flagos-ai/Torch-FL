# Measured: can a CPU torch + external libtorch_npu.so reuse Ascend operators?

> Measured: 2026-07-20
> Machine: Ascend 910 (8× 910, npu-smi 25.5.0, CANN 9.0.0, aarch64)
> Verdict: **no — this path cannot simply copy the CUDA approach.** torch_npu registers
> its operators under the **PrivateUse1** dispatch key, which **collides with the same key**
> used by torch_fl's `flagos` backend. It cannot act as a drop-in fallback the way
> `libtorch_cuda.so` does (which registers under the separate `CUDA` key).

## Background

The CUDA approach (see [../cuda/external-libtorch-cuda.md](../cuda/external-libtorch-cuda.md))
rests on one **fundamental precondition**:

- PyTorch's CUDA kernels register under a **dedicated `CUDA` dispatch key**.
- torch_fl's `vm`/`flagos` backend occupies **`PrivateUse1`**.
- The two keys do not overlap. Once the external `libtorch_cuda.so` has pushed the CUDA
  kernels into the dispatcher, the boxing path rewrites a `flagos` tensor's metadata to
  CUDA and calls `structured_*_out_cuda`. The layering is natural; nothing contends.

The question: can Ascend do the same — extract `libtorch_npu.so` from the torch_npu wheel,
load it externally, and let NPU kernels enter the dispatcher as a boxing fallback?

## What was measured

```bash
# 1. Clean conda env with CPU-only torch (version aligned with torch_npu)
conda create -n libtorch_npu_test python=3.10
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
#   torch/lib holds only libc10/libtorch/libtorch_cpu/libtorch_python/libshm/libtorch_global_deps

# 2. Download (do NOT install) the version-matched torch_npu wheel, extract the .so
pip download torch_npu==2.7.1 -d /tmp/npu_wheel --no-deps
#   torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl (22.6 MB)
#   the artifact that matters: torch_npu/lib/libtorch_npu.so (~51 MB), the libtorch_cuda.so analogue

# 3. CANN runtime
source "$ASCEND_HOME/set_env.sh"
```

## Key results

### Symbol resolution ✅ (it loads)

`libtorch_npu.so`'s `NEEDED` entries include `libtorch.so / libtorch_cpu.so / libc10.so /
libtorch_python.so` plus the CANN-side `libhccl / libascendcl / libge_runner / libgraph`.
With the CANN library paths added to `LD_LIBRARY_PATH`,
`ctypes.CDLL(libtorch_npu.so, RTLD_GLOBAL)` **succeeds** with no undefined symbols — same as
CUDA. The CPU wheel satisfies every symbol it needs.

> Note: the CUDA approach carries a hard constraint that the library must be `LD_PRELOAD`ed
> *before* `import torch` (the cached CUDAHooks stub problem). In the NPU measurement,
> loading *after* `import torch` still registered the kernels into the table (see below), but
> device initialization goes through `PrivateUse1HooksInterface` — and that hooks path
> collides with the hooks torch_fl registers itself.

### Kernel registration ❌ (same-key collision — the decisive difference)

```python
import ctypes, torch
def has(op, key): return torch._C._dispatch_has_kernel_for_dispatch_key(op, key)

# before loading
#   aten::mm         PrivateUse1=False  CPU=True
#   aten::add.Tensor PrivateUse1=False  CPU=True
ctypes.CDLL(".../libtorch_npu.so", ctypes.RTLD_GLOBAL)
# after loading
#   aten::mm         PrivateUse1=True     <- registered into PrivateUse1!
#   aten::add.Tensor PrivateUse1=True
```

`strings` over `libtorch_npu.so`: `PrivateUse1` appears 25 times, `CUDA` only 6. It also
carries `c10_npu::impl::rename_privateuse1_backend()`,
`at::RegisterPrivateUse1HooksInterface`, and `c10::register_privateuse1_backend(...)`.
**torch_npu's entire device/operator/hooks stack is built on PrivateUse1** — which matches the
community understanding that torch_npu is a PrivateUse1 out-of-tree backend.

And torch_fl, in `torch_fl/__init__.py`, does exactly:

```python
torch.utils.rename_privateuse1_backend("flagos")
torch._register_device_module("flagos", flagos)
```

**A single PrivateUse1 key can only be claimed by one backend.** Loading `libtorch_npu.so`
externally would:

1. Overwrite / race the NPU kernels into `PrivateUse1`, clobbering flagos's own PrivateUse1
   registrations and vice versa.
2. Collide on `register_privateuse1_backend` / `PrivateUse1HooksInterface` — PyTorch allows
   the PrivateUse1 backend name and its hooks to be registered only once.

In short: **the "separate keys, natural layering" precondition is absent**, so boxing's
"rewrite metadata → call the native kernel" model has nowhere to stand on NPU. The target key
is the one we already occupy.

## CUDA vs Ascend

| Dimension | CUDA (`libtorch_cuda.so`) | Ascend (`libtorch_npu.so`) |
|---|---|---|
| Kernel registration key | **`CUDA`** (dedicated) | **`PrivateUse1`** (collides with flagos) |
| Relation to flagos (PrivateUse1) | Orthogonal; boxing can layer | Same key; direct conflict |
| Extract .so from wheel and load | ✅ symbols complete | ✅ symbols complete |
| Device hooks | `CUDAHooks` (preload solves the caching) | `PrivateUse1HooksInterface` (conflicts with flagos hooks) |
| Does "external .so as fallback" work? | ✅ yes | ❌ no |

## Conclusion and recommendation

- **The CUDA approach cannot be copied directly.** CUDA works because the `CUDA` key and the
  `PrivateUse1` key layer naturally. torch_npu implemented itself as **another PrivateUse1
  backend**, contending with torch_fl for the same key.
- For Ascend, the existing route stands: hand-written / CANN-backed operators under
  `csrc/aten/backends/ascend/`, or FlagGems on FlagTree. `libtorch_npu.so` cannot be
  loaded externally as a zero-cost fallback layer.
- If reusing torch_npu's already-implemented NPU kernels is genuinely desirable, the mechanism
  is not "load the .so" but **explicitly forwarding to torch_npu's op implementations at the
  C++ level**, bypassing the dispatcher's single-PrivateUse1-key limit. That is a separate
  body of work, tightly coupled to a torch_npu version, and its cost/benefit needs its own
  evaluation.

## One-line summary

> `libtorch_npu.so` loads fine under CPU torch and every symbol resolves, but it registers NPU
> operators under **PrivateUse1** — precisely the key torch_fl's `flagos` already holds. The
> "separate key layering" precondition the CUDA approach depends on does not exist on Ascend,
> so **"extract the .so and get a fallback for free" does not work there**.
