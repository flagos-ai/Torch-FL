# Integrating TileOPs into torch_fl (flagos / PrivateUse1)

This document records the feasibility findings and the implementation plan for
integrating [TileOPs](https://github.com/tile-ai/TileOPs) -- a GPU operator
library built on [TileLang](https://github.com/tile-ai/tilelang) -- into
`torch_fl`. Every key conclusion below was verified on real H800 hardware
(torch 2.10.0+cu130, CUDA 13.0, SM90) rather than derived on paper; the
verification scripts and measured numbers are in section 2.

> Status: research and feasibility verification complete. The integration
> implementation is described in
> [`tileops_codegen_design.md`](./tileops_codegen_design.md) (PR1/PR2 landed,
> all 60 aten-aligned routes verified on hardware). Branch `h800-2.10`, based on
> `flagos/main` (3fa2057).

---

## 1. What TileOPs is, and how it differs from FlagGems

| | FlagGems | TileOPs |
|---|---|---|
| Kernel language | Triton | TileLang (TVM-FFI backend) |
| Operator entry point | `flag_gems.ops.<fn>` module-level functions | `tileops.ops.<XxxOp>` classes, must be instantiated |
| Signature alignment | Broadly matches the aten schema | **Does not match**; designed around workload semantics |
| Shape / dtype | Inferred at runtime | Fixed at **construction time** (ctor) for some ops |
| Hardware scope | Multi-vendor (nvidia/metax/ascend/...) | **Hopper SM_90 only** |
| Operator count | ~200 aten routes | 165 `implemented` (per manifest) |

TileOPs layers its code as `Op` (L2, the Python entry point, handling
validation, dtype and layout) plus `Kernel` (L1, the TileLang implementation),
and uses `tileops/manifest/*.yaml` as the single source of truth for specs
(signatures, workloads, roofline formulas). That manifest is directly valuable
to us: it is a **machine-readable operator inventory** that can drive route
table generation, so we do not need to walk `_FULL_CONFIG` by reflection the way
the FlagGems path does.

### 1.1 Key difference: TileOPs is not a drop-in aten backend

The FlagGems Python path can auto-discover and generate 2000+ routes because its
function signatures broadly line up with aten schemas. TileOPs cannot, for three
hard reasons:

1. **Operators are stateful objects.** You must do `op = GemmOp()` and then
   `op(a, b)`; they are not free functions.
2. **Some operators fix shape/dtype in the ctor.** For example
   `ReluFwdOp(N_total, dtype)`, `UnaryOp(N_total, dtype)`,
   `RMSNormFwdOp(normalized_shape, ...)`. One instance serves one shape; a
   different shape needs a different instance. `GemmOp` / `BmmFwdOp` are good
   citizens (the ctor takes no shapes and they cache internally by
   `(m,n,k,dtype)`).
3. **Semantics are not aten-equivalent.** `GemmOp` defaults to the NT layout
   (`a @ b.T`), not `aten::mm`'s NN; the `ref_api` field often points at
   `torch.nn.functional.*` rather than an aten operator. Each operator needs its
   adapter confirmed by hand.

So the integration shape **cannot** copy FlagGems' "full auto-discovery plus
2000 conf entries". It has to be **per-operator allowlisting with hand-written
adapters**.

---

## 2. Findings from hardware verification

Environment: H800 xN, torch 2.10.0+cu130, CUDA 13.0, SM90, Python 3.12.
`torch_fl` builds cleanly with `ACCELERATOR=cuda CUDA_KERNEL=1 FLAGGEMS_KERNEL=0
FLAGGEMS_PYTHON=0 python setup.py build_ext --inplace`.

### 2.1 The dependency stack must be pinned exactly (or import crashes)

Installing TileLang 0.1.11 into an isolated directory hits two traps, both
reproduced and resolved:

| Package | Required | Symptom with the wrong version |
|---|---|---|
| `apache-tvm-ffi` | `==0.1.11` | 0.1.12 aborts at import: `TypeAttr __ffi_repr__ is already registered for type index 130` (core dump) |
| `z3-solver` | `==4.15.4.0` | 5.x names its library differently: `OSError: libz3.so.4.15: cannot open shared object file` |

TileOPs' own `constraints.txt` already states that `apache-tvm-ffi` and the
tilelang wheel are ABI-coupled. With the versions pinned, a hand-written
TileLang fp16 GEMM and TileOPs' `GemmOp` / `SoftmaxFwdOp` all run and produce
correct numerics.

### 2.2 Feeding flagos tensors straight to TileOPs -- fails (as expected)

```
ROUTE A failed: RuntimeError kernel _gemm_main input a device_type mismatch,
                expected cuda; expected: 12, got: 2
```

The cause is on the TileLang side:
`jit/adapter/cython/adapter.py::_process_buffer_device()` hardcodes
`device = torch.device("cuda")` for cuda targets, and the subsequent
`_check_buffer_device` compares `device.type == tensor_device.type` and fails.
PrivateUse1 cannot get through, and there is no env switch to bypass it
(`skip_tensor_validation` only exists on the cython backend's internal API and
is not exposed by the default tvm_ffi backend).

### 2.3 Zero-copy boxing -- works, and the overhead is negligible

The repository already has the primitives exposed to Python
(`torch_fl/csrc/module.cc`): `_C._flagos_to_cuda_view(t)` /
`_C._cuda_to_flagos_view(t, idx)`. Both only rewrite the TensorImpl's device
metadata and hold the source tensor via `shared_ptr` to keep it alive; **no data
is copied**.

Measured:

```
boxed: cuda:0 torch.float16 (1024,512)  data_ptr same: True
ROUTE B ok: (1024,1024) flagos:0 fp16   max err 0.0312  (normal for fp16 accum)
box   (flagos->cuda view): 0.41 us
unbox (cuda->flagos view): 0.57 us
```

Registering `GemmOp` as PrivateUse1's `aten::mm` via `torch.library` makes both
`torch.mm(a,b)` and `a @ b` hit correctly (hits=2), with rel err 2.8e-4 (normal
for fp16).

### 2.4 Streams and the memory pool -- already consistent, no extra plumbing

Both concerns turned out to be non-issues in practice:

- **Streams**: flagos' `current_stream().cuda_stream == 0` and
  `torch._C._cuda_getCurrentRawStream(0) == 0`. `_cuda_compat.py`'s
  `_StreamShim` pins both sides to the null stream, so TileLang reads the same
  one.
- **Memory pool**: TileOPs allocates outputs with `torch.empty(device="cuda")`.
  An 8 MiB cuda allocation increases `flagos.memory_allocated` by exactly
  8388608 and drops back to zero on free -- meaning it lands in the **same**
  flagos caching allocator pool. Boxed outputs are therefore accounted for and
  safe to unbox, and no custom output allocator is needed.

### 2.5 Performance: the bottleneck is Python overhead in the L2 Op layer, not the kernel

All measurements below run on flagos tensors through zero-copy boxing, fp16,
H800/SM90, `TILELANG_DISABLE_CACHE=1`. `Op(L2)` is the normal `op(x)` entry
point; `Kernel` bypasses L2 and calls `op.kernel(x)` directly:

| Operator | Shape | Op(L2) | Kernel | torch | L2/torch | Kernel/torch | max err |
|---|---|---|---|---|---|---|---|
| `GemmOp` (NT) | 4096³ | 0.274 ms | 0.285 ms | 0.179 ms | 0.65x | 0.63x | 0 |
| `SoftmaxFwdOp` | 8192×4096 | 0.150 ms | 0.098 ms | 0.109 ms | 0.72x | **1.11x** | 3.8e-06 |
| `LogSoftmaxFwdOp` | 8192×4096 | 0.150 ms | 0.100 ms | 0.095 ms | 0.63x | 0.95x | 7.8e-03 |
| `LayerNormFwdOp` | 8192×4096 | 0.145 ms | 0.100 ms | 0.063 ms | 0.43x | 0.63x | 3.9e-03 |
| `RMSNormFwdOp` | 8192×4096 | 0.137 ms | -- | 0.061 ms | 0.44x | -- | 3.9e-03 |
| `SumFwdOp` | 8192×4096 | 0.142 ms | -- | 0.026 ms | 0.19x | -- | 9.8e-04 |
| `AmaxFwdOp` | 8192×4096 | 0.138 ms | -- | 0.029 ms | 0.21x | -- | 0 |
| `ReluFwdOp` | 4194304 | 0.096 ms | 0.030 ms | 0.018 ms | 0.19x | 0.60x | 0 |
| `SiluFwdOp` | 4194304 | 0.095 ms | 0.031 ms | 0.019 ms | 0.20x | 0.62x | 3.9e-03 |
| `LogSumExpFwdOp` | 8192×4096 | 0.043 ms | -- | 0.202 ms | **4.68x** | -- | 7.8e-03 |
| `CumsumFwdOp` | 8192×4096 | 0.140 ms | -- | 0.190 ms | **1.36x** | -- | 2.12 (!) |
| `BmmFwdOp` | 16×512³ | 0.038 ms | -- | 0.037 ms | 0.97x | -- | 0 |
| `ArgmaxFwdOp` | 8192×4096 | 7.826 ms | -- | 0.044 ms | **0.01x** (!) | -- | 0 |

Key observations:

1. **The L2 Op layer carries a fixed Python overhead of roughly 0.05-0.09 ms**,
   independent of input size. The same `SoftmaxFwdOp` instance takes a constant
   0.147-0.178 ms from 8×8 all the way to 16384×4096, whereas `torch.softmax`
   scales linearly from 0.012 ms to 0.212 ms. `GemmOp` behaves the same way:
   0.055 ms flat from 128³ to 2048³. This means that **at small and medium sizes
   we are measuring dispatch overhead exclusively** -- the kernel itself is
   nowhere near saturated.
2. **Bypass L2 and the kernels are actually competitive**: softmax at 0.098 ms
   beats torch's 0.109 ms (1.11x). In a separate measurement,
   `Op.forward` 0.129 ms vs `Op.__call__` 0.158 ms vs `kernel()` 0.068 ms --
   each layer eats tens of microseconds. `_cache_key()` alone measures in the
   7 ms range (slow paths such as `Tensor.__contains__`) and is the prime
   suspect.
3. **GEMM is genuinely slow** (0.65x), and L2 and kernel are equally slow -- this
   is not an overhead problem, the kernel simply loses to cuBLAS.
4. **`LogSumExpFwdOp` is the clear winner** (4.68x), because torch's logsumexp is
   itself a multi-kernel composition -- exactly the scenario a fused operator
   should win.
5. **Two red flags**: `ArgmaxFwdOp` is 100x slower (0.01x), suggesting a
   degenerate implementation; `CumsumFwdOp` reaches 1.36x but has max err 2.12,
   which is unacceptable fp16 accumulation precision. Neither can enter the
   allowlist.

Revised conclusion: **we still cannot replace the cuda routes wholesale**, but
the reason needs restating. It is not that "TileOPs kernels are generally slow",
it is that (a) L2 Python overhead consumes the entire benefit for
memory-bound operators, and (b) GEMM-class operators lose head-to-head against
vendor libraries. The value therefore still lies in **fused operators**
(logsumexp verified, plus rms_norm, sdpa variants, mamba/SSD, moe, lightning
indexer), and if the integration layer wants to cover memory-bound operators it
**must consider bypassing L2 and holding `op.kernel` directly**.

### 2.6 JIT cost is high, so instance caching is a hard requirement

Measured with `ReluFwdOp`, split by cold and warm disk cache (previously only the
warm column was recorded, which was easy to misread):

| | ctor | first call | warm | new instance, new shape |
|---|---|---|---|---|
| Cold cache (first compile) | 4169.8 ms | 897.4 ms | 0.060 ms | 4115.8 ms |
| Warm cache (second process, same `TILELANG_CACHE_DIR`) | 54.0 ms | 628.2 ms | 0.056 ms | -- |

Two conclusions:

- Warm calls are **4-5 orders of magnitude** faster than the first call, so Op
  instances **must** be cached by `(op_class, shape_key, dtype)` and must never
  be constructed inside a dispatch.
- With a cold cache the ctor alone costs 4 seconds (TileLang compilation happens
  in the ctor), dropping to 54 ms once warm. That means CI and first startup need
  a warmup pass, and the `TILELANG_DISABLE_CACHE=1` workaround in section 2.7 is
  expensive -- every process pays the compile cost again. Only after the upstream
  cache key is fixed can the disk cache be used for real.

One related correction: with the disk cache off, reusing an instance with a
mismatched N changes the error from
`RuntimeError: ... violates packed ABI constraint` to
`ValueError: Expected 4194304 elements, got 2097152` -- because with the cache
disabled the compiled kernel is correct, and what stops the call is TileOPs' own
argument validation rather than the mismatched artifact from 2.7.

### 2.7 Upstream bug: TileLang JIT frontend cache key collision (unrelated to flagos; fixed upstream)

This is the problem most in need of an upstream fix, and it **reproduces on pure
CUDA with flagos not involved at all**:

```
N=    1024 ok
N= 2097152 FAILED: kernel main input x shape[0] violates packed ABI constraint;
                   expected: 2097152, got: 1024
N= 4194304 FAILED: ... expected: 4194304, got: 1024
N=    1024 ok          <- correct again, showing the first artifact is being reused
```

**Root cause corrected (2026-07-31, re-checked online plus local instrumentation).**
The earlier claim that "identical script bytes produce a sha256 collision in
`cache/kernel_cache.py::_generate_key()`" is inaccurate; the collision is not in
the disk kernel cache layer:

- Instrumenting `KernelCache._generate_key` shows the three different values of N
  trigger only **one** call (the N=1024 one); the other two never reach that
  layer.
- Instrumenting `load_frontend_cached` gives the decisive evidence: N=2097152
  returns **HIT=True**, and both keys are identical --
  `((('npt_arg', 8), ('threads_arg', 256)), None)` -- **`N` is not in the key**.
  `tileops/kernels/elementwise.py` wraps `_make_unary_direct(N, dtype, ...)` in
  `lru_cache`, and the inner `@tilelang.jit def kernel(threads_arg)` captures `N`
  as a **closure free variable**, while `JITImpl._frontend_cache_key_data` only
  covers explicit parameters. The two `JITImpl` objects have distinct
  `_kernel_cache` objects, so the leak travels solely through the **frontend disk
  cache** layer.
- The upstream issue is exactly this:
  **[tilelang#2358](https://github.com/tile-ai/tilelang/issues/2358)
  "[JIT] Include closure free vars in frontend cache key"**, whose minimal repro
  matches ours; fixed by **#2363 "[JIT] Remove frontend disk cache"**,
  **closed 2026-06-09**.
- The timing missed by exactly one day: `tilelang==0.1.11` shipped on
  **2026-06-08**, the fix merged the next day, so our pinned version necessarily
  carries the bug. `0.1.12` (2026-07-08) should contain the fix (not yet verified
  locally).

- Blast radius: operators that fix their shape in the ctor (elementwise,
  softmax). `GemmOp` passes `(m,n,k)` as **explicit parameters** rather than
  closure variables and is therefore **unaffected** -- which confirms the root
  cause.
- The workaround has been upgraded: instead of a blanket
  `TILELANG_DISABLE_CACHE=1`, we make only the frontend cache's load/store
  no-ops (equivalent to #2363) and **keep the kernel disk cache**. Measured: all
  shapes correct, and a second process drops from 12.0s to 0.3s (14.2s to 1.2s
  through the full aten route). See
  `torch_fl/tileops_backend.py::_disable_frontend_cache()`, with escape hatch
  `FLAGOS_TILEOPS_DISABLE_ALL_CACHE=1`.
- Disposition: **no new issue needed**. Once the pin moves to 0.1.12 (both
  packages together, see 2.1) the workaround function can be deleted.

### 2.8 Operator coverage

The manifest has 184 entries: 165 `implemented` and 19 spec-only. Intersecting
the last segment of `ref_api` with the 1009 aten base names in
`backends_cuda.conf`:

- **84** aten operators have a direct same-name counterpart, including
  `mm` (via matmul), `bmm`, `add`, `mul`, `div`, `sub`, `pow`, `sum`, `mean`,
  `prod`, `amax`, `amin`, `argmax`, `argmin`, `all`, `any`, `cumsum`, `cumprod`,
  `softmax`, `log_softmax`, `logsumexp`, `var`, `std`, `var_mean`, `topk`,
  `where`, `masked_fill`, `clamp`, `gelu`, `silu`, `relu`, `sigmoid`, `tanh`,
  `elu`, `leaky_relu`, `hardswish`, `hardsigmoid`, `mish`, `softplus`,
  `avg_pool{1,2,3}d`, `bitwise_*`, `logical_*`.
- The rest are **fused or non-aten semantics**: `rms_norm`, `layer_norm`,
  `group_norm`, `instance_norm`, `batch_norm`,
  `scaled_dot_product_attention`, `conv{1,2,3}d`, `max_pool{1,2,3}d`, `dropout`,
  `fft`, `prelu`, `selu`, `vector_norm`, plus things aten has no equivalent for
  at all: moe, mamba, SSD, deltanet, GLA, NSA, engram, fp8_quant and the
  lightning indexer.

---

## 3. Integration plan

### 3.1 Positioning: add a fourth dispatch path, `kTileOps`

The existing `Backend` enum in `csrc/aten/common.h` already has
`kCuda / kFlagOs / kFlagOsPython / kAscend / kMusa / kMetax / kTsingMicro / kGcu`.
`Dispatcher` stores one function pointer per backend, routing is driven by
`op_name = backend` lines in `backends*.conf`, and `FLAGOS_OP_<op>=<backend>`
provides a per-operator env override.

TileOPs naturally belongs to the "Python implementation that needs boxing"
category, structurally identical to `kFlagOsPython` (the FlagGems Python path).
The plan is to **reuse that machinery**, adding:

- The `Backend::kTileOps` enum value, the `tileops_fn_` slot in `Dispatcher`, the
  `"tileops"` name in `LogDispatch`, and `"tileops"` string parsing in
  `common.cc` (recognized both as a conf value and in `FLAGOS_OP_*` overrides).
- `torch_fl/configs/backends_tileops.conf`: structurally identical to
  `backends_cuda.conf`, with only the allowlisted operators changed to
  `= tileops` and everything else left at `cuda`.
- A `FLAGOS_USE_TILEOPS=1` branch in
  `torch_fl/__init__.py::_select_backend_config()` selecting that conf.

The benefit: routing granularity, env overrides, `FLAGOS_LOG_DISPATCH=1` logging
and the existing dispatch test conventions (`tests/integration/ops/test_*_dispatch.py`
has mature templates) all come for free, without introducing a second mechanism.

### 3.2 Choosing between the two implementation paths

**Recommendation: Stage A is pure Python registration, no C++ changes.**

Rationale: TileOPs objects are stateful, so calling them from C++ requires the
pybind bridge in `python_op_caller.h`, whose design assumes "module-level free
functions reachable by qualname". That does not fit "instantiate first, then
call, with instances cached by shape"; forcing it would push the instance cache
into C++ for no worthwhile return.

Stage A shape (new file `torch_fl/tileops_backend.py`):

```python
# pseudocode, structure only
_INSTANCES: dict[tuple, object] = {}

def _get_op(cls, key, *ctor_args, **ctor_kw):
    """Cache Op instances by (class, shape/dtype key) -- see 2.6, the 645 ms
    first call must not be paid repeatedly."""
    ck = (cls, key)
    op = _INSTANCES.get(ck)
    if op is None:
        op = _INSTANCES[ck] = cls(*ctor_args, **ctor_kw)
    return op

def _box(t):
    return _C._flagos_to_cuda_view(t) if t.device.type == "flagos" else t

def _unbox(t, idx):
    return _C._cuda_to_flagos_view(t, idx)

def _mm(self, mat2):                       # aten::mm  -> GemmOp(NN)
    idx = self.device.index or 0
    op = _get_op(GemmOp, ("nn",), trans_a=False, trans_b=False)
    return _unbox(op(_box(self).contiguous(), _box(mat2).contiguous()), idx)

_ROUTES = {"mm": _mm, "bmm": _bmm, ...}    # allowlist, adapted one by one

def enable_tileops_for_flagos(include=None, exclude=None) -> int:
    lib = torch.library.Library("aten", "IMPL")
    for name, fn in _selected(include, exclude):
        lib.impl(name, fn, "PrivateUse1")
```

Note that `torch.library` PrivateUse1 registration **takes precedence over** the
C++ dispatcher (confirmed in 2.3: the conf was still `backends_cuda.conf`, yet
the Python-registered `mm` was hit). So Stage A works even without the conf; the
conf and enum exist to make routing **observable and revertible per operator**.
The two complement each other rather than compete.

Stage B (optional, only once Python overhead is proven to be the bottleneck):
add C++ kernels in the `kTileOps` slot for the operators that have stabilized,
going through `DeviceBoxingGuard` plus a TileOPs caller similar to
`python_op_caller`. **Not to be done without profiling evidence.**

### 3.3 Allowlist selection criteria

The first batch only admits operators satisfying all of:

1. The ctor does not fix the shape (avoiding the cache collision in 2.7), or the
   shape key can be enumerated stably;
2. The semantics map one-to-one onto the aten schema (including confirmed
   layout/transpose differences);
3. There is benchmark evidence it is no slower than the existing cuda route, or
   the cuda route does not exist at all (fused operators).

Applying those criteria (with the measurements from 2.5), the tiers are:

- **Eligible for the first batch**: `logsumexp` (`LogSumExpFwdOp`, 4.68x, the
  only clear win among same-name aten operators); `mm` / `bmm` (`GemmOp`,
  `BmmFwdOp` carry their own shape caches with err=0, kept as a minimal credible
  sample proving the path works, but **off by default**, enabled explicitly via
  `FLAGOS_OP_mm=tileops`).
- **Worth admitting only after the L2 overhead is solved**: `softmax` /
  `log_softmax` / `layer_norm` / `rms_norm` / `relu` / `silu` -- 0.60-1.11x at
  the kernel layer but only 0.19-0.72x at L2. If the integration layer calls
  `op.kernel` directly (skipping L2 validation and dispatch), softmax and friends
  turn positive; otherwise admitting them is a pure regression.
- **Explicitly excluded**: `ArgmaxFwdOp` (0.01x); `CumsumFwdOp` (max err 2.12,
  unacceptable fp16 accumulation precision); pure memory-bound reductions
  (`sum` / `amax`, 0.19-0.21x with no measured advantage at the kernel layer
  either).

The batch genuinely worth pushing is the second one: fused operators (sdpa
variants, moe, SSD, mamba, lightning indexer). They have no same-name aten
counterpart, so they need to be exposed as custom ops via
`torch.library.custom_op` and called explicitly by models rather than
intercepting aten -- and `LogSumExpFwdOp`'s 4.68x is exactly the evidence for
that direction (it wins because the torch side is a multi-kernel composition).
That work should be a separate PR, after the Stage A skeleton lands.

### 3.4 Environment and dependencies

- TileOPs and TileLang are both **optional dependencies**. When missing,
  `enable_tileops_for_flagos()` silently returns 0, consistent with how
  `is_flaggems_available()` behaves, and `import torch_fl` is unaffected.
- Dependencies are pinned per 2.1: `tilelang==0.1.11`, `apache-tvm-ffi==0.1.11`,
  `z3-solver==4.15.4.0`. Record these in the docs and the CI install script; do
  not use version ranges.
- The integration layer sets `TILELANG_DISABLE_CACHE=1` at startup (with a
  comment pointing at 2.7), to be removed once upstream is fixed.
- Hardware gate: TileOPs supports SM_90 only; on non-Hopper we skip registration
  and warn.

---

## 4. Work items (suggested PR split)

1. **PR1 skeleton**: `Backend::kTileOps` enum, dispatcher slot, conf parsing,
   `backends_tileops.conf`, and the `FLAGOS_USE_TILEOPS` switch. Plumbing only,
   no operators.
2. **PR2 Stage A**: `torch_fl/tileops_backend.py` (instance cache, boxing,
   `mm`/`bmm` adapters), dispatch and numerical tests following the
   `tests/integration/ops/test_*_dispatch.py` template, the SM90 gate, plus docs
   and dependency pins.
3. **PR3 fused operators**: `rms_norm` and sdpa via `custom_op`, evaluated
   separately.
4. **Upstream issue**: report the cache key collision from 2.7 to TileLang, with
   the minimal repro (pure CUDA, `ReluFwdOp` at two different values of N).

## 5. Reproduction scripts

The one-off probe scripts written during research were not committed (TileLang
self-check, TileOPs on CUDA, routes A and B, `torch.library` registration of
`aten::mm`, performance plus boxing overhead plus streams, and the minimal cache
collision repro). The parts worth keeping long-term now live in the repository:

- `scripts/run_tileops_checks.py` -- numerical and fallback checks for all 60
  routes, with no pytest dependency.
- `tests/integration/ops/test_tileops_generated.py` -- the full generated suite.
- `scripts/codegen_tileops.py --check` -- a gate on generated-artifact
  consistency.

For the environment these need, see `tileops_codegen_design.md` section 6.3.
