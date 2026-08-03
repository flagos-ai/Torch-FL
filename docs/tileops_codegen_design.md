# TileOPs integration design: generating aten-aligned operator routes by codegen

This document is the implementation design derived from
[`tileops_integration_plan.md`](./tileops_integration_plan.md). Its scope is
narrowed to: **take the TileOPs operators that align exactly with torch aten and
generate them all at once by codegen, integrated as an optional operator
library.** The fused and non-aten operators (sdpa, moe, mamba, SSD, lightning
indexer and 63 others) are out of scope here.

Every grouping and feasibility conclusion below was measured on H800
(torch 2.10.0+cu130, CUDA 13.0, SM90, Python 3.12), not derived on paper.

---

## 1. Why codegen is viable

The research phase concluded that "the only option is per-operator allowlisting
with hand-written adapters". **That conclusion needs correcting.** "Hand-written"
applies to the act of confirming each operator's semantics by hand, but
measurement shows the ctor parameters are highly regular and can be generated
mechanically by **recipe grouping** -- one adapter per operator is not needed.

Four supporting facts:

### 1.1 The manifest is a complete single source of truth

`tileops/manifest/*.yaml` contains 23 files and 184 entries (165 `implemented`).
For the 102 aten-aligned operators we filtered out:

| Field | Coverage | Use |
|---|---|---|
| `ref_api` | 102/102 | Map to the aten base name |
| `signature.inputs[].dtype` (concrete dtype, not `same_as`) | **102/102** | Generate dtype guards |
| `signature.params` | 102/102 | Validate the ctor/forward parameter split |
| `workloads` (shape + dtype + label) | **102/102** | CI warmup and benchmark shapes |
| `status` | 184/184 | Filter out spec-only entries |

Conclusion: **no reflection-based discovery is needed** (the FlagGems Python path
has to walk `_FULL_CONFIG`). The manifest is directly readable, and all 165 Op
classes resolve by name (0 unresolved, measured).

### 1.2 The aten-aligned surface: 102 Ops mapping to 90 unique aten names

Intersecting the last segment of `ref_api` (with an alias table:
`torch.matmul -> mm`, `torch.nn.functional.softmax -> _softmax`,
`layer_norm -> native_layer_norm`, ...) against the 1009 aten base names in
`backends_cuda.conf`:

- **102 Ops align** (90 unique aten names, 8.9% of the conf), distributed by
  family: `elementwise_unary_math` 23, `elementwise_binary` 23, `reduction` 19,
  `elementwise_unary_activation` 15, `convolution` 6, `normalization` 5,
  `pool` 3, `gemm` 2, `scan` 2, `elementwise_multi_input` 2, `bmm` 1,
  `attention_indexing` 1.
- 63 do not align (attention 14, moe 7, normalization 7, sequence_modeling 6,
  position_encoding 6, pool 6, linear_attention 4, ...) and are not covered here.

### 1.3 Ctor arguments can be derived mechanically from the aten call site -- measured

This is the key fact that makes codegen work. Ctor parameter frequencies are
highly concentrated: `dtype` 82 times, `N_total` 33, `a_shape`/`b_shape` 21 each,
`dim` 21, `keepdim` 16. Grouped by ctor signature, **68 of 102 operators fall
into 4 recipes**:

| Recipe | n | Ctor shape | Derived from |
|---|---|---|---|
| `UNARY` | 28 | `(N_total, dtype[, inplace])` | `x.numel()`, `x.dtype` |
| `BINARY` | 19 | `(a_shape, b_shape, dtype[, alpha])` | `tuple(a.shape)`, `tuple(b.shape)`, `a.dtype` |
| `REDUCE` | 19 | `(dtype, dim[, keepdim/correction/ord])` | aten arguments passed through |
| `SOFTMAX` | 2 | `(dim,)` | aten arguments passed through |

Measured verification (`TILELANG_DISABLE_CACHE=1`, all err=0 or within normal
fp16 range):

```
UNARY   relu/sigmoid/tanh/abs/exp x {(512,128), (33,7), (1024,)} x {fp16,bf16}  all OK
BINARY  add/mul x {same shape, (256,128)+(128,), (256,1)+(1,128) broadcast, 3D} all OK
REDUCE  sum/amax x {dim=-1, dim=0, keepdim=True, 3D dim=1}                      all OK
SOFTMAX softmax x {dim=-1, dim=0, 3D dim=1}                                     all OK
```

Three additional confirmations:

- **Broadcasting works**: both `(256,128)+(128,)` and `(256,1)+(1,128)` are
  correct, so the `a_shape/b_shape` recipe does not require us to broadcast
  manually.
- **Non-contiguous inputs work**: `relu` on a transposed view is correct (via
  `reshape(-1)`).
- **dtype guards are necessary and well-defined**: `relu` passes on fp32/bf16 and
  raises `ValueError: ReluFwdKernel only supports dtypes [...]` on int32/int64.
  The dtype set is statically readable -- 50/102 are available from
  `op_cls.kernel_cls.SUPPORTED_DTYPES` (4 distinct value sets), and the other 52
  from the manifest's `signature.inputs[].dtype` (102/102 coverage).
  **Combining the two sources gives 100% static guard generation** with no
  instantiation required.

The remaining 34 operators spread across 25 ctor shapes (conv 6, pool 3, clamp 4,
masked_fill 2, group_norm 2, `GemmOp`, `LayerNormFwdOp`, `WhereFwdOp`, ...) and
are handled as "hand-written recipes"; see section 3.4.

### 1.4 The admission bar set by performance data

See `tileops_integration_plan.md` section 2.5. The core fact: **the L2 `Op` layer
carries a fixed Python overhead of roughly 0.05-0.09 ms independent of size**,
and bypassing L2 to call `op.kernel(x)` directly turns softmax from 0.72x to
1.11x. The generated adapter layer therefore **takes the direct-kernel path by
default**, with L2 only as a fallback.

---

## 2. Overall architecture

```
tileops/manifest/*.yaml ──┐
                          ├─► scripts/codegen_tileops.py ─┬─► torch_fl/generated/tileops_routes.py
torch_fl/tileops_spec.py ─┘   (run offline, output committed)
  (recipe table + alias table                             ├─► torch_fl/configs/backends_tileops.conf
   + exclusion table, hand-maintained)                    └─► tests/integration/ops/test_tileops_generated.py

                                          runtime
                                            │
torch_fl/__init__.py ──FLAGOS_USE_TILEOPS=1──► tileops_backend.py
                                                 ├─ SM90 + dependency gate (silently returns 0 if missing)
                                                 ├─ instance cache (op_cls, ctor_key)
                                                 ├─ boxing: _flagos_to_cuda_view / _cuda_to_flagos_view
                                                 └─ torch.library.Library("aten","IMPL").impl(..., "PrivateUse1")
```

Three design principles:

1. **The manifest is the only spec source.** `tileops_spec.py` holds only what
   the manifest does not have (recipe grouping, aten aliases, exclusion
   reasons). When TileOPs is upgraded, re-run codegen and diff.
2. **Pure Python, no C++ changes.** Same rationale as integration_plan section
   3.2: TileOPs objects are stateful and instances must be cached by shape, so
   pushing them through the `python_op_caller.h` pybind bridge is not worth it.
   `torch.library` PrivateUse1 registration was measured to take precedence over
   the C++ dispatcher, which is sufficient.
3. **Generated artifacts are committed** (not generated at build time),
   consistent with `csrc/aten/generated/`. This keeps route changes visible in
   review and avoids adding a TileOPs dependency to the build.

### 2.1 The conf and the Backend enum

Reuse the existing machinery, adding `Backend::kTileOps`:

- Add `kTileOps` to the enum at `csrc/aten/common.h:20`.
- Recognize `"tileops"` in the two parse sites in `csrc/aten/common.cc` (conf
  values around `common.cc:95`, `FLAGOS_OP_*` overrides around `common.cc:128`).
- Add the `tileops_fn_` slot and the `"tileops"` name at the three sites in
  `csrc/aten/dispatcher.h` (`:52` `RegisterKernel`, `:80` `DispatchAs`,
  `:100` `LogDispatch`).
- `torch_fl/configs/backends_tileops.conf` (generated): structurally identical to
  `backends_cuda.conf` with 2033 entries, only the allowlisted operators set to
  `= tileops`.
- Add the `FLAGOS_USE_TILEOPS=1` branch to
  `torch_fl/__init__.py::_select_backend_config()`.

The C++ side has **no kernels registered** at this stage (an empty slot means
fallback). The value of the enum is that `FLAGOS_LOG_DISPATCH=1` can observe
routing and `FLAGOS_OP_<op>=cuda` can revert individual operators -- complementing
Python registration rather than competing with it.

> **However, neither of those two features works automatically for TileOPs
> routes on the C++ side; each has to be implemented again in Python.** The gaps
> found by measurement (now fixed): a `torch.library` PrivateUse1 binding
> **intercepts before the C++ dispatcher**, so a bound operator never reaches the
> dispatcher at all. Even though `common.cc` correctly parses
> `FLAGOS_OP_relu=cuda` and prints `[flagos] env override: relu -> cuda`,
> `torch.relu` still built a TileOPs instance. Likewise `FLAGOS_LOG_DISPATCH=1`
> only shows the operators that fell through to cuda, which reads as if TileOPs
> were not wired up at all.
>
> The fix: `enable_tileops_for_flagos()` consults `_env_override_backend()`
> before registering, and **does not bind** operators pointed elsewhere
> (measured: under `FLAGOS_OP_relu=cuda`, registered drops 60 -> 59, relu builds
> no instance, results are correct, and the un-overridden sigmoid still routes
> through TileOPs). `build_impl()` wraps the impl in a logging layer when
> `FLAGOS_LOG_DISPATCH=1` is set, and the fallback logs a line too, so int32 relu
> prints `relu -> tileops` followed by `relu -> cuda (tileops declined)`.
>
> The lesson: the "free reuse" the conf and enum provide only covers the C++
> path. Any backend registered from Python has to wire up those two
> cross-cutting features itself, or the escape hatches promised by the docs are
> hollow.

---

## 3. Codegen design

### 3.1 Input: `torch_fl/tileops_spec.py` (hand-maintained)

```python
# aten aliases: the part of ref_api that cannot be derived mechanically
ATEN_ALIAS = {
    "torch.matmul": "mm",
    "torch.nn.functional.softmax": "_softmax",
    "torch.nn.functional.log_softmax": "_log_softmax",
    "torch.nn.functional.layer_norm": "native_layer_norm",
    ...
}

# ctor signature -> recipe
RECIPES = {
    ("N_total", "dtype"): "UNARY",
    ("N_total", "dtype", "inplace"): "UNARY",
    ("a_shape", "b_shape", "dtype"): "BINARY",
    ("a_shape", "b_shape", "dtype", "alpha"): "BINARY",
    ("dtype", "dim", "keepdim"): "REDUCE",
    ("dtype", "dim"): "REDUCE",
    ("dtype", "dim", "correction", "keepdim"): "REDUCE",
    ("dtype", "ord", "dim", "keepdim"): "REDUCE",
    ("dim",): "SOFTMAX",
}

# exclusion table: every entry must carry a measured reason
EXCLUDE = {
    "ArgmaxFwdOp":  "perf 0.01x (7.826ms vs 0.044ms @8192x4096) -- likely degenerate impl",
    "ArgminFwdOp":  "same as ArgmaxFwdOp, not measured separately, excluded conservatively",
    "CumsumFwdOp":  "max err 2.12 @fp16 8192x4096 -- accumulation precision unacceptable",
    "CumprodFwdOp": "same accumulation class as CumsumFwdOp, excluded conservatively",
    "GemmFp8Op":    "fp8 semantics do not align with aten::mm",
}

# off by default (generated into the conf but valued cuda; enable via FLAGOS_OP_*)
DEFAULT_OFF = {"GemmOp", "BmmFwdOp"}   # the GEMM kernel itself is 0.65x, see 1.4
```

### 3.2 Generator: `scripts/codegen_tileops.py`

The flow mirrors how `scripts/codegen_ops.py` is organized (read spec ->
classify -> call a generator per category -> write files and conf):

1. Read every manifest yaml, filter to `status == "implemented"`.
2. Map `ref_api` to an aten base name via `ATEN_ALIAS`; intersect with the base
   names in `backends_cuda.conf` and drop anything outside it (-> 102).
3. Take ctor parameters from `inspect.signature(cls.__init__)` (dropping
   `self/kernel_map/tune`) and look up `RECIPES` to pick a recipe; anything not
   found falls to `MANUAL` (34 operators, see section 3.4).
4. dtype guards: prefer `op_cls.kernel_cls.SUPPORTED_DTYPES`, falling back to
   parsing the manifest's `signature.inputs[].dtype` (tokenizing
   `"float16 | bfloat16 | float32"`).
5. Apply `EXCLUDE`, render per recipe, write the three artifacts.
6. **Self-check**: assert that each generated entry's aten schema argument count
   matches what the recipe expects, failing outright rather than emitting broken
   code.
7. **Format**: pipe the two Python products through `ruff format` before writing
   or comparing. The repo's lint job runs `ruff format --check .` over
   everything including generated files, so without this step `ruff format .`
   rewrites them and `--check` immediately reports them stale -- the lint gate
   and the codegen gate would contradict each other. The conf is not Python and
   is written as-is.

### 3.3 Output: `torch_fl/generated/tileops_routes.py`

Recipes render as table-driven entries rather than one function body per
operator:

```python
# AUTO-GENERATED by scripts/codegen_tileops.py -- DO NOT EDIT
from torch_fl.tileops_backend import UNARY, BINARY, REDUCE, SOFTMAX

FP = (torch.float16, torch.bfloat16, torch.float32)
INT = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

ROUTES = [
    # (aten_name, recipe, op_module, op_class, dtypes, extra)
    ("relu",        UNARY,  "tileops.ops.elementwise.activations", "ReluFwdOp",   FP,       {}),
    ("silu",        UNARY,  "tileops.ops.elementwise.activations", "SiluFwdOp",   FP,       {}),
    ("abs",         UNARY,  "tileops.ops.elementwise.math_unary",  "AbsFwdOp",    FP + INT, {}),
    ("add.Tensor",  BINARY, "tileops.ops.elementwise.arithmetic",  "AddFwdOp",    FP + INT, {"alpha": True}),
    ("sum.dim_IntList", REDUCE, "tileops.ops.reduction.reduce",    "SumFwdOp",    FP,       {"keepdim": True}),
    ("logsumexp",   REDUCE, "tileops.ops.reduction.softmax",       "LogSumExpFwdOp", FP,    {"keepdim": True}),
    ("_softmax",    SOFTMAX,"tileops.ops.reduction.softmax",       "SoftmaxFwdOp",FP,       {}),
    ...
]
```

`op_module` is a string rather than an import, so `import torch_fl` does not
trigger a TileOPs import (the module stays safely importable when TileOPs is
absent, it just registers 0 routes).

### 3.4 Hand-written recipes (the remaining 30) -- grouped by measured pattern

After codegen there are **30 operators across 22 ctor shapes** left. The grouping
below is not by literal ctor signature but by "the shape of the adapter code we
have to write", and **every group was measured on H800** (fp16, calling L2
directly on CUDA tensors and comparing against CPU aten; conv/pool use an fp32
reference). Conclusion: **26 can be covered by 5 more recipes, 3 have a hard
blocker, and 1 stays excluded.**

#### (1) `SCALAR_UNARY` -- ctor `(N_total, dtype, <scalars...>)`, fwd `(x)`, 7 operators

`EluFwdOp`, `LeakyReluFwdOp`, `HardtanhFwdOp`, `SoftplusFwdOp`, `NanToNumFwdOp`,
`GeluFwdOp`, `ClampScalarFwdOp`. This is UNARY plus "pass the aten scalar
arguments into the ctor and into the ctor_key", roughly 15 lines.

```
elu alpha=1.0 / 0.5      err=0        leaky_relu slope=0.01   err=3.05e-05
hardtanh(-1,1)           err=0        softplus(beta=1,thr=20) err=0
gelu approximate=none    err=3.05e-05 gelu approximate=tanh   err=0
nan_to_num(nan=0)        err=0        clamp.Scalar(-0.5,0.5)  err=0
```

Two details: `GeluFwdOp.__init__` is `(*args, **kwargs)`, so
`inspect.signature` cannot see the real signature (which lives on
`_GeluApproximateBase`: `(N_total, dtype, *, approximate='none')`). Codegen's
ctor classification therefore **must walk the MRO for the first named
`__init__`**. And `approximate` is keyword-only.

#### (2) `BROADCAST_TENSORS` -- ctor takes **shape tuples**, fwd takes tensors, 7 operators

`ClampFwdOp`, `ClampMinFwdOp`, `ClampMaxFwdOp`, `WhereFwdOp`, `MaskedFillFwdOp`,
`MaskedFillScalarFwdOp`, `LerpTensorFwdOp`. The ctor parameters are named
`input`/`min`/`condition`, which reads as if they take tensors; they actually
take `tuple` shapes, and broadcasting is computed in the ctor with
`torch.broadcast_shapes`.

```
clamp_min/clamp_max/clamp (scalar materialized as a 0-dim tensor)   err=0
clamp_min(tensor bound, (64,32)+(32,) broadcast)                    err=0
where(cond,x,y) and cond=(64,1) broadcast                           err=0
masked_fill.Scalar(-inf) / .Tensor(0-dim)                           err=0
lerp.Tensor                                                         err=0.00195
```

One semantic point: **`clamp.Scalar` has two routes** -- `ClampScalarFwdOp`
takes Python scalars directly (group (1)), while `ClampMinFwdOp`/`ClampFwdOp`
require materializing the scalar as a 0-dim CUDA tensor. Prefer the former and
save an H2D copy. `aten::masked_fill.Scalar`'s `value` lives in the ctor, so
ctor_key must include the value or different fill values will alias.

#### (3) `BINARY_EXTRA` -- BINARY plus one ctor scalar, 2 operators

`DivFwdOp` (`rounding_mode`) and `LerpFwdOp` (`weight`). The existing BINARY
recipe just needs an extra pass-through.

```
div rounding_mode=None / trunc / floor   err=0
lerp.Scalar weight=0.3                   err=0.000977
```

Note the aten side has three distinct overloads (`div.Tensor` /
`div.Tensor_mode`); `rounding_mode` only appears on the latter, so they must be
bound separately per overload.

#### (4) `SHAPELESS` -- ctor takes no shapes (internally cached), 4 classes

`GemmOp`, `BmmFwdOp` plus the six `conv{1,2,3}d` classes. This is the easiest
group: **instances can be global singletons** (keyed by ctor configuration) and
are unaffected by the cache collision in section 2.7.

```
mm via GemmOp(trans_a=False, trans_b=False)   err=6.1e-05
conv2d no-bias s1p1                            err=0.00766
conv2d bias   s1p1                             err=0.00764
conv2d s2 groups=2                             err=0.00381
```

`GemmOp`'s ctor defaults to `trans_b=True` (NT), so binding `aten::mm` requires
passing `trans_b=False` explicitly -- confirmed by measurement. Three points on
the conv side: with and without bias are **two different classes**
(`Conv2dFwdOp` / `Conv2dBiasFwdOp`), and the adapter picks based on
`bias is None`; `aten::convolution` also has `transposed` and `output_padding`
parameters that TileOPs does not, so the adapter **must guard** them (falling
back to aten when `transposed=True` or `output_padding != 0`); and the first
conv compilation emits a TileLang data-race warning (`out(...)` written by
multiple threads) -- the numerics are correct but this **should be reported
upstream**.

#### (5) `POOL` -- ctor `(kernel_size, stride, padding, ceil_mode, ...)`, 3 operators

`AvgPool1dFwdOp`, `AvgPool2dFwdOp`, `AvgPool3dFwdOp`. The ctor maps one-to-one
onto the aten parameters; the most straightforward group.

```
avg_pool1d k=2  err=0     avg_pool2d k=2  err=0
avg_pool3d k=2  err=2.44e-04 (fp32 reference; CPU has no fp16 avg_pool3d)
```

**A bonus opportunity codegen missed**: `MaxPool{1,2,3}dIndicesFwdOp` in the
manifest returns `(out, indices)` from `forward`, which lines up exactly with
`aten::max_pool2d_with_indices` (present in the conf), and the indices dtype is
already `int64`. Measured, both `out` and `idx` of `max_pool2d_with_indices` are
**bit-exact**. These 3 were previously classified as "not aten-aligned" and left
outside the 102; they should be brought in.

#### (6) `ORD_DISPATCH` -- 3 classes sharing one aten overload, 3 operators

`L1NormFwdOp`, `L2NormFwdOp`, `InfNormFwdOp` -> `aten::linalg_vector_norm`,
selected by the `ord` argument, with other `ord` values falling back to aten.

```
vector_norm ord=1 / 2 / inf   err=0
```

These three are the "false alignments" caught by the generator's self-check in
section 6.1; they are now confirmed **not to be a semantic problem, just in need
of an ord dispatch layer**.

#### (7) `ROUND` -- `decimals` is on forward, not the ctor, 1 operator

`RoundFwdOp(N_total, dtype)` with `forward(input, decimals)`, corresponding to
`aten::round` (decimals=0) and `aten::round.decimals`. Measured `err=0`.

#### Hard blocker: the 3 norm operators (the `native_*` schema does not match)

`LayerNormFwdOp`, `GroupNormFwdOp`, `GroupNormNoAffineFwdOp` -- these **cannot be
bound to `aten::native_layer_norm` / `native_group_norm`** as originally planned.
This is a new finding from measurement, and it overturns this section's earlier
judgment that listed `LayerNormFwdOp` as "first batch, semantics confirmed":

| | TileOPs returns | aten requires |
|---|---|---|
| `LayerNormFwdOp` | a single Tensor `(128,256)` | `(out, mean, rstd)` = `[(128,256),(128,1),(128,1)]` |
| `GroupNormFwdOp` | a single Tensor `(4,32,8,8)` | `(out, mean, rstd)` = `[(4,32,8,8),(4,4),(4,4)]` |

Probing confirms the Op has **no** mean/rstd attributes; `forward` returns only
the normalized result, and the statistics are discarded inside the kernel and
cannot be recovered afterwards. Three ways out:

1. **Recompute** mean/rstd in the adapter (`x.mean(dim)` / `rsqrt(var+eps)`) --
   correct, but it costs another pass over the input, and given that
   `LayerNormFwdOp` is only 0.43x to begin with (section 2.5) this is
   guaranteed to be slower. **Not worth it.**
2. Bind only the **inference path**: `aten::layer_norm` (the composite operator,
   which does not require returning statistics) instead of `native_layer_norm`.
   The cost is that under autograd torch decomposes straight to `native_*`, so
   this route only takes effect under `inference_mode` / `no_grad`.
3. Request the feature from TileOPs: add an optional `return_stats=True` to
   `LayerNormFwdOp`.

**Recommendation: option 3, with option 2 as a stopgap**, and add this to the
upstream feedback list.

#### Still excluded

`BatchNormFwdOp`/`BatchNormBwdOp` (the in-place `running_mean/var` update
semantics need checking), `TopkSelectorOp` (`forward(index_score, starts, ends)`
does not align with `aten::topk`), `GemmFp8Op`, `Argmax`/`Argmin`, and
`Cumsum`/`Cumprod` (reasons in `tileops_spec.EXCLUDE`).

### 3.5 Runtime: `torch_fl/tileops_backend.py` (hand-written)

> The following is the design intent; the implementation deviates in three
> necessary ways, see section 6.2 (the kernel fast path must be verified rather
> than assumed, the fallback cannot call aten directly, and the cache env var
> must be set before import).

```python
_INSTANCES: dict[tuple, object] = {}     # (op_cls, ctor_key) -> (op, kernel)

def _get(op_cls, ctor_key, *args, **kw):
    """Instance cache. Cold-cache ctor 4169.8 ms / warm 54.0 ms, warm call
    0.056 ms -- see integration_plan section 2.6. Never construct inside a
    dispatch."""
    ck = (op_cls, ctor_key)
    ent = _INSTANCES.get(ck)
    if ent is None:
        op = op_cls(*args, **kw)
        ent = _INSTANCES[ck] = (op, getattr(op, "kernel", None))
    return ent

def _box(t):
    return _C._flagos_to_cuda_view(t) if t is not None and t.device.type == "flagos" else t

def _unbox(t, idx):
    return _C._cuda_to_flagos_view(t, idx)

def _call(op, kernel, *args):
    """Default to calling the kernel directly (saving L2's fixed 0.05-0.09 ms,
    see section 1.4), falling back to L2 on failure."""
    if kernel is not None and _BYPASS_L2:
        try:
            return kernel(*args)
        except Exception:
            pass
    return op(*args)
```

Each of the four recipes is about 10 lines, e.g. `UNARY`:

```python
def _unary(op_cls, dtypes, aten_fn):
    def impl(x):
        if x.dtype not in dtypes:
            return aten_fn(x)                       # dtype guard -> fallback
        idx = x.device.index or 0
        xb = _box(x).contiguous().reshape(-1)
        op, kern = _get(op_cls, (x.numel(), x.dtype), x.numel(), x.dtype)
        return _unbox(_call(op, kern, xb).view_as(_box(x)), idx)
    return impl
```

### 3.6 Gates

`enable_tileops_for_flagos()` **silently returns 0** if any condition is unmet
(consistent with `is_flaggems_available()`, leaving `import torch_fl`
unaffected):

1. `import tileops` succeeds;
2. `torch.cuda.get_device_capability() == (9, 0)` (TileOPs is SM_90 only; on
   non-Hopper, warn and skip);
3. Dependency versions match the pins in section 4.

---

## 4. Dependencies and known upstream issues

Per `tileops_integration_plan.md` section 2.1, all three pins must be exact --
do not use ranges:

```
tilelang==0.1.11
apache-tvm-ffi==0.1.11     # 0.1.12 aborts at import (ABI-coupled to the tilelang wheel)
z3-solver==4.15.4.0        # 5.x names libz3.so.4.15 differently
```

**Cache key collision** (integration_plan section 2.7): elementwise/softmax Ops
with different `N_total` reuse the first compiled artifact. This matters a great
deal here -- the 28 `UNARY` recipe operators build instances keyed exactly by
`N_total`.

**The root cause has been pinned to a specific layer, upstream has fixed it, and
our side now uses a targeted workaround (measured 2026-07-31).** The earlier note
that "TileLang's disk cache keys `_generate_key` off the script sha256" is
inaccurate:

- Instrumenting `KernelCache._generate_key`: three different values of N trigger
  only **one** call, and the other two never reach the disk kernel cache at all.
  So the collision is not in that layer.
- Instrumenting `load_frontend_cached` gives the decisive evidence -- N=2097152
  returns **HIT=True**, and both keys are identical:

  ```
  key = ((('npt_arg', 8), ('threads_arg', 256)), None)
  ```

  The key contains only explicit parameters; **`N` is not in it at all**. The
  reason is that `tileops/kernels/elementwise.py` wraps
  `_make_unary_direct(N, dtype, ...)` in `@functools.lru_cache`, and the inner
  `@tilelang.jit def kernel(threads_arg)` captures `N` as a **closure free
  variable**, while `JITImpl._frontend_cache_key_data` only covers explicit
  parameters. The two `JITImpl` objects have distinct `_kernel_cache` objects
  (different ids), so the only leak path is the **frontend disk cache** layer.
- The upstream issue is exactly this:
  **[tile-ai/tilelang#2358](https://github.com/tile-ai/tilelang/issues/2358)
  "[JIT] Include closure free vars in frontend cache key"**, with a matching
  minimal repro, fixed by **#2363 "[JIT] Remove frontend disk cache"**,
  **closed 2026-06-09**.
- The timing missed by exactly one day: `tilelang==0.1.11` shipped on
  **2026-06-08** and the fix merged the next day. It also explains why `GemmOp`
  is immune -- it passes `(m,n,k)` down as **explicit parameters**.

Disposition (implemented in `tileops_backend._disable_frontend_cache()`):

Instead of the blanket `TILELANG_DISABLE_CACHE=1`, make **only**
`load_frontend_cached`/`store_frontend_cache` no-ops (equivalent to what #2363
does) and **keep the kernel disk cache**. Measured across three `ReluFwdOp`
shapes: all correct, and the per-process recompilation cost disappears:

| Approach | First process | Second process | Correctness |
|---|---|---|---|
| `TILELANG_DISABLE_CACHE=1` (original) | 12.1s | **12.0s** | all correct |
| Frontend cache only (current) | 12.7s | **0.3s** | all correct |

Re-checked through the full aten route (`torch.relu` on flagos, three different
shapes): all results correct, second process **14.2s -> 1.2s**.

`FLAGOS_TILEOPS_DISABLE_ALL_CACHE=1` is retained as an escape hatch (back to the
blanket disable). `warmup()` is consequently downgraded from "required" to
"convenient when the cache is cold".

**No new issue needs filing** (#2358 is already fixed). Once the pin moves to
`tilelang==0.1.12` + `apache-tvm-ffi==0.1.12`, `_disable_frontend_cache()` can be
deleted -- note that the "0.1.12 aborts at import" recorded above is an ABI
mismatch between **tilelang 0.1.11 and tvm-ffi 0.1.12**; upgrading both together
is the supported combination. The 0.1.12 wheel downloads at roughly 11 KB/s
through the proxy and 50 MB has not finished, so **"0.1.12 is fixed" is currently
inferred from the upstream timeline, not verified on this machine**.

---

## 5. Testing

`tests/integration/ops/test_tileops_generated.py` is generated following the
`tests/integration/ops/test_*_dispatch.py` template, with two cases per route
entry:

1. **Dispatch hit**: confirm the tileops path is taken under
   `FLAGOS_LOG_DISPATCH=1`.
2. **Numerical agreement**: compare against aten on CPU/cuda, with tolerances
   graded by dtype (relative tolerance for fp16/bf16, bit-exact for integer
   types). Shapes are the first entry from the manifest's `workloads`, plus one
   small shape (covering the unaligned tail) and one non-contiguous view.

Plus two non-generated cases: dtype guard fallback (int32 through `relu` should
fall back to aten without erroring) and, when TileOPs is absent,
`enable_tileops_for_flagos()` returning 0 with `import torch_fl` still working.

Benchmarks are generated from the manifest `workloads`. The **admission bar**:
the direct-kernel path must be no slower than the existing cuda route, otherwise
the operator goes into `DEFAULT_OFF`.

---

## 6. PR split and landing status

| PR | Content | Status |
|---|---|---|
| 1 | `Backend::kTileOps` enum + dispatcher slot + conf parsing + `FLAGOS_USE_TILEOPS` | **Done** |
| 2 | `tileops_spec.py` + `codegen_tileops.py` + 4 recipes + generated artifacts | **Done, 60 routes** |
| 3 | Groups (1)(2)(3)(7) from section 3.4: `SCALAR_UNARY` 7 + `BROADCAST_TENSORS` 7 + `BINARY_EXTRA` 2 + `ROUND` 1 = **17** | To do, patterns measured |
| 4 | Groups (4)(5)(6) from section 3.4: `SHAPELESS` 8 (mm/bmm/conv x6) + `POOL` 3 + `ORD_DISPATCH` 3, plus the newly admitted `max_pool*_with_indices` 3 = **17** | To do, patterns measured |
| 5 | Upstream feedback: TileLang cache key collision (pure CUDA minimal repro), conv data-race warning, TileOPs norm `return_stats` request | To do |
| 6 | The norm operators (hard blocker in section 3.4): wait for `return_stats`, or bind the `aten::layer_norm` inference path first | Blocked |

Per the measured grouping in section 3.4, **34 more routes can land** across the
30 remaining operators (17 + 17; `SHAPELESS` and `ROUND` each cover several
overloads), leaving 3 norm operators blocked and 7 still excluded. On top of the
60 already landed, the aten-aligned surface reaches **94 routes**.

### 6.1 What PR1/PR2 actually landed

```
routed 60 ops: {UNARY: 27, BINARY: 19, REDUCE: 12, SOFTMAX: 2}
excluded 12 (each with a measured reason, see tileops_spec.EXCLUDE)
manual recipes still needed: 30 ops across 22 ctor shapes
```

That is 4 fewer than the 64 the design predicted, because **the generator's
self-check (step 6 of section 3.2) caught two classes of false alignment the
design phase did not anticipate** -- which is precisely the value of putting
validation in the generator:

- `InfNormFwdOp` / `L1NormFwdOp` / `L2NormFwdOp` all declare
  `ref_api = torch.linalg.vector_norm` and collide on the same aten overload;
  they actually need dispatch by `ord` -> moved to hand-written.
- `RoundFwdOp`'s `forward(input, decimals)` takes 2 arguments, violating UNARY's
  1-argument convention -> moved to hand-written.

Verification results (H800, `FLAGOS_USE_TILEOPS=1`): **all 60 routes numerically
correct** (42 floating-point routes via `assert_close`, 18 integer/bool routes
required bit-exact), plus broadcasting, transposed non-contiguous inputs, int32
dtype-guard fallback and registration idempotence -- all passing.

### 6.2 Two runtime problems found during implementation (not anticipated in design)

**(a) L2 cannot be bypassed unconditionally.** Section 3.5 originally planned to
"call the kernel directly by default". Measurement shows `Op.forward` reshapes
before handing off to the kernel for some operators -- the BINARY family's L2
flattens first, so calling `op.kernel` with the original 2-D tensor raises
`ndim mismatch, expected 1; got 2`. Furthermore `op.kernel` is `None` until the
Op has run once, so it is unavailable at construction time.

Changed to **verify per instance rather than assume** (`_Entry`): the first call
goes through L2 (always correct), then the kernel is tried with the same
arguments and the results compared; the fast path is enabled only on an exact
match, otherwise that instance stays on L2 permanently. This costs one extra
call, negligible against the ~4 s ctor. Measured, all four representative
operators enabled the fast path successfully:

| Operator | 8192×4096 fp16 through the full aten route | Kernel fast path active |
|---|---|---|
| `relu` | 0.049 ms | yes |
| `softmax` | 0.100 ms | yes |
| `logsumexp` | 0.101 ms | yes |
| `sum` | 0.150 ms | yes |

(`softmax` at 0.100 ms compares against 0.150 ms for the L2 path in the section
2.5 table and 0.109 ms for torch -- the fast path does pull softmax slightly
ahead of torch.)

**(b) The fallback cannot call `torch.ops.aten.<op>` directly.** Our
implementation is registered on PrivateUse1, so calling aten directly on a flagos
tensor **recurses infinitely** (measured: `RecursionError`). Changed to
`_make_fallback()`: box to a CUDA view with zero copy, run aten on CUDA (which is
what `backends_cuda.conf` does by default anyway), then unbox.

**(c) `TILELANG_DISABLE_CACHE=1` must be set before importing tileops.** TileLang
reads the variable at import time; setting it later has no effect, and the
symptom is exactly the ndim/shape mismatch from section 2.7. It has been moved
into `is_tileops_available()` ahead of the import.

### 6.3 Test environment and reproduction

The TileOPs stack is an optional dependency; install it into an isolated
directory per the pins in section 4 (do not install into the main environment --
`apache-tvm-ffi` and the tilelang wheel are ABI-coupled):

```bash
python -m venv --system-site-packages "$VENV"     # reuse the host's torch/CUDA
# if the host lacks ensurepip, add --without-pip and install with the host pip
# plus --target
pip install --target "$VENV/lib/python3.12/site-packages" pytest

# put the tilelang stack in $STACK and bring it in via a .pth or PYTHONPATH;
# z3 needs LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$STACK/z3/lib:$LD_LIBRARY_PATH"
export FLAGOS_USE_TILEOPS=1
```

On an offline machine, put the stack directory on a persistent path rather than
`/tmp`.

Two verification paths:

```bash
# the full suite
$ pytest tests/integration/ops/test_tileops_generated.py -q
123 passed

# the equivalent script with no pytest dependency (for hosts where pytest
# cannot be installed)
$ python scripts/run_tileops_checks.py
registered=60 checked=60 passed=60 skipped=0 failed=0
dtype-guard fallback: OK

# generated-artifact consistency (CI gate)
$ python scripts/codegen_tileops.py --check
codegen_tileops: products up to date

# lint, exactly what .github/workflows/lint.yml runs (ruff must be on PATH for
# the codegen gate above to agree with this one)
$ ruff check . && ruff format --check .
All checks passed!
105 files already formatted
```

---

## 7. Two corrections to the research conclusions

1. **integration_plan section 1.1 says "the only option is per-operator
   allowlisting with hand-written adapters"** -- corrected: ctor arguments can be
   derived mechanically by recipe, and 68/102 are generatable (section 1.3 has
   all four recipes measured and passing, including broadcasting and
   non-contiguous inputs). "Hand-written" applies only to the remaining 34.
2. **integration_plan section 3.3 puts `mm`/`bmm` in the first batch** --
   corrected: GEMM is kept but `DEFAULT_OFF` (0.63x at the kernel layer, which is
   genuine slowness rather than overhead). What should actually be enabled is
   `logsumexp` (4.68x) and the softmax family, which turns positive once L2 is
   bypassed.
