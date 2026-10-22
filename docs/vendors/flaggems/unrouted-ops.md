# FlagGems unrouted operator inventory

FlagGems' `_FULL_CONFIG` holds **433** operators. **320 are currently routed** to the
`flagos_python` path; **113 do not appear in the routing table under their own name**. This
document buckets them by cause, for working through in batches.

> **2026-07 update (+13 this round, 307 → 320)**: ③varargs and part of rng are now handled.
> - **varargs unary in-place (7)**: `asinh_`/`sinh_`/`log1p_`/`digamma_`/`sgn_`/`hardswish_`/`logit_`.
>   The gems wrapper is `(*args,**kwargs)` so arity cannot be introspected, but the aten schema is
>   the authoritative arity. An explicit `_FLAGGEMS_ARITY_OVERRIDE` allowlist (restricted to simple
>   elementwise ops verified to run, produce correct numerics, and drop no arguments) bypasses the
>   npos gate. `logit_` has npos=2 (self and eps both passed positionally; eps=None is fine).
>   Measured numerics match CPU (maxdiff ≤ 1e-6).
> - **rng (6)**: `rand`/`randn` (factory), `rand_like`/`randn_like` (like_factory), `randperm`
>   (factory; this torch version's schema has no generator argument), and `multinomial` (a new
>   `rng_dropgen` category that strips the trailing `Generator?`). The blocker was that gems'
>   `philox_backend_seed_offset(increment)` reads the empty `torch.cuda.default_generators`
>   (CPU-torch + cuda shim, len=0) and raises IndexError. At runtime,
>   `torch_fl/__init__.py._patch_flaggems_philox()` monkeypatches that function to inject a
>   fallback CUDA generator, unlocking all six at once. Measured: distributions are correct and
>   consecutive calls differ (the offset advances).
> - **Still excluded**: `i0_`, `zero`, `zero.out` (the gems kernel hard-asserts `tensor.is_cuda` /
>   "Input tensor must be on a CUDA device", which flagos as PrivateUse1 can never satisfy;
>   `_FLAGGEMS_ARITY_OVERRIDE` only records arity, so these actually land in
>   `FLAGGEMS_PYTHON_SKIP`), and `normal_`/`normal.*` (gems hardcodes `generator=None` and does
>   not forward it — an upstream bug).

The data comes from reconciling the per-branch rejection logic in `discover_flaggems_ops()`
(`scripts/codegen/codegen_ops.py`); the buckets match the actual codegen rejections exactly.

> **Important correction (measured 2026-07)**: the ①no_dispatcher bucket (88) is **mostly not a
> functional gap**. Probing them individually on real hardware under flagos
> (`FLAGOS_USE_FLAGGEMS=1`, 55 representative ops) gave **54 PASS, 0 numerical errors, 0 genuine
> crashes**. The reason is that these ops are `composite_implicit_autograd`: PyTorch decomposes
> them into leaf ops **above** the PrivateUse1 dispatch key (conv2d→convolution, divide→div,
> var→…), and those leaves are already routed. Adding a dispatcher for them is useless (it would
> never be hit — dead code) and potentially harmful. **"Not in the routing table" ≠ "does not
> work".**

| Bucket | Count | Cause in one line |
|---|---|---|
| ① no_dispatcher | 88 | **Mostly by design, not a gap**: composite decomposition to already-routed leaves; already works (see the correction above) |
| ② type_unsupported_kwarg | 13 | An argument type the generic caller cannot express (Generator?/Device?/Layout?/MemoryFormat?/Tensor?) |
| ③ varargs | 12 | gems signature is `(*args, **kwargs)`; arity cannot be introspected |
| ④ manual_skip | 12 | Crashes at runtime; excluded by hand (device assert / mandatory out / rng) |
| ⑤ name_mismatch | 2 | Trailing aten parameter names do not match gems' keyword-only parameter names |
| ⑥ Miscellaneous | 5 | foreach / optlist / arity-reordering traps |
| **Total** | **132** | |

---

## ① no_dispatcher — 88 operators (measured: most already work via leaf decomposition)

gems has an implementation, but the aten-side codegen **generated no dispatcher under that op
name**. torchgen's classification plus on-hardware verification splits these 88 into four groups
by "should a dispatcher be added":

| Subgroup | Count | Measured conclusion | Worth adding? |
|---|---|---|---|
| **composite_implicit_autograd** | 61 | Decomposed above the PrivateUse1 key into already-routed leaf ops; measured conv1/2/3d, divide, true_divide, var, square, clip, selu, pad, one_hot, hstack, vstack, isfinite, kron, diag, tile, absolute, arcsinh, etc. **all run with correct numerics** | **No.** A dispatcher would be dead code that is never hit |
| **no_cuda_kernel** | 12 | No CUDA leaf to reuse; measured alias_copy/t_copy/diag_embed/pixel_unshuffle/select_scatter/slice_scatter/select_backward/lift_fresh_copy/equal **also run** (via cpu_fallback or composite decomposition) | Only max_pool2d_backward is blocked, by an upstream flaggems max_pool2d **forward** bug (unrelated to this bucket) |
| **NOT_IN_YAML** | 9 | The op name is absent from this torch version's native_functions.yaml (alias / version drift); bitwise_left/right_shift, copysign, new_full.Tensor, nll_loss_nd_* measured **working** via equivalent leaves | **No**, there is no corresponding schema |
| **composite_explicit_autograd** | 6 | repeat / allclose / _to_copy / copy_ / index_put / index_put_ measured **working**; repeat.out is already generated | Possible in principle, but they already work; low value |

**Core conclusion**: the no_dispatcher bucket is barely a real gap. Of 55 representative ops
measured, 54 PASS; the only failure, `max_pool2d_backward`, is blocked by an upstream stride
parsing bug in flaggems' `max_pool2d_with_indices` **forward** and is not this bucket's
responsibility. This bucket's **priority should therefore drop to the lowest** — adding
dispatchers yields near-zero benefit.

(The full original list of 88 ops is retained below for lookup.)

```
__ior__.Scalar          __ior__.Tensor          __or__.Scalar
__or__.Tensor           _assert_async           _index_put_impl_
_to_copy                absolute                alias_copy
allclose                arcsinh                 arcsinh.out
arcsinh_                arctanh_                bitwise_left_shift
bitwise_right_shift     clip                    clip_
conj_physical           conv1d                  conv1d.padding
conv2d                  conv2d.padding          conv3d
conv3d.padding          copy_                   copysign
diag                    diag_embed              divide.Scalar
divide.Scalar_mode      divide.Tensor           divide.Tensor_mode
divide_.Scalar          divide_.Scalar_mode     divide_.Tensor
divide_.Tensor_mode     embedding_backward      equal
gather_backward         greater.Scalar          greater.Scalar_out
greater.Tensor          greater.out             hstack
index_put               index_put_              isclose
isfinite                kron                    lift_fresh_copy
log_sigmoid             margin_ranking_loss     max_pool2d_backward
new_full.Tensor         nll_loss_nd_backward    nll_loss_nd_forward
one_hot                 pad                     pixel_unshuffle
prelu                   quantile                relu6
repeat                  repeat_interleave.self_Tensor
repeat_interleave.self_int                      resolve_conj
resolve_neg             rms_norm                scaled_softmax_backward
scaled_softmax_forward  select_backward         select_scatter
selu                    selu_                   slice_scatter
square                  square.out              square_
t_copy                  tile                    true_divide.Scalar
true_divide.Tensor      true_divide_.Scalar     true_divide_.Tensor
var                     var.dim                 vstack
```

---

## ② type_unsupported_kwarg — 13 operators

gems accepts the arguments, but one argument type cannot be expressed by the generic caller.

**`Generator?` (7)** — random operators. PrivateUse1 has no default generator; same root cause as
rand/randn:

| op | gems qualname |
|---|---|
| `bernoulli_.float` | `bernoulli_.bernoulli_` |
| `exponential_` | `exponential_.exponential_` |
| `normal.Tensor_Tensor` | `normal.normal_tensor_tensor` |
| `normal.Tensor_float` | `normal.normal_tensor_float` |
| `normal.float_Tensor` | `normal.normal_float_tensor` |
| `normal_` | `normal.normal_` |
| `uniform_` | `uniform.uniform_` |

**`Device?` / `Layout?` / `MemoryFormat?` (5)** — factory metadata, but with no positional shape
argument to infer from, the factory caller does not apply:

| op | gems qualname |
|---|---|
| `full_like` | `full_like.full_like` |
| `ones_like` | `ones_like.ones_like` |
| `rand_like` | `rand_like.rand_like` |
| `randn_like` | `randn_like.randn_like` |
| `zeros_like` | `zeros_like.zeros_like` |

**`Tensor?` (1)**:

| op | gems qualname |
|---|---|
| `_flash_attention_forward` | `attention.flash_attention_forward` |

---

## ③ varargs — 12 operators

The gems function signature is `(*args, **kwargs)`, so `inspect.signature` cannot determine arity
and the op fails the arity safety gate (dropping a trailing argument would be a silent error).

```
_functional_sym_constrain_range_for_size    _upsample_nearest_exact1d
asinh_          digamma_        hardswish_      i0_
log1p_          logit_          sgn_            sinh_
zero            zero.out
```

---

## ④ manual_skip — 12 operators

These crash at runtime and are excluded by hand (`FLAGGEMS_PYTHON_SKIP`).

**device assert (8)** — gems asserts `device == "cuda"` internally and rejects PrivateUse1:

```
maximum   minimum   _safe_softmax   upsample_linear1d
upsample_nearest1d   upsample_nearest2d   upsample_nearest3d
_upsample_bicubic2d_aa
```

**required out kwarg (1)** — `mm.out`: gems' `mm_out(a, b, *, out)` requires out, which the
positional caller cannot supply.

**rng (3)** — `rand`, `randn`, `randperm`: gems reads `default_generators[device]` and raises
IndexError (PrivateUse1 has no default generator); `randperm` additionally asserts an int dtype.

---

## ⑤ name_mismatch — 2 operators

Arity-short: the trailing aten parameter names do not match gems' keyword-only parameter names,
so forwarding by name is impossible.

| op | gems qualname | aten trailing args | gems kwonly args | Cause |
|---|---|---|---|---|
| `_grouped_mm` | `group_gemm.group_mm` | `bias`, `out_dtype` | *(none)* | gems has no kwonly args; nowhere to put them |
| `multinomial` | `multinomial.multinomial` | `generator` | `gen` | Name mismatch (and it also hits the Generator? group) |

---

## ⑥ Miscellaneous — 5 operators

**foreach_tensorlist (2)** — the TensorList category is unsupported:

| op | gems qualname |
|---|---|
| `cat` | `cat.cat` |
| `stack` | `stack.stack` |

**special_optlist (1)** — a `Tensor?[]` index list:

| op | gems qualname |
|---|---|
| `index.Tensor` | `index.index` |

**arity_other (2)** — argument-reordering traps:

| op | gems qualname | Cause |
|---|---|---|
| `gather` | `gather.gather` | gems places `out=None` third, so aten's fourth argument would land in the out slot |
| `t_copy.out` | `t_copy.t_copy_out` | gems' out is a mandatory positional argument, so `npos > with_out` |

---

## Priority guidance (revised by the 2026-07 measurements)

- **~~no_dispatcher (88)~~ → lowest priority**: measurement proves this bucket is not a gap —
  composites decompose to already-routed leaves and run with correct numerics (54 of 55 probes
  PASS). Adding dispatchers would be dead code that is never hit; **not recommended**. Only if a
  specific no_cuda_kernel op genuinely needs a dedicated flaggems kernel should it be integrated
  one at a time through flaggems python (not by adding a CUDA dispatcher).
- **`*_like` Device?/Layout?/MemoryFormat? (5 originally, 3 done)** ✅ `zeros_like`/`ones_like`/
  `full_like` are integrated (commit 4906d22); `rand_like`/`randn_like` have no generator entry
  point and are excluded.
- **Random in-place ops (part of the original Generator? group, 3 done)** ✅ `uniform_`/
  `exponential_`/`bernoulli_.float` are integrated (by explicitly injecting a CUDA generator).
  `normal_`/`normal.*` are excluded because gems hardcodes `generator=None` and does not forward
  it.
- **rng factories (rand/randn/randperm/multinomial)** have no generator injection point (no
  generator argument in the signature); solving this in one shot requires registering per-device
  generators for PrivateUse1, which is work at the runtime layer.
- **varargs (12)** need an explicit arity table, maintained either upstream in gems or locally,
  before they can be integrated safely.
- **name_mismatch / arity_other / optlist / foreach (10)** each require a special case; low value.
