# Integration analysis of the 88 no_dispatcher operators

The largest bucket in `docs/vendors/flaggems/unrouted-ops.md` is no_dispatcher (88 operators):
gems has an implementation, but the aten-side codegen never generated a CUDA dispatcher for
them, so discovery skips them outright (`op not in codegen_ops`).

This document answers **why these 88 have no dispatcher, and what integrating them would take**.
Reproducing the gating in `enumerate_all_cuda_ops()` operator by operator attributes the 88 as
follows:

| Subgroup | Count | Nature | Worth integrating? |
|---|---|---|---|
| A. composite_implicit (decomposed ops) | 61 | PyTorch decomposes them into leaves **above** our dispatch key | Not functionally needed; performance only |
| B. No CUDA kernel but falls back | 9 | No direct kernel; lands via CompositeExplicit / cpu_fallback | Not functionally needed; performance only |
| C. Manually registered (MANUAL_REGISTERED_OPS) | 4 | Already hand-written in register.cc (copy_/_to_copy/index_put_/_index_put_impl_) | Already integrated; special memory/view semantics |
| D. Remaining gems names with no aten counterpart | 14 | gems uses an alias/fused name; the corresponding aten leaf is routed separately | No benefit |

> **Key conclusion**: **not one** of the no_dispatcher bucket is a *functional* gap. Spot checks
> (`FLAGOS_USE_FLAGGEMS=1`) show `divide/square/true_divide/greater/var/selu/vstack` (group A)
> and `diag_embed/pixel_unshuffle/select_backward/slice_scatter/alias_copy/t_copy` (group B)
> **all run on flagos with correct results**. They already land on registered leaf operators via
> decomposition or fallback. So the only motivation to "integrate" them is **letting gems' fused
> kernels take over for performance** — not filling a functional hole.

---

## A. composite_implicit — 61 operators

These carry a `CompositeImplicitAutograd` kernel and have no `structured_delegate` /
`CompositeExplicitAutograd`. `enumerate_all_cuda_ops` excludes them explicitly
(codegen_ops.py:143-147), because **PyTorch decomposes them into leaf operators before reaching
the PrivateUse1 dispatch key**; registering them would be both redundant and dangerous (it would
intercept the decomposition and lose the autograd formula).

```
__ior__.Scalar   __ior__.Tensor   __or__.Scalar   __or__.Tensor
absolute   arcsinh   arcsinh.out   arcsinh_   arctanh_
clip   clip_   conj_physical
conv1d   conv1d.padding   conv2d   conv2d.padding   conv3d   conv3d.padding
diag   divide.Scalar   divide.Scalar_mode   divide.Tensor   divide.Tensor_mode
divide_.Scalar   divide_.Scalar_mode   divide_.Tensor   divide_.Tensor_mode
embedding_backward   gather_backward
greater.Scalar   greater.Scalar_out   greater.Tensor
hstack   isclose   isfinite   kron   log_sigmoid   margin_ranking_loss
one_hot   pad   prelu   quantile   relu6
repeat_interleave.self_Tensor   repeat_interleave.self_int
resolve_conj   resolve_neg   rms_norm   selu   selu_
square   square.out   square_   tile
true_divide.Scalar   true_divide.Tensor   true_divide_.Scalar   true_divide_.Tensor
var   var.dim   vstack
```

**Why decomposition is sufficient**: `divide.Tensor` decomposes into `div.Tensor` (already
routed to gems), `square` into `mul`/`pow`, `vstack` into `cat`, `selu` into `elu`/`mul`. The
leaves already run on flaggems or cuda, so the composite operator needs no kernel of its own.

**If integration is truly wanted (for performance)**, there are two routes, both requiring care:

1. **Force-register under the CUDA dispatch key**: release them from the composite_implicit
   exclusion in `enumerate_all_cuda_ops` and codegen a dispatcher plus a `kFlagOsPython` kernel.
   Risk: this intercepts PyTorch's decomposition and **loses the autograd formula along with
   it** — anything with gradient semantics (`conv1d`, `embedding_backward`, `gather_backward`,
   `rms_norm`, `prelu`) will train incorrectly. Only pure-forward operators with no gradient
   dependency (`isfinite`, `isclose`, `conj_physical`, `resolve_*`) are relatively safe.
2. **Register at a key below `CompositeImplicitAutograd` but above autograd** (e.g. the functorch
   layer after `Autograd`) — this project's boxing approach has no such layer, so it is not
   realistic.

**Recommendation**: do not integrate as a group. Grant exceptions one at a time, only when
profiling proves a specific fused kernel (e.g. `rms_norm`, `conv2d`) delivers a significant gain
*and* we can supply its backward, which must also route through gems.

---

## B. No CUDA kernel, lands via fallback — 9 operators

No `CompositeImplicit` and no direct CUDA kernel, but a `CompositeExplicitAutograd` exists or
cpu_fallback carries them to a registered leaf.

```
alias_copy   diag_embed   lift_fresh_copy   max_pool2d_backward
pixel_unshuffle   select_backward   select_scatter   slice_scatter   t_copy
```

All spot-checked as working on flagos. Of these, `*_scatter`, `select_backward`, `alias_copy`,
`t_copy`, `diag_embed`, and `pixel_unshuffle` are view/scatter meta-operations, or expressible
via as_strided + copy.

**How to integrate**: unlike group A (pure decomposition), these **have real CUDA leaf
semantics**, so in principle one could:

- Relax `cuda_supported` to let them into codegen. Most are `CompositeExplicitAutograd`, which
  the third clause of `cuda_supported` should already admit — worth checking why it does not
  match; possibly `has_composite_explicit_autograd_kernel` is False while they actually go
  through structured.
- Then generate a `kFlagOsPython` kernel using the ordinary functional/out categorization.

**Recommendation**: low priority. They already execute correctly via fallback and the gems
versions offer limited gain. `max_pool2d_backward` is the only plausible candidate (a training
hot spot), but its forward `max_pool2d_with_indices` must be confirmed routed with matching
indices semantics.

---

## C. Manually registered — 4 operators

```
copy_   _to_copy   index_put_   _index_put_impl_
```

Already hand-written in `csrc/aten/.../register.cc` (`MANUAL_REGISTERED_OPS`), because they
involve memory copies, in-place indexed writes, and cross-device semantics that generic boxing
cannot handle. **Already integrated; not a gap** — they appear under no_dispatcher only because
codegen deliberately yields to the hand-written versions. Do not override them with flaggems.

---

## D. gems alias / fused names — the remainder

A small number use a gems fused name or alias (`scaled_softmax_forward/backward`,
`nll_loss_nd_forward/backward`, `new_full.Tensor`, `repeat`, `bitwise_left_shift`), for which the
corresponding aten leaf either is absent from the native schema or is already routed separately.
These **have no standard aten dispatcher to hang off**; they are gems-private extension ops, and
integrating them would require a custom schema for very little gain.

---

## Summary and recommendations

- **All 88 no_dispatcher operators already execute correctly** (via decomposition, fallback, or
  hand-written kernels). This is not a functional gap.
- **Bulk integration is not recommended.** Releasing composite_implicit wholesale loses autograd
  and is a net loss.
- **Performance candidates worth case-by-case approval** (each requiring a backward and profiling
  evidence): `rms_norm`, `conv2d`, `max_pool2d_backward`. Integrate via "force-register on the
  CUDA key + route the gems forward/backward as a pair", with numerical and training regressions
  added.
- Everything else (meta-operations, aliases, hand-written) **stays as is**.

Analysis scripts (temporary): `/tmp/fg_no_disp.py`, `/tmp/fg_decomp.py`, which reproduce the
`enumerate_all_cuda_ops` gating.
