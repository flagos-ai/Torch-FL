# Operator Support

This reference records measured operator coverage for torch-fl accelerator
backends. The current baseline measures the generic FlagGems Python routing
surface on four hardware platforms. It is an availability and correctness
survey, not a claim of complete PyTorch conformance, autograd coverage, or
performance quality.

The measurement unit is an active, unique, exact ATen overload such as
`sum.dim_IntList`. It is different from an OpInfo base operation, so historical
OpInfo totals such as 158 must not be compared with the 546-overload denominator
below.

Routing-table presence alone is not proof that an overload executes correctly.
Conversely, an overload without a direct route may still execute through a
composite decomposition or fallback. See the [Compatibility Matrix](compatibility.md),
[unrouted operator analysis](../vendors/flaggems/unrouted-ops.md), and
[no-dispatcher analysis](../vendors/flaggems/no-dispatcher-analysis.md) for those
separate concerns.

Historical evidence entries cite `tests/integration/test_amp.py` and the
per-vendor `tests/integration/test_profiler_*.py` modules under the names used
when the measurement was taken. Those suites are now the single cross-backend
contracts `tests/integration/test_amp_contract.py` (`-m amp`) and
`tests/integration/test_profiler_contract.py` (`-m profiler`); the recorded
results are unchanged.

## Verdicts

The manual survey first rejects synthesized invocations that are invalid on the
CPU reference. It then classifies each overload from the remaining valid cases:

| Verdict | Definition |
|---|---|
| `STRICT` | Every CPU-valid synthesized case passed on the target hardware. |
| `BASIC_ONLY` | At least one CPU-valid case passed, but one or more other valid cases failed. |
| `FAILED` | Valid cases existed and none passed. |
| `UNTESTED` | No CPU-valid synthesized case existed; this is neither a pass nor a failure. |

**Basic executable** is `STRICT + BASIC_ONLY`.

`PASS`, `INVALID_CASE`, `UNVERIFIABLE`, `ERROR`, `WRONG`, `CRASH`, and
`TIMEOUT` are case-level statuses, not additional operator verdicts.
`INVALID_CASE` and `UNVERIFIABLE` are excluded from support classification.

## Baseline Cohort

All hardware rows in this baseline use the same active route set and survey
methodology. These revisions identify the measured cohort; they do not describe
the current repository HEAD.

| Field | Value |
|---|---|
| torch-fl source | `fe2272b5fd1313eff00017c3f8242afe6c9a2cf6` |
| FlagGems source | `7fb49bad47116434961bfb2b912811716d383eaf` |
| Generic config | `torch_fl/configs/backends_flaggems.conf` |
| Generic config SHA-256 | `f97686deec8aa4863ecd04d359960804cbdf5862d27449e6345e3451512db9d8` |
| Active route-set SHA-256 | `8a1649e79ef7c419c050d65465c46dcf25575303c74d61dc194c5838ea847456` |
| Survey harness | `tests/manual/flaggems_overload_survey.py`, version 4 |
| Survey harness SHA-256 | `2354d4f76a6b37831492979dae25b9318cbe94fb48e08cdf100a4cab09cebd13` |
| FlagGems `_FULL_CONFIG` entries | 866 |
| Generated Python routes | 572 |
| Active surveyed routes | 546 |
| Forced CUDA fallbacks | 26 |
| Profiles per overload | 7 |

Full generation discovers 572 Python routes. The generic production
configuration activates 546 as `flagos_python` and forces 26 to CUDA fallback,
which explains the 546-route survey denominator.

## Hardware Summary

Rates use all 546 active routes as the denominator and are rounded to one
decimal place.

| Hardware | Total | STRICT | BASIC_ONLY | FAILED | UNTESTED | Basic executable | Basic rate | Strict rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA A100 | 546 | 348 | 54 | 46 | 98 | 402 | 73.6% | 63.7% |
| MetaX mc550 | 546 | 260 | 33 | 155 | 98 | 293 | 53.7% | 47.6% |
| PPU 810e | 546 | 347 | 54 | 47 | 98 | 401 | 73.4% | 63.6% |
| Hygon DCU bw1000 | 546 | 321 | 53 | 74 | 98 | 374 | 68.5% | 58.8% |

For every row, `STRICT + BASIC_ONLY + FAILED + UNTESTED = Total`, and
`Basic executable = STRICT + BASIC_ONLY`.

## Raw Case Evidence

These counts cover seven synthesized profiles per overload. They are case-level
data and therefore do not share the 546-overload denominator of the hardware
summary.

| Hardware | PASS | INVALID_CASE | UNVERIFIABLE | ERROR | WRONG | CRASH | TIMEOUT | Context poison |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA A100 | 2163 | 1309 | 0 | 184 | 152 | 14 | 0 | 0 |
| MetaX mc550 | 1597 | 1309 | 0 | 812 | 90 | 14 | 0 | 0 |
| PPU 810e | 2158 | 1305 | 0 | 184 | 154 | 21 | 0 | 0 |
| Hygon DCU bw1000 | 2016 | 1309 | 0 | 337 | 146 | 14 | 0 | 0 |

The bw1000 baseline excludes 108 initial records that failed before operator
execution because the child process could not load its MPI runtime. Exactly
those routes were rerun with the correct runtime environment; the corrected
result has 14 remaining `CRASH` cases, all return code `-11`.

## Reproducing and Updating the Report

Run the manual survey on each target hardware platform:

```bash
python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_flaggems.conf \
  --out /tmp/flaggems-overloads.json
```

When operator routing or implementation changes:

1. Run from an identified torch-fl revision with an identified FlagGems
   revision on every affected hardware platform.
2. Record the exact hardware model, run date, source revisions, configuration
   SHA-256, active route-set SHA-256, and harness version/SHA-256.
3. Keep the active route set fixed for cross-hardware comparisons. If cohorts
   differ, label that difference explicitly rather than presenting the rows as
   directly comparable.
4. Recompute the four overload verdicts centrally from raw cases. Do not treat
   `CRASH` or `TIMEOUT` as operator verdicts and do not infer support from a
   configured route.
5. Update both tables, verify their arithmetic, and append an update-history
   entry describing the affected hardware and evidence.
6. If hardware is unavailable, mark the affected row **not revalidated** and
   document the evidence gap in this report and the PR.

Keep the per-overload JSON as the auditable evidence. Do not expand this report
into a 546-row inventory; the aggregate tables are the maintained human-facing
record.

## Native Backend Route Changes

The generic FlagGems survey above does not exercise vendor-native routes such as
`ascend` or `gcu`. Native route changes are tracked here separately so they are
not misrepresented as part of the 546-overload FlagGems cohort.

### Enflame GCU S60 unified RNG parity routes (2026-08-24)

The GCU RNG route set went from 15 to 47 overloads so the platform answers the
same unified RNG contract the other backends do. The 32 newly routed overloads,
all declared in `HANDWRITTEN_OPS` in
[`scripts/codegen/codegen_gcu.py`](../../scripts/codegen/codegen_gcu.py) with the registration and
conf lines regenerated (idempotent on a repeat run):

- uniform family: `uniform_`, `rand`, `rand.generator`, `rand.out`,
  `rand.names_out`, `rand_like`, `rand_like.generator`, `rand_like.out`
- normal family: `normal_`, `normal.float_float`, `normal.Tensor_float`,
  `normal.Tensor_Tensor`, `randn.names_out`, `randn_like`, `randn_like.out`
- integer family: `randint`, `randint.low`, `randint.out`,
  `randint.low_out`, `randint_like`, `randint_like.low_dtype`,
  `randint_like.out`, `randint_like.low_dtype_out`, `randperm`,
  `randperm.out`, `random_.from`
- dropout and discrete: `native_dropout`, `native_dropout_backward`,
  `bernoulli_.Tensor`, `binomial`
- CPU-reference distributions (no topsaten entry point, seeded from the flagos
  generator exactly as Ascend does): `_standard_gamma`, `_sample_dirichlet`

Two correctness fixes came with the routing, both reported in
[issue #161](https://github.com/flagos-ai/Torch-FL/issues/161):

- **Generator identity.** Seeds are now reserved through
  `c10::flagos::ReserveSeed` instead of a private philox state. The previous code
  path went through `at::check_generator<at::CPUGeneratorImpl>`, which compares
  `device_type()` — kCPU for the flagos generator — so an explicit
  `torch.Generator(device="flagos")` was rejected outright, and generator-less
  draws advanced a second state that `torch.flagos.manual_seed`,
  `get_rng_state`, and `set_rng_state` never touched. All three now drive the
  same per-device stream. A flagos generator paired with a CPU tensor raises
  instead of silently redispatching.
- **`random_` default bounds.** `topsatenRandom`'s no-bound overload fills the
  dtype's full signed range; ATen's contract is `[0, iinfo(dtype).max]`. The
  default overloads now pass an explicit per-dtype upper bound (the same table
  Ascend uses) through the bounded overload.

Measured on S60 against the installed TopsRider release, running the CI steps
verbatim: `tests/integration/ops/test_rng_dispatch.py` is
`111 passed, 5 skipped, 1 xpassed` (was `7 failed, 104 passed, 5 skipped,
1 xpassed`), the full operator suite is `568 passed, 33 skipped, 4 xpassed`,
`test_factory_ops.py` is `46 passed`, and `test_amp.py` is `25 passed`.
The generic FlagGems cohort was **not revalidated** — the standard
`flaggems_overload_survey.py` harness selects only `flagos_python` overloads and
does not exercise these native routes.

### Enflame GCU S60 AMP routes (2026-08-24)

The GCU backend now routes both GradScaler unscale overloads through the
native `topsatenAmpForeachNonFiniteCheckAndUnscale` API when tensor lists are
contiguous, non-empty, same-device, and use supported dtypes. Unsupported
layouts and dtypes retain the CPU correctness fallback. The shared
`AutocastPrivateUse1` policy registration covers FP16/BF16 autocast.

Measured on S60 against the installed TopsRider release:
`tests/integration/test_amp.py` is `25 passed`. Three things were needed beyond
the unscale routes themselves:

- **float64 gate.** topsaten has no F64 kernels, so `TopsatenSupportsDtype` now
  excludes `at::kDouble` as well as `at::kLong`, sending float64 to the CPU
  fallback across all gated kernels. GradScaler needs this: it computes the
  inverse scale as `scale.double().reciprocal().float()`.
- **`.out` overload semantics.** topsaten writes `found_inf` for the `.out`
  overload; the CPU reference does not. The native path now passes a scratch
  flag so the observable contract matches other backends. GradScaler itself uses
  the in-place overload for overflow detection.
- **convolution routes.** `aten::convolution` dispatches PrivateUse1 to
  `convolution_overrideable`, which has no composite fallback, so conv2d raised
  `NotImplementedError` and the autocast lower-precision policy could not be
  exercised. Added `convolution_overrideable` (native `topsatenConvolution`,
  within 3.9e-6 of the CPU reference across stride/padding/dilation/group/bias
  variants) and `convolution_backward_overrideable` (CPU fallback; grads match
  the reference exactly).

`topsatenConvolutionBackward` is exported by `libtopsaten.so.3` but returns
`NOT_SUPPORT` for every input measured — fp32 and fp16, grouped and ungrouped,
padded and unpadded, with both the caller's `output_mask` and an all-true mask —
hence the CPU fallback for that one route. Switch it to native once a TopsRider
release implements it.

Not fixed here and still failing: `torch.neg` on uint8 and bool
(`topsatenNeg` returns `NOT_SUPPORT` and those dtypes are not routed to the
fallback). Pre-existing and outside the AMP contract.

### Enflame GCU S60 RNG routes (2026-08-17)

The GCU backend added native topsaten routes for the following RNG overloads:

- `bernoulli`, `bernoulli_.float`
- `exponential`, `exponential_`
- `multinomial`
- `poisson`
- `randn`, `randn.generator`
- `randn_like.generator`, `randn_like.generator_out`
- `randint.generator`, `randint.low_generator`
- `randperm.generator`
- `random_`, `random_.to`

Generator-less calls on these routes consume the same explicit topsaten
`{seed, offset}` stream used by FlagGems; explicit generators remain isolated.
Unsupported dtypes continue through the CPU fallback.

Targeted validation ran on an Enflame S60 with the installed TopsRider SDK:

- `tests/integration/ops/test_rng_dispatch.py`: `104 passed, 2 skipped, 1 xpassed`.
- Mixed route probe with `randn -> flagos_python` and `exponential_ -> gcu`:
  shared state advanced `(1234, 0) -> (1234, 8) -> (1234, 40)`; same-seed
  replay, different-seed sensitivity, and mixed-state replay all passed.

The standard `flaggems_overload_survey.py` harness is not applicable to these
native routes because it selects only `flagos_python` overloads. This targeted
RNG evidence does not revalidate the separate generic FlagGems support cohort.

### Ascend FSDP2 routes (2026-08-14)

The Ascend backend added or enabled the following FSDP2 paths:

- `_chunk_cat`
- `_chunk_cat.out`
- `_foreach_copy_`
- `cat.out`
- `split.Tensor`
- `split_with_sizes`
- `split_with_sizes_copy.out`

The standard `flaggems_overload_survey.py` harness cannot measure these routes:
it deliberately selects only `flagos_python` entries. Instead, these native
routes were exercised end-to-end on two physical Ascend 910 devices with CANN
9.0 and `ASCEND_RT_VISIBLE_DEVICES=2,3`:

- FlagCX collective test: passed all-reduce, broadcast, all-gather,
  reduce-scatter, and barrier.
- DDP test: passed forward, backward, gradient synchronization, and optimizer
  step (final losses `0.061326` and `0.118651`).
- FSDP2 test: passed parameter all-gather, gradient reduce-scatter, forward,
  backward, and optimizer step; each rank produced four finite gradient tensors
  (final losses `0.044212` and `0.063512`).

The generic FlagGems rows are **not revalidated** by this change because their
active route cohort is unchanged. The evidence gap is that there is no
per-overload synthesized survey for vendor-native Ascend routes; the available
evidence is the targeted FSDP2/DDP/collective workload described above.

### Ascend lazy math-bit view routes (2026-08-27)

The Ascend backend adds the two lazy math-bit view operators:

- `_conj`
- `_neg_view`

Both are metadata-only aliases that set PyTorch's Conjugate / Negative
dispatcher bit and leave storage untouched, so they route to the same
`at::native::` stride implementations as `alias` and `detach` rather than to an
ACLNN kernel. They need an explicit route because a view operator cannot reach
`cpu_fallback` -- storage is not shareable across devices -- so leaving them
unregistered made the dispatcher raise `_conj: backend not registered` for every
operation that resolves a math bit, including `copy_`, `clone`, `contiguous`,
`resolve_conj`, and `resolve_neg`.

Measured on Ascend 910 (CANN 9.0.0, torch 2.10.0+cpu) with
`ASCEND_RT_VISIBLE_DEVICES=1`:

- `tests/integration/test_math_bits_contract.py`: **5 passed, 7 skipped**. The
  five Negative-bit cases (clone, device-to-host copy, device-to-device copy,
  `resolve_neg`, and the plain-tensor fast-path regression) pass with bit-exact
  values.

The seven Conjugate cases skip: CANN 9.0.0 accepts complex *storage* but
provides no complex *compute*, so `_conj_physical` -- the operator that
materializes the bit -- has no ACLNN kernel, and `aclnnAdd`/`aclnnMul` reject
`ComplexFloat` and `ComplexDouble` outright. Complex dtypes remain outside the
Ascend cohort, unchanged from the 2026-08-18 dtype work below. The generic
FlagGems rows are **not revalidated** by this change because no FlagGems route
is affected; the evidence gap is that vendor-native Ascend view routes have no
per-overload synthesized survey, so the shared contract above is the evidence.

### Ascend AMP and dtype routes (2026-08-18)

The Ascend dtype work adds generated support for the PrivateUse1 AMP workflow
and corrects dtype behavior around the CANN capability boundary:

- `_amp_foreach_non_finite_check_and_unscale_`
- `_amp_foreach_non_finite_check_and_unscale.out`
- `_foreach_add_.List`
- Tensor-tensor binary promotion now uses PyTorch `result_type` semantics.
- Ascend float64 copies and casts preserve float64 instead of being clamped to
  float32.
- Ascend matmul-family float64 and unsupported integer inputs use the CPU
  fallback and return a correctly typed Ascend tensor.
- Ascend unary dtypes rejected by CANN use the CPU fallback; supported native
  paths remain unchanged.

Measured on Ascend 910 with CANN 9.0 and `ASCEND_RT_VISIBLE_DEVICES=2`:

- `tests/integration/test_amp.py`: **25 passed** (including both float16 and
  bfloat16 autocast, non-finite detection, and all GradScaler step/overflow
  paths).
- `tests/integration/test_dtype_coverage.py`: **174 passed**.
- The focused probe confirmed exact float64 round trips (including `1e300`),
  float16 + float32 -> float32 promotion, int16/uint8 negation parity, and
  float64 matmul parity through CPU fallback.

This is targeted dtype evidence for CANN 9.0, not a claim that every ACLNN
operator accepts every ACL dtype. Complex and quantized dtypes remain outside
this cohort.

### MUSA integer division: mudnn `TRUEDIV` promotion and FlagGems floor-divide tail store (2026-09-15)

[Issue #266](https://github.com/flagos-ai/Torch-FL/issues/266) reported two
distinct integer-division defects on MUSA, both reproduced on the eight-device
MTT S5000 host. They are fixed in the platform code generator, not with
handwritten kernels:

- **`int64 / int64` raised.** `a / b`, `torch.div(a, b)`, and `a.div_(b)` on
  integer tensors failed with
  `Failed: Unsupported binary mode: TRUEDIV, with left data type: INT64`. The
  generated `Binary` kernels derived `result_dtype = at::result_type(self, other)`,
  which for two integers is the integer type itself; mudnn's `TRUEDIV` has no
  integer overload, so the status was `NOT_SUPPORTED`. ATen's own semantics are
  different: TensorIterator builds the true-division kernel with
  `promote_integer_inputs_to_float`, so `int64 / int64` yields `float32` even
  though `at::result_type(int64, int64)` is `int64`.
- **Integer floor division returned a stale trailing element.** `a // b`,
  `a // 2`, `torch.floor_divide(a, b)`, and `a.clone().floor_divide_(b)` gave a
  wrong last value on inputs whose `numel` is not a power of two (wrong at
  `n = 3, 5, 6, 7, 9, 15, 17, 31, 33, 100`; correct at `n = 1, 2, 4, 8, 16, 32,
  64, 1024`). FlagGems' Triton kernel loses the final store on this stack.
  Float inputs were correct, the in-place form failed identically, and the same
  defect reached `torch.div(a, b, rounding_mode='floor'|'trunc')` and
  `a.div_(b, rounding_mode='floor')`, which route through the `div.Tensor_mode`
  / `div_.Tensor_mode` overloads.

**Generator changes** (`scripts/codegen/codegen_mudnn.py`, `scripts/codegen/gen_vendor_confs.py`):

- `_TRUEDIV_INT_TO_FLOAT` widens an integral `result_dtype` to
  `at::get_default_dtype_as_scalartype()` on the true-division path, so the
  computation happens in float and the result is cast back per ATen's
  `result_type` contract. `_TRUEDIV_INT_TO_FLOAT_IF_UNROUNDED` applies the same
  widening to the `*_mode` categories but only when `rounding_mode` is absent:
  `'floor'` and `'trunc'` are defined on integers and must keep `int64`. This
  is measured CPU behaviour, not an inference — `torch.div(a, b,
  rounding_mode=None)` on `int64` returns `float32` while `rounding_mode='floor'`
  returns `int64`.
- Two new template categories, `binary_mode` and `binary_inplace_mode`, generate
  `div.Tensor_mode` and `div_.Tensor_mode` against the native kernel. The mudnn
  mode comes from ATen's runtime `rounding_mode` string through
  `musa_ops::SetMudnnDivMode` (`nullopt -> TRUEDIV`, `"floor" -> FLOORDIV`,
  `"trunc" -> TRUNCATEDIV`). ATen validates that string before dispatch
  (`div expected rounding_mode to be one of None, 'trunc', or 'floor'`), so the
  helper's final arm is only there to keep it total. The Scalar spellings
  (`torch.div(a, 2, rounding_mode='floor')`) decompose into the Tensor overloads
  before dispatch, so no separate Scalar template is needed.
- `floor_divide_.Tensor` is added to the native `OPS` table.
- `NATIVE_TRITON_GAPS["musa"]` gains four entries — `div.Tensor_mode`,
  `div_.Tensor_mode`, `floor_divide`, `floor_divide_.Tensor` — so
  `gen_vendor_confs.py` routes them to `musa` and
  `codegen_musa_flaggems.py` drops them from the FlagGems registration.

**Route delta.** MUSA `flaggems` 468 -> **464**, `musa` 47 -> **51**, `none`
1521 unchanged (2036 routable ops). The registered-op set is unchanged at 518:
three overloads moved from `musa_flaggems_register.inc` (362 -> 359 `m.impl`
lines) to `musa_register.inc` (156 -> 159). The `int64` in-place true-division
forms keep ATen's own error, `result type Float can't be cast to the desired
output type Long`, which the in-place prologue's `c10::promoteTypes` +
`c10::canCast` check reproduces exactly — measured byte-identical on CPU.

**Measured on the MTT S5000 host** with CPU PyTorch 2.10.0, mudnn v3300,
FlagGems `4d9c34775` (5.4.0rc2.post1+g4d9c34775) and flagtree
`0.6.2a3+mthreads3.6` (Triton 3.6, backend `mthreads`):

- A CPU-parity probe covering 59 integer and float division cases — out-of-place,
  in-place, scalar and tensor operands, both rounding modes, negative operands,
  `out=`, broadcasting, and `floor_divide` at
  `n = 2, 3, 4, 5, 7, 8, 15, 17, 33, 100` — was run against both the fixed tree
  and a second worktree built at the base commit (`6b978c0`). Before:
  **39 exact, 7 float-approximate, 13 mismatches**. After: **43 exact,
  14 float-approximate, 1 error-text match, 1 mismatch**. Every integer
  floor-division and `rounding_mode` case is exact, and the in-place `int64`
  true-division case reproduces ATen's own
  `result type Float can't be cast to the desired output type Long` byte for
  byte. Two qualifications, both measured: (a) the 14 float-approximate cases are
  `truediv` results differing from CPU by exactly one float32 ULP (`5.960e-08`)
  at `n = 5, 7, 8, 15, 17, 33, 100`, and the pure-float spellings — which never
  touched the FlagGems floor-divide kernel — show the identical `5.960e-08` on
  the base tree, so this is mudnn `TRUEDIV` arithmetic versus CPU libm and
  predates the change; (b) the remaining mismatch is the probe's own `out=`
  harness passing CPU tensors to a Triton path and raising identically on both
  trees, not a property of the operators.
- `FLAGOS_LOG_DISPATCH=1` confirms the routes at runtime: `div.Tensor`,
  `div.Tensor_mode`, `div_.Tensor`, `floor_divide`, and `floor_divide_.Tensor`
  all resolve to `-> musa`.
- The two defects are independent, and the routing half is causal. Pinning the
  four rerouted overloads back onto FlagGems with `FLAGOS_OP_*` reproduces the
  trailing-store loss exactly and nothing else: at `n = 3`, `a // b`, `a // 2`,
  `torch.floor_divide(a, b)`, both `rounding_mode` values, and both in-place
  spellings return `[5, 5, 0]` where CPU returns `[5, 5, 6]`, while true division
  — never on that kernel — stays correct. All ten cases are correct on the
  shipped route.
- The full `.github/configs/musa.yml` manifest run locally: dispatch
  **104 passed, 1 skipped**; factory **46 passed**; AMP **27 passed**;
  math-bits **12 passed**; profiler **10 passed, 1 skipped, 1 xpassed**;
  operator cohort **493 passed, 1 skipped, 513 deselected, 2 xfailed,
  1 xpassed**; RNG **80 passed, 37 deselected**.
- Generator idempotency: `codegen_mudnn.py` run twice produces byte-identical
  `musa_kernels.cc`, `musa_register.inc` and `musa_flaggems_register.inc`;
  `codegen_musa_flaggems.py --check` reports "is up to date";
  `gen_vendor_confs.py --check` is clean for MUSA.

**Evidence gap.** `tests/manual/flaggems_overload_survey.py` cannot measure this
change. The harness selects overloads whose conf value is `flagos_python`, and
the four rerouted overloads are precisely the ones that are no longer on that
route; the unified per-platform confs also spell the FlagGems route `flaggems`,
and the `backends_flaggems.conf` the harness was written against was removed by
`d0e2d1a`. The evidence above is targeted CPU-parity probing plus the full CI
manifest, not a synthesized overload survey. The generic FlagGems baseline rows
are unchanged by this work and are **not revalidated**; no FlagGems route was
altered for any other platform.

**Two pre-existing conditions, unchanged by this work.** Three assertions in
`tests/integration/ops/test_flaggems_conf_consistency.py`
(`test_every_conf_op_maps_to_a_dispatcher`, `test_no_orphan_flagos_python_kernels`,
`test_counts_match`) fail against the pristine conf as well, on `mm`/`bmm`/`addmm`
dispatcher drift in `csrc/aten/generated/` that this change does not touch. And
mixed-device operands on the `*_out` overloads (`mul.out`, `add.out`, `div.out`)
fail generically for every `flaggems`-routed op on this stack; both operands must
be on `flagos`. Neither is in scope here.

### MUSA FlagGems routing restored, in-place arithmetic routed back to mudnn (2026-09-14)

The MUSA FlagGems registration generator was restored
(`scripts/codegen/codegen_musa_flaggems.py` -> `csrc/aten/backends/musa/generated/musa_flaggems_register.inc`,
included from `csrc/aten/register.cc`), so all 482 FlagGems Python ops are again
registered on MUSA's PrivateUse1 device and the wrappers route through the
FlagGems Python dispatcher slot. MUSA moves from 158 to **515 registered ops**
and from 122 to **468 `flaggems` routes**; `musa` route count goes 36 -> 47 and
`none` 1878 -> 1521. The restored path is the same one the `d0e2d1a` full-coverage
unification assumed: without it, `FLAGGEMS_PYTHON_OPS` was a coverage ceiling
MUSA could not reach.

**Ops that do not stay on FlagGems.** FlagGems is not patched anywhere. Fourteen
ops are listed in `NATIVE_TRITON_GAPS["musa"]` and route back to the mudnn native
kernel, with three distinct root causes:

- **bf16 wrapped-number promotion (11 ops).** `add.Tensor`, `sub.Tensor`,
  `div.Tensor`, and their in-place forms, plus `mul_.Tensor`. ATen boxes a
  Python-float operand into a float64 0-dim tensor (`is_wrapped_number`);
  FlagGems' pointwise promotion does not honour that flag, promotes the result to
  fp64, and mthreads' LLVM lowering declares `llvm.musa.float2bfloat16(float)`
  with no double overload. The four in-place entries are the ones that matter at
  runtime: ATen boxes `add_.Scalar` — and `_foreach_add_`, and therefore AdamW's
  foreach step — onto `add_.Tensor`, so routing only the out-of-place op leaves
  every in-place caller on the failing kernel.
- **`randn`, `randn_like`.** FlagGems crashes unpacking generator state. The
  native muRAND routes already exist and were measured on 2026-08-17.
- **`sort`, `sort.stable`.** FlagGems' radix sort casts its histogram to uint32
  internally, and mudnn's `Unary::CAST` has no UInt16/32/64 case, so the cast
  raises before the sort runs. mudnn's own sort is a real kernel; argsort and
  msort decompose onto sort, so one entry covers all four.

`_conj`, `index_add`, and `index_add_` are in the same gap set but route to
`none`: mudnn has no kernel for them either, so the call reaches ATen's CPU
fallback rather than a registered-but-incorrect dispatcher slot. `_conj` is a
contract case, not a compile one — ATen's `conj` is a lazy view that sets the
Conjugate bit, and FlagGems materializes it, which breaks
`tests/integration/test_math_bits_contract.py`'s `is_conj()` assertion.

**Measured on the eight-device MTT S5000 host** with CPU PyTorch 2.10.0, mudnn
v3300, FlagGems `4d9c34775` (5.4.0rc2.post1+g4d9c34775) and flagtree
`0.6.2a3+mthreads3.6` (Triton 3.6, backend `mthreads`), running every group of
`.github/configs/musa.yml` locally:

- `tests/integration/ops/test_musa_dispatch.py -m musa`: **104 passed, 1 skipped**.
- `tests/integration/test_factory_ops.py`: **46 passed**; `test_amp_contract.py -m amp`:
  **27 passed**; `test_math_bits_contract.py -m math_bits`: **12 passed**;
  `test_profiler_contract.py -m profiler`: **10 passed, 1 skipped, 1 xpassed**.
- The operator cohort in a wheel-only workspace: **490 passed, 2 skipped, 512 deselected,
  2 xfailed, 1 xpassed**; `test_rng_dispatch.py -m main_ops`: **80 passed, 37 deselected**.
  The cohort is run with `FLAGOS_USE_FLAGGEMS=1`, as `.github/scripts/set_env_musa.sh`
  sets it; without that variable the `flaggems`-marked tests skip rather than run.
- Routing was confirmed at runtime with `FLAGOS_LOG_DISPATCH=1`: `add_.Scalar`,
  `_foreach_add_.Scalar`, `sub_.Scalar`, `mul_.Scalar`, and `div_.Scalar` all
  resolve to `[flagos dispatch] <op>.Tensor -> musa`, and the numeric results match
  the CPU reference.

**Root-cause evidence for the routes.** The bf16 failure was reproduced locally
by forcing the route back with `FLAGOS_OP_add__Tensor=flaggems`:
`test_autocast_fp32_policy[dtype1]` then fails with
`RuntimeError: failed to translate module to LLVM IR ... intrinsic call operand #0
has type double but "llvm.musa.float2bfloat16" expects float`, and passes with the
shipped `add.Tensor = musa` route. That is the same failure the remote MUSA CI
reported on the pre-fix revision of this branch.

`tests/integration/ops/test_flaggems_conf_consistency.py` was repointed from the
deleted `torch_fl/configs/backends_flaggems.conf` to `scripts/codegen/backend_coverage.py`,
which is where `d0e2d1a` moved `FLAGGEMS_PYTHON_OPS`. Three of its assertions
(`test_every_conf_op_maps_to_a_dispatcher`, `test_no_orphan_flagos_python_kernels`,
`test_counts_match`) still fail on a pre-existing drift: `addmm` and `bmm` are
listed in `FLAGGEMS_CPP_OPS` while their dispatchers are registered with
`Backend::kFlagGems`. The same three fail at `400cf865`, before `d0e2d1a`, so the
drift is not introduced here. It is invisible to CI because the wheel-only
workspace has no `scripts/` or `csrc/` and the module skips.

### MUSA native empty-tensor handling (2026-08-30)

The native mudnn kernels now handle zero-element tensors without a CPU fallback. mudnn v3300 rejects zero-element operands for its Unary, Binary, and Reduce modes, returning `NOT_SUPPORTED`; the generated kernels therefore return an already device-allocated empty output without launching. Whole-tensor `sum`, `mean`, and `prod` additionally fill their CPU-defined identities (`0`, `nan`, and `1`) on the device when the input is empty. This covers the zero-length `narrow` autograd path from issue #214, where `square().sum()` previously failed in the pow kernel and then in the reduction.

Measured on the eight-device Moore Threads MTT S5000 host with CPU PyTorch 2.10.0 and mudnn v3300:

- `tests/integration/ops/test_pow_dispatch.py -m anyplatform`: **22 passed, 4 deselected**; coverage includes both `pow.Tensor_Scalar` and `pow.Tensor_Tensor` empty outputs, empty broadcasting, the zero-length narrow backward path, and non-empty parity.
- `tests/integration/ops/test_narrow_dispatch.py -m anyplatform`: **17 passed**, including `test_narrow_backward_edge_cases[1-2-0]`, which is no longer deselected in the MUSA CI manifest.
- `tests/integration/ops/test_musa_dispatch.py -m musa`: **89 passed**.
- The CI operator cohort (excluding the separately triaged RNG file and the known float64 `mm` gap) reached **480 passed, 14 skipped, and 3 xpassed**; three existing FlagGems configuration-consistency assertions failed because they inspect unrelated generic FlagGems routes, not MUSA native kernels.

The empty-output path stays on `flagos:0` and preserves shape and dtype; no host round trip is used. The MUSA operator support cohort is otherwise unchanged.

### MUSA native RNG routes (2026-08-17)

The MUSA route configuration includes native muRAND/mudnn implementations for the core RNG families (`rand`, `randn`, `rand_like`, `randn_like`, `randint`, `normal_`, `uniform_`, `random_`, and native dropout). They share the authoritative per-device PrivateUse1 generator with the optional FlagGems Philox bridge. `randperm` and unsupported distribution overloads remain on CPU fallback and are not counted as native support.

These native routes were measured on an eight-device Moore Threads MTT S5000 host. Device 0 reported capability 3.1, 60 multiprocessors, and 85,813,358,592 bytes of memory. With CPU PyTorch 2.10.0 and the installed `/usr/local/musa` toolkit (`mudnn` v3300):

- `tests/integration/ops/test_rng_dispatch.py`: the shared RNG suite covers same-seed reproducibility, `torch.manual_seed`, `torch.flagos.manual_seed`/`manual_seed_all`, state round trips, explicit generators, integer/out/like variants, full-width int64 ranges, `[0, 1)` uniform bounds, native dropout forward/backward, shared native/FlagGems reservation ordering, and per-device sequence isolation. MUSA-specific generator and reservation cases are selected through the `musa` mark in this same file.
- `tests/integration/ops/test_musa_dispatch.py`: **89 passed**.
- `tests/unit/test_vendor_routing.py` plus `tests/unit/test_musa_rng_bridge.py`: **24 passed**; the bridge unit test remains focused on MUSA FlagGems patching rather than duplicating integration coverage.

The target cohort is the available MTT S5000 host; no S6000 claim is made.

The MUSA hybrid config adds seven non-overlapping FlagGems Python routes (`all`, `all.dims`, `any`, `any.dims`, `index_add`, `index_add_`, and `repeat_interleave.Tensor`) while retaining native RNG precedence. They were execution-validated with FlagGems 5.0.2 and the vendor `flagtree-0.5.0+mthreads3.1` wheel (Triton 3.1.0, backend `mthreads`; SHA-256 `197b0c6954ad8b3edef51138311a8c4f3aea75b90ba0f69d3c2fda95a76b6b1b`). `tests/integration/ops/test_musa_flaggems.py` passed **2 tests in 5.33 seconds** on `flagos:0`: instrumentation observed every configured wrapper, it compares selected route outputs against CPU, includes duplicate-index `index_add`, checks in-place `index_add_`, and launches FlagGems `randn` on `flagos:0` between native `rand` calls. Repeating after `torch.flagos.manual_seed(20260817)` reproduced all outputs and confirmed the two shared C++ generator reservations. Native and hybrid suites must run in separate pytest processes because the C++ `BackendTable()` caches the backend configuration on first use. The generic installed Triton 3.7.1 is not MThreads-capable and is not execution evidence.

### Full-coverage vendor configurations (2026-09-10)

The MUSA, GCU and Ascend configurations became **full-coverage**: all 2036 ops
torch_fl can route are listed exactly once under one of four keys, with routing
priority `flaggems_cpp > flaggems > <vendor> > none`. Previously these files were
sparse, so "absent from the file" and "known to be unsupported" looked identical
and support could not be counted from the configuration. Omission was never a
safe way to say "unsupported" either: `GetBackendForOp()` returns `kFlagOs` on a
table miss, so an unlisted op claimed a FlagGems kernel by default.

What a configuration may claim is bounded by what the platform registers on
PrivateUse1. FlagGems coverage is measured on CUDA and is only a **ceiling**: the
FlagGems wrapper is reached *through* the op's PrivateUse1 registration, so an op
the platform does not register cannot reach any kernel, FlagGems included. Each
platform's registration set is therefore read from the generated
`*_register.inc` files that `csrc/aten/register.cc` includes — the same list the
compiler sees — and every accelerated route is required to be in it.

| Platform | flaggems | vendor | none | Registered / total |
|---|---:|---:|---:|---:|
| Ascend 910 | 248 | 126 | 1662 | 374 / 2036 (18%) |
| Enflame GCU S60 | 88 | 64 | 1884 | 152 / 2036 (7%) |
| MTT S5000 (MUSA) | 122 | 36 | 1878 | 158 / 2036 (7%) |

**The MUSA row was superseded on 2026-09-14** — the MUSA FlagGems registration
generator was restored, taking MUSA to 468 `flaggems` / 47 `musa` / 1521 `none`
and 515 registered ops. See "MUSA FlagGems routing restored, in-place arithmetic
routed back to mudnn (2026-09-14)" below. It moved again on 2026-09-15 to
464 `flaggems` / 51 `musa` / 1521 `none`, with the registered-op set unchanged
at 518. See "MUSA integer division: mudnn `TRUEDIV` promotion and FlagGems
floor-divide tail store (2026-09-15)". The Ascend and GCU numbers are the ones
committed in their shipped configurations; re-running `gen_vendor_confs.py`
today would move Ascend to 243 `flaggems` / 131 `ascend`, a pre-existing drift
that predates this work and is out of scope here.

The FlagGems count differs per platform because it is now the intersection of the
shared coverage set with that platform's registrations, not the shared set
itself. Ops the vendor also implements but that FlagGems covers are routed to
FlagGems by priority; a trailing `# <vendor>` annotation records the kernel so it
stays recoverable and `ALL_USE_VENDOR` can find it (248 such kernels on Ascend,
88 on GCU, 115 on MUSA — MUSA's remaining 7 FlagGems routes come from
`musa_flaggems_register.inc`, which registers wrappers without native kernels
behind them).

`none` means no accelerated implementation on that platform. It is honest only
where registration *skips* the op, so the call reaches `cpu_fallback` instead of
a registered-but-empty dispatcher slot. That is what limits generation to these
three platforms: **MetaX and Tsingmicro register the full generated op list** via
the `#else` branch of `csrc/aten/register.cc`, so a `none` entry there would
reach the dispatcher and raise instead of falling back. Their configurations stay
hand-written and sparse; MetaX's supported path is its boxing configurations.
Relative to the sparse files this is not a regression for MUSA/GCU/Ascend — an
absent op reached the same fallback, it just could not be counted.

The two boxing configurations (`metax`, `dcu`) are generated in the same
full-coverage shape, but their fallback key is `cuda` and they contain no `none`:
a CUDA-compatible platform can box every op. Their per-op key distribution is
byte-for-byte equivalent to the previous revision — only the shape and key
spellings changed.

The `flaggems_cpp` key is emitted **only** in `backends_metax.conf`. That slot
is `Backend::kFlagOs`, registered behind `#ifdef FLAGOS_FLAGGEMS_CPP`, which is
defined only for a `FLAGGEMS_KERNEL=ON` build; `CMakeLists.txt` force-sets it
`OFF` for ascend, dcu, musa, bpu, tsingmicro and non-boxing metax. When a build
without that slot reads a `flaggems_cpp` entry, `Dispatcher::GetFn` degrades to
the boxing kernel instead of raising, so the file is safe for both opt-in and
plain boxing builds. No coverage is lost: the C++ op set is a subset of the
Python set, so those ops take the Python path to the same FlagGems kernels and
only the GIL-free entry point is given up.

**Evidence status: not revalidated on any accelerator.** No route was measured on
hardware for this change. `tests/manual/flaggems_overload_survey.py` could not
run: the work was done on a CPU-only host whose Triton 3.7.1 exposes only the
`amd` and `nvidia` backends, so `import flag_gems` fails there. This applies to
every platform named above — Ascend 910, Enflame GCU S60, MTT S5000, MetaX C550
and Hygon DCU data in the sections above predate this change and were **not**
re-measured against it.

No coverage set is newly measured either. The vendor sets are read from the
committed codegen artifacts, and the boxing platforms' measured facts (each
platform's Triton gap set, the MetaX 17-of-18 C++ subset that keeps `mm` boxed)
are recovered from the configurations the generator rewrites. That is what makes
the change auditable by regeneration rather than by hardware. Confirmed
mechanically:

- `scripts/codegen/gen_vendor_confs.py` twice in a row produces an empty diff and
  `--check` exits 0, so no measured fact is lost across the round trip.
- Routing equals registration exactly on all three generated vendors
  (Ascend 374/374, GCU 152/152, MUSA 158/158, with no op routed outside its
  registration set and none registered-but-left-`none`).
- `tests/unit/test_gen_vendor_confs.py`: 27 passed, pinning the four-key
  priority, the registration gate, the annotation round trip, the `flaggems_cpp`
  build gate, and that MetaX/Tsingmicro stay hand-written.
- Per-op key distribution on `metax` and
  `dcu` is unchanged from the previous revision.

Before any of these platforms is described as validated under the new
configurations, rerun the survey on that hardware and replace this entry's status.

## FlagGems Route Removals

The 10 stale-qualname routes fixed on 2026-08-31 are omitted from this section
because the source cohort is not hardware-revalidated; the update history records
the routing change and its focused MetaX evidence. The four-platform summary tables
above still describe the baseline cohort; affected rows are **not revalidated**
against the reduced route set because A100, mc550, and 810e hardware is unavailable
to this change.

### `index_select` rerouted to CUDA boxing (2026-08-19, Hygon DCU)

`index_select` was moved from `flagos_python` to `cuda` in all FlagGems
configurations (`backends_flaggems.conf`, `backends_flaggems_cpp.conf`,
        `backends_metax.conf`, `backends_dcu.conf`,
`backends_dcu.conf`; `index_select.out` was already CUDA-routed).
The generic configuration now activates 545 FlagGems Python routes with 27
forced CUDA fallbacks (SHA-256
`14b4c64c0d2684b126fe06c6f39f42b62c571a94813a684cb63d4a09b909b60c`).

Reason: the FlagGems triton launch is not stream-ordered against the flagos
(PrivateUse1) stream that produced the index tensor. Under a busy allocation
stream (HF cached beam search, `DynamicCache.reorder_cache` ->
`index_select(0, beam_idx)`) the kernel can read a stale index entry, fail its
`indices < N` validity mask, and leave the output column unwritten, poisoning
KV caches with recycled `torch.empty` bytes and NaNs. This is a launch
integration race, not a kernel arithmetic defect.

Targeted evidence on Hygon DCU bw1000 (FlagGems 5.4.0.dev0 hygon build,
DTK triton, harness v4):

- `flaggems_overload_survey.py --ops index_select` against a conf that still
  routes `index_select = flagos_python`: **STRICT** (all CPU-valid synthesized
  cases pass standalone), confirming the kernel math is correct and the hazard
  is the missing stream ordering, which the synthesized-case harness does not
  reproduce.
- Failing HF UT nodes on the FlagGems route before the reroute:
  `T5ModelTest::test_generate_with_past_key_values` (deterministic),
  `Qwen3ModelTest::test_generate_from_inputs_embeds_1_beam_search` (flaky,
  ~1/3), `Gemma3Vision2TextModelTest::test_generate_from_inputs_embeds_1_beam_search`
  (deterministic). After the reroute all three pass (Qwen3 verified 3/3).
- Minimal reproducer (tiny T5, `num_beams=2, use_cache=True`): NaN logits from
  decoder step 1 before the reroute, 3/3 clean after.

The generic four-platform FlagGems rows are **not revalidated** by this
change; the evidence gap is that no A100/mc550/810e re-survey was run, and the
545-route denominator applies only from this change forward. Note that PR #108
(`native_layer_norm_backward` rerouted to CUDA boxing, 2026-08-14) previously
reduced the same cohort 546 -> 545 without a re-survey; the baseline tables
therefore describe the original 546-route cohort, not the current HEAD.

### MetaX AMP routes (2026-08-21)

The shared `AutocastPrivateUse1` registrations now have explicit MetaX boxing
coverage. They use the same PyTorch policy groups as CUDA and redispatch through
the existing PrivateUse1-to-CUDA boxing kernels; no handwritten MetaX operator
was added or rerouted.

Measured on MetaX C550 with MACA 3.8.0 in boxing mode:

- `tests/integration/test_amp.py`: **25 passed**.
- The suite covered FP16 and BF16 lower-precision, FP32, optional-dtype, and
  promote policies; nested autocast state; BCE fallthrough; non-finite unscale;
  finite scale growth; overflow backoff; and a forward/backward optimizer step.

The generic FlagGems route cohort is unchanged and was **not revalidated** by
this work. The AMP result does not establish support for the legacy handwritten
MetaX kernel mode or for additional MACA releases and devices.

## Update History

| Date | Hardware | Cohort | Change | Evidence |
|---|---|---|---|---|
| 2026-09-15 | MTT S5000 (8 devices) | MUSA integer division (issue #266) | Fixed two integer-division defects in the generator, not with handwritten kernels. `int64 / int64` raised `Unsupported binary mode: TRUEDIV, with left data type: INT64` because the generated kernels took `result_dtype` from `at::result_type` (int64) while ATen promotes integer true division to float32; `_TRUEDIV_INT_TO_FLOAT` now widens integral results, guarded on `!rounding_mode.has_value()` so `'floor'`/`'trunc'` keep int64. Integer `//`, `floor_divide`, and `floor_divide_` silently lost the trailing element on non-power-of-two `numel` in FlagGems; new `binary_mode` / `binary_inplace_mode` categories plus the `floor_divide_.Tensor` native entry route `div.Tensor_mode`, `div_.Tensor_mode`, `floor_divide` and `floor_divide_.Tensor` through mudnn `FLOORDIV`/`TRUNCATEDIV`/`TRUEDIV` via `SetMudnnDivMode`. MUSA `flaggems` 468 -> 464, `musa` 47 -> 51, `none` 1521; registered-op set unchanged at 518 (three overloads moved from the FlagGems registration to the native one). FlagGems is not patched. Other platforms are **not revalidated** and no FlagGems route changed for them. | 59-case CPU-parity probe on `flagos:0` run against both this tree and a base-commit worktree (out-of-place, in-place, scalar and tensor operands, both rounding modes, negatives, `out=`, broadcasting, and `floor_divide` at `n = 2,3,4,5,7,8,15,17,33,100`): 39 exact / 7 float-approximate / 13 mismatches before, 43 exact / 14 float-approximate / 1 error-text match / 1 probe-harness mismatch after. Integer floor division and every `rounding_mode` case exact; the float-approximate cases are true division one float32 ULP from CPU and reproduce identically on pure-float inputs on the base tree (pre-existing mudnn `TRUEDIV` arithmetic, not this change). `FLAGOS_LOG_DISPATCH=1` shows all five overloads on `-> musa`; pinning the four rerouted overloads back onto FlagGems via `FLAGOS_OP_*` reproduces the tail loss (`[5, 5, 0]` for `[5, 5, 6]` at n=3) and leaves true division correct, isolating the routing fix causally. Full `.github/configs/musa.yml` run locally: dispatch 104 passed/1 skipped, factory 46 passed, AMP 27 passed, math-bits 12 passed, profiler 10 passed/1 skipped/1 xpassed, operator cohort 493 passed/1 skipped/513 deselected/2 xfailed/1 xpassed, RNG 80 passed/37 deselected. Generator idempotent (two runs byte-identical; `codegen_musa_flaggems.py --check` and `gen_vendor_confs.py --check` clean for MUSA). `flaggems_overload_survey.py` cannot measure these routes: it selects `flagos_python` entries, and the rerouted overloads are exactly the ones that left that route — evidence gap recorded in the section above. Three pre-existing `test_flaggems_conf_consistency.py` failures (`mm`/`bmm`/`addmm` dispatcher drift) reproduce byte-identically against the pristine conf. |
| 2026-09-14 | MTT S5000 (8 devices) | MUSA FlagGems routing and in-place arithmetic fallback | Restored the MUSA FlagGems registration generator, taking MUSA from 158 to 515 registered ops and from 122 to 468 `flaggems` routes (`musa` 36 -> 47, `none` 1878 -> 1521). Moved 14 ops into `NATIVE_TRITON_GAPS["musa"]` so they fall back to mudnn instead: `add/sub/div.Tensor` and their in-place forms plus `mul_.Tensor` (bf16 wrapped-number promotion reaches `llvm.musa.float2bfloat16` with a double operand), `randn`/`randn_like`, `sort`/`sort.stable`, and `_conj`/`index_add`/`index_add_`, which route to `none` because mudnn has no kernel for them. FlagGems is not patched. Ascend, GCU, DCU, MetaX and PPU rows are **not revalidated** by this change and no FlagGems route was altered for them. | Every group of `.github/configs/musa.yml` run locally on hardware: dispatch 104 passed/1 skipped, factory 46 passed, AMP 27 passed, math-bits 12 passed, profiler 10 passed/1 skipped/1 xpassed, operator cohort 490 passed/2 skipped/512 deselected/2 xfailed/1 xpassed, RNG 80 passed/37 deselected. The bf16 gap was reproduced causally with `FLAGOS_OP_add__Tensor=flaggems`, which reproduces the remote CI's `failed to translate module to LLVM IR` on `test_autocast_fp32_policy[dtype1]` and passes on the shipped route. Three `flaggems`-marked dispatch-log tests that hard-coded `flagos_python`/`cuda` were rewritten to read the route from the platform conf (`tests/integration/ops/backend_conf.py`); they were the only failures in CI group 7 on `6f8128e` and pass on every platform's conf afterwards. Generator idempotent (`codegen_mudnn.py` twice, byte-identical; `codegen_musa_flaggems.py --check` and `gen_vendor_confs.py --check` clean for MUSA). `tests/unit/test_gen_vendor_confs.py`: 34 passed, 1 pre-existing failure (ascend/gcu conf staleness, unrelated). Three pre-existing `test_flaggems_conf_consistency.py` failures reproduce byte-identically against `d0e2d1a`'s data files, so they are not introduced by this change. |
| 2026-09-11 | None (CPU-only host) | Unified MetaX confs (refactor/unified-vendor-confs) | Collapsed `backends_metax_flaggems.conf` and `backends_metax_flaggems_cpp.conf` into a single `backends_metax.conf`. The 17 on-device-verified C++ routes are now in the file unconditionally; a build without `FLAGGEMS_KERNEL=ON` degrades them to the boxing kernel via `Dispatcher::GetFn` instead of raising. `METAX_CPP_MEASURED` in `gen_vendor_confs.py` records the measured set explicitly since the file it was formerly recovered from no longer exists. `mm` remains on the boxing kernel (MetaX C550 shared-memory limit). `_select_backend_config()` now routes both `FLAGOS_USE_FLAGGEMS` and `FLAGOS_USE_FLAGGEMS_CPP` to the same `backends_metax.conf` under `FLAGOS_METAX_BOXING=1`. **All hardware rows not revalidated.** | Mechanical evidence only — generator idempotent (two runs, empty diff; `--check` exits 0), `tests/unit/test_gen_vendor_confs.py` passes with updated test names. |
| 2026-09-10 | None (CPU-only host) | Full-coverage MUSA/GCU/Ascend and boxing configurations | Converted the MUSA, GCU, Ascend and boxing configurations to full coverage: all 2036 routable ops listed exactly once under `flaggems_cpp` / `flaggems` / `<vendor>` / `none`, priority in that order, generated by `scripts/codegen/gen_vendor_confs.py`. Every accelerated route is now gated on the platform's real PrivateUse1 registration set, read from the generated `*_register.inc` files, because CUDA-measured FlagGems coverage is a ceiling and not a per-platform routing set (Ascend 374, GCU 152, MUSA 158 registered of 2036). MetaX and Tsingmicro register the full generated list, so `none` would raise there instead of boxing to `cpu_fallback`; Tsingmicro's configuration stays hand-written. **All hardware rows not revalidated.** | No route measured. `flaggems_overload_survey.py` cannot run on this host: Triton 3.7.1 exposes only `amd`/`nvidia` backends and `import flag_gems` fails. Mechanical evidence only — generator idempotent (two runs, empty diff; `--check` exits 0), routing equals registration exactly on all three vendors, `tests/unit/test_gen_vendor_confs.py`: 27 passed, `tests/unit/`: 303 passed, 96 skipped, 2 pre-existing profiler failures (`CXXABI_1.3.15` libstdc++ skew) unrelated to routing. |
| 2026-08-31 | MetaX C550 (8 devices) | FlagGems qualname/cohort skew | Rerouted 10 FlagGems entries whose generated Python qualnames are absent from the current FlagGems tree to the CUDA boxing path in the generic, DCU, and MetaX FlagGems configurations. The generic FlagGems cohort was not revalidated on the other platforms. | On MetaX, `x[None]`, `binary_cross_entropy_with_logits`, and the affected dispatch paths now resolve through CUDA boxing; the issue #218 `mul_` reproducer still passes. `special_bessel_j1` retains a pre-existing MACA boxing failure unrelated to FlagGems. The 10 routes were not measured by the standard overload survey. |
| 2026-08-30 | MTT S5000 (8 devices) | Native MUSA empty-tensor handling | Added generated on-device handling for zero-element Unary/Binary/Reduce outputs and on-device identities for whole-tensor empty `sum`, `mean`, and `prod`; no CPU fallback is used. | `test_pow_dispatch.py`: 22 passed, 4 deselected; `test_narrow_dispatch.py`: 17 passed including the restored zero-length backward case; `test_musa_dispatch.py`: 89 passed. The full operator cohort reached 480 passed, 14 skipped, and 3 xpassed; three unrelated FlagGems consistency assertions remain environment/configuration failures. |
| 2026-08-27 | Ascend 910 (CANN 9.0.0) | Native Ascend view routes | Added `_conj` and `_neg_view` as metadata-only view routes; without them every math-bit resolution raised `backend not registered`. Generic FlagGems cohort **not revalidated** because no FlagGems route changed. | `tests/integration/test_math_bits_contract.py`: 5 passed, 7 skipped. Negative-bit clone/copy/resolve are bit-exact; the Conjugate cases skip because CANN 9.0.0 has no complex compute (`_conj_physical` absent, `aclnnAdd`/`aclnnMul` reject Complex{Float,Double}). |
| 2026-08-26 | MetaX mc550 (C550), MACA 3.8.0 | Shared soft-lowp matrix wrappers | Enabled the CUDA-boxing build gate for scalar FP8 and packed FP4 `mm`/`bmm`/`addmm`; ordinary dtypes retain MACA boxing and unsupported scaled-mm metadata remains fail-closed. The generic FlagGems survey was not rerun because it does not exercise these wrappers. | `tests/integration/ops/test_soft_lowp_gate_dispatch.py -m soft_lowp -v -s --tb=short`: 37 passed. Coverage includes five FP8 formats, packed FP4, matrix overloads, non-square packed layouts, in-place `addmm_`, and fail-closed scaled-mm. |
| 2026-08-21 | MetaX C550 (MACA 3.8.0) | CUDA-boxing AMP routes | Enabled the shared AMP integration contract for MetaX and added it to the MetaX CI manifest; no operator route changed. Generic FlagGems routes were **not revalidated**. | `tests/integration/test_amp.py`: 25 passed, covering FP16/BF16 autocast policies and GradScaler finite/overflow training paths. |
| 2026-08-19 | Hygon DCU bw1000 | Generic FlagGems routes | Rerouted `index_select` from `flagos_python` to `cuda` in all FlagGems configs (cross-stream launch race drops output stores under load); generic cohort 546 -> 545 active routes, 26 -> 27 forced CUDA fallbacks. Four-platform rows **not revalidated** (A100/mc550/810e unavailable). | Targeted survey `--ops index_select` on the flagos_python route: STRICT (standalone math correct); three failing HF v5.5.0 UT nodes (T5/Qwen3/Gemma3 beam search) pass after the reroute; tiny-T5 NaN reproducer clean 3/3. |
| 2026-08-18 | MTT S5000 (8 devices) | Native MUSA RNG, MThreads FlagGems hybrid, and MUPTI profiler | Added optional MUPTI activity tracing; the operator route cohort is unchanged. | `tests/integration/test_profiler_musa.py`: 1 passed with real positive-duration MUPTI kernel/runtime/memcpy activities and valid Chrome JSON. CPU-only Kineto resolver behavior remains environment-dependent; generic FlagGems operator coverage was not revalidated by this profiler change. |
| 2026-08-18 | Ascend 910 (CANN 9.0) | Ascend AMP and dtype routes | Added generated AMP unscale and foreach list-add routes; fixed promotion-aware binary outputs, float64 copies, and CPU fallback for unsupported matmul/unary dtypes. | `test_amp.py`: 25 passed; `test_dtype_coverage.py`: 174 passed; targeted float64, promotion, and fallback parity probes passed. |
| 2026-08-17 | MTT S5000 (8 devices) | Native MUSA RNG and MThreads FlagGems hybrid | Added shared per-device RNG reservations, muRAND/mudnn native RNG, shared stream compatibility, and seven non-overlapping FlagGems routes. | Unified RNG suite passed on the MUSA-marked cases; MUSA dispatch: 89 passed; routing/bridge units: 24 passed; real hybrid FlagGems: 2 passed, including selected reductions, duplicate-index `index_add`, and FlagGems `randn` mixed with native RNG. Vendor FlagTree wheel required; generic Triton 3.7.1 is not evidence. |
| 2026-08-17 | Enflame S60 | Native GCU RNG routes | Added 16 topsaten RNG routes; generic FlagGems cohort not revalidated. | Targeted mixed native/FlagGems probe verified shared seed/offset progression and replay; `tests/integration/ops/test_rng_dispatch.py`: `104 passed, 2 skipped, 1 xpassed`. |
| 2026-08-14 | Ascend 910 (2 devices) | Native Ascend FSDP2 routes | Added `_chunk_cat`, `_chunk_cat.out`, `_foreach_copy_`, `cat.out`, `split.Tensor`, `split_with_sizes`, and `split_with_sizes_copy.out`; generic FlagGems cohort not revalidated because it is unchanged. | Manual FlagCX collective, DDP, and FSDP2 tests on CANN 9.0; standard FlagGems harness is not applicable to native routes. |
| 2026-08-13 | A100, mc550, 810e, bw1000 | torch-fl `fe2272b5`, FlagGems `7fb49bad`, harness v4 | Established the verified 546-overload four-platform baseline. | Manual survey JSON; aggregate and raw counts recorded above. |
