# Operator Support

This reference records measured operator coverage for torch-fl accelerator
backends. The current baseline measures the generic FlagGems Python routing
surface on four hardware platforms. It is an availability and correctness
survey, not a claim of complete PyTorch conformance, autograd coverage, or
performance quality. A second, separate cohort measures each platform's own
full-coverage configuration; those rows are recorded under
[Native Backend Route Changes](#native-backend-route-changes) and carry their own
denominator, so they must not be read against the 546-overload tables below.

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

These four rows are the FlagGems overload-survey cohort, which does not include
MUSA. The MUSA routing changes recorded under **Native Backend Route Changes** are
**not revalidated** against them: no route in this cohort was altered for A100,
mc550, 810e or bw1000 by that work, and no MUSA row exists here to update. The
same caveat applies to the raw case counts below.

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
not misrepresented as part of the 546-overload FlagGems cohort. A change against
a platform's own full-coverage configuration is a third cohort again — its
denominator is that file's active route set, not 546 — and the entry states which
one it measured.

### Enflame GCU S60 FlagGems routing (2026-09-15)

The GCU configuration is now FlagGems-first. Before this change
`backends_gcu.conf` contained no `flaggems` route at all: every accelerated
overload was a topsaten kernel and everything else was `none`. That describes the
committed conf -- what the wheel ships -- which is also the version that is
consistent with the build: the generator already wanted to emit 88 GCU
`flaggems` routes, but GCU had no FlagGems registration to route to, so
regenerating the conf alone would have produced 88 routes pointing at an empty
dispatcher slot. The conf is stale against its own generator at the branch point
and still at `flagos/main` `bc39a83`
(`scripts/codegen/gen_vendor_confs.py --check` reports `backends_gcu.conf` stale
there),
and this change closes that gap by supplying the registration below before
regenerating. FlagGems is reached through a new generated registration file,
`csrc/aten/backends/gcu/generated/gcu_flaggems_register.inc`, emitted by
[`scripts/codegen/codegen_gcu_flaggems.py`](../../scripts/codegen/codegen_gcu_flaggems.py) and
included by `csrc/aten/register.cc` directly after `gcu_register.inc`. The two
lists together are what GCU claims on PrivateUse1: `gcu_register.inc` (152
`m.impl` lines) takes the ops topsaten has a kernel for, the new file (249
`m.impl` lines) takes the rest of the shared FlagGems coverage. An op routed to
`flaggems` that neither file claims would reach the dispatcher with an empty
`kFlagGems` slot and raise `backend not registered` instead of falling back, so
generation gates every accelerated route on the registration set.

**Route delta.** GCU `flaggems` 0 -> **257**, `gcu` 152 -> 144, `none`
1884 -> 1635 (2036 routable ops), so accelerated routes go from **152 to 401**
(7% -> 19.7%). Eight of the 257 `flaggems` routes (`_softmax`, `clamp`,
`fmod.Tensor`, `gelu`, `mean`, `mean.dim`, `remainder.Tensor`, `silu`) are
bridged by the existing handwritten wrappers in `csrc/aten/register.cc` rather
than by the new `.inc`; that is why the file has 249 lines rather than 257.

**Why 225 ops are gapped.** `NATIVE_TRITON_GAPS["gcu"]` grew from 108 to 225
entries. Every addition is a measured failure on the S60, not an inference, and
they fall into four families:

- **The GCU300 front end rejects 64-bit types in kernel IR** — `error: 64-bit
  data type not supported on GCU300!`, surfaced as `RuntimeError: Pipeline run
  failed: PassManager execution failed`. The rejected type sits *inside* the
  kernel, so it is the operand dtype that has to be avoided, and since routing
  is per operator, any op a real caller hands an `int64` tensor moves as a whole.
  This is the largest family: reductions, scans and index ops that carry an index
  accumulator (`nonzero`, `argmax`, `argmin`, `count_nonzero`, `_unique2`,
  `unique_dim`, `unique_consecutive`, `topk`, `median`, `nanmedian`, `max.dim`,
  `min.dim`, `mode`, `kthvalue`, `range`, `unfold_backward`, `index_copy(_)`,
  `var.correction`, `var_mean.correction`, `norm.ScalarOpt_dim`,
  `native_layer_norm`, `mse_loss`, `scatter`, `renorm`).
- **Unimplemented GCU300 lowering, hard compiler abort.** `_adaptive_avg_pool2d`
  aborts in `PtrAnalysis.cpp:1711` after `add logic to support op arith.maxsi`;
  `asin` aborts in `ElementwiseFusionOpToGCU.cpp:874` after `unsupported extern
  elementwise: __nv_asinf`; the `special_chebyshev_*`, `special_shifted_chebyshev_*`
  and `special_hermite_polynomial_h` group aborts the same way. These are SIGABRT,
  not exceptions, so they cannot be caught at the Python level.
- **Driver-level process death.** `addr` aborts with
  `dtu_context_obj.cc:693:submit_sip_assertion_task ##abort as Detected SIP
  assert###` followed by `detected SIP assert!!!`; `native_batch_norm` and
  `_batch_norm_no_update` die with SIGSEGV.
- **Measured wrong answers on float profiles**, which no "did it raise" test
  would catch: `elu`/`elu_`/`elu_backward` (`max_diff` 0.38-0.89),
  `histc` (`max_diff` up to 1536), `_softmax_backward_data` and
  `_log_softmax_backward_data` (return `int8` where `float32` is required),
  `sum.out` (returns `(32, 32)` where the scalar shape `()` is required),
  `addmm.*`, `silu_backward`, `tril.*`, `triu.*`.

The gap decision was not the survey verdict. A route that is wrong only for
`int64` still carries a working `float16`/`float32` path, so the classifier was
profile-aware and the rule was split in two:

- **Wrong on any `float16`/`float32` profile** (81 ops): unusable on GCU, moved
  back to the vendor route.
- **Wrong only for `int64`/`bool` and a topsaten kernel exists** (36 ops):
  gapped, because `TopsatenSupportsDtype` in `topsaten_common.h` round-trips
  unsupported dtypes through the CPU, so the vendor route is correct where
  FlagGems raised and no slower at `float16`/`float32`.
- **Wrong only for `int64`/`bool` with no topsaten kernel** (76 ops):
  deliberately **left on FlagGems**. Gapping them would demote their
  `float16`/`float32` path to `cpu_fallback`, a strictly larger regression than
  the `int64` raise it would avoid. The consequence is explicit and is a
  behaviour change: these overloads raise `Pipeline run failed` for an `int64`
  operand where the previous configuration served the same call through
  `cpu_fallback`. The set is dominated by ops a float model never hands an
  `int64` tensor, and its highest-traffic members are `threshold_backward`,
  `relu_`, `clamp_min`, `clamp_max`, `max`, `min`, `nan_to_num` and
  `masked_scatter`. The full list is in the test-results document.

`abs`, `neg` and `sum.dim_IntList` are in the group that *is* gapped, so the
three dispatch-log tests that asserted `flagos_python` for them were rewritten to
read the route from the platform configuration
(`tests/integration/ops/backend_conf.py`), the same pattern already used for the
MUSA `add.Tensor` case; hard-coding the route "describes the platform the test
was written on and silently becomes wrong on every other one".

**Measured on the S60** with `tests/manual/flaggems_overload_survey.py`
(harness v4), seven profiles per overload
(`2d-f32`, `4d-f32`, `1d-f32`, `2d-f16`, `2d-i64`, `2d-bool`, `2d-f32-strided`),
all 374 `flag_gems` routes measured: 374 registered, 314 tested, 239
basic-executable, 121 strict. Of the 374 routes, 121 are clean on every exercised
profile, 81 are wrong at `float16`/`float32`, 112 are wrong only for
`int64`/`bool`, and 60 produced no valid case on any profile.

Those 374 routes are the un-gapped candidate set, not the shipped configuration:
the survey ran against a transient draft of `backends_gcu.conf` taken before the
measured gap set was applied. The draft is not byte-recoverable -- the survey
recorded `meta.conf` as this repository's `backends_gcu.conf` with
`meta.conf_sha256` `82f801778c…`, which matches neither the base commit's conf
(`039fb323…`) nor the one this change ships (`4c5082d6…`) -- but it reconciles
with the shipped conf exactly: 374 - 117 = 257, where 117 = 81 + 36 is the number
of ops the measurement returned to the vendor. Per-op results survive in the
survey JSON. The environment was
Python 3.12.13, CPU PyTorch 2.10.0,
flagtree `0.6.1+enflame3.6` (Triton 3.6, backend `enflame`) and FlagGems master
`3c6f7537d`. Full per-op evidence and the raw failure families are in
[docs/vendors/gcu/flaggems-test-results.md](../vendors/gcu/flaggems-test-results.md).

**CI.** Every pytest group of the GCU manifest was run locally on the S60 against
this tree, in one uninterrupted pass, all exiting 0: vendor operator cohort **595
passed, 32 skipped, 499 deselected, 2 xfailed, 2 xpassed**; FlagGems runtime path
**9 passed, 4 skipped, 1116 deselected, 1 xpassed**; unified RNG **111 passed,
4 skipped, 1 deselected, 1 xpassed**; general/factory **46 passed**; AMP **27
passed**; math-bits **12 passed**; `torch.compile` **29 passed, 18 skipped**. The
conf-consistency test is **7 passed**, and routing equals registration on GCU:
no op is routed to `gcu` without an entry in `gcu_register.inc`.

Group 3 initially failed three tests (`test_abs_dispatch.py`,
`test_neg_dispatch.py`, `test_sum_dispatch.py`) because they asserted
`-> flagos_python` for ops this change moved to the vendor route; the rewrite
described above is what makes them pass, and they are the evidence that the
rerouting is real rather than a configuration-only edit. Group 8's single failure
was `test_torch_backends_entry_point_is_registered` against a stale local
`torch_fl.egg-info` baked without `entry_points.txt`; `setup.py egg_info`
regenerated it and the group passes. Both are recorded because neither was a
defect in the shipped change and a reader should be able to tell that from the
numbers alone.

**Generator idempotency.** `codegen_gcu_flaggems.py` and `gen_vendor_confs.py`
each run twice produce byte-identical output
(`gcu_flaggems_register.inc` `a02b46d9…`, `backends_gcu.conf` `4c5082d6…`) and
both `--check` modes exit 0. The banner line naming the generator's own path is
the only thing the merge with `flagos/main` moved in either file, so the hashes
recorded before that merge (`39cd03e3…` and `10b8ab4b…`) differ without any route
or registration changing.

**Evidence gaps.** Four, all recorded rather than papered over:

- 60 of the 374 FlagGems routes are `UNTESTED` on every profile — the generic
  harness cannot construct a valid call for them (shape- and metadata-driven ops:
  `avg_pool2d(_backward)`, `col2im`, `reflection_pad*`, `native_group_norm*`,
  `scatter.reduce`, `_thnn_fused_lstm_cell`, `_scaled_dot_product_*_backward`,
  and others). They are **not** gapped and **not** measured; they stay on
  FlagGems and are listed in the test-results document so the gap is auditable.
- Two of the four process-death entries (`native_batch_norm`,
  `_batch_norm_no_update`) are gapped on exit status alone. The harness truncates
  captured stderr at 300 bytes, so the recorded evidence is SIGSEGV plus the last
  stderr line and **not** the faulting frame.
- No route was measured on any other platform, and no route was changed for one.
  The Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro rows are **not revalidated**
  by this change.
- The environment group (`set_env_gcu.sh`) was reproduced into a scratch venv
  rather than by CI, and its final Triton import check needed a local,
  never-committed retarget of one glibc-2.38 symbol in `libtriton.so` because
  the measurement host is Ubuntu 22.04. Nothing in this change has been executed
  by CI yet.

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

### MUSA FlagGems gap re-measurement: four ops promoted back to FlagGems (2026-09-15)

`NATIVE_TRITON_GAPS["musa"]` was introduced on 2026-09-14 with 18 entries, each
carrying a failure signature recorded at the time ("`index_add` returns all zeros
instead of accumulating", "`randn` crashes unpacking generator state", ...). Every
entry was re-probed against the FlagGems revision the MUSA CI job actually
installs, on the documented signature rather than on "some shape now works", so a
pass verdict means the specific defect is gone. Four entries no longer reproduce
and are promoted out of the set; the remaining fourteen keep their routes and have
their provenance updated to the current revision.

**Promoted out (4).**

| Op | Route before | Route now | Measured |
|---|---|---|---|
| `index_add` | `none` | `flaggems` | bit-exact vs CPU for duplicate indices, `dim` 0 and 1, float64, and `alpha = 2.5`; 7/7 probe cases pass |
| `index_add_` | `none` | `flaggems` | same 7/7, both spellings, in-place alias preserved |
| `randn` | `musa` | `flaggems  # musa` | finite, seed-reproducible and seed-sensitive over 65536 samples; f32/f16/bf16 all `std ≈ 0.97`; 4/4 pass |
| `randn_like` | `musa` | `flaggems  # musa` | inherits shape and dtype; 4/4 pass |

Neither `index_add` nor `index_add_` has a mudnn kernel, so before this change they
could not be registered at all: they routed to `none`, and the `FLAGOS_OP_*`
override could not reach them either, which is why the old "returns all zeros"
signature had to be measured by hand. The recorded signature has the shape of a
cross-stream read — the in-place wrapper is a FlagGems kernel writing a clone
followed by a copy into `self`, and the split default stream was fixed in the same
change that added the entry — so the promotion is a fix that landed upstream of
this repository, not a re-routing decision.

**Kept (14).** All of these reproduce their recorded signature exactly, in process,
with `FLAGOS_OP_*` pinning the op back onto FlagGems:

| Op | Reproduced failure | Root cause |
|---|---|---|
| `add.Tensor`, `add_.Tensor`, `sub.Tensor`, `sub_.Tensor`, `div.Tensor`, `div_.Tensor` | `failed to translate module to LLVM IR` (bf16 wrapped-number operand only) | FlagGems pointwise promotion ignores `is_wrapped_number`, promotes to fp64; mthreads' LLVM lowering has no double `float2bfloat16` |
| `mul_.Tensor` | `no fallback function is registered for schema aten::mul.out` (f32 and bf16, every shape) | `flag_gems/ops/mul.py:587` gates on its own device name (`'flagos' != 'musa'`) and redispatches to `aten.mul.out`, which has no kernel at that dispatch key |
| `div.Tensor_mode`, `div_.Tensor_mode` | wrong trailing element at `n = 3, 5, 6, 7, 9, 15, 17, 31, 33, 100` in int64 `floor`/`trunc` and int32 `floor` | FlagGems computes the integer rounding-mode forms on the same kernel that loses its final store |
| `floor_divide`, `floor_divide_.Tensor` | wrong trailing element at the same `n`, int64 and int32 | same trailing-store loss |
| `sort`, `sort.stable` | `RuntimeError: MudnnCopy: unsupported dtype Long -> UInt32` | FlagGems' radix sort casts its histogram to uint32 internally; mudnn `Unary::CAST` stops at `kBool` |
| `_conj` | probe *passes*, and that is the point | `ATen` `conj` is a lazy view that sets the Conjugate bit (`is_conj() == True`); `flag_gems.ops._conj` materializes (`is_conj() == False`), so registering it breaks `test_math_bits_contract.py`. mudnn has no Conjugate-bit path either, so `none` — unregistered, ATen's composite — is the correct route |

Two probes establish the `mul_.Tensor` reason directly rather than by inference:
`aten.mul.out` called on MUSA at the top level *works* (so the kernel exists and
the only missing piece is the redispatch target), and the guard is confirmed live —
`flag_gems.ops.mul._DEVICE_NAME == 'musa'`, the FlagGems runtime device name is
`'musa'`, and the tensor's `device.type` is `'flagos'`.

**Route delta.** MUSA `flaggems` 464 -> **468**, `musa` 51 -> **49**, `none` 1521 ->
**1519** (2036 routable ops). Two of the four were previously on the vendor route
(`randn`, `randn_like`, which keep a native kernel and so are annotated
`flaggems  # musa`); the `index_add` pair was on `none`. The registered-op set is
unchanged at 518: `index_add` and `index_add_` are added to
`musa_flaggems_register.inc` (357 -> **359** `m.impl` lines) at the same time as the
four entries leave the gap set, which is what `codegen_musa_flaggems.py` keys on.

**Measured on the eight-device MTT S5000 host** with CPU PyTorch 2.10.0, mudnn
v3300, FlagGems `4d9c34775` (5.4.0rc2.post1+g4d9c34775) and flagtree
`0.6.2a3+mthreads3.6` (Triton 3.6, backend `mthreads`):

- Per-op probe, one fresh process per op (`FLAGOS_LOG_DISPATCH=1`,
  `FLAGOS_LOG_FALLBACK=1`, `PYTHONUNBUFFERED=1` so the `--- CASE` markers and the
  dispatcher's own stderr lines interleave). Verdicts for all 18 entries:

  | Op | Verdict | Cases | Dispatch / fallback lines |
  |---|---|---|---|
  | `index_add` | PASS | 0/7 failed | 28 / 7 |
  | `index_add_` | PASS | 0/7 failed | 28 / 7 |
  | `randn` | PASS | 0/4 failed | 15 / 0 |
  | `randn_like` | PASS | 0/4 failed | 15 / 0 |
  | `_conj` | PASS (contract entry) | 0/1 failed | 0 / 2 |
  | `add.Tensor` | FAIL (expected) | 1/5 failed | 5 / 0 |
  | `add_.Tensor` | FAIL (expected) | 2/6 failed | 6 / 0 |
  | `sub.Tensor` | FAIL (expected) | 1/5 failed | 5 / 0 |
  | `sub_.Tensor` | FAIL (expected) | 2/6 failed | 6 / 0 |
  | `mul_.Tensor` | FAIL (expected) | 3/6 failed | 5 / 3 |
  | `div.Tensor` | FAIL (expected) | 1/5 failed | 5 / 0 |
  | `div_.Tensor` | FAIL (expected) | 1/6 failed | 7 / 0 |
  | `div.Tensor_mode` | FAIL (expected) | 3/3 failed | 54 / 0 |
  | `div_.Tensor_mode` | FAIL (expected) | 3/3 failed | 54 / 0 |
  | `floor_divide` | FAIL (expected) | 2/3 failed | 37 / 0 |
  | `floor_divide_.Tensor` | FAIL (expected) | 2/3 failed | 37 / 0 |
  | `sort` | FAIL (expected) | 6/7 failed | 24 / 0 |
  | `sort.stable` | FAIL (expected) | 2/7 failed | 12 / 0 |

  The comparison control matters as much as the verdicts do: a deliberately
  perturbed reference is rejected by the same comparator (`perturbed reference
  rejected=True`, `1/24` element outside tolerance), so a green table is not a
  comparator that cannot fail. The `_conj` row's two fallback lines are
  `aten::view_as_real` — no slot is claimed for the op, which is the intended
  state.
- Provenance, not just verdicts: the flag_gems code cache was snapshotted before
  and after each op. Both `index_add` and `index_add_` add a new entry
  (`index_add_rank_2_pid_*.py`), so the mthreads kernel compiled and ran on device;
  `sort`/`sort.stable` add six and two triton cache directories, i.e. the failing
  kernels really are being built and then rejected by the cast. All 7 fallback
  lines in the `index_add` rows are the probe's own CPU comparison.
- End-to-end on the rebuilt library, driving the public API with no `FLAGOS_OP_*`
  override so the op reaches whatever the shipped conf routes it to: **14/14 cases
  pass**. The dispatch log shows `index_add -> flagos_python` (22),
  `index_add_ -> flagos_python` (5), `randn -> flagos_python` (10),
  `randn_like -> flagos_python` (1), against the regression controls
  `sort -> musa` and `add.Tensor -> musa`; the 27 `cpu_fallback` lines are all
  `aten::equal`, the harness's own host comparison. Coverage includes duplicate
  indices, `dim=1`, float64, a non-zero base (returns `sum 60.0`, so a kernel
  returning the input unchanged cannot pass), `torch.manual_seed` reproducibility,
  a 200k-sample statistics check against the CPU draw, and sort/index bit-exactness.
- `index_add` with duplicate indices and `alpha != 1` is *not* bit-exact and the
  bound was measured rather than assumed: over 20 seeds, `alpha = 2.5` differs from
  the CPU reference in 11/20 trials with a worst absolute error of `4.768e-07` —
  one float32 ULP at `|x| < 4` — and only on the duplicated rows. `alpha == 1` is a
  plain add and is bit-exact in every trial. ATen documents duplicate-index
  `index_add` as order-free (it is on the CUDA non-determinism list), so this is
  reassociation of the fused multiply, not an accumulation defect; the probe
  carries both a 1-ULP budgeted comparator and a dedicated case that fails if the
  error ever leaves that bound.
- The four MUSA CI groups that exercise these routes were re-run locally: dispatch
  **113 passed**; factory **46 passed**; operator cohort **493 passed, 1 skipped,
  521 deselected, 2 xfailed, 1 xpassed, 3 failed**; RNG with the manifest's own
  `-k` filter **80 passed, 37 deselected**. The cohort's three failures are the
  pre-existing `test_flaggems_conf_consistency.py` assertions described below and
  are the same three that fail on the pristine conf.
- Generator ordering, which is load-bearing here: `gen_vendor_confs.py` reads the
  on-disk `musa_flaggems_register.inc` to decide which ops the platform registers,
  so `codegen_musa_flaggems.py` must run **before** it. Running them the other way
  round leaves `index_add` at `none`. Idempotency: both generators re-run to
  byte-identical output, `codegen_musa_flaggems.py --check` reports "is up to
  date", and `gen_vendor_confs.py --check` is clean for MUSA.
- `tests/unit/test_gen_vendor_confs.py`: **34 passed, 1 failed**. The failure is
  `test_shipped_confs_are_up_to_date` reporting `stale: ['backends_gcu.conf',
  'backends_ascend.conf']`, which also fails on a stashed pristine tree.

**Evidence gap.** `tests/manual/flaggems_overload_survey.py` cannot measure this
change, for the reason recorded in the section below: it selects overloads whose
conf value is `flagos_python`, the unified per-platform confs spell the FlagGems
route `flaggems`, and the `backends_flaggems.conf` it was written against was
removed by `d0e2d1a`. The evidence here is targeted per-op probing plus the
end-to-end route check plus the CI groups, not a synthesized overload survey. The
generic FlagGems baseline rows (A100, mc550, PPU, DCU, 546-route cohort) are
unchanged by this work and are **not revalidated**; no FlagGems route was altered
for any other platform, and `NATIVE_TRITON_GAPS` has no non-MUSA entry beyond the
pre-existing Ascend `pow`/`rsqrt` set.

**Two pre-existing conditions, unchanged by this work.** Three assertions in
`tests/integration/ops/test_flaggems_conf_consistency.py`
(`test_every_conf_op_maps_to_a_dispatcher`, `test_no_orphan_flagos_python_kernels`,
`test_counts_match`) fail against the pristine conf as well — re-measured here with
`backends_musa.conf` stashed, same three tests, byte-identical text — on
`mm`/`bmm`/`addmm` dispatcher drift in `csrc/aten/generated/` that this change does
not touch. Separately, the stale `backends_gcu.conf` / `backends_ascend.conf`
reported by `test_shipped_confs_are_up_to_date` are pre-existing generator drift on
two out-of-scope platforms. Neither is in scope here.

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

### CUDA FlagGems-first routing with FlagTree Triton 3.6 (2026-09-15)

CUDA full-coverage code generation routes every FlagGems Python wrapper whose ATen
schema the boxed adapter can satisfy to `flaggems` instead of CUDA boxing, and
returns to CUDA boxing the ones that then measured worse there.
`torch_fl/configs/backends_cuda.conf` moves from 13 to **416 `flaggems` routes**
and from 2021 to **1618 `cuda` routes**. The TileOPs annotation moves with the
surviving routes: 40 of the 51 annotated ops now read `flaggems  # tileops` and 11
read `cuda  # tileops`, where before all 51 read `cuda  # tileops`. Route priority,
`flaggems_cpp > flaggems > tileops > <vendor> > none`, is unchanged, no CUDA boxing
kernel was added, removed, or reimplemented, and no other platform's configuration
was touched. Two of the 13 `flaggems` routes `main` already carried, `embedding`
and `sum.dim_IntList`, measured worse on FlagGems and are returned to CUDA boxing;
the other 11 (`_softmax`, `abs`, `add.Tensor`, `bmm`, `mean.dim`, `mm`, `neg`,
`silu`, `sin`, `sqrt`, `where.self`) keep their route.

**Routing is a guess; the rollback is the measurement.** The generator's first
pass is mechanical -- it checks that the ATen schema is one the boxed adapter can
satisfy -- and 98 of the 514 candidate routes failed that check in practice: each
one failed a case on the FlagGems route that CUDA boxing answers correctly, or
crashed, hung, or recursed. Those 98 are listed in `measured_flaggems_rollback` in
`scripts/codegen/codegen_ops.py` and route to `cuda` in the checked-in
configuration, so the file is derived from the generator rather than hand-edited.
The criterion is a paired measurement, not a threshold on the FlagGems verdict
alone: every op in the set was run twice by the same harness, once on each route.
An op whose failure vector is identical on both routes is **not** rolled back --
returning it to CUDA boxing would buy nothing -- and stays on `flaggems` as
`BASIC_ONLY`. Seven ops are in that state; they are listed below.

**Cohort.** This is a second cohort, not a re-measurement of the baseline tables
above. Those measure the generic `backends_flaggems.conf` route set (546 active
routes, harness version 4) on four platforms; the numbers below measure the CUDA
full-coverage configuration (416 active routes, harness version 6) on A100. The
denominators differ, so no row of one cohort may be compared with, or subtracted
from, a row of the other.

| Field | Value |
|---|---|
| torch-fl source | `93568ac` |
| FlagGems source | `7fb49bad47116434961bfb2b912811716d383eaf` (`flag_gems` 5.3.4.post1.dev1+g7fb49bad4) |
| Triton provider | `flagtree==0.6.2a2` (source-free; provides `triton` 3.6.0, `is_flagtree_active()` true) |
| CPU PyTorch | `2.10.0+cpu` with staged `cu130` accelerator assets |
| Configuration | `torch_fl/configs/backends_cuda.conf` |
| Configuration SHA-256 | `ab2522b7fec9363699452249900b28188ce78ba3ca16b481be34927dbe933ede` |
| Active route-set SHA-256 | `0b344884e9bb318a29d28a5b99b36b652f48f3d773576a9dade0391e3a9912e5` |
| Survey harness | `tests/manual/flaggems_overload_survey.py`, version 6 |
| Survey harness SHA-256 | `31334631cc42d3e9df947fa101bd2a5e905f690ba5cc7e5dfcab3d9feb2a709f` |
| Registered and active routes | 416 |
| Profiles per overload | 7 |

Measured on one host with 8 x NVIDIA A100-SXM4-40GB. This row is a revalidation:
the hardware was available and the survey was rerun against the changed
configuration. The library the survey exercised is the FlagGems Python build
(`CUDA_KERNEL=ON`, `FLAGGEMS_PYTHON=ON`), so the FlagGems dispatcher slot is
populated and the `flaggems` routes in this cohort really execute FlagGems. The
A/B runs that set the rollback list used the same library on the same host.

| Hardware | Total | STRICT | BASIC_ONLY | FAILED | UNTESTED | Basic executable | Basic rate | Strict rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA A100 | 416 | 321 | 7 | 0 | 88 | 328 | 78.8% | 77.2% |

| Hardware | PASS | INVALID_CASE | UNVERIFIABLE | ERROR | WRONG | CRASH | TIMEOUT | Context poison |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA A100 | 1817 | 1085 | 0 | 2 | 8 | 0 | 0 | 0 |

For this row, `STRICT + BASIC_ONLY + FAILED + UNTESTED = 416` and
`Basic executable = STRICT + BASIC_ONLY = 328`. The case-level counts sum to 2912,
which is the 416 routes x 7 profiles the harness ran.

**Partial overloads.** No overload passed zero valid cases, so there is no
`FAILED` list. Seven recorded `PASS` on some valid cases and something else on
others, and each of the seven also produced exactly that status vector on the CUDA
boxing route in the paired run: `kthvalue`, `median.dim`, `mm`, `mm.out`, `mode`,
`sort`, `sort.stable`. Five of them (`kthvalue`, `median.dim`, `mode`, `sort`,
`sort.stable`) disagree with CPU on the `2d-i64` profile (`mode`, `sort` and
`sort.stable` also on `2d-bool`), where the harness compares duplicate values by
identity and reports `unequal` on both routes; `mm` and `mm.out` raise
`RuntimeError: self must be a matrix` on the same profile on both routes. None of
the seven is a FlagGems-route defect, so they are reported `BASIC_ONLY` rather
than rolled back or claimed as `STRICT`. The remaining 88 overloads are
`UNTESTED`: no CPU-valid synthesized case existed for them under the seven
measured profiles, which is neither a pass nor a failure. That set is dominated
by backward and pooling overloads whose schemas the synthesizer cannot fill
(`_flash_attention_backward`, `avg_pool3d_backward`, `nll_loss_backward`, the
`bitwise_*_.Scalar` family).

**The 98 overloads returned to CUDA boxing.** The groups below are the ones
`measured_flaggems_rollback` itself carries; an op that fails differently on
different profiles is counted once, under the failure the rollback was decided
on.

| Group | Ops |
|---|---:|
| Rejected by a `is_cuda` device guard | 13 |
| Wrong result on a profile where CUDA boxing passes | 43 |
| Rejected by an argument or domain contract narrower than ATen | 12 |
| Triton `CompilationError` | 8 |
| Other runtime error | 12 |
| Exceeds the per-op time budget | 5 |
| Unbounded recursion | 2 |
| Process crash | 2 |
| Silently wrong where CUDA boxing also fails | 1 |

*Device guard (13).* The FlagGems kernel rejects its operands before doing any
work, because `Tensor.is_cuda` is false on the `flagos` PrivateUse1 device even
though `torch.cuda` is live: `i0`, `i0.out`, `im2col`, `smooth_l1_loss`,
`smooth_l1_loss.out`, `smooth_l1_loss_backward`, `special_i0e`, `special_i1`,
`special_modified_bessel_k0`, `special_modified_bessel_k0.out`,
`special_scaled_modified_bessel_k1`, `special_scaled_modified_bessel_k1.out`,
`upsample_bicubic2d`. Representative messages: `ValueError: i0: input tensor must
be on cuda device`, `AssertionError: im2col: Inputs must be CUDA tensors`,
`AssertionError: smooth_l1_loss: input and target must be CUDA tensors.`,
`ValueError: special_i0e: Tensors must be cuda tensors`, `ValueError:
upsample_bicubic2d: This Triton kernel requires CUDA tensors`. CUDA boxing runs
the same cases correctly.

*Wrong result on a profile where CUDA boxing passes (43).* Two subgroups. The
larger is integer input, where `flag_gems` returns the input dtype (int64) while
ATen's type promotion returns float32, so the dtype and the values are both wrong
(13): `acosh`, `atan2`, `atanh`, `digamma`, `erf`, `erfinv`, `log`, `log1p`,
`log2`, `rad2deg`, `special_airy_ai`, `special_bessel_j1`, `special_xlog1py`. The
other 30 are `_log_softmax_backward_data`, `_pdist_backward`, `_softmax_backward_data`,
`_unique2`, `_weight_norm_interface`, `_weight_norm_interface_backward`, `elu`,
`elu_`, `floor_divide.Scalar`, `histc`, `igammac_`, `index_copy`, `index_copy_`,
`leaky_relu_`, `logsumexp`, `median.dim_values`, `mse_loss_backward`,
`nanmedian.out`, `native_layer_norm`, `nll_loss_forward`, `prod`, `range`,
`scatter.src`, `scatter_.src`, `special_chebyshev_polynomial_v`,
`special_shifted_chebyshev_polynomial_u`, `special_shifted_chebyshev_polynomial_w`,
`sum.out`, `unfold_backward`, `unique_dim`. Measured deviations include `elu` at
`max_diff 0.690`, `histc` at `max_diff 1024.0`, `_weight_norm_interface` at
`max_diff 8.1e34`, `nll_loss_forward` wrong on `2d-f16`, `_unique2` returning
three values where ATen returns one, `sum.out` returning a `(32, 32)` tensor for a
scalar reduction, and `range` returning float64 for a float32 input.

*Argument or domain contract narrower than ATen (12).* The kernel asserts a dtype
set, calls `torch.finfo`, or otherwise rejects an input ATen accepts:
`_euclidean_dist` (`AssertionError: x1 must be a 2D tensor`),
`_upsample_bilinear2d_aa` (bare `AssertionError`), `randperm` (bare
`AssertionError`), `special_shifted_chebyshev_polynomial_v`, `topk`
(`AssertionError: Currently only support topk in last dimension`),
`soft_margin_loss` (`AssertionError: soft_margin_loss: input and target must be
cuda tensors for Triton kernel.`), `soft_margin_loss_backward` (`AssertionError:
soft_margin_loss_backward: grad_output, self, and target must have the same number
of elements`), `amin` (`AssertionError: amin only supports float dtypes`), `logit`
(`TypeError: logit expected a floating point tensor as input`), `nan_to_num`,
`special_chebyshev_polynomial_u.n_scalar`, `special_modified_bessel_k1`.

*Triton `CompilationError` (8).* `norm.ScalarOpt_dim`, `randint`, `randint_like`,
`special_chebyshev_polynomial_w`, and the four bool-input ops `cummax`, `cummin`,
`index_add`, `index_add_`. The compiler names the generated line: the philox seed
conversion (`philox_seed = philox_seed.to(tl.int64)`) for `randint`/`randint_like`,
`X = X + pid * N` for `norm.ScalarOpt_dim`, `offset0 = (tile_id0 * ...)` for
`special_chebyshev_polynomial_w`, and the int8 element-type check for the bool
inputs. All of them compile on the CUDA boxing route.

*Other runtime error (12).* `_cdist_backward` (`IndexError: tuple index out of
range`), `_log_softmax_backward_data.out` and `_softmax_backward_data.out`
(`RuntimeError: ...: expected grad_input dtype torch.int8, got torch.float32`),
`cosh.out` (`TypeError: cosh_out() missing 1 required positional argument:
'out'`), `dequantize.self` (`NotImplementedError: Could not run 'aten::int_repr'
with arguments from the 'CPU' backend`), `elu_backward` and `embedding`
(`RuntimeError: Triton Error [CUDA]: context is destroyed`), `mul_.Tensor`
(`NotImplementedError: There were no tensor arguments to this function`),
`nanmedian.dim_values` (`RuntimeError: shape '[32]' is invalid for input of size
1024`), `norm.Scalar` (`RuntimeError: Please look up dimensions by name, got: name
= None.`), `special_chebyshev_polynomial_u` (`ValueError: Chebyshev polynomial
order n must be in [0, 5], got values in [-3, 4]`), `special_hermite_polynomial_h`
(`ValueError: special_hermite_polynomial_h only supports n in [0, 9], got n in
[-3, 3]`).

*Time budget (5).* `lcm`, `lcm_`, `prod.dim_int`, `sum.IntList_out` and
`sum.dim_IntList` exceed the harness's per-op budget. The first two hang on
`2d-i64` and the last three on `2d-bool`; both are profiles where the CUDA boxing
route returns immediately.

*Unbounded recursion (2).* `unique_consecutive` and `sgn_` raise
`RecursionError: maximum recursion depth exceeded`: `flag_gems` falls back to the
torch op it is patching.

*Process crash (2).* `native_batch_norm` and `_batch_norm_no_update` exit with
return code `-11` on all seven profiles. They do so on the CUDA boxing route as
well, so this is not a FlagGems-specific defect; they are rolled back because a
7/7 crash is not a support claim, not because CUDA boxing answers them.

*Silently wrong where CUDA boxing also fails (1).* `native_dropout_backward`
returns a wrong tensor where CUDA boxing raises `NotImplementedError:
"masked_scale" not implemented for 'Long'`. Both routes fail that case, so the
paired criterion alone would keep it on `flaggems`; it is returned to CUDA boxing
anyway, because a loud error is the contract ATen defines there and a wrong tensor
is not.

**Withdrawn cohort.** An earlier entry in this report recorded 520 active routes
for the same change, at configuration SHA-256
`224f9d7c17f84db4e2a3aac5ab21efd2c34651083c701478a288489600a41397` and harness
SHA-256 `11e219b9ed0ed8d40cc03b1e2a8921490d6ff2ad35f3f15a8c0d5f1fcbed4cce`. That
measurement is withdrawn. It ran against a `libtorch_fl.so` whose FlagGems Python
dispatcher slot was empty, and `Dispatcher::GetFn` (`csrc/aten/dispatcher.h`)
degrades `Backend::kFlagGems` to `cuda_fn_` when the slot is absent, so every
`flaggems` route in that run executed CUDA boxing and the cohort recorded
CUDA-boxing behaviour under a FlagGems label. The raw cases show it: of the 98
overloads this change returns to CUDA boxing, 88 recorded `PASS` in the withdrawn
cohort where the same overload on the same hardware fails on the FlagGems route.
The withdrawn cohort's own 36-overload re-route check -- which forced every
failing overload onto CUDA boxing -- found 34 of 36 case-status vectors unchanged,
which is what a degraded route set predicts and the one result a populated
FlagGems route cannot produce. This is also why the rollback criterion had to
become a paired measurement: one survey cannot tell a FlagGems failure from a
CUDA-boxing success if the FlagGems slot may be empty. No number from the
withdrawn cohort is carried forward anywhere in this report.

**Generator reproducibility and cohort size, with an evidence gap.** The
checked-in configuration is the generator's output over a FlagGems cohort of 520
wrappers: masking the wrappers the locally installed FlagGems exposes on top of
that cohort and rerunning `FLAGOS_CODEGEN_ALL=1 scripts/codegen/codegen_ops.py`
reproduces `torch_fl/configs/backends_cuda.conf`'s route values exactly, apart
from nine `# tileops` annotation lines the checked-in file does not carry -- the
same nine that `main`'s configuration is already missing, since `TILEOPS_OPS` in
`scripts/codegen/backend_coverage.py` lists 60 ops while the file annotates 51. In
this environment the recorded FlagGems revision exposes 52 further wrappers
(`_cdist_forward`, `addbmm`, `cholesky_solve`, `huber_loss`, `linalg_lstsq`,
`polygamma`, `scatter_add`, `sign`, `take`, `_fused_rms_norm` and 42 others), and
the checked-in configuration routes all of them to `cuda`. They were never
candidates for the FlagGems route in this cohort, so this measurement says nothing
about them in either direction: they are **not revalidated**. Putting them on the
FlagGems route requires regenerating *and* rebuilding `libtorch_fl.so` -- the
generated `flaggems_python_kernels.cc` is what populates the dispatcher slot, so
routing them without rebuilding would reproduce exactly the silent degradation
described above -- followed by a fresh survey. That work is not part of this
change.

**FlagTree in CI.** The CUDA manifest installs the FlagTree Triton provider and
the FlagGems overloads in the job, on top of a pinned image. A FlagTree wheel is
a source-free build that links its bundled `libtriton.so` against the glibc
symbol versions of the distribution it was built on, which puts a floor under the
userland the job can run in: every published NVIDIA wheel binds the C23 `strtol`
family at `GLIBC_2.38` — `__isoc23_strtol` in 0.5.0/0.5.1,
`__isoc23_strtol`/`__isoc23_strtoll`/`__isoc23_strtoull` from 0.6.0 through
0.6.2a2 — so on an older userland the wheel installs and then `import triton`
fails on a missing symbol. `LD_PRELOAD` cannot substitute for it: `DT_VERNEED` is
resolved against the named file `libc.so.6`, so a shim under another soname is
never consulted.

The CUDA CI image carried by both entry points is now Ubuntu 24.04 (glibc 2.39),
so the image satisfies that floor and the manifest's FlagTree steps — the
`Check FlagTree and FlagGems` gate and the `FLAGOS_USE_FLAGTREE=1` torch.compile
run — can execute there. The A100 numbers above were measured on a local Ubuntu
24.04 host (glibc 2.39), so this cohort and the CI image now share a userland.
`.github/scripts/set_env_cuda.sh` compares the image glibc against
`TORCH_FL_FLAGTREE_MIN_GLIBC` (default `2.38`) before installing FlagTree, so a
future rebuild on an older userland fails naming the image requirement instead of
reporting a Triton symbol error.

`set_env_cuda.sh` takes the accelerator PyTorch, and the FlagGems C++ operators
that go with it, from the interpreter the image already ships
(`TORCH_FL_CUDA_VENDOR_MODE=auto` resolves to the image when it imports a CUDA
`torch`), and falls back to `bootstrap` — installing the cu130 build into a
job-local interpreter and compiling the operators there — for an image that
carries neither. The MetaX, PPU, DCU, Ascend and GCU rows are **not revalidated**
by this change: their configurations are untouched and their numbers still
describe the baseline cohort.

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

**The Enflame GCU S60 row was superseded on 2026-09-15** — a generated FlagGems
registration for GCU took the platform to 257 `flaggems` / 144 `gcu` / 1635
`none` and 401 registered ops. See "Enflame GCU S60 FlagGems routing
(2026-09-15)" above.

**The MUSA row was superseded on 2026-09-14** — the MUSA FlagGems registration
generator was restored, taking MUSA to 468 `flaggems` / 47 `musa` / 1521 `none`
and 515 registered ops. See "MUSA FlagGems routing restored, in-place arithmetic
routed back to mudnn (2026-09-14)" below. It moved again on 2026-09-15 to
464 `flaggems` / 51 `musa` / 1521 `none`, with the registered-op set unchanged
at 518. See "MUSA integer division: mudnn `TRUEDIV` promotion and FlagGems
floor-divide tail store (2026-09-15)". The Ascend numbers are the ones committed
in its shipped configuration; re-running `gen_vendor_confs.py` today would move
Ascend to 243 `flaggems` / 131 `ascend`, a pre-existing drift that predates this
work and is out of scope here. The GCU column of this table is now stale in the
same way and is kept only as the historical 2026-09-10 baseline.

The FlagGems count differs per platform because it is now the intersection of the
shared coverage set with that platform's registrations, not the shared set
itself. Ops the vendor also implements but that FlagGems covers are routed to
FlagGems by priority; a trailing `# <vendor>` annotation records the kernel so it
stays recoverable and `ALL_USE_VENDOR` can find it (248 such kernels on Ascend, 88 on GCU, 115 on
MUSA — MUSA's remaining 7 FlagGems routes come from
`musa_flaggems_register.inc`, which registers wrappers without native kernels
behind them). On GCU that 88 became 144 with the 2026-09-15 rerouting.

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

**Superseded for GCU.** The Enflame GCU S60 half of the table was measured on
hardware on 2026-09-15 and replaced by the routes in "Enflame GCU S60 FlagGems
routing (2026-09-15)" above; the GCU figures below are the 2026-09-10 baseline.
Ascend, MUSA, MetaX, DCU, PPU and Tsingmicro remain **not revalidated** against
either revision.

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
  registration set and none registered-but-left-`none`). GCU is 401/401 as of
  2026-09-15.
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
| 2026-09-15 | MTT S5000 (8 devices) | MUSA FlagGems gap re-measurement | Re-probed all 18 `NATIVE_TRITON_GAPS["musa"]` entries against the FlagGems revision the MUSA CI job installs, on each entry's recorded failure signature. Four no longer reproduce and are promoted out of the set: `index_add` and `index_add_` (recorded as "returns all zeros") now route to `flaggems` from `none`, and `randn`/`randn_like` (recorded as "crashes unpacking generator state") route to `flaggems` with the mudnn kernel retained as `flaggems  # musa`. The other fourteen keep their routes with provenance updated to `4d9c34775`; `_conj` stays because its probe *passes* (flag_gems materializes the conjugate where ATen's lazy view must set the Conjugate bit). MUSA `flaggems` 464 -> 468, `musa` 51 -> 49, `none` 1521 -> 1519; registered-op set unchanged at 518, `musa_flaggems_register.inc` 357 -> 359 `m.impl` lines. FlagGems is not patched. A100/mc550/PPU/DCU rows and every non-MUSA platform are **not revalidated**. | Per-op probe, one fresh process each, `FLAGOS_OP_*` pinning the op back to FlagGems, `FLAGOS_LOG_DISPATCH=1`/`FLAGOS_LOG_FALLBACK=1`: `index_add`, `index_add_`, `randn`, `randn_like` PASS (0/7, 0/7, 0/4, 0/4) and the 13 entries kept in the set reproduce their recorded signature exactly (bf16 `failed to translate module to LLVM IR`; `no fallback function is registered for schema aten::mul.out` for f32 and bf16, with `aten.mul.out` itself verified usable on MUSA and the `flag_gems/ops/mul.py:587` device-name guard confirmed live; the trailing-store loss at `n = 3,5,6,7,9,15,17,31,33,100`; `RuntimeError: MudnnCopy: unsupported dtype Long -> UInt32`). Comparator control rejects a perturbed reference. Provenance beyond verdicts: `index_add`/`index_add_` each add a new flag_gems code-cache entry, so the mthreads kernel compiled and ran on device, and every fallback line in those rows is the probe's own CPU comparison. End-to-end on the rebuilt library with the shipped conf: 14/14 cases pass, dispatch log showing `index_add`/`index_add_`/`randn`/`randn_like -> flagos_python` against `sort`/`add.Tensor -> musa` regression controls. `index_add` with duplicate indices and `alpha = 2.5` is bounded, not assumed: 11/20 seeds differ from CPU by at most `4.768e-07` (one float32 ULP) on the duplicated rows only, `alpha == 1` bit-exact, matching ATen's documented order-freedom for duplicate indices. CI groups re-run: dispatch 113 passed; factory 46 passed; operator cohort 493 passed/1 skipped/521 deselected/2 xfailed/1 xpassed plus the 3 pre-existing consistency failures; RNG 80 passed/37 deselected with the manifest's `-k` filter. Generators idempotent (`codegen_musa_flaggems.py --check` "is up to date", `gen_vendor_confs.py --check` clean for MUSA); `codegen_musa_flaggems.py` must run before `gen_vendor_confs.py`. `tests/unit/test_gen_vendor_confs.py`: 34 passed, 1 pre-existing ascend/gcu drift failure. `flaggems_overload_survey.py` cannot measure these routes — evidence gap recorded in the section above. |
| 2026-09-15 | Enflame GCU S60 (8 `flagos` devices) | GCU FlagGems routing | Made `backends_gcu.conf` FlagGems-first via a new generated registration file (`scripts/codegen/codegen_gcu_flaggems.py` -> `csrc/aten/backends/gcu/generated/gcu_flaggems_register.inc`, 249 `m.impl` lines), included by `csrc/aten/register.cc` after `gcu_register.inc`. GCU `flaggems` 0 -> 257, `gcu` 152 -> 144, `none` 1884 -> 1635; accelerated routes 152 -> 401 (7% -> 19.7%). `NATIVE_TRITON_GAPS["gcu"]` 108 -> 225: 81 routes measured wrong at `float16`/`float32`, plus 36 that fail only for `int64`/`bool` and have a topsaten kernel to fall back to. 76 `int64`-only routes with no topsaten kernel are deliberately **left on FlagGems** rather than demoted to `cpu_fallback` for float too; they now raise `Pipeline run failed` for an `int64` operand where the previous configuration served the call through `cpu_fallback`. FlagGems is not patched or forked. Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are **not revalidated** and no route changed for them. | `flaggems_overload_survey.py` (harness v4) on the S60 against flagtree `0.6.1+enflame3.6` (Triton 3.6, backend `enflame`, FlagGems master `3c6f7537d`), 7 profiles per overload over all 374 FlagGems routes (a transient un-gapped draft of `backends_gcu.conf`, `meta.conf_sha256` `82f801778c…`; it reconciles with the shipped conf as 374 - 117 = 257 and is not byte-recoverable): 314 tested, 121 strict, 121 clean on every exercised profile, 81 wrong at `float16`/`float32`, 112 wrong only for `int64`/`bool`, 60 with no constructible case. Failure families reproduced and recorded: GCU300 `64-bit data type not supported` / `Pipeline run failed: PassManager execution failed` (largest family), `arith.maxsi` UNREACHABLE at `PtrAnalysis.cpp:1711` (`_adaptive_avg_pool2d`), `unsupported extern elementwise: __nv_asinf` UNREACHABLE at `ElementwiseFusionOpToGCU.cpp:874` (`asin`), SIP abort at `dtu_context_obj.cc:693` (`addr`), SIGSEGV (`native_batch_norm`, `_batch_norm_no_update`), and measured wrong values on float profiles (`elu` `max_diff` 0.38-0.89, `histc` up to 1536, `_softmax_backward_data` returning `int8`, `sum.out` returning `(32, 32)` for `()`). Full `.github/configs/gcu.yml` pytest manifest run locally in one pass, all seven groups rc=0: vendor operator cohort 595 passed/32 skipped/499 deselected/2 xfailed/2 xpassed, FlagGems runtime path 9 passed/4 skipped/1116 deselected/1 xpassed, unified RNG 111 passed/4 skipped/1 deselected/1 xpassed, general 46 passed, AMP 27 passed, math-bits 12 passed, `torch.compile` 29 passed/18 skipped; conf consistency 7 passed; routing equals registration (no op routed to `gcu` without a `gcu_register.inc` entry, none registered-but-left-`none`). Both generators idempotent (two runs byte-identical; `--check` exit 0). The environment group (`set_env_gcu.sh`, `CI_STAGE=integration`) was reproduced into a scratch venv: TopsRider discovery, `/dev/gcu0`, the venv bootstrap, CPU torch 2.10.0, flagtree from the FlagOS index and FlagGems `3c6f7537d` from git all succeed, and its Triton/flag_gems verification snippet passes; on the measurement host alone it needs a local, uncommitted retarget of libtriton.so's single glibc-2.38 symbol, because that host is Ubuntu 22.04 while the wheel and the pinned ubuntu24.04 CI image are not. Evidence gaps recorded: the 60 unconstructible routes are not measured, the two batch-norm process deaths are gapped on exit status alone because the harness truncates stderr at 300 bytes, and no part of this change has been executed by CI yet. |
| 2026-09-15 | MTT S5000 (8 devices) | MUSA integer division (issue #266) | Fixed two integer-division defects in the generator, not with handwritten kernels. `int64 / int64` raised `Unsupported binary mode: TRUEDIV, with left data type: INT64` because the generated kernels took `result_dtype` from `at::result_type` (int64) while ATen promotes integer true division to float32; `_TRUEDIV_INT_TO_FLOAT` now widens integral results, guarded on `!rounding_mode.has_value()` so `'floor'`/`'trunc'` keep int64. Integer `//`, `floor_divide`, and `floor_divide_` silently lost the trailing element on non-power-of-two `numel` in FlagGems; new `binary_mode` / `binary_inplace_mode` categories plus the `floor_divide_.Tensor` native entry route `div.Tensor_mode`, `div_.Tensor_mode`, `floor_divide` and `floor_divide_.Tensor` through mudnn `FLOORDIV`/`TRUNCATEDIV`/`TRUEDIV` via `SetMudnnDivMode`. MUSA `flaggems` 468 -> 464, `musa` 47 -> 51, `none` 1521; registered-op set unchanged at 518 (three overloads moved from the FlagGems registration to the native one). FlagGems is not patched. Other platforms are **not revalidated** and no FlagGems route changed for them. | 59-case CPU-parity probe on `flagos:0` run against both this tree and a base-commit worktree (out-of-place, in-place, scalar and tensor operands, both rounding modes, negatives, `out=`, broadcasting, and `floor_divide` at `n = 2,3,4,5,7,8,15,17,33,100`): 39 exact / 7 float-approximate / 13 mismatches before, 43 exact / 14 float-approximate / 1 error-text match / 1 probe-harness mismatch after. Integer floor division and every `rounding_mode` case exact; the float-approximate cases are true division one float32 ULP from CPU and reproduce identically on pure-float inputs on the base tree (pre-existing mudnn `TRUEDIV` arithmetic, not this change). `FLAGOS_LOG_DISPATCH=1` shows all five overloads on `-> musa`; pinning the four rerouted overloads back onto FlagGems via `FLAGOS_OP_*` reproduces the tail loss (`[5, 5, 0]` for `[5, 5, 6]` at n=3) and leaves true division correct, isolating the routing fix causally. Full `.github/configs/musa.yml` run locally: dispatch 104 passed/1 skipped, factory 46 passed, AMP 27 passed, math-bits 12 passed, profiler 10 passed/1 skipped/1 xpassed, operator cohort 493 passed/1 skipped/513 deselected/2 xfailed/1 xpassed, RNG 80 passed/37 deselected. Generator idempotent (two runs byte-identical; `codegen_musa_flaggems.py --check` and `gen_vendor_confs.py --check` clean for MUSA). `flaggems_overload_survey.py` cannot measure these routes: it selects `flagos_python` entries, and the rerouted overloads are exactly the ones that left that route — evidence gap recorded in the section above. Three pre-existing `test_flaggems_conf_consistency.py` failures (`mm`/`bmm`/`addmm` dispatcher drift) reproduce byte-identically against the pristine conf. |
| 2026-09-14 | MTT S5000 (8 devices) | MUSA FlagGems routing and in-place arithmetic fallback | Restored the MUSA FlagGems registration generator, taking MUSA from 158 to 515 registered ops and from 122 to 468 `flaggems` routes (`musa` 36 -> 47, `none` 1878 -> 1521). Moved 14 ops into `NATIVE_TRITON_GAPS["musa"]` so they fall back to mudnn instead: `add/sub/div.Tensor` and their in-place forms plus `mul_.Tensor` (bf16 wrapped-number promotion reaches `llvm.musa.float2bfloat16` with a double operand), `randn`/`randn_like`, `sort`/`sort.stable`, and `_conj`/`index_add`/`index_add_`, which route to `none` because mudnn has no kernel for them. FlagGems is not patched. Ascend, GCU, DCU, MetaX and PPU rows are **not revalidated** by this change and no FlagGems route was altered for them. | Every group of `.github/configs/musa.yml` run locally on hardware: dispatch 104 passed/1 skipped, factory 46 passed, AMP 27 passed, math-bits 12 passed, profiler 10 passed/1 skipped/1 xpassed, operator cohort 490 passed/2 skipped/512 deselected/2 xfailed/1 xpassed, RNG 80 passed/37 deselected. The bf16 gap was reproduced causally with `FLAGOS_OP_add__Tensor=flaggems`, which reproduces the remote CI's `failed to translate module to LLVM IR` on `test_autocast_fp32_policy[dtype1]` and passes on the shipped route. Three `flaggems`-marked dispatch-log tests that hard-coded `flagos_python`/`cuda` were rewritten to read the route from the platform conf (`tests/integration/ops/backend_conf.py`); they were the only failures in CI group 7 on `6f8128e` and pass on every platform's conf afterwards. Generator idempotent (`codegen_mudnn.py` twice, byte-identical; `codegen_musa_flaggems.py --check` and `gen_vendor_confs.py --check` clean for MUSA). `tests/unit/test_gen_vendor_confs.py`: 34 passed, 1 pre-existing failure (ascend/gcu conf staleness, unrelated). Three pre-existing `test_flaggems_conf_consistency.py` failures reproduce byte-identically against `d0e2d1a`'s data files, so they are not introduced by this change. |
| 2026-09-15 | NVIDIA A100-SXM4-40GB (8 devices) | CUDA full-coverage configuration (416 active routes, harness v6) | Full CUDA code generation now routes schema-compatible FlagGems Python wrappers to `flaggems`, with the overloads that then measured worse there returned to CUDA boxing; the checked-in CUDA configuration moves from 13 to 416 `flaggems` routes and from 2021 to 1618 `cuda` routes, with 40 of the 51 TileOPs-annotated routes following it. Two of the 13 pre-existing `flaggems` routes, `embedding` and `sum.dim_IntList`, go back to CUDA boxing; the other 11 keep theirs. The 98 returned overloads are recorded in `scripts/codegen/codegen_ops.py:measured_flaggems_rollback`, so the configuration is generator output rather than a hand edit. CUDA CI installs the NVIDIA source-free `flagtree==0.6.2a2` wheel and the current FlagGems default branch (`master`; the repository has no `main` branch). Route priority is unchanged and no other platform's configuration was altered. **MetaX, PPU, DCU, Ascend and GCU are not revalidated.** | Measured with `tests/manual/flaggems_overload_survey.py` (v6, SHA-256 `31334631`) against `torch_fl/configs/backends_cuda.conf` (SHA-256 `ab2522b7`, active route-set SHA-256 `0b344884`) at torch-fl `93568ac` with FlagGems `7fb49bad47116434961bfb2b912811716d383eaf`: 416 registered, STRICT 321, BASIC_ONLY 7, FAILED 0, UNTESTED 88; case-level PASS 1817 / INVALID_CASE 1085 / ERROR 2 / WRONG 8 / CRASH 0 / TIMEOUT 0 / UNVERIFIABLE 0 / context poison 0 (2912 = 416 x 7). Every rollback was decided by a paired run of the same harness on the same host, once per route: an overload goes back to CUDA boxing when the FlagGems route fails a case CUDA boxing answers correctly, or crashes, hangs, or recurses; an overload whose failure vector is identical on both routes stays on `flaggems` as BASIC_ONLY. The seven partial overloads (`kthvalue`, `median.dim`, `mm`, `mm.out`, `mode`, `sort`, `sort.stable`) produced identical case-status vectors on both routes, so no residual failure is attributable to the routing. The generator reproduces the configuration's route values exactly over its 520-wrapper FlagGems cohort; the locally installed FlagGems exposes 52 wrappers beyond that cohort, which stay on `cuda` and are **not revalidated** (evidence gap recorded in the section). FlagTree Triton 3.6 needs glibc >= 2.38 and the CUDA CI image is now Ubuntu 24.04, so the manifest's FlagTree steps run there; the survey ran on a local Ubuntu 24.04 host (glibc 2.39). The 2026-09-14 CUDA row below is **withdrawn**: it was measured while the FlagGems Python dispatcher slot was empty, so its `flaggems` routes executed CUDA boxing. See "CUDA FlagGems-first routing with FlagTree Triton 3.6 (2026-09-15)" for the cohort definition and the per-group rollback list. |
| 2026-09-14 | NVIDIA A100-SXM4-40GB (8 devices) | CUDA full-coverage configuration (520 active routes, harness v6) — **withdrawn** | **Withdrawn cohort:** measured while the FlagGems Python dispatcher slot was empty, so every `flaggems` route in this row executed CUDA boxing and its verdicts are not FlagGems results. Superseded by the 2026-09-15 row. Updated full CUDA code generation so schema-compatible FlagGems Python wrappers are routed to `flaggems` instead of CUDA boxing; the checked-in CUDA configuration moves from 13 to 520 `flaggems` routes and from 2021 to 1514 `cuda` routes, with 46 of the 51 TileOPs-annotated routes following it. CUDA CI installs the NVIDIA source-free `flagtree===0.6.2a2` wheel and the current FlagGems default branch (`master`; the repository has no `main` branch). Route priority is unchanged and no other platform's configuration was altered. **MetaX, PPU, DCU, Ascend and GCU are not revalidated.** | Measured with `tests/manual/flaggems_overload_survey.py` (v6, SHA-256 `11e219b9`) against `torch_fl/configs/backends_cuda.conf` (SHA-256 `224f9d7c`, active route-set SHA-256 `290e7c90`) at torch-fl `13cbf4b` with FlagGems `7fb49bad47116434961bfb2b912811716d383eaf`: 520 registered, STRICT 393, BASIC_ONLY 23, FAILED 13, UNTESTED 91; case-level PASS 2281 / INVALID_CASE 1266 / ERROR 33 / WRONG 46 / CRASH 14 / TIMEOUT 0 / UNVERIFIABLE 0 / context poison 0 (3640 = 520 x 7). 36 overloads recorded a failure; the same 36 rerun on the CUDA-boxing route set produced 34 identical case statuses, with `index_copy`/`index_copy_` trading one `2d-i64` status. Under the degraded route set that identity is expected and is not evidence that the rerouting is safe; the paired re-measurement in the 2026-09-15 row replaces this attribution. FlagTree Triton 3.6 could not run on the pinned CI image (Ubuntu 22.04, glibc 2.35, against the `GLIBC_2.38` the FlagTree wheels bind); the survey ran on a local Ubuntu 24.04 host (glibc 2.39) and `.github/scripts/set_env_cuda.sh` guards the image glibc before installing FlagTree. See "Withdrawn cohort" in "CUDA FlagGems-first routing with FlagTree Triton 3.6 (2026-09-15)" for why this row cannot be compared with the current configuration. |
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
