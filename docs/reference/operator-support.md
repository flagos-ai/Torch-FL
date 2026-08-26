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

The NVIDIA, MetaX, PPU, and Hygon rows below use the same generic active route
set and survey methodology. Kunlun is reported separately with its measured
five-route cohort. These revisions identify the measured cohorts; they do not
describe the current repository HEAD.

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

Rates use each row's own active route count as the denominator and are rounded
to one decimal place. That denominator is 546 for every row except Kunlun P800,
which is a separate five-route cohort.

| Hardware | Total | STRICT | BASIC_ONLY | FAILED | UNTESTED | Basic executable | Basic rate | Strict rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NVIDIA A100 | 546 | 348 | 54 | 46 | 98 | 402 | 73.6% | 63.7% |
| MetaX mc550 | 546 | 260 | 33 | 155 | 98 | 293 | 53.7% | 47.6% |
| PPU 810e | 546 | 347 | 54 | 47 | 98 | 401 | 73.4% | 63.6% |
| Hygon DCU bw1000 | 546 | 321 | 53 | 74 | 98 | 374 | 68.5% | 58.8% |
| Kunlun P800 | 5* | 4 | 1 | 0 | 0 | 5 | 100.0% | 80.0% |

*Kunlun uses a separate five-overload measured cohort, not the generic 546-route
cohort above. The active routes are `add.Tensor`, `add_.Tensor`, `cos`,
`div.Tensor`, and `mean`. `add_.Tensor` is `BASIC_ONLY` because its float16
case was `WRONG` (`max_diff=1.8424072265625`); all other valid cases passed.
The unvalidated generic routes remain on CUDA boxing in
`backends_kunlun_flaggems.conf` and are not included in this row. Evidence
provenance: config SHA-256
`db34e51d1fcdb06f778ff84475fc77c41ac24b9bd6db8235de37227932dd4fa3`, active
route-set SHA-256
`0f29789c27931e007ff189272a5f0f8f0b07473d265a8b23e0109e66ee2ba1f4`, and raw
survey JSON SHA-256
`9d4e746d3566ea1a39601815e190aa6332ac41abd5f72f82c16e1767c53936c6`.

For every measured row, `STRICT + BASIC_ONLY + FAILED + UNTESTED = Total`, and
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
| Kunlun P800 (5-route cohort) | 28 | 6 | 0 | 0 | 1 | 0 | 0 | 0 |

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

## Update History

| Date | Hardware | Cohort | Change | Evidence |
|---|---|---|---|---|
| 2026-08-17 | Enflame S60 | Native GCU RNG routes | Added 16 topsaten RNG routes; generic FlagGems cohort not revalidated. | Targeted mixed native/FlagGems probe verified shared seed/offset progression and replay; `tests/integration/ops/test_rng_dispatch.py`: `104 passed, 2 skipped, 1 xpassed`. |
| 2026-08-26 | Kunlun P800 XPU | Kunlun AMP routes (`_to_copy`, `_copy_from`, `_amp_foreach_non_finite_check_and_unscale_`) | Registered the generic `AutocastPrivateUse1` policy table for Kunlun, rerouted Kunlun dtype conversion to the portable host-staged path, and routed the AMP non-finite/unscale overload to a Kunlun-safe helper. FlagGems and generic 546-route cohorts are **not revalidated** because their active routes are unchanged. | Measured on P800 OAM (XPU-RT 5.37.1, torch 2.9.0+cu129, `ACCELERATOR=kunlun`, `FLAGOS_USE_FLAGGEMS=0`, device `flagos:0`, 8 devices). `tests/integration/test_amp.py`: `15 passed in 1.51s` (FP16 and BF16 parametrizations). Targeted probe measured autocast nesting/exception cleanup, FP32↔FP16 casts, float64 copy, the lower-precision/FP32-preserving/explicit-dtype/promote policy matrix against CPU references, `binary_cross_entropy` banning, non-finite detection (`found_inf` 0.0 finite / 1.0 non-finite with unscale to 1.0), GradScaler finite step and growth (param 1.6, scale 16.0), overflow step-skip and backoff (param 2.0, scale 4.0), and a single-device training step against a CPU FP32 reference. Distributed AMP (DDP/FSDP2 with GradScaler) and AMP performance remain **not validated** on this hardware. |
| 2026-08-19 | Kunlun P800 | Kunlun FlagGems cohort (5 routes) | Enabled the FlagGems Python build and added `backends_kunlun_flaggems.conf` with 5 measured routes; the generic 546-route cohort stays **not revalidated** on this hardware. | `flaggems_overload_survey.py` on P800 OAM (XPU-RT 5.37.1, driver 5.0.21.43, torch 2.9.0+cu129, FlagTree Triton at `/opt/flagtree`, harness v4, 7 profiles): `registered=5, tested=5, basic_executable=5, strict_support=4`; raw cases `28 PASS / 6 INVALID_CASE / 1 WRONG`. `exp`, `mm`, `mm.out`, `addmm`, and `mul_.Tensor` measured as FAILED on the generic config and are routed to CUDA boxing instead. `test_flaggems_conf_consistency.py`: `8 passed`. The RNG dispatch suite is **not** part of this evidence: on P800 it gives `51 failed, 41 passed, 11 skipped, 1 xfailed` with `TestRngMultiDevice` deselected and aborts inside that class otherwise, identically with FlagGems on and off and identically at the branch base, so Kunlun RNG is a pre-existing gap and Kunlun FlagGems RNG remains unvalidated. |
| 2026-08-18 | Kunlun P800 | Generic FlagGems cohort | Added Kunlun CUDA-boxing support; overload cohort **not revalidated**. | P800 runtime smoke measured 8 devices, allocation, H2D/D2H, stream creation, synchronization, cache release, and float32 `mm` against CPU reference. The build disables FlagGems pending a dedicated survey. |
| 2026-08-14 | Ascend 910 (2 devices) | Native Ascend FSDP2 routes | Added `_chunk_cat`, `_chunk_cat.out`, `_foreach_copy_`, `cat.out`, `split.Tensor`, `split_with_sizes`, and `split_with_sizes_copy.out`; generic FlagGems cohort not revalidated because it is unchanged. | Manual FlagCX collective, DDP, and FSDP2 tests on CANN 9.0; standard FlagGems harness is not applicable to native routes. |
| 2026-08-13 | A100, mc550, 810e, bw1000 | torch-fl `fe2272b5`, FlagGems `7fb49bad`, harness v4 | Established the verified 546-overload four-platform baseline. | Manual survey JSON; aggregate and raw counts recorded above. |
