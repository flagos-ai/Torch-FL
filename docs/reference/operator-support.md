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

**The shipped tree has moved on from this cohort.** The harness in
`tests/manual/flaggems_overload_survey.py` is version 6 (SHA-256
`7b01c22ce3a94315f1364df242323e9faac27f2585debfb05030670c7c756cc7`), and the
shared coverage set has been widened to 639 overloads on the FlagGems master
cohort (`5a58df410`), which is what the MetaX configuration below is measured
against. The four hardware rows in this section were **not** re-measured against
that cohort and remain the `fe2272b5` / `7fb49bad` baseline, as the table says.
The harness version and hash above are the ones this tree ships; every measurement
recorded against "harness version 5, SHA-256 `cfd09e50…`" by an earlier revision
of this report was taken with the same version 6 file, because no version 5 of
this harness exists in the repository history.

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

### MetaX: `scaled_dot_product_attention` routed to FlagGems (2026-09-18, MetaX C550)

`scaled_dot_product_attention` is a `flaggems` route in `backends_metax.conf`,
which now reads **592 `flaggems` / 12 `flaggems_cpp` / 1433 `cuda`** over a
**2037**-op list (SHA-256
`bb1dc5c4550339dcd44ac438b2981e703c882025e477221e86cac37d833f58f2`), superseding
the 591 / 12 / 1433 over 2036 ops the `slice.Tensor` entry below records. Every
op in the list is still accelerated. MetaX still routes 584 of the 639 overloads
in the raised FlagGems ceiling through the Python path and 12 through the C++
slot, and still holds 43 on the cuda boxing kernel; this op is outside that
ceiling, so those three numbers do not move.

**Why the op is not a generated kernel.** `aten::scaled_dot_product_attention` is
a composite, and the generated kernels this report otherwise counts are leaves.
Its fused-backend selection runs *inside* the composite and then branches on
`query.device().type()`, so on PrivateUse1 the leaves are never consulted and
routing them per-op can never reach a fused kernel. The override that fixes that
is hand-written — `csrc/aten/sdp_choice_stub.cc` registers the composite itself
on PrivateUse1 and decides inside it — and it is registered only on the
CUDA-boxing builds, which is why `EXTRA_ROUTED` in
[`scripts/codegen/gen_vendor_confs.py`](../../scripts/codegen/gen_vendor_confs.py)
exists: the op appears in no `.inc`, so no coverage scan can see it and the op
list has to be widened by hand. A second hand-written set,
`METAX_COMPOSITE_FLAGGEMS`, holds it from every other platform's conf exactly the
way `METAX_FLAGGEMS_MEASURED` holds the measured leaf routes — a measurement
taken on one platform is not a route for another.

**Measured.** Eight-device C550 host, `flagtree 0.6.1+metax3.6`, MACA 3.8.0 in
CUDA-boxing mode, `flag_gems 5.4.0rc2.post1+g5a58df410`, Triton 3.6.0 with the
`metax` backend. Qwen-Image-2512, 1024x1024, 50-step denoise, one seed:

- `run-on.json` / `run-off.json`, `build_pipeline` and placement held constant:
  steady **78.49 s against 184.02 s**, first step 90.26 s against 189.34 s
  (**2.34x**, 57.3% off the loop), with `pipeline_load_s` 134.74 in both arms.
- The same window re-run as three arms that also write latents and pixels:
  route active **84.46 s**, route off **183.68 s** and **182.39 s**.
- Op level, `(1, 24, 4114, 128)` bf16, the shape the route serves, min of five
  calls after warm-up: FlagGems `5.908 ms/call` against the boxing route's
  `23.313 ms/call` (**3.95x**).

**The envelope is the probe's predicate.** The route serves bf16, 4-D, head_dim
128 exactly, query seq >= 1024, no mask, no causal, no explicit scale, no dropout,
no gqa — Qwen-Image-2512's joint attention, 120 calls per denoise step. Nineteen
calls bracket that shape on each side and every one of them takes the route the
envelope says it should; `route hits: 6` of 19, the six being the joint shape and
its seq-4096/2048/1024/non-contiguous variants:

```text
match q(1,24,4114,128) bf16  gems   gems    ok  out (1, 24, 4114, 128) bfloat16
head_dim 64                  box    box     ok  out (1, 24, 4114, 64) bfloat16
head_dim 256                 box    box     ok  out (1, 24, 4114, 256) bfloat16
head_dim 512 (VAE)           box    box     ok  out (1, 24, 4114, 512) bfloat16
seq 1023 (just under)        box    box     ok  out (1, 24, 1023, 128) bfloat16
fp16 / fp32                  box    box     ok
mask present / is_causal     box    box     ok
scale given / dropout 0.1    box    box     ok
gqa 24q/8kv enable_gqa       box    box     ok
3-D operands                 box    box     ok  out (24, 4114, 128) bfloat16
```

**head_dim 128 is measured, not argued, in both directions.** At the VAE's
head_dim 512 the FlagGems kernel has no configuration that compiles on this part:

```text
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 294912, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
```

`attention.py:173` narrows `_attn_fwd`'s autotune set with `keep()` from
`flag_gems/ops/flash_kernel.py` — the always-kept tiles `(128, 32, 4)` and
`(128, 128, 8)` plus the 24 explicit `SMALL_HEAD_DIM_CONFIGS` (`BLOCK_M` in
{64, 128} × `BLOCK_N` in {16, 32} × `num_stages` in {2, 3, 4} × `num_warps` in
{4, 8}) — so every surviving configuration is capped at `BLOCK_N <= 32`, and an
`_attn_fwd` tile's shared-memory block grows with `BLOCK_DMODEL`. At head_dim 128
some admitted configurations are already over this part's limit (one asks for
`98304` B against `65536`) while a legal tile remains, so the op runs; at 512 the
same code path reports `294912` B and the whole surviving set is over the limit,
so the failure is the tile set rather than one autotuner candidate. At head_dim 128 the same call completes and
matches the unfused fp32 math path (`max 2.189e-05 mean 2.365e-06` over the same
`(1, 24, 4114, 128)` operand). The clause is not the kernel's outer boundary: at
head_dim 64 the same call also completes and matches the fp32 math path
(`max 2.153e-05 mean 2.421e-06`), at `2.983 ms/call` for a direct
`flag_gems.scaled_dot_product_attention` against `22.037 ms/call` for the same
shape through the boxing route in a `FLAGOS_OP_scaled_dot_product_attention=cuda`
arm, and a second arm at that head_dim, whose conf names `flaggems` while the
head_dim clause turns the call back to boxing, agrees at `3.010` against
`21.958 ms/call`. head_dim 128 is therefore the head_dim this route was measured
at, not the largest the kernel accepts, and widening the clause is a measurement
rather than an edit.

**Numerics.** The three op-level arms load the same seeded operands — digest
`37efe866eb24ada5` in all three — and write them out for comparison. The FlagGems
route against the boxing route differs by `max 9.766e-04 mean 3.274e-05`, the
delta of a fused bf16 attention against the vendor's own selection. The two
boxing arms agree **exactly** (`max 0.000e+00 mean 0.000e+00`): one is
`FLAGOS_OP_scaled_dot_product_attention=cuda` on the shipped conf, the other is a
conf that never names the op, and a zero delta between them is what shows the
unlisted op kept the boxing path instead of reading `kFlagGems` from
`GetBackendForOp`'s table miss. That reading is what the new
`HasBackendForOp()` in `csrc/aten/common.h` supplies, and it is why the route
cannot arrive silently on a conf — or a third party's wheel — that predates it.
Turning the route off is one line, at build time or at runtime
(`FLAGOS_OP_scaled_dot_product_attention=cuda`).

**Image cost, with its own control.** All three pixel arms share one seed and one
window, so the two route-off arms bound what the window can resolve at all:

| Comparison | max | mean (/255) | over 1/255 | over 4/255 |
|---|---:|---:|---:|---:|
| off#arm1 vs off#arm2 | 0.0000 | 0.00000 | 0 of 3145728 | 0 |
| off#arm1 vs on#arm1 | 170.3320 | 2.04607 | 1150098 of 3145728 | 312147 |

The measurement floor is exactly zero, so the whole second row is the route
decision. Both route-off arms give identical numbers against the route-on arm.
The prototype of this route, on the arms recorded in `sdpaend_img`, measured
`1.39572` for route-off against route-on and `1.25745` for route-off against
vendor torch — the same window, a different seed. The route's image cost is
therefore of the same order as the flagos-vs-vendor difference the wheel already
carries, which is a property of the route being taken and not of this change;
the shipped figure is the `2.04607` above.

**Survey.** `tests/manual/flaggems_overload_survey.py` (version 6, SHA-256
`7b01c22ce3a94315f1364df242323e9faac27f2585debfb05030670c7c756cc7`) was rerun on
the C550 with `backends_metax.conf` and scoped to the changed route:
`registered 1`, `tested 1`, and the verdict is **`FAILED`** — `basic_executable 0`,
`strict_support 0`. The report must record that, and the cause is measured: the
harness synthesizes this op's `dropout_p` as `0.5` — the argument is `float`, and
`default_for()`'s `base in ("float", "Scalar")` branch ends in a catch-all
`return 0.5` that names `p` but not `dropout_p` — and a dropout call is
non-deterministic, so `max_diff` against the CPU reference is not a measurement
of anything. Rerunning
the same seven profiles with `dropout_p = 0.0` turns all four `WRONG` verdicts
into `PASS` — `2d-f32` `1.4507` -> pass, `4d-f32` `3.3939` -> pass, `2d-f16`
`1.5486` -> pass, `2d-f32-strided` `2.2727` -> pass — and the three
`INVALID_CASE` profiles (`1d-f32`, `2d-i64`, `2d-bool`) stay CPU-reference
rejections at either dropout, so they are neither passes nor failures. Every one
of the four `WRONG` cases is also 2-D or otherwise outside the route's envelope,
so none of them entered the FlagGems path at all: the `FAILED` verdict is the
harness's, and it is the same verdict the op would draw on the boxing route. The
harness defect is not fixed here; it is reported separately.

**Guard tests.** `tests/integration/ops/test_metax_flaggems.py` gains
`TestMetaXFlaggemsSdpaRoute`, and `_MEASURED_FLAGGEMS_ROUTES` moves 591 -> 592 to
match the conf. Each arm runs every case in one fresh interpreter — the shipped
conf, then the shipped conf with the documented switch set — and asserts the
membership the envelope draws rather than that the op merely runs: one case is
the joint attention's shape class (bf16, 4-D, head_dim 128, no mask) at the
smallest seq the route admits, and each of the other seven differs from it in
exactly one respect — head_dim 64, seq 512, float32, float16, 2-D, `is_causal`,
an all-zero additive mask — so a case that reaches the kernel names the clause
that let it through. The class also pins the shapes the kernel saw, bounds every
case against a float32 host reference at `2e-2`, and checks that with the switch
off the kernel is never called and no case reaches it. On the C550 that class
reports **10 passed in 43.28s, 0 failed**, and the whole file reports
**101 passed in 854.53s, 0 failed** (exit 0) with the route active, against the
90- and 91-test cohorts the two MetaX entries below record.

`gen_vendor_confs.py` is idempotent: two consecutive runs leave all nine confs
byte-identical, and `--check` reports `all vendor confs up to date`. Ascend, GCU,
MUSA, DCU and PPU are **not revalidated** and no route changed for them — each
gained one line, `scaled_dot_product_attention` as `none` on ascend/gcu/musa and
`cuda` on dcu/ppu, with SHA-256 `311445fd…`, `9d144ca4…`, `e62cac76…`,
`dcd7b149…` and `e1f34144…` respectively. `backends_cuda.conf`,
`backends_bpu.conf` and `backends_tsingmicro.conf` do not carry the op: the CUDA
conf is written one line per generated wrapper and this op has none, TsingMicro's
is a copy of CUDA's, and BPU's is intentionally empty.

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

### Hygon DCU FlagGems path enabled by default in CI (2026-09-15)

`.github/scripts/set_env_dcu.sh` now installs the FlagTree wheel carrying the
`hcu` backend and FlagGems, and exports `FLAGOS_USE_FLAGGEMS=1` through
`$GITHUB_ENV`; `.github/configs/dcu.yml` no longer re-enables the path inline.
Before this, DCU's FlagGems group set `FLAGOS_USE_FLAGGEMS=1` against a venv the
setup script had populated with the image's Triton *by copy*, without its
`dist-info`. Triton discovers backends through `[triton.backends]` entry points,
so no backend was registered and the group could not exercise FlagGems at all.

**Three route values change.** `torch_fl/configs/backends_dcu.conf` is `main`'s
file with three edits, all in the same direction: `_conj`, `relu` and `relu_`
move from `flaggems` to `cuda`, for the contract reasons recorded below. The
routed-route counts move `flaggems 473 -> 470` and `cuda 1563 -> 1566`.
Everything else, including the `silu_backward` and `slice_backward` fallbacks to
the cuda boxing kernel, is unchanged; the rest of this change is what makes the
routes that file already carried reachable.

**A fourth route crosses once the MetaX coverage rebuild is merged in.** The
paragraph above describes `main`'s change on its own. On the branch carrying the
MetaX coverage-ceiling rebuild, `FLAGGEMS_PYTHON_OPS` no longer lists
`mul_.Tensor`: `flaggems_runtime_broken` in `scripts/codegen/codegen_ops.py`
holds it on the CUDA boxing kernel for every platform, because the FlagGems
kernel reaches `flag_gems/ops/mul.py`'s `out is not None` branch and redispatches
to `aten.mul.out`, which has no kernel registered for that keyset (measured on
MetaX C550; Ascend routes around the same kernel). `boxing_triton_gaps()` cannot
hold the op on FlagGems here either -- the gap set is recovered from the conf and
intersected with the ceiling, so an op outside the ceiling has no route to the
FlagGems path at all. The regenerated `backends_dcu.conf` therefore also moves
`mul_.Tensor` from `flaggems` to `cuda`, and the merged file stands at
469 `flaggems` / 1567 `cuda`. DCU is **not re-measured** by this: the direction is
the same as `_conj`/`relu` above, the op is a regression on the FlagGems route on
the hardware where it was measured, and the update-history row records the move.

Measured on Hygon DCU bw1000 with the revision in this change (FlagTree
`0.6.2a1+hcu3.6`, FlagGems `e7b4a865`, PyTorch 2.10.0), over the 13 operator
files whose FlagGems routes became reachable:

```bash
ACCELERATOR=dcu FLAGOS_USE_FLAGGEMS=1 python -m pytest \
  tests/integration/ops/test_{abs,add,bmm,cat,embedding,mean,mm,mul,neg,silu,softmax,sum,where}_dispatch.py \
  -m "not flaggems and not flaggems_python and not flaggems_cpp" -k "not dispatch_log" -q
```

`1 failed, 121 passed, 14 skipped, 52 deselected in 49.21s`. The single failure is
`test_mm_dispatch.py::TestMmDispatch::test_mm_half_hgemm_strict`, which asserts
`returncode == 0` on a child process; the child printed its dispatch line and
then died with signal 11 having run no test. That is the same host-level fault
described in the bw1000 raw-case note above — a bare `import torch_fl, torch`
with no operator executed also exits 139 there — not an operator verdict.

Two dispatch-log assertions needed the same treatment. `test_mm_dispatch.py` and
`test_bmm_dispatch.py` hard-coded `[flagos dispatch] mm -> flagos_python` and
`[flagos dispatch] bmm -> flagos_python`, but `backends_dcu.conf` pins `mm`,
`mm.out`, `bmm` and `bmm.out` to the cuda boxing kernel (FlagGems #6227), so DCU
correctly logs `-> cuda` and the assertion was wrong about the platform rather
than about the route. Both now assert `routed_backend("<op>")`, the helper the
repo already used for `cat`, `add` and `mul`. This is not a DCU-only regression
introduced here: `main`'s own DCU CI job 104211346098 (run 34867798275) reports
`2 failed, 11 passed, 1 skipped, 1109 deselected, 1 xpassed in 351.35s`, the two
failures being exactly those two assertions.

The first CI run of this branch produced no test evidence at all. The new
triton-cleanup step used `while pip uninstall -y triton triton_kernels; do :; done`,
and `pip uninstall -y` exits 0 even when it skips every named package, so that
loop cannot terminate. Job 104213552362 sat on it with every byte of output
redirected to `/dev/null` from 01:27:18Z until the 60-minute job timeout
cancelled the job at 02:26:59Z, before a single test ran. The loop is now bounded
by triton `dist-info` presence, and `tests/unit/test_dcu_env_script.py` guards
the shape of it on any host. The loop is no longer what blocks the group: the
sixth run of this branch (run 34931465738, job 104260457173) provisions the
intended stack on the runner (`Triton: 3.6.0 (backends: ['hcu'])`, `FlagGems:
5.4.0rc2.post1+ge7b4a865f (vendor: hygon)`), passes the device-availability
group, and runs `tests/unit/` to completion at 26/27 -- but
`run_integration_tests.py` returns on the first failing group, and the one
failing file is a repository-wide `gen_vendor_confs.py --check` on conf drift
belonging to Ascend and GCU, not to DCU. Both halves of that drift were cured
upstream while this branch was open -- GCU by #285, Ascend by #288 -- so this
change carries no conf edit of its own; at the time of that run, though, the
drift was what stopped the manifest, and
the FlagGems group and every group after it are never reached. The manifest therefore now runs the unit group
last: `run_integration_tests.py` returns on the first failing group, so a
repository-wide check with no DCU content was deciding whether any DCU hardware
group ran at all.

Run 34935660930 (job 104272971051, head `a89e869`) is the first on this branch
whose FlagGems group executes, and it passes:

```text
[3/11] Run operator tests (FlagGems runtime path, main ops)
==== 13 passed, 1 skipped, 1109 deselected, 1 xpassed in 367.26s (0:06:07) ====
```

The groups around it pass as well: `[2/11]` vendor backend **140 passed, 2
skipped, 981 deselected, 1 xpassed, 30 warnings in 803.62s (0:13:23)**, `[4/11]`
low-precision matrix **30 passed, 7 deselected**, `[5/11]` unified RNG **114
passed, 1 skipped, 1 deselected, 1 xpassed**, `[6/11]` general **46 passed**,
`[7/11]` AMP **25 passed, 1 skipped, 1 xpassed**. The run stops at `[8/11]`
math-bits on `_conj` (below), so `[9/11]` through `[11/11]` -- including the unit
group -- have still not run on that head.

Run 34939743596 (job 104285567077, head `d81d5b5`) is the first at a revision
carrying the `_conj` reroute. `[8/11]` is green there, and `[9/11]` executes for
the first time at any revision and passes:

```text
[8/11] Unified math-bits contract    12 passed in 2.17s
[9/11] Unified profiler contract     9 passed, 2 skipped, 1 xpassed, 1 warning in 10.55s
[10/11] Profiler parity test         1 failed, 5 passed, 1 xpassed, 1 warning in 8.69s
Integration test 'Profiler parity test' failed with exit code 1
```

`[10/11]` is the next blocker and the `relu` reroute below is what it turns on:
`test_kernel_names_are_demangled` fails as vacuous with `relu` on FlagGems.

Run 34946627774 (job 104307731540, head `076ab48`) is the first at a revision
carrying the `relu` reroute, and the first at any revision to execute all eleven
groups. `[10/11]` goes green at the reroute -- `6 passed, 1 xpassed, 1 warning in
8.03s`, the exact count the local A/B predicted for the cuda arm -- and the
groups in between stay green, including `[3/11]` FlagGems runtime path at `13
passed, 1 skipped, 1116 deselected, 1 xpassed in 391.93s`. `[11/11]`, the unit
group, executes for the first time and reports **26 of 27 files**; the exception
is `tests/unit/test_gen_vendor_confs.py`, a repository-wide check that failed
identically on a clean `main` worktree for Ascend/GCU conf drift this change does
not touch. No DCU-measured group is red at that head. Both halves of that drift
have since been cured upstream -- GCU by #285, Ascend by #288 -- so at this head
that file is green as well: `gen_vendor_confs.py --check` exits 0 with `all
vendor confs up to date` and `tests/unit/test_gen_vendor_confs.py` reports 35
passed locally. The job's only remaining red is gone, and this change carries no
conf edit of its own.

The 546-overload FlagGems cohort row for bw1000 is **not revalidated** here. Its
denominator is the generic `backends_flaggems.conf` cohort, and this change does
not touch that cohort or its routes; the numbers above are a targeted run over
the DCU dispatch path, not a re-measurement of the table. The evidence gap is
that `tests/manual/flaggems_overload_survey.py::active_routes()` only recognises
the literal `flagos_python` backend and cannot enumerate `backends_dcu.conf`'s
`= flaggems` routes, so the standard survey cannot reproduce this change's cohort
even on this hardware.

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

- Per-op probe, one fresh process per op (`FLAGOS_LOG=dispatch`,
  `FLAGOS_LOG=fallback`, `PYTHONUNBUFFERED=1` so the `--- CASE` markers and the
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
### Ascend moves from triton-ascend to FlagTree, and widens the FlagGems route (2026-09-15)

Ascend's FlagGems route used to run on `triton-ascend 3.2.2`. It now runs on the
vendor's own Triton distribution, FlagTree `0.6.2a1+ascend3.5` (Triton 3.5), and
the FlagGems coverage was re-measured on that stack rather than carried over from
the old one. Two changes follow from the re-measurement:

- **`pow` and `rsqrt` come back to FlagGems.** They were excluded under
  triton-ascend 3.2.2 because they crashed `bishengir-compile` with
  `LLVM ERROR: unsupported datatype for arith::ExtFOp to hfusion` (CI run
  34792677968, filed as FlagGems issue #6226). On FlagTree they compile: all
  five overloads — `pow.Scalar`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor`,
  `rsqrt`, `rsqrt_` — match the CPU reference for shapes `(1,)`, `(7,)`,
  `(128, 256)` and `(3, 5, 17)` in fp32/fp16/bf16, three seeds each, plus the
  backward through the FlagGems kernels.
- **Twenty-three overloads move the other way, back to aclnn.** Each was
  measured on the new stack and each has a native kernel, so the route back
  costs no coverage. They fall into five groups:
  - `mm`, `mm.out` — the Ascend tune config tunes a kernel that cannot accept one
    of its own keys (`tune_configs.yaml` declares `SPLIT_K` for `mm:` while
    `mm_kernel_general` does not take it), so the first call dies inside the
    autotuner's own benchmark run with `KeyError: 'Keyword argument SPLIT_K was
    specified but unrecognised'` before any kernel is compiled. The rest of the
    family is fine: `bmm`, `bmm.out` and `addmm` run on FlagGems and match a
    float64 CPU reference to `1.6e-7` relative, three orders tighter than the
    aclnn path's `1.7e-4`.
  - twelve comparison overloads (`eq`/`ge`/`gt`/`le`/`lt`/`ne`, `.Scalar` and
    `.Tensor`) — the kernels evaluate in float32 (`x.to(tl.float32)`), which is
    silently wrong for integer operands wider than float32's 24-bit mantissa:
    `ge` over `[2**53+1, 2**53]` reports `[True, True]` where ATen returns
    `[False, False]`, and a Python scalar operand outside int32 range collapses
    to 0 before the comparison. Both the cast and the scalar path reproduce in a
    one-line Triton kernel with no FlagGems involved.
  - `rand`, `rand_like`, `randperm`, `exponential_`, `native_dropout`,
    `native_dropout_backward` — each dies inside BiShengHIR's compilation of the
    FlagGems kernel, so the op never launches. `rand`/`rand_like` and the
    large-tensor dropout path hit the unified-buffer budget
    (`ub overflow, requires 2294016 bits while 1572864 bits available!`);
    `exponential_` is rejected on its own `arith.cmpi` attribute and is already
    in FlagGems' Ascend `CUSTOMIZED_UNUSED_OPS`; `randperm` is wrong before it is
    slow (`randperm(50)` returns all zeros for two different seeds, while
    `randperm(2000)` fails to compile through `topk.py`). The dropout failure is
    shape-dependent — the 100 000-element case needs 5112576 bits while the
    256-element case in the same test file fits — so a small-N check proves
    nothing about the route.
  - `sort`, `sort.stable` — rejected by BiShengHIR with `ub overflow, requires
    4620288 bits while 1572864 bits available!`. This is the same kernel MUSA
    routes around for a different reason; here it is the compiler's
    unified-buffer budget, not a missing cast. `argsort`/`msort` are composites
    over `sort`, so one entry covers all four.
  - `mul_.Tensor` — a FlagGems defect, not a backend one. `flag_gems/ops/mul.py`
    is the only operator module that gates its Triton path on the runtime device
    *name* and then re-dispatches with `torch.ops.aten.mul.out.redispatch(...,
    a, b, out=out)`. Here the runtime name is `npu` while the tensor's device
    type is `flagos`, so the fallback always fires and hands the boxed schema the
    caller's raw operand. `mul_.Scalar` is boxed onto `mul_.Tensor` by ATen, so
    one entry covers both spellings; `add_`/`div_`/`sub_` are unaffected because
    no other module carries that gate.

**Runtime dtype escape.** The exclusion above is expressible in a conf because it
is per-op. A per-dtype exception is not: a conf has no way to say "FlagGems,
except for float64". On this stack that exception is real and broad — BiShengHIR
rejects the float64 instantiation of nearly every pointwise kernel FlagGems
emits. Measured: `add`, `sub`, `div`, `neg`, `abs`, `exp`, `log`, `sqrt`,
`reciprocal`, `where`, `clamp`, `fill_`, `zeros_like`, `ones_like`, `ones`,
`full` and `arange` over float64 all raise `MLIRCompilationError`, while `mul`,
`cat`, `eq` and `lt` compile. The exception therefore lives at runtime:
`FlagGemsRejectsDtype` in `csrc/aten/common.cc`, consulted by
`Dispatcher::ResolveFn` (`csrc/aten/dispatcher.h`) where the arguments are still
visible. It is a dtype-only predicate, and it sees Tensors, `optional<Tensor>`,
Tensor lists, `optional<ScalarType>` and bare `ScalarType`, so the factories that
carry the dtype as an argument rather than in a tensor are covered too. When the
dtype is rejected and the platform has a vendor kernel, the call resolves to the
vendor slot and `FLAGOS_LOG=dispatch` logs that backend, not `flaggems`.

**Route delta.** Ascend `flaggems` 241 -> **225**, `ascend` 133 -> **149**,
`none` 1662 unchanged (2036 routable ops); 30 overloads moved, 7 to FlagGems and
23 back. Conf SHA-256
`04a5380ab55c127d82c0657c2a02c20593257364c582be154cbfdb060c252412` (was
`8ce7c8c733c7b0b040a5ac38e0ba1a2f6fc997f230209cbc9de24c74ba3384`). The generated
conf remains byte-identical across two runs of
`scripts/codegen/gen_vendor_confs.py`.

**Other Ascend changes in the same cohort**, all measured on the same stack:

- Ascend defaults to FlagTree instead of `triton-ascend`, and FlagGems is
  imported after `torch_fl` so the `torch.npu` shim absorbs the backend's
  discovery-time import without a real torch-npu. A thin `triton.experimental.tle`
  placeholder keeps `import flag_gems` working where AscendSHMEM is absent.
- RNG: the generator state contract FlagGems' RNG kernels expect is bridged onto
  the platform's default generators, which is what lets the generator-consuming
  tests reach the dtype and compilation failures above instead of failing earlier
  on the state shape.
- Per-device default ACL streams, so a drain on one device no longer synchronizes
  another device's stream, and the executor cache key carries the device index.
  A failed `aclrtCreateStream` is no longer cached as a null stream, which would
  have pinned that device to the runtime default stream — and left it undrained,
  since `DrainDefaultAclStreams` skips null entries — for the life of the process.

**Measured on an Ascend 910 host** (4 devices, server-class 910/910B, **not** the
910C the CI image targets) with CANN 9.0.0, FlagTree `0.6.2a1+ascend3.5`
(Triton 3.5.1) and FlagGems `5.4.0rc2.post1+g6d31db9aa` — the CI pin is the
different revision `d45285ba`, and every gap above was re-checked against both:

- A 22-op float64/float32 probe over `flagos:0`: **22/22 float32** and
  **22/22 float64** pass. Before the runtime escape, 17 of the 22 float64 cases
  raised `MLIRCompilationError`.
- `tests/integration/ops/` with `-m ascend`: **38 passed, 1099 deselected**;
  **44 passed** with the new `test_dtype_route_fallback.py` (below) included.
- `tests/integration/ops/test_rng_dispatch.py -m main_ops`: **112 passed, 3
  skipped, 1 deselected, 1 xpassed**.
- `tests/integration/test_factory_ops.py`: **46 passed**.
- `tests/integration/test_amp_contract.py -m amp`: **27 passed** (4 failing / 23
  passing before the runtime escape).
- `tests/integration/test_math_bits_contract.py -m math_bits`: **5 passed, 7
  skipped**.
- `tests/integration/test_profiler_contract.py -m profiler` with the MSPTI
  preload: **2 passed, 10 skipped**.

**New regression test.** `tests/integration/ops/test_dtype_route_fallback.py`
(6 cases) is the CI-visible contract for the runtime escape: it drives one
subprocess probe over both dtypes with `FLAGOS_LOG=dispatch`, asserts that every
float64 call the conf sends to FlagGems is answered by the native backend, that
float32 keeps whatever route the conf chose, that a route the conf made itself is
untouched, and that the float64 answers match a CPU reference rather than merely
not raising. It also pins that the per-op backend cache does not pin an op to one
backend for good — both dtypes run in one process and take different routes.

**Evidence gaps.** Two, both recorded rather than papered over:

- The CI target is a 910C image; the host used here is a 910/910B. The route
  table, the FlagGems revision and the compiler are the ones CI uses, but the
  silicon is not, so no 910C row is claimed and the CI run is the only 910C
  evidence for this change.
- `tests/manual/flaggems_overload_survey.py` cannot measure these routes. It
  selects overloads whose conf value is the FlagGems route, and the float64
  escape is a runtime decision that no conf value reflects; the Ascend rows of
  the generic FlagGems baseline above are unchanged by this work and are **not
  revalidated**. The evidence is targeted float64 probing plus the full CI
  manifest, not a synthesized overload survey.

**One pre-existing failure, unchanged by this work.**
`tests/unit/test_gen_vendor_confs.py::test_shipped_confs_are_up_to_date` still
reports `backends_gcu.conf` as stale. This is measured, not assumed: the branch
generator and the base-commit generator produce **byte-identical GCU output**
(`head==now True`), and the shipped GCU conf is stale under both, so the drift
predates this change and is orthogonal to Ascend. It is left alone rather than
regenerated, because a GCU conf regeneration is an 88-route, 590-line diff that
belongs with a GCU change. Ascend and MUSA are clean under the branch generator
(`--check` reports only `backends_gcu.conf`).

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
- `FLAGOS_LOG=dispatch` confirms the routes at runtime: `div.Tensor`,
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

**One device, two names.** FlagGems resolves its own device name from the vendor
descriptor (`flag_gems/runtime/backend/_nvidia/__init__.py` sets
`device_name = "cuda"`) and caches it in a process-wide singleton, so a build that
registers the PrivateUse1 backend as `flagos` carries two names for one device.
Most of the FlagGems tree reads the singleton at call time and is unaffected by
that. The exceptions compare `tensor.device.type` against the name: some against
`flag_gems.device` at call time (`ops/i0.py`, `ops/special_i0e.py`,
`ops/special_i1.py`, `ops/special_scaled_modified_bessel_k1.py`), others against a
module-level snapshot taken at import (`ops/mul.py:34`, `ops/smooth_l1_loss.py`).
On a `flagos` tensor that comparison is always false. In `ops/mul.py` the false
branch redispatches to the module's own aten reference path,
`torch.ops.aten.mul.Tensor.redispatch(CompositeExplicitAutograd, a, b)`, which
requires a Tensor; a Python scalar reaches it only as `float` because
`csrc/aten/backends/flagos/python_op_caller.cc` unwraps a wrapped number to
`ScalarToPython(t.item())`. With `mul.Tensor` routed to `flaggems`, every
`tensor * python_float` therefore raised `RuntimeError: aten::mul() Expected a
value of type 'Tensor' for argument 'other' but instead found type 'float'`.

`torch_fl/accelerator/cuda/_cuda_compat.py:patch_flaggems_device_name()`, called
from `torch_fl.flagos.init()` before the first route executes, rewrites the
singleton's name and every module-level copy of it (`device`, `_DEVICE_NAME`) in
the loaded `flag_gems` modules to the registered backend name. Nothing in FlagGems
is patched or forked; the rewrite is applied to the imported modules from
torch-fl. It is deliberately narrow: it acts only when FlagGems resolved the
`nvidia` vendor and the name is the vendor literal, so a build whose registration
already matches, and every other vendor, is left alone.

**The aligned cohort was re-measured and is verdict-identical.** The whole survey
was rerun on the same host with the alignment active, against the same
configuration SHA and the same FlagGems revision. All 416 routes were measured in
both runs, and every one of them returned the same verdict and the same per-case
status vector: 0 verdict differences and 0 status differences over the 2912
cases the two runs share. Fifteen cases differ only in the *text* of the error
they report while carrying `INVALID_CASE` in both runs -- an ATen internal source
line, the internal function name a `NotImplementedError` mentions, and raw
pointer addresses on a padding error. The tables below are the aligned run and are
byte-identical to the pre-fix numbers, so the alignment changes which code path
FlagGems takes and not what the harness observes. That is the honest scope of this
evidence: on this host the pre-fix mismatch was reachable through the FlagGems
entry point but not through the routed path, because the staged accelerator
library predates the wrapped-number conversion described above, so `tensor *
python_float` never reached the guard here. The routed failure was reproduced by
rewriting the names in process (below); the end-to-end crash on a build that does
carry the conversion is an evidence gap, recorded with the update-history entry.

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
work. Two different guards produce that rejection, and only one of them is about
the device *name*:

- **Comparison against the FlagGems device name (9).** `i0`, `i0.out`,
  `smooth_l1_loss`, `smooth_l1_loss.out`, `smooth_l1_loss_backward`,
  `special_i0e`, `special_i1`, `special_scaled_modified_bessel_k1`,
  `special_scaled_modified_bessel_k1.out` compare `tensor.device.type` against the
  name FlagGems resolved, which is the stale `"cuda"`. With the name aligned they
  accept a `flagos` operand and run. Measured per op in both arms of an in-process
  A/B on the same host: `i0`, `special_i0e`, `special_i1`,
  `special_scaled_modified_bessel_k1` and `smooth_l1_loss` each raise their guard
  message with the vendor literal restored, and each returns a tensor with the
  alignment in place. The `smooth_l1_loss` case is the module-level snapshot
  shape, the other four read the singleton at call time, so both rewrite paths are
  exercised.
- **`Tensor.is_cuda` (4).** `im2col`, `special_modified_bessel_k0`,
  `special_modified_bessel_k0.out` and `upsample_bicubic2d` assert on `is_cuda`
  itself, which is false on a PrivateUse1 device under any name. Measured in the
  same A/B: the first, second and fourth still raise `AssertionError: Inputs must
  be CUDA tensors`, `AssertionError: Tensors must be CUDA tensors` and
  `ValueError: This Triton kernel requires CUDA tensors` with the alignment in
  place. These four cannot be routed to FlagGems without an upstream change.

Representative messages from the cohort run: `ValueError: i0: input tensor must
be on cuda device`, `AssertionError: im2col: Inputs must be CUDA tensors`,
`AssertionError: smooth_l1_loss: input and target must be CUDA tensors.`,
`ValueError: special_i0e: Tensors must be cuda tensors`, `ValueError:
upsample_bicubic2d: This Triton kernel requires CUDA tensors`. CUDA boxing runs
the same cases correctly. All thirteen stay on CUDA boxing in the shipped
configuration: unblocking the guard is not the same as measuring the kernel, and
the nine name-guarded ops have not been re-measured for correctness on the
FlagGems route.

The 13 are the subset of guarded ops that *failed* in the cohort run. The guard
also sits on routes that measured clean and therefore appear in no rollback
group, because the harness profiles that reach them pass a Tensor where the guard
is satisfied by a different branch. Eleven `flaggems` routes are in that state:
`_embedding_bag_dense_backward`, `_upsample_nearest_exact2d_backward`, `eq.Scalar`,
`eq.Tensor`, `mul.Tensor`, `reflection_pad2d`, `reflection_pad2d.out`,
`reflection_pad3d`, `reflection_pad3d.out`, `upsample_trilinear3d` and `zero_`.
All eleven are name-guarded, and they are the routes the alignment changes the
behaviour of; `mul.Tensor` is the one that failed in CUDA CI, through the scalar
operand path rather than through any profile the harness synthesizes.

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
- Routing was confirmed at runtime with `FLAGOS_LOG=dispatch`: `add_.Scalar`,
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

### PPU FlagGems routing on the FlagTree triton stack (2026-09-15)

PPU's routing configuration went from 11 to **478** `flaggems` routes. Exactly
467 overloads moved `cuda` -> `flaggems` and nothing moved in the other
direction: `flaggems` 11 -> 478, `cuda` 2025 -> 1558, `none` 0 unchanged, over
the same 2036-op list, so the platform still reads as 100% covered.

That widened set was then surveyed, and the routes the survey could not run on
the FlagGems path were pinned back to the boxing kernel. Running the result
through CI then exposed two further route-dependent failures the survey cannot
see, which are pinned in the same place, and a source-level audit of the routed
set added four more, so the shipped configuration is **435 `flaggems` / 1601
`cuda`**: 47 of the 482 FlagGems-covered ops are held on the vendor kernel — the
33 route-dependent failures measured below, the mm/bmm family, five `addmm`
overloads, `_conj`, and the four reflection-padding routes the harness cannot
reach — and the other 435 route to FlagGems. The shipped file is idempotent
(SHA-256 `0aa2c5ba9825ec57852f63ed7c5437d40b2982c9402515540877bc9ca579bf6d`,
reproduced byte-for-byte by a second `gen_vendor_confs.py` run).

**PPU's op list no longer comes from CUDA's configuration.** The set of ops each
conf enumerates is read from `csrc/aten/generated/register.inc` — the
registration list the CUDA build and the PPU build both compile, and the same
artifact `gen_vendor_confs.py` already reads for each vendor's native kernel set
— instead of being read back out of `backends_cuda.conf`, which was the op-list
source for every platform. The two agreed only because `codegen_ops.py` rewrites
`backends_cuda.conf` with one line per generated wrapper, which made PPU's op
universe a function of another platform's routing table: a CUDA-side edit that
added or dropped an op line would have resized `backends_ppu.conf` without
touching PPU, and enumerating from a file PPU also overrides is the round trip
this generator avoids everywhere else. The substitution is provenance-only and
was verified as such — with the new source, regenerating every conf leaves all
nine of them byte-identical, so no route moved.

No other platform's configuration changed either — `git diff --stat HEAD --
torch_fl/configs/` touches `backends_ppu.conf` alone.

**Why PPU was held at 11 routes.** The limit was environmental, not a measured
FlagGems limitation on the hardware. The PPU job venv took its triton and its
`flag_gems` from the container filesystem: the *vendor* triton
(3.5.0+v0.2.0.ppu2.1.0, backend registry `['amd', 'nvidia']`) and a PEP 660
editable `flag_gems` resolving to a `/workspace/FlagGems` host bind mount. When
the runner pod stopped carrying that mount, `set_env_ppu.sh` aborted in
environment setup — it probed a hard-coded source list and called `exit 1` with
`FlagGems source is not available under /workspace/FlagGems` — so the job never
reached the operator steps at all.

**The stack is now installed, not mounted.** `set_env_ppu.sh` installs both
packages into the job venv in the integration stage:

- `flagtree===0.6.2a2+ppu3.6` from the FlagOS index. Its `ppu` variant *is* the
  `triton` package rather than a plugin beside it, so it replaces the vendor
  build: installed as triton 3.6.0 with backend registry `['ppu']`.
- FlagGems master from git, `--no-deps` (measured
  `5.4.0rc2.post1+gd45285ba6`, vendor `thead`, auto-detected from `PPU_SDK` at
  import).

`--no-deps` is load-bearing on both: a normal `flag_gems` resolve pulls PyPI's
NVIDIA triton over the FlagTree build and re-resolves `torch`, replacing the
pinned CPU wheel that the PPU core libraries are symlinked over at import.
`triton` and `flag_gems` were also removed from the vendor-site copy loop, which
would otherwise copy the image's vendor triton and its `/workspace/FlagGems`
editable path over the freshly installed ones. An assertion after
`bundle_ppu_libtorch.sh` now proves the three properties that broke during
bring-up: `triton.__file__` and `flag_gems.__file__` resolve inside the venv,
`'ppu' in triton.backends.backends`, and flag_gems' vendor is `thead`.

**One platform-side defect had to be fixed first.** Enabling the FlagGems route
broadly exposed a hole in `_StreamShim`
(`torch_fl/accelerator/cuda/_cuda_compat.py`), the stand-in that
`torch.cuda.current_stream()` / `default_stream()` return on the PPU CPU-wheel
boxing path. It exposed only `.cuda_stream` and `synchronize()`, while
`torch.cuda.Stream.wait_stream` is implemented as
`self.wait_event(stream.record_event())` and `torch.cuda.StreamContext` compares
`current_stream().device` on entry and restores through
`current_stream().stream_id` on exit. FlagTree's triton benchmarks every
autotuned FlagGems kernel through `triton.testing.do_bench_cudagraph`, which
uses exactly that surface (`torch.cuda.current_stream()`, `torch.cuda.Stream()`,
`benchmark_stream.wait_stream(caller_stream)`, `with torch.cuda.stream(...)`), so
the autotuner died with `AttributeError: '_StreamShim' object has no attribute
'record_event'` — and, once that was granted by hand, on `.device` next. The
shim now delegates the event/ordering half to the real flagos stream (resolved
via `torch.flagos.current_stream`, i.e. the same physical stream the boxing
kernels submit to) and keeps `.cuda_stream == 0`, the null stream, for the
launch-side half. This is the `_MetaxStreamShim` design in
`torch_fl/accelerator/metax/_metax_compat.py`, ported; the fix retires the whole
family of failures rather than one op, since any FlagGems op whose autotuner
routes through `do_bench_cudagraph` would fail identically. The reported symptom
before the fix was three `exponential_` RNG cases
(`TestRngReproducible::test_same_seed_same_draw[exponential_]`,
`...::test_different_seed_differs[exponential_]`,
`TestRngDistribution::test_exponential_rate_1`); afterwards the RNG file is
**113 passed, 2 skipped, 1 deselected, 1 xpassed in 5.03s**.

**The route exceptions.** 47 FlagGems-covered ops stay on the `cuda` boxing
route, recorded in `BOXING_TRITON_GAPS["ppu"]` in
`scripts/codegen/gen_vendor_confs.py`, which carries each op's measured failure
next to it. The first four (`mm`, `mm.out`, `bmm`, `bmm.out`) predate this work
and are not a FlagTree finding: FlagGems' `_hygon` mm/bmm kernel passes a
`num_ldmatrixes` keyword the triton `mm_kernel` does not accept, which is
`KeyError` at `triton/runtime/jit.py:_pack_args` for `mm` and a 60 s compile
timeout for `bmm`. Filed upstream as FlagGems issue #6225. The two `--deselect`
entries in the PPU manifest cover only the dispatch-log tests that assert the
FlagGems route for those two ops. The other 33 are the survey's route-dependent
failures, listed in the survey paragraph below.

**The four exceptions the survey cannot reach, found by audit rather than by
running them.** `reflection_pad2d`, `reflection_pad2d.out`, `reflection_pad3d`
and `reflection_pad3d.out` are routed to FlagGems by the widened table and
cannot execute there, but every one of their seven profiles is `INVALID_CASE`,
so the survey records nothing about them and no rollback group contains them.
The harness derives `padding` from the tensor's rank — one element for the
`2d-f32` and `1d-f32` profiles, two for `4d-f32` — and ATen's arity check
rejects that before the operator body runs (`RuntimeError: padding size is
expected to be 4, but got: 1`; `... 6, but got: 1` for the 3d overloads). That
is the *argument* check, which runs before the device check underneath it:

```python
# flag_gems/ops/reflection_pad2d.py:106,136 — reflection_pad3d.py:125,158
if input.device.type != flag_gems.device:
    raise ValueError(f"input must be a {flag_gems.device} tensor")
```

`flag_gems.device` is a module-level string, set once from
`runtime.device.name` at import, and on PPU it resolves to `"cuda"`: the
`_thead` vendor descriptor declares `device_name="cuda"` for a device whose
tensors report `"flagos"`. The survey's own evidence names the value —
`i0` failed its profiles with `ValueError: i0: input tensor must be on cuda
device`, a message that interpolates `flag_gems.device` — and `i0` is one of
the ten device-guard refusals already pinned above for the same reason. The four
reflection-padding routes are the remainder of that class:
`torch_fl.accelerator.cuda._cuda_compat.patch_flaggems_device_name`, which
realigns the name for NVIDIA and is what unblocks the same four routes on CUDA
(upstream #291), returns immediately unless FlagGems resolved the `nvidia`
vendor, so on PPU the guard fires on every call. They are pinned to `cuda`
rather than left on FlagGems because the boxing kernel needs no such alignment
and is the route the platform already used, and because the alternative —
widening the alignment to `thead` — would newly enable every guarded FlagGems
kernel on PPU on the strength of measurements taken on another platform, which
this report does not have. `reflection_pad1d`, `reflection_pad1d.out` and
`reflection_pad3d_backward` carry no device-name guard — their `out` checks
compare tensor to tensor — and stay on FlagGems.

The audit that found them is worth stating, because it is what makes the four
numbers rather than a sample: the generated
`csrc/aten/generated/flaggems_python_kernels.cc` names the Python entry point
each route calls, so the 435 routes were each resolved to their FlagGems
function through `flag_gems._FULL_CONFIG` and their source scanned for a raise
or assert that tests `is_cuda`, `_DEVICE_NAME` or `flag_gems.device`. Four
routes matched; the rest of the guarded modules fall back to ATen instead of
raising, which is why an operand on the flagos device makes them slower rather
than broken. The scan was run against the FlagGems installed on the development
host (`5.3.1.post1.dev212+g7fb49bad4`); the CI job installs master, so the four
are a lower bound on that revision, not a statement about it.

**The addmm and `_conj` exceptions the survey cannot see.** `addmm`,
`addmm.dtype`, `addmm.dtype_out`, `addmm.out`, `addmm_` and `_conj` were pinned
after the first CI run on the 445-route configuration, because both failures are
invisible to a value-level overload survey and only the integration suites reach
them.

The `addmm` family fails on small-K shapes. FlagGems' `addmm` autotune selects a
`BLOCK_SIZE_K < 16` configuration, and the FlagTree ppu backend's
`min_dot_size[2]` is 16, so `triton/language/semantic.py` rejects the tile inside
`tl.dot` with `CompilationError: Input shapes should have M >= 1, N >= 1 and
K >= 16`. The error surfaces from `flag_gems/ops/addmm.py:170`, i.e. inside the
autotuner's `do_bench_cudagraph` replay, so the op exits through the autotuner
rather than through a fallback. It is config selection rather than `K < 16`
outright: measured on the PPU stack below, M/N/K = 4/8/8, 128/128/8, 2/2/1 and
4/4/4 fail while 4/8/12, 8/8/15, 4/8/16, 8/8/17, 4/8/31, 16/16/16 and 32/32/12
pass, and `baddbmm` — the same dot structure — passes at both K = 8 and K = 16.
The shape that reaches it from CI is `nn.Linear(8, 8)` over a `(4, 8)` input in
`tests/integration/test_factory_ops.py::TestCopyTransfer::
test_module_cpu_after_forward`. All five overloads are pinned together rather
than the one that failed, since they share the autotuner.

`_conj` fails a contract test with correct values. `torch.conj()` is a metadata
operator in PyTorch: it sets the Conjugate bit and leaves storage untouched.
FlagGems' `_conj` (`flag_gems/ops/_conj.py`) computes the conjugate into a fresh
tensor instead, so on the FlagGems route `torch.conj(x).is_conj` reads False —
the numbers are right and the laziness contract is broken.
`tests/integration/test_math_bits_contract.py` asserts that contract for every
backend and takes it as a precondition for its own cases, so the file went from
12 passed to 5 passed / 7 errors. The survey cannot see it by construction:
eager materialization is numerically indistinguishable from the lazy view. The
same signature is recorded for MUSA in `NATIVE_TRITON_GAPS["musa"]`, where
`_conj` routes to `none` for exactly this reason.

**Measured on PPU 810e hardware** (16 `PPU-ZW810E` devices, torch 2.10.0 CPU
wheel with the PPU core swapped in at import) with the installed stack above:

- Operator step 1 (vendor backend, `main_ops and not flaggems_python and not
  flaggems_cpp`): **126 passed, 15 skipped, 1002 deselected, 1 xpassed in
  125.33s**.
- Operator step 2 (FlagGems runtime path, `FLAGOS_USE_FLAGGEMS=1`,
  `flaggems and main_ops`): **11 passed, 1132 deselected, 1 xpassed in 99.09s,
  exit 0**. No op in this cohort needed a vendor fallback.
- The two suites the six later pins came from, re-run against the shipped
  configuration: `tests/integration/test_factory_ops.py` **46 passed in 1.95s**
  (1 failed / 45 passed on the 445-route configuration) and
  `tests/integration/test_math_bits_contract.py -m math_bits` **12 passed in
  1.00s** (5 passed / 7 errors before). Both were confirmed causally before the
  pins were written: re-running with `FLAGOS_OP_<op>=cuda` on the unpinned
  configuration makes them pass, which is what identifies the route rather than
  the shape or the revision as the cause.
- The two model-level steps of the PPU manifest, which no operator cohort
  covers, run locally against the same hardware and the local Qwen3-0.6B
  snapshot on the shipped commit:
  `tests/integration/test_qwen3_infer.py` **4 passed in 32.78s** and
  `tests/integration/test_qwen3_train.py` **3 passed in 12.82s** (an earlier
  run on the pre-rebase tree also passed, in 87.08s and 55.80s). Both are
  needed because the widened route is what these steps exercise — the model's
  Linear, RMSNorm, SiLU, rotary, attention and cross-entropy paths all now run
  through FlagGems kernels rather than the boxing ones.
- The failures above were found by the CI PPU job and reproduced locally against
  the same hardware before being pinned.
- **In CI**, the PPU job on the commit that carries this change (`79d88aaf`,
  PPU 810e runner, FlagTree `0.6.2a2+ppu3.6` and FlagGems master installed into
  the job venv by `set_env_ppu.sh`) reproduced the operator steps and got past
  `[6/8]` of the manifest for the first time: operator step 1 **126 passed, 16
  skipped, 995 deselected, 1 xpassed in 122.50s**, operator step 2 **11 passed,
  1 skipped, 1125 deselected, 1 xpassed in 87.33s**, general tests **46 passed
  in 16.17s** (the `addmm` pin) and the math-bits contract **12 passed in
  1.61s** (the `_conj` pin). The CI deselection counts are seven lower than the
  local ones because the job venv's collection differs; the pass counts are the
  same.
- Both CI operator steps report `1 xpassed` where the pre-change CI reported
  `1 xfailed`. That is
  `tests/integration/ops/test_rng_dispatch.py::TestRngDropout::test_dropout_reproducible`,
  whose non-strict xfail is conditioned on `FLAGOS_USE_FLAGGEMS` and whose own
  reason text says that closing the vendor-path gap should surface as an xpass.
  `native_dropout` now routes to `flaggems` in `backends_ppu.conf`, so dropout
  is reproducible on the configured route with no runtime flag, and the xfail no
  longer describes the configured behaviour. It is not a PPU-specific anomaly:
  CUDA has reported the same xpass since #276 gave it the same route.
- **The two model-level steps are not revalidated in CI.** The job reached the
  first of them; it failed in setup, before any test body ran, on the model
  directory rather than on the routing. `[7/8] Run inference tests` reported
  `4 errors in 1.01s`, all of them
  `ValueError: Unrecognized model in /models/Qwen3-0.6B. Should have a
  model_type key in its config.json, or contain one of the following strings in
  its name: ...`, and the runner stops the manifest at the first failing step, so
  `[8/8] Run training tests` never ran at all. That is the error transformers
  raises for a directory whose `config.json` parses but declares no
  `model_type`; an empty directory raises the missing-weights `OSError` instead,
  so the runner's bind-mount source is not a Qwen3 snapshot — while being
  non-empty enough for `[1/8] Check model availability` (a bare
  `test -d "$MODEL_PATH"`) to pass. The same mount, on the same runner, with the
  same test file passed earlier the same day — `4 passed in 30.64s` on `main` at
  `bc39a832` — and nothing between that commit and this one changes the model
  path, its mount line, or the test, so this is a runner-host data problem that
  this change neither causes nor repairs. The local numbers above remain the only
  evidence for these two steps.
- Generator idempotency: a second `gen_vendor_confs.py` run leaves
  `backends_ppu.conf` byte-identical; `--check` does not list PPU.
- Lint: `ruff check .` — "All checks passed!"; `ruff format --check .` — "250
  files already formatted".

**Evidence gap.** The two summary tables above describe the 546-overload
*generic* cohort measured at torch-fl `fe2272b5` with FlagGems `7fb49bad` and
`backends_flaggems.conf`, which no longer exists (`d0e2d1a` removed it). This
change moves PPU to the platform's own 478-route set at a different FlagGems
revision, so the **PPU 810e rows are not revalidated** against the current
configuration and are retained only as the historical baseline; the same is true
of the 26 forced-CUDA-fallback count, which is a property of the removed generic
config. The baseline cohort cannot be rebuilt either: neither the removed
configuration nor the FlagGems revision it pinned is reproducible from HEAD.

**Survey result.** `tests/manual/flaggems_overload_survey.py` (harness v4) was run
over the new 478-route set on `flagos:0`: **312 STRICT / 46 BASIC_ONLY / 42
FAILED / 78 UNTESTED** over 400 routes with at least one CPU-valid case, of 478
registered (358 basic-executable). A route's verdict is the worst of its cases:
STRICT means every CPU-valid case matched the reference, BASIC_ONLY means at least
one did, FAILED means none did, and UNTESTED means no case could be synthesized.
The 42 FAILED routes were then re-run on the same overloads with
`FLAGOS_OP_<op>=cuda`, which returned **29 STRICT / 4 BASIC_ONLY / 9 FAILED** over
the same 42. That separates 33 route-dependent failures — passing on the boxing
kernel and failing on the FlagGems route, so caused by opening the route — from 9
that fail on both routes. The 33 are the `BOXING_TRITON_GAPS["ppu"]` entries added
above. The 9 stay on FlagGems, because pinning an op that fails on both routes
would record a routing fix that does not exist: `_batch_norm_no_update`,
`_log_softmax_backward_data`, `_softmax_backward_data`, `linalg_ldl_factor_ex`,
`mse_loss_backward`, `native_batch_norm`, `scatter.src`, `scatter_.src`,
`unique_dim`. (`_batch_norm_no_update` is the segfault of this cohort:
`returncode -11` on all seven profiles on both routes.) The 33 split by what the
FlagGems path does with them: ten refuse the tensor's device before reaching a
kernel — FlagGems tests `is_cuda`, and a tensor on the flagos device is
PrivateUse1 — three fail to compile on the FlagTree ppu backend (three
`CompilationError`s: `randint`, `randint_like`, `norm.ScalarOpt_dim`), three are
`out=` aliases that do not write through or do not accept the schema's arguments
(`cosh.out`, `sum.out`, `mul_.Tensor`), ten return numerically wrong results on
every profile that ran, and seven raise.

`tests/manual/flaggems_overload_survey.py` selects routes by the
literal conf value `flagos_python`, and the unified per-platform confs spell that
route `flaggems`, so the survey was pointed at a copy of `backends_ppu.conf` with
the 478 `flaggems` values rewritten to `flagos_python` (SHA-256
`5825602605bd65223419b330f6529e15c4be3c59f738b541513c74de517e1ec3`). The two
spellings map to the same enum slot (`ParseBackendName` in
`csrc/aten/common.cc`, `Backend::kFlagGems`) and the dispatch log prints
`flagos_python` for both, so the substitution changes no routing; it is recorded
here because the harness hash and the conf hash in the provenance table do not
describe this run. The survey measured the 478-route configuration, i.e. the
table *before* the 33-op pin, which is its own output; the shipped file is the
435-route table hashed above, and the two hashes therefore differ by design. The
ten ops pinned after that run are by definition outside its scope — the survey
never measured them, and the paragraphs above record what did.

**Survey measurements must pin `torch_fl` explicitly on hosts with a stale
editable install.** The harness runs each overload in a child process with
`cwd="/tmp"`, so `sys.path[0]` is not the repository root and `import torch_fl`
falls through to whatever the interpreter has installed. On the 810e host the
shared conda environment carries `__editable__.torch_fl-0.1.0+ppu.pth` pointing
at a `.claude/worktrees/fix-issue-92` checkout three weeks stale, and the venv
inherits that environment's `site-packages`; a survey run from `/tmp` therefore
measures that build, silently and with no provenance in the output. Every run
recorded here was invoked with `PYTHONPATH` pinned to the repository root so that
the child resolves the tree under test. The first pass over the new routes was
discarded for exactly this reason.

**Pre-existing conditions.** `gen_vendor_confs.py --check` reported
`backends_ascend.conf` and `backends_gcu.conf` stale while this work was in
progress; that reproduced on a pristine `HEAD` worktree with the unmodified
generator, so it was checked-in drift rather than a consequence of this change.
The Ascend and GCU pipelines have since regenerated those files upstream (#285,
#288), so on the branch as rebased the check is clean,
`tests/unit/test_gen_vendor_confs.py` is **35 passed**, and a regeneration run
leaves every conf byte-identical. None of that work moves PPU: the conf
regenerates to the same SHA-256 on either base, which is the property the
op-list decoupling above was for.

### PPU conf regenerated against the widened FlagGems cohort (2026-09-16, not hardware-revalidated)

`backends_ppu.conf` was stale: `gen_vendor_confs.py --check` (and
`tests/unit/test_gen_vendor_confs.py::test_shipped_confs_are_up_to_date`, which
runs as the last group of the DCU pipeline) reported it after the
shared-coverage widening. Regenerating with the unmodified generator moves
**158** overloads `cuda` -> `flaggems` and nothing in the other direction:
`flaggems` 435 -> 593, `cuda` 1601 -> 1443, `none` 0 unchanged, over the same
2036-op list. New SHA-256
`d4906256972fbd7204ea703873b40c8e730886e9fea88c8dde2bcba90b1195c0`,
reproduced byte-for-byte by a second generator run.

No survey pin was dropped to get there: `BOXING_TRITON_GAPS["ppu"]` is identical
to the 47-entry set that shipped the 435-route conf, all 47 still route `cuda`
in the regenerated file (checked entry by entry, including the mm/bmm family,
the five `addmm` overloads, `_conj`, and the four reflection-padding routes),
and the 158 flips are exactly the newly FlagGems-covered overloads. The
widening comes from the shared cohort side, not from a PPU re-measurement.

**Not revalidated on hardware.** No PPU runner was available for this change,
so the 158 newly FlagGems-routed overloads have no device evidence here; the
PPU CI FlagGems group (`-m "flaggems and main_ops"`) is the measurement vehicle
that will confirm or roll back individual routes on the next run. Until then
the PPU survey rows above remain the 435-route evidence, and this subsection is
the recorded evidence gap for the 593-route file.

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
stays recoverable and `FLAGOS_FORCE_BACKEND=vendor` can find it (248 such kernels on Ascend, 88 on GCU, 115 on
MUSA — MUSA's remaining 7 FlagGems routes come from
`musa_flaggems_register.inc`, which registers wrappers without native kernels
behind them). On GCU that 88 became 144 with the 2026-09-15 rerouting.

`none` means no accelerated implementation on that platform. It is honest only
where registration *skips* the op, so the call reaches `cpu_fallback` instead of
a registered-but-empty dispatcher slot. That is what limits `none` to these
three platforms: **MetaX and Tsingmicro register the full generated op list**
via the `#else` branch of `csrc/aten/register.cc`, so a `none` entry there would
reach the dispatcher and raise instead of falling back. MetaX's configuration is
nevertheless generated in the same full-coverage shape as these three, with
`cuda` in the vendor slot instead of `none` (a CUDA-compatible platform can box
every op); Tsingmicro's remains hand-written. Relative to the sparse files this
is not a regression for MUSA/GCU/Ascend — an absent op reached the same fallback,
it just could not be counted.

The two boxing configurations (`metax`, `dcu`) are generated in the same
full-coverage shape, but their fallback key is `cuda` and they contain no `none`:
a CUDA-compatible platform can box every op. Their per-op key distribution was
byte-for-byte equivalent to the previous revision at this date — only the shape
and key spellings changed. **The MetaX distribution was superseded on
2026-09-15** by the widened FlagGems cohort recorded below; `dcu` additionally
lost `mul_.Tensor` to `cuda` when that op left the coverage set.

The `flaggems_cpp` key is emitted **only** in `backends_metax.conf`. That slot
is `Backend::kFlagGemsCpp`, registered behind `#ifdef FLAGOS_FLAGGEMS_CPP`, which
`csrc/CMakeLists.txt` defines only for a `FLAGOS_BUILD_FLAGGEMS_CPP=ON` build.
That switch defaults ON for cuda and tsingmicro and OFF everywhere else, and
`CMakeLists.txt` pins it OFF for dcu, musa and bpu because the path needs
FlagGems' `liboperators.so` built for that vendor's own toolkit. When a build
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

### MetaX: generated FlagGems calls name the package-level entry point (2026-09-15, MetaX C550)

`_normalize_flaggems_qualname` in [`scripts/codegen_ops.py`](../../scripts/codegen_ops.py)
used to rewrite every discovered FlagGems alias to the canonical
`flag_gems.ops.<module>.<fn>` path, on the reasoning that the alias itself
(`_metax.ops.mm`, `gcu300.ops.count_nonzero`, `_hygon.ops.mul`) must not be frozen
into a portable artifact. The rewrite was portable but it pinned the wrong
implementation. FlagGems re-exports every operator at package level, and that
top-level name is the one the active backend has already rebound at import
(`runtime.backend.SpecOpRegistrar` writes into the package globals); the
`flag_gems.ops.<module>` path holds the *generic* implementation. So a generated
kernel that named the module path ran the generic kernel on a platform that ships
an override — 72 of the 666 qualnames in the checked-in kernels resolve to a
`_metax.ops.*` callable.

The generator now emits `flag_gems.<fn.__name__>`, which
`PythonOpCache::GetFunc` resolves through its dotted branch (import the prefix,
take the attribute) against the backend that loads the extension. That keeps the
artifact portable *and* lets the vendor override win. Measured over the cohort the
checked-in kernels were generated from: 666 distinct qualnames, every one a
two-component `flag_gems.<op>` name and every one resolvable on the package; 594
resolve to the callable the module path gave, 72 to the vendor override; no two
distinct qualnames collapse onto the same name.

The failure this removes is visible on hardware. `_log_softmax_backward_data`
raised `TypeError: dynamic_func() missing 1 required positional argument:
'BLOCK_N'` on C550 through the generic module — `ops/log_softmax.py` takes
`BLOCK_N` from `runtime.get_heuristic_config("softmax_non_inner")`, while
`runtime/backend/_metax/ops/log_softmax.py` supplies it from a hard-coded
`triton.heuristics` callback, so the generic kernel is entered with a tuning
config it cannot satisfy.

The same change makes `scripts/codegen_ops.py` the writer of the coverage
ceiling: `render_flaggems_coverage` rewrites `FLAGGEMS_PYTHON_OPS` in
`scripts/backend_coverage.py` from the discovery, minus the override-only ops
(`flaggems_forced_cuda`), which must keep their kernels but must not be a default
route. Previously that literal was hand-carried, so a re-discovery against a newer
FlagGems grew the generated kernels while the ceiling stayed put — and since
`gen_vendor_confs.py` intersects every platform conf with that set, the new ops
could never reach a conf.

### MUSA: the FlagGems RNG bridge reaches the vendor op modules (2026-09-16, MUSA MTT S5000)

The package-level qualname change above applies to every FlagGems platform, and
on MUSA it moved the generated kernels onto a different *module* than the one
torch_fl's RNG bridge was patching.

`flag_gems` namespaces each vendor backend by putting the backend directory on
`sys.path` and importing `_<vendor>.ops` from it, so the MUSA op modules are
`_mthreads.ops.<op>` — not `flag_gems.ops.<op>`. `SpecOpRegistrar` then
republishes their entries as package-level attributes, which is why
`flag_gems.randn` is `_mthreads.ops.randn.randn`. Before the qualname change the
generated kernel named `flag_gems.ops.randn.randn`, the generic module;
afterwards it names `flag_gems.randn`, the vendor override.

`_patch_flaggems_philox()` in [`torch_fl/__init__.py`](../../torch_fl/__init__.py)
rebinds `philox_backend_seed_offset` on every module that imported it with
`from ... import`, and selected those modules with
`mod.__name__.startswith("flag_gems")`. `_mthreads.ops.randn` does not match, so
it kept the unpatched function. That function reads the generator through
`state_copy.view(torch.int64)` and unpacks it into exactly two values — the CUDA
seed/offset layout. torch_fl's flagos generators are CPU Mersenne-Twister
generators whose state views to 632 int64s, so every dispatched MUSA `randn`
raised `ValueError: too many values to unpack (expected 2)`.

**Measured failure.** Pipeline run [35048300960](https://github.com/flagos-ai/PyTorch-Plugin-FL/actions/runs/35048300960)
(the push of `5e4b78e` to `main`), job `104643022596` — `Platform pipeline (musa)
/ Build and test (MUSA)`: **14 failed, 99 passed, 28 warnings in 193.65s**. The
two preceding runs on `main` (35047336702, 35041546614) both had the MUSA job
green, so the regression is attributable to `5e4b78e`. Both signatures resolve to
the same call: four tests failed in-process on a literal
`torch.randn(2, 1, device="flagos:0")`, and ten `test_musa_dispatch.py`
subprocesses — `test_dispatch_log_musa[add.Tensor|mm|mul.Tensor|relu|softmax]`,
`test_dispatch_log_musa_override`, `test_dispatch_log_mm_out_musa`,
`test_flaggems_only_ops_route_and_stay_correct[asin|cosh|sinh]` — each begin with
`torch.randn(8, 8, device='flagos:0')` as their first statement, so the child
died before reaching the op under test. The traceback:

```text
torch_fl/__init__.py:1262: in __torch_function__
    return func(*args, **kwargs)
flag_gems/runtime/backend/_mthreads/ops/randn.py:96: in randn
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
flag_gems/utils/random_utils.py:75: in philox_backend_seed_offset
    c0, c1 = state_copy.view(torch.int64)
E   ValueError: too many values to unpack (expected 2)
```

**Fix.** The rebinding loop now selects modules by the identity of the bound
object (`getattr(mod, "philox_backend_seed_offset", None) is _orig`) instead of
by module name. Every module that imported the FlagGems function by value holds
that exact object and is rebound, whatever it is called; a module that does not
is left alone. `flag_gems/ops/*` and `flag_gems/runtime/backend/_<vendor>/ops/*`
are covered by the same rule, and so is any backend added later. Only
`flag_gems/utils/random_utils.py` defines the function — every other reference in
the tree is a `from flag_gems.utils.random_utils import philox_backend_seed_offset`
— so the identity set is exactly the set that needs rebinding.

**No route changed.** `randn`, `randn_like`, `rand`, `rand_like`, `randperm` and
`native_dropout` stay `flaggems  # musa` in
[`torch_fl/configs/backends_musa.conf`](../../torch_fl/configs/backends_musa.conf).
The affected MUSA FlagGems row (`randn`, `randn_like` -> `flaggems # musa`, finite
and seed-reproducible over 65536 samples, 4/4) is unchanged and is what the fix
restores; no operator was added, enabled, removed, disabled or rerouted.

**Evidence gap.** The fix is validated by unit coverage of the rebinding
(`tests/unit/test_musa_rng_bridge.py::test_flaggems_philox_reaches_vendor_backend_modules`),
which fails against the pre-fix selector, and by the MUSA pipeline on this
change. The broader MUSA FlagGems cohort table below was **not revalidated**:
MTT S5000 hardware is not available to this change, and no
`tests/manual/flaggems_overload_survey.py` re-survey was run.

### MetaX: FlagGems cohort widened to FlagGems master, sixteen ops withdrawn (2026-09-15, MetaX C550)

The shared Python coverage set `FLAGGEMS_PYTHON_OPS` in
[`scripts/backend_coverage.py`](../../scripts/backend_coverage.py) was rebuilt on
the FlagGems master cohort pinned at
`5a58df410c551c4f4eb41d31887cd75fd596804a` (2026-09-15), which raised it from
482 to 639 overloads — 158 added and exactly one removed, `mul_.Tensor`, which
that cohort does not cover. The ceiling is measured on CUDA and is shared by
every FlagGems platform, so the widening is not a MetaX change; what follows is
the MetaX measurement of it.

**Provenance.** MetaX C550 (eight devices), MACA 3.8.0 in CUDA-boxing mode,
`flag_gems 5.4.0rc2.post1+g5a58df410`, `flagtree 0.6.1+metax3.6`, Triton 3.6.0
with the `metax` backend, `tests/manual/flaggems_overload_survey.py` harness
version 6, SHA-256
`7b01c22ce3a94315f1364df242323e9faac27f2585debfb05030670c7c756cc7` as shipped.
(The revision that produced the rows below was identical apart from the
illustrative route count its module docstring quotes, which is not read by any
code path; the hash above is the one an auditor can reproduce from this tree.
An earlier revision of this entry recorded version 5 and a different hash for
the same file; no version 5 exists in the repository history, and the blob at
the revision these rows were measured on is the version 6 hash above.)

**Screening survey.** 166 overloads changed route in `backends_metax.conf` when
the set was widened. All 166 were run through the survey's `2d-f32` profile
against the pre-change configuration (SHA-256
`80202b8add16979e00319383333aa5f572125f3d517b19fdbabd44ad964ba3bc`), giving
`{"registered": 166, "tested": 97, "STRICT": 76, "FAILED": 21, "UNTESTED": 69}`,
`basic_executable` 76. The 69 `UNTESTED` overloads are those whose synthesized
invocation the CPU reference rejects, so they are neither passes nor failures and
carry no verdict here.

**Differential probe.** The 21 `FAILED` overloads were then re-run through the
same profile with their route forced to the cuda boxing kernel
(`FLAGOS_OP_<op>=cuda`), one input pair built on the host and moved with
`.to("flagos")` so both arms see identical values. Sixteen of the twenty-one
**pass on cuda and fail on flaggems** — that asymmetry, not a preference, is what
makes the withdrawal a correction. Per-op, `flaggems` -> `cuda`:

| Cause | Overload | `flaggems` verdict | `cuda` verdict |
|---|---|---|---|
| Kernel asserts its input is a real CUDA tensor | `special_bessel_j0` | `ERROR` "Tensors must be CUDA tensors" | `PASS` |
| | `special_i1e` | `ERROR` "Tensors must be cuda tensors" | `PASS` |
| | `special_i1e.out` | `ERROR` "Tensors must be cuda tensors" | `PASS` |
| | `special_chebyshev_polynomial_w.out` | `ERROR` "input x must be on cuda device" | `PASS` |
| The gems wrapper raises on its own argument handling | `nansum.out` | `ERROR` `'NoneType' object has no attribute 'copy_'` | `PASS` |
| | `lu_unpack.out` | `ERROR` size 32 vs 0 at dim 1 | `PASS` |
| | `linalg_matrix_exp.out` | `ERROR` "out must be provided for out variant" | `PASS` |
| | `_cdist_forward` | `ERROR` "None is not a valid value for compute_mode" | `PASS` |
| Wrong result, no exception | `sum.out` | `WRONG` shape `(32, 32)` against `()` | `PASS` |
| | `_compute_linear_combination` | `WRONG` max_diff 22.12 | `PASS` |
| | `_compute_linear_combination.out` | `WRONG` max_diff 1.91e+37 | `PASS` |
| | `_fused_rms_norm` | `WRONG` shape `(32,)` against `(32, 1)` | `PASS` |
| | `igamma` | `WRONG` max_diff `nan` | `PASS` |
| | `igamma_` | `WRONG` max_diff `nan` | `PASS` |
| | `logit_backward` | `WRONG` max_diff `nan` | `PASS` |
| | `special_shifted_chebyshev_polynomial_t` | `WRONG` max_diff 361.53 | `PASS` |

`sum.out` is the mildest of the third group and is shape-only: the gems wrapper
returns the `(32, 32)` `out` buffer where ATen returns the 0-dim result view, so
the sum itself lands and only the returned shape is wrong. That is precisely the
class a routing-only check cannot see, which is why the hardware guard for this
group runs the call and requires the boxing route to complete it.

All sixteen now route to the cuda boxing kernel in `backends_metax.conf`, listed
individually in the `metax_triton_fallback` literal in
[`scripts/codegen_ops.py`](../../scripts/codegen_ops.py) with the same cause
grouping. Re-running them through the shipped configuration — no route override —
reproduces the `cuda` column above: 16 `PASS`, 0 failures.

**The five not withdrawn.** They fail on **both** routes, so holding them would
not fix anything and the route they already had is kept:

- `_native_batch_norm_legit.no_stats` `CRASH` on both.
- `linalg_lstsq` `WRONG` on both (shape `(0,)` against `()`).
- `log_sigmoid_backward` and `log_sigmoid_backward.grad_input` `WRONG` on both
  (max_diff 560.86 / 279.29 on flaggems).
- `linalg_eig` is `WRONG` on flaggems and `ERROR` on cuda, but flaggems is the
  better route and it is kept there: the boxing route raises
  `RuntimeError: Calling torch.linalg.eig with MAGMA requires compiling PyTorch
  with MAGMA`, while a targeted probe showed the gems eigenvalues match the host
  exactly (`sorted real allclose: True`, unsorted also `True`); only the
  eigenvectors differ, which is the phase ambiguity inherent to `eig`.

**Resulting configuration.** Against the committed baseline of 443 `flaggems` /
11 `flaggems_cpp` / 1582 `cuda`, the widening takes `backends_metax.conf` to a
608 / 12 / 1416 intermediate and the sixteen withdrawals bring it to
**592 `flaggems` / 12 `flaggems_cpp` / 1432
`cuda`** (SHA-256 `6962f023dffbe8731d55ae582d54aa966cfc4835d1f835b911081b900a91f075`,
2036 ops total). Of the 639 overloads in the raised ceiling, MetaX routes 585
through the Python FlagGems path and 12 through the C++ slot, and holds 42 on the
cuda boxing kernel — the 16 withdrawals above plus 26 overloads that were already
cuda-only Triton gaps on this platform (`mm`, `mm.out`, `sort`, `sort.stable`,
`relu`, `add.Tensor`, `bmm`, `reflection_pad2d`, and similar). The file's
`flaggems` count reads 592 rather than 585 because 7 of its Python-path routes —
the `METAX_FLAGGEMS_MEASURED` overloads — are not in the shared ceiling at all;
they are MetaX-only promotions measured against the earlier cohort.

**Evidence status.** MetaX is measured as above. Ascend, GCU, MUSA, DCU and PPU
are **not revalidated** against the raised ceiling and no flagos route on those
platforms was altered by this change: the newly covered overloads are withheld
from the Ascend, GCU and MUSA configurations by
`FLAGGEMS_PENDING_NATIVE_VENDORS` / `FLAGGEMS_PENDING_NATIVE_OPS` in
`scripts/gen_vendor_confs.py`, so their shipped `flaggems` counts are unchanged,
and regenerating the DCU configuration moves three lines (`mul_.Tensor`, which
left the coverage set) and nothing else. Those platforms' rows in the summary
tables above still describe the 546-overload baseline cohort.

**Mechanical confirmation.** `scripts/gen_vendor_confs.py` run twice produces an
empty diff and `--check` exits 0 for the MetaX configuration. From a clean
checkout of the same revision, `codegen_ops.py` plus `gen_vendor_confs.py`
reproduce `backends_metax.conf` byte-for-byte, along with every generated
artifact (`csrc/aten/generated/flaggems_python_kernels.cc`, `register.inc`,
`ops.h`, `ops.cc`, `cuda_kernels.cc`) and `backend_coverage.py`. The
out-of-scope configurations were restored to their committed state, so the only
MetaX change is the one described here.

The scope of that reproducibility statement is the MetaX configuration, and it
was measured per file rather than assumed. Running the two generators over this
tree leaves `backends_metax.conf` at
`6962f023dffbe8731d55ae582d54aa966cfc4835d1f835b911081b900a91f075` — the shipped
bytes — and leaves the generated C++ artifacts and `backend_coverage.py`
unchanged, but it moves four out-of-scope configurations on that first pass:
334 lines in `backends_dcu.conf`, 176 in `backends_gcu.conf`, 26 in
`backends_cuda.conf` and 10 in `backends_ascend.conf`. A second pass over the
result changes nothing in any of the five, so the pipeline is idempotent; it has
simply more than one fixed point, and the committed DCU/Ascend/GCU files are not
the ones the previous revision's generator would have produced from this tree.
That is why the out-of-scope configurations were restored rather than
regenerated, and it is a second reason not to treat `gen_vendor_confs.py
--check` as a scope guard: it seeds each boxing configuration from the shipped
file, so it agrees with whichever fixed point the file is already at.

One ordering constraint is worth recording, because it silently changes the
result. `codegen_ops.py` writes the platform configurations in the legacy
`flagos_python` / `flagos` key spelling; `gen_vendor_confs.py` is what
normalizes them to `flaggems` / `flaggems_cpp`. So both must run, in that order.
More importantly, `backends_dcu.conf` is a boxing configuration, and
`boxing_triton_gaps` reads the triton-gap set back out of the file it is about
to rewrite. Regenerating DCU from a copy that already carries the widened
ceiling therefore *keeps* the widening rather than returning to the measured
set — the run that produced this change yields 637 `flaggems` routes from a
widened seed and 472 from the committed seed. The 472 state above is reproduced
by restoring `backends_dcu.conf` to its committed state and re-running
`gen_vendor_confs.py`, which is the order the configuration was actually
arrived at. Neither seed is wrong on its own; only the seed the DCU
configuration is *intended* to track is, and it is the committed one, since DCU
is not revalidated here.

Hardware re-check: `tests/integration/ops/test_metax_flaggems.py` on the
eight-device C550 host with `flagtree 0.6.1+metax3.6` / `flag_gems
5.4.0rc2.post1+g5a58df410` reports **90 passed in 756.07s**, 0 failed. That
includes one routing case per withdrawn overload plus three exclusion cases that
run the call on the boxing route — one per cause group above — so a regression
in any of the three failure modes fails a named test rather than only moving a
count.

### `igammac_` rerouted to CUDA boxing (2026-09-15, MetaX C550)

`igammac_` was moved from `flaggems` to `cuda` in `backends_metax.conf` (608
FlagGems Python routes, 12 C++ routes, 1416 cuda boxing routes — the state at
this change; the cohort widening recorded above moved the same file to
592 / 12 / 1432 later the same day, and this withdrawal survives it). It had been
promoted to the FlagGems path earlier the same day, on a probe whose inputs were
strictly positive (`torch.rand(4, 4) + 0.5`); re-measuring on the full argument
domain showed the FlagGems kernel returning finite values where ATen returns
NaN.

Targeted A/B probe, one input pair built on the host and moved with `.to(DEVICE)`
so both arms see identical values, route forced with `FLAGOS_OP_igammac_`:

- `torch.randn(8, 8)` pair (`a` positive and negative, `b` positive and
  negative), route `cuda`: host NaN 51, device NaN 51, one-sided NaN 0.
- Same pair, route `flaggems`: host NaN 51, device NaN 15, one-sided NaN 36.
  `a=-1.1524, b=+0.9200` -> device `0.0275` against host NaN;
  `a=+0.8487, b=-1.4782` and `a=+0.3223, b=-1.6293` -> device `1.0` against NaN.
  Every disagreement is one-sided: no input produced a device NaN that the host
  called finite.
- Same probe on `torch.randn(32, 32)`: 776 host NaNs against 212 device NaNs,
  564 one-sided.

Where both arms are finite the two routes agree to `2.98e-07` (`cuda`: `5.96e-08`),
so this is a domain question, not a precision one: the gems kernel computes a
value on inputs ATen defines as NaN. There is no reverse disagreement on either
probe -- the device never reports NaN where the host is finite. The boxing route
reproduces the host mask exactly, which is why the op keeps a route and only
changes which one.

The change is guarded by `tests/integration/ops/test_metax_flaggems.py`:
`igammac_` is listed in `_FORCED_OFF_FLAGGEMS` (the conf must not route it to
`flaggems`) and in `_FORCED_OFF_DISPATCH` (its dispatch line must read `cuda` on
hardware). The operator-support row for MetaX is unaffected -- this op was never
part of the four-platform FlagGems baseline cohort.

### `slice.Tensor` rerouted to CUDA boxing (2026-09-16, MetaX C550)

`slice.Tensor` moved from `flaggems` to `cuda` in `backends_metax.conf`, which
now reads **591 `flaggems` / 12 `flaggems_cpp` / 1433 `cuda`** over the same
2036-op list (SHA-256
`0d6be6d0fb3aff0aa26293ddf8719bb0f811ea50acbba8a62d02a833b8b154a3`), superseding
the 592 / 12 / 1432 state the cohort-widening section above records. Of the 639
overloads in the raised ceiling, MetaX now routes 584 through the Python
FlagGems path and 12 through the C++ slot, and holds 43 on the cuda boxing
kernel.

The op reached the FlagGems Python path through that widening, and the route
cannot serve it. `flag_gems/ops/slice.py` opens with

```python
    assert input_tensor.dtype not in (
        torch.complex64,
        torch.complex128,
    ), f"slice: unsupported dtype {input_tensor.dtype}"
```

but the function is a pure metadata operation: it returns a zero-copy
`torch.as_strided` view built from the input's shape, strides and storage
offset, and never consults the dtype. The assertion is vestigial, and it is the
only dtype-sensitive line in the function.

What surfaced it is Qwen-Image-2512. `diffusers`'
`QwenImageTransformer2DModel` precomputes its rotary frequencies as complex
tensors and slices them per frame in `_compute_video_freqs`
(`freqs_pos[0][idx : idx + frame]`), so the transformer step aborted at the
assertion on the FlagGems route:

```text
  File ".../diffusers/models/transformers/transformer_qwenimage.py", line 347, in _compute_video_freqs
    freqs_pos = freqs_pos[0][idx : idx + frame]
  File ".../torch_fl/flagos/__init__.py", line 201, in _patched_getitem
  File ".../flag_gems/ops/slice.py", line 199, in slice
AssertionError: slice: unsupported dtype torch.complex64
```

Measured on the eight-device C550 host with `flagtree 0.6.1+metax3.6` /
`flag_gems 5.4.0rc2.post1+g5a58df410`, one complex `(8, 16)` operand built on
the host and moved to the device, then sliced in the shape the model's call
arrives in:

- Shipped configuration, no override: `complex64 slice -> ok (4,) torch.complex64`.
- `FLAGOS_OP_slice__Tensor=flaggems` -- the route the file carried before this
  change: `AssertionError: slice: unsupported dtype torch.complex64`.
- A float32 slice of the same shape is `ok` and shares storage with its input on
  both routes, so it is the dtype check inside the route that fails, not the
  route.

That the assertion is not a kernel limitation was measured rather than inferred.
A probe that deletes only that statement from
`inspect.getsource(flag_gems.ops.slice.slice)` and compiles the remainder --
FlagGems' own code otherwise -- returns a `torch.complex64` result equal to
`torch.Tensor` slicing on the same operand, and the view shares storage with its
input. On a float32 operand the shipped and the patched function agree exactly.

This is not MetaX-specific: `complex64` and `complex128` are rejected on every
device name, and MetaX was the one platform whose conf had the op on FlagGems.
At the commit before this change `slice.Tensor` was already `cuda` in
`backends_cuda.conf`, `backends_ppu.conf` and `backends_dcu.conf`, `none` in
`backends_musa.conf` and `backends_gcu.conf`, and `ascend` in
`backends_ascend.conf` -- the last three because the op is in
`FLAGGEMS_PENDING_NATIVE_OPS`, so it is withheld from their FlagGems routes and
from their generated registrations. The remaining two configurations do not
carry the op at all: `backends_bpu.conf` is intentionally empty, and
`backends_tsingmicro.conf` lists no `slice.Tensor` route. The entry is held in
`metax_triton_fallback`
in `scripts/codegen/codegen_ops.py` rather than in the shared
`flaggems_runtime_broken` set, so the change stays inside MetaX the way the
`special_bessel_j0` group does; it is the counterpart of `slice_backward`, which
is already there for an unrelated MetaX-specific fault. Reported upstream as
FlagGems issue #6356 -- the remaining half of #6049 / #6061, where the same
assertion was trimmed for `torch.bool` and the complex entries were kept.

The change is guarded by `tests/integration/ops/test_metax_flaggems.py`:
`slice.Tensor` is listed in `_FORCED_OFF_FLAGGEMS` (the conf must not route it to
`flaggems`) and in `_FORCED_OFF_DISPATCH` (a complex operand, which is the call
form that failed, must dispatch to `cuda`), and `_MEASURED_FLAGGEMS_ROUTES`
moves 592 -> 591 to match the conf. On the C550 host, with `flagtree
0.6.1+metax3.6` and `flag_gems 5.4.0rc2.post1+g5a58df410`, that file reports **91
passed in 751.82s, 0 failed** -- one case more than the 90-test cohort it
replaces, the added one being `slice.Tensor`'s dispatch check.
`gen_vendor_confs.py` is idempotent for the MetaX configuration, and no
out-of-scope configuration is regenerated: the boxing gap is recovered by
diffing the conf the generator rewrites, which is why `backends_metax.conf` was
edited first. Ascend, GCU, MUSA, DCU and PPU are **not revalidated** and no route
changed for them.

### `_conj` rerouted to CUDA boxing (2026-09-15, Hygon DCU)

`backends_dcu.conf` carried `_conj = flaggems`. ATen's `conj` is a lazy view --
it sets the Conjugate bit and leaves storage untouched -- and
`tests/integration/test_math_bits_contract.py` pins that with
`assert lazy.is_conj()`. FlagGems' `_conj` is a real kernel that materializes the
conjugated values, so registering it on PrivateUse1 replaced the view with an
eager copy and `is_conj()` came back `False`. This is the same defect
`gen_vendor_confs.py` already records for MUSA in `NATIVE_TRITON_GAPS["musa"]`,
where mudnn has no kernel either and the entry routes to `none`.

The DCU manifest's math-bits group is where it surfaced (run 34935660930, job
104272971051, head `a89e869`):

```text
[8/11] Unified math-bits contract      5 passed, 7 errors in 7.37s
tests/integration/test_math_bits_contract.py:74: in conj_tensor
    assert lazy.is_conj(), "torch.conj must stay lazy for this contract to apply"
E   AssertionError: torch.conj must stay lazy for this contract to apply
```

`_conj` now routes to the cuda boxing kernel in `backends_dcu.conf` only. On a
CUDA-boxing platform the fallback is `cuda` rather than `none`, because the
boxing kernel is ATen's own view implementation. This is the DCU conf's
established mechanism: `boxing_triton_gaps()` recovers the pinned set from the
conf itself, so regeneration preserves the entry, and the diagnosis is carried in
`BOXING_GAP_NOTES["dcu"]` for the same reason the `mm`/`bmm` (#6227), `mse_loss`
(#6221) and `transpose.int` (#6219) notes are. The generator stays idempotent for
DCU: `gen_vendor_confs.py --check` was clean for the DCU conf at every head of
this branch. It named `backends_ascend.conf` and `backends_gcu.conf`, a
pre-existing drift on those two platforms rather than on DCU; upstream cured both
halves afterwards, by #285 and #288.

Measured on the eight-device Hygon DCU bw1000 host (PyTorch 2.10.0, FlagGems
`e7b4a865`), holding the build and environment constant and changing only the
route through `FLAGOS_BACKEND_CONFIG`:

| `_conj` route | `torch.conj(t).is_conj()` | result shares `t`'s storage |
| --- | --- | --- |
| `flaggems` (before) | `False` -- materialized | `False` |
| `cuda` (after) | `True` | `True` |

The same A/B over `tests/integration/test_math_bits_contract.py -m math_bits`
passes all seven Conjugate cases under `_conj = cuda`; under `_conj = flaggems`
every one of them errors in the `conj_tensor` fixture before any assertion runs.
That local harness drives the primary checkout's DCU build rather than this
branch's wheel, so its Negative-bit cases fail on `neg`/`arange` with
`no kernel registered for backend 'flagos'` in **both** arms -- an artifact of
that build's FlagGems Python route, not a `_conj` result. The authoritative
numbers for this change remain the CI groups above. CI confirms the reroute: on
run 34939743596 (job 104285567077, head `d81d5b5`), `[8/11]` Unified math-bits
contract reports `12 passed in 2.17s`, against `5 passed, 7 errors in 7.37s` on
the previous head.

The generic `backends_flaggems.conf` cohort and the 546-overload bw1000 row are
**not revalidated** by this change; `_conj` is pinned in `backends_dcu.conf`
alone, and the generic configurations still route it to FlagGems.

### `relu`/`relu_` rerouted to CUDA boxing (2026-09-15, Hygon DCU)

`backends_dcu.conf` carried `relu = flaggems` and `relu_ = flaggems`.
`tests/integration/test_profiler_parity.py::test_kernel_names_are_demangled`
guards against its own vacuity by requiring a kernel name in the workload's trace
to contain `::`. The workload is `(x @ y).relu()` plus a `sort`, and on a CUDA
build the `::` comes from ATen's `at::native::vectorized_elementwise_kernel`
template. With `relu` on FlagGems that kernel is replaced by gems'
`relu_forward_kernel_rank_1`, so no name in the trace contains `::` and the guard
fails.

The DCU manifest's profiler-parity group is where it surfaced (run 34939743596,
job 104285567077, head `d81d5b5`):

```text
[10/11] Profiler parity test        1 failed, 5 passed, 1 xpassed in 8.69s
tests/integration/test_profiler_parity.py:414: in test_kernel_names_are_demangled
E   AssertionError: No kernel name contains '::', so no name in this trace
    required demangling and the check above is vacuous. The workload is
    expected to launch at::native template kernels.
```

`relu` and `relu_` now route to the cuda boxing kernel in `backends_dcu.conf`
only, through the same mechanism as `_conj` above: `boxing_triton_gaps()`
recovers them from the conf, so regeneration preserves the entries, and
`BOXING_GAP_NOTES["dcu"]` carries the diagnosis. `relu.out` was already `cuda`,
so the whole `relu` family now sits on the boxing kernel. The generator stayed
idempotent for DCU at every head of this branch: `gen_vendor_confs.py --check`
was clean for the DCU conf and named only `backends_ascend.conf` and
`backends_gcu.conf` -- a pre-existing drift on those two platforms, which this
change never edits and which upstream cured in #285 and #288.

Measured on the eight-device Hygon DCU bw1000 host (PyTorch 2.10.0, FlagTree
`0.6.2a1+hcu3.6`, FlagGems `e7b4a865`), holding the build and environment
constant and changing only the route through `FLAGOS_BACKEND_CONFIG`, over the
parity workload's device kernel names:

| `relu`/`relu_` route | elementwise kernels in the trace | `::` present |
| --- | --- | --- |
| `flaggems` (before) | `relu_forward_kernel_rank_1` | no |
| `cuda` (after) | `at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(...)>` | yes |

The same A/B over `tests/integration/test_profiler_parity.py -m main_ops`
reproduces the CI result in the `flaggems` arm exactly -- `1 failed, 5 passed,
1 xpassed` -- and is green in the `cuda` arm: `6 passed, 1 xpassed`.

Cross-checks at this revision with the pinned stack, on the same host: `[8/11]`
math-bits `12 passed`, `[9/11]` profiler contract `11 passed, 1 xpassed`,
`[6/11]` general (`test_factory_ops.py`) `46 passed`, and the two files that
exercise `relu`/`relu_` as such -- `tests/integration/test_ops.py` `58 passed`
and `tests/integration/test_compute_device_index.py` `15 passed`. That harness
drives the primary checkout's DCU build rather than this branch's wheel, so these
are cross-checks and not the authority; the CI groups are.

The generic `backends_flaggems.conf` cohort and the 546-overload bw1000 row are
**not revalidated** by this change; `relu` is pinned in `backends_dcu.conf`
alone, and the generic and MUSA configurations still route it to FlagGems.

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

Dated records of work already done, newest first. The environment variable names
in the Evidence column are written in their current spelling, so a re-run uses
the names the tree reads today; where a row predates the `FLAGOS_*` unification
that renamed them, the command is the same command under a different name and
the recorded result is unchanged.

| Date | Hardware | Cohort | Change | Evidence |
|---|---|---|---|---|
| 2026-09-18 | MetaX C550 (8 devices) | MetaX composite SDPA routed to FlagGems (hand-written whole-op override, outside the 639-overload ceiling) | `aten::scaled_dot_product_attention` became a `flaggems` route in `backends_metax.conf`, taking the file to **592 `flaggems` / 12 `flaggems_cpp` / 1433 `cuda`** over a **2037**-op list (SHA-256 `bb1dc5c4550339dcd44ac438b2981e703c882025e477221e86cac37d833f58f2`), superseding 591 / 12 / 1433 over 2036. The op cannot be routed leaf by leaf: it is a composite whose fused-backend selection runs *inside* it and then branches on `query.device().type()`, so on PrivateUse1 the generated leaves are never consulted and no per-op route can reach a fused kernel. The override is hand-written — `csrc/aten/sdp_choice_stub.cc` registers the composite itself on PrivateUse1 and decides inside it — and it is registered only on the CUDA-boxing builds, which is why `gen_vendor_confs.py` gains `EXTRA_ROUTED` (an op that appears in no `.inc` is invisible to every coverage scan, so the op list has to be widened by hand) and `METAX_COMPOSITE_FLAGGEMS` to hold it from every other platform's conf the way `METAX_FLAGGEMS_MEASURED` holds the measured leaf routes. It is gated on a new `HasBackendForOp()` in `csrc/aten/common.h`, which tells a conf that names the op from one that is silent about it: without it the op would read `kFlagGems` from `GetBackendForOp`'s table-miss default, and a conf written before this change — including a third party's wheel — would take the route silently. The envelope is bf16, 4-D, head_dim 128 exactly, query seq >= 1024, no mask, no causal, no explicit scale, no dropout, no gqa; every shape outside it falls through to the pre-existing boxing path. Ascend, GCU, MUSA, DCU and PPU are **not revalidated** — each gained exactly one line, `none` on ascend/gcu/musa and `cuda` on dcu/ppu, and no measurement transfers to them. | Qwen-Image-2512, 1024x1024, 50-step denoise, one seed, on the 8-device C550 host with `flagtree 0.6.1+metax3.6`, MACA 3.8.0 in CUDA-boxing mode, `flag_gems 5.4.0rc2.post1+g5a58df410` and Triton 3.6.0 (`metax`). Pipeline, `build_pipeline` and placement held constant: steady **78.49 s against 184.02 s**, first step 90.26 s against 189.34 s (**2.34x**, 57.3% off the loop), `pipeline_load_s` 134.74 in both arms; the same window re-run as three arms that also write latents and pixels gives 84.46 s with the route active against 183.68 s and 182.39 s with it off. Op level at the shape the route serves, `(1, 24, 4114, 128)` bf16, min of five calls after warm-up: FlagGems `5.908 ms/call` against the boxing route's `23.313 ms/call` (**3.95x**). A 19-clause envelope probe reports `route hits: 6` — the joint shape and its seq-4096/2048/1024/non-contiguous variants — while head_dim 64/256/512, seq 1023, fp16, fp32, 2-D, 3-D, mask, `is_causal`, scale, dropout and gqa all box. Head_dim is measured in both directions: at 512 the kernel has no configuration that compiles on this part (`triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 294912, Hardware limit: 65536`, against an autotune set `attention.py:173` caps at `BLOCK_N <= 32`), while 128 and 64 both complete and match the unfused fp32 math path (`max 2.189e-05 mean 2.365e-06` and `max 2.153e-05 mean 2.421e-06`), so 128 is where the route was measured and not the kernel's outer boundary. Numerics over the same seeded operands (digest `37efe866eb24ada5` in all three arms): route against boxing `max 9.766e-04 mean 3.274e-05`, and the two boxing arms — `FLAGOS_OP_scaled_dot_product_attention=cuda` on the shipped conf and a conf that never names the op — agree exactly (`max 0.000e+00 mean 0.000e+00`), which is the measurement of the `HasBackendForOp()` rule. Image cost with its own control in the same window: the two route-off arms differ by `max 0.0000 mean 0.00000`, and route-on against them by `max 170.3320 mean 2.04607 (/255)`, 1150098 of 3145728 pixels over 1/255 and 312147 over 4/255, against the prototype arms' `1.39572` route effect and `1.25745` flagos-vs-vendor. Survey: `tests/manual/flaggems_overload_survey.py` v6 (`7b01c22c…`) scoped to the changed route reports `registered 1`, `tested 1`, verdict **`FAILED`**, `basic_executable 0`, `strict_support 0`; the cause is measured as the harness's `default_for()` catch-all `return 0.5` synthesizing `dropout_p = 0.5` (the argument's name is `dropout_p` but the branch ends on `return 0.5`, which names `p`), and the same seven profiles at `dropout_p = 0.0` turn all four `WRONG` verdicts into `PASS` with the three `INVALID_CASE` profiles unchanged — and none of the four is inside the route's envelope, so the verdict is the harness's and not the route's. Regression coverage: `tests/integration/ops/test_metax_flaggems.py` gains `TestMetaXFlaggemsSdpaRoute` and `_MEASURED_FLAGGEMS_ROUTES` moves 591 -> 592. On the C550 host with the route active the whole file reports **101 passed in 854.53s, 0 failed** (exit 0), the class subset **10 passed in 43.28s**, against 90- and 91-test cohorts on the two entries below. `gen_vendor_confs.py` idempotent (two runs, all nine confs byte-identical; `--check` reports `all vendor confs up to date`). Full detail: "MetaX: `scaled_dot_product_attention` routed to FlagGems" above. |
| 2026-09-16 | MetaX C550 (8 devices) | FlagGems `slice` dtype assertion, found by Qwen-Image-2512 | `slice.Tensor` moved from `flaggems` to `cuda` in `backends_metax.conf`: 592 `flaggems` / 12 `flaggems_cpp` / 1432 `cuda` -> **591 / 12 / 1433** (SHA-256 `0d6be6d0fb3aff0aa26293ddf8719bb0f811ea50acbba8a62d02a833b8b154a3`), so the measured Python cohort moves 585 -> 584 and the cuda boxing hold 42 -> 43. The entry is held in `metax_triton_fallback` in `scripts/codegen/codegen_ops.py` rather than in the shared `flaggems_runtime_broken` set, the way the `special_bessel_j0` group is, so no other platform moves; it sits next to `slice_backward`, which is already there for an unrelated MetaX fault. At the base commit `slice.Tensor` was already `cuda` in `backends_cuda.conf`, `backends_ppu.conf` and `backends_dcu.conf`, `none` in `backends_musa.conf` and `backends_gcu.conf` and `ascend` in `backends_ascend.conf` (the last three through `FLAGGEMS_PENDING_NATIVE_OPS`), and does not appear at all in `backends_bpu.conf` (intentionally empty) or `backends_tsingmicro.conf`, so MetaX is the one platform whose conf carried it on FlagGems and this is a convergence, not a new exception. Ascend, GCU, MUSA, DCU and PPU are **not revalidated**. | `flag_gems/ops/slice.py::slice` asserts against `complex64`/`complex128` at line 199 while its body is `torch.as_strided(input, size, strides, storage_offset)` and reads no dtype. Qwen-Image-2512 reproduces it at model level: diffusers' `QwenImageTransformer2DModel._compute_video_freqs` (`/diffusers/models/transformers/transformer_qwenimage.py:347`, `freqs_pos[0][idx : idx + frame]`) reaches `flag_gems/ops/slice.py:199` through `torch_fl/flagos/__init__.py:201` and raises `AssertionError: slice: unsupported dtype torch.complex64`, which aborts the transformer step. Per-op A/B on C550 with `flagtree 0.6.1+metax3.6` / `flag_gems 5.4.0rc2.post1+g5a58df410`, one host-built complex `(8, 16)` operand moved with `.to("flagos")`: shipped conf (no override) `ok (4,) torch.complex64`; `FLAGOS_OP_slice__Tensor=flaggems` `AssertionError: slice: unsupported dtype torch.complex64`; float32 the same shape `ok` with storage shared on both routes, so the dtype check inside the route is the failure and not the route. The assertion is not a kernel limit: an in-process probe that deletes only that statement from `inspect.getsource(...)` and compiles the rest of FlagGems' own function returns a `complex64` result equal to `torch.Tensor` slicing with storage shared, and agrees with the shipped function exactly on float32. Regression coverage in `tests/integration/ops/test_metax_flaggems.py`: `slice.Tensor` added to `_FORCED_OFF_FLAGGEMS` and to `_FORCED_OFF_DISPATCH` (complex operand), `_MEASURED_FLAGGEMS_ROUTES` 592 -> 591. `gen_vendor_confs.py` idempotent for MetaX (two runs, byte-identical), `--check` clean for the MetaX file; the out-of-scope configurations were left at their committed state. `tests/integration/ops/test_metax_flaggems.py` on the C550 host reports **91 passed in 751.82s, 0 failed** (the 90-test cohort plus the new dispatch case), including both `slice.Tensor` guards. Filed upstream as FlagGems issue #6356. |
| 2026-09-16 | MUSA MTT S5000 | MUSA FlagGems RNG bridge | `5e4b78e` (the qualname change above) moved MUSA's generated kernels from `flag_gems.ops.randn.randn` to `flag_gems.randn`, which `SpecOpRegistrar` has rebound to the vendor override `_mthreads.ops.randn.randn`. `_patch_flaggems_philox()` selected the modules to rebind with `mod.__name__.startswith("flag_gems")`, which that module does not match, so the vendor kernel reached the unpatched `philox_backend_seed_offset` and raised `ValueError: too many values to unpack (expected 2)` unpacking the flagos MT19937 state. The loop now matches the bound object's identity instead of the module name. No route changed: `randn`, `randn_like`, `rand`, `rand_like`, `randperm`, `native_dropout` stay `flaggems  # musa`. | `Platform pipeline (musa) / Build and test (MUSA)` on run `35048300960` (push of `5e4b78e`) failed with **14 failed, 99 passed, 28 warnings in 193.65s**; the preceding `main` runs on `35047336702` and `35041546614` were green on MUSA. Both signatures are the same `torch.randn(..., device="flagos:0")` call: 4 in-process failures on `ValueError` at `flag_gems/utils/random_utils.py:75`, and 10 `test_musa_dispatch.py` subprocesses whose first statement is that call. Regression coverage: `tests/unit/test_musa_rng_bridge.py::test_flaggems_philox_reaches_vendor_backend_modules` fails against the pre-fix selector (`999 != -9223372036854775803`) and passes after; `tests/integration/ops/test_musa_flaggems.py::test_flaggems_randn_shares_native_generator_reservations` now drives `flag_gems.randn` rather than the generic module, which the MUSA dispatch never reached. Full detail: "MUSA: the FlagGems RNG bridge reaches the vendor op modules" above. |
| 2026-09-15 | MetaX C550 (8 devices) | FlagGems entry-point resolution (`5a58df410`) | `_normalize_flaggems_qualname` in `scripts/codegen/codegen_ops.py` now emits `flag_gems.<fn>` instead of `flag_gems.ops.<module>.<fn>`, so a generated kernel reaches the entry point the active backend has rebound rather than the generic module the alias rewrite pinned. 72 of the 666 qualnames in the checked-in kernels resolve to a `_metax.ops.*` override and were running the generic kernel before this. `codegen_ops.py` also becomes the writer of the `FLAGGEMS_PYTHON_OPS` ceiling in `scripts/codegen/backend_coverage.py` (`render_flaggems_coverage`, minus the override-only ops), which was previously a hand-carried literal that capped every conf built from it. Both apply to every FlagGems platform; no route changed on Ascend, GCU, MUSA, DCU or PPU. | Counted over `csrc/aten/generated/flaggems_python_kernels.cc` with `flag_gems 5.4.0rc2.post1+g5a58df410` on the C550 host: 688 call sites, 666 distinct qualnames, 0 that are not two-component `flag_gems.<op>`, 0 unresolvable on the package, 594 resolving inside `flag_gems` and 72 to a `_metax.ops.*` module. `tests/integration/ops/test_flaggems_conf_consistency.py` requires the two-component form and now compares the conf, the override-only routes and the generated kernels as sets (7 passed); `tests/integration/ops/test_metax_flaggems.py` on C550 reports **90 passed in 756.07s**, 0 failed. Full detail: "MetaX: generated FlagGems calls name the package-level entry point" above. |
| 2026-09-15 | MetaX C550 (8 devices) | FlagGems master coverage cohort (`5a58df410`) | Rebuilt `FLAGGEMS_PYTHON_OPS` on the FlagGems master cohort pinned at `5a58df410c551c4f4eb41d31887cd75fd596804a`: 482 -> 639 overloads, 158 added and `mul_.Tensor` removed because that cohort does not cover it. The newly covered overloads are withheld from the Ascend, GCU and MUSA configurations by `FLAGGEMS_PENDING_NATIVE_VENDORS` / `FLAGGEMS_PENDING_NATIVE_OPS` so their shipped counts do not move without hardware; DCU loses `mul_.Tensor` to `cuda` (three lines) for the same reason as MetaX. MetaX was re-measured against the raised ceiling and **sixteen overloads were withdrawn back to the CUDA boxing kernel** after a differential A/B probe showed each one passing on `cuda` and failing on `flaggems`: `special_bessel_j0`, `special_i1e`, `special_i1e.out`, `special_chebyshev_polynomial_w.out` (kernel asserts its input is a real CUDA tensor), `nansum.out`, `lu_unpack.out`, `linalg_matrix_exp.out`, `sum.out`, `_cdist_forward` (the gems wrapper cannot serve the caller's call form), and `_compute_linear_combination`, `_compute_linear_combination.out`, `_fused_rms_norm`, `igamma`, `igamma_`, `logit_backward`, `special_shifted_chebyshev_polynomial_t` (wrong result). `backends_metax.conf`: 443 `flaggems` / 11 `flaggems_cpp` / 1582 `cuda` (committed) -> 592 / 12 / 1432, via the widened intermediate 608 / 12 / 1416. Ascend, GCU, MUSA, DCU and PPU are **not revalidated** against the raised ceiling; only DCU's `mul_.Tensor` line moves and no MetaX measurement is transferred to them. | Screening survey over the 166 overloads whose route changed in `backends_metax.conf`, `2d-f32` profile, harness v5: `{"registered": 166, "tested": 97, "STRICT": 76, "FAILED": 21, "UNTESTED": 69}`, `basic_executable` 76. The 21 `FAILED` overloads re-run with `FLAGOS_OP_<op>=cuda` (one host-built input pair moved with `.to("flagos")`, both arms identical values): 16 `cuda` PASS with the `flaggems` verdicts in the table above, 5 fail on both routes so they keep their route. Replaying the 16 through the shipped configuration with no override reproduces 16 PASS. `gen_vendor_confs.py` idempotent (two runs, empty diff; `--check` exits 0 for the MetaX file), and running the two generators over this tree leaves `backends_metax.conf`, every generated artifact and `backend_coverage.py` byte-identical — the out-of-scope configurations do move on that first pass, which the ordering note above records. Full detail: "MetaX: FlagGems cohort widened to FlagGems master, sixteen ops withdrawn" above. |
| 2026-09-15 | MetaX C550 (8 devices) | MetaX FlagGems hybrid path | Promoted 8 overloads to the Python FlagGems path on MetaX only, through `METAX_FLAGGEMS_MEASURED` in `scripts/codegen/gen_vendor_confs.py`, because their `flag_gems.<name>` entry points exist in the pinned cohort while the shared hold was written against an older one: `_embedding_bag_per_sample_weights_backward`, `_native_batch_norm_legit_functional`, `binary_cross_entropy_with_logits`, `linalg_ldl_solve`, `special_bessel_j1`, `unsqueeze`, `unsqueeze_`. Two of them were failing outright on the cuda boxing route before this, so the promotion is a fix and not a preference: `special_bessel_j1` raises `cudaErrorMemoryValueTooLarge` through maca, and `linalg_ldl_solve` needs a `cusolverDnXsytrs_bufferSize` symbol maca does not provide. The eighth, `igammac_`, was promoted and then withdrawn the same day (see "`igammac_` rerouted to CUDA boxing" above). No other platform's routes changed. | `tests/integration/ops/test_metax_flaggems.py` on C550 with `flagtree 0.6.1+metax3.6` / `flag_gems 5.4.0rc2.post1+g5a58df410`: **90 passed in 756.07s**, 0 failed — the routing cases, the execution cases, and the exclusion cases including the three representative withdrawals added by the cohort widening recorded above. `gen_vendor_confs.py` idempotent for the MetaX configuration. Ascend, GCU, MUSA, DCU and PPU are **not revalidated** by this change -- `METAX_FLAGGEMS_MEASURED` is consulted only for `backends_metax.conf`. |
| 2026-09-15 | NVIDIA A100-SXM4-40GB (8 devices) | CUDA FlagGems device-name alignment | Made FlagGems' device name equal the name torch_fl registers, so FlagGems' own device guards stop rejecting `flagos` operands. Its nvidia descriptor names the device `cuda` (`flag_gems/runtime/backend/_nvidia/__init__.py`) and caches that in a process-wide singleton, so the two guards that compare `tensor.device.type` against the name were always false on a `flagos` tensor. `torch_fl/accelerator/cuda/_cuda_compat.py:patch_flaggems_device_name()`, called from `torch_fl.flagos.init()` before the first route executes, rewrites the singleton and every module-level copy of the name (`device`, `_DEVICE_NAME`) in the already-imported `flag_gems` modules. FlagGems is not patched or forked; the rewrite is applied to the imported modules from torch-fl. It is deliberately narrow: it acts only when FlagGems resolved the `nvidia` vendor and the name is the vendor literal, and it leaves every other vendor and every already-matching registration alone. **No route value changed** — `backends_cuda.conf` is byte-identical (`ab2522b7`, 416 `flaggems` / 1618 `cuda` before and after); what changed is which code path eleven of those routes take. Thirteen guarded overloads stay on CUDA boxing either way: nine of them compare against the device *name* (the alignment unblocks the guard, but they have not been re-measured for correctness on the FlagGems route) and four assert on `Tensor.is_cuda`, which no device name can satisfy. `scripts/codegen/codegen_ops.py:measured_flaggems_rollback` records that split. MetaX, PPU, DCU, Ascend, GCU, MUSA and Tsingmicro are **not revalidated** and no FlagGems route changed for them. | Full survey rerun on the same host against the same conf SHA and FlagGems `7fb49bad47116434961bfb2b912811716d383eaf` with the alignment active: all 416 routes measured in both runs, **0 verdict differences and 0 per-case status differences over the 2912 shared cases**; the two tables above are byte-identical and unchanged at 416 / 321 / 7 / 0 / 88 / 328. Fifteen cases differ only in the *text* of their error (an ATen internal source line, the internal function name a `NotImplementedError` names, raw pointer addresses on a padding error) while carrying `INVALID_CASE` in both runs. The blast radius was measured rather than assumed: of the 2034 routed overloads, 75 sit on a FlagGems module that guards on the device and 11 of those are `flaggems`-routed (`_embedding_bag_dense_backward`, `_upsample_nearest_exact2d_backward`, `eq.Scalar`, `eq.Tensor`, `mul.Tensor`, `reflection_pad2d`, `reflection_pad2d.out`, `reflection_pad3d`, `reflection_pad3d.out`, `upsample_trilinear3d`, `zero_`). The guarded rollback group was re-measured per op in both arms of an in-process A/B that only changes the name: the nine name-guarded overloads raise their guard with the vendor literal restored and return a tensor with the alignment in place (covering both the call-time and the import-time `_DEVICE_NAME` snapshot shapes), and the four `Tensor.is_cuda` overloads are blocked in both arms. New `tests/integration/ops/test_flaggems_device_name.py`: 4 passed in 1.92s with the fix, 3 failed / 1 passed against the reverted source (`assert 'cuda' == 'flagos'` twice, plus the `aten::mul()` RuntimeError). **Evidence gap:** the routed end-to-end crash is not reproducible on this host, because the staged `libtorch_fl.so` predates the wrapped-number conversion (`TensorToPython`) that makes a Python scalar reach `mul.Tensor` as a `float`; here the pre-fix mismatch was reachable through the FlagGems entry point but not through the routed path, so the routed failure is evidenced by the in-process A/B and by the new test rather than by a survey case. |
| 2026-09-15 | PPU 810e (16 devices), FlagTree `0.6.2a2+ppu3.6`, FlagGems `5.4.0rc2.post1+gd45285ba6` | PPU FlagGems-first routing and op-list provenance | Two changes to PPU's generated routing. (1) `gen_vendor_confs.py` now reads the op list every conf must cover from `csrc/aten/generated/register.inc` instead of `backends_cuda.conf`, so PPU's op universe is no longer a function of the CUDA platform's routing table; proven provenance-only, since regenerating with the new source leaves all nine confs byte-identical. (2) The 478-route FlagGems-first set was surveyed and the routes that could not run on the FlagGems path were pinned back to the boxing kernel: PPU `flaggems` 478 -> 435, `cuda` 1558 -> 1601, `none` 0, over the same 2036-op list, still 100% covered. 47 FlagGems-covered ops stay on `cuda`, recorded per-op in `BOXING_TRITON_GAPS["ppu"]`: the pre-existing mm/bmm family (FlagGems issue #6225, not a FlagTree finding), 33 survey-measured route-dependent failures, six the survey cannot reach, and four reflection-padding routes that no survey profile reaches. FlagGems is not patched. Every other platform's conf is untouched and every non-PPU hardware row, including the historical 546-overload PPU cohort, is **not revalidated**. | `tests/manual/flaggems_overload_survey.py` (harness v4) over the widened 478-route set on `flagos:0`: **312 STRICT / 46 BASIC_ONLY / 42 FAILED / 78 UNTESTED** over the 400 routes with at least one CPU-valid case of 478 registered (358 basic-executable). The 42 FAILED routes re-run on the same overloads with `FLAGOS_OP_<op>=cuda` returned **29 STRICT / 4 BASIC_ONLY / 9 FAILED**, which separates the 33 route-dependent failures (pass on the boxing kernel, fail on FlagGems) from 9 that fail on both routes and therefore stay on FlagGems: `_batch_norm_no_update`, `_log_softmax_backward_data`, `_softmax_backward_data`, `linalg_ldl_factor_ex`, `mse_loss_backward`, `native_batch_norm`, `scatter.src`, `scatter_.src`, `unique_dim` — `_batch_norm_no_update` segfaults on both (`returncode -11` on all seven profiles). The 33 split by failure mode into ten FlagGems device-guard refusals (`is_cuda` is false on a PrivateUse1 tensor), three FlagTree ppu `CompilationError`s (`randint`, `randint_like`, `norm.ScalarOpt_dim`), three `out=`/alias failures (`cosh.out`, `sum.out`, `mul_.Tensor`), ten numerically wrong results every profile that ran, and seven raises; each is recorded with its measured failure inline in `BOXING_TRITON_GAPS["ppu"]`. On the shipped conf: operator step 1 (vendor backend) **126 passed, 15 skipped, 1002 deselected, 1 xpassed in 125.33s**, operator step 2 (FlagGems runtime path, `FLAGOS_USE_FLAGGEMS=1`) **11 passed, 1132 deselected, 1 xpassed, in 99.09s**, exit 0, with no op in that cohort needing a vendor fallback. That shipped conf then failed two CI steps the operator cohorts do not cover, so a second round pinned six more ops after reproducing both failures locally and confirming each causally with `FLAGOS_OP_<op>=cuda`: the five `addmm` overloads, whose FlagGems autotune picks a `BLOCK_SIZE_K < 16` config the FlagTree ppu backend rejects in `tl.dot` (`tests/integration/test_factory_ops.py::TestCopyTransfer::test_module_cpu_after_forward`, `nn.Linear(8, 8)` over a `(4, 8)` input: 1 failed / 45 passed before, **46 passed in 1.95s** after), and `_conj`, which FlagGems materializes eagerly where ATen's metadata operator must leave the Conjugate bit set (`tests/integration/test_math_bits_contract.py -m math_bits`: 5 passed / 7 errors before, **12 passed in 1.00s** after). The two model-level manifest steps were run locally against the same hardware and the local Qwen3-0.6B snapshot, because the widened route is exactly what they exercise: on the shipped commit `test_qwen3_infer.py` **4 passed in 32.78s** and `test_qwen3_train.py` **3 passed in 12.82s** (an earlier run on the pre-rebase tree passed in 87.08s and 55.80s). In CI on the same commit (PPU 810e runner, run 35006023949) the manifest reproduced steps [3/8] through [6/8] — **126 passed, 16 skipped, 995 deselected, 1 xpassed in 122.50s**; **11 passed, 1 skipped, 1125 deselected, 1 xpassed in 87.33s**; **46 passed in 16.17s**; **12 passed in 1.61s** — and then failed [7/8] at collection because the runner's own `/models/Qwen3-0.6B` mount no longer holds a Qwen3 snapshot, so [8/8] never ran; the two model steps therefore remain local evidence. Generator idempotent (second run byte-identical, SHA-256 `0aa2c5ba9825ec57852f63ed7c5437d40b2982c9402515540877bc9ca579bf6d`). The last four pins were not measured but audited: the four `reflection_pad` routes were resolved to their Python entry points through the generated `flaggems_python_kernels.cc` and `flag_gems._FULL_CONFIG`, and each was found to raise `input must be a cuda tensor` against a device name PPU's `_thead` descriptor declares as `"cuda"` while its tensors report `"flagos"` -- the alias `i0` and the other nine device-guard refusals were already pinned for. The survey cannot see them because the harness derives `padding` from the rank, so all seven profiles on each route are rejected by ATen's arity check before the guard is reached. The audit was run against the host's FlagGems (`5.3.1.post1.dev212+g7fb49bad4`), older than the master the CI job installs, so four is a lower bound on that revision. The boxing route they are pinned to is not re-measured on PPU; it is the route the platform used before the widening. `ruff check .` -- "All checks passed!"; `ruff format --check .` -- 251 files already formatted. `tests/unit/test_gen_vendor_confs.py`: 35 passed — the ascend/gcu conf-staleness failure this work saw while in progress was fixed upstream by #285/#288, and `gen_vendor_confs.py --check` is clean on the rebased base. |
| 2026-09-15 | MTT S5000 (8 devices) | MUSA FlagGems gap re-measurement | Re-probed all 18 `NATIVE_TRITON_GAPS["musa"]` entries against the FlagGems revision the MUSA CI job installs, on each entry's recorded failure signature. Four no longer reproduce and are promoted out of the set: `index_add` and `index_add_` (recorded as "returns all zeros") now route to `flaggems` from `none`, and `randn`/`randn_like` (recorded as "crashes unpacking generator state") route to `flaggems` with the mudnn kernel retained as `flaggems  # musa`. The other fourteen keep their routes with provenance updated to `4d9c34775`; `_conj` stays because its probe *passes* (flag_gems materializes the conjugate where ATen's lazy view must set the Conjugate bit). MUSA `flaggems` 464 -> 468, `musa` 51 -> 49, `none` 1521 -> 1519; registered-op set unchanged at 518, `musa_flaggems_register.inc` 357 -> 359 `m.impl` lines. FlagGems is not patched. A100/mc550/PPU/DCU rows and every non-MUSA platform are **not revalidated**. | Per-op probe, one fresh process each, `FLAGOS_OP_*` pinning the op back to FlagGems, `FLAGOS_LOG=dispatch`/`FLAGOS_LOG=fallback`: `index_add`, `index_add_`, `randn`, `randn_like` PASS (0/7, 0/7, 0/4, 0/4) and the 13 entries kept in the set reproduce their recorded signature exactly (bf16 `failed to translate module to LLVM IR`; `no fallback function is registered for schema aten::mul.out` for f32 and bf16, with `aten.mul.out` itself verified usable on MUSA and the `flag_gems/ops/mul.py:587` device-name guard confirmed live; the trailing-store loss at `n = 3,5,6,7,9,15,17,31,33,100`; `RuntimeError: MudnnCopy: unsupported dtype Long -> UInt32`). Comparator control rejects a perturbed reference. Provenance beyond verdicts: `index_add`/`index_add_` each add a new flag_gems code-cache entry, so the mthreads kernel compiled and ran on device, and every fallback line in those rows is the probe's own CPU comparison. End-to-end on the rebuilt library with the shipped conf: 14/14 cases pass, dispatch log showing `index_add`/`index_add_`/`randn`/`randn_like -> flagos_python` against `sort`/`add.Tensor -> musa` regression controls. `index_add` with duplicate indices and `alpha = 2.5` is bounded, not assumed: 11/20 seeds differ from CPU by at most `4.768e-07` (one float32 ULP) on the duplicated rows only, `alpha == 1` bit-exact, matching ATen's documented order-freedom for duplicate indices. CI groups re-run: dispatch 113 passed; factory 46 passed; operator cohort 493 passed/1 skipped/521 deselected/2 xfailed/1 xpassed plus the 3 pre-existing consistency failures; RNG 80 passed/37 deselected with the manifest's `-k` filter. Generators idempotent (`codegen_musa_flaggems.py --check` "is up to date", `gen_vendor_confs.py --check` clean for MUSA); `codegen_musa_flaggems.py` must run before `gen_vendor_confs.py`. `tests/unit/test_gen_vendor_confs.py`: 34 passed, 1 pre-existing ascend/gcu drift failure. `flaggems_overload_survey.py` cannot measure these routes — evidence gap recorded in the section above. |
| 2026-09-15 | Enflame GCU S60 (8 `flagos` devices) | GCU FlagGems routing | Made `backends_gcu.conf` FlagGems-first via a new generated registration file (`scripts/codegen/codegen_gcu_flaggems.py` -> `csrc/aten/backends/gcu/generated/gcu_flaggems_register.inc`, 249 `m.impl` lines), included by `csrc/aten/register.cc` after `gcu_register.inc`. GCU `flaggems` 0 -> 257, `gcu` 152 -> 144, `none` 1884 -> 1635; accelerated routes 152 -> 401 (7% -> 19.7%). `NATIVE_TRITON_GAPS["gcu"]` 108 -> 225: 81 routes measured wrong at `float16`/`float32`, plus 36 that fail only for `int64`/`bool` and have a topsaten kernel to fall back to. 76 `int64`-only routes with no topsaten kernel are deliberately **left on FlagGems** rather than demoted to `cpu_fallback` for float too; they now raise `Pipeline run failed` for an `int64` operand where the previous configuration served the call through `cpu_fallback`. FlagGems is not patched or forked. Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are **not revalidated** and no route changed for them. | `flaggems_overload_survey.py` (harness v4) on the S60 against flagtree `0.6.1+enflame3.6` (Triton 3.6, backend `enflame`, FlagGems master `3c6f7537d`), 7 profiles per overload over all 374 FlagGems routes (a transient un-gapped draft of `backends_gcu.conf`, `meta.conf_sha256` `82f801778c…`; it reconciles with the shipped conf as 374 - 117 = 257 and is not byte-recoverable): 314 tested, 121 strict, 121 clean on every exercised profile, 81 wrong at `float16`/`float32`, 112 wrong only for `int64`/`bool`, 60 with no constructible case. Failure families reproduced and recorded: GCU300 `64-bit data type not supported` / `Pipeline run failed: PassManager execution failed` (largest family), `arith.maxsi` UNREACHABLE at `PtrAnalysis.cpp:1711` (`_adaptive_avg_pool2d`), `unsupported extern elementwise: __nv_asinf` UNREACHABLE at `ElementwiseFusionOpToGCU.cpp:874` (`asin`), SIP abort at `dtu_context_obj.cc:693` (`addr`), SIGSEGV (`native_batch_norm`, `_batch_norm_no_update`), and measured wrong values on float profiles (`elu` `max_diff` 0.38-0.89, `histc` up to 1536, `_softmax_backward_data` returning `int8`, `sum.out` returning `(32, 32)` for `()`). Full `.github/configs/gcu.yml` pytest manifest run locally in one pass, all seven groups rc=0: vendor operator cohort 595 passed/32 skipped/499 deselected/2 xfailed/2 xpassed, FlagGems runtime path 9 passed/4 skipped/1116 deselected/1 xpassed, unified RNG 111 passed/4 skipped/1 deselected/1 xpassed, general 46 passed, AMP 27 passed, math-bits 12 passed, `torch.compile` 29 passed/18 skipped; conf consistency 7 passed; routing equals registration (no op routed to `gcu` without a `gcu_register.inc` entry, none registered-but-left-`none`). Both generators idempotent (two runs byte-identical; `--check` exit 0). The environment group (`set_env_gcu.sh`, `CI_STAGE=integration`) was reproduced into a scratch venv: TopsRider discovery, `/dev/gcu0`, the venv bootstrap, CPU torch 2.10.0, flagtree from the FlagOS index and FlagGems `3c6f7537d` from git all succeed, and its Triton/flag_gems verification snippet passes; on the measurement host alone it needs a local, uncommitted retarget of libtriton.so's single glibc-2.38 symbol, because that host is Ubuntu 22.04 while the wheel and the pinned ubuntu24.04 CI image are not. Evidence gaps recorded: the 60 unconstructible routes are not measured, the two batch-norm process deaths are gapped on exit status alone because the harness truncates stderr at 300 bytes, and no part of this change has been executed by CI yet. |
| 2026-09-15 | Ascend 910 (910/910B host, CANN 9.0.0) — **910C not revalidated** | Ascend FlagTree migration, widened FlagGems route, and a runtime float64 escape | Moved Ascend's FlagGems route from `triton-ascend 3.2.2` to FlagTree `0.6.2a1+ascend3.5` (Triton 3.5) and re-measured the coverage on the new stack instead of inheriting it. `pow.Scalar`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor`, `rsqrt`, `rsqrt_` return to FlagGems (the triton-ascend crash behind FlagGems issue #6226 does not reproduce). 23 overloads return to aclnn: `mm`/`mm.out` (Ascend tune config passes `SPLIT_K` to a kernel that does not take it), the twelve `eq`/`ge`/`gt`/`le`/`lt`/`ne` comparison overloads (float32 evaluation is silently wrong above 2**24), `rand`/`rand_like`/`randperm`/`exponential_`/`native_dropout`/`native_dropout_backward` and `sort`/`sort.stable` (all rejected by BiShengHIR, mostly on the unified-buffer budget), and `mul_.Tensor` (FlagGems' `mul.py` gates on the runtime device *name* and mis-redispatches). Ascend `flaggems` 241 -> 225, `ascend` 133 -> 149, `none` 1662; conf SHA-256 `04a5380a...c252412` (was `8ce7c8c7...4ba3384`). Because a conf cannot express a per-dtype exception, the float64 gap is handled at runtime: `FlagGemsRejectsDtype` in `csrc/aten/common.cc`, consulted by `Dispatcher::ResolveFn`, which sees Tensor, `optional<Tensor>`, Tensor-list, `optional<ScalarType>` and bare `ScalarType` arguments. Also fixed per-device default ACL streams and the executor-cache device key. FlagGems is not patched. All other platform rows are **not revalidated** and no FlagGems route changed for them. | 22-op float64/float32 probe on `flagos:0`: 22/22 float32 and 22/22 float64 pass, against 17 of 22 float64 cases raising `MLIRCompilationError` before the escape. Full `.github/configs/ascend.yml` manifest run locally on an Ascend 910: operator cohort `-m ascend` 38 passed / 1099 deselected, 44 passed with the new test; RNG `-m main_ops` 112 passed / 3 skipped / 1 deselected / 1 xpassed; factory 46 passed; AMP contract 27 passed (4 failing / 23 passing before); math-bits 5 passed / 7 skipped; profiler contract 2 passed / 10 skipped with the MSPTI preload. New `tests/integration/ops/test_dtype_route_fallback.py` (6 passed) pins the float64 escape through `FLAGOS_LOG=dispatch` in both dtypes in one process. Generator idempotent (two runs byte-identical; `gen_vendor_confs.py --check` clean for Ascend and MUSA). `flaggems_overload_survey.py` cannot measure these routes, so the generic Ascend FlagGems rows are not revalidated — evidence gap recorded in the section above. `tests/unit/test_gen_vendor_confs.py::test_shipped_confs_are_up_to_date` still fails on `backends_gcu.conf`; measured as pre-existing, since the base-commit and branch generators emit byte-identical GCU output. |
| 2026-09-15 | Hygon DCU bw1000 | DCU FlagGems path enabled by default in CI | `.github/scripts/set_env_dcu.sh` installs FlagTree (`0.6.2a1+hcu3.6`) and FlagGems (`e7b4a865`) and exports `FLAGOS_USE_FLAGGEMS=1` through `$GITHUB_ENV`, replacing an inline `FLAGOS_USE_FLAGGEMS=1` in `.github/configs/dcu.yml` that ran against a venv with no `flag_gems` and no registered `hcu` backend. `backends_dcu.conf` carries **three route changes**, all from `flaggems` to `cuda`: `_conj` after the math-bits group failed 7/12 on FlagGems' materializing `_conj` destroying ATen's lazy Conjugate view, and `relu`/`relu_` after the profiler-parity group's demangling guard failed as vacuous because gems' `relu_forward_kernel_rank_1` displaces ATen's `at::native::vectorized_elementwise_kernel` from the trace; the rest of the change makes the routes the file already carried reachable. DCU's `silu_backward` and `slice_backward` cuda-boxing fallbacks are unchanged. The 546-overload bw1000 row is **not revalidated**; the generic cohort and its routes are untouched. | Targeted run over the 13 operator files whose FlagGems routes became reachable: `1 failed, 121 passed, 14 skipped, 52 deselected in 49.21s`. The failure is `test_mm_half_hgemm_strict`, which asserts `returncode == 0` on a child process that printed its dispatch line and then died with signal 11 without running a test — the same host fault as the bw1000 raw-case note above, reproducible with `python -c "import torch_fl, torch"` and no operator. No operator produced a wrong result in that run, so none was demoted for one — the route demotions above are contract reasons, each traced to its own failing CI group. Evidence gap: `flaggems_overload_survey.py::active_routes()` recognises only the literal `flagos_python` backend and cannot enumerate this conf's `= flaggems` routes, so the standard survey cannot reproduce this cohort even on this hardware. Provisioning verified end to end on the CI runner by run 34931465738 (job 104260457173): `Successfully installed flagtree-0.6.2a1+hcu3.6`, `Successfully installed flag_gems-5.4.0rc2.post1+ge7b4a865f`, then `Triton: 3.6.0 (backends: ['hcu'])` and `FlagGems: 5.4.0rc2.post1+ge7b4a865f (vendor: hygon)` from the setup script's resolved-stack assertion, with `FLAGOS_USE_FLAGGEMS=1` in every group's environment and the device-availability group passing on 8 devices. That run's unit group stopped at 26/27 on the pre-existing `gen_vendor_confs.py --check` drift for Ascend and GCU, so the FlagGems group itself still did not execute; the manifest was reordered to run the unit group last so that the DCU hardware groups are no longer gated on it. The next run (34935660930, job 104272971051, head `a89e869`) is the first whose FlagGems group executes, and it passes: `13 passed, 1 skipped, 1109 deselected, 1 xpassed in 367.26s`, with the vendor-backend (140 passed/2 skipped/1 xpassed), low-precision matrix (30 passed), RNG (114 passed/1 skipped/1 xpassed), general (46 passed) and AMP (25 passed/1 skipped/1 xpassed) groups passing too. That run stops at the math-bits group on `_conj`, which is the reroute above. Run 34939743596 (job 104285567077, head `d81d5b5`) is the first at a revision carrying that reroute: `[8/11]` math-bits is green (`12 passed in 2.17s`), `[9/11]` profiler contract executes for the first time and passes (`9 passed, 2 skipped, 1 xpassed in 10.55s`), and the run stops at `[10/11]` profiler parity (`1 failed, 5 passed, 1 xpassed in 8.69s`) on the demangling guard, which is the `relu` reroute above. Run 34946627774 (job 104307731540, head `076ab48`) is the first at a revision carrying the `relu` reroute and the first at any revision to execute all eleven groups: `[10/11]` profiler parity goes green (`6 passed, 1 xpassed, 1 warning in 8.03s`), confirming the reroute's A/B prediction (`1 failed, 5 passed, 1 xpassed` with `relu` on FlagGems, `6 passed, 1 xpassed` with it on cuda) on CI, and `[11/11]` unit tests execute for the first time, at 26 of 27 files. Every DCU-measured group is green at that head; the job exits 1 only on `tests/unit/test_gen_vendor_confs.py`, whose single failure is conf drift on Ascend and GCU -- two platforms this change does not edit; upstream cured the GCU half in #285 and the Ascend half in #288, so at the head of this change `gen_vendor_confs.py --check` exits 0 and that file is green. |
| 2026-09-15 | MTT S5000 (8 devices) | MUSA integer division (issue #266) | Fixed two integer-division defects in the generator, not with handwritten kernels. `int64 / int64` raised `Unsupported binary mode: TRUEDIV, with left data type: INT64` because the generated kernels took `result_dtype` from `at::result_type` (int64) while ATen promotes integer true division to float32; `_TRUEDIV_INT_TO_FLOAT` now widens integral results, guarded on `!rounding_mode.has_value()` so `'floor'`/`'trunc'` keep int64. Integer `//`, `floor_divide`, and `floor_divide_` silently lost the trailing element on non-power-of-two `numel` in FlagGems; new `binary_mode` / `binary_inplace_mode` categories plus the `floor_divide_.Tensor` native entry route `div.Tensor_mode`, `div_.Tensor_mode`, `floor_divide` and `floor_divide_.Tensor` through mudnn `FLOORDIV`/`TRUNCATEDIV`/`TRUEDIV` via `SetMudnnDivMode`. MUSA `flaggems` 468 -> 464, `musa` 47 -> 51, `none` 1521; registered-op set unchanged at 518 (three overloads moved from the FlagGems registration to the native one). FlagGems is not patched. Other platforms are **not revalidated** and no FlagGems route changed for them. | 59-case CPU-parity probe on `flagos:0` run against both this tree and a base-commit worktree (out-of-place, in-place, scalar and tensor operands, both rounding modes, negatives, `out=`, broadcasting, and `floor_divide` at `n = 2,3,4,5,7,8,15,17,33,100`): 39 exact / 7 float-approximate / 13 mismatches before, 43 exact / 14 float-approximate / 1 error-text match / 1 probe-harness mismatch after. Integer floor division and every `rounding_mode` case exact; the float-approximate cases are true division one float32 ULP from CPU and reproduce identically on pure-float inputs on the base tree (pre-existing mudnn `TRUEDIV` arithmetic, not this change). `FLAGOS_LOG=dispatch` shows all five overloads on `-> musa`; pinning the four rerouted overloads back onto FlagGems via `FLAGOS_OP_*` reproduces the tail loss (`[5, 5, 0]` for `[5, 5, 6]` at n=3) and leaves true division correct, isolating the routing fix causally. Full `.github/configs/musa.yml` run locally: dispatch 104 passed/1 skipped, factory 46 passed, AMP 27 passed, math-bits 12 passed, profiler 10 passed/1 skipped/1 xpassed, operator cohort 493 passed/1 skipped/513 deselected/2 xfailed/1 xpassed, RNG 80 passed/37 deselected. Generator idempotent (two runs byte-identical; `codegen_musa_flaggems.py --check` and `gen_vendor_confs.py --check` clean for MUSA). `flaggems_overload_survey.py` cannot measure these routes: it selects `flagos_python` entries, and the rerouted overloads are exactly the ones that left that route — evidence gap recorded in the section above. Three pre-existing `test_flaggems_conf_consistency.py` failures (`mm`/`bmm`/`addmm` dispatcher drift) reproduce byte-identically against the pristine conf. |
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
