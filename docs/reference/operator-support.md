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

### DCU: FlagGems launch, layout and autotune costs in the Qwen-Image-2.1 denoise loop (2026-09-19, Hygon DCU bw1000)

Four changes to the DCU FlagGems path, none of which moves an operator between
routes. `torch_fl/configs/backends_dcu.conf` is **not modified** by this work
and still reads **458 `flaggems` / 1579 `cuda`** over a **2037**-entry list
(SHA-256 `7b82cb492de80f3dc2dcb93deeba14764a986368460a63cae9d1deb90fee9163`;
active route-set SHA-256
`ededd42387eef3b74473eef358515a1848c153b3d83896c13168676332c3f69c`, the SHA-256
of the newline-joined active route list). Each change makes a route the conf
already hands to FlagGems cheaper, and each is gated on both
`_build_accelerator() == "dcu"` and the conf actually routing to FlagGems, so on
every other build these functions return before touching anything.

The cohort is therefore the DCU configuration's own active route set (**458**),
a third cohort against the 546-overload generic set and the 639-overload
FlagGems master set. The **Hardware Summary** and **Raw Case Evidence** rows for
bw1000 keep their 546 denominators and are **not revalidated** by this change;
see the re-survey note at the end of this entry.

**1. `index_put_` runs FlagGems' Triton kernel instead of a CPU round-trip.**
`index_put_`/`_index_put_impl_` are registered by hand in
`csrc/aten/register.cc` (`WrapperIndexPut_`) as `self.cpu() -> CPU index_put_ ->
self.copy_(self_cpu)`, and they sit in the codegen's `MANUAL_REGISTERED_OPS`
because a `Tensor?[]` index list has no `IValueToPython` conversion, so no
generated kernel can be handed to the bridge. FlagGems ships the op as a Triton
kernel, so the round-trip is a per-call cost with no coverage reason behind it;
it is now installed on the dispatcher for DCU. On the same six-step driver, the
op falls from 15501.54 to 388.79 us per call on the steady four-step block and
16720.84 to 381.17 on the last step, the steady step wall from 1.272 to 1.180 s,
and the last step's exclusive host sum by 0.093 s against 0.098 s for the op
alone — the two agree, so nothing else moved. Against the round-trip on the same
shapes with fresh operands, the wide calls are 68.583 to 0.291 ms
(`[:, i] <- (4096, 4096)` f32), 72.360 to 0.638 ms (`[:, m]` f32) and 43.442 to
0.578 ms (bf16), while the narrow calls are a wash on both sides (0.085-0.180 ms).
All of them agree bitwise with the round-trip (`equal=True`, `max|delta|=0`, the
same for `accumulate=True`), and the rendered image is byte-identical to the
unpatched one (md5 `6fc5c62c451ecf5772f11bb0e1652cfd`, 1875231 bytes). The op is
in no configuration — it is not in the generated registration list that every
conf enumerates — so no conf entry changed and no FlagGems route moved.

**2. L2 `linalg_vector_norm` no longer materialises a transposed copy.**
`flag_gems.ops.vector_norm` sends every partial reduction through
`dim_compress`, which permutes the reduced dim innermost and then calls
`.contiguous()`; on a strided reduction that permute is not the identity and the
tensor is gathered into a fresh buffer a few hundred bytes at a time. On a
`(1, 144, 1, 2048, 2048)` fp32 input (2.42 GB), against the same device's
1339.8 GB/s on a plain contiguous copy of the same buffer:

| Expression | Time | Rate |
|---|---:|---:|
| `dim_compress(x, [1])` | 354869 us | 13.6 GB/s |
| `x.clone()` | 3606 us | 1339.8 GB/s |
| `flag_gems.vector_norm(x, 2, [1])` | 357307 us | — |
| `sqrt(sum_dim(x*x, [1]))` | 5801 us | — |

99.3% of the op is the copy and the `l2_norm_kernel` it was copied for is
~2.4 ms. The one-dim, `ord=2`, fp32/fp64 path is therefore spelled at the aten
level as `sqrt(sum(x*x, dim, keepdim))`, which reduces the strided axis in place;
`sum.dim_IntList` and `sqrt` are both already `flaggems` in the conf, so no
sub-op leaves the FlagGems route. The op runs 36 times in the VAE decode and
not at all in the denoise loop, and there it falls from 139999.20 to 6248.30 us
per call — 5.040 s of that phase's 6.620 s of exclusive host time down to
0.225 s — against the vendor build's 247.26 us per call. The two
spellings agree to a maximum absolute difference of 6.676e-06. Any other `ord`,
a multi-dim or full reduction, an explicit `dtype`, a half input, and every
reduction whose dim is already innermost still go to FlagGems' own
`vector_norm` unchanged.

**3. The FlagGems pointwise launch path stops rebuilding per-call state.** Every
FlagGems elementwise op is a generated wrapper around a Triton kernel launched
through `flag_gems.utils.libentry.LibEntry`, and on a DCU bw1000 the host side
of that, not the kernel, is what a launch costs: in the denoise loop the
FlagGems pointwise ops are 402 ms of the 837 ms of exclusive host time across
four steady steps (`where` 196 ms over 520 calls, `pow` 61, `tanh` 58, `rsqrt`
56, `silu` 31), where the vendor build spends 34.94 us on each `where` against
FlagGems' 376.48. The kernels are not what differs — the FlagGems `where` kernel
is 74.0 us of device time against 62.1 for the vendor's, while the enqueue is
262.4 us against 11.5. What is left is Python that recomputes per launch what
the kernel fixed at construction: `LibEntry.run` re-derives the specialised,
non-specialised and constexpr argument sets and walks the signature twice,
`LibEntry.key` rebuilds two closures per call and re-resolves the vendor's
tensor-specialisation hook per argument, and `_descriptor_cache_key` re-imports
a module once per argument. All three are memoised on values that cannot change
between two launches of the same kernel on the same shapes. Because every
FlagGems launch goes through that same `LibEntry`, the saving is not confined to
the elementwise ops — the routed composites drop with them (`native_layer_norm`
204.31 to 169.88 us per call, `tanh` 226.51 to 212.80, `rsqrt` 216.23 to 202.41,
`silu` 220.45 to 204.89) and the steady block's exclusive sum with them, 0.837 s
to 0.767 s. That is host time: the steady step on this model is device-bound, so
the four-step wall does not move with it (1.179 s to 1.178 s; the vendor build
is at 1.098 s). No op changes route and no kernel, grid or cache key changes.

**4. Triton's persisted autotune cache is enabled.** FlagGems'
`layer_norm_persistent_kernel` carries a three-way `@triton.autotune`
(`num_warps` 4/8/16), and Triton re-benchmarks all three on the first call for
each tuning key in every process. The knob gates the disk cache in both
directions: `triton/runtime/autotuner.py:262` only reaches `check_disk_cache`
when `Autotuner.cache_results` was set at construction from
`knobs.autotuning.cache`, and `check_disk_cache` is also what writes the entry.
Three fresh processes, `torch.native_layer_norm` on `(1, 4096, 4096)` bf16:

| Arm | First call | Profile |
|---|---:|---|
| `TRITON_CACHE_AUTOTUNING=0`, no entry on disk | 454.2 ms | `:236(run) -> :252(benchmark) -> :132(_bench) x3`, 0.452 s |
| cache enabled, still no entry on disk | 462.5 ms | `:236(run) -> :194(check_disk_cache)` 0.460 s `-> :252(benchmark) -> :132(_bench) x3`, 0.446 s |
| cache enabled, the entry the previous arm wrote on disk | 33.6 ms | `:236(run)` 0.032 s `-> :194(check_disk_cache)` 0.014 s, no benchmark |

The disk lookup hashes the Triton build, the backend target, the kernel's cache
key, the cache-invalidating environment variables, the tuning key and the config
list, so a stale entry cannot survive a compiler or configuration change.
Qwen-Image-2.1 issues exactly two tuning keys over a whole run, measured at
0.38 s and 0.47 s, so this is 0.85 s paid once per process. An explicit
`TRITON_CACHE_AUTOTUNING=0` is honoured — the existing value is parsed with
Triton's own `getenv_bool` rather than left to `setdefault`, because assigning
the knob writes the environment variable back.

**Re-survey.** A focused re-survey of the twelve active routes this change
touches — `linalg_vector_norm`, `sum.dim_IntList`, `sqrt`, `where.self`,
`where.self_out`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor`, `tanh`, `rsqrt`,
`silu`, `native_layer_norm`, `mean.dim` — was run with
`tests/manual/flaggems_overload_survey.py` (harness version 6, SHA-256
`7b01c22ce3a94315f1364df242323e9faac27f2585debfb05030670c7c756cc7`) against the
same conf SHA-256 at torch-fl `873f516` (the `main` this change is based on)
with FlagGems `5.4.0rc2.post1+g437ba3938`
and FlagTree `0.6.0+hcu.git46341ffa`: **12 registered, 12 STRICT, 0 BASIC_ONLY, 0
FAILED, 0 UNTESTED**; case-level 63 `PASS` / 21 `INVALID_CASE` / 0 `ERROR` / 0
`WRONG` / 0 `CRASH` / 0 `TIMEOUT` / 0 `VERIFIABLE` over the 84 synthesized
cases, the 21 `INVALID_CASE` entries being inputs ATen itself rejects before
dispatch (`linalg.vector_norm` on an integer or bool tensor, `where` with a
non-bool condition). `index_put_` is in no conf and so is in no survey cohort.

The **full 458-overload DCU re-survey then ran to completion** on the same
hardware, the same conf SHA-256, the same harness version and the same torch-fl
revision, with these four patches applied, over
`torch_fl/configs/backends_dcu.conf` in full:

| Total | STRICT | BASIC_ONLY | FAILED | UNTESTED | Basic executable | Basic rate | Strict rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 458 | 297 | 45 | 39 | 77 | 342 | 74.7% | 64.8% |

`STRICT + BASIC_ONLY + FAILED + UNTESTED = 458` and
`Basic executable = STRICT + BASIC_ONLY = 342`, matching the run's own
`registered 458 / tested 381 / basic_executable 342` totals. Its 3206 synthesized
cases resolve as **1848 `PASS` / 1059 `INVALID_CASE` / 147 `ERROR` /
138 `WRONG` / 14 `CRASH`**, with no `TIMEOUT` and no `VERIFIABLE`. This is the
third cohort the introduction above describes: its denominator is this
configuration's own 458 active routes, not the 546 of the generic FlagGems set,
so it is recorded here rather than folded into the tables above. **All twelve
affected routes are `STRICT`** (`linalg_vector_norm` 5 `PASS` / 2
`INVALID_CASE`, `sum.dim_IntList` 7 `PASS`, `sqrt` 7, `where.self` 1 / 6,
`where.self_out` 1 / 6, `pow.Tensor_Scalar` 7, `pow.Tensor_Tensor` 6 / 1,
`tanh` 7, `rsqrt` 7, `silu` 5 / 2, `native_layer_norm` 5 / 2, `mean.dim` 5 / 2)
and none is among the 39 `FAILED`, so no failure in this cohort is on a route
this change touches.

The 39 `FAILED` routes group by the first error each recorded: nine as a
FlagGems kernel gated on a CUDA tensor — `i0`, `i0.out`, `special_i0e`,
`special_i1`, `special_modified_bessel_k0` and its `.out`, the same for
`special_scaled_modified_bessel_k1`, and `soft_margin_loss`; eleven as a missing
dtype on the device `*_cpu` implementation — `elu`, `elu_`, `elu_backward`,
`gcd_`, `histc`, `_log_softmax_backward_data`, `_softmax_backward_data`, and the
four `special_{chebyshev_polynomial_v,w,shifted_chebyshev_polynomial_u,w}` in
`Half`; three as a Triton `CompilationError` in lowering — `norm.ScalarOpt_dim`,
`randint`, `randint_like`; nine as a schema or synthesizer mismatch —
`scatter.src`, `scatter_.src`, `sum.out`, `upsample_bicubic2d`,
`_pdist_backward`, `norm.Scalar`, `dequantize.self`,
`special_chebyshev_polynomial_u` and `special_hermite_polynomial_h` (both
rejecting the synthesized order argument); four with no error text at all —
`_unique2`, `range`, `unique_dim`, `randperm`; two with a bare `.cpp:6` —
`_batch_norm_no_update`, `native_batch_norm`; and `unique_consecutive` with a
`RecursionError`. They are recorded with the text that raises them rather than
diagnosed further here, and none is on a route this change touches. The 77
`UNTESTED` routes are those where no CPU-valid synthesized case existed under
the seven profiles, which is neither a pass nor a failure.

The per-overload JSON this table is computed from is the run's `--out`
(`/tmp/flaggems-overloads-dcu-20260919.json`), kept as the auditable evidence in
the sense the introduction requires; it is not expanded into a 458-row
inventory here.

The bw1000 rows in the **Hardware Summary** and **Raw Case Evidence** above
nevertheless keep their 546 denominators and remain the `fe2272b5` / `7fb49bad`
baseline cohort: they measure the generic `backends_flaggems.conf` route set,
which this change neither touched nor surveyed, so this run does not revalidate
them and they are left as they were rather than restated against a denominator
they were not measured on.

**Other platforms are not revalidated.** The four functions are no-ops outside a
DCU build whose conf routes to FlagGems, and no conf file other than
`backends_dcu.conf` is touched — and that one is not touched either. Ascend,
GCU, MUSA, MetaX, PPU and CUDA were not measured against this change and their
rows are unchanged.

### Enflame GCU S60 forward SDPA on the vendor flash op, and the broadcast materialisation (2026-09-20)

The entry below left the S60 at **6.0-6.1x** the vendor `torch_gcu` leg with one
term dominating what remained: Qwen-Image-2512's transformer issues **60**
`F.scaled_dot_product_attention` calls per forward pass and PyTorch served every
one of them with the math decomposition. That decomposition runs **on the
device** — there is no `cpu_fallback` line for it and this is not a CPU-fallback
change — but it is the wrong kernel by a factor of forty: **90.71 ms per call**
against **1.39 ms** for the vendor flash op, **11.637 s** of the transformer step
against **7.240 s**. This change takes it. The same commit removes a
materialising `expand().contiguous()` from the generated elementwise kernels,
which is a second, smaller cost on the same step.

**The measurement.** A within-process A/B on one card, over the same 60 calls and
the same operands, with a sentinel planted in the output buffer and counted at the
end of every run (`sentinel-left 0` of 12 638 208, every run):

| Step | Leaf that ran | Calls | Per call |
|---|---|---|---|
| before | `aten::_scaled_dot_product_attention_math` | 60/60 | 5442.9 ms -> **90.71 ms** |
| after, with the drain | `aten::_scaled_dot_product_efficient_attention` | 60/60 | 331.3 ms -> **5.52 ms** |
| after, drains suppressed | same | 60/60 | 83.1 ms -> **1.39 ms** |

**4.397 s of 11.637 s, 37.8 %**, and the two "after" runs are byte-identical in
their outputs (`max|d| 3.125e-02`, `mean|d| 2.743e-03` against `|math|max
5.750e+00`); against an independent float64 reference on the model's own envelope
the kernel reads `max|d| 1.555e-02`, `mean|d| 1.005e-03` against `|ref|max
5.315e+00`, which is the bf16 noise floor. `EXEC_TOPSATEN_CMD`'s pre/post stream
sync costs **248 ms/step (3.4 %)** on this op; it is kept, because the
correctness of every other call in the process rests on the same macro.

That table is the projection. The shipped route was then measured in the real
graph: warm-up 9.51 s, profiled **7.83 s**, summed self time **7.50 s/step over
64 ops**, with `aten::_scaled_dot_product_efficient_attention` at **170.813 ms /
60 calls** — the same rate through the composed route as through the probe — and
`_fused_sdp_choice REACHED -> 2` in the log before every batch.

**The projection, confirmed as a paired end-to-end run.** Two builds of this
tree, one variable apart — whether `_scaled_dot_product_efficient_attention`
reaches the vendor flash op or ATen's math decomposition — against
`torch_gcu + diffusers`, over an 8-step 1024x1024 50-block `--stage full` run.
Every leg takes `QWEN_IMAGE_REAL_ROPE=1` (the vendor leg cannot start without it:
`topsaten` has no complex kernel) and every leg consumes **the same initial
latents, the same prompt embeds and the same seed**: the first leg writes them
with `--save-latents` / `--save-inputs`, the other two read them back with
`--load-latents` / `--load-inputs`. So the two `torch_fl` legs differ in one
kernel and nothing else — no RNG stream, no prompt, no placement. The layout is
the vendor baseline's own, `TOPS_VISIBLE_DEVICES=0,1,2` with encoder and VAE on
the third card and blocks 0..29 / 30..59 on the first two.

| Leg | Leaf that ran | `s/it` at 8 steps | vs `torch_gcu` |
|---|---|---|---|
| before | ATen's math decomposition | **19.33** | 5.07x |
| after | the vendor flash op | **8.48** (8.81 on a repeat) | **2.23x** |
| reference | `torch_gcu + diffusers` | **3.81** | 1.00x |

**Every leg of this table takes `QWEN_IMAGE_REAL_ROPE=1`, and that is the harness
switch rather than the shipped configuration.** It has to be here — the vendor leg
cannot start without it — and it is what makes these three legs a controlled
comparison of one kernel, so the `2.28x` this table reports for the reroute is
unaffected. What it does affect is the absolute `s/it` and the ratio to the
vendor: since the rotation registration below ships, a `flagos` leg with neither
variable set reads **`5.01 s/it`, `1.31x` the vendor**, on the same 8-step
`--stage full` shape and with its own reference leg reading the same `3.81 s/it`
as this table's. Read the `2.23x` on the `after` row, and the `2.31x` it is
qualified with, as readings of the switch rather than of what a user gets.

**19.33 -> 8.48 s/it is 2.28x, and 56.1 % off the denoising loop** — the 10.9
s/step the projection above predicts, arriving as measured rather than inferred.
The repeat leg is the same build run again and it produces a **byte-identical
image** (`md5 d47974b1…` both times, 8.81 s/it), which is what makes the 8.48 a
reading rather than a lucky one; quoted against the slower of the two the change
is 2.19x and 54.4 %, and the shipped state is then **2.31x** the vendor rather
than the 2.23x the faster leg gives. That the 8-step delta is larger than the
37.8 % the per-op table above predicts is expected and is not a contradiction:
the projection priced the 60 leaf calls of one forward shape, while the loop
delta covers every attention call the step makes plus whatever the math
decomposition drags in behind it — an `aten::_scaled_dot_product_attention_math`
call materialises intermediates the flash leaf never builds, so its cost is not
confined to its own self time. The two numbers measure the same reroute at two
resolutions and the loop number is the one that decides the comparison. Reverting
the route is not a runtime switch and cannot be:
`gen_vendor_confs.route()` gates every route on the PrivateUse1 registration, so
the `m.impl` line in `gcu_register.inc`, the `= gcu` line in the conf and the
hand-written translation unit that carries the `_fused_sdp_choice` registrar all
move together, and the "before" leg is a separate build of this same tree with
the op out of `HANDWRITTEN_OPS`. Everything else in that build — including the
broadcast half below — is identical. Pinning ATen's own selector instead
(`torch.nn.attention.sdpa_kernel([SDPBackend.MATH])`) does not work and was
measured not to: that run produced a **byte-identical image to the shipped leg**
(`md5 d47974b1…`) and still logged **960** `[flagos dispatch]
_scaled_dot_product_efficient_attention -> gcu` lines, because the composite's
route is chosen by torch_fl's `__torch_function__` before ATen's backend pin is
consulted. The conf route and the PrivateUse1 registration also have to agree,
so `FLAGOS_OP__scaled_dot_product_efficient_attention=none` is refused outright
with *"routed to 'none' … but the op is registered on PrivateUse1 — regenerate
the vendor conf so registration and routing agree"*. The rebuild is the only
honest A/B for this op, and it is the one reported above.

**And the image moves toward the vendor, not away from it.** The same three legs,
compared at the end of the eighth step:

| Pair | mean abs delta | PSNR |
|---|---|---|
| shipped vs route reverted | 3.568/255 | 30.10 dB |
| shipped vs `torch_gcu` | 5.089/255 | **26.96 dB** |
| route reverted vs `torch_gcu` | 5.102/255 | 26.91 dB |

The shipped leg agrees with `torch_gcu` **better than the math path it replaced
does** — 26.96 dB against 26.91 dB — so the flash kernel is not a numerical
shortcut away from the reference. The ~27 dB that remains to the vendor is the
rest of the graph (its VAE decode, its elementwise kernels) and not this op: the
math-versus-flash delta the probe measured (`max|d| 3.125e-02`, `mean|d|
2.743e-03` against `|math|max 5.750e+00`) is the bf16 noise floor over one call,
and eight diffusion steps amplify it to a 30 dB image difference, the same
amplification the RoPE A/B below runs into at 28.58 dB over 50 steps.

**Why the math path was in use, and why the stub is the fix.**
`F.scaled_dot_product_attention` is a composite that asks ATen's
`_fused_sdp_choice` **DispatchStub** which backend the device supports, and a
vendor backend claims that by registering the stub's PrivateUse1 slot with
`REGISTER_PRIVATEUSE1_DISPATCH`. Registering a PrivateUse1 *ATen op* of the same
name is not enough, and neither is registering the leaf: the composite consults
the stub, not the op. With the slot empty `is_device_supported(PrivateUse1)` is
false and the composite takes the math branch however many PrivateUse1 kernels
exist for it. Measured, holding everything else fixed:

| `_fused_sdp_choice_stub` slot | stub's answer | `_scaled_dot_product_efficient_attention` leaf fires |
|---|---|---|
| filled | 2 | **1** per call — `_fused_sdp_choice REACHED -> 2` in the log |
| filled | 0 | **0** per call, and the output stays correct |
| empty | (the registered op answers 2) | **0** per call |

The middle row is what makes the change safe rather than merely faster: the
stub's return value **is** the backend selection, so it doubles as a decline
mechanism. Every call this kernel cannot serve — a mask, `is_causal`, GQA, fp16,
`q_len != kv_len`, anything under `torch.enable_grad()` where the backward leaf is
a FlagGems kernel this forward was never validated against — returns `math` and
keeps exactly today's behaviour, with no host round trip and no CPU fallback. The
predicate behind the stub's answer and the kernel's own `TORCH_CHECK` is one
function, so the backend the composite is told about is always one that can serve
the call.

**Why not FlagGems.** FlagGems does carry an SDPA route, and it was measured at
this shape rather than assumed: **133.84 ms per call** against the math path's
90.54 ms in the same run. It is slower than what it would replace, so the vendor
op is the route taken, and `_scaled_dot_product_efficient_attention_backward`
stays where it is, on FlagGems.

**A hand-written translation unit, and the limitation that requires it.**
CLAUDE.md requires a handwritten kernel to be justified by a concrete codegen
limitation and to receive explicit human approval before it is added; both are
satisfied here, and the approval was obtained in session. `codegen_gcu.py` cannot
express two things this op needs:

1. a `REGISTER_PRIVATEUSE1_DISPATCH` registrar into an **ATen DispatchStub** —
   every one of the generator's ~50 category templates ends in
   `REGISTER_IMPL_TO_DISPATCHER(..., {disp}, Backend::kGcu, {kernel})`, which
   registers into the flagos dispatcher table instead;
2. a kernel **body returning a 4-tuple** — every template emits a body whose
   return type is a single `at::Tensor`.

Everything else stays generated. The `m.impl` registration and the `= gcu` route
come from listing the op in the generator's `HANDWRITTEN_OPS`, the same mechanism
that wires `rng.cc` and `native_dropout`. The kernel is
`csrc/aten/backends/gcu/scaled_dot_product_attention.cc`; the full record,
including the vendor header's own contradictory shape documentation and the
measured layout envelope, is in `docs/vendors/gcu/scaled-dot-product-attention.md`.

**Route delta.** Exactly one route moves: `_scaled_dot_product_efficient_attention`
`none` -> `gcu`. GCU `gcu` **177 -> 178**, `none` **1606 -> 1605**, `flaggems`
**254** unchanged, over the same **2037** routable ops, so accelerated routes go
**431 -> 432** (**21.2 %**). The shipped conf is **1605 `none` / 254 `flaggems` /
178 `gcu`**, and both generated registration files reconcile against it:
`254 = 247 + 7` (`gcu_flaggems_register.inc` carries 247 `m.impl` lines and the
conf marks 7 further lines `# gcu`) and `178 = 185 - 7` (`gcu_register.inc`
carries 185). The seven markers are unchanged — `clamp`, `fmod.Tensor`, `gelu`,
`mean`, `mean.dim`, `remainder.Tensor`, `silu` — and there are no orphans and no
overlaps in either direction. The backward sibling does **not** move: it is
registered by `gcu_flaggems_register.inc` and stays on FlagGems, which is also why
the stub declines grad-mode calls.

| Artifact | Before | After |
|---|---|---|
| `torch_fl/configs/backends_gcu.conf` | `476825dba7f3050f4d91590bce29362c50f7d23df6f8c952fba1417b51afc05f` | `28f4656c30b39b7a60128cf581426f968c077aa5895d23d6193c642799a4ef1b` |
| `gcu_kernels.cc` | `57e6edc95b6ec6cbd0f20c19d17214bf5f6283aa6562f59744622d0533efacc1` | `f3cb0ee87ac91bd6ef1952c73152cb4b50ec614998cfbc3566d6af3b8dee518e` |
| `gcu_register.inc` | `dc88d5bca113669315be122703f791bc6f1624918324d55fdb99138d2c763b0e` | `dcab7a4e87bcb63a3273d63be5cd5ba5e57edfb6ab46164ff06024ac981bdd0c` |
| `gcu_flaggems_register.inc` | `be8431800f21fab5038633e4dc79baa84dc317ca7aa9425f05607233b6e88364` | *(unchanged)* |

Generator idempotency: two runs leave all four byte-identical (`md5sum -c` ->
`OK` on the three that move). `gcu_register.inc` gains exactly one line —
`m.impl("_scaled_dot_product_efficient_attention",
WrapperPrivScaledDotProductEfficientAttention);` between `_sample_dirichlet` and
`_softmax`. The hand-written `.cc` adds itself to the build with no CMake edit:
`csrc/CMakeLists.txt` globs the vendor backend directories, and the build log
shows `Building CXX object
csrc/CMakeFiles/torch_fl.dir/aten/backends/gcu/scaled_dot_product_attention.cc.o`.

**The broadcast materialisation.** The generated elementwise kernels brought
operands to a common shape with `tensor.expand(shape).contiguous()` on the
premise that "topsaten does not broadcast for us". The vendor disagrees in its own
documentation — "if lhs or rhs needs to broadcast, it will be processed in this
step" (`/opt/tops/include/gcu/topsaten/topsaten_ops.h:491`) — and the
materialisation is not free: `.contiguous()` on an expanded view allocates and
fills the whole common shape, so `x * 2.0`, whose operand is a 0-dim tensor, pays
an allocator round trip plus a full elementwise copy before the multiply starts.
Measured on an S60 at `(4096, 2560)` bf16, per call: `a * a` **0.212 ms**, `a *
0-dim device` **0.322 ms**, `a * 0-dim host` **0.416 ms**, while `.contiguous()`
on an already-contiguous tensor is **0.004 ms**. A `gcu::BroadcastTo` helper in
`topsaten_common.h` now hands over the view and only materialises a
non-contiguous operand, which costs nothing for the operands these kernels
actually receive after their `.to(device, dtype)` prologues. Handing over the view
is bounded, not assumed: a probe at the rank-4 attention shapes checked every
spelling of interest — all-zero strides, the mixed `(0,0,1,0)` per-token form, a
rank-1 `(1)` with stride 0, a rank-2 and a rank-3 operand against a rank-4 one,
and a small base buffer such a view really points into — against the same op run
on the materialised operand, and found bit-identical output in fp32 and in bf16
with no form refused. This half is not partitioned in the end-to-end table above
and deliberately so: **both** the "before" and the "after" legs carry it, so the
2.19x is attributable to the SDPA reroute alone, and the broadcast half is left
with its per-call evidence (`a * 0-dim device` 0.322 ms against `a * a` 0.212 ms,
the 0.110 ms saved per broadcast operand) rather than an end-to-end number it
cannot be given without a third build.

**What is left, and why it is not attention.** With the leaf at 170.8 ms of a
7.50 s step, the remaining gap to the vendor leg is host-side, and it is the next
workstream. The same harness on the two legs, profiler self time in ms with call
counts in parentheses:

| Op | `torch_fl` (shipped) | `torch_gcu` |
|---|---|---|
| step | 7.83 s | 2.60 s |
| `topsStreamSynchronize` | 1873.2 (13980) | — |
| `aten::_to_copy` | 1822.7 (2995) | 3.1 (1304) |
| `topsMemcpy` | 1256.9 (1208) | — |
| `aten::contiguous` | 454.3 (846) | — |
| attention | 170.8 (60) | 2.4 (60) |
| `GCU::_copy_from` | — | 53.6 (1783), ~2 us/call |
| `aten::addmm` | 516.3 (846) | 34.0 (846) |
| `aten::mul` | 408.7 (1446) | 38.5 (1446) |

Two caveats, and they matter for how far the table can be pushed. The profiler
serialises and inflates runtime calls (the same step is 7.24 s unprofiled against
7.83 s profiled), and the two flagos legs ran on cards 0/3/6 while the `torch_gcu`
leg ran on 0/1/2, so the columns are not a controlled comparison — the
within-process A/B above is. A same-card, profiler-off pair is the first step of
that census.

**The first host-side candidate is the rotary embedding, and it is a third of the
step.** The largest single bucket in the `torch_fl` column is the 240
`aten::_to_copy` calls on `(1, 4096, 24, 64)` — 1848.0 ms of the second session's
7.77 s profiled step — and 240 is exactly four rotary applications per block over
60 blocks. The call chain runs to diffusers' **complex** branch of
`apply_rotary_emb_qwen`, which is what a device type outside `ROPE_PER_DEVICE`
gets; the table lists `cuda` and `neuron` only, and `flagos` is neither. Timed at
the call site, one transformer forward, transformer on `flagos:1`, 1024x1024 (4096
image tokens and 18 text tokens, 24 heads of head_dim 128):

| | complex rotation (default) | real rotation (`QWEN_IMAGE_REAL_ROPE=1`) |
|---|---|---|
| forward | 6.629 / 6.639 s | 4.236 / 4.278 s |
| rotary calls per forward | 240 | 240 |
| rotary total | 3812.6 / 3852.8 ms | 1441.4 / 1469.6 ms |
| — mean | 15.886 / 16.053 ms | 6.006 / 6.123 ms |
| — share of the forward | **57.5 / 58.0 %** | 34.0 / 34.3 % |
| `freqs` dtype at the call | `complex64` | `float32` |

The switch is the repository's own, already documented in
`tests/manual/qwen_image_2512/README.md` §3.3, and it is off by default. As a
switch it reroutes nothing: `FLAGOS_LOG=dispatch` prints
`mul.Tensor -> gcu` for the complex multiply exactly as before, and
`FLAGOS_LOG=fallback` prints nothing at all for it, so both branches stay on the
accelerator and the route delta above is unaffected. What the measurement does do
is price the branch the census had only seen the shadow of. Per call on the
model's shape, ten calls each, minimum reported, operands 50.3 MB unless stated:

| Op | ms/call |
|---|---|
| `complex64` mul, `(1,4096,24,64)` x itself | **50.166** |
| `complex64` mul, `(1,4096,24,64)` x `(4096,1,64)` | 36.412 |
| `complex64` `clone` — the same bytes, copied | **0.547** |
| `float32` mul, `(1,4096,24,128)` x itself | 0.495 |
| `out_real = ar*wr - ai*wi` | 1.557 |
| `out_imag = ar*wi + ai*wr` | 1.571 |
| `torch.angle(complex64)` | 134.091 |

So the complex multiply is **~91x the cost of copying its own operands** and
**~16x the same rotation written as two real multiplies**, which is what the
angle path computes. That is a property of this build's complex kernel rather
than of the rotation, and it is worth a separate report: a `complex64` multiply
that is 91x a `complex64` copy of identical size is not bandwidth-bound, not a
host round trip, and not a wrong route. `torch.angle`'s 134 ms is amortised — it
runs once per forward and all 240 rotations reuse it — but only because
`_get_device_freqs` is the cached one; a wrapper that dropped its `lru_cache`
would put that cost on every call and measure the host instead of the kernel.

**End to end, on 50 steps, the branch is worth 38 % of the run and it is the same
rotation.** Both legs of a 1024x1024 `--stage full`, 50-step, true-CFG run,
`QWEN_IMAGE_REAL_ROPE=0` and `=1`, one variable apart, back to back on
`TOPS_VISIBLE_DEVICES=0,1,2` with encoder and VAE on the third card and blocks
0..29 / 30..59 on the first two, i.e. the vendor baseline's own layout: **13.51
s/it against 8.38 s/it**, 11:15 against 06:58 for the loop — 38.0 % off the whole
run, the same order as the 36 % the wrapped forward measured, so the wrapper was
not manufacturing it. That is 3.37x and 2.09x the vendor's `4.01 s/it`. The two
outputs are not bit-identical, and the question that leaves is answered directly
rather than through a 50-step rollout: on the model's own frequency table
(`QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56])`, `pos_freqs` `(4096, 64)`
`complex64`) and the model's own shape `(1, 4096, 24, 128)` bf16 on `flagos:0`,
both spellings measure `max|d| 1.561453e-02` and `mean|d| 1.121671e-03` against a
float64 reference of `(xr + i.xi)(cos + i.sin)` — identical to six digits, so
neither is the more accurate one — and against each other they agree **bit for
bit on 12,582,343 of 12,582,912 elements (99.9955 %)**, differ by at most one bf16
ulp (`1.5625e-02` at magnitude ~5), and have `mean|d| 3.2e-08`. The switch is a
different spelling of the same rotation, not a numerical shortcut, and what the
two images differ by (28.58 dB over 50 steps against 33.86 dB over 8 for the same
pair) is the rollout's amplification of a one-ulp residue rather than the
arithmetic.

Two consequences for the table above. Its `aten::mul` row at 408.7 ms (1446 calls)
is *not* where the rotation lives: `torch.profiler` attributes a GCU kernel's wait
to `topsStreamSynchronize`, so the same complex multiply reports 2.6 ms a call of
CPU time while costing 15.9 ms of wall — which is why 1873.2 ms of
`topsStreamSynchronize` and much of the 1256.9 ms of `topsMemcpy` belong to this
one branch as well. And the census below should be read with the switch's state
recorded: the two columns are not the same run configuration unless both legs took
it, and the vendor leg needs it too (`README.md` §3.3), so it is the leg with the
switch on that the vendor's 4.01 s/it should be compared against.

**On the leg that ships, the same rotation is worth 6.80 s per step.** A step of
`--stage full` is not a forward: `stage_full` calls the pipeline with
`true_cfg_scale=4.0` whenever a negative prompt is present, so a step is **two**
transformer forwards under true CFG. Three legs, one session, back to back on the
same cards with a warm page cache, one environment variable apart, 8 steps at
1024x1024, the harness's default seed:

| Leg | Configuration | `s/it` at 8 steps | stage wall | vs the vendor |
|---|---|---|---|---|
| opt out | `FLAGOS_DISABLE_QWENIMAGE_ROPE=1` | **11.81** | 267.0 s | 3.10x |
| shipped | registration installed at import, nothing set | **5.01** | 212.4 s | **1.31x** |
| reference | `torch_gcu` + `diffusers` | **3.81** | 197.0 s | 1.00x |

**6.80 s per step, 57.6 % of the loop**, or 3.40 s per forward, against the
2.956 s the three-leg isolated-forward table above measures for the same change.
The step-level gain is larger than twice the isolated one, which is the direction
the `cat` row of the Qwen-Image copy-path workstream reports for a null result of
its own (`README.md` §5.2): it removes 240 of the 780 copies a step issues and the
step wall does not move, because a drain waits for whatever the device has already
queued rather than paying a fixed cost per operation. `5.01` reproduced as `5.00`
on a separate run in the same session, so the reading is stable to 0.2 %, and the
reference leg reads the same `3.81 s/it` as the paired 8-step run reported above,
which is what makes the two runs comparable.

**This is the reading for the configuration a user gets, and it supersedes the
`8.38 s/it` / `2.09x` and `8.48 s/it` / `2.23x` above.** Both of those legs ran
the harness switch `QWEN_IMAGE_REAL_ROPE=1`, which selects `diffusers`'
`apply_rotary_emb_qwen_neuron` — neither the consumer this change installs nor the
operand it installs beside it — and the table above prices the shipped
registration **0.937 s per forward below that switch** (2.455 s against 3.392 s),
measured in one session with one process per leg. The 50-step
`QWEN_IMAGE_REAL_ROPE=0` / `=1` pair stays a valid controlled comparison of the two
*spellings* it names, and its 38.0 % is the price of the angle path over the
complex one; neither of its legs is the shipped configuration, and both are
slower than it.

A companion run split the step the other way — timing `text-encoder` and `vae`
alone so that `wall(vae) - wall(text-encoder)` would isolate the 1024x1024
decode. That subtraction carries no signal and the arm is best not repeated: both
stages pay the same ~155-165 s pipeline load (164.1 s and 161.2 s on `flagos`,
156.3 s and 153.7 s on the vendor), which is larger than the decode and moves more
between repeats than the decode is worth, so the only timing either arm
contributes is the `s/it` line of its `full` stage — and there is no large
non-transformer term left for a change to attack.

**The rotation now ships installed by the plugin, and its expansion is cheaper
than the one `diffusers` provides for a device without a complex dtype.** The
measurement above was taken through the harness switch, so it priced the rotation
without changing what a user gets: a `flagos` tensor was still taking
`ROPE_PER_DEVICE`'s `cuda` fallback. This change registers the `flagos` device in
that table from production code, in the GCU branch of `torch_fl/__init__.py`, so
the angle path is the default and the complex exponential is not reachable by
accident. Two halves are installed and they have to be, because `diffusers` keys
the rotation on device type in two places: `ROPE_PER_DEVICE` picks the rotation at
the attention call site (`transformer_qwenimage.py:591`), while
`QwenEmbedRope._get_device_freqs` (`:269`, `@lru_cache_unless_export`) and
`QwenEmbedLayer3DRope._get_device_freqs` (`:398`) produce the operand and return
**complex** `pos_freqs`/`neg_freqs` for every device except `neuron`
(`:271-276`). Registering the consumer alone would multiply complex frequencies
with a real-valued kernel; registering the operand alone would feed a complex
operand to a kernel that reads it as angles. The operand half therefore delegates
to the stock method for every other device, and both halves are installed once
under a module flag so a second call is a no-op.

The consumer is the same rotation as `apply_rotary_emb_qwen_neuron` — the entry
`diffusers` provides for a backend with no complex dtype — but it does not expand
each angle into two adjacent features with `repeat_interleave(2, dim=-1)`. torch
has no accelerator kernel for that overload on this backend: `aten`'s
`repeat_interleave.self_int` is a composite that lowers to
`unsqueeze(-1).expand(..., 2).reshape(...)`, and the `reshape` materialises a
stride-0 view through `StridedCopy`, which drains the stream on both sides of
every copy it makes — 2.249 ms per call while those calls are the ones paying the
drain, against 25.596 us once they are not. The consumer instead reads each angle
once into `cos`/`sin` and writes the doubled layout with
`torch.stack([angle, angle], dim=-1).flatten(-2, -1)`. `.flatten` is a view, so
`cos`/`sin` stay contiguous, and no stride-0 broadcast reaches the copy path.

**Three legs, one warm forward each, each in its own process**, on an S60 with the
transformer on `flagos`, 1024x1024 (4096 image tokens and 18 text tokens, 24 heads
of head_dim 128), operands and latents loaded from the same files so the only
difference is the registration. The rope columns come from a second pass with the
resolved entry point wrapped, timed so each call pays its own drain:

| Leg | `ROPE_PER_DEVICE['flagos']` | `freqs` dtype at the call | forward | rope total | — share |
|---|---|---|---|---|---|
| complex (old default) | *absent* — `cuda` fallback | `complex64` | 5.411 s | 3818.7 ms | **70.6 %** |
| neuron (harness `QWEN_IMAGE_REAL_ROPE=1`) | `apply_rotary_emb_qwen_neuron` | `float32` | 3.392 s | 1448.2 ms | 42.7 % |
| shipped | `flagos_qwenimage_rotary_emb` | `float32` | **2.455 s** | **546.8 ms** | **22.3 %** |

240 calls per forward on every leg. Against the fallback this change removes,
**2.956 s per forward, 54.6 %**; against the angle path it replaces,
**0.937 s per forward, 27.6 %** — the harness switch alone was worth 2.019 s of
that, and the expansion is worth the remaining 0.937 s. All three legs produce a
finite output on the same envelope (`absmax 5.3125`).

**And the shipped leg is bit-identical to the neuron leg, end to end.** The three
forwards were saved and compared as full model outputs, `(1, 4096, 64)` bf16:

| Pair | `torch.equal` | max abs delta | elements differing |
|---|---|---|---|
| neuron vs shipped | **True** | `0.000e+00` | **0 / 262144** |
| complex vs shipped | False | `3.125e-02` | 142636 / 262144 |
| complex vs neuron | False | `3.125e-02` | 142636 / 262144 |

The complex path and the real-valued contraction are not the same floating-point
contraction — `apply_rotary_emb_qwen(..., use_real=False)` forms `(a+bi)(c+di)`,
which rounds differently from the two-product sum — so the last two rows are two
bf16 ulps at the output's own magnitude (`2^-6` at `absmax 5.3125`) accumulated
over 60 attention blocks, not a defect in either. The contract this consumer is
held to is the neuron expansion, and it meets it exactly; the same assertion is
pinned in `tests/unit/test_gcu_qwenimage_rope.py`, which checks both spellings
against a pinned copy of that expansion, against the `diffusers` copy actually
installed, and against the complex path at a documented tolerance.

**Route delta: no `backends_gcu.conf` route moves, and none could.** This change
is a `diffusers` registration, not a torch_fl route: it edits no conf file and no
generated registration, so the shipped conf is unchanged at **1605 `none` / 254
`flaggems` / 178 `gcu`** over 2037 routable ops, and the reconciliation above
still holds. What does change is which operators the graph contains. Leaving the
rotation are the complex family the fallback used — the `complex64` `mul`, its
`clone`, and the `(1, 4096, 24, 64)` `_to_copy` pair per rotation — and
`repeat_interleave.self_int`, which was never routed off the accelerator (the
dispatch census over one warm forward on the affected leg logs no `cpu_fallback`
line at all) but whose composite lowering reached the drained copy path 480 times
per forward. Entering are `cos`, `sin`, `stack` and a view-only `flatten`, all of
which take the backend they were already configured for. This is the same category
the DCU `FLAGOS_DCU_SDPA_FLASH` row occupies: a switch on another library's table
rather than a route switch in torch_fl's, and it is why nothing in the
conf/registration reconciliation above moves.

**A switch, because the A/B has to stay available.** `FLAGOS_DISABLE_QWENIMAGE_ROPE=1`
leaves `ROPE_PER_DEVICE` exactly as `diffusers` ships it, which is the `complex`
leg of the first table; `QWEN_IMAGE_REAL_ROPE=1` still provides the `neuron` leg
for every device kind, and the manual harness no longer double-wraps the operand
producer or overwrites the registered consumer. Both are documented in
[environment-variables.md](environment-variables.md).

Provenance: torch-fl `f48a1459`, `torch_fl/accelerator/gcu/_gcu_compat.py` sha256
`7bd3f2d156a0baa7c3d84adffb86a44fcd2ca8d38bceb8efe612c1f67d0cd984`,
`torch_fl/_env.py` (registry row) and
`tests/unit/test_gcu_qwenimage_rope.py` sha256
`c4848b23af688bb73832540519fb06cbc01f8ad74748243ebb1e5863a2756006`; probe
`/tmp/probe_rope_shipped.py`, one process per leg, `TOPS_VISIBLE_DEVICES=0,1,2`,
warm-up 1 and 3 timed forwards, minimum reported. `f48a1459` is the tip these were
measured at; the branch was rebased onto upstream `main` after them, which rewrote
every SHA on it and brought in `#353` — MetaX's `scaled_dot_product_attention`
composite and its docs, on neither file hashed above.

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
is a copy of CUDA's, and BPU's is intentionally empty. The guard cohort and the
whole-file figure are restated in the entry below, which widens the envelope; the
route counts, the conf SHA-256 and the survey verdict above are unchanged by it.

### MetaX: the SDPA route takes Qwen-Image-2.1's key-valid mask (2026-09-19, MetaX C550)

The route above admitted no mask at all, which left Qwen-Image-2.1's prefill
attention on the boxing path. `QwenImage21AttnProcessor.__call__`
(`transformer_qwenimage21.py:478-560`) issues one `dispatch_attention_fn` per
prefix segment and hands every one of them the joint key-valid row as an explicit
`(1,1,1,KV)` bool mask, so an envelope that only knows "no mask" never fires for
the shape it was built for. This change widens the mask clause to exactly that
class and leaves every other clause alone. **No routing configuration changed** —
the clause lives in `csrc/aten/sdp_choice_stub.cc`, so the conf SHA-256 and the
592 / 12 / 1433 counts above still stand.

**The kernel cannot read a bool mask as-is.** FlagGems converts one itself at
`attention.py:928-929` with `attn_mask.to(query.dtype) * -1.0e6`, which maps
`True` to `-1e6` — the inverse of torch's convention, where `True` is the key that
attends — and it skips that conversion for a float mask. Measured on this part
against `F.scaled_dot_product_attention` on a materialized `(1,2,1024,1024)` bool
mask, with the same operands throughout: passing the bool through unchanged and
passing `-1e6 where True` both land at `max|d| 5.840e-01`, while `0 where True`
lands at `1.953e-03` — which is exactly the `1.953e-03` that
`F.scaled_dot_product_attention` itself sits from an fp64 CPU reference over the
same rows, so that polarity is the one whose residual is torch's own bf16 floor.
The route therefore builds the additive itself: `where(mask, 0, -1e6)`, in fp32
because the kernel adds it onto an fp32 accumulator.

**The strides are the second load-bearing detail.** The caller's `(1,1,1,KV)`
tensor carries a real stride on both size-1 axes — the routed call below logs
`(4122, 4122, 4122, 1)` — while `_attn_fwd` indexes the mask at
`batch_id*stride(0) + head_id*stride(1) + offs_m*stride(2) + offs_n*stride(3)`
with nothing bounding the first three (`attention.py:240`, `262-272`). An
`expand` alone keeps those strides, because the axes are size 1 either way, so
the route reshapes to `(1,1,1,KV)` first and expands to `(B,H,Q,KV)` afterwards;
the reshape is what leaves a contiguous tensor whose broadcast axes then take
stride 0 (`expanded (4122, 0, 0, 1)` in the probe below). Handing that row
straight to the kernel is measurably wrong: over `q(1,32,4096,128)` /
`kv(1,32,4122,128)` with 2748 of 4122 keys valid, the route's additive lands
`max|d| 9.766e-04` from `F.scaled_dot_product_attention` while the same operands
with no mask at all land `2.441e-01` away, which is the size of the mistake the
stride fix removes.

**Numerics of the conversion, at four mask densities.** `q(1,32,4096,128)` /
`kv(1,32,4122,128)`, 2748, 4096 and 1 valid keys plus all-4122, against
`F.scaled_dot_product_attention` with the bool mask and against an fp64 CPU
reference over head 0:

| Keys valid | route vs `F.sdpa` | no mask vs `F.sdpa` | route vs fp64 CPU | `F.sdpa` vs fp64 CPU |
|---|---:|---:|---:|---:|
| 4122 / 4122 | 9.766e-04 | 9.766e-04 | 4.883e-04 | 4.883e-04 |
| 2748 (mid-block) | 1.953e-03 | 2.441e-01 | 4.883e-04 | 4.883e-04 |
| 4096 (block-aligned) | 9.766e-04 | 1.333e-01 | 4.883e-04 | 4.883e-04 |
| 1 | 0.000e+00 | 4.056e+00 | 0.000e+00 | 0.000e+00 |

The route tracks the fp64 reference exactly as closely as torch's own masked math
path does, at every density including a single surviving key; the last column is
what says the residual is the model's bf16 floor rather than this conversion. The
materialized `(B,H,Q,KV)` form of the same additive agrees with the reshaped view
bit-for-bit (`max|d| 0.000e+00`) in all four.

**Cost of the additive, at the spelling the route uses.** On this part at
KV 4122, submit time: `zeros_like` `0.024 ms`, `full_like` `0.037 ms`,
`where.self` `0.105 ms`, and `0.010 ms` more for the reshape and expand — about
`0.18 ms` per routed call. All three are `flaggems` routes in the conf
(`zeros_like` 2081, `full_like` 1060, `where.self` 2064), and that is checked
rather than assumed: a `FLAGOS_LOG=dispatch` census of the two spellings the
route serves reports

```text
[flagos dispatch] full_like  -> flagos_python
[flagos dispatch] zeros_like -> flagos_python
[flagos dispatch] where.self -> flagos_python
... twice, once for the masked call and once for the unmasked one ...
[flagos dispatch] empty_like -> cuda
```

with no `fallback` line anywhere in the log. The `empty_like -> cuda` lines are
the conf's own value for the allocation (`empty_like = cuda` at line 969) and not
a demotion: FlagGems has no allocator kernel, and this is where the additive's
storage comes from.

**Op level, in the pipeline.** Qwen-Image-2.1 prefill, the shape the clause
admits, `(1,1,1,4122)` mask:

| Arm | ms/call |
|---|---:|
| boxing (`FLAGOS_OP_scaled_dot_product_attention=cuda`) | 40.239 |
| this route, live | 11.808 |

The routed arm's 96 calls at `q(1,4096,32,128)` / `kv(1,4122,32,128)` /
`mask(1,1,1,4122)` total `1133.57 ms` against `1126.15 ms` for
`flag_gems.scaled_dot_product_attention` called directly over the same operands —
`max|d| 0.000e+00`, `rel 0.00e+00` — and the NaN count on q/kv/out/alt is
`0/0/0/0`. (A steps-1 arm of the same harness reports `16777216` on every
tensor; that arm's flow-match schedule is degenerate and does not run attention,
which is why the figure above comes from a three-step run.)

**End to end.** Qwen-Image-2.1, 1024x1024, 40 steps, batch 1, seed 42,
`--warmup 1`, card 0 on every arm, `--min-run-time` 40 s:

| Arm | latency | loop per step |
|---|---:|---:|
| flagos, this route | 31.62 s/image (4 calls) | **776.9 ms** |
| vendor MACA torch | 66.96 s/image (2 calls) | 1661.1 ms |
| flagos, route off (`=cuda`) | 68.66 s/image (2 calls) | 1702.8 ms |

A second window at 50 s reproduces all three (773.1 / 1660.2 / 1687.5 ms), so the
route is **2.13x** the vendor's own torch on this model and **2.19x** the same
wheel with the clause unemployed.

**Paired latents.** The three arms above run their own RNG streams, so their
pixels are not comparable. Re-running all three against one injected latent file
(`--load-latents`) gives `1661.0` / `774.8` / `1687.4 ms per step` and makes the
images comparable:

| Comparison | max | MAE (/255) | PSNR |
|---|---:|---:|---:|
| route vs boxed | 47.0000 | 0.43833 | **49.05 dB** |
| cuda vs boxed | 56.0000 | 0.56115 | 46.58 dB |
| cuda vs route | 62.0000 | 0.55726 | 46.47 dB |

The route's own image cost, `49.05 dB`, is *below* the pre-existing
flagos-vs-vendor difference `46.58 dB` the wheel already carries, and `cuda vs
boxed` reproduces the `46.58 dB` recorded when this model was brought up, which
is the control showing the window resolves what it claims to.

**Guard tests.** `TestMetaXFlaggemsSdpaRoute` grows from eight cases to eleven:
the admitted `masked` is new — the joint shape at `(1,2,1024,128)` carrying the
`(1,1,1,1024)` bool key row 2.1 passes, whose last 128 keys are dropped, so the
mask is live rather than a row of True — the old `mask` case is renamed
`mask_additive` now that there is more than one mask case, and two rejection
cases are added: `mask_4d_bool` (a bool mask with one entry per query-key pair)
and `mask_row_f32` (a per-key row in float32, whose polarity the kernel's own
conversion would invert). `_SDPA_ORACLE`'s host reference now moves a bool mask
as a bool, since the CPU reference reads keep/drop from it. The shape-census test
becomes `test_only_the_admitted_shapes_reach_the_kernel` and expects both admitted
shape classes. On the C550 that class reports **13 passed in 41.19s, 0 failed**,
and the whole file reports **104 passed in 795.16s, 0 failed** (exit 0) with the
route active, against the 101-test cohort the entry above records.

**Survey.** `tests/manual/flaggems_overload_survey.py` (version 6, SHA-256
`7b01c22ce3a94315f1364df242323e9faac27f2585debfb05030670c7c756cc7`) was rerun on
the C550 against the changed route: `registered 1`, `tested 1`, verdict
**`FAILED`** with `strict_support 0` and `basic_executable 0`. The verdict and its
cause are the ones the entry above records — the harness synthesizes `dropout_p`
as `0.5`, and a dropout call is non-deterministic — and they are unchanged by
this clause, because every case the harness synthesizes is outside the new mask
clause as well: none of them passes a `(1,1,1,KV)` bool mask.

**Other platforms.** The clause is in the MetaX-only composite override, so no
other platform's route or conf changes. Ascend, GCU, MUSA, DCU, PPU, CUDA and
TsingMicro are **not revalidated** by this change and their rows are unchanged.

### Enflame GCU S60 Qwen-Image-2512 throughput: five host round-trips and a FlagGems route (2026-09-19)

The paired Qwen-Image-2512 cohort on the S60 measured the `torch_fl` leg at
**301.77-304.81 s/it** against the vendor `torch_gcu` leg's **3.98-4.01 s/it** —
about **76x** per denoise step. This change removes five distinct costs and lands
at **24.19-24.29 s/it**, which is **6.0-6.1x** the vendor rate and therefore
inside the one-order-of-magnitude budget. Four of the five are the same mistake in
different places — work that belongs on the card performed through the host, or a
dtype decided from the wrong side of a promotion — and the fifth is a route
FlagGems' `where.self_out` could not serve. Only one of the five is a routing
change, so this entry is the first here whose subject is throughput rather than
coverage.

**The measurement.** The first two rows are the two legs of one 50-step paired
run; the last three are one script and one fixed latent, so only the tree differs
between them. Every number is the denoise-loop rate, which is the progress bar
`diffusers` draws around the transformer.

| Leg | Run | Denoise step |
|---|---|---|
| vendor `torch_gcu` + `diffusers` | 50-step paired | **3.98-4.01 s/it** (`50/50 [03:18<00:00, 4.01s/it]`) |
| `torch_fl` + `diffusers`, as found | 50-step paired | 301.77-304.81 s/it (`50/50 [4:11:28<00:00, 304.81s/it]`) |
| after the copy path | 8-step fixed latent | 156.78-156.80 s/it |
| after `where.self_out` | 8-step fixed latent | 24.29-24.40 s/it |
| after the wrapped-number, comparison and clamp fixes | 8-step fixed latent | **24.19-24.29 s/it** |

The 8-step runs are `stage: full` — text encoder, transformer and VAE — at the
same `(1, 4096, 64)` latent and the same 50-step schedule cut to 8 steps, and all
three save an image. The vendor row prints both its running and its final rate
(`4.01` and `3.98`), which is the source of the range.

**This table is the state this change left, not the configuration that ships.**
Two later changes in this report carry the same workload further — the
`_scaled_dot_product_efficient_attention` reroute onto the vendor flash op, and
the Qwen-Image rotation registration — and with both in place an 8-step
1024x1024 `--stage full` run reads **`5.01 s/it`, `1.31x` the vendor** (the
rotation section above). The `6.0-6.1x` on the last row is what this change
alone reached from 76x.

**1. Every dtype cast and every strided copy went through the host.** GCU had no
ATen kernel for either, and the no-on-device-path arm of `csrc/aten/copy_ops.cc`
and `csrc/aten/contiguous_ops.cc` is a full round trip: D2H the operand, copy on
the CPU, H2D the result. At the `(1, 24, 4114, 4114)` fp32 attention matrix — the
shape that dominates this forward at 1.5 GiB — that measured **437.9 ms** for
`_to_copy` f32 -> bf16 against the vendor plugin's **5.8 ms**, and **975.6 ms**
for `small.expand(full).contiguous()` against **14.2 ms** for the same 1.5 GiB as
a clone. Both are one vendor call away: `topsatenCopy` takes a strided source and
a broadcastable destination, and `topsatenToCopy` takes an explicit output dtype.
Issued directly from a probe the same two operations cost **5.6 ms** and
**4.6 ms**. `csrc/aten/backends/gcu/gcu_copy.{h,cc}` are that route — two
functions, `StridedCopy` and `DtypeCast`, in the shape Ascend's `ascend_copy.h`
already has — and `_copy_from`, `contiguous`, `clone` and `_to_copy` call them
before their host arm. In the pipeline `_to_copy` f32 -> bf16 across the six
attention layers of a step measures **5.90-5.93 ms**, against the vendor's
5.813 ms and 5.919 ms for the two directions, so the cast is now within 1.3 %.

**2. A full-size host tensor was materialized for every scalar operand.** Every
`binary_scalar_as_tensor` kernel stages its scalar as a full-size device tensor,
because the vendor takes no scalar there, and `ScalarToDeviceTensor` built it with
`at::full(sizes, scalar, options.device(at::kCPU)).to(options.device())`: a
full-size host allocation, a full-size CPU fill, a full-size H2D transfer and a
second full-size device allocation, once per scalar operand. The fill now happens
on the device. At `(1, 24, 4114, 4114)` fp32 the host fill alone is **265.2 ms**
against **3.7 ms** for the device one, and `a + 1.0` through it is **630.1 ms**
against **17.2 ms** at the `(1, 4096, 3072)` bf16 activation shape — where the
fill is **4.6 ms** against **0.3 ms**. A forward pays this once per
`x + 1.0`-style expression in the model, not once per attention layer.

**3. Every binary op with a Python-number operand fell to the host, silently.**
This is the largest single finding and it was invisible: `x * 2.0` dispatches to
`aten.mul.Tensor`, not `mul.Scalar`, with a **wrapped-number** second operand —
a real 0-dim tensor carrying the number's own dtype, f64 for a float and i64 for
an int, flagged `is_wrapped_number`, which promotion deliberately stops from
widening `self`. The generated guard tested `TopsatenSupportsDtype(other.scalar_type())`,
so it read f64 for the most ordinary spelling there is, rejected the operand, and
its fallback moved the whole tensor to the CPU and back — with no `cpu_fallback`
line to show for it, because the fallback is inside the kernel. Measured at
`(1, 24, 4114, 4114)` fp32:

| Spelling | Before | After |
|---|---|---|
| `a * 2.0` (Python float) | 575.474 ms, 5.3 GiB/s | **21.131 ms** |
| `a * 2` (Python int) | 615.780 ms, 4.9 GiB/s | **21.106 ms** |
| `a + 1.0` (Python float) | 630.091 ms | **21.133 ms** |
| `a * <f32 0-dim device tensor>` (the reference) | 19.993 ms, 151.4 GiB/s | 20.699 ms |
| `a.cpu().mul(2.0).to(dev)` (what the fallback ran) | 616.449 ms | — |

`torch.profiler` on the before-state shows the shape of it exactly: on `a * 2.0`
two `topsMemcpy` of 219.6 ms each and **zero** `topsLaunchKernel`. `at::result_type`
is wrapped-number aware (`ResultTypeState::wrappedResult`) and reports f32 for
both `mul(f32, wrapped_f64)` and `mul(f32, wrapped_i64)`, which is the type the
arithmetic is actually done in, so the guard now reads it through a shared
`_binary_dtype_guard` helper that keeps `TopsatenSupportsDtype(self.scalar_type())`
as well — `self` is read as well as written, so an i64 `self` still has to fall
back even when its promotion lands on f32. A genuine f64 operand, wrapped or
not, still promotes to f64 and still falls back, which is what the CPU does too:
topsaten has no float64 kernels. This fix is why the third and fourth rows of the
step-time table are the same number as the fifth row's predecessor — the e2e cost
of a 27x per-op win was below the run's own noise, and the correctness evidence
below is what establishes it.

**4. A fractional scalar bound was truncated on the comparison and clamp paths.**
A different bug with the same root. `x < 0.5` on an integral tensor dispatches to
`aten.lt.Scalar`, whose template guarded on `at::result_type` — correctly — and
then converted the scalar with `ToTopsatenScalar(other, self.scalar_type())`,
which truncates `0.5` to `0`; so `int32 < 0.5` answered `False` for every value
in `[0, 1)`. `lt` and `le` now cast `self` to the promoted dtype and convert the
scalar into that, so `int32 < 0.5` compares `0.0 < 0.5`. `clamp` had the same
shape of error on its absent bound: it filled an absent limit with the *dtype's*
extreme, and a wider constant truncates through `topsatenScalar_t`'s int64 member
— INT64_MAX handed to an int32 clamp arrives as -1 (its low 32 bits), which turns
`clamp(min=0)` on an int32 tensor into a clamp to -1. `gcu::IntegralExtreme` now
supplies the extreme at the tensor's own width, and `gcu::ClampComputeDtype`
promotes the bounds into the result the way ATen's clamp meta does, so
`int32.clamp_min(0.5)` answers f32 while `int32.clamp_min(0)` stays int32.
`TopsatenScalar_t` is a plain `{dtype, union{double fval; int64_t ival;}}`, which
is why every one of these has to be decided before the wrapper, not inside it.

**5. `where.self_out` was on FlagGems, and it is the last op of `_safe_softmax`.**
ATen's eager `_safe_softmax` — which the SDPA math path runs once per attention
layer on this platform — is a `CompositeExplicitAutograd` C++ kernel whose last op
is exactly this overload, so the whole op inherited FlagGems' cost here. Measured
at `(1, 24, 4114, 4114)` fp32 with a `(1, 24, 4114, 1)` bool condition, drained
on both sides of every call: **1098.6 ms** through `flag_gems.where_self_out` with
a 0-dim value operand, **69.8 ms** for the same call with a full-size one, and
**23.7 ms** for the `gcu` `where.self` kernel. The 0-dim operand is what falls off
the cliff, not its device: 1105.2 ms with the scalar on the host against 1098.6 ms
with it on the card. Summed over the composite, one `_safe_softmax` costs
**1135 ms** through FlagGems and **37.8 ms** once this op is taken out of it, and
at that shape the op costs the same alone or after the other four — so this is the
whole of the gap, not an amplifier of it. Confirmed on the full composite:

| Spelling of `_safe_softmax`'s body | Before | After |
|---|---|---|
| `SIAW` (softmax + in-place add + `where.self_out`) | 1130.40 ms | **61.96 ms** |
| `SW` | 1101.19 ms | **32.00 ms** |
| `IW` | 1120.22 ms | **51.28 ms** |
| `AW` | 1095.36 ms | **27.09 ms** |
| `_safe_softmax` itself | 1131.07 ms | **61.85 ms** |
| `SIA` (no `where`) | 37.74 ms | 37.77 ms |

The row that does not move is the answer: `SIA` is the same composite with the
`where` removed and it is 37.7 ms both before and after. The kernel itself is the
out-of-place `where.self` template plus the out= contract, and it carries forward
that template's two ordering decisions — it expands all three operands including
the condition, because ATen broadcasts the condition against the values' common
shape rather than against the condition's own, and it sizes `out` before the
empty early-return. Two things differ from `T_BINARY_OUT` on purpose. It tests
`out`'s device *before* resizing, because `resize_` takes no device argument and
allocates out of the *current* device's pool, so a mismatched `out` would be grown
out of the wrong card's pool while still reporting its own. And its device guard
is installed on `out` rather than on `self`: `out` is both the tensor being grown
and the one naming the device the work belongs on, while `self` is a host scalar
for the composite this kernel exists for. That last point produced a change to the
out-of-place template too — `where.self` took `self.device()` as its compute
device, which for a 0-dim host `self` **silently** selects the host and hands the
vendor device pointers, a `RUNTIME_ERROR` out of `topsatenWhere` at best. Both
templates now resolve the device through `gcu::TopsatenComputeDevice`, which
returns the first operand that is actually on a card and `kCPU` only when none is.

**Route delta.** Exactly one route moves: `where.self_out` `flaggems` -> `gcu`.
GCU `flaggems` **255 -> 254**, `gcu` **176 -> 177**, `none` **1606** unchanged,
over the same **2037** routable ops, so accelerated routes stay **431**
(**21.2 %**) — this change buys throughput and not coverage. The shipped conf
is **1606 `none` / 254 `flaggems` / 177 `gcu`** over 2081 file lines, and both
generated registration files reconcile against it: `254 = 247 + 7` (`gcu_flaggems_register.inc`
carries 247 `m.impl` lines and the conf marks 7 further lines `# gcu`, so that
these ops are routed to a native kernel the FlagGems file defers to) and
`177 = 184 - 7` (`gcu_register.inc` carries 184). The seven markers are the same
seven as before — `clamp`, `fmod.Tensor`, `gelu`, `mean`, `mean.dim`,
`remainder.Tensor`, `silu` — and there are no orphans and no overlaps in either
direction. `where.self_out` also leaves `gcu_flaggems_register.inc` for
`gcu_register.inc`, which moves that file's provenance banner from "248 ops
registered here, 100 further FlagGems ops already claimed by `gcu_register.inc`"
to **247 and 101**. Its `NATIVE_TRITON_GAPS["gcu"]` entry is added, recording the
1098.6 ms against 23.7 ms measurement and "~1.1 s/layer"; the route and the gap
entry are two halves of one claim, and a gap entry without the route would leave
the op on FlagGems.

| Artifact | Before | After |
|---|---|---|
| `torch_fl/configs/backends_gcu.conf` | `4e8265afbf12b472097588ea1f7ae67f627b37711f2e3c17200f190588e0ea40` | `476825dba7f3050f4d91590bce29362c50f7d23df6f8c952fba1417b51afc05f` |
| `gcu_kernels.cc` | `1d7d4f9b6f2830fb67a78bfeeb48582fa59a4e369287669a3486ad92745ef843` | `57e6edc95b6ec6cbd0f20c19d17214bf5f6283aa6562f59744622d0533efacc1` |
| `gcu_register.inc` | `638619ee4b080a0839ad225f2b82d56684dcee219d95f903899014e5c3d5f954` | `dc88d5bca113669315be122703f791bc6f1624918324d55fdb99138d2c763b0e` |
| `gcu_flaggems_register.inc` | `c520a93a3ab23a196d836f349547c6535ba6a84b0a1490cf04f5c2309b7f210d` | `be8431800f21fab5038633e4dc79baa84dc317ca7aa9425f05607233b6e88364` |

Generator idempotency: two runs leave all four byte-identical (`IDEMPOTENT=YES`).
`ruff check .` — "All checks passed!"; `ruff format --check .` — 272 files already
formatted. The generator change, `gcu_copy.{h,cc}` and the helpers added to
`csrc/aten/backends/gcu/topsaten_common.h` are not hashed: they are
version-controlled sources rather than generated output, so the diff is the record.

**Bit-exactness, and why the image changes.** The wrapped-number, comparison and
clamp fixes move real arithmetic from the CPU onto the card, and the two do not
agree bit for bit. A probe over 45 spellings finds **41 non-bit-exact** ones,
every one of them at **1 ulp**, and they are not a rounding-mode curiosity: the
device is the more accurate side on `a / 3.0`, `a / 7.0` and `a ** 0.5` at f32
(327-560 of 1000 elements differ) because it keeps the division in higher
precision, while the CPU is the more accurate side on `* 0.1` and `* 1.1` at bf16
and f16. The spellings that matter for this pipeline — `* 2.0`, `* 2`, `+ 1.0`,
`- 1.0`, `/ 2.0`, `** 2.0`, and `/ 3.0` in the sigma schedule — are **exact**
(0 of 8192 and 0 of 262144 elements differ). Consequence: the copy and
`where.self_out` fixes are bit-identical to their predecessors, and the 8-step
fixed-latent image at `ff69df70…` is unchanged through both, while the scalar
fixes move it to `526f05c1…` — 26.16 % of bytes differing, mean |delta| 0.441/255,
max 40/255, against a 0.441 global mean shift. That is the signature of a
chaotic system amplifying 1-ulp differences across 8 steps and not of a defect,
and it is a change toward the vendor's own arithmetic rather than away from it:
the vendor leg computes these on the card as well. It is recorded here because it
is a real behavioural change a reviewer should see, not because it is a
regression.

**Verification.** Five probes, all against the same ATen call on the CPU, all on
card 6 (card 5 is defective and hangs any `topsaten` op):

- `/tmp/guard_verify.py` — **35 cases, 0 failures** — the wrapped-number fix over
  f32/f16/bf16/i32/i64 cross products of `* 2.0`, `* 2`, `+ 1.0`, `== 2`, `< 0.5`,
  plus a genuine f64 tensor operand, a broadcast operand, a non-contiguous
  operand, the empty case, two in-place spellings and two `out=` spellings.
- `/tmp/clamp_probe.py` — **60 cases, 0 failures** — nine dtypes across
  `clamp(0.5, 1.5)`, `clamp(0, 1)`, `clamp(-1.0, 1.0)`, the one-sided forms,
  an inverted pair, a large int32 bound, non-contiguous and empty, asserting the
  result dtype as well as the result.
- `/tmp/cmp_int32.py` — **0 mismatches** on int32, int64, int16 and uint8 against
  twelve fractional bounds, with the raw cast and comparison printed both ways.
- `/tmp/copy_probe.py` — **36 cases, 0 failures** — `StridedCopy` and `DtypeCast`
  over seven dtype pairs, six strided/offset/empty/broadcast copies, three
  cross-card peer round trips and a 0.03 GiB H2D + D2H sanity check.
- `/tmp/whereout_probe.py` — **13 cases, 0 failures** — the `out=` contract,
  including the 0-dim `self` that `_safe_softmax` passes, a wider condition, a
  grown empty `out`, a non-contiguous `out` taking the host path, and the
  uncastable-dtype `TORCH_CHECK` raising the message ATen would.

Three integration test classes are added for the three correctness fixes, **18
cases** in total: `TestMulPythonNumberOperand` in
`tests/integration/ops/test_mul_dispatch.py` (6), `TestLeScalarCorrectness` in
`tests/integration/ops/test_le_dispatch.py` (7, one of them parametrized over the
four bounds that straddle the promotion) and `TestWhereOutCorrectness` in
`tests/integration/ops/test_where_dispatch.py` (5). Those three files report
**34 passed, 9 skipped in 29.50s**, and the GCU operator cohort under the two
marker selections `.github/configs/gcu.yml` uses reports **621 passed, 26
skipped, 633 deselected, 2 xfailed, 2 xpassed** for the vendor-backend group and
**7 passed, 4 skipped, 1272 deselected, 1 xpassed** for the FlagGems-runtime
group. Both groups exit 1 on the same two cases,
`TestMeanDimDispatch::test_dispatch_log_flaggems_runtime` and
`TestSiluDispatch::test_dispatch_log_flaggems_runtime`, and those failures are
this host's rather than this change's: each spawns a child process with
`sys.executable` and an `os.environ.copy()`, and the child dies importing Triton —
`ImportError: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.38' not found
(required by .../triton/_C/libtriton.so)` — before any dispatch assertion runs.
The host is Ubuntu 22.04 (glibc 2.35), which is why this session runs its own
interpreter under a private glibc 2.39 loader; a `sys.executable` child does not
inherit that. The CI image is ubuntu 24.04. Neither failing op is rerouted here,
and an unscoped `tests/integration/ops/` run additionally fails two
`flaggems_python` cases, `test_bmm_dispatch.py` and
`test_silu_backward_dispatch.py`, with the same import error — **4 failed, 650
passed, 627 skipped, 2 xfailed, 3 xpassed in 311.57s**. The `-m anyplatform`
selection, which excludes the FlagGems-runtime subprocess tests, is **614 passed,
8 skipped, 661 deselected, 2 xfailed, 1 xpassed in 22.89s** with no failure.
`tests/unit/` — **486 passed, 98 skipped, 2 failed**, and both failures
are a pre-existing ordering leak in `tests/unit/test_musa_rng_bridge.py` that this
change does not touch:
`test_flaggems_philox_uses_flagos_reservations` and
`test_flaggems_philox_reaches_vendor_backend_modules`. `tests/unit/test_ascend_platform_marker.py`
writes the module global `_BACKEND_CONFIG_PATH` and leaves it pointing at a
nonexistent temp conf, so a later `_conf_routes_to_flaggems()` returns False and
the MUSA RNG bridge tests take their early return. The MUSA file passes alone
(`3 passed in 7.88s`) and the pair reproduces deterministically (`2 failed,
5 passed in 8.45s`), so the leak is in the suite's ordering and not in this
change; the file is byte-identical to the base commit's copy.

**FlagGems overload survey.** The route this change moves is
`_scaled_dot_product_efficient_attention`, `none` -> `gcu`. No route moves *out*
of FlagGems and the FlagGems route set is byte-identical before and after, so the
S60-wide survey is a provenance re-run rather than a new cohort — and it was run
against exactly the shipped configuration rather than inferred from the routing
table. `tests/manual/flaggems_overload_survey.py` v6 (hash
`7b01c22ce3a94315f1364df242323e9faac27f2585debfb05030670c7c756cc7`),
`torch_fl/configs/backends_gcu.conf` at
`28f4656c30b39b7a60128cf581426f968c077aa5895d23d6193c642799a4ef1b` — the hash
this change ships — flag-gems 5.3.2, FlagTree 0.6.1+enflame3.6, on card 6:

| Verdict | Routes |
|---|---|
| registered | 254 |
| tested | 194 |
| basic-executable | 191 |
| strict | 121 |
| basic-only | 70 |
| failed | 3 |
| untested | 60 |

1778 cases in all — 981 pass, 705 invalid case, 80 error, 12 wrong (artifact
`/tmp/gcu-overloads-post.json`, `6d1120be…`). Every one of those 1778 verdicts is
**identical to the base cohort's, case for case and route for route**: the base is
`/tmp/flaggems_gcu_survey.json` (`1ee7ea4f…`) against
`476825dba7f3050f4d91590bce29362c50f7d23df6f8c952fba1417b51afc05f`, the copy the
parent commit ships and the conf this change starts from, and a case-level
comparison of the two artifacts finds the same key set and **0 differing cases**.
That is the number to read the section by, because a summary that agrees can still
hide swapped verdicts; this one does not, so the change is route-neutral for
FlagGems by measurement and not merely by inspection of the conf diff. The
comparison with the 2026-09-15 FlagGems-routing entry below still stands: that run
was harness **v4** over all 374 routes of a transient, un-gapped draft of the same
file (`meta.conf_sha256` `82f801778c…`, not byte-recoverable), while this is
harness **v6** over the 254 routes the shipped file claims. The route set shrank
in between — the native generator took 117 back (`374 - 117 = 257`) and three more
left FlagGems afterwards, `_softmax` and `linalg_vector_norm` on 2026-09-17 and
`where.self_out` on 2026-09-19, for 254 — so the two are different cohorts and
their counts do not subtract. It is also not comparable with the four rows in the
hardware summary above, which are the generic FlagGems configuration over 546
overloads.

The three failed routes are `gcd_`, `lcm` and `lcm_`. Each fails only on the
`2d-i64` profile, with its other six profiles passing or being invalid cases, and
each fails with `RuntimeError: Pipeline run failed: PassManager execution failed`
raised out of `triton/backends/enflame/compiler.py:253 make_gcuir`. That is a
FlagTree i64-lowering failure of the same family as the vendor SDK's missing
int64 kernels — the reason `clamp`, `fmod.Tensor`, `gelu`, `mean`, `mean.dim`,
`remainder.Tensor` and `silu` are marked `# gcu` in the FlagGems file at all. It
is on three routes this change does not touch, and the route it does touch,
`where.self_out`, is no longer a FlagGems route. The 60 untested routes are ones
where every profile was rejected as an invalid case, so their verdict is absent
rather than negative.

**Other platforms.** The GCU generator, the GCU backend helpers, the two GCU copy
files and the GCU conf are the whole of it, except for `copy_ops.cc` and
`contiguous_ops.cc`, which are shared. Those two are touched only in the shape
Ascend's `ascend_copy.h` already established — a `#if defined(USE_GCU)` arm beside
the existing `#else` ascend arm, with both headers supplying inline no-op
fallbacks so every caller stays total — so no other platform's routes, kernels or
behaviour move *in this half of the change*: Ascend, DCU, MetaX, PPU and
Tsingmicro are unaffected rather than unvalidated. MUSA is unaffected by it too,
but the same pull request carries a MUSA half as well — the `where.self_out`
route this entry moves on GCU fails on MUSA for a different reason, and the
section below records that route change and its own cohort.

**Evidence gaps.** The vendor rate is the 50-step paired run's and the three
fixed-latent rows are 8-step runs, so the ratio in the headline is a per-step
comparison across two runs rather than within one, and no vendor 8-step run was
taken to make it a same-run figure; the two rows at 301.77 and 304.81 s/it are
`paired_full.out`'s running and final rates for one leg, not two runs; the 8-step
image is a fixed latent and not a prompt cohort, so it establishes determinism
rather than quality; the 1-ulp survey is 45 spellings on one shape family rather
than a dtype-by-operator matrix; card 5 is defective, so nothing was measured on
it; the FlagGems survey covers the 254 ops GCU routes to FlagGems and is not the
generic-conf 546-overload cohort the hardware summary measures, and its three
failures are recorded with the compiler frame that raises them rather than
diagnosed further into FlagTree; the four integration failures in the local runs
are the runner's and not this change's — the suite's subprocess tests spawn
`sys.executable`, which does not inherit the private glibc 2.39 loader this
session runs its interpreter under, so the child fails on the host's glibc 2.35
before any dispatch assertion runs, while CI runs the ubuntu 24.04 image; and the
probes are uncommitted.

### MUSA: `where.self_out` moved off FlagGems after the `out=` contract failure (2026-09-19, not hardware-revalidated)

`where.self_out` moves `flaggems` -> `musa` in
`torch_fl/configs/backends_musa.conf` (SHA-256
`87d150533c73e4ca40a24c2588aed51387d257044290d1dd85e8cc9a9d40ffad`), through a new
`NATIVE_TRITON_GAPS["musa"]` entry and a new `where_out` category in
[`scripts/codegen/codegen_mudnn.py`](../../scripts/codegen/codegen_mudnn.py). This
is the same overload the GCU entry above moves, and the two are independent: GCU's
is a throughput route change over a measured **46x**, this one is a *contract*
route change over a failure, and it was selected by a CI run rather than by a
sweep.

**The defect, and why it is the `out=` contract.** ATen's `out=` overload grows a
destination whose shape does not match the result, including one that starts
empty. FlagGems' `where_self_out` computes the broadcast shape only when `out is
None`; handed a destination it passes it straight to
`where_inner(..., out0=out)`, whose `pointwise_dynamic.prepare_args` validates the
shape instead of resizing it. The out-of-place spelling is unaffected and stays on
FlagGems — `where_self` sizes its own destination — so the gap is exactly the
`out=` overload and not the op.

**Measured.** The MUSA lane of PR #351, group *Run operator tests (native mudnn,
main ops)*, before the change:
`tests/integration/ops/test_where_dispatch.py::TestWhereOutCorrectness::test_out_grows_from_empty`
fails with

```text
RuntimeError: out tensor at index 0 shape is invalid, should be (2, 4) but is torch.Size([0])!
```

raised out of `flag_gems/utils/pointwise_dynamic.py:1699` and reached through
`torch_fl/__init__.py:1321` (`_CudaAliasMode.__torch_function__`) and
`flag_gems/ops/where.py:81`, on flag_gems `5.4.0rc2.post1+g437ba3938`. The group
reports **1 failed, 507 passed, 6 skipped, 646 deselected, 2 xfailed, 1 xpassed in
228.66s**, and that case is the only failure in it; every other `where` case
passes, including the same overload at a matching shape. The case is the one this
pull request added.

**Why it surfaced on MUSA and not on CUDA.** The case carries
`@pytest.mark.anyplatform` alone, and the per-platform manifests select
differently: MUSA's group selects `(anyplatform or main_ops)` while CUDA's selects
`main_ops` and the FlagGems marks, so the case is collected here and not there.
The same route is still a FlagGems route on `cuda`, `dcu`, `metax` and `ppu`, so
the contract gap is latent on those four and this change does not fix them; they
are recorded under **Evidence gaps** below.

**The fix is generated, not handwritten.** `T_WHERE_OUT` is the out-of-place
ternary template plus the contract: the broadcast shape is inferred with
`at::infer_size` over all three operands, `out` is resized when it differs, the
standard empty-output guard follows, and the body runs `mudnn::Ternary` in
`SELECT` mode. It reuses the two idioms the MUSA `mm`/`bmm` `out=` fallbacks
already established in the same generated file — `out.resize_(...)`
(`musa_kernels.cc:1782`, `:1809`) and `out.copy_(host)` for the host path — and
takes the route through the gap set rather than by editing the conf, since the
conf is generated output.

**The second half of the contract: the destination's dtype.** Growing the
destination is not the whole of ATen's `out=` contract, so the template carries
the dtype rule too, and ATen's rule is *equality* rather than castability.
Measured on CPU with `torch.where(cond, a, b, out=out)` over an f32 result,
ATen's own overload raises `RuntimeError: Expected out type to be Float but got
Double` for a destination of *any* other dtype — and likewise for `Half`, `Int`,
`Long`, `Bool` and `Byte` — whether or not the result would fit there. That check
is TensorIterator's; the regression test this pull request adds asserts only the
part that matters for safety, that a result is never *narrowed* silently, and
deliberately does not pin the exception class. `T_WHERE_OUT` enforces the same
equality, with ATen's own phrasing, and it does so *before* the fallback branch
rather than inside it: that branch writes the result through `out.copy_(host)`,
which for an f16 destination would narrow an f32 result instead of raising, and
placing the check first also means an unsupported input operand cannot turn into
a silent cast either — the FlagGems wrapper this overload used to route to
asserted on the same case rather than writing a truncated result.

This is deliberately *stricter* than the sibling GCU template `T_WHERE_SELF_OUT`,
which admits any destination the result can be cast into (`c10::canCast`) and
writes the castable-but-different case through the same host copy; for an f16
destination over an f32 result that route narrows rather than raises on GCU. The
MUSA half takes ATen's rule because it replaces a route that raised, so matching
ATen is what the change is for; tightening GCU's is a behaviour change on a
platform whose kernel bytes carry their own measured evidence and is **not** part
of this change.

**Route delta.** MUSA `flaggems` 468 -> **467**, `musa` 51 -> **52**, `none`
**1518** unchanged over the same **2037** routable ops; accelerated stays **519
(25.5%)** — this buys correctness and not coverage. `musa_register.inc` gains one
`m.impl` line (160 -> 161) and `musa_flaggems_register.inc` loses one (359 -> 358)
and gains the op in its excluded list (14 -> 15, banner 121 -> 122), so the two
reconciliations read `467 = 358 + 109` and `52 = 161 - 109`: the 109 is the conf's
`# musa` annotations, the ops a native kernel exists for and that FlagGems still
wins, and the 52 left over are the native ops FlagGems has no kernel for. Before
the regeneration the two files overlapped in exactly one op, `where.self_out`,
which registers twice on PrivateUse1 and warns; after it, they are disjoint in
both directions.

**Regenerated artifacts (SHA-256 before -> after).** `backends_musa.conf`
`e62cac76…` -> `87d150533c73e4ca40a24c2588aed51387d257044290d1dd85e8cc9a9d40ffad`;
`musa_kernels.cc` `f0edc538…` ->
`2002b5fdb087b56385b7ee27fe604c902ae43146da46f376c54f691aec3a7483`;
`musa_register.inc` `22dd2327…` ->
`7b4e0480f0fc1777a61cda1c5010aa2a0c0ff2e6a9e43cf92c6b4f5413b4ed8a`;
`musa_flaggems_register.inc` `14ffc529…` ->
`13474b812da85f88dbb4d3ef6a2f44ac04a88cf7c286b63e2d194021c9dfb4ab`. Generator
idempotency: a second run of `codegen_mudnn.py`, `codegen_musa_flaggems.py` and
`gen_vendor_confs.py` leaves all four artifacts byte-identical
(`IDEMPOTENT=YES`).

**Local verification.** `tests/unit/test_gen_vendor_confs.py` — **36 passed**,
after the pinned MUSA gap set in that test is widened by the entry and its
docstring records the diagnosis, which is what the file's own strategy asks for
(*register everything, run CI, move failures to the gap set*);
`tests/unit/` — **486 passed, 98 skipped, 2 failed**, both failures the
pre-existing ordering leak in `tests/unit/test_musa_rng_bridge.py`
(byte-identical to the base commit's copy, **3 passed in 7.97s** alone, the same
two failing when `tests/unit/test_ascend_platform_marker.py` runs first);
`ruff check .` — "All checks passed!"; `ruff format --check .` — 272 files already
formatted.

**The kernel executed on an MTT S5000 and the group now passes.** The CI lane that
carried the failure was re-run against the fixed tree — PR #351 run
`35417994902` at `e8bd278`, job *Platform pipeline (musa) / Build and test (MUSA)*,
group *Run operator tests (native mudnn, main ops)*, on the same manifest as the
failing run — and the group reports **508 passed, 6 skipped, 646 deselected, 2
xfailed, 1 xpassed in 227.81s**, with **no failure**: one more passing case than
the failing run's 507 and none failing. Both `out=` cases this pull request adds
are in that group and both pass, `test_out_grows_from_empty` — the case that raised
`out tensor at index 0 shape is invalid` before the change — and
`test_out_rejects_uncastable_dtype`, which the earlier run did not exercise:

```text
tests/integration/ops/test_where_dispatch.py::TestWhereOutCorrectness::test_out_grows_from_empty PASSED
tests/integration/ops/test_where_dispatch.py::TestWhereOutCorrectness::test_out_rejects_uncastable_dtype PASSED
```

That is device evidence for the kernel body: the mudnn `SELECT` runs, the contract
resize grows the empty destination, and the strict dtype check raises where ATen
raises rather than writing a narrowed result. The whole MUSA job completed
`success`, and the run's other lane groups pass with it. What it is *not* is the
survey cohort described below — the FlagGems support table is still the one
measured against the 2026-09-15 configuration.

**Evidence gaps.** This is a **not revalidated** platform in the sense the
introduction fixes: no MTT S5000 run of `flaggems_overload_survey.py` or of its
cohort was taken against the new configuration, so MUSA's FlagGems support table
above is unchanged and stands as measured against its own 2026-09-15 cohort rather
than against the 467-route one this change ships. The entry's evidence is the CI
failure, the fix, and the CI re-run on the fixed tree described above — a real
S5000 run of the affected group, but through the lane's manifest rather than a
local device session, which this host has none of. The generated kernel was
compiled and executed by that lane, so it is no longer an unbuilt artifact; it is
still **not locally compilable here**, the MUDNN toolchain not being installed. No
timing was taken, so nothing is claimed about whether mudnn's `SELECT` is faster
than FlagGems' `where_self_out` at the shapes this overload is called with; the
change is justified by the contract alone. And the same contract gap remains on
`cuda`, `dcu`, `metax` and `ppu`, which still route the overload to FlagGems and
whose manifests do not collect the case — the two candidate remedies (the same
per-platform gap plus a native kernel, or resizing in the shared generated FlagGems
wrapper that all five platforms call) are a separate change, and neither is
attempted here.

### Enflame GCU S60 native kernels for the four remaining Qwen-Image-2512 CPU fallbacks (2026-09-18)

The Qwen-Image-2512 pipeline still reached ATen's `cpu_fallback` in four operators
after the three route changes below, and each of them now has a topsaten kernel.
Five overloads move from `none` to `gcu` — `all`, `where.self`, `index.Tensor`,
`nonzero` and `nonzero_static` — which is the first route change on this platform
selected by a census of the pipeline under test rather than by a probe standing in
for it.

**The census.** Both halves of the paired Qwen-Image-2512 flagos cohort on the S60
(`torch_fl` + `diffusers`, `FLAGOS_LOG=dispatch,fallback`) log `cpu_fallback`
before their first prompt, and the two logs agree operator for operator: **62
`cpu_fallback` calls in exactly four operators — `aten::where` 56, `aten::nonzero`
2, `aten::index` 2 and `aten::all` 2** — against 489,770 dispatch records in the
first half (366,633 `gcu` and 122,583 `flagos_python`; the second half's log is
429,692 records with the identical fallback census). No fifth operator appears in
either log. `csrc/aten/fallback.cc:21` prints `op.schema().name()` — the schema
name and not the overload — so `aten::where`, `aten::index` and `aten::all` each
stand for one spelling here: the same log carries **4,215 `where.self_out` and
4,215 `all.dim` dispatch lines on `flagos_python`**, so the `out=` and `dim`
siblings were already routed and it is the `where.self` and whole-tensor `all`
spellings the census counted. The fifth overload, `nonzero_static`, is in neither
log; it comes from the training probe the entry above used, which recorded it as
the one `cpu_fallback` call it had left, and it is in this change because the
reason that entry gave for declining it turned out to be partly wrong — see
"Correction" below.

**`where.self`.** `topsatenWhere` serves it, with two conditions checked rather
than assumed: the `condition` must be bool, because a mask arriving as anything
else is a different operation, and the result dtype must be one
`TopsatenWhereDtype` lists (Float, Double, Half, BFloat16, Long, Int, Short,
Char, Byte, Bool), which excludes the float8 and complex results
`ToTopsatenDataType` would raise on. Anything else takes the host path. The vendor
does not broadcast, so all three operands are expanded to
`at::infer_size(at::infer_size(condition, self), other)` — **all three, the
condition included**: ATen broadcasts the condition against the values' common
shape, which is not the condition's own shape when the condition is the wider one,
so a kernel expanding only the two values would be right on the common case and
wrong on that one. The expansion is materialised with `contiguous()` because the
wrapper honours strides while the vendor reads a dense buffer.

**`all`, the whole-tensor spelling.** `topsatenAll` needs a destination it can
write a single PRED into and aborts on a bare rank-0 descriptor;
`TopsatenTensorWrapper` already rewrites a rank-0 tensor to a one-element vector,
so the 0-dim ATen result is passed straight through and no buffer or metadata
`reshape` is needed — which matters, because `at::Tensor::reshape` and `squeeze`
are themselves dispatcher calls GCU carries no kernel for. The output's dtype is
**ATen's, not bool**, and that is a measurement rather than a reading:
against ATen's own `meta` function, `all` on a `uint8` operand returns `uint8` and
on every other dtype — bool, int8, int16, int32, int64, float16, bfloat16,
float32, float64 — returns `bool`. That is the meta function, not a CPU kernel
quirk, so a backend answering `bool` unconditionally is the deviant one, and the
kernel allocates `kByte` for a byte operand and `kBool` otherwise. A dedicated
probe that calls `topsaten::topsatenAll` directly on card 4 established that the
vendor writes the correct 0/1 into *either* a `U8` or a `PRED` output descriptor —
an all-ones buffer gave 1 and a buffer with one zero in it gave 0 in both, with
the untouched sentinel bytes left where they were and `SUCCESS` returned each
time — so no staging cast is needed. Half and bfloat16 operands are widened to
float32 first, which is what makes the Qwen-Image transformer's bf16 mask reach
the vendor at all; an empty or otherwise declined operand takes the host path,
where the vacuous `True` costs nothing to compute.

**`index.Tensor`.** ATen's advanced indexing is one dispatcher entry for several
different operations and the vendor's kernel implements exactly one of them: a
single rank-1 index tensor, which is `index_select` on dim 0. The template
recognises that spelling — exactly one index, present and defined, `dim() == 1`,
an index dtype `TopsatenIndexDtype` admits, and every coordinate non-negative —
and delegates to the `IndexSelectKernelGcu` this generator already emits, so it
inherits that kernel's own guard and declines for a 0-dim `self`, an unsupported
operand dtype or a non-contiguous operand on the same host path as this template's
own fallback. A 0-dim index is excluded by the rank test and is not an oversight:
`t[tensor(1)]` drops the indexed dimension (`(4,)` for a `(3, 4)`) where
`index_select` keeps it (`(1, 4)`), so it is a different operation. A bool index
is excluded by the dtype test — it is a mask, not a coordinate list. Negative
coordinates are excluded because ATen wraps them into the tensor while the vendor
resolves them outside, which is what `TopsatenIndexNonNegative` reads back to
check. Every other spelling marshals the index tensors to the host and calls
ATen's CPU advanced indexing, then moves the result back: **the indices travel
with the operand**, since ATen's host implementation runs on whichever device
holds the data and a device index would reach it as a device pointer. That is the
marshalling `register.cc`'s `WrapperIndexPut_` already uses.

**`nonzero` and `nonzero_static`.** Both need three vendor calls rather than the
one an operator of this shape suggests, because the output's first dimension is
the *count* of nonzeros and nothing about the answer is known before the data has
been read once: `topsatenCountNonzero` writes the count into a device int32,
`topsatenNonzero` writes the coordinates as int32, and `topsatenTo` widens them to
the int64 ATen declares. The count is read back to the host rather than used as
the output size directly, which is not defensiveness — see "the status is not the
answer" below. `nonzero` sizes its output to the true count. `nonzero_static`
sizes it to the caller's `size` and pads, and the padding is why it stages through
int32: the tail has to be filled and the vendor has no int64 fill, so
`topsatenFill_` fills the staging before `topsatenNonzero` writes the coordinate
rows into it. `nonzero_static` declines a `fill_value` outside int32 range when a
fill is actually needed, because a truncating narrowing would round-trip wrong,
and declines a negative `size`, a rank-0 operand and an empty one; the first two
of those take the host path so that ATen's own message is the only spelling of
that error that cannot drift from ATen.

**The status is not the answer.** `topsatenNonzero` refuses a partial write
outright: handed an output narrower than the count it returns `BAD_PARAM` and
writes nothing — `(2, 3)` over a rank-3 input with four nonzeros failed, while
`(4, 3)` and `(6, 3)` both succeeded. So the coordinates always go into an
`(nnz, rank)` description of the staging and it is the *read-back* that gets
shortened, in either direction, through the `TopsatenRowView` helper. This is the
same family as the earlier finding that `topsatenCountNonzero` returns SUCCESS
with a wrong count on a dtype it does not support, and it is why the count is
rounded through the host instead of being handed straight to `at::empty`.

**Verification.** A 67-case probe (`/tmp/verify_five.py`, one process, card 4,
`FLAGOS_LOG=dispatch,fallback`) runs every case against the same ATen call on the
CPU and compares shape, dtype and values — exact for bool and integer results,
`allclose(1e-3, 1e-3)` for floating ones — reporting **`RESULT: all 67 cases
passed` with 0 failures, 0 `cpu_fallback` lines and 67 dispatch lines, every one
of them `-> gcu`** (`where.self` 15, `all` 15, `index.Tensor` 15, `nonzero` 12,
`nonzero_static` 10). Coverage is by case family rather than by operator: `where`
over same-shape, a narrower and a wider condition, a scalar `self`, eight dtypes,
a mixed float32/int64 pair, the empty case and rank 0; `all` over ten dtypes on
all-ones, a buffer containing a zero, two empty shapes and both rank-0 answers;
`index` over 1-D int32 and int64 indices, duplicates, empty rows, a bool mask, a
0-dim index, two 1-D indices, the `try`-then-index spelling, six `self` dtypes and
a 3-D operand with an int64 index; `nonzero` over six dtypes, an all-zero buffer,
ranks 1 and 3, two empty shapes and rank 0; and `nonzero_static` over an exact
fit, a pad, a truncate, `size = 0`, a large pad, an int64 `fill_value` large
enough that the narrowing matters, two empty shapes, rank 0 and rank 3. One of the
67 cases is the host path on purpose — it is checked for ATen's answer rather than
accepted for running.

**One case found a bug, and the fix is the interesting half.** The first run was
66 of 67: `all` on a uint8 operand came back `tensor(True)` where the CPU returns
`tensor(1, dtype=torch.uint8)`. The generated kernel had hardcoded `at::kBool`,
which reads correct and is not — ATen's dtype inference for this operator is not
uniform, and the ten-dtype measurement above, taken on both the meta and the CPU
function, is what turned "the test is too strict" into "the kernel is wrong". The
fix allocates ATen's declared dtype rather than adding a staging cast, because the
vendor probe showed either descriptor receives the right 0/1 and a cast would be
work for nothing.

**Route delta.** GCU `flaggems` stays **248**, `gcu` 171 -> **176**, `none` 1611 ->
**1606**, over the same 2037 routable ops, so accelerated routes go 426 -> **431**
(20.9% -> **21.2%**). `gcu_register.inc` grows by five `m.impl` lines (178 -> 183)
and `gcu_flaggems_register.inc` is unchanged at 248, so the reconciliations read
`248 = 248 + 0` and `176 = 183 - 7` — **no route moved *from* `flaggems`**, since
this change only takes operators off `none`, which is why no S60-wide
`flaggems_overload_survey.py` run is claimed for it. Three of the five do move the
second file's provenance banner, 97 -> 100 "further FlagGems ops already claimed
by `gcu_register.inc`": `all`, `nonzero` and `where.self` are inside FlagGems' own
coverage, so claiming them natively is what takes them off that list, while
`index.Tensor` and `nonzero_static` were never on it. Their
`NATIVE_TRITON_GAPS["gcu"]` entries **stay**, for the reason the factory entry
below records: a gap entry only takes an operator off FlagGems, and deleting one
would put it back on a FlagGems path that cannot serve it.

| Artifact | Before | After |
|---|---|---|
| `torch_fl/configs/backends_gcu.conf` | `366d986f0c7aa292eb398e9de8b3dd9a1533240fc4f026472299a3560245e40d` | `4e8265afbf12b472097588ea1f7ae67f627b37711f2e3c17200f190588e0ea40` |
| `gcu_kernels.cc` | `7f34288a4af8b3da7231c600eea8446c7870fa7312d2bb369a5585de16e3a517` | `1d7d4f9b6f2830fb67a78bfeeb48582fa59a4e369287669a3486ad92745ef843` |
| `gcu_register.inc` | `806d7ce79d075060f53a57c9f8982238e0925d9098da8ffd18a5b6f31a84a6be` | `638619ee4b080a0839ad225f2b82d56684dcee219d95f903899014e5c3d5f954` |
| `gcu_flaggems_register.inc` | `9c9c99d0d2eab6ba158090ef16fff3566422d688b850a50e6bcc97568d2e0b71` | `c520a93a3ab23a196d836f349547c6535ba6a84b0a1490cf04f5c2309b7f210d` |

**Rebase note.** This branch was rebased onto upstream `2f6b10c` after these
measurements were taken. That commit adds `aten::scaled_dot_product_attention` to
the routable op list with a `none` GCU route, so the list is **2037** rather than
the 2036 the entries below record, every `none` count is one higher than the
measurement log shows, and the accelerated counts and their percentages are
unchanged, because no `gcu` or `flaggems` route moved. `backends_gcu.conf` is the
only generated artifact the shift touches, since it is the file that carries the
op list: the conf this branch ships is
`4e8265afbf12b472097588ea1f7ae67f627b37711f2e3c17200f190588e0ea40`, 431/2037
(**21.2%**), and every conf hash recorded below for a state this branch *commits*
has been restated from the rebased tree. Hashes recorded for intermediate
working-tree states rather than for commits are left as measured.

The generator change and the helpers it adds to
`csrc/aten/backends/gcu/topsaten_common.h` are not hashed: both are
version-controlled sources rather than generated output, so the diff is the
record. Only the three generated artifacts and the generated conf are hashed, as
in the entries below. Generator idempotency: two runs leave all four byte-identical
(`IDEMPOTENT=YES`). `ruff check .` — "All checks passed!"; `ruff format --check .`
— 270 files already formatted. `tests/unit/test_gen_vendor_confs.py` — 36 passed;
the rest of `tests/unit/` under the glibc 2.39 loader — 482 passed, 98 skipped,
2 failed, both of them in `tests/unit/test_musa_rng_bridge.py`, which is
byte-identical to the base commit's copy of that file and imports nothing this
change touches.

**Correction to the entry below.** That entry closed with "`aten::nonzero_static`
is deliberately left on `none` … its schema has no `out=` overload, so the FL
layer's `out=` lane cannot serve it either". The `out=` half of that sentence is
wrong: `nonzero_static.out` exists in the schema, and it is a line in
`backends_gcu.conf` (still `none` here, as is `nonzero.out`). The other two halves
stand — the result is int64, and FlagGems ships no `nonzero_static` module — so
the operator's route to the device is a native kernel that widens int32
coordinates to int64 through `topsatenTo`, which is what this change adds rather
than a route the older entry missed.

**Other platforms.** The whole change is inside the GCU backend and the GCU
generator — `scripts/codegen/codegen_gcu.py`,
`csrc/aten/backends/gcu/topsaten_common.h` and the three generated files under
`csrc/aten/backends/gcu/generated/`, plus the GCU conf — so no shared code is
touched and no other platform's routes or kernels move. Ascend, MUSA, DCU, MetaX,
PPU and Tsingmicro are therefore unaffected rather than unvalidated.

**Evidence gaps.** The census is the pipeline's own warm-up region rather than a
full sweep — the cohort's `cpu_fallback` records precede its first prompt — so 62
is a floor on what the pipeline costs and not a per-prompt or per-step total; card
5 is defective and hangs any `topsaten` op on a tensor resident there, so nothing
was measured on it and the verification ran on card 4; the 67 cases are one shape
family per case rather than a shape sweep, and `where`'s broadcasting is exercised
at three shapes rather than over a general broadcast matrix; the probe scripts and
the `topsatenAll` descriptor probe are uncommitted; and no timing was taken, so
what each of the five round trips was worth is not measured — only that the calls
stop reaching the host.

### Enflame GCU S60 native kernels for the index, embedding and upsample fallbacks (2026-09-17)

The GCU configuration was regenerated once more to claim the operators that still
reached ATen's `cpu_fallback` after the three route changes below. Twelve
overloads moved from `none` to `gcu` — `index_select[.out]`,
`index_fill.int_Scalar[_out]`, `index_fill.int_Tensor[_out]`,
`index_fill_.int_Scalar`, `index_fill_.int_Tensor`,
`embedding_dense_backward[.out]` and `_upsample_nearest_exact2d[.out]` — the
largest of the four additions. None of the twelve had a PrivateUse1 registration
before this change, which is why every call to any of them was a host round trip
by construction rather than by accident.

**The census.** No single workload reaches all twelve, so two were used. A training
step — two Adam steps over a five-parameter model with an embedding lookup and an
`index_select` path — run with `FLAGOS_LOG=fallback` recorded **11
`cpu_fallback` calls in four operators**: `aten::index_select` 4,
`aten::embedding_dense_backward` 4, `aten::index_fill_` 2 and
`aten::nonzero_static` 1. That is the probe the `out=` entry below used, at the
step where it had already taken the total from 45 to 11. The names are the schema
names `csrc/aten/fallback.cc:21` logs and not the overloads, which is why the
eight `index_*` / `embedding_dense_backward` spellings above appear as three: the
2 `aten::index_fill_` are the probe's `index_fill_.int_Scalar` and
`index_fill_.int_Tensor` calls, one each. A Qwen-Image-2512 VAE decode at
1024x1024 supplied the rest of the set: it makes three
`_upsample_nearest_exact2d` calls, and on the pre-change tree all three reported
`cpu_fallback`.

**`index_select` and `index_select.out`.** `topsatenIndexSelect` serves both, with
`out` sized from the index's length before the call because the `out=` contract
lets a caller pass a zero-size tensor. The index is the one integer operand this
backend does not decline for being wide: a coordinate is consumed as an integer
rather than as element data, so the "topsaten has no int64 kernel" rule the rest
of the GCU backend is built around does not reach it, and `TopsatenIndexDtype`
accepts int32 and int64 alike — both measured at dim 0, 1 and 2 and with a 0-dim,
empty and non-contiguous index, each matching the host result. Two limits the
vendor's dtype table does not express are checked rather than discovered: every
dim holds fewer than 2\*\*24 elements and every buffer stays under 4.0 GB, past
either of which `topsatenIndexSelect` returns an error instead of a result, so a
call that trips one is served by the host path where ATen had an answer. The two
rejections ATen owns — an out-of-range dim and an index that is not a vector or a
scalar — are left to it, and are conditions of the decline rather than checks of
their own, so the messages a caller sees are ATen's. A 0-dim index is a legal
one-element vector (`index_select` on a `(3, 4)` at dim 0 with a scalar index
returns a `(1, 4)`, measured), so the flatten is a reshape and deliberately not a
rank check.

**`index_fill_` and `index_fill`, all six spellings.** `topsatenIndexFill` fills
its first operand in place, so the in-place spellings pass `self` twice and the
`out=` spellings compute into a temporary through the same kernel and `copy_` into
`out` after resizing it. This is the one operator in the group whose operand types
the two sides disagree on outright: ATen takes an int64 index and nothing else
(`index_fill_(): Expected dtype int64 for index.`) and the vendor takes an int32
one and nothing else (`BAD_PARAM` for an i64 index at every dim, rank, self dtype
and scalar tag measured on the S60). The vendor's own bounds check is not safe to
lean on either — an index equal to the dim size satisfies it and writes one element
past the end of the tensor, and a larger index aborts the process (`Index
out-of-bounds! bound=2, index=5`) — while ATen accepts a negative index here and
wraps it (measured: -1 in dim 0 of a `(2, 3, 4)` fills row 1, where -3 raises). So
`TopsatenIndexFillIndex` reads the index back once, wraps negatives and narrows in
that single pass, and returns an undefined tensor for a value that is still out of
range or not representable as int32 — a truncating cast would fold an out-of-range
index back into range — which sends that call to the host path where ATen raises
its own message. The transfer is O(num_indices) and `self` stays on the device,
which is the point: the host path this replaces copies the tensor being filled to
the host and back. The tensor-valued spellings additionally require a 0-dimensional
`value`, because ATen does: `index_fill_ only supports a 0-dimensional value
tensor, but got tensor with 1 dimension(s)`, measured at both one and four
elements. An empty index is a no-op after validation rather than before it, since
ATen rejects an int32 index even when it is empty.

**`embedding_dense_backward` and its `out=` form.**
`topsatenEmbeddingDenseBackward` again wants int32 where ATen supplies int64, and
again the narrowing cannot be a plain cast. ATen does not raise on an index outside
`[0, num_weights)`: it *ignores* that element of `grad` — measured for 9, -2 and
2\*\*32 + 1, each against a 6-row table, all three leaving the table holding only
the in-range contributions. A truncating cast folds 2\*\*32 + 1 back to 1 and the
vendor would then accumulate that row of `grad` into table row 1, a silently wrong
gradient rather than an ignored element, so `TopsatenEmbeddingIndex` validates in
the same pass that narrows and returns undefined for any index outside the range —
and for a `num_weights` above `INT32_MAX`, a table the vendor cannot address at
all. Declined the same way: a rank-0 `grad`, a non-positive `num_weights`, a
`padding_idx` below -1 or at or above `num_weights`, and a bool `grad`, the last
through a new `TopsatenEmbeddingDenseBackwardDtype` that is `TopsatenSupportsDtype`
minus PRED, because the vendor's parameter table omits bool even though its arange
and index kernels take one and a bool embedding weight is legal in ATen. A `grad`
whose element count does not match `num_indices * embed_dim` also declines, since
the vendor answers that mismatch with a status error where ATen raises a message a
caller can act on. The empty case is the one place the vendor's own flow is not
sufficient: it zeroes its buffer as step 1 and then writes the rows the index
names, so with no index at all nothing is written and the kernel zeroes the output
itself rather than assume a caller's `out=` starts at zero.

**`_upsample_nearest_exact2d` and its `out=` form.**
`topsatenUpsampleNearestExact2d` takes the output size and both scales separately,
which is the shape ATen's schema has, and it serves the 4-D input the Qwen-Image
VAE decoder produces. `scales_h` and `scales_w` are optional in the schema and
required by the vendor, so a call that omits either declines, as do a non-4-D input
and any dtype `TopsatenSupportsDtype` disallows. It is the one operator in the group
that the training probe cannot see at all — its only GCU caller is the VAE
decoder's `QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact")`
blocks — which is why it was measured on that path rather than inferred from the
probe.

**Verification.** A 60-case semantic probe over the twelve overloads, every case
against the same ATen call on the CPU, reports `ALL_OK` with 0 failures and 0
`cpu_fallback` lines. Twelve of its cases are the host path on purpose — an int64
and a float64 `self` for `index_select`, an int64 `self` for `index_fill_`, an
`edb` index above `num_weights`, negative and past int32, a bool `grad`, a
`padding_idx` of -4 through `out=`, and an upsample with no scales — and each is
checked for ATen's answer rather than accepted for running. The training probe
re-run against this change logs **exactly one `cpu_fallback` call**,
`aten::nonzero_static`, against 4 `index_select -> gcu` and 4
`embedding_dense_backward -> gcu` dispatch lines, one `index_fill_.int_Scalar ->
gcu` and one `index_fill_.int_Tensor -> gcu`; both step losses are unchanged to
the printed digits (`-32.7952`, `-51.3676`), the same readings the pre-change run
printed. The VAE decode re-run on the current tree reports **`cpu_fallback ops:
0`** where the pre-change tree reported three `_upsample_nearest_exact2d` round
trips, and runs to completion (exit 0): 315 dispatch records over 19 distinct ATen
operators, 246 `gcu` and 69 `flagos_python`, the 1024x1024 image saved with
`mean=0.3489 std=0.2659`. What each route is worth was measured with the kernel's
own declining branch as the in-run control — an int64 operand takes the host path
through the same entry point, at the same shapes, in the same process — because a
`FLAGOS_OP_*=none` override is not available for an op that is registered: it
raises `routed to 'none' (no accelerated impl on this platform) but the op is
registered on PrivateUse1` rather than falling back. Two runs each, at shapes
`index_select` and `embedding_dense_backward` are actually called with: a
20,000 x 512 table gathering 8,192 rows is **0.155-0.159 ms/call** on the device
against **36.558-37.810 ms/call** on the host, **229.4x-244.2x**; an 8,192-row
`embedding_dense_backward` against a 20,000-row table is **2.183-2.197** against
**31.860-32.416 ms/call**, **14.6x-14.8x**; and `index_fill_` over 1,024 rows of an
8,192 x 512 buffer is **0.578-0.584** against **17.349-17.785 ms/call**,
**29.7x-30.8x**. The generator was extended rather than a kernel handwritten, and
run a second time: all three generated artifacts are byte-identical across the two
runs (`gcu_kernels.cc` md5 `32a05182…`, `gcu_register.inc` `1d9b0b91…`,
`backends_gcu.conf` `b51665a7…`). `ruff check .` reports "All checks passed!" and
`ruff format --check .` "263 files already formatted"; `tests/unit/` gives 426
passed and 98 skipped.

**The one remaining fallback.** `aten::nonzero_static` is deliberately left on
`none`, and is now the only `cpu_fallback` operator on either measured path. Its
result is int64, which `TopsatenSupportsDtype` declines everywhere, so no `gcu`
kernel can be written for it; `flag_gems` ships no `nonzero_static` module, so
there is no `flaggems` kernel to route it to instead; and the schema has no `out=`
overload, so the FL layer's `out=` lane cannot serve it either. The conf says
`none` because that is the only thing it could truthfully say, and the operator is
named here so the next census does not read it as an oversight.

**Corrected 2026-09-18.** The third clause above is wrong: the schema **does** have
a `nonzero_static.out` overload, and it is a line in `backends_gcu.conf` (still
`none`, as is `nonzero.out`). The first two clauses stand — the result is int64 and
FlagGems ships no such module — but "int64, therefore no kernel" does not follow:
int64 is the *output's* dtype, and `topsatenNonzero` writes int32 coordinates that
`topsatenTo` widens, which is how `nonzero_static` was taken off this list. See the
2026-09-18 entry at the top of this section, which also routes `all`, `where.self`,
`index.Tensor` and `nonzero`.

**Route delta.** Measured against the state after the `out=` change below, because
all four are uncommitted in the same tree: GCU `flaggems` stays **255**, `gcu`
159 -> **171**, `none` 1622 -> **1610**, so accelerated routes go **414 -> 426**
(20.3% -> **20.9%**). `gcu_register.inc` grows by twelve `m.impl` lines (166 ->
178) and `gcu_flaggems_register.inc` is unchanged at 248 — no `flaggems` route was
touched — so the reconciliations read `255 = 248 + 7` and `171 = 178 - 7`.

| Artifact | After the `out=` change | In this change |
|---|---|---|
| `torch_fl/configs/backends_gcu.conf` | `f7c2e309df4265246a3f17e3800c5b2529ac1110fa6675a86222a4bc2e648ce0` | `366d986f0c7aa292eb398e9de8b3dd9a1533240fc4f026472299a3560245e40d` |
| `gcu_kernels.cc` | `35c8a37e80f82f377ab671e32b11a2c488b246c200a37b16a462dcd181f6b4d0` | `7f34288a4af8b3da7231c600eea8446c7870fa7312d2bb369a5585de16e3a517` |
| `gcu_register.inc` | `f961aff0c9c1db8dbbc1166f8be58a43329c2917e833de0c618d7c32584ae99f` | `806d7ce79d075060f53a57c9f8982238e0925d9098da8ffd18a5b6f31a84a6be` |
| `gcu_flaggems_register.inc` | `9c9c99d0d2eab6ba158090ef16fff3566422d688b850a50e6bcc97568d2e0b71` | unchanged |

The generator change and the five helpers in
`csrc/aten/backends/gcu/topsaten_common.h` are not hashed here: both are
version-controlled sources rather than generated output, so the diff is the
record. Only the three generated artifacts and the generated conf are hashed, as
in the entries below.

**Other platforms.** The whole change is inside the GCU backend and the GCU
generator — `scripts/codegen/codegen_gcu.py`,
`csrc/aten/backends/gcu/topsaten_common.h` and the two generated files under
`csrc/aten/backends/gcu/generated/`, plus the GCU conf — so no shared code is
touched and no other platform's routes or kernels move. Ascend, MUSA, DCU, MetaX,
PPU and Tsingmicro are therefore unaffected rather than unvalidated, and no
S60-wide `flaggems_overload_survey.py` run is claimed for this change: it is the
right call rather than a gap, because no route moved *from* `flaggems`.

### Enflame GCU S60 native kernels for the six measured CPU fallbacks (2026-09-17)

The GCU configuration was regenerated again to claim topsaten kernels or
metadata registrations for the six operators that still reached ATen's
`cpu_fallback` on a real workload. The set was not chosen by inspection. The
flagos leg of the paired Qwen-Image-2512 run below logs every fallback, and over
the whole 50-step pipeline at 1024x1024 it issued **80,536 `cpu_fallback` calls
across exactly six operators** — `fill_` 23,570, `zero_` 19,221,
`view_as_complex` 18,794, `view_as_real` 18,794, `arange` 79 and
`linalg_vector_norm` 78 — against 468,227 device dispatches (331,263 `gcu`,
136,964 `flaggems`). That is 14.7% of all dispatches, and the six are the whole
of the host traffic: no seventh operator appears in the log. Five of the six were
worth a kernel for their own call count; `fill_` and `zero_` were worth much more
than that, because a device-side `torch.zeros`, `torch.ones` or `torch.full`
decomposes into them, so a host `fill_` turns every factory call in a model into
a round trip.

**`arange`, all three overloads** (`arange`, `arange.start`,
`arange.start_step`). This is the one operator in the group whose vendor entry
point takes no size: `topsatenArange` fills an output the caller has already
allocated, so the kernel computes the length host-side with ATen's own rule,
`at::native::compute_arange_size` from `<ATen/native/RangeUtils.h>`, which is
also what carries ATen's two bound errors (`step must be nonzero`, `upper bound
and lower bound inconsistent with step sign`) onto this route. Which dtypes go to
the vendor is a new predicate, `TopsatenArangeDtype` in `topsaten_common.h`,
deliberately narrower than `TopsatenSupportsDtype`: it is measured on the S60
(`float32`, `int32`, `int16`, `int8`, `uint8`, `float16`, `bfloat16`) rather than
inferred from `ToTopsatenDataType`, which only says a dtype is representable, and
a dtype it declines is built on the CPU and copied back. `int64` — what
`torch.arange(n)` infers with no dtype, and what Qwen-Image's text-position path
asks for — and `float64` therefore always take that host path, because topsaten
has no kernel for either. Bool is absent on purpose: ATen has no arange kernel
for Bool either (`arange_cpu not implemented for 'Bool'`), so declining here
preserves that error instead of inventing a result CPU and CUDA both refuse to
produce. Because a wrong length would look like a correct arange of the wrong
size rather than like a crash, the length is the first thing the probe checks.

**The values.** The vendor evaluates `out[i] = (T)(double((T)start) + i *
double((T)step))` — both operands cast to the output dtype `T` first, then one
rounding to `T` — which is not the CPU kernel's formulation. Every
integer-valued step is consequently bit-exact (measured `max|diff|` 0.000e+00 for
`arange(1.5, 4.0)`, `arange(10., 0., -2.)`, `arange(0., 100., 7.)`,
`arange(0, 128, 2)`, `arange(4096)`, and all seven vendor dtypes), and a
fractional step agrees to **1 ULP and no better**, which is not a defect to fix:
the CPU kernel is not length-invariant, so there are no bits to match.
`torch.arange(0., 1., 0.1)[9]` is `0.8999999761581421` while
`torch.arange(0., 1000., 0.1)[9]` is `0.9000000357627869` for the same
mathematical element. The measured bound is 1.000 ULP on both a 10,000-element
(`arange(0., 1000., 0.1)`, `max|diff|` 6.104e-05) and a 1,000,000-element
(`arange(0., 1., 1e-6)`, `max|diff|` 5.960e-08) range, so the error does not
grow with length: casting the step to `T` perturbs it by at most 2\*\*-24
relative, which over `i` elements is about one ULP of the largest value in the
range. Every arange Qwen-Image issues is integer-valued — `arange(0, dim, 2)`
for the block frequencies and `arange(4096)` for the position table — so the
pipeline is on the exact side of that line.

**The five in-place writes.** `zero_` to `topsatenZero`; `fill_.Scalar` and
`fill_.Tensor` to `topsatenFill_`, the scalar staged through
`ToTopsatenScalar`; `masked_fill_.Scalar` and `masked_fill_.Tensor` to
`topsatenMasked_fill`, called with `out` aliasing `self` as the vendor expects
for an in-place form. Measured on the S60 at both the transformer's
`(1, 4096, 24, 128)` and `(16, 128, 128)`, including the empty `(0,)` case, all
three float dtypes, the int64 host path for `zero_`, and a transposed input for
`masked_fill_.Scalar`; every one reports its `-> gcu` dispatch line and matches
the CPU reference. `torch.zeros`, `torch.ones`, `torch.full`, `torch.zeros_like`
and `torch.ones_like` were then re-run and each reaches the device with the right
values through this path; `zeros_like`/`ones_like` do not decompose at all, since
`zeros_like` and `ones_like` are themselves claimed. The three gap entries for
`zero_`, `fill_.Scalar` and `fill_.Tensor` **stay** in
`NATIVE_TRITON_GAPS["gcu"]`, and `arange`'s with them: a gap entry only takes an
op off FlagGems, and where it goes next is the native side's decision, so
deleting the entry would put `arange` back on the FlagGems path that fails to
compile. For the factory family the gap is what *selects* the vendor.

**The two complex views** (`view_as_real`, `view_as_complex`). These are
metadata-only — sizes, strides and dtype are recomputed in `at::native`'s
`ComplexHelper.h` over the input's own storage — and they are therefore the one
place in this change where a handwritten kernel is legitimate: every category
template in `scripts/codegen/codegen_gcu.py` ends in a `topsaten::` call, so the
generator cannot express an op that moves no data, and it lists both in
`METADATA_OPS` to get them into `gcu_register.inc`. That is the concrete codegen
limitation CLAUDE.md requires for a handwritten kernel, and the registration in
`csrc/aten/strided_ops.cc` carries the human approval recorded in the PR. The
consequence of leaving them on the host path was not latency but correctness: a
view op that reaches `cpu_fallback` is *copied*, so the result silently stops
aliasing the input. Measured on the S60: `view_as_complex` returns
`torch.complex64` with a matching shape, stride and `data_ptr` against both the
source and the CPU result, a write through the real view is visible through the
complex one, `view_as_real` keeps its 3-D strides and its non-contiguous layout,
the whole RoPE sequence (`randn`, `view_as_complex`, `mul.Tensor` against a
complex frequency table, `view_as_real`) logs `-> gcu` at every step and stays on
`flagos:7` with `max|diff|` 0.000e+00 against a CPU round trip, and the two error
contracts (`odd stride`, bad last dimension) raise the same message ATen does.

**`linalg_vector_norm`** is served by `topsatenLinalgVectorNorm`, which takes the
dim list and the order directly. The vendor writes the `keepdim=True` shape, so
the shape ATen promises is reached with a metadata `reshape`. This supersedes the
claim two entries below that "there is no topsaten kernel behind
`linalg_vector_norm`, so that gap lands on `none`": the vendor API was there and
the missing piece was the codegen. Measured on the S60, with a CPU reference:
the VAE decode call site `(1, 128, 1, 1024, 1024)` ord=2 dim=[1] now reports
`linalg_vector_norm -> gcu` at `max|rel diff|` 5.048e-07 where it previously
reported `cpu_fallback`, `F.normalize(x, dim=1)` and `(dim=-1)` each match the
CPU result at 1.192e-07 while logging `-> gcu`, and the sweep covers
`ord ∈ {1, 1.5, 2, 3, inf, -inf, 0}`, `dim` as a list, a negative index and
`None` (rank-0 output, exactly matching), keepdim and no-keepdim, a
non-contiguous input, and the empty case. `float16`, `bfloat16` and `float64`
operands take the host path and are exact.

**`_softmax` is not part of this change**; the launch-grid entry below moved it
to native `gcu` in the same branch. It is named here only because the counts
that follow include it.

**Route delta.** Measured against the committed conf at `082afa8`, because both
this change and the launch-grid entry below are uncommitted in the same tree and
no commit contains the state between them: GCU `flaggems` 257 -> **255**, `gcu`
144 -> **156**, `none` 1635 -> **1625** over 2036 routable ops, so accelerated
routes go **401 -> 411** (19.7% -> **20.2%**). The `gcu` gain of 12 decomposes as
`_softmax` from the launch-grid work plus the 11 overloads claimed here, and the
`flaggems` loss of 2 is `_softmax` **and** `linalg_vector_norm`, the latter of
which left FlagGems for `none` in the launch-grid entry and is now on `gcu`
instead. `gcu_register.inc` grows by 11 `m.impl` lines (152 -> 163) — `arange`,
`arange.start`, `arange.start_step`, `fill_.Scalar`, `fill_.Tensor`,
`linalg_vector_norm`, `masked_fill_.Scalar`, `masked_fill_.Tensor`,
`view_as_complex`, `view_as_real` and `zero_` — and `gcu_flaggems_register.inc`
shrinks from 249 to 248 lines by exactly one entry: `linalg_vector_norm` was
registered in the generated FlagGems file and is now registered natively, so it
moves from that file's first list to its second, the FlagGems ops
`gcu_register.inc` already claims. Both are checked by diffing the sorted
`m.impl` op-name lists, not by reading the file's banner. The `flaggems` route
count is unchanged because nothing moved *from* `flaggems`: `linalg_vector_norm`
was on `none` after the launch-grid entry. These are this change's own counts;
the `out=` entry below moves three more overloads, and the index/embedding entry
at the top of this section moves twelve, leaving `gcu` 171, `none` 1610 and
accelerated 426 (20.9%). Four artifact hashes, before -> after, all taken from
this worktree. The third column is the same four files as the two later entries
leave them: both are uncommitted in this same tree, so it is the column a reader
of the tree sees, while the middle one is what this change alone produced. All
three agree on `gcu_flaggems_register.inc`, which neither later entry touches.

| Artifact | At `082afa8` | After this change | In the final tree |
|---|---|---|---|
| `torch_fl/configs/backends_gcu.conf` | `4c5082d60295b3134b084c349aedbb9d9c2e449afcd834143d79e9496b09b8f1` | `f7c2e309df4265246a3f17e3800c5b2529ac1110fa6675a86222a4bc2e648ce0` | `366d986f0c7aa292eb398e9de8b3dd9a1533240fc4f026472299a3560245e40d` |
| `gcu_kernels.cc` | `01437b68764987bb64a01e1f608cf45197f79e47e3cc7081d113984b6365237e` | `947c1b113f9656abe67ab19eb49bf23097b2be12fc0dd23ba321114c706beffe` | `7f34288a4af8b3da7231c600eea8446c7870fa7312d2bb369a5585de16e3a517` |
| `gcu_register.inc` | `756f6134ea148f7450886cf9b3dba87930728eba9a7359e13b3a639d8f76eb5b` | `c6526ed3cf885d8209970a20913fd4b82352d6c55b537faf3e89eab9afa3f12a` | `806d7ce79d075060f53a57c9f8982238e0925d9098da8ffd18a5b6f31a84a6be` |
| `gcu_flaggems_register.inc` | `a02b46d91c507eac2fee34ef3105046a6ad7ed2d37bf9c509d3f668a6671f9ee` | `9c9c99d0d2eab6ba158090ef16fff3566422d688b850a50e6bcc97568d2e0b71` | `9c9c99d0d2eab6ba158090ef16fff3566422d688b850a50e6bcc97568d2e0b71` |

**Scope boundary.** The `out=` forms of these operators (`arange.out`,
`arange.start_out`, `linalg_vector_norm.out`, `_softmax.out`,
`view_as_real_copy.out`, `view_as_complex_copy.out`) stay on `none`. The
pipeline does not call them, and each would need its own kernel signature, so
they are deliberately left rather than implied to be covered by the
non-`out` claims above.

**Generator idempotency.** All three generators run twice produce byte-identical
output, and both `--check` modes exit 0. The edit to `codegen_gcu.py` in the
second pass was comment-only, which the hashes above confirm: they were taken
after the regeneration that followed it.

**Evidence gaps.**

- Two of the six operators — `fill_` and `linalg_vector_norm` — are the only ones
  whose routing was re-measured at a *call site the pipeline actually uses*
  rather than at a probe's own shapes. `zero_`, the two complex views and
  `arange` are measured at their own shapes, at the same sizes
  (`(1, 4096, 24, 128)` for the transformer, `(16, 128, 128)` for the VAE) and
  the same dtypes, but through the probe rather than through the model.
- `arange`'s fractional-step agreement is a bound, 1 ULP, and is measured at the
  two lengths above plus the five fractional cases in the probe. It is not a
  proof over all lengths; it is a proof that the error does not grow with length
  over the three lengths that were measured (10, 10k, 1M).
- The probe that isolated card 5 was run against `flagos:0`, `flagos:3` and
  `flagos:6` only. Device 5 on this host hangs any topsaten op, including
  `sum` and other kernels that predate this change, so nothing here was measured
  on it and a defect specific to that device is not ruled out for the new
  kernels either — it is simply indistinguishable from the pre-existing one.
- No S60-wide survey was re-run, and none was needed for the FlagGems half:
  no `flaggems` route was added, removed or changed by this change, and
  `flaggems_overload_survey.py` measures FlagGems routes only. The 255 remaining
  `flaggems` routes are exactly the ones the launch-grid entry measured.
- The Qwen-Image census is one pipeline at one shape, one seed and one prompt.
  It is the strongest evidence in this entry that the six operators are the whole
  of the host traffic, and it is still one workload.
- Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are **not revalidated** and no
  route changed for them. Ascend has the identical `view_as_real` /
  `view_as_complex` gap in its own conf (both route to `none` there) and was left
  as it stands rather than widened from a measurement taken on the S60.

### Enflame GCU S60 native kernels for the `out=` forms of the arithmetic operators (2026-09-17)

The entry above claims a kernel for every operator a *Qwen-Image-2512 inference*
census saw reach `cpu_fallback`. This entry is the same kind of census run
against a *training* step, and it found a second, disjoint set: the optimizer's
in-place arithmetic. Two Adam steps over a five-parameter model, run once with
`foreach=True` and once with `foreach=False`, log **25 `aten::add` and 10
`aten::mul` `cpu_fallback` calls** — the state updates (`exp_avg.lerp_`,
`exp_avg_sq.mul_(beta2)`, `denom.add_(eps)`) — a device -> host -> device round
trip per parameter per step, and the whole of the fallback traffic the optimizer
itself produces.

**Why they were invisible as `add` and `mul`.** ATen gives `add_.Tensor`,
`sub_.Tensor` and `mul_.Tensor` no `PrivateUse1` registration of their own: they
are structured in-place ops whose composite redispatches onto `add.out` /
`sub.out` / `mul.out` with `out=self`. `csrc/aten/fallback.cc` logs
`op.schema().name()`, which is the schema and not the overload, so every one of
them — in-place, scalar, explicit `out=`, every overload — prints as
`aten::add`. The census therefore reads as "the out-of-place `add`, which is
already native" unless the conf is consulted, and in that conf `add_.Tensor` and
`add_.Scalar` are `none` for a different reason: the dispatcher never sees the
in-place spelling, so those two lines are inert for the arithmetic path. The
route that decides is `add.out`. One kernel per operator consequently serves
both spellings and every explicit `out=` caller in the stack.

**The three new kernels.** `mul.out` takes a new `binary_out` category template
(`topsatenMul(out, self, other)`), and `add.out` and `sub.out` a new
`binary_alpha_out` one (`topsatenAdd` / `topsatenSub` with the alpha staged
through `ToTopsatenScalar`). Measured on the S60 at 4096x4096 float32, an
in-place `add_` costs **0.507 ms/call** on the device path against **114.185
ms/call** for an int64 operand, which takes the host round trip by construction
— **225x**, and that int64 case is the in-run control: it is what says the fast
branch is the device one rather than a short-circuited one.

**The `out=` contract.** An `out=` kernel sizes the output: a caller may legally
pass a zero-sized (`torch.empty(0)`) or differently-shaped tensor and expects
the result's shape back, which is what an `out=` caller in this stack does after
a `resize_`. Both branches therefore resize `out` to the result shape before
writing, and both do it after the `out` argument has been checked, so a rejected
call leaves `out` untouched.

**Correctness on the S60.** All nine in-place spellings (`add_` / `sub_` / `mul_`
with a tensor, with alpha, with a scalar, and the self-aliasing `add_(t)`),
both broadcast directions, fp16, bf16, int32, an int64 operand on the host path,
a strided non-contiguous `out`, and the explicit `out=` forms — from a zero-size
`out`, to a different shape, widened to `float64`, and broadcast — match a CPU
reference exactly, each reporting its `-> gcu` dispatch line and no
`cpu_fallback`. `mm.out` and `addmm.out` measure 9.537e-06 and 7.629e-06 against
the CPU at float32, which is reduction order and not route: their `float64`
siblings, which cannot take the device path and are computed the same way as the
reference, are exact 0.0.

**The cast guard.** ATen refuses an `out` it cannot be cast into rather than
truncating into it, and the in-place spellings inherit that rule through
`add.out`: `int32_tensor.add_(float_tensor)` raises on the CPU with `result type
Float can't be cast to the desired output type Int`. Each `*_out` kernel runs
`c10::canCast(result_dtype, out.scalar_type())` first — ahead of the
unsupported-dtype branch, so that path is covered too, and ahead of the resize —
and raises the same message. It is load-bearing rather than defensive: the host
branch ends in an ordinary `copy_`, which casts, so before the check
`torch.mul(f32, f32, out=int32)` measured `max|diff|` 1.19e9 instead of raising.
Measured after it, `torch.mul(a, b, out=int32)` is rejected on the device with
the CPU's own message.

**The device guard.** Every `*_out` kernel opens with `gcu::TopsDeviceGuard
out_guard(self)` — including the three that predate this change (`bmm.out`,
`mm.out`, `addmm.out`), which also gain the cast guard and a
`out.scalar_type() != result_dtype || !out.is_contiguous() || out.device() !=
self.device()` check before their topsaten call. Those three previously handed
topsaten a descriptor built from `out` without consulting its dtype, contiguity
or device, and built one without any guard on the current device. The guard is
needed because `resize_` takes no device argument and installs no guard of its
own: it allocates through the block pool, which resolves the device from the
*current* one at call time, so a multi-device process sitting on device 0 that
grows a `flagos:6` `out` takes the block out of device 0's pool and topsaten then
refuses the descriptor — `FindMemObj DeviceId[0] of memory VA[...] is not match
for DeviceId[6] of stream`. Measured: `torch.add(a, a,
out=torch.empty(0, device="flagos:6"))` failed with the current device at 0 and
passed once device 6 was current. The guard is a no-op when the current device
already matches.

**The `resize_` crash, reached from this path and fixed here.** `out.resize_(...)`
is the first thing the new kernels do when the caller's `out` is the wrong
shape, and growing a flagos tensor that already had content crashed with
SIGSEGV. The defect is neither new nor specific to `out=`: any `Tensor.resize_`
on a non-empty flagos tensor did it, but nothing in the pipeline had grown one
before. `at::native::flagos::resize_` delegated to `at::native::resize_`, the
**CPU** implementation, and its growth helper `maybe_resize_storage_cpu` ->
`resize_bytes_cpu` allocates the new block through the storage's own allocator
(correct here — a flagos storage carries the PrivateUse1 allocator) and then
moves the existing bytes with a **host** `memcpy`: `libtorch_cpu.so+0x2164723:
call memcpy@plt`, called from `resize_bytes_cpu+0x168`, against bytes that live
in HBM. That is why growing from empty never crashed — with an empty new shape
the helper returns before it allocates or copies anything — and why neither a
shrink nor an unchanged shape reached it. CUDA does not have this problem because
it has its own `ATen/native/cuda/Resize.h` with `maybe_resize_storage_cuda` /
`resize_bytes_cuda`, which performs that copy with `cudaMemcpy`. The fix is those
same two steps written for this backend: `maybe_resize_storage_flagos` in
`csrc/aten/strided_ops.cc` computes the new byte count with ATen's own
`computeStorageNbytesContiguous`, allocates through the storage's allocator under
a `c10::DeviceGuard`, moves the old bytes with `Allocator::copy_data` — the
device memcpy, `CachingDeviceAllocator::copy_data` -> `MemcpyDeviceToDevice` —
and installs the block with `set_data_ptr_noswap` / `set_nbytes`.
`at::native::resize_` then runs as before and finds the storage already large
enough, so `resize_bytes_cpu` is never entered.

**Correctness of the growth, measured on the S60** at `flagos:0` and `flagos:6`,
one fresh process each, eight case groups per device, all pass. Growing a filled
`(4, 16)` to `(32, 64)` went from `rc=-11` (SIGSEGV) to `rc=0`, with the first 64
elements `torch.equal` to the pre-growth contents, the storage at 8192 or more
bytes, the tensor still on `flagos` and usable afterwards. The same holds for a
growth from empty, a shrink, an unchanged shape, `float16` / `bfloat16` /
`int32` / `int64`, and a slice with a non-zero `storage_offset` — which keeps its
prefix and leaves its base tensor untouched. A live neighbour tensor's contents
survive the reallocation. The `out=` growth cases are among them: `out=` from
empty, `out=` to a wider shape, and `out=` widening a dtype.

**The `copy_` cross-dtype defect, also reached from this path and fixed here.**
The host branches of the new kernels end in `out.copy_(...)`, and the widening
`out=` case makes that copy cross dtypes. `_copy_from` in `csrc/aten/copy_ops.cc`
issued every `Memcpy` with the byte count taken from the **source**, so a
differing dtype gave the driver a count that did not match the destination's
element width. Measured before the fix on the S60: `torch.empty(6,
dtype=torch.float64, device="flagos:6").copy_(torch.arange(6) + 0.5)` returned
`[0.125, 128.0, 4096.0, ...]` — two float32 values reinterpreted as one float64,
`max|diff|` 4.094e+03 — the `dev f32 <- cpu f64` direction gave `max|diff|`
4.500e+00 and `dev i32 <- cpu f32` 1.085e+09, while the same-dtype control was
exact. The device -> host direction took its byte count from the source by the
same rule. The fix converts a CPU source to the destination's dtype on the host
before the memcpy, and lands a cross-dtype device -> host copy in a CPU tensor of
the *source's* dtype before letting the host `copy_` cast into `dst`. All eleven
cases of the probe now read `max|diff|` 0.000e+00: host -> device widening and
narrowing, device -> host in both directions, a strided host destination,
same-device `dev f16 <- dev f32`, and the `.to(device, dtype)` spelling, which
was already correct and is the control.

**Route delta.** Measured against the state after the six-fallback change above,
because both are uncommitted in the same tree: GCU `flaggems` stays **255**,
`gcu` 156 -> **159**, `none` 1625 -> **1622**, so accelerated routes go
**411 -> 414** (20.2% -> **20.3%**). The three are `add.out`, `mul.out` and
`sub.out`. `gcu_register.inc` grows by three `m.impl` lines (163 -> 166) and
`gcu_flaggems_register.inc` is unchanged at 248 — no `flaggems` route was
touched — so the reconciliations above still hold as `255 = 248 + 7` and
`159 = 166 - 7`. Three hashes move and the fourth does not. The full hashes are
in the table in the entry above; the before column here is that table's middle
column. These are this change's own counts: the entry at the top of this section
moves twelve more overloads and leaves `gcu` 171, `none` 1610 and accelerated 426
(20.9%).

| Artifact | After the six-fallback change | In this change |
|---|---|---|
| `torch_fl/configs/backends_gcu.conf` | `7b87541ebfd2fcc159ffff60edc213c3d4ce135215fdb40a9acbcd0dc850fdb3` | `74aab449194a7495b6a93725bb66a2ccd3f53c2d3a2505d7ab46e1f171e7aefc` |
| `gcu_kernels.cc` | `947c1b113f9656abe67ab19eb49bf23097b2be12fc0dd23ba321114c706beffe` | `35c8a37e80f82f377ab671e32b11a2c488b246c200a37b16a462dcd181f6b4d0` |
| `gcu_register.inc` | `c6526ed3cf885d8209970a20913fd4b82352d6c55b537faf3e89eab9afa3f12a` | `f961aff0c9c1db8dbbc1166f8be58a43329c2917e833de0c618d7c32584ae99f` |
| `gcu_flaggems_register.inc` | `9c9c99d0d2eab6ba158090ef16fff3566422d688b850a50e6bcc97568d2e0b71` | unchanged |

**The census, re-run against this change.** The same training probe, post-fix,
logs **11 `cpu_fallback` calls instead of 45**, with no `aten::add` and no
`aten::mul` at all: the 25 and the 10 are now 25 `add.out -> gcu` and 10
`mul.out -> gcu` dispatch lines. The two losses are unchanged to the printed
digits against the pre-fix run (-32.7952 and -51.3676), so the rerouted optimizer
updates produce the same numbers the host path did rather than merely stopping
being slow. The eleven that remain are the training path's next set and are
**not** part of this change: `index_select` 4, `embedding_dense_backward` 4,
`index_fill_` 2 and `nonzero_static` 1.

**Scope boundary.** The remaining `out=` forms stay on `none` — among them
`arange.out`, `arange.start_out`, `linalg_vector_norm.out`, `_softmax.out`,
`fill.Scalar_out`, `fill.Tensor_out`, `masked_fill.Scalar_out`,
`masked_fill.Tensor_out`, `view_as_real_copy.out` and
`view_as_complex_copy.out`. Nothing in the two censuses calls them, and each
would need its own kernel signature. The conf's `add_.Tensor`, `add_.Scalar`,
`mul_.*`, `sub_.*` lines also remain `none` and are inert for the arithmetic
path, for the dispatch reason in the second paragraph; they are not evidence that
the in-place ops are uncovered.

**Evidence gaps.**

- The probes are ad-hoc scripts run out of a scratch directory and are not
  committed to the repository, the same practice as the six-fallback entry
  above. The route counts and the artifact hashes they are read against are
  reproducible from the tree; the per-case readings are not.
- The device -> host half of the `copy_` dtype fix is measured after the fix
  only. The host -> device half was measured both before and after. The device
  -> host defect is read off the same source-derived byte count in the same
  function rather than from a pre-fix run of that direction.
- The pre-fix and post-fix readings of the `resize_` crash and of the `copy_`
  values are single-shot: what is reported is the change between two builds of
  this tree, not a sweep over configurations.
- Device 5 on this host hangs any topsaten op that touches a tensor resident
  there, `resize` and `sum` included, and that predates this change. Every
  reading above is from cards 0 and 6, so a defect specific to card 5 is not
  ruled out for the new kernels — only indistinguishable from the pre-existing
  one.
- The optimizer census is one model, one optimizer, two steps and one seed. It
  establishes that the chosen set is a real training-step cost and that the new
  kernels remove it; it does not enumerate every in-place op a larger training
  stack would use.
- Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are **not revalidated** and no
  route changed for them. The `resize_` bug, the `copy_` defect and the
  device-guard requirement are in shared flagos code rather than in a
  GCU-specific file, so every platform whose PrivateUse1 backend reaches
  `at::native::flagos::resize_` had the same crash and this fix applies to it,
  but no other platform's hardware was available to confirm that and no platform
  is claimed here.

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
`m.impl` lines) takes the ops topsaten has a kernel for, the new file (248
`m.impl` lines) takes the rest of the shared FlagGems coverage. An op routed to
`flaggems` that neither file claims would reach the dispatcher with an empty
`kFlagGems` slot and raise `backend not registered` instead of falling back, so
generation gates every accelerated route on the registration set.

**Route delta.** GCU `flaggems` 0 -> **255**, `gcu` 152 -> 145, `none`
1884 -> 1636 (2036 routable ops), so accelerated routes go from **152 to 400**
(7% -> 19.6%). Seven of the 255 `flaggems` routes (`clamp`, `fmod.Tensor`,
`gelu`, `mean`, `mean.dim`, `remainder.Tensor`, `silu`) are claimed natively by
`gcu_register.inc` rather than by the new `.inc`: `codegen_gcu_flaggems.py`
computes its list as `(FLAGGEMS_PYTHON_OPS - gaps) - native - pending` and omits
them deliberately, because a second `m.impl()` for the same op and dispatch key
would only override the first and warn at import. That is why the file has 248
lines rather than 255, and why those seven are routed `flaggems` in the conf
while the native kernel serves them.

**Why 227 ops are gapped.** `NATIVE_TRITON_GAPS["gcu"]` grew from 108 to 227
entries. Every addition is a measured failure on the S60, not an inference, and
they fall into five families:

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
- **Launch-grid axis limits, correct at every dtype and broken at scale** (2 ops,
  `_softmax` and `linalg_vector_norm`). This family was *added after* the
  seven-profile sweep and is invisible to it: every profile that harness
  constructs is at most 32 rows wide, and the defect is in the grid the enflame
  backend derives from a data dimension. GCU's limits are far below CUDA's --
  `grid.x` 65535 against `2**31-1`, `grid.y` 255 against 65535 -- so a kernel
  launched once per row (or per 8-row tile) works in the sweep and raises on a
  real workload. `_softmax`'s `softmax_kernel_inner` launches with
  `grid = (M, 1, 1)`, which passes at `M = 65496` (`(1, 24, 2729, 2729)`) and
  fails at `M = 65544` (`grid.x Required 65544`) and `M = 98736`
  (`grid.x Required 98736`); its non-inner kernel has the same defect on the
  other axis (`grid = (M, cdiv(K, TILE_K), 1)`, so `(24, 4114, 4114)` over dim 0
  fails with `grid.y Required 2067`). `linalg_vector_norm`'s `l2_norm_kernel`
  launches with `grid = (cdiv(M, BLOCK_M),)` and the enflame tune config for it
  offers only `BLOCK_M` in {1, 2, 4, 8}, so no config fits `M` above
  65535 * 8 = 524280. Both were reached from Qwen-Image-2512 on the S60:
  `_softmax` on `(1, 24, 4114, 4114)` from `F.scaled_dot_product_attention` in
  every one of the 60 transformer blocks, and `linalg_vector_norm` via
  `F.normalize(x, dim=1)` in diffusers' `QwenImageRMS_norm` on the VAE decode at
  `(1, 128, 1, 1024, 1024)`, `M = 1048576`, failing with
  `grid.x Required 131072`. `topsatenSoftmaxForward` has no grid to size and
  serves every shape above, so `_softmax` goes back to the vendor kernel. There
  is no topsaten kernel behind `linalg_vector_norm`, so that gap lands on `none`
  and the call reaches `cpu_fallback`; topsaten does carry `topsteL2Norm`, but
  `topste` is not wired into this generator, so the missing piece is the codegen,
  not the vendor API.

  **Superseded for `linalg_vector_norm`** by the entry above, written later in the
  same branch: the generator now emits a kernel for it against
  `topsatenLinalgVectorNorm`, so the op routes to `gcu` rather than `none` and the
  host round trip described here no longer happens. The `_softmax` half of this
  family is unchanged. The paragraph is kept as written because the measurement
  it records — the grid limits, the failing shapes, the call sites — is what
  identified the defect; only its routing conclusion is out of date.

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
(`039fb323…`) nor the one this change ships (`10a85c9e…`) -- but it reconciles
with the shipped conf: 374 - 117 - 2 = 255, where 117 = 81 + 36 is the number of
ops the measurement returned to the vendor and the further 2 are the launch-grid
family above, added after the survey and outside its profile range by
construction. Per-op results survive in the survey JSON. The environment was
Python 3.12.13, CPU PyTorch 2.10.0,
flagtree `0.6.1+enflame3.6` (Triton 3.6, backend `enflame`) and FlagGems master
`3c6f7537d`. Full per-op evidence and the raw failure families are in
[docs/vendors/gcu/flaggems-test-results.md](../vendors/gcu/flaggems-test-results.md).

**The two launch-grid reroutes were re-measured on the S60 after regeneration**
(`bash /tmp/group_c_probe.sh`, `FLAGOS_LOG=dispatch FLAGOS_LOG=fallback`),
against the conf this change ships, at the shapes that broke FlagGems:

| Call | Shape | `M` | Route taken | Result |
|---|---|---:|---|---|
| `torch._softmax(x, -1, False)` | `(1, 24, 2729, 2729)` bf16 | 65496 | `_softmax -> gcu` | `max|diff|` 6.10e-05 |
| `torch._softmax(x, -1, False)` | `(1, 24, 2731, 2731)` bf16 | 65544 | `_softmax -> gcu` | `max|diff|` 6.10e-05 |
| `torch._softmax(x, -1, False)` | `(1, 24, 4114, 4114)` bf16 | 98736 | `_softmax -> gcu` | `max|diff|` 3.05e-05 |
| `F.normalize(x, dim=1)` | `(1, 128, 1, 1024, 1024)` bf16 | 1048576 | `linalg_vector_norm -> cpu_fallback`, `clamp_min -> flagos_python`, `div.Tensor -> gcu` | `max|diff|` 0.00e+00 |

The three `_softmax` cases are the ones that previously raised
`RuntimeError: grid.x Required 65544` / `98736`; the `F.normalize` case is the
one that previously raised `grid.x Required 131072` inside the VAE decode. The
`linalg_vector_norm` step now reaches the boxed `cpu_fallback`, which is the
documented cost of having no topsaten kernel wired into the generator.

**The fourth row's `linalg_vector_norm` route is superseded** by the entry above,
which was written later in the same branch: the same call site now reports
`linalg_vector_norm -> gcu` at `max|rel diff|` 5.048e-07 against a CPU reference,
because the generator gained a kernel for it. The `F.normalize` result and the
VAE statistics below are unaffected; they were measured through the
`cpu_fallback` route, and the same call is now served on the device.

The third row is also the current Qwen-Image-2512 VAE evidence on the flagos
route: the whole decode stage runs at `mean=0.3475 std=0.2618` against the
vendor stage's `mean=0.3505 std=0.2635`, and a paired run that shares one latent
tensor between the two backends matches at **MAE 0.1646, PSNR 55.90 dB, max 6.0**
(8-bit).

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
(`gcu_flaggems_register.inc` `f8b73835…`, `backends_gcu.conf` `10a85c9e…`) and
both `--check` modes exit 0. The banner line naming the generator's own path is
the only thing the merge with `flagos/main` moved in either file, so the hashes
recorded before that merge (`39cd03e3…` and `10b8ab4b…`) differ without any route
or registration changing. The two hashes above were taken after the launch-grid
gap was added; the pair recorded before it (`a02b46d9…`, `4c5082d6…`) differs
from these by exactly the two reroutes that gap describes.

**Evidence gaps.** Five, all recorded rather than papered over:

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
- The launch-grid family is measured through its two Qwen-Image callers and the
  shapes above, not through the seven-profile survey, which cannot reach it. The
  boundary on `grid.x` is bracketed for `_softmax` at 65496 / 65544 and for
  `linalg_vector_norm` derived from the tune config rather than measured at
  524280. The 254 remaining `flaggems` routes carry no equivalent boundary test,
  so a kernel with the same defect and no Qwen-Image caller would not have been
  found. One reading during the re-measurement reported `nan` for the
  `F.normalize` case and has not been reproduced: eight consecutive repetitions
  of that exact sequence (three large `_softmax` calls, then `F.normalize` at
  `M = 1048576`) came back bit-identical to the CPU reference, so the single
  `nan` is recorded here as unattributed rather than explained.

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
registration for GCU took the platform to 255 `flaggems` / 145 `gcu` / 1636
`none` and 400 registered ops. See "Enflame GCU S60 FlagGems routing
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
  registration set and none registered-but-left-`none`). GCU is 400/400 as of
  2026-09-17.
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

### `scaled_dot_product_attention` moved onto DTK's CUTLASS flash adapter (2026-09-17, Hygon DCU)

The DCU boxing path already carries the composite override in
`csrc/aten/sdp_choice_stub.cc`: it boxes q/k/v to CUDA and calls the native
composite, which then consults *DTK's* selector (`at::cuda::_fused_sdp_choice`).
What that selector answers is decided by
`at::globalContext().getROCmFAPreferredBackend()`, and on a decoupled wheel that
member is `Default`: `at::ROCmFABackend` is DTK's to define and lives in
`libtorch_hip.so`, while the `Context` singleton holding the member belongs to the
*official* CPU `libtorch_cpu.so`, which does not carry DTK's enum. Every fused
call therefore landed in the aotriton branch DTK did not compile —

```text
RuntimeError: Non't compile aotrition fa, please compile aotriton fa before use it
```

reached from `transformers`' `sdpa_attention_forward` — and memory-efficient
attention is not compiled at all ("USE_MEM_EFF_ATTENTION was not enabled for
build"). The shim had been disabling both fused backends up front and letting the
composite fall through to the math decomposition, which is the route the previous
revision documented. It now reaches the vendor fused leaf instead.

The mechanism is one member write. DTK's hipified `flash_api.h` reaches its
CUTLASS adapter only when that member equals `at::ROCmFABackend::Cutlass`, whose
ordinal is `1` in DTK's `{Default, Cutlass, AOTriton, Ck}`. Upstream's enum is
`{Default, AOTriton, Ck}`, where the same literal is `AOTriton`, so
`_prefer_dtk_cutlass_flash()` in `torch_fl/accelerator/dcu/_dcu_compat.py`
resolves the value by name when the binding is DTK's own and by ordinal when it is
upstream's, then writes it through
`torch._C._set_rocm_fa_preferred_backend`. DTK's `cutlassfa_adapter.h` `dlopen`s
`flash_attn_2_cuda*.so` (`FA_SO_PATH` first, else three directories above the
mapped `libtorch_hip.so`) behind a hard `TORCH_CHECK` rather than a decline, so
`_dtk_flash_attn_lib()` probes for that file first and the shim stays on the math
decomposition when it is absent. `FLAGOS_DCU_SDPA_FLASH=0` forces the math-only
behaviour for a stack whose adapter misbehaves.

Measured on this host (DTK 6.3.26113, torch 2.10.0+cpu, `flagos:7`, bf16, no
mask, no causal, no explicit scale, no dropout, no gqa), one process per arm,
the same call `torch.ops.aten.scaled_dot_product_attention.default`, median of
five calls after two warm-up calls, every window closed by
`torch.cuda.synchronize(flagos:7)`:

| SDPA route | `(1, 24, 4114, 128)` | `(1, 24, 12576, 128)` |
| --- | --- | --- |
| math decomposition (`FLAGOS_DCU_SDPA_FLASH=0`) | 16.7 ms | 167.5 ms |
| DTK CUTLASS flash adapter (shipped conf) | **1.3 ms** | **10.2 ms** |
| FlagGems Triton (`FLAGOS_OP_scaled_dot_product_attention=flaggems`) | 3.2 ms | 25.9 ms |

`(1, 24, 12576, 128)` is the per-head shape Qwen-Image-2512's 1664×928 pass
runs. Peak device memory for the two shipped-route arms was 33,865 MiB against
463 MiB, sampled off-device with `rocm-smi` as the card's own counter rather
than the allocator's reserved/peak figures.

**Two figures this section carried were corrected on 2026-09-18, because the
harness behind them was wrong, not because a route moved.** They came from a loop
that closed each window with `torch.cuda.synchronize()`, which on this stack
synchronizes `torch.cuda.current_device()` — and `current_device()` is `0` while
the tensors live on `flagos:7`, so it returns immediately. Rerun through
`tests/manual/dcu_sdpa_ab.py --routes cuda --sync-check --device flagos:7`, which
adds the fourth window that isolates the cause:

```text
tensors_on=flagos:7 current_device_at_start=0
SYNC no-arg(current=0)=0.04ms sync(flagos:7)=10.18ms sync(flagos:0)=0.05ms no-arg(after set_device)=10.18ms
```

The no-argument form timed against the process default is `0.04 ms` — roughly
the idle-card number, and the one the old harness recorded; the same call after
`set_device(flagos:7)` is `10.18 ms`, agreeing with the explicit form. The math
decomposition measured 102.1 ms against a true 167.5, and the adapter 6.1 ms
against a true 10.2: both absolutes were low, while the ~16x ratio between them
happened to survive, because the math path submits enough kernels per call to
stall on queue depth even without a synchronization point. The peak-VRAM figures
are unaffected — those were sampled off the device while the op was held running,
and do not depend on the process ever waiting.

**A/B against the FlagGems composite route: not adopted on DCU (2026-09-18).**
PR #347 gave `aten::scaled_dot_product_attention` a `flaggems` route in
`backends_metax.conf`, and its conf regeneration wrote the op into
`backends_dcu.conf` as `cuda` — so DCU is the one other CUDA-boxing platform
where the same whole-op override is reachable, and the question of whether to
follow MetaX is a conf line plus a `DCU_COMPOSITE_FLAGGEMS`-style hold. It is
measurably not worth it:

* The adapter is **2.5x faster than FlagGems on both shapes** — 1.3 ms against
  3.2 ms at `(1, 24, 4114, 128)` and 10.2 ms against 25.9 ms at
  `(1, 24, 12576, 128)` (table above). At `(1, 24, 12576, 128)` that is 15.7x for
  the adapter and 6.5x for FlagGems, both against the same math decomposition.
* The attribution is kernel-level, not inferred from latency. The FlagGems arm's
  routed output is **bit-identical** to a direct
  `flag_gems.scaled_dot_product_attention(q, k, v)` call
  (`torch.equal` → `True`, `max|diff|` 0.000e+00), and the shipped arm's is not
  (`max|diff|` 9.766e-04), so the arm really took the route it was asked for and
  the latency difference is the kernel's. A profiled run of the shipped arm
  reports `aten::_flash_attention_forward` and
  `flash_fwd_kernel_16x64_prefetch<...cutlass::bfloat16...>` with zero
  triton/flaggems kernels.
* #347's MetaX case does not transfer. There the boxing route had no *reachable*
  fused kernel, so FlagGems was the only way to a fused one; on DCU DTK ships a
  CUTLASS flash adapter and the defect was that the composite never selected it.
  Once it does, the fused kernel the override was written to reach is already
  reached, and the override would only replace it with a slower one.

So `backends_dcu.conf` keeps `scaled_dot_product_attention = cuda`, and
`METAX_COMPOSITE_FLAGGEMS` stays MetaX-only. `FLAGOS_OP_scaled_dot_product_attention`
remains a working A/B knob on DCU — the FlagGems branch of the stub is compiled
into the DCU wheel (`FLAGOS_BUILD_FLAGGEMS=1`) and reachable through the
override — it is simply not the shipped route. The three arms above are
reproducible from the tree:

```bash
python tests/manual/dcu_sdpa_ab.py --device flagos:7
```

The numerics are the adapter's own, because the composite is intercepted rather
than reimplemented: `max|diff|` 4.9e-4 against `max|ref|` 0.109, i.e. bf16
rounding, and shapes DTK's selector refuses still fall back to its own math path
exactly as on the vendor route. `torch.backends.cuda.is_flash_attention_available()`
and `torch._C._can_use_flash_attention()` both answer `False` on this stack — in
the same process that ran the adapter, with the adapter's own warning printed —
so they are the official wheel's CPU-only bindings rather than part of the
decision, and are deliberately not patched.

No conf entry changed and no FlagGems route moved: the vendor fused leaf is
reached through DTK's selector inside the boxed composite, not through a routed
overload. Ascend, MUSA, GCU and BPU keep their own `_fused_sdp_choice` paths, and
every other platform is **not revalidated** by this change.

### Pointwise `add`/`sub`/`div` overloads rerouted to CUDA boxing (2026-09-17, Hygon DCU)

`backends_dcu.conf` carried `flag_gems` routes for eleven pointwise overloads:
`add.Tensor`, `add_.Tensor`, `sub.Tensor`, `sub_.Tensor`, `div.Tensor`,
`div_.Tensor`, `div.out`, `div.Tensor_mode`, `div_.Tensor_mode`,
`div.Scalar_mode` and `div_.Scalar_mode`. All eleven now route to the CUDA
boxing kernel in `backends_dcu.conf` only. They fail for two unrelated reasons,
both measured on the eight-device Hygon DCU bw1000 host, and the split is the
whole point of the entry: a single cause would have been wrong for four of them.

**Seven entries abort in the HCU backend's f64 conversion.** ATen boxes a Python
float as an f64 wrapped number, and `.Tensor` spellings are the overloads the
dispatcher reaches when one is present. With a bf16 tensor and a Python float
operand, the FlagGems pointwise kernel carries an `f64` alongside the `bf16`, and
`TruncFOpConversion::createDestOps` asserts:

```text
/root/codes/flagtree-hygon/third_party/hcu/lib/TritonHCUGPUToLLVM/ElementwiseOpHCUToLLVM.cpp:2358:
  ... Assertion `inElemTy.isF32() && "unsupported conversion"' failed.
```

The IR dump that follows shows the promotion directly —
`arith.extf %111 : tensor<...xf32> to tensor<...xf64>`, then
`arith.addf %112, %109 : tensor<...xf64>` and `arith.truncf` back to bf16 — and
the failure surfaces as `RuntimeError` at `triton/backends/hcu/compiler_hcu.py`
`make_llir` (`compiler_hcu.py:594`). This is the same defect GCU already carries
in `NATIVE_TRITON_GAPS["gcu"]` ("broken by ATen's float64 wrapped-number
boxing") and the one FlagGems tracks as issue #6212, reported for MetaX.

**Four entries die in FlagGems' `div_rn` shim.** `div.Tensor_mode`,
`div_.Tensor_mode`, `div.Scalar_mode` and `div_.Scalar_mode` fail with an
unrelated `CompilationError` whose cause chain ends in
`TypeError: cannot convert None of type <class 'NoneType'> to tensor`. Under a
named rounding mode — the `trunc` the harness passes, and the only mode that
reaches these kernels — `flag_gems/ops/div.py` routes to
`trunc_div_func`/`trunc_div_func_tensor_scalar`, whose entire body is
`return trunc(div_rn(x, y))`. The `div_rn` in that body is
`flag_gems.utils.triton_lang_extension.div_rn`, and `use_tl_extra` resolves it to
`triton.language.extra.hip.libdevice.div_rn` because the shim carries the symbol:
`triton_lang_extension.div_rn is triton.language.extra.hip.libdevice.div_rn` is
`True` on this host, so FlagGems' own `x / y` + `tl.floor` fallback never runs.
That symbol exists at the Python level but lowers to `None` in the kernel, so the
`trunc(...)` call receives `None`. The floor path dies the same way inside
`_float_floordiv`, one statement later, with
`unsupported operand type(s) for -: 'NoneType' and 'int'` on `q - 1`. This is the
failure mode FlagGems documents in its own source — `triton_lang_helper.py`
describes it for `asin` on another fork as "it exists at the Python level but
lowers to None, so the kernel fails to compile with *cannot convert None ... to
tensor*". With `rounding_mode=None` these four entries route to `true_divide`,
which is plain `x / y`, and run; the named mode is what breaks them.

Reachability in Qwen-Image-2512, which is what put this on the board:
`QwenImageRMS_norm` in diffusers' `autoencoder_kl_qwenimage.py` returns
`normalized * self.scale * self.gamma + self.bias`, and `self.bias` is the Python
float `0.0` whenever the layer is built without a bias (`nn.Parameter(...) if
bias else 0.0`) — which is how `norm_out` on the VAE decoder is built. With the
pipeline loaded as bfloat16 (`sweep.py`'s `torch_dtype`), that expression is
`add.Tensor(bf16_tensor, 0.0)`, the aborting case.

Both directions of the A/B are measured, on the fixed build, with
`tests/manual/dcu_pointwise_replay.py`, which replays one entry at a time onto
the FlagGems route through `FLAGOS_OP_<op>=flaggems` (dots doubled) and leaves
the other ten shipped:

| entry | FlagGems route (replayed) | shipped route |
| --- | --- | --- |
| `add.Tensor`, `add_.Tensor`, `sub.Tensor`, `sub_.Tensor`, `div.Tensor`, `div_.Tensor`, `div.out` | `RuntimeError`, HCU `TruncFOpConversion` assertion | `cuda` boxing, runs |
| `div.Tensor_mode`, `div_.Tensor_mode`, `div.Scalar_mode`, `div_.Scalar_mode` | `CompilationError`, cause `cannot convert None of type <class 'NoneType'> to tensor` | `cuda` boxing, runs |

Every one of the twenty-three entries in the family — the eleven pinned ones plus
`add.out`, `sub.out`, `div.out_mode`, `mul.Tensor`, `clamp`, `clamp_`,
`clamp_max`, `clamp_min`, `clamp.Tensor`, `clamp_.Tensor`, `div.Scalar` and
`div_.Scalar` — runs on the shipped conf. The entries deliberately left on
FlagGems are `div.Scalar`, `div_.Scalar` and the four `clamp*` spellings;
`mul.Tensor`, `add.out`, `sub.out` and `div.out_mode` were already on `cuda`.
Python `int` operands, f16/f32 tensor operands and tensor operands all stay off
this path.

**The 546-overload survey cohort cannot see either defect, which is worth
recording rather than glossing.** Two independent gaps, both confirmed by
running the survey on this host rather than inferred from the routing:

* `tests/manual/flaggems_overload_survey.py` enumerates the ops a conf *file*
  spells `flaggems` (`active_routes`), so pinning an op removes it from the
  survey's denominator by construction — a run against the shipped conf refuses
  the eleven pinned names outright. Against a copy of the conf with the pins
  reverted, all eleven report `PASS` on the FlagGems route on every profile the
  harness can synthesize (seven of seven for the `div` spellings; five of seven
  for the `add`/`sub` ones, whose `int64` and `bool` profiles ATen itself
  rejects), because none of the profiles is bf16 and every `.Tensor` operand the
  harness synthesizes is a tensor, not a Python float.
* The survey's `default_for` yields `rounding_mode=None` for the `*_mode`
  entries, which routes to `true_divide` and never touches `div_rn`. The same
  survey entry that measures `PASS` here fails under `rounding_mode="trunc"`.

So the pin is justified by the targeted probes above, not by the survey, and the
survey's passing verdict for these names should not be read as contradicting it.
The control arm — the entries left on FlagGems — measured `PASS` on the shipped
conf: `div.Scalar` 7/7, `clamp_max` 7/7, `clamp_min` 7/7, and `div_.Scalar` 5/7
with 2 `INVALID_CASE`. The four `clamp`/`clamp_` spellings, which are also left
on FlagGems, are `INVALID_CASE` for all seven profiles in that harness: it
synthesizes `min=None, max=None`, so ATen rejects the call before any device
kernel is reached. That is a synthesis gap in the survey, not evidence about
those routes; `tests/manual/dcu_pointwise_replay.py`'s sibling probe passes real
bounds and they run.

Regeneration preserves all eleven: `boxing_triton_gaps()` recovers the gap set
by diffing an existing conf against FlagGems coverage, so entries spelled `cuda`
survive, and `BOXING_GAP_NOTES["dcu"]` in `scripts/codegen/gen_vendor_confs.py`
carries the diagnosis above. `gen_vendor_confs.py --check` reports
`all vendor confs up to date` both before and after a regeneration on this
branch.

The generic `backends_flaggems.conf` cohort, the 546-overload bw1000 row, and
every non-DCU platform are **not revalidated** by this change. The eleven entries
are pinned in `backends_dcu.conf` alone; the generic and vendor configurations
still route them to FlagGems, and the two FlagGems-side defects are reported
against the DCU HCU backend where they were measured.

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

### DCU: `scaled_dot_product_attention` routed to FlagGems, converting the Qwen-Image-2.1 key-valid mask (2026-09-20, Hygon DCU bw1000)

`scaled_dot_product_attention` is a `flaggems` route in `backends_dcu.conf`, which
now reads **459 `flaggems` / 1578 `cuda`** over a **2037**-op list (SHA-256
`8849ce31ca6e517b6e7f71057f90dfa917f1b76068501554f40b7806d5600369`), superseding
the 458 / 1579 over 2037 (SHA-256
`7b82cb492de80f3dc2dcb93deeba14764a986368460a63cae9d1deb90fee9163`) recorded
below. The active route set moves with it, from SHA-256
`ededd42387eef3b74473eef358515a1848c153b3d83896c13168676332c3f69c` to
`b646c47b5d6643ed7a0ef753af24f402cf741d598ca3dca662946b89977288e6`. One line
changes -- `scaled_dot_product_attention = cuda` becomes `flaggems` -- no op loses
a route and every op in the list is still accelerated.

**Why the op cannot be routed leaf by leaf.** `aten::scaled_dot_product_attention`
is a composite, and the generated kernels this report otherwise counts are
leaves. Its fused-backend selection runs *inside* the composite and then branches
on `query.device().type()`, so on PrivateUse1 the leaves are never consulted and
routing them per-op can never reach a fused kernel. The override that fixes that
is hand-written -- `csrc/aten/sdp_choice_stub.cc` registers the composite itself
on PrivateUse1 and decides inside it -- and it is registered only on the
CUDA-boxing builds, which is why `EXTRA_ROUTED` in
[`scripts/codegen/gen_vendor_confs.py`](../../scripts/codegen/gen_vendor_confs.py)
exists: the op appears in no `.inc`, so no coverage scan can see it and the op
list has to be widened by hand. A second hand-written set,
`DCU_COMPOSITE_FLAGGEMS`, holds it from every other platform's conf exactly the
way `METAX_COMPOSITE_FLAGGEMS` does -- a measurement taken on one platform is not
a route for another, and `tests/unit/test_gen_vendor_confs.py` pins the
containment in both directions.

**What DCU adds to the MetaX measurement: a mask.** MetaX's Qwen-Image-2512
arrives maskless, so its clause is "no mask". Qwen-Image-2.1 does not:
`build_token_metadata` fills the joint sequence's `key_valid` with `torch.ones`
over the padded length whenever the caller supplies no
`encoder_hidden_states_mask`, as the text-to-image bench does, and every one of
the model's 32 blocks is handed that same `(1, 1, 1, 4122)` all-true bool mask.
Read as "a mask, keep the boxing route" the model's largest attention call --
`q(1, 32, 4096, 128) x kv(1, 32, 4122, 128)` -- lands on the composite's math
decomposition, which materialises the fp32 score matrix: 32 x 4096 x 4122 x 4 B
is **2.16 GB per call, 32 calls per forward**, where the Triton kernel keeps its
tiles in shared memory and never does. So the clause admits the mask and the
route converts it.

The clause is a shape test and nothing else: 4-D, `torch.bool`, one entry per
key. `size(3) == key.size(2)` together with `numel() == key.size(2)` is what pins
it to `(1,1,1,KV)`. That is deliberate -- no reduction runs over the mask and no
value in it is read on the host, so admitting a call costs no device sync. An
additive float mask is refused for the mirror-image reason: the kernel reads one
as `scores + mask` (`attention.py:102`, `113`), so "all zeros" would be a value
claim the route could only establish by a pass over it.

**The caller's row cannot be forwarded as it stands.** Two independent reasons,
both in the DCU backend's own copy of the kernel
(`flag_gems/runtime/backend/_hygon/ops/attention.py`):

- **The shape carries unbounded strides.** The mask block pointer is
  `attn_mask + batch_id*stride(0) + head_id*stride(1) + offs_m*stride(2) +
  offs_n*stride(3)` (`attention.py:251-259`), guarded against the *scores*
  extents and not the mask's own. A mask broadcast along any of those axes reads
  past its buffer. Measured on a DCU bw1000, `q(1, 4, 256, 64) x kv(1, 4, 320,
  64)` bf16 against an fp32 no-mask reference:

  ```
  attn_mask=None                 sum  105.7207   max|err| 0.001875
  attn_mask=(1, 1, 1, 320) bool  sum  nan        max|err| 3.195073
  attn_mask=(1, 4, 256, 320)     sum  105.9962   max|err| 0.007508
  ```

  At the model's own shape the same read is far enough past the allocation to
  take the process down rather than return NaN: a `(1, 1, 1, 4122)` mask against
  `(1, 32, 4096, 4122)` scores is a reproducibly fatal kernel VMFault on the
  first call.

- **A bool mask means the opposite of what torch means by it.** The kernel folds
  one in as `attn_mask.to(query.dtype) * -1.0e6` (`attention.py:816-817`), so
  `True` -- torch's "this key may be attended to" -- becomes the additive that
  removes the key. Only a non-bool mask is taken as the additive itself.

**What the route forwards instead.** `RouteMask()` builds that additive from the
caller's row and nothing else: `0 where True`, `kRoutedMaskFill` (`-1.0e6`) where
False, in fp32, reshaped to `(1,1,1,KV)` and expanded to the call's `(B,H,Q,KV)`.
Two properties do the work:

- **The expansion is what makes it safe.** What the kernel's first three strides
  have to satisfy is that the last index each of them can reach contributes no
  offset, i.e. `stride(i) * (size(i) - 1) == 0`; `expand` gives that for free,
  because an axis that grows from 1 takes stride 0 and an axis that stays at 1 is
  only ever indexed at 0. Measured on this part, a `(1,1,1,KV)` mask leaves
  `expand` as `(KV, 0, 0, 1)` at batch 1 -- the batch stride surviving because
  that axis was never expanded -- and as `(0, 0, 0, 1)` above it. `where()`
  already returns a contiguous tensor, so the `reshape` in front of the `expand`
  is a shape no-op on every mask this clause admits.
- **fp32, and finite.** The kernel adds this onto an fp32 accumulator
  (`attention.py:102`), so the additive is built in fp32 rather than the query's
  bf16. `-1.0e6` rather than `-inf` for the dropped keys: a fully dropped block
  would take `-inf` as its running max and then divide 0 by 0, while `-1e6`
  underflows to a zero weight after `exp2` against any logit this model produces
  and stays comparable as a maximum. MetaX measured the three spellings against
  `F.scaled_dot_product_attention` on a materialised bool mask: `as-is bool` and
  `-1e6 where True` both at max|d| 5.840e-01, `0 where True` at 1.953e-03 -- the
  bf16 floor the unfused reference itself sits at.

**The converted mask does not move the answer.** Measured on a DCU bw1000 at the
model's class of shape, `q(1, 2, 1024, 128) x kv(1, 2, 1040, 128)` bf16, against
the same call with no mask at all:

```
all-true row (1,1,1,1040)   max|d| 0.0001220703125
half the row dropped        max|d| 0.249023
```

The all-true row does not return the maskless call's tensor bit for bit, and it
cannot: the two calls are handed different arguments -- `attn_mask=None` against
a 1040-wide fp32 additive -- the entry point branches on that, and the same
reduction comes back reassociated. 0.0001220703125 is 2^-13, one bf16 ulp just
under the output's 0.287 peak, over 8 elements of 2.1M, and both arms land on the
same 0.000694 against an fp64 reference. Dropping half the row moves the answer
2000x further, which is what says the additive is read rather than only built.

**The envelope is the probes' predicates one for one.** bf16, 4-D, head_dim 128
exactly, query seq >= 1024, non-causal, no gqa, no dropout, no explicit scale, and
a mask the clause above admits. Widening any clause is a measurement rather
than an edit. Head_dim in particular is bounded below by the kernel, not by the
route: `_attn_fwd`'s shared-memory block grows with `BLOCK_DMODEL` and the
autotune set `keep()` admits (`flag_gems/ops/attention.py:173`) is capped at
`BLOCK_N <= 32`, so the tile cannot be traded down; at head_dim 128 some admitted
configurations already ask for 98304 B against this part's 65536 B limit while a
legal tile remains, and at the VAE's head_dim 512 the same path reports 294912 B
with nothing in the set fitting, so that call keeps the boxing route. The route
is drawn where it was measured, not where the kernel could be argued to fit.

**Four changes to the FlagGems Python path**, none of which moves an operator
between routes. All live in `torch_fl/flagos/__init__.py`, are gated on both
`_build_accelerator() == "dcu"` and the conf actually routing the op to FlagGems,
and are no-ops elsewhere.

1. `_patch_flaggems_pow_scalar_square` -- `pow.Tensor_Scalar` reaches
   `flag_gems.pow_tensor_scalar`, which on this backend is not upstream's
   `flag_gems/ops/pow.py` but the backend registrar's swap-in
   `flag_gems/runtime/backend/_hygon/ops/pow.py`, where every non-half input goes
   through `_pow(x.to(tl.float64), exponent.to(tl.float64))` -- an fp64 libdevice
   `powf` per element, for an exponent that is a runtime operand and so decides
   nothing at compile time. The Qwen-Image RMSNorm squares its input. Measured on
   a DCU bw1000 at fp32 `(1, 4122, 32, 128)`, 16.9M elements, 67.6 MB, events
   around a back-to-back launch loop:

   | op | host | device |
   |---|---|---|
   | `pow_tensor_scalar(z, 2.0)` | 185.6 us | 418.3 us |
   | `flag_gems.mul(z, z)` | 20.1 us | 104.8 us |

   and on bf16, 384.6/169.7 us against 56.5/20.1 us. The replacement is at least
   as accurate rather than merely close: `x * x` is correctly rounded where
   `exp2(2 * log2(x))` is not, and the two agree bit for bit at 0, -0, inf and
   nan. It is spelled with FlagGems' own `mul` rather than `torch.mul` to keep the
   whole op on FlagGems Triton kernels -- a deliberate ~50 us, because
   `mul.Tensor` is a `cuda` route in this conf and taking it would move a routed
   op off FlagGems.

2. `_patch_flaggems_mean_last_dim` -- `flag_gems.ops.mean.mean_dim_comm` picks its
   kernel by `K = numel/M/N`, the length of the *unreduced* slice, and `K == 1`
   (a reduction over the last axis of a contiguous tensor) gets
   `mean_dim_kernel_inner`, one CTA per row. At the RMSNorm's shape that is a CTA
   per 512 bytes and the launch cost dominates. Measured on the same host at fp32
   `(1, 4122, 32, 128)`, 131904 rows of 128:

   | op | host | device |
   |---|---|---|
   | `mean_dim(z, -1, True)`, shipped dispatch | 77.7 us | 443.5 us |
   | `mean_dim_kernel`, tiled | 47.8 us | 59.8 us |

   and on bf16 the shipped path is the same 443.5 us. Both kernels sum in fp32 and
   divide by N, so bf16 agrees bit for bit; fp32 differs by reassociation only
   (max |delta| 1.5e-8 on unit-variance input, below the bf16 rounding the model
   then applies). The tiled kernel is the one `mean_dim_comm` already uses for
   multi-dim reductions and for `sum`/`amax`/`prod`, so this is a dispatch fix,
   not a new kernel. Measured flat across M -- 46 us against 76 us at M=1 as well
   as at M=16384 -- so it needs only an upper bound on the row width, which is
   `_MEAN_TILED_MAX_N = 1024`; `_mean_is_last_dim` also rejects an out-of-range
   `dim` rather than let `%` normalise it into a mean over the wrong axis.

3. `_prewarm_triton_key` -- `triton.runtime.cache.triton_key` is `lru_cache`d and
   is the first thing `get_cache_key` asks for, so it runs inside whichever kernel
   compiles first. It sha256s every file under `triton/compiler`,
   `triton/backends` and `triton/language` and then the whole of
   `libtriton.<ext>`, which on the DTK 6.3 wheel is **900,284,304 bytes**.
   Measured with the venv's own interpreter, importing only
   `triton.runtime.cache`: `import triton.runtime.cache` 0.196 s, `triton_key()`
   **1.013 s**. In the model the same second shows up as a first denoise step of
   3030.0 ms of enqueue against 254.0 ms on the second, with `get_cache_key` at
   1.325 s of a first-step profile and one `triton_key` call at 1.085 s. Nothing
   about the value depends on this process -- it is a function of the installed
   files and `triton.__version__` -- so it is computed on a daemon thread at
   device init, while the caller still has ~33 GB of weights to materialise, and
   `hashlib` releases the GIL for buffers this size.

4. **The defect this branch's own refactor introduced, and its fix.** Changes 1
   and 3 route their rebinding through a new shared helper,
   `_rebind_flag_gems_name`, and so does `_patch_flaggems_vector_norm`, which
   `#352` merged with its sweep inlined. The helper reads the module-global `sys`,
   while each patch function carried its own function-local `import sys` for its
   other imports -- and a local `import sys` binds a local name, leaving the
   helper's global unbound. The helper raised `NameError` on every call, and every
   caller wraps it in `except Exception: pass`, so **all three patches installed
   nothing and said nothing**. The end-to-end arm labelled *patches inert* below
   is this branch measured in that state; it is a control, not a released
   revision. `import sys` is now a module-level import, the redundant function-local
   ones are gone, and `tests/unit/test_flaggems_dcu_costs.py` covers the helper
   directly -- including that a `_LazyModule` which only computes the name in
   `__getattr__` is never asked, and that a `sys.modules` entry of `None` is
   skipped rather than fatal.

**Measured end to end.** Eight-device DCU bw1000 host, DTK 6.3.26113 in
CUDA-boxing mode, torch 2.10.0+cpu decoupled, `flag_gems`
5.4.0rc2.post1+g437ba3938, Triton 3.6.0 with the `hcu` backend. Qwen-Image-2.1,
6-step denoise, batch 1, seed 42, `true_cfg_scale=1.0`, initial latents injected,
`torch.utils.benchmark.Timer.blocked_autorange` with two discarded warm-up calls;
median of the calls the timer's 60 s budget fits:

| arm | 1024x1024 | 1664x928 |
|---|---|---|
| DTK's own torch (vendor) | **6.998 s** (n=3) | **14.938 s** (n=2) |
| torch_fl, route off | 7.899 s (n=3) | 16.167 s (n=4) |
| torch_fl, route on, patches inert | 3.772 s (n=6) | 5.713 s (n=4) |
| torch_fl, route on, shipped | **3.197 s** (n=19) | **4.979 s** (n=13) |

**2.19x** and **3.00x** against the vendor build. The `n` differs per arm because
the timer sizes the call count from its own budget, not from the protocol: the
faster the arm, the more calls fit, and every arm ran at least two. Per phase, the
denoise loop is where the two builds separate -- 1095.8 ms/step against 436.3 ms
at 1024x1024 and 2389.2 against 699.0 at 1664x928 (**2.51x** and **3.42x**) --
while the text encoder stays the vendor's (0.057 s against 0.146 s) and VAE decode
lands at 0.292 against 0.352 s, the width the `mean_dim` patch closes most of. The
route-off arm is the control that makes this the route's doing and not the build's:
with the op boxed, this build is *slower* than the vendor's at both resolutions
(7.899 against 6.998, 16.167 against 14.938), which is the pre-existing per-op
launch overhead recorded in the entry below.

**The image does not move with the route.** One 1024x1024 image per arm from the
same injected latents, seed and 6-step schedule, so build and route are the only
variables, each arm bit-reproducible against itself (`determinism.identical` true,
one `image_sha256` per arm, `min_run_time` so short only the fast arm fits more
than one call):

| pair | max abs d | mean abs d | pixels over 1/255 | over 4 | over 16 |
|---|---|---|---|---|---|
| vendor vs route off | 48/255 | 0.790 | 22.67% | 4.21% | 0.18% |
| vendor vs route on | 43/255 | 0.684 | 19.48% | 1.66% | 0.07% |
| route off vs route on | 45/255 | 0.587 | 15.56% | 3.36% | 0.17% |

The route's own contribution is the same order as the build's, and the shipped arm
is the *closest* of the three to the vendor image on every threshold past 1/255 --
1.66% of pixels differ by more than 4/255 where the same build with the op boxed
shows 4.21%. Six bf16 steps amplify a one-ulp reassociation in either direction;
what the difference tracks is kernel selection on both sides, not a route that
computes something else. Peak allocation is the same 36.6 GiB in all three arms:
the score matrix the route avoids is traffic, not footprint.

**Survey.** `tests/manual/flaggems_overload_survey.py` v6 (`7b01c22c...`) scoped
to the changed route reports `registered 1`, `tested 1`, verdict **`FAILED`**,
`basic_executable 0`, `strict_support 0`, over 7 cases: 4 `WRONG` and 3
`INVALID_CASE`. The cause is the harness's `default_for()` catch-all, which ends
in `return 0.5` for a `float`/`Scalar` argument whose name it does not recognise,
and the SDPA schema's argument is named `dropout_p` -- so the harness
synthesizes `dropout_p = 0.5`, both arms of every profile draw from their own
dropout RNG, and a deterministic reference reports all four executable profiles
as wrong. Re-run with the catch-all's `0.5` changed to `0.0`, which is the value
that names `dropout_p`, the same seven profiles report **`STRICT`, 1 of 1, four
PASS and the same three `INVALID_CASE`** -- the verdict was the harness's
synthetic argument, not the route. The three `INVALID_CASE` profiles are the
harness's synthesis too, and are identical under both runs: 1-D input
(`Expected query, key, and value to all be at least 2 dimensional`), and int64 and
bool operands (`expected m1 and m2 to have the same dtype`), which ATen rejects
before any device kernel is reached. None of the seven profiles is inside the
route's envelope, so the verdict is the harness's either way. The full 459-overload
DCU survey over the shipped conf reports **459 registered, 382 tested, 342
`basic_executable`, 297 `strict_support`** -- 74.5% and 64.7% -- over 3213
cases (1850 `PASS`, 1062 `INVALID_CASE`, 147 `ERROR`, 140 `WRONG`, 14 `CRASH`).
The previous revision measured 458/381/342/297 (74.7% / 64.8%) over 3206 cases.
The one route added is `scaled_dot_product_attention`, and its `FAILED` is the
harness's synthetic `dropout_p` described above, not the route. The only other
records that differ between the two runs are `index_copy` and `index_copy_`,
each of which moved one profile between `PASS` and `WRONG` and stayed `FAILED`
in both: both synthesize their index argument as `randint(0, 2, ...)`, so
duplicate indices are guaranteed and which of them wins is undefined. Neither
moves a cohort total. The run was sharded -- eight processes over disjoint
`--ops` slices, merged with the harness's own `summarize()` over the same
459-route list -- which is sound because `run_overload()` measures every
overload in a fresh child process, so a per-overload verdict does not depend on
which process ran it or on run order.

**Regression coverage.** `tests/integration/ops/test_dcu_flaggems_sdpa.py` runs 15
call shapes through the shipped conf in a fresh interpreter, each differing from
the model's own in exactly one clause, and asserts the number of calls that
reached `flag_gems.scaled_dot_product_attention` against the clause that decides
it: `routed=1` for `eligible`, `joint`, `mask_bcast_true` and `mask_false`, and
`routed=0` for the eleven refusals (`mask_full_true`, whose `numel` is not KV;
`mask_float_zeros`; `head_dim64`; `seq512`; `float32`; `float16`; `rank2`;
`causal`; `scale`; `dropout`; `gqa`). **19 passed** on the hardware described
above. The mask each of the seven routed calls carried is checked too -- four
carried `RouteMask`'s fp32 `(B,H,Q,KV)` additive and three carried none -- and
every mask the kernel was handed is checked to satisfy `stride(i) * (size(i) - 1)
== 0` on the batch, head and query axes, the condition that keeps the block
pointer inside the KV-long buffer. Max deviation against a float32 host reference
is `0.001007` at the model's shape and `0.005810` at the worst of the fifteen, all
inside the file's `5e-2` bound; on one set of operands the all-true-row call stays
within `0.000122` of the maskless one while a half-dropped row moves it
`0.249023`. With `FLAGOS_OP_scaled_dot_product_attention=cuda` the same shapes
reach the kernel 0 times, which is what pins the route to the conf key and not to
the shape alone. Host-side, `tests/unit/test_flaggems_dcu_costs.py` (66 tests)
covers the three patched callables and the rebinding helper without a device, and
`tests/unit/test_gen_vendor_confs.py` (37 tests) gains the two-platform
containment pin for the composite set.

**Not revalidated.** Nothing outside DCU moves: `backends_metax.conf`,
`backends_cuda.conf`, TsingMicro and the remaining vendor confs are byte-identical
to the previous revision of this report, and no measurement transfers to them.
Their rows are carried over unchanged and are **not revalidated** by this change.

## Update History

Dated records of work already done, newest first. The environment variable names
in the Evidence column are written in their current spelling, so a re-run uses
the names the tree reads today; where a row predates the `FLAGOS_*` unification
that renamed them, the command is the same command under a different name and
the recorded result is unchanged.

| Date | Hardware | Cohort | Change | Evidence |
|---|---|---|---|---|
| 2026-09-20 | Hygon DCU bw1000 (8 devices) | The DCU full-coverage configuration (459 active routes, harness v6) and the Qwen-Image-2.1 6-step flow it serves | Routed `scaled_dot_product_attention` from `cuda` to `flaggems` in `backends_dcu.conf` — one line, taking the file to **459 `flaggems` / 1578 `cuda`** over the same **2037** entries (SHA-256 `8849ce31ca6e517b6e7f71057f90dfa917f1b76068501554f40b7806d5600369`, from `7b82cb492de80f3dc2dcb93deeba14764a986368460a63cae9d1deb90fee9163`) with the active route set moving from SHA-256 `ededd42387eef3b74473eef358515a1848c153b3d83896c13168676332c3f69c` to `b646c47b5d6643ed7a0ef753af24f402cf741d598ca3dca662946b89977288e6` — and widened the `attn_mask` clause of the shared hand-written override (`csrc/aten/sdp_choice_stub.cc`) from "no mask" to the one class Qwen-Image-2.1 passes: 4-D bool with `size(3) == key.size(2)` and `numel == key.size(2)`, which the size-1 axes pin to `(1,1,1,KV)`. The clause stayed a shape test — no reduction runs over the mask and no value in it is read on the host, so admitting a call costs no device sync — and a materialized float mask is still refused, because the kernel reads one as `scores + mask` and "all zeros" would be a value claim the route could only establish with a pass over it. The kernel cannot take the caller's row as it stands for two independent reasons, both in the backend's own copy at `flag_gems/runtime/backend/_hygon/ops/attention.py`: it forms the mask block pointer as `batch_id*stride(0) + head_id*stride(1) + offs_m*stride(2) + offs_n*stride(3)` (lines 251-259) guarded against the *scores* extents and not the mask's own, so a mask broadcast along any of those axes reads past its buffer — measured as `sum nan` / `max|err| 3.195073` against `0.001875` for no mask at all, and at the model's own shape a fatal kernel VMFault on the first call — and it folds a bool mask in as `attn_mask.to(query.dtype) * -1.0e6` (lines 816-817), the inverse of torch's convention, so `True` becomes the additive that *removes* the key. `RouteMask()` therefore builds the additive itself in fp32 — `0 where True`, `kRoutedMaskFill` (`-1.0e6`) where False, reshaped to `(1,1,1,KV)` and expanded to the call's `(B,H,Q,KV)` — and the contract that makes it safe is `stride(i) * (size(i) - 1) == 0` on each axis, **not** `stride(i) == 0`: an axis that stays at size 1 is only ever indexed at 0, which is what `expand` supplies and what a measured `(1,1,1,KV)` mask shows, leaving `(KV, 0, 0, 1)` at batch 1 (the batch stride surviving because that axis was never expanded) and `(0, 0, 0, 1)` above it. Three Python-side patches cut the per-op host cost that had this build *slower* than the vendor's with the route off: `pow.Tensor_Scalar` on this backend runs an fp64 libdevice `powf` per element for a runtime exponent, `mean_dim_comm` sends a last-dim reduction to one CTA per row, and `triton_key()` sha256s 900,284,304 bytes inside whichever kernel compiles first. Every route the additive needs (`zeros_like`, `full_like`, `where.self`) is a `flaggems` route in this conf, so nothing falls back to the vendor, and **no operator leaves FlagGems for CUDA boxing**. `DCU_COMPOSITE_FLAGGEMS` in `scripts/codegen/gen_vendor_confs.py` holds the route from every other platform's conf the way `METAX_COMPOSITE_FLAGGEMS` does, and `tests/unit/test_gen_vendor_confs.py` pins the containment in both directions. The full 459-overload DCU re-survey over this configuration's own route set ran to completion: **459 registered, 382 tested, 342 `basic_executable`, 297 `strict_support`** (74.5% / 64.7%) over 3213 cases (1850 `PASS`, 1062 `INVALID_CASE`, 147 `ERROR`, 140 `WRONG`, 14 `CRASH`), against the previous revision's 458/381/342/297 (74.7% / 64.8%) over 3206; the one route added is `scaled_dot_product_attention`, whose `FAILED` is the harness's synthetic `dropout_p` recorded in this entry, and the only other records that moved (`index_copy`, `index_copy_`) each flipped one profile between `PASS` and `WRONG` while staying `FAILED`, which the duplicate indices their synthesized `randint(0, 2, ...)` argument guarantees; the run was sharded across eight processes over disjoint `--ops` slices and merged with the harness's own `summarize()`. **Ascend, GCU, MUSA, MetaX, PPU and TsingMicro are not revalidated** — except for the two `backends_metax.conf` / `backends_dcu.conf` divergences this entry records, every other conf is byte-identical to the previous revision of this report and no measurement transfers to them. | Measured on the Qwen-Image-2.1 1024x1024 6-step flow (batch 1, seed 42, `true_cfg_scale=1.0`, injected initial latents) on a Hygon DCU bw1000 with DTK 6.3.26113 in CUDA-boxing mode, torch 2.10.0+cpu decoupled, `flag_gems` `5.4.0rc2.post1+g437ba3938` and Triton 3.6.0 (`hcu`), `torch.utils.benchmark.Timer.blocked_autorange` with two discarded warm-up calls, one process per arm: vendor DTK torch **6.998 s** (n=3) / **14.938 s** (n=2) at 1024x1024 / 1664x928, this build with the op boxed **7.899 s** (n=3) / **16.167 s** (n=4), this build with the route on and the patches inert **3.772 s** (n=6) / **5.713 s** (n=4), and this build shipped **3.197 s** (n=19) / **4.979 s** (n=13) — **2.19x** and **3.00x** against the vendor, where the route-off arm is the control that makes this the route's doing and not the build's. Per phase the denoise loop is where the builds separate, 1095.8 against 436.3 ms/step at 1024x1024 and 2389.2 against 699.0 at 1664x928 (**2.51x** / **3.42x**). Per op, all on one card: `pow_tensor_scalar(z, 2.0)` at fp32 `(1, 4122, 32, 128)` — 16.9M elements, 67.6 MB — **185.6 us host / 418.3 us device** against `flag_gems.mul(z, z)` at **20.1 / 104.8**, and on bf16 384.6/169.7 against 56.5/20.1; `mean_dim(z, -1, True)` at the same shape, 131904 rows of 128, **77.7 / 443.5** shipped against the tiled kernel's **47.8 / 59.8**, flat across M so it needs only `_MEAN_TILED_MAX_N = 1024` as a row-width bound; `triton_key()` measured alone with the venv's own interpreter **1.013 s** after a 0.196 s import, showing up in the model as a first denoise step of 3030.0 ms of enqueue against 254.0 ms on the second. The route's own cost is measured rather than assumed: all three ops the additive calls are `flaggems` routes in this same conf (`zeros_like`, `full_like`, `where.self`; `RouteMask` calls the out-of-place `where`, and the `zeros_like.out` / `full_like.out` spellings it does not call are `cuda`), so the conversion adds no boxed launch, and a `FLAGOS_LOG=dispatch` census of a masked call on the MetaX sibling of this route shows all three on `flagos_python` with no `fallback` line. The converted mask does not move the answer: on `q(1,2,1024,128) x kv(1,2,1040,128)` bf16 against the same call with no mask, an all-true row reads `max|d| 0.0001220703125` — one bf16 ulp just under the output's 0.287 peak, over 8 elements of 2.1M, with both arms landing on the same `0.000694` against an fp64 reference — while dropping half the row moves it `0.249023`, 2000x further, which is what says the additive is read rather than only built. The image moves no further than the build does: one 1024x1024 image per arm from the same latents, seed and schedule, vendor against route-off `48/255` max / `0.790` mean / 22.67% of pixels over 1/255, vendor against route-on `43/255` / `0.684` / 19.48%, route-off against route-on `45/255` / `0.587` / 15.56% — the shipped arm is the closest of the three to the vendor image at every threshold past 1/255 (1.66% of pixels over 4/255 against the boxed build's 4.21%), and peak allocation is the same 36.6 GiB in all three, since the score matrix the route avoids is traffic and not footprint. Re-surveyed because a route moved: `tests/manual/flaggems_overload_survey.py` v6 (`7b01c22c...`) scoped to the changed route reports `registered 1`, `tested 1`, verdict **`FAILED`**, `basic_executable 0`, `strict_support 0` over 7 cases (4 `WRONG`, 3 `INVALID_CASE`) — and the cause is the harness, not the route: its `default_for()` catch-all ends in `return 0.5` for a `float` argument whose name it does not recognise, and the SDPA schema's argument is named `dropout_p`, so both arms of every profile draw from their own dropout RNG and a deterministic reference reports all four executable profiles wrong; with the catch-all changed to `0.0`, which is the value that names `dropout_p`, the same seven profiles report **`STRICT`** with four `PASS` and the same three `INVALID_CASE` (1-D input, int64 and bool operands — calls ATen rejects before any device kernel is reached). None of the seven profiles is inside the route's envelope either way. Regression coverage: `tests/integration/ops/test_dcu_flaggems_sdpa.py` runs 15 call shapes through the shipped conf in a fresh interpreter, each differing from the model's own in exactly one clause, asserting the count that reached `flag_gems.scaled_dot_product_attention` against the clause that decides it (`routed=1` for `eligible`, `joint`, `mask_bcast_true`, `mask_false`; `routed=0` for `mask_full_true`, `mask_float_zeros`, `head_dim64`, `seq512`, `float32`, `float16`, `rank2`, `causal`, `scale`, `dropout`, `gqa`), the form of the mask each routed call carried, the `stride(i) * (size(i) - 1) == 0` condition on every mask the kernel was handed, and that the additive is informative — **19 passed in 79.51s**, and with `FLAGOS_OP_scaled_dot_product_attention=cuda` the same shapes reach the kernel 0 times, which pins the route to the conf key and not to the shape alone. Host-side, `tests/unit/test_flaggems_dcu_costs.py` (66 tests) covers the three patched callables and the rebinding helper without a device and `tests/unit/test_gen_vendor_confs.py` (37 tests) the two-platform containment pin — **103 passed in 24.92s** together. `ruff check` — "All checks passed!"; `ruff format --check` — 285 files already formatted; `gen_vendor_confs.py --check` — `all vendor confs up to date`. **Evidence gaps:** the mask conversion is measured on the class of shape the route admits and not on a prompt cohort; the three per-op figures are one shape and one dtype pair each rather than a sweep; the `pow` and `mean_dim` patches are silent no-ops if a FlagGems upgrade renames what they rebind, which is the same failure mode as the `NameError` this branch itself introduced and fixed; the survey harness cannot measure this op at all through `active_routes()`, which enumerates the ops a conf *file* spells `flaggems`, so the verdict above is the harness's synthetic `dropout_p`; and no vendor-side 1664x928 repeat was taken in the same session as the route-on arm. |
| 2026-09-20 | Enflame GCU S60 (8 `flagos` devices) | The 171 `F.scaled_dot_product_attention` calls one Qwen-Image-2.1 forward makes, logged by shape, stride, mask dtype and mask innermost stride from a real run | Extended the GCU SDPA predicate to admit calls that carry a mask, and dropped the `q_len != kv_len` refusal that lived in the same clause. On 2.1 **every one of the transformer's 98 attention calls carries a mask** (`attn_mask` absent was the 2512-shaped predicate's first requirement), so the Reroute above served all 98 mask-free-only calls through ATen's math decomposition. The new clause is one uniform rule, `SupportedVendorMask`: the mask must be bool, **or** additive in the query's dtype (bf16); rank 1, 2 or 4; right-aligned on `(batch, q_head_num, q_seq_len, kv_seq_len)` with every dim either singleton or full; and its innermost axis must be of size 1 or stride 1. It is deliberately dtype-symmetric rather than bool-only, because **the `_fused_sdp_choice` stub sees the caller's mask while the leaf sees the composite's converted one**: two caller spellings that pass the shape test and fail only on the innermost stride (`(4096, 4122)` bool `.t()` at stride 4096, and `1x1x64x80` bool with an expanded innermost axis at stride 0) are declined by the rule, and an admitted conversion would have replaced that stride with 1 and sent them to the leaf, whose own refusal is a hard `RuntimeError` rather than a fallback. The conversion itself can never produce a spelling the leaf refuses — it always yields the caller's shape right-aligned on `(B, H, q, kv)`, size-1-or-full, innermost stride 1 — which was verified on 16 caller spellings: every admitted one correct on the leaf, every declined one correct on math, none raising. **No route moves and no generated artifact changes**: the predicate lives in the hand-written translation unit, not in the generated conf, so the shipped conf stays 1605 `none` / 254 `flaggems` / 178 `gcu` over 2037 routable ops, `254 = 247 + 7` and `178 = 185 - 7` unchanged, and the GCU FlagGems cohort is **not revalidated** here: no survey run was taken with this clause in the tree, and the reason one is not needed is that `backends_gcu.conf` is byte-identical (SHA-256 `28f4656c`...) to the file the existing S60-wide cohort below was measured against, so no route verdict it records can move -- the existing cohort stands as the last measurement rather than as a re-measurement. FlagGems is still not the route, and was measured again rather than assumed: **133.84 ms/call** mask-free at the 2512 shape (against the math path's 90.54 in that run) and it **aborts with `exit 134` on a masked call**, because Triton-GCU rejects the zero-valued strides `expand` produces for the broadcast mask. | **The census**, from one real 2.1 forward (`probe_q21_sdpa_wrap`, 171 calls in the process, 4 distinct geometries): **98 admitted to the fused lane, 73 left on math** — 65 x target-image `q=(1,32,4096,128) kv=(1,32,4122,128) mask=(1,1,1,4122)` bool strides `(4122,4122,4122,1)`, and 33 x prefix-segment `q=(1,32,26,128)` `mask=(1,1,26,26)` bool strides `(676,676,26,1)`; declined are 72 x text-encoder `q=(1,32,40,128)` `k/v=(1,8,40,128)` (GQA and `is_causal`, 0.0002 GiB, not where the time goes) and 1 x VAE decode `q=(1,1,4096,1152)` strides `(14155776,14155776,3456,1)`. The 65 target-image calls are the reason this mattered: on math each materialises an fp32 `(1,32,4096,4122)` score matrix, **2.0127 GiB per call and 130.828 GiB of allocation churn over one forward**, and the driver refuses it — `itopsMalloc required :2164260864` (**2.0156 GiB**) — so the leg could not complete a single denoise step on one card. **A/B at that shape** (`probe_gcu_attn_target`, `flagos:0`, 3 calls per row, peak from `torch.flagos.memory_stats()["peak_allocated_bytes"]`, which is flat where `torch.cuda`'s is nested): fused no-mask **5.46 ms / peak 0.031 GiB**; fused with an admitted bf16 mask **13.81 ms / 0.031 GiB**; math with the fp32 spelling of the *same* allow-mask **147.14 ms / 4.785 GiB** — **10.65x per call and 4.754 GiB off the peak**, the two mask rows differing only in dtype and therefore a controlled pair, since the predicate is the only thing that sends them apart. A masked call costs 2.5x the unmasked one on the same op; that gap is not attributed here and does not affect the route decision, both being far below math. **Accuracy**: the masked fused calls agree with an fp32 CPU reference to `2.092e-03` relative at the target-image shape and `5.376e-03` on the prefix group, the same order as the mask-free cells (`3.650e-03`), so the mask is read as an additive bias and not as something else; discriminating padded-plane cells read additive `4.302e-03` / `4.134e-03` where a boolean reading of the same buffer gives `4.4e-01`-`1.64e+00`. The 73 declines stay on math and stay correct. **One silent-wrongness finding, which is why the predicate demands bool or the query dtype**: an fp32 additive mask is accepted by the leaf with `SUCCESS` and read as bf16 bytes (relative `1.1908` for the no-mask row and `1.2072` for the additive row, where correct is ~`4e-3`); it is unreachable through `F.scaled_dot_product_attention`, which converts the mask to the query dtype first, and the leaf therefore carries the dtype clause as a guard rather than the composite. New `tests/integration/ops/test_gcu_sdpa_mask.py`: 13 tests, 13 passed, asserting both the route (a `TorchDispatchMode` census over `_scaled_dot_product_efficient_attention` against `_safe_softmax`/`bmm`) and the values, including the two stride declines, the grad-mode decline, the four maskless-drop cases, a per-head mask that has to be read on every head, and the finite-offset deviation below. `ruff check .` — "All checks passed!"; `ruff format --check .` — all files already formatted. **A second, later change in the same clause: an inert mask is dropped before the vendor op is asked.** Everything above prices a masked call at 2.5x the mask-free one and attributes the gap to the leaf's own cost; a sync-bracketed in-model probe then showed that the mask the composite builds from an all-allow caller mask is a `(1,32,4096,4122)` bf16 view at strides `(4128,0,0,1)` -- one stored row of 4122 zeros expanded over 540M logical elements -- and that the vendor op's maskless entry point (`topsatenScaledDotProductFlashAttention`) answers it in **5.2 ms/call against the masked entry point's 19.3**, i.e. 166.3 ms against 617.3 over the 32 target-image calls of one forward. `AdditiveMaskIsAllZero` therefore narrows every stride-0 axis of size > 1 down to index 0 (a view, not a copy), declines outright when nothing was narrowed, and evaluates `(narrowed != 0).any()` through the routed `ne` and `any` kernels before handing the mask on; a dense mask is declined and a bool mask is never treated as inert, since zero is the additive identity only for an additive mask. The gate itself costs **0.52 ms/call** at the target-image geometry and **0.88** at the prefix geometry (0.19 / 0.63 on the vendor leg, ~11 ms on the first call while the modules load), **~45 ms per forward** against the 451.0 ms those 32 calls were costing. **The equivalence is measured.** On a nine-geometry off-model grid (`q` 1..4096, `kv` 26..4122, the model's own `kv + 6` padded strides) masked and maskless are bitwise equal on **9 of 9**, and each geometry carries a `-1000.0` control that differs. In-model, replaying every masked call of one real forward maskless is bitwise equal at **32 of 32** calls and moves the whole-model output by relative L2 `0.000000e+00` (`100.00 %` of elements within one bf16 ulp), where the rejected *broadcast-narrowing* spelling -- cheaper still at 343.0 ms on the leaf -- is **0 of 32** bitwise at the target-image geometry (one bf16 ulp, amplified by 32 layers to max abs `3.125e-01`, relative L2 `7.102801e-03`, `26.25 %` within one ulp), so it is not in the tree. **Census A/B, one tree one op apart** (`/tmp/op-census-shipped.log` against `/tmp/op-census-maskless.log`, `--stage transformer-step`, same prompt and seed): forward **1971.8 -> 1550.9 ms**, exclusive total 2281.9 -> 1946.7, the 64 attention leaves **551.4 -> 200.8 ms**, dispatch counts identical at 4753 / 9506 and the census output bitwise the plain output's on both legs, so the whole **-420.9 ms (21.3 %)** is the leaf; the rope-matched pair (`QWEN_IMAGE_REAL_ROPE=1`, 5521 / 11042 on both legs) reads 1897.5 -> **1468.4 ms** with the leaf at 548.2 -> 200.6. Against the vendor leg at the same rope configuration and the same 64 calls the fused leaf is now **200.6 against 407.6 ms, 207 ms cheaper than torch_gcu's**, and the flagos forward remains **+254.3 ms** behind at 1468.4 against 1214.1: the residual is a per-op premium on ops both legs call the same number of times, **65 % of it in `aten.mul.Tensor` alone** (+162.7 ms over 549 calls on each leg), which is why the maskless drop is a step and not parity. Approval to change what the translation unit hands the vendor op was requested and given explicitly, and is recorded in `docs/vendors/gcu/scaled-dot-product-attention.md`. **A finite constant offset in the mask is not exact, and is recorded rather than declined**: softmax is shift-invariant, so a row shifted by a constant is a no-op in exact arithmetic, yet the op applies the additive row at reduced precision -- `max|d|` against an fp32 CPU reference on the regression test's geometry (2 heads, 64 queries, 80 keys, bf16) is `1.6e-2` at `-8`, `2.2e-2` at `-16`, `3.5e-2` at `-30`, `1.29e-1` at `-100` and `1.33` at `-1000`, while the two spellings 2.1 produces are exact: an all-zero mask is bitwise identical to no mask, and `-inf`, which is what the bool conversion builds, agrees to `0.000e+00`. No clause declines it -- a threshold on the shift's magnitude would be arbitrary and a uniformly shifted row is a degenerate input -- so the deviation is pinned by a test that fails if it goes away, rather than living on as a number past its measurement. **Evidence gaps:** there is no paired end-to-end run on 2.1 with this clause in the tree yet, so the per-call A/B above is the whole of the measured claim and the single-card fit it predicts is **not** confirmed by a completed leg; the census is one forward of one prompt and one seed rather than a cohort; the SDPA leaf's own masked-call cost was priced at the 2512-adjacent shape rather than inside the 2.1 graph; the device run uses card 0, with card 5 defective and excluded; and on this platform the composite's `attn_mask` conversion is not exercised by the 2.1 graph at all — every admitted mask is already rank-4 with an innermost stride of 1 when the stub sees it — so the conversion path is verified by the 16-row probe rather than by the model. |
| 2026-09-20 | Enflame GCU S60 (8 `flagos` devices) | The complex64 `aten.mul.Tensor` calls of the Qwen-Image-2.1 rope path, disaggregated from a per-op census of one forward | `topsatenMul` computes a complex64 product on the device, but not at the kernel's cost: at `topsaten_common.h:97` a complex64 call takes a decomposed path and the promotion and argument marshalling run on the host, so a complex64 multiply of two `(1,4096,24,64)` operands measured **50.166 ms against 2.27 ms for the two real multiplies carrying the same arithmetic**. The generated `MulTensorKernelGcu` therefore takes `view_as_real` of both operands, builds `(ar*br - ai*bi) + i(ar*bi + ai*br)` from four real multiplies, and `view_as_complex`es the stack -- views over the stack's own buffer, every multiply a topsaten kernel -- **only** when the promoted result is `complex64` and both operands are contiguous after the cast, since `view_as_real` cannot be taken of a strided operand; a mixed real x complex64 pair takes the same branch because the test is on `at::result_type`, complex128 is left alone, and only `mul.Tensor` is listed (`div.Tensor` needs a shared real denominator, `add`/`sub` are not where the time goes, and `mul.out` / `mul._Tensor` would need a complex `copy_`). The change is in the generator, not in the generated file: `_binary_dtype_guard` now takes the complex body as a parameter and a module-level `COMPLEX_ARITHMETIC` dict supplies it, so `gcu_kernels.cc` is regenerated rather than hand-edited, and three includes (`ATen/ops/stack.h`, `ATen/ops/view_as_complex.h`, `ATen/ops/view_as_real.h`) were added to the `FILE_HEADER`. **No route moves**: `mul.Tensor` is `= gcu` in `backends_gcu.conf:1425` before and after, the appended conf line is a no-op, no new backend is introduced, and the generator produces an empty second diff. **Measured** by per-op census of one real forward (`/tmp/op-census-flagos.log` against `/tmp/op-census-shipped.log`, same prompt and seed, identical rope configuration and identical `4753` / `9506` dispatch and sync counts): `aten.mul.Tensor` **3405.2 -> 545.2 ms** at 357 calls (**9538.3 -> 1527.1 us/call**, 66.0 % -> 23.9 % of the census's exclusive self time) and the exclusive total **5161.7 -> 2281.9 ms**, so `-2860.0 ms` of a `-2879.8 ms` change is that one row; the forward reads 4801.7 -> 1971.8 ms. The same row is 65 % of the remaining flagos-versus-vendor per-op premium at an identical call count. **Bit-identical**: the rewritten product was checked against the complex product it replaced at `max|d| 0.000e+00` (`bitwise equal: True`) and against the host's own real-view product in `tests/integration/ops/test_mul_dispatch.py`. New `TestMulTensorComplex64` there: 9 tests, 9 passed -- values against CPU at four shapes plus a broadcast and a mixed real x complex64 pair, bit-identity against the spelled-out real-view product, the route read from the dispatch log in a subprocess (two `view_as_real` and one `view_as_complex` for a contiguous pair, none for a transposed one, with the multiply logged on the card in both), and a transposed operand's values proving the contiguity test stands the branch down. The route is read from the log because **the nested dispatches a routed GCU kernel makes are invisible to a Python `TorchDispatchMode`** -- it sees only the outermost `mul.Tensor`, measured with a contiguous pair whose log shows the two `view_as_real` calls that the mode does not -- so a mode-based assertion would pass whether or not the branch ran. **Evidence gaps:** the micro-measurement is one shape at one dtype rather than a cohort; the census pair is one forward of one prompt and one seed; and the rope path's 4-deep complex64 multiply is the only complex64 call this model makes, so `complex128` and the `div` / `add` / `sub` / `out=` / `_` spellings are deliberately out of scope and unmeasured. |
| 2026-09-20 | Enflame GCU S60 (8 `flagos` devices) | The Qwen-Image-2512 transformer's 240 rotary-embedding applications per forward pass, priced at the call site and again as full model outputs | Registered the `flagos` device in `diffusers`' Qwen-Image rope table from `torch_fl`'s GCU branch, which turns the angle path the entry above measured through a manual switch into the default and replaces the expansion it used. `diffusers` keys the rotation on device type in two places that must agree — `ROPE_PER_DEVICE` selects the rotation at the attention call site, while `QwenEmbedRope._get_device_freqs` produces the operand and returns **complex** frequencies for every device but `neuron` — so both halves are installed together, the operand half delegating to the stock method for every other device and both guarded by a module flag so a second call is a no-op. The consumer is the same rotation as `apply_rotary_emb_qwen_neuron` but expands each angle with `stack`/`flatten` instead of `repeat_interleave(2, dim=-1)`, whose composite lowering (`unsqueeze(-1).expand(..., 2).reshape(...)`) materialises a stride-0 view through `StridedCopy` and reaches the drained copy path 480 times per forward — 2.249 ms per call while those calls are the ones paying the drain, against 25.596 us once they are not. Three legs, one process each, warm-up 1 and 3 timed forwards, transformer on `flagos`, 1024x1024: complex (old default) 5.411 s forward with 3818.7 ms of rope (70.6 %), neuron via `QWEN_IMAGE_REAL_ROPE=1` 3.392 s with 1448.2 ms (42.7 %), shipped 2.455 s with 546.8 ms (22.3 %). **2.956 s per forward (54.6 %) against the fallback it removes and 0.937 s (27.6 %) against the angle path it replaces**, of which the harness switch alone was worth 2.019 s. **On the leg that ships**, where a step is two forwards under true CFG, the same registration is worth **6.80 s per step, 57.6 % of the loop**: one session, three legs one environment variable apart, 8-step 1024x1024 `--stage full` at the harness's default seed -- `FLAGOS_DISABLE_QWENIMAGE_ROPE=1` **11.81 s/it** (3.10x), shipped **5.01 s/it** (**1.31x**), `torch_gcu` + `diffusers` **3.81 s/it** (1.00x), with the shipped leg reproducing at `5.00`. That is the configuration a user gets: the `8.48 s/it` / `2.23x` on the entry below and the `8.38 s/it` / `2.09x` in the rotation section above both ran the `QWEN_IMAGE_REAL_ROPE=1` harness switch instead, and are not the shipped state. The shipped and neuron legs are **bit-identical end to end** (`torch.equal` true on `(1, 4096, 64)` bf16, 0 of 262144 elements differing); the complex leg differs from both by `3.125e-02` at `absmax 5.3125`, two bf16 ulps from a different product ordering accumulated over 60 blocks. No conf file or generated registration is touched, so no route moves — the shipped conf stays 1605 `none` / 254 `flaggems` / 178 `gcu` over 2037 routable ops. `FLAGOS_DISABLE_QWENIMAGE_ROPE=1` leaves the table as `diffusers` ships it. Evidence: [the rotation entry](#enflame-gcu-s60-forward-sdpa-on-the-vendor-flash-op-and-the-broadcast-materialisation-2026-09-20) above, `tests/unit/test_gcu_qwenimage_rope.py` (19 tests: bit-for-bit against a pinned copy of the neuron expansion and against the installed `diffusers` copy, `allclose` against the complex path at a documented one-ulp tolerance, and the registration's idempotence, disable switch and operand/consumer split), `tests/unit/test_env_registry.py`, and `/tmp/probe_rope_shipped.py` with `/tmp/rope_out_{complex,neuron,shipped}.pt` as the saved evidence; `/tmp/run_rope_ab.sh` with `/tmp/rope_ab_{rope_on,rope_off,vendor}.log` for the 8-step shipped-leg reading and `/tmp/run_stage_decomp.sh` with `/tmp/decomp_{flagos,vendor}.log` for the discarded stage-split arm, the reference leg of both on `TOPS_VISIBLE_DEVICES=0,1,2` through its own interpreter. **Evidence gaps:** the shipped-leg A/B carries the harness's default inputs and seed rather than the paired latents the SDPA entry's 8-step runs use, so the two runs are comparable through the reference leg's reproduced `3.81 s/it` and not through identical inputs; `FLAGOS_DISABLE_QWENIMAGE_ROPE=1` returns before installing either half, so the opt-out leg prices the consumer and the operand producer together and not separately; and the stage-split arm's walls are recorded as a discarded result rather than as evidence about the decode. |
| 2026-09-20 | Enflame GCU S60 (8 `flagos` devices) | The Qwen-Image-2512 transformer's own 60 `F.scaled_dot_product_attention` calls per forward pass, measured on the model's envelope rather than on a stand-in shape | Routed `_scaled_dot_product_efficient_attention` from `none` to `gcu` so the composite stops serving every attention call with the math decomposition. That decomposition runs on the device — there was no `cpu_fallback` line for it — but at the model's shape it is 90.71 ms per call against 1.39 ms for the vendor flash op, and it was the largest single term left in the transformer step after #351. `F.scaled_dot_product_attention` selects a backend by asking ATen's `_fused_sdp_choice` **DispatchStub**, which only a `REGISTER_PRIVATEUSE1_DISPATCH` registrar fills: registering the leaf, or an ATen op of the same name, leaves `is_device_supported(PrivateUse1)` false and the composite on the math branch. The stub's return value *is* the backend selection, so it also serves as the decline mechanism — a mask, `is_causal`, GQA, fp16, `q_len != kv_len` or any grad-mode call returns `math` and keeps today's behaviour with no host round trip and no CPU fallback, and the predicate behind the stub and the kernel's `TORCH_CHECK` is one function. FlagGems carries an SDPA route and it was measured rather than assumed: 133.84 ms per call against the math path's 90.54 ms, so it is not the route taken and `_scaled_dot_product_efficient_attention_backward` stays on FlagGems. The kernel is a hand-written translation unit, which CLAUDE.md allows only for a concrete codegen limitation and with explicit human approval (both met): `codegen_gcu.py` cannot express a `REGISTER_PRIVATEUSE1_DISPATCH` registrar into an ATen DispatchStub, nor a kernel body returning a 4-tuple. Everything else stays generated — the `m.impl` line and the `= gcu` route come from listing the op in `HANDWRITTEN_OPS`. The same commit stops the generated elementwise kernels materialising broadcasts with `expand().contiguous()`, which the vendor documents it does not need. Route delta: `gcu` 177 -> **178**, `none` 1606 -> **1605**, `flaggems` **254** unchanged, accelerated 431 -> **432** (21.2 %); `gcu_register.inc` 184 -> **185** `m.impl` lines, so the reconciliations read `254 = 247 + 7` and `178 = 185 - 7`. No route moves *from* FlagGems and the FlagGems route set is byte-identical, so the S60-wide survey below is a provenance re-run rather than a new cohort — and it reproduces the base cohort's **1778 verdicts with 0 differing cases** (`/tmp/gcu-overloads-post.json` `6d1120be…` against `/tmp/flaggems_gcu_survey.json` `1ee7ea4f…`). | Within-process A/B on one card, same 60 calls and operands, sentinel-planted output buffer read back per run (`sentinel-left 0` of 12 638 208 every run): math leaf **11.637 s** (60/60, 5442.9 ms, 90.71 ms/call) -> vendor leaf **7.240 s** with the pre/post drain (60/60, 331.3 ms, 5.52 ms/call) -> **7.356 s** with drains suppressed (83.1 ms, 1.39 ms/call), i.e. **4.397 s of 11.637 s = 37.8 %**, with the drain costing 248 ms/step (3.4 %). The two vendor runs are byte-identical in their outputs (`max|d| 3.125e-02`, `mean|d| 2.743e-03` against `|math|max 5.750e+00`). Against an independent float64 reference on the model's envelope the kernel reads `max|d| 1.555e-02`, `mean|d| 1.005e-03` against `|ref|max 5.315e+00`, and every declined case — fp16, `attn_mask`, `is_causal`, GQA, `q_len != kv_len`, grad enabled — reaches the leaf **0** times while staying correct on the math path; the direct `compute_log_sumexp=True` call returns `out (1,24,4114,128)`, `lse (1,24,4114,4114)` bf16, seed and offset `(0,)`, which is ATen's own convention for that slot and is delegated to the math op rather than re-derived. Shipped route in the real graph: warm-up 9.51 s, profiled **7.83 s**, summed **7.50 s/step over 64 ops**, leaf **170.813 ms / 60 calls**, `_fused_sdp_choice REACHED -> 2` before every batch. **End to end, paired, two builds of one tree:** 8-step 1024x1024 `--stage full`, seed 42, `QWEN_IMAGE_REAL_ROPE=1` on every leg, `TOPS_VISIBLE_DEVICES=0,1,2`, and the same latents and prompt embeds carried between the legs by `--save-latents`/`--save-inputs` then `--load-latents`/`--load-inputs` — math leaf **19.33 s/it**, vendor leaf **8.48 s/it** (8.81 on a repeat that produced a byte-identical image, `md5 d47974b1…`), `torch_gcu + diffusers` **3.81 s/it**. So the reroute is **2.28x** and **56.1 % off the denoising loop** (2.19x / 54.4 % against the slower repeat, which is 2.31x the vendor against the faster leg's 2.23x), and the loop delta is larger than the 37.8 % the per-call table projects because the projection priced the 60 leaf calls of one forward shape while the math decomposition also materialises intermediates the flash leaf never builds. The three output images: shipped against reverted `MAE 3.568/255`, `PSNR 30.10 dB`; shipped against `torch_gcu` `5.089/255`, `26.96 dB`; reverted against `torch_gcu` `5.102/255`, `26.91 dB` — the shipped leg is marginally the closer of the two to the reference, so the flash kernel is a real image-level change and not a step away from it. The route cannot be A/B'd at runtime and two attempts are recorded so they are not repeated: `FLAGOS_OP__scaled_dot_product_efficient_attention=none` is refused with *"routed to 'none' … but the op is registered on PrivateUse1"*, and `torch.nn.attention.sdpa_kernel([SDPBackend.MATH])` is a silent no-op — that leg returned a byte-identical image to the shipped leg while still logging 960 `-> gcu` dispatches, because `__torch_function__` selects the route before ATen's backend pin is consulted. `ruff check .` — "All checks passed!"; `ruff format --check .` — 272 files already formatted. Generator idempotency: two runs leave the three moving artifacts byte-identical (`md5sum -c` -> `OK`). Artifact SHA-256 before -> after: `backends_gcu.conf` `476825db…` -> `28f4656c…`; `gcu_kernels.cc` `57e6edc9…` -> `f3cb0ee8…`; `gcu_register.inc` `dc88d5bc…` -> `dcab7a4e…`; `gcu_flaggems_register.inc` unchanged at `be843180…`. **Evidence gaps:** the broadcast half is measured per call (`a * a` 0.212 ms, `a * 0-dim device` 0.322 ms, `a * 0-dim host` 0.416 ms, `.contiguous()` on a contiguous tensor 0.004 ms at `(4096, 2560)` bf16) and not end-to-end — deliberately and not by omission, since the paired run above carries it in **both** legs, which is what makes that run's 2.19x attributable to the SDPA reroute alone; partitioning it out would need a third build. The 8-step paired run is one prompt and one seed and not the 12-prompt cohort; the two `torch_fl` legs are the same tree built twice rather than a runtime switch, since no runtime switch exists; the residual-gap table's two columns are not a controlled comparison (different card sets, profiler on, and it inflates runtime calls — 7.24 s unprofiled against 7.83 s profiled); and the device run uses card 0, with card 5 defective and excluded. |
| 2026-09-19 | MetaX C550 (8 devices) | MetaX `scaled_dot_product_attention` composite route | Widened the `attn_mask` clause of the MetaX `scaled_dot_product_attention` composite override (`csrc/aten/sdp_choice_stub.cc`) from "no mask" to the one class Qwen-Image-2.1 passes: 4-D bool with `size(3) == key.size(2)` and `numel == key.size(2)`, which pins the shape to `(1,1,1,KV)` once the size-1 axes are. The kernel cannot read a bool mask as-is — FlagGems converts one itself at `attention.py:928-929` with `attn_mask.to(query.dtype) * -1.0e6`, the inverse of torch's convention, and skips that for a float mask — so `RouteMask()` builds the additive itself in fp32, `where(mask, 0, -1e6)`, and reshapes to `(1,1,1,KV)` before expanding because the caller's tensor carries a real stride on both size-1 axes and `_attn_fwd` indexes the mask by stride with nothing bounding batch, head or query row. **No routing configuration changed**: the clause lives in the hand-written override, so `backends_metax.conf` still reads 592 `flaggems` / 12 `flaggems_cpp` / 1433 `cuda` over 2037 entries (SHA-256 `bb1dc5c4550339dcd44ac438b2981e703c882025e477221e86cac37d833f58f2`) and no other platform's conf or route moves. Nothing falls back to the vendor: `zeros_like`, `full_like` and `where.self` are `flaggems` routes in that conf, and a `FLAGOS_LOG=dispatch` census of both spellings the route serves shows all three on `flagos_python` with no `fallback` line in the log. Ascend, GCU, MUSA, DCU, PPU, CUDA and TsingMicro are **not revalidated**. | Qwen-Image-2.1, 1024x1024, 40 steps, batch 1, seed 42, `--warmup 1`, card 0 on every arm: flagos with this route **776.9 ms/step / 31.62 s/image** against vendor MACA torch's **1661.1 ms/step / 66.96 s/image** and the same wheel with the clause unemployed (`FLAGOS_OP_scaled_dot_product_attention=cuda`) at **1702.8 ms/step** — **2.13x** and **2.19x**; a second window at `--min-run-time 50` reproduces all three (773.1 / 1660.2 / 1687.5 ms). Paired-latents arms (`--load-latents`, 774.8 / 1661.0 / 1687.4 ms/step): route vs boxed **49.05 dB / MAE 0.43833**, cuda vs boxed **46.58 dB / 0.56115** (the bring-up figure, reproducing as a control), cuda vs route **46.47 dB / 0.55726** — so the route's own image cost is below the flagos-vs-vendor difference the wheel already carries. Op level in the pipeline at `q(1,4096,32,128)` / `kv(1,4122,32,128)` / `mask(1,1,1,4122)`: **11.808 ms/call over 96 calls** against the boxing arm's **40.239 ms/call**, at `max|d| 0.000e+00` / `rel 0.00e+00` and NaN `0/0/0/0` against a direct `flag_gems.scaled_dot_product_attention` over the same operands; the mask's logged stride is `(4122, 4122, 4122, 1)`. The additive costs, on this part at KV 4122, `zeros_like 0.024 + full_like 0.037 + where.self 0.105` ms plus `0.010` ms of views — about 0.18 ms per routed call. `pytest tests/integration/ops/test_metax_flaggems.py -k SdpaRoute -x -q`: **13 passed, 91 deselected in 41.19s**; whole file **104 passed in 795.16s, 0 failed** (exit 0). `flaggems_overload_survey.py` v6 (SHA-256 `7b01c22c…`) rerun against the changed route: `registered 1` / `tested 1` / **`FAILED`**, unchanged from the entry above — its cause is the harness's synthesized `dropout_p = 0.5` in a non-deterministic call, and every case it synthesizes is outside the new mask clause as well. Every figure is a measurement from the shipped conf on the built wheel; none is inferred from the routing table. |
| 2026-09-19 | Hygon DCU bw1000 (8 devices) | DCU full-coverage configuration (458 active routes, harness v6) | Four changes to the DCU FlagGems path, none of which moves an operator between routes: `index_put_`/`_index_put_impl_`, hand-registered in `csrc/aten/register.cc` as a CPU round-trip and in the codegen's `MANUAL_REGISTERED_OPS` because a `Tensor?[]` index list has no `IValueToPython` conversion, now run FlagGems' Triton kernel through the dispatcher; L2 one-dim `linalg_vector_norm` stops going through FlagGems' `dim_compress` permute-and-`contiguous`, which was 99.3% of the op, and is spelled `sqrt(sum(x*x, dim, keepdim))` with both sub-ops already `flaggems`; the FlagGems `LibEntry` launch path (`run`, `key`, `_descriptor_cache_key`) stops rebuilding per-call state that cannot change between two launches of the same kernel on the same shapes; and Triton's persisted autotune cache is enabled for DCU builds whose conf routes to FlagGems, with an explicit `TRITON_CACHE_AUTOTUNING=0` still honoured. `torch_fl/configs/backends_dcu.conf` is not modified and still reads 458 `flaggems` / 1579 `cuda` over 2037 entries (SHA-256 `7b82cb492de80f3dc2dcb93deeba14764a986368460a63cae9d1deb90fee9163`). No op leaves the FlagGems route for CUDA boxing, no kernel, grid or cache key changes, and every function is gated on both `_build_accelerator() == "dcu"` and the conf actually routing to FlagGems, so it is a no-op elsewhere. The full 458-overload DCU re-survey ran to completion over this configuration's own route set — **458 registered / 381 tested / 342 basic executable / 297 `STRICT` / 45 `BASIC_ONLY` / 39 `FAILED` / 77 `UNTESTED`**, a basic rate of 74.7% and a strict rate of 64.8% over the same 458 active routes — so this is a third cohort with its own denominator rather than a revalidation of the 546-overload generic set. **The bw1000 rows in Hardware Summary and Raw Case Evidence keep their 546 denominators and are left as they were**, since they measure `backends_flaggems.conf`, which this change neither touched nor surveyed. **Ascend, GCU, MUSA, MetaX, PPU and CUDA are not revalidated** — no configuration other than `backends_dcu.conf` was involved and that one was not edited. | Measured on the Qwen-Image-2.1 1024x1024 6-step flow (the `tests/manual/qwen_image_21` driver added in #342) on a Hygon DCU bw1000 with DTK 6.3.26113, torch_fl `873f516`, FlagGems `5.4.0rc2.post1+g437ba3938`, FlagTree `0.6.0+hcu.git46341ffa`. `index_put_`: steady four-step block 24 calls at 15501.54 -> 388.79 us/call, last step 16720.84 -> 381.17, step wall 1.272 -> 1.180 s, last-step exclusive host sum 0.302 -> 0.209 s against 0.098 s for the op; bitwise equal to the round-trip (`equal=True`, `max|delta|=0`, also with `accumulate=True`) and the rendered image is byte-identical (md5 `6fc5c62c451ecf5772f11bb0e1652cfd`, 1875231 bytes). `linalg_vector_norm` (36 calls in the VAE decode, none in the denoise loop): 139999.20 -> 6248.30 us/call, 5.040 -> 0.225 s for the phase, against the vendor build's 247.26 us/call and a probe on `(1,144,1,2048,2048)` fp32 showing `dim_compress` 354869 us / 13.6 GB/s and `x.clone()` 3606 us / 1339.8 GB/s, with the two spellings agreeing to `max|delta| = 6.676e-06`. Pointwise launch path: the FlagGems pointwise ops are 402 ms of 837 ms of exclusive host time over four steady steps (`where` 196 ms over 520 calls at 376.48 us against the vendor's 34.94, `pow` 61, `tanh` 58, `rsqrt` 56, `silu` 31), with the kernels not the difference (`where` 74.0 us device on FlagGems against 62.1 on the vendor, enqueue 262.4 against 11.5); the block's exclusive sum falls to 0.767 s and the composites fall with it (`native_layer_norm` 204.31 -> 169.88 us/call, `tanh` 226.51 -> 212.80, `rsqrt` 216.23 -> 202.41, `silu` 220.45 -> 204.89) while the device-bound step wall does not move (1.179 -> 1.178 s). Autotune cache: three fresh processes on `torch.native_layer_norm` `(1,4096,4096)` bf16 give 454.2 ms (knob off, no entry) / 462.5 ms (knob on, cache miss) / 33.6 ms (knob on, the miss's entry on disk), the third profiling as `:236(run)` 0.032 s -> `:194(check_disk_cache)` 0.014 s with no benchmark; Qwen-Image-2.1 issues two tuning keys, measured at 0.38 s and 0.47 s. Focused re-survey of the twelve active routes this change touches (`linalg_vector_norm`, `sum.dim_IntList`, `sqrt`, `where.self`, `where.self_out`, `pow.Tensor_Scalar`, `pow.Tensor_Tensor`, `tanh`, `rsqrt`, `silu`, `native_layer_norm`, `mean.dim`) with `tests/manual/flaggems_overload_survey.py` (v6, SHA-256 `7b01c22ce3a94315f1364df242323e9faac27f2585debfb05030670c7c756cc7`) and `--ops` against the same conf SHA-256: 12 registered, 12 STRICT, 0 BASIC_ONLY, 0 FAILED, 0 UNTESTED; case-level 63 PASS / 21 INVALID_CASE / 0 ERROR / 0 WRONG / 0 CRASH / 0 TIMEOUT (84 = 12 x 7), the invalid cases being inputs ATen rejects before dispatch. `index_put_` is in no conf and no cohort. The full 458-overload DCU re-survey then ran to completion on the same hardware, conf SHA-256, harness version and torch-fl revision with the four patches applied: **458 registered / 381 tested / 342 basic executable / 297 `STRICT` / 45 `BASIC_ONLY` / 39 `FAILED` / 77 `UNTESTED`** over the configuration's own 458 active routes (basic rate 74.7%, strict rate 64.8%; `STRICT + BASIC_ONLY + FAILED + UNTESTED = 458`, `Basic executable = STRICT + BASIC_ONLY = 342`), its 3206 synthesized cases resolving as 1848 `PASS` / 1059 `INVALID_CASE` / 147 `ERROR` / 138 `WRONG` / 14 `CRASH` and no `TIMEOUT` or `VERIFIABLE`. All twelve affected routes came back `STRICT` — `linalg_vector_norm` 5 `PASS` / 2 `INVALID_CASE`, `sum.dim_IntList` 7, `sqrt` 7, `where.self` 1 / 6, `where.self_out` 1 / 6, `pow.Tensor_Scalar` 7, `pow.Tensor_Tensor` 6 / 1, `tanh` 7, `rsqrt` 7, `silu` 5 / 2, `native_layer_norm` 5 / 2, `mean.dim` 5 / 2 — and none is among the 39 `FAILED`, which are the cohort's pre-existing gaps on routes this change does not touch. |
| 2026-09-19 | Enflame GCU S60 (8 `flagos` devices) | The Qwen-Image-2512 50-step paired `torch_fl`-against-`torch_gcu` throughput cohort, and the five costs it measured | Cut the five costs behind the `torch_fl` leg's **301.77-304.81 s/it** against the vendor `torch_gcu` leg's **3.98-4.01 s/it** — about **76x** per denoise step — landing at **24.19-24.29 s/it**, **6.0-6.1x** and inside the one-order-of-magnitude budget. Four of the five are work that belongs on the card done through the host. (1) Every dtype cast and every strided copy, which GCU had no ATen kernel for and which the no-on-device-path arm of `csrc/aten/copy_ops.cc` and `csrc/aten/contiguous_ops.cc` served as a full D2H/copy/H2D round trip: `csrc/aten/backends/gcu/gcu_copy.{h,cc}` add `StridedCopy` and `DtypeCast` over `topsatenCopy` / `topsatenToCopy`, in the shape Ascend's `ascend_copy.h` already has, and `_copy_from`, `contiguous`, `clone` and `_to_copy` call them before their host arm. (2) `ScalarToDeviceTensor` staged every scalar operand as a full-size host tensor (`at::full(sizes, scalar, options.device(at::kCPU)).to(options.device())`, a host allocation, a host fill, an H2D transfer and a second device allocation); the fill now happens on the device. (3) Every binary op with a **Python-number** operand fell to the host silently, because `x * 2.0` is `aten.mul.Tensor` with a wrapped-number 0-dim operand carrying the number's own dtype — f64 for a float — while promotion deliberately stops it widening `self`; the generated guard read `other.scalar_type()`, saw f64 for the most ordinary spelling there is, and its fallback moved the tensor to the CPU and back with **no `cpu_fallback` line to show for it**, since the fallback is inside the kernel. `at::result_type` is wrapped-number aware, so the guard now reads it through a shared `_binary_dtype_guard` that keeps the `self` dtype test as well. (4) A fractional scalar bound was truncated on two paths: `int32 < 0.5` dispatches to `aten.lt.Scalar`, whose template guarded on `at::result_type` correctly and then converted the scalar with `ToTopsatenScalar(other, self.scalar_type())`, so it answered `False` for every value in `[0, 1)` — `lt` and `le` now cast `self` to the promoted dtype; and `clamp` filled an absent bound with the dtype's extreme, which truncates through `topsatenScalar_t`'s int64 member (INT64_MAX into an int32 clamp arrives as -1) — `gcu::IntegralExtreme` supplies the extreme at the tensor's own width and `gcu::ClampComputeDtype` promotes the bounds as ATen's clamp meta does, so `int32.clamp_min(0.5)` answers f32 while `int32.clamp_min(0)` stays int32. The fifth is the one routing change: `where.self_out` was on FlagGems, and ATen's eager `_safe_softmax` — the SDPA math path's once-per-layer composite — ends in exactly that overload, so the whole op inherited FlagGems' cost at a 0-dim value operand; it moves to `gcu` with the out-of-place template plus the `out=` contract, and both `where` templates now resolve their device through `gcu::TopsatenComputeDevice`, because `where.self` took `self.device()` and a 0-dim host `self` silently selects the host while handing the vendor device pointers. Route delta: exactly one route moves, `where.self_out` `flaggems` -> `gcu`, so GCU `flaggems` **255 -> 254**, `gcu` **176 -> 177**, `none` **1606** unchanged over the same **2037** routable ops and accelerated routes stay **431** (**21.2%**) — this buys throughput and not coverage. `gcu_register.inc` gains one `m.impl` line (183 -> 184) and `gcu_flaggems_register.inc` loses one (248 -> 247), so the reconciliations read `254 = 247 + 7` and `177 = 184 - 7` with the same seven `# gcu` markers (`clamp`, `fmod.Tensor`, `gelu`, `mean`, `mean.dim`, `remainder.Tensor`, `silu`) and no orphans and no overlaps in either direction; the op's `NATIVE_TRITON_GAPS["gcu"]` entry is added, since the route and the gap entry are two halves of one claim. The whole change is inside the GCU backend, the GCU generator and the GCU configuration, except `copy_ops.cc` and `contiguous_ops.cc`, which are shared and touched only in the shape Ascend's `ascend_copy.h` established — a `#if defined(USE_GCU)` arm beside the existing `#else` ascend arm, both headers supplying inline no-op fallbacks so every caller stays total — so no other platform's routes, kernels or behaviour move in this half of the change: Ascend, DCU, MetaX, PPU and Tsingmicro are unaffected rather than unvalidated, and MUSA is unaffected by it as well. The pull request carries a second, independent MUSA half — the same overload fails there for a contract reason rather than a throughput one — recorded in its own row below and in its own section above. | Denoise-loop rates, `(1, 4096, 64)` latent, one seed: the same 50-step paired run gives the vendor leg **3.98-4.01 s/it** (`50/50 [03:18<00:00, 4.01s/it]`) against the `torch_fl` leg's **301.77-304.81 s/it** (`50/50 [4:11:28<00:00, 304.81s/it]`), and three 8-step `stage: full` fixed-latent runs on one script and one latent measure **156.78-156.80** after the copy path, **24.29-24.40** after `where.self_out` and **24.19-24.29 s/it** after the scalar fixes. Per-op, every figure against the same ATen call on the CPU, on card 6 (card 5 is defective and hangs any `topsaten` op): `_to_copy` f32 -> bf16 at `(1, 24, 4114, 4114)` — 1.5 GiB — **437.9 ms** against the vendor plugin's **5.8 ms** and **5.6 ms** issued directly, and **5.90-5.93 ms** in the pipeline across the six attention layers against the vendor's 5.813 and 5.919; `small.expand(full).contiguous()` **975.6 ms** against **14.2 ms** for the same 1.5 GiB as a clone and **4.6 ms** directly; the scalar host fill **265.2 ms** against **3.7 ms**, with `a + 1.0` **630.1 ms** against **17.2 ms** at `(1, 4096, 3072)` bf16 (fill 4.6 against 0.3 ms); `a * 2.0` **575.474 -> 21.131 ms** and `a * 2` **615.780 -> 21.106 ms** against a **19.993 ms** full-size-device-tensor reference, with `torch.profiler` on the before-state showing two 219.6 ms `topsMemcpy` and **zero** `topsLaunchKernel`; and `_safe_softmax` **1131.07 -> 61.85 ms**, the four spellings of its body that contain `where` moving 1095.36-1130.40 -> 27.09-61.96 ms while `SIA`, the one without it, is 37.74 -> 37.77 ms. Correctness, one process each, all comparing against the CPU reference: `/tmp/guard_verify.py` **35 cases, 0 failures** — f32/f16/bf16/i32/i64 cross products of `* 2.0`, `* 2`, `+ 1.0`, `== 2` and `< 0.5`, plus a genuine f64 operand, a broadcast, a non-contiguous operand, the empty case, two in-place and two `out=` spellings; `/tmp/clamp_probe.py` **60 cases, 0 failures** — nine dtypes over `clamp(0.5, 1.5)`, `clamp(0, 1)`, `clamp(-1.0, 1.0)`, the one-sided and inverted forms, a large int32 bound, non-contiguous and empty, asserting the result dtype as well as the result; `/tmp/cmp_int32.py` **0 mismatches** on int32, int64, int16 and uint8 against twelve fractional bounds; `/tmp/copy_probe.py` **36 cases, 0 failures** — `StridedCopy` and `DtypeCast` over seven dtype pairs, six strided/offset/empty/broadcast copies, three cross-card peer round trips and a 0.03 GiB H2D + D2H check; `/tmp/whereout_probe.py` **13 cases, 0 failures** — the `out=` contract including the 0-dim `self` `_safe_softmax` passes, a wider condition, a grown empty `out`, a non-contiguous `out` on the host path and the uncastable-dtype `TORCH_CHECK`. Bit-exactness, because the wrapped-number, comparison and clamp fixes move real arithmetic onto the card and the two do not agree bit for bit: a 45-spelling probe finds **41 non-bit-exact** spellings, every one at **1 ulp**, and the spellings this pipeline uses (`* 2.0`, `* 2`, `+ 1.0`, `- 1.0`, `/ 2.0`, `** 2.0`, the sigma schedule's `/ 3.0`) are **exact**; the 8-step fixed-latent image at `ff69df70…` is unchanged through the copy and `where.self_out` fixes and moves to `526f05c1…` through the scalar ones — 26.16 % of bytes differing, mean \|delta\| 0.441/255, max 40/255 — which is a chaotic system amplifying 1 ulp across 8 steps and a move toward the vendor's own on-card arithmetic rather than away from it. Regression coverage: `TestMulPythonNumberOperand`, `TestLeScalarCorrectness` and `TestWhereOutCorrectness` add **18 cases** in `tests/integration/ops/test_mul_dispatch.py`, `test_le_dispatch.py` and `test_where_dispatch.py`; those three files report **34 passed, 9 skipped in 29.50s**, the two marker selections `.github/configs/gcu.yml` uses report **621 passed, 26 skipped, 633 deselected, 2 xfailed, 2 xpassed** and **7 passed, 4 skipped, 1272 deselected, 1 xpassed** and both exit 1 on the same two FlagGems-runtime subprocess cases (`TestMeanDimDispatch`, `TestSiluDispatch`) that die importing Triton because the child is spawned with `sys.executable` and does not inherit the private glibc 2.39 loader this session runs under — an artifact of this Ubuntu 22.04 host rather than of the change, since CI runs ubuntu 24.04 — while `-m anyplatform` is **614 passed, 8 skipped, 661 deselected, 2 xfailed, 1 xpassed** with no failure, and `tests/unit/` reports **486 passed, 98 skipped, 2 failed**, both failures a pre-existing ordering leak in `tests/unit/test_musa_rng_bridge.py` that this change does not touch (the file is byte-identical to the base commit's copy, passes alone at **3 passed in 7.88s** and reproduces at **2 failed, 5 passed in 8.45s** when `tests/unit/test_ascend_platform_marker.py` runs first and leaves the module global `_BACKEND_CONFIG_PATH` on a nonexistent temp conf). Survey, because one route moves *from* `flaggems`: `tests/manual/flaggems_overload_survey.py` v6 (`7b01c22c…`) against exactly the shipped `backends_gcu.conf` (`476825db…`), flag-gems 5.3.2, flagtree `0.6.1+enflame3.6`, ~60 minutes — **254 registered / 194 tested / 191 basic-executable / 121 strict / 70 basic-only / 3 failed / 60 untested** over **1778 cases** (981 pass, 705 invalid case, 80 error, 12 wrong), superseding the 2026-09-15 GCU cohort's harness-v4 run over a transient 374-route draft of the same file; the three failures are `gcd_`, `lcm` and `lcm_`, each on the `2d-i64` profile only, each `RuntimeError: Pipeline run failed: PassManager execution failed` from `triton/backends/enflame/compiler.py:253 make_gcuir`, a FlagTree i64-lowering failure on three routes this change does not touch. Generator idempotency: two runs leave all four artifacts byte-identical (`IDEMPOTENT=YES`). Artifact SHA-256 before -> after: `backends_gcu.conf` `4e8265af…` -> `476825dba7f3050f4d91590bce29362c50f7d23df6f8c952fba1417b51afc05f`; `gcu_kernels.cc` `1d7d4f9b…` -> `57e6edc95b6ec6cbd0f20c19d17214bf5f6283aa6562f59744622d0533efacc1`; `gcu_register.inc` `638619ee…` -> `dc88d5bca113669315be122703f791bc6f1624918324d55fdb99138d2c763b0e`; `gcu_flaggems_register.inc` `c520a93a…` -> `be8431800f21fab5038633e4dc79baa84dc317ca7aa9425f05607233b6e88364`. `ruff check .` — "All checks passed!"; `ruff format --check .` — 272 files already formatted. **Evidence gaps:** the vendor rate is the 50-step paired run's and the three fixed-latent rows are 8-step runs, so the headline ratio compares two runs per denoise step rather than one, and no vendor 8-step run was taken to make it a same-run figure; the 8-step image is a fixed latent and not a prompt cohort, so it establishes determinism rather than quality; the 1-ulp probe is 45 spellings on one shape family rather than a dtype-by-operator matrix; the survey covers the 254 ops GCU routes to FlagGems and not the generic 546-overload cohort the hardware summary measures, and its three failures are recorded with the compiler frame that raises them rather than diagnosed further into FlagTree; card 5 is defective, so nothing was measured on it; the four integration failures in the local runs are the runner's and not this change's, since the suite's subprocess tests spawn `sys.executable` and so lose the private glibc 2.39 loader this session runs under while CI runs ubuntu 24.04; and the probes are uncommitted. |
| 2026-09-19 | MUSA MTT S5000 (CI lane, not hardware-revalidated) | The one `out=` case this pull request added, and the FlagGems route it fails on | Moved `where.self_out` off FlagGems on MUSA, `flaggems` -> `musa`, because FlagGems' wrapper does not implement ATen's `out=` contract: it computes the broadcast shape only when `out is None` and otherwise hands the caller's destination to `where_inner(..., out0=out)`, whose `pointwise_dynamic.prepare_args` validates instead of resizing, so a destination that does not already match — including the empty one ATen's own overload grows — raises instead of being grown. It is the same overload the GCU half moves, for a different reason: GCU's is a measured **46x** throughput route change, this is a correctness one selected by a CI failure rather than a sweep. `T_WHERE_OUT`, a new `where_out` category in `scripts/codegen/codegen_mudnn.py`, is the out-of-place ternary template plus the contract — `at::infer_size` over all three operands, `out.resize_()` when the shape differs, the standard empty-output guard, and a `TORCH_CHECK` that the destination's dtype already equals the promoted result dtype — ATen's rule, raising ATen's phrasing — before either arm runs, then `mudnn::Ternary` `SELECT` — and it reuses the `out.resize_` / `out.copy_(host)` idioms the MUSA `mm`/`bmm` `out=` fallbacks in the same generated file already use. Generated, not handwritten: a new `NATIVE_TRITON_GAPS["musa"]` entry carries the route, and the conf is regenerated rather than edited. Route delta: `flaggems` 468 -> **467**, `musa` 51 -> **52**, `none` **1518** unchanged over the same **2037** routable ops, accelerated **519 (25.5%)** unchanged — correctness and not coverage. `musa_register.inc` 160 -> **161** `m.impl` lines and `musa_flaggems_register.inc` 359 -> **358** with its excluded list 14 -> **15**, so the reconciliations read `467 = 358 + 109` (the 109 being the conf's `# musa` annotations, native kernels FlagGems still outranks) and `52 = 161 - 109`; before the regeneration the two register files overlapped in exactly one op — this one, which registers twice on PrivateUse1 and warns — and after it they are disjoint. The other FlagGems-first confs still route the overload to FlagGems (`cuda`, `dcu`, `metax`, `ppu`), so the gap is latent there and is **not** fixed here. MUSA is **not hardware-revalidated** by this entry in the sense the introduction fixes: no MTT S5000 `flaggems_overload_survey.py` cohort was taken against the new 467-route configuration. It *is* device-tested: the CI lane that caught the failure was re-run on the fixed tree and the group passes with no failure. | The MUSA lane of PR #351, group *Run operator tests (native mudnn, main ops)*: `tests/integration/ops/test_where_dispatch.py::TestWhereOutCorrectness::test_out_grows_from_empty` — `torch.where(cond, a, b, out=torch.empty(0, device=flagos))`, expecting the `(2, 4)` broadcast shape — fails with `RuntimeError: out tensor at index 0 shape is invalid, should be (2, 4) but is torch.Size([0])!`, raised out of `flag_gems/utils/pointwise_dynamic.py:1699` and reached through `torch_fl/__init__.py:1321` (`_CudaAliasMode.__torch_function__`) and `flag_gems/ops/where.py:81`, on flag_gems `5.4.0rc2.post1+g437ba3938`; the group reports **1 failed, 507 passed, 6 skipped, 646 deselected, 2 xfailed, 1 xpassed in 228.66s** and that case is its only failure, while every other `where` case including the same overload at a matching shape passes. It surfaced on MUSA and not on CUDA because the case carries `@pytest.mark.anyplatform` alone and MUSA's manifest selects `(anyplatform or main_ops)` where CUDA's selects `main_ops`. The same lane re-run against the fixed tree — PR #351 run `35417994902` at `e8bd278`, same job and group — reports **508 passed, 6 skipped, 646 deselected, 2 xfailed, 1 xpassed in 227.81s** with no failure, one more passing case than before: both `TestWhereOutCorrectness::test_out_grows_from_empty` and `::test_out_rejects_uncastable_dtype` are `PASSED` in its per-case listing, so the kernel's resize, its mudnn `SELECT` and its strict dtype check all ran on an MTT S5000, and the job completed `success`. Locally (no MUSA host in that session): `tests/unit/test_gen_vendor_confs.py` **36 passed** with the pinned gap set widened and the diagnosis recorded in its docstring; `tests/unit/` **486 passed, 98 skipped, 2 failed**, both the pre-existing ordering leak in `tests/unit/test_musa_rng_bridge.py` (byte-identical to the base commit's copy, **3 passed in 7.97s** alone); `ruff check .` — "All checks passed!"; `ruff format --check .` — 272 files already formatted. Generator idempotency: a second run of `codegen_mudnn.py`, `codegen_musa_flaggems.py` and `gen_vendor_confs.py` leaves all four artifacts byte-identical (`IDEMPOTENT=YES`). The destination-dtype rule the template carries comes from a CPU probe over six destination dtypes (scratch, uncommitted): over an f32 result ATen's own overload raises `Expected out type to be Float but got Double` — and the same for `Half`, `Int`, `Long`, `Bool`, `Byte` — for every dtype that is not the result's, whether or not the result would fit there, so the template enforces that same equality; this is deliberately tighter than the sibling GCU template `T_WHERE_SELF_OUT`, which admits anything `c10::canCast` accepts and so narrows an f32 result into an f16 destination rather than raising. Artifact SHA-256 before -> after: `backends_musa.conf` `e62cac76…` -> `87d150533c73e4ca40a24c2588aed51387d257044290d1dd85e8cc9a9d40ffad`; `musa_kernels.cc` `f0edc538…` -> `2002b5fdb087b56385b7ee27fe604c902ae43146da46f376c54f691aec3a7483`; `musa_register.inc` `22dd2327…` -> `7b4e0480f0fc1777a61cda1c5010aa2a0c0ff2e6a9e43cf92c6b4f5413b4ed8a`; `musa_flaggems_register.inc` `14ffc529…` -> `13474b812da85f88dbb4d3ef6a2f44ac04a88cf7c286b63e2d194021c9dfb4ab`. **Evidence gaps:** the failing run is the CI lane and the passing evidence is the same lane re-run on the fixed tree, so the kernel was compiled and executed on device by CI but was never exercised from a session on this host; no MTT S5000 `flaggems_overload_survey.py` cohort was taken against the new 467-route configuration, so MUSA's FlagGems support table is unchanged and remains measured against its own 2026-09-15 cohort; the generated kernel is not locally compilable here, the MUDNN toolchain not being installed on this host; no timing was taken, so nothing is claimed about mudnn's `SELECT` against FlagGems' `where_self_out` at the shapes the overload is called with; and the same `out=` contract gap is left in place on `cuda`, `dcu`, `metax` and `ppu`. |
| 2026-09-18 | MetaX C550 (8 devices) | MetaX composite SDPA routed to FlagGems (hand-written whole-op override, outside the 639-overload ceiling) | `aten::scaled_dot_product_attention` became a `flaggems` route in `backends_metax.conf`, taking the file to **592 `flaggems` / 12 `flaggems_cpp` / 1433 `cuda`** over a **2037**-op list (SHA-256 `bb1dc5c4550339dcd44ac438b2981e703c882025e477221e86cac37d833f58f2`), superseding 591 / 12 / 1433 over 2036. The op cannot be routed leaf by leaf: it is a composite whose fused-backend selection runs *inside* it and then branches on `query.device().type()`, so on PrivateUse1 the generated leaves are never consulted and no per-op route can reach a fused kernel. The override is hand-written — `csrc/aten/sdp_choice_stub.cc` registers the composite itself on PrivateUse1 and decides inside it — and it is registered only on the CUDA-boxing builds, which is why `gen_vendor_confs.py` gains `EXTRA_ROUTED` (an op that appears in no `.inc` is invisible to every coverage scan, so the op list has to be widened by hand) and `METAX_COMPOSITE_FLAGGEMS` to hold it from every other platform's conf the way `METAX_FLAGGEMS_MEASURED` holds the measured leaf routes. It is gated on a new `HasBackendForOp()` in `csrc/aten/common.h`, which tells a conf that names the op from one that is silent about it: without it the op would read `kFlagGems` from `GetBackendForOp`'s table-miss default, and a conf written before this change — including a third party's wheel — would take the route silently. The envelope is bf16, 4-D, head_dim 128 exactly, query seq >= 1024, no mask, no causal, no explicit scale, no dropout, no gqa; every shape outside it falls through to the pre-existing boxing path. Ascend, GCU, MUSA, DCU and PPU are **not revalidated** — each gained exactly one line, `none` on ascend/gcu/musa and `cuda` on dcu/ppu, and no measurement transfers to them. | Qwen-Image-2512, 1024x1024, 50-step denoise, one seed, on the 8-device C550 host with `flagtree 0.6.1+metax3.6`, MACA 3.8.0 in CUDA-boxing mode, `flag_gems 5.4.0rc2.post1+g5a58df410` and Triton 3.6.0 (`metax`). Pipeline, `build_pipeline` and placement held constant: steady **78.49 s against 184.02 s**, first step 90.26 s against 189.34 s (**2.34x**, 57.3% off the loop), `pipeline_load_s` 134.74 in both arms; the same window re-run as three arms that also write latents and pixels gives 84.46 s with the route active against 183.68 s and 182.39 s with it off. Op level at the shape the route serves, `(1, 24, 4114, 128)` bf16, min of five calls after warm-up: FlagGems `5.908 ms/call` against the boxing route's `23.313 ms/call` (**3.95x**). A 19-clause envelope probe reports `route hits: 6` — the joint shape and its seq-4096/2048/1024/non-contiguous variants — while head_dim 64/256/512, seq 1023, fp16, fp32, 2-D, 3-D, mask, `is_causal`, scale, dropout and gqa all box. Head_dim is measured in both directions: at 512 the kernel has no configuration that compiles on this part (`triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 294912, Hardware limit: 65536`, against an autotune set `attention.py:173` caps at `BLOCK_N <= 32`), while 128 and 64 both complete and match the unfused fp32 math path (`max 2.189e-05 mean 2.365e-06` and `max 2.153e-05 mean 2.421e-06`), so 128 is where the route was measured and not the kernel's outer boundary. Numerics over the same seeded operands (digest `37efe866eb24ada5` in all three arms): route against boxing `max 9.766e-04 mean 3.274e-05`, and the two boxing arms — `FLAGOS_OP_scaled_dot_product_attention=cuda` on the shipped conf and a conf that never names the op — agree exactly (`max 0.000e+00 mean 0.000e+00`), which is the measurement of the `HasBackendForOp()` rule. Image cost with its own control in the same window: the two route-off arms differ by `max 0.0000 mean 0.00000`, and route-on against them by `max 170.3320 mean 2.04607 (/255)`, 1150098 of 3145728 pixels over 1/255 and 312147 over 4/255, against the prototype arms' `1.39572` route effect and `1.25745` flagos-vs-vendor. Survey: `tests/manual/flaggems_overload_survey.py` v6 (`7b01c22c…`) scoped to the changed route reports `registered 1`, `tested 1`, verdict **`FAILED`**, `basic_executable 0`, `strict_support 0`; the cause is measured as the harness's `default_for()` catch-all `return 0.5` synthesizing `dropout_p = 0.5` (the argument's name is `dropout_p` but the branch ends on `return 0.5`, which names `p`), and the same seven profiles at `dropout_p = 0.0` turn all four `WRONG` verdicts into `PASS` with the three `INVALID_CASE` profiles unchanged — and none of the four is inside the route's envelope, so the verdict is the harness's and not the route's. Regression coverage: `tests/integration/ops/test_metax_flaggems.py` gains `TestMetaXFlaggemsSdpaRoute` and `_MEASURED_FLAGGEMS_ROUTES` moves 591 -> 592. On the C550 host with the route active the whole file reports **101 passed in 854.53s, 0 failed** (exit 0), the class subset **10 passed in 43.28s**, against 90- and 91-test cohorts on the two entries below. `gen_vendor_confs.py` idempotent (two runs, all nine confs byte-identical; `--check` reports `all vendor confs up to date`). Full detail: "MetaX: `scaled_dot_product_attention` routed to FlagGems" above. |
| 2026-09-18 | Enflame GCU S60 (8 `flagos` devices) | The four CPU fallbacks left in the Qwen-Image-2512 `torch_fl` + `diffusers` cohort, measured on the pipeline's own warm-up log rather than on a stand-in probe | Took the last four `cpu_fallback` operators of the Qwen-Image-2512 pipeline off the host, with five overloads routed to `gcu`: `where.self`, `all`, `index.Tensor`, `nonzero` and `nonzero_static`. This is the first route change on this platform selected by a census of the workload under test rather than by a probe standing in for it. **The census:** both halves of the paired cohort (`FLAGOS_LOG=dispatch,fallback`) log `cpu_fallback` before their first prompt and agree operator for operator -- **62 `cpu_fallback` calls in exactly four operators**: `aten::where` 56, `aten::nonzero` 2, `aten::index` 2, `aten::all` 2, against 489,770 dispatch records in the first half (366,633 `gcu`, 122,583 `flagos_python`; the second half logs 429,692 records with the identical fallback census) and no fifth operator in either log. `csrc/aten/fallback.cc:21` prints the schema name and not the overload, so the same log's **4,215 `where.self_out` and 4,215 `all.dim` `flagos_python` dispatch lines** identify the counted spellings as `where.self` and whole-tensor `all`: the `out=` and `dim` siblings were already routed. `nonzero_static` is in neither log -- it is the one call the training probe below left, and it is here because the reason that entry gave for declining it was partly wrong. **`where.self`:** `topsatenWhere` requires a bool condition and a result dtype `TopsatenWhereDtype` lists (Float, Double, Half, BFloat16, Long, Int, Short, Char, Byte, Bool), which excludes the float8 and complex results `ToTopsatenDataType` raises on; anything else takes the host path. The vendor does not broadcast, so **all three** operands are expanded to `at::infer_size(infer_size(cond, self), other)` -- the condition included, because ATen broadcasts it against the values' common shape and that is not the condition's own shape when the condition is the wider one, so a kernel expanding only the two values is right on the common case and wrong on that one. **`all`:** `topsatenAll` aborts on a bare rank-0 descriptor, and `TopsatenTensorWrapper` already rewrites rank 0 to a one-element vector, so the 0-dim ATen result passes straight through -- which matters because `reshape` and `squeeze` are themselves dispatcher calls GCU carries no kernel for. The output dtype is ATen's and **not uniformly bool**, a measurement rather than a reading: against ATen's own `meta` function, `all` on a uint8 operand returns uint8 and on every other dtype (bool, int8, int16, int32, int64, float16, bfloat16, float32, float64) returns bool, so a backend answering bool unconditionally is the deviant one and the kernel allocates `kByte` for a byte operand. A card-4 probe calling `topsatenAll` directly showed the vendor writing the correct 0/1 into either a `U8` or a `PRED` descriptor, so no staging cast is needed; half and bfloat16 operands are widened to float32, which is what lets the transformer's bf16 mask reach the vendor. **`index.Tensor`:** one dispatcher entry for several operations, and the vendor implements one -- a single rank-1 index, which is `index_select` on dim 0. The template recognises that spelling (exactly one index, present, defined, `dim() == 1`, a dtype `TopsatenIndexDtype` admits, every coordinate non-negative) and delegates to the `IndexSelectKernelGcu` this generator already emits, inheriting its guard, so it declines on the same host path for a 0-dim `self`, an unsupported dtype or a non-contiguous operand. A 0-dim index is excluded by the rank test and not by oversight -- `t[tensor(1)]` drops the indexed dimension where `index_select` keeps it, so it is a different operation; a bool index is a mask rather than a coordinate list; negative coordinates are excluded because ATen wraps them and the vendor resolves them outside. Every other spelling marshals the index tensors to the **host alongside the operand**, since ATen's host implementation runs on whichever device holds the data and a device index would reach it as a device pointer. **`nonzero` / `nonzero_static`:** both need three vendor calls rather than one, because the output's first dimension is the count and nothing about the answer is known before the data has been read once -- `topsatenCountNonzero` into a device int32, `topsatenNonzero` for the int32 coordinates, `topsatenTo` to widen to the int64 ATen declares. The count is read back to the host rather than used as the output size directly, and that is not defensiveness: `topsatenNonzero` **refuses a partial write outright** -- handed an output narrower than the count it returns `BAD_PARAM` and writes nothing, measured as `(2, 3)` failing over a rank-3 input with four nonzeros while `(4, 3)` and `(6, 3)` both succeeded -- so the coordinates always go into an `(nnz, rank)` staging description and the *read-back* is what gets shortened, in either direction, through `TopsatenRowView`. `nonzero_static` pads through that staging, which is why it stages in int32 at all: the tail must be filled and the vendor has no int64 fill, so `topsatenFill_` fills before the coordinate rows are written. It declines a `fill_value` outside int32 range when a fill is needed, a negative `size`, a rank-0 operand and an empty one; the first two take the host path so ATen's own message is the only spelling of that error that cannot drift. **Correction to the 2026-09-17 entry:** its closing claim that `nonzero_static`'s "schema has no `out=` overload, so there is nothing to route it to" is wrong -- `nonzero_static.out` exists and is a conf line -- and "int64, therefore no kernel" does not follow, since int64 is the *output's* dtype and only the coordinates need widening. Everything was added to `scripts/codegen/codegen_gcu.py` and `csrc/aten/backends/gcu/topsaten_common.h` rather than handwritten. Route delta: `flaggems` stays **248**, `gcu` 171 -> **176**, `none` 1611 -> **1606** over the same 2037 routable ops, accelerated 426 -> **431** (20.9% -> **21.2%**); `gcu_register.inc` 178 -> **183** `m.impl` lines and `gcu_flaggems_register.inc` unchanged at 248, so the reconciliations read `248 = 248 + 0` and `176 = 183 - 7` and **no route moved *from* `flaggems`** -- which is why no S60-wide `flaggems_overload_survey.py` run is claimed here. Three of the five do move the second file's provenance banner, 97 -> 100 "further FlagGems ops already claimed by `gcu_register.inc`" (`all`, `nonzero` and `where.self` are inside FlagGems' own coverage; `index.Tensor` and `nonzero_static` never were), and their `NATIVE_TRITON_GAPS` entries **stay**, since a gap entry only takes an operator off FlagGems and deleting one would put it back on a path that cannot serve it. The whole change is inside the GCU backend and the GCU generator, so Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are unaffected rather than unvalidated. | A 67-case probe (`/tmp/verify_five.py`, one process, card 4, `FLAGOS_LOG=dispatch,fallback`) runs every case against the same ATen call on the CPU and compares shape, dtype and values -- exact for bool and integer results, `allclose(1e-3, 1e-3)` for floating ones -- reporting **`RESULT: all 67 cases passed`, 0 failures, 0 `cpu_fallback` lines and 67 dispatch lines, every one `-> gcu`** (`where.self` 15, `all` 15, `index.Tensor` 15, `nonzero` 12, `nonzero_static` 10), with one case on the host path on purpose and checked for ATen's answer rather than accepted for running. Coverage by family: `where` over same-shape, a narrower and a wider condition, a scalar `self`, eight dtypes, a mixed float32/int64 pair, the empty case and rank 0; `all` over ten dtypes on all-ones, a buffer containing a zero, two empty shapes and both rank-0 answers; `index` over 1-D int32 and int64 indices, duplicates, empty rows, a bool mask, a 0-dim index, two 1-D indices, the `try`-then-index spelling, six `self` dtypes and a 3-D operand with an int64 index; `nonzero` over six dtypes, an all-zero buffer, ranks 1 and 3, two empty shapes and rank 0; `nonzero_static` over an exact fit, a pad, a truncate, `size = 0`, a large pad, an int64 `fill_value` large enough that the narrowing matters, two empty shapes, rank 0 and rank 3. One case found a real bug and the fix is the interesting half: the first run was 66 of 67, with `all` on a uint8 operand returning `tensor(True)` where the CPU returns `tensor(1, dtype=torch.uint8)` -- the generated kernel had hardcoded `at::kBool`, which reads correct and is not, and the ten-dtype meta-and-CPU measurement is what turned "the test is too strict" into "the kernel is wrong". Generator idempotency: two runs leave all four artifacts byte-identical (`IDEMPOTENT=YES`). Artifact SHA-256 before -> after: `backends_gcu.conf` `366d986f…` -> `4e8265afbf12b472097588ea1f7ae67f627b37711f2e3c17200f190588e0ea40`; `gcu_kernels.cc` `7f34288a…` -> `1d7d4f9b6f2830fb67a78bfeeb48582fa59a4e369287669a3486ad92745ef843`; `gcu_register.inc` `806d7ce7…` -> `638619ee4b080a0839ad225f2b82d56684dcee219d95f903899014e5c3d5f954`; `gcu_flaggems_register.inc` `9c9c99d0…` -> `c520a93a3ab23a196d836f349547c6535ba6a84b0a1490cf04f5c2309b7f210d`. `ruff check .` -- "All checks passed!"; `ruff format --check .` -- 270 files already formatted; `tests/unit/test_gen_vendor_confs.py` -- 36 passed; the rest of `tests/unit/` under the glibc 2.39 loader -- 482 passed, 98 skipped, 2 failed, both in `tests/unit/test_musa_rng_bridge.py`, byte-identical to the base commit's copy of that file. **Evidence gaps:** the census is the pipeline's warm-up region rather than a full sweep, so 62 is a floor on the pipeline's cost and not a per-prompt or per-step total; card 5 is defective and hangs any `topsaten` op on a tensor resident there, so the verification ran on card 4; the 67 cases are one shape family each rather than a shape sweep; the probe scripts are uncommitted; and no timing was taken, so what each round trip was worth is not measured -- only that the calls stop reaching the host. |
| 2026-09-18 | Hygon DCU bw1000 (8 devices) | `scaled_dot_product_attention` on DCU: the MetaX FlagGems composite route measured and rejected | **No route changed.** `backends_dcu.conf` keeps `scaled_dot_product_attention = cuda` and `METAX_COMPOSITE_FLAGGEMS` in `scripts/codegen/gen_vendor_confs.py` stays MetaX-only; the conf's SHA-256 is unchanged by this entry. PR #347 made the same whole-op override reachable on DCU — `HasBackendForOp()` sees the op because the regenerated conf names it, and the FlagGems branch of `csrc/aten/sdp_choice_stub.cc` is compiled into the DCU wheel (`FLAGOS_BUILD_FLAGGEMS=1`) — so following MetaX would have been one conf line plus a `DCU_COMPOSITE_FLAGGEMS`-style hold, and it is measurably not worth it. Ascend, GCU, MUSA and PPU are untouched and **not revalidated**. | Host and build held fixed; only the route moves. Three arms at bf16, no mask/causal/scale/dropout/gqa, `torch.ops.aten.scaled_dot_product_attention.default`, one process per arm, median of five calls after two warm-up calls, every window closed by `torch.cuda.synchronize(flagos:7)`; DTK 6.3.26113 / torch 2.10.0+cpu on `flagos:7`. Median ms/iter, `(1, 24, 4114, 128)` against `(1, 24, 12576, 128)`: math decomposition (`FLAGOS_DCU_SDPA_FLASH=0`) **16.7 / 167.5**, DTK CUTLASS flash adapter (shipped conf) **1.3 / 10.2**, FlagGems Triton (`FLAGOS_OP_scaled_dot_product_attention=flaggems`) **3.2 / 25.9** — the adapter is **2.5x** faster than FlagGems at both shapes, 15.7x against the math decomposition at the larger one where FlagGems is 6.5x. The arms are attributed to kernels rather than to latency: the FlagGems arm's routed output is **bit-identical** to a direct `flag_gems.scaled_dot_product_attention(q, k, v)` call (`torch.equal` → `True`, `max\|diff\|` 0.000e+00) and the shipped arm's is not (`max\|diff\|` 9.766e-04), and a profiled run of the shipped arm reports `aten::_flash_attention_forward` plus `flash_fwd_kernel_16x64_prefetch<…cutlass::bfloat16…>` with zero triton/flaggems kernels. #347's MetaX case does not transfer: there the boxing route had no *reachable* fused kernel, while on DCU DTK ships a CUTLASS flash adapter and the defect was that the composite never selected it. The same session corrected two figures the 2026-09-17 entry below carried: its harness closed each window with `torch.cuda.synchronize()` and no device argument, which syncs `current_device()` — `0`, while the tensors were on `flagos:7` — and returns immediately (`no-arg(current=0)=0.04ms sync(flagos:7)=10.18ms sync(flagos:0)=0.05ms no-arg(after set_device)=10.18ms`), so 102.1/6.1 were submit times against true values of 167.5/10.2; the ratio and the VRAM figures were unaffected. The A/B is reproducible from `tests/manual/dcu_sdpa_ab.py --device flagos:7`. Full detail: "`scaled_dot_product_attention` moved onto DTK's CUTLASS flash adapter" above. |
| 2026-09-17 | Enflame GCU S60 (8 `flagos` devices) | The index, embedding and upsample CPU fallbacks, measured on a training step and on the Qwen-Image-2512 VAE decode | Gave the last twelve measured fallback overloads a topsaten kernel, so the training probe and the VAE decode both stop reaching the host: `index_select[.out]`, `index_fill.int_Scalar[_out]`, `index_fill.int_Tensor[_out]`, `index_fill_.int_Scalar`, `index_fill_.int_Tensor`, `embedding_dense_backward[.out]` and `_upsample_nearest_exact2d[.out]`. None of the twelve had a PrivateUse1 registration before this change, which is why every call to any of them was a host round trip by construction rather than by accident. The set came from two censuses rather than from a survey. A training step -- two Adam steps over a five-parameter model with an embedding lookup and an `index_select` path, run with `FLAGOS_LOG=fallback` -- recorded **11 `cpu_fallback` calls in four operators**: `aten::index_select` 4, `aten::embedding_dense_backward` 4, `aten::index_fill_` 2, `aten::nonzero_static` 1; the names are the schema names `csrc/aten/fallback.cc:21` logs and not the overloads, so the eight `index_*` / `embedding_dense_backward` spellings above appear as three and the 2 `aten::index_fill_` are its `index_fill_.int_Scalar` and `index_fill_.int_Tensor` calls, one each. A Qwen-Image-2512 VAE decode at 1024x1024 supplied the fourth name: it makes three `_upsample_nearest_exact2d` calls, all of them `cpu_fallback` on the pre-change tree, and it is the only caller of that op in either workload. Three findings are worth recording because each contradicts the rule the rest of the GCU backend follows. (1) `topsatenIndexSelect` takes **both** int32 and int64 indices, which the "topsaten has no int64 kernel" rule would forbid: a coordinate is consumed as an integer rather than as element data, so `TopsatenIndexDtype` admits both -- measured at dim 0, 1 and 2 and with a 0-dim, empty and non-contiguous index, each matching the host result in both dtypes -- and two limits the vendor's dtype table does not express (every dim under 2\*\*24 elements, every buffer under 4.0 GB) are checked rather than discovered, since past either the kernel returns an error instead of a result. (2) `topsatenIndexFill` wants an int32 index where ATen takes int64 and nothing else (`index_fill_(): Expected dtype int64 for index.`), and its own bounds check is **not safe to lean on**: an index equal to the dim size satisfies it and writes one element past the end of the tensor, and a larger index aborts the process (`Index out-of-bounds! bound=2, index=5`), while ATen accepts a negative index here and wraps it (-1 in dim 0 of a `(2, 3, 4)` fills row 1, where -3 raises). `TopsatenIndexFillIndex` therefore reads the index back once, wraps negatives and narrows in the same pass, and returns an undefined tensor -- which sends that call to the host path -- for a value still out of range or not representable as int32, because a truncating cast would fold an out-of-range index back into range. (3) `topsatenEmbeddingDenseBackward` wants int32 where ATen supplies int64, and the narrowing again cannot be a cast: ATen does not raise on an index outside `[0, num_weights)`, it **ignores** that element of `grad` (measured for 9, -2 and 2\*\*32 + 1 against a 6-row table, all three leaving only the in-range contributions), so 2\*\*32 + 1 would fold back to 1 and the vendor would accumulate that row into table row 1 -- a silently wrong gradient rather than an ignored element. `TopsatenEmbeddingIndex` validates in the same pass that narrows, and a `num_weights` above `INT32_MAX` declines for the same reason. The `index_fill` `out=` spellings compute into a temporary through the same kernel and `copy_` into a resized `out`, because the vendor entry point fills its first operand in place; the tensor-valued spellings require a 0-dimensional `value` because ATen does (`index_fill_ only supports a 0-dimensional value tensor, but got tensor with 1 dimension(s)`); an empty index is a no-op after validation rather than before it, since ATen rejects an int32 index even when it is empty; and `embedding_dense_backward` with no index zeroes its output itself, because the vendor zeroes its buffer as step 1 of its own flow and then writes the rows the index names, so with no index nothing is written. Everything was added to `scripts/codegen/codegen_gcu.py` -- twelve overloads, twelve templates, five helpers in `csrc/aten/backends/gcu/topsaten_common.h` -- rather than handwritten, and the generator re-run leaves all three generated artifacts byte-identical. Route delta against the state the `out=` entry below leaves: `flaggems` **255** unchanged, `gcu` 159 -> **171**, `none` 1622 -> **1610** over the same 2036 routable ops, accelerated 414 -> **426** (20.3% -> **20.9%**); `gcu_register.inc` 166 -> **178** `m.impl` lines (`f961aff0…` -> `806d7ce7…`), `gcu_kernels.cc` `35c8a37e…` -> `7f34288a…`, `gcu_flaggems_register.inc` unchanged at 248, so the reconciliations read `255 = 248 + 7` and `171 = 178 - 7` and no route moved *from* `flaggems`. `aten::nonzero_static` is deliberately left on `none` and is now the only `cpu_fallback` operator on either path: its result is int64, which `TopsatenSupportsDtype` declines everywhere, `flag_gems` ships no `nonzero_static` module, and its schema has no `out=` overload, so there is nothing to route it to. The whole change is inside the GCU backend and the GCU generator, so Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are unaffected rather than unvalidated -- no shared code is touched and no route moves for them. | A 60-case semantic probe (`/tmp/newops_gcu.py`) over the twelve overloads, every case against the same ATen call on the CPU, reports **`ALL_OK` with 0 failures and 0 `cpu_fallback` lines**; twelve of its cases are the host path on purpose (an int64 and a float64 `self` for `index_select`, an int64 `self` for `index_fill_`, an `edb` index above `num_weights`, negative and past int32, a bool `grad`, `padding_idx=-4` through `out=`, an upsample with no scales) and each is checked for ATen's answer rather than accepted for running. The training probe (`/tmp/train_fallback_probe.py`) re-run against this change logs **exactly one `cpu_fallback` call**, `aten::nonzero_static`, against 4 `index_select -> gcu` and 4 `embedding_dense_backward -> gcu` dispatch lines and one each of `index_fill_.int_Scalar -> gcu` and `index_fill_.int_Tensor -> gcu`; both step losses are unchanged to the printed digits (`-32.7952`, `-51.3676`), the same readings the pre-change run printed. The VAE decode re-run on the current tree (`tests/manual/qwen_image_2512/run.sh infer --stage vae --device flagos`) reports **`cpu_fallback ops: 0`** where the pre-change tree reported three `_upsample_nearest_exact2d` round trips, and runs to completion (exit 0): 315 dispatch records over 19 distinct ATen operators, 246 `gcu` and 69 `flagos_python`, the 1024x1024 image saved with `mean=0.3489 std=0.2659`. What a round trip was worth was measured with the kernel's own declining branch as the in-run control -- an int64 operand takes the host path through the same entry point, at the same shapes, in the same process -- because a `FLAGOS_OP_*=none` override is **not available** for an op that is registered: it raises `routed to 'none' (no accelerated impl on this platform) but the op is registered on PrivateUse1` rather than falling back, which was measured directly by running the VAE stage against the pre-change conf. Two runs each, at shapes the ops are actually called with: a 20,000 x 512 table gathering 8,192 rows is **0.155-0.159 ms/call** on the device against **36.558-37.810 ms/call** on the host (**229.4x-244.2x**); an 8,192-row `embedding_dense_backward` against a 20,000-row table is **2.183-2.197** against **31.860-32.416 ms/call** (**14.6x-14.8x**); and `index_fill_` over 1,024 rows of an 8,192 x 512 buffer is **0.578-0.584** against **17.349-17.785 ms/call** (**29.7x-30.8x**). Generator idempotency: two runs leave `gcu_kernels.cc` (md5 `32a05182…`), `gcu_register.inc` (`1d9b0b91…`) and `backends_gcu.conf` (`b51665a7…`) byte-identical. `ruff check .` -- "All checks passed!"; `ruff format --check .` -- 263 files already formatted; `tests/unit/` -- 426 passed, 98 skipped (under the glibc 2.39 loader `tests/manual/` documents, since the env's `libtriton.so` needs GLIBC_2.38 and the host has 2.35). **Evidence gaps:** the three ratios are one shape and one dtype pair each, so they bound the cost of the round trip rather than describe a workload; card 5 is defective and hangs any `topsaten` op on a tensor resident there, so nothing was measured on it and the session used cards 0, 3, 6 and 7; the census is one probe and one pipeline, so 11 and 3 are not per-op totals for the stack; what `aten::nonzero_static` costs is not measured, only that it is the one call left; and the probe scripts are uncommitted. |
| 2026-09-17 | Enflame GCU S60 (8 `flagos` devices) | The `out=` forms of the arithmetic operators, and the training-step fallback census | Gave the three in-place arithmetic operators a kernel on the device -- the whole of the fallback traffic a training step produces -- and fixed the two defects that path reaches on the way. A census over two Adam steps on a five-parameter model, run once with `foreach=True` and once with `foreach=False`, found **25 `aten::add` and 10 `aten::mul` `cpu_fallback` calls against 45 in total**, all of them from the optimizer's own state updates (`exp_avg.lerp_`, `exp_avg_sq.mul_(beta2)`, `denom.add_(eps)`), so every one was a device->host->device round trip per parameter per step. They were invisible as `add` and `mul` for two reasons that compound: `csrc/aten/fallback.cc:21` logs `op.schema().name()`, the schema name and not the overload, so `add_.Tensor`, `add.out` and `add.Tensor` all print as `aten::add`; and ATen gives `add_.Tensor` / `sub_.Tensor` / `mul_.Tensor` **no PrivateUse1 registration of their own** -- they are structured in-place ops whose composite redispatches onto `add.out` / `sub.out` / `mul.out` with `out=self` -- so the conf's `add_.Tensor = none`, `add_.Scalar = none` and the `mul_.*` / `sub_.*` lines are **inert** and the dispatcher never sees those spellings. `scripts/codegen/codegen_gcu.py` gains an `out=` category: `mul.out` -> `binary_out` -> `topsatenMul(out, self, other)` and `add.out` / `sub.out` -> `binary_alpha_out` -> `topsatenAdd` / `topsatenSub`, one template serving both the explicit `out=` spelling and the in-place one, since `topsaten` accepts an `out` that aliases an input -- which the `zero_`, `fill_` and `masked_fill_` kernels above already rely on. A new `_out_cast_check` emits the `c10::canCast(result_dtype, out.scalar_type())` guard the ATen `out=` contract requires before any element is touched, and a new `_OUT_DEVICE_GUARD` emits `gcu::TopsDeviceGuard out_guard(self)` because the `topsaten` output pointer is resolved under the current device. The three pre-existing `out=` kernels -- `bmm.out`, `mm.out` and `addmm.out` -- gain both, plus a `out.scalar_type() != result_dtype || !out.is_contiguous() || out.device() != self.device()` check before the `topsaten` call, which they had none of. The kernel sizes the output, which is the `out=` contract: a caller may legally pass a zero-size or differently-shaped `out`, and `torch.add(a, b, out=torch.empty(0, device="flagos:6"))` has to come back shaped like `a`. Route delta against the conf this tree carried before the change: `backends_gcu.conf` `7b87541e…` -> `74aab449…`, `flaggems` 255 unchanged, `gcu` 156 -> **159**, `none` 1625 -> **1622** over the same 2036 routable ops, accelerated 411 -> **414** (20.2% -> **20.3%**). `gcu_register.inc` 163 -> **166** `m.impl` lines (`c6526ed3…` -> `f961aff0…`), the three-name `gcu` gain being `add.out`, `sub.out` and `mul.out`; `gcu_kernels.cc` `947c1b11…` -> `35c8a37e…`; `gcu_flaggems_register.inc` unchanged at 248 (`9c9c99d0…`), so the reconciliation moves from `156 = 163 - 7` to `159 = 166 - 7` and no route moved *from* `flaggems`. Two defects are fixed here because this path is what reaches them. (1) `at::native::flagos::resize_` delegated straight to the **CPU** `at::native::resize_`, whose `maybe_resize_storage_cpu` -> `resize_bytes_cpu` allocates the larger block through the storage's allocator -- correct -- but then copies the old bytes with a plain host `memcpy`, so on a flagos storage the source is HBM and the load faults. Every `out=` call with a zero-size or too-small `out` grows a storage, so `torch.add(a, b, out=torch.empty(4, 16, device=...))` segfaulted. A new `maybe_resize_storage_flagos` in `csrc/aten/strided_ops.cc` computes the needed contiguous nbytes, bails out below the current allocation, otherwise allocates through `storage.allocator()->allocate` and copies with `Allocator::copy_data`, and only then delegates to `at::native::resize_` for the size/stride metadata -- the device-correct shape of `ATen/native/cuda/Resize.h`'s `maybe_resize_storage_cuda`, which this backend never used. (2) `at::native::flagos::_copy_from` took its byte count from the source while writing into the destination's dtype in both directions, so `torch.add(a, b, out=double)` and a `dev f32 <- cpu f64` transfer copied `nbytes(f64)` bytes into an `f32` destination. Both directions now stage through a dtype-matched buffer when the two dtypes differ. Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are **not revalidated** and no route changed for them; both fixes are in shared code, so they inherit them unmeasured, and the `out=` category is GCU-only. | The census re-run against this change, same model, optimizer and seed on `flagos:6`: `EXIT=0`, **11 `cpu_fallback` calls instead of 45** -- `index_select` 4, `embedding_dense_backward` 4, `index_fill_` 2, `nonzero_static` 1 -- with **zero** `aten::add` and **zero** `aten::mul`, against **25 `add.out -> gcu` and 10 `mul.out -> gcu`** dispatch lines. Both losses are byte-identical to the pre-fix run (`-32.7952`, `-51.3676`), so the rerouted optimizer produces the same numbers rather than merely the same shapes. `_OUT_ROUTE` probe (`/tmp/out_route_probe.py`), one process, `FLAGOS_LOG=dispatch` / `FLAGOS_LOG=fallback`, every case against a CPU reference: **`FAILURES: none`**, exit 0. The nine in-place spellings (`add_`/`sub_`/`mul_` with a tensor, an alpha and a scalar, plus `add_(self)` with `out` aliasing both operands) are all `max|diff|` 0.000e+00, as are both broadcasts, float16, bfloat16, int32, int64 and a strided-view write; the explicit `out=` forms match for a zero-size `out`, a wrong-shaped `out`, a widening `out=double` and a broadcasting call; the narrowing rejections (`torch.mul(f32, f32, out=int32)`, `torch.mm(..., out=int32)`, `torch.addmm(..., out=int32)`) raise, and raise for the CPU's reason -- `result type Float can't be cast to the desired output type Int` -- which is what the cast guard exists for, since the host path ends in a casting `copy_` and would otherwise narrow silently. `mm.out` (9.537e-06) and `addmm.out` (7.629e-06) on f32 are compared with a tolerance for reduction order while their `out=double` siblings are **exact** (`0.000e+00`): an f64 out takes the host path by construction and is computed the way the reference is, so the pair of readings is itself evidence of which branch each case took. Branch evidence by timing at 4096x4096 in-place `add_`: **0.507 ms/call** on the device path against **114.185 ms/call** for an int64 operand, which takes the host path by construction and is the in-run control -- **225.3x**. The `resize_` fix is measured by `RESIZE_GROW_FAILURES=0` on `flagos:0` and `flagos:6` across grow-into-live-storage, grow-from-empty, shrink, same-numel reshape, four dtypes, a nonzero-`storage_offset` slice, neighbouring allocations and the `out=` growth cases; the sentinel-filled ones check the old bytes actually moved rather than garbage being left behind, and the standalone diagnostic that crashed with `rc=-11` on `grow 4x16 -> 32x64` now reports `rc=0` on every case. The `copy_` fix is measured by eleven cases in both directions -- host->device widening and narrowing, device->host in both directions, a strided host destination, and same-device `dev f16 <- dev f32` -- all `max|diff|` 0.000e+00 against the pre-fix readings of 4.094e+03, 4.500e+00 and 1.085e+09, with the `.to(device, dtype)` spelling as a control. Both generators are idempotent: `--check` exits 0 for each, and `codegen_gcu.py` re-run three times is byte-identical. `ruff check .` -- "All checks passed!"; `ruff format --check .` -- 263 files already formatted. `tests/unit/test_gen_vendor_confs.py`: 35 passed; the rest of `tests/unit/`: 418 passed, 100 skipped. **Evidence gaps:** the probes are uncommitted; card 5 is defective and hangs any `topsaten` op on a tensor resident there, so the probes used cards 0 and 6 and the session as a whole used 0, 3, 6 and 7; the census is one model, one optimizer and one seed, so 45 is not a per-op total for the stack; the 225.3x ratio is one shape, one dtype pair and one run; no probe copies between two devices, so the D2D arm of the device guard -- which derives its device from `src` in both `csrc/runtime/accelerator/gcu/memory.cc:71` and the `memcpy` override in `csrc/runtime/allocator/backends/gcu_memory.h` -- is reasoned from the source rather than measured, and a cross-device `copy_` is the case that would exercise it; and the `resize_` and `copy_` fixes sit in shared code, so the other six platforms inherit them without revalidation. |
| 2026-09-17 | Enflame GCU S60 (8 `flagos` devices) | The six measured CPU fallbacks in the Qwen-Image-2512 denoise loop | Gave every operator in the measured fallback census a route to a kernel on the device, so nothing in the pipeline reaches ATen's `cpu_fallback` any more. The census that selected the set, over the whole 50-step 1024x1024 pipeline, found **80,536 `cpu_fallback` calls in exactly six operators and no seventh anywhere in the log** -- `fill_` 23,570, `zero_` 19,221, `view_as_complex` 18,794, `view_as_real` 18,794, `arange` 79, `linalg_vector_norm` 78 -- against 468,227 device dispatches (331,263 `gcu`, 136,964 `flaggems`), 14.7% of all dispatches. `fill_` and `zero_` are worth far more than their own call count because a device-side `torch.zeros` / `ones` / `full` decomposes into them. `scripts/codegen/codegen_gcu.py` gains an `arange` category and a fill/in-place category: `arange` (all three overloads) computes its length on the host through `at::native::compute_arange_size` -- `topsatenArange` takes no size and the caller pre-sizes `output` -- and calls `topsatenArange`; `zero_` -> `topsatenZero`; `fill_.Scalar` / `fill_.Tensor` -> `topsatenFill_` (the scalar through `ToTopsatenScalar`); `masked_fill_.Scalar` / `masked_fill_.Tensor` -> `topsatenMasked_fill` with `out` aliasing `self`; `linalg_vector_norm` -> `topsatenLinalgVectorNorm`, which takes the dim list and the order and whose output is pre-sized to the keepdim shape and then metadata-`reshape`d, superseding the older "no `topsaten` kernel, so `none`" claim for that op. The factory family resolves an absent or index-less `device` to the **current** device (`c10::flagos::CurrentDevice()`), never to index 0, because resolving to 0 would be a silent cross-device write on GCU. `view_as_real` and `view_as_complex` are metadata-only aliases and are the two handwritten kernels this change adds, in `csrc/aten/strided_ops.cc` under `Backend::kGcu`: every category template in the generator ends in a `topsaten::` call, so `codegen_gcu.py` cannot express an alias and lists both in `METADATA_OPS`, and that concrete limitation plus explicit human approval is what the project rule requires for a handwritten kernel. It matters because a *view* op on the `cpu_fallback` path is **copied**, so the result silently stops aliasing its input, and the Qwen-Image RoPE path calls each of them 488 times per denoise step (19.3 ms and 18.5 ms per call at `(1, 4096, 24, 128)`). A new `TopsatenArangeDtype` accepts Float / Int / Short / Char / Byte / Half / BFloat16 only: int64 and float64 stay on the host path because `topsaten` has no kernel for them, and Bool is declined deliberately because ATen has no Bool arange kernel either. Measured against the committed conf at `082afa8`: `backends_gcu.conf` `4c5082d6…` -> `7b87541e…`, `flaggems` 257 -> 255, `gcu` 144 -> 156, `none` 1635 -> 1625 over the same 2036 routable ops, accelerated 401 -> 411 (19.7% -> 20.2%). The launch-grid entry below is uncommitted in this tree and its two route moves are in the same diff, so the split is: that entry takes `_softmax` from `flaggems` to `gcu` and `linalg_vector_norm` from `flaggems` to `none` (257 / 144 / 1635 -> 255 / 145 / 1636), and this entry then takes `linalg_vector_norm` and the nine other `none` overloads to `gcu` (255 / 145 / 1636 -> 255 / 156 / 1625), leaving the `flaggems` count untouched here. `gcu_register.inc` 152 -> 163 `m.impl` lines (`756f6134…` -> `c6526ed3…`), the twelve-name `gcu` gain being `_softmax` plus `arange`, `arange.start`, `arange.start_step`, `fill_.Scalar`, `fill_.Tensor`, `masked_fill_.Scalar`, `masked_fill_.Tensor`, `zero_`, `view_as_real`, `view_as_complex` and `linalg_vector_norm`. `gcu_flaggems_register.inc` 249 -> 248 (`a02b46d9…` -> `9c9c99d0…`) and `gcu_kernels.cc` `01437b68…` -> `947c1b11…`: the single `m.impl` removal there is `linalg_vector_norm` and belongs to the launch-grid entry, since it is that entry's gap addition which takes the op off FlagGems, while this change adds nothing to that file's registrations and moves its "further FlagGems ops already claimed by `gcu_register.inc`" banner 88 -> 97, which is exactly these nine (`arange` x3, `fill_` x2, `masked_fill_` x2, `zero_`, `linalg_vector_norm`) -- `view_as_real` is not in FlagGems' coverage and `view_as_complex` is held in `FLAGGEMS_PENDING_NATIVE_OPS`. The `NATIVE_TRITON_GAPS["gcu"]` entries for `arange`, `zero_`, `fill_.Scalar` and `fill_.Tensor` **stay**: a gap entry only takes the op off FlagGems, and for the factory family it is what *selects* the vendor, so deleting one would put the op back on the FlagGems path that cannot compile it. Out of scope and still `none`: the `out=` forms (`arange.out`, `arange.start_out`, `linalg_vector_norm.out`, `view_as_real_copy.out`, `view_as_complex_copy.out`). Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are **not revalidated** and no route moved for them; Ascend carries the identical `view_as_real` / `view_as_complex` gap and it is left as it stands. | Per-op probes on the S60 with `torch_fl` + diffusers, flagtree `0.6.1+enflame3.6`, `FLAGOS_LOG=dispatch` / `FLAGOS_LOG=fallback`, one fresh process per probe, every case against a CPU reference. `arange`: **48 PASS / 0 FAIL** over the three overloads, three dtypes, integer-valued and fractional steps, the empty length and the single-element length. The vendor formula was identified exactly -- `out[i] = (T)(double((T)start) + i * double((T)step))`, operands cast to the output dtype first and rounded once -- and it reproduces all five measured GCU points bit for bit. Integer-valued steps are bit-exact against the CPU kernel on both routes (`1.5, 4.0, 0.5`, `10., 0., -2.`, `0., 100., 7.`, `arange(0, 128, 2)`, `arange(4096)`). A fractional step agrees to **1 ULP and no better**, and bit-exactness with the host path is unreachable for it by construction: the ATen CPU kernel is not length-invariant -- `torch.arange(0., 1., 0.1)[9] = 0.8999999761581421` while `torch.arange(0., 1000., 0.1)[9] = 0.9000000357627869` -- and the measured bound is 1.000 ULP at 10,000 elements (`max|diff|` 6.104e-05) and at 1,000,000 elements (5.960e-08). `zero_` / `fill_` / `masked_fill_` (`INPLACE_ROUTE_PASS`): all 20 case comparisons match CPU, re-measured at `(1, 4096, 24, 128)` and `(16, 128, 128)`, on the empty `(0,)`, on three float dtypes, on the int64 host path and on a transposed input -- and the five `cpu_fallback` lines that probe logs are its own `_unique2` verification step, not an op under test. Device-side `torch.zeros` / `ones` / `full` / `zeros_like` / `ones_like` re-run and reach the device with the right values. `view_as_real` / `view_as_complex` (`COMPLEX_VIEW_PASS`, 23 PASS / 0 FAIL): the metadata-alias claim is checked by `data_ptr` rather than inferred from the dispatch table -- the complex tensor shares storage with its real source, shape and stride and 3-D strides all agree, a write through the real view is visible through the complex one, and the full RoPE sequence runs `-> gcu` at every step on `flagos:7` with `max|diff|` 0.000e+00; both error contracts raise the same message as the host path. `linalg_vector_norm` (`LINALG_NORM_PASS`, 42 PASS / 0 FAIL, 0 `cpu_fallback`): the VAE call site `(1, 128, 1, 1024, 1024)` ord=2 dim=[1] gives `max|rel diff|` 5.048e-07 and `F.normalize(dim=1)` / `(dim=-1)` 1.192e-07, the ord / dim / keepdim / non-contiguous / empty sweep stays on `flagos` throughout, and fp16, bf16 and fp64 are bit-exact on the host path. The view and decomposition probes both exit 0 with 0 `cpu_fallback` lines. Both generators are idempotent (second run byte-identical, `--check` exit 0). **Evidence gaps:** the probes use their own shapes except where a model call site is named above; the 1-ULP `arange` bound is measured at three lengths rather than proved; cards 0, 3, 6 and 7 were probed because card 5 is defective and hangs any `topsaten` op on a tensor resident there, so nothing was measured on card 5; the census is one pipeline, one seed and one prompt; and no S60-wide FlagGems survey was re-run, which is the right call rather than a gap, because no route moved *from* `flaggems` in this change. |
| 2026-09-17 | Enflame GCU S60 (8 `flagos` devices) | GCU launch-grid routing (`_softmax`, `linalg_vector_norm`) | Added two measured exceptions to `NATIVE_TRITON_GAPS["gcu"]` (225 -> 227) and regenerated the GCU configuration. Both are launch-grid axis limits rather than dtype or numerics faults: they are correct at every dtype and on every profile `flaggems_overload_survey.py` can construct, and fail only at the scale Qwen-Image-2512 drives them to. FlagGems' `_softmax` launches one program per leading-dimension product and takes that product in `grid.x`, which the GCU caps at 65535 against CUDA's 2\*\*31-1; `linalg_vector_norm` meets the same cap, and its non-inner reduction puts 2067 on `grid.y`, capped at 255 against CUDA's 65535. GCU `flaggems` 257 -> 255, `gcu` 144 -> 145, `none` 1635 -> 1636; accelerated routes 401 -> 400 (19.7% -> 19.6%). `_softmax` moves to native `gcu` because `topsaten` serves the full shape; `linalg_vector_norm` has no `topsaten` kernel, so it moves to `none` and the boxed fallback serves it on the host. `gcu_flaggems_register.inc` 249 -> 248 `m.impl` lines, and the reconciliation moves from 257 = 249 + 8 to 255 = 248 + 7, because one of the two ops was registered in the generated file and the other was already among the native-claimed ops `codegen_gcu_flaggems.py` omits on purpose. `backends_gcu.conf` SHA-256 `4c5082d6…` -> `10a85c9e48a0a34a1b8c3796db91f5ae3660b0dfbda54d2f94611d01b4e39e8f`; `gcu_flaggems_register.inc` `a02b46d9…` -> `f8b73835a99cc97856a622a0ebb5037eae07b1dde81f80fedf5a32847e3846dd`. FlagGems is not patched, and `topste` is not wired into the GCU generator, so the L2-norm route stays on the host rather than being retargeted by hand. Ascend, MUSA, DCU, MetaX, PPU and Tsingmicro are **not revalidated** and no route changed for them. | `tests/manual/qwen_image_2512/` on the S60 with `torch_fl` + diffusers, flagtree `0.6.1+enflame3.6` / Triton 3.6 backend `enflame`. Both reroutes re-measured at the shapes that broke FlagGems, one process, `FLAGOS_LOG=dispatch` / `FLAGOS_LOG=fallback`, each against a CPU reference: `torch._softmax(x, -1, False)` bf16 at M = 65496 (`max|diff|` 6.10e-05), 65544 (6.10e-05) and 98736 (3.05e-05) each report `_softmax -> gcu`; `F.normalize(x, dim=1)` bf16 at `(1, 128, 1, 1024, 1024)` (M = 1048576) reports `linalg_vector_norm -> cpu_fallback`, `clamp_min -> flagos_python`, `div.Tensor -> gcu`, and is bit-exact (`max|diff|` 0.00e+00). The boundary is bracketed rather than assumed: M = 65496 passes on FlagGems and M = 65544 raises `grid.x Required 65544`, so the 65535 cap is the failure and not a shape-specific one. Qwen-Image-2512 reaches both, at two call sites -- `_softmax` on the 60-block MMDiT joint attention and `linalg_vector_norm` on the VAE's `F.normalize`. Paired VAE decode, seed 42, shared latents: vendor `mean=0.3505 std=0.2635` against flagos `mean=0.3475 std=0.2618`, **MAE 0.1646, PSNR 55.90 dB, max 6.0**. Both generators idempotent (second run byte-identical; `--check` exit 0). Evidence gaps: `_softmax` is bracketed only at its two ends, so nothing is measured between 65496 and 98736; `linalg_vector_norm`'s 524280 ceiling is derived from the `BLOCK_M <= 8` schedule rather than measured at that size; the remaining 255 `flaggems` routes were exercised only at the survey's profiles, whose leading-dimension products stay far below the caps, so none of them is ruled out for the same defect and this family is a lower bound; and one probe run reported `normalize(1,128,1,1024,1024) max|diff|=nan`, which did not reproduce in eight further attempts over the same sequence with NaN attribution on both operands (`nan(out)=0 nan(ref)=0`), so it is recorded as unattributed rather than explained. |
| 2026-09-17 | Hygon DCU bw1000 (8 devices) | Qwen-Image-2512 SDPA backend selection | `aten::scaled_dot_product_attention` on the DCU boxing route now executes DTK's CUTLASS flash adapter instead of the math decomposition. **No conf entry changed and no FlagGems route moved**: the DCU boxing path already intercepts the composite in `csrc/aten/sdp_choice_stub.cc` and boxes q/k/v to CUDA, so what changed is which backend *DTK's own* selector answers inside it. DTK's hipified `flash_api.h` reaches the adapter only when `at::globalContext().getROCmFAPreferredBackend() == at::ROCmFABackend::Cutlass`, a member of the `Context` singleton that the official CPU `libtorch_cpu.so` constructs without DTK's enum, so it initialised to `Default` and every fused call landed in the aotriton branch DTK did not compile (`RuntimeError: Non't compile aotrition fa, please compile aotriton fa before use it`, reached from `transformers`' `sdpa_attention_forward`); memory-efficient attention is not compiled either ("USE_MEM_EFF_ATTENTION was not enabled for build"). `_prefer_dtk_cutlass_flash()` in `torch_fl/accelerator/dcu/_dcu_compat.py` writes DTK's ordinal for `Cutlass` (the literal `1`; upstream's narrower `{Default, AOTriton, Ck}` spells that literal `AOTriton`) through `torch._C._set_rocm_fa_preferred_backend`, resolving by name when the binding is DTK's own enum and by ordinal when it is upstream's. It is gated on the adapter's `dlopen` target actually existing (`_dtk_flash_attn_lib()`, mirroring `get_so_path()` in DTK's `cutlassfa_adapter.h`, which `TORCH_CHECK`s rather than declining), and `FLAGOS_DCU_SDPA_FLASH=0` restores the previous math-only behaviour. Ascend, MUSA, GCU and BPU keep their own PrivateUse1 `_fused_sdp_choice` registrations; MetaX, PPU, Tsingmicro and the other platforms are **not revalidated**. | Measured on DTK 6.3.26113 / torch 2.10.0+cpu, `flagos:7`, bf16 `(1, 24, 12576, 128)` — the per-head shape Qwen-Image-2512's 1664×928 pass runs: math decomposition **167.5 ms/iter, 33,865 MiB** peak against DTK CUTLASS flash adapter **10.2 ms/iter, 463 MiB** peak, peak VRAM sampled off-device with `rocm-smi` as the card's own figure rather than the allocator's reserved/peak counters; both per-iteration figures were re-measured on 2026-09-18 and the pair first published here (102.1 against 6.1) came from a loop whose `torch.cuda.synchronize()` had no device argument and therefore synchronized `current_device()` — `0`, while the tensors lived on `flagos:7` — so both absolutes were submit times; the ~16x ratio between them was unaffected and the route conclusions do not move. The numerics are the adapter's own, since the composite is intercepted rather than reimplemented: `max(diff)` 4.9e-4 against `max(ref)` 0.109, i.e. bf16 rounding, and shapes DTK's selector refuses still fall back to its own math path. `torch.backends.cuda.is_flash_attention_available()` and `torch._C._can_use_flash_attention()` both answer `False` in the same process that ran the adapter and printed its warning, so they are the official wheel's CPU-only bindings and are deliberately left unpatched. Full detail: "`scaled_dot_product_attention` moved onto DTK's CUTLASS flash adapter" above. |
| 2026-09-17 | Hygon DCU bw1000 (8 devices) | Pointwise `add`/`sub`/`div` overloads reached with a Python float, found by Qwen-Image-2512 | Eleven overloads moved from `flaggems` to `cuda` in `backends_dcu.conf` only: 469 `flaggems` / 1567 `cuda` -> **458 / 1578**, `none` 0, over the same 2036-entry op list. `backends_dcu.conf` is the only file that moves, so the digests are its own: SHA-256 `103529363fce2795a769fc7a4003cc9d481f2fcd50adc48f968e3bee569a77ec` on `main` -> `e6ac1851c558495d898514abfe4f6d304c10e99440fe1ccc5f2159bc268ff19d` here. Pinned: `add.Tensor`, `add_.Tensor`, `sub.Tensor`, `sub_.Tensor`, `div.Tensor`, `div_.Tensor`, `div.out`, `div.Tensor_mode`, `div_.Tensor_mode`, `div.Scalar_mode`, `div_.Scalar_mode`. **They fail for two unrelated reasons, and the split is the point of the entry** — a single diagnosis would have been wrong for four of them. Seven abort in the HCU backend because ATen boxes a Python float as an f64 wrapped number, so a bf16 tensor beside a Python float puts an `f64` in the FlagGems pointwise kernel and `TruncFOpConversion::createDestOps` asserts `inElemTy.isF32() && "unsupported conversion"` (`third_party/hcu/lib/TritonHCUGPUToLLVM/ElementwiseOpHCUToLLVM.cpp:2358`); the IR dump shows `arith.extf … to tensor<…xf64>` / `arith.addf : tensor<…xf64>` / `arith.truncf`, and it surfaces as `RuntimeError` at `triton/backends/hcu/compiler_hcu.py` `make_llir`. That is the defect GCU already carries in `NATIVE_TRITON_GAPS["gcu"]` and the one FlagGems tracks as issue **#6212** (reported for MetaX). The other four fail under a *named* rounding mode in FlagGems' own `div_rn` shim: `'trunc'` routes `flag_gems/ops/div.py` to `trunc_div_func`/`trunc_div_func_tensor_scalar`, whose whole body is `trunc(div_rn(x, y))`, and `use_tl_extra` resolves that `div_rn` to `triton.language.extra.hip.libdevice.div_rn` because the shim carries the symbol — so FlagGems' own `x / y` + `tl.floor` fallback never runs — while the symbol lowers to `None` on the HCU backend (`TypeError: cannot convert None of type <class 'NoneType'> to tensor`; the floor path dies one statement later in `_float_floordiv` on `q - 1` with `unsupported operand type(s) for -: 'NoneType' and 'int'`). Exactly the mode FlagGems documents for `asin` on another fork in its own `triton_lang_helper.py`. Python `int` operands, f16/f32 tensors and tensor operands are unaffected: `add.Scalar`, `div.Scalar`, `div_.Scalar` and the six `clamp` spellings stay on `flaggems`, all measured passing, and `mul.Tensor`/`add.out`/`sub.out`/`div.out_mode` were already on `cuda`. Reachable in Qwen-Image-2512 because `QwenImageRMS_norm` in diffusers' `autoencoder_kl_qwenimage.py` returns `normalized * self.scale * self.gamma + self.bias` with `self.bias` the Python float `0.0` whenever the layer is built without a bias, which is how the VAE decoder's `norm_out` is built. Regeneration preserves all eleven: `boxing_triton_gaps()` recovers the gap set by diffing an existing conf against FlagGems coverage, and the diagnosis is carried in `BOXING_GAP_NOTES["dcu"]` in `scripts/codegen/gen_vendor_confs.py`. The generic `backends_flaggems.conf` cohort, the historical 546-overload bw1000 row, and every non-DCU platform are **not revalidated** by this change. | Both directions of the A/B are re-runnable in-tree: `tests/manual/dcu_pointwise_replay.py` replays one entry at a time onto the FlagGems route through `FLAGOS_OP_<op>=flaggems` (dots doubled; `LoadBackendConfig` in `csrc/aten/common.cc`) and leaves the other ten shipped, one subprocess per case because the f64 path aborts the interpreter rather than raising. It reports `pinned to cuda in backends_dcu.conf: 11/11`, then seven `RuntimeError … ElementwiseOpHCUToLLVM TruncFOpConversion` and four `CompilationError … cannot convert None of type <class 'NoneType'> to tensor`, each `[shipped: cuda]`, at exit 0. The 23-entry family probe (`/tmp/probe_aten_overloads.py`, session evidence) reports `ran` for every one of the eleven pins' siblings on the shipped conf. The named-mode split was isolated with `--rounding-mode {none,trunc,floor}` on `/tmp/probe_one_overload.py`: `none` routes to `true_divide` (plain `x / y`) and runs for all four `*_mode` entries, `trunc` fails as above, `floor` fails in `_float_floordiv`. The shim resolution was read off the host, not inferred: `flag_gems.utils.triton_lang_extension.div_rn is triton.language.extra.hip.libdevice.div_rn` -> `True`. **Evidence gap, recorded rather than glossed:** `flaggems_overload_survey.py` cannot measure any of this — `active_routes()` enumerates the ops a conf *file* spells `flaggems`, so pinning an entry removes it from the survey's denominator by construction (a run against the shipped conf refuses the eleven names outright), and against a copy of the conf with the pins reverted all eleven report `PASS`, because none of the seven profiles is bf16, every `.Tensor` operand the harness synthesizes is a tensor rather than a Python float, and `default_for` yields `rounding_mode=None`. The four `clamp` spellings, left on FlagGems, are `INVALID_CASE` 7/7 in the same harness because it synthesizes `min=None, max=None` and ATen rejects the call before any device kernel is reached — a synthesis gap in the harness, not evidence about those routes. The control arm on the shipped conf measured `div.Scalar` 7/7, `clamp_max` 7/7, `clamp_min` 7/7, `div_.Scalar` 5/7 with 2 `INVALID_CASE`. `gen_vendor_confs.py --check` reports `all vendor confs up to date` before and after a regeneration on this branch. Full detail: "Pointwise `add`/`sub`/`div` overloads rerouted to CUDA boxing" above. |
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
