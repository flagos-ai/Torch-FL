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

**The shipped tree has moved on from this cohort.** The harness in
`tests/manual/flaggems_overload_survey.py` is now version 5 (SHA-256
`cfd09e50b915700cc39d2a78e7076e8008e3f24d1c16187529d4dc40e2b201bd`), and the
shared coverage set has been widened to 639 overloads on the FlagGems master
cohort (`5a58df410`), which is what the MetaX configuration below is measured
against. The four hardware rows in this section were **not** re-measured against
that cohort and remain the `fe2272b5` / `7fb49bad` baseline, as the table says.

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
[`scripts/codegen_gcu.py`](../../scripts/codegen_gcu.py) with the registration and
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

### MUSA FlagGems routing restored, in-place arithmetic routed back to mudnn (2026-09-14)

The MUSA FlagGems registration generator was restored
(`scripts/codegen_musa_flaggems.py` -> `csrc/aten/backends/musa/generated/musa_flaggems_register.inc`,
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
deleted `torch_fl/configs/backends_flaggems.conf` to `scripts/backend_coverage.py`,
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
routed back to mudnn (2026-09-14)" below. The Ascend and GCU numbers are the ones
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

- `scripts/gen_vendor_confs.py` twice in a row produces an empty diff and
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
version 5, SHA-256
`cfd09e50b915700cc39d2a78e7076e8008e3f24d1c16187529d4dc40e2b201bd` as shipped.
(The revision that produced the rows below was identical apart from the
illustrative route count its module docstring quotes, which is not read by any
code path; the hash above is the one an auditor can reproduce from this tree.)

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
| 2026-09-15 | MetaX C550 (8 devices) | FlagGems entry-point resolution (`5a58df410`) | `_normalize_flaggems_qualname` in `scripts/codegen_ops.py` now emits `flag_gems.<fn>` instead of `flag_gems.ops.<module>.<fn>`, so a generated kernel reaches the entry point the active backend has rebound rather than the generic module the alias rewrite pinned. 72 of the 666 qualnames in the checked-in kernels resolve to a `_metax.ops.*` override and were running the generic kernel before this. `codegen_ops.py` also becomes the writer of the `FLAGGEMS_PYTHON_OPS` ceiling in `scripts/backend_coverage.py` (`render_flaggems_coverage`, minus the override-only ops), which was previously a hand-carried literal that capped every conf built from it. Both apply to every FlagGems platform; no route changed on Ascend, GCU, MUSA, DCU or PPU. | Counted over `csrc/aten/generated/flaggems_python_kernels.cc` with `flag_gems 5.4.0rc2.post1+g5a58df410` on the C550 host: 688 call sites, 666 distinct qualnames, 0 that are not two-component `flag_gems.<op>`, 0 unresolvable on the package, 594 resolving inside `flag_gems` and 72 to a `_metax.ops.*` module. `tests/integration/ops/test_flaggems_conf_consistency.py` requires the two-component form and now compares the conf, the override-only routes and the generated kernels as sets (7 passed); `tests/integration/ops/test_metax_flaggems.py` on C550 reports **90 passed in 756.07s**, 0 failed. Full detail: "MetaX: generated FlagGems calls name the package-level entry point" above. |
| 2026-09-15 | MetaX C550 (8 devices) | FlagGems master coverage cohort (`5a58df410`) | Rebuilt `FLAGGEMS_PYTHON_OPS` on the FlagGems master cohort pinned at `5a58df410c551c4f4eb41d31887cd75fd596804a`: 482 -> 639 overloads, 158 added and `mul_.Tensor` removed because that cohort does not cover it. The newly covered overloads are withheld from the Ascend, GCU and MUSA configurations by `FLAGGEMS_PENDING_NATIVE_VENDORS` / `FLAGGEMS_PENDING_NATIVE_OPS` so their shipped counts do not move without hardware; DCU loses `mul_.Tensor` to `cuda` (three lines) for the same reason as MetaX. MetaX was re-measured against the raised ceiling and **sixteen overloads were withdrawn back to the CUDA boxing kernel** after a differential A/B probe showed each one passing on `cuda` and failing on `flaggems`: `special_bessel_j0`, `special_i1e`, `special_i1e.out`, `special_chebyshev_polynomial_w.out` (kernel asserts its input is a real CUDA tensor), `nansum.out`, `lu_unpack.out`, `linalg_matrix_exp.out`, `sum.out`, `_cdist_forward` (the gems wrapper cannot serve the caller's call form), and `_compute_linear_combination`, `_compute_linear_combination.out`, `_fused_rms_norm`, `igamma`, `igamma_`, `logit_backward`, `special_shifted_chebyshev_polynomial_t` (wrong result). `backends_metax.conf`: 443 `flaggems` / 11 `flaggems_cpp` / 1582 `cuda` (committed) -> 592 / 12 / 1432, via the widened intermediate 608 / 12 / 1416. Ascend, GCU, MUSA, DCU and PPU are **not revalidated** against the raised ceiling; only DCU's `mul_.Tensor` line moves and no MetaX measurement is transferred to them. | Screening survey over the 166 overloads whose route changed in `backends_metax.conf`, `2d-f32` profile, harness v5: `{"registered": 166, "tested": 97, "STRICT": 76, "FAILED": 21, "UNTESTED": 69}`, `basic_executable` 76. The 21 `FAILED` overloads re-run with `FLAGOS_OP_<op>=cuda` (one host-built input pair moved with `.to("flagos")`, both arms identical values): 16 `cuda` PASS with the `flaggems` verdicts in the table above, 5 fail on both routes so they keep their route. Replaying the 16 through the shipped configuration with no override reproduces 16 PASS. `gen_vendor_confs.py` idempotent (two runs, empty diff; `--check` exits 0 for the MetaX file), and running the two generators over this tree leaves `backends_metax.conf`, every generated artifact and `backend_coverage.py` byte-identical — the out-of-scope configurations do move on that first pass, which the ordering note above records. Full detail: "MetaX: FlagGems cohort widened to FlagGems master, sixteen ops withdrawn" above. |
| 2026-09-15 | MetaX C550 (8 devices) | MetaX FlagGems hybrid path | Promoted 8 overloads to the Python FlagGems path on MetaX only, through `METAX_FLAGGEMS_MEASURED` in `scripts/gen_vendor_confs.py`, because their `flag_gems.<name>` entry points exist in the pinned cohort while the shared hold was written against an older one: `_embedding_bag_per_sample_weights_backward`, `_native_batch_norm_legit_functional`, `binary_cross_entropy_with_logits`, `linalg_ldl_solve`, `special_bessel_j1`, `unsqueeze`, `unsqueeze_`. Two of them were failing outright on the cuda boxing route before this, so the promotion is a fix and not a preference: `special_bessel_j1` raises `cudaErrorMemoryValueTooLarge` through maca, and `linalg_ldl_solve` needs a `cusolverDnXsytrs_bufferSize` symbol maca does not provide. The eighth, `igammac_`, was promoted and then withdrawn the same day (see "`igammac_` rerouted to CUDA boxing" above). No other platform's routes changed. | `tests/integration/ops/test_metax_flaggems.py` on C550 with `flagtree 0.6.1+metax3.6` / `flag_gems 5.4.0rc2.post1+g5a58df410`: **90 passed in 756.07s**, 0 failed — the routing cases, the execution cases, and the exclusion cases including the three representative withdrawals added by the cohort widening recorded above. `gen_vendor_confs.py` idempotent for the MetaX configuration. Ascend, GCU, MUSA, DCU and PPU are **not revalidated** by this change -- `METAX_FLAGGEMS_MEASURED` is consulted only for `backends_metax.conf`. |
| 2026-09-14 | MTT S5000 (8 devices) | MUSA FlagGems routing and in-place arithmetic fallback | Restored the MUSA FlagGems registration generator, taking MUSA from 158 to 515 registered ops and from 122 to 468 `flaggems` routes (`musa` 36 -> 47, `none` 1878 -> 1521). Moved 14 ops into `NATIVE_TRITON_GAPS["musa"]` so they fall back to mudnn instead: `add/sub/div.Tensor` and their in-place forms plus `mul_.Tensor` (bf16 wrapped-number promotion reaches `llvm.musa.float2bfloat16` with a double operand), `randn`/`randn_like`, `sort`/`sort.stable`, and `_conj`/`index_add`/`index_add_`, which route to `none` because mudnn has no kernel for them. FlagGems is not patched. Ascend, GCU, DCU, MetaX and PPU rows are **not revalidated** by this change and no FlagGems route was altered for them. | Every group of `.github/configs/musa.yml` run locally on hardware: dispatch 104 passed/1 skipped, factory 46 passed, AMP 27 passed, math-bits 12 passed, profiler 10 passed/1 skipped/1 xpassed, operator cohort 490 passed/2 skipped/512 deselected/2 xfailed/1 xpassed, RNG 80 passed/37 deselected. The bf16 gap was reproduced causally with `FLAGOS_OP_add__Tensor=flaggems`, which reproduces the remote CI's `failed to translate module to LLVM IR` on `test_autocast_fp32_policy[dtype1]` and passes on the shipped route. Three `flaggems`-marked dispatch-log tests that hard-coded `flagos_python`/`cuda` were rewritten to read the route from the platform conf (`tests/integration/ops/backend_conf.py`); they were the only failures in CI group 7 on `6f8128e` and pass on every platform's conf afterwards. Generator idempotent (`codegen_mudnn.py` twice, byte-identical; `codegen_musa_flaggems.py --check` and `gen_vendor_confs.py --check` clean for MUSA). `tests/unit/test_gen_vendor_confs.py`: 34 passed, 1 pre-existing failure (ascend/gcu conf staleness, unrelated). Three pre-existing `test_flaggems_conf_consistency.py` failures reproduce byte-identically against `d0e2d1a`'s data files, so they are not introduced by this change. |
| 2026-09-11 | None (CPU-only host) | Unified MetaX confs (refactor/unified-vendor-confs) | Collapsed `backends_metax_flaggems.conf` and `backends_metax_flaggems_cpp.conf` into a single `backends_metax.conf`. The 17 on-device-verified C++ routes are now in the file unconditionally; a build without `FLAGGEMS_KERNEL=ON` degrades them to the boxing kernel via `Dispatcher::GetFn` instead of raising. `METAX_CPP_MEASURED` in `gen_vendor_confs.py` records the measured set explicitly since the file it was formerly recovered from no longer exists. `mm` remains on the boxing kernel (MetaX C550 shared-memory limit). `_select_backend_config()` now routes both `FLAGOS_USE_FLAGGEMS` and `FLAGOS_USE_FLAGGEMS_CPP` to the same `backends_metax.conf` under `FLAGOS_METAX_BOXING=1`. **All hardware rows not revalidated.** | Mechanical evidence only — generator idempotent (two runs, empty diff; `--check` exits 0), `tests/unit/test_gen_vendor_confs.py` passes with updated test names. |
| 2026-09-10 | None (CPU-only host) | Full-coverage MUSA/GCU/Ascend and boxing configurations | Converted the MUSA, GCU, Ascend and boxing configurations to full coverage: all 2036 routable ops listed exactly once under `flaggems_cpp` / `flaggems` / `<vendor>` / `none`, priority in that order, generated by `scripts/gen_vendor_confs.py`. Every accelerated route is now gated on the platform's real PrivateUse1 registration set, read from the generated `*_register.inc` files, because CUDA-measured FlagGems coverage is a ceiling and not a per-platform routing set (Ascend 374, GCU 152, MUSA 158 registered of 2036). MetaX and Tsingmicro register the full generated list, so `none` would raise there instead of boxing to `cpu_fallback`; Tsingmicro's configuration stays hand-written. **All hardware rows not revalidated.** | No route measured. `flaggems_overload_survey.py` cannot run on this host: Triton 3.7.1 exposes only `amd`/`nvidia` backends and `import flag_gems` fails. Mechanical evidence only — generator idempotent (two runs, empty diff; `--check` exits 0), routing equals registration exactly on all three vendors, `tests/unit/test_gen_vendor_confs.py`: 27 passed, `tests/unit/`: 303 passed, 96 skipped, 2 pre-existing profiler failures (`CXXABI_1.3.15` libstdc++ skew) unrelated to routing. |
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
