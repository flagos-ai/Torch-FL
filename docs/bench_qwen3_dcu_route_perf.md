# Qwen3-0.6B three-route decode performance analysis on DCU

**Status: P0 lever implemented and validated.** The fused SDPA path is restored
on the boxing route through a composite-level override
(`csrc/aten/sdp_choice_stub.cc`). Boxing e2e now measures **0.952x vendor**
(interleaved, drift-cancelled) on the acceptance workload, meeting the
`e2e_tps >= 0.95 x vendor` line. This document records the Step 0 baseline,
the Step 1 attribution, the delivered P0 change, and where the residual stands.

Workload and acceptance line (fixed by the perf-tune-dcu-routes skill): Qwen3-0.6B
bf16, batch=1, greedy decode, sdpa attention, single Hygon DCU card,
target e2e_tps >= 0.95 x vendor.

## Environment

| Item | Value |
|---|---|
| torch_fl commit | `4a95268` (branch `feat/transformers-test-command`) + uncommitted working tree (the P0 override is in the measured state, not yet on HEAD) |
| torch_fl package | 0.1.0+dtk (installed in `/opt/fl-envs/boxing`) |
| vendor torch | 2.10.0 (DTK build, system python3.10, `cuda` device) |
| boxing venv torch | 2.10.0+cpu frontend, DTK-fork libtorch relinked by torch_fl (`flagos` device) |
| flag_gems | 5.4.0.dev0 (`FLAGOS_USE_FLAGGEMS=1 TRITON_BACKENDS_IN_TREE=1`) |
| transformers | 5.5.0 |
| DTK | 6.3.26113 |
| Model | Qwen3-0.6B bf16, `attn_implementation="sdpa"`, eval, card 0 (`HIP_VISIBLE_DEVICES=0`) |
| Bench | `benchmarks/bench_qwen3_routes.py`, 3 prompts x 5 timed rounds, 128 new tokens, greedy |

Runtime requirement discovered this change: the boxing (and flaggems) route
must run with
`FA_SO_PATH=/usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`
set. DTK's libtorch defers flash attention to an external `flash_attn_2_cuda*.so`
(`cutlassfa_adapter.h`); its loader computes the directory of the loaded
`libtorch_hip.so` and walks up three parents, which finds the .so for the
vendor's `torch/lib` copy but *not* for torch_fl's staged copy in the venv.
`FA_SO_PATH` names the file directly. A directory value makes `dlopen` fail.

Raw JSON / logs / traces live under `/root/bench/perf/` (outside the repo).

## P0 lever delivered: fused SDPA on the boxing route (composite override)

### Why a composite override (measured root cause)

`aten::scaled_dot_product_attention` is a **composite** op
(`ATen/native/transformers/attention.cpp`). Its fused-backend selection runs
inside the composite, and each chosen backend's leaf is then dispatched by
`query_.device().type()`: only CUDA/XPU tensors reach
`at::_scaled_dot_product_flash_attention`; every other device (including
PrivateUse1) is sent to the CPU kernel
`_scaled_dot_product_flash_attention_for_cpu`. On the boxing route the tensors
are `flagos` (PrivateUse1), so **no per-op conf routing of the attention leaf
ops can ever reach the vendor fused kernel** — the composite picks the CPU
flash leaf before dispatch sees the flagos tensors. The earlier
`_fused_sdp_choice` DispatchStub PrivateUse1 slot had the same flaw: the stub
only picks a *backend*; the backend's leaf is still routed by device type.

### The fix

Register a PrivateUse1 kernel for the composite op itself
(`csrc/aten/sdp_choice_stub.cc`, the `WrapperMatmul` pattern from
`csrc/aten/register.cc`): one `DeviceBoxingGuard` round-trip boxes q/k/v(+mask)
to CUDA, the native composite runs on CUDA tensors — the vendor's own
`_fused_sdp_choice` selector then picks flash exactly as on the vendor route —
and the output is unboxed. Numerics match vendor by construction; shapes the
vendor selector refuses fall back to the composite's math path exactly as on
CUDA. The file is guarded to CUDA-boxing builds so Ascend (which registers the
same stub slot for its aclnn efficient kernel) and MUSA/GCU/BPU do not double
register.

Scope: **inference**. Under grad mode with a grad-requiring input the wrapper
falls through to the composite on the flagos tensors (the pre-change math
decomposition), because a boxed CUDA forward would leave backward a CUDA
autograd graph over tensors the guard unboxes to PrivateUse1 (a
`ToCopyBackward0` device mismatch). The fall-through is byte-identical to the
pre-change path; see "Training-path limitation" below.

### Correctness gates (all green, before any benchmark)

1. **Numerics** (`/root/bench/perf/sdpa_gate_cases.py`, vs CPU fp64 ref):
   ALL PASS — GQA decode 1.998e-03, non-GQA 1.148e-03, causal prefill
   8.445e-03, attn-mask 7.486e-03, head_dim-80 4.817e-03 (bf16, tol 0.05),
   fp32 8.649e-07 (tol 1e-4). The traced backend op is only
   `aten.scaled_dot_product_attention.default` — fused, no decomposition
   leaves. `/root/bench/dump_logits.py` fp32 last-token logits are **bitwise
   identical** to vendor (max|diff| = 0.0), argmax/top-5 match; bf16 in tol.
2. **Op dispatch suite**: `pytest tests/integration/ops -q -m anyplatform` →
   590 passed, 7 skipped, 4 xpassed, 2 failed. Both failures are in
   `test_flaggems_conf_consistency.py` (flaggems conf vs generated-kernel drift
   for the mm/bmm/addmm dispatchers) and are **pre-existing at HEAD** — the
   same two tests plus a third (`test_counts_match`) fail on a clean HEAD
   checkout of the conf + generated files; the working tree's regeneration
   actually improved it 3->2. They are unrelated to this lever (the test reads
   only `backends_flaggems.conf` + `flaggems_python_kernels.cc`, neither of
   which this lever touches).
3. **Training smoke** (`/root/bench/perf/sdpa_train_smoke.py`): ALL PASS —
   backward raises no exception and grads are defined + finite on both GQA and
   non-GQA cases. Nonzero SDPA grads on the boxing route are a pre-existing
   cpu_fallback math-path limitation (the composite's math decomposition
   relocates to CPU via `ToCopyBackward0`), identical before and after this
   lever — out of scope for this inference change.

### Op-level effect

`/root/bench/perf/op_microbench.py`, `sdpa_gqa_decode` (Qwen3 decode shape,
1x16x1x128 q, 1x8x100x128 kv, enable_gqa):

| | before | after |
|---|---|---|
| vendor | 57.9 us | 55.6 us |
| boxing | **275.9 us** (4.8x vendor) | **60.9 us** (1.10x vendor) |

The ~5 us residual on boxing is the guard + double-dispatch round-trip for the
whole attention.

### Op-census effect

Before P0 the boxing decode step issued **3370** aten ops vs vendor **2586**
(+784/step: the SDPA math decomposition — `_safe_softmax` x28, extra `bmm`
x57, `mul.Scalar` x56, `clone` x56 — and no fused flash op). After P0 the
boxing census is **byte-identical to vendor: 2586 ops/step**, with
`aten.scaled_dot_product_attention.default` x28 and none of the decomposition
leaves. All SDPA op amplification is gone.

### End-to-end effect (this is the acceptance measurement)

Baseline (recorded Step 0, 2026-09-06; note the orphan-profiler contamination
caveat in "Measurement gaps"): boxing 20.94 e2e vs vendor 23.57 same-session =
**0.888**. After P0 the card was found to be shared with a stuck 29-hour
profiler process; with it killed, a clean four-round vendor/boxing interleave
(`iters=3` per leg, short_zh) cancels the session thermal drift that corrupts
single-shot comparisons:

| round | vendor e2e tps | boxing e2e tps | ratio |
|---|---|---|---|
| R1 | 24.2 | 22.9 | 0.946 |
| R2 | 23.1 | 22.4 | 0.970 |
| R3 | 24.1 | 23.0 | 0.954 |
| R4 | 24.5 | 23.0 | 0.939 |
| **mean** | **23.98** | **22.83** | **0.952** |

**Boxing e2e = 0.952x vendor (range 0.939-0.970), fwd 47.7 -> 43.8 ms/step —
the 0.95 acceptance line is met on average.** Per-prompt spread from a
3-prompt clean run at the session start: 0.955 / 0.955 / 0.942 (short_zh /
medium_zh / medium_en), i.e. all within ~1% of the line. Reported ratios must
be close-in-time interleaved pairs; see "Measurement gaps" for why.

### Where the residual stands now

With the op census identical to vendor, the remaining ~2 ms/step is the
structural **per-op PrivateUse1 boxing CPU cost** (~1 us x 2586 ops) on a
decode that is CPU-bound (boxing CPU dispatch ~44 ms > device ~41 ms). It is
invisible in per-op microbenchmarks (back-to-back launches overlap CPU with
GPU) and only surfaces in the CPU-bound e2e. A clean copy-family microbench
shows `copy_` alone carries a wrapper tax (~2x) while `_to_copy`/`clone`/`.to`
are at parity and `copy_` is rare in the census — so the previously-projected
P2 (copy wrapper) lever is **not** the residual driver. The named next lever
is a reduction of the per-op boxing dispatch cost (e.g. the generated wrappers
calling the native CUDA function directly instead of re-entering `at::`), a
broad codegen/C++ change deferred beyond this round.

## Step 0 — end-to-end baseline (recorded, historical)

Recorded 2026-09-06 under a stuck orphan profiler sharing the card (see
Measurement gaps). Mean over 3 prompts x 5 rounds:

| Route | TTFT (ms) | fwd ms/step | decode tps | e2e tps | e2e vs vendor |
|---|---|---|---|---|---|
| vendor | 42.6 | 41.02 | 24.38 | 24.37 | 100% |
| boxing | 53.3 | 47.68 | 20.96 | 20.94 | **85.9%** (-14.1%) |
| flaggems | 240.5 | 201.52 | 4.96 | 4.95 | **20.3%** (-79.7%) |

Gap in step time: boxing +6.66 ms/step, flaggems +160.5 ms/step vs vendor.
The same-session JSON pair (vendor.json/boxing.json, both under the orphan)
reads boxing 20.94 vs vendor 23.57 = 0.888, the before-ratio used in the P0
section above. After the P0 lever the flaggems route was re-measured clean at
5.3 tps (unchanged; see below).

## Anatomy of a vendor decode step (the reference)

From the vendor profiler trace and the op census (see the original revision of
this file for the full table): 2586 aten ops/step, 1419 kernel launches/step,
device busy only ~10 ms of the 41 ms step — **the vendor decode is itself
CPU/dispatch-bound**, not device-bound. Top device consumers mm 24.3%, mul
23.8%, copy_ 11.1%, cat 8.9%; SDPA runs fused flash (28 calls/step). Consequence:
the currency that matters is **per-op CPU cost x op count**.

## Op census — routing changes the op stream itself (pre-P0)

| op (per step) | vendor | boxing (pre-P0) | flaggems |
|---|---|---|---|
| total aten ops | **2586** | **3370** | **3370** |
| view | 311 | 479 | 479 |
| _to_copy | 229 | 369 | 369 |
| _unsafe_view | 198 | 254 | 254 |
| expand | 3 | 171 | 171 |
| unsqueeze | 60 | 116 | 116 |
| transpose | 113 | 141 | 141 |
| bmm | 1 | 57 | 57 |
| mul.Scalar | 0 | 56 | 56 |
| clone | 0 | 56 | 56 |
| _safe_softmax | 0 | 28 | 28 |
| _scaled_dot_product_flash_attention | 28 | **0** | **0** |

The +784 ops/step on both flagos routes is the signature of one mechanism:
**SDPA runs the math decomposition instead of fused flash**. This census
predates the P0 override. **After P0 the boxing census is identical to vendor
(2586 ops/step)** — see the P0 section.

## Op-level A/B microbench (us/op, pre-P0; see P0 section for sdpa after)

| op | vendor | boxing | flaggems |
|---|---|---|---|
| add 1x1024 | 9.89 | 8.88 | 179.27 |
| pow 1x1024 | 10.55 | 9.79 | 160.92 |
| rsqrt 1x1024 | 8.92 | 8.12 | 155.25 |
| silu 1x3072 | 9.29 | 8.66 | 153.84 |
| matmul q (1024x2048) | 14.45 | 13.12 | 129.20 |
| matmul lm-head (1024x151936) | 277.81 | 277.66 | 274.85 |
| copy_ 1x1024 | 6.79 | 14.19 | 14.61 |
| sdpa GQA decode | **57.89** | **275.93** | **593.57** |
| sdpa GQA decode (after P0) | 55.58 | **60.90** | (routed to boxing fused) |

Facts: boxing per-op tax is ~0 for launch-bound ops (only `copy_` and the old
`sdpa` fallback regressed); the flaggems `flagos_python` bridge costs
~75-170 us/call on every routed op; the bridge is hidden only under
device-bound kernels (lm-head GEMM at parity).

## Attribution (historical, pre-P0)

- **Boxing +6.66 ms/step vs vendor**: ~92% one mechanism — the SDPA math
  fallback (28 attn/step x ~+218 us = +6.11 ms). This is exactly what P0
  removes; the residual is the per-op boxing CPU cost described above.
- **Flaggems +160.5 ms/step**: ~77% attributed to the Python bridge (~116
  ms/step from add/mm/pow/mean/rsqrt/silu/bmm/softmax crossing C++->Python);
  + boxing baseline. No evidence of Triton kernel inferiority — every
  cuda-routed op is at parity, and the device-bound lm-head GEMM is at parity
  with vendor. Post-P0 clean re-measure: flaggems 5.3 tps (~0.22x vendor) —
  unchanged, because its ~190 ms/step is bridge-bound across thousands of ops
  and the SDPA decomposition was a small slice of that.

## Optimization priorities (reranked after P0)

**P0 — DONE.** Restore fused SDPA on the boxing route. Delivered via the
composite override; boxing e2e 0.889 -> **0.952x vendor**, meeting the target.

**P1 — Reroute the flaggems hot tiny-op cohort from flagos_python to cuda.**
~950 bridge calls/step cost ~116 ms/step; at 0.6B decode shapes no measured
flagos_python op beats its boxing counterpart, so the honest expectation is
"flaggems ~ selective boxing" (skill Step 5). Conf/codegen change with the
operator-support.md revalidation obligations. Expected: flaggems -> boxing-like
(~23 tps).

**P1.5 — Reduce the per-op boxing CPU dispatch cost (only if the target
tightens).** The residual after P0 is ~1 us x 2586 ops of PrivateUse1 boxing
CPU cost on a CPU-bound decode. Named candidate: generated wrappers call the
CUDA native function directly instead of re-entering `at::` (one dispatcher
round saved per op). Broad codegen change, measured-ROI-uncertain, deferred.

**P2 (copy_ wrapper) — measured NOT a driver; dropped.** Only `copy_` carries a
wrapper tax; `_to_copy`/`clone` are at parity and `copy_` is rare in the census.

**P3 — Op-count reduction via composite routing — moot on boxing now** (census
matches vendor); still open for the flaggems bridge bound (P1 supersedes).

**Not recommended (measured evidence against):** KV-cache persistent boxing /
already-boxed sets — the per-op boxing tax outside copy_/sdpa measures ~0 and
the KV cache is a fresh-tensor-per-`cat`, so a step-scoped guard cannot amortize
it; Python bridge cost reduction (moot once P1 reroutes the hot cohort).

## Measurement gaps and tooling findings

- **Stuck orphan profiler (contamination of every pre-P0 number).** A
  `--mode profile --device flagos` bench process was found wedged on the card
  for ~29 hours. It was alive during the recorded Step 0 baseline and every
  earlier microbench, so pre-P0 absolute numbers are depressed (the vendor was
  ~23.5 instead of a clean ~24.3). It was killed before all clean after-P0
  measurements. The ratio-based P0 delta (0.888 -> 0.952) is robust to this
  because vendor and boxing were equally contaminated.
- **Session thermal drift corrupts single-shot route comparisons.** Both routes
  slow over a long benchmark session (vendor 24.3 -> 23.3, boxing 22.9 -> 21.7
  over ~an hour), and the CPU-bound boxing route drifts slightly more. A
  single vendor-then-boxing comparison can read anywhere from 0.90 to 0.955
  depending on window. **The correct protocol is close-in-time interleaved
  pairs averaged** — that is what the 0.952 figure uses.
- **torch.profiler deadlocks on the flagos routes** (parks in futex_wait at
  teardown; reproduced). All flagos-route attribution rests on census +
  microbench + e2e triangulation.
- **Training-path limitation (pre-existing):** SDPA under grad mode on the
  boxing route yields defined-but-zero q/k/v grads because the composite's
  math decomposition is relocated to CPU by cpu_fallback (`ToCopyBackward0`),
  terminating the graph before it reaches the flagos tensors. The override's
  grad fall-through preserves exactly the pre-change semantics (no crash).
  Nonzero SDPA grad flow on the boxing route is a separate follow-up.
- **Dispatch suite pre-existing failures:** 2 in `test_flaggems_conf_consistency`
  (3 on clean HEAD) from conf-vs-generated-kernel drift in the flaggems
  mm/bmm/addmm routing; unrelated to this lever.
- **FA_SO_PATH** runtime requirement (see Environment) is not in the skill's
  route matrix and must be set on boxing/flaggems runs or flash silently
  degrades to the math path / fails to load.

## Reproduction

```bash
# P0-enabled boxing e2e, interleaved against vendor to cancel drift
FA_SO=/usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so
for i in 1 2 3 4; do
  source /opt/dtk/env.sh
  export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch.libs:$LD_LIBRARY_PATH
  python3 benchmarks/bench_qwen3_routes.py --device cuda --iters 3 --out-json /root/bench/perf/v_$i.json
  cd /root/bench
  HIP_VISIBLE_DEVICES=0 FA_SO_PATH=$FA_SO /opt/fl-envs/boxing/bin/python \
    /workspace/PyTorch-Plugin-FL/benchmarks/bench_qwen3_routes.py --device flagos \
    --iters 3 --out-json /root/bench/perf/b_$i.json
  cd /workspace/PyTorch-Plugin-FL
done
# gates
/opt/fl-envs/boxing/bin/python /root/bench/perf/sdpa_gate_cases.py
/opt/fl-envs/boxing/bin/python /root/bench/perf/sdpa_train_smoke.py
```

## Next steps

P1 (flaggems hot-cohort reroute to cuda) is the remaining lever that can move
the flaggems route to a boxing-like number. The boxing route has met the 0.95
line; P1.5 (per-op boxing CPU cost) is available if the target tightens.
