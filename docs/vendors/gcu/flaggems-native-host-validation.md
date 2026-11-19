# FlagGems-first Enflame native-host validation

Original measurement: 2026-11-18 (Asia/Shanghai). Working-tree candidate based on
`eba6b82e95b4ae3cc7f1784c54fd8724596f3845`. The
[measurement record](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_native_host.json)
contains source/artifact hashes, environment, gates and raw-report references.

Raw JSON reports are archived on the contributor fork's
[evidence branch](https://github.com/lvyufeng/Torch-FL/tree/codex/enflame-native-host-evidence-20261119/benchmarks/host_compiler/results).
Report links below use the immutable measurement commit. The implementation
PR includes the code, tests and this readable summary; it carries no raw result
JSON files.

**The November 18 complete Qwen3-0.6B measurement met the 80% median-performance target with
real FlagGems numerical kernels: 82.6% for prefill and 80.7% for cached decode.
All nine logit comparisons are bitwise identical to vendor PyTorch.**
The [post-rebase qualification](#post-rebase-qualification-november-19) preserves
exact logits but has not stably reproduced the performance gate. The original
measurement is historical evidence and does not establish the current PR's
performance acceptance.
The final candidate process records 87,032 successful FlagGems submissions across
19 kernel names. This count includes validation, sampling, warm-up, timings and
the warm-host audit; it is not a per-forward count.

The [November 17 result](native-host-validation.md) remains historical evidence
for the vendor numerical path. Its 121.5%/118.5% result did not qualify FlagGems
model performance.

## Model and acceptance

All 28 layers and 596,049,920 parameters execute in ordinary eager mode:
BF16, eager attention, batch one, 23 prompt tokens and eight cached decode steps.
The vendor and candidate run sequentially in isolated environments on device 0,
with CPU affinity 64-71 and one Torch thread. Both use the same checkpoint,
Transformers, math settings and device-side token selection.

Seven measured repetitions follow one unmeasured repetition: seven prefill and
56 decode observations per provider. Acceptance has no profiler, dispatch mode,
model hooks, preload or model graph capture.

| Synchronized forward median | Vendor PyTorch | FlagGems-first candidate | Vendor / candidate | Maximum candidate latency for 80% |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 51.397 ms | 62.200 ms | 82.63% | 64.246 ms |
| One cached decode step | 45.085 ms | 55.872 ms | 80.69% | 56.357 ms |

Relative performance is vendor latency divided by candidate latency for fixed
work. CPU submission medians are 61.552/55.209 ms for the candidate and
50.758/44.399 ms for the vendor. These are forward latencies, not end-to-end
generation throughput or cold compilation measurements.

The gate is on medians. Decode has only 0.485 ms of median headroom in this run;
tail samples exceed the latency bound. Candidate p95 is 68.156 ms for prefill
and 60.153 ms for decode. This result does not guarantee 80% on every step,
other workloads or other dependency cohorts.

Full prefill logits at every position and all eight requested decode outputs
pass the unchanged `atol=rtol=0.001` contract with maximum absolute error zero.
An independent CPU comparison of contiguous byte views verifies the complete
prefill array, all nine last-token arrays, all eight decode inputs and prompt
IDs. Autoregressive tokens also match, producing:

```text
The sky appears blue because of the way
```

Raw results: [candidate](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_enflame_flaggems_final.json),
[vendor](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_enflame_vendor_flaggems_comparison.json),
and [independent byte verification](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_bitwise_verification.json).

## Post-rebase qualification (November 19)

The PR candidate was rebased onto upstream
`f0b41362390a41e0d1d598379f658910eaf02ac1` and rebuilt with the same published
dependency and SDK cohort. Both the default and experiment installed-wheel
conformance runs pass all 67 cases. The expanded Linux unit selection passes
695 cases. Two GCU codegen runs and generated-config checks preserve all five
artifact hashes; all 41 changed source/config/test hashes match the remote tree.

The first fresh seven-round paired comparison records vendor prefill/decode
medians of 50.872/44.465 ms and candidate medians of 73.272/66.354 ms:
**69.43%/67.01%, failing the 80% gate**. All nine complete logit comparisons
remain exact, all 19 independent tensor byte comparisons pass, and the candidate
still records 87,032 successful FlagGems enqueues over 19 kernel names.

Within this process decode latencies shift from approximately 56 ms to 66 ms
and return to 56 ms, with unchanged operator/submission counts. The shared
TOPS launch/device sources and warm Python-call counts match the historical
candidate. This indicates run-dependent overhead but does not establish its
root cause or rule out a code-related contribution. A reverse-order control
and its complete results are recorded alongside the initial run; no slow
samples are discarded to assert acceptance.

| Post-rebase measurement | Vendor prefill / decode median | Candidate prefill / decode median | Relative prefill / decode performance |
| --- | ---: | ---: | ---: |
| Initial vendor-then-candidate run | 50.872 / 44.465 ms | 73.272 / 66.354 ms | 69.43% / 67.01% |
| Reverse-order control | 51.718 / 45.209 ms | 63.188 / 56.935 ms | 81.85% / 79.40% |
| All retained observations pooled | 51.505 / 44.843 ms | 63.289 / 57.049 ms | 81.38% / 78.60% |

The pooled result uses 14 prefill and 112 decode observations per provider.
Decode misses the 80% latency bound by 0.424 ms in the reverse-order control
and 0.995 ms in the pooled observations. Both candidate processes have exact
logits and 87,032 successful FlagGems enqueues each. CPU cgroup snapshots show
no quota or throttling; this excludes that specific explanation, not other
host/device interference. The remaining variability and decode gap are open
performance work.

Current evidence: [initial candidate](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_enflame_flaggems_pr_initial_20261119.json),
[initial vendor](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_enflame_vendor_pr_initial_20261119.json),
and [post-rebase validation record](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_pr_validation_20261119.json).
The [reverse-order candidate](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_enflame_flaggems_pr_repeat_20261119.json)
and [reverse-order vendor](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_enflame_vendor_pr_repeat_20261119.json)
retain every measured sample.
The source-generated host adapter and accuracy gates are qualified for this
model; stable 80% performance acceptance remains open.

Independent published-wheel probes on original torch_gcu, without Torch-FL,
also reproduce a non-contiguous broadcast-add bug for both Int32 and Float32
and the cache-cat compiler failure. All sliced add cases fail while contiguous
controls pass; GPU-side comparison confirms that this is not a CPU readback
artifact. The Int32 vendor guard fixes the measured model composition only.
The experiment config does not qualify arbitrary sliced Float32 add layouts.
See the [add sweep](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_strided_add_direct_20261119.json)
and [operator/source report](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_published_gems_direct_20261119.json).
Direct norm/pow/mm differences satisfy per-op `atol=rtol=1e-3`; their amplified
full-model differences motivate a precision compatibility request, not a claim
that vendor non-equivalence alone proves a FlagGems defect. In the direct decode
matmul case, FlagGems matches the rounded CPU Float64 reference exactly while
the vendor differs in one value.

## Actual operator routes

The generated, opt-in
[experiment config](../../../benchmarks/host_compiler/configs/enflame_native_host.conf)
routes nine model operators directly to FlagGems:
`mul.Tensor`, `add.Tensor`, `neg`, `cos`, `sin`, `where.self`, `silu`, `_softmax`
and `bmm`. The generated GCU scalar-pow category also expresses FP32 `pow(2)`
through the selected multiply route, so this model operation executes the
published FlagGems multiply kernel. Other powers and dtypes retain topsaten.
No handwritten numerical device kernel or dependency source patch is added.

The following holds have measured reasons:

| Operation | Evidence and retained route |
| --- | --- |
| `mean.dim` | Isolated FlagGems substitution changes full-model logits by up to 1.0; retain topsaten. |
| `rsqrt` | Isolated substitution changes logits by up to 1.0; retain topsaten. |
| `mm` | Isolated substitution changes logits by up to 0.9375; retain topsaten. |
| `cat` | Published GCU kernel fails compilation on cache concatenation, with an int32/int64 `dim_size_in` branch mismatch; retain topsaten. |
| Int32/Int64 `add.Tensor` | Int32 FlagGems add breaks the vendor int64 carry/word composition. Generated dtype clauses preserve the vendor path; floating add remains FlagGems. |

SiLU alone is exact: the earlier joint mean/SiLU hold was too broad. SiLU is
restored to FlagGems in the default Enflame config. The other experiment routes
do not change default coverage, which retains its older-cohort and large-grid
holds. In particular, this experiment's small softmax shapes do not qualify
long-context or Qwen-Image softmax grids.

The [isolated substitutions](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_operator_probe.json)
and [square composition](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_square_probe.json)
record real submissions. Launcher construction is a separate provenance metric.
The [zero-launch control](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_zero_launch_control.json)
reproduces all logits through a generic multiply that redispatches to ATen,
but submits zero FlagGems kernels and is rejected. The older generic diagnostic's
`individually_passing` field represented accuracy only; it must not be read as
FlagGems execution evidence. The current probe requires a zero-submission vendor
control and positive successful submissions before admitting an operator.

## Host and device bottlenecks

The first ordinary nine-operator FlagGems trial took 161.87/152.18 ms. Separate
diagnostics identified expensive pointwise preparation and 2,582 stream drains
across prefill and the first decode. C++ argument preparation and the GCU300
schedule reduce the trial to 94.27/86.56 ms. Source-derived C++ entries, including
safe expansion of the published read-only variadic dtype predicate, reduce it
to 78.08/70.25 ms. These one-repetition trials are diagnostic progress, not the
final acceptance result.

The pointwise adapter preserves the published DMA predicate, both device
variants, dtype-dependent tile budget, MMU bound, warp rule, strides, extra tile
count and `ENABLE_I64`. Unsupported preparation keeps the compiled published
fallback. Numerical device source remains from the published FlagGems wheel.

Stream interop ABI 2 serializes producer submissions and records/waits a
per-stream TOPS event at topsaten/Triton transitions. It preserves device
ordering without a host drain at every transition. ABI 1 still supports
transition drains; ordinary untracked builds retain legacy pre/post drains.
Event errors release the lock, and forgetting a stream destroys its event.
The launch-backend ABI is a separate contract. These measurements used version
1; current builds use version 2, which also supplies kernel metadata to the
descriptor factory. Enflame's software descriptor does not consume that extra
metadata. Rebuild the host bundle when changing this ABI.

The final first-prefill/first-decode API scopes contain 22 stream synchronizes,
even with the additional FlagGems square path. A temporary event prototype was
used for diagnosis; final acceptance uses the rebuilt installed wheel without
that prototype or any preload.

Separate profiling-capable execs collect real TOPSPTI device durations:

| Provider / phase | Profiled wall | Device kernel sum | Device copy sum | Host/submission/device-idle remainder |
| --- | ---: | ---: | ---: | ---: |
| Candidate prefill | 84.725 ms | 43.393 ms | 0.141 ms | 41.191 ms |
| Candidate decode | 77.071 ms | 36.100 ms | 0.186 ms | 40.785 ms |
| Vendor prefill | 85.538 ms | 35.306 ms | 0.054 ms | 50.178 ms |
| Vendor decode | 76.842 ms | 30.091 ms | 0.099 ms | 46.652 ms |

These instrumented durations are not acceptance timings. The remainder uses the
union of device intervals and includes host work, waits and profiler overhead;
it is not pure Python time. FlagGems `softmax_kernel_inner` consumes 7.402 ms
over 28 prefill calls, and rank-3 broadcast multiply consumes 7.744 ms over
115 decode calls. The independent API observer records 145.527 ms in event
record calls over nine forwards. Device kernels and event submission remain
material costs even after removing repeated host drains.

See the [candidate device profile](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_device_profile.json)
and [vendor device profile](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_vendor_device_profile.json).

## CPU fallback audit

The [full-forward API audit](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_flaggems_full_forward_api.json)
captures one warmed complete prefill and all eight cached decode steps through
model hooks, excluding benchmark logit reads and checkpoint loading. No
diagnostic dispatch mode is active. Python `Tensor.cpu` is also rejected inside
all model forwards in that diagnostic process.

All observed D2H transfers are **18 one-byte booleans**: nine embedding bounds
checks and nine dtype-cast range predicates, with resolved GCU call stacks.
There is no observed model tensor CPU numerical fallback. Nine two-byte H2D
scalar uploads and 100 four-byte D2D word-composition transfers remain. This
evidence qualifies these model shapes; it is not a claim that every unsupported
platform dtype or operator has lost its CPU fallback.

## Cohort and reproduction

The S60 asset reports `ZIXIAOC200` / GCU300, driver 1.10.6. Its existing Ubuntu
22.04 container has TOPS runtime/compiler 1.7.2.2402 and topsaten 3.6.20260403.
The native environment uses CPU Torch 2.10.0, FlagGems 5.4.0,
FlagTree 0.7.0+enflame3.6 (Triton 3.6), FlagCX 0.14.0+tops1.9.10 and
Transformers 5.19.0. Dependencies are published wheels. The isolated vendor
environment uses published torch-gcu 2.10.0+3.7.20260408 with the same CPU Torch
and Transformers, without importing torch_fl or FlagGems.

Build as described in the
[native-host build record](native-host-validation.md#reproduction-and-verification).
The new wheel SHA256 is
`61451a990ef2cf7a1de45840a9c0d5536c40e71e2b930204102957378f3f6655`.
The filename's `+tops1.9.10` segment is a legacy build variant; verified SDK
metadata is **1.7.2.2402**, not inferred from that filename.

Regenerate the explicit experiment config, then run both providers outside the
source checkout. Set `SOURCE`, `MODEL`, `RESULTS`, `VENDOR_PYTHON` and
`NATIVE_PYTHON` to the corresponding paths:

```bash
python "$SOURCE/scripts/codegen/gen_vendor_confs.py" \
  --enflame-native-host-conf "$SOURCE/benchmarks/host_compiler/configs/enflame_native_host.conf"
export TOPS_VISIBLE_DEVICES=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /tmp
taskset -c 64-71 "$VENDOR_PYTHON" "$SOURCE/benchmarks/host_compiler/qwen_strict_eager.py" \
  --model "$MODEL" --device gcu:0 --dtype bfloat16 --steps 8 --rounds 7 \
  --output "$RESULTS/vendor-final.json"
FLAGOS_BACKEND_CONFIG="$SOURCE/benchmarks/host_compiler/configs/enflame_native_host.conf" \
taskset -c 64-71 "$NATIVE_PYTHON" "$SOURCE/benchmarks/host_compiler/qwen_strict_eager.py" \
  --model "$MODEL" --device flagos:0 --dtype bfloat16 --steps 8 --rounds 7 \
  --reference "$RESULTS/vendor-final.pt" --check-logits --output "$RESULTS/native-final.json"
```

Keep device profiling and the forwarding API observer in separate processes.
For the nine-forward audit, use `qwen_operator_census.py --census-forwards 0
--reject-model-readbacks --forward-api-dir PATH --forward-api-start 9
--forward-api-count 9` with one diagnostic timing repetition and the forwarding
observer preloaded. Those timings are excluded from acceptance.

Verification output for the final candidate:

```text
ruff check: All checks passed!
ruff format --check: 631 files already formatted
Linux unit suites: 478 passed, 116 warnings in 34.02s
Installed wheel, experiment config: 67 passed, 15 warnings in 26.38s
Installed wheel, default config: 67 passed, 15 warnings in 26.41s
Wheel preflight: exit 0, including build environment and native backend checks
Repeated codegen/config checks and 39-file source hash verification: exit 0
```

Twelve hardware cases compare launch plans rather than numerical int64 FlagGems
coverage. Tests also cover live tensors, cold/warm calls, nondefault streams,
mixed producers, square special values/layouts and vendor int64 composition.
Generic setup unit tests run after unsetting the build-only SDK override.
Training, other architectures, long contexts and additional model families
remain outside this qualification.
