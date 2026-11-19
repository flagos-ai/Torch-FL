# Enflame native-host validation

**Historical vendor-kernel result.** The November 17 model comparison below
uses topsaten for its numerical kernels and does not qualify FlagGems model
performance. Continue with the
[FlagGems-first validation](flaggems-native-host-validation.md) for the measured
FlagGems routes, actual submission counts and the November 18 comparison.

Date: 2026-11-17 (Asia/Shanghai). This is an unpublished working-tree candidate
based on `eba6b82e95b4ae3cc7f1784c54fd8724596f3845`.
The [optimized measurement record](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_native_host_optimized.json)
contains verified source and artifact hashes, environment, commands and gates.

**The complete Qwen3-0.6B comparison passes the 80% performance gate and vendor
precision acceptance. Prefill reaches 121.5% and cached
decode reaches 118.5% of vendor performance. All nine
logit comparisons have maximum absolute error zero. All 63 hardware conformance
cases pass.** The observed tensor CPU fallbacks have been replaced with vendor
device paths; scalar exception/range predicates still return to the host.

All 28 layers and 596,049,920 parameters execute in ordinary eager mode, following
the [build-time host design](../../design/flaggems-build-time-hosts.md). This
qualifies this workload and cohort, not every model or operator on Enflame.
The [earlier failure record](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_native_host_validation.json)
and its raw reports are preserved as historical evidence.

## Full-model performance and precision

Both providers ran sequentially on physical device 0 with the same checkpoint,
BF16, eager attention, batch one, 23 prompt tokens and eight cached decode steps.
CPU affinity is 64-71, Torch threads are one, and Transformers/dependencies match.
Seven measured repetitions follow one unmeasured repetition: seven prefill and
56 decode observations per provider. Acceptance runs have no profiler or
`LD_PRELOAD` observer.

| Synchronized forward median | Vendor PyTorch | Optimized native-host wheel | Vendor / candidate performance | Maximum latency for 80% |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 51.61 ms | 42.47 ms | 121.5% | 64.52 ms |
| One cached decode step | 45.19 ms | 38.13 ms | 118.5% | 56.49 ms |

For fixed work, relative performance is vendor latency divided by candidate
latency. CPU submission medians are 41.80/37.46 ms for
the candidate and 50.95/44.50 ms for the vendor.
Device-side token selection runs on both providers, outside timed forwards.
These are model-forward latencies, not end-to-end generation throughput or
empty-cache compilation measurements.

The checkpoint, precision contract and math settings match. Full prefill logits
at every position and all eight requested decode outputs pass `atol=rtol=0.001`;
every compared value is equal, with zero violations and no nonfinite values.
The autoregressive token arrays also match exactly:

```text
The sky appears blue because of the way
```

Both processes exit with status zero. Raw reports are the
[vendor result](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_vendor_enflame_optimized.json)
and [candidate result](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/qwen3_eager_source_hosts_enflame_optimized.json).

## Bottleneck measurements and optimization

The original 149.92/139.87 ms candidate drained its stream before and after every
topsaten/copy call. Two diagnostic forwards recorded **7,176 stream synchronizes**
and 173.40 ms aggregate API duration. The native launcher now enables ABI-checked
producer tracking after a device drain. Submissions from the same producer stay
asynchronous; a topsaten/Triton transition drains before the consumer. A shared
submission lock prevents concurrent host threads from crossing an open producer
transition, and stream destruction clears its record. Ordinary builds retain
their conservative pre/post waits.

The final separate forwarding API observer records **22 stream synchronizes**
across the same two model scopes. It forwards SDK calls unchanged. No wait-skipping
preload or substitute kernel participates in acceptance. FP32 weak-scalar Add now
uses the measured vendor scalar overload, removing 226 small blocking scalar
uploads seen in the preceding diagnostic candidate. Half/BF16 Add retains its
previous staging because the scalar overload differs on some rounding cases.

Real TOPSPTI kernel durations give the following **instrumented-run** breakdown:

| Provider / phase | Profiled wall | Device kernel sum | Device copy sum | Host/submission/idle remainder |
| --- | ---: | ---: | ---: | ---: |
| Candidate prefill | 74.56 ms | 38.54 ms | 0.32 ms | 35.69 ms |
| Candidate decode | 68.52 ms | 34.30 ms | 0.20 ms | 34.03 ms |
| Vendor prefill | 102.90 ms | 35.99 ms | 0.06 ms | 66.85 ms |
| Vendor decode | 100.07 ms | 30.80 ms | 0.08 ms | 69.19 ms |

The remainder is wall time minus the union of device kernel/copy intervals. It
includes profiling overhead, host submission, waits and other device gaps; it is
not pure Python CPU time. These values must not be subtracted from unprofiled
medians. The candidate still launches more numerical kernels
(2519/2357 versus
2188/1936) and has slightly
higher aggregate device time. Whole-forward performance improves by removing
host waits and scalar transfers. TOPS device queries remain frequent (72,472 in
the API diagnostic); kernel launches, casts and broadcasts are further potential
optimization targets, not additional completed improvements.

The [native profile](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_native_host_profile.json)
and [vendor profile](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_vendor_profile.json)
record top kernels, API counts and raw-trace hashes/paths. Native traces use
`torch.profiler`; vendor traces use `torch_gcu.gcu.profiler`. Device activity
required profiling-capable privileged execs; ordinary-container attempts returned
unusable timestamps. The host `topsprof` 1.9.10 CLI could not import this SDK 1.7
capture, so none of its attempted results supply timings. Compute SDK versions
and numerical dependencies remain unchanged.

## CPU fallback and precision investigation

An explicit audit of SDK transfer call stacks found hidden CPU computation
inside registered native operators. The generated device paths now handle:

- Bounded int64 position/cache `arange` through the vendor int32 factory and
  device sign extension; empty ranges allocate directly on the device.
- Int64 Add/Sub with alpha one and all six comparisons through vendor int32
  operations over both storage words, including carry/borrow and signed ordering.
- Embedding forward bounds validation on device, followed by safe int32 indexing.
  Invalid int64 indices raise before the vendor's fatal device assertion.
- Int64 casts within the measured int32 range, with device validation before
  narrowing; strided int64 copies preserve both words on device.
- Device `argmax`/`argmax.out` via vendor int32 indices and zero extension into
  logical int64 output, including noncontiguous output handling.

The final API observer and independent TOPSPTI memory trace show only **four
one-byte D2H bool predicates**, for embedding bounds and conversion range checks,
and two two-byte H2D scalar constants. All other traced copies are device to
device. There are no tensor readbacks for CPU numerical computation in these
observed model scopes. This is stronger evidence than absence of a fallback log
message, which misses internal `.cpu()` branches. Unsupported operators and
wider int64 cast/factory ranges outside this workload still retain existing
fallbacks; this result does not claim platform-wide removal.

Precision was localized with first-attention operand snapshots. The first large
error was BF16 attention-score scaling: the exact scalar
`128**-0.5 = 0.08838834764831845` had been staged as BF16
`0.08837890625`, introducing score error up to 0.25. The generated Mul scalar
route now preserves FP32 opmath, matching vendor PyTorch. The
[operator diagnostic](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_attention_precision_diagnostic.json)
records the captured sequence. Snapshot readbacks deliberately perturb execution
and never provide acceptance timings.

After correcting Mul, the published FlagGems `mean.dim`/SiLU combination still
fails full-model vendor precision. Both measured routes therefore use topsaten
in the generated default conf. Their policy has a removal condition: a published
cohort must pass this same model precision contract. No reduced-precision setting
was changed to pass the check. The older original-host control matched the
pre-optimization adapter exactly, establishing that those numerical differences
predated the new launcher.

**The optimized model's numerical path launches vendor kernels.** Its FlagGems
kernel audit is empty after those precision-driven route changes. The compiled
typed-entry counters include ATen/composite hosts and must not be reported as
FlagGems numerical coverage or as proof of a typed-host speedup. Actual FlagGems
TOPS submissions are verified separately by the hardware adapter tests.

## Adapter, environment and artifact

The [Python adapter](../../../torch_fl/native_host/backends/enflame.py) and
[native launcher](../../../csrc/native_host/backends/enflame.cc) package
the TOPS module loader and unchanged published `gcu_utils` at wheel build time.
They preserve the published Triton 3.6 software-descriptor and integer-pointer
ABI, launch geometry, live output binding and stream refresh. Unsupported launch
features fail explicitly, and runtime C/C++ launcher compilation is blocked.
Vendor pointwise hosts retain compiled Cython object semantics; new ranks generate
device code only. Hardware cases cover the measured GCU300 path and nondefault
`TopsStream` contexts, not every architecture supported by source branches.

The node is labeled S60; EFSMI reports ZIXIAOC200, driver 1.10.6 and 42,976 MiB.
Device 0 had no process before each paired acceptance run and after diagnostics.
`TOPS_VISIBLE_DEVICES=0` selects it; snapshots are not a hardware reservation.
Both environments derive from the same Python 3.12.13 / CPU Torch 2.10.0+cpu
environment with Transformers 5.19.0. The candidate uses published FlagGems 5.4.0,
FlagTree 0.7.0+enflame3.6 / Triton 3.6.0 and FlagCX 0.14.0+tops1.9.10. The isolated
baseline imports torch-gcu 2.10.0+3.7.20260408 and neither Torch-FL nor FlagGems.
Cython 3.2.4 is build-only. No dependency was rebuilt from source.

The existing Ubuntu 22.04.5 container image is
`registry-egc.enflame-tech.com/artifacts/torch_gcu:v2.10.0-TR3.7.107-ubuntu2204`.
Installed topsruntime/topscc are **1.7.2.2402**, topsaten is 3.6.20260403 and the
runtime is `/opt/tops/lib/libtopsrt.so.1.7`. This is not the current pinned CI
image. The candidate wheel is
`torch_fl-2.10.0+tops1.9.10-cp312-cp312-linux_x86_64.whl`, SHA256
`ec842db83479dc09eced9a495d6d9a2cbe165a8e706ca7fa1a26d298ec9fa306`.
The filename uses the existing platform-table variant; the generated compatibility
manifest records verified SDK **1.7.2.2402**, CPU Torch and
`KERNELS=('flaggems', 'vendor')`. Preflight passes with build-environment and
native-backend checks. This wheel has not been published.

The checkpoint SHA256 is
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`,
and configuration SHA256 is
`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
The native report records 797 hashed host artifacts,
684 typed variants and 990 compiled-object fallbacks. Model-capture calls and
legacy host-program counters are zero.

## Reproduction and verification

Build with the verified published dependency cohort and CPU Torch:

```bash
export FLAGOS_ACCELERATOR=enflame FLAGOS_BUILD_NATIVE_HOST=1
export FLAGOS_BUILD_FLAGGEMS=1 FLAGOS_BUILD_VENDOR=1
export FLAGOS_BUILD_BOXING=0 FLAGOS_BUILD_FLAGGEMS_CPP=0 FLAGOS_BUILD_TILEOPS=0
export FLAGOS_DISABLE_CUDA_ASSETS=1 TOPS_HOME=/opt/tops
export TOPSATEN_LIB=/usr/lib/libtopsaten.so FLAGOS_SDK_VERSION=1.7.2.2402
export CPATH=/opt/tops/include/gcu:/opt/tops/include
export LIBRARY_PATH=/opt/tops/lib:/usr/lib
export MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8
python setup.py bdist_wheel
python scripts/tools/torch-fl-preflight --wheel dist/*.whl \
  --check-build-env --check-native-host-backend --sdk-version 1.7.2.2402
python -m pip install --no-deps --force-reinstall dist/*.whl
unset FLAGOS_SDK_VERSION
```

Set `LD_LIBRARY_PATH` to TOPS and the CPU-Torch environment's library directory.
`SOURCE`, `MODEL`, `RESULTS`, `VENDOR_PYTHON` and `NATIVE_PYTHON` name the candidate
tree, checkpoint, output directory and isolated interpreters:

```bash
export TOPS_VISIBLE_DEVICES=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /tmp
taskset -c 64-71 "$VENDOR_PYTHON" "$SOURCE/benchmarks/host_compiler/qwen_strict_eager.py" \
  --model "$MODEL" --device gcu:0 --dtype bfloat16 --steps 8 --rounds 7 \
  --output "$RESULTS/vendor.json"
taskset -c 64-71 "$NATIVE_PYTHON" "$SOURCE/benchmarks/host_compiler/qwen_strict_eager.py" \
  --model "$MODEL" --device flagos:0 --dtype bfloat16 --steps 8 --rounds 7 \
  --reference "$RESULTS/vendor.pt" --check-logits --output "$RESULTS/native.json"
```

For separate device diagnostics, use `--rounds 0 --skip-sampling --profile-trace`
in a profiling-capable exec. The native command may retain `--reference` and
`--check-logits`. For the forwarding API audit, build the optional observer and
run another diagnostic process with `LD_PRELOAD` and `--api-profile`:

```bash
g++ -shared -fPIC -O3 -std=c++17 -pthread -I/opt/tops/include \
  "$SOURCE/benchmarks/host_compiler/enflame_api_trace.cc" \
  -L/opt/tops/lib -Wl,-rpath,/opt/tops/lib -ltopsrt -ldl -o enflame_api_trace.so
LD_PRELOAD="$PWD/enflame_api_trace.so" "$NATIVE_PYTHON" \
  "$SOURCE/benchmarks/host_compiler/qwen_strict_eager.py" \
  --model "$MODEL" --device flagos:0 --dtype bfloat16 --steps 8 --rounds 0 \
  --skip-sampling --reference "$RESULTS/vendor.pt" --check-logits \
  --api-profile "$RESULTS/native-api.json" --output "$RESULTS/native-api-report.json"
"$NATIVE_PYTHON" "$SOURCE/benchmarks/host_compiler/enflame_profile_summary.py" \
  --trace "$RESULTS/native-pti.trace.json" --api-trace "$RESULTS/native-api.json" \
  --output "$RESULTS/native-profile-summary.json"
```

The [hardware suite](../../../tests/integration/ops/test_enflame_native_hosts.py)
covers live launches, compiled ABI/manifest agreement, scalar/descriptor packing,
stream rebinding, compiler rejection, precision, int64 carry/overflow/comparison,
embedding bounds, NaN/empty/scalar argmax and forty real mixed-producer chains.
Twelve schedule cases compare packed launches against the published wrapper;
they do not claim numerical int64 Triton support.

```text
$ ruff check
All checks passed!
$ ruff format --check
629 files already formatted
$ python -m pytest <native-host and related unit suites> -q -rs --disable-warnings
303 passed, 4 skipped, 54 warnings in 2.74s
$ python -m pytest <codegen, conf consistency and stream-state unit suites> -q --disable-warnings
165 passed, 21 warnings in 4.17s
$ python -m pytest <Torch-dependent unit suites on Linux> -q --disable-warnings
88 passed, 39 warnings in 30.13s
$ python -m pytest tests/integration/ops/test_enflame_native_hosts.py -q --disable-warnings
63 passed in 5.69s
```

Exact commands are in the machine-readable record. The macOS skips are three
modules requiring absent Torch/torchgen and one MetaX case requiring GNU ld.
The Linux run covers those Torch-dependent modules. Warnings concern collection
markers, existing operator registration and the published capability helper.
Repeated codegen leaves all four generated artifacts unchanged. Source hashes
for 24 implementation, benchmark and hardware-test files match the remote tree.
The earlier [readiness snapshot](https://github.com/lvyufeng/Torch-FL/blob/9b409b673488f7864eb9d84e59a43f88d862496d/benchmarks/host_compiler/results/enflame_native_host_readiness.json)
is historical and superseded by the implemented adapter and current measurements.
