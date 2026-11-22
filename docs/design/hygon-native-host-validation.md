# Hygon native-host validation

## PR qualification: 2026-11-19

**The repeated performance gate remains open.** The maintainer has authorized
merging before the performance follow-up is complete and checking CI. Fresh
qualification after rebasing onto upstream `f0b41362390a41e0d1d598379f658910eaf02ac1`
retains bitwise-identical logits, but short pairs `s03` and `s05` exceed the
1.25x latency limit. Worst short prefill/decode ratios are **1.264/1.260**;
both 512-step pairs pass, with worst ratios **1.212/1.210**. The earlier passing
qualification below does not override these fresh failures. The remaining
latency variation has not been isolated to a specific rebased change.

The [PR validation record](../../benchmarks/host_compiler/results/hygon_native_host_pr_validation.json)
contains all eight fresh pairs, commands, timings, actual test output, source and
wheel fingerprints, and the independent saved-logit byte checks. It reports
`performance_gap_remaining`, `correctness_passed=true` and
`performance_passed=false`. No failing pair is discarded or hidden by pooling.

The measured source commit is
`200d07988035a454b889c1960718148f70ea0771`; 1,113 transported source files match
between local and remote checkouts. Subsequent changes update documentation and
evidence. The rebuilt wheel is `torch_fl-2.10.0+dtk2604-cp310-cp310-linux_x86_64.whl`,
SHA-256 `c8c9f8793cded49158cbdf7a66042a5fe9d66f55170f5c1e6c9baa519ee88793`.
It has not been published.

The protocol is unchanged: six independent short pairs each use 15 timed rounds,
two independent 512-step pairs each use three timed rounds, one full warmup is
excluded in every process, and vendor/native order alternates. Every forward
is synchronized before and after timing. This yields 90/720 short
prefill/decode samples and 6/3,072 long samples per backend.

Complete-forward medians in milliseconds:

| Pair | Steps | Order | Vendor prefill | Native prefill | Ratio | Vendor decode | Native decode | Ratio | Gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| s01 | 8 | vendor then native | 21.216 | 25.978 | 1.224 | 20.174 | 24.690 | 1.224 | PASS |
| s02 | 8 | native then vendor | 21.037 | 25.754 | 1.224 | 20.056 | 24.613 | 1.227 | PASS |
| s03 | 8 | vendor then native | 21.033 | 26.580 | 1.264 | 20.111 | 25.322 | 1.259 | **FAIL** |
| s04 | 8 | native then vendor | 21.647 | 25.794 | 1.192 | 20.775 | 24.588 | 1.184 | PASS |
| s05 | 8 | vendor then native | 21.122 | 26.635 | 1.261 | 20.153 | 25.387 | 1.260 | **FAIL** |
| s06 | 8 | native then vendor | 21.834 | 26.976 | 1.235 | 20.838 | 25.695 | 1.233 | PASS |
| l01 | 512 | vendor then native | 22.188 | 26.886 | 1.212 | 21.064 | 25.479 | 1.210 | PASS |
| l02 | 512 | native then vendor | 22.584 | 26.305 | 1.165 | 21.232 | 25.268 | 1.190 | PASS |

All eight pairs preserve every prefill position and requested decode step:
1,080 checkpoint comparisons and 190,831,616 logit elements have identical bytes,
with maximum absolute error zero. All 20 paired operator cases also have
identical inputs and output bytes. Warm model execution launches 22 distinct
published FlagGems kernel variants and has no FlagGems/Triton Python calls.
Numerical CPU fallback, boxing and model capture remain forbidden; host-plan
and capture counters are zero. Eight boxing attempts are rejected before
metadata or value changes.

```text
Installed HCU regressions: 216 passed in 44.82s
Linux CPU unit cohort:    791 passed, 4 skipped, 183 warnings in 119.69s (0:01:59)
ruff check:               All checks passed!
ruff format --check:      641 files already formatted
```

Two generator reruns leave artifacts unchanged and all six consistency checks
pass. Wheel preflight passes. A macOS/Python 3.14 comparison without Torch has
the same eight upstream iterator-region failures on base and candidate: base
470 passed/24 skipped, candidate 507 passed/29 skipped, with no new failing test
IDs. Those tests pass in the selected Linux Python 3.10 cohort. NVIDIA, MetaX
and Ascend hardware have not been rerun for shared ABI/compiler changes; older
native-host bundles require rebuilding.

## Earlier qualification: 2026-11-18

Date: 2026-11-18. Base: `eba6b82e95b4ae3cc7f1784c54fd8724596f3845`,
plus the HCU changes in the original measured source snapshot.

This earlier validation uses a non-boxing HCU native-host wheel, independent published
DTK PyTorch, and the complete official Qwen3-0.6B BF16 eager model. The gate
requires elementwise `atol=rtol=1e-3` over every prefill position and requested
decode step, and synchronized complete-forward latency no greater than 1.25
times the baseline for both prefill and decode.

The [acceptance record](../../benchmarks/host_compiler/results/hygon_native_host_acceptance.json)
contains paired timings, artifact and source fingerprints, checkpoint contracts,
precision ablations, test output and raw evidence hashes. The
[earlier validation](../../benchmarks/host_compiler/results/hygon_native_host_validation.json)
and [readiness record](../../benchmarks/host_compiler/results/hygon_native_host_readiness.json)
are historical failures preceding these repairs. Their CPU-fallback diagnostic
timings do not count toward acceptance.

**The 2026-11-18 wheel passed every independent correctness and performance pair.**
Six short-sequence pairs and two 512-step pairs alternate vendor-first and
native-first execution. Every pair passes both prefill and decode latency gates.
Direct byte comparison of the saved tensors verifies all prefill positions and
every requested decode step: 1,080 checkpoint comparisons and 190,831,616 logit
elements are identical, with maximum absolute error zero.

Synchronized complete-forward medians, milliseconds:

| Pair | Steps | Order | Vendor prefill | Native prefill | Vendor decode | Native decode |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| s01 | 8 | Vendor then native | 21.699 | 25.743 | 20.741 | 24.521 |
| s02 | 8 | Native then vendor | 21.168 | 26.005 | 20.218 | 24.691 |
| s03 | 8 | Vendor then native | 21.231 | 26.176 | 20.270 | 24.824 |
| s04 | 8 | Native then vendor | 21.982 | 25.668 | 20.968 | 24.464 |
| s05 | 8 | Vendor then native | 20.768 | 25.626 | 19.883 | 24.410 |
| s06 | 8 | Native then vendor | 21.169 | 26.081 | 20.240 | 24.828 |
| l01 | 512 | Vendor then native | 21.822 | 26.172 | 20.627 | 24.990 |
| l02 | 512 | Native then vendor | 22.278 | 25.925 | 20.394 | 24.676 |

The worst native/vendor prefill/decode ratios across all six short pairs are
**1.234/1.228**. The worst ratios across both long pairs are **1.199/1.212**.
Each short process has 15 timed rounds, giving 90 prefill and 720 decode samples
per backend across the six pairs. Each long process has three timed rounds,
giving six prefill and 3,072 decode samples per backend across both pairs.
Each process excludes an additional complete warmup round. The gate applies
separately to every pair's synchronized complete-forward medians; passing pairs
are not pooled to conceal a failing one.

2026-11-18 artifact: `torch_fl-2.10.0+dtk2604-cp310-cp310-linux_x86_64.whl`, SHA-256
`d8a6115e6c004f1f5a436d5c3e0cb529a890312605bca1cd4292c1fef1c568f7`.
The wheel was installed for validation and has not been published.

Twenty paired operator cases have identical input hashes and bitwise-identical
outputs. Warm submission medians for `add.out`, `mul.out`, `mean.out` and
`cat.out` are 6.68, 6.56, 7.96 and 10.10 microseconds respectively. These are
operator measurements, not individual operator performance-gate claims.

## Short-sequence host investigation

The previous candidate passed some pairs but failed a short repeat at
1.270/1.267. Isolated C++ timers located repeated specialization-key work in
about 1,400 launches per forward. Diagnostic self times, excluding nested launch
work, were:

| Phase | Queries | Serialized lookup, microseconds | Metadata mirror, microseconds |
| --- | ---: | ---: | ---: |
| Prefill | 1,452 | 1,543.032 | 995.941 |
| Decode | 1,366 | 1,323.430 | 888.650 |

The diagnostic clone used the same published numerical kernels and reported
1,452/1,366 metadata-mirror hits with zero serialized lookups in the sampled
prefill/decode forwards. These intrusive timer measurements are excluded from
acceptance; the final production wheel contains no diagnostic timer bindings.

The launcher now creates an immutable metadata mirror only after reproducing
the original serialized key exactly. A coarse index selects a bounded candidate
bucket, then full checks validate the launch device, tensor dtype/device index,
shape, strides, pointer alignment and scalar type/value bits. NaN payloads and
signed zeros remain distinct. Keyword reordering, unsupported containers and
large collision buckets retain the original serialized lookup.

The mirror stores no tensor references or data pointers. Every hit still binds
the current arguments, evaluates the current grid and prehooks, and reads the
current stream. Registration and lookup share the existing mutex. No numerical
kernel, operator route, dtype exception, model graph or output allocation policy
changed in this optimization.

## Environment and execution boundary

The target is Hygon BW1101 144 GB, architecture `gfx938`, wavefront width 64.
The cohort is Python 3.10.12, CPU Torch 2.10.0+cpu, FlagGems 5.4.0,
FlagTree 0.7.0+hcu3.6 / Triton 3.6.0, Cython 3.2.4, Transformers 5.5.0,
and verified DTK 26.04. The vendor distribution is
`2.10.0+das.opt1.dtk2604.2606151054.g98b1a8`, runtime Torch 2.10.0,
HIP 6.3.26113. CPU and vendor Torch Git revisions are checked separately.
Dependencies come from published wheels; no dependency Git source was built.

The candidate records kernels `flaggems` and `vendor`, with boxing, FlagGems C++
and TileOPs disabled. Tensors retain PrivateUse1 storage and dispatch keys.
HCU non-boxing builds reject a PrivateUse1-to-CUDA metadata conversion before
changing the tensor, including attempts through the historical boxing helpers.
The HCU adapter uses the published HIP loader and TDM codec. Backend ABI 2
supplies kernel metadata to descriptor factories, including the TDM warp count.
Ordinary, cooperative and multi-CTA launch descriptions use their published
driver interfaces. Global scratch stays null; supported profile scratch uses
the published allocation rule. The adapter implements no numerical kernel.

Measured native precision entries call original DTK structured implementations
or reduction stubs on PrivateUse1 storage. TensorIterator's physical pointer
descriptors describe HIP memory; TensorImpl device types and dispatch keys
remain PrivateUse1. There is no CUDA ATen redispatch or tensor boxing. The
independent baseline uses the manufacturer's HIP PyTorch in another environment
and process, with `torch_fl` absent.

## Route and host repairs

The generated Hygon configuration and registrations enable measured functional
`add.Tensor`, `mul.Tensor`, `mean.dim` and `cat` routes. Generated adapters for
`add.out`, `mul.out`, `mean.out` and `cat.out` reuse those configured functional
routes, then copy into the caller's output. They preserve storage identity,
resizing, output dtype rules, legal full aliases and overlap rejection.
Integer mean, incompatible explicit mean dtype, invalid alpha and cat aliases
retain ATen's validation behavior.

Supported same-device copies and casts use the compiled published FlagGems copy
host, including contiguous output copies. This removes device-wide waits from
the measured `out` copies. Same-device casts allocate their result directly,
preserving dense input strides or the requested memory format, then pass the
original source strides to the copy host. This avoids a separate contiguous
gather and repeated allocation dispatch. Integral non-boolean `add` alpha values
are valid for every result dtype, so their validation avoids a redundant
`result_type` dispatch; boolean, floating and complex alpha values retain the
original checks. Generated metadata adapters reuse ATen primitives for
`_unsafe_view`, `transpose.int`, `unsqueeze` and `slice.Tensor`, preserving sizes,
strides, storage offsets and aliasing. An incompatible `_unsafe_view` raises
instead of materializing a reshape.

Float/BFloat16 multiply compiles the unchanged published
`_sunrise.ops.mul.mul` host, pinned by source SHA-256. Its guard admits CPU
zero-dimensional real tensors marked as wrapped numbers and requires a real
PrivateUse1 operand. Dtype-constrained lowering removes the unreachable complex
branch only for unchanged root parameters. Ordinary CPU tensors, unconstrained
parameters, complex wrapped scalars and rebound parameters retain their original
behavior. Floating
square uses the existing multiply decomposition through the configured route;
power is not a vendor precision exception.

CPython 3.10/3.11 pointwise rank caches validate dictionary identity and version
under the GIL, including mutations while rank instantiation releases the GIL.
Wrapped scalar reads and supported empty factories avoid redundant dispatcher
traversals, preserving lazy tensor values and device guards. Results remain
fresh allocations; no model tensors or outputs are cached.

Ordinary launch arguments use eight-byte scalar slots instead of clearing
128-byte descriptor slots. The original wide path remains for descriptors;
zero scratch arguments are immutable metadata. Tensor pointers, launch grids
and streams are read from the current call. Under no-grad, an already-contiguous
plain PrivateUse1 tensor can return directly from the compiled contiguous host;
other layouts, symbolic sizes and Python/functional dispatch retain ATen calls.

HCU's PrivateUse1 guard now carries actual stream handles. Creation, exchange,
query, synchronization and events use the runtime registry. Allocations bind
the physical HIP stream; `record_stream` tracks cross-stream storage lifetime
with the original DTK allocator. The non-boxing Hygon default stream is
nonblocking, with explicit synchronization for blocking host transfers.

The strict benchmark binds the DTK library stream once for an eager region:

```python
import torch_fl  # Load HCU device libraries before importing Torch.
import torch

with torch.no_grad(), torch_fl.native_host.execution_scope(0):
    output = model(inputs)
```

The context restores the caller's stream on normal or exceptional exit and
supports nested scopes on another stream. Individual native entries also
establish the correct stream outside it. It does not capture, compile or replay
a model graph and stores no tensors.

## Precision investigation

The original first-layer normalization output matched the vendor, but its BF16
projection diverged. Removing each exception separately from an otherwise
passing short-sequence diagnostic restored the precision failure:

| Removed exception | Passing comparisons | Maximum prefill absolute error |
| --- | ---: | ---: |
| BF16 `mm` | 0 / 9 | 0.7734375 |
| FP32 `mean.dim` | 0 / 9 | 0.71875 |
| FP32 `rsqrt` | 0 / 9 | 0.75 |
| FP32 `_softmax` | 0 / 9 | 0.8515625 |

Four exceptions sufficed for eight decode steps. The 512-step run exposed BF16
`bmm`: 458/513 comparisons passed, with steps 458 through 512 failing.
Input-matched attention operands first differed at layer index 17 for
`(16,1,481) @ (16,481,128)`, maximum error `0.0009765625`. Replacing only BF16
BMM with original DTK restored all 513 comparisons exactly, including every
prefill position. Power remained on FlagGems.

The single policy in `scripts/codegen/native_host_policy.py` therefore lists
BF16 `mm`/`bmm` and FP32 `mean.dim`/`rsqrt`/`_softmax`. Generated configuration
selects a native entry only for a matching dtype and registered slot, for example
`bmm = flaggems; BFloat16:hygon`. Plain FlagGems config/environment overrides
remove those conditions. Registration installs kernels without creating a
separate routing policy. Remove each exception after a published FlagGems cohort
passes paired precision and repeated eager performance checks. This policy does
not claim general Hygon native-operator coverage.

## Verification and reproduction

Build with the pinned published dependencies and verified metadata:

```bash
FLAGOS_ACCELERATOR=hygon FLAGOS_BUILD_NATIVE_HOST=1 \
FLAGOS_BUILD_VENDOR=1 FLAGOS_BUILD_FLAGGEMS=1 \
FLAGOS_BUILD_FLAGGEMS_CPP=0 FLAGOS_BUILD_TILEOPS=0 \
FLAGOS_VENDOR_TORCH_LIB=/path/to/published/vendor/torch/lib \
FLAGOS_SDK_VERSION=26.04 \
FLAGOS_VENDOR_TORCH_VERSION=2.10.0+das.opt1.dtk2604.2606151054.g98b1a8 \
python setup.py bdist_wheel
python scripts/tools/torch-fl-preflight --wheel dist/<wheel>.whl \
  --check-build-env --check-native-host-backend --sdk-version 26.04
```

Run the installed wheel outside the checkout, importing `torch_fl` before Torch.
Keep the separate vendor Torch library directory out of the candidate's
`LD_LIBRARY_PATH`: a second `libmagma.so` caused a shutdown double-free during
investigation. The clean environment maps one copy and exits normally. Finish
running processes before rebuilding bundled libraries.

Use `hygon_operator_benchmark.py` for paired operators and
`qwen_strict_eager.py` for the complete model. The independent vendor run writes
the reference consumed by the candidate:

```bash
python qwen_strict_eager.py --model /path/to/Qwen3-0.6B \
  --device cuda:0 --mode plain --steps 512 --rounds 3 --skip-sampling \
  --output vendor.json
python qwen_strict_eager.py --model /path/to/Qwen3-0.6B \
  --device flagos:0 --mode plain --steps 512 --rounds 3 --skip-sampling \
  --reference vendor.pt --check-logits --output native.json
```

Measurements use physical GPU 1, CPU affinity `8-15`, and one OpenMP/MKL thread.
Runs are sequential after tests/builders finish and the node is checked idle.
Both use 28 layers, 596,049,920 parameters, a 23-token prompt, matching checkpoint
hashes, teacher-forced inputs, eager attention and math settings. Loading, tuning
and sampling are excluded on both sides; every measured forward is synchronized
before and after execution. Native hosts reject CPU fallback. Model compilation
entry points are blocked, requiring zero capture attempts and legacy host-plan
calls.

The acceptance record stores every subprocess command. Six short pairs use
`--steps 8 --rounds 15`; two long pairs use `--steps 512 --rounds 3`. Both orders
run each backend in a fresh process. A native-first pair consumes a previously
verified independent vendor reference; after both processes finish, saved native
logits are compared byte for byte with that pair's fresh vendor output. No extra
main-thread pinning or inference-mode substitution is used.

Hardware checks cover launch ABI, scalar widths, descriptors, retained outputs,
transposed inputs, cache mutation/concurrency, out contracts, metadata aliasing,
stream restoration and two-device operation. Event/lifetime tests release an
input after recording its consumer stream, then allocate a replacement on the
producer while retaining the result, also with a different current device.

```text
Hygon hardware cohort: 217 passed in 44.53s
Hygon unit cohort:     593 passed, 137 warnings in 98.00s
Local unit cohort:     428 passed, 28 skipped, 79 warnings in 5.77s
ruff check:            All checks passed!
ruff format --check:   631 files already formatted
```

Two generator reruns leave artifacts unchanged. Native-vendor, out-adapter,
native-route, configuration and registration consistency checks pass. Wheel
preflight validates the generated compatibility record and HCU backend.
An independently compiled header probe rejects eight boxing attempts across
two devices, two dtypes and contiguous/transposed inputs. Device/storage
metadata, pointers, dispatch keys and values are unchanged after rejection.
New cache regressions cover tensor metadata changes, scalar type distinctions,
NaN payloads, signed zero, keyword order, collision buckets, concurrent
preparation and release of input tensors after specialization.

## Scope and limitations

The 2026-11-18 acceptance record reports `passed` for the fixed six-short/two-long protocol
and retains the previous failed acceptance's artifact hash and measurements.
Earlier main-thread pinning and inference-mode diagnostics did not resolve the
gap. Warm Python-call tracing records no FlagGems/Triton Python calls. C++ timer
results above explain the host optimization and are excluded from gate timings.
The repeated median gate does not claim every individual sample stays below
1.25 times vendor latency.

Correctness evidence covers the recorded full Qwen3-0.6B BF16 eager workload and
paired operator shapes, without establishing other models, dtypes or training.
Individual operator submission latency may exceed 1.25 times vendor latency;
the performance gate concerns synchronized complete-model forwards.

TDM, cooperative and multi-CTA descriptors were checked for construction and
rejection; no numerical workload for these features was measured. NVIDIA, MetaX and Ascend hardware were not rerun for shared ABI/compiler changes.
Older native-host bundles require rebuilding.
