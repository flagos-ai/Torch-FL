# Build-time FlagGems hosts

Status: experimental; the eager performance target has not been met. Acceptance
requires synchronized warm prefill and decode latency at most 30 ms against the
approximately 24 ms vendor workload, with the original model remaining eager.
The approximately 26 ms measurements below use Dynamo model graph capture and
cannot count toward that requirement. The measurements are also separate from
the capture-based operator prototype merged in PR #824.

The installed FlagGems wheel is the source of both host semantics and numerical
kernels. The build generates native host code from its source, including shape
branches, loops, allocations, views and launch schedules. Runtime specialization
may compile the original FlagGems Triton kernels, but may not compile C/C++ or
generate a shape-specific host shared library. Neither handwritten Triton nor
Inductor numerical kernels are permitted.

## Compilation boundary

The wheel builder discovers the installed registry automatically. Cython provides
compiled Python object semantics for the complete discovered host closure. A
second source compiler lowers supported control flow to typed C++ callbacks:
AST transformation, build-time typed IR, alias analysis, and C++ emission. Typed
callbacks enter before the Python operator bridge. Unsupported language constructs
retain the compiled original host; adding an operator requires no handwritten
wrapper or list of benchmark-specific operators.

Branches, loops, current shapes, allocations and views remain in the generated
program. Read-only shape lists use stack storage only when alias analysis finds
no writes. Mutable lists and dictionaries retain shared identity, including across
compiled object helpers and their return values. Kernel arguments and descriptor
prehooks are evaluated on every call. Immutable version/API metadata is folded
only by a restricted AST interpreter, with the dependency source hashes checked
again at load time. Primitive module initializers that are assigned once in the
pinned source bind from the running module when the hosts load, rather than from
the build machine's device. Functions with mutable module state keep compiled
object semantics. This does not promise arbitrary external monkeypatching of
otherwise immutable module attributes.

Kernel definitions remain the original JIT functions from the pinned wheel.
The build identifies host functions and their transitive helpers separately
from device functions. Generated host references bind to compiled helpers and
a common launcher. No per-operator native implementation is added.

PointwiseDynamic uses a shared rank-independent host adapter. Its original
KernelGenerator specializes device code for a new rank. No Python host wrapper
is generated for that rank. The source-derived preparation code retains the
wheel's schema, promotion, broadcasting and layout decisions.

The CUDA launcher is compiled once. Signature, scalar widths, descriptor layout,
shared memory, scratch requirements and cluster launch configuration are data
provided by kernel compilation. The installed FlagTree wheel's CUDA utility
source is also compiled at build time. Runtime native compiler invocations from
the host/kernel path are rejected, and each launch must refer to an admitted
FlagGems JIT source hash. Unrelated application subprocesses are unaffected.

The native launch ABI takes borrowed argument views. It reads current tensor
addresses and streams, reevaluates grids and hooks, and allocates fresh outputs.
Caches contain original device specializations and the original pointwise
generator's rank definitions. They contain no output tensors, captured launch
sequences, or shape-specific host programs. On CPython 3.12, dictionary watchers
invalidate rank definitions when configuration or overload definitions change;
without watcher support the original instantiation path runs on every call.

## Optional Dynamo model-graph comparison

`torch_fl.native_host.compile(model)` additionally compiles the model's Python
host graph through Dynamo and TorchScript's ATen executor. It returns the same
module with a compiled `forward`, so methods such as generation use that forward.
Calls require `torch.no_grad()`. Dynamic-shape guards remain active and new model
control-flow paths may compile another graph. This produces host bytecode and
does not compile another native host shared library.

Numerical TorchScript fusers are disabled for this explicit process mode. Source
and executed graphs are checked for fusion nodes, and the ordinary ATen dispatch
still selects FlagGems. No Inductor backend is called and no attention or other
numerical operator is substituted. The measured Qwen traversal needs three guarded
model graphs across 512 decode steps, rather than a graph per token. This model
mode is distinct from ordinary eager execution, which retains additional Python
overhead; benchmark reports record the execution mode explicitly. It changes the
model execution route and is retained only as a diagnostic comparison, not as
evidence that the source-generated operator hosts meet the eager target.

Use `torch.Stream(device="flagos:0")` and its context manager for nondefault
streams on the CPU-Torch NVIDIA build. The native runtime, allocator, kernel
launcher and CUDA shim's raw-stream lookup use the same current-stream registry.
Stream selection uses the generic `torch.Stream` API; the older CUDA-shaped
Python stream shims are not the stream-selection API validated for this mode.

## Build and loading

`FLAGOS_BUILD_NATIVE_HOST=1` adds generation after torch-fl's native extension is
built. The default discovery reads FlagGems' registered functions and their
source dependencies; it does not enumerate tensor shapes. `--functions` on the
development builder restricts probes only. The wheel records `NATIVE_HOSTS` in
its build configuration and packages the generated extensions, source manifest,
and common launcher under `torch_fl.native_host._generated`.

Initialization installs the prebuilt launcher before importing FlagGems, then
binds the generated host functions after the normal backend setup. The build
requires a working NVIDIA CPU-Torch environment, the pinned published dependency
wheels, Cython 3.2.4, and the CUDA toolkit. Cython is an optional build requirement
selected by `FLAGOS_BUILD_NATIVE_HOST`, not a dependency of ordinary platform
builds. Install it explicitly when building without isolation. No FlagGems or
FlagTree Git checkout is used. The manifest binds the hosts to their
source/dependency cohort; changed
source or an incompatible driver ABI requires rebuilding the wheel.

The feature is currently validated for inference on H100 with Torch 2.10.0+cpu,
FlagGems 5.4.0 and FlagTree 0.7.0. Boxing, the upstream C++ wrapper slot and
TileOPs are disabled for this build. A missing numerical route is an error.
This work does not add new vendor-native fallback operators or claim coverage
for other accelerator backends.

## Registry coverage and generalization

[Project 14](https://github.com/orgs/flagos-ai/projects/14) and
[issue #822](https://github.com/flagos-ai/Torch-FL/issues/822) track full registry
coverage, typed eager entries, shared host-language semantics, portable launching
and per-vendor validation. Domestic accelerators must reach at least 80% of their
manufacturer's native PyTorch performance on the same physical card, model and
settings. For fixed work this means latency at most 1.25 times that baseline;
for throughput it means at least 0.80 times the baseline. Record the actual
vendor PyTorch distribution/version, SDK, precision, attention implementation,
inputs, batch/context/decode lengths and CPU placement. NVIDIA's existing Qwen
workload retains its explicit 30 ms eager limit. Graph-mode comparisons do not
qualify for either gate.

Each typed build writes `coverage.json` beside `diagnostics.json` and the typed
manifest. The versioned report contains every installed `_FULL_CONFIG` row,
including schema aliases and duplicate registrations. It records the ATen
operator/overload, registered and resolved host identities, inspected source
hash, candidate signatures, exclusion reason and lowering outcomes. The cohort
records the vendor, Torch/FlagGems/FlagTree/Triton versions, selected host names
and a hash of the source-host manifest. Selection refers to the host bundle; it
does not assert that an excluded operator is unsupported by the device.

Rows have these states:

- `not_selected`: the callable is outside this host bundle, including filtered
  development builds. The full registry remains the denominator.
- `excluded`: no candidate signature could be generated, with an explicit reason
  such as missing schema/host, unsupported return/type/default, variadic parameters
  or unavailable source.
- `pending`: lowering has not completed for every candidate signature.
- `lowered`: all generated signatures for the row passed typed host lowering.
- `partial`: some signatures lowered and others retained compiled object semantics.
- `compiler_fallback`: every candidate retained compiled object semantics; detailed
  errors remain in `diagnostics.json`.

Summary counts distinguish registry rows, unique operator names and unique host
signatures. Aliases retain separate rows but share a compilation result. Missing,
duplicate or unknown signature results prevent a completed report. The initial
report is written before lowering, and even a build with zero successful roots
retains diagnostics. `native_artifact_built` becomes true only after the C++
extension builds; emit-only reports keep it false. Manifest summaries carry the
same reconciled counts. None of these states establishes numerical correctness,
all-input coverage, warm native execution or performance. Genuine FlagGems gaps
and compiler limitations must remain separate when selecting vendor-native routes.

## Schema-typed eager entries

The operator generator emits one static slot for each existing generated
FlagGems ATen overload. Its catalog records the resolved host, faithful C++
argument types, positional/keyword mapping and result shape. An ABI hash covers
that contract. The native-host builder binds source-derived variants to these
slots and emits typed C++ callbacks. Import validates the host identity and ABI
before publishing each callback address. Replacing an already registered address
is rejected; repeated registration of the same address is idempotent.

An admitted warm call loads its slot directly, checks the device and argument
types, then invokes the source-generated host. It does not construct an IValue
argument array or search for a host by name. The original generic call is inside
a lazy fallback lambda. A variant can decline before execution; an exception
after entering the host propagates without retrying side effects. Current-device
and stream behavior, tensor subclass/functionalization guards, and the existing
GradMode fallback are retained. This remains inference-only native admission;
training semantics are tracked in #829.

The first binding slice supports Tensor results and fixed Tensor tuples, exact
bool/int/float scalar admission, optional values, integer/Tensor lists, keyword
defaults and existing out/in-place adapters. Tuple handles retain ownership and
order. It preserves all 688 existing generated registrations and their routes.
The measured NVIDIA build binds 400 overload entries, including 13 with multiple
outputs, from 997 lowered signatures. These are generation counts, not proof
that every registered input combination has been executed.

`native_source_typed_stats()` reports registered slots and successful calls;
`native_source_typed_hits(op)` exposes a per-overload counter for audits outside
timing loops. The manifest records `typed_entry_abi`, `typed_entries` and
`typed_rejected`. General scalar/list/optional **result** bindings, the remaining
handwritten dispatcher bridges and unlowered host variants remain work under
#828 and #829. They do not acquire handwritten host implementations or get
silently rerouted to vendor kernels.

For example, the pinned `var_mean` source assigns `1.0` when `correction` is
`None`. TorchScript rejects its Optional[int] variant at that assignment, while
Optional[float] lowers. Integer arguments retain the original FlagGems path;
admission never coerces them to float to hide the compiler gap. Separately,
schema-required parameters keep their schema type when a Python helper has a
`None` default; generated facades pass all arguments explicitly.

## Generated source-call bridges

Some ATen operators, including copy and view plumbing, are registered manually
because they coordinate device transfers, aliases or storage. The operator
generator now examines those schemas against the same installed FlagGems
registry and argument-admission rules as ordinary generated entries. Eligible
single-Tensor source calls are emitted into `flaggems_source_bridges.h/.cc`.
These are callable adapters, not new ATen registrations or routing entries.

The NVIDIA device-copy helper calls the generated `copy_` adapter. Its existing
device, dtype, conjugate/negative, overlap, alias, empty-input and recursion
checks remain in the helper. The adapter borrows input handles and calls the
original FlagGems host once; it does not copy the returned tensor back into the
destination, rewrite its device metadata or re-enter the dispatcher. Ordinary
wrapper code still supplies the original return identity and version increments.
Unsupported typed variants and GradMode-enabled calls retain the original FlagGems path.

Host signature discovery also respects ATen's positional/keyword boundary.
A Python helper's extra defaulted positional parameter must not consume an
unrelated schema keyword: for example, the default `step=1` on the helper for
`arange.start` remains an integer default rather than acquiring dtype's optional
type. A positional-or-keyword Python parameter can still bind an ATen keyword
by name, preserving scalar variants such as `alpha`. This signature repair is
separate from host-language lowering: `arange` device defaults and scalar_tensor
optional comparisons still have compiler gaps under #829.

The build binds 404 typed entries from 1,008 lowered signatures, with 553
signature fallbacks. The two emitted bridge functions are `copy_` and
`_reshape_alias`; only the existing copy helper is connected in this change.
The schema repair also admits `round` and `round_`, whose extra `decimals=0`
helper default previously had no schema argument. The default-argument paths
are checked against the original FlagGems implementation in three floating
dtypes, including ties, infinities, NaNs and in-place identity. The pinned
registry does not expose `aten::round.decimals`; this change does not add that
route or claim its coverage.

The fresh paired H100 run retains the complete workload described below:

| Synchronized complete forward | Prior typed-entry wheel | Source bridges | Vendor PyTorch |
|---|---:|---:|---:|
| Prefill median | 33.24 ms | 32.15 ms | 25.25 ms |
| Decode median | 31.14 ms | 30.31 ms | 23.71 ms |
| Decode p99 | 32.64 ms | 31.98 ms | 24.57 ms |

The medians are lower in this run, but both still exceed 30 ms. These are
whole-wheel measurements with process-level variation, not an isolated ABI
speedup. In an interleaved A-B-B-A microbenchmark, cast submission medians range
from 8.61–8.65 us before and 8.42–8.95 us after, so no reliable cast latency
improvement is established. Every measured cast in the new wheel hits the typed
bridge; the old wheel records zero typed cast hits. All 513 logits match the
original FlagGems oracle bitwise, model-capture calls and legacy plans are zero,
and the separate trace still contains 3,814 original FlagGems launches with no
unmatched kernels. The [source-bridge evidence](../../benchmarks/host_compiler/results/qwen3_source_bridges_h100.json)
retains the raw round timings, tail distributions, cohort and artifact hashes.

## Typed-entry eager measurement: target not met

The paired H100 rerun uses the same complete Qwen3-0.6B model, 60-token prompt,
512 teacher-forced decode inputs, BF16 eager attention and CPU affinity. Each
environment runs sequentially on the same idle physical GPU, with normal
allocations, three timed traversals and synchronization around every forward.
Token selection, comparisons and profiling remain outside the timed passes.

| Synchronized complete forward | Prior native-host wheel | Typed entries | Vendor PyTorch |
|---|---:|---:|---:|
| Prefill median | 34.87 ms | 32.66 ms | 24.58 ms |
| Decode median | 32.86 ms | 31.12 ms | 23.14 ms |
| Decode p99 | 33.96 ms | 32.26 ms | 24.17 ms |

Prefill and decode medians decrease by 6.35% and 5.31%, respectively. Both still
exceed 30 ms. This compares complete development wheels, including the generic
signature/default fixes; it does not isolate the entry ABI's contribution.
The [typed-entry evidence](../../benchmarks/host_compiler/results/qwen3_typed_entries_h100.json)
records distributions, cohort, artifact hashes and per-operator measurements.
Representative CPU submission medians decrease from 26.29 to 23.22 us for mm,
8.79 to 7.72 us for SiLU and 7.95 to 6.82 us for softmax. Cast, arange and
scalar_tensor still report zero typed-entry hits in this workload.

All 513 logits remain bitwise equal to the original FlagGems oracle. The
independent capture guard reports zero calls to ten model compilation/tracing
entry points; no legacy host plans are present. A separate two-forward trace
contains 3,814 launches from 25 original FlagGems kernel names, with no unmatched
names. Host artifact hashes remain unchanged, and the warm FlagGems/Triton
Python-frame audit is empty. Vendor timing uses native Torch 2.10.0+cu130 with
backend autoload disabled and identical teacher-forced inputs; numerical
differences between vendor Torch and FlagGems are recorded separately.

Reproduce with `benchmarks/host_compiler/qwen_strict_eager.py`, using the normal
`qwen_eager.py` arguments, `--mode plain`, `--steps 512`, `--rounds 3`, the same
prompt/reference and each environment's device. The wrapper rejects model
capture and legacy host-plan modes. It adds no changes to model code.

## Earlier eager baseline: target not met

The same installed wheel was rerun with the original model forward and no
model-host compilation. A separate validation wrapper rejected calls to
`torch.compile`, Dynamo optimize/export entry points, and TorchScript model
script/trace entry points. None was called. The ordinary benchmark used the
same 60-token prompt, 512 teacher-forced decode inputs, BF16 eager attention,
H100 GPU and CPU affinity as the graph comparison below.

| Synchronized complete eager forward | Latency |
|---|---:|
| Prefill median | 35.65 ms |
| Decode median | 33.54 ms |
| Decode p99 | 34.94 ms |

The three prefill samples were 35.65, 36.43 and 35.19 ms. The 1,536 decode
samples came from three complete traversals. All 513 logits matched the
unmodified FlagGems oracle bitwise, all 773 host libraries remained unchanged,
and the warm FlagGems/Triton Python-frame audit was empty. There were no legacy
host plans. The separate two-forward trace recorded 3,814 original FlagGems
kernel launches, with no unmatched kernel names.

This earlier baseline is eligible for the requested eager comparison, and both
prefill and decode exceed 30 ms. The [eager evidence record](../../benchmarks/host_compiler/results/qwen3_eager_source_hosts_h100.json)
retains the capture guards, timings and artifact hashes. The graph-mode result
below must not be substituted for it.

## Typed hosts with Dynamo: comparison measurements

The final measurements use the complete official Qwen3-0.6B model, BF16, eager
attention, 28 layers and 596,049,920 parameters on an H100 80GB. The model host is
enabled explicitly. The vendor baseline uses Torch 2.10.0+cu130, the same model,
input tokens, attention implementation, GPU and eight pinned CPU cores.

Each timed call includes the full model forward, KV-cache updates, allocations,
dispatch and a device synchronization before the clock stops. Tokenization,
token selection and correctness comparisons are outside these timings on both
sides. After an untimed traversal, three complete 512-step traversals provide
three prefill and 1,536 decode samples. Profiling is a separate, untimed pass.

| Synchronized complete forward | Vendor Torch | FlagGems model host |
|---|---:|---:|
| 60-token prefill median | 24.39 ms | 26.71 ms |
| 512-step decode median | 22.78 ms | 25.98 ms |
| Decode mean | 22.87 ms | 26.02 ms |
| Decode p99 | 25.05 ms | 27.52 ms |
| Maximum observed decode sample | 37.33 ms | 30.65 ms |

All three graph-mode prefill samples are below 30 ms. These measurements do not
satisfy eager acceptance because the model was captured by Dynamo. They also do
not guarantee that every individual call finishes within 30 ms. The maximum
samples above are retained rather than discarded.

The 60-token prompt and 512 decode steps produce 513 logits tensors, all bitwise
equal to the unmodified FlagGems oracle. The 23-token, eight-step check also
matches all nine logits bitwise. The benchmark uses teacher-forced oracle tokens
for both correctness and timing; it does not present those predictions as an
autoregressive text-generation quality result.

The installed wheel contains 773 host libraries, whose set and SHA-256 hashes
remain unchanged. It registers 521 typed operator entry points from 959 compiled
signatures; 450 unsupported signatures retain compiled object semantics. Those
counts describe generation coverage, not correctness testing of every operator.
The 512-step traversal uses three guarded model graphs, no legacy host plans,
and no warm FlagGems/Triton Python frames. A separate two-forward GPU trace has
3,806 kernel launches and 25 distinct kernel names, all matched to admitted
original FlagGems source or its original pointwise generator. CPU numerical
fallback raises in this mode; boxing, the upstream C++ wrapper slot and TileOPs
are disabled.

Startup remains expensive: the measured process imports torch-fl in about five
seconds and its first prefill takes several seconds, including model-host graph
compilation and original kernel specialization. Its device cache was already
populated by earlier validation, so this is not a fresh-cache cold-start claim.
The generated wheel is a development artifact, not a release artifact.

The [model-host evidence record](../../benchmarks/host_compiler/results/qwen3_model_hosts_h100.json)
retains the distributions, source provenance, checkpoint identity, artifact
hashes and raw-report hashes. The historical Cython-only measurements follow.

## Original Cython-only evidence

This section records the original PR baseline, before typed C++ callbacks and
the explicit model host. Its timings do not satisfy the 30 ms requirement.

The source-host probe was built before introducing the held-out prompt. It used
63 precompiled host extensions for the Qwen call closure. The later full
registry build packaged 772 native extensions and 1032 host entry points; those
counts are generation coverage, not a correctness claim for every operator.

The complete official Qwen3-0.6B model has 28 layers and 596,049,920 parameters.
All measurements below use BF16, eager attention, normal allocations and current
streams; there are no CUDA graphs, output-buffer replay or layer truncation.

| Check | Result |
|---|---|
| Initial 23-token prompt, 8 decode steps | All 9 logits bitwise equal to FlagGems |
| Held-out 60-token prompt, 512 decode steps | All 513 logits and all input tokens bitwise equal to FlagGems |
| Frozen host artifact set | All 63 hashes unchanged |
| Held-out runtime cache | 3222 kernel binaries; no host shared libraries |
| Warm FlagGems/Triton Python calls | Zero |
| CPU fallback | Zero |
| Host loading for held-out probe | 59.7 ms |
| First held-out prefill, fresh kernel cache | 17.92 s, including kernel JIT/tuning |

The cold prefill is not the cost of all 512 unseen lengths. Their first traversal
also specializes many copy kernels. Stack samples during that traversal were in
FlagTree's Triton IR compiler. This remaining kernel specialization cost is real
and is not described as a fast cold start.

The installed full-registry wheel repeated the 60-token, 512-step run: all 513
logits were bitwise equal to FlagGems, with no warm FlagGems/Triton Python frames
or CPU fallback. Its import took 4.83 seconds; a separate import smoke took
6.02 seconds. This startup cost is considerably larger than the selective probe.

| CPU submission median | Installed native-host wheel | Vendor Torch | Ratio |
|---|---:|---:|---:|
| 60-token prefill | 172.36 ms | 25.91 ms | 6.65x |
| 512-step decode | 167.35 ms | 24.04 ms | 6.96x |

These timings include host control flow, allocation, specialization selection,
argument packing and output ownership. The warm run issued only ptxas version
queries, and its 513 logits remained bitwise equal to FlagGems. Comparisons with
vendor Torch remain separate: upstream FlagGems BF16 differences are not hidden
by changing tolerances.

The [portable evidence record](../../benchmarks/host_compiler/results/qwen3_source_hosts_h100.json)
records the checkpoint, dependency cohort, artifact hashes, kernel provenance
and benchmark results. The installed wheel also passes 11 targeted integration
cases, including direct JIT warmup/return semantics and malformed nested launch
arguments. The 1032 generated entries have not all been correctness-tested.

## Validation gates

- Freeze the generator and installed dependency versions before introducing
  held-out shapes, strides, scalar values and decode lengths.
- Record all host binaries and compiler invocations. New inputs may add kernel
  binaries, but must leave the host binary set and hashes unchanged.
- Audit every numerical launch back to its FlagGems JIT source. Fail on
  handwritten or Inductor-generated substitute kernels.
- Profile warm execution for Python host frames, including transitive helpers
  and pointwise wrappers. A compiled outer function alone is insufficient.
- Validate outputs, mutation, view aliases, retained outputs, empty inputs and
  nondefault streams. Autotuning must not corrupt user inputs.
- Run the complete official Qwen3-0.6B model with held-out prefill shapes and
  long decode. Include allocation, specialization lookup, argument packing and
  output ownership in CPU submission time. Compare against the same vendor
  baseline and report host loading, kernel JIT/tuning and warm time separately.
- Require synchronized complete-model prefill and decode latency at most 30 ms
  on the approximately 24 ms vendor workload. Submission time alone, a favorable
  individual run, or a relaxed numerical tolerance cannot satisfy this gate.
- Keep the original model eager. Do not use Dynamo capture, model tracing,
  runtime model-graph compilation or replay to satisfy the latency gate.
  Build-time typed IR of individual FlagGems host source remains allowed.
- Compare actual GPU trace kernel names against admitted FlagGems specializations,
  including kernels emitted by its original pointwise generator. Keep the trace
  outside the timed measurement.
- Require the model's numerical operators to use FlagGems. Any missing route is
  an explicit failure; boxing, vendor Torch redispatch and CPU fallback cannot
  be counted as coverage. The general routing design may use an actual vendor
  native API for an operator absent from FlagGems, as previously specified.

The implementation stays opt-in until all gates have measured evidence.
