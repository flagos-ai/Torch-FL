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

## Current eager baseline: target not met

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

This is the relevant baseline for the requested eager solution, and both
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
