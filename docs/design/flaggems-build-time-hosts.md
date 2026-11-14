# Build-time FlagGems hosts

Status: opt-in implementation. The measurements below are separate from the
capture-based prototype merged in PR #824.

The installed FlagGems wheel is the source of both host semantics and numerical
kernels. The build generates native host code from its source, including shape
branches, loops, allocations, views and launch schedules. Runtime specialization
may compile the original FlagGems Triton kernels, but may not compile C/C++ or
generate a shape-specific host shared library. Neither handwritten Triton nor
Inductor numerical kernels are permitted.

## Compilation boundary

The implementation uses Cython source compilation rather than execution tracing.
Python control flow remains explicit, including branches never taken during the
build. This retains Python object semantics and the GIL. It is substantially
slower than the path-specialized C++ prototype in PR #824, but the measured CPU
submission time remains within one order of magnitude of the vendor baseline.

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
wheels, Cython, and the CUDA toolkit. No FlagGems or FlagTree Git checkout is
used. The manifest binds the hosts to their source/dependency cohort; changed
source or an incompatible driver ABI requires rebuilding the wheel.

The feature is currently validated for inference on H100 with Torch 2.10.0+cpu,
FlagGems 5.4.0 and FlagTree 0.7.0. Boxing, the upstream C++ wrapper slot and
TileOPs are disabled for this build. A missing numerical route is an error.
This work does not add new vendor-native fallback operators or claim coverage
for other accelerator backends.

## Measured source-host evidence

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
- Require the model's numerical operators to use FlagGems. Any missing route is
  an explicit failure; boxing, vendor Torch redispatch and CPU fallback cannot
  be counted as coverage. The general routing design may use an actual vendor
  native API for an operator absent from FlagGems, as previously specified.

The implementation stays opt-in until all gates have measured evidence.
