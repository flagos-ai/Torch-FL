# Native host programs for eager FlagGems inference

## Purpose and scope

Torch-FL's FlagGems route enters Python for allocation, metadata calculations,
heuristics and Triton launch preparation on every eager operator call. These
costs dominate small inference operators. Upstream C++ wrappers cover only part
of FlagGems and require individual implementations.

This experimental implementation generates native host programs from the
installed FlagGems callable through one shared capture and lowering pipeline.
Users keep their ordinary eager model; they do not call `torch.compile`.
Programs compute allocation sizes and launch geometry from current inputs,
within generated guard domains. No user tensor contents, fixed device pointers,
output pools or streams are retained for replay.

The validated scope is NVIDIA inference with CPU PyTorch 2.10.0, FlagGems 5.4.0
and FlagTree 0.7.0 on H100. The API is opt-in and disabled by default.
FlagGems remains the numerical provider. Unsupported capture uses the original
FlagGems Python path; it does not lower numerical ATen operations to replacement
Inductor kernels. Existing routing for operators outside FlagGems is unchanged.
This change adds no general NVIDIA vendor-native fallback library.

## Architecture

```text
Ordinary eager operator
  -> native specialization-key lookup and tensor-metadata guard
       hit  -> allocate current outputs and launch original FlagGems kernels
       miss -> run the original FlagGems callable
                 -> record metadata and scalar specialization constants
                 -> isolated worker with fresh dummy inputs
                 -> symbolic make_fx capture of the real callable
                 -> shared Inductor user-Triton C++ wrapper generation
                 -> native ABI adapter, guards and numerical/alias validation
                 -> admit the generated library for later calls
```

The C++ lookup precedes GIL acquisition in the ordinary positional, keyword and
multiple-output callers. Keys preserve function identity, scalar values and
types, argument-list structure, keyword names and output arity. Tensor sizes,
strides, offsets, dtype, device, alignment and math bits are checked by native
guards. Scalars and genuine Triton constexpr domains remain specialization
conditions. A single generated multiplication program was tested across four
different shapes; the mechanism is not limited to a concrete invocation.

PrivateUse1 tensors reach the generated entry through the existing AOTI tensor
handle ABI. Entries use the current Torch-FL device and stream on each call and
return normally owned tensors. No DeviceBoxingGuard or vendor libtorch is used.
Views retain shared storage and version counters with fresh TensorImpl identities
where required. Generated entries take ownership of input handles; an entry
failure is propagated without retrying after possible device submission.

### Shared capture adapters

The worker handles LibEntry and direct JIT launches, nested heuristics and
autotuners, default arguments, and metadata-only custom JIT `run` hooks. Autotuning
touches only worker-owned dummy buffers. The selected configuration specializes
a kernel; it does not record a user allocation or stream.

Common adapters provide:

- Symbolic metadata and alignment queries for FakeTensor and StridedBuffer.
- Clone, cast and contiguous copies through the installed FlagGems copy kernel.
- Runtime scalar ABI preservation in PyTorch 2.10's Triton mutation analysis.
- Stable TMA descriptors using the compiled FlagTree kernel's physical tile,
  element type, swizzle, global-shape and stride ABI.
- Original user-kernel source serialization, including referenced modules.
- Native guards with symbolic range checks before derived arithmetic and CSE.

All modifications to PyTorch/Triton compiler internals live in the isolated
worker process. The inference process does not import those adapters on a hit.
The worker validates generated values, input mutation, output aliasing and tensor
identity against the original callable before writing an artifact manifest.

### Admission and resource policy

Numerical ATen residuals are rejected during capture. Generated kernel launch
names must match the captured original FlagGems kernels. Admission repeats that
check, rejects Inductor fused-kernel markers in the actual library bytes and
external `aoti_torch_copy_` calls, and verifies the library checksum when present.
New manifests record the generated source checksum separately from the build
fingerprint, which also includes the owned headers and PyTorch version.
These checks catch accidental provider substitution; they are not a sandbox for
untrusted native libraries. Use only caches created by your own installation.

One stale exploratory matmul artifact containing two Inductor copy kernels was
detected during the audit, retired and regenerated through the corrected TMA
lowering. All results below were rerun after that correction. The final audit
covered 188 programs and their actual shared libraries.

Admission is bounded to 256 resident programs and 512 MiB of wrapper-library
files; these limits do not measure total host or device memory. Up to 512
distinct compilation requests are attempted per process, independently of up to
512 metadata-only collection requests. Deduplication includes the selected cache
directory, so collection can be followed by compilation in the same process.
Repeated enablement reuses the two runtime library handles. Modules remain resident
until process exit so disabling the feature cannot unload code in use. There is
no eviction. Capture, compilation or admission failure preserves the already
computed FlagGems result. Failure records remain beside the artifact, with
warnings for observer failures. Regenerate caches after dependency or compiler
changes; cross-version cache reuse is outside the validated contract.

## Supporting device-route fixes

The NVIDIA default configuration now routes embedding to FlagGems. The generated
forward adapter passes `sparse=False`: sparsity selects ATen's backward gradient
representation and does not change forward gather values. The original schema
argument remains above the backend for autograd. The forward cases cover FP32,
FP16 and BF16 with both sparse flags.

Supported same-device strided copies and dtype conversions now use the FlagGems
copy kernel before the existing host-roundtrip implementation. Unsupported
dtypes, complex/math-bit views, aliases, large tensors and cross-device copies
retain the existing path. Overlapping destinations are rejected before launch.
This integration is gated to NVIDIA builds with FlagGems enabled.

## Full-model measurements

The complete official Qwen3-0.6B checkpoint was evaluated: 28 layers,
596,049,920 parameters, BF16, eager attention, a 23-token chat prompt and eight
sequential KV-cache decode steps. Both environments used the same checkpoint,
prompt and attention implementation. Checkpoint revision, file checksum, raw
timing samples and numerical comparisons are recorded in
[the benchmark JSON](../../benchmarks/host_compiler/results/qwen3_0_6b_h100.json).

The candidate used CPU Torch 2.10.0, FlagGems 5.4.0, FlagTree 0.7.0,
Transformers 5.17.0, SDK 13.1, and `KERNELS = ('flaggems', 'vendor')`.
Boxing, FlagGems C++ wrappers and TileOPs were disabled. The separate vendor
reference used Torch 2.10.0+cu130. The GPU was checked idle before measurement.

Median warm timings in milliseconds:

| Execution | Prefill CPU submission | Decode CPU submission | Prefill completed | Decode completed |
| --- | ---: | ---: | ---: | ---: |
| Vendor Torch reference | 24.426 | 22.956 | 24.459 | 23.036 |
| FlagGems Python hosts, with device-route fixes | 211.514 | 203.916 | 211.529 | 203.974 |
| Generated native hosts | **28.465** | **27.152** | **28.500** | **27.229** |

Three prefill repetitions and 24 decode steps were measured after an unmeasured
warm repetition. Submission ends before synchronization; completed latency
includes it. Normal eager dispatch, guards, allocations and output lifetimes are
included. Native submission is 1.17x/1.18x the vendor reference and 7.43x/7.51x
faster than the corrected Python-host path. These are single-host measurements,
not performance guarantees across all shapes or platforms.

The earlier uncorrected Torch-FL baseline completed prefill/decode in about
398.15/380.05 ms. Its CPU embedding fallback and device-copy round trips must not
be attributed entirely to Python wrapper overhead. The native-host comparison
above uses the corrected Python path to isolate the remaining host cost.

All nine last-token logit tensors matched the original FlagGems path byte for
byte. A separate autoregressive sample produced the same tokens. Teacher-forced
predictions are reported separately. This does not establish numerical identity
with vendor Torch: the existing FlagGems BF16 path had maximum logit differences
of 0.375–0.625 and one teacher-forced next-token difference. The first layer
matched through attention and the MLP gate/up projections; the first difference
was the MLP down projection (maximum absolute difference 0.001953125).

The final model log contained no CPU-fallback messages. The process admitted
188 programs, accounting for 10,943,784 wrapper-library bytes, and recorded
108,236 native hits across validation, warmup, measurement and profiling.
A profiled prefill plus one decode retained ten arange and two scalar-tensor
factory launches through FlagGems Python. Other measured model wrappers used
native hosts.

Representative eager CPU submission, median microseconds over seven batches of
200 calls after 30 warmups:

| Operator / shape | Torch-FL | Vendor | Ratio |
| --- | ---: | ---: | ---: |
| BF16 mm `(23,1024) @ (1024,3072)` | 9.457 | 12.957 | 0.73x |
| BF16-to-FP32 cast `(1,1,1024)` | 8.758 | 8.209 | 1.07x |
| BF16 silu `(1,23,3072)` | 7.721 | 6.997 | 1.10x |
| FP32 softmax `(1,16,1,24)` | 7.925 | 6.692 | 1.18x |
| int64 arange, 24 elements, Python factory | 23.001 | 8.056 | 2.86x |
| FP32 scalar tensor, Python factory | 28.240 | 7.658 | 3.69x |

Each numerical case recorded 1,430 native hits, including warmup. The factories
recorded zero. Remaining Python factories also met the same-order-of-magnitude
submission target in this workload.

## Usage and reproduction

```python
import torch
import torch_fl
from torch_fl import host_compiler

host_compiler.enable("host-program-cache", mode="compile")
with torch.no_grad():
    output = model(inputs)
host_compiler.disable()
```

`mode="collect"` records missed-call metadata without compiling. `mode="load"`
loads admitted programs and prohibits further compilation. Both preserve the
original FlagGems route on a miss.

Use an installed NVIDIA CPU-Torch wheel, the dependency versions above, a C++
compiler and the matching NVIDIA toolkit. The tested build ran
`python setup.py bdist_wheel` with `FLAGOS_BUILD_BOXING=0`,
`FLAGOS_BUILD_FLAGGEMS_CPP=0`, `FLAGOS_BUILD_TILEOPS=0`, and the verified
`FLAGOS_SDK_VERSION=13.1`. Wheel preflight with `--check-build-env --sdk-version
13.1` passed before installation. The existing platform table still labels
this experimental wheel `+cuda13.3`; the manifest records the actual SDK 13.1.
No wheel was published. Unset the build-only SDK variable for runtime.

Run from outside the checkout so the installed native extension is imported.
Set `REPO_DIR` and `MODEL_DIR` to your checkout and downloaded checkpoint:

```bash
# Run in the separate vendor Torch environment first.
python "$REPO_DIR/benchmarks/host_compiler/qwen_eager.py" \
  --model "$MODEL_DIR" --device cuda:0 --output vendor.json

# Run in the CPU-Torch + Torch-FL environment.
python "$REPO_DIR/benchmarks/host_compiler/qwen_eager.py" \
  --model "$MODEL_DIR" --device flagos:0 --reference vendor.pt --output plain.json
python "$REPO_DIR/benchmarks/host_compiler/qwen_eager.py" \
  --model "$MODEL_DIR" --device flagos:0 --mode compile --cache host-program-cache \
  --reference plain.pt --output compile.json
python "$REPO_DIR/benchmarks/host_compiler/qwen_eager.py" \
  --model "$MODEL_DIR" --device flagos:0 --mode load --cache host-program-cache \
  --reference plain.pt --output native.json
python "$REPO_DIR/benchmarks/host_compiler/model_operator_submission.py" \
  --device flagos:0 --cache host-program-cache --output operators.json
```

The integration tests cover dynamic shapes served by one program, live outputs,
zero FlagGems/Triton Python events on native hits, invalid guard domains,
non-default streams, embedding flags, strided copies, overlapping destinations
and fresh views with shared version counters. Admission unit tests require no
PyTorch installation. The PR records exact lint, test and codegen-check output.
The integration manifest runs both hardware test files in the default NVIDIA
PR and nightly lanes, with a 53-pass minimum; the boxing lane does not select
this compiler-specific group. Admission and lifecycle unit tests run in the
shared CPU workflow without requiring PyTorch or an accelerator.

## Remaining work

Cold compilation remains expensive. An exploratory run took 63.840 seconds
upgrading existing broadcast guard variants. The final load-only run took
832.248 ms for its first prefill, excluding checkpoint loading and artifact
admission. Neither is a measurement from an empty compiler cache. Warm figures
must not be read as first-call or unseen-specialization latency guarantees.

Production work should separate reusable host control code from kernel
specializations and prepare hosts before serving. A new constexpr value should
not require recompiling the entire C++ wrapper. A persistent compiler process
can reduce startup cost but does not by itself provide that separation.

Training, arbitrary Python effects, data-dependent output shapes, concurrent
first-launch initialization, eviction, cross-version cache compatibility and
general vendor-native fallback coverage remain outside the validated contract.
