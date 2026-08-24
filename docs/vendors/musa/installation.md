# Moore Threads MUSA Installation Guide

Moore Threads MUSA uses native operator kernels calling `libmudnn.so`, not CUDA boxing. The MUSA toolkit ships no CUDA runtime, and there is no vendor dispatch key to box into.

## Prerequisites

- **CPU PyTorch 2.10.x**: `torch==2.10.0` from the upstream CPU index
- **MUSA toolkit**: Moore Threads SDK with `musart` runtime, `mudnn` operator library, and `murand` RNG library under `/usr/local/musa`
- **Python**: 3.8 or later
- **Operating System**: Linux

The MUSA toolkit provides:
- `musart`: runtime layer (device/memory/stream)
- `mudnn` (`libmudnn.so`): operator library with category-driven API (Unary/Binary/Reduce/MatMul/Softmax/Convolution/Dropout)
- `murand` (`libmurand.so`): device-side Philox random generation for uniform, normal, and integer draws

`mudnn` links against `musart` only and pulls in **no torch symbols at all**, making this backend torch-version-agnostic (tested against both 2.9.1 and 2.10.0 from the same source tree, though only 2.10.0 is officially supported).

## Installation

### 1. Install CPU PyTorch

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Build and install torch_fl

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

ACCELERATOR=musa pip install --no-build-isolation -v -e .
```

Build flags:
- `ACCELERATOR=musa`: selects the MUSA build path and enables `MUSA_KERNEL=ON`
- `MUSA_KERNEL=ON`: compiles generated `mudnn` operator kernels (automatic when `ACCELERATOR=musa`)
- `FLAGGEMS_PYTHON=ON`: compiles the optional Python dispatcher callers into the same wheel; runtime routing stays native unless `FLAGOS_USE_FLAGGEMS=1`
- `CUDA_KERNEL=OFF`: automatically disabled (the MUSA toolkit exports no CUDA symbols)
- `FLAGGEMS_KERNEL=OFF`: automatically disabled because the FlagGems C++ runtime is not built for MUSA
- `--no-build-isolation`: **required** (without it, pip resolves its own torch into a build overlay, and the extension links against that instead of your installed torch, causing `import torch_fl` to fail with `undefined symbol: c10::ValueError`)

The build runs `scripts/codegen_mudnn.py` to generate kernels. Coverage is **64 generated ops** plus 2 handwritten convolution kernels; native RNG kernels add muRAND-backed `rand`/`randn`, `rand_like`/`randn_like`, `randint`, `normal_`, `uniform_`, `random_`, and mudnn dropout paths. Everything outside those sets reaches the `cpu_fallback`.

### Why no-build-isolation is required

Without `--no-build-isolation`, pip creates a temporary build environment and installs its own copy of torch there. The C++ extension then links against that temporary torch, not the one you installed. When you later `import torch_fl` in your actual environment, the torch C++ object layouts don't match, causing symbol resolution failures.

This constraint is specific to MUSA because `mudnn` has no torch dependency at all. Earlier versions of this backend called `torch_musa`'s flat `at::musa::*` API from `libmusa_python.so`, which links against torch and embeds its C++ object layout—pinning the plugin to one exact torch build (`sizeof(c10::MessageLogger)` changed 408 → 400 between 2.9.1 and 2.10, corrupting the vendor `.so`'s stack). `mudnn` avoids that coupling entirely.

## Verification

### Import order with torch_musa

If the `torch_musa` package is installed alongside `torch_fl` (not recommended, but possible), you must **import `torch_fl` before `torch`**, or export `TORCH_DEVICE_BACKEND_AUTOLOAD=0`:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

`torch_musa` registers a `torch.backends` entry point, so a bare `import torch` autoloads it and claims the `PrivateUse1` backend name first. `torch_fl` sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0` internally when imported first, which covers the torch_fl-first order. The other order fails with an explicit message.

**Recommendation**: do not import the top-level `torch_musa` plugin in the same process. It is optional for normal torch-fl operation; only its `distributed` submodule may be discovered lazily for the MCCL fallback when it is installed.

### Import and device availability

```python
import torch_fl  # Must come before torch if torch_musa is present
import torch

print(f"flagos available: {torch_fl.flagos.is_available()}")
print(f"flagos devices: {torch_fl.flagos.device_count()}")

x = torch.randn(64, 64, device="flagos:0")
y = torch.abs(x)
print(f"abs matches CPU: {torch.allclose(y.cpu(), x.cpu().abs())}")
```

### Runtime backend selection

`torch_fl` installs a `lib/flagos_platform` marker so the runtime picks `backends_musa.conf` automatically. This native-only mode requires no environment variable override.

The same wheel also contains a narrow hybrid route set. Enable it with:

```bash
export FLAGOS_USE_FLAGGEMS=1
```

This selects `backends_musa_flagos_py.conf`: native mudnn/muRAND remains authoritative for every native schema, including RNG and dropout, while `all`, `all.dims`, `any`, `any.dims`, `index_add`, `index_add_`, and `repeat_interleave.Tensor` route to the FlagGems Python caller. The hybrid routes require FlagGems and the MThreads FlagTree compiler/runtime. On the measured host, FlagGems 5.0.2 executed with the vendor `flagtree-0.5.0+mthreads3.1` wheel (Triton 3.1.0, backend `mthreads`). The generic installed Triton 3.7.1 is not sufficient and must not be used for this path. The vendor wheel SHA-256 was `197b0c6954ad8b3edef51138311a8c4f3aea75b90ba0f69d3c2fda95a76b6b1b`.

## Testing

Run the MUSA dispatch suite:

```bash
pytest tests/integration/ops/test_musa_dispatch.py -v
```

This test file checks that ops route to the `musa` backend, exercises per-op environment overrides, and validates results against CPU references. It replaces the generic per-op tests in `tests/integration/ops/`, which assert `-> cuda` routing that MUSA builds cannot produce (no CUDA boxing kernels are compiled).

Run the autocast and GradScaler suite:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACCELERATOR=musa \
  LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH \
  pytest tests/integration/test_amp.py -v
```

Run common operator smoke tests:

```bash
pytest tests/integration/ops/test_common_ops.py -v -s --tb=short
```

**Note**: MUSA has no CI configuration file (`.github/configs/musa.yml` does not exist), so all validation is manual.

## Operator Coverage

### Generated operators (64 ops)

Category-driven codegen via `scripts/codegen_mudnn.py` covers:

- **Unary**: abs, sqrt, rsqrt, exp, log, log2, log10, log1p, sin, cos, acos, atan, tanh, sigmoid, silu, relu, gelu, erf, floor, ceil, sign
- **Binary**: add, mul, sub, div, pow, eq, ne, lt, le, gt, ge, maximum, minimum, logical_and, logical_or, logical_xor
- **Reductions**: sum, mean, max, min, argmax, argmin (with dim support)
- **MatMul**: mm, addmm, bmm, baddbmm
- **Softmax**: softmax, log_softmax
- **Composed from single mode**: neg (MUL by -1), trunc (TRUNCATEDIV by 1), expm1 (EXP then SUB 1)

### Handwritten operators (2 ops)

**Convolution** (`csrc/aten/backends/musa/mudnn_conv.cc`):
- `convolution_overrideable` cannot be left unregistered (ATen's default raises rather than being boxable to CPU)
- `mudnn`'s `Convolution` covers 2 spatial dims only:
  - conv1d runs as 2D conv with unit `H` dim (exact against CPU)
  - conv3d takes the CPU fallback
- Bias is a separate broadcast `Binary::ADD`; `RunFusion` accepts bias only for non-grouped 2D
- Algorithm is chosen by trial and cached per shape (since `GetRecommendForwardAlgorithm` can name one that `Run` then rejects)

**Copy and cast** (`csrc/aten/backends/musa/mudnn_copy.cc`):
- Strided copies and dtype casts use `mudnn`'s `Unary::IDENTITY` / `Unary::CAST`
- Handles both in a single device pass
- Without this, `copy_`/`clone`/`contiguous` would reach the CUDA `DispatchStub` and fail

### AMP and GradScaler

MUSA uses the shared `AutocastPrivateUse1` policies with its native mudnn routes.
Both `torch.float16` and `torch.bfloat16` are supported autocast targets:

```python
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401

model = torch.nn.Linear(8, 4, device="flagos")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("flagos")
inputs = torch.randn(2, 8, device="flagos")
targets = torch.randn(2, 4, device="flagos")

with torch.autocast("flagos", dtype=torch.bfloat16):
    output = model(inputs)
    loss = F.mse_loss(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

The standard policy groups cast matmul, linear, and convolution to the selected
lower-precision dtype. Logarithm, layer normalization, MSE loss, and the default
softmax policy run in float32; operations in the promote group use the widest
input dtype. Nested autocast state and explicit softmax dtypes follow PyTorch's
PrivateUse1 behavior.

GradScaler's `_amp_foreach_non_finite_check_and_unscale_` currently reaches the
existing CPU fallback because mudnn has no fused AMP-foreach primitive. The
fallback copies tensor-list operands and scalar state to CPU, invokes PyTorch's
reference kernel, and copies the mutated gradients and `found_inf` flag back to
MUSA. This preserves finite-step growth, overflow backoff, and optimizer-step
skipping, but it is not a native device-side performance path.

### Native RNG and CPU fallback

Native RNG uses a per-device Philox generator from `libmurand.so`. Each operation consumes one seed from the selected PyTorch `flagos` generator under its mutex, then configures a cached muRAND generator on the shared MUSA stream. This preserves `torch.manual_seed`, `torch.flagos.manual_seed`, `manual_seed_all`, `get_rng_state`/`set_rng_state`, explicit `generator=` isolation, and independent per-device sequences without maintaining a second Python-side RNG state. The optional FlagGems bridge reserves operation seeds through the same C++ API and starts each FlagGems Philox invocation at offset zero, so mixed native/FlagGems call order has one authoritative state.

Odd-sized normal outputs are generated with one extra sample because muRAND's normal API requires an even length. muRAND produces uniform samples in `(0, 1]`, so the native wrapper maps the endpoint to zero to provide PyTorch's required `[0, 1)` interval. S5000's Philox implementation returns `MURAND_STATUS_TYPE_ERROR` from `murandGenerateLongLong`; integer RNG therefore combines two supported 32-bit outputs and applies ATen-compatible range transforms, including int64 ranges wider than `INT64_MAX`.

The current native RNG set covers `rand`/`randn`, their generator and like variants, integer factories and in-place `random_`, `normal_`, `uniform_`, and `native_dropout` forward/backward. `randperm`, tensor-parameter distributions, and rejection-sampled distributions remain unregistered until a device implementation can provide their exact ATen contract. They therefore use `cpu_fallback` rather than being routed to an absent kernel.

Other ops with no `mudnn` mode are deliberately left unregistered, so they reach the `cpu_fallback` and stay correct. Examples:
- `sinh`, `cosh`, `asin`: no `mudnn` mode exists

Registering an op with no kernel behind it would trip the dispatcher's "backend not registered" check.

## Platform-specific Behavior

### Stride and broadcasting support

`mudnn` Tensors carry strides on **both** operands and honor 0-strides. This means:
- Broadcasting is just `expand()` (a view, no copy)
- Non-contiguous inputs are read in place
- No `.contiguous()` materialization needed (contrast with GCU's `topsaten`)

### int64 support

int64 works across Unary/Binary/Reduce/MatMul categories. Unlike GCU's `topsaten` (which has no int64 kernels at all), MUSA only falls back for genuinely unmapped dtypes (complex, quantized).

### TF32 matmul behavior

`mudnn` enables TF32 by default, whereas PyTorch defaults `torch.backends.cuda.matmul.allow_tf32` **off**. The handle is refreshed from torch's flag on every op to match PyTorch semantics. Without this, a 64×64 float `mm` drifts ~2e-2 from CPU.

### Broadcast-reduction SIGFPE workaround

`mudnn` v3300's `Reduce` raises `SIGFPE` (an uncatchable crash, not an error status) when reducing over more than one dim of a tensor that is a broadcast of a single element. Conv bias gradients hit exactly that case, since autograd feeds `ones.expand(...)` into the reduction.

A fully broadcast input is materialized before a multi-dim reduction to avoid this crash.

### Caching allocator

`flagos` keeps its own caching allocator over raw `musaMalloc`. `mudnn` allocates nothing on its own beyond op workspaces, which are served from the same allocator via a `MemoryMaintainer`.

## Limitations

### Manual S5000 validation

MUSA has no CI runner (no `.github/configs/musa.yml` config exists), so all tests are manual. The platform status is **Experimental** (see [`docs/reference/compatibility.md`](../../reference/compatibility.md) line 31).

The native RNG and hybrid FlagGems implementation were measured on 2026-08-17 on an eight-device Moore Threads MTT S5000 host. Device 0 reported capability 3.1, 60 multiprocessors, and 85,813,358,592 bytes of memory. The environment used CPU PyTorch 2.10.0, the installed `/usr/local/musa` toolkit (`mudnn` v3300), FlagGems 5.0.2, and the vendor `flagtree-0.5.0+mthreads3.1` wheel (Triton 3.1.0, backend `mthreads`). Results:

- `pytest tests/integration/test_amp.py -v`: **25 passed** in 9.00 seconds. Both FP16 and BF16 passed lower-precision matmul/linear/convolution, float32 and promote policies, nested state, mutable non-finite unscale, finite scale growth, overflow backoff, optimizer-step skipping, and autocast/GradScaler training.
- `pytest tests/integration/ops/test_rng_dispatch.py -m "main_ops" -v`: the unified RNG suite passed, including explicit generators, state round trips, `torch.manual_seed`, full-width int64 ranges, native dropout, shared native/FlagGems reservation ordering, and multi-device sequence isolation. MUSA-only generator and reservation cases are selected by the `musa` mark.
- `pytest tests/integration/ops/test_musa_dispatch.py -v`: **89 passed** in 36.31 seconds.
- `pytest tests/unit/test_vendor_routing.py tests/unit/test_musa_rng_bridge.py -v`: **24 passed** in 2.19 seconds.
- `pytest tests/integration/ops/test_musa_flaggems.py -q`: **2 passed** in 5.33 seconds with `FLAGOS_USE_FLAGGEMS=1`. The test instruments and observes all seven selected Python callables (`all`, `all.dims`, `any`, `any.dims`, `repeat_interleave.Tensor`, `index_add`, and `index_add_`) on `flagos:0`, verifies CPU-equivalent outputs including duplicate indices, and executes FlagGems `randn` on `flagos:0` between native `rand` calls. Repeating the sequence after `torch.flagos.manual_seed(20260817)` reproduces all three outputs and confirms two shared C++ generator reservations.
- Native and hybrid suites ran in separate pytest processes. Backend configuration is cached by the process-static C++ `BackendTable()`, so changing `FLAGOS_USE_FLAGGEMS` after a native import cannot switch the active routes.
- The MThreads driver obtained the same nonzero raw `musaStream_t` that native mudnn/muRAND uses, so native and FlagGems launches share the torch-fl stream. In the `torch.compile` path this handle comes from `torch_fl.compile.musa_runtime.get_current_raw_stream()`, with no `torch_musa` module involved.

The generic `triton` 3.7.1 installation remains unsuitable because it does not ship the MThreads backend. FlagGems stochastic ATen routing is intentionally still native-first; the end-to-end FlagGems RNG evidence comes from its real `flag_gems.ops.randn.randn` kernel and the shared reservation bridge, not an expanded RNG dispatcher route.

### Distributed support

**Status**: FlagCX-first routing and MCCL fallback are implemented; this RNG validation did not rerun multi-process collectives.

MUSA distributed communication routes through FlagCX via the identity view path. The `_flagos_identity_view` binding passes privateuseone tensors directly to FlagCX's MUSA adaptor without storage conversion, since FlagCX extracts `data_ptr()` and `musaStream_t` without validating `device().type()`. If FlagCX initialization is unavailable, the MUSA profile attempts `ProcessGroupMCCL` from `torch.distributed` or `torch_musa.distributed`. The FlagGems compatibility shim preserves access to an installed `torch_musa` package's submodules without importing its top-level plugin and taking ownership of PrivateUse1.

The routing unit suite covers FlagCX preference, MCCL fallback, the `mthreads` vendor alias, missing identity bindings, and the no-backend error. End-to-end multi-process validation still requires FlagCX or MCCL built for this host and should use `tests/manual/test_comm_device_index.py --world-size 2`. See [`docs/architecture/distributed-flagcx.md`](../../architecture/distributed-flagcx.md) for the ProcessGroupFlagOS architecture.

### Profiler

MUSA profiling uses the optional MUPTI activity API from the MUSA toolkit. A profiler-capable
build detects `mupti_activity.h` under `MUSA_HOME` and dynamically loads `libmupti.so` when a
`torch.profiler` session starts. The extension does not link against MUPTI, so importing
`torch_fl` remains possible on a host that has the MUSA runtime but no profiler library.

The tracer enables concurrent and serialized kernels, runtime and driver API records, memcpy,
memset, and external-correlation records. MUPTI timestamps are converted into the realtime clock
domain used by Kineto. Runtime/device vendor correlation IDs remain separate from torch external
correlation IDs: the former draws flow arrows and the latter attributes device time to CPU ops.

Run the focused hardware test on an MTT S5000 host:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACCELERATOR=musa \
LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH \
pytest tests/integration/test_profiler_musa.py -q
```

Measured with CPU PyTorch 2.10.0 and the installed MUSA 5.1.0 toolkit, the test captured real
MUPTI kernel, runtime, and memcpy events with positive duration, valid names, device/stream
metadata, and a valid Chrome trace JSON document. The same run exposed torch external IDs on the
captured device events. CPU-only Kineto builds may not invoke the PrivateUse1 resolver, so this is
an MUPTI device-timeline validation rather than a claim of full torch-cuda profiler parity.

Useful diagnostics are `FLAGOS_MUPTI_DEBUG=1` for activity setup and session lifecycle logging and
`FLAGOS_MUPTI_LIBRARY=/path/to/libmupti.so` to select a specific MUPTI library. MUPTI subscriber
ownership remains process-global; an external MUSA profiling tool may therefore reject a concurrent
`torch.profiler` session. `torch.compile` with the vendor FlagTree runtime is
validated separately below; a stock Triton wheel remains insufficient for MUSA.

### FlagTree and `torch.compile`

MUSA `torch.compile(backend="flagos")` uses TorchInductor without rewriting the
graph to CUDA. The graph and autograd device stay `flagos`, while the MThreads
FlagTree runtime receives a native `musa` target and launches through the shared
`musaStream_t`. The tested compiler is `flagtree-0.5.0+mthreads3.1` (Triton 3.1,
backend `mthreads`); generic Triton 3.7.1 is not MUSA compiler evidence.

**`torch_musa` is not required and must not be installed in the same process.**
FlagTree reaches the MUSA device through `torch_fl` alone: the vendor MThreads
driver normally reads device availability, current device, capability, and the
raw stream from `torch_musa`, and `torch_fl.compile.musa_runtime` rebinds those
lookups onto its own runtime before the first Triton driver is created. See
[torch-compile-integration.md](../../architecture/torch-compile-integration.md)
for why the plugin cannot coexist with `torch_fl`.

`mode="max-autotune"` compiles and runs, but its runtime coordinate-descent
tuning is disabled on MUSA: it times candidate configurations through Inductor's
CUDA benchmarker (`torch.cuda.synchronize` plus `torch.cuda.Event`), which the
CPU PyTorch wheel cannot provide. Kernels are compiled and executed, just not
runtime-tuned.

Use a process with the vendor runtime before running the focused compile tests:

```bash
PYTHONPATH=/path/to/flagtree-mthreads-runtime:$PWD \\
LD_LIBRARY_PATH=/path/to/flagtree-mthreads-runtime/triton/_C:/usr/local/musa/lib:$LD_LIBRARY_PATH \\
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACCELERATOR=musa FLAGOS_USE_FLAGTREE=1 \\
pytest tests/integration/test_compile.py -v
```

On the MTT S5000 this suite passed in full (22 tests), covering eager
equivalence, forward/backward compilation, FP32/FP16 paths, max-autotune,
recompilation, FakeTensor tracing, output/gradient placement, and the assertion
that `torch_musa` is absent from `sys.modules` throughout. MThreads FlagTree queries
the MUSA runtime during compiler setup, so `torch_fl` serializes compilation to
one Inductor worker by default. Set `TORCHINDUCTOR_COMPILE_THREADS` or the
`compile_threads` option only when the installed vendor driver is known to be
fork-safe. `FLAGOS_COMPILE_FALLBACK_EAGER=1` is useful for diagnosing an
unsupported graph, but a fallback pass does not count as compiler validation.

The result is specific to the measured MTT S5000, MUSA 5.1.0, CPU PyTorch 2.10,
and the vendor FlagTree wheel above; it does not imply support for other MUSA
SDK, PyTorch, or FlagTree combinations.

### FlagGems runtime prerequisite

The wheel compiles the narrow FlagGems Python dispatcher set, but a generic Triton wheel does not imply MUSA kernel support. Use the vendor `flagtree-0.5.0+mthreads3.1` wheel compatible with FlagGems 5.0.2 and expose its `triton` package and `triton/_C` directory to the process. The measured setup used:

```bash
PYTHONPATH=/path/to/flagtree-runtime:$PWD \
LD_LIBRARY_PATH=/path/to/flagtree-runtime/triton/_C:/path/to/flagtree-runtime/triton:$CONDA_PREFIX/lib:/usr/local/musa/lib \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLAGOS_USE_FLAGGEMS=1 ACCELERATOR=musa \
pytest tests/integration/ops/test_musa_flaggems.py -q
```

If that vendor compiler/runtime is unavailable, keep `FLAGOS_USE_FLAGGEMS` unset and use `backends_musa.conf`; native mudnn/muRAND and CPU fallback remain usable.

## Build without native kernels

To build the runtime layer only (device/memory/stream support) with no native operator kernels:

```bash
ACCELERATOR=musa MUSA_KERNEL=OFF pip install --no-build-isolation -v -e .
```

All compute ops will fall back to CPU. This mode is useful for testing the runtime layer in isolation.

## Reference Documentation

- [Codegen source](../../../scripts/codegen_mudnn.py): category-driven kernel generation for `mudnn`
- [Compatibility matrix](../../reference/compatibility.md): platform status and limitations
- [Environment variables](../../reference/environment-variables.md): runtime environment variables and backend selection
