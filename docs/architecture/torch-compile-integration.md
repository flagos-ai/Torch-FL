# torch.compile Integration for flagos Device

This document describes the `torch.compile` integration for the flagos device, enabling automatic kernel fusion and optimization via TorchInductor.

## Overview

The flagos device now supports PyTorch 2.0+ `torch.compile` for automatic performance optimization:

```python
import torch_fl  # Import first on MetaX.
import torch

model = MyModel().to("flagos:0")
compiled_model = torch.compile(model, backend="flagos")

# Automatic fusion of elementwise ops, reduced dispatch overhead
output = compiled_model(input)
```

**Key benefits**:
- Automatic kernel fusion (no manual optimization needed), cutting per-op dispatch overhead
- Graph stays on flagos: no cuda round trip, no copy at the graph boundary
- Compatible with existing flagos dispatch (FlagGems Python/C++, CUDA boxing)
- Optional FlagTree compilation for multi-backend kernel generation (drop-in: it
  replaces `triton` at install time, so no code change is needed)

## Quick Start

### Basic Usage

```python
import torch_fl  # Import first on MetaX.
import torch

# Standard model definition
model = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
).to("flagos:0")

# Compile with flagos backend
model = torch.compile(model, backend="flagos")

# Use as normal
x = torch.randn(64, 512, device="flagos:0")
y = model(x)  # Automatically fused kernels
```

### Compilation Modes

```python
# Default: balanced optimization
model = torch.compile(model, backend="flagos")

# Maximum performance (longer compile, better runtime)
model = torch.compile(model, backend="flagos", mode="max-autotune")

# Explicit inductor config overrides
model = torch.compile(model, backend="flagos", options={"max_autotune": True})
```

`mode` and `options` are expanded into inductor config patches scoped to that
compile. Note that CUDA graphs are always forced off (see Limitations), so
`mode="reduce-overhead"` -- whose main lever is cudagraphs -- has little effect
here.

### FlagTree Integration

[FlagTree](https://github.com/flagos-ai/FlagTree) is a Triton fork whose kernel
compiler targets many vendor backends. It integrates by **substitution at install
time**, which is the whole thing to understand about it:

- Its wheel is named `flagtree`, but the module it installs is **`triton`**.
- Installing it **uninstalls the official `triton`** and takes its place.
- So `import flagtree` never works, and inductor's own `import triton` already
  resolves to FlagTree once installed. Nothing in `torch_fl` patches `sys.modules`.

There is no `flagtree` package on PyPI; build it from source. On a machine whose
`triton` is in use by FlagGems, build into a separate virtualenv, since the
install removes the existing `triton`:

```bash
# System deps (nlohmann-json must match cmake/json-version.txt)
apt install zlib1g-dev libxml2-dev nlohmann-json3-dev

git clone https://github.com/flagos-ai/FlagTree.git
cd FlagTree
pip install -r python/requirements.txt

# Pick the vendor backend. Do NOT set this for nvidia or amd.
# export FLAGTREE_BACKEND=metax

# Repo root for Triton 3.4+ (branch main); use `cd python` first on 3.1-3.3.
MAX_JOBS=64 pip install . --no-build-isolation -v
```

LLVM arrives as a prebuilt tarball (~4.5 GB unpacked), so this does not build
LLVM from source. Verify, from any directory other than `FlagTree/python`:

```bash
pip show flagtree                                  # wheel name
python -c 'import triton; print(triton.__path__)'  # module name
```

Branch selects the Triton base version, which must match what your torch
expects: `main` is Triton 3.6 (correct for torch 2.10), `triton_v3.5.x` is 3.5
and carries the Ascend backend.

#### Using a FlagTree build without replacing your triton

Since FlagTree's install removes the existing `triton`, the cheapest way to test
against it is to build into a separate virtualenv and shadow `triton` by path,
keeping torch and `torch_fl` from your main environment. Triton does not link
against torch, so the two mix freely as long as the Python versions match:

```bash
PYTHONPATH=/path/to/flagtree-venv/lib/python3.12/site-packages \
  FLAGOS_USE_FLAGTREE=1 pytest tests/integration/test_compile.py
```

This avoids rebuilding `torch_fl` inside the FlagTree venv, and leaves the
FlagGems-facing `triton` in the main environment untouched.

Once built, compilation goes through FlagTree with no further configuration.
Setting `FLAGOS_USE_FLAGTREE=1` **asserts** that the active `triton` is FlagTree
and errors out if it is stock Triton — it cannot switch FlagTree on, because that
was decided when the environment was built:

```bash
FLAGOS_USE_FLAGTREE=1 python your_script.py
```

Detection uses the FlagTree-only module `triton._flagtree_spec`. Note that
`triton._flagtree_backend.FLAGTREE_BACKEND` is *not* a usable signal: it is the
empty string on nvidia/amd builds, since upstream directs you not to set
`FLAGTREE_BACKEND` for those.

PPU FlagTree currently queries `torch.cuda.current_device()` while selecting
compiler hints. Inductor's asynchronous compile workers can make that query
after a PPU CUDA context was initialized in the parent, which PyTorch rejects as
CUDA reinitialization after `fork` ([FlagTree #1031](https://github.com/flagos-ai/FlagTree/issues/1031)).
The flagos backend therefore defaults PPU FlagTree to one compile thread until
the upstream driver no longer initializes CUDA in a worker. An explicit
`TORCHINDUCTOR_COMPILE_THREADS` value or `compile_threads` compile option remains
authoritative.

#### MUSA / MThreads FlagTree

MUSA uses the vendor FlagTree runtime (`flagtree-0.5.0+mthreads3.1`, Triton 3.1)
and its `mthreads` Python backend. The compiler emits a `musa` target. A stock
Triton wheel only provides `nvidia`/`amd` backends and cannot compile MUSA
kernels.

FlagTree talks to `torch_fl` directly; the separate `torch_musa` plugin is not
required and its `__init__` is never imported. The vendor MThreads driver reads
four runtime facts from `torch_musa` — device availability (which is what makes
Triton select the backend at all), current device, compute capability, and the
raw `musaStream_t`. Running that plugin's `__init__` in this process is not an
option, because it claims the process-global PrivateUse1 hooks that `torch_fl`
must own. So
[`torch_fl/compile/flagtree_shim.py`](../../torch_fl/compile/flagtree_shim.py)
answers those four questions from `torch_fl`'s own MUSA runtime and rebinds them
onto the vendor driver class before the first driver instance exists. The
consequence that matters: compiled kernels are submitted to the same stream the
native mudnn kernels use, so a compiled graph is ordered against eager MUSA work
without an explicit synchronize between them.

The binding must happen before Triton instantiates its driver, since the driver's
`__init__` copies these attributes onto the instance. `flagos_compile_backend`
therefore binds at module load and again per compile, both idempotently.
`tests/integration/test_compile.py::test_musa_flagtree_binds_to_torch_fl_runtime`
asserts the binding took effect by checking that every rebound driver callable
resolves inside `torch_fl`. That, rather than `torch_musa` being absent from
`sys.modules`, is the property worth pinning: the driver would also "work"
against some other runtime, and `torch_fl` itself publishes a small
compatibility surface under that module name when FlagGems is enabled (see
`_install_musa_flaggems_compat`).

MThreads FlagTree currently queries the MUSA runtime while compiling. The flagos
backend therefore defaults this path to one Inductor compile thread, preserving
an explicit `TORCHINDUCTOR_COMPILE_THREADS` or `compile_threads` setting. This
serialization is a safety workaround for the vendor driver, not a performance
claim; remove it only after the relevant FlagTree driver is verified fork-safe.

On the measured MTT S5000 setup, validation requires the vendor FlagTree runtime,
`ACCELERATOR=musa`, and `FLAGOS_USE_FLAGTREE=1`. The generic Triton 3.7.1 runtime
is not MThreads execution evidence.

Example environment:

```bash
PYTHONPATH=/path/to/flagtree-mthreads-runtime:$PWD \
LD_LIBRARY_PATH=/path/to/flagtree-mthreads-runtime/triton/_C:/usr/local/musa/lib:$LD_LIBRARY_PATH \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 ACCELERATOR=musa FLAGOS_USE_FLAGTREE=1 \
pytest tests/integration/test_compile.py -v
```

The focused FlagTree test must compare compiled output with eager output and
assert that outputs and gradients remain on `flagos`; CPU-only tests can cover
registration and vendor target selection but do not establish MUSA compiler
support.

## Architecture

### Phase 1: Inductor Integration

flagos is registered with TorchInductor as a **first-class GPU device**. The
traced graph is handed to `compile_fx` unchanged -- still on flagos -- and
inductor generates Triton kernels that operate on flagos tensors directly.

**Components**:
1. **Backend registration** (`torch_fl/compile/inductor_backend.py`)
   - Registers `"flagos"` with `torch._dynamo.register_backend`
   - Expands `mode` / `options` into inductor `config_patches`
   - Delegates to `compile_fx` with no graph rewriting

2. **Device interface** (`torch_fl/compile/device_interface.py`)
   - `DeviceInterface` subclass: device state from `torch.flagos`, hardware
     properties from `torch.cuda` (the same physical GPU)
   - Adds `"flagos"` to inductor's `GPU_TYPES` so `is_gpu()` is True and the
     Triton codegen path is taken instead of C++/CPU
   - Reports the underlying compiler target at the Triton boundary
     (`DeviceProperties.create`): `cuda`/`nvidia` on CUDA-compatible builds and
     `maca`/`metax` on MetaX

3. **Codegen registration** (`torch_fl/compile/inductor_codegen.py`)
   - Device op overrides (guards, streams, sync) inheriting the CUDA ones
   - Scheduling + wrapper codegen: the stock CUDA/Triton pipeline

4. **Dispatch integration**
   - Ops inductor does not fuse fall back to eager flagos dispatch
     (FlagGems Python/C++ or CUDA boxing) with no changes needed

**Flow**:
```
torch.compile(model, backend="flagos")
  → dynamo captures FX graph (on flagos)
  → compile_fx / AOT autograd, graph never leaves flagos
  → inductor generates fused Triton kernels for flagos tensors
  → unfused ops fall back to flagos eager dispatch
```

**Why the graph is not rewritten to cuda.** An earlier version converted the
graph and example inputs to cuda first. Beyond the copy per call, this breaks
backward: `at::getAccelerator()` is PrivateUse1/flagos, and
`torch::autograd::Node::stream()` only yields a stream when a node's input
device type equals the accelerator, so a cuda-rewritten graph produces
stream-less autograd nodes and AOT autograd's backward trace trips
`opt_ready_stream && opt_parent_stream` (engine.cpp:1085).

### FlagTree Compilation

**Components**:
1. **Detection only** (`torch_fl/compile/flagtree_shim.py`)
   - No import patching: FlagTree *is* the `triton` module once installed, so
     inductor picks it up with no interception
   - `is_flagtree_active()` tests for the FlagTree-only `triton._flagtree_spec`
   - `FLAGOS_USE_FLAGTREE=1` calls `require_flagtree()`, which errors if the
     active Triton is stock

2. **Backend selection**
   - Chosen at FlagTree build time via `FLAGTREE_BACKEND` (unset for nvidia/amd),
     not at runtime
   - Same Triton kernel code, different backend compiler

**Benefits**:
- Multi-backend: same compiled model runs on NVIDIA/Ascend/Cambricon/MetaX
- Future-proof for non-NVIDIA hardware

## Performance

**Not yet measured.** Correctness is verified (`tests/integration/test_compile.py`);
benchmarking the fusion gain, and comparing it against stock `inductor` on cuda,
is still open work. Structurally the two should land close together -- same
inductor fusion passes, same Triton codegen, and since the graph stays on flagos
there is no per-call copy -- but that is an expectation, not a measurement.

### Benchmarking

```bash
# Run performance benchmark
python tests/perf/bench_compile.py --model=mlp --batch-size=64

# Compare with CUDA baseline
python tests/perf/bench_compile.py --model=transformer --compare-cuda

# Test FlagTree integration
FLAGOS_USE_FLAGTREE=1 python tests/perf/bench_compile.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLAGOS_USE_FLAGTREE` | `0` | Require the active triton to be FlagTree (assert, not switch) |
| `FLAGOS_COMPILE_FALLBACK_EAGER` | `0` | Fall back to eager on compile errors |

Existing dispatch variables (`FLAGOS_USE_FLAGGEMS`, `FLAGOS_BACKEND_CONFIG`) still apply to compiled kernels.

## Troubleshooting

### Compilation Errors

**Symptom**: `torch.compile` raises errors during graph capture or codegen.

**Solutions**:
1. Enable fallback to eager: `FLAGOS_COMPILE_FALLBACK_EAGER=1`
2. Check for unsupported ops (dynamic shapes, custom ops)
3. Verify meta implementations for custom ops

### No Speedup

**Symptom**: Compiled model runs at same speed as eager.

**Possible causes**:
1. Model is compute-bound (large matmuls) — fusion won't help much
2. Compilation didn't fuse ops (check inductor logs)
3. Dispatch overhead is small relative to kernel time

**Debug**: Run with `TORCH_LOGS="+inductor"` to see fusion decisions.

### FlagTree Not Active

**Symptom**: `FLAGOS_USE_FLAGTREE=1` raises "the active 'triton' module is stock
Triton, not FlagTree".

The env's `triton` is the official one. FlagTree is selected when the environment
is built, so this cannot be fixed at runtime — build FlagTree per the section
above, into this interpreter. Check what you have with:

```bash
python -c 'import triton; print(triton.__path__)'
python -c 'import importlib.util as u; print(u.find_spec("triton._flagtree_spec") is not None)'
```

Do not reach for `import flagtree` when debugging this; that module does not
exist at any point, whether or not FlagTree is installed.

## Testing

```bash
# Run integration tests
pytest tests/integration/test_compile.py -v

# Test specific scenarios
pytest tests/integration/test_compile.py::test_basic_compile
pytest tests/integration/test_compile.py::test_compile_backward

# Regression guards for the two codegen fixes this integration required
pytest tests/integration/test_compile.py -k fake_tensor
pytest tests/integration/ops/test_clamp_dispatch.py -v

# Test FlagTree compilation (requires a FlagTree-built env)
FLAGOS_USE_FLAGTREE=1 pytest tests/integration/test_compile.py::test_flagtree_compiles_correct_results
```

### Codegen fixes this integration required

Two generated-kernel bugs only surface under compilation, so their regression
tests live alongside it:

- **`detach` re-dispatch.** The generated kernel called `at::detach(self)`, which
  is registered on PrivateUse1 too and so dispatched back into itself. Eager hid
  the recursion because `DeviceBoxingGuard` rewrites self's device metadata
  first; under FakeTensor it cannot, since the Python dispatch key sits *above*
  the backend key. Dynamo traces every `nn.Linear` through detach, so this was a
  stack-overflow segfault at trace time. Fixed by emitting `at::native::detach`
  (`NATIVE_DIRECT_VIEW_OPS` in `scripts/codegen_ops.py`).
- **`optional<Tensor>` boxing in in-place kernels.** `gen_inplace` handed only
  plain `at::Tensor` args to `DeviceBoxingGuard`, so `clamp_.Tensor` passed
  unboxed flagos `min`/`max` into a CUDA `self` and crashed. Fixed by
  materializing each optional into a holder, matching `gen_functional_pure`.

## Limitations

1. **torch >= 2.0 required**: Older PyTorch versions don't have `torch.compile`
2. **Inductor-compatible ops only**: Custom C++ ops may not fuse
3. **Dynamic shapes**: Some models with dynamic shapes may not compile
4. **CUDA graphs off**: `torch.cuda.CUDAGraph` is a dummy class in the CPU torch
   wheel, so `triton.cudagraphs` is forced off even under `mode="max-autotune"`
5. **FlagTree maturity**: Backend support varies by hardware; Hygon HCU is
   validated on `gfx936` and MetaX on C550/MACA 3.8.0, while other vendor
   backends remain untested here

## Roadmap

- [x] Phase 1: Inductor integration (flagos as a first-class GPU device)
- [x] Phase 2: FlagTree integration — needs no shim; FlagTree substitutes for
      triton at install time, so inductor picks it up unaided. Built from source
      on H800 (`flagtree-0.6.0`, Triton 3.6, nvidia backend).
- [x] `torch.compile(backend="flagos")` end-to-end on FlagTree — forward and
      backward compile and match eager on NVIDIA and MetaX, with outputs and
      gradients remaining on `flagos`. The complete compile suite passes on both
      targets (the MetaX-specific event regression adds one case there).
- [ ] Benchmark fusion gains vs. stock inductor+triton on cuda
- [ ] Phase 3: FlagGems-aware fusion (recognize pre-optimized patterns)
- [ ] Phase 4: Custom fusion patterns for flagos-specific ops

## See Also

- [PyTorch 2.0 torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchInductor overview](https://pytorch.org/docs/stable/torch.compiler_inductor_overview.html)
- [FlagTree repository](https://github.com/flagos-ai/FlagTree)
- [CPU torch + external libtorch_cuda.so](../vendors/cuda/external-libtorch-cuda.md) — why several `torch.cuda` bindings need shimming
- [`torch_fl/compile/README.md`](../../torch_fl/compile/README.md) — the registration surface in detail
