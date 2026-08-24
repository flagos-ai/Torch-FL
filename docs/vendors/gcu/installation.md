# Enflame GCU Installation Guide

Enflame GCU uses native operator kernels calling `libtopsaten.so`, not CUDA boxing. The TopsRider stack has no CUDA runtime, and the vendor `torch-gcu` wheel claims `PrivateUse1` for itself, so it cannot coexist with `torch_fl`.

## Prerequisites

- **CPU PyTorch 2.10.x**: `torch==2.10.0` from the upstream CPU index
- **TopsRider SDK**: GCU toolkit with `topsrt` runtime and `topsaten` operator library
- **Python**: 3.8 or later
- **Operating System**: Linux

The TopsRider SDK provides:
- `libtopsrt.so`: device/memory/stream runtime layer
- `libtopsaten.so`: ATen-style operator library with single-call execution (no workspace/executor phase like ACLNN)

## Installation

### 1. Install CPU PyTorch

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Build and install torch_fl

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

ACCELERATOR=gcu pip install --no-build-isolation -v -e .
```

Build flags:
- `ACCELERATOR=gcu`: selects the GCU build path and enables `GCU_KERNEL=ON`
- `GCU_KERNEL=ON`: compiles generated `topsaten` operator kernels (automatic when `ACCELERATOR=gcu`)
- `CUDA_KERNEL=OFF`: automatically disabled (no CUDA runtime exists on GCU)
- `FLAGGEMS_PYTHON=ON`: compiled into the same PrivateUse1 wrapper set as native GCU kernels; runtime routing selects native or FlagGems implementations without duplicate registration
- `--no-build-isolation`: ensures the build uses your installed CPU torch

The build runs `scripts/codegen_gcu.py` to generate kernels. Each op is validated against the demangled `topsaten::topsatenXxx` symbols actually present in `libtopsaten.so`; ops missing from the SDK are skipped with a warning.

### Codegen validation

The generator matches derived `topsaten<Name>` symbols (e.g., `topsatenAdd`, `topsatenSqrt`) against the installed SDK via:

```bash
nm -DC /path/to/libtopsaten.so | grep 'topsaten::topsaten'
```

Only ops with verified symbols are generated. Coverage extends as the SDK expands.

## Verification

### Import and device availability

```python
import torch_fl
import torch

print(f"flagos available: {torch_fl.flagos.is_available()}")
print(f"flagos devices: {torch_fl.flagos.device_count()}")

x = torch.randn(64, 64, device="flagos:0")
y = torch.abs(x)
print(f"abs matches CPU: {torch.allclose(y.cpu(), x.cpu().abs())}")
```

### Runtime backend selection

`torch_fl` installs a `lib/flagos_platform` marker so the runtime picks `backends_gcu.conf` automatically. No `FLAGOS_BACKEND_CONFIG` override is needed.

## Testing

Run smoke tests over generated operators and factory functions:

```bash
# General factory operator tests
pytest tests/integration/test_factory_ops.py -v -s --tb=short

# Common operator smoke tests (elementwise, reductions, matmul)
pytest tests/integration/ops/test_common_ops.py -v -s --tb=short
```

Run the shared AMP contract, which every supported platform executes with the same command:

```bash
pytest tests/integration/test_amp_contract.py -m amp -v --tb=short
```

Cases needing a capability GCU does not provide skip with a reason naming the platform.

The same suites run in CI through [`.github/configs/gcu.yml`](../../../.github/configs/gcu.yml); see [CI scope](#ci-scope) for what that manifest covers and what it deliberately leaves out.

## Platform-specific Behavior

### CPU fallback for missing kernels

Ops without a `topsaten` kernel are **not registered** on `PrivateUse1` at all, so they reach the `cpu_fallback` dispatcher hook instead of raising an error. This keeps models working even when coverage is incomplete. Adding an op is a matter of extending the `OPS` table in `scripts/codegen_gcu.py`.

### int64 limitations

`topsaten` has no int64 kernels: every op returns `NOT_SUPPORT` for an `I64` operand. Each generated kernel guards on dtype and runs the op on CPU for int64 inputs. This keeps indices, masks, and counters working transparently.

### Rank-0 tensor workaround

`topsaten` rejects rank-0 shapes, so 0-dim tensors are described as 1-element vectors to the library.

### Device-scoped pointers

GCU device pointers are device-scoped (no unified addressing): a pointer only resolves against the **current** device. The allocator and every kernel select the device first, and the default stream is per-device.

### Contiguous input requirement

Unlike `mudnn` (MUSA), `topsaten` does not honor strides on non-contiguous inputs. Generated kernels call `.contiguous()` where necessary to materialize a contiguous copy before passing to `topsaten`.

## Optional: FlagGems via Triton-GCU

FlagGems can provide Triton-compiled kernels when the Enflame `triton_gcu` plugin and `/opt/triton_gcu` toolchain are installed. GCU builds compile the FlagGems Python caller alongside native topsaten wrappers, while the backend configuration selects which implementation runs for each exact ATen overload.

The GCU compatibility layer prepares the Triton-GCU runtime but does not call `flag_gems.enable()` to register a competing PrivateUse1 implementation. This keeps one wrapper per overload and allows native and FlagGems RNG paths to share the same per-device seed/offset stream.

The FlagGems path remains experimental and requires validation on the target S60 software stack.

## Limitations

### CI scope

[`.github/configs/gcu.yml`](../../../.github/configs/gcu.yml) runs an S60 runner against an isolated CPU-PyTorch wheel, selecting the same contract suites the other platforms run by marker rather than by a file allowlist: the operator suite, the full `tests/integration/ops/test_rng_dispatch.py`, `tests/integration/test_factory_ops.py`, and the shared `tests/integration/test_amp_contract.py`. FlagGems markers are excluded because the CI image does not install the vendor Triton stack. Profiler contract tests and Qwen3 smoke are not in the manifest yet; see the notes below and the comment block at the end of that file.

### Distributed support not validated

Distributed collectives are not validated on GCU hardware. The `_VENDOR_PROFILES` routing table in `torch_fl/comm/process_group.py` lists GCU as FlagCX-only unless live evidence proves otherwise.

### Profiler is runtime only; torch.compile not validated

The TOPSPTI tracer collects activities on S60, but none surface as device events on the CPU-only PyTorch/Kineto build used here, which supplies no PrivateUse1 resolver. This is an environment limitation rather than a tracer defect, so `test_profiler_contract.py` stays out of CI until it is measured end to end. `torch.compile` has not been validated on GCU.

## Build without native kernels

To build the runtime layer only (device/memory/stream support) with no native operator kernels:

```bash
ACCELERATOR=gcu GCU_KERNEL=OFF pip install --no-build-isolation -v -e .
```

All compute ops will fall back to CPU. This mode is useful for testing the runtime layer in isolation.

## Reference Documentation

- [Codegen source](../../../scripts/codegen_gcu.py): category-driven kernel generation for `topsaten`
- [Compatibility matrix](../../reference/compatibility.md): platform status and limitations
- [Environment variables](../../reference/environment-variables.md): runtime environment variables and backend selection
