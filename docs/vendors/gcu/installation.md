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

FLAGOS_ACCELERATOR=gcu pip install --no-build-isolation -v -e .
```

Build flags:
- `FLAGOS_ACCELERATOR=gcu`: selects the GCU build path and enables `FLAGOS_BUILD_VENDOR=ON`
- `FLAGOS_BUILD_VENDOR=ON`: compiles generated `topsaten` operator kernels (automatic when `FLAGOS_ACCELERATOR=gcu`)
- `FLAGOS_BUILD_BOXING=OFF`: automatically disabled (no CUDA runtime exists on GCU)
- `FLAGOS_BUILD_FLAGGEMS=ON`: compiled into the same PrivateUse1 wrapper set as native GCU kernels; runtime routing selects native or FlagGems implementations without duplicate registration. It is on by default under `FLAGOS_ACCELERATOR=gcu`, but the environment variable is applied afterwards, so export `FLAGOS_BUILD_FLAGGEMS=1` explicitly if you also set the other kernel flags in the same shell
- `--no-build-isolation`: ensures the build uses your installed CPU torch

The build runs `scripts/codegen/codegen_gcu.py` to generate kernels. Each op is validated against the demangled `topsaten::topsatenXxx` symbols actually present in `libtopsaten.so`; ops missing from the SDK are skipped with a warning. `scripts/codegen/codegen_gcu_flaggems.py` separately writes the FlagGems wrapper registrations into `csrc/aten/backends/gcu/generated/gcu_flaggems_register.inc`; both generators must be re-run when the routing sets change.

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

Ops without a `topsaten` kernel are **not registered** on `PrivateUse1` at all, so they reach the `cpu_fallback` dispatcher hook instead of raising an error. This keeps models working even when coverage is incomplete. Adding an op is a matter of extending the `OPS` table in `scripts/codegen/codegen_gcu.py`.

### int64 limitations

`topsaten` has no int64 kernels: every op returns `NOT_SUPPORT` for an `I64` operand. Each generated kernel guards on dtype and runs the op on CPU for int64 inputs. This keeps indices, masks, and counters working transparently.

### Rank-0 tensor workaround

`topsaten` rejects rank-0 shapes, so 0-dim tensors are described as 1-element vectors to the library.

### Device-scoped pointers

GCU device pointers are device-scoped (no unified addressing): a pointer only resolves against the **current** device. The allocator and every kernel select the device first, and the default stream is per-device.

A copy between two cards is a separate case that selecting the device does not fix. `topsMemcpy` with a device-to-device kind returns an error and leaves the destination byte-for-byte unchanged when the two pointers live on different cards, under every candidate current device, so a copy routed through it moves nothing while reporting a failure that callers commonly ignore — the visible symptom is a tensor that keeps its previous contents, not an exception. `topsMemcpyPeer` is the entry point that transfers the bytes, and it requires the current device to be one of the two cards. Both copy paths in the runtime (`csrc/runtime/accelerator/gcu/memory.cc` and `csrc/runtime/allocator/backends/gcu_memory.h`) resolve both pointers and take the peer route when their devices differ. The asynchronous peer entry point is stricter still: it also needs the caller's stream to belong to the source's device, and stalls the caller rather than returning an error when it does not.

### Contiguous input requirement

Unlike `mudnn` (MUSA), `topsaten` does not honor strides on non-contiguous inputs. Generated kernels call `.contiguous()` where necessary to materialize a contiguous copy before passing to `topsaten`.

## FlagGems via the enflame Triton backend

FlagGems provides Triton-compiled kernels for the operators `topsaten` does not
cover, and the generated `backends_gcu.conf` routes to them by default wherever
FlagGems can execute the operator correctly on GCU. GCU builds compile the
FlagGems Python caller alongside the native `topsaten` wrappers, and the backend
configuration selects which implementation runs for each exact ATen overload.

The FlagGems path needs a vendor Triton backend that carries the `enflame`
backend — `flagtree==0.6.1+enflame3.6` in CI, Enflame's older `triton_gcu`
plugin elsewhere — plus FlagGems itself. Without one, no `flaggems` route can
execute.

The GCU compatibility layer prepares the vendor Triton runtime but does not call
`flag_gems.enable()` to register a competing PrivateUse1 implementation. This
keeps one wrapper per overload and allows native and FlagGems RNG paths to share
the same per-device seed/offset stream.

### One stream, two producers

`topsaten` and FlagGems submit to the **same** tops stream, so a run mixes two
producers on one queue. `EXEC_TOPSATEN_CMD` submits to `GetCurrentTopsStream()`
and Triton's enflame driver reads the handle back from
`torch.gcu.current_stream(idx).gcu_stream`; measured on an S60 the two were equal
process after process.

Sharing a stream is not by itself enough to order them. A `topsaten` op that is
merely *submitted* behind an already-queued Triton kernel does not read that
kernel's output on this hardware, so a FlagGems producer followed by a `topsaten`
consumer races unless the stream is drained in between:

- `EXEC_TOPSATEN_CMD` (`csrc/aten/backends/gcu/topsaten_common.h`) synchronises
  the stream **before** issuing its op as well as after it.
- `BlockingCopyGuard::DrainCurrentQueue` (`csrc/aten/copy_ops.cc`) drains the
  same stream on GCU, so a blocking device-to-host copy waits for a FlagGems
  kernel rather than reading the buffer ahead of it.

Only the consumer side needs this. The trailing synchronise in
`EXEC_TOPSATEN_CMD` has already emptied the stream when the next FlagGems kernel
is launched, so a FlagGems op that follows a `topsaten` op is ordered without
help. The cost is close to free in the steady state for the same reason: only
work issued since the last `topsaten` op can be in flight, which is exactly the
FlagGems work the barrier exists to wait for.

The failure mode this prevents is not a stale read that looks like a small
numerical difference. Measured at `transformer_blocks.0.attn.norm_q`, the
Qwen-Image transformer's attention RMSNorm, an unordered consumer of a FlagGems
`mean` produced NaN in 131072 places (1024 rows of 128 lanes) and finite values
up to 5.66e7, and the pipeline decoded a black image.

Installation, routing, and operator-specific failure modes are documented in
[flaggems-setup.md](flaggems-setup.md); the measured per-operator results are in
[flaggems-test-results.md](flaggems-test-results.md).

## Limitations

### CI scope

[`.github/configs/gcu.yml`](../../../.github/configs/gcu.yml) runs an S60 runner against an isolated CPU-PyTorch wheel, selecting the same contract suites the other platforms run by marker rather than by a file allowlist: the operator suite twice — once on the `anyplatform`/`main_ops` cohort with the FlagGems markers excluded, once on the `flaggems and main_ops` cohort — plus the full `tests/integration/ops/test_rng_dispatch.py`, `tests/integration/test_factory_ops.py`, the shared `tests/integration/test_amp_contract.py`, the shared `tests/integration/test_math_bits_contract.py`, and `tests/integration/test_compile.py`. `.github/scripts/set_env_gcu.sh` provisions FlagTree and FlagGems into the isolated venv for both the build and integration stages, so the FlagGems cohort runs against the real Triton stack rather than being excluded. Profiler contract tests and Qwen3 smoke are not in the manifest yet; see the notes below and the comment block at the end of that file.

### Distributed support not validated

Distributed collectives are not validated on GCU hardware. The `_VENDOR_PROFILES` routing table in `torch_fl/comm/process_group.py` lists GCU as FlagCX-only unless live evidence proves otherwise.

### Profiler is runtime only; torch.compile not validated

The TOPSPTI tracer collects activities on S60, but none surface as device events on the CPU-only PyTorch/Kineto build used here, which supplies no PrivateUse1 resolver. This is an environment limitation rather than a tracer defect, so `test_profiler_contract.py` stays out of CI until it is measured end to end. `torch.compile` has not been validated on GCU.

## Build without native kernels

To build the runtime layer only (device/memory/stream support) with no native operator kernels:

```bash
FLAGOS_ACCELERATOR=gcu FLAGOS_BUILD_VENDOR=OFF pip install --no-build-isolation -v -e .
```

All compute ops will fall back to CPU. This mode is useful for testing the runtime layer in isolation.

## Reference Documentation

- [Codegen source](../../../scripts/codegen/codegen_gcu.py): category-driven kernel generation for `topsaten`
- [Compatibility matrix](../../reference/compatibility.md): platform status and limitations
- [Environment variables](../../reference/environment-variables.md): runtime environment variables and backend selection
