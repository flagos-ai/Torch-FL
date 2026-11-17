# torch-fl

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.x-orange.svg)](https://pytorch.org/)
[![CI](https://github.com/flagos-ai/PyTorch-Plugin-FL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/flagos-ai/PyTorch-Plugin-FL/actions/workflows/ci.yml?query=branch%3Amain)

[Documentation](docs/) · [Installation](docs/getting-started/installation.md) · [Quick Start](docs/getting-started/quickstart.md) · [Compatibility](docs/reference/compatibility.md)

A PyTorch device plugin for the FlagOS software stack. torch-fl exposes a single `flagos` device that routes operators among reusable native kernels, portable compiler kernels, vendor-native implementations, and explicit CPU fallback.

## Overview

Accelerator vendors provide different runtimes, compiler stacks, and PyTorch integration strategies. torch-fl offers a PyTorch-native device interface that isolates these differences behind a unified runtime and operator-routing layer.

Users program against standard PyTorch APIs and the `flagos` device. The plugin selects kernel implementations per operator based on platform capabilities and configuration, without exposing vendor differences as distinct device names or requiring workload changes when moving between accelerators.

## Design Philosophy

torch-fl is built on five principles:

1. **PyTorch-native interface** — Standard PyTorch APIs work unchanged; users target the `flagos` device rather than vendor-specific extensions.

2. **One logical device** — A single device name (`flagos`) abstracts vendor differences. Platform-specific routing happens transparently at the operator level.

3. **Layered operator backends** — Each operation may dispatch to a different implementation. Routing decisions are per-operator, not per-device or per-model.

4. **Reuse before reimplementation** — Established kernels and compiler stacks are integrated where their dispatch and ABI boundaries permit, rather than rewriting functionality that already exists.

5. **Explicit capability boundaries** — Unsupported operations and CPU fallback paths are documented rather than presented as complete native coverage. Status levels distinguish validated support from experimental integrations.

### Execution Paths

torch-fl implements three principal operator execution strategies:

- **Native vendor kernels**: Direct calls into vendor runtime and operator libraries (ACLNN, mudnn, topsaten). The plugin generates bindings to each vendor's C/C++ API.

- **Compatibility boxing**: Zero-copy metadata conversion into an independent PyTorch dispatch key when the vendor stack exposes one that can coexist with PrivateUse1. CUDA boxing reuses the vendor's CUDA-ABI kernels via an external `libtorch_cuda.so`. It is an **opt-in** (`FLAGOS_BUILD_BOXING=1`): the default wheel for every CUDA-ABI platform is FlagGems-primary, so it compiles no boxing kernels and bundles no vendor libtorch.

- **Portable compiler kernels**: FlagGems kernels generated through Triton or a compatible compiler backend. These kernels target multiple accelerator families without per-platform rewrites.

A platform may combine execution paths. These are implementation strategies, not user-selectable product tiers.

## Architecture

```text
PyTorch API
    |
flagos device (PrivateUse1)
    |
device runtime + per-operator routing
    |
FlagGems/compiler kernels | compatibility boxing | native vendor kernels | CPU fallback
    |
accelerator runtime
```

This diagram is conceptual. Detailed component architecture, dispatch internals, distributed collectives, compilation integration, and profiler design are documented under [docs/architecture/](docs/architecture/).

## Capabilities

torch-fl provides project-level support for:

- PyTorch eager tensor operations and device management
- Autograd and model training
- `torch.compile` integration
- Distributed collectives and DDP through `ProcessGroupFlagOS`
- `torch.profiler` integration
- FlagGems/Triton operator integration
- Explicit CPU fallback for uncovered operations where PyTorch semantics permit

Capability availability and validation status vary by platform. A feature existing in the torch-fl codebase does not imply that every platform implements or validates it. See the [Compatibility Matrix](docs/reference/compatibility.md) for per-platform detail.

## Hardware Support

| Platform | Execution path | Validated capabilities | Status | Guide |
|---|---|---|---|---|
| NVIDIA CUDA | FlagGems-primary; CUDA boxing over an external `libtorch_cuda.so` is opt-in | Eager, autograd, distributed (FlagCX/NCCL), profiler (CUPTI), FlagGems (Python + C++) | **Stable** | [CUDA](docs/vendors/cuda/installation.md) |
| MetaX | FlagGems-primary; CUDA boxing via cu-bridge is opt-in, native MetaX kernels retired | Eager, autograd | **Stable** | [MetaX](docs/vendors/metax/installation.md) |
| Ascend | Native ACLNN backend, FlagGems via FlagTree (Triton 3.5) | Eager, autograd, RNG suite | **Beta** | [Ascend](docs/vendors/ascend/installation.md) |
| PPU | FlagGems-primary; CUDA boxing against PPU's CUDA-13-compatible SDK is opt-in | Eager, autograd | **Experimental** | [PPU](docs/vendors/ppu/installation.md) |
| Hygon DCU | FlagGems-primary; CUDA boxing over the hipified DTK torch is opt-in | Eager, autograd, FP16/BF16 AMP, profiler | **Beta** | [DCU](docs/vendors/dcu/installation.md) |
| Kunlunxin (Kunlun P800) | Native XPU backend over the vendor xdnn operator library | Eager, autograd | **Experimental** | [Kunlunxin](docs/vendors/kunlunxin/installation.md) |
| Enflame GCU | Native topsaten backend, CPU fallback for unrouted/int64 ops | Eager | **Experimental** | [GCU](docs/vendors/gcu/installation.md) |
| Moore Threads MUSA | FlagGems-first Triton kernels, native mudnn fallback, CPU fallback for unrouted ops | Eager, FP16/BF16 AMP | **Experimental** | [MUSA](docs/vendors/musa/installation.md) |
| D-Robotics BPU | No eager kernels; `torch.compile` graph path via hbdk4 | Graph compilation only | **Runtime only** | [BPU](docs/vendors/bpu/installation.md) |
| TsingMicro | FlagGems-primary over the Kuiper TX runtime; no vendor-native operator library | Eager (FlagGems routes), RNG (host-draw kernels), Qwen-Image-2.1 rotary embedding | **Experimental** | [TsingMicro](docs/vendors/tsingmicro/installation.md) |

**Status definitions:**

- **Stable**: Critical paths are continuously tested and the supported version combination is documented.
- **Beta**: The primary path is validated, but coverage, packaging, or release procedures are not yet stable.
- **Experimental**: Validation exists for a specific setup, model, or hardware environment; interfaces or build procedures may change.
- **Runtime only**: Device runtime support exists, but the platform is not a general eager operator backend.

Detailed capability breakdowns (eager execution, training, compile, distributed, profiler, FlagGems) and on-hardware validation evidence are in the [Compatibility Matrix](docs/reference/compatibility.md).

## Compatibility

| Component | Supported range | Notes |
|---|---|---|
| Python | 3.8 or later | Platform SDKs and available wheels may impose a narrower range. |
| PyTorch | 2.10.x (`>=2.10,<2.11`) | Generated ATen bindings are tied to this minor line. |
| FlagGems | Platform dependent | Installed from PyPI or a vendor-compatible build only where the platform route uses it. |
| Triton/compiler | Platform dependent | Use the compiler distribution required by the selected accelerator. |

### ATen Minor-Line Pinning

torch-fl generates native bindings to PyTorch's internal ATen operator registry. These bindings are sensitive to C++ ABI and operator schema changes, so the project pins to a PyTorch minor line. The current pinned line is **2.10.x**.

Using a different PyTorch minor version (e.g., 2.11.x) will result in build or runtime failures. Patch versions within the same minor line (e.g., 2.10.0 to 2.10.1) are compatible.

Vendor SDK versions, CUDA toolkit versions, and other platform-specific requirements are documented in each platform's installation guide.

## Quick Start

Choose your platform in the [Installation Guide](docs/getting-started/installation.md), then try the following:

```python
import torch
import torch_fl

# Create a tensor on the flagos device
x = torch.randn(4, 4, device="flagos:0")

# Operations route to platform-appropriate kernels
y = torch.relu(x @ x)

# Move result back to CPU
print(y.cpu())
```

Operator routing (FlagGems, vendor kernels, compatibility boxing, or CPU fallback) is determined by platform detection and runtime configuration. The code above works unchanged across all supported accelerators.

For device queries, synchronization, and multi-device usage patterns, see the [Quick Start Guide](docs/getting-started/quickstart.md).

### Experimental native FlagGems hosts

To compile source-derived hosts into a NVIDIA CPU-Torch wheel, build with the
published, pinned FlagGems/FlagTree wheels already installed:

```bash
FLAGOS_ACCELERATOR=nvidia FLAGOS_BUILD_NATIVE_HOST=1 \
FLAGOS_BUILD_FLAGGEMS=1 FLAGOS_BUILD_FLAGGEMS_CPP=0 \
FLAGOS_BUILD_TILEOPS=0 FLAGOS_BUILD_BOXING=0 python setup.py bdist_wheel
```

This opt-in build uses Cython 3.2.4 and a CUDA toolkit. The generated hosts and
the common launcher are installed with the wheel and activated on import.
Eager inference uses the generated operator hosts under `torch.no_grad()`.
Generated ATen entries bind directly to matching typed hosts at import time;
warm admitted calls avoid generic argument packing and host-name lookup.
Tensor and fixed Tensor-tuple results, scalar/optional/list inputs and existing
out/in-place adapters are supported. Other signatures retain the original
FlagGems path and are reported as compiler gaps, not vendor coverage.
Existing NVIDIA device-copy plumbing also uses a generated source bridge, so
supported casts can reach the same typed hosts without generic argument packing.
Supported scalar rebindings preserve Python bool/int/float types across branches,
including Optional defaults; unsafe constructs retain the original host path.
Guarded exception helpers keep supported prefixes native and run only the reached
original tail; existing metadata regions retain their native remainder.
The acceptance target is synchronized complete-model latency at most 30 ms
against the roughly 24 ms vendor workload, with the model remaining eager.
Logit acceptance separately requires elementwise `atol=rtol=1e-3` against the
accelerator manufacturer's native Torch at the same precision. The unchanged
#839 BF16 cohort fails this gate; the optional precision cohort below passes it.
Use the strict-eager benchmark's
`--check-logits --reference <vendor-reference.pt>` to enforce it; see the
[paired validation procedure](docs/design/flaggems-build-time-hosts.md#vendor-logit-acceptance).

An optional native-host build adds measured vendor precision exceptions. Set
`FLAGOS_VENDOR_TORCH_LIB` to the `torch/lib` directory from the published vendor
Torch wheel when running the build command above. The build validates the exact
cohort, bundles its numerical libraries and generates typed eager entries from
ATen schemas. The policy currently covers FP32 power, mean, reciprocal square
root and softmax, plus BF16 matrix products on NVIDIA; other routes remain
FlagGems. No model-code change, Dynamo capture or tensor boxing is involved.
The generated conf states the dtype conditions explicitly, such as
`mm = flaggems; BFloat16:nvidia`; registering a vendor entry cannot override
a plain FlagGems route or an explicit FlagGems environment override.
This cohort also selects an unchanged published FlagGems pointwise host for
measured `mul.Tensor` dtypes through the same build-time compiler.
On H100, two independent installed-wheel Qwen3-0.6B BF16 runs pass all prefill
positions and 512 decode steps with maximum logit error 0. Synchronized
prefill/decode medians are **29.15/27.34 ms** and **29.24/27.28 ms**, versus
**25.09/23.40 ms** for paired vendor Torch. These are complete forward timings
with identical teacher-forced inputs; loading, tuning and sampling are excluded
on both sides. See the [recorded acceptance evidence in issue #840](https://github.com/flagos-ai/Torch-FL/issues/840).
See the [precision policy](docs/design/flaggems-build-time-hosts.md#native-vendor-precision-exceptions)
for cohort requirements, audit evidence and limitations.

Native-host builds write `native_host/_generated/typed/coverage.json`, accounting
for every FlagGems registry row, excluded signatures and lowering outcomes.
Its counts describe build coverage, not operator correctness or performance.
The [generalization project](https://github.com/orgs/flagos-ai/projects/14)
tracks broader eager and accelerator support. Domestic accelerator acceptance
compares against the manufacturer's native PyTorch on the same card and workload:
at least 80% of its throughput, or at most 1.25 times its latency for fixed work.
The shared host launcher uses a versioned backend contract. NVIDIA and the
experimental MetaX and Ascend adapters are available; other vendors require separate validation. New native-host
wheels fingerprint their adapter, published compiler driver and native ABI before
activation. See the [backend launch contract](docs/design/flaggems-build-time-hosts.md#backend-launch-contract)
for its boundary and wheel preflight command.

For the MetaX pilot, use the same native-host build switches with
`FLAGOS_ACCELERATOR=metax` and `MACA_PATH` pointing at the installed MACA SDK.
The adapter submits the original published MetaX FlagGems kernels directly through
MACA, with boxing disabled. MetaX CI builds a separate opt-in wheel and retains
its generated coverage, preflight and launch-conformance evidence. Full-model
accuracy and the 80% vendor-performance gate remain tracked in
[issue #832](https://github.com/flagos-ai/Torch-FL/issues/832); adapter availability
does not establish those results. NVIDIA's existing precision policy is unchanged.

The Ascend pilot uses the same native-host switches with
`FLAGOS_ACCELERATOR=ascend` and `ASCEND_HOME` pointing at the installed CANN SDK.
It builds the original published FlagTree module-loader utility and submits
original FlagGems kernels through the Ascend runtime, without loading the real
`torch_npu` extension or using boxing. Ascend CI builds a separate opt-in wheel
after testing the ordinary wheel and retains coverage and conformance artifacts.
Full-model accuracy and the 80% vendor-performance gate remain open in
[issue #833](https://github.com/flagos-ai/Torch-FL/issues/833); build coverage is
not evidence of those results. NVIDIA's existing precision policy is unchanged.

An optional graph-mode comparison uses Dynamo to capture the model and
TorchScript to execute its ATen graph:

```python
model = torch_fl.native_host.compile(model.eval().to("flagos"))
with torch.no_grad():
    output = model(inputs)
```

This backend runs the graph through ATen and FlagGems with numerical fusers
disabled. It generates no Inductor or replacement Triton kernels. Runtime
operator misses compile only original FlagGems device kernels, including its
pointwise generator's kernels; they do not compile host C/C++ libraries.
Kernel provenance is checked and CPU fallback is rejected. Its measured
approximately 26 ms latency depends on model graph capture and does not satisfy
the eager acceptance requirement. Eager and graph-mode results are reported
separately.
See the
[build-time host design and validation](docs/design/flaggems-build-time-hosts.md)
for the current H100 evidence, compatibility requirements and performance tradeoffs.

The earlier capture-based experiment remains available on ordinary wheels:

The NVIDIA CPU-Torch build can generate native host programs from the installed
FlagGems implementations. The model remains ordinary eager Python:

```python
from torch_fl import host_compiler

host_compiler.enable("/path/to/host-program-cache")
with torch.no_grad():
    output = model(inputs)
```

This experimental path requires PyTorch 2.10.0, the NVIDIA toolkit and a C++
compiler. It is currently validated on H100 inference. Initial compilation is
expensive: a separate process captures metadata, compiles the original Triton
kernels and validates outputs before admission. Warm calls use native guards,
allocation and launches; guard misses keep the original FlagGems route. Use
`mode="load"` to prohibit further compilation, or `host_compiler.disable()` to
disable native lookup. Programs are bounded and remain resident until process
exit. Use a fresh cache after changing dependencies or the host compiler.

See the [native host design](docs/design/flaggems-native-host-programs.md)
and `benchmarks/host_compiler/qwen_eager.py` for full-model validation and its
limitations. The feature does not enable boxing or replace uncovered operations
with Inductor-generated kernels.

## Documentation

### Getting Started

- [Installation](docs/getting-started/installation.md) — Platform selection and source builds
- [Quick Start](docs/getting-started/quickstart.md) — Basic usage patterns

### Reference

- [Compatibility Matrix](docs/reference/compatibility.md) — Per-platform capability validation and status
- [Operator Support](docs/reference/operator-support.md) — Per-platform routed-operator support counts
- [Environment Variables](docs/reference/environment-variables.md) — Build and runtime configuration

### Architecture

- [Startup Lifecycle](docs/architecture/startup-lifecycle.md) — Mandatory bootstrap and explicit framework integration activation
- [Distributed Collectives](docs/architecture/distributed-flagcx.md) — ProcessGroupFlagOS, FlagCX, and vendor fallbacks
- [Profiler Integration](docs/architecture/profiler.md) — torch.profiler parity and CUPTI integration
- [torch.compile Integration](docs/architecture/torch-compile-integration.md) — Inductor GPU device registration

### Platform Guides

- [CUDA (NVIDIA)](docs/vendors/cuda/installation.md)
- [MetaX](docs/vendors/metax/installation.md)
- [Ascend (Huawei)](docs/vendors/ascend/installation.md)
- [PPU](docs/vendors/ppu/installation.md)
- [Hygon DCU](docs/vendors/dcu/installation.md)
- [Kunlunxin (Kunlun P800)](docs/vendors/kunlunxin/installation.md)
- [Enflame GCU](docs/vendors/gcu/installation.md)
- [Moore Threads MUSA](docs/vendors/musa/installation.md)
- [D-Robotics BPU](docs/vendors/bpu/installation.md)

### Development

- [Releasing](docs/development/releasing.md) — What a release tag builds and where each wheel is published
- [Testing](docs/development/testing.md) — Test organization, marks, and how to run each suite
- [Integration Harness](docs/development/integration-harness-design.md) — How the per-platform integration manifests are structured

## Contributing

Contributions are welcome in these areas:

- **Operators**: Add missing operators or optimize existing implementations for specific backends.
- **Runtime and platform integration**: Port torch-fl to new accelerators or improve existing backend support.
- **Compiler integration**: Extend torch.compile support, improve Triton codegen, or add new compilation backends.
- **Distributed and profiler work**: Enhance collective communication backends or profiling integration.
- **Tests**: Add operator correctness tests, model integration tests, or performance benchmarks.
- **Documentation**: Improve setup guides, troubleshooting docs, or vendor-specific integration notes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, code generation procedures, testing requirements, and PR guidelines.

**All GitHub-facing text must be written in English** — PR titles, PR descriptions, commit messages, issue text, and code review comments. This repository's code, comments, and history are in English, and PRs are reviewed by contributors who do not read other languages.

## Acknowledgements

torch-fl builds on several upstream projects:

- **PyTorch** — The PrivateUse1 extension mechanism and ATen operator surface
- **FlagGems** — Triton-based portable operator kernels
- **Triton** (via FlagTree and vendor distributions) — GPU kernel compilation infrastructure
- **FlagCX** — Heterogeneous collective communication library
- **Vendor runtime and operator libraries** — ACLNN (Ascend), mudnn (MUSA), topsaten (GCU), cu-bridge (MetaX), DTK (DCU), and others

These acknowledgements do not imply endorsement by the named projects or vendors.

## License

torch-fl is licensed under the [Apache License 2.0](LICENSE).
