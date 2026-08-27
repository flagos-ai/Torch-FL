# torch-fl

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.x-orange.svg)](https://pytorch.org/)
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

- **Compatibility boxing**: Zero-copy metadata conversion into an independent PyTorch dispatch key when the vendor stack exposes one that can coexist with PrivateUse1. CUDA boxing reuses NVIDIA kernels via an external `libtorch_cuda.so`.

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
| NVIDIA CUDA | CUDA boxing over external `libtorch_cuda.so` | Eager, autograd, distributed (FlagCX/NCCL), profiler (CUPTI), FlagGems (Python + C++) | **Stable** | [CUDA](docs/vendors/cuda/installation.md) |
| MetaX | CUDA boxing via cu-bridge, or native MetaX kernels | Eager, autograd | **Stable** | [MetaX](docs/vendors/metax/installation.md) |
| Ascend | Native ACLNN backend, optional FlagGems via triton-ascend | Eager, autograd, RNG suite | **Beta** | [Ascend](docs/vendors/ascend/installation.md) |
| PPU | CUDA boxing against PPU's CUDA-13-compatible SDK | Eager, autograd | **Experimental** | [PPU](docs/vendors/ppu/installation.md) |
| Hygon DCU | CUDA boxing over hipified DTK torch | Eager, autograd, FP16/BF16 AMP, profiler | **Beta** | [DCU](docs/vendors/dcu/installation.md) |
| Kunlun P800 | CUDA boxing over the XPU CUDA-compatibility runtime | Runtime, allocation, copies, streams/events, `mm` smoke test, FP16/BF16 AMP | **Experimental** | [Kunlun](docs/vendors/kunlun/installation.md) |
| Enflame GCU | Native topsaten backend, CPU fallback for unrouted/int64 ops | Eager | **Experimental** | [GCU](docs/vendors/gcu/installation.md) |
| Moore Threads MUSA | Native mudnn backend, CPU fallback for unrouted ops | Eager | **Experimental** | [MUSA](docs/vendors/musa/installation.md) |
| D-Robotics BPU | No eager kernels; `torch.compile` graph path via hbdk4 | Graph compilation only | **Runtime only** | [BPU](docs/vendors/bpu/installation.md) |
| TsingMicro | Runtime build selector exists | Setup not documented | **Runtime only** | — |

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
| PyTorch | 2.9.x (`>=2.9,<2.10`) | Generated ATen bindings are tied to this minor line. |
| FlagGems | Platform dependent | Installed from PyPI or a vendor-compatible build only where the platform route uses it. |
| Triton/compiler | Platform dependent | Use the compiler distribution required by the selected accelerator. |

### ATen Minor-Line Pinning

torch-fl generates native bindings to PyTorch's internal ATen operator registry. These bindings are sensitive to C++ ABI and operator schema changes, so the project pins to a PyTorch minor line. The current pinned line is **2.9.x**.

Using a different PyTorch minor version (e.g., 2.10.x) will result in build or runtime failures. Patch versions within the same minor line (e.g., 2.9.0 to 2.9.1) are compatible.

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

## Documentation

### Getting Started

- [Installation](docs/getting-started/installation.md) — Platform selection and source builds
- [Quick Start](docs/getting-started/quickstart.md) — Basic usage patterns

### Reference

- [Compatibility Matrix](docs/reference/compatibility.md) — Per-platform capability validation and status
- [Operator Support](docs/reference/operator-support.md) — Measured per-hardware FlagGems overload coverage
- [Environment Variables](docs/reference/environment-variables.md) — Build and runtime configuration

### Architecture

- [Distributed Collectives](docs/architecture/distributed-flagcx.md) — ProcessGroupFlagOS, FlagCX, and vendor fallbacks
- [Profiler Integration](docs/architecture/profiler.md) — torch.profiler parity and CUPTI integration
- [torch.compile Integration](docs/architecture/torch-compile-integration.md) — Inductor GPU device registration

### Platform Guides

- [CUDA (NVIDIA)](docs/vendors/cuda/installation.md)
- [MetaX](docs/vendors/metax/installation.md)
- [Ascend (Huawei)](docs/vendors/ascend/installation.md)
- [PPU](docs/vendors/ppu/installation.md)
- [Hygon DCU](docs/vendors/dcu/installation.md)
- [Kunlun P800](docs/vendors/kunlun/installation.md)
- [Enflame GCU](docs/vendors/gcu/installation.md)
- [Moore Threads MUSA](docs/vendors/musa/installation.md)
- [D-Robotics BPU](docs/vendors/bpu/installation.md)

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
