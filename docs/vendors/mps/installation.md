# Apple Silicon (M-series) Installation

The MPS platform targets Apple Silicon through Metal. It ships the **device
runtime floor** -- real device memory, real Metal streams and events,
host/device copies and synchronization -- plus the first two stages of the
MPSGraph operator layer, so a small set of aten ops runs on the GPU and
everything else reaches `cpu_fallback`. See [Current status](#current-status)
below for what that set is.

## Supported target

- **Hardware:** any Apple Silicon Mac whose `MTLCreateSystemDefaultDevice()`
  returns an Apple GPU (M1 through M5; one device, index 0)
- **Operating system:** macOS
- **Architecture:** arm64 (the wheel built here is `macosx_*_arm64`)
- **PyTorch:** 2.10.x (`>=2.10,<2.11`), the stock macOS arm64 wheel
- **Python:** any version that PyTorch publishes a macOS arm64 wheel for. There
  is no FlagTree build for Metal, so nothing pins the interpreter to a vendor
  cp tag.

## Common requirements

The Xcode Command Line Tools provide `clang` and the Metal SDK
(`Metal.framework`, `MetalPerformanceShadersGraph.framework`). Verify that the
SDK is present -- the build fails at the cmake configure stage if it is not:

```bash
xcrun --show-sdk-path                       # the SDK root
ls "$(xcrun --show-sdk-path)/System/Library/Frameworks/Metal.framework"
```

## Installation

Install PyTorch from the CPU index, then build `torch_fl` with the MPS
accelerator:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
FLAGOS_ACCELERATOR=mps pip install --no-build-isolation -e .
```

There is **no vendor SDK to point at** and **no package-index lane**: Apple
supplies the Metal stack with the operating system, and no FlagOS lane publishes
an Apple wheel. The MPS wheel therefore declares only `torch` and `packaging`
-- no `flagtree`, no `flag_gems`, no `flagcx`. Declaring Triton would be
unsatisfiable, because it publishes no macOS wheel.

## What the build does and does not compile

The kernel switches are fixed for this platform in `cmake/flagos_platforms.json`
and are not per-build choices:

| Switch | Value | Why |
|---|---|---|
| `FLAGOS_BUILD_VENDOR` | on | The native operator tree lives in `csrc/aten/backends/mps/` and links `-framework MetalPerformanceShadersGraph` for the MPSGraph kernels. |
| `FLAGOS_BUILD_BOXING` | **off** | This is a native backend. It never re-tags a `flagos` tensor onto `DispatchKey::MPS` or `DispatchKey::CUDA` and never builds the CUDA boxing kernels. |
| `FLAGOS_BUILD_FLAGGEMS` | **off** | No FlagGems/Triton line targets Metal. |
| `FLAGOS_BUILD_FLAGGEMS_CPP` | **off** | No `liboperators.so` exists for the Metal stack. |
| `FLAGOS_BUILD_TILEOPS` | **off** | No TileOps line for Metal. |

The device runtime is Objective-C++ (`csrc/runtime/accelerator/mps/*.mm`)
because Metal is an Objective-C API; the runtime links `-framework Metal` and
`-framework Foundation`. Metal is also why the `mps` entry is the only platform
whose `project_languages` includes `OBJCXX`.

## Verification

Import and check the device runtime:

```python
import torch, torch_fl
from torch_fl import flagos

print(flagos.device_count())     # 1 on an Apple GPU, 0 without one
print(flagos.is_available())

x = torch.empty(4, 4, device="flagos:0")
print(x.device.type)             # flagos

y = torch.arange(16, dtype=torch.float32).reshape(4, 4).to("flagos:0")
print(y.cpu())                   # host/device copy round trip

z = torch.tensor([-3.5, 0.0, 2.25], device="flagos:0")
print(torch.abs(z).cpu())        # the MPSGraph operator layer
```

The stage-0 gate -- allocate, host/device copy, stream and event
synchronization, allocator reuse -- is exercised by:

```bash
pytest tests/integration/test_mps_runtime.py -q
```

Every case in that file self-skips off an MPS build, so a broad
`pytest tests/integration` collection on a non-Apple machine stays quiet.

## Current status

Stages 0-2 of
`docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md` have
landed: the 28-function C ABI plus the allocator, and the `unary`, `binary`,
`binary_scalar` and `reduce` categories of the MPSGraph operator layer -- nine
aten ops: `abs`; `add`/`sub`/`mul` over two tensors and over a tensor and a host
scalar (with aten's `alpha`); and `sum.dim_IntList`/`mean.dim`. Those nine are
claimed on the `PrivateUse1` `mps` slot by `csrc/aten/register.cc`, and
`torch_fl/configs/backends_mps.conf` routes exactly them to `mps` and every
other op to `none`, which is what sends the rest to `cpu_fallback`. The routes
are generated from `csrc/aten/backends/mps/generated/mps_register.inc`, so the
conf cannot name an op the operator library does not register.
`torch_fl/accelerator/mps/` holds `mps_stream.py`, the stream/event wrappers
`torch.flagos.Stream` dispatches to.

Results are correct on every op: the accelerated nine are compared against the
CPU reference in `tests/integration/ops/test_mps_unary.py`,
`test_mps_binary.py` and `test_mps_reduce.py`, and a case their kernels decline
takes the CPU round-trip rather than a wrong answer. Stage 3 (gemm/matmul, conv,
softmax -- the lanes a small model needs) is the next increment. A platform
whose GPU never runs a kernel is a "Runtime only" row, not operator support;
nine ops is operator support for the unary, elementwise and reduction families
and nothing more.

## MPSGraph answers measured on an M5

The accepted strategy is MPSGraph-first with a hand-written MSL fallback
(design doc section 6), delivered by category. The build-time answers the
operator layer rests on were measured on an M5 (macOS 26, 2026-10-15):

| Question | Answer |
|---|---|
| Tensor descriptor | `MTLBuffer` (from the flagos allocator) wrapped in `MPSGraphTensorData` via `initWithMTLBuffer:shape:dataType:`. There is no vendor tensor struct. |
| Call convention | Single call. Build a one-node `MPSGraph`, then `runWithFeeds:targetTensors:targetOperations:`, which blocks and returns results synchronized to CPU memory. |
| Stream hook | `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` takes the flagos `MTLCommandQueue`, which is how a kernel runs on the flagos stream. |
| Output ownership | The caller. Allocate the output through the flagos allocator and hand the graph a destination `MTLBuffer`, or read the returned `MPSGraphTensorData` back through `[data mpsndarray]` + `readBytes:strideBytes:`. |
| dtype enum | `MPSDataTypeOf` maps the eight torch dtypes MPSGraph has an element type for; `float64` and the complex pair have none, and a kernel declines them rather than guessing at a widening. |

A float32 `absoluteWithTensor:` over `[-3.5, 0, 2.25, -0.125]` was measured to
return `[3.5, 0, 2.25, 0.125]` through a shared `MTLBuffer`, so the graph
round trip was confirmed before any wrapper was written.

The bring-up order, the category system and the per-op CPU-comparison
requirement are in `.claude/skills/native-op-backend/SKILL.md`.
