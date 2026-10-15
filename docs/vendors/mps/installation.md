# Apple Silicon (M-series) Installation

The MPS platform targets Apple Silicon through Metal. It currently ships the
**device runtime floor only** -- real device memory, real Metal streams and
events, host/device copies and synchronization -- with **no operator claimed
yet**, so every aten call reaches `cpu_fallback`. See
[Current status](#current-status) below before expecting GPU acceleration.

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
| `FLAGOS_BUILD_VENDOR` | on | The native operator tree lives in `csrc/aten/backends/mps/`; it is a no-op until that directory exists. |
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
```

The stage-0 gate -- allocate, host/device copy, stream and event
synchronization, allocator reuse -- is exercised by:

```bash
pytest tests/integration/test_mps_runtime.py -q
```

Every case in that file self-skips off an MPS build, so a broad
`pytest tests/integration` collection on a non-Apple machine stays quiet.

## Current status

This is **stage 0** of
`docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md`: the
28-function C ABI plus the allocator, with zero aten ops claimed on
`PrivateUse1`. `torch_fl/configs/backends_mps.conf` is intentionally empty and
`csrc/aten/register.cc`'s `USE_MPS` arm claims nothing, which is what routes
every op to `cpu_fallback`. `torch_fl/accelerator/mps/` holds only
`mps_stream.py`, the stream/event wrappers `torch.flagos.Stream` dispatches to.

Expect a working device with correct results and **no GPU acceleration**. A
platform whose GPU never runs a kernel is a "Runtime only" row, not operator
support.

## Next: the operator layer

The accepted strategy is MPSGraph-first with a hand-written MSL fallback
(design doc section 6), delivered by category starting from `unary`. The
build-time answers a first kernel needs were measured on an M5 (macOS 26,
2026-10-15):

| Question | Answer |
|---|---|
| Tensor descriptor | `MTLBuffer` (from the flagos allocator) wrapped in `MPSGraphTensorData` via `initWithMTLBuffer:shape:dataType:`. There is no vendor tensor struct. |
| Call convention | Single call. Build a one-node `MPSGraph`, then `runWithFeeds:targetTensors:targetOperations:`, which blocks and returns results synchronized to CPU memory. |
| Stream hook | `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` takes the flagos `MTLCommandQueue`, which is how a kernel runs on the flagos stream. |
| Output ownership | The caller. Allocate the output through the flagos allocator and hand the graph a destination `MTLBuffer`, or read the returned `MPSGraphTensorData` back through `[data mpsndarray]` + `readBytes:strideBytes:`. |
| dtype enum | `torch.float32` -> `MPSDataTypeFloat32`; the rest of the map is enumerated as the `unary` category widens, and a missing dtype must raise. |

A float32 `absoluteWithTensor:` over `[-3.5, 0, 2.25, -0.125]` was measured to
return `[3.5, 0, 2.25, 0.125]` through a shared `MTLBuffer`, so the graph
round trip is confirmed before any wrapper is written.

The bring-up order, the category system and the per-op CPU-comparison
requirement are in `.claude/skills/native-op-backend/SKILL.md`.
