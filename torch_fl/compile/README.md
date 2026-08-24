# torch.compile Integration for FlagOS

This directory contains the torch.compile backend integration for the flagos device.

## Overview

The `flagos` backend registers flagos with TorchInductor as a **first-class GPU
device**, then hands the traced graph to `compile_fx` unchanged. Inductor
generates Triton kernels that operate on flagos tensors directly -- there is no
conversion to cuda and no copy at the graph boundary.

This works because flagos runs on the physical GPU that `torch.cuda` describes:
its allocator delegates to `c10::cuda::CUDACachingAllocator`, so a flagos
tensor's storage already *is* CUDA memory, and device indices line up
(`flagos.set_device(i)` moves the CUDA current device).

## Usage

```python
import torch_fl  # Import first on MetaX.
import torch

def my_model(x):
    z = x + 1.0
    z = torch.nn.functional.relu(z)
    z = z * 2.0
    return z

x = torch.randn(4096, 4096, device='flagos:0')

# Compile with flagos backend
compiled_model = torch.compile(my_model, backend='flagos')
result = compiled_model(x)

# mode / options are forwarded to inductor as config patches
compiled_model = torch.compile(my_model, backend='flagos', mode='max-autotune')
```

## Implementation Notes

### Why the graph stays on flagos

An earlier version rewrote the graph and its example inputs to cuda before
calling `compile_fx`. That is not merely a copy cost -- it breaks backward.
`at::getAccelerator()` is PrivateUse1/flagos in this build, and
`torch::autograd::Node::stream()` only yields a stream when a node's input
device type equals the accelerator. A cuda-rewritten graph therefore produces
stream-less autograd nodes, and AOT autograd's backward trace inside
`compile_fx` trips `opt_ready_stream && opt_parent_stream` (engine.cpp:1085).

Keeping the graph on flagos avoids that, and removes a copy-in/copy-out per call.

### Registration surface (`device_interface.py`, `inductor_codegen.py`)

| What | Why |
|---|---|
| `GPU_TYPES.append("flagos")` | `is_gpu()` is a membership test on this list; without it inductor picks the C++/CPU codegen path and never emits Triton. Must be in place -- callers captured the list object at import. |
| Prime `get_gpu_type()`'s cache | It asserts at most one GPU type is available, and the torch.cuda shim reports available alongside flagos. |
| `register_interface_for_device` | Inductor's `DeviceInterface`: device state from `torch.flagos`, hardware properties from `torch.cuda` (same GPU). |
| `DeviceProperties.create` wrap | Reports the underlying compiler target at the Triton boundary: `cuda` for NVIDIA or `maca` for MetaX. A literal `flagos` target finds no compatible Triton backend. |
| `register_device_op_overrides` | Device guard / stream / synchronize snippets spliced into generated code. Inherits the CUDA ones; only Python-level device manipulation routes through `torch.flagos`. |
| `register_backend_for_device` | Scheduling + wrapper codegen -- the stock CUDA/Triton pipeline under the `"flagos"` key. |

The four codegen classes are also published on `torch.flagos` so inductor's
official PrivateUse1 hook (`init_backend_registration`, `codegen/common.py:578`)
can register flagos on its own.

### CPU-torch wheel accommodations

This build pairs a CPU-only pip torch with an externally supplied
`libtorch_cuda.so`, so several `torch.cuda` Python bindings are missing. The
backend compensates:

- `use_static_cuda_launcher = False` -- `torch._C._StaticCudaLauncher` is not built.
- `triton.cudagraphs = False` -- `torch.cuda.CUDAGraph` is a dummy base class
  that raises on construction; `mode="max-autotune"` would otherwise enable it.
- `CudaInterface.get_raw_stream` is re-attached -- the binding exists, but the
  import-time `torch.cuda._is_compiled()` probe left it at `None`.

See `torch_fl/accelerator/cuda/_cuda_compat.py` and
`torch_fl/accelerator/metax/_metax_compat.py` for the memory-stats and
Event/Stream shims that inductor's autotuner needs.

## AMP and Dtype Support

The flagos device implements PyTorch's standard `AutocastPrivateUse1` policies.
Autocast supports `torch.float16` and `torch.bfloat16` targets:

```python
model = torch.nn.Linear(512, 512, device="flagos:0")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("flagos")

inputs = torch.randn(8, 512, device="flagos:0")
target = torch.randn(8, 512, device="flagos:0")
with torch.autocast("flagos", dtype=torch.bfloat16):
    output = model(inputs)
    loss = torch.nn.functional.mse_loss(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Matrix operations such as `mm`, `linear`, and `addmm` use the selected
lower-precision dtype. Numerically sensitive operations covered by the standard
policy list, such as `softmax`, normalization, and `mse_loss`, run in float32.
Operations outside that list, such as binary cross entropy, use backend
fallthrough. Explicit `dtype=` arguments
remain authoritative for policy wrappers. Unsupported autocast targets such as
float32 and float64 are disabled by PyTorch with its standard warning.

Outside autocast, the backend preserves the dtypes exercised by the current
contract: float16, bfloat16, float32, float64, int64, and bool. Complex and
float8 behavior is not part of this tested contract. AMP and dtype behavior was
validated on NVIDIA H800/sm90; other vendor backends require their own
validation.

## Environment Variables

- `FLAGOS_USE_FLAGTREE=1` - Assert that the active `triton` is FlagTree, erroring
  out if it is stock Triton. FlagTree replaces `triton` at install time, so this
  can only check; it cannot switch anything on. See
  [`docs/architecture/torch-compile-integration.md`](../../docs/architecture/torch-compile-integration.md).
- `FLAGOS_COMPILE_FALLBACK_EAGER=1` - Fall back to eager mode on compile errors

## Limitations

1. Single device - multi-GPU compilation not yet exercised
2. FlagTree is validated on NVIDIA `sm90`, Hygon `gfx936` with the HCU
   backend, and MetaX C550 with MACA 3.8.0; other vendor backends remain
   untested here

## Future Work

- [x] Exercise `torch.compile(backend="flagos")` on FlagTree-built NVIDIA,
      Hygon HCU, and MetaX environments
- [ ] Benchmark fusion gains against stock inductor+triton on cuda
- [ ] Multi-GPU compilation support
