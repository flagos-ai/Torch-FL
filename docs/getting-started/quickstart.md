# Quick Start Guide

This guide shows platform-independent usage patterns for `torch_fl`. After installing for your platform, these examples work the same way across all supported accelerators.

## Basic Usage

```python
import torch
import torch_fl

x = torch.randn(4, 4, device="flagos:0")
y = torch.relu(x @ x)
print(y.cpu())
```

This creates tensors on the first `flagos` device, performs a matrix multiplication and ReLU activation, and moves the result back to CPU for printing.

## Moving Tensors Between Devices

Move tensors to the accelerator with `.to()`:

```python
import torch
import torch_fl

x = torch.randn(4, 4)  # CPU tensor
x_flagos = x.to("flagos")  # Move to flagos device 0
x_flagos_1 = x.to("flagos:1")  # Move to flagos device 1
```

Move tensors back to CPU:

```python
y = torch.randn(4, 4, device="flagos")
y_cpu = y.cpu()  # Move back to CPU
```

## Selecting a Device

Use `torch.flagos.device()` to set the current device context:

```python
import torch
import torch_fl

with torch.flagos.device(0):
    x = torch.randn(4, 4, device="flagos")  # Created on device 0

with torch.flagos.device(1):
    y = torch.randn(4, 4, device="flagos")  # Created on device 1
```

## Synchronization

Wait for all operations on the current device to complete:

```python
import torch
import torch_fl

x = torch.randn(1000, 1000, device="flagos")
y = x @ x  # Asynchronous operation
torch.flagos.synchronize()  # Wait for completion
```

## Device Queries

Check device availability and count:

```python
import torch
import torch_fl

# Check if any flagos devices are available
if torch.flagos.is_available():
    print(f"Found {torch.flagos.device_count()} device(s)")
    print(f"Current device: {torch.flagos.current_device()}")
else:
    print("No flagos devices available")
```

Query device properties:

```python
import torch
import torch_fl

if torch.flagos.is_available():
    props = torch.flagos.get_device_properties(0)
    print(f"Device name: {props.name}")
    print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
```

## Operator Routing

Operations on `flagos` tensors are routed to different kernel implementations depending on platform and configuration:

- **Portable compiler kernels**: FlagGems Triton kernels that work across platforms (when enabled)
- **Native vendor kernels**: Platform-specific optimized implementations (e.g., ACLNN for Ascend, mudnn for MUSA)
- **Compatibility boxing**: Generated wrapper kernels that delegate to an external vendor backend (e.g., CUDA boxing)
- **CPU fallback**: Operations without a device kernel fall back to CPU (documented per-platform; not all platforms use this)

The routing is transparent to your code. No changes are needed when switching between platforms or kernel sources.

## Next Steps

- Explore the [Compatibility Matrix](../reference/compatibility.md) to understand which features are validated for your platform
- Review platform-specific notes in your [installation guide](installation.md)
- Learn about distributed training in the [Distributed Guide](../architecture/distributed-flagcx.md)
- Profile your workload with the [Profiler Guide](../architecture/profiler.md)
