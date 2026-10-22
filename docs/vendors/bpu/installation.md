# BPU (D-Robotics RDK) Installation

The BPU platform provides **runtime acceleration only** — native device memory and streams, graph-level compilation through `torch.compile`, and a separate prebuilt-HBM LLM runtime. Eager operators execute on the CPU via fallback.

## Supported Target

- **Hardware:** D-Robotics RDK board with Horizon BPU (nash-p, nash-e, or nash-m)
- **PyTorch:** `torch==2.10.0+cpu` (aarch64 cp314 wheel)
- **Python:** 3.14 aarch64
- **Runtime:** `libhbucp.so` and `libbpu.so` at `/usr/hobot/lib` (shipped in board image)

The torch version pin is enforced: checked-in ATen bindings target the 2.10 series (`torch>=2.10,<2.11`), and a newer torch fails at build time with signature mismatches.

## Installation

Install the upstream CPU torch wheel, then build torch_fl with the BPU accelerator:

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
FLAGOS_ACCELERATOR=bpu pip install --no-build-isolation -e .
```

No SDK root configuration is needed — the runtime ships at standard system paths.

## Execution Modes

The BPU backend provides three distinct execution paths:

1. **Eager CPU fallback:** Individual operators on flagos tensors execute on the CPU. No per-operator BPU kernels exist because the BPU executes whole compiled graphs, not single ops.

2. **Graph acceleration via `torch.compile`:** Traced graphs compile to `.hbm` artifacts through hbdk4 and execute on the BPU. Quantization (int8 Q/DQ insertion) is applied by default to move convolution and pooling work onto the device.

   ```python
   import torch, torch_fl
   
   model = MyNet().eval()
   compiled = torch.compile(model, backend="bpu")
   out = compiled(torch.randn(1, 3, 224, 224))
   ```

3. **Prebuilt-HBM LLM runtime:** `torch_fl.accelerator.bpu.infer` drives vendor-supplied `.hbm` artifacts with device-resident buffers and a sliding-window KV cache, bypassing `torch.compile` entirely.

## Compiler Setup (Optional)

Graph compilation requires hbdk4, which ships x86_64-only wheels. On-board compilation runs under box64 emulation:

```bash
scripts/vendor/setup_bpu_hbdk4.sh --wheels /path/to/oe/wheels
export FLAGOS_BPU_X86_PYTHON=~/hbdk4-x86/python/bin/python3.11
export FLAGOS_BPU_X86_EMULATOR=~/hbdk4-x86/bin/box64
```

The wheels are in the D-Robotics OE package (`oe-package-*.tgz` on `ftp://oeftp@sdk.d-robotics.cc/`). The script needs `hbdk4_compiler-*cp311*` and `hbdk4_march-*`.

**Without a reachable hbdk4**, the backend logs a warning and executes every partition on the CPU — the install remains usable, just without BPU acceleration.

Verify compiler availability:

```bash
python -c "from torch_fl.accelerator.bpu.compiler import find_hbdk; print(find_hbdk())"
# -> x86-emul (if box64 setup succeeded)
# -> cli (if native x86 python with hbdk4 is on PATH)
# -> unavailable (hbdk4 not found; graphs will run on CPU)
```

## Verification

Basic import and device runtime:

```python
import torch, torch_fl

# Device runtime is present
assert torch.utils.backend_registration.is_registered_backend("flagos")
print(torch.cuda.device_count())  # 1 if BPU runtime is available

# Create a flagos tensor and verify it runs (on CPU, eagerly)
x = torch.randn(3, 3, device="flagos")
y = x + x
print(y.device)  # flagos:0
```

Focused unit tests covering device memory, streams, the `cuda` alias, weight freezing, partitioning, Q/DQ insertion, and the KV cache layout:

```bash
pytest tests/unit/bpu/ -v
```

Graph compilation (requires hbdk4 setup):

```bash
python benchmarks/bpu_resnet18_bench.py  # first run compiles (~25 min), then cached
```

## Further Documentation

**For architecture details, partitioning strategy, quantization, calibration, zero-copy paths, fixed-shape constraints, benchmark results, and LLM runtime internals**, see [integration.md](integration.md).

That document covers:
- Why there are no per-op kernels
- The full compile pipeline (Dynamo → AOTAutograd → ONNX → hbdk4)
- Why quantization is a precondition rather than an optimization
- On-board compilation with box64 (why source-built box64 is required, the `libhbtl.so` preload, numba/torch stubs)
- Calibration for better int8 accuracy
- Runtime notes (host-mapped device memory, synchronous execution, the cached-block leak at exit, FX/Dynamo identity fixes for the `cuda` alias)
- Measured performance against D-Robotics' own artifacts
- Stock transformers under `torch.compile` (correct but not yet fast; coverage capped by symbolic shapes)
- Known limitations (single device, int8 only, synchronous, prebuilt LLM path separate from torch.compile)

Environment variables, cache layout, and the sliding-window KV cache geometry are all documented there.
