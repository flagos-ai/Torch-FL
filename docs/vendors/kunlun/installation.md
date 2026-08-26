# Kunlun P800 Installation Guide

## Overview

Kunlun P800 uses the CUDA-compatible boxing route on the PyTorch 2.9 branch. The
P800 XPU SDK exposes CUDA-shaped headers and a `libcudart.so.12` compatibility
runtime, while the Kunlun vendor torch registers its device kernels under the
CUDA dispatch key. torch-fl therefore reuses the generated
PrivateUse1-to-CUDA boxing kernels and the existing CUDA-shaped runtime wrapper;
it does not add handwritten per-operator kernels.

**Status:** Experimental. The measured scope currently covers device discovery,
allocation, host/device copies, streams, events, multi-device selection, cache
release, and a boxed `mm` smoke test on a P800 OAM. Full operator coverage and
FlagGems support are not claimed by this guide.

## Prerequisites

- Kunlun P800 hardware and XPU-RT driver
- Kunlun XPU SDK, normally `/usr/local/xpu` or `/usr/local/xpu-5.37.1.0`
- XPU CUDA compatibility runtime, normally `/usr/local/xcudart`
- Kunlun vendor PyTorch 2.9 (`torch==2.9.0+cu129` in the measured environment)
- Python, CMake, and a host C++ compiler compatible with the vendor torch
- `XPU_ENABLE_PROFILER_TRACING=1` during import on XPU-RT 5.37.1

The Python build environment must use the vendor torch. Do not replace it with
an NVIDIA CUDA wheel: the vendor CUDA registrations and XPU runtime are the
measured operator path.

## Build From Source

```bash
export XPU_ROOT=/usr/local/xpu
export XCUDART_ROOT=/usr/local/xcudart
export ACCELERATOR=kunlun
export FLAGOS_BUILD_JOBS=4

ACCELERATOR=kunlun \
  XPU_ROOT="$XPU_ROOT" \
  XCUDART_ROOT="$XCUDART_ROOT" \
  pip install --no-build-isolation -e .
```

The build uses the host compiler and does not require `nvcc`. It enables the
CUDA boxing kernels and the FlagGems Python wrappers; FlagGems C++ remains
disabled because the Kunlun stack does not provide the required `liboperators.so`.
The runtime default remains CUDA boxing. Set `FLAGOS_USE_FLAGGEMS=1` to select
the Kunlun-specific config, which enables only the overloads measured on P800
and keeps the remaining generated operators on CUDA boxing. The SDK libraries
remain supplied by the target installation; they are not copied into the wheel.

For an in-place source checkout, use `setup.py build_ext --inplace` instead of
`pip install`.

## Runtime Environment

Set the compatibility runtime on the loader path and enable the vendor profiler
runtime check before importing torch-fl:

```bash
export LD_LIBRARY_PATH=/usr/local/xcudart/lib:/usr/local/xpu/lib:$LD_LIBRARY_PATH
export XPU_ENABLE_PROFILER_TRACING=1
```

`XPU_CUPTI_ENABLE_DEVICE` is optional. If it is unset, XPU-RT emits a warning
that no devices are selected for profiling; this does not prevent ordinary
allocation or operator execution.

## Verification

Run the checked-in smoke harness on the P800:

```bash
ACCELERATOR=kunlun \
  XPU_ROOT=/usr/local/xpu \
  XCUDART_ROOT=/usr/local/xcudart \
  XPU_ENABLE_PROFILER_TRACING=1 \
  LD_LIBRARY_PATH=/usr/local/xcudart/lib:/usr/local/xpu/lib:$LD_LIBRARY_PATH \
  /path/to/vendor/python tests/manual/kunlun_runtime_smoke.py
```

The harness verifies device count, device index 1, allocation, host-to-device
and device-to-host copies, a CUDA-shaped stream, synchronization, cache release,
and `torch.mm` against a CPU reference. It exits nonzero on the first failed
check.

The measured P800 environment reported:

- Product: `P800 OAM` (`KUNLUNXIN`, architecture `KL3`)
- Attached devices: `8`
- XPU-RT: `5.37.1`
- Driver: `5.0.21.43`
- Device memory: `98304 MiB` per reported device

## Operator Scope

The default routing config is `torch_fl/configs/backends_cuda.conf`. This is
intentional: the vendor torch exposes CUDA registrations, so a route to `cuda`
reaches the vendor kernel through the generated boxing implementation. A measured
8x8 float32 matrix multiplication passed against the CPU reference.

## Automatic Mixed Precision

The generic `AutocastPrivateUse1` policy table is compiled for Kunlun, so
`torch.autocast("flagos")` and `torch.amp.GradScaler("flagos")` work on the
`flagos` device. Run the AMP suite with:

```bash
ACCELERATOR=kunlun \
  XPU_ENABLE_PROFILER_TRACING=1 \
  /path/to/vendor/python -m pytest tests/integration/test_amp.py -v
```

The measured P800 result is `15 passed`, covering FP16 and BF16 autocast policy,
FP32-preserving operators, explicit output dtype, type promotion, autocast
nesting, `binary_cross_entropy` banning, non-finite gradient detection, and
GradScaler finite/overflow behavior.

Two Kunlun-specific routes make this work. The XPU runtime rejects the native
CUDA TensorIterator cast kernel with `cudaErrorInvalidDeviceFunction`, so
`_to_copy` and `_copy_from` use the portable host-staged conversion instead. The
boxed `_amp_foreach_non_finite_check_and_unscale_` overload segfaults on this
stack, so it routes to a Kunlun-safe helper that stages the check and unscale
through the host. Both paths add host round-trips, so AMP throughput on Kunlun is
not comparable to a native device-side implementation.

Distributed AMP (DDP or FSDP2 combined with GradScaler) is **not validated** on
Kunlun.

## Optional: FlagGems on Kunlun

The vendor stack ships a FlagTree Triton at `/opt/flagtree` that `flag_gems`
imports and runs on. Enable it per process:

```bash
export PYTHONPATH=/opt/flagtree:$PYTHONPATH
export FLAGOS_USE_FLAGGEMS=1
```

`import torch_fl` then selects `torch_fl/configs/backends_kunlun_flaggems.conf`.
That config is deliberately narrow: only the overloads with measured CPU-reference
evidence on P800 route to `flagos_python`, and every other generated operator
uses CUDA boxing. The generic `backends_flaggems.conf` is **not** the Kunlun
default, because several of its routes fail on the current XPU stack:

| Route | Observed failure on P800 |
|---|---|
| `exp`, `exp.out` | `xpuLaunchKernel("exp_func_kernel_rank_1", ...) -> Operation not permitted` |
| `mm`, `mm.out` | Triton `OutOfResources: out of resource: uni_sram` |
| `addmm` | Compiler abort inside the FlagTree LLVM backend |
| `mul_.Tensor` | gems exits via `torch.mul(..., out=)`, which the XPU path leaves unregistered |

Do not add a route to the Kunlun config from FlagGems availability alone.
Extend `kunlun_verified_flaggems` in [`scripts/codegen_ops.py`](../../../scripts/codegen_ops.py),
regenerate, and re-run the survey on hardware:

```bash
python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_kunlun_flaggems.conf \
  --out /tmp/kunlun-flaggems-overloads.json
```

An individual overload can still be forced onto FlagGems for debugging with
`FLAGOS_OP_<op>=flaggems_python`, because every discovered wrapper stays compiled.

FlagGems RNG is **not** validated on this hardware, so no RNG overload is part
of the measured Kunlun scope. `tests/integration/ops/test_rng_dispatch.py` gives
`51 failed, 41 passed, 11 skipped, 1 xfailed` on P800 with `TestRngMultiDevice`
deselected, and aborts inside that class when it is not
(`test_reproducible_on_second_device`: a `randn` on `flagos:1` returns garbage
such as `1.4013e-45` once device 0 has been used). Those results are identical
with FlagGems on and off, and identical at the branch base, so they are a
pre-existing Kunlun RNG gap rather than a FlagGems regression. A focused
seed-replay probe on device 0 passed, which is not a substitute for the suite.

Record the torch-fl revision, FlagGems revision, hardware model, date, config
hashes, harness version, and raw JSON evidence in
[`operator-support.md`](../../reference/operator-support.md). Kunlun remains
**not revalidated** against the aggregate 546-overload FlagGems cohort; its
measured scope is the Kunlun config only.

## Troubleshooting

### Import exits with `Runtime profiler is disabled`

Set `XPU_ENABLE_PROFILER_TRACING=1` before the first `import torch` or
`import torch_fl`. This is an XPU-RT initialization requirement on the measured
SDK version.

### `cuda_runtime.h` or `libcudart.so.12` is missing

Set `XPU_ROOT` to the directory containing `include/cuda_runtime.h` and
`XCUDART_ROOT` to the directory containing `lib/libcudart.so.12`. The build
fails early when either path is invalid.

### Import cannot find `torch_fl._C`

A plain `build_ext` command leaves the extension under setuptools' temporary
build directory. Use `build_ext --inplace` or install the package before running
from the source checkout.

### Profiler warning about no selected devices

Set `XPU_CUPTI_ENABLE_DEVICE` according to the vendor profiler documentation if
profiling is required. The warning is not a runtime or operator failure when
profiling is not under test.
