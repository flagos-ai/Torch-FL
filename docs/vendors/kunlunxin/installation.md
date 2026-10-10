# Kunlunxin (Kunlun P800 XPU) Installation Guide

## Overview

Kunlunxin is a **CUDA-ABI boxing** platform, like DCU, PPU and MetaX. Two
properties of the vendor stack make that route the right one:

- The Kunlun P800 XPU SDK exposes **CUDA-shaped headers** (`cuda_runtime.h`) and a
  `libcudart.so*` compatibility runtime — the soname tracks the SDK's CUDA-compat
  version (`libcudart.so.12` on XPU SDK 5.37.1.0, whose CUDA-compatible version is
  12.9) — so the shared CUDA runtime sources compile against the vendor SDK with
  plain host `g++`, no `nvcc`.
- The Kunlun vendor `torch` registers its device kernels **under the CUDA
  dispatch key**. The generated PrivateUse1 → CUDA boxing kernels dispatch into
  the vendor's CUDA-ABI libtorch unchanged, so torch-fl adds no handwritten
  per-operator kernels.

`FLAGOS_ACCELERATOR=kunlunxin` selects this build and is recorded in
`torch_fl/_build_config.py`, which is what selects
`torch_fl/configs/backends_kunlunxin.conf` at runtime.

**Status:** Experimental. The measured scope on a P800 OAM covers device
discovery and indexing, allocation, host/device copies, a CUDA-shaped stream,
synchronization, and cache release, driven by the port's own CUDA boxing device
layer. Eager *operators* that reach ATen's CUDA dispatch (for example `torch.mm`
into the vendor BLAS) are **not validated** here, because the vendor XPU runtime
on the measured box is built against torch 2.9 and does not load against torch
2.10 — see [Runtime measurement boundary](#runtime-measurement-boundary). Full
operator coverage and FlagGems support are not claimed by this guide.

## Prerequisites

- Kunlun P800 hardware and XPU-RT driver
- Kunlun XPU SDK, normally `/usr/local/xpu` (a versioned `xpu-<ver>` directory
  symlinked from there; `5.37.1.0` on the measured install, CUDA-compatible
  version 12.9)
- XPU CUDA compatibility runtime, normally the SDK's own `/usr/local/xpu/so`
  (`XCUDART_ROOT` may point at the SDK `so/` directory, or at a directory with
  the shim one level down in `lib/`)
- Kunlun vendor PyTorch on the **2.10** line (torch-fl's supported minor)
- Python 3.8 or later matching the vendor torch build
- `cmake >= 3.18`, `ninja`, and a host C++17 compiler
- Optional: `XPU_ENABLE_PROFILER_TRACING=1` if you want the device tracer armed
  (import no longer requires it — see [Runtime Environment](#runtime-environment))

The Python build environment must use the vendor torch. Do not replace it with an
NVIDIA CUDA wheel: the vendor CUDA registrations and the XPU runtime are the
measured operator path.

## Build From Source

```bash
git clone https://github.com/flagos-ai/Torch-FL.git
cd Torch-FL

export XPU_ROOT=/usr/local/xpu
export XCUDART_ROOT="$XPU_ROOT/so"

FLAGOS_ACCELERATOR=kunlunxin \
  XPU_ROOT="$XPU_ROOT" \
  XCUDART_ROOT="$XCUDART_ROOT" \
  pip install --no-build-isolation -vvv -e .
```

The build uses the host compiler and does not require `nvcc`. It enables the
generated CUDA boxing kernels. FlagGems C++ stays off by default because the
Kunlun stack ships no `liboperators.so`. The SDK libraries remain supplied by the
target installation; they are not copied into the wheel.

For an in-place source checkout, use `setup.py build_ext --inplace` instead of
`pip install`.

### Build Notes

- **`XPU_ROOT`** must contain `include/cuda_runtime.h`; **`XCUDART_ROOT`** must
  contain a `libcudart.so*` shim, either directly or one level down in `lib/`. The
  build fails early when either is invalid, and the build log reports the exact
  runtime directory it resolved.
- **`FLAGOS_BUILD_FLAGGEMS_CPP=OFF`** for kunlunxin: the XPU SDK provides no
  FlagGems `liboperators.so`.
- The runtime default is CUDA boxing. No operator is rerouted to a vendor-native
  or FlagGems backend by this port.

## Runtime Environment

Set the compatibility runtime on the loader path before importing torch-fl:

```bash
export LD_LIBRARY_PATH=/usr/local/xpu/so:/usr/local/xpu/lib:$LD_LIBRARY_PATH
```

`XPU_ENABLE_PROFILER_TRACING=1` is required only if you want the device tracer to
run. The vendor CUPTI compatibility layer raises — rather than returning a
`CUptiResult` — when the profiler is unconfigured, so
`csrc/profiler/cupti_device_tracer.cc` arms the activity kinds inside a guarded
helper and leaves the tracer *unarmed* when the vendor refuses. Import therefore
succeeds either way; without the variable the tracer reports itself unavailable
and `start()` is a no-op. `XPU_CUPTI_ENABLE_DEVICE` is likewise optional. If it is
unset, XPU-RT emits a warning that no devices are selected for profiling; this
does not prevent ordinary allocation or operator execution.

## Verification

Run the checked-in smoke harness on the P800:

```bash
FLAGOS_ACCELERATOR=kunlunxin \
  XPU_ROOT=/usr/local/xpu \
  XCUDART_ROOT=/usr/local/xpu/so \
  LD_LIBRARY_PATH=/usr/local/xpu/so:/usr/local/xpu/lib:$LD_LIBRARY_PATH \
  /path/to/vendor/python tests/manual/kunlunxin_runtime_smoke.py
```

The harness verifies device count, device index 1, allocation, host-to-device and
device-to-host copies, a CUDA-shaped stream, synchronization, cache release, and
`torch.mm` against a CPU reference. It exits nonzero on the first failed check.
Every check except `mm` runs through the port's own CUDA boxing *device* layer and
is expected to pass on a stock-torch 2.10 build; `mm` reaches ATen's CUDA
dispatch and needs a vendor torch 2.10 runtime — see
[Runtime measurement boundary](#runtime-measurement-boundary).

The measured P800 environment reported:

- Attached devices: `8`
- XPU SDK / XPU-RT: `5.37.1.0` (CUDA-compatible version `12.9`)
- CUDA compatibility runtime: `libcudart.so.12` → `libcudart.so.12.9.1.kunlun`
- Vendor torch reported by the box: `2.9.0+cu129` (CUDA `12.9`)
- Vendor device name: `GPU`; device memory: `98304 MiB` per reported device

### Fleet environment status

A 2.10 fleet environment was built from this port on shared storage: a
`torch-fl-<vendor>-py<X>` conda environment carrying torch `2.10.0+cu129`,
`flagtree` as the Triton provider, and `FLAGOS_ACCELERATOR=kunlunxin`. It is
**build/import validated only**. The wheel builds from a clean tree, the
`lib/flagos_platform` marker records `kunlunxin`, `import torch_fl` succeeds,
torch-fl selects `backends_kunlunxin.conf`, and `device_count()` returns `8`.
Runtime operator execution is gated on a vendor XPU-RT plus XMLIR overlay built
against torch 2.10.

Three independent layers were measured between the fleet environment and a
working kernel launch, in order:

1. **Runtime RPATH shadowing** — `libtorch_cuda.so` resolves `libcudart.so.12`
   from a pip `nvidia/cuda_runtime` ahead of the SDK shim. Preloading the SDK
   runtime (`LD_PRELOAD=/usr/local/xpu/so/libcudart.so.12`) clears the
   driver-version gate and brings up `device_count() == 8`.
2. **A stale error latch** — the SDK's `cudaDeviceGetStreamPriorityRange`
   returns success while leaving `cudaErrorNotSupported` latched, which
   `CUDACachingAllocator`'s defensive `cudaGetLastError()` check turns into a
   failure on every allocation.
3. **Kernel-launch refusal** — with both layers above addressed, allocation and
   metadata operators succeed (`empty`, view/reshape/transpose, host→device
   copy) but every kernel-launching operator fails with
   `CUDA error: operation not supported` (`zeros`/`fill`, `add`, `mul`, `sum`,
   `matmul`, `cat`, `softmax`, device→host copy).

Layers 1 and 2 are runtime-configuration workarounds outside this port; layer 3
is a vendor toolchain gap. Nothing in this port addresses layer 3, and the
port's own verification scope is unaffected — it is a missing vendor torch 2.10
runtime, not a defect in the CUDA-boxing port.

## Operator Scope

The build selects
[`torch_fl/configs/backends_kunlunxin.conf`](../../../torch_fl/configs/backends_kunlunxin.conf),
a generated full-coverage config that routes every covered op to the `cuda`
boxing key. This is intentional: the vendor torch exposes CUDA registrations, so
a route to `cuda` reaches the vendor kernel through the generated boxing
implementation. That mapping is audited by regenerating the config; executing the
routed operators on the XPU requires a vendor torch 2.10 runtime (see below).

## Runtime measurement boundary

The vendor XPU runtime on the measured box is **torch-2.9-bound**: the vendor
`torch/lib` plus the `torch_plugin` / `torch_xmlir` overlay are compiled against
torch 2.9 internals. Its compiled module
`torch_xmlir/_XMLIRC.cpython-310-x86_64-linux-gnu.so` fails to load against a
stock torch 2.10 build with
`undefined symbol: _ZN2at4_ops5triu_10redispatchEN3c1014DispatchKeySetERNS_6TensorEl`
(`at::_ops::triu_::redispatch`, whose 2.10 signature differs from 2.9). The
overlay's import hook catches that error and only warns, so `import torch_fl`
still succeeds, but `torch_xmlir` stays uninitialized and eager operators that
reach ATen's `CUDA` dispatch raise
`NotImplementedError: Could not run 'aten::empty.memory_format' ... 'CUDA' backend`.

What **is** measured on the ported 2.10 build therefore stops at the boxing
device layer: `import torch_fl`, accelerator/conf selection
(`backends_kunlunxin.conf`), `device_count() == 8`, and on `flagos:1`
`set_device` / `current_device` / allocation / host↔device copy / `Stream` /
`synchronize` / `empty_cache`. What is **not** validated is any operator that
dispatches into ATen's CUDA backend, including the harness's `mm` check.
Validating those requires a vendor torch 2.10 runtime with a matching overlay; a
pip `nvidia-cublas-cu12` is not a substitute for the vendor BLAS.

## FlagGems on Kunlunxin

FlagGems is **not** part of this port. Kunlunxin is deliberately absent from
`FLAGGEMS_PYTHON_PLATFORMS` in
[`scripts/codegen/gen_vendor_confs.py`](../../../scripts/codegen/gen_vendor_confs.py),
so `backends_kunlunxin.conf` claims no FlagGems route and every covered operator
stays on the CUDA boxing kernel. The FlagGems Python allowlist measured on the 2.9
line is a follow-up PR: adding a route requires codegen coverage for kunlunxin,
so the gate and the config must move together.

Do not add a kunlunxin FlagGems route from FlagGems availability alone. Extend
the generator, regenerate, and re-run the survey on hardware:

```bash
python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_kunlunxin.conf \
  --out /tmp/kunlunxin-flaggems-overloads.json
```

Record the torch-fl revision, FlagGems revision, hardware model, date, config
hashes, harness version, and raw JSON evidence in
[`operator-support.md`](../../reference/operator-support.md). Kunlunxin remains
**not revalidated** against the aggregate FlagGems cohort; its measured scope is
the CUDA-boxing route only.

## Troubleshooting

### Import exits with `Runtime profiler is disabled`

This was the original symptom of the vendor CUPTI layer throwing from a static
initializer. `csrc/profiler/cupti_device_tracer.cc` now arms the activity kinds
inside a try/catch, so `import torch_fl` succeeds whether or not the vendor
profiler is configured. Set `XPU_ENABLE_PROFILER_TRACING=1` before the first
`import torch` only when you actually want the tracer armed.

### `cuda_runtime.h` or `libcudart.so*` is missing

Set `XPU_ROOT` to the directory containing `include/cuda_runtime.h` and
`XCUDART_ROOT` to the directory containing the `libcudart.so*` shim (directly, or
one level down in `lib/`). The build fails early when either path is invalid.

### Import cannot find `torch_fl._C`

A plain `build_ext` command leaves the extension under setuptools' temporary
build directory. Use `build_ext --inplace` or install the package before running
from the source checkout.

### Profiler warning about no selected devices

Set `XPU_CUPTI_ENABLE_DEVICE` according to the vendor profiler documentation if
profiling is required. The warning is not a runtime or operator failure when
profiling is not under test.
