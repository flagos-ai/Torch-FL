# Kunlunxin (Kunlun P800 XPU) Installation Guide

## Overview

Kunlunxin is a **native PrivateUse1 backend**, shaped like the MUSA, GCU and
Ascend backends: its own route key (`kunlunxin`) over the vendor XPU SDK
operator library, and **no CUDA boxing**. Two properties of the vendor stack
make that the right shape:

- The Kunlun P800 XPU SDK ships a native C runtime — `libxpurt` (headers under
  `<XPU_ROOT>/include/xpu/runtime.h`) — that the port's own runtime sources
  (`csrc/runtime/accelerator/kunlunxin/`) call directly for device discovery,
  allocation, host/device copies and streams. No tensor is re-dispatched onto
  the CUDA dispatch key, so the build compiles no boxing kernels.
- The vendor op library is a standalone xdnn library (`libxdnn_pytorch.so`,
  headers under `include/xdnn_pytorch/`). It ships inside the vendor torch
  package (`torch_xmlir/`), *not* the XPU SDK, and the native backend wraps it
  behind the `kunlunxin` route key.

`FLAGOS_ACCELERATOR=kunlunxin` selects this build and is recorded in
`torch_fl/_build_config.py`, which is what selects
`torch_fl/configs/backends_kunlunxin.conf` at runtime.

**Status:** Experimental. The port implements the device layer natively: device
discovery and indexing, allocation, host/device copies, an XPU stream,
synchronization, and cache release all run through
`csrc/runtime/accelerator/kunlunxin/` on the XPU SDK's own `libxpurt` runtime
rather than the CUDA-shaped shim. Compute coverage comes from the vendor xdnn op
library behind the `kunlunxin` route key. Neither the native device layer nor the
vendor route has been measured on P800 hardware in this change — the checked-in
harness to run it is
[`tests/manual/kunlunxin_runtime_smoke.py`](../../../tests/manual/kunlunxin_runtime_smoke.py).
The FlagGems cohort has **not** been revalidated on the P800 — its route is a
follow-up (see [FlagGems on Kunlunxin](#flaggems-on-kunlunxin)).

## Prerequisites

- Kunlun P800 hardware and the XPU-RT driver
- Kunlun XPU SDK, normally `/usr/local/xpu`. It must provide both the native
  runtime headers (`include/xpu/runtime.h`) and the CUDA-shaped surface
  (`include/cuda_runtime.h`) — see [Build Notes](#build-notes) for why the
  latter is still needed on a de-boxed backend.
- The vendor xdnn op library (`libxdnn_pytorch.so` plus
  `include/xdnn_pytorch/xdnn_pytorch.h`); it ships inside the vendor torch
  package, not the XPU SDK.
- Kunlun vendor PyTorch, if you are taking the xdnn op library from a vendor
  install. The build itself never imports it, so its version is not part of the
  requirement (see below)
- Python 3.8 or later, matching the FlagTree XPU wheel's interpreter (cp310 in
  the pinned configuration)
- `cmake >= 3.18`, `ninja`, and a host C++17 compiler

The build never imports or links the vendor torch. All a vendor install supplies
is the xdnn op library and its header, and `setup.py` finds those two files by
presence — `XDNN_ROOT`, then `FLAGOS_VENDOR_TORCH_LIB`, then the `torch_xmlir`
package in site-packages — rather than by importing anything. Those files happen
to ship inside `torch_xmlir`, but a plain CPU torch wheel is enough to build and
run the backend, and the vendor `torch_xmlir` extension does not load in that
interpreter. The vendor torch is therefore a convenient place to find xdnn, not a
build requirement: any supported torch wheel works provided `XDNN_ROOT` (or
`FLAGOS_VENDOR_TORCH_LIB`) resolves to the library.

## Build From Source

```bash
git clone https://github.com/flagos-ai/Torch-FL.git
cd Torch-FL

export XPU_ROOT=/usr/local/xpu
# Optional: point XDNN_ROOT at the vendor torch package root explicitly.
# Unset, setup.py discovers it (see Build Notes).
# export XDNN_ROOT=/path/to/vendor/torch_xmlir

FLAGOS_ACCELERATOR=kunlunxin \
  XPU_ROOT="$XPU_ROOT" \
  pip install --no-build-isolation -vvv -e .
```

The build uses the host compiler and does not require `nvcc` or a CUDA toolkit.
It compiles the native XPU runtime layer and links the vendor xdnn op library.
FlagGems C++ stays off by default because the XPU SDK ships no
`liboperators.so`. The SDK and vendor libraries remain supplied by the target
installation; they are not copied into the wheel.

For an in-place source checkout, use `setup.py build_ext --inplace` instead of
`pip install`.

### Build Notes

- **`XPU_ROOT`** must contain `include/xpu/runtime.h` (the native `libxpurt`
  headers the backend links) and `include/cuda_runtime.h`. The build fails early
  when either is missing.
- **`XDNN_ROOT`** must contain `libxdnn_pytorch.so` (directly or one level down
  in `lib/`) and `include/xdnn_pytorch/xdnn_pytorch.h`. It is not derived from
  `XPU_ROOT` because the library ships inside the vendor torch package. If it is
  unset, `setup.py` probes `FLAGOS_VENDOR_TORCH_LIB` and then the `torch_xmlir`
  package in site-packages, and the configure step fails early when none of them
  holds the library. Only these two files are needed from a vendor install, so
  any torch wheel can sit alongside them.
- **`XCUDART_ROOT`** locates the CUDA-compat `libcudart.so*` shim (the SDK's own
  `so/` directory on a P800 install). This is the *only* remaining CUDA-shaped
  build input: `csrc/runtime/hooks.h`'s `isPinnedPtr()` probes
  `cudaPointerGetAttributes` as the DDP fallback for pointers the flagos host
  allocator did not register. Neither the header nor the shim carries a kernel
  and neither is reachable from the routing layer, so neither can re-introduce
  boxing.
- **`FLAGOS_BUILD_VENDOR=ON`** (pinned): xdnn is the only operator library the
  XPU SDK provides.
- **`FLAGOS_BUILD_BOXING=OFF`**: the conf routes nothing to the CUDA boxing
  kernels.
- **`FLAGOS_BUILD_FLAGGEMS_CPP=OFF`** (pinned): no FlagGems `liboperators.so`
  exists for the XPU SDK.

## Runtime Environment

The build embeds the runtime paths (the XPU SDK's `so/` directory, the xdnn
directory, and the vendor torch package's nested dependency directory that
`libxdnn_pytorch.so` itself needs) in the install RPATH. That is a **build-host**
answer: a wheel installed into a different virtualenv has RPATH entries pointing
at directories that may not exist, and the import then fails. `libxdnn_pytorch.so`
is not self-contained — its `DT_NEEDED` names

```
libxpu_blas.so  libxpu_dnn.so  libxpu_flash_attention.so  libcudart.so.12
```

and the vendor torch package keeps those component libraries **inside its Python
package directory**, as `<site-packages>/torch_xmlir/*.so`, together with the
nested `xhpc/*/dependency_so/` the CUDA-compat runtime comes from. `libxpuml.so.1`
is the exception: it ships with the XPU SDK, on the install RPATH. Importing the
`torch_xmlir` Python package is *not* required and never happens on this path;
its directories simply have to be on the loader's search path. `XCUDART_ROOT`
does not cover them, because the XPU SDK ships only its own CUDA-11-shaped shim
while the op library needs the CUDA-12-compat one.

When the directories are missing, `import torch_fl` reports a bare

```
ImportError: libxpu_blas.so: cannot open shared object file: No such file or directory
```

that names neither the variable nor the directory. Put the vendor package's
directory on the loader path — this is the one runtime environment variable
kunlunxin needs, and it must be set **before** the interpreter starts, because
`DT_NEEDED` is resolved before any of the wheel's code runs:

```bash
export FLAGOS_KUNLUNXIN_XDNN_LIB_DIR=<venv>/lib/python3.10/site-packages/torch_xmlir
export LD_LIBRARY_PATH="$FLAGOS_KUNLUNXIN_XDNN_LIB_DIR:/usr/local/xpu/so:/usr/local/xcudart/lib:$LD_LIBRARY_PATH"
```

`FLAGOS_KUNLUNXIN_XDNN_LIB_DIR` also lets the wheel load the components itself
(colon-separated for more than one directory) and turns the
`libxpu_blas.so: cannot open shared object file` message into one that names the
missing libraries and where they live. It cannot replace `LD_LIBRARY_PATH` for
the first import: by the time any code in `libtorch_fl.so` runs, the loader has
already resolved — or failed to resolve — the `DT_NEEDED` list.

The component libraries have to be loaded **after** the XPU SDK's CUDA-compat
runtime, not before it. `libxpu_blas.so` carries an undefined, unversioned
`cudaGetDriverEntryPoint` and declares no `DT_NEEDED` on any CUDA library, so
that reference is satisfied only by what the process has already mapped
`RTLD_GLOBAL`. Measured on a P800, `dlopen("libxpu_blas.so")` in a bare
interpreter fails with

```
libxpu_blas.so: undefined symbol: cudaGetDriverEntryPoint
```

and succeeds as soon as `libxpucuda.so`, `libcudart.so.12` and `libxpurt.so.2`
(from the XPU SDK, found by `SONAME`) are loaded. The wheel's pre-load honours
that order. In the ordinary path it does not matter, because the vendor package
is already mapped by the time `import torch_fl` runs; it matters for a
diagnostic that would otherwise report `libxpu_blas.so` missing in a directory
that contains it.

Verify with the smoke below; a healthy stack reports the device count and an
`mm` value, and the package is not needed in `sys.modules`:

```bash
python -c "import sys, torch, torch_fl; \
  print('torch_xmlir imported?', 'torch_xmlir' in sys.modules); \
  print('devices', torch_fl.flagos.device_count()); \
  print('add', float((torch.ones(8,8,device='flagos')*2).sum().cpu()))"
```

Kunlunxin is not in the CUPTI tracer arm of the selection in `csrc/CMakeLists.txt`,
so the build compiles the `unavailable_device_tracer`. The de-box is why: the
vendor's CUPTI-compatible shim instruments the CUDA dispatch key, which the
native backend no longer uses, so arming it would report an empty device timeline
instead of the XPU one. Import is unaffected, and `start()` is a no-op; device
profiling is not available on this platform.

## Verification

Run the checked-in smoke harness on the P800:

```bash
FLAGOS_ACCELERATOR=kunlunxin \
  XPU_ROOT=/usr/local/xpu \
  FLAGOS_KUNLUNXIN_XDNN_LIB_DIR=/path/to/venv/lib/python3.10/site-packages/torch_xmlir \
  LD_LIBRARY_PATH=/usr/local/xpu/so:/usr/local/xpu/lib:$LD_LIBRARY_PATH \
  /path/to/vendor/python tests/manual/kunlunxin_runtime_smoke.py
```

The harness verifies device count, device index 1, allocation, host-to-device and
device-to-host copies, an XPU stream, synchronization, cache release, and
`torch.mm` against a CPU reference. It exits nonzero on the first failed check.
Every check runs through the port's own native XPU device layer; `mm` reaches the
vendor xdnn op library behind the native `kunlunxin` route.

### Superseded CUDA-boxing baseline

An earlier revision of this guide recorded what a P800 OAM reported under the
removed CUDA-boxing backend: 8 attached devices; XPU SDK / XPU-RT `5.37.1.0`
(CUDA-compatible version `12.9`); the CUDA compatibility runtime resolving to
`libcudart.so.12`; and a per-device memory size of 98304 MiB. The device layer it
measured was the port's own CUDA sources over the XPU driver shim, and operators
that reached ATen's CUDA dispatch were not validated. **That measurement does not
transfer to the native backend** — the de-box replaces the runtime sources it
exercised — so it is recorded here only as provenance for the 2026-10-11 entry in
[`operator-support.md`](../../reference/operator-support.md), not as evidence for
the current route list.

That baseline also quoted a vendor torch version (`2.9.0+cu129`) as if it were a
requirement. It is not one: the de-boxed backend links only the xdnn op library
and never imports the vendor torch, so the vendor torch's own version — and the
CUDA-compat version it was built against — are details of the removed baseline.
The XPU SDK / XPU-RT version above is the only version fact here that still
applies.

## Operator Scope

The build selects
[`torch_fl/configs/backends_kunlunxin.conf`](../../../torch_fl/configs/backends_kunlunxin.conf),
a generated full-coverage config. Kunlunxin is a native-kernel vendor like musa,
gcu and ascend, so every covered op routes to the native `kunlunxin` key — which
reaches the vendor xdnn op library — or is left at `none`, which sends the op to
`cpu_fallback`. There is **no `cuda` route** in the conf. That mapping is audited
by regenerating the config and by
`tests/unit/test_conf_registration_consistency.py`, which checks that every
accelerated route is an op the native registration claims.

## FlagGems on Kunlunxin

`backends_kunlunxin.conf` routes **315 operators** to FlagGems on the P800, out of
a 618-operator ceiling. The count is a measurement, not the cohort: the whole
ceiling was surveyed on the chip and only the operators that matched a CPU
reference on the device are routed. `KUNLUNXIN_FLAGGEMS_MEASURED` in
`scripts/codegen/gen_vendor_confs.py` is the list, and the rest of this section
records how it was produced and what it does and does not claim.

The stack it runs on is provisioned by `set_env_kunlunxin.sh`, which installs
FlagTree `0.7.0+xpu3.6` (the Triton build carrying the `xpu` backend) and
FlagGems from the `flagos-pypi-kunlunxin` lane, and the build defines
`FLAGGOS_FLAGGEMS_PYTHON` for this platform. The registration path is in place
and empty rather than absent:

- `csrc/aten/backends/kunlunxin/generated/kunlunxin_flaggems_register.inc`
  (generated by `scripts/codegen/codegen_kunlunxin_flaggems.py`) claims those 315
  ops on PrivateUse1.
- `register.cc`'s `USE_KUNLUNXIN` arm includes it under
  `FLAGOS_FLAGGEMS_PYTHON`, and `VENDORS` in `scripts/codegen/gen_vendor_confs.py`
  names it, so the conf generator and the compiler read the same registration
  surface.
- `torch_fl/accelerator/kunlunxin/_klx_compat.py` supplies the two runtime pieces
  a CPU-torch FlagGems build needs — the device-name realignment and the
  `torch.cuda` shim — and `_patch_flaggems_codegen_config`'s kunlunxin arm
  installs them. Before this module existed that arm labelled the vendor and
  returned before any shim ran, so a routed op reached FlagGems' kernel and then
  died in `torch.cuda._exchange_device`.

Both registration files are driven by one value, `KUNLUNXIN_FLAGGEMS_MEASURED` in
`scripts/codegen/gen_vendor_confs.py`. Adding an op there routes it to `flaggems`
in the conf **and** emits its `m.impl()` line, so the two cannot drift. Kunlunxin
stays out of `FLAGGEMS_PYTHON_PLATFORMS`, which keeps the unrouted remainder of
the ceiling off the conf: joining that set is the statement "the whole cohort has
been measured", and 315 of 618 is not that.

### What the survey measured

The measurement is per overload, through `torch.ops.aten`, with the device result
compared against the same call on CPU:

```bash
python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_kunlunxin.conf \
  --out /tmp/kunlunxin-flaggems-overloads.json --profiles 2d-f32
```

Measured 2026-11 on a P800 (Kunlun P800 XPU, `10.21.1.13`) with torch 2.10.0+cpu,
FlagGems 5.4.0 and FlagTree 0.6.1+xpu3.6. All 618 ops were registered and
surveyed, which is why the outcome is a distribution rather than a pass count:

| Outcome | Count | Meaning |
| --- | --- | --- |
| STRICT | 316 | every CPU-valid case matched the CPU reference on device |
| WRONG | 22 | ran, returned a silently wrong result |
| CRASH | 9 | poisoned the device (`addmm`, `fmax`, `native_batch_norm`, …) |
| ERROR | 95 | raised |
| TIMEOUT | 5 | did not finish |
| UNTESTED | 171 | no CPU-valid synthesized case (internal ops) |

315 of the 316 STRICT ops are routed. The WRONG and CRASH classes are the reason
this table exists: routing the cohort wholesale would have silently corrupted
results in 22 operators and taken the device down in 9 more. A sample of 12 was
re-run on a different device with a cold Triton cache and reproduced 12/12.

The one STRICT op held back is `mean.dim`. It passed the screen and was routed,
and this lane's own `test_mean_dim_keepdim` then rejected it: FlagGems' kernel
reduces axis 0 for 2D+ shapes and the FlagTree XPU backend will not compile that,
failing with `OutOfResources: out of resource: uni_sram` / "axis must not be 0
for 2D+ shapes". The survey's synthesized arguments never reached that path --
the concrete cost of screening on one profile. Iluvatar records the same operator
as unusable for the same kernel.

**Scope of the evidence.** `2d-f32` is the screening profile the survey documents
for large route sets, not the full seven-profile matrix, so "STRICT" here means
every CPU-valid case at the `2d-f32` profile — float32, contiguous, 2-D — matched.
Ruling out a dtype- or layout-specific defect in one of these ops is a full-matrix
follow-up, not something this run established.

Two facts a future survey run needs. Device 0 on this host is about seven times
slower per allocation than the other cards (11.2 s versus 1.6 s for a first
`torch.abs`), and the survey's child processes default to it, so pin the device or
most of the wall clock goes to allocation. The route set is also large enough that
one process takes hours; the run above was split into three `--ops` shards on
separate devices and merged, which is valid because each shard records the same
`harness_version` and `conf_sha256`.

To add an op, put it in `KUNLUNXIN_FLAGGEMS_MEASURED` and regenerate — the `.inc`
first, then the conf, because `route()` gates on the registration the conf
generator reads from disk:

```bash
python scripts/codegen/codegen_kunlunxin_flaggems.py
python scripts/codegen/gen_vendor_confs.py
```

The generator refuses an op that is also served by a native xdnn kernel, rather
than subtracting it: the two `.inc` files share one `TORCH_LIBRARY_IMPL` body, so
a duplicate `m.impl` will not compile, and silently dropping the FlagGems half
would leave the conf routing `flaggems` at an empty slot. Moving such an op means
editing the native table in `scripts/codegen/codegen_kunlunxin.py` in the same
change.

Kunlunxin's FlagGems scope is therefore the 315 routed ops, not the aggregate
cohort; everything else stays on the native xdnn route or at `none`.

## Troubleshooting

### `xpu/runtime.h` or `cuda_runtime.h` is missing

Set `XPU_ROOT` to the XPU SDK root. The build fails early: `xpu/runtime.h` is the
native `libxpurt` header the backend links, and `cuda_runtime.h` is still needed
for the pinned-pointer probe. Both live under `$XPU_ROOT/include`.

### The vendor xdnn op library is not found

`FLAGOS_BUILD_VENDOR` is pinned on for this platform, so the build needs
`libxdnn_pytorch.so` and `include/xdnn_pytorch/xdnn_pytorch.h`. Set `XDNN_ROOT` to
the directory holding them (the vendor torch package root `torch_xmlir/` is one
such directory), or `FLAGOS_VENDOR_TORCH_LIB` to a directory holding the `.so`.
The configure step fails early instead of falling back to CPU for compute ops.

### `ImportError: libxpu_blas.so: cannot open shared object file`

This is the runtime analogue of the section above, and the library named is one of
`libxdnn_pytorch.so`'s components, not a file you are expected to install. They
ship inside the vendor torch package at
`<venv>/lib/python*/site-packages/torch_xmlir/*.so`. Set
`FLAGOS_KUNLUNXIN_XDNN_LIB_DIR` to that directory (it is not a Python import — do
not add it to `PYTHONPATH` expecting an import) and put it first on
`LD_LIBRARY_PATH` before starting the interpreter, as shown in
[Runtime Environment](#runtime-environment). A path that happens to be baked into
the wheel from the build host does not help: it refers to the build machine's
virtualenv.

### `libcudart.so*` is missing

Set `XCUDART_ROOT` to the directory containing the CUDA-compat shim (directly, or
one level down in `lib/`). Only the pinned-pointer probe uses it; no kernel is
compiled against it.

### Import cannot find `torch_fl._C`

A plain `build_ext` command leaves the extension under setuptools' temporary
build directory. Use `build_ext --inplace` or install the package before running
from the source checkout.
