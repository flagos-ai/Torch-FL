---
name: cuda-compat-vendor
description: >
  Enable operators on a CUDA-compatible accelerator (MetaX, Hygon DCU, PPU,
  or any chip whose SDK ships a CUDA-shaped torch build) by extracting
  libtorch_cuda.so from the vendor's own torch wheel and bundling it, instead of
  writing kernels. Use this after runtime-bringup passes, once you have confirmed
  the vendor ships a CUDA-compatible torch. Covers: proving compatibility before
  committing, extracting the vendor .so set, the mandatory LD_PRELOAD-before-
  import-torch constraint, wheel bundling via setup.py, and the boxing-path tests.
---

# CUDA-compatible vendor enablement (torch_fl)

## What this achieves

For a chip whose SDK provides a CUDA-shaped PyTorch build, torch_fl writes
**zero operator kernels**. It instead:

1. **Boxes** flagos (PrivateUse1) tensors to CUDA device metadata — no data copy,
   since flagos and the vendor's CUDA-compatible device share the same memory —
   calls the vendor's already-compiled kernel through the public `at::` API, then
   **unboxes** the result back to flagos.
2. **Reuses** the generated `csrc/aten/generated/cuda_kernels.cc` produced by
   [[torch-version-port]]. Nothing per-operator is written here.
3. Runs against an **external `libtorch_cuda.so`** extracted from the vendor
   wheel and `LD_PRELOAD`ed before `import torch`, so the pip environment stays
   CPU-only.

This is roughly an order of magnitude less work than [[native-op-backend]], which
is why proving compatibility (Step 1) is worth real effort before defaulting to
the native route.

**Prerequisite:** [[runtime-bringup]] must be green. Boxing does not remove the
need for the 28-function runtime contract — allocation, streams and copies still
go through flagos.

## Step 1 — prove compatibility before committing to this path

Do not infer compatibility from marketing claims about "CUDA ecosystem support".
The question is narrow and mechanically checkable: **does the vendor ship a
`libtorch_cuda.so` whose ATen symbols register kernels under the CUDA dispatch
key?**

```bash
# 1. Find the vendor's torch build (usually a vendor pip index or SDK tarball)
find /opt/<vendor> -name 'libtorch_cuda.so*' -o -name 'libtorch_*.so*' | head

# 2. Are the ATen CUDA symbols actually present?
nm -D --defined-only <path>/libtorch_cuda.so | grep -cE ' T .*at.*(add|mm|cat)'

# 3. Decisive test: load it and ask the dispatcher what it registered.
#    Note the load must happen before torch initializes -- see Step 3.
python - <<'PY'
import ctypes, torch
print(torch._C._dispatch_dump("aten::mm"))   # before: CPU only
ctypes.CDLL("<path>/libtorch_cuda.so", mode=ctypes.RTLD_GLOBAL)
print(torch._C._dispatch_dump("aten::mm"))   # after: a CUDA entry must appear
PY
```

If a CUDA entry appears for `mm`, `add`, `_softmax` and `bmm`, this path is
viable. If not, stop and use [[native-op-backend]].

`docs/vendors/cuda/external-libtorch-cuda.md` records the same four gates
measured on NVIDIA, including the exact commands and the four hard constraints.
Read it once before proceeding — the constraints are not obvious and one of them
is fatal if missed.

### Which shape is this vendor?

Three shapes exist, in increasing order of work:

| Shape | Example | Runtime dir | Operator work |
|---|---|---|---|
| SDK ships a `libcudart` shim, `cuda_runtime.h` compiles | Hygon DCU (DTK) | **reuses `accelerator/cuda/*.cc` verbatim** — no vendor dir | none |
| CUDA-compatible SDK, own detection | PPU | reuses `cuda` sources, `ACCELERATOR=cuda` + `PPU_SDK` detection | none |
| CUDA-compatible but needs a shim layer | MetaX (cu-bridge/mxcc) | `accelerator/metax/` incl. `cudart_shim.c` + a `.version` script | none |

Read `csrc/runtime/accelerator/CMakeLists.txt` for how each is wired; DCU's
branch (which globs `cuda/*.cc` from a non-cuda ACCELERATOR) is the cleanest
precedent and worth copying when the shim exists.

## Step 2 — extract the vendor .so set

Download-only, then extract. Do **not** `pip install` the vendor torch into the
working env — that replaces your CPU-only pin and reintroduces exactly the
duplicate-symbol problem the scheme avoids.

```bash
pip download <vendor-torch>==<exact-version> -d /tmp/vt --no-deps
cd /tmp/vt && unzip -o *.whl -d unpacked
ls -la unpacked/torch/lib/libtorch_cuda.so unpacked/torch/lib/libc10_cuda.so
```

Stage what you extracted into the assets directory `setup.py` looks for:

```bash
mkdir -p .libtorch_cuda_assets
cp unpacked/torch/lib/libtorch_cuda.so \
   unpacked/torch/lib/libc10_cuda.so .libtorch_cuda_assets/
```

`setup.py:_bundle_cuda_assets()` copies every `*.so*` from
`.libtorch_cuda_assets` (override with `FLAGOS_CUDA_ASSETS_DIR`) into
`torch_fl/lib/` at build time, skipping same-size files so the ~1GB copy does not
repeat. `FLAGOS_SKIP_CUDA_ASSETS=1` produces a slim wheel for machines that
supply the `.so` out-of-band.

### Version matching is exact, not approximate

Constraint 3 in the CUDA doc: the `libtorch_cuda.so` version must match the
installed CPU torch **exactly**. `2.9.0+cpu` pairs only with a `2.9.0` CUDA
build. A minor mismatch does not fail cleanly at load — it fails as an ABI
crash somewhere later, which is a much worse debugging experience.

Also note constraint 4: the scheme relies on the CPU wheel exposing a complete
enough symbol set for the CUDA library to link against. That is not officially
guaranteed by PyTorch. It has held across the versions measured, but it is the
reason this path warrants an on-hardware test rather than a build-only check.

## Step 3 — the load-timing constraint (the one that bites)

**`libtorch_cuda.so` must be loaded before `import torch`.** This is constraint 1
and it is hard, not advisory. Loading afterwards leaves the dispatcher table
already built without CUDA entries, and no later `CDLL` repairs it. The failure
is silent — ops quietly run on CPU, or you get a device-mismatch error far from
the cause.

`scripts/vendor/with_cuda_libtorch.sh` exists precisely so nobody has to remember the
`LD_PRELOAD` + `LD_LIBRARY_PATH` incantation. Run **everything** through it:

```bash
scripts/vendor/with_cuda_libtorch.sh python -c "import torch_fl, torch; \
    print(torch.randn(4,4,device='flagos') @ torch.randn(4,4,device='flagos'))"
scripts/vendor/with_cuda_libtorch.sh pytest tests/integration/ops/ -v
```

If the vendor's `.so` lives somewhere non-default, point the script at it rather
than reinventing the preload — read the script first and extend it if the vendor
needs extra libs (MetaX needs its shim, DCU pulls from the DTK tree).

## Step 4 — build wiring

Which `ACCELERATOR` value to use depends on the shape from Step 1, and the choice
is not cosmetic:

- **Shim present, `cuda_runtime.h` compiles** — follow DCU: add an `ACCELERATOR`
  branch in `csrc/runtime/accelerator/CMakeLists.txt` that globs
  `${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cc`, and add no vendor runtime directory at
  all. Least code, least drift.
- **CUDA-compatible with its own detection** — follow PPU: keep
  `ACCELERATOR=cuda` and add SDK detection in `setup.py`.
- **Needs a shim layer** — follow MetaX: a real `accelerator/<vendor>/` directory
  with `cudart_shim.c` and a linker `.version` script.

The build must still be **g++ only, no nvcc**, linking only `torch_cpu_library`.
CUDA symbols resolve at runtime from the preloaded library. If the build starts
wanting nvcc, something has pulled in a real CUDA dependency and the scheme's main
benefit is gone.

```bash
ACCELERATOR=<vendor> FLAGGEMS_KERNEL=OFF FLAGGEMS_CPP=OFF \
  pip install -e . --no-build-isolation
```

## Step 5 — verify on hardware

Build-only success proves very little here — constraint 4 means the interesting
failures are at runtime. Compare against CPU:

```bash
scripts/vendor/with_cuda_libtorch.sh pytest tests/integration/ops/ \
  -m "not flaggems and not flaggems_python" -v
```

Then confirm boxing is actually happening rather than silently falling back to
CPU. A passing numerical test does **not** prove the vendor kernel ran:

```bash
scripts/vendor/with_cuda_libtorch.sh python - <<'PY'
import torch_fl, torch
x = torch.randn(64, 64, device="flagos")
y = (x @ x).sum()
print(y.item(), x.device)      # device must stay flagos, not cpu
PY
```

Check `torch.__version__` still ends in `+cpu`. If it does not, a vendor torch got
installed somewhere and the measurement is invalid.

### Known cold-start artifact

Under this scheme the first CUDA op in a fresh process can hit
`Allocator not initialized for device`, because PyTorch normally primes its CUDA
caching allocator inside `torch.cuda._lazy_init()` — which this path deliberately
never calls. It surfaces on **out-variant** ops (`mm.out`, `bmm.out`) run as the
very first CUDA op; non-out ops warm the allocator as a side effect. This is an
environment artifact, not a codegen defect. Accept/xfail those log-only tests, or
prime once at import with a throwaway functional op. Do **not** hand-initialize
PyTorch's CUDA allocator — that reaches into the `torch.cuda` internals the whole
scheme exists to avoid.

## Done criteria

- Step 1's dispatcher dump shows CUDA entries for `mm`/`add`/`_softmax`/`bmm`
- Build succeeds with g++ only, no nvcc, linking only `torch_cpu_library`
- `import torch_fl` clean through `scripts/vendor/with_cuda_libtorch.sh`
- Operator suite passes, modulo the cold-start artifact above
- Tensors stay on `flagos` through a compute chain (boxing confirmed, not fallback)
- `torch.__version__` still ends in `+cpu`
- `docs/vendors/<vendor>/installation.md` written, and
  `docs/reference/compatibility.md` plus `docs/reference/operator-support.md`
  updated from **measured** results — per CLAUDE.md, unavailable hardware is
  marked *not revalidated* rather than assumed

## Related

[[runtime-bringup]] (prerequisite) · [[torch-version-port]] (supplies the
generated kernels) · [[native-op-backend]] (the fallback if Step 1 fails) ·
[[cuda-op-integration]] (the NVIDIA-specific version of this path) ·
[[pre-pr-checks]] (before opening the PR)
