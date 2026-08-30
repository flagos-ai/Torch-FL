# DCU without DTK's core libraries

A DCU wheel runs DTK's HIP kernels on the **official** PyTorch core. Only DTK's
*device* libraries are bundled; `libc10.so`, `libtorch_cpu.so`, `libtorch.so`,
`libtorch_global_deps.so`, `libtorch_python.so` and `libshm.so` come from the
stock `torch` wheel, which is never modified.

This is the default. `FLAGOS_DCU_VENDOR_CORE=1` selects the previous behaviour
(symlink DTK's whole core set over the installed wheel) as a rollback path.

- Measured: 2026-08-29, Hygon DCU (8 cards), DTK 2604
- Vendor torch: `2.10.0+das.opt1.dtk2604`; official torch: `2.10.0+cpu`
- Operator coverage: unchanged. Every HIP kernel in `libtorch_hip.so` still
  registers under the `CUDA` dispatch key, which is what the PrivateUse1 boxing
  path re-dispatches into.

## Why the vendor core was needed before

DTK forked the core libraries, not just the device ones. On this box DTK's
`libtorch_cpu.so` exports 128 hip symbols and carries
`DT_NEEDED: libgalaxyhip.so.5`; it resolves 2101 of `libtorch_fl.so`'s undefined
symbols while `libtorch_hip.so` resolves none. So the original DCU route had to
take the core as a set, which meant:

- ~1.8 GB of vendor libraries in the wheel;
- `torch.__version__` and all Python-side behaviour coming from the fork;
- in-place mutation of the user's torch install (reversible via
  `torch/lib/_orig_backup/`, but invasive and order-sensitive);
- every torch upgrade blocked on the vendor publishing a matching fork.

## What the linker actually requires

Counting dynamic symbols with `nm -D` and diffing against the stock core:

| Consumer | Undefined syms resolved by the DTK core | Of those, absent from `2.10.0+cpu` |
|---|---|---|
| `libc10_hip.so` | 39 | **0** |
| `libtorch_hip.so` | 2517 | **16** |
| `libtorch_fl.so` (this plugin) | 2267 | **16** |

The DTK core fork is, from the linker's point of view, almost entirely
upstream-compatible. The 32-symbol gap is not spread across the ABI; it is one
feature, and both halves trace to the same DTK patch to `ATen/autocast_mode.h`,
which adds 32 entries to the fp32-cast policy macro lists:

- `libtorch_hip.so` imports 16 `at::_ops::native_fuse_*::call` wrappers. The
  fused *kernels* live in `libtorch_hip.so` itself; only the generated schema
  wrappers live in the vendor's `libtorch_cpu.so`, which is why the device library
  imports them from the core.
- `libtorch_fl.so` imports 16 `at::_ops::fuse_*::call` wrappers, from one object
  file (`aten/autocast.cc.o`): `csrc/aten/autocast.cc` expands the macro list, so
  every DTK-added entry becomes an `at::_ops::<op>::call` reference. The plugin is
  compiled against DTK's patched headers, so it inherits them.

Full symbol list: `torch_fl/accelerator/dcu/dtk_core_compat_symbols.txt`.

## How the decoupled mode works

### 1. A 32-symbol compatibility shim

`csrc/runtime/accelerator/dcu/dtk_core_compat.cc` defines all 32 symbols. Each is
a C function aliased to the C++ mangled name:

```cpp
extern "C" [[noreturn]] void identifier() __asm__("_ZN2at4_ops12fuse_rmsnorm4callE...");
```

The body throws a `std::runtime_error` naming the op and pointing at
`FLAGOS_DCU_VENDOR_CORE=1`. Nothing on any torch_fl route reaches them:
`torch_fl/codegen_skip_ops.txt` already excludes every `native_fuse_*` entry from
codegen, because the DTK-only schemas do not exist on other platforms. The shim
therefore satisfies the loader without pretending the ops work.

It builds as `libflagos_dtk_core_compat.so` (~16 KB) and installs into
`torch_fl/lib_dcu/`.

### 2. A build-time ABI guard

`scripts/check_dcu_core_abi.py` recomputes the gap from the binaries rather than
trusting the checked-in list. It reports

```text
(undefined syms of the device libs + the plugin)
  ∩ (DTK core exports) − (official core exports)
```

and fails unless every symbol in that set is allowed by the manifest and exported
by the shim. The manifest is deliberately a compatibility superset: its 16
`native_fuse_*` symbols are imported by DTK's device library in every build, while
its 16 `fuse_*` symbols are imported by `libtorch_fl.so` only when the plugin was
compiled against DTK's patched headers. CI compiles the plugin against official
headers, so that valid build has a 16-symbol gap rather than 32.
`scripts/bundle_dcu_libtorch.sh` runs the guard on every decoupled bundle, so a DTK
release that adds a fused op fails the build with the new symbol
named, instead of failing at `import torch_fl` with a mangled undefined symbol.
The guard needs the vendor core's *exports*, so it is pointed at DTK's own
`torch/lib`, not at the pruned bundle.

`tests/unit/test_dcu_core_abi.py` drives the guard over purpose-built `.so` files
and asserts the shipped manifest still matches the shim source (16 + 16).

### 3. Preload instead of relink

`torch_fl/accelerator/dcu/_dcu_libtorch_link.py` dlopens, in order, before
`import torch`:

```text
official torch/lib:  libc10.so  libtorch_cpu.so  libtorch.so
bundled lib_dcu:     libflagos_dtk_core_compat.so  libcaffe2_nvrtc.so
                     libc10_hip.so  libmagma.so  libtorch_hip.so
```

All `RTLD_GLOBAL`. Three ordering constraints, each measured:

- The official core must be mapped first. The bundle's RUNPATH is `$ORIGIN` plus
  the DTK driver directories and deliberately does not name the official
  `torch/lib`, so `libtorch_hip.so`'s hard `DT_NEEDED` on `libc10.so` would
  otherwise be unresolvable. Loading by path up front satisfies it by soname
  against the already-mapped object — the same inode `import torch` maps a moment
  later, so there is no second copy and no duplicate static init.
- The shim must precede `libtorch_hip.so`, else its 32 exports are not in the
  global scope when the loader binds the device library.
- All of it must precede `import torch`. PyTorch caches its CUDAHooks on first
  import; loading afterwards leaves device init failing with "Cannot initialize
  CUDA without ATen_cuda library" even though the kernels did register. See
  constraint 1 in [../cuda/external-libtorch-cuda.md](../cuda/external-libtorch-cuda.md).

`libcaffe2_nvrtc.so` is in the list for a different reason: ATen's lazy NVRTC stub
reaches it through a `dlopen` **by soname**, not through any `DT_NEEDED`, so
RUNPATH does not help. Without it already mapped, constructing a CUDA generator
fails with `Error in dlopen: libcaffe2_nvrtc.so`.

Nothing under the official `torch/lib` is touched, so torch_fl installs beside a
stock torch and uninstalling leaves no trace.

Legacy mode has one extra ordering constraint, for the opposite reason: it *does*
symlink the bundle into `torch/lib`, and `LD_LIBRARY_PATH` (which names `torch/lib`
in a source checkout, and does so in CI) is searched ahead of every RUNPATH. So a
transitive `DT_NEEDED` resolves to the symlink, and glibc expands `$ORIGIN` from the
path the object was *opened by* — `torch/lib`, where the bundle's hash-suffixed deps
do not exist. Measured on the CI image: `libtorch.so` -> `libtorch_hip.so` ->
`libmagma.so` picked up through the symlink dies with

```text
OSError: libmkl_gf_lp64-e350bb11.so: cannot open shared object file
```

even though that MKL library sits in `lib_dcu` right next to `libmagma.so`. Opening
the same file from the bundle path loads it, and the later by-soname resolution then
matches the already-mapped object by inode. The legacy preload list therefore covers
every `.so` it symlinks, `libmagma.so` included, ahead of `libtorch.so`.

### 4. A `torch.cuda` shim

The official `+cpu` wheel's `libtorch_python.so` was compiled without CUDA, so
`torch.cuda.is_available()` is False and `_lazy_init()` raises "Torch not compiled
with CUDA enabled". The compute path never notices — it goes through C++ boxing
into DTK's CUDA-key kernels — but everything that asks *Python* whether a CUDA
device exists does:

- triton's hcu backend gates `is_active()` on
  `torch.cuda.is_available() and torch.version.hip is not None`, so False means
  zero active drivers and every FlagGems op dies in triton's driver factory;
- inductor's `codecache` fingerprint reads `gcnArchName` off the device
  properties whenever `torch.version.cuda is None`;
- `torch.manual_seed()` walks `torch.cuda.default_generators`, empty on the CPU
  wheel;
- `torch.cuda.Event` is a dummy base class that raises "Tried to instantiate dummy
  base class Event", and triton's autotuner times every candidate config with
  `Event(enable_timing=True)` — so making `is_available()` True without this makes
  each autotuning FlagGems op fail one step later, inside `do_bench`.

`patch_torch_cuda_for_dcu()` in `torch_fl/accelerator/dcu/_dcu_compat.py` answers
those from the flagos runtime and the HIP driver instead, mirroring the existing
MetaX shim. Device properties are read through `libgalaxyhip.so.5` by attribute
**id** (layout-independent), except `gcnArchName`, which is scraped from the
`hipGetDeviceProperties` blob because the `GcnArchName` attribute id errors on
DTK. Measured, the shim reports exactly what DTK's own torch reports:

```text
BW  9 3  80 CUs  warp 64  L2 8388608  gfx936:sramecc+:xnack-  68702699520 bytes
```

Generators are real CUDA generators (16-byte philox state), which is what
FlagGems' hygon backend unpacks as `(seed, offset)`; the flagos PrivateUse1
generator is a CPUGeneratorImpl whose state is the ~5 KB mt19937 blob and would
blow that unpack up.

`torch.cuda.Event` is pointed at `torch.flagos.Event`, which falls back to host
timing when no native CUDA event is reachable. Timing only decides which autotune
config wins, so a host clock is sufficient; the kernels still run on the DCU.

`torch.version.hip` is restored separately, from the vendor's own `version.py`
copied into the bundle: it is pure Python generated at build time, so swapping
`.so` files cannot change it.

### 5. A post-import runtime check

`torch_fl/accelerator/dcu/_dcu_runtime_check.py` runs right after `import torch`:

1. **Base version alignment.** A mismatched official wheel is an ABI mismatch that
   `dlopen` does *not* reject, because the symbols it needs exist with the same
   names. The bundled `vendor_version.py` records which torch DTK built against.
2. **CUDA dispatch presence.** `aten::mm`, `aten::add.Tensor`, `aten::_softmax`
   and `aten::bmm` must all have a `CUDA` kernel. If `libtorch_hip.so` did not
   load, or loaded `RTLD_LOCAL`, they do not, and every boxed op would fail with
   "Could not run 'aten::mm' with arguments from the 'CUDA' backend".

Both raise. `FLAGOS_DCU_SKIP_RUNTIME_CHECK=1` bypasses them, for deliberately
testing a non-matching wheel pair.

## Measured results

Bundle contents, decoupled vs legacy:

| | Files | Size |
|---|---|---|
| Legacy (`FLAGOS_DCU_VENDOR_CORE=1`) | 27 | 1.8 GB |
| Decoupled (default) | 11 | 1.3 GB |

What remains vendor-side: `libtorch_hip.so` (881 MB, the kernels themselves),
`libmagma.so` (360 MB), `libc10_hip.so`, `libcaffe2_nvrtc.so`, the shim, and four
hash-suffixed common libs the device libraries `DT_NEEDED`
(`libgflags-*`, `libglog-*`, `libmkl_*`). `libmagma.so` cannot be dropped: the
stock CPU wheel has no magma, `libtorch_hip.so` needs it, and DTK ships no copy
outside its torch wheel.

Runtime, on the official `torch 2.10.0+cpu`:

```text
torch 2.10.0+cpu   hip 6.3.26113
cuda avail True  count 8
mm err 4.77e-06   softmax err 2.98e-08
props BW 9 3 80 64 gfx936:sramecc+:xnack- 68702699520
gens 8 x 16 bytes
torch/lib symlinks 0   _orig_backup False
```

FlagGems on the decoupled runtime (`FLAGOS_USE_FLAGGEMS=1`, `GEMS_VENDOR=hygon`),
which is what the `torch.cuda` shim exists for:

```text
hcu is_active True   driver triton.backends.hcu.driver.HIPDriver
[flagos dispatch] randn -> flagos_python
[flagos dispatch] add.Tensor -> flagos_python
[flagos dispatch] mm -> flagos_python
add err 0.0   mm err 9.54e-07
```

Non-zero device: `flagos:3` mm err 2.86e-06. Bundle gates:
`DCU core ABI check passed: 32 vendor-private imports are covered by
libflagos_dtk_core_compat.so`, `Verified: no vendor core lib in the bundle`, and
the `DT_NEEDED` self-check reports no unresolved names.

Legacy rollback re-verified end to end (`legacy torch 2.10.0+cpu 6.3.26113 err
1.19e-06`), with `restore_original_libtorch()` leaving `torch/lib` clean.

The packaged wheel was also installed on its own, into a venv holding nothing but
the official `torch 2.10.0+cpu`, and run from a directory outside the checkout, so
that no part of the result can come from the source tree. Both modes work from the
same wheel:

```text
# default (decoupled)
torch 2.10.0+cpu hip 6.3.26113   cuda.is_available True
Event <class 'torch_fl.flagos.Event'>   device_count 8
add err 0.0   mm err 5.72e-06
grad device flagos:0   grad err 0.0
vendor_core_mode False

# FLAGOS_DCU_VENDOR_CORE=1
vendor_core_mode True   add err 0.0   mm err 2.86e-06
```

## Choosing a mode

```bash
# Default: device libs only, on the official torch wheel.
bash scripts/bundle_dcu_libtorch.sh

# Rollback: DTK's full core set, relinked into the torch install.
FLAGOS_DCU_VENDOR_CORE=1 bash scripts/bundle_dcu_libtorch.sh
```

The env var must match at build time and at import time; a mismatch is rejected
with the command needed to fix it rather than a missing-`libc10.so` error. CI
exercises both: `.github/scripts/set_env_dcu.sh` runs the decoupled wheel
invariants and an import/compute gate, then smoke-tests the legacy path and
restores the decoupled bundle.

Legacy mode is the only mode where DTK-private schemas such as
`aten::native_fuse_rmsnorm` work, since their schema wrappers and autograd
registrations live in the core fork.

## Limits

- **The shim is DTK-version-coupled.** If a DTK release adds a fused op, the list
  grows. The ABI guard fails the build with the new symbol named, and a unit test
  keeps the manifest and the shim source in agreement, but the list still has to
  be regenerated by hand.
- **It relies on the stock CPU wheel's symbol completeness**, which PyTorch does
  not guarantee (constraint 4 in the CUDA doc). It held for 2.10.0 with a measured
  32-symbol delta; re-run the guard on every torch bump.
- **`libtorch_hip.so` remains ABI-bound to its torch minor.** This is "track
  upstream without waiting for a core fork", not "one binary across versions" —
  hence the version check.
- **DTK's fused ops are unreachable** in decoupled mode. They are on no torch_fl
  route today. Making them callable would mean replacing the throwing stubs with
  forwarding ones: declare the schemas with `TORCH_LIBRARY_FRAGMENT` using DTK's
  signatures and implement `::call` as
  `Dispatcher::findSchemaOrThrow(...).call(...)`, since the kernels are already
  registered in `libtorch_hip.so`.
- **Zero vendor `.so` is not achievable** while keeping the vendor operator set.
  The operator machine code *is* `libtorch_hip.so`. What decoupling removes is the
  vendor's fork of the core runtime.
