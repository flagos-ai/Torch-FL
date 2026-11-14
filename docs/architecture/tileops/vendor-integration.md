# TileOPs vendor integration for Ascend, MUSA, and MetaX

This document describes the opt-in TileOPs experiment for three domestic
accelerators: Huawei Ascend 910C, Moore Threads MUSA on MTT S5000, and MetaX.
It separates the torch-fl routing and device boundary from the provider's
kernel implementation, and records the provider revisions used for each run.

The experiment uses dedicated backend configuration files:

| Platform | Configuration | Provider | Target |
|---|---|---|---|
| Ascend | `backends_ascend_tileops.conf` | `https://github.com/yyttt6/TileOPs` at `06fc97a46e7664b3f30bbee6d66e2abd1061972b` | `ascend` |
| MUSA (vendor `mthreads`) | `backends_mthreads_tileops.conf` | `https://github.com/ChipMind-CAS/tileops-musa` at `9f69c4f5f9cde6695a3ac9036865c7b9f45557cb` | `musa` |
| MetaX | `backends_metax_tileops.conf` | `https://github.com/ChipMind-CAS/tileops-metax` at `39342151a7aaa62647e975664aeb6cd152e10c11` | `tileops_maca` |

The upstream compiler adapters are public and separately versioned:

| Platform | TileLang adapter | Adapter revision used for source review |
|---|---|---|
| Ascend | `https://github.com/tile-ai/tilelang-ascend` | `4805777d691b0c586e647fc0f975fa4cc306c00e` |
| MUSA | `https://github.com/tile-ai/tilelang-musa` | `a7a0f5f119e10e45fb899af5dd9ddfdc7f58042a` |
| MetaX | `https://github.com/tile-ai/tilelang-metax` | `90d49f353ec88bd0d346474f143003d865dd3b79` |

All three files contain only `tileops`, the platform's own kernel, and `none`.
`none` stays unclaimed on PrivateUse1 so ATen reaches `cpu_fallback`. The
experiment never selects FlagGems and never uses CUDA boxing for an ordinary
operation.

## Build

TileOPs remains an experimental compiler artifact. Build it explicitly for the
chosen platform:

```bash
FLAGOS_ACCELERATOR=ascend FLAGOS_BUILD_TILEOPS=1 python setup.py bdist_wheel
FLAGOS_ACCELERATOR=musa FLAGOS_BUILD_TILEOPS=1 python setup.py bdist_wheel
FLAGOS_ACCELERATOR=metax FLAGOS_BUILD_TILEOPS=1 \
  FLAGOS_BUILD_BOXING=0 python setup.py bdist_wheel
```

The MetaX build must keep boxing off. Its experiment configuration depends on
`register_tileops_metax.inc`, which claims exactly the 60 TileOPs operators and
leaves every other operator unclaimed for CPU fallback. A normal boxing build
continues to use the full generated wrapper set.

Select the experiment at runtime after installing the matching provider:

```bash
export FLAGOS_BACKEND_CONFIG=/absolute/path/to/backends_<vendor>_tileops.conf
export FLAGOS_TILEOPS_PROVIDER=tileops_musa    # MUSA
export FLAGOS_TILEOPS_PROVIDER=tileops_maca    # MetaX
export FLAGOS_TILEOPS_PROVIDER=tileops         # official TileOPs or Ascend fork
```

The provider variable selects the Python module imported first. The target and
its builders are discovered from the installed TileOPs provider. A default
Torch-FL wheel continues to omit the TileOPs key.

## Dispatch and fallback

The generated shim calls `resolve_impl()`, which checks three facts before an
operator reaches a provider:

1. The provider claims a target for the device.
2. Torch-FL has a tensor and stream adapter for that target.
3. The provider registers the Op class on that target.

Failure of any check returns to the configured vendor route or CPU fallback.
Instances are cached by target and constructor arguments, so kernels built for
one accelerator cannot be reused by another. Enable `FLAGOS_LOG=dispatch` to
see both the C++ route decision and a declined TileOPs call.

The runtime accepts both provider generations used by this delivery:

- The July TileOPs API commits the output dtype and, for elementwise operators,
  the flattened size or operand shapes in the Op constructor.
- The September API moved those facts to the first forward call, leaves
  `strategy` and tuning internal, and accepts an explicit `target`.

`target` is consumer plumbing in either generation. `strategy` selects a
kernel variant and is intentionally absent from the generated recipes.

## Device boundaries

MetaX C500 and C550 use the CUDA ABI. Torch-FL exposes flagos storage through the
zero-copy `_flagos_to_cuda_view` boundary; the MACA provider consumes that view
as a CUDA-compatible tensor. Results return through `_cuda_to_flagos_view`. A
C550 gives the same CUDA-ABI route but is not a C500 performance or qualification
result. The validation below ran on C550 and is recorded with that distinction.

MUSA and Ascend expose their own device types (`musa` and `npu`). Torch-FL owns
PrivateUse1 as `flagos`, so a provider that compares `TensorSpec.device.type` or
the input tensor's device against `musa` / `npu` rejects a flagos tensor. The
runtime carries a native adapter contract with zero-copy conversions and a raw
stream lookup, but the pinned provider and TileLang-MUSA/Ascend revisions keep
exact device-type checks and do not publish a `flagos` storage conversion. The
Ascend adapter has a usable raw-payload launch path for an NPU tensor, while
the MUSA adapter's launch code consumes `torch.musa` storage and streams; both
still need the same cross-plugin storage identity. Torch-FL does not monkeypatch
those checks or the TileLang adapters; doing so would hide a boundary mismatch
and would violate the provider repositories' integration contracts.

The MUSA experiment therefore has two explicit prerequisites before it can run
on hardware:

- a provider revision that describes a flagos tensor as its internal `musa`
  storage while preserving S5000 qualification checks;
- a TileLang-MUSA kernel adapter that consumes torch-fl's flagos storage and raw
  stream without rewriting the upstream device type.

The Ascend experiment has the equivalent `flagos` / `npu` requirements. Its
detector is intentionally disabled upstream until all operators have builders,
so Torch-FL selects the target by name after confirming that `ascend` is in the
provider's registered-target table. This selection does not bypass the
provider's per-call builder registration or the device boundary.

## Validation

The dedicated CI entrypoint is `.github/workflows/tileops-ci.yml`. Relevant PRs
run the generated-source, configuration and runtime-contract checks on a CPU
runner. The MetaX smoke is manual or nightly only and runs a five-case pytest
file that requires no skips. It verifies the wheel build record contains only
`tileops`, executes two calls of each representative TileOPs route, checks the
returned `flagos` device and CPU values, and requires `transpose.int -> none`.
The private provider is pinned by commit and fetched through
`TILEOPS_GITHUB_TOKEN` without placing the token in a URL or command line.

Run the source and configuration checks before a hardware run:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 GEMS_VENDOR=cuda \
  python scripts/codegen/codegen_tileops.py --check
TORCH_DEVICE_BACKEND_AUTOLOAD=0 GEMS_VENDOR=cuda \
  PYTHONPATH=$PWD python scripts/codegen/gen_vendor_confs.py --check
TORCH_DEVICE_BACKEND_AUTOLOAD=0 GEMS_VENDOR=cuda \
  pytest tests/unit/test_tileops_runtime.py tests/unit/test_gen_vendor_confs.py \
         tests/unit/test_conf_registration_consistency.py -q
```

On each accelerator, verify at least one unary, binary, reduction, and softmax
route. For each, record the provider package path and commit, target selected,
the `FLAGOS_LOG=dispatch` output, output device type, numerical comparison
against the vendor or CPU reference, stream ordering, and one deliberately
unsupported route reaching `cpu_fallback`.

Do not report a source registration, successful import, or generated
configuration as device support. The distinction matters most for MUSA and
Ascend, where the current device-type boundary is unresolved.

The MetaX C550 validation used torch `2.10.0+metax3.8.1.0`, MACA `3.8.1.3`,
TileOPs `0.0.1.dev1+g04d08e8b5` from public commit
`04d08e8b52bf9fa98c55f862534b1b31de85b1eb`, `tileops-maca` at
`39342151a7aaa62647e975664aeb6cd152e10c11`, and TileLang-MetaX at
`ee99db668929e2da70b1e01ae70d0f59bb0e455d`. With the non-boxing
`backends_metax_tileops.conf`, two consecutive calls of each route produced:

| Route | Target | Output device | Maximum absolute error |
|---|---|---|---|
| `relu` | `tileops_maca` | `flagos:0` | 0 |
| `add.Tensor` | `tileops_maca` | `flagos:0` | 0 |
| `sum.dim_IntList` | `tileops_maca` | `flagos:0` | 3.81e-6 |
| `_softmax` | `tileops_maca` | `flagos:0` | 1.19e-7 |

The generated dispatch log reported `relu`, `add.Tensor`, `sum.dim_IntList`,
and `_softmax` as `tileops` for both calls. A `transpose.int` call, configured
as `none`, returned a `flagos:0` tensor after CPU fallback. These are C550
functional results, not C500 qualification or performance results.
