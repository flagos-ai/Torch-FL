# Iluvatar BI-V150 CI bring-up

The lane uses the shared integration workflow and
[platform config](../../../.github/configs/iluvatar.yml), with runner label
`ts-v150-cicd-pytorch`. Each job builds and installs the revision under test,
then runs tests against the installed wheel outside the source checkout.

## CPU torch and independent CoreX runtime

Upstream [PR #593](https://github.com/flagos-ai/Torch-FL/pull/593) introduced
`csrc/runtime/accelerator/iluvatar/`, its stream registry, allocator and Python
compatibility layer. It calls the CoreX C ABI without linking vendor
`libtorch_cuda.so` or `libc10_cuda.so`. Build headers, CMake files, core libraries
and the C++ ABI come from the active official CPU torch installation. The
compatibility manifest records C++ ABI true, `bundle_libdir=lib`, and
`vendor_torch_libraries=false` for this image.

`torch==2.10.0+cpu` remains both the Python frontend and the actual runtime
core. Its files are not overwritten and `torch.version.cuda` remains `None`.
The Python shim provides the CUDA-shaped device APIs used by FlagGems/Triton;
it does not turn CPU torch into a CUDA build. Iluvatar remains FlagGems-primary
with native vendor kernels, boxing disabled, and no FlagGems C++ or TileOps.
This CI adaptation changes no operator routes or expected numeric results.

The hook creates a fresh job-local Python 3.12 venv under
`${RUNNER_TEMP}/torch-fl-iluvatar-${CI_STAGE}` (or `TORCH_FL_VENV_ROOT`). System
and user site packages are disabled. The official runtime's `/flagos` contains
vendor Torch `2.10.0+corex.4.5.0.20260804`; it stays outside CI, which installs
the official `2.10.0+cpu` wheel instead. No vendor libtorch is copied or loaded.

Before installing packages, the shared `seed_runtime.py` copies compatible
distributions from the runtime by their RECORD entries. Each copy is independent;
the runtime's complete site-packages and compiler side directories are never
added to the CI import path. This can reuse FlagGems 5.4.0, NumPy 1.26.4 and
matching tools. CI's FlagTree pin matches the image's `0.7.1a1+iluvatar3.6`,
so its compiler can be copied instead of downloading the older release. A
replacement removes the complete old Triton provider. FlagCX is
not copied merely because the runtime includes it; this lane has no FlagCX pin.

`.github/requirements-iluvatar.txt` defines Python build/test tools and FlagGems
dependencies. SQLAlchemy follows FlagGems 5.4.0's `>=1.4.31,<2.1` requirement,
which accepts the image's 2.0.55 instead of forcing a downgrade to 2.0.48.
A local version/dependency check skips pip entirely for satisfied
requirements; pip fills gaps under the checkout's CPU-Torch constraints.
System gcc/g++, make and readelf remain available from the image. CMake's Python
distribution is needed by the build frontend, with a compatible 3.x version
rather than the old development image's exact patch pin. `pip check` validates
the resulting environment, and setup logs report copy/preparation time.

The runtime also contains NVIDIA compatibility headers under
`/usr/local/cuda/include`, which the profiler's occupancy probe may add to the
build. The Iluvatar target places the CoreX include directory first: otherwise
`cublas_v2.h` resolves to the NVIDIA header and compilation loses CoreX's
`ixblasDataType_t` and `ixblasStatus_t`, even though the isolated SDK probe passes.
The first new-runtime run reproduced this header precedence failure after its
dependency checks succeeded; a host-only CMake regression covers the conflict.

After the image initializes its SDK environment, the hook persists
`BASH_ENV=/dev/null` so later Bash steps cannot select `/flagos` again.
It clears inherited PYTHONPATH and vendor Torch library/header/CMake paths,
preserves CoreX paths, and exports the venv and SDK settings to subsequent steps.
`FLAGOS_VENDOR_TORCH_LIB` and vendor version metadata remain empty. The checkout
[verification script](../../../.github/scripts/verify_iluvatar.py) checks the
isolated CPU environment and installed runtime used by CI.

## Host device access

The inspected u121 host has 16 BI-V150 cards, Ubuntu 20.04, driver/IX-ML 5.0.0
and CoreX SDK 4.5.0. The validated container configuration maps `/dev/itrctl`
and `/dev/iluvatar0` through `/dev/iluvatar15`, uses explicit runc, unlimited
memlock and 16 GiB shared memory. Computation stays reserved to physical cards
8 and 9 through `CUDA_VISIBLE_DEVICES=8,9`; keep that reservation on the runner.

Read-only mounts from `/usr/local/corex-4.5.0` provide:

- `lib64/libcuda.so.1` and `lib64/libixml.so` at the runtime's
  `/usr/local/corex/lib/` paths;
- `bin/ixsmi` at `/usr/local/corex/bin/ixsmi`.

Omitting `/dev/itrctl` caused initialization error 1001 even with compute nodes
mapped. The old development base's zero-byte ixsmi placeholder caused Triton's matmul clock
query to fail; the real host tool is required. The new runtime ships driver
user-space libraries under `lib`, so mounts target that directory to prevent
the image's copies from preceding the host's matching driver. These mounts remain necessary
with the independent runtime. The SDK header probe uses native
`ixblasDataType_t` and `IX_R_16BF`, matching upstream's corrected type handling.

## CI image and runtime acceptance

CI uses the official COREX 4.5.0 runtime 2.3.0, pinned to the registry manifest
inspected on 2026-11-10:

```text
harbor.baai.ac.cn/flagos-runtime/flagos-runtime-iluvatar-corex4.5.0:2.3.0@sha256:41e27e638326bfe97adc45e35dcb8cfacd27487a15d3ce395f005be84f62e08d
```

Its [official package list](https://github.com/flagos-ai/build-infra/blob/main/docs/content/zh-cn/runtime/iluvatar-corex4.5.0.md)
and [base build recipe](https://github.com/flagos-ai/build-infra/blob/main/base/iluvatar-corex4.5.0)
describe Python 3.12, the SDK, compiler and driver layout. CI uses the image
directly; no derived image or Dockerfile is needed. The historical results below
used the former development image and do not validate the new runtime. Native
build, driver loading, cold Triton compilation and complete integration results
must be confirmed by its first CI run. There is no Iluvatar boxing lane;
the build and environment assertions keep boxing disabled.

The previous vendor-core lane reached the runner, built a wheel and passed
startup and multi-device groups. Its last completed run
[#1567](https://github.com/flagos-ai/Torch-FL/actions/runs/39248604889) stopped at
the portable group with **3 failed, 233 passed, 1 xfailed**. Those failures are
tracked in [issue #570](https://github.com/flagos-ai/Torch-FL/issues/570): two
float64 matmul cases and one incorrect keepdim mean reduction. These are
historical vendor-core results, not validation of the new CPU-core lane.

The adapted CPU-core lane was exercised in
[run #1584](https://github.com/flagos-ai/Torch-FL/actions/runs/39284684295/job/118168010926)
at revision `a92e8ead5df3148cee084cecbe750a98adda7ebe`. Build, wheel installation,
the CPU-core runtime audit, startup and multi-device checks passed. The portable
group still reported **3 failed, 233 passed, 1 xfailed**, reproducing the same
three cases tracked in #570 after upstream #593.

The runtime audit imports CPU torch before torch_fl, confirms official mapped
core-library paths and no CUDA libtorch linkage, checks the real BI-V150 target
(corex, arch 71, warp 64), and runs Triton addition plus flagos arithmetic. It
runs after the shared environment check and before the remaining test groups.
There are 16 selected groups including this audit; no tests are deselected to
hide #570. Groups after the failing portable group were not reached.

Full hardware acceptance remains pending successful execution of all selected
groups and verification of #570's reproductions. The existing groups cover
environment, startup, multi-device, portable contracts, vendor/FlagGems ops,
RNG, factory, AMP, math-bits, profiler contract/parity, compile, runtime and
FlagGems conformance. AMP follows the capability table's existing guard. Qwen3
inference/training still need a model mount; the DCU communication bridge,
MUSA-specific checks and DCU per-file unit sweep keep their existing scopes.

Use `platform:iluvatar` to select this lane for a PR against main; shared checks
still run. Hardware dispatch for fork PRs follows the trusted-maintainer controls
in [CI controls](../../development/ci-controls.md). A mergeable PR or a successful
image pull alone is not evidence of green hardware CI.
