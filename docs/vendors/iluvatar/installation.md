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

The isolated `/opt/torch-fl-iluvatar-venv` disables system and user site packages.
The image preinstalls FlagTree `0.6.2a3+iluvatar3.6` (with Triton 3.6), FlagGems
`5.4.0` and build/test dependencies. The hook clears inherited PYTHONPATH,
checks requested versions against checkout pins/overrides, and replaces only
mismatched components in the job container. A FlagTree change replaces its
Triton provider as a pair. `pip check` validates the resulting dependencies;
the checkout's constraints protect CPU torch. Stable dependency updates can be
baked into a later image. SDK, Python or CPU torch changes require a rebuild.

The published image still contains the old, separately staged vendor libtorch
assets. The hook clears `FLAGOS_VENDOR_TORCH_LIB` and vendor version metadata,
including in subsequent workflow steps. Those assets are neither selected nor
bundled. The old Iluvatar facade helper, bundler and setup.py ABI overrides have
been removed in favor of the upstream CPU-core implementation. A new image push
is not required to use the existing digest with this hook. The checkout
[verification script](../../../.github/scripts/verify_iluvatar.py) checks the
isolated CPU environment and installed runtime used by CI.

## Host device access

The inspected u121 host has 16 BI-V150 cards, Ubuntu 20.04, driver/IX-ML 5.0.0
and CoreX SDK 4.5.0. The validated container configuration maps `/dev/itrctl`
and `/dev/iluvatar0` through `/dev/iluvatar15`, uses explicit runc, unlimited
memlock and 16 GiB shared memory. Computation stays reserved to physical cards
8 and 9 through `CUDA_VISIBLE_DEVICES=8,9`; keep that reservation on the runner.

Read-only mounts from `/usr/local/corex-4.5.0` provide:

- `lib64/libcuda.so.1` and `lib64/libixml.so` at matching CoreX paths;
- `bin/ixsmi` at `/usr/local/corex/bin/ixsmi`.

Omitting `/dev/itrctl` caused initialization error 1001 even with compute nodes
mapped. The base's zero-byte ixsmi placeholder caused Triton's matmul clock
query to fail; the real host tool is required. These mounts remain necessary
with the independent runtime. The SDK header probe uses native
`ixblasDataType_t` and `IX_R_16BF`, matching upstream's corrected type handling.

## CI image and runtime acceptance

The operator published the baseline on 2026-10-17:

```text
harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl@sha256:bc1b3c0beb22b301ec501896ea1c77903879526fb7303358669321d609458023
```

CI consumes this published image; image build recipes are maintained outside
this repository. When adopting a validated replacement, update `ci_image` to
its registry manifest digest.

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
