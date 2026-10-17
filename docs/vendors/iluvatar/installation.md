# Iluvatar BI-V150 CI bring-up

The lane uses the existing shared integration workflow, not a separate test
pipeline. The active config is
[`.github/configs/iluvatar.yml`](../../../.github/configs/iluvatar.yml).
It is discovered by PR CI and by pushes to `iluvatar_build_and_test`. Image
publication is complete; the first PR run must establish runner availability
and shared integration results. Configuring a lane does not imply green CI.

## Official CPU frontend and separate CoreX binaries

The image provides a clean `/opt/torch-fl-iluvatar-venv`, with system-site
packages disabled, official `torch==2.10.0+cpu`, FlagTree
`0.6.2a3+iluvatar3.6` and FlagGems `5.4.0`. The hook reuses these packages when
they match the checkout's version pins; mismatches install the requested binary
release in the job container using the common retry helpers. A FlagTree change
removes the previous FlagTree/Triton provider first. Build/test tools and Python
dependencies are baked into the venv;
PyYAML 6.0.1 and SQLAlchemy 2.0.48 satisfy FlagGems' dependency bounds.

The checkout's constraints protect CPU torch during provisioning, and its
verifier checks the requested versions, the Iluvatar Triton backend, isolation
and SDK headers. `pip check` rejects incompatible dependency cohorts. Newly
required dependencies must be provisioned explicitly if that check fails; a
compiler upgrade does not justify installing a vendor torch frontend or stock
Triton. Publish a refreshed baseline image after the new cohort is validated.
SDK, Python or CPU torch changes require a rebuilt and revalidated image.

The base remains the pinned CoreX development image in the
[Dockerfile](../../../docker/iluvatar/Dockerfile). The smaller runtime 2.3.0
failed native compilation because SDK header types were missing. That failure
was not a CPU-torch compatibility experiment.

A separate experiment installed the official CPU wheel in a clean venv.
CoreX's `libc10_cuda.so` failed direct loading against its core with an undefined
old-ABI `c10::Error` symbol. The official CPU core reports C++ ABI true, whereas
CoreX 4.5.0 requires ABI false. Reusing the existing private-facade helper with
the matched CoreX binary set then passed CUDA initialization and device add.

Native headers/CMake files/libraries are staged under
`/opt/torch-fl-iluvatar-libtorch`. `setup.py` uses that explicit native SDK root
for Iluvatar while the Python build frontend remains official CPU torch. The
compatibility manifest records the CPU distribution separately from native
C++ ABI false and vendor binary provenance.

The hook runs `scripts/vendor/bundle_iluvatar_libtorch.sh` before wheel staging.
The wheel carries the matched native binaries in `torch_fl/lib_iluvatar/`.
At import, the plugin selects those through the same private torch facade used
by PPU/MetaX; it never overwrites the installed official torch files. Actual
runtime core libraries are CoreX binaries, not the official CPU core. No
separate vendor torch Python package is installed into the venv.

Import `torch_fl` before `torch` in fresh interpreters. The image disables
`TORCH_DEVICE_BACKEND_AUTOLOAD` because the torch-first backend entry point runs
too late for a core replacement. Children inherit the selected facade. The
base image's global Python packages, including vLLM, remain outside the clean
venv's import path.

Iluvatar stays FlagGems-primary with a native vendor slot and no CUDA boxing,
FlagGems C++ or TileOps. No operator route changes in this bring-up. See
[operator support](../../reference/operator-support.md) and the
[capability matrix](../../reference/platform-capability-matrix.md).

## Host device access

The inspected u121 host has 16 BI-V150 cards, Ubuntu 20.04, driver/IX-ML 5.0.0
and CoreX SDK 4.5.0. The older application guide's `/dev/corex` mount does not
apply. The validated launch maps `/dev/itrctl` alongside `/dev/iluvatar0` through
`/dev/iluvatar15`, uses explicit runc, unlimited memlock and 16 GiB shared memory.
Computation stays reserved to physical cards 8 and 9 using
`CUDA_VISIBLE_DEVICES=8,9`. Confirm that reservation for the registered runner.

Read-only mounts from `/usr/local/corex-4.5.0` supply:

- `lib64/libcuda.so.1` and `lib64/libixml.so` at the corresponding CoreX paths;
- `bin/ixsmi` at `/usr/local/corex/bin/ixsmi`.

A host cuInit trace identified `/dev/itrctl`: omitting it caused error 1001 even
with all compute nodes mapped. The base's ixsmi is a zero-byte placeholder;
omitting the real host tool caused Triton's matmul clock query to fail with
`Exec format error`. Neither issue requires privileged containers on this host.

## Validation status

The revised CPU-frontend image passed its isolation check and `pip check`.
A fresh container ran the actual shared setup hook, complete native/wheel build,
compatibility preflight and wheel installation. Cold Triton JIT and flagos add
passed; mapped core libraries came from the installed wheel's `lib_iluvatar`,
and hashes proved the official CPU torch libraries were unchanged.

An initial clean CPU-frontend run of conf-consistency, factory and runtime tests
finished with 76 passed, 2 failed, 1 skipped (129.74 seconds). The final-image
run also included the shared startup subprocess contract and finished with
**78 passed, 2 failed, 4 skipped** (151.38 seconds). The two startup profiles
passed; its three CUDA-only import-order cases skipped through the existing
platform guard. The runtime subset reported 30 passed and 2 failed. Local
relevant unit suites passed 219 tests; Ruff, shell syntax, YAML and whitespace
checks passed.

Two existing failures from the shared runtime contract are retained:

- `test_dispatch_matches_conf_route[mul.Tensor]`: its conf route is `none`,
  whereas the common runtime contract requires an accelerator route.
- `test_dispatch_matches_conf_route[bmm]`: the contract's small shape fails
  CoreX Triton compilation at `tl.dot` with a `K >= 16` requirement. The larger
  factory-suite bmm shape passed in the earlier environment.

Neither is hidden by deselection. Other shared groups still need measurement;
smoke results alone do not establish a green CI lane.

## Operator-managed publication

Use the revised tag; do not resume the interrupted push of the old `-dev` tag.
The operator performs publication and returns the manifest digest:

```bash
image=harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20261016-iluvatar-cpu-dev
docker image inspect --format '{{.Id}}' "$image"
docker login harbor.baai.ac.cn
docker push "$image"
docker image inspect --format '{{json .RepoDigests}}' "$image"
```

The local image ID is not the registry manifest digest. Record the pushed
repository digest in `ci_image`; credentials belong in neither source nor chat.

The operator confirmed the completed push on 2026-10-17:

```text
harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl@sha256:bc1b3c0beb22b301ec501896ea1c77903879526fb7303358669321d609458023
```

The agreed runner label is `ts-v150-cicd-pytorch`. The active config now
records both values; this does not establish that the runner is online.

Validated candidate local image ID:
`sha256:9ac45c1978ba7d699cc2eec3a836e51532113c3e872529c4fa08885e5a17c3c7`.
The final tag's filesystem and environment match the tested image; the last
build only clarified that GPU validation requires an installed wheel. No
Harbor push was performed by the agent.

## Shared pipeline and PR validation

Keep the existing `integration-tests.yml` sequence: common setup, native
prebuild, wheel build, compatibility preflight, install, wheel-only test
workspace and the composed shared integration manifest. Each job builds the
revision under test; the image contains no prebuilt torch_fl wheel.

The config and manifest are wired together. The lane selects the shared
environment, startup, multi-device, portable top-level, vendor and FlagGems
ops, RNG, factory, AMP, math-bits, profiler contract/parity, compile, runtime
and FlagGems conformance groups. No additional deselections hide known failures.
The AMP group follows the existing capability table: its device-independent
API cases run, while Iluvatar device AMP remains gated pending measurement.
The software low-precision group remains DCU/MetaX-specific. Qwen3 inference
and training remain pending a model mount; the DCU native communication bridge,
MUSA-specific dispatch and DCU's full per-file unit sweep retain their existing
platform scopes. Hardware-independent regression tests run in agnostic checks.

For the first PR:

1. Confirm that the self-hosted runner is online with label
   `ts-v150-cicd-pytorch`, and check its device reservation and Docker access.
2. Verify that the runner can pull the published image at the recorded digest.
3. Measure shared groups against the installed wheel, preserving actual failures
   and documenting proven capability gaps with re-enable conditions.
4. Inspect every selected shared group; local smoke results do not cover the
   full manifest, profiler parity or compile behavior on this host.
5. Use `platform:iluvatar` to select this lane for a PR against `main`. Shared
   checks still run. Fork hardware dispatch follows the repository's trusted
   maintainer controls in [CI controls](../../development/ci-controls.md).

The ops collection gate distinguishes native Iluvatar from CUDA boxing while
keeping FlagGems coverage enabled. The previous survey's removed routes and
synthetic-case failures are not a blanket pytest exclusion list.
