# Iluvatar CI image

The published CPU-frontend baseline is
`harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20261016-iluvatar-cpu-dev`.
It supersedes the earlier `manual-20261016-iluvatar-dev` development experiment.
The operator confirmed publication on 2026-10-17. The CI config pins
`harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl@sha256:bc1b3c0beb22b301ec501896ea1c77903879526fb7303358669321d609458023`
and uses the agreed runner label `ts-v150-cicd-pytorch`.
The Dockerfile pins the CoreX development base by digest, preserving the SDK
headers that the smaller runtime 2.3.0 image lacked.

## Python isolation and native assets

`/opt/torch-fl-iluvatar-venv` is a normal venv with
`include-system-site-packages=false`. It installs official
`torch==2.10.0+cpu`, clears inherited PYTHONPATH and disables user-site packages.
Base-image vLLM and vendor Python packages are not on its import path.

CoreX native headers, CMake files and shared libraries are staged separately
under `/opt/torch-fl-iluvatar-libtorch`. These are build/runtime assets, not an
installed vendor Python frontend. The pinned base's exact FlagTree distribution
is copied into the clean venv by its wheel RECORD; no global site directory is
added to sys.path. FlagGems and build/test dependencies are installed there.
The exact installed versions are saved in `/opt/torch-fl-ci/installed.txt`.

CI reuses the preinstalled FlagTree/Gems cohort when it matches the checkout's
`.github/version-pins.env`. A changed pin replaces only that component inside
the job container. FlagTree supplies Triton as a pair; stock Triton is not
installed separately. The hook checks dependencies and runs the checkout's
verifier after provisioning, so stale image pins do not block a new cohort.
Keep the SDK, Python and CPU torch baseline fixed until a new image is validated.

CoreX 4.5.0's device libraries use the old C++ ABI and fail direct loading against
the official CPU core. The plugin therefore bundles the matched binaries under
`torch_fl/lib_iluvatar/`, using the existing private-facade mechanism also used
by PPU/MetaX. The official CPU torch installation remains unchanged, but its
process runs the bundled CoreX core after `import torch_fl`. This is not a claim
that all loaded core libraries are official CPU binaries.

Import `torch_fl` before `torch` when selecting the CoreX core in a fresh
interpreter. The image disables torch backend autoload because a torch-first
entry point runs too late to select a different core. Neither torch_fl source
nor its wheel is baked into the image: CI builds the revision under test.

## Build and validate

```bash
DOCKER_BUILDKIT=0 docker build -f docker/iluvatar/Dockerfile \
  -t harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20261016-iluvatar-cpu-dev .
```

The default CPU-wheel mirror is the same SJTU mirror supported by the shared
CI helper. Set `CPU_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` to use
the official index directly. Ordinary packages use `PIP_INDEX_URL`. Do not put
proxy credentials in build arguments or the image environment.

Build-time `verify.py` checks the untouched CPU frontend, venv isolation,
FlagTree/Gems pins and SDK headers without requiring devices. After building
and installing a wheel, `verify.py --device` starts a separate interpreter to
check bundle selection, actual mapped core-library paths, cold Triton JIT and
flagos tensor arithmetic. The CPU and CoreX checks cannot share an interpreter.

See [the bring-up guide](../../docs/vendors/iluvatar/installation.md) for the
host device/control-node and driver-tool mounts, measured results and push
steps. The active config is [`.github/configs/iluvatar.yml`](../../.github/configs/iluvatar.yml).
Its first PR run must confirm runner access and the remaining shared hardware
groups; the previously measured runtime failures remain visible.
