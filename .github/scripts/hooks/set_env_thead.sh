# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PPU (T-Head Jianwu ZW810E) environment bootstrap for CI.
#
# Build against the official CPU torch core and use the PPU SDK runtime.
# The isolated venv supplies the published T-Head FlagTree and FlagGems wheels.

# SDK startup has already run in the image. Child shells must not restore its
# vendor Python/compiler paths after CI activates the isolated environment.
export BASH_ENV=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
# Use the official indexes by default: the PPU runner pod's HTTP proxy returns
# 500 on HTTPS CONNECT to *.tuna.tsinghua.edu.cn, so the Tsinghua PyPI and
# pytorch-wheels mirrors are unreachable from the pod (the proxy whitelists
# pypi.org and download.pytorch.org, which CI was green on before). The PPU
# image's own pip.conf points at an internal mirror that 503s, so the explicit
# index-url here bypasses it. Both stay overridable via env: flip to the
# Tsinghua mirrors once the pod proxy allows them.
#
# Setting this is not enough on its own, and that is why these lanes still saw
# Tsinghua. install_cpu_torch probes a *second* index -- the shared
# CPU_TORCH_MIRROR_INDEX_URL_DEFAULT -- and prefers it when it measures faster.
# That policy is unchanged; the default it points at is now mirror.nju.edu.cn,
# because mirror.sjtu.edu.cn redirected into the one host this comment set out
# to avoid, which is how a filelock 403 failed 6 of 14 consecutive runs across
# the T-Head lane. The fallback in install_cpu_torch still
# returns to this pin when the mirror does not serve the wheel, and the mirror
# stays enabled, because it is also what keeps the 146-190 MB CPU-torch wheel
# from taking 17-21 minutes.
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
PIP_INDEX_URL="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"

# FlagTree is the FlagOS Triton distribution, and its `ppu` variant *is* the
# `triton` package: the wheel ships triton/ (with triton/FLAGTREE_BACKEND =
# ppu), not a triton plugin, so it replaces the vendor triton rather than
# sitting beside it. cp312 only, hosted on the FlagOS index. The runner's HTTP
# proxy does not allowlist that index (it answers CONNECT with 500), so the
# install below reaches it directly -- see prefer_direct_route. Install it as:
#   python3.12 -m pip install flagtree===0.7.0+ppu3.6 \
#     --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_thead}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"

# The T-Head index publishes a Python wheel for the pinned FlagGems release.
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-thead/simple}"

# The runtime SDK lives under /usr/local/PPU_SDK. Keep the legacy hyphenated
# location as a local-development fallback. A preset PPU_SDK wins; scan candidates
# and pick the first with CUDA_SDK/lib64/libcudart.so. Mirrors
# set_env_ascend.sh's CANN toolkit scan so layout changes do not break the env.
_ppu_candidates=(
  "${PPU_SDK:-}"
  /usr/local/PPU_SDK
  /usr/local/PPU-SDK
  /opt/PPU-SDK
)
PPU_SDK=""
for _cand in "${_ppu_candidates[@]}"; do
  [[ -z "$_cand" ]] && continue
  if [[ -d "$_cand" && -e "$_cand/CUDA_SDK/lib64/libcudart.so" ]]; then
    PPU_SDK="$_cand"
    break
  fi
done
if [[ -z "$PPU_SDK" ]]; then
  echo "::error::PPU SDK not found; none of the candidates had CUDA_SDK/lib64/libcudart.so."
  echo "::error::Tried: ${_ppu_candidates[*]}. Set PPU_SDK to the SDK root."
  exit 1
fi

# Use the image interpreter for venv creation and package reuse. It need not
# have Torch: the new venv reuses or installs the official CPU wheel.
select_runtime_python() {
  local candidate
  for candidate in "${TORCH_FL_VENDOR_PYTHON:-}" python python3.12; do
    [[ -n "$candidate" ]] || continue
    if [[ "$candidate" != */* ]]; then
      candidate="$(command -v "$candidate" 2>/dev/null || true)"
    fi
    if [[ -x "$candidate" ]] && "$candidate" -I -c \
        'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "::error::PPU FlagTree requires a Python 3.12 interpreter" >&2
  return 1
}

RUNTIME_PYTHON="$(select_runtime_python)"
# If the image has Torch, identify its prefix only to remove inherited paths.
# Reading distribution metadata does not load the package or its native libs.
VENDOR_TORCH_ROOT="$("$RUNTIME_PYTHON" -I - <<'PYCODE'
from importlib import metadata
from pathlib import Path

try:
    print(Path(metadata.distribution("torch").locate_file("torch")).resolve())
except metadata.PackageNotFoundError:
    pass
PYCODE
)"

# Device nodes: three classes on the runner -- /dev/alixpu (base),
# /dev/alixpu_ctl (control), /dev/alixpu_ppu0-15 (compute). The CI container
# mounts all of them via container_options --device=...; probe alixpu_ppu0 as
# the "devices are visible" canary.
if [[ ! -c /dev/alixpu_ppu0 ]]; then
  echo "::error::PPU device node /dev/alixpu_ppu0 is unavailable"
  exit 1
fi

# The ordinary PPU build uses the SDK's CUDA-compatible runtime.
export FLAGOS_SKIP_CUDA_ASSETS=1
export FLAGOS_DISABLE_CUDA_ASSETS=1

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-ppu-${CI_STAGE}}"
create_runtime_venv "$RUNTIME_PYTHON"
# The pod's proxied egress to files.pythonhosted.org has been observed at
# ~90 kB/s, well below what pip's 15s default read timeout tolerates for a
# single wheel (e.g. the ~30 MB cmake wheel stalled mid-download and tripped
# ReadTimeoutError). Raise it for every venv pip install below so a slow but
# still-progressing download is not killed early; this is independent of the
# index-url choice and applies regardless of which index ends up serving it.
export PIP_DEFAULT_TIMEOUT=120

# Selectively copy RECORD-owned wheels. Never expose /flagos or /opt/flagtree
# on the target import path, and never copy vendor Torch or editable finders.
"$RUNTIME_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
  --target-python "$VENV_PYTHON" \
  --torch-version "${CPU_TORCH_VERSION%+cpu}" \
  --flagtree-version "$FLAGTREE_VERSION" \
  --flaggems-version "$FLAGGEMS_VERSION" \
  --requirements-file "$REPO_ROOT/.github/requirements-thead.txt"

ensure_runtime_requirements "$REPO_ROOT/.github/requirements-thead.txt" "$PIP_INDEX_URL"

# Retain the existing integration-only compiler, operator and model packages.
if [[ "$CI_STAGE" == "integration" ]]; then
  for requirement in pytest 'transformers>=4.51,<5' sentencepiece tiktoken protobuf; do
    ensure_runtime_requirement "$requirement" "$PIP_INDEX_URL"
  done
  if runtime_flagtree_is_usable ppu; then
    echo "Using prepared FlagTree $FLAGTREE_VERSION"
  else
    "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
    install_flagtree
  fi
  install_flag_gems
fi

if ! runtime_cpu_torch_is_usable; then
  repair_runtime_cpu_torch
fi
CPU_TORCH_ROOT="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV_PYTHON" -I - <<'PY'
from pathlib import Path
import sys

import torch

torch_root = Path(torch.__file__).resolve().parent
assert torch_root.is_relative_to(Path(sys.prefix).resolve()), torch_root
print(torch_root)
PY
)"

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export FLAGOS_ACCELERATOR=ppu
export CUDA_HOME="$PPU_SDK/CUDA_SDK"
export CUDA_PATH="$CUDA_HOME"
export PPU_SDK="$PPU_SDK"
export FLAGOS_VENDOR_TORCH_LIB=""
export FLAGOS_VENDOR_TORCH_VERSION=""
export FLAGOS_SDK_VERSION="${PPU_VERSION:-}"
FLAGOS_SDK_VERSION="${FLAGOS_SDK_VERSION#v}"
if [[ "$FLAGOS_SDK_VERSION" != 2.1.0 ]]; then
  echo "::error::Expected runtime PPU_VERSION=v2.1.0, got ${PPU_VERSION:-unset}" >&2
  exit 1
fi
# FLAGOS_WHEEL_LOCAL is not set here: the wheel's local segment is the vendor
# name, taken from the wheel_local column of cmake/flagos_platforms.json
# (currently "ppu2.0.0"). The manifest records the actual SDK separately;
# changing the shared release naming policy is outside this image migration.
# Setting an override here would make the
# artifact name depend on the build script instead of the platform table.
# PPU runtime ships the Python FlagGems wheel without liboperators.so /
# FlagGemsConfig.cmake, so the C++ kFlagOs dispatch (FLAGOS_BUILD_FLAGGEMS_CPP) must be off.
# setup.py's `ppu` branch already forces it OFF; exporting it here keeps the
# environment consistent for the pre-build assertions and any manual cmake run.
# The Python-path kernels (FLAGOS_BUILD_FLAGGEMS) stay at the default ON -- they
# compile without importing flag_gems, and the FlagGems runtime test step needs
# them.
export FLAGOS_BUILD_FLAGGEMS_CPP=0

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
CLEAN_CPATH="$(strip_vendor_paths "${CPATH:-}")"
export CPATH="$CUDA_HOME/include${CLEAN_CPATH:+:$CLEAN_CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$PPU_SDK/lib:$PPU_SDK/lib64${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$PPU_SDK/lib:$PPU_SDK/lib64:$CPU_TORCH_ROOT/lib${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so
  # before the common workflow invokes python -m build.
  build_flagos_inplace
fi

if command -v ppu-smi >/dev/null 2>&1; then
  ppu-smi
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "::error::Neither ppu-smi nor nvidia-smi is available in the PPU runtime" >&2
  exit 1
fi

if [[ -e torch_fl/lib_thead/libtorch_cuda.so ]]; then
  echo "::error::The ordinary PPU build must not bundle libtorch_cuda.so" >&2
  exit 1
fi

# Prove the installed FlagGems stack is importable and wired to PPU, so a broken
# install fails here instead of turning into a wall of identical failures in the
# FlagGems runtime step. Import torch_fl first to initialize the PPU runtime,
# then check the published wheels provisioned for integration tests.
#
# The three assertions are the ones that actually went wrong while bringing this
# up: the installed `triton` must be FlagTree's (its registry exposes the `ppu`
# backend, the vendor build exposes ['amd', 'nvidia']), flag_gems must have picked
# the `thead` vendor from PPU_SDK rather than falling back to nvidia, and it must
# resolve to the venv rather than the image's editable-install path.
if [[ "$CI_STAGE" == "integration" ]]; then
  python - <<'PY'
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

lib = Path(importlib.util.find_spec("torch").submodule_search_locations[0]) / "lib"
before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
          for path in lib.iterdir() if path.is_file()}
import torch_fl  # noqa: F401  -- register the PPU backend before FlagGems
import flag_gems
import triton
import triton.backends

venv = sys.prefix
assert str(Path(triton.__file__).resolve()).startswith(venv), triton.__file__
assert "ppu" in triton.backends.backends, list(triton.backends.backends)
assert str(Path(flag_gems.__file__).resolve()).startswith(venv), flag_gems.__file__

vendor = flag_gems.runtime.backend.device_finder.DeviceDetector()._get_vendor_from_env()
assert vendor == "thead", f"flag_gems selected vendor {vendor!r}; PPU_SDK={os.environ.get('PPU_SDK')!r}"

after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
         for path in lib.iterdir() if path.is_file()}
assert after == before, "PPU import changed the installed torch/lib"

import torch_fl._env as _env

kernels = _env.build_kernels()
conf = torch_fl.backend_config_path()
assert "boxing" not in kernels, kernels
assert conf.endswith("backends_thead.conf"), conf
assert "flaggems" in kernels, kernels
print(f"triton (FlagTree): {triton.__version__} -> {triton.__file__}")
print(f"triton backends: {sorted(triton.backends.backends)}")
print(f"flag_gems: {flag_gems.__version__} -> {flag_gems.__file__}")
print(f"flag_gems vendor: {vendor}")
PY
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then

  : "${FLAGOS_BACKEND_CONFIG:=}"
  export_ci_env PATH VIRTUAL_ENV BASH_ENV TORCH_DEVICE_BACKEND_AUTOLOAD FLAGOS_SDK_VERSION FLAGOS_VENDOR_TORCH_VERSION PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR FLAGOS_BACKEND_CONFIG CUDA_HOME CUDA_PATH PPU_SDK FLAGOS_VENDOR_TORCH_LIB FLAGOS_SKIP_CUDA_ASSETS FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_BUILD_FLAGGEMS_CPP CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH
fi
