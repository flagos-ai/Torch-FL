#!/usr/bin/env bash
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

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-2.10.0}"
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
# The NVIDIA FlagTree 3.6 wheel is the source-free artifact documented by
# FlagTree's User-Manual. It replaces the stock Triton module in the venv.
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.6.2a2}"
# FlagGems currently uses master as its default branch; the repository has no
# main branch. Keep this overrideable so a tested revision can be pinned by CI.
FLAGGEMS_REPOSITORY="${TORCH_FL_FLAGGEMS_REPOSITORY:-https://github.com/flagos-ai/FlagGems.git}"
FLAGGEMS_REVISION="${TORCH_FL_FLAGGEMS_REVISION:-master}"

select_vendor_python() {
  local candidate="${TORCH_FL_VENDOR_PYTHON:-}"
  if [[ -n "$candidate" && "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$(command -v python 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" || ! -x "$candidate" ]]; then
    echo "::error::Unable to find the vendor Python interpreter" >&2
    exit 1
  fi
  if ! "$candidate" -c "import torch" >/dev/null 2>&1; then
    echo "::error::Vendor Python cannot import torch: $candidate" >&2
    exit 1
  fi
  printf '%s' "$candidate"
}

VENDOR_PYTHON="$(select_vendor_python)"
VENDOR_INFO="$("$VENDOR_PYTHON" - <<'PY'
import json
from pathlib import Path
import site
import sys

import torch

print(json.dumps({
    "version": torch.__version__,
    "base_version": torch.__version__.split("+", 1)[0],
    "cuda": torch.version.cuda,
    "root": str(Path(torch.__file__).resolve().parent),
    "site": site.getsitepackages()[0],
    "python": sys.version.split()[0],
}, sort_keys=True))
PY
)"

readarray -t VENDOR_FIELDS < <(
  VENDOR_INFO="$VENDOR_INFO" "$VENDOR_PYTHON" - <<'PY'
import json
import os

info = json.loads(os.environ["VENDOR_INFO"])
for key in ("version", "base_version", "cuda", "root", "site", "python"):
    print(info.get(key) or "")
PY
)
VENDOR_TORCH_VERSION="${VENDOR_FIELDS[0]}"
VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[1]}"
VENDOR_CUDA_VERSION="${VENDOR_FIELDS[2]}"
VENDOR_TORCH_ROOT="${VENDOR_FIELDS[3]}"
VENDOR_SITE="${VENDOR_FIELDS[4]}"
VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[5]}"
VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

VENDOR_NVIDIA_LIBS=""
for nvidia_lib in "$VENDOR_SITE"/nvidia/*/lib; do
  [[ -d "$nvidia_lib" ]] || continue
  VENDOR_NVIDIA_LIBS="${VENDOR_NVIDIA_LIBS:+$VENDOR_NVIDIA_LIBS:}$nvidia_lib"
done

if [[ "$VENDOR_TORCH_BASE_VERSION" != "$CPU_TORCH_VERSION" ]]; then
  echo "::error::Vendor torch is $VENDOR_TORCH_VERSION; expected $CPU_TORCH_VERSION"
  exit 1
fi
if [[ "$VENDOR_CUDA_VERSION" != "13.0" ]]; then
  echo "::error::Vendor torch CUDA runtime is $VENDOR_CUDA_VERSION; expected 13.0"
  exit 1
fi

echo "Vendor Python: $VENDOR_PYTHON ($VENDOR_PYTHON_VERSION)"
echo "Vendor PyTorch: $VENDOR_TORCH_VERSION"
echo "Vendor torch root: $VENDOR_TORCH_ROOT"

VENDOR_FLAGGEMS_DIR="$("$VENDOR_PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("flag_gems")
if spec is None or not spec.submodule_search_locations:
    print("")
else:
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    candidate = root / "lib" / "cmake" / "FlagGems"
    print(candidate if (candidate / "FlagGemsConfig.cmake").is_file() else "")
PY
)"
if [[ -z "$VENDOR_FLAGGEMS_DIR" ]]; then
  echo "::error::FlagGems C++ package was not found in the vendor image"
  exit 1
fi
VENDOR_FLAGGEMS_LIB="$(cd "$VENDOR_FLAGGEMS_DIR/../.." && pwd)"
if ! compgen -G "$VENDOR_FLAGGEMS_LIB/liboperators.so*" >/dev/null; then
  echo "::error::FlagGems liboperators.so was not found under $VENDOR_FLAGGEMS_LIB"
  exit 1
fi

# Copy only accelerator-side PyTorch libraries. libc10.so, libtorch.so,
# libtorch_cpu.so and libtorch_python.so deliberately come from the upstream
# CPU wheel installed below. setup.py copies these assets into torch_fl/lib,
# and torch_fl preloads them before importing torch.
CUDA_ASSETS_DIR="$REPO_ROOT/.libtorch_cuda_assets"
rm -rf "$CUDA_ASSETS_DIR"
mkdir -p "$CUDA_ASSETS_DIR"
shopt -s nullglob
for pattern in \
  'libc10_cuda.so*' \
  'libtorch_cuda.so*' \
  'libtorch_cuda_linalg.so*' \
  'libtorch_nvshmem.so*' \
  'libcaffe2_nvrtc.so*'; do
  for source in "$VENDOR_TORCH_LIB"/$pattern; do
    cp -a "$source" "$CUDA_ASSETS_DIR/"
  done
done
shopt -u nullglob

# The CUDA dispatcher library is the only mandatory accelerator asset. Some
# vendor Torch layouts (including cu130 images) do not ship a standalone
# libc10_cuda.so; torch_fl treats that library as optional and loads it when
# present. Keep the check layout-agnostic instead of requiring a fixed set of
# files on every CUDA image.
if [[ ! -e "$CUDA_ASSETS_DIR/libtorch_cuda.so" ]]; then
  echo "::error::Required CUDA asset was not found: $VENDOR_TORCH_LIB/libtorch_cuda.so"
  exit 1
fi
if [[ ! -e "$CUDA_ASSETS_DIR/libc10_cuda.so" ]]; then
  echo "::notice::Optional CUDA asset not present: $VENDOR_TORCH_LIB/libc10_cuda.so"
fi
echo "CUDA assets staged: $(find "$CUDA_ASSETS_DIR" -maxdepth 1 -type f -name '*.so*' | wc -l)"

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-cuda-${CI_STAGE}}"
if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
  echo "::warning::Vendor Python cannot create a venv; trying uv"
  if ! command -v uv >/dev/null 2>&1; then
    "$VENDOR_PYTHON" -m pip install --upgrade uv
  fi
  uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip "setuptools>=64,<77" "setuptools-scm>=8,<10" "wheel==0.46.2" cmake build
"$VENV_PYTHON" -m pip install \
  --index-url "$CPU_TORCH_INDEX_URL" \
  "torch==$CPU_TORCH_VERSION"

# FlagTree 3.6 requires Python 3.12. If vendor Python is older, install python3.12
# from deadsnakes PPA and create a separate venv for FlagTree, then symlink packages.
FLAGTREE_PYTHON="$VENV_PYTHON"
if [[ "$VENDOR_PYTHON_VERSION" != 3.12.* ]]; then
  echo "::warning::Vendor Python is $VENDOR_PYTHON_VERSION; FlagTree 3.6 requires Python 3.12"
  if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Installing Python 3.12 from deadsnakes PPA..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
  fi
  FLAGTREE_VENV="$VENV_ROOT-py312"
  rm -rf "$FLAGTREE_VENV"
  python3.12 -m venv "$FLAGTREE_VENV"
  FLAGTREE_PYTHON="$FLAGTREE_VENV/bin/python"
  echo "Created Python 3.12 venv for FlagTree: $FLAGTREE_VENV"

  # Install basic dependencies in Python 3.12 venv
  "$FLAGTREE_PYTHON" -m pip install --upgrade pip "setuptools>=64,<77" "setuptools-scm>=8,<10" "wheel==0.46.2"
  "$FLAGTREE_PYTHON" -m pip install \
    --index-url "$CPU_TORCH_INDEX_URL" \
    "torch==$CPU_TORCH_VERSION"
fi

# Install FlagTree with the appropriate Python version
while "$FLAGTREE_PYTHON" -m pip show triton >/dev/null 2>&1; do
  "$FLAGTREE_PYTHON" -m pip uninstall -y triton
done
pip_retry() {
  local python_exe="$1"
  shift
  local attempt=1
  while true; do
    if "$python_exe" -m pip install --retries 10 --timeout 600 "$@"; then
      return 0
    fi
    if (( attempt >= 5 )); then
      echo "::error::pip install failed after $attempt attempts: $*"
      return 1
    fi
    echo "::warning::pip install attempt $attempt failed; retrying: $*"
    attempt=$((attempt + 1))
    sleep 10
  done
}
pip_retry "$FLAGTREE_PYTHON" --no-deps --index-url "$FLAGTREE_INDEX_URL" \
  "flagtree===${FLAGTREE_VERSION}"

# If using separate Python 3.12 venv, symlink FlagTree packages to main venv
if [[ "$FLAGTREE_PYTHON" != "$VENV_PYTHON" ]]; then
  FLAGTREE_SITE="$("$FLAGTREE_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "Symlinking FlagTree packages from $FLAGTREE_SITE to $VENV_SITE"
  for pkg in flagtree triton; do
    if [[ -d "$FLAGTREE_SITE/$pkg" ]]; then
      ln -sf "$FLAGTREE_SITE/$pkg" "$VENV_SITE/"
    fi
    for metadata in "$FLAGTREE_SITE"/"$pkg"-*.dist-info; do
      [[ -e "$metadata" ]] || continue
      ln -sf "$metadata" "$VENV_SITE/"
    done
  done
fi

# Keep only vendor packages that are not provided by FlagTree. In particular,
# do not copy vendor `triton` or `triton_kernels`: either would contaminate the
# source-free FlagTree installation with a potentially incompatible Triton.
VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for package in flagcx; do
  if [[ -d "$VENDOR_SITE/$package" ]]; then
    cp -a "$VENDOR_SITE/$package" "$VENV_SITE/"
  fi
  for metadata in "$VENDOR_SITE"/"$package"-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    cp -a "$metadata" "$VENV_SITE/"
  done
done

# Install the current FlagGems source after FlagTree. FlagGems has no `main`
# branch at present; `master` is its default branch and can be overridden with
# TORCH_FL_FLAGGEMS_REVISION for reproducible CI experiments. --no-deps keeps
# the CPU-only torch ABI intact; install its non-torch dependencies explicitly.
pip_retry "$VENV_PYTHON" packaging 'PyYAML==6.0.1' 'sqlalchemy==2.0.48' numpy
FLAGGEMS_SOURCE_ROOT="${RUNNER_TEMP:-/tmp}/flag-gems-${CI_STAGE}"
rm -rf "$FLAGGEMS_SOURCE_ROOT"
git clone --depth 1 --branch "$FLAGGEMS_REVISION" \
  "$FLAGGEMS_REPOSITORY" "$FLAGGEMS_SOURCE_ROOT"
FLAGGEMS_COMMIT="$(git -C "$FLAGGEMS_SOURCE_ROOT" rev-parse HEAD)"
echo "FlagGems source: ${FLAGGEMS_REPOSITORY}@${FLAGGEMS_REVISION} (${FLAGGEMS_COMMIT})"
pip_retry "$VENV_PYTHON" --no-deps --no-build-isolation "$FLAGGEMS_SOURCE_ROOT"
if [[ "$CI_STAGE" == "integration" ]]; then
  pip_retry "$VENV_PYTHON" pytest transformers
fi

CPU_TORCH_ROOT="$("$VENV_PYTHON" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
PY
)"

strip_vendor_paths() {
  local value="${1:-}"
  local entry
  local -a entries=()
  local -a kept=()
  IFS=: read -ra entries <<< "$value"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      "$VENDOR_TORCH_ROOT"|"$VENDOR_TORCH_ROOT"/*) ;;
      *) kept+=("$entry") ;;
    esac
  done
  local joined=""
  for entry in "${kept[@]}"; do
    joined="${joined:+$joined:}$entry"
  done
  printf '%s' "$joined"
}

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export ACCELERATOR=cuda
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export FLAGOS_CUDA_ASSETS_DIR="$CUDA_ASSETS_DIR"
export FLAGGEMS_DIR="$VENDOR_FLAGGEMS_DIR"
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"
export FLAGTREE_VERSION
export GEMS_VENDOR="${GEMS_VENDOR:-nvidia}"

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake:$VENDOR_FLAGGEMS_DIR${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/lib64:$CUDA_ASSETS_DIR${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_ASSETS_DIR${VENDOR_NVIDIA_LIBS:+:$VENDOR_NVIDIA_LIBS}:$CPU_TORCH_ROOT/lib:$VENDOR_FLAGGEMS_LIB:$CUDA_HOME/lib64${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
# Verify that the source-free wheel installed the expected Triton provider,
# rather than silently falling back to stock Triton.
python - <<'PY'
import importlib.metadata
import importlib.util
import os

import triton

flagtree_version = importlib.metadata.version("flagtree")
assert flagtree_version == os.environ["FLAGTREE_VERSION"], flagtree_version
assert importlib.util.find_spec("triton.flagtree_spec") is not None
print(f"FlagTree: {flagtree_version}")
print(f"Triton module: {triton.__file__}")
print(f"Triton version: {getattr(triton, '__version__', 'unknown')}")
PY

if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so and the bundled CUDA assets
  # before the common workflow invokes python -m build.
  python setup.py build_ext --inplace
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - <<'PY'
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
assert Path(".libtorch_cuda_assets/libtorch_cuda.so").is_file()
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"CUDA assets: {Path('.libtorch_cuda_assets').resolve()}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR CUDA_HOME CUDA_PATH \
    FLAGOS_CUDA_ASSETS_DIR FLAGGEMS_DIR FLAGCX_PATH FLAGTREE_VERSION GEMS_VENDOR \
    CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
