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

# Prepare the CI environment for the CUDA platform against the native
# torch-2.9 ABI image built by .github/workflows/integration-test-cuda.yml
# (docker/cuda/Dockerfile).
#
# The image ships a full GPU torch==2.9.0+cu130 plus FlagGems C++ extensions
# and FlagCX compiled against that same torch. torch_fl therefore runs in
# "active-torch-supplies-libtorch_cuda" mode: we set FLAGOS_SKIP_CUDA_ASSETS=1
# (setup.py does not bundle libtorch_cuda into the wheel) and
# FLAGOS_DISABLE_CUDA_ASSETS=1 (torch_fl/__init__.py does not ctypes-preload
# it before import torch). The full GPU torch loads its own libtorch_cuda via
# import torch, so the CUDAHooks caching constraint that forced the old
# CPU-torch + external-libtorch_cuda hack no longer applies.
#
# This retires the fix_2.9_watch hack: no venv, no libtorch_cuda extraction,
# no LD_LIBRARY_PATH stripping, no TORCH_DEVICE_BACKEND_AUTOLOAD=0, no
# FLAGGEMS_KERNEL=0 (the C++ FlagGems path is the default ON for CUDA again).

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The image's /opt/venv is the active interpreter (PATH is set by the image).
PYTHON_BIN="$(command -v python)"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "::error::python interpreter not found on PATH"
  exit 1
fi

# FlagGems ships its CMake config + liboperators.so in the installed package.
# Locate it the same way setup.py's _find_flaggems_dir() does so the torch_fl
# build links the in-image 2.9-ABI C++ FlagGems path (FLAGGEMS_KERNEL=ON).
FLAGGEMS_DIR="$("$PYTHON_BIN" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("flag_gems")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("flag_gems package not found in the image")
root = Path(next(iter(spec.submodule_search_locations))).resolve()
candidate = root / "lib" / "cmake" / "FlagGems"
if not (candidate / "FlagGemsConfig.cmake").is_file():
    raise SystemExit(f"FlagGemsConfig.cmake not found under {candidate}")
print(candidate)
PY
)"
echo "FlagGems CMake dir: $FLAGGEMS_DIR"

# torch_fl "active-torch-supplies-libtorch_cuda" mode (Mode 2). See header.
export FLAGOS_SKIP_CUDA_ASSETS=1
export FLAGOS_DISABLE_CUDA_ASSETS=1

export ACCELERATOR=cuda
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export FLAGGEMS_DIR="$FLAGGEMS_DIR"
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"

# CMake needs torch's config dir for target resolution and the FlagGems dir
# for the C++ operators. torch's own lib is already on the loader path via the
# image, so no LD_LIBRARY_PATH surgery is required.
TORCH_CMAKE_PATH="$("$PYTHON_BIN" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH:$FLAGGEMS_DIR${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"

if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so before python -m build runs.
  python setup.py build_ext --inplace
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - <<'PY'
import sys
import torch

assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.9.0", torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
print(f"Python: {sys.executable}")
print(f"GPU PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
PY

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    ACCELERATOR CUDA_HOME CUDA_PATH FLAGGEMS_DIR FLAGCX_PATH \
    CMAKE_PREFIX_PATH CPATH \
    FLAGOS_SKIP_CUDA_ASSETS FLAGOS_DISABLE_CUDA_ASSETS; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
