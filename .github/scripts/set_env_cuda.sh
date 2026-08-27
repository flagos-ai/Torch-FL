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

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel cmake build
"$VENV_PYTHON" -m pip install \
  --index-url "$CPU_TORCH_INDEX_URL" \
  "torch==$CPU_TORCH_VERSION"
# transformers is pinned to [4.51, 5): 4.51 is where Qwen3 model_type support
# landed, and 5.x carries a TokenizersBackend regression that breaks the Qwen3
# slow-to-fast tokenizer conversion. sentencepiece/tiktoken/protobuf drive that
# conversion when the mounted model dir has no tokenizer.json. numpy stays on 1.x
# so the +cpu torch C extensions keep importing. triton is excluded: the vendor
# Triton is carried in from VENDOR_SITE below.
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" -m pip install \
    pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf
fi

# Keep the vendor FlagGems/FlagCX Python packages available without copying the
# vendor torch package. They are pure-Python/extension packages used by the
# CUDA runtime path; the active torch package remains the CPU wheel below.
VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for package in flag_gems triton triton_kernels flagcx sqlalchemy; do
  if [[ -d "$VENDOR_SITE/$package" ]]; then
    cp -a "$VENDOR_SITE/$package" "$VENV_SITE/"
  fi
  for metadata in "$VENDOR_SITE"/"$package"-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    cp -a "$metadata" "$VENV_SITE/"
  done
done

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

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake:$VENDOR_FLAGGEMS_DIR${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/lib64:$CUDA_ASSETS_DIR${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_ASSETS_DIR${VENDOR_NVIDIA_LIBS:+:$VENDOR_NVIDIA_LIBS}:$CPU_TORCH_ROOT/lib:$VENDOR_FLAGGEMS_LIB:$CUDA_HOME/lib64${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so and the bundled CUDA assets
  # before the common workflow invokes python -m build.
  python setup.py build_ext --inplace
fi

# Device probes only run during integration tests. Build-only runs do not need
# the device, and manifests have a more complete device-availability check.
if [[ "$CI_STAGE" == "integration" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "::error::nvidia-smi is unavailable"
    exit 1
  fi
  nvidia-smi
fi

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
    FLAGOS_CUDA_ASSETS_DIR FLAGGEMS_DIR FLAGCX_PATH \
    CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

# Report the integration stack once, here, rather than letting a group fail on a
# missing import and read as a platform defect. Triton is carried in from the
# vendor site-packages by this script.
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" .github/scripts/check_integration_deps.py \
    --require pytest transformers safetensors triton
fi
