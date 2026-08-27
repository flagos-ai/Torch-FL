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
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

# Locate the DTK install root. The image may ship DTK under /opt/dtk,
# /opt/dtk-26.04, or a versioned directory; honor an explicit DTK_ROOT or
# ROCM_PATH, then probe common roots, then fall back to a filesystem search.
discover_dtk_root() {
  local candidate
  local -a candidates=(
    "${DTK_ROOT:-}"
    "${ROCM_PATH:-}"
    "/opt/dtk"
    "/opt/dtk-26.04"
    "/opt/dtk26.04"
    "/opt/dtk-26.04.4"
    "/opt/dtk-25.04.4"
  )
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate/env.sh" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  local found
  found="$(find /opt /usr/local -maxdepth 3 -type f -name env.sh \
    \( -path '*dtk*' -o -path '*rocm*' \) -print -quit 2>/dev/null | head -n1)"
  if [[ -n "$found" ]]; then
    printf '%s' "$(dirname "$found")"
    return 0
  fi
  return 1
}

if ! DTK_ROOT="$(discover_dtk_root)"; then
  echo "::error::DTK env.sh not found under /opt or /usr/local; set DTK_ROOT explicitly"
  exit 1
fi
echo "DTK_ROOT discovered: $DTK_ROOT"
set +u
# shellcheck disable=SC1091
source "$DTK_ROOT/env.sh"
set -u
export DTK_ROOT
export ROCM_PATH="${ROCM_PATH:-$DTK_ROOT}"

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
    echo "::error::Vendor Python cannot import torch: $candidate (was DTK env.sh sourced?)" >&2
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
for key in ("version", "base_version", "root", "site", "python"):
    print(info.get(key) or "")
PY
)
VENDOR_TORCH_VERSION="${VENDOR_FIELDS[0]}"
VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[1]}"
VENDOR_TORCH_ROOT="${VENDOR_FIELDS[2]}"
VENDOR_SITE="${VENDOR_FIELDS[3]}"
VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[4]}"
VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

# DCU is a boxing build: the DTK torch wheel is a hipified build whose HIP
# kernels are registered under the CUDA dispatch key, so torch_fl reaches them
# via PrivateUse1->CUDA boxing kernels without shipping its own kernels.
# libtorch_hip.so is the DTK analogue of CUDA's libtorch_cuda.so; the bundle
# script stages it (plus the core .so) into torch_fl/lib_dcu for a
# self-contained wheel.
if [[ ! -f "$VENDOR_TORCH_LIB/libtorch_hip.so" ]]; then
  echo "::error::$VENDOR_TORCH_LIB has no libtorch_hip.so, not a DTK torch/lib" >&2
  exit 1
fi
export FLAGOS_DCU_TORCH_LIB="$VENDOR_TORCH_LIB"

# CPU torch base version must match the vendor's for ABI compatibility. The
# DTK wheel version (e.g. 2.10.0+das on the current CI image) may differ across
# DTK releases and from the CUDA/MetaX images, so derive the target from the
# vendor wheel instead of hard-coding it.
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$VENDOR_TORCH_BASE_VERSION}"

# FlagGems C++ package is optional for DCU (FLAGGEMS_KERNEL=OFF in setup.py);
# the flaggems runtime path uses the DTK triton hcu backend + Python flag_gems.
# Probe it for CMAKE_PREFIX_PATH but never fail when absent.
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
VENDOR_FLAGGEMS_LIB=""
if [[ -n "$VENDOR_FLAGGEMS_DIR" ]]; then
  VENDOR_FLAGGEMS_LIB="$(cd "$VENDOR_FLAGGEMS_DIR/../.." && pwd)"
  if ! compgen -G "$VENDOR_FLAGGEMS_LIB/liboperators.so*" >/dev/null; then
    echo "::notice::FlagGems C++ liboperators.so not found under $VENDOR_FLAGGEMS_LIB (optional for DCU)"
    VENDOR_FLAGGEMS_LIB=""
  fi
else
  echo "::notice::FlagGems C++ package not found in vendor image (optional for DCU)"
fi

echo "Vendor Python: $VENDOR_PYTHON ($VENDOR_PYTHON_VERSION)"
echo "Vendor PyTorch: $VENDOR_TORCH_VERSION (base $VENDOR_TORCH_BASE_VERSION)"
echo "Vendor torch root: $VENDOR_TORCH_ROOT"
echo "CPU torch target: $CPU_TORCH_VERSION+cpu"

# Isolated venv with the matching CPU-only torch wheel. Unlike CUDA, DCU does
# not stage libtorch_cuda.so into .libtorch_cuda_assets; the DTK forked libtorch
# .so are bundled into torch_fl/lib_dcu by bundle_dcu_libtorch.sh below.
PREBUILT_VENV="${TORCH_FL_PREBUILT_DCU_VENV:-/opt/torch-fl-dcu-venv}"
if [[ -z "${TORCH_FL_VENV_ROOT:-}" && -x "$PREBUILT_VENV/bin/python" ]]; then
  VENV_ROOT="$PREBUILT_VENV"
  echo "Using prebuilt DCU venv: $VENV_ROOT"
else
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-dcu-${CI_STAGE}}"
  if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
    echo "::warning::Vendor Python cannot create a venv; trying uv"
    if ! command -v uv >/dev/null 2>&1; then
      "$VENDOR_PYTHON" -m pip install --upgrade uv
    fi
    uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
  fi
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

if [[ "$VENV_ROOT" != "$PREBUILT_VENV" ]]; then
  # patchelf is required by setup.py (rewrites _C.so RPATH to $ORIGIN/lib and
  # $ORIGIN/lib_dcu after copy) and by bundle_dcu_libtorch.sh (sets RPATH on the
  # bundled DTK .so). The pinned CI image ships it at /usr/bin/patchelf; the pip
  # install here is a redundant fallback in case the image lacks it.
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel cmake build patchelf
  "$VENV_PYTHON" -m pip install \
    --index-url "$CPU_TORCH_INDEX_URL" \
    "torch==$CPU_TORCH_VERSION"
fi

# Test dependencies, installed whether or not the venv came prebuilt: a prebuilt
# venv missing them makes the inference and training groups fail as an environment
# problem that reads as a DTK problem.
#
# transformers is pinned to [4.51, 5): 4.51 is where Qwen3 model_type support
# landed, and 5.x carries a TokenizersBackend regression that breaks the Qwen3
# slow-to-fast tokenizer conversion. sentencepiece/tiktoken/protobuf drive that
# conversion when the mounted model dir has no tokenizer.json. numpy stays on 1.x:
# the CPU torch wheel is built against the NumPy 1.x ABI and transformers
# otherwise pulls 2.x, which breaks torch's C extensions at import.
#
# triton is excluded: on dcu it comes from DTK (see _vendor_supplies_triton in
# setup.py), and the vendor package is carried into the venv below.
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" -m pip install \
    pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf
fi

# Carry the vendor FlagGems/FlagCX Python packages into the venv so the DTK
# triton (hcu backend) and flag_gems runtime path work under the CPU torch.
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

CPU_TORCH_ROOT="$(CPU_TORCH_VERSION="$CPU_TORCH_VERSION" "$VENV_PYTHON" - <<'PY'
import os
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == os.environ["CPU_TORCH_VERSION"], torch.__version__
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
export ACCELERATOR=dcu
# DTK's Triton (hcu backend) is copied into the venv without its dist-info
# (the vendor image ships none), so Triton's entry-point backend discovery
# finds nothing and the hcu backend never loads. Force in-tree loading so the
# backend under triton/backends is used -- required for the FlagGems runtime
# path (integration [3/5]). See README "Enabling FlagGems on DCU".
export TRITON_BACKENDS_IN_TREE=1
export FLAGGEMS_DIR="$VENDOR_FLAGGEMS_DIR"
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"

# libhydmi.so (Hygon DMI device management) ships in the host hyhal driver,
# which dcu.yml bind-mounts into the container at the standard hyhal root.
# DTK's env.sh does not add it to the search path, so flagos device init's
# dlopen(libhydmi.so) fails with "cannot open shared object file". Probe the
# standard roots (honoring an explicit HYHAL_ROOT) and prepend them so the
# loader finds libhydmi.so plus its sibling driver libs (libhydrm, libamd_smi).
HYHAL_LD_PATH=""
for hyhal_root in "${HYHAL_ROOT:-}" /usr/local/hyhal /opt/hyhal; do
  [[ -z "$hyhal_root" ]] && continue
  [[ -d "$hyhal_root/lib" ]] || continue
  HYHAL_LD_PATH="${HYHAL_LD_PATH:+$HYHAL_LD_PATH:}$hyhal_root/lib"
done
# Recreate the host's /opt/hyhal -> /usr/local/hyhal symlink inside the
# container (the flagos image lacks it). dlopen(libhydmi.so) goes through
# LD_LIBRARY_PATH above, but sibling hyhal libs or host code may resolve .so
# by absolute /opt/hyhal/... paths; mirror the host layout so those resolve.
if [[ -d /usr/local/hyhal && ! -e /opt/hyhal ]]; then
  ln -sf /usr/local/hyhal /opt/hyhal 2>/dev/null || \
    echo "::warning::failed to create /opt/hyhal -> /usr/local/hyhal symlink"
fi
if [[ -n "$HYHAL_LD_PATH" ]]; then
  echo "hyhal lib path: $HYHAL_LD_PATH"
else
  echo "::warning::No hyhal/lib found; flagos device init may fail to dlopen libhydmi.so"
fi

export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake${VENDOR_FLAGGEMS_DIR:+:$VENDOR_FLAGGEMS_DIR}${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export LIBRARY_PATH="${HYHAL_LD_PATH:+$HYHAL_LD_PATH:}$CPU_TORCH_ROOT/lib${VENDOR_FLAGGEMS_LIB:+:$VENDOR_FLAGGEMS_LIB}${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${HYHAL_LD_PATH:+$HYHAL_LD_PATH:}$CPU_TORCH_ROOT/lib${VENDOR_FLAGGEMS_LIB:+:$VENDOR_FLAGGEMS_LIB}${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so before python -m build.
  python setup.py build_ext --inplace
fi

# Bundle DTK's forked libtorch core + HIP .so into torch_fl/lib_dcu for a
# self-contained wheel (replaces CUDA's .libtorch_cuda_assets staging step).
DTK_ROOT="$DTK_ROOT" FLAGOS_DCU_TORCH_LIB="$FLAGOS_DCU_TORCH_LIB" \
  bash scripts/bundle_dcu_libtorch.sh

# Device probes only run during integration tests. Build-only runs do not need
# the device, and manifests have a more complete device-availability check.
if [[ "$CI_STAGE" == "integration" ]]; then
  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "::error::rocm-smi is unavailable"
    exit 1
  fi
  rocm-smi
fi

python - <<'PY'
import os
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
assert Path("torch_fl/lib/libtorch_fl.so").is_file()
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"DCU torch lib: {os.environ.get('FLAGOS_DCU_TORCH_LIB', '?')}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR DTK_ROOT ROCM_PATH \
    FLAGOS_DCU_TORCH_LIB FLAGGEMS_DIR FLAGCX_PATH TRITON_BACKENDS_IN_TREE \
    CMAKE_PREFIX_PATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

# Report the integration stack once, here, rather than letting a group fail on a
# missing import and read as a platform defect. Triton comes from DTK and is
# carried into the venv by this script.
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" .github/scripts/check_integration_deps.py \
    --require pytest transformers safetensors triton
fi
