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

# Enflame GCU environment for the common build/integration workflow.
#
# GCU uses the TopsRider runtime directly. The vendor torch-gcu package that
# may be present in the base image is intentionally not copied into the
# isolated environment: this backend builds against stock CPU PyTorch and
# links TopsRider's libtopsrt/libtopsaten libraries instead.
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
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-2.10.0}"

discover_tops_root() {
  local candidate found
  local -a candidates=(
    "${TOPS_HOME:-}"
    "/opt/tops"
    "/opt/topsrider"
    "/opt/tops-rider"
    "/usr/local/tops"
  )

  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate/lib/libtopsrt.so" &&
          -f "$candidate/include/gcu/topsaten/topsaten.h" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  found="$(find /opt /usr/local -maxdepth 5 -type f -name libtopsrt.so \
    -print -quit 2>/dev/null || true)"
  if [[ -n "$found" ]]; then
    candidate="$(dirname "$(dirname "$found")")"
    if [[ -f "$candidate/include/gcu/topsaten/topsaten.h" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  fi
  return 1
}

if ! TOPS_HOME="$(discover_tops_root)"; then
  echo "::error::TopsRider SDK not found; expected lib/libtopsrt.so and include/gcu/topsaten/topsaten.h" >&2
  exit 1
fi
export TOPS_HOME
echo "TOPS_HOME=$TOPS_HOME"

TOPSATEN_LIB="${TOPSATEN_LIB:-}"
if [[ -z "$TOPSATEN_LIB" ]]; then
  for candidate in \
    "$TOPS_HOME/lib/libtopsaten.so" \
    /usr/lib/libtopsaten.so \
    /usr/lib64/libtopsaten.so; do
    if [[ -f "$candidate" ]]; then
      TOPSATEN_LIB="$candidate"
      break
    fi
  done
fi
if [[ -z "$TOPSATEN_LIB" || ! -f "$TOPSATEN_LIB" ]]; then
  echo "::error::libtopsaten.so was not found under $TOPS_HOME/lib, /usr/lib, or /usr/lib64" >&2
  exit 1
fi
export TOPSATEN_LIB
TOPSATEN_LIB_DIR="$(dirname "$(readlink -f "$TOPSATEN_LIB")")"
echo "TOPSATEN_LIB=$TOPSATEN_LIB"

if [[ "$CI_STAGE" == "integration" && ! -c /dev/gcu0 ]]; then
  echo "::error::Enflame device node /dev/gcu0 is unavailable"
  exit 1
fi

export ACCELERATOR=gcu
export GCU_KERNEL=1
export CUDA_KERNEL=0
export METAX_KERNEL=0
export ASCEND_KERNEL=0
export MUSA_KERNEL=0
# The base image currently has no triton_gcu/FlagGems package. Keep the native
# Topsaten path deterministic; FlagGems can be enabled later when the vendor
# Triton stack is explicitly provisioned and validated.
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON="${FLAGOS_GCU_FLAGGEMS_PYTHON:-0}"
export FLAGOS_USE_FLAGGEMS=0
export FLAGOS_USE_FLAGGEMS_CPP=0
export FLAGOS_DISABLE_CUDA_ASSETS=1
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

export CPATH="$TOPS_HOME/include/gcu:$TOPS_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$TOPS_HOME/lib:$TOPSATEN_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TOPS_HOME/lib:$TOPSATEN_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VENDOR_PYTHON="${TORCH_FL_VENDOR_PYTHON:-}"
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      VENDOR_PYTHON="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  echo "::error::Unable to find Python 3.12 to bootstrap the isolated environment"
  exit 1
fi

PREBUILT_VENV="${TORCH_FL_PREBUILT_GCU_VENV:-/opt/torch-fl-gcu-venv}"
if [[ -z "${TORCH_FL_VENV_ROOT:-}" && -x "$PREBUILT_VENV/bin/python" ]]; then
  VENV_ROOT="$PREBUILT_VENV"
  echo "Using prebuilt GCU venv: $VENV_ROOT"
else
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-gcu-${CI_STAGE}}"
  "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT" || true
fi

VENV_PYTHON="$VENV_ROOT/bin/python"
venv_is_usable() {
  [[ -x "$VENV_PYTHON" ]] || return 1
  "$VENV_PYTHON" -m pip --version >/dev/null 2>&1
}

if ! venv_is_usable; then
  # The current TopsRider base image does not ship python3.12-venv.  Keep the
  # dependency in the chip-specific setup path so the common workflow remains
  # image/workflow agnostic; a derived CI image should still bake this package
  # in to avoid the apt fallback on every job.
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    python_mm="$($VENDOR_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "python${python_mm}-venv"
    "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"
  fi
fi

if ! venv_is_usable; then
  echo "::error::Isolated Python was not created at $VENV_ROOT; install python3.12-venv in the CI image" >&2
  exit 1
fi

if [[ "$VENV_ROOT" != "$PREBUILT_VENV" ]]; then
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel cmake ninja
  "$VENV_PYTHON" -m pip install \
    --index-url "$CPU_TORCH_INDEX_URL" \
    "torch==$CPU_TORCH_VERSION"
fi

# Test dependencies, installed whether or not the venv was prebuilt: a prebuilt
# venv missing them makes the inference and training groups fail as an
# environment problem that reads as a topsaten problem.
#
# transformers is pinned to [4.51, 5) for Qwen3 model_type support, below the
# 5.x TokenizersBackend regression; sentencepiece/tiktoken/protobuf drive the
# slow-to-fast tokenizer conversion for a model dir with no tokenizer.json, and
# numpy stays on 1.x so the +cpu torch C extensions keep importing.
#
# triton is not installed here. On GCU it comes from the vendor triton_gcu /
# flagtree build in the image; stock PyPI triton targets NVIDIA and would shadow
# it (see _vendor_supplies_triton in setup.py for the same rule on dcu/ascend).
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" -m pip install \
    pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import importlib.util
import os
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/usr/local/lib/python3.12/dist-packages" not in str(torch_path), torch_path
assert importlib.util.find_spec("torch_gcu") is None, "torch_gcu leaked into isolated venv"
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"Topsaten library: {os.environ['TOPSATEN_LIB']}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR GCU_KERNEL \
    CUDA_KERNEL METAX_KERNEL ASCEND_KERNEL MUSA_KERNEL FLAGGEMS_KERNEL \
    FLAGGEMS_PYTHON FLAGOS_USE_FLAGGEMS FLAGOS_USE_FLAGGEMS_CPP \
    FLAGOS_DISABLE_CUDA_ASSETS TOPS_HOME TOPSATEN_LIB CPATH LIBRARY_PATH \
    LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

# Expose the vendor Triton (triton_gcu / flagtree) to the isolated venv, matching
# how set_env_metax.sh exposes triton-metax: the active torch is the CPU wheel and
# ships no Triton, and the vendor build is installed in the image rather than
# resolved by pip.
#
# A missing vendor Triton is a warning, not a setup failure: compile-tests then
# fails on its own and the remaining nine groups still produce evidence, whereas
# exiting here would take the whole run down over a torch.compile gap.
if [[ "$CI_STAGE" == "integration" ]]; then
  VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  if [[ ! -e "$VENV_SITE/triton" ]]; then
    for candidate in /usr/local/lib/python3.*/dist-packages \
                     /usr/local/lib/python3.*/site-packages \
                     /usr/lib/python3.*/dist-packages \
                     /usr/lib/python3.*/site-packages; do
      if [[ -d "$candidate/triton" ]]; then
        ln -s "$candidate/triton" "$VENV_SITE/triton"
        for metadata in "$candidate"/triton*-*.dist-info; do
          [[ -e "$metadata" ]] || continue
          [[ -e "$VENV_SITE/$(basename "$metadata")" ]] || ln -s "$metadata" "$VENV_SITE/"
        done
        echo "Vendor Triton linked from $candidate/triton"
        break
      fi
    done
  fi

  # Verify triton is available through vendor stack (triton_gcu/flagtree).
  # Stock PyPI triton targets NVIDIA and is not a substitute. triton is not in
  # the --require list: a missing vendor Triton is a warning, not a setup
  # failure (the compile-tests group will fail on its own, documenting the gap).
  "$VENV_PYTHON" .github/scripts/check_integration_deps.py \
    --require pytest transformers safetensors
fi

cd "$REPO_ROOT"
# build_ext runs in both stages: it produces the libtorch_fl.so that package_data
# stages into the wheel, and build and test now share one container job, so
# gating it to the build stage would leave the integration stage building a wheel
# with no native library. The device is checked by the device-availability
# preflight against the installed wheel.
python setup.py build_ext --inplace
