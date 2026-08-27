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

# Moore Threads MUSA environment for the common build/integration workflow.
#
# MUSA runs native mudnn operator kernels, not CUDA boxing: the toolkit ships no
# CUDA runtime and there is no vendor dispatch key to box into. mudnn links
# against musart only and pulls in no torch symbols, so this backend builds
# against a stock CPU PyTorch.
#
# The vendor torch_musa package present in the base image is deliberately never
# imported, linked, or copied into the isolated environment: its libtorch is a
# 2.9.1 build whose C++ object layout differs from 2.10 (sizeof(c10::MessageLogger)
# 408 -> 400), so mixing the two ABIs corrupts memory at runtime rather than
# failing to link. See docs/vendors/musa/installation.md.
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
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-https://pypi.org/simple}"

# --- MUSA toolkit ------------------------------------------------------------
export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
if [[ ! -d "$MUSA_HOME" ]]; then
  echo "::error::MUSA toolkit not found at MUSA_HOME=$MUSA_HOME" >&2
  exit 1
fi
echo "MUSA_HOME=$MUSA_HOME"

# MUSA_KERNEL selects the native mudnn build. Default ON to match setup.py and
# the documented native-only contract, and fail loudly when the image lacks the
# C++ assets rather than silently shipping a wheel that looks native but reaches
# cpu_fallback for every compute op.
export MUSA_KERNEL="${MUSA_KERNEL:-1}"
if [[ "$MUSA_KERNEL" != "0" ]]; then
  missing=()
  for asset in \
    "include/mudnncxx/mudnn.h" \
    "include/murand.h" \
    "lib/libmudnn.so" \
    "lib/libmurand.so"; do
    [[ -e "$MUSA_HOME/$asset" ]] || missing+=("$MUSA_HOME/$asset")
  done
  if (( ${#missing[@]} > 0 )); then
    echo "::error::MUSA_KERNEL=$MUSA_KERNEL requires the mudnn/murand C++ assets, but this image is missing:" >&2
    printf '::error::  %s\n' "${missing[@]}" >&2
    echo "::error::Install the mudnn and murand components of the MUSA toolkit." >&2
    echo "::error::Building with MUSA_KERNEL=0 is not a substitute: the MUSA path also excludes the CUDA backend and the generated CUDA boxing kernels, so the result is a fallback-only wheel, not a native MUSA wheel." >&2
    exit 1
  fi
  echo "mudnn C++ assets: present"
fi

# --- Device node -------------------------------------------------------------
# Only the integration stage needs a card; a build can run on any MUSA host.
if [[ "$CI_STAGE" == "integration" ]] && ! ls /dev/mtgpu* >/dev/null 2>&1; then
  echo "::error::No Moore Threads device node (/dev/mtgpu*) is visible"
  exit 1
fi

# --- Backend selection -------------------------------------------------------
export ACCELERATOR=musa
export CUDA_KERNEL=0
export METAX_KERNEL=0
export ASCEND_KERNEL=0
export GCU_KERNEL=0
# This image ships no MThreads Triton/FlagGems stack (the vendor
# flagtree-0.5.0+mthreads3.1 wheel is absent). Keep the native mudnn path
# deterministic; the hybrid routing documented in
# docs/vendors/musa/installation.md can be enabled once that backend is
# provisioned and validated. A generic PyPI triton is not a substitute for the
# MThreads build and is never installed in its place.
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON=0
export FLAGOS_USE_FLAGGEMS=0
export FLAGOS_USE_FLAGGEMS_CPP=0
# MUSA bundles no libtorch_cuda.so and the toolkit exports no cuda symbols, so
# the CUDA asset preload has nothing to open.
export FLAGOS_DISABLE_CUDA_ASSETS=1
export MTHREADS_VISIBLE_DEVICES="${MTHREADS_VISIBLE_DEVICES:-all}"
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

export PATH="$MUSA_HOME/bin:$PATH"
export CPATH="$MUSA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$MUSA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$MUSA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Isolated Python ---------------------------------------------------------
# The vendor image Python has torch_musa (and its 2.9.1 libtorch) installed, so
# the build runs in a venv holding nothing but stock CPU torch.
BOOTSTRAP_PYTHON="${TORCH_FL_BOOTSTRAP_PYTHON:-}"
if [[ -z "$BOOTSTRAP_PYTHON" || ! -x "$BOOTSTRAP_PYTHON" ]]; then
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      BOOTSTRAP_PYTHON="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$BOOTSTRAP_PYTHON" || ! -x "$BOOTSTRAP_PYTHON" ]]; then
  echo "::error::Unable to find Python to bootstrap the isolated environment"
  exit 1
fi

PREBUILT_VENV="${TORCH_FL_PREBUILT_MUSA_VENV:-/opt/torch-fl-musa-venv}"
USING_PREBUILT=0
if [[ -z "${TORCH_FL_VENV_ROOT:-}" && -x "$PREBUILT_VENV/bin/python" ]]; then
  VENV_ROOT="$PREBUILT_VENV"
  USING_PREBUILT=1
  echo "Using prebuilt MUSA venv: $VENV_ROOT"
else
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-musa-${CI_STAGE}}"
  "$BOOTSTRAP_PYTHON" -m venv --clear "$VENV_ROOT" || true
fi

VENV_PYTHON="$VENV_ROOT/bin/python"
venv_is_usable() {
  [[ -x "$VENV_PYTHON" ]] || return 1
  "$VENV_PYTHON" -m pip --version >/dev/null 2>&1
}

if ! venv_is_usable; then
  # The vendor base image may not ship the matching python*-venv package. Keep
  # that dependency in the chip-specific setup path so the common workflow stays
  # image agnostic; a derived CI image should bake it in.
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    python_mm="$($BOOTSTRAP_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "python${python_mm}-venv"
    "$BOOTSTRAP_PYTHON" -m venv --clear "$VENV_ROOT"
  fi
fi

if ! venv_is_usable; then
  echo "::error::Isolated Python was not created at $VENV_ROOT; install the matching python*-venv package in the CI image" >&2
  exit 1
fi

if (( USING_PREBUILT == 0 )); then
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL_ARG" \
    --upgrade pip setuptools wheel cmake ninja
  "$VENV_PYTHON" -m pip install \
    --index-url "$CPU_TORCH_INDEX_URL" \
    "torch==$CPU_TORCH_VERSION"
fi

if [[ "$CI_STAGE" == "integration" ]]; then
  # Test dependencies. transformers pulls numpy 2.x, which breaks the stock
  # +cpu torch C extensions at import, so numpy stays on 1.x.
  #
  # sentencepiece + tiktoken + protobuf: the Qwen3 tests load the tokenizer via
  # AutoTokenizer, and the mounted model dir has no tokenizer.json, so
  # transformers converts the slow tokenizer to a fast one and that conversion
  # needs one of these.
  #
  # transformers is pinned to [4.51, 5): 4.51 is where Qwen3 model_type support
  # landed (older releases raise "Unrecognized model" on AutoConfig), and 5.x has
  # an unresolved TokenizersBackend regression that breaks the Qwen3 slow-to-fast
  # conversion even with sentencepiece installed. An unpinned install picks
  # whichever of those two failure modes is latest on PyPI.
  #
  # Installed even into a prebuilt venv: a venv baked without them turns the
  # inference and training groups into an environment failure that looks like a
  # platform failure. pip is a no-op when they are already present.
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL_ARG" \
    pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# --- Verify the isolation held ----------------------------------------------
CI_STAGE="$CI_STAGE" CPU_TORCH_VERSION="$CPU_TORCH_VERSION" "$VENV_PYTHON" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

import torch

torch_path = Path(torch.__file__).resolve()
assert torch.__version__.split("+", 1)[0] == os.environ["CPU_TORCH_VERSION"], torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
# The 2.9.1 vendor ABI must not be reachable from this interpreter.
assert importlib.util.find_spec("torch_musa") is None, "torch_musa leaked into the isolated venv"

print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"MUSA_HOME: {os.environ['MUSA_HOME']}")
print(f"MUSA_KERNEL: {os.environ['MUSA_KERNEL']}")
PY

if command -v mthreads-gmi >/dev/null 2>&1; then
  mthreads-gmi
fi

# Report the integration stack once, here, rather than letting a group fail on a
# missing import and read as a platform defect. Triton is absent on this line:
# the image ships no MThreads flagtree build and stock PyPI triton targets NVIDIA,
# so compile-tests will fail when torch.compile is invoked, and that failure is
# the environment-gap record the platform owners act on.
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" .github/scripts/check_integration_deps.py \
    --require pytest transformers safetensors
fi

# --- Export to later workflow steps ------------------------------------------
if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR MUSA_HOME \
    MUSA_KERNEL CUDA_KERNEL METAX_KERNEL ASCEND_KERNEL GCU_KERNEL \
    FLAGGEMS_KERNEL FLAGGEMS_PYTHON FLAGOS_USE_FLAGGEMS \
    FLAGOS_USE_FLAGGEMS_CPP FLAGOS_DISABLE_CUDA_ASSETS \
    MTHREADS_VISIBLE_DEVICES CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

cd "$REPO_ROOT"
# build_ext produces the libtorch_fl.so that package_data stages into the wheel,
# and build and test share one job, so it must run in both stages. No device is
# required here; the card was checked above for the integration stage.
python setup.py build_ext --inplace
