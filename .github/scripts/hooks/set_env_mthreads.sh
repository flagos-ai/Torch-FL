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
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"

# --- MUSA toolkit ------------------------------------------------------------
export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
if [[ ! -d "$MUSA_HOME" ]]; then
  echo "::error::MUSA toolkit not found at MUSA_HOME=$MUSA_HOME" >&2
  exit 1
fi
echo "MUSA_HOME=$MUSA_HOME"

# FLAGOS_BUILD_VENDOR selects the native mudnn build. Default ON to match setup.py and
# the documented native-only contract, and fail loudly when the image lacks the
# C++ assets rather than silently shipping a wheel that looks native but reaches
# cpu_fallback for every compute op.
export FLAGOS_BUILD_VENDOR="${FLAGOS_BUILD_VENDOR:-1}"
if [[ "$FLAGOS_BUILD_VENDOR" != "0" ]]; then
  missing=()
  for asset in \
    "include/mudnncxx/mudnn.h" \
    "include/murand.h" \
    "lib/libmudnn.so" \
    "lib/libmurand.so"; do
    [[ -e "$MUSA_HOME/$asset" ]] || missing+=("$MUSA_HOME/$asset")
  done
  if (( ${#missing[@]} > 0 )); then
    echo "::error::FLAGOS_BUILD_VENDOR=$FLAGOS_BUILD_VENDOR requires the mudnn/murand C++ assets, but this image is missing:" >&2
    printf '::error::  %s\n' "${missing[@]}" >&2
    echo "::error::Install the mudnn and murand components of the MUSA toolkit." >&2
    echo "::error::Building with FLAGOS_BUILD_VENDOR=0 is not a substitute: the MUSA path also excludes the CUDA backend and the generated CUDA boxing kernels, so the result is a fallback-only wheel, not a native MUSA wheel." >&2
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
# Chip selection is FLAGOS_ACCELERATOR's job alone: FLAGOS_BUILD_VENDOR covers this
# platform's native kernels and FLAGOS_BUILD_BOXING=OFF (set in setup.py) drops the
# generated CUDA boxing kernels. No per-chip switches.
export FLAGOS_ACCELERATOR=musa
# FlagGems Python ops are provisioned and validated on MUSA via the MThreads
# Triton build (flagtree). The C++ kernels (FLAGOS_BUILD_FLAGGEMS_CPP=1) require
# FLAGOS_BUILD_FLAGGEMS_CPP=ON at build time and are not yet available.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1
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
# Clear inherited Python paths before probing or provisioning any interpreter.
# Backend autoload is disabled only for probes, not for the integration tests.
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
# The common set_env entrypoint selects and exports the pip download cache.

# The official runtime keeps vendor Torch in /flagos and FlagTree in
# /opt/flagtree. Reuse individual distributions, never either import path.
prepare_musa_environment() {
  local vendor_python="${TORCH_FL_VENDOR_PYTHON:-$(command -v python3.10 || true)}"
  export BASH_ENV=/dev/null
  export PYTHONPATH=""
  export PYTHONNOUSERSITE=1
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  if [[ ! -x "$vendor_python" ]] || ! "$vendor_python" -I -c \
      'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
    echo "::error::The MUSA runtime requires Python 3.10 for its FlagTree wheel" >&2
    return 1
  fi
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-musa-${CI_STAGE}}"
  create_runtime_venv "$vendor_python" torch_musa
  "$vendor_python" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --target-python "$VENV_PYTHON" \
    --torch-version "${CPU_TORCH_VERSION%+cpu}" \
    --flagtree-version "${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_mthreads}" \
    --flaggems-version "${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}" \
    --requirements-file "$REPO_ROOT/.github/requirements-mthreads.txt"
  export VIRTUAL_ENV="$VENV_ROOT"
  export PATH="$VENV_ROOT/bin:$PATH"
  # A Python CMake distribution is required by pyproject.toml even when the
  # image supplies /usr/bin/cmake. Bootstrap pip before resolving CPU Torch.
  local requirement
  ensure_runtime_requirements "$REPO_ROOT/.github/requirements-mthreads.txt" "$PIP_INDEX_URL_ARG"
  if ! runtime_cpu_torch_is_usable 2>/dev/null; then
    CPU_TORCH_VERSION="${CPU_TORCH_VERSION%+cpu}+cpu"
    install_cpu_torch
  fi
  validate_runtime_build_tools
  if [[ "$CI_STAGE" == integration ]]; then
    for requirement in 'transformers>=4.51,<5' safetensors; do
      ensure_runtime_requirement "$requirement" "$PIP_INDEX_URL_ARG"
    done
    if [[ "${TORCH_FL_INSTALL_QWEN_DEPS:-0}" == 1 ]]; then
      for requirement in sentencepiece tiktoken protobuf; do
        ensure_runtime_requirement "$requirement" "$PIP_INDEX_URL_ARG"
      done
    fi
  fi
  export FLAGOS_SDK_VERSION=5.2.0
}

prepare_musa_environment

# --no-deps on both binary packages so pip cannot replace the pinned CPU torch
# 2.10 with something a transitive requirement prefers.
#
# flagtree is the Triton build carrying the "mthreads" backend. 3.6 is not a
# preference but a requirement: current FlagGems uses tl.map_elementwise and
# triton.knobs, which flagtree 0.5.x (Triton 3.1) does not have -- that pair
# fails at import, so the two pins move together.
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_mthreads}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
if runtime_flagtree_is_usable mthreads; then
  echo "Using runtime-seeded FlagTree $FLAGTREE_VERSION"
else
  "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
  install_flagtree
fi

# flagtree may bring torch_musa as a dependency or in its wheel. Uninstall it
# again to ensure isolation.
"$VENV_PYTHON" -m pip uninstall -y torch_musa 2>/dev/null || true

# Install the MThreads-indexed wheel instead of cloning FlagGems from GitHub.
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-mthreads/simple}"
install_flag_gems

# Published FlagCX still requires the vendor Torch extension excluded from
# this CPU-Torch venv. Keep it deferred until a compatible wheel is available.
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

# --- Verify the isolation held ----------------------------------------------
TORCH_DEVICE_BACKEND_AUTOLOAD=0 CPU_TORCH_VERSION="$CPU_TORCH_VERSION" "$VENV_PYTHON" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

import torch

torch_path = Path(torch.__file__).resolve()
expected = os.environ["CPU_TORCH_VERSION"].removesuffix("+cpu") + "+cpu"
assert torch.__version__ == expected, torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
# The 2.9.1 vendor ABI must not be reachable from this interpreter.
assert importlib.util.find_spec("torch_musa") is None, "torch_musa leaked into the isolated venv"

print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"MUSA_HOME: {os.environ['MUSA_HOME']}")
print(f"FLAGOS_BUILD_VENDOR: {os.environ['FLAGOS_BUILD_VENDOR']}")
PY

# --- Verify the FlagGems stack imports --------------------------------------
# Integration only: torch_fl._C does not exist until the wheel is built, and the
# import order below needs it. Check for the extension module before attempting
# import -- CI calls set_env_mthreads.sh before building the wheel, so torch_fl is
# not yet importable at that stage.
#
# torch_fl must be imported before flag_gems. FlagGems 5.x selects its MThreads
# backend by reading torch.musa, which stock PyTorch does not have -- torch_fl
# installs that surface as a shim (_install_musa_flaggems_compat). Importing
# flag_gems first raises AttributeError: module 'torch' has no attribute 'musa'.
#
# That shim is itself conditional on the active conf routing at least one op to
# FlagGems, so this check also fails if backends_mthreads.conf has no flaggems route
# -- which is exactly the state this environment is being provisioned to leave.
if "$VENV_PYTHON" -c "import torch_fl._C" 2>/dev/null; then
  "$VENV_PYTHON" - <<'PY'
import torch_fl  # noqa: F401  -- must precede flag_gems; installs the torch.musa shim

import triton
import flag_gems

assert "mthreads" in triton.backends.backends, sorted(triton.backends.backends)
print(f"Triton: {triton.__version__} (backends: {sorted(triton.backends.backends)})")
print(f"FlagGems: {flag_gems.__version__} (vendor: {flag_gems.vendor_name})")
PY
fi

if command -v mthreads-gmi >/dev/null 2>&1; then
  mthreads-gmi
fi

# Integration keeps the existing operator/contract baseline, including BERT.
# Qwen3 tokenizer dependencies are opt-in. FlagTree supplies MThreads Triton.

# --- Export to later workflow steps ------------------------------------------
if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  export_ci_env BASH_ENV FLAGOS_SDK_VERSION
  export_ci_env PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR MUSA_HOME FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS FLAGOS_DISABLE_CUDA_ASSETS MTHREADS_VISIBLE_DEVICES CPATH LIBRARY_PATH LD_LIBRARY_PATH
  if [[ -n "${PIP_CACHE_DIR:-}" ]]; then
    export_ci_env PIP_CACHE_DIR
  fi
  if [[ -n "${TORCH_DEVICE_BACKEND_AUTOLOAD:-}" ]]; then
    printf 'TORCH_DEVICE_BACKEND_AUTOLOAD=%s\n' "$TORCH_DEVICE_BACKEND_AUTOLOAD" >> "$GITHUB_ENV"
  fi
fi

cd "$REPO_ROOT"
# build_ext produces the libtorch_fl.so that package_data stages into the wheel,
# and build and test share one job, so it must run in both stages. No device is
# required here; the card was checked above for the integration stage.
build_flagos_inplace
