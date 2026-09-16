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
# FlagGems Python ops are provisioned and validated on MUSA via the MThreads
# Triton build (flagtree). The C++ kernels (FLAGGEMS_KERNEL=1) require
# FLAGGEMS_KERNEL=ON at build time and are not yet available.
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON=1
export FLAGOS_USE_FLAGGEMS=1
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
  # Ensure system site-packages are not inherited (torch_musa from base image)
  if [[ -f "$VENV_ROOT/pyvenv.cfg" ]]; then
    # Remove any existing include-system-site-packages line and add our own
    grep -v '^include-system-site-packages' "$VENV_ROOT/pyvenv.cfg" > "$VENV_ROOT/pyvenv.cfg.tmp"
    echo "include-system-site-packages = false" >> "$VENV_ROOT/pyvenv.cfg.tmp"
    mv "$VENV_ROOT/pyvenv.cfg.tmp" "$VENV_ROOT/pyvenv.cfg"
  fi
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
    # Ensure system site-packages are not inherited (torch_musa from base image)
    if [[ -f "$VENV_ROOT/pyvenv.cfg" ]]; then
      echo "::debug::pyvenv.cfg before modification:"
      cat "$VENV_ROOT/pyvenv.cfg"
      # Remove any existing include-system-site-packages line and add our own
      grep -v '^include-system-site-packages' "$VENV_ROOT/pyvenv.cfg" > "$VENV_ROOT/pyvenv.cfg.tmp"
      echo "include-system-site-packages = false" >> "$VENV_ROOT/pyvenv.cfg.tmp"
      mv "$VENV_ROOT/pyvenv.cfg.tmp" "$VENV_ROOT/pyvenv.cfg"
      echo "::debug::pyvenv.cfg after modification:"
      cat "$VENV_ROOT/pyvenv.cfg"
    else
      echo "::warning::pyvenv.cfg not found at $VENV_ROOT/pyvenv.cfg after venv creation"
    fi
  fi
fi

if ! venv_is_usable; then
  echo "::error::Isolated Python was not created at $VENV_ROOT; install the matching python*-venv package in the CI image" >&2
  exit 1
fi

if (( USING_PREBUILT == 0 )); then
  # build (pypa/build) is the PEP 517 frontend the common "Build wheel" step
  # invokes via `python -m build --wheel --no-isolation`. MetaX gets it from the
  # prebuilt /opt/venv; this fresh venv must ship it itself. A derived CI image
  # that bakes a prebuilt musa venv must bake `build` into it too.
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL_ARG" \
    --upgrade pip setuptools wheel cmake ninja build
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

# Persist venv activation to subsequent workflow steps via GITHUB_ENV
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "VIRTUAL_ENV=$VENV_ROOT"
    echo "PATH=$VENV_ROOT/bin:$PATH"
    echo "PYTHONNOUSERSITE=1"
    echo "PYTHONPATH="
  } >> "$GITHUB_ENV"
fi

# Ensure torch_musa from the base image is not importable in the venv.
# The venv should be isolated by default, but explicitly uninstall if present.
if "$VENV_PYTHON" -c "import importlib.util; exit(0 if importlib.util.find_spec('torch_musa') is None else 1)" 2>/dev/null; then
  : # torch_musa is not visible, isolation is working
else
  echo "::warning::torch_musa is visible in the venv; attempting to uninstall"
  echo "::debug::sys.path from venv:"
  "$VENV_PYTHON" -c "import sys; print('\n'.join(sys.path))"
  echo "::debug::pyvenv.cfg content:"
  cat "$VENV_ROOT/pyvenv.cfg" || echo "pyvenv.cfg not found"
  "$VENV_PYTHON" -m pip uninstall -y torch_musa 2>/dev/null || true
  # Verify torch_musa is now gone
  if "$VENV_PYTHON" -c "import importlib.util; exit(0 if importlib.util.find_spec('torch_musa') is None else 1)" 2>/dev/null; then
    echo "::notice::torch_musa successfully removed from venv"
  else
    echo "::error::torch_musa still visible after uninstall attempt"
    exit 1
  fi
fi

# --- MThreads Triton (flagtree) + FlagGems -----------------------------------
# Installed for both stages, not just integration: build and integration share
# one platform job, and restricting this to CI_STAGE=integration would leave the
# build job's venv without flag_gems -- the wheel then fails as soon as a
# FlagGems route dispatches. Same reasoning as set_env_ascend.sh.
#
# --no-deps on both source packages so pip cannot replace the pinned CPU torch
# 2.10 with something a transitive requirement prefers.
#
# Retries are deliberate: the flagtree wheel is 180 MB and the shared mirror can
# close a large-wheel response early (IncompleteRead) even though the package is
# there. Retrying just the failed package beats restarting all of setup.
pip_retry() {
  local attempt=1
  while true; do
    if "$VENV_PYTHON" -m pip install --retries 10 --timeout 300 --no-cache-dir "$@"; then
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

# flagtree is the Triton build carrying the "mthreads" backend. 3.6 is not a
# preference but a requirement: current FlagGems uses tl.map_elementwise and
# triton.knobs, which flagtree 0.5.x (Triton 3.1) does not have -- that pair
# fails at import, so the two pins move together.
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.6.2a3+mthreads3.6}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
pip_retry --no-deps --index-url "$FLAGTREE_INDEX_URL" "flagtree===$FLAGTREE_VERSION"

# flagtree may bring torch_musa as a dependency or in its wheel. Uninstall it
# again to ensure isolation.
"$VENV_PYTHON" -m pip uninstall -y torch_musa 2>/dev/null || true

# FlagGems from the flagos-ai fork, tracking master by policy: every CI run
# measures the current master, not a pinned snapshot. Override with
# TORCH_FL_FLAGGEMS_REVISION to pin a commit for a reproducible run.
#
# TEMPORARY PIN -- revert the default to `master` once upstream fixes
# flagos-ai/FlagGems: d312aa02 (2026-09-16) added
# ("argsort.stable", argsort_stable) to the module-level _FULL_CONFIG in
# flag_gems/__init__.py, but argsort_stable is only defined by the kunlunxin
# backend package, so `import flag_gems` raises
# NameError: name 'argsort_stable' is not defined on every other vendor.
# 437ba393 is the last good master (d312aa02's parent).
FLAGGEMS_REVISION="${TORCH_FL_FLAGGEMS_REVISION:-437ba39387ddc681dc884259ef9dbf0c1802bccc}"
FLAGGEMS_REPO="${TORCH_FL_FLAGGEMS_REPO:-https://github.com/flagos-ai/FlagGems.git}"
pip_retry --no-deps "git+${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"

# FlagGems' own runtime deps, installed one at a time for the IncompleteRead
# reason above. numpy stays <2 for the same reason as the test deps: 2.x breaks
# the stock +cpu torch C extensions at import.
pip_retry --index-url "$PIP_INDEX_URL_ARG" packaging
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'PyYAML==6.0.1'
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'sqlalchemy==2.0.48'
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'numpy<2'

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

# --- Verify the FlagGems stack imports --------------------------------------
# Integration only: torch_fl._C does not exist until the wheel is built, and the
# import order below needs it. Check for the extension module before attempting
# import -- CI calls set_env_musa.sh before building the wheel, so torch_fl is
# not yet importable at that stage.
#
# torch_fl must be imported before flag_gems. FlagGems 5.x selects its MThreads
# backend by reading torch.musa, which stock PyTorch does not have -- torch_fl
# installs that surface as a shim (_install_musa_flaggems_compat). Importing
# flag_gems first raises AttributeError: module 'torch' has no attribute 'musa'.
#
# That shim is itself conditional on the active conf routing at least one op to
# FlagGems, so this check also fails if backends_musa.conf has no flaggems route
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

# Integration deps (pytest, transformers, numpy<2, safetensors, sentencepiece,
# tiktoken, protobuf) are pip-installed above. Triton is deliberately absent on
# this line: the image ships no MThreads flagtree build and stock PyPI triton
# targets NVIDIA, so torch.compile will fail when invoked -- that failure is the
# environment-gap record the platform owners act on (compile-tests is withheld in
# the manifest until the image bakes the vendor triton stack).

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
