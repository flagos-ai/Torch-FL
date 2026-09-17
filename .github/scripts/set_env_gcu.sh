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
#
# The FlagGems Python path is provisioned here the same way set_env_musa.sh
# provisions it: the generated configs/backends_gcu.conf routes most overloads
# to a FlagGems Triton kernel and the rest to the native topsaten kernel, so the
# isolated venv needs a Triton build carrying the "enflame" backend (flagtree)
# plus FlagGems itself. See docs/vendors/gcu/flaggems-setup.md.
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
export VENDOR_KERNEL=1
# FLAGGEMS_CPP=0: the FlagGems C++ kernels (liboperators.so) need
# FLAGGEMS_CPP=ON, which is not built here. FLAGGEMS_KERNEL=1 is what makes
# the flaggems routes in backends_gcu.conf resolvable at all: setup.py turns
# FLAGGEMS_KERNEL on for ACCELERATOR=gcu, but this environment variable is
# applied afterwards and would otherwise switch it back off, producing a wheel
# whose conf routes ops to dispatcher slots that were never compiled in.
export FLAGGEMS_CPP=0
export FLAGGEMS_KERNEL=1
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

# --- Enflame Triton (flagtree) + FlagGems ------------------------------------
# Installed for both stages, not just integration: build and integration share
# one platform job, and restricting this to CI_STAGE=integration would leave the
# build job's venv without flag_gems -- the wheel then fails as soon as a
# FlagGems route dispatches. Same reasoning as set_env_musa.sh.
#
# --no-deps on both source packages so pip cannot replace the pinned CPU torch
# 2.10 with something a transitive requirement prefers.
#
# Retries are deliberate: the flagtree wheel is large and the shared mirror can
# close the response early (IncompleteRead) even though the package is there.
# Retrying just the failed package beats restarting all of setup.
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

# An NVIDIA triton wheel must not be present: it carries no "enflame" backend,
# so flag_gems' backend discovery fails and every FlagGems route raises at
# import. flagtree installs its own `triton` package under the same import name
# without registering a `triton` distribution, so remove any real one first --
# repeatedly, because a partially uninstalled triton leaves a dist record that
# keeps the stale files in place. This is the uninstall step from the vendor
# install instructions, run against the isolated interpreter.
while "$VENV_PYTHON" -m pip show triton >/dev/null 2>&1; do
  "$VENV_PYTHON" -m pip uninstall -y triton || break
done
"$VENV_PYTHON" -m pip uninstall -y triton_gcu torch_gcu 2>/dev/null || true

# flagtree is the Triton build carrying the "enflame" backend. 3.6 is not a
# preference but a requirement: current FlagGems uses tl.map_elementwise and
# triton.knobs, which a Triton 3.1 build does not have -- that pair fails at
# import, so the flagtree and FlagGems pins move together.
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.6.1+enflame3.6}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
pip_retry --no-deps --index-url "$FLAGTREE_INDEX_URL" "flagtree===$FLAGTREE_VERSION"

# flagtree may bring torch_gcu as a dependency or bundle it in the wheel.
# Uninstall it again to keep the isolated venv free of the vendor ABI.
"$VENV_PYTHON" -m pip uninstall -y torch_gcu 2>/dev/null || true

# FlagGems from the flagos-ai fork, tracking master by policy: every CI run
# measures the current master, not a pinned snapshot. Override with
# TORCH_FL_FLAGGEMS_REVISION to pin a commit for a reproducible run. The routing
# measured against the old 3c6f7537d pin is recorded in
# docs/vendors/gcu/flaggems-test-results.md.
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
# reason above. numpy is deliberately not listed: the CPU torch wheel does not
# pull it in, and nothing on this path imports it -- FlagGems' GCU route and the
# integration tests are both numpy-free. pip still warns that flag_gems declares
# a numpy requirement it cannot satisfy; that warning is expected here and is not
# a failure, because flag_gems is installed with --no-deps.
pip_retry --index-url "$PIP_INDEX_URL_ARG" packaging
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'PyYAML==6.0.1'
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'sqlalchemy==2.0.48'

# --- Verify the Triton stack imports -----------------------------------------
# flagtree must be the only provider of the `triton` package: a stock wheel left
# behind by the base image carries no "enflame" backend and FlagGems' backend
# discovery would fail against it.
#
# flag_gems itself is not imported here. FlagGems 5.x resolves its vendor through
# the torch_fl device surface, which does not exist until the wheel is installed,
# so a bare `import flag_gems` raises "No device were detected on your machine".
# The integration "Check isolated GCU environment" group does that import once
# torch_fl is importable, and is the check that matters.
"$VENV_PYTHON" - <<'PY'
import importlib.util

import triton

assert "enflame" in triton.backends.backends, sorted(triton.backends.backends)
assert importlib.util.find_spec("flag_gems") is not None, "flag_gems is not installed"
print(f"Triton: {triton.__version__} (backends: {sorted(triton.backends.backends)})")
print("flag_gems: installed")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR VENDOR_KERNEL \
    FLAGGEMS_CPP FLAGGEMS_KERNEL FLAGOS_USE_FLAGGEMS_CPP \
    FLAGOS_DISABLE_CUDA_ASSETS TOPS_HOME TOPSATEN_LIB CPATH LIBRARY_PATH \
    LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" ]]; then
  # Prebuild so package_data contains the generated libtorch_fl.so before the
  # common wheel workflow stages the final artifact.
  python setup.py build_ext --inplace

  python - <<'PY'
import torch_fl

assert torch_fl.flagos.is_available(), "flagos device is unavailable"
print(f"flagos devices: {torch_fl.flagos.device_count()}")
PY
fi
