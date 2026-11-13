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
# The FlagGems Python path is provisioned here the same way set_env_mthreads.sh
# provisions it: the generated configs/backends_enflame.conf routes most overloads
# to a FlagGems Triton kernel and the rest to the native topsaten kernel, so the
# isolated venv needs a Triton build carrying the "enflame" backend (flagtree)
# plus FlagGems itself. See docs/vendors/gcu/flaggems-setup.md.
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"

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
    # TOPS ATen headers may live in /usr/include/gcu independently of the
    # runtime prefix. Existing CMake discovery validates those headers.
    if [[ -f "$candidate/lib/libtopsrt.so" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  found="$(find /opt /usr/local -maxdepth 5 -type f -name libtopsrt.so \
    -print -quit 2>/dev/null || true)"
  if [[ -n "$found" ]]; then
    candidate="$(dirname "$(dirname "$found")")"
    printf '%s' "$candidate"
    return 0
  fi
  return 1
}

if ! TOPS_HOME="$(discover_tops_root)"; then
  echo "::error::TopsRider SDK not found; expected lib/libtopsrt.so under the runtime prefix" >&2
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

export FLAGOS_ACCELERATOR=gcu
export FLAGOS_BUILD_VENDOR=1
# FLAGOS_BUILD_FLAGGEMS_CPP=0: the FlagGems C++ kernels (liboperators.so) need
# FLAGOS_BUILD_FLAGGEMS_CPP=ON, which is not built here. FLAGOS_BUILD_FLAGGEMS=1 is what makes
# the flaggems routes in backends_enflame.conf resolvable at all: setup.py turns
# FLAGOS_BUILD_FLAGGEMS on for FLAGOS_ACCELERATOR=gcu, but this environment variable is
# applied afterwards and would otherwise switch it back off, producing a wheel
# whose conf routes ops to dispatcher slots that were never compiled in.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1
export FLAGOS_DISABLE_CUDA_ASSETS=1
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

export CPATH="$TOPS_HOME/include/gcu:$TOPS_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$TOPS_HOME/lib:$TOPSATEN_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TOPS_HOME/lib:$TOPSATEN_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Official runtime packages are copied individually into a fresh venv, never
# exposed through the runtime's site-packages or compiler activation scripts.

gcu_record_runtime_sdk() {
  local version=""
  version="$(dpkg-query -W -f='${Version}' topsruntime)" || return 1
  version="${version#*:}"
  version="${version%%-*}"
  if [[ "$version" != 1.10.6 ]]; then
    echo "::error::Expected TOPS runtime 1.10.6, found $version" >&2
    return 1
  fi
  export FLAGOS_SDK_VERSION="$version"
}

prepare_gcu_environment() {
  gcu_record_runtime_sdk
  local vendor_python="${TORCH_FL_VENDOR_PYTHON:-$(command -v python3.12 || true)}"
  export BASH_ENV=/dev/null
  export PYTHONPATH=""
  export PYTHONNOUSERSITE=1
  if [[ ! -x "$vendor_python" ]] || ! "$vendor_python" -I -c \
      'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    echo "::error::The GCU runtime requires Python 3.12 for its FlagTree wheel" >&2
    return 1
  fi
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-gcu-${CI_STAGE}}"
  create_runtime_venv "$vendor_python" torch_gcu
  "$vendor_python" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --target-python "$VENV_PYTHON" \
    --torch-version "${CPU_TORCH_VERSION%+cpu}" \
    --flagtree-version "${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_enflame}" \
    --flaggems-version "${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}" \
    --requirements-file "$REPO_ROOT/.github/requirements-enflame.txt"
  export VIRTUAL_ENV="$VENV_ROOT"
  export PATH="$VENV_ROOT/bin:$PATH"
  # A Python CMake distribution is required by pyproject.toml even when the
  # image supplies /usr/bin/cmake. Bootstrap pip before resolving CPU Torch.
  ensure_runtime_requirements "$REPO_ROOT/.github/requirements-enflame.txt" "$PIP_INDEX_URL_ARG"
  validate_runtime_build_tools
  if ! runtime_cpu_torch_is_usable 2>/dev/null; then
    CPU_TORCH_VERSION="${CPU_TORCH_VERSION%+cpu}+cpu"
    install_cpu_torch
  fi
}

prepare_gcu_environment

export PYTHONNOUSERSITE=1
export PYTHONPATH=""

TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV_PYTHON" - <<'PY'
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
# FlagGems route dispatches. Same reasoning as set_env_mthreads.sh.
#
# --no-deps on both source packages so pip cannot replace the pinned CPU torch
# 2.10 with something a transitive requirement prefers.
# An NVIDIA triton wheel must not be present: it carries no "enflame" backend,
# so FlagGems backend discovery must resolve the Enflame FlagTree provider.
#
# flagtree is the Triton build carrying the "enflame" backend. 3.6 is not a
# preference but a requirement: current FlagGems uses tl.map_elementwise and
# triton.knobs, which a Triton 3.1 build does not have -- that pair fails at
# import, so the flagtree and FlagGems pins move together.
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_enflame}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
if runtime_flagtree_is_usable enflame; then
  echo "Using runtime-seeded FlagTree $FLAGTREE_VERSION"
else
  "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
  install_flagtree
fi

# flagtree may bring torch_gcu as a dependency or bundle it in the wheel.
# Uninstall it again to keep the isolated venv free of the vendor ABI.
"$VENV_PYTHON" -m pip uninstall -y torch_gcu 2>/dev/null || true

# The Hygon, Ascend and Enflame indexes publish the same 5.4.0 Python wheel.
# Select the Enflame index so this job never needs a GitHub source checkout.
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-enflame/simple}"
install_flag_gems

# Published FlagCX still requires the vendor Torch extension excluded from
# this CPU-Torch venv. Keep it deferred until a compatible wheel is available.
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

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
  export_ci_env BASH_ENV FLAGOS_SDK_VERSION
  export_ci_env PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS FLAGOS_DISABLE_CUDA_ASSETS TOPS_HOME TOPSATEN_LIB CPATH LIBRARY_PATH LD_LIBRARY_PATH
  if [[ -n "${FLAGCX_TORCH_BACKEND:-}" ]]; then
    printf 'FLAGCX_TORCH_BACKEND=%s\n' "$FLAGCX_TORCH_BACKEND" >> "$GITHUB_ENV"
  fi
  if [[ -n "${TORCH_DEVICE_BACKEND_AUTOLOAD:-}" ]]; then
    printf 'TORCH_DEVICE_BACKEND_AUTOLOAD=%s\n' "$TORCH_DEVICE_BACKEND_AUTOLOAD" >> "$GITHUB_ENV"
  fi
fi

# Print the device/driver inventory without pretending the container upgrades
# the host kernel driver. The existing hardware checks still determine success.
if command -v efsmi >/dev/null 2>&1; then
  efsmi
fi

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" ]]; then
  # Prebuild so package_data contains the generated libtorch_fl.so before the
  # common wheel workflow stages the final artifact.
  build_flagos_inplace

  python - <<'PY'
import torch_fl

assert torch_fl.flagos.is_available(), "flagos device is unavailable"
print(f"flagos devices: {torch_fl.flagos.device_count()}")
PY
fi
