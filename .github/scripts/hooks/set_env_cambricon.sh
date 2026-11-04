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

# Cambricon MLU (BANG) environment for the common build/integration workflow.
#
# Cambricon is a native PrivateUse1 backend built over the Neuware SDK: the
# runtime sources under csrc/runtime/accelerator/cambricon/ call libcnrt
# directly (cnrtGetDeviceCount, cnrtMalloc, cnrtQueue_t, cnrtNotifier_t), and
# the compute kernels will come from CNNL behind the native `cambricon` route
# key. There is no CUDA runtime and no boxing, so nothing here needs a CUDA
# toolkit.
#
# The vendor torch_mlu package is deliberately never installed: it hard-links
# its own libtorch and is ABI-bound to one vendor torch minor. The torch.mlu /
# torch_mlu surface FlagGems and Triton's mlu driver read is synthesized at
# import time by torch_fl/accelerator/cambricon/_cambricon_compat.py, so a stock
# CPU PyTorch wheel is enough to build and run the backend.
#
# FlagGems needs a Triton build carrying the "mlu" backend. On this platform that
# comes from the lane's `triton` wheel rather than from a `flagtree` package:
# flagtree is not installed here.
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"

# --- Neuware SDK -------------------------------------------------------------
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
if [[ ! -f "$NEUWARE_HOME/include/cnrt.h" ]]; then
  echo "::error::Neuware toolkit not found at NEUWARE_HOME=$NEUWARE_HOME (no include/cnrt.h)" >&2
  exit 1
fi
echo "NEUWARE_HOME=$NEUWARE_HOME"

# The toolkit keeps its libraries in lib64/ on current releases and lib/ on some
# older layouts. The same pair is resolved in CMakeLists.txt, with the same
# order, so setup.py and the configure step cannot disagree.
if [[ -f "$NEUWARE_HOME/lib64/libcnrt.so" ]]; then
  NEUWARE_LIB_DIR="$NEUWARE_HOME/lib64"
elif [[ -f "$NEUWARE_HOME/lib/libcnrt.so" ]]; then
  NEUWARE_LIB_DIR="$NEUWARE_HOME/lib"
else
  echo "::error::Neuware found at $NEUWARE_HOME but no libcnrt.so under lib64/ or lib/" >&2
  exit 1
fi
echo "NEUWARE_LIB_DIR=$NEUWARE_LIB_DIR"

# FLAGOS_BUILD_VENDOR selects the native CNNL build. Default ON to match setup.py.
# The CNNL kernels under csrc/aten/backends/cambricon/ are still being added, so
# only the library itself has to be present today; requiring a kernel .inc here
# would fail the lane on a platform state that is deliberately incomplete.
export FLAGOS_BUILD_VENDOR="${FLAGOS_BUILD_VENDOR:-1}"
if [[ "$FLAGOS_BUILD_VENDOR" != "0" ]]; then
  missing=()
  for asset in "include/cnrt.h" "include/cnnl.h" "libcnnl.so"; do
    case "$asset" in
      include/*) [[ -e "$NEUWARE_HOME/$asset" ]] || missing+=("$NEUWARE_HOME/$asset") ;;
      *) [[ -e "$NEUWARE_LIB_DIR/$asset" ]] || missing+=("$NEUWARE_LIB_DIR/$asset") ;;
    esac
  done
  if (( ${#missing[@]} > 0 )); then
    echo "::error::FLAGOS_BUILD_VENDOR=$FLAGOS_BUILD_VENDOR requires the CNNL assets, but this image is missing:" >&2
    printf '::error::  %s\n' "${missing[@]}" >&2
    echo "::error::Install the CNNL component of the Neuware toolkit." >&2
    exit 1
  fi
  echo "cnnl assets: present"
fi

# --- Device node -------------------------------------------------------------
# Only the integration stage needs a card; a build can run on any MLU host.
if [[ "$CI_STAGE" == "integration" ]] && ! ls /dev/cambricon_dev* >/dev/null 2>&1; then
  echo "::error::No Cambricon device node (/dev/cambricon_dev*) is visible"
  exit 1
fi

# --- Backend selection -------------------------------------------------------
# Chip selection is FLAGOS_ACCELERATOR's job alone: one wheel serves every MLU
# part, exactly like the other native-vendor lanes.
export FLAGOS_ACCELERATOR=cambricon
# FlagGems Python ops run through the mlu Triton backend. The C++ wrappers need
# liboperators.so built for the Neuware SDK, which does not exist.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1
# No CUDA runtime on this platform, so the CUDA asset preload has nothing to open.
export FLAGOS_DISABLE_CUDA_ASSETS=1
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

export PATH="$NEUWARE_HOME/bin:$PATH"
export CPATH="$NEUWARE_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NEUWARE_LIB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NEUWARE_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Isolated Python ---------------------------------------------------------
# The vendor image carries torch_mlu, which must not leak into the build: the
# wheel is built against stock CPU torch and torch_fl publishes a torch_mlu shim
# of its own, so a real vendor package in the same interpreter is a conflict.
BOOTSTRAP_PYTHON="${TORCH_FL_BOOTSTRAP_PYTHON:-}"
if [[ -z "$BOOTSTRAP_PYTHON" || ! -x "$BOOTSTRAP_PYTHON" ]]; then
  for candidate in python3.12 python3 python; do
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

PREBUILT_VENV="${TORCH_FL_PREBUILT_CAMBRICON_VENV:-/opt/torch-fl-cambricon-venv}"
USING_PREBUILT=0
if [[ -z "${TORCH_FL_VENV_ROOT:-}" && -x "$PREBUILT_VENV/bin/python" ]]; then
  VENV_ROOT="$PREBUILT_VENV"
  USING_PREBUILT=1
  echo "Using prebuilt Cambricon venv: $VENV_ROOT"
else
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-cambricon-${CI_STAGE}}"
  "$BOOTSTRAP_PYTHON" -m venv --clear "$VENV_ROOT" || true
  # Ensure system site-packages are not inherited (torch_mlu from base image)
  if [[ -f "$VENV_ROOT/pyvenv.cfg" ]]; then
    grep -v '^include-system-site-packages' "$VENV_ROOT/pyvenv.cfg" > "$VENV_ROOT/pyvenv.cfg.tmp"
    echo "include-system-site-packages = false" >> "$VENV_ROOT/pyvenv.cfg.tmp"
    mv "$VENV_ROOT/pyvenv.cfg.tmp" "$VENV_ROOT/pyvenv.cfg"
  fi
fi

VENV_PYTHON="$VENV_ROOT/bin/python"
if ! venv_is_usable; then
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    python_mm="$($BOOTSTRAP_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "python${python_mm}-venv"
    "$BOOTSTRAP_PYTHON" -m venv --clear "$VENV_ROOT"
    if [[ -f "$VENV_ROOT/pyvenv.cfg" ]]; then
      grep -v '^include-system-site-packages' "$VENV_ROOT/pyvenv.cfg" > "$VENV_ROOT/pyvenv.cfg.tmp"
      echo "include-system-site-packages = false" >> "$VENV_ROOT/pyvenv.cfg.tmp"
      mv "$VENV_ROOT/pyvenv.cfg.tmp" "$VENV_ROOT/pyvenv.cfg"
    fi
  fi
fi

if ! venv_is_usable; then
  echo "::error::Isolated Python was not created at $VENV_ROOT; install the matching python*-venv package in the CI image" >&2
  exit 1
fi

if (( USING_PREBUILT == 0 )); then
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL_ARG" \
    --upgrade pip setuptools wheel cmake ninja build
  install_cpu_torch
fi

if [[ "$CI_STAGE" == "integration" ]]; then
  # numpy stays <2 for the same reason the other lanes pin it: 2.x breaks the
  # stock +cpu torch C extensions at import.
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL_ARG" \
    pytest "numpy<2" pyyaml
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "VIRTUAL_ENV=$VENV_ROOT"
    echo "PATH=$VENV_ROOT/bin:$PATH"
    echo "PYTHONNOUSERSITE=1"
    echo "PYTHONPATH="
  } >> "$GITHUB_ENV"
fi

# Ensure torch_mlu from the base image is not importable in the venv.
if "$VENV_PYTHON" -c "import importlib.util; exit(0 if importlib.util.find_spec('torch_mlu') is None else 1)" 2>/dev/null; then
  : # torch_mlu is not visible, isolation is working
else
  echo "::warning::torch_mlu is visible in the venv; attempting to uninstall"
  "$VENV_PYTHON" -m pip uninstall -y torch_mlu torch_mlu_ops 2>/dev/null || true
  if "$VENV_PYTHON" -c "import importlib.util; exit(0 if importlib.util.find_spec('torch_mlu') is None else 1)" 2>/dev/null; then
    echo "::notice::torch_mlu successfully removed from venv"
  else
    echo "::error::torch_mlu still visible after uninstall attempt"
    exit 1
  fi
fi

# --- MLU Triton (lane `triton` wheel) + FlagGems ------------------------------
# Installed for both stages, not just integration: build and integration share
# one platform job, and restricting this to CI_STAGE=integration would leave the
# build job's venv without flag_gems -- the wheel then fails as soon as a
# FlagGems route dispatches.
#
# Deliberately a `triton` wheel and not `flagtree`: no flagtree build exists for
# cambricon on the lane, and the mlu Triton backend is what FlagGems' cambricon
# kernel path needs. --no-deps so pip cannot replace the pinned CPU torch.
# No flagtree build exists for cambricon, and what FlagGems' cambricon backend
# and the driver it launches through need is `triton.backends.mlu`. The lane
# publishes that as a `triton` wheel instead, so that is what gets installed.
#
# The shared install_flagtree phase still runs first, pointed at the same lane:
# every hook calls it, a wheel missing Triton would otherwise surface only when a
# FlagGems route first dispatches. Today it finds no cambricon flagtree and is
# skipped rather than failing the job, so the triton wheel below is the one that
# actually carries the backend. When a `flagtree 0.7.0+cambricon3.6` is published
# to FLAGTREE_INDEX_URL, this hook stops installing triton and releases version
# alignment with the other lanes with no other change.
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.7.0+cambricon3.6}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
install_flagtree || echo "::notice::no cambricon flagtree on the lane; using the triton wheel below"

FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-cambricon/simple}"
TRITON_VERSION="${TORCH_FL_CAMBRICON_TRITON_VERSION:-$TRITON_VERSION_cambricon}"
pip_retry --no-deps --only-binary=:all: --index-url "$FLAGGEMS_INDEX_URL" \
  "triton===$TRITON_VERSION"

# The triton wheel may pull torch_mlu; uninstall again to keep the isolation this
# hook establishes above.
"$VENV_PYTHON" -m pip uninstall -y torch_mlu torch_mlu_ops 2>/dev/null || true

FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
install_flag_gems

pip_retry --index-url "$PIP_INDEX_URL_ARG" 'packaging>=26.0'
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'PyYAML==6.0.1'

if command -v cnmon >/dev/null 2>&1; then
  cnmon version || true
fi

# --- Export to later workflow steps ------------------------------------------
if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  export_ci_env PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR NEUWARE_HOME FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS FLAGOS_DISABLE_CUDA_ASSETS CPATH LIBRARY_PATH LD_LIBRARY_PATH
fi

cd "$REPO_ROOT"
# build_ext produces the libtorch_fl.so that package_data stages into the wheel,
# and build and test share one job, so it must run in both stages. No device is
# required here; the card was checked above for the integration stage.
build_flagos_inplace
