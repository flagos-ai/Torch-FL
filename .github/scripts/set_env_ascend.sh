#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ascend NPU environment for the common build/integration workflow.
#
# Ascend is a standalone vendor backend (pure CANN ACLNN). Unlike CUDA it has no
# boxing layer, and unlike MetaX it does not shim a CUDA runtime. The wheel is
# built against a stock CPU PyTorch (2.10.0+cpu); the ACLNN shared libraries
# from the CANN toolkit are linked at runtime via LD_LIBRARY_PATH.
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
# Default PyPI index for build deps (pip/setuptools/wheel/cmake/build/pytest).
# CPU torch is installed from CPU_TORCH_INDEX_URL, not this generic PyPI mirror.
export PIP_INDEX_URL="${TORCH_FL_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT="${TORCH_FL_PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${TORCH_FL_PIP_RETRIES:-10}"

# --- CANN toolkit root -------------------------------------------------------
# CANN images ship several layouts; pick the first candidate that actually has
# both lib64/ and include/. A preset ASCEND_HOME wins, otherwise try the common
# roots. ASCEND_HOME must expose $ASCEND_HOME/include and $ASCEND_HOME/lib64.
_ascend_candidates=(
  "${ASCEND_HOME:-}"
  /usr/local/Ascend/ascend-toolkit/latest
  /usr/local/ascend/ascend-toolkit/latest
  /usr/local/Ascend/cann-9.0.0/aarch64-linux
  /usr/local/Ascend/cann-9.0.0
  /usr/local/Ascend/latest
)
ASCEND_HOME=""
for _cand in "${_ascend_candidates[@]}"; do
  [[ -z "$_cand" ]] && continue
  if [[ -d "$_cand/lib64" && -d "$_cand/include" ]]; then
    ASCEND_HOME="$_cand"
    break
  fi
done
if [[ -z "$ASCEND_HOME" ]]; then
  echo "::error::CANN toolkit not found; none of the candidates had lib64/ + include/."
  echo "::error::Tried: ${_ascend_candidates[*]}. Set ASCEND_HOME to the CANN root."
  exit 1
fi
echo "ASCEND_HOME=$ASCEND_HOME"

# Runtime ACLNN libraries actually linked by libtorch_fl.so. Their presence is
# the minimum proof that the image can drive an Ascend kernel.
for lib in libascendcl.so libopapi.so libnnopbase.so; do
  if ! compgen -G "$ASCEND_HOME/lib64/$lib*" >/dev/null \
     && ! compgen -G "$ASCEND_HOME/acllib/lib64/$lib*" >/dev/null; then
    echo "::error::Required ACLNN library missing: $lib (looked in $ASCEND_HOME/lib64 and $ASCEND_HOME/acllib/lib64)"
    exit 1
  fi
done

# --- Environment -------------------------------------------------------------
export ACCELERATOR=ascend
export ASCEND_HOME
# Ascend has no CUDA assets, no CUDA runtime, and (in the first-version wheel)
# no FlagGems C++/Python path. Disable all of them explicitly so dispatch
# resolves every op to the ascend ACLNN backend.
export FLAGOS_DISABLE_CUDA_ASSETS=1
export FLAGOS_USE_FLAGGEMS=0
export FLAGOS_USE_FLAGGEMS_CPP=0
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON=0
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

# --- Ascend driver (HAL) -----------------------------------------------------
# libascend_hal.so (the hardware abstraction layer) lives in the DRIVER layer,
# which is host-side and bind-mounted into the container (see ascend.yml
# container_options). The toolkit lib64 does NOT ship it, so without these
# paths the built .so imports with "libascend_hal.so: cannot open shared object
# file". Add every driver lib dir that actually exists.
ASCEND_DRIVER_HOME="${ASCEND_DRIVER_HOME:-/usr/local/Ascend/driver}"
DRIVER_LIBS=""
if [[ -d "$ASCEND_DRIVER_HOME" ]]; then
  for _d in \
    "$ASCEND_DRIVER_HOME/lib64" \
    "$ASCEND_DRIVER_HOME/lib64/driver" \
    "$ASCEND_DRIVER_HOME/lib64/extra" \
    "$ASCEND_DRIVER_HOME/lib64/common" \
    "$ASCEND_DRIVER_HOME/lib64/fwcek"; do
    [[ -d "$_d" ]] && DRIVER_LIBS="$DRIVER_LIBS:$_d"
  done
  DRIVER_LIBS="${DRIVER_LIBS#:}"
  echo "ASCEND_DRIVER_HOME=$ASCEND_DRIVER_HOME"
else
  echo "::warning::Ascend driver dir not found at $ASCEND_DRIVER_HOME; libascend_hal.so may be missing at runtime"
fi

# ACLNN headers for the build (aclnnop/*.h, acl/*.h).
export CPATH="$ASCEND_HOME/include${CPATH:+:$CPATH}"
# Link-time and runtime library path. lib64 holds libascendcl/libopapi/libnnopbase;
# acllib/lib64 is the legacy fallback on some CANN images; DRIVER_LIBS adds the
# HAL/runtime libs (libascend_hal.so) from the driver layer.
export LIBRARY_PATH="$ASCEND_HOME/lib64:$ASCEND_HOME/acllib/lib64${DRIVER_LIBS:+:$DRIVER_LIBS}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ASCEND_HOME/lib64:$ASCEND_HOME/acllib/lib64${DRIVER_LIBS:+:$DRIVER_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Build Python / CPU torch ------------------------------------------------
# The CANN image's system Python is used only to bootstrap a venv; it does NOT
# need torch pre-installed. CPU torch (2.10.0+cpu) is installed into the venv
# below: ascend needs no accelerator-linked torch, only the CPU dispatcher plus
# the ACLNN runtime libraries exported above. (The cuda script requires the
# build python to import a vendor torch; that contract does not apply here.)
VENDOR_PYTHON="${TORCH_FL_VENDOR_PYTHON:-}"
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  for _pyc in python python3 python3.12 python3.11; do
    if command -v "$_pyc" >/dev/null 2>&1; then
      VENDOR_PYTHON="$(command -v "$_pyc")"
      break
    fi
  done
fi
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  echo "::error::Unable to find a Python interpreter to bootstrap the venv"
  exit 1
fi

PREBUILT_VENV="${TORCH_FL_PREBUILT_ASCEND_VENV:-/opt/torch-fl-ascend-venv}"
if [[ -z "${TORCH_FL_VENV_ROOT:-}" && -x "$PREBUILT_VENV/bin/python" ]]; then
  VENV_ROOT="$PREBUILT_VENV"
  echo "Using prebuilt Ascend venv: $VENV_ROOT"
else
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-ascend-${CI_STAGE}}"
  if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
    echo "::warning::Build Python cannot create a venv; trying uv"
    if ! command -v uv >/dev/null 2>&1; then
      "$VENDOR_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" --upgrade uv
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
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" --upgrade pip setuptools wheel cmake
  "$VENV_PYTHON" -m pip install --index-url "$CPU_TORCH_INDEX_URL" \
    "torch==${TORCH_FL_CPU_TORCH_VERSION:-2.10.0}"
fi

# Test dependencies, installed whether or not the venv was prebuilt: a venv baked
# without them makes the inference and training groups fail as an environment
# problem that reads as an ACLNN problem. pip is a no-op when they are present.
#
# transformers is pinned to [4.51, 5): 4.51 is where Qwen3 model_type support
# landed, and 5.x carries a TokenizersBackend regression that breaks the Qwen3
# slow-to-fast tokenizer conversion. sentencepiece/tiktoken/protobuf are what
# that conversion needs, since the mounted model dir has no tokenizer.json.
# numpy stays on 1.x: transformers pulls 2.x, which breaks the +cpu torch C
# extensions at import.
#
# triton is deliberately absent from this list. On ascend it must come from
# triton-ascend (see _vendor_supplies_triton in setup.py); stock PyPI triton
# installs over it and then every Triton entry point dies with "0 active
# drivers", which converts a missing-stack failure into a wrong-stack failure
# that looks like a compiler bug.
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" \
    pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# Sanity: the isolated torch must be the CPU wheel, not a CUDA vendor build.
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR ASCEND_HOME \
    FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_USE_FLAGGEMS FLAGOS_USE_FLAGGEMS_CPP \
    FLAGGEMS_KERNEL FLAGGEMS_PYTHON PIP_INDEX_URL PIP_DEFAULT_TIMEOUT PIP_RETRIES \
    CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

# Expose the vendor Triton (triton-ascend) to the CPU torch venv, the way
# set_env_metax.sh exposes triton-metax. torch.compile needs it: the active torch
# is the CPU wheel, which ships no Triton, and triton-ascend has no PyPI release
# satisfying torch's pin, so it is installed into the image out of band rather
# than resolved by pip.
#
# The image at manual-20260816-ascend-ci did not bake triton-ascend at a
# discoverable site, so the symlink strategy fails. Install a real vendor Triton
# or skip torch.compile until the image is updated. A missing vendor Triton is a
# warning, not a setup failure: exiting here would take the other nine groups
# down and leave the run with no evidence at all; instead compile-tests fails on
# its own and the operator, RNG, AMP, profiler, inference and training groups
# still report.
if [[ "$CI_STAGE" == "integration" ]]; then
  VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  if [[ ! -e "$VENV_SITE/triton" ]]; then
    for candidate in /usr/local/lib/python3.*/site-packages \
                     /usr/local/lib/python3.*/dist-packages \
                     /usr/lib/python3.*/site-packages \
                     /usr/lib/python3.*/dist-packages \
                     /opt/conda/lib/python3.*/site-packages; do
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

  # Verify the mandatory test deps against the interpreter that will run the
  # tests, including when the venv above was adopted prebuilt. Triton is probed
  # advisorially (--triton-backend ascend): a missing or misconfigured vendor
  # triton only breaks torch.compile (compile-tests), so it emits ::warning::
  # rather than aborting setup -- see check_integration_deps.py. triton is
  # intentionally absent from --require for the same reason.
  "$VENV_PYTHON" .github/scripts/check_integration_deps.py \
    --require pytest transformers safetensors \
    --triton-backend ascend

  # Device node check: Ascend exposes a manager device plus per-card davinci
  # nodes. Require the manager node; card count is asserted later from
  # torch_fl.flagos.device_count().
  if [[ ! -c /dev/davinci_manager ]]; then
    echo "::error::Ascend device node /dev/davinci_manager is unavailable"
    exit 1
  fi
fi

cd "$REPO_ROOT"
# build_ext runs in both stages. It produces the libtorch_fl.so that package_data
# stages into the wheel, and build and test share one container job, so gating it
# to the build stage would leave the integration stage building a wheel with no
# native library. No device is needed here; the card is checked above for the
# integration stage and again by the device-availability preflight.
python setup.py build_ext --inplace
