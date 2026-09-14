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
export PIP_INDEX_URL="${TORCH_FL_PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
export PIP_DEFAULT_TIMEOUT="${TORCH_FL_PIP_DEFAULT_TIMEOUT:-300}"
export PIP_RETRIES="${TORCH_FL_PIP_RETRIES:-20}"

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

# --- Device node -------------------------------------------------------------
# Ascend exposes a manager device plus per-card davinci nodes. Require the
# manager node; card count is asserted later from torch_fl.flagos.device_count().
if [[ ! -c /dev/davinci_manager ]]; then
  echo "::error::Ascend device node /dev/davinci_manager is unavailable"
  exit 1
fi

# --- Environment -------------------------------------------------------------
export ACCELERATOR=ascend
export ASCEND_HOME
# Ascend has no CUDA assets or CUDA runtime. Keep the ACLNN backend as the
# native fallback and enable the patched FlagGems Python path by default.
export FLAGOS_DISABLE_CUDA_ASSETS=1
export FLAGOS_USE_FLAGGEMS=1
export FLAGOS_USE_FLAGGEMS_CPP=0
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON=1
# triton-ascend's taskqueue launch path calls at_npu::native::OpCommand, a
# torch_npu symbol torch_fl does not provide. Turn it off through the env var
# triton-ascend already reads rather than relying on patch_triton_ascend.py to
# rewrite the default: that rewrite is an exact-string replace written against
# triton-ascend 3.2.0 and silently stopped matching on the 3.2.2 wheel pinned
# below, which left taskqueue on. The empty OpCommand.h stub written by
# scripts/setup_torch_npu_stubs.sh is only sufficient while taskqueue is off --
# with it on, the generated launcher references the symbol and the JIT compile
# fails with "'at_npu' has not been declared", which is what broke 11 Ascend
# operator tests in run 34786387238. An env var cannot version-rot.
export TRITON_ENABLE_TASKQUEUE=false
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

# --- MSPTI profiler interposer -----------------------------------------------
# CANN intercepts aclrtMemcpy*/aclrtMemset* by symbol interposition, so
# libmspti.so must already be in the ELF link map when libascendcl.so resolves
# those calls. The tracer's own dlopen at profiler-session start is too late:
# measured with a standalone C program, dlopen'ing libmspti before libascendcl
# still yields zero memcpy/memset records, while a process-start preload yields
# real ones. Kernel and runtime activities are unaffected either way.
#
# Do NOT export this as LD_PRELOAD for the whole Ascend job. CANN 9.0's
# interposer is not inert for ordinary operator processes: CI observed sporadic
# SIGSEGV exits in add, embedding, le, mean, and mm subprocesses when every
# process inherited it. Keep the prepared value under an inert variable; the
# integration runner applies it only to the profiler contract process through
# that test's `environment` mapping. The visible pytest command remains exactly
# the same on every platform, without destabilizing unrelated Ascend tests.
#
# Guarded on the file existing so a CANN image without the profiling tools does
# not get an LD_PRELOAD that ld.so can only warn about. The variable is always
# assigned because the GITHUB_ENV loop reads each name with ${!name} under
# `set -u`. See docs/architecture/profiler.md.
ASCEND_MSPTI_LIB="$ASCEND_HOME/tools/mspti/lib64/libmspti.so"
if [[ -f "$ASCEND_MSPTI_LIB" ]]; then
  export ASCEND_MSPTI_PRELOAD="$ASCEND_MSPTI_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "MSPTI interposer prepared for profiler process: $ASCEND_MSPTI_LIB"
else
  export ASCEND_MSPTI_PRELOAD="${LD_PRELOAD:-}"
  echo "::warning::libmspti.so not found at $ASCEND_MSPTI_LIB; profiler memcpy/memset activity will be unavailable"
fi

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
  if [[ "$CI_STAGE" == "integration" ]]; then
    "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" pytest
  fi
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# Both the build and integration jobs execute this setup script in the same
# combined platform job. Install FlagGems before either stage builds or tests
# the wheel; restricting this block to CI_STAGE=integration leaves the build
# job's prebuilt venv without flag_gems, and the resulting wheel fails as soon
# as the FlagGems-first Ascend conf dispatches an operator.
#
# Use --no-deps for the source packages so pip cannot replace the pinned CPU
# torch. Triton-Ascend is supplied by its vendor index and also installs
# without dependency resolution; the venv already contains the compatible
# Python/Torch base. Individual retries are intentional: the shared mirror can
# close a large-wheel response early, producing IncompleteRead even though the
# package is available. Retrying the failed package avoids restarting all setup.
pip_retry() {
  local attempt=1
  while true; do
    # Increase pip's own retry limit and timeout for large wheels on unstable networks.
    # triton-ascend 3.2.2 is 270 MB; the default timeout (15s) and retries (5) are
    # insufficient when the mirror link drops mid-download.
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

# FlagGems commit 8e1fce4 adds triton.knobs, and 4ff8a0f adds tl.map_elementwise,
# both unavailable in triton-ascend 3.2.x. Pin to 82b4b0c, the last commit before
# those Triton APIs were used.
FLAGGEMS_REVISION="82b4b0c10e6bb628ba9d0d4054913f3bdb6f87a2"
pip_retry --no-deps "git+https://github.com/flagos-ai/FlagGems.git@${FLAGGEMS_REVISION}"
# Install dependencies one by one to avoid IncompleteRead causing full batch retry
pip_retry --index-url "$PIP_INDEX_URL" pybind11
pip_retry --index-url "$PIP_INDEX_URL" packaging
pip_retry --index-url "$PIP_INDEX_URL" 'PyYAML==6.0.1'
pip_retry --index-url "$PIP_INDEX_URL" 'sqlalchemy==2.0.48'
pip_retry --index-url "$PIP_INDEX_URL" 'numpy>=1.20,<2.0'
# Install the latest published triton-ascend wheel (3.2.2). Source builds from
# the 3.5 branch require unreliable gitcode.com submodules and exceed the CI
# timeout; the 3.2.x wheel is stable and fast.
pip_retry --no-deps --extra-index-url 'https://triton-ascend.osinfra.cn/pypi/simple' 'triton-ascend==3.2.2'
"$VENV_PYTHON" "$REPO_ROOT/scripts/patch_triton_ascend.py"

# Fail early in the wheel-only stage if the pinned FlagGems/Triton pair
# can be imported together. The build stage cannot run this check because
# torch_fl._C does not exist until the wheel has been built and installed.
if [[ "$CI_STAGE" == "integration" ]]; then
  # torch_fl must be imported first so its PrivateUse1 shim owns the device key.
  "$VENV_PYTHON" - <<'PY'
import torch_fl
import flag_gems

print(f"FlagGems import: {flag_gems.__file__}")
PY
fi

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
    CPATH LIBRARY_PATH LD_LIBRARY_PATH ASCEND_MSPTI_PRELOAD; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

cd "$REPO_ROOT"

if [[ "$CI_STAGE" == "build" ]]; then
  # Prebuild so package_data sees libtorch_fl.so before the common workflow
  # invokes python -m build. The following wheel build is incremental. The
  # integration job downloads this artifact and must not rebuild from source.
  python setup.py build_ext --inplace

  # Build-stage availability check. Integration repeats this after installing
  # the artifact wheel from an isolated test workspace.
  python - <<'PY'
import torch_fl
import torch

assert torch_fl.flagos.is_available(), "flagos device is unavailable"
n = torch_fl.flagos.device_count()
assert n >= 1, f"expected >=1 flagos device, got {n}"
print(f"flagos devices: {n}")
PY
fi
