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
#
# FlagGems runs here on FlagTree, not on triton-ascend. triton-ascend 3.2.x was
# the previous provider and is gone: it lags the Triton APIs current FlagGems
# uses, its task-queue launch path calls at_npu::native::OpCommand (a torch_npu
# symbol torch_fl must not link), and the exact-string patch that used to strip
# those calls stopped matching on 3.2.2 and silently no-opped, which broke 11
# operator tests in run 34786387238. FlagTree 0.6.2a1+ascend3.5 is the Triton
# 3.5 build for this backend, and torch_fl carries a torch_npu-free backend
# policy for it (torch_fl/compile/flagtree_ascend_policy.py), so no torch_npu
# appears anywhere in this environment. See docs/vendors/ascend/installation.md.
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
# Default PyPI index for build deps (pip/setuptools/wheel/cmake/build/pytest).
# CPU torch is installed from CPU_TORCH_INDEX_URL, not this generic PyPI mirror.
export PIP_INDEX_URL="${TORCH_FL_PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}"
export PIP_DEFAULT_TIMEOUT="${TORCH_FL_PIP_DEFAULT_TIMEOUT:-300}"
export PIP_RETRIES="${TORCH_FL_PIP_RETRIES:-20}"
# FlagTree and FlagGems both come from the FlagOS index.
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"

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
export FLAGOS_ACCELERATOR=ascend
export ASCEND_HOME
# Ascend has no CUDA assets or CUDA runtime. Keep the ACLNN backend as the
# native fallback and enable the FlagGems Python path by default.
export FLAGOS_DISABLE_CUDA_ASSETS=1
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1
# FlagTree's task queue launches through at_npu::native::OpCommand, a torch_npu
# symbol this environment does not have (and must not have). torch_fl's FlagTree
# backend policy turns the task queue off itself when it is installed; setting it
# here as well keeps the environment correct before the first torch_fl import,
# which matters because FlagTree's launcher __init__ is reached from the first
# FlagGems kernel rather than from a hook the policy could wrap.
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
#
# 3.11 specifically: the FlagTree Ascend 3.5 wheel is published for cp311 only
# (see the FlagTree user manual's install table), so a 3.12 venv would simply
# fail to resolve flagtree. Prefer the interpreter the wheel needs instead of
# discovering that at pip time.
VENDOR_PYTHON="${TORCH_FL_VENDOR_PYTHON:-}"
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  for _pyc in python3.11 python3 python; do
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
if ! "$VENDOR_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
  echo "::error::Ascend needs Python 3.11: the FlagTree ascend3.5 wheel is cp311-only, and $VENDOR_PYTHON is $("$VENDOR_PYTHON" -V 2>&1)"
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
  # `build` is explicit rather than assumed: the job runs `python -m build
  # --wheel --no-isolation`, and the FlagTree image does not ship it. A missing
  # `build` does not fail cleanly -- the repo's own CMake `build/` directory is
  # found first as a namespace package and the step dies with the misleading
  # "No module named build.__main__; 'build' is a package and cannot be directly
  # executed".
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" --upgrade \
    pip setuptools wheel cmake build
  "$VENV_PYTHON" -m pip install --index-url "$CPU_TORCH_INDEX_URL" \
    "torch==$CPU_TORCH_VERSION"
  # Unconditional, unlike the other vendors: Ascend's job builds the wheel and
  # runs the tests under one CI_STAGE=build invocation, so gating pytest on
  # CI_STAGE=integration installs it in no job at all. The previous CI image hid
  # this by shipping a prebuilt /opt/torch-fl-ascend-venv that already had it;
  # the FlagTree image does not, and the flagtree wheel is cp311-only anyway, so
  # that Python 3.12 venv could not have been reused here.
  "$VENV_PYTHON" -m pip install --index-url "$PIP_INDEX_URL" pytest
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
# torch. Individual retries are intentional: the shared mirror can close a
# large-wheel response early, producing IncompleteRead even though the package
# is available. Retrying the failed package avoids restarting all setup.
pip_retry() {
  local attempt=1
  while true; do
    # Raise pip's own retry limit and timeout for large wheels on unstable
    # networks: the flagtree wheel is ~200 MB, and the default timeout (15s) and
    # retries (5) are not enough when the mirror link drops mid-download.
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

# The FlagGems install is a VCS install, and pip reports the exit status of its
# last step -- the wheel build of whichever tree it managed to fetch. A checkout
# the runner's proxy truncated therefore still ends in `Successfully installed`,
# and pip never notices. On 2026-09-18 the Ascend runner's clone spent ten
# minutes printing
#   fatal: unable to access 'https://github.com/flagos-ai/FlagGems.git/':
#   Proxy CONNECT aborted
# (364 times), alongside `error: unable to read sha1 file of ...` and
# `error: invalid object 100644 2e574121... for
# '.github/workflows/rule-check.yaml'` for the blobs it never received, and then
# reported
#   Successfully installed flaggems_setup-0.0.0
# -- a 2.1 MB stub named after the build scaffolding rather than the project,
# where the same revision produced the 10 MB
# flag_gems-5.4.0rc2.post1+g437ba3938 on every other platform that ran that
# morning. Nothing failed until the integration suite took its first FlagGems
# route, four minutes later.
#
# So check the install instead of trusting pip's status, and reinstall when it
# is wrong: the conf routes this platform's operators to flagos_python, so an
# unusable flag_gems is not a state this script may leave behind.
flag_gems_installed() {
  "$VENV_PYTHON" - "${FLAGGEMS_REVISION:0:9}" <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

try:
    version = metadata.version("flag_gems")
except metadata.PackageNotFoundError:
    raise SystemExit("flag_gems is not installed")

# A distribution can be installed with no importable package behind it, which
# is what a truncated checkout produces.
if importlib.util.find_spec("flag_gems") is None:
    raise SystemExit(f"flag_gems {version} has no importable package")

print(f"flag_gems {version}")
if sys.argv[1] not in version:
    # Not a failure: a revision given as a branch name, or a tarball without
    # git metadata, lands on a version string that names neither. Only warn --
    # reinstalling cannot change how the version was written.
    print(
        f"::warning::flag_gems {version} does not name the requested revision "
        f"{sys.argv[1]}; the checkout it was built from may be incomplete",
        file=sys.stderr,
    )
PY
}

# Three attempts at most, and only for an install pip called successful: a pip
# failure has already been retried five times by pip_retry, and repeating that
# spends the job's budget on a link that is down rather than on a bad checkout.
install_flag_gems() {
  local attempt=1
  while true; do
    if ! pip_retry --no-deps "git+${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"; then
      echo "::error::could not install FlagGems from ${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"
      return 1
    fi
    if flag_gems_installed; then
      return 0
    fi
    if (( attempt >= 3 )); then
      echo "::error::no usable flag_gems after $attempt installs of ${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"
      return 1
    fi
    echo "::warning::install attempt $attempt left no usable flag_gems; reinstalling"
    attempt=$((attempt + 1))
    # A truncated tree installs under the build scaffolding's name, so both
    # names have to go for the next attempt to be read as a fresh result.
    "$VENV_PYTHON" -m pip uninstall -y flag_gems flaggems_setup >/dev/null 2>&1 || true
    sleep 10
  done
}

# FlagTree installs the module named `triton`, so any stock or vendor Triton
# already present would be shadowed rather than replaced, and the user manual
# asks for it to be removed first. The venv is fresh and has none, but the
# prebuilt-venv path may, and `pip uninstall triton` is a no-op when it does not.
for _ in 1 2 3; do
  "$VENV_PYTHON" -m pip uninstall -y triton >/dev/null 2>&1 || true
done

# FlagTree, the Triton build carrying the Ascend backend. Pinned, not tracking a
# moving tag: the FlagGems revision below is validated against this exact wheel,
# and the Triton minor (3.5) has to match the backend the wheel was built for.
# From the FlagTree user manual ("ascend", Triton 3.5 row).
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.6.2a1+ascend3.5}"
pip_retry --no-deps --index-url "$FLAGTREE_INDEX_URL" "flagtree===${FLAGTREE_VERSION}"

# FlagGems from the flagos-ai fork, tracking master by policy: every CI run
# measures the current master, not a pinned snapshot. Override with
# TORCH_FL_FLAGGEMS_REVISION to pin a commit for a reproducible run, e.g.:
#   TORCH_FL_FLAGGEMS_REVISION=$(git ls-remote https://github.com/flagos-ai/FlagGems.git HEAD | cut -f1)
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
install_flag_gems

# FlagGems' own runtime deps, installed one at a time to avoid IncompleteRead
# failing the whole batch. numpy stays <2: 2.x breaks the stock +cpu torch C
# extensions at import, and the Ascend test groups import both.
pip_retry --index-url "$PIP_INDEX_URL" pybind11
pip_retry --index-url "$PIP_INDEX_URL" packaging
pip_retry --index-url "$PIP_INDEX_URL" 'PyYAML==6.0.1'
pip_retry --index-url "$PIP_INDEX_URL" 'sqlalchemy==2.0.48'
pip_retry --index-url "$PIP_INDEX_URL" 'numpy<2'

# --- Verify the isolation held ----------------------------------------------
# torch_fl does not depend on torch_npu -- that is a hard requirement of this
# backend, not a preference: the real extension claims PrivateUse1 on import and
# would lock `flagos` out of the device key. Checked before torch_fl is imported,
# because importing torch_fl installs a stub under that name for FlagGems' sake.
"$VENV_PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

import torch

torch_path = Path(torch.__file__).resolve()
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
assert importlib.util.find_spec("torch_npu") is None, (
    "torch_npu is importable in the isolated venv; it claims PrivateUse1 on "
    "import and would make the flagos device unregisterable"
)
print(f"CPU torch path: {torch_path}")
PY

# --- Verify the FlagTree + FlagGems stack imports ----------------------------
# Integration only: torch_fl._C does not exist until the wheel has been built,
# and the import order below needs it. CI calls this script before building.
#
# torch_fl must be imported before flag_gems. FlagTree's Ascend backend picks
# its host-side implementation by importing torch_npu at discovery time (and
# again from the launcher on the first kernel launch); torch_fl's stub absorbs
# that import, and torch_fl installs its own torch_npu-free backend policy in
# the same step. Importing flag_gems first, or importing triton first, gets the
# torch_npu policy instead and the first FlagGems kernel dies on the stub.
if "$VENV_PYTHON" -c "import torch_fl._C" 2>/dev/null; then
  "$VENV_PYTHON" - <<'PY'
import torch_fl  # noqa: F401  -- must precede triton/flag_gems; installs the backend policy

import torch
import triton
import flag_gems

assert "ascend" in triton.backends.backends, sorted(triton.backends.backends)
assert torch.version.cuda is None, torch.version.cuda
# FlagGems resolves its operator set against this name; a mismatch means the
# Ascend vendor package is missing from the install.
print(f"Triton: {triton.__version__} (backends: {sorted(triton.backends.backends)})")
print(f"FlagGems: {flag_gems.__version__} (vendor: {flag_gems.vendor_name})")
PY
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR ASCEND_HOME \
    FLAGOS_DISABLE_CUDA_ASSETS \
    FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS TRITON_ENABLE_TASKQUEUE \
    PIP_INDEX_URL PIP_DEFAULT_TIMEOUT PIP_RETRIES \
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
