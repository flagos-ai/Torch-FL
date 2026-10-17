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

# Apple Silicon (MPS) environment for the common build/integration workflow.
#
# MPS is the platform this abstraction was not written for, and the differences
# are all in one direction: there is no vendor SDK to source, no container to
# enter, and no vendor index to install from. What remains is a stock macOS
# runner, a PyPI torch, and the system Metal toolchain, which is exactly what
# this hook asserts before either stage runs.
#
# There is no FlagTree/FlagGems cohort here and none is installed: PyPI
# publishes no `triton` wheel for macOS, and there is no Metal FlagTree line, so
# torch_fl's MPS backend is native PrivateUse1 kernels with no operator-source
# dependency (see NO_OPERATOR_COHORT in tests/unit/test_ci_version_pins.py and
# the mps arm of setup.py). The absence is the platform's design, not an
# omission, so this hook installs none and does not check for any.

# --- Runner, toolchain, and the device ---------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "::error::MPS requires a macOS runner, found $(uname -s)" >&2
  exit 1
fi

# The accelerator is the Apple GPU, so an x86_64 runner has none. Failing here
# names the label that is wrong once, instead of leaving every MPS test to skip
# and the suite to report success over an empty run.
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "::error::MPS requires an arm64 runner (Apple Silicon), found $(uname -m); check runner_labels in .github/configs/mps.yml" >&2
  exit 1
fi

# The Metal SDK is the one toolchain this backend links against, and the CMake
# arm refuses to configure without it. Xcode command line tools supply it.
if ! xcode-select -p >/dev/null 2>&1; then
  echo "::error::Xcode command line tools are required for the Metal SDK (xcrun clang++ -framework Metal)" >&2
  exit 1
fi
if ! xcrun --sdk macosx --show-sdk-path >/dev/null 2>&1; then
  echo "::error::No macOS SDK found through xcrun; install Xcode command line tools" >&2
  exit 1
fi
echo "macOS SDK: $(xcrun --sdk macosx --show-sdk-path)"

# --- Backend selection -------------------------------------------------------
# Chip selection is FLAGOS_ACCELERATOR's job alone. FLAGOS_BUILD_VENDOR selects
# this platform's native kernels and defaults ON to match setup.py; boxing stays
# off, which is pinned in cmake/flagos_platforms.json for mps rather than left
# to the environment. MPS has no libtorch_cuda.so to preload.
export FLAGOS_ACCELERATOR=mps
export FLAGOS_BUILD_VENDOR="${FLAGOS_BUILD_VENDOR:-1}"
export FLAGOS_DISABLE_CUDA_ASSETS=1
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

# --- One architecture, the one this runner has -------------------------------
# The python.org installer on a GitHub macOS runner is a universal2 build, so
# distutils tags everything it compiles `macosx-10.13-universal2` and passes
# `-arch arm64 -arch x86_64` to clang. CMake compiles for the host alone, so
# the dylibs it installs are arm64-only and the second `-arch` only produces a
# broken link: `ld: warning: ignoring file 'libtorch_bindings.dylib': found
# architecture 'arm64', required architecture 'x86_64'`, after which `_C.so`
# silently carries no reference to the backend it exists to load. A wheel can
# only be as native as its unstripped half, so the wheel is built for one
# architecture: arm64, which is what the runner this lane selects has, what
# Metal exists for, and what PyPI's torch ships for macOS.
#
# ARCHFLAGS is the knob distutils/`_osx_support` reads (it overrides the
# per-interpreter default); setuptools' bdist_wheel consults
# `_PYTHON_HOST_PLATFORM` for the tag, so both are set -- otherwise the wheel
# would still be *labelled* universal2 while containing only arm64, which
# `check_platform_environment.py`'s arm64 assertion (and `pip`, on an x86_64
# host) would then trust.
export ARCHFLAGS="-arch arm64"
export _PYTHON_HOST_PLATFORM="macosx-$(sw_vers -productVersion | cut -d. -f1,2)-arm64"
export CMAKE_OSX_ARCHITECTURES=arm64

# --- Isolated Python ---------------------------------------------------------
# The runner image Python is fine to use directly: unlike the vendor images
# there is no conflicting vendor torch to keep out, and isolating would only add
# a venv whose interpreter has to match the one the macOS runner provisions.
# VENV_ROOT is still honoured so a developer can point this at a local venv.
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
  echo "::error::Unable to find Python to bootstrap the environment" >&2
  exit 1
fi

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-mps-${CI_STAGE}}"
"$BOOTSTRAP_PYTHON" -m venv --clear "$VENV_ROOT"
VENV_PYTHON="$VENV_ROOT/bin/python"

if ! venv_is_usable; then
  echo "::error::Isolated Python was not created at $VENV_ROOT" >&2
  exit 1
fi

# torch on macOS comes from PyPI un-suffixed: there is no `2.10.0+cpu` macosx
# wheel, and the MPS backend lives in the default build. The Linux index the
# other hooks point at serves no arm64 macOS wheel at all -- and the shared
# helper measures that index by default -- so both are repointed at PyPI, whose
# resolver picks the right macosx arm64 wheel for the local tag. numpy<2 for the
# same reason the other hooks pin it: 2.x breaks the stock torch C extensions at
# import.
CPU_TORCH_INDEX_URL="https://pypi.org/simple"
CPU_TORCH_MIRROR_INDEX_URL_DEFAULT=""
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
pip_retry --upgrade pip
# The wheel step runs `python -m build --wheel --no-isolation`, which reads the
# build requirements from pyproject.toml instead of provisioning them in an
# isolated env -- so `cmake>=3.18` and `wheel` have to be present in this venv or
# that step fails with `ERROR Unmet dependencies ... found: not installed`. The
# vendor images happen to ship cmake; a stock macOS runner image does not, and
# the runner's own cmake is a Homebrew binary outside this venv where
# `--no-isolation` will not look for it. Installed from PyPI rather than
# Homebrew so the version the build backend resolves is the one it verified.
#
# `setuptools` is upgraded alongside because a macOS runner image can carry an
# older one than the pyproject build requirements ask for, and `--upgrade` there
# is what keeps this list matching what `python -m build` will demand.
pip_retry --upgrade setuptools "cmake>=3.18" wheel
install_cpu_torch
CPU_TORCH_INDEX_URL="${CPU_TORCH_INDEX_URL_DEFAULT}"
CPU_TORCH_VERSION="${CPU_TORCH_VERSION_DEFAULT}"

if [[ "$CI_STAGE" == "integration" ]]; then
  # Test dependencies. transformers is pinned to [4.51, 5) exactly as the other
  # hooks pin it: 4.51 is where Qwen3 model_type support landed and 5.x has the
  # TokenizersBackend regression. The MPS lane runs the operator suite and the
  # runtime traversal rather than the model groups, but the ops conftest imports
  # the same shared fixtures.
  pip_retry pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf
fi

# --- Native backtraces on a fatal signal -------------------------------------
# A segfault inside an MPS op surfaces in the pytest log as a Python traceback
# whose deepest frame is torch_fl's __torch_function__ trampoline, so the log
# names the aten call that was in flight and nothing below it. On a hosted runner
# there is no debugger attachment and no core file, which leaves the crash
# address-less and the fix a guess. The MPS runtime installs a SIGSEGV/SIGBUS/
# SIGABRT/SIGILL/SIGFPE handler that prints a backtrace when this variable is
# set (see CrashTraceHandler in csrc/runtime/accelerator/mps/mps_runtime.mm), and
# the handler re-raises with the default disposition so the exit status and the
# reported failure are unchanged.
#
# Integration only: the build stage never runs an op, so there is nothing there
# for it to catch. Assigned unconditionally as well as exported, because
# export_ci_env reads every name it is given with ${!name} under `set -u`, so an
# unset FLAGOS_MPS_CRASH_TRACE would abort this hook on the build stage.
FLAGOS_MPS_CRASH_TRACE=""
if [[ "$CI_STAGE" == "integration" ]]; then
  FLAGOS_MPS_CRASH_TRACE=1
fi
export FLAGOS_MPS_CRASH_TRACE

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

# --- Verify the interpreter --------------------------------------------------
CI_STAGE="$CI_STAGE" "$VENV_PYTHON" - <<'PY'
import os
import platform
import sys

import torch

# The un-suffixed PyPI build, not the `+cpu` Linux wheel: torch.version.cuda is
# None and the version carries no local tag.
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert torch.backends.mps.is_built(), "this torch build has no MPS support"
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"Machine: {platform.machine()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
PY

# --- Verify the accelerator is visible through torch_fl ----------------------
if "$VENV_PYTHON" -c "import torch_fl._C" 2>/dev/null; then
  "$VENV_PYTHON" - <<'PY'
import torch_fl

assert torch_fl.flagos.is_available(), "flagos device is unavailable"
assert torch_fl.flagos.device_count() >= 1, torch_fl.flagos.device_count()
print(f"flagos devices: {torch_fl.flagos.device_count()}")
PY
fi

if command -v system_profiler >/dev/null 2>&1; then
  system_profiler SPDisplaysDataType 2>/dev/null | head -20 || true
fi

# --- Export to later workflow steps ------------------------------------------
if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  export_ci_env PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR FLAGOS_BUILD_VENDOR FLAGOS_DISABLE_CUDA_ASSETS ARCHFLAGS _PYTHON_HOST_PLATFORM CMAKE_OSX_ARCHITECTURES FLAGOS_MPS_CRASH_TRACE
fi

cd "$REPO_ROOT"
# build_ext produces the libtorch_fl.so the wheel stages; build and test share
# one job. No device is required here beyond the Metal SDK asserted above.
build_flagos_inplace
