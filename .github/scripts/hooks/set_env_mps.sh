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
# This platform has no vendor SDK to source, no container to enter, and no vendor
# index to install from. What remains is a stock macOS runner, a PyPI torch, and
# the system Metal toolchain, which is exactly what this hook asserts before
# either stage runs.
#
# This hook installs no FlagTree or FlagGems build: there is no Metal FlagTree
# line, and the platform table (cmake/flagos_platforms.json) turns both off for
# MPS. The native runtime and operator kernels link to Metal directly.

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
# this platform's native kernels and defaults ON to match setup.py. MPS has no
# libtorch_cuda.so to preload.
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

# --- Python and build requirements -------------------------------------------
# The workflow-dispatch test runs on a developer runner too, so this hook is the
# only place that provisions the macOS toolchain from scratch.
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

# torch on macOS comes from PyPI without a local version suffix, and the default
# build includes MPS. Repoint the helper at PyPI, which serves the arm64 wheel.
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
  # Test dependencies, including the shared conftest imports.
  pip_retry pytest "transformers>=4.51,<5" "numpy<2" safetensors sentencepiece tiktoken protobuf
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

# --- Metal buffer wrapper ----------------------------------------------------
METAL_DEVICE_WRAPPER_TYPE_WAS_SET=0
if [[ -n "${METAL_DEVICE_WRAPPER_TYPE:-}" ]]; then
  METAL_DEVICE_WRAPPER_TYPE_WAS_SET=1
elif [[ "$(sw_vers -productVersion | cut -d. -f1)" -ge 14 ]]; then
  # Select the MetalTools wrapper on systems that provide it.
  METAL_DEVICE_WRAPPER_TYPE=1
else
  METAL_DEVICE_WRAPPER_TYPE=0
fi
if (( ! METAL_DEVICE_WRAPPER_TYPE_WAS_SET )); then
  export METAL_DEVICE_WRAPPER_TYPE
fi
echo "Metal buffer wrapper: $METAL_DEVICE_WRAPPER_TYPE"

# --- MPS allocator limits ----------------------------------------------------
TORCH_FL_MPS_HIGH_WATERMARK_RATIO=0.0
TORCH_FL_MPS_LOW_WATERMARK_RATIO=0.5

# --- Verify the interpreter --------------------------------------------------
CI_STAGE="$CI_STAGE" \
TORCH_FL_MPS_HIGH_WATERMARK_RATIO="$TORCH_FL_MPS_HIGH_WATERMARK_RATIO" \
TORCH_FL_MPS_LOW_WATERMARK_RATIO="$TORCH_FL_MPS_LOW_WATERMARK_RATIO" \
"$VENV_PYTHON" - <<'PY'
import os
import platform
import sys


high_ratio = os.environ["TORCH_FL_MPS_HIGH_WATERMARK_RATIO"]
low_ratio = os.environ["TORCH_FL_MPS_LOW_WATERMARK_RATIO"]
# Both knobs are read when the MPS allocator is created, so set them before
# the import. The explicit test below uses a fresh process to change them.
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = high_ratio
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = low_ratio

import torch  # noqa: E402

# The un-suffixed PyPI build, not the `+cpu` Linux wheel: torch.version.cuda is
# None and the version carries no local tag.
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"Machine: {platform.machine()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Recommended MPS memory: {torch.mps.recommended_max_memory()} bytes")
print(f"MPS high watermark: {high_ratio}")
print(f"MPS low watermark: {low_ratio}")
print(f"METAL_DEVICE_WRAPPER_TYPE: {os.environ.get('METAL_DEVICE_WRAPPER_TYPE')}")

# Report allocator state in the same process as a failed allocation.
try:
    tensor = torch.arange(12, dtype=torch.float32, device="mps")
    tensor_cpu = tensor.cpu()
except RuntimeError:
    print("MPS probe failed:", file=sys.stderr)
    print(
        f"  driver allocated: {torch.mps.driver_allocated_memory()} bytes",
        file=sys.stderr,
    )
    print(
        f"  current allocated: {torch.mps.current_allocated_memory()} bytes",
        file=sys.stderr,
    )
    raise
torch.mps.synchronize()

assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert torch.backends.mps.is_built(), "this torch build has no MPS support"
print("MPS allocation: ok")
PY

if command -v system_profiler >/dev/null 2>&1; then
  system_profiler SPDisplaysDataType 2>/dev/null | head -20 || true
fi

if [[ "$CI_STAGE" == "build" ]]; then
  # Prebuild before the integration job stages the wheel, as the CUDA lane does.
  build_flagos_inplace
fi

# --- Export to later workflow steps ------------------------------------------
if [[ -n "${GITHUB_ENV:-}" ]]; then
  export_ci_env VIRTUAL_ENV PATH PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR FLAGOS_BUILD_VENDOR FLAGOS_DISABLE_CUDA_ASSETS METAL_DEVICE_WRAPPER_TYPE TORCH_FL_MPS_HIGH_WATERMARK_RATIO TORCH_FL_MPS_LOW_WATERMARK_RATIO ARCHFLAGS _PYTHON_HOST_PLATFORM CMAKE_OSX_ARCHITECTURES
fi
