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


# Cache ordinary and CPU Torch downloads through the shared CI policy.
export PIP_RETRY_TIMEOUT=600
export BASH_ENV=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
CPU_TORCH_VERSION="${CPU_TORCH_VERSION%+cpu}"
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"
# FlagTree provides Triton support. The source-free wheel pairs with Triton
# 3.6 and is published as cp312 only, so the isolated test environment
# below has to run on Python 3.12.
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_nvidia}"
FLAGTREE_PYTHON_VERSION="${TORCH_FL_FLAGTREE_PYTHON_VERSION:-$FLAGTREE_PYTHON_VERSION_cuda}"
FLAGTREE_MIN_GLIBC="${TORCH_FL_FLAGTREE_MIN_GLIBC:-$FLAGTREE_MIN_GLIBC_DEFAULT}"
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-nvidia/simple}"

# An image may supply a matching CUDA PyTorch and FlagGems C++ library.
VENDOR_MODE="${TORCH_FL_CUDA_VENDOR_MODE:-auto}"

find_image_vendor_python() {
  local candidate="${TORCH_FL_VENDOR_PYTHON:-}"
  if [[ -n "$candidate" && "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$(command -v python3 2>/dev/null || true)"
  fi
  if [[ -n "$candidate" && -x "$candidate" ]] \
    && "$candidate" - "$CPU_TORCH_VERSION" <<'PY' >/dev/null 2>&1
import sys
import torch

assert torch.__version__.split("+", 1)[0] == sys.argv[1]
assert torch.version.cuda == "13.0"
PY
  then
    printf '%s' "$candidate"
  fi
}

python_matches_version() {
  "$1" - "$FLAGTREE_PYTHON_VERSION" <<'PY'
import sys

expected = tuple(int(part) for part in sys.argv[1].split("."))
raise SystemExit(0 if sys.version_info[:2] == expected else 1)
PY
}

select_test_python() {
  local candidate
  for candidate in \
    "${TORCH_FL_TEST_PYTHON:-}" \
    "python$FLAGTREE_PYTHON_VERSION" \
    "/usr/bin/python$FLAGTREE_PYTHON_VERSION"; do
    [[ -n "$candidate" ]] || continue
    if [[ "$candidate" != */* ]]; then
      candidate="$(command -v "$candidate" 2>/dev/null || true)"
      [[ -n "$candidate" ]] || continue
    fi
    if [[ -x "$candidate" ]] && python_matches_version "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

python_has_headers() {
  "$1" -c 'import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' \
    >/dev/null 2>&1
}

resolve_vendor_torch() {
  if [[ ! -x "$VENDOR_PYTHON" ]]; then
    echo "::error::The accelerator interpreter resolved to '$VENDOR_PYTHON', which is not an executable file"
    exit 1
  fi
  VENDOR_INFO="$("$VENDOR_PYTHON" - <<'PY'
import json
from pathlib import Path
import site
import sys

import torch

print(json.dumps({
    "version": torch.__version__,
    "base_version": torch.__version__.split("+", 1)[0],
    "cuda": torch.version.cuda,
    "root": str(Path(torch.__file__).resolve().parent),
    "site": site.getsitepackages()[0],
    "python": sys.version.split()[0],
}, sort_keys=True))
PY
)"

  readarray -t VENDOR_FIELDS < <(
    VENDOR_INFO="$VENDOR_INFO" "$VENDOR_PYTHON" - <<'PY'
import json
import os

info = json.loads(os.environ["VENDOR_INFO"])
for key in ("version", "base_version", "cuda", "root", "site", "python"):
    print(info.get(key) or "")
PY
  )
  VENDOR_TORCH_VERSION="${VENDOR_FIELDS[0]}"
  VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[1]}"
  VENDOR_CUDA_VERSION="${VENDOR_FIELDS[2]}"
  VENDOR_TORCH_ROOT="${VENDOR_FIELDS[3]}"
  VENDOR_SITE="${VENDOR_FIELDS[4]}"
  VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[5]}"
  VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

  for nvidia_lib in "$VENDOR_SITE"/nvidia/*/lib; do
    [[ -d "$nvidia_lib" ]] || continue
    VENDOR_NVIDIA_LIBS="${VENDOR_NVIDIA_LIBS:+$VENDOR_NVIDIA_LIBS:}$nvidia_lib"
  done

  if [[ "$VENDOR_TORCH_BASE_VERSION" != "$CPU_TORCH_VERSION" ]]; then
    echo "::error::Vendor torch is $VENDOR_TORCH_VERSION; expected $CPU_TORCH_VERSION"
    exit 1
  fi
  if [[ "$VENDOR_CUDA_VERSION" != "13.0" ]]; then
    echo "::error::Vendor torch CUDA runtime is $VENDOR_CUDA_VERSION; expected 13.0"
    exit 1
  fi

  echo "Vendor Python: $VENDOR_PYTHON ($VENDOR_PYTHON_VERSION)"
  echo "Vendor PyTorch: $VENDOR_TORCH_VERSION"
  echo "Vendor torch root: $VENDOR_TORCH_ROOT"
}


VENDOR_PYTHON="$(find_image_vendor_python)"
case "$VENDOR_MODE" in
  image)
    if [[ -z "$VENDOR_PYTHON" ]]; then
      echo "::error::This image provides no accelerator PyTorch and TORCH_FL_CUDA_VENDOR_MODE=image"
      exit 1
    fi
    ;;
  bootstrap)
    echo "::error::TORCH_FL_CUDA_VENDOR_MODE=bootstrap has been removed. Use auto or an image with the vendor libraries."
    exit 1
    ;;
  auto) ;;
  *)
    echo "::error::TORCH_FL_CUDA_VENDOR_MODE must be auto or image (got '$VENDOR_MODE')"
    exit 1
    ;;
esac
if [[ -n "$VENDOR_PYTHON" ]]; then
  VENDOR_SOURCE="image"
else
  # The CPU test wheel does not require an accelerator Torch install.
  VENDOR_SOURCE="none"
fi
echo "Accelerator PyTorch source: $VENDOR_SOURCE ($VENDOR_PYTHON)"
# The optional image supplies FlagGems C++ and CUDA runtime libraries.
VENDOR_TORCH_VERSION=""
VENDOR_TORCH_BASE_VERSION=""
VENDOR_CUDA_VERSION=""
VENDOR_TORCH_ROOT=""
VENDOR_SITE=""
VENDOR_PYTHON_VERSION=""
VENDOR_TORCH_LIB=""
VENDOR_NVIDIA_LIBS=""
if [[ "$VENDOR_SOURCE" != "none" ]]; then
  resolve_vendor_torch
fi

# A matching prebuilt image may carry the FlagGems C++ operators. The only
# published CUDA native wheel has a libtriton_jit.so built against a different
# PyTorch C++ ABI and cannot be loaded with this job's torch 2.10.0, so the
# bootstrapped path uses the Python FlagGems wheel without the C++ route.
VENDOR_FLAGGEMS_DIR=""
VENDOR_FLAGGEMS_LIB=""
if [[ "$VENDOR_SOURCE" == "image" ]]; then
  export FLAGOS_BUILD_FLAGGEMS_CPP=1
  VENDOR_FLAGGEMS_DIR="$("$VENDOR_PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("flag_gems")
if spec is None or not spec.submodule_search_locations:
    print("")
else:
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    candidate = root / "lib" / "cmake" / "FlagGems"
    print(candidate if (candidate / "FlagGemsConfig.cmake").is_file() else "")
PY
)"
  if [[ -z "$VENDOR_FLAGGEMS_DIR" ]]; then
    echo "::error::FlagGems C++ package was not found in the vendor image"
    exit 1
  fi
  VENDOR_FLAGGEMS_LIB="$(cd "$VENDOR_FLAGGEMS_DIR/../.." && pwd)"
  if ! compgen -G "$VENDOR_FLAGGEMS_LIB/liboperators.so*" >/dev/null; then
    echo "::error::FlagGems liboperators.so was not found under $VENDOR_FLAGGEMS_LIB"
    exit 1
  fi
else
  export FLAGOS_BUILD_FLAGGEMS_CPP=0
fi

# Sets TEST_PYTHON, after installing the interpreter through uv or the distribution.
install_test_python() {
  if command -v uv >/dev/null 2>&1; then
    echo "Installing Python $FLAGTREE_PYTHON_VERSION with uv..."
    if uv python install "$FLAGTREE_PYTHON_VERSION"; then
      local installed
      installed="$(uv python find "$FLAGTREE_PYTHON_VERSION" 2>/dev/null || true)"
      if [[ -n "$installed" && -x "$installed" ]] && python_matches_version "$installed"; then
        TEST_PYTHON="$installed"
        return 0
      fi
    fi
  fi
  echo "Installing Python $FLAGTREE_PYTHON_VERSION from the deadsnakes PPA..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
  apt-get install -y -qq "python$FLAGTREE_PYTHON_VERSION" \
    "python$FLAGTREE_PYTHON_VERSION-venv" "python$FLAGTREE_PYTHON_VERSION-dev"
  TEST_PYTHON="$(command -v "python$FLAGTREE_PYTHON_VERSION")"
}

check_image_requirement() {
  local requirement="$1"
  if ! "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --check-installed "$requirement"; then
    echo "::error::The NVIDIA CI image venv is missing a compatible $requirement"
    return 1
  fi
}

IMAGE_VENV_ROOT=/opt/torch-fl-ci-venv
NVIDIA_REQUIREMENTS="$REPO_ROOT/.github/requirements-nvidia.txt"
VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-cuda-${CI_STAGE}}"
USE_IMAGE_VENV=0
# Explicit provisioning overrides still use an independent job venv.
if [[ ( -d "$IMAGE_VENV_ROOT" || "${VIRTUAL_ENV:-}" == "$IMAGE_VENV_ROOT" ) &&
      "$VENDOR_SOURCE" == none &&
      -z "${TORCH_FL_VENV_ROOT:-}${TORCH_FL_TEST_PYTHON:-}${TORCH_FL_CPU_TORCH_WHEELHOUSE:-}" &&
      -z "${TORCH_FL_CPU_TORCH_VERSION:-}${TORCH_FL_FLAGTREE_VERSION:-}${TORCH_FL_FLAGTREE_PYTHON_VERSION:-}${TORCH_FL_FLAGGEMS_VERSION:-}" ]]; then
  USE_IMAGE_VENV=1
  VENV_ROOT="$IMAGE_VENV_ROOT"
  VENV_PYTHON="$VENV_ROOT/bin/python"
  if ! python_matches_version "$VENV_PYTHON" || ! python_has_headers "$VENV_PYTHON" || ! runtime_venv_is_isolated flagcx; then
    echo "::error::The NVIDIA CI image requires an isolated Python $FLAGTREE_PYTHON_VERSION venv with development headers at $VENV_ROOT"
    exit 1
  fi
  echo "Using the NVIDIA CI image venv: $VENV_ROOT"
else
  if [[ -d "$IMAGE_VENV_ROOT" ]]; then
    if [[ "$VENV_ROOT" -ef "$IMAGE_VENV_ROOT" ]]; then
      echo "::error::A fresh NVIDIA job venv must be separate from $IMAGE_VENV_ROOT"
      exit 1
    fi
    # A job-specific Torch must also supply its own CMake files and libraries.
    IMAGE_TORCH_ROOT="$IMAGE_VENV_ROOT/lib/python$FLAGTREE_PYTHON_VERSION_cuda/site-packages/torch"
    export CMAKE_PREFIX_PATH="$(VENDOR_TORCH_ROOT="$IMAGE_TORCH_ROOT" strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
    export LD_LIBRARY_PATH="$(VENDOR_TORCH_ROOT="$IMAGE_TORCH_ROOT" strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
  fi
  TEST_PYTHON="$(select_test_python || true)"
  if [[ -z "$TEST_PYTHON" ]]; then
    if ! install_test_python; then
      echo "::error::Python $FLAGTREE_PYTHON_VERSION is required by FlagTree $FLAGTREE_VERSION"
      exit 1
    fi
  fi
  echo "Test Python: $TEST_PYTHON ($("$TEST_PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

  create_runtime_venv "$TEST_PYTHON" "" "$VENDOR_TORCH_ROOT"
  # Copy only compatible RECORD-owned distributions. CUDA Torch from the image
  # is a device-library source, never a replacement for the CPU test frontend.
  "${VENDOR_PYTHON:-$TEST_PYTHON}" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --target-python "$VENV_PYTHON" \
    --torch-version "$CPU_TORCH_VERSION" \
    --flagtree-version "$FLAGTREE_VERSION" \
    --flaggems-version "$FLAGGEMS_VERSION" \
    --requirements-file "$NVIDIA_REQUIREMENTS"
fi
export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
if (( USE_IMAGE_VENV )); then
  while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    [[ -z "$requirement" || "$requirement" == \#* ]] && continue
    check_image_requirement "$requirement"
  done < "$NVIDIA_REQUIREMENTS"
else
  ensure_runtime_requirements "$NVIDIA_REQUIREMENTS" "$PIP_INDEX_URL_ARG"
fi
validate_runtime_build_tools
if ! runtime_cpu_torch_is_usable; then
  if (( USE_IMAGE_VENV )); then
    echo "::error::The NVIDIA CI image venv requires CPU PyTorch $CPU_TORCH_VERSION+cpu"
    exit 1
  fi
  repair_runtime_cpu_torch
fi

# FlagTree ships its own Triton provider; nothing else in the environment may
# shadow it, so it is installed last with --no-deps.
#
# The source-free NVIDIA FlagTree wheel links its bundled libtriton.so against
# the glibc symbol versions of the distribution it was built on, and the current
# wheels require GLIBC_2.38. An image older than that installs the wheel
# successfully and then fails while importing triton, reporting the missing
# symbol from a C extension instead of the actual image requirement. Check the
# image up front so the failure names the requirement and its fix.
detect_glibc_version() {
  local version=""
  if command -v getconf >/dev/null 2>&1; then
    version="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $NF}')"
  fi
  if [[ -z "$version" ]] && command -v ldd >/dev/null 2>&1; then
    version="$(ldd --version 2>/dev/null | head -n 1 | awk '{print $NF}')"
  fi
  printf '%s' "${version%%[^0-9.]*}"
}

IMAGE_GLIBC="$(detect_glibc_version)"
if [[ -z "$IMAGE_GLIBC" ]]; then
  echo "::error::Unable to determine the glibc version of this image"
  exit 1
fi
if [[ "$(printf '%s\n%s\n' "$FLAGTREE_MIN_GLIBC" "$IMAGE_GLIBC" | sort -V | head -n 1)" != "$FLAGTREE_MIN_GLIBC" ]]; then
  echo "::error::FlagTree $FLAGTREE_VERSION requires glibc >= $FLAGTREE_MIN_GLIBC; this image provides $IMAGE_GLIBC"
  exit 1
fi
echo "Image glibc: $IMAGE_GLIBC (FlagTree requires >= $FLAGTREE_MIN_GLIBC)"

if ! runtime_flagtree_is_usable nvidia; then
  if (( USE_IMAGE_VENV )); then
    echo "::error::The NVIDIA CI image venv requires FlagTree $FLAGTREE_VERSION with its NVIDIA Triton backend"
    exit 1
  fi
  "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
  install_flagtree
fi

# FlagCX is deferred across CI. FlagTree can find a system libflagcx.so
# even without the Python package, so disable its optional distributed loader.
export USE_TLE_DIST=0
echo "::notice::CI scope: FlagCX communication is deferred"

# Install the published FlagGems wheel after FlagTree. --no-deps keeps the
# CPU-only torch ABI intact; its non-torch dependencies use the common policy.
if (( USE_IMAGE_VENV )); then
  if ! flag_gems_installed; then
    echo "::error::The NVIDIA CI image venv requires FlagGems $FLAGGEMS_VERSION"
    exit 1
  fi
else
  install_flag_gems
fi

if [[ "$CI_STAGE" == "integration" ]]; then
  # The requirements file already provides pytest and Ninja for host-program
  # tests. Reuse Transformers only when it matches the upstream Qwen3 pin.
  ensure_runtime_requirement "transformers==$QWEN3_TRANSFORMERS_VERSION_DEFAULT" "$PIP_INDEX_URL_ARG"
fi

"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

CPU_TORCH_ROOT="$("$VENV_PYTHON" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
PY
)"

export FLAGOS_ACCELERATOR=cuda
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export FLAGGEMS_DIR="$VENDOR_FLAGGEMS_DIR"
export FLAGTREE_VERSION
export GEMS_VENDOR="${GEMS_VENDOR:-nvidia}"
# Keep the CUDA FlagTune workaround until the new FlagTree wheel has passed an
# on-device autotune run. The old wheel lacked the manifest machinery; without
# this override, every FlagGems-routed kernel whose autotuner takes the
# Cost Model path dies loading the model bundle: the old wheel has no
# FLAGTUNE_MANIFEST_URL default and the runner provides none, so
# model_sources raises ManifestFetchError (TestAutograd goes red on it).
# USE_FLAGTUNE=0 forces TuningMode.DEFAULT in flag_gems.runtime.resolve_
# tuning_mode, restoring the legacy default autotune policy -- the behavior
# this CI was green with before FlagTune. Must read exactly 0/1.
# CUDA-only exception: the other six wheels resolve the probe (their jobs are
# green), so they keep tuning coverage; do NOT copy this export elsewhere.
export USE_FLAGTUNE=0

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake${VENDOR_FLAGGEMS_DIR:+:$VENDOR_FLAGGEMS_DIR}${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/lib64${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
# Load the CPU wheel's core. Putting the accelerator torch's entire lib/ on
# this path can load its libtorch_cpu first and duplicate dispatcher registrations.
export LD_LIBRARY_PATH="${VENDOR_NVIDIA_LIBS:+$VENDOR_NVIDIA_LIBS:}$CPU_TORCH_ROOT/lib${VENDOR_FLAGGEMS_LIB:+:$VENDOR_FLAGGEMS_LIB}:$CUDA_HOME/lib64${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
# Verify that the source-free wheel installed the Triton provider it ships,
# rather than a stock Triton left over in the environment.
python - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import triton

if os.environ.get("USE_TLE_DIST") == "0":
    # Check the import that FlagGems triggers before spending time on build_ext.
    from triton.experimental.tle import language  # noqa: F401

expected = os.environ["FLAGTREE_VERSION"]
flagtree_version = importlib.metadata.version("flagtree")
assert flagtree_version == expected, (flagtree_version, expected)

distribution = importlib.metadata.distribution("flagtree")
provided = {str(entry).split("/", 1)[0] for entry in distribution.files or ()}
assert "triton" in provided, sorted(provided)

triton_root = Path(triton.__file__).resolve().parent
assert triton_root == Path(distribution.locate_file("triton")).resolve(), triton_root
print(f"FlagTree: {flagtree_version}")
print(f"Triton module: {triton_root}")
print(f"Triton version: {triton.__version__}")
PY

if (( ! USE_IMAGE_VENV )); then
  build_flagos_inplace
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - "$USE_IMAGE_VENV" <<'PY'
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path

print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
if sys.argv[1] == "1":
    # The wheel preflight validates the build record after python -m build.
    assert torch_path.is_relative_to(Path(sys.prefix).resolve()), torch_path
else:
    # Check the legacy prebuild without importing or activating Torch-FL.
    record = Path("torch_fl/_build_config.py")
    assert record.is_file(), record
    loader = importlib.machinery.SourceFileLoader("_flagos_build_config", str(record))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    assert "boxing" not in tuple(config.KERNELS), config.KERNELS
    assert "flaggems" in tuple(config.KERNELS), config.KERNELS
    print(f"Kernel sets: {tuple(config.KERNELS)}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  # Later steps resolve `bash` and `python` through a PATH the runner assembles
  # from these lines and from the image config:
  #   PATH=<lines, reversed, ':'-joined>:<PATH reported by `docker inspect`>
  # (actions/runner, Handlers/StepHost.cs). Publish the full PATH since older
  # job images may not declare one and later steps still need /usr/bin and /bin.
  printf '%s\n' "$PATH" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then

  : "${FLAGOS_BACKEND_CONFIG:=}"
  : "${FLAGOS_CUDA_ASSETS_DIR:=}"
  : "${FLAGOS_SKIP_CUDA_ASSETS:=}"
  : "${FLAGOS_DISABLE_CUDA_ASSETS:=}"
  : "${FLAGOS_VENDOR_TORCH_LIB:=}"
  : "${TORCH_FL_VENDOR_TORCH_VERSION:=}"
  export_ci_env BASH_ENV PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR CUDA_HOME CUDA_PATH FLAGGEMS_DIR FLAGTREE_VERSION GEMS_VENDOR FLAGOS_BUILD_FLAGGEMS_CPP USE_FLAGTUNE CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH FLAGOS_BACKEND_CONFIG FLAGOS_CUDA_ASSETS_DIR FLAGOS_SKIP_CUDA_ASSETS FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_VENDOR_TORCH_LIB TORCH_FL_VENDOR_TORCH_VERSION
  if [[ -n "${TORCH_DEVICE_BACKEND_AUTOLOAD:-}" ]]; then
    printf 'TORCH_DEVICE_BACKEND_AUTOLOAD=%s\n' "$TORCH_DEVICE_BACKEND_AUTOLOAD" >> "$GITHUB_ENV"
  fi
  if [[ -n "${USE_TLE_DIST:-}" ]]; then
    printf 'USE_TLE_DIST=%s\n' "$USE_TLE_DIST" >> "$GITHUB_ENV"
  fi
fi
