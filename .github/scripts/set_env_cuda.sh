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

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-2.10.0}"
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
# FlagTree provides Triton support. The source-free 0.6.2a2 wheel pairs with
# Triton 3.6 and is published as cp312 only, so the isolated test environment
# below has to run on Python 3.12. The accelerator interpreter resolved further
# down must be the same interpreter: the FlagGems C++ extension it compiles is
# imported by the test environment.
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.6.2a2}"
FLAGTREE_PYTHON_VERSION="${TORCH_FL_FLAGTREE_PYTHON_VERSION:-3.12}"
FLAGTREE_MIN_GLIBC="${TORCH_FL_FLAGTREE_MIN_GLIBC:-2.38}"
# FlagGems currently uses master as its default branch; the repository has no
# main branch. Keep this overrideable so a tested revision can be pinned by CI.
FLAGGEMS_REPOSITORY="${TORCH_FL_FLAGGEMS_REPOSITORY:-https://github.com/flagos-ai/FlagGems.git}"
# TEMPORARY PIN -- revert the default to `master` once upstream fixes
# flagos-ai/FlagGems: d312aa02 (2026-09-16) added
# ("argsort.stable", argsort_stable) to the module-level _FULL_CONFIG in
# flag_gems/__init__.py, but argsort_stable is only defined by the kunlunxin
# backend package, so `import flag_gems` raises
# NameError: name 'argsort_stable' is not defined on every other vendor.
# 437ba393 is the last good master (d312aa02's parent).
FLAGGEMS_REVISION="${TORCH_FL_FLAGGEMS_REVISION:-437ba39387ddc681dc884259ef9dbf0c1802bccc}"
# Ninja parallelism for the in-job FlagGems C++ build. The CUDA translation
# units are the bulk of it, so it is worth leaving headroom on a shared runner.
FLAGGEMS_CPP_JOBS="${TORCH_FL_FLAGGEMS_CPP_JOBS:-$(nproc 2>/dev/null || echo 4)}"

# Where the accelerator-side PyTorch comes from. A FlagOS-built image already
# carries one (`image`); the stock NVIDIA CUDA image does not, so `bootstrap`
# installs the matching CUDA build into a job-local interpreter and the pieces
# derived from it — the staged CUDA assets and the FlagGems C++ operators — are
# produced in the job instead of being copied out of the image. `auto` picks
# whichever applies.
VENDOR_MODE="${TORCH_FL_CUDA_VENDOR_MODE:-auto}"
VENDOR_TORCH_INDEX_URL="${TORCH_FL_CUDA_VENDOR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

pip_retry() {
  local python_exe="$1"
  shift
  local attempt=1
  while true; do
    if "$python_exe" -m pip install --retries 10 --timeout 600 "$@"; then
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

find_image_vendor_python() {
  local candidate="${TORCH_FL_VENDOR_PYTHON:-}"
  if [[ -n "$candidate" && "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$(command -v python3 2>/dev/null || true)"
  fi
  if [[ -n "$candidate" && -x "$candidate" ]] \
    && "$candidate" -c "import torch" >/dev/null 2>&1; then
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

# A stock NVIDIA CUDA image is a compiler image rather than an application
# runtime: it ships nvcc and the CUDA libraries but neither Python nor git,
# while everything on the bootstrap path below is cloned with git and installed
# with pip. Only that path needs this — a FlagOS-built image already carries
# both, and calls this function nowhere.
ensure_system_prerequisites() {
  local python_exe="python$FLAGTREE_PYTHON_VERSION"
  local python_path
  local attempt
  local -a packages=()

  command -v git >/dev/null 2>&1 || packages+=(git)

  python_path="$(command -v "$python_exe" 2>/dev/null || true)"
  if [[ -z "$python_path" ]]; then
    packages+=("$python_exe" "$python_exe-venv" "$python_exe-dev")
  else
    # Debian splits two pieces out of the interpreter package: `python -m venv`
    # needs ensurepip from the -venv half, and compiling the extension modules
    # below needs Python.h from the -dev half.
    "$python_path" -c 'import ensurepip' >/dev/null 2>&1 || packages+=("$python_exe-venv")
    python_has_headers "$python_path" || packages+=("$python_exe-dev")
  fi

  if (( ${#packages[@]} == 0 )); then
    return 0
  fi

  echo "Installing system packages required by this image's bootstrap path: ${packages[*]}"
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "::error::This image provides no ${packages[*]} and no apt-get to install them; use an image that ships Python $FLAGTREE_PYTHON_VERSION with its development files, or prebuild the image"
    exit 1
  fi
  export DEBIAN_FRONTEND=noninteractive
  for attempt in 1 2 3; do
    if apt-get update -qq && apt-get install -y -qq --no-install-recommends "${packages[@]}"; then
      return 0
    fi
    echo "::warning::apt-get attempt $attempt failed; retrying: ${packages[*]}"
    sleep 10
  done
  echo "::error::Could not install ${packages[*]}; this image has to reach the distribution package archive"
  exit 1
}

# Sets VENDOR_PYTHON. The interpreter is returned through that global rather than
# on stdout: everything this function runs — the venv bootstrap, pip, the CUDA
# wheel download — writes to stdout, and a command substitution would fold all of
# it into the interpreter path (which fails as `File name too long` on the first
# exec). Progress therefore stays visible in the job log and cannot reach the
# variable.
bootstrap_vendor_python() {
  # The FlagGems C++ operators are compiled by this interpreter and imported by
  # the test environment, so both have to run the same Python minor version or
  # the extension module ABI will not match: resolve it exactly the way the test
  # environment does.
  local base_python="${TORCH_FL_VENDOR_PYTHON:-}"
  if [[ -n "$base_python" && "$base_python" != */* ]]; then
    base_python="$(command -v "$base_python" 2>/dev/null || true)"
  fi
  if [[ -z "$base_python" ]]; then
    base_python="$(select_test_python || true)"
  fi
  if [[ -z "$base_python" ]]; then
    echo "::error::This CUDA image has no accelerator PyTorch and no Python $FLAGTREE_PYTHON_VERSION to bootstrap one with; use an image that provides it or prebuild the image"
    exit 1
  fi
  local base_version
  base_version="$("$base_python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [[ "$base_version" != "$FLAGTREE_PYTHON_VERSION" ]]; then
    echo "::error::The bootstrapped accelerator PyTorch must run the Python that executes the tests ($FLAGTREE_PYTHON_VERSION), but $base_python is $base_version; use an image whose Python is $FLAGTREE_PYTHON_VERSION or prebuild the image"
    exit 1
  fi
  local vendor_venv="${TORCH_FL_VENDOR_VENV:-${RUNNER_TEMP:-/tmp}/torch-fl-cuda-vendor}"
  echo "Installing CUDA PyTorch $CPU_TORCH_VERSION into $vendor_venv"
  rm -rf "$vendor_venv"
  "$base_python" -m venv "$vendor_venv"
  local vendor_python="$vendor_venv/bin/python"
  "$vendor_python" -m pip install --upgrade pip
  pip_retry "$vendor_python" --index-url "$VENDOR_TORCH_INDEX_URL" \
    "torch==$CPU_TORCH_VERSION"
  VENDOR_PYTHON="$vendor_python"
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
    VENDOR_PYTHON=""
    ;;
  auto) ;;
  *)
    echo "::error::TORCH_FL_CUDA_VENDOR_MODE must be auto, image or bootstrap (got '$VENDOR_MODE')"
    exit 1
    ;;
esac
if [[ -n "$VENDOR_PYTHON" ]]; then
  VENDOR_SOURCE="image"
else
  VENDOR_SOURCE="bootstrap"
  ensure_system_prerequisites
  bootstrap_vendor_python
fi
echo "Accelerator PyTorch source: $VENDOR_SOURCE ($VENDOR_PYTHON)"
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

VENDOR_NVIDIA_LIBS=""
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

# A prebuilt image carries the FlagGems C++ operators next to flag_gems. A
# bootstrapped environment has no such copy: the operators are built from the
# same FlagGems revision this job installs as a Python package, once that
# revision is known further down.
VENDOR_FLAGGEMS_DIR=""
VENDOR_FLAGGEMS_LIB=""
if [[ "$VENDOR_SOURCE" == "image" ]]; then
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
fi

# Copy only accelerator-side PyTorch libraries. libc10.so, libtorch.so,
# libtorch_cpu.so and libtorch_python.so deliberately come from the upstream
# CPU wheel installed below. setup.py copies these assets into torch_fl/lib,
# and torch_fl preloads them before importing torch.
CUDA_ASSETS_DIR="$REPO_ROOT/.libtorch_cuda_assets"
rm -rf "$CUDA_ASSETS_DIR"
mkdir -p "$CUDA_ASSETS_DIR"
shopt -s nullglob
for pattern in \
  'libc10_cuda.so*' \
  'libtorch_cuda.so*' \
  'libtorch_cuda_linalg.so*' \
  'libtorch_nvshmem.so*' \
  'libcaffe2_nvrtc.so*'; do
  for source in "$VENDOR_TORCH_LIB"/$pattern; do
    cp -a "$source" "$CUDA_ASSETS_DIR/"
  done
done
shopt -u nullglob

# The CUDA dispatcher library is the only mandatory accelerator asset. Some
# vendor Torch layouts (including cu130 images) do not ship a standalone
# libc10_cuda.so; torch_fl treats that library as optional and loads it when
# present. Keep the check layout-agnostic instead of requiring a fixed set of
# files on every CUDA image.
if [[ ! -e "$CUDA_ASSETS_DIR/libtorch_cuda.so" ]]; then
  echo "::error::Required CUDA asset was not found: $VENDOR_TORCH_LIB/libtorch_cuda.so"
  exit 1
fi
if [[ ! -e "$CUDA_ASSETS_DIR/libc10_cuda.so" ]]; then
  echo "::notice::Optional CUDA asset not present: $VENDOR_TORCH_LIB/libc10_cuda.so"
fi
echo "CUDA assets staged: $(find "$CUDA_ASSETS_DIR" -maxdepth 1 -type f -name '*.so*' | wc -l)"

# The isolated environment runs on the Python version FlagTree is built for, so
# `python` inside it is the same interpreter that owns the Triton provider. The
# accelerator interpreter above runs the same minor version (a bootstrapped one
# is required to), so both can compile and import the same extension modules.
VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-cuda-${CI_STAGE}}"

# Sets TEST_PYTHON, for the same reason bootstrap_vendor_python sets
# VENDOR_PYTHON: the uv and apt-get branches below both write to stdout, and
# capturing them would return a transcript instead of an interpreter path.
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

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-cuda-${CI_STAGE}}"
TEST_PYTHON="$(select_test_python || true)"
if [[ -z "$TEST_PYTHON" ]]; then
  if ! install_test_python; then
    echo "::error::Python $FLAGTREE_PYTHON_VERSION is required by FlagTree $FLAGTREE_VERSION"
    exit 1
  fi
fi
echo "Test Python: $TEST_PYTHON ($("$TEST_PYTHON" -c 'import sys; print(sys.version.split()[0])'))"

if ! "$TEST_PYTHON" -m venv --clear "$VENV_ROOT"; then
  echo "::warning::$TEST_PYTHON cannot create a venv; trying uv"
  if ! command -v uv >/dev/null 2>&1; then
    "$TEST_PYTHON" -m pip install --upgrade uv
  fi
  uv venv --clear --seed --python "$TEST_PYTHON" "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip "setuptools>=64,<77" "setuptools-scm>=8,<10" "wheel==0.46.2" cmake build
"$VENV_PYTHON" -m pip install \
  --index-url "$CPU_TORCH_INDEX_URL" \
  "torch==$CPU_TORCH_VERSION"

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

pip_retry "$VENV_PYTHON" --no-deps --index-url "$FLAGTREE_INDEX_URL" \
  "flagtree===${FLAGTREE_VERSION}"

# Keep only vendor packages that are not provided by FlagTree. In particular,
# do not copy vendor `triton` or `triton_kernels`: either would contaminate the
# source-free FlagTree installation with a potentially incompatible Triton.
VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
FLAGCX_COPIED=0
for package in flagcx; do
  if [[ -d "$VENDOR_SITE/$package" ]]; then
    cp -a "$VENDOR_SITE/$package" "$VENV_SITE/"
    FLAGCX_COPIED=1
  fi
  for metadata in "$VENDOR_SITE"/"$package"-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    cp -a "$metadata" "$VENV_SITE/"
  done
done
# FlagCX backs the distributed paths only (`torch_fl.comm` imports it lazily),
# and no test in the CUDA manifest builds a process group. A bootstrapped
# environment therefore runs without it instead of paying for a source build;
# `tests/manual/` covers the FlagCX paths where it is present.
if [[ "$FLAGCX_COPIED" == "0" ]]; then
  echo "::notice::flagcx is not installed in this environment; the distributed backend is unavailable"
fi

# Install the current FlagGems source after FlagTree. FlagGems has no `main`
# branch at present; `master` is its default branch and can be overridden with
# TORCH_FL_FLAGGEMS_REVISION for reproducible CI experiments. --no-deps keeps
# the CPU-only torch ABI intact; install its non-torch dependencies explicitly.
pip_retry "$VENV_PYTHON" packaging 'PyYAML==6.0.1' 'sqlalchemy==2.0.48' numpy
FLAGGEMS_SOURCE_ROOT="${RUNNER_TEMP:-/tmp}/flag-gems-${CI_STAGE}"
rm -rf "$FLAGGEMS_SOURCE_ROOT"
git clone --depth 1 --branch master \
  "$FLAGGEMS_REPOSITORY" "$FLAGGEMS_SOURCE_ROOT"
if [[ "$FLAGGEMS_REVISION" != "master" ]]; then
  # A raw SHA cannot be passed to `git clone --branch`; fetch it explicitly.
  # (Only the temporary pin above takes this path -- `master` stays a plain
  # shallow clone.)
  git -C "$FLAGGEMS_SOURCE_ROOT" fetch --depth 1 origin "$FLAGGEMS_REVISION"
  git -C "$FLAGGEMS_SOURCE_ROOT" checkout --detach "$FLAGGEMS_REVISION"
fi
FLAGGEMS_COMMIT="$(git -C "$FLAGGEMS_SOURCE_ROOT" rev-parse HEAD)"
echo "FlagGems source: ${FLAGGEMS_REPOSITORY}@${FLAGGEMS_REVISION} (${FLAGGEMS_COMMIT})"
pip_retry "$VENV_PYTHON" --no-deps --no-build-isolation "$FLAGGEMS_SOURCE_ROOT"

# The FlagGems C++ operators are the native half of FlagGems support: a prebuilt
# FlagOS image carries them next to flag_gems, a bootstrapped environment builds
# them from the revision just installed. That revision matters — the native
# package drops its modules into the flag_gems namespace — so this runs after
# the Python package above, not before it.
if [[ "$VENDOR_SOURCE" == "bootstrap" ]]; then
  echo "Building the FlagGems C++ operators from $FLAGGEMS_COMMIT"
  # The operators are compiled against c10/cuda, and only a CUDA-enabled torch
  # ships those headers: the CUDA wheel generates
  # `c10/cuda/impl/cuda_cmake_macros.h`, the CPU wheel does not. This is why the
  # vendor interpreter — the one that owns the accelerator torch — runs the
  # build, exactly as the FlagOS image that carries a prebuilt copy does. The
  # wheel it produces is installed into the isolated environment afterwards,
  # which keeps running the CPU wheel plus the staged CUDA assets.
  #
  # scikit-build-core drives the build with --no-build-isolation, which means
  # pip does not resolve cpp/pyproject.toml's build requirements: every one of
  # them has to be in this environment already. `setuptools-scm` in particular is
  # needed at metadata time, because that project takes its version from the
  # parent repository through that provider, and cmake is needed because a CUDA
  # development image ships a compiler, not a build system. The versions mirror
  # what the isolated environment installs so both halves of the build agree.
  pip_retry "$VENDOR_PYTHON" \
    "setuptools>=64,<77" "setuptools-scm>=8,<10" cmake \
    "scikit-build-core==0.12.2" "pybind11==3.0.3" "ninja==1.13.0"
  # Configuring the cpp package probes `import triton` in the building
  # interpreter and aborts without it, so this environment carries the same
  # Triton provider the isolated one runs on.
  pip_retry "$VENDOR_PYTHON" --no-deps --index-url "$FLAGTREE_INDEX_URL" \
    "flagtree===$FLAGTREE_VERSION"
  # PEP 621 requires a static project name, so the per-vendor suffix is injected
  # into cpp/pyproject.toml before building (flag-gems-cpp-cuda).
  (cd "$FLAGGEMS_SOURCE_ROOT" && bash tools/set_cpp_vendor.sh cuda)
  # The native package takes its version from the parent repository's tags, and
  # this clone is shallow and untagged; pin the metadata to the version of the
  # Python package installed from the same commit so the pair stays in lockstep.
  FLAGGEMS_CPP_VERSION="$("$VENV_PYTHON" -c 'import importlib.metadata; print(importlib.metadata.version("flag_gems"))')"
  FLAGGEMS_CPP_WHEEL_DIR="${RUNNER_TEMP:-/tmp}/flag-gems-cpp-wheel-${CI_STAGE}"
  rm -rf "$FLAGGEMS_CPP_WHEEL_DIR"
  mkdir -p "$FLAGGEMS_CPP_WHEEL_DIR"
  # nvcc is not guaranteed to be on PATH in a bootstrapped environment, so name
  # it explicitly rather than relying on the image having exported it.
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  # cpp/CMakeLists.txt defaults FLAGGEMS_BUILD_CTESTS to the value of
  # FLAGGEMS_BUILD_C_EXTENSIONS, which would pull in that project's own test
  # suite: it is not installed, nothing here runs ctest over it (upstream has a
  # separate workflow for that), and it fails to compile without CUDA include
  # directories that only its own build sets up. Build what the wheel ships.
  env \
    CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BUILD_CTESTS=OFF -DFLAGGEMS_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=$CUDA_HOME" \
    CMAKE_BUILD_PARALLEL_LEVEL="$FLAGGEMS_CPP_JOBS" \
    MAX_JOBS="$FLAGGEMS_CPP_JOBS" \
    CUDACXX="$CUDA_HOME/bin/nvcc" \
    SETUPTOOLS_SCM_PRETEND_VERSION="$FLAGGEMS_CPP_VERSION" \
    "$VENDOR_PYTHON" -m pip wheel "$FLAGGEMS_SOURCE_ROOT/cpp" \
      --no-deps --no-build-isolation -w "$FLAGGEMS_CPP_WHEEL_DIR"
  # Both interpreters run Python $FLAGTREE_PYTHON_VERSION, so the native wheel
  # the vendor interpreter built installs into the isolated environment as-is.
  "$VENV_PYTHON" -m pip install --no-deps --no-build-isolation \
    "$FLAGGEMS_CPP_WHEEL_DIR"/flag_gems_cpp_cuda-*.whl
  VENDOR_FLAGGEMS_LIB="$VENV_SITE/flag_gems/lib"
  VENDOR_FLAGGEMS_DIR="$VENDOR_FLAGGEMS_LIB/cmake/FlagGems"
  if [[ ! -f "$VENDOR_FLAGGEMS_DIR/FlagGemsConfig.cmake" ]] \
    || ! compgen -G "$VENDOR_FLAGGEMS_LIB/liboperators.so*" >/dev/null; then
    echo "::error::The FlagGems C++ build produced no FlagGemsConfig.cmake / liboperators.so under $VENDOR_FLAGGEMS_LIB"
    exit 1
  fi
  echo "FlagGems C++ operators: $FLAGGEMS_CPP_VERSION ($VENDOR_FLAGGEMS_LIB)"
fi

if [[ "$CI_STAGE" == "integration" ]]; then
  pip_retry "$VENV_PYTHON" pytest transformers
fi

CPU_TORCH_ROOT="$("$VENV_PYTHON" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
PY
)"

strip_vendor_paths() {
  local value="${1:-}"
  local entry
  local -a entries=()
  local -a kept=()
  IFS=: read -ra entries <<< "$value"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      "$VENDOR_TORCH_ROOT"|"$VENDOR_TORCH_ROOT"/*) ;;
      *) kept+=("$entry") ;;
    esac
  done
  local joined=""
  for entry in "${kept[@]}"; do
    joined="${joined:+$joined:}$entry"
  done
  printf '%s' "$joined"
}

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export ACCELERATOR=cuda
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export FLAGOS_CUDA_ASSETS_DIR="$CUDA_ASSETS_DIR"
export FLAGGEMS_DIR="$VENDOR_FLAGGEMS_DIR"
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"
export FLAGTREE_VERSION
export GEMS_VENDOR="${GEMS_VENDOR:-nvidia}"

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake:$VENDOR_FLAGGEMS_DIR${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/lib64:$CUDA_ASSETS_DIR${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_ASSETS_DIR${VENDOR_NVIDIA_LIBS:+:$VENDOR_NVIDIA_LIBS}:$CPU_TORCH_ROOT/lib:$VENDOR_FLAGGEMS_LIB:$CUDA_HOME/lib64${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
# Verify that the source-free wheel installed the Triton provider it ships,
# rather than a stock Triton left over in the environment.
python - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import triton

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

if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so and the bundled CUDA assets
  # before the common workflow invokes python -m build.
  python setup.py build_ext --inplace
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - <<'PY'
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
assert Path(".libtorch_cuda_assets/libtorch_cuda.so").is_file()
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"CUDA assets: {Path('.libtorch_cuda_assets').resolve()}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  # Later steps resolve `bash` and `python` through a PATH the runner assembles
  # from these lines and from the image config:
  #   PATH=<lines, reversed, ':'-joined>:<PATH reported by `docker inspect`>
  # (actions/runner, Handlers/StepHost.cs). This job image declares no PATH in
  # its image config, so whatever is written here becomes the whole PATH: a bare
  # `$VENV_ROOT/bin` line would drop `/usr/bin` and `/bin`, and the next
  # `docker exec` fails with `exec: "bash": executable file not found in $PATH`.
  # Publish the interpreter's own PATH, venv first, so later steps resolve
  # commands exactly as this script does.
  printf '%s\n' "$PATH" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR CUDA_HOME CUDA_PATH \
    FLAGOS_CUDA_ASSETS_DIR FLAGGEMS_DIR FLAGCX_PATH FLAGTREE_VERSION GEMS_VENDOR \
    CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
