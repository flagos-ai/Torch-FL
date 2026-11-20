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

# Install the exact TileOPs/MetaX runtime stack. This runs after set_env.sh has
# selected the MetaX Python and exported MACA paths, and before setup.py builds
# the wheel. The private provider is fetched with a token-safe askpass helper.

set -euo pipefail

: "${REPO_ROOT:?source set_env.sh before this script}"
: "${TILEOPS_GITHUB_TOKEN:?TILEOPS_GITHUB_TOKEN is required to fetch the private MetaX provider}"

source "${REPO_ROOT}/.github/version-pins.env"
# The shared Torch-FL constraints pin torch's base version and NumPy<2. The
# TileLang wheel and TileOPs both declare a torch range; every package install
# below explicitly applies the same constraints the platform environment uses.

TILEOPS_COMMIT="${TILEOPS_OFFICIAL_COMMIT:-$TILEOPS_OFFICIAL_COMMIT_DEFAULT}"
TILELANG_VERSION="${TILELANG_METAX_VERSION:-$TILELANG_METAX_VERSION_DEFAULT}"
TILELANG_COMMIT="${TILELANG_METAX_COMMIT:-$TILELANG_METAX_COMMIT_DEFAULT}"
PROVIDER_COMMIT="${TILEOPS_METAX_COMMIT:-$TILEOPS_METAX_COMMIT_DEFAULT}"
TVM_FFI_VERSION="${APACHE_TVM_FFI_VERSION:-$APACHE_TVM_FFI_VERSION_DEFAULT}"
METAX_PYPI_INDEX_URL="${METAX_PYPI_INDEX_URL:-https://repos.metax-tech.com/r/maca-pypi/simple}"
METAX_PYPI_TRUSTED_HOST="${METAX_PYPI_TRUSTED_HOST:-repos.metax-tech.com}"

work_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/tileops-metax.XXXXXX")"
askpass="$(mktemp "${RUNNER_TEMP:-/tmp}/tileops-askpass.XXXXXX")"
# The installed wheel declares the normal Torch-FL FlagTree/FlagGems/FlagCX
# dependencies. This experiment supplies TileLang instead and installs the wheel
# with --no-deps, so the global build constraint must not reject that metadata
# while setup.py is queried.
cleanup() {
  rm -rf -- "$work_root" "$askpass"
}
trap cleanup EXIT

cat >"$askpass" <<'SH'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "$TILEOPS_GITHUB_TOKEN" ;;
  *) exit 1 ;;
esac
SH
chmod 0700 "$askpass"

clone_pinned() {
  local name="$1" url="$2" commit="$3" destination="$4"
  mkdir -p "$destination"
  git -C "$destination" init --quiet
  git -C "$destination" remote add origin "$url"
  GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
    git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach FETCH_HEAD
  actual="$(git -C "$destination" rev-parse HEAD)"
  [[ "$actual" == "$commit" ]] || {
    echo "::error::$name resolved $actual, expected $commit" >&2
    exit 1
  }
}

mkdir -p "$work_root"
clone_pinned official-tileops \
  "https://github.com/tile-ai/TileOPs.git" \
  "$TILEOPS_COMMIT" "$work_root/TileOPs"
clone_pinned metax-provider \
  "https://github.com/ChipMind-CAS/tileops-metax.git" \
  "$PROVIDER_COMMIT" "$work_root/tileops-metax"
clone_pinned tilelang-metax \
  "https://github.com/tile-ai/tilelang-metax.git" \
  "$TILELANG_COMMIT" "$work_root/tilelang-metax"
GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
  git -C "$work_root/tilelang-metax" submodule sync --recursive
GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
  git -C "$work_root/tilelang-metax" submodule update --init --recursive --depth 1

PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt" python -m pip install \
  --index-url "$PIP_INDEX_URL_DEFAULT" \
  "apache-tvm-ffi==$TVM_FFI_VERSION" \
  "build>=1.2" \
  "cloudpickle" \
  "cython>=3.1.0" \
  "cmake>=3.26.1,<4" \
  "einops" \
  "ml-dtypes" \
  "ninja>=1.11" \
  "numpy<2" \
  "packaging>=26.2" \
  "pytest>=8.0" \
  "psutil" \
  "PyYAML>=6.0" \
  "scikit-build-core>=0.10.0" \
  "tqdm" \
  "typing-extensions>=4.10.0" \
  "z3-solver>=4.13.0,<4.15.5"
PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt" python -m pip install --no-build-isolation \
  "setuptools>=77" \
  "setuptools-scm>=8" \
  "wheel"
PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt" python -m pip install --no-cache-dir \
  --index-url "$METAX_PYPI_INDEX_URL" \
  --trusted-host "$METAX_PYPI_TRUSTED_HOST" \
  "torch-c-dlpack-ext==0.1.5"
PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt" USE_MACA=ON CIBUILDWHEEL=1 \
  python -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$work_root/wheels" "$work_root/tilelang-metax"
PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt" \
  python -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$work_root/wheels" \
  "$work_root/tilelang-metax/3rdparty/tvm/3rdparty/tvm-ffi"
python -m pip install --no-deps --force-reinstall \
  "$work_root"/wheels/apache_tvm_ffi-*.whl \
  "$work_root"/wheels/tilelang-*.whl
python -m pip install --no-deps --force-reinstall "$work_root/TileOPs"
python -m pip install --no-deps --force-reinstall "$work_root/tileops-metax"

python - "$TILELANG_VERSION" "$TVM_FFI_VERSION" "$TILEOPS_COMMIT" "$PROVIDER_COMMIT" <<'PY'
import importlib.metadata as metadata
import sys
import tilelang
import torch
from tilelang import tvm
from tileops.backend import registry
from tileops.backend.registry import ensure_loaded

expected_tilelang, expected_tvm_ffi, expected_tileops, expected_provider = sys.argv[1:]
installed_tilelang = metadata.version("tilelang")
assert installed_tilelang.split("+", 1)[0] == expected_tilelang, (
    installed_tilelang, expected_tilelang
)
assert "+maca" in installed_tilelang, installed_tilelang
installed_tvm_ffi = metadata.version("apache-tvm-ffi")
assert installed_tvm_ffi.split("+", 1)[0] == expected_tvm_ffi, (
    installed_tvm_ffi, expected_tvm_ffi
)
assert tvm.get_global_func("target.build.tilelang_maca", allow_missing=True)
assert getattr(torch.version, "maca", None), "torch.version.maca is absent"
version = metadata.version("tileops")
assert version.endswith("+" + expected_tileops[:10]), (version, expected_tileops)
metadata.distribution("tileops-maca")
ensure_loaded()
known = registry.known_targets()
assert "tileops_maca" in known, known
print(f"TileLang: {installed_tilelang}")
print(f"TVM FFI: {installed_tvm_ffi}")
print(f"TileOPs: {version} ({expected_tileops})")
print(f"Provider: tileops-maca ({expected_provider})")
PY

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    printf 'FLAGOS_BUILD_TILEOPS=1\n'
    printf 'FLAGOS_BUILD_BOXING=0\n'
    printf 'FLAGOS_BUILD_FLAGGEMS=0\n'
    printf 'FLAGOS_BUILD_FLAGGEMS_CPP=0\n'
    printf 'FLAGOS_TILEOPS_PROVIDER=tileops_maca\n'
    printf 'FLAGOS_BACKEND_CONFIG=%s/torch_fl/configs/backends_metax_tileops.conf\n' "$REPO_ROOT"
    printf 'FLAGOS_LOG=dispatch\n'
    printf 'TILELANG_VERSION=%s\n' "$TILELANG_VERSION"
    printf 'TILEOPS_VERSION=%s\n' "$TILEOPS_COMMIT"
    printf 'TILEOPS_PROVIDER_VERSION=%s\n' "$PROVIDER_COMMIT"
  } >>"$GITHUB_ENV"
fi
