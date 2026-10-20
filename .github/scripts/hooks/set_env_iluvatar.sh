# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Official CPU torch core with the independent CoreX C-ABI runtime.
if [[ ! -x /opt/torch-fl-iluvatar-venv/bin/python || ! -f /opt/torch-fl-ci/verify.py ]]; then
  echo "::error::The Iluvatar CI image is required (/opt/torch-fl-iluvatar-venv and /opt/torch-fl-ci)" >&2
  exit 1
fi

export VIRTUAL_ENV=/opt/torch-fl-iluvatar-venv
VENV_PYTHON="$VIRTUAL_ENV/bin/python"
export PATH="/opt/torch-fl-iluvatar-venv/bin:/usr/local/corex/bin:$PATH"
export PYTHONPATH=""
export PYTHONNOUSERSITE=1
export COREX_PATH=/usr/local/corex
# Clear legacy image defaults in this shell and subsequent Actions steps.
# The independent runtime must never select a vendor torch facade.
export FLAGOS_VENDOR_TORCH_LIB=""
export FLAGOS_VENDOR_TORCH_VERSION=""
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLAGOS_ACCELERATOR=iluvatar
export FLAGOS_BUILD_VENDOR=1
export FLAGOS_BUILD_FLAGGEMS=1
export FLAGOS_BUILD_BOXING=0
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_TILEOPS=0
export FLAGOS_SKIP_CUDA_ASSETS=1
export FLAGOS_DISABLE_CUDA_ASSETS=1
export FLAGOS_BUILD_JOBS="${FLAGOS_BUILD_JOBS:-8}"
export FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_iluvatar}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
export FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-iluvatar/simple}"
# The pinned development image supplies CoreX 4.5.0 independently of the
# host kernel driver's version. Record that SDK, not the host's driver 5.0.0.
export FLAGOS_SDK_VERSION=4.5.0

for directory in "$COREX_PATH/lib64" "$COREX_PATH/lib"; do
  if [[ -d "$directory" ]]; then
    export LIBRARY_PATH="$directory${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
export CPATH="$COREX_PATH/include${CPATH:+:$CPATH}"

# The development base leaves an empty ixsmi placeholder. Triton matmul tuning
# needs the host driver tool, even when the basic CUDA/Triton add probe passes.
if [[ "$CI_STAGE" == integration ]]; then
  ixsmi_path="$(command -v ixsmi || true)"
  if [[ ! -s "$ixsmi_path" || ! -x "$ixsmi_path" ]]; then
    echo "::error::Mount the host CoreX bin/ixsmi read-only at /usr/local/corex/bin/ixsmi" >&2
    exit 1
  fi
fi

# set_env.sh replaces the baked constraints with the checkout's constraints,
# which protect CPU torch without pinning FlagTree/Gems to the old image.
# shellcheck source=.github/scripts/lib/iluvatar_packages.sh
source "$REPO_ROOT/.github/scripts/lib/iluvatar_packages.sh"
prepare_iluvatar_packages

# Use the checkout's verifier and pins, not the older copy baked into Harbor.
TORCH_FL_VERSION_PINS="$REPO_ROOT/.github/version-pins.env" \
  "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/verify_iluvatar.py"

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' /opt/torch-fl-iluvatar-venv/bin /usr/local/corex/bin >> "$GITHUB_PATH"
fi
export_ci_env PATH VIRTUAL_ENV PYTHONPATH PYTHONNOUSERSITE COREX_PATH \
  FLAGOS_ACCELERATOR FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS \
  FLAGOS_BUILD_BOXING FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_TILEOPS \
  FLAGOS_SKIP_CUDA_ASSETS FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_BUILD_JOBS \
  FLAGTREE_VERSION FLAGGEMS_VERSION PIP_CONSTRAINT FLAGOS_SDK_VERSION FLAGOS_VENDOR_TORCH_VERSION \
  FLAGOS_VENDOR_TORCH_LIB TORCH_DEVICE_BACKEND_AUTOLOAD CPATH LIBRARY_PATH LD_LIBRARY_PATH

# Build against the active CPU torch headers, CMake config and C++ ABI.
build_flagos_inplace
