# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

if [[ "${BOXING_LANE:-}" == "1" ]]; then
  echo "::error::Iluvatar uses the independent CoreX runtime and has no boxing lane" >&2
  exit 1
fi

# The runtime initializes CoreX before this hook. Later Bash shells must not
# replay /etc/bash_env.sh and select the runtime's vendor torch in /flagos.
ILUVATAR_PREPARE_STARTED=$SECONDS
export BASH_ENV=/dev/null
export PYTHONPATH=""
export PYTHONNOUSERSITE=1
export COREX_PATH="${COREX_PATH:-${COREX_ROOT:-/usr/local/corex}}"
export COREX_ROOT="$COREX_PATH"
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
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
PIP_INDEX_URL="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"
export FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_iluvatar}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
export FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-iluvatar/simple}"
# The runtime image supplies CoreX 4.5.0 independently of the
# host kernel driver's version. Record that SDK, not the host's driver 5.0.0.
export FLAGOS_SDK_VERSION=4.5.0

# Metadata inspection must not import the runtime's vendor Torch. Its native
# library paths must not leak into the isolated CPU core's build or loader.
VENDOR_PYTHON="${TORCH_FL_VENDOR_PYTHON:-$(command -v python3.12 || true)}"
if [[ ! -x "$VENDOR_PYTHON" ]] || ! "$VENDOR_PYTHON" -I -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "::error::Iluvatar requires a Python 3.12 interpreter to create the CI venv" >&2
  exit 1
fi
VENDOR_TORCH_ROOT="$("$VENDOR_PYTHON" -I - <<'PY'
from importlib import metadata
from pathlib import Path
try:
    print(Path(metadata.distribution("torch").locate_file("torch")).resolve())
except metadata.PackageNotFoundError:
    print("/nonexistent/vendor-torch")
PY
)"
export CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
export LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
export LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CPATH="$(strip_vendor_paths "${CPATH:-}")"

for directory in "$COREX_PATH/lib64" "$COREX_PATH/lib"; do
  if [[ -d "$directory" ]]; then
    export LIBRARY_PATH="$directory${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
export CPATH="$COREX_PATH/include${CPATH:+:$CPATH}"

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-iluvatar-${CI_STAGE}}"
# --clear must never target the source runtime or a parent of its prefix.
"$VENDOR_PYTHON" -I - "$VENV_ROOT" <<'PY'
from pathlib import Path
import sys
source, target = Path(sys.prefix).resolve(), Path(sys.argv[1]).resolve()
if source.is_relative_to(target) or target.is_relative_to(source):
    raise SystemExit("CI venv must be separate from the runtime Python prefix")
PY
if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "::error::Python venv creation failed and the runtime has no uv fallback" >&2
    exit 1
  fi
  uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
"$VENDOR_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
  --target-python "$VENV_PYTHON" \
  --torch-version "$CPU_TORCH_VERSION" \
  --flagtree-version "$FLAGTREE_VERSION" \
  --flaggems-version "$FLAGGEMS_VERSION" \
  --requirements-file "$REPO_ROOT/.github/requirements-iluvatar.txt"
export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$COREX_PATH/bin:$PATH"

# The normal runtime carries vendor Torch, so the CPU wheel usually needs
# installation. A future matching runtime copy can skip even the index probe.
if "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --check-installed "torch==${CPU_TORCH_VERSION}+cpu"; then
  echo "Using prepared CPU torch $CPU_TORCH_VERSION"
else
  install_cpu_torch
fi

while IFS= read -r requirement || [[ -n "$requirement" ]]; do
  [[ -z "$requirement" || "$requirement" == \#* ]] && continue
  if "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
      --check-installed "$requirement"; then
    echo "Using prepared requirement: $requirement"
  else
    pip_retry --index-url "$PIP_INDEX_URL" "$requirement"
  fi
done < "$REPO_ROOT/.github/requirements-iluvatar.txt"

# Triton matmul tuning needs the real host driver tool, not a placeholder.
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

echo "Prepared Iluvatar CI environment (runtime reuse and pip installs above):"
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

# Use the checkout's verifier and pins, not the older copy baked into Harbor.
TORCH_FL_VERSION_PINS="$REPO_ROOT/.github/version-pins.env" \
  "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/verify_iluvatar.py"

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" "$COREX_PATH/bin" >> "$GITHUB_PATH"
fi
export_ci_env PATH VIRTUAL_ENV BASH_ENV PYTHONPATH PYTHONNOUSERSITE COREX_PATH COREX_ROOT \
  FLAGOS_ACCELERATOR FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS \
  FLAGOS_BUILD_BOXING FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_TILEOPS \
  FLAGOS_SKIP_CUDA_ASSETS FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_BUILD_JOBS \
  FLAGTREE_VERSION FLAGGEMS_VERSION PIP_CONSTRAINT FLAGOS_SDK_VERSION FLAGOS_VENDOR_TORCH_VERSION \
  FLAGOS_VENDOR_TORCH_LIB TORCH_DEVICE_BACKEND_AUTOLOAD CPATH LIBRARY_PATH LD_LIBRARY_PATH \
  CMAKE_PREFIX_PATH
echo "Iluvatar dependency preparation completed in $((SECONDS - ILUVATAR_PREPARE_STARTED))s"

# The previous CI image had no discoverable cuda_occupancy.h. Keep that same
# optional branch off: the new runtime's CUDA compatibility headers must not
# add compiler include paths or enable occupancy while only changing images.
# An empty find_path cache entry survives both the in-place and wheel builds.
mkdir -p "$REPO_ROOT/build"
printf '\nCUDA_OCCUPANCY_INCLUDE_DIR:PATH=\n' >> "$REPO_ROOT/build/CMakeCache.txt"

# Build against the active CPU torch headers, CMake config and C++ ABI.
build_flagos_inplace
