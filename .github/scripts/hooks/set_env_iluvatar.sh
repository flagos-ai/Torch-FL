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
create_runtime_venv "$VENDOR_PYTHON"
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

ensure_runtime_requirements "$REPO_ROOT/.github/requirements-iluvatar.txt" "$PIP_INDEX_URL"

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
# The job container owns its venv; the image remains an immutable warm baseline.
# Caller supplies the requested versions, indexes and common install helpers.
prepare_iluvatar_packages() {
  if "$VENV_PYTHON" - "$FLAGTREE_VERSION" <<'PY'
import importlib.metadata
import sys

try:
    import triton.backends
    assert importlib.metadata.version("flagtree") == sys.argv[1]
    assert "iluvatar" in triton.backends.backends
except (AssertionError, ImportError, importlib.metadata.PackageNotFoundError):
    raise SystemExit(1)
PY
  then
    echo "Using preinstalled FlagTree $FLAGTREE_VERSION"
  else
    # FlagTree owns triton's modules. Remove both metadata owners before
    # replacing the provider; installing stock Triton separately is invalid.
    "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
    install_flagtree
  fi

  install_flag_gems
  # New releases may require a different dependency cohort. Fail explicitly
  # instead of letting an incompatible install reach the build or hardware.
  "$VENV_PYTHON" -m pip check
}
prepare_iluvatar_packages

echo "Prepared Iluvatar CI environment (runtime reuse and pip installs above):"
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

# Validate build inputs here; the common hardware gate owns installed-wheel
# checks, and the integration test owns the original computation smoke.
"$VENV_PYTHON" - "$REPO_ROOT" "$FLAGTREE_PYTHON_VERSION_iluvatar" "$CPU_TORCH_VERSION_DEFAULT" <<'PY'
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import torch
import triton
import yaml
from packaging.version import Version

sys.path.insert(0, str(Path(sys.argv[1]) / ".github/scripts"))
import check_platform_environment as checks

spec = yaml.safe_load(
    (Path(sys.argv[1]) / ".github/configs/iluvatar.yml").read_text()
)["environment_check"]
checks.validate_spec(spec)
assert sys.argv[3] == "2.10.0", (
    "Changing CPU torch requires validation of the independent CoreX runtime"
)
assert sys.version_info[:2] == tuple(map(int, sys.argv[2].split("."))), sys.version
dependencies = SimpleNamespace(torch=torch, triton=triton)
checks.assert_torch_version(dependencies, spec)
checks.assert_cuda_version(dependencies, spec)
assert torch._C._GLIBCXX_USE_CXX11_ABI == spec["torch_cxx11_abi"], (
    "The untouched CPU core must be active"
)
root = checks.isolated_venv_root(sys)
checks.assert_venv_package(root, torch.__file__)
checks.assert_venv_package(root, triton.__file__)
checks.assert_import_isolation(root, sys.path)
assert importlib.util.find_spec("vllm") is None, "Base Python packages leaked"
assert importlib.metadata.version("flagtree") == os.environ["FLAGTREE_VERSION"]
assert importlib.metadata.version("flag_gems") == os.environ["FLAGGEMS_VERSION"]
checks.assert_triton_backend(dependencies, spec["triton_backend"])
cmake_version = Version(importlib.metadata.version("cmake"))
assert Version("3.19") <= cmake_version < Version("4"), cmake_version
checks.assert_venv_package(root, shutil.which("cmake"))

sdk = Path(os.environ["COREX_PATH"])
for header in ("cuda_runtime.h", "cudnn.h", "cublas_v2.h"):
    assert (sdk / "include" / header).is_file(), header
for library in ("libcudart.so*", "libcudnn.so*", "libcublas.so*"):
    assert any(
        path.is_file()
        for directory in ("lib", "lib64")
        for path in (sdk / directory).glob(library)
    ), library
subprocess.run(
    ["c++", "-fsyntax-only", "-x", "c++", "-I", str(sdk / "include"), "-"],
    input=(
        "#include <cublas_v2.h>\n#include <cudnn.h>\n"
        "int main() { ixblasStatus_t s = IXBLAS_STATUS_SUCCESS; "
        "ixblasDataType_t d = IX_R_16BF; return int(s) + int(d); }\n"
    ),
    text=True,
    check=True,
)
PY

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
