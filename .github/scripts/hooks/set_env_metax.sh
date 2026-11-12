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


# Resolve the operator names the checked-in Python kernels call against the
# FlagGems an interpreter in $1 imports, and print "<missing>/<total>".
#
# A generated kernel calls its operator by package-level name
# (`flag_gems.<name>`), resolved at dispatch time by
# csrc/aten/backends/flagos/python_op_caller.cc:GetFunc -- by getattr on the
# package. A cohort that does not define one of those names therefore raises
# AttributeError on the routes that use it, which is why "a FlagGems is
# installed" is not the property worth checking. Must run from the repo root:
# that is where the generated kernels are.
#
# $2, when "torch_fl", imports torch_fl in the probe process before flag_gems.
# Only the venv call passes it; see the cohort check at the end of this script
# for why the ordering is load-bearing on MetaX. The vendor interpreter must not
# get it: it has its own MetaX torch, and this checkout's build-tree torch_fl is
# not built for that interpreter.
gems_cohort_gap() {
  (
    cd "$REPO_ROOT" || exit 1
    GEMS_COHORT_PRELUDE="${2:-}" "$1" - <<'PY'
import importlib
import os
import re
from pathlib import Path

if os.environ.get("GEMS_COHORT_PRELUDE") == "torch_fl":
    # Must precede flag_gems: torch_fl selects the MetaX core via a private facade.
    # Without that, MetaX FlagTree Triton resolves no
    # active driver, so flag_gems raises at import. See the cohort check below.
    import torch_fl  # noqa: F401, E402

source = Path("csrc/aten/generated/flaggems_python_kernels.cc").read_text()
names = sorted(set(re.findall(r'"(flag_gems\.[A-Za-z0-9_.]+)"', source)))

import flag_gems  # noqa: E402  -- the cohort under test

missing = []
for qualname in names:
    func = qualname.split(".", 1)[1]
    try:
        getattr(flag_gems, func)
        continue
    except Exception:  # noqa: BLE001  -- unresolvable here, try the module form
        pass
    # GetFunc's fallback: a dotted name imports the prefix and takes the last
    # component, so flag_gems.sum.dim_IntList may resolve as flag_gems.sum.
    if "." in func:
        prefix, last = func.rsplit(".", 1)
        try:
            getattr(importlib.import_module(f"flag_gems.{prefix}"), last)
            continue
        except Exception:  # noqa: BLE001
            pass
    missing.append(qualname)

print(f"{len(missing)}/{len(names)}")
PY
  )
}

# Runtime preparation follows the same isolated package/cache path as the
# other migrated lanes. Vendor libtorch remains a read-only build input.
metax_validate_build_tools() {
  # setup.py imports editable_wheel even for a non-editable native build.
  # Validate real imports after provisioning, before any large wheel download.
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV_PYTHON" - <<'PY'
import build
import setuptools
from setuptools.command.editable_wheel import editable_wheel

print(f"Build frontend: {build.__version__}")
print(f"Setuptools: {setuptools.__version__} (editable_wheel available)")
PY
}

metax_venv_is_isolated() {
  venv_is_usable || return 1
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV_PYTHON" - "$VENV_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
assert Path(sys.prefix).resolve() == root
assert sys.prefix != sys.base_prefix
settings = {}
for line in (root / "pyvenv.cfg").read_text().splitlines():
    key, separator, value = line.partition("=")
    if separator:
        settings[key.strip().lower()] = value.strip().lower()
assert settings.get("include-system-site-packages") == "false"
PY
}

metax_cpu_torch_is_usable() {
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV_PYTHON" - "$CPU_TORCH_VERSION" <<'PY'
import sys
import torch

expected = sys.argv[1].removesuffix("+cpu") + "+cpu"
assert torch.__version__ == expected, torch.__version__
assert torch.version.cuda is None, torch.version.cuda
PY
}

metax_runtime_requirement() {
  if "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
      --check-installed "$1"; then
    echo "Using prepared requirement: $1"
  else
    pip_retry --index-url "$PIP_INDEX_URL_ARG" "$1"
  fi
  "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --check-installed "$1"
}

metax_runtime_flagtree_is_usable() {
  "$VENV_PYTHON" - "$FLAGTREE_VERSION" <<'PY'
from importlib import metadata, util
from pathlib import Path
import sys

try:
    dist = metadata.distribution("flagtree")
    root = Path(dist.locate_file("triton")).resolve()
    spec = util.find_spec("triton")
    assert dist.version == sys.argv[1]
    assert root.is_relative_to(Path(sys.prefix).resolve())
    assert spec is not None and spec.origin
    assert Path(spec.origin).resolve().parent == root
    assert (root / "backends/metax").is_dir()
    assert (root / "_flagtree_spec.py").is_file()
except (metadata.PackageNotFoundError, AssertionError):
    raise SystemExit(1)
PY
}

metax_record_runtime_sdk() {
  local version=""
  version="$(cat "${MACA_PATH:-/opt/maca}/Version.txt")" || return 1
  version="${version%$'\r'}"
  version="${version#Version:}"
  if [[ "$version" != 3.8.1.3 ]]; then
    echo "::error::Expected MACA SDK 3.8.1.3, found $version" >&2
    return 1
  fi
  export FLAGOS_SDK_VERSION="$version"
}

metax_read_vendor_torch() {
  local details=""
  # Read distribution metadata, never import the vendor Python torch/backend.
  details="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$1" -I - "$CPU_TORCH_VERSION" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

dist = metadata.distribution("torch")
version = dist.version
assert version.split("+", 1)[0] == sys.argv[1].removesuffix("+cpu"), version
assert "metax" in version, version
root = Path(dist.locate_file("torch")).resolve()
assert root.is_relative_to(Path(sys.prefix).resolve()), root
assert (root / "version.py").is_file(), root
print(f"{root / 'lib'}\t{version}")
PY
)" || return 1
  IFS=$'\t' read -r FLAGOS_VENDOR_TORCH_LIB FLAGOS_VENDOR_TORCH_VERSION <<< "$details"
  export FLAGOS_VENDOR_TORCH_LIB FLAGOS_VENDOR_TORCH_VERSION
  export TORCH_FL_VENDOR_TORCH_VERSION="${FLAGOS_VENDOR_TORCH_VERSION%%+*}"
}

prepare_metax_runtime_environment() {
  metax_record_runtime_sdk
  local vendor_python="${TORCH_FL_VENDOR_PYTHON:-$(command -v python3.12 || true)}"
  export BASH_ENV=/dev/null
  export PYTHONPATH=""
  export PYTHONNOUSERSITE=1
  if [[ ! -x "$vendor_python" ]] || ! "$vendor_python" -I -c \
      'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    echo "::error::The MetaX runtime requires Python 3.12 for its FlagTree wheel" >&2
    return 1
  fi
  metax_read_vendor_torch "$vendor_python"
  local suffix=""
  if [[ "${BOXING_LANE:-}" == 1 ]]; then suffix=-boxing; fi
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-metax-${CI_STAGE}${suffix}}"
  "$vendor_python" -I - "$VENV_ROOT" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

target, repo = (Path(p).resolve() for p in sys.argv[1:])
for prefix in (sys.prefix, sys.base_prefix):
    source = Path(prefix).resolve()
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise SystemExit("CI venv must be separate from the runtime Python prefixes")
if repo.is_relative_to(target):
    raise SystemExit("CI venv must not contain the source checkout")
PY
  if ! "$vendor_python" -m venv --clear "$VENV_ROOT"; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "::error::Python venv creation failed and the runtime has no uv fallback" >&2
      return 1
    fi
    uv venv --clear --seed --python "$vendor_python" "$VENV_ROOT"
  fi
  VENV_PYTHON="$VENV_ROOT/bin/python"
  metax_venv_is_isolated
  "$vendor_python" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --target-python "$VENV_PYTHON" \
    --torch-version "${CPU_TORCH_VERSION%+cpu}" \
    --flagtree-version "${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_metax}" \
    --flaggems-version "${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}" \
    --requirements-file "$REPO_ROOT/.github/requirements-metax.txt"
  export VIRTUAL_ENV="$VENV_ROOT"
  export PATH="$VENV_ROOT/bin:$PATH"
  # A Python CMake distribution is required by pyproject.toml even when the
  # image supplies /usr/bin/cmake. Bootstrap pip before resolving CPU Torch.
  local requirement
  while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    [[ -z "$requirement" || "$requirement" == \#* ]] && continue
    metax_runtime_requirement "$requirement"
  done < "$REPO_ROOT/.github/requirements-metax.txt"
  metax_validate_build_tools
  if ! metax_cpu_torch_is_usable 2>/dev/null; then
    CPU_TORCH_VERSION="${CPU_TORCH_VERSION%+cpu}+cpu"
    install_cpu_torch
  fi
}

# The runtime path uses a job-owned venv; explicit legacy development-image
# users retain their prebuilt /opt/venv and library-only input below. Both paths
# install CPU Torch through the same shared index/mirror helper.
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"
METAX_RUNTIME_REUSE=0
if [[ "${VIRTUAL_ENV:-}" == /flagos || -n "${TORCH_FL_VENDOR_PYTHON:-}" ]]; then
  METAX_RUNTIME_REUSE=1
  prepare_metax_runtime_environment
else
VENV_ROOT=/opt/venv
VENV_PYTHON=/opt/venv/bin/python
if [[ ! -x /opt/venv/bin/python && -x /flagos/bin/python ]]; then
  FLAGOS_VENDOR_TORCH_LIB="$(/flagos/bin/python - <<'PY'
from pathlib import Path
import torch

assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
  export FLAGOS_VENDOR_TORCH_LIB
  if ! /flagos/bin/python -m venv /opt/venv; then
    uv venv --clear --seed --python /flagos/bin/python /opt/venv
  fi
  install_cpu_torch
  pip_retry --index-url "$PIP_INDEX_URL_DEFAULT" \
    build cmake ninja setuptools wheel pytest patchelf
  export PYTHONPATH=""
else
  export FLAGOS_VENDOR_TORCH_LIB=/opt/vendor-libtorch/lib
  # This image stages only the shared libraries, without torch/version.py.
  # The image's vendor core is built against the pinned 2.10.0 Python wheel.
  export TORCH_FL_VENDOR_TORCH_VERSION=2.10.0
fi
fi
# The wheel label follows the shared SDK table; record the inspected runtime's
# actual SDK and vendor Torch independently in the compatibility manifest.
export PATH="$VENV_ROOT/bin:/opt/maca/tools/cu-bridge/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/bin:$PATH"
export VIRTUAL_ENV="$VENV_ROOT"
export PYTHONNOUSERSITE=1

export FLAGOS_ACCELERATOR=metax
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca

# MetaX builds no native mxcc kernels, so FLAGOS_BUILD_VENDOR=OFF is one
# build-side statement on both lanes. The exports above are build inputs --
# setup.py records them in torch_fl/_build_config.py, and the runtime derives
# its conf from that record (torch_fl._select_backend_config), so there is no
# mode variable to keep in agreement with the build any more.
#
# MetaX runs both builds of the wheel, and BOXING_LANE says which one this job
# is:
#
#   vendor lane  the default build: FlagGems routes only, no generated boxing
#                kernels. What a `pip install` gets.
#
#   boxing lane  FLAGOS_BUILD_BOXING=1: the generated boxing kernels, the
#                cu-bridge shim that lets them reach MetaX's MXVM, and the day-0
#                conf. A boxing wheel registers every op its boxed table names
#                on PrivateUse1, and backends_metax.conf -- the file a default
#                wheel selects -- leaves the uncovered ones on `none`, which on
#                that wheel is a raise rather than a cpu_fallback. Measured on
#                the MetaX lane (run 40093392100 onwards first showed it; run
#                40107997127 is the collapse): `empty_like: routed to 'none'
#                ... but the op is registered on PrivateUse1` for every
#                unary/binary case of the portable group.
#                backends_metax_boxing.conf is the same routing table with
#                FlagGems where this build has it (594 ops) and boxing for the
#                other 1442.
export FLAGOS_BUILD_VENDOR=OFF
# Both legacy CI lanes inherited these values from the old development image.
# State them explicitly instead of relying on that image-only environment.
export FLAGOS_METAX_CUDART_SHIM="${FLAGOS_METAX_CUDART_SHIM:-1}"
export FLAGOS_DISABLE_CUDA_ASSETS="${FLAGOS_DISABLE_CUDA_ASSETS:-1}"
if [[ "${BOXING_LANE:-}" == "1" ]]; then
  export FLAGOS_BUILD_BOXING=1
  export FLAGOS_METAX_CUDART_SHIM=1
  export FLAGOS_DISABLE_CUDA_ASSETS=1
  export FLAGOS_BACKEND_CONFIG="${FLAGOS_BACKEND_CONFIG:-$REPO_ROOT/torch_fl/configs/backends_metax_boxing.conf}"
  if [[ ! -f "$FLAGOS_BACKEND_CONFIG" ]]; then
    echo "::error::the MetaX boxing lane needs the day-0 conf $FLAGOS_BACKEND_CONFIG"
    exit 1
  fi
fi
# Which op takes which backend is stated in backends_metax.conf, not here: that
# file is full-coverage and lists all five keys per op
# (flaggems_cpp > flaggems > tileops > cuda), selected from the accelerator
# record. The retired FLAGOS_USE_FLAGGEMS switch
# used to select a separate backends_flaggems.conf; nothing reads it any more,
# so setting it here would misdescribe the build -- the FlagGems Python path is
# on for the 594 ops the conf routes to it either way.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1

export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/mxshmem/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="/opt/maca/tools/cu-bridge/include:/opt/maca/include:/opt/maca/include/mcr${CPATH:+:$CPATH}"

for path in \
  "$VENV_PYTHON" \
  "$FLAGOS_VENDOR_TORCH_LIB/libc10.so" \
  "$FLAGOS_VENDOR_TORCH_LIB/libtorch_cpu.so" \
  "$FLAGOS_VENDOR_TORCH_LIB/libtorch.so" \
  "$FLAGOS_VENDOR_TORCH_LIB/libtorch_global_deps.so" \
  "$FLAGOS_VENDOR_TORCH_LIB/libtorch_python.so" \
  "$FLAGOS_VENDOR_TORCH_LIB/libc10_cuda.so" \
  "$FLAGOS_VENDOR_TORCH_LIB/libtorch_cuda.so" \
  "$FLAGOS_VENDOR_TORCH_LIB/libtorch_cuda_linalg.so"; do
  if [[ ! -e "$path" ]]; then
    echo "::error::Required MetaX image asset is missing: $path"
    exit 1
  fi
done

if [[ ! -c /dev/mxcd ]]; then
  echo "::error::MetaX device node /dev/mxcd is unavailable"
  exit 1
fi

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - <<'PY'
from pathlib import Path
import sys

import torch

torch_path = Path(torch.__file__).resolve()
assert torch.__version__.startswith("2.10.0+cpu"), torch.__version__
assert torch_path.is_relative_to(Path(sys.prefix).resolve()), torch_path
assert "/opt/conda/" not in str(torch_path), torch_path

print(f"Build Python: {Path(__import__('sys').executable).resolve()}")
print(f"Build PyTorch: {torch.__version__}")
print(f"Build torch path: {torch_path}")
PY

# Expose a FlagTree Triton and a FlagGems cohort to the CPU torch venv.
# torch.compile needs Triton: the active torch is the CPU wheel, which ships no
# Triton, so inductor raises TritonMissing without this. The Triton has to be
# the MetaX FlagTree build -- the one whose "metax" backend is the one the
# generated kernels were measured on -- rather than the triton-metax the image
# carries beside its own MetaX torch install. FlagGems is required because
# backends_metax.conf routes 594 ops to the Python FlagGems path by default
# (FLAGOS_BUILD_FLAGGEMS=1 above compiles the dispatcher slot).
#
# Both are installed into the venv rather than linked out of the image, so the
# packages the tests import are the ones this script put there. Linking is what
# let a stale image cohort shadow the pinned revision: the venv resolved
# whatever the image's editable install pointed at, and the pin below never ran.
if [[ "$CI_STAGE" == "integration" || "$METAX_RUNTIME_REUSE" == 1 ]]; then
  VENV_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

  # Install FlagGems' runtime dependencies into the venv. flag_gems imports
  # packaging, yaml (PyYAML), sqlalchemy and numpy at or shortly after import.
  # Use --no-deps to protect the torch ABI (same rationale as test-dependencies).
  # numpy<2 because numpy 2.x breaks the stock +cpu torch C extensions at import
  # (documented in set_env_mthreads.sh:176-178).
  if [[ "$METAX_RUNTIME_REUSE" == 0 ]]; then
    python -m pip install --no-deps 'packaging>=26.0' 'PyYAML==6.0.1' 'sqlalchemy==2.0.48' 'numpy>=1.20,<2.0'
  fi

  # --- FlagTree: the Triton build carrying the "metax" backend ----------------
  #
  # The MetaX wheel supplies the backend used by FlagGems. Install only a
  # published binary so CI cannot fall back to a source build.
  FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_metax}"
  FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
  if [[ "$METAX_RUNTIME_REUSE" == 1 ]] && metax_runtime_flagtree_is_usable; then
    echo "Using runtime-seeded FlagTree $FLAGTREE_VERSION"
  else
    if [[ "$METAX_RUNTIME_REUSE" == 1 ]]; then
      "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
    fi
    install_flagtree
  fi

  # flagtree installs itself as the `triton` module, so assert on the resolved
  # module and not on the distribution name: an image triton-metax winning the
  # import, or a stock PyPI triton arriving as a dependency, would leave every
  # triton test running on a Triton the conf was not generated against --
  # silently, since both import as `triton`.
  #
  # Nothing is uninstalled first. If the venv inherits system site-packages, a
  # `pip uninstall triton` would reach into the image's own MetaX torch install,
  # and this script deliberately consumes only libtorch from it. flagtree's files
  # win on their own: the venv's site-packages precede the system one on
  # sys.path, which is what the assertion below checks.
  python - <<'PY'
from pathlib import Path
import sysconfig

import triton
import triton.backends

site = Path(triton.__file__).resolve().parent
venv_site = Path(sysconfig.get_paths()["purelib"]).resolve()

assert site.parent == venv_site, (
    f"the venv resolves triton from {site}, not from {venv_site}: another "
    "triton shadows the flagtree install"
)
assert "metax" in triton.backends.backends, sorted(triton.backends.backends)
assert (site / "_flagtree_spec.py").is_file(), (
    f"the active triton is not a FlagTree build: {site}"
)

print(f"FlagTree Triton: {triton.__version__} ({site})")
PY

  # A generated kernel resolves each flag_gems.<name> at dispatch time. Keep
  # the cohort-gap probe at the end of this script, after torch_fl is built.
  FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
  FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-metax/simple}"
  # Older images linked a vendor source checkout into this venv. A matching
  # dist-info alone would otherwise make that link look like the wheel.
  if [[ -L "$VENV_SITE/flag_gems" ]]; then
    rm "$VENV_SITE/flag_gems"
  fi
  for metadata in "$VENV_SITE"/flag_gems-*.dist-info; do
    [[ -L "$metadata" ]] && rm "$metadata"
  done
  install_flag_gems

  # Preserve the legacy FlagCX branch. A new runtime carrying the wheel does
  # not opt either CI lane into communication.
  if [[ "$METAX_RUNTIME_REUSE" == 0 && "$FLAGOS_VENDOR_TORCH_LIB" == /flagos/* ]]; then
    FLAGCX_VERSION="${TORCH_FL_FLAGCX_VERSION:-$FLAGCX_VERSION_metax}"
    pip_retry --no-deps --only-binary=:all: --index-url "$FLAGGEMS_INDEX_URL" \
      "flagcx===$FLAGCX_VERSION"
    export FLAGCX_TORCH_BACKEND=flagos
    # The extension is built after provisioning, so defer PyTorch's FlagCX
    # entry-point import until later workflow steps.
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  fi
fi

if [[ "$METAX_RUNTIME_REUSE" == 1 ]]; then
  "$VENV_PYTHON" -m pip check
  "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment
fi
# Preserve the existing fallback metadata for legacy library-only images.
: "${TORCH_FL_VENDOR_TORCH_VERSION:=2.10.0}"
if command -v mx-smi >/dev/null 2>&1; then
  mx-smi
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' \
    "$VENV_ROOT/bin" \
    /opt/maca/tools/cu-bridge/bin \
    /opt/maca/mxgpu_llvm/bin \
    /opt/maca/bin >> "$GITHUB_PATH"
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf '%s=%s\n' PATH "$PATH" >> "$GITHUB_ENV"

  # The boxing lane assigns these; the vendor lane leaves them unset, and
  # export_ci_env reads every name under `set -u`. Assign the empty default here
  # so both lanes persist an explicit value.
  : "${FLAGOS_BUILD_BOXING:=}"
  : "${FLAGOS_BACKEND_CONFIG:=}"
  if [[ "$METAX_RUNTIME_REUSE" == 1 ]]; then
    export_ci_env BASH_ENV PYTHONPATH FLAGOS_SDK_VERSION FLAGOS_VENDOR_TORCH_VERSION
  fi
  export_ci_env VIRTUAL_ENV PYTHONNOUSERSITE FLAGOS_ACCELERATOR FLAGOS_BACKEND_CONFIG MACA_PATH MACA_HOME FLAGOS_BUILD_VENDOR FLAGOS_BUILD_BOXING FLAGOS_METAX_CUDART_SHIM FLAGOS_DISABLE_CUDA_ASSETS FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS FLAGOS_VENDOR_TORCH_LIB TORCH_FL_VENDOR_TORCH_VERSION LD_LIBRARY_PATH LIBRARY_PATH CPATH
  if [[ -n "${FLAGCX_TORCH_BACKEND:-}" ]]; then
    printf 'FLAGCX_TORCH_BACKEND=%s\n' "$FLAGCX_TORCH_BACKEND" >> "$GITHUB_ENV"
  fi
  if [[ -n "${TORCH_DEVICE_BACKEND_AUTOLOAD:-}" ]]; then
    printf 'TORCH_DEVICE_BACKEND_AUTOLOAD=%s\n' "$TORCH_DEVICE_BACKEND_AUTOLOAD" >> "$GITHUB_ENV"
  fi
fi

cd "$REPO_ROOT"

if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # setuptools collects package_data before build_ext on the first wheel build.
  # Prebuilding makes torch_fl/lib/*.so available when the common workflow
  # packages the local wheel for either stage. The following python -m build
  # is incremental.
  build_flagos_inplace
fi

# Populate the ignored package-data directory after the optional native
# prebuild. This also normalizes the native RPATHs before wheel/editable install.
bash scripts/vendor/bundle_metax_libtorch.sh

# --- FlagGems cohort ---------------------------------------------------------
#
# Confirm the vendor packages actually import against the CPU torch wheel, and
# that the FlagGems the venv resolves is the one that will answer dispatch.
# Failing here names the gap once instead of as a dispatch error per op.
#
# This runs after the native build, and not with the FlagGems install above,
# because on MetaX flag_gems cannot be imported by a process that has not
# imported torch_fl first -- set_env_mthreads.sh documents the same ordering for
# MThreads, for a different reason. Both halves are load-bearing:
#
#   * the venv's torch is the stock +cpu wheel, and a stock +cpu torch/lib has no
#     libtorch_cuda.so at all, so torch.cuda.is_available() is False there. The
#     MetaX FlagTree Triton then resolves zero active drivers.
#   * flag_gems reaches that resolution during its own import:
#     flag_gems/__init__.py -> flag_gems.fused -> fused/beam_search_score.py ->
#     utils/pointwise_dynamic.py (scalar_fn.cache_key) -> triton.runtime.jit
#     .parse -> triton/compiler/hint_manager.py:hint_get_flagtree_backend ->
#     hasattr(triton.runtime.driver, "active"). That attribute is a property
#     that builds the driver, and the hint manager catches only ImportError, so
#     the RuntimeError escapes the import:
#
#       RuntimeError: 0 active drivers ([]). There should only be one.
#         triton/spec/metax/triton/runtime/driver.py:14, in _create_driver
#
#     Seen in this job with flagtree 0.6.1+metax3.6 and torch 2.10.0+cpu in
#     /opt/venv, and reproduced on the C550 host by hiding the device
#     (CUDA_VISIBLE_DEVICES="" python -c "import flag_gems"). The shared input
#     is triton/backends/metax/driver.py:515 is_active() ->
#     torch.cuda.is_available(); it is False for a torch/lib with no
#     libtorch_cuda.so and for a device that is not visible alike.
#
#     Importing torch_fl is what removes that condition: it selects the MetaX
#     libtorch through a private facade (the bundle
#     bundle_metax_libtorch.sh just wrote, or FLAGOS_VENDOR_TORCH_LIB), and it has
#     to happen before `import torch` -- which is exactly the order the probe
#     uses. Child interpreters inherit the facade through PYTHONPATH; other
#     processes select it independently when they import torch_fl first.
#
# torch_fl._C is what makes torch_fl importable at all, and build_ext has just
# built it. The gate mirrors set_env_mthreads.sh's `import torch_fl._C` guard: on a
# tree without the extension there is nothing to probe, and that is not a
# FlagGems failure. The probe is skipped with a warning rather than failing the
# job for a state this script has not reached yet.
if [[ "$CI_STAGE" == "integration" ]]; then
  if ! python -c "import torch_fl._C" >/dev/null 2>&1; then
    echo "::warning::torch_fl._C is not importable from $REPO_ROOT, so the" \
         "FlagGems cohort cannot be probed here; the installed wheel answers" \
         "for it instead."
  else
    FLAGGEMS_GAP="$(gems_cohort_gap python torch_fl || true)"
    if [[ -z "$FLAGGEMS_GAP" ]]; then
      echo "::error::The venv cannot import flag_gems even with torch_fl" \
           "imported first; the 594 FlagGems routes in backends_metax.conf have" \
           "nothing to call."
      exit 1
    fi
    if [[ "$FLAGGEMS_GAP" != "0/"* ]]; then
      echo "::error::The FlagGems the venv resolves does not define $FLAGGEMS_GAP" \
           "of the names csrc/aten/generated/flaggems_python_kernels.cc calls;" \
           "those routes would raise AttributeError at dispatch."
      exit 1
    fi
    echo "FlagGems cohort: $FLAGGEMS_GAP kernel names resolve"

    # Which build each lane got, from the record setup.py wrote.
    python - "$([[ "${BOXING_LANE:-}" == "1" ]] && echo boxing || echo vendor)" <<'PY'
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_record = Path("torch_fl/_build_config.py")
assert _record.is_file(), _record
_loader = importlib.machinery.SourceFileLoader("_flagos_build_config", str(_record))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
_config = importlib.util.module_from_spec(_spec)
_loader.exec_module(_config)
kernels = tuple(_config.KERNELS)
assert "flaggems" in kernels, kernels
if sys.argv[1] == "boxing":
    assert "boxing" in kernels, kernels
else:
    assert "boxing" not in kernels, kernels
print(f"Kernel sets: {kernels}")
PY

    python - <<'PY'
import hashlib
import importlib.util
from pathlib import Path

lib = Path(importlib.util.find_spec("torch").submodule_search_locations[0]) / "lib"
before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
          for path in lib.iterdir() if path.is_file()}
import torch_fl  # noqa: F401  -- must precede flag_gems; see the cohort check above

import triton

import flag_gems

after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
         for path in lib.iterdir() if path.is_file()}
assert after == before, "MetaX import changed the installed torch/lib"

print(f"Triton: {triton.__version__} ({triton.__file__})")
print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")
PY
  fi
fi
