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

# Shared version pins (torch, FlagTree, FlagGems); see .github/version-pins.env.
# shellcheck source=.github/version-pins.env
source "${REPO_ROOT}/.github/version-pins.env"

# Shared set_env helpers (pip_retry, FlagGems install, path stripping).
# shellcheck source=.github/scripts/lib/set_env_common.sh
source "${REPO_ROOT}/.github/scripts/lib/set_env_common.sh"

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
    # Must precede flag_gems: torch_fl points the stock +cpu wheel's torch/lib at
    # the MetaX libtorch, and without that the MetaX FlagTree Triton resolves no
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

# The published MACA 3.8.1.3 build image carries vendor torch in /flagos,
# whereas the older CI image carries a prebuilt CPU venv in /opt/venv. Prepare
# the same isolated layout when the image is upgraded; keep vendor libtorch as
# a link input and never import its Python package in the build/test venv.
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
  PIP_RETRY_PYTHON=/opt/venv/bin/python pip_retry --index-url "$CPU_TORCH_INDEX_URL_DEFAULT" \
    "torch==$CPU_TORCH_VERSION_DEFAULT"
  PIP_RETRY_PYTHON=/opt/venv/bin/python pip_retry --index-url "$PIP_INDEX_URL_DEFAULT" \
    build cmake ninja setuptools wheel pytest patchelf
  export FLAGOS_WHEEL_LOCAL=metax3.8.1.3
  export PYTHONPATH=""
else
  export FLAGOS_VENDOR_TORCH_LIB=/opt/vendor-libtorch/lib
  export FLAGOS_WHEEL_LOCAL=metax3.8.0
fi
export PATH="/opt/venv/bin:/opt/maca/tools/cu-bridge/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/bin:$PATH"
export VIRTUAL_ENV=/opt/venv
export PYTHONNOUSERSITE=1

export FLAGOS_ACCELERATOR=metax
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca

# MetaX is a boxing-only build: no native mxcc kernels compile in, so
# FLAGOS_BUILD_VENDOR=OFF is the one build-side statement. The export above is a
# build input -- setup.py records it in torch_fl/_build_config.py, and the runtime
# derives its conf from that record (torch_fl._select_backend_config), so there is
# no mode variable to keep in agreement with the build any more.
export FLAGOS_BUILD_VENDOR=OFF
export FLAGOS_METAX_CUDART_SHIM=1
export FLAGOS_DISABLE_CUDA_ASSETS=1
# Which op takes which backend is stated in backends_metax.conf, not here: that
# file is full-coverage and lists all five keys per op
# (flaggems_cpp > flaggems > tileops > cuda), selected from the accelerator
# record. The retired FLAGOS_USE_FLAGGEMS switch
# used to select a separate backends_flaggems.conf; nothing reads it any more,
# so setting it here would misdescribe the build -- the FlagGems Python path is
# on for the 592 ops the conf routes to it either way.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1

export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/mxshmem/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="/opt/maca/tools/cu-bridge/include:/opt/maca/include:/opt/maca/include/mcr${CPATH:+:$CPATH}"

for path in \
  /opt/venv/bin/python \
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

python - <<'PY'
from pathlib import Path

import torch

torch_path = Path(torch.__file__).resolve()
assert torch.__version__.startswith("2.10.0+cpu"), torch.__version__
assert str(torch_path).startswith("/opt/venv/"), torch_path
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
# backends_metax.conf routes 592 ops to the Python FlagGems path by default
# (FLAGOS_BUILD_FLAGGEMS=1 above compiles the dispatcher slot).
#
# Both are installed into the venv rather than linked out of the image, so the
# packages the tests import are the ones this script put there. Linking is what
# let a stale image cohort shadow the pinned revision: the venv resolved
# whatever the image's editable install pointed at, and the pin below never ran.
if [[ "$CI_STAGE" == "integration" ]]; then
  VENV_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

  # Install FlagGems' runtime dependencies into the venv. flag_gems imports
  # packaging, yaml (PyYAML), sqlalchemy and numpy at or shortly after import.
  # Use --no-deps to protect the torch ABI (same rationale as test-dependencies).
  # numpy<2 because numpy 2.x breaks the stock +cpu torch C extensions at import
  # (documented in set_env_musa.sh:176-178).
  python -m pip install --no-deps 'packaging>=26.0' 'PyYAML==6.0.1' 'sqlalchemy==2.0.48' 'numpy>=1.20,<2.0'

  # --- FlagTree: the Triton build carrying the "metax" backend ----------------
  #
  # The MetaX wheel supplies the backend used by FlagGems. Install only a
  # published binary so CI cannot fall back to a source build.
  FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_metax}"
  FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
  pip_retry --no-deps --only-binary=:all: --index-url "$FLAGTREE_INDEX_URL" "flagtree===$FLAGTREE_VERSION"

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

  # The published FlagCX wheel requires MACA 3.8.1.3. The current 3.8.0 CI
  # image remains in use until the runner driver is confirmed compatible.
  if [[ "$FLAGOS_VENDOR_TORCH_LIB" == /flagos/* ]]; then
    FLAGCX_VERSION="${TORCH_FL_FLAGCX_VERSION:-$FLAGCX_VERSION_metax}"
    pip_retry --no-deps --only-binary=:all: --index-url "$FLAGGEMS_INDEX_URL" \
      "flagcx===$FLAGCX_VERSION"
    export FLAGCX_TORCH_BACKEND=flagos
    # The extension is built after provisioning, so defer PyTorch's FlagCX
    # entry-point import until later workflow steps.
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  fi
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' \
    /opt/venv/bin \
    /opt/maca/tools/cu-bridge/bin \
    /opt/maca/mxgpu_llvm/bin \
    /opt/maca/bin >> "$GITHUB_PATH"
fi

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf '%s=%s\n' PATH "$PATH" >> "$GITHUB_ENV"
  for name in \
    VIRTUAL_ENV PYTHONNOUSERSITE FLAGOS_ACCELERATOR MACA_PATH MACA_HOME \
    FLAGOS_BUILD_VENDOR FLAGOS_METAX_CUDART_SHIM \
    FLAGOS_DISABLE_CUDA_ASSETS \
    FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS FLAGOS_WHEEL_LOCAL \
    FLAGOS_VENDOR_TORCH_LIB LD_LIBRARY_PATH LIBRARY_PATH CPATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
  if [[ -n "${FLAGCX_TORCH_BACKEND:-}" ]]; then
    printf 'FLAGCX_TORCH_BACKEND=%s\n' "$FLAGCX_TORCH_BACKEND" >> "$GITHUB_ENV"
  fi
fi

cd "$REPO_ROOT"

if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # setuptools collects package_data before build_ext on the first wheel build.
  # Prebuilding makes torch_fl/lib/*.so available when the common workflow
  # packages the local wheel for either stage. The following python -m build
  # is incremental.
  python setup.py build_ext --inplace
fi

# Populate the ignored package-data directory after the optional native
# prebuild. This also normalizes the native RPATHs before wheel/editable install.
bash scripts/vendor/bundle_maca_libtorch.sh

# --- FlagGems cohort ---------------------------------------------------------
#
# Confirm the vendor packages actually import against the CPU torch wheel, and
# that the FlagGems the venv resolves is the one that will answer dispatch.
# Failing here names the gap once instead of as a dispatch error per op.
#
# This runs after the native build, and not with the FlagGems install above,
# because on MetaX flag_gems cannot be imported by a process that has not
# imported torch_fl first -- set_env_musa.sh documents the same ordering for
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
#     Importing torch_fl is what removes that condition: it points the stock
#     wheel's torch/lib at the MetaX libtorch (the bundle
#     bundle_maca_libtorch.sh just wrote, or FLAGOS_VENDOR_TORCH_LIB), and it has
#     to happen before `import torch` -- which is exactly the order the probe
#     uses. The relink is on disk, so it holds for every later process, the
#     tests included.
#
# torch_fl._C is what makes torch_fl importable at all, and build_ext has just
# built it. The gate mirrors set_env_musa.sh's `import torch_fl._C` guard: on a
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
           "imported first; the 592 FlagGems routes in backends_metax.conf have" \
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

    python - <<'PY'
import torch_fl  # noqa: F401  -- must precede flag_gems; see the cohort check above

import triton

import flag_gems

print(f"Triton: {triton.__version__} ({triton.__file__})")
print(f"FlagGems: {flag_gems.__version__} ({flag_gems.__file__})")
PY
  fi
fi
