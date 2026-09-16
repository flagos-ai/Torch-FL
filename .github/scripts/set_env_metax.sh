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

# Retries are deliberate: the flagtree wheel is ~180MB and the shared mirror can
# close a large-wheel response early (IncompleteRead) even though the package is
# there. Retrying just the failed package beats restarting all of setup.
pip_retry() {
  local attempt=1
  while true; do
    if python -m pip install --retries 10 --timeout 300 --no-cache-dir "$@"; then
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

export PATH="/opt/venv/bin:/opt/maca/tools/cu-bridge/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/bin:$PATH"
export VIRTUAL_ENV=/opt/venv
export PYTHONNOUSERSITE=1

export ACCELERATOR=metax
export METAX_PATH=/opt/maca
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca

export FLAGOS_METAX_BOXING=1
export FLAGOS_METAX_CUDART_SHIM=1
export FLAGOS_DISABLE_CUDA_ASSETS=1
# Which op takes which backend is stated in backends_metax.conf, not here: that
# file is full-coverage and lists all five keys per op
# (flaggems_cpp > flaggems > tileops > cuda), and _select_backend_config() picks
# it from FLAGOS_METAX_BOXING alone. FLAGOS_USE_FLAGGEMS used to select a
# separate backends_flaggems.conf and no longer selects anything, so setting it
# here would misdescribe the build -- the FlagGems Python path is on for the 592
# ops the conf routes to it either way. It is still a test gate
# (tests/integration/ops/conftest.py:_flaggems_enabled), which is why the
# @pytest.mark.flaggems group in .github/configs/metax.yml sets it on its own
# command line rather than globally: that group is the measurement, and the
# rest of the suite is not about the runtime switch.
export FLAGGEMS_KERNEL=0
export FLAGGEMS_PYTHON=1
export FLAGOS_WHEEL_LOCAL=metax3.8.0
export FLAGOS_MACA_TORCH_LIB=/opt/vendor-libtorch/lib

export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/mxshmem/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="/opt/maca/lib:/opt/maca/tools/cu-bridge/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CPATH="/opt/maca/tools/cu-bridge/include:/opt/maca/include:/opt/maca/include/mcr${CPATH:+:$CPATH}"

for path in \
  /opt/venv/bin/python \
  /opt/maca/tools/cu-bridge/bin/cucc \
  /opt/vendor-libtorch/lib/libc10.so \
  /opt/vendor-libtorch/lib/libtorch_cpu.so \
  /opt/vendor-libtorch/lib/libtorch.so \
  /opt/vendor-libtorch/lib/libtorch_global_deps.so \
  /opt/vendor-libtorch/lib/libtorch_python.so \
  /opt/vendor-libtorch/lib/libc10_cuda.so \
  /opt/vendor-libtorch/lib/libtorch_cuda.so \
  /opt/vendor-libtorch/lib/libtorch_cuda_linalg.so; do
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
# (FLAGGEMS_PYTHON=1 above compiles the dispatcher slot).
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
  python -m pip install --no-deps 'packaging>=20.0' 'PyYAML>=5.0' 'sqlalchemy>=1.4' 'numpy>=1.20,<2.0'

  # --- FlagTree: the Triton build carrying the "metax" backend ----------------
  #
  # 0.6.1 is the newest MetaX build on the FlagOS index and the one every number
  # in .github/configs/metax.yml and docs/vendors/metax was measured on. 3.6 is
  # not a preference: FlagGems uses tl.map_elementwise and triton.knobs, which
  # flagtree 0.5.x (Triton 3.1) does not have -- the two pins move together.
  # Same install as set_env_musa.sh, from the same index.
  FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.6.1+metax3.6}"
  FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
  pip_retry --no-deps --index-url "$FLAGTREE_INDEX_URL" "flagtree===$FLAGTREE_VERSION"

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

  # --- FlagGems --------------------------------------------------------------
  #
  # FlagGems from the flagos-ai fork, tracking master by policy: every CI run
  # measures the current master, not a pinned snapshot. Override with
  # TORCH_FL_FLAGGEMS_REVISION to pin a commit for a reproducible run.
  #
  # TEMPORARY PIN -- revert the default to `master` once upstream fixes
  # flagos-ai/FlagGems: d312aa02 (2026-09-16) added
  # ("argsort.stable", argsort_stable) to the module-level _FULL_CONFIG in
  # flag_gems/__init__.py, but argsort_stable is only defined by the kunlunxin
  # backend package, so `import flag_gems` raises
  # NameError: name 'argsort_stable' is not defined on every other vendor.
  # 437ba393 is the last good master (d312aa02's parent).
  #
  # A generated kernel calls its operator by the package-level name
  # (`flag_gems.<name>`, see scripts/codegen/codegen_ops.py) so a name only
  # resolves if the installed FlagGems defines it. The cohort-gap probe below
  # (gems_cohort_gap) enforces that against the installed revision on every
  # run: if it drops a name the checked-in kernels call, this fails naming the
  # gap instead of erroring per op at dispatch.
  FLAGGEMS_REVISION="${TORCH_FL_FLAGGEMS_REVISION:-437ba39387ddc681dc884259ef9dbf0c1802bccc}"
  FLAGGEMS_REPO="${TORCH_FL_FLAGGEMS_REPO:-https://github.com/flagos-ai/FlagGems.git}"

  # Discover flag_gems via interpreter query, not directory probe. FlagGems is
  # normally an editable install, so there is no site-packages/flag_gems directory
  # to test -- only a .pth file and a finder module pointing at a source tree.
  VENDOR_FLAGGEMS_PYTHON=""
  VENDOR_FLAGGEMS_ROOT=""
  for candidate_python in /opt/conda/bin/python3 /opt/conda/bin/python \
                          /opt/vendor-torch/bin/python3 /opt/vendor-torch/bin/python \
                          /usr/bin/python3 /usr/local/bin/python3; do
    [[ -x "$candidate_python" ]] || continue
    VENDOR_FLAGGEMS_ROOT="$("$candidate_python" - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("flag_gems")
if spec is None or not spec.submodule_search_locations:
    print("")
else:
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    print(root if (root / "__init__.py").is_file() else "")
PY
)"
    if [[ -n "$VENDOR_FLAGGEMS_ROOT" ]]; then
      VENDOR_FLAGGEMS_PYTHON="$candidate_python"
      break
    fi
  done

  # The image's copy is a fast path, not an assumption: it is kept only if it
  # resolves every name the checked-in kernels call. The version string is not
  # usable as the test -- the stale cohort measured on 2026-09-15 reports
  # 0.0.0, and a cohort that claims the right version is still only useful if
  # the names are there.
  VENDOR_GAP=""
  if [[ -n "$VENDOR_FLAGGEMS_ROOT" ]]; then
    VENDOR_GAP="$(gems_cohort_gap "$VENDOR_FLAGGEMS_PYTHON" || true)"
    if [[ -z "$VENDOR_GAP" ]]; then
      echo "::warning::Could not probe the image's FlagGems cohort with" \
           "$VENDOR_FLAGGEMS_PYTHON; treating it as unusable and installing the" \
           "pinned revision."
    fi
  fi

  if [[ -n "$VENDOR_GAP" && "$VENDOR_GAP" == "0/"* ]]; then
    echo "Image FlagGems cohort resolves all $VENDOR_GAP kernel names ($VENDOR_FLAGGEMS_ROOT)"
  else
    if [[ -n "$VENDOR_FLAGGEMS_ROOT" ]]; then
      echo "::warning::The image's FlagGems at $VENDOR_FLAGGEMS_ROOT does not" \
           "resolve $VENDOR_GAP of the names csrc/aten/generated/" \
           "flaggems_python_kernels.cc calls; installing @$FLAGGEMS_REVISION" \
           "into the venv instead."
    else
      echo "FlagGems not found in vendor interpreters. Installing from source..."
    fi

    # FlagGems is not available on PyPI. Install from GitHub.
    pip_retry --no-deps "git+${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"

    # After installation, resolve the package location in the venv itself
    VENDOR_FLAGGEMS_ROOT="$(python - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("flag_gems")
if spec is None or not spec.submodule_search_locations:
    print("")
else:
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    print(root if (root / "__init__.py").is_file() else "")
PY
)"

    if [[ -z "$VENDOR_FLAGGEMS_ROOT" ]]; then
      echo "::error::Failed to install FlagGems from source. The MetaX backend" \
           "requires FlagGems because backends_metax.conf routes 592 ops to the" \
           "Python FlagGems path."
      exit 1
    fi
  fi

  # Link the resolved flag_gems root (if from vendor) so the venv imports the
  # same tree the probe above approved; a venv install is already in place.
  if [[ ! -e "$VENV_SITE/flag_gems" && "$VENDOR_FLAGGEMS_ROOT" != "$VENV_SITE"* ]]; then
    ln -s "$VENDOR_FLAGGEMS_ROOT" "$VENV_SITE/flag_gems"
  fi
  # Link dist-info metadata if it exists alongside the package (for non-editable installs)
  VENDOR_FLAGGEMS_PARENT="$(dirname "$VENDOR_FLAGGEMS_ROOT")"
  for metadata in "$VENDOR_FLAGGEMS_PARENT"/flag_gems-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    [[ -e "$VENV_SITE/$(basename "$metadata")" ]] || ln -s "$metadata" "$VENV_SITE/"
  done

  # The cohort check is deliberately not here: it has to import flag_gems, which
  # needs torch_fl first, which needs the native build. It runs at the end of
  # this script instead. See the comment there.
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
    VIRTUAL_ENV PYTHONNOUSERSITE ACCELERATOR METAX_PATH MACA_PATH MACA_HOME \
    FLAGOS_METAX_BOXING FLAGOS_METAX_CUDART_SHIM \
    FLAGOS_DISABLE_CUDA_ASSETS \
    FLAGGEMS_KERNEL FLAGGEMS_PYTHON FLAGOS_WHEEL_LOCAL \
    FLAGOS_MACA_TORCH_LIB LD_LIBRARY_PATH LIBRARY_PATH CPATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
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
#     bundle_maca_libtorch.sh just wrote, or FLAGOS_MACA_TORCH_LIB), and it has
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
