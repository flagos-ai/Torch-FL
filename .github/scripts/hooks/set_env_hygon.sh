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

CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"

# The runtime owns /flagos and /opt/flagtree; neither is the CI environment.
# Capture its interpreter before DTK env.sh changes PATH, and prevent child
# shells from restoring the image-wide activation after CI selects its venv.
TORCH_FL_VENDOR_PYTHON="${TORCH_FL_VENDOR_PYTHON:-/flagos/bin/python}"
export BASH_ENV=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# Locate the DTK install root. The image may ship DTK under /opt/dtk,
# /opt/dtk-26.04, or a versioned directory; honor an explicit ROCM_PATH,
# then probe common roots, then fall back to a filesystem search.
discover_dtk_root() {
  local candidate
  local -a candidates=(
    "${ROCM_PATH:-}"
    "/opt/dtk"
    "/opt/dtk-26.10"
    "/opt/dtk-26.04"
    "/opt/dtk26.04"
    "/opt/dtk-26.04.4"
    "/opt/dtk-25.04.4"
  )
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate/env.sh" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  local found
  found="$(find /opt /usr/local -maxdepth 3 -type f -name env.sh \
    \( -path '*dtk*' -o -path '*rocm*' \) -print -quit 2>/dev/null | head -n1)"
  if [[ -n "$found" ]]; then
    printf '%s' "$(dirname "$found")"
    return 0
  fi
  return 1
}

if ! ROCM_PATH="$(discover_dtk_root)"; then
  echo "::error::DTK env.sh not found under /opt or /usr/local; set ROCM_PATH explicitly"
  exit 1
fi
echo "ROCM_PATH discovered: $ROCM_PATH"
set +u
# shellcheck disable=SC1091
source "$ROCM_PATH/env.sh"
set -u
export ROCM_PATH
export BASH_ENV=/dev/null
export PYTHONPATH=""
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

select_vendor_python() {
  local candidate="${TORCH_FL_VENDOR_PYTHON:-}"
  if [[ -n "$candidate" && "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" ]]; then
    candidate="$(command -v python 2>/dev/null || true)"
  fi
  if [[ -z "$candidate" || ! -x "$candidate" ]]; then
    echo "::error::Unable to find the vendor Python interpreter" >&2
    exit 1
  fi
  if ! "$candidate" -I -c 'from importlib.metadata import distribution; distribution("torch")' >/dev/null 2>&1; then
    echo "::error::Vendor Python has no torch distribution: $candidate" >&2
    exit 1
  fi
  printf '%s' "$candidate"
}

VENDOR_PYTHON="$(select_vendor_python)"
VENDOR_INFO="$("$VENDOR_PYTHON" -I - <<'PY'
from importlib import metadata
import json
from pathlib import Path
import sys

dist = metadata.distribution("torch")
print(json.dumps({
    "version": dist.version,
    "base_version": dist.version.split("+", 1)[0],
    "root": str(Path(dist.locate_file("torch")).resolve()),
    "python": sys.version.split()[0],
}, sort_keys=True))
PY
)"

readarray -t VENDOR_FIELDS < <(
  VENDOR_INFO="$VENDOR_INFO" "$VENDOR_PYTHON" - <<'PY'
import json
import os

info = json.loads(os.environ["VENDOR_INFO"])
for key in ("version", "base_version", "root", "python"):
    print(info.get(key) or "")
PY
)
VENDOR_TORCH_VERSION="${VENDOR_FIELDS[0]}"
VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[1]}"
VENDOR_TORCH_ROOT="${VENDOR_FIELDS[2]}"
VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[3]}"
VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

# DCU is a CUDA-ABI platform: the DTK torch wheel is a hipified build whose HIP
# kernels are registered under the CUDA dispatch key, so torch_fl reaches them
# through that dispatch key -- with the generated boxing kernels on the boxing
# lane (BOXING_LANE=1), and with the DTK-native routes backends_hygon.conf names
# on the vendor lane. libtorch_hip.so is the DTK analogue of CUDA's
# libtorch_cuda.so; the bundle script stages it (plus the core .so) into
# torch_fl/lib_hygon for a self-contained wheel on both lanes.
if [[ ! -f "$VENDOR_TORCH_LIB/libtorch_hip.so" ]]; then
  echo "::error::$VENDOR_TORCH_LIB has no libtorch_hip.so, not a DTK torch/lib" >&2
  exit 1
fi
export FLAGOS_VENDOR_TORCH_LIB="$VENDOR_TORCH_LIB"
export FLAGOS_VENDOR_TORCH_VERSION="$VENDOR_TORCH_VERSION"

# Legacy images select a matching frontend. The runtime trial below explicitly
# keeps the project's 2.10 frontend while measuring 2.13 device libraries.
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
CPU_TORCH_VERSION="${CPU_TORCH_VERSION%+cpu}"
if [[ "$CPU_TORCH_VERSION" != 2.10.0 || "$VENDOR_TORCH_VERSION" != 2.13.0+dtk2610.* ]]; then
  echo "::error::The DTK 26.10 trial requires CPU Torch 2.10.0 and DTK Torch 2.13.0" >&2
  exit 1
fi
if ! "$VENDOR_PYTHON" -I -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo "::error::The DTK 26.10 runtime cohort requires Python 3.12" >&2
  exit 1
fi
# The existing opt-out allows this explicitly experimental cross-version
# combination to reach the original import/computation gates and hardware
# suite. Those gates below still assert all four CUDA dispatch registrations;
# the bundle's symbol ABI guard remains mandatory and unchanged.
export FLAGOS_DCU_SKIP_RUNTIME_CHECK=1
# A full vendor-core facade is a different configuration and requires a
# matching Python frontend. Do not claim its legacy smoke passed on 2.10.
export FLAGOS_DCU_SKIP_LEGACY_SMOKE=1
echo "::warning::Experimental CPU Torch 2.10 / DTK Torch 2.13 device-library cohort; version-equality gate bypassed, ABI and hardware checks retained"
echo "::warning::Legacy full-vendor-core smoke is excluded from this decoupled-only trial"
# Record only a version actually provided by the sourced SDK. A missing
# SDK version remains unknown rather than being inferred from the image tag.
if [[ "${DTK_VERSION:-}" =~ ^26\.10(\.[0-9]+)*$ ]]; then
  export FLAGOS_SDK_VERSION="$DTK_VERSION"
else
  unset FLAGOS_SDK_VERSION
  echo "::warning::DTK env.sh did not provide a verified DTK_VERSION; SDK metadata remains unknown"
fi

# FlagGems C++ package is optional for DCU (FLAGOS_BUILD_FLAGGEMS_CPP=OFF in setup.py);
# the flaggems runtime path uses the DTK triton hcu backend + Python flag_gems.
# Probe it for CMAKE_PREFIX_PATH but never fail when absent.
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
VENDOR_FLAGGEMS_LIB=""
if [[ -n "$VENDOR_FLAGGEMS_DIR" ]]; then
  VENDOR_FLAGGEMS_LIB="$(cd "$VENDOR_FLAGGEMS_DIR/../.." && pwd)"
  if ! compgen -G "$VENDOR_FLAGGEMS_LIB/liboperators.so*" >/dev/null; then
    echo "::notice::FlagGems C++ liboperators.so not found under $VENDOR_FLAGGEMS_LIB (optional for DCU)"
    VENDOR_FLAGGEMS_LIB=""
  fi
else
  echo "::notice::FlagGems C++ package not found in vendor image (optional for DCU)"
fi

echo "Vendor Python: $VENDOR_PYTHON ($VENDOR_PYTHON_VERSION)"
echo "Vendor PyTorch: $VENDOR_TORCH_VERSION (base $VENDOR_TORCH_BASE_VERSION)"
echo "Vendor torch root: $VENDOR_TORCH_ROOT"
echo "CPU torch target: $CPU_TORCH_VERSION+cpu"

# Isolated venv with the matching CPU-only torch wheel. Unlike CUDA, DCU does
# not stage libtorch_cuda.so into .libtorch_cuda_assets; the DTK forked libtorch
# .so are bundled into torch_fl/lib_hygon by bundle_hygon_libtorch.sh below.
suffix=""
if [[ "${BOXING_LANE:-}" == 1 ]]; then suffix=-boxing; fi
VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-hygon-${CI_STAGE}${suffix}}"
create_runtime_venv "$VENDOR_PYTHON"
"$VENDOR_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
  --target-python "$VENV_PYTHON" \
  --torch-version "$CPU_TORCH_VERSION" \
  --flagtree-version "${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_hygon}" \
  --flaggems-version "${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}" \
  --requirements-file "$REPO_ROOT/.github/requirements-hygon.txt"
export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
ensure_runtime_requirements "$REPO_ROOT/.github/requirements-hygon.txt" "$PIP_INDEX_URL_ARG"
validate_runtime_build_tools
CPU_TORCH_VERSION="$CPU_TORCH_VERSION+cpu"
install_cpu_torch
CPU_TORCH_VERSION="${CPU_TORCH_VERSION%+cpu}"
if [[ "$CI_STAGE" == integration ]]; then
  ensure_runtime_requirement transformers "$PIP_INDEX_URL_ARG"
fi

# The FlagTree wheel supplies the hcu backend and its entry-point metadata;
# do not copy a conflicting Triton provider from the vendor image.
VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# --- Published Hygon wheels --------------------------------------------------
# Installed for both stages, not just integration: build and integration share
# one platform job, and restricting this to CI_STAGE=integration would leave the
# build job's venv without flag_gems -- the wheel then fails as soon as a
# FlagGems route dispatches. Same reasoning as set_env_mthreads.sh.
#
# --no-deps on the wheels prevents their metadata from replacing the pinned
# CPU torch with a vendor wheel or another PyTorch version.
# Reuse a correctly provisioned image. Older images copied the vendor's triton
# without dist-info, so only a matching FlagTree distribution with the hcu
# backend is safe to keep. Otherwise remove the old tree by path before pip
# installs the wheel, and bound the uninstall loop by remaining dist-info.
#
# `pip uninstall -y` exits 0 even when it skips every named package ("WARNING:
# Skipping triton as it is not installed"), so a loop keyed on the pip command
# itself never terminates. It did exactly that on the first run of this
# manifest: the DCU job sat in this line with all output redirected to
# /dev/null for the remainder of its 60-minute budget and was cancelled before
# reaching a single test.
# flagtree is the Triton build carrying the "hcu" (Hygon) backend. 3.6 is not a
# preference but a requirement: current FlagGems uses tl.map_elementwise and
# triton.knobs, which flagtree 0.5.x (Triton 3.1) does not have -- that pair
# fails at import, so the two pins move together.
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_hygon}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
if "$VENV_PYTHON" - "$FLAGTREE_VERSION" <<'PY'
import importlib.metadata as metadata
import sys

try:
    import triton.backends
    assert metadata.version("flagtree") == sys.argv[1]
    assert "hcu" in triton.backends.backends
except (AssertionError, ImportError, metadata.PackageNotFoundError):
    raise SystemExit(1)
PY
then
  echo "Using preinstalled FlagTree $FLAGTREE_VERSION with hcu backend"
else
  # Copies are RECORD-owned wheels in a fresh venv. Remove the FlagTree
  # metadata owner as well as Triton before replacing an unusable provider;
  # otherwise pip can treat the same pinned version as already installed.
  "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
  install_flagtree
fi

HYGON_INDEX_URL="${TORCH_FL_HYGON_INDEX_URL:-$HYGON_INDEX_URL_DEFAULT}"
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="$HYGON_INDEX_URL"

# The release wheel avoids a GitHub clone on every job. The index advertises a
# SHA-256 for each wheel, so pip verifies the download before installing it.
install_flag_gems

# Native Torch-FL extensions are imported explicitly after wheel installation.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" --report-environment

CPU_TORCH_ROOT="$(CPU_TORCH_VERSION="$CPU_TORCH_VERSION" "$VENV_PYTHON" - <<'PY'
import os
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == os.environ["CPU_TORCH_VERSION"], torch.__version__
assert torch.version.cuda is None, torch.version.cuda
PY
)"

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export FLAGOS_ACCELERATOR=dcu
# Both lanes stage the hipified DTK libtorch into the bundle and assert
# libtorch_hip.so/libc10_hip.so: that is the platform's vendor path, not the
# boxing kernels. What BOXING_LANE adds is the generated CUDA boxing kernel set
# (off by default for every accelerator, cmake/flagos_platforms.json), which is
# the explicit opt-in a user makes with FLAGOS_BUILD_BOXING=1.
if [[ "${BOXING_LANE:-}" == "1" ]]; then
  export FLAGOS_BUILD_BOXING=1
  # The day-0 conf a boxing wheel needs (AGENTS.md, "Day-0 boxing confs"):
  # FlagGems where this build has it, boxing for everything else, instead of
  # the default conf's DTK-native routes.
  export FLAGOS_BACKEND_CONFIG="${FLAGOS_BACKEND_CONFIG:-$REPO_ROOT/torch_fl/configs/backends_hygon_boxing.conf}"
  if [[ ! -f "$FLAGOS_BACKEND_CONFIG" ]]; then
    echo "::error::the DCU boxing lane needs the day-0 conf $FLAGOS_BACKEND_CONFIG"
    exit 1
  fi
fi
# FlagGems on DCU goes through the Python (Triton) path, not the C++ wrapped
# one: the flagtree hcu backend lands the kernels on the HIP runtime, which is
# exactly the CUDA-compatible path a kernel library needs here. Mirrors the
# backend selection set_env_mthreads.sh performs -- FLAGOS_BUILD_FLAGGEMS_CPP is the C++
# switch and stays off so nothing tries to rebuild flag_gems
# against the DCU toolchain.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1
export FLAGGEMS_DIR="$VENDOR_FLAGGEMS_DIR"

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"

# libhydmi.so (Hygon DMI device management) ships in the host hyhal driver,
# which dcu.yml bind-mounts into the container at the standard hyhal root.
# DTK's env.sh does not add it to the search path, so flagos device init's
# dlopen(libhydmi.so) fails with "cannot open shared object file". Probe the
# standard roots (honoring an explicit HYHAL_ROOT) and prepend them so the
# loader finds libhydmi.so plus its sibling driver libs (libhydrm, libamd_smi).
HYHAL_LD_PATH=""
for hyhal_root in "${HYHAL_ROOT:-}" /usr/local/hyhal /opt/hyhal; do
  [[ -z "$hyhal_root" ]] && continue
  [[ -d "$hyhal_root/lib" ]] || continue
  HYHAL_LD_PATH="${HYHAL_LD_PATH:+$HYHAL_LD_PATH:}$hyhal_root/lib"
done
# Recreate the host's /opt/hyhal -> /usr/local/hyhal symlink inside the
# container (the flagos image lacks it). dlopen(libhydmi.so) goes through
# LD_LIBRARY_PATH above, but sibling hyhal libs or host code may resolve .so
# by absolute /opt/hyhal/... paths; mirror the host layout so those resolve.
if [[ -d /usr/local/hyhal && ! -e /opt/hyhal ]]; then
  ln -sf /usr/local/hyhal /opt/hyhal 2>/dev/null || \
    echo "::warning::failed to create /opt/hyhal -> /usr/local/hyhal symlink"
fi
if [[ -n "$HYHAL_LD_PATH" ]]; then
  echo "hyhal lib path: $HYHAL_LD_PATH"
else
  echo "::warning::No hyhal/lib found; flagos device init may fail to dlopen libhydmi.so"
fi

export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake${VENDOR_FLAGGEMS_DIR:+:$VENDOR_FLAGGEMS_DIR}${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export LIBRARY_PATH="${HYHAL_LD_PATH:+$HYHAL_LD_PATH:}$CPU_TORCH_ROOT/lib${VENDOR_FLAGGEMS_LIB:+:$VENDOR_FLAGGEMS_LIB}${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${HYHAL_LD_PATH:+$HYHAL_LD_PATH:}$CPU_TORCH_ROOT/lib${VENDOR_FLAGGEMS_LIB:+:$VENDOR_FLAGGEMS_LIB}${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so before python -m build.
  build_flagos_inplace
fi

# Bundle DTK's device .so into torch_fl/lib_hygon for a self-contained wheel
# (replaces CUDA's .libtorch_cuda_assets staging step). Default is the decoupled
# mode: device libraries only, running on the official torch+cpu core installed
# above, with the DTK-private ATen symbols supplied by
# libflagos_dtk_core_compat.so. The script's own ABI guard fails the build if a
# DTK release ever needs more from the vendor core than the shim covers.
# FLAGOS_DCU_VENDOR_CORE=1 selects the full-core bundle and private facade; CI
# smoke-tests that path separately below.
FLAGOS_VENDOR_TORCH_LIB="$FLAGOS_VENDOR_TORCH_LIB" \
  PYTHON="$VENV_PYTHON" bash scripts/vendor/bundle_hygon_libtorch.sh

# Kernel sets are a build input (setup.py records them in
# torch_fl/_build_config.py), so a lane that silently got the other one would run
# the other lane's tests. Asserted here, before the import gate below.
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

# Wheel invariants for the decoupled default. A vendor core .so in lib_hygon means
# the wheel would shadow the official core through RPATH order, i.e. silently
# fall back to the old coupled layout; the shim's absence means the DTK device
# libraries have nothing to satisfy their DTK-private imports with.
if [[ "${FLAGOS_DCU_VENDOR_CORE:-0}" != "1" ]]; then
  python - <<'PY'
from pathlib import Path
import sys

bundle = Path("torch_fl/lib_hygon")
core = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
    "libshm.so",
)
leaked = [name for name in core if (bundle / name).exists()]
if leaked:
    print(f"::error::decoupled DCU bundle contains vendor core libs: {leaked}")
    sys.exit(1)
for required in ("libtorch_hip.so", "libc10_hip.so", "libflagos_dtk_core_compat.so"):
    if not (bundle / required).is_file():
        print(f"::error::{bundle / required} is missing from the DCU bundle")
        sys.exit(1)
print(f"Decoupled DCU bundle verified: {len(list(bundle.glob('*.so*')))} .so, no vendor core")
PY

  # Import gate: the device libraries must bind to the official core and register
  # CUDA-key kernels. _validate_dcu_decoupled_runtime() in torch_fl/__init__.py
  # enforces this, so a regression fails here rather than inside a later test's
  # dispatch. Also checks the torch.cuda shim, which is what triton's hcu backend
  # and therefore all of FlagGems gate on.
  python - <<'PY'
import torch_fl  # noqa: F401  (must precede torch: preload happens at import)
import torch

assert torch.cuda.is_available(), "torch.cuda shim did not activate on DCU"
assert torch.version.hip, f"torch.version.hip not restored: {torch.version.hip!r}"
# triton's autotuner times candidate configs with this, so every FlagGems op that
# autotunes depends on it being constructible (the +cpu wheel ships a dummy).
torch.cuda.Event(enable_timing=True)
for op in ("aten::mm", "aten::add.Tensor", "aten::_softmax", "aten::bmm"):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA"), op
a = torch.randn(64, 64, device="flagos")
b = torch.randn(64, 64, device="flagos")
err = (a @ b).cpu() - (a.cpu() @ b.cpu())
assert err.abs().max().item() < 1e-3, err.abs().max().item()
props = torch.cuda.get_device_properties(0)
print(
    f"Decoupled DCU runtime OK: torch {torch.__version__} hip {torch.version.hip}, "
    f"{props.name} {props.gcnArchName} x{props.multi_processor_count}, "
    f"mm err {err.abs().max().item():.3g}"
)
PY

  # Legacy-mode smoke path (FLAGOS_DCU_VENDOR_CORE=1). It is the documented
  # rollback for anything the decoupled mode cannot express -- DTK-private
  # schemas such as aten::native_fuse_rmsnorm, whose wrappers live in the core
  # fork -- so it must not be allowed to rot. Two properties are checked:
  #
  #   1. Selecting legacy mode against a decoupled bundle fails fast with the
  #      mode-mismatch message, without touching torch/lib.
  #   2. A full legacy bundle selects the vendor core and computes cleanly.
  #
  # Order matters: this runs *after* the decoupled gates because it rebuilds
  # lib_hygon. The last step restores the decoupled bundle, which is what the wheel
  # is built from, and re-runs the invariant check to prove it.
  if [[ "${FLAGOS_DCU_SKIP_LEGACY_SMOKE:-0}" != "1" ]]; then
    echo "Legacy-mode smoke: refusing a decoupled bundle"
    FLAGOS_DCU_VENDOR_CORE=1 python - <<'PY'
try:
    import torch_fl  # noqa: F401
except RuntimeError as exc:
    assert "bundled in decoupled mode" in str(exc), exc
    print(f"Legacy mode correctly refused the decoupled bundle: {exc}")
else:
    raise SystemExit(
        "::error::FLAGOS_DCU_VENDOR_CORE=1 silently accepted a decoupled bundle"
    )
PY

    echo "Legacy-mode smoke: full-core bundle"
    FLAGOS_VENDOR_TORCH_LIB="$FLAGOS_VENDOR_TORCH_LIB" \
      PYTHON="$VENV_PYTHON" FLAGOS_DCU_VENDOR_CORE=1 \
      bash scripts/vendor/bundle_hygon_libtorch.sh
    FLAGOS_DCU_VENDOR_CORE=1 python - <<'PY'
import hashlib
import importlib.util
from pathlib import Path

lib = Path(importlib.util.find_spec("torch").submodule_search_locations[0]) / "lib"
before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
          for path in lib.iterdir() if path.is_file()}
import torch_fl  # noqa: F401  (selects DTK's core without changing torch/lib)
import torch

from torch_fl.accelerator.hygon._dcu_libtorch_link import restore_original_libtorch

try:
    assert torch.version.hip, f"torch.version.hip not restored: {torch.version.hip!r}"
    a = torch.randn(64, 64, device="flagos")
    b = torch.randn(64, 64, device="flagos")
    err = (a @ b).cpu() - (a.cpu() @ b.cpu())
    assert err.abs().max().item() < 1e-3, err.abs().max().item()
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in lib.iterdir() if path.is_file()}
    assert after == before, "DCU vendor-core import changed the installed torch/lib"
    print(
        f"Legacy DCU runtime OK: torch {torch.__version__} hip {torch.version.hip}, "
        f"mm err {err.abs().max().item():.3g}"
    )
finally:
    # Kept for older callers. The private facade leaves the installed wheel intact.
    restore_original_libtorch()
PY
    CPU_TORCH_ROOT="$CPU_TORCH_ROOT" python - <<'PY'
import os
from pathlib import Path
import sys

lib = Path(os.environ["CPU_TORCH_ROOT"]) / "lib"
links = sorted(entry.name for entry in os.scandir(lib) if entry.is_symlink())
if links or (lib / "_orig_backup").exists():
    print(f"::error::legacy smoke left {lib} modified: {links}")
    sys.exit(1)
print(f"Legacy smoke left the installed torch/lib unchanged: {lib}")
PY

    echo "Restoring the decoupled bundle for the wheel"
    FLAGOS_VENDOR_TORCH_LIB="$FLAGOS_VENDOR_TORCH_LIB" \
      PYTHON="$VENV_PYTHON" bash scripts/vendor/bundle_hygon_libtorch.sh
    python - <<'PY'
from pathlib import Path
import sys

bundle = Path("torch_fl/lib_hygon")
leaked = [
    name
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch.so", "libtorch_python.so")
    if (bundle / name).exists()
]
if leaked or not (bundle / "libflagos_dtk_core_compat.so").is_file():
    print(f"::error::decoupled bundle not restored after the legacy smoke: {leaked}")
    sys.exit(1)
print("Decoupled bundle restored")
PY
  fi
fi

# --- Verify the FlagGems stack imports ---------------------------------------
# Integration only: torch_fl._C does not exist until the wheel is built, and the
# import below needs it. set_env_hygon.sh runs before the wheel build, so torch_fl
# is not importable at that stage.
#
# torch_fl is imported before flag_gems on purpose. It installs the torch.cuda
# shim that makes this box look like the CUDA-compatible device FlagGems expects;
# importing flag_gems first would resolve the vendor against a stock torch and
# pick the wrong backend.
#
# The assertions cover the three ways this provisioning can silently regress:
# the flagtree wheel failing to replace the deleted vendor triton, the hcu
# backend not being discoverable, and FlagGems falling back to a non-Hygon
# vendor. Each of those would otherwise surface as a confusing per-op failure
# deep inside the FlagGems test group.
if "$VENV_PYTHON" -c "import torch_fl._C" 2>/dev/null; then
  "$VENV_PYTHON" - <<'PY'
import torch_fl  # noqa: F401  -- must precede flag_gems; installs the torch.cuda shim

import triton
import triton.backends
import flag_gems

assert "hcu" in triton.backends.backends, sorted(triton.backends.backends)
assert flag_gems.vendor_name == "hygon", flag_gems.vendor_name
print(f"Triton: {triton.__version__} (backends: {sorted(triton.backends.backends)})")
print(f"FlagGems: {flag_gems.__version__} (vendor: {flag_gems.vendor_name})")
PY
fi

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "::error::rocm-smi is unavailable"
  exit 1
fi
rocm-smi

python - <<'PY'
import os
from pathlib import Path
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
assert Path("torch_fl/lib/libtorch_fl.so").is_file()
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"DCU torch lib: {os.environ.get('FLAGOS_VENDOR_TORCH_LIB', '?')}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  export_ci_env BASH_ENV FLAGOS_VENDOR_TORCH_VERSION FLAGOS_DCU_SKIP_RUNTIME_CHECK FLAGOS_DCU_SKIP_LEGACY_SMOKE
  if [[ -n "${FLAGOS_SDK_VERSION:-}" ]]; then
    export_ci_env FLAGOS_SDK_VERSION
  fi

  # The boxing lane assigns these; the vendor lane leaves them unset, and
  # export_ci_env reads every name under `set -u`. Assign the empty default here
  # so both lanes persist an explicit value.
  : "${FLAGOS_BUILD_BOXING:=}"
  : "${FLAGOS_BACKEND_CONFIG:=}"
  export_ci_env PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR FLAGOS_BUILD_BOXING FLAGOS_BACKEND_CONFIG ROCM_PATH FLAGOS_VENDOR_TORCH_LIB FLAGGEMS_DIR FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS TORCH_DEVICE_BACKEND_AUTOLOAD CMAKE_PREFIX_PATH LIBRARY_PATH LD_LIBRARY_PATH
fi
