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
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

# Locate the DTK install root. The image may ship DTK under /opt/dtk,
# /opt/dtk-26.04, or a versioned directory; honor an explicit DTK_ROOT or
# ROCM_PATH, then probe common roots, then fall back to a filesystem search.
discover_dtk_root() {
  local candidate
  local -a candidates=(
    "${DTK_ROOT:-}"
    "${ROCM_PATH:-}"
    "/opt/dtk"
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

if ! DTK_ROOT="$(discover_dtk_root)"; then
  echo "::error::DTK env.sh not found under /opt or /usr/local; set DTK_ROOT explicitly"
  exit 1
fi
echo "DTK_ROOT discovered: $DTK_ROOT"
set +u
# shellcheck disable=SC1091
source "$DTK_ROOT/env.sh"
set -u
export DTK_ROOT
export ROCM_PATH="${ROCM_PATH:-$DTK_ROOT}"

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
  if ! "$candidate" -c "import torch" >/dev/null 2>&1; then
    echo "::error::Vendor Python cannot import torch: $candidate (was DTK env.sh sourced?)" >&2
    exit 1
  fi
  printf '%s' "$candidate"
}

VENDOR_PYTHON="$(select_vendor_python)"
VENDOR_INFO="$("$VENDOR_PYTHON" - <<'PY'
import json
from pathlib import Path
import site
import sys

import torch

print(json.dumps({
    "version": torch.__version__,
    "base_version": torch.__version__.split("+", 1)[0],
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
for key in ("version", "base_version", "root", "site", "python"):
    print(info.get(key) or "")
PY
)
VENDOR_TORCH_VERSION="${VENDOR_FIELDS[0]}"
VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[1]}"
VENDOR_TORCH_ROOT="${VENDOR_FIELDS[2]}"
VENDOR_SITE="${VENDOR_FIELDS[3]}"
VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[4]}"
VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

# DCU is a boxing build: the DTK torch wheel is a hipified build whose HIP
# kernels are registered under the CUDA dispatch key, so torch_fl reaches them
# via PrivateUse1->CUDA boxing kernels without shipping its own kernels.
# libtorch_hip.so is the DTK analogue of CUDA's libtorch_cuda.so; the bundle
# script stages it (plus the core .so) into torch_fl/lib_dcu for a
# self-contained wheel.
if [[ ! -f "$VENDOR_TORCH_LIB/libtorch_hip.so" ]]; then
  echo "::error::$VENDOR_TORCH_LIB has no libtorch_hip.so, not a DTK torch/lib" >&2
  exit 1
fi
export FLAGOS_DCU_TORCH_LIB="$VENDOR_TORCH_LIB"

# CPU torch base version must match the vendor's for ABI compatibility. The
# DTK wheel version (e.g. 2.10.0+das on the current CI image) may differ across
# DTK releases and from the CUDA/MetaX images, so derive the target from the
# vendor wheel instead of hard-coding it.
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$VENDOR_TORCH_BASE_VERSION}"

# FlagGems C++ package is optional for DCU (FLAGGEMS_KERNEL=OFF in setup.py);
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
# .so are bundled into torch_fl/lib_dcu by bundle_dcu_libtorch.sh below.
PREBUILT_VENV="${TORCH_FL_PREBUILT_DCU_VENV:-/opt/torch-fl-dcu-venv}"
if [[ -z "${TORCH_FL_VENV_ROOT:-}" && -x "$PREBUILT_VENV/bin/python" ]]; then
  VENV_ROOT="$PREBUILT_VENV"
  echo "Using prebuilt DCU venv: $VENV_ROOT"
else
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-dcu-${CI_STAGE}}"
  if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
    echo "::warning::Vendor Python cannot create a venv; trying uv"
    if ! command -v uv >/dev/null 2>&1; then
      "$VENDOR_PYTHON" -m pip install --upgrade uv
    fi
    uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
  fi
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

if [[ "$VENV_ROOT" != "$PREBUILT_VENV" ]]; then
  # patchelf is required by setup.py (rewrites _C.so RPATH to $ORIGIN/lib and
  # $ORIGIN/lib_dcu after copy) and by bundle_dcu_libtorch.sh (sets RPATH on the
  # bundled DTK .so). The pinned CI image ships it at /usr/bin/patchelf; the pip
  # install here is a redundant fallback in case the image lacks it.
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel cmake build patchelf
  "$VENV_PYTHON" -m pip install \
    --index-url "$CPU_TORCH_INDEX_URL" \
    "torch==$CPU_TORCH_VERSION"
  if [[ "$CI_STAGE" == "integration" ]]; then
    # Pin numpy<2: the CPU torch wheel is built against the NumPy 1.x ABI, and
    # transformers pulls numpy 2.x which breaks torch's C extensions at import.
    "$VENV_PYTHON" -m pip install pytest transformers "numpy<2"
  fi
fi

# Carry the vendor FlagGems/FlagCX Python packages into the venv so the DTK
# triton (hcu backend) and flag_gems runtime path work under the CPU torch.
VENV_SITE="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for package in flag_gems triton triton_kernels flagcx sqlalchemy; do
  if [[ -d "$VENDOR_SITE/$package" ]]; then
    cp -a "$VENDOR_SITE/$package" "$VENV_SITE/"
  fi
  for metadata in "$VENDOR_SITE"/"$package"-*.dist-info; do
    [[ -e "$metadata" ]] || continue
    cp -a "$metadata" "$VENV_SITE/"
  done
done

CPU_TORCH_ROOT="$(CPU_TORCH_VERSION="$CPU_TORCH_VERSION" "$VENV_PYTHON" - <<'PY'
import os
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == os.environ["CPU_TORCH_VERSION"], torch.__version__
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
export ACCELERATOR=dcu
# DTK's Triton (hcu backend) is copied into the venv without its dist-info
# (the vendor image ships none), so Triton's entry-point backend discovery
# finds nothing and the hcu backend never loads. Force in-tree loading so the
# backend under triton/backends is used -- required for the FlagGems runtime
# path (integration [3/5]). See README "Enabling FlagGems on DCU".
export TRITON_BACKENDS_IN_TREE=1
export FLAGGEMS_DIR="$VENDOR_FLAGGEMS_DIR"
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"

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
  python setup.py build_ext --inplace
fi

# Bundle DTK's device .so into torch_fl/lib_dcu for a self-contained wheel
# (replaces CUDA's .libtorch_cuda_assets staging step). Default is the decoupled
# mode: device libraries only, running on the official torch+cpu core installed
# above, with the DTK-private ATen symbols supplied by
# libflagos_dtk_core_compat.so. The script's own ABI guard fails the build if a
# DTK release ever needs more from the vendor core than the shim covers.
# FLAGOS_DCU_VENDOR_CORE=1 selects the legacy full-core bundle + relink; CI
# smoke-tests that path separately below.
DTK_ROOT="$DTK_ROOT" FLAGOS_DCU_TORCH_LIB="$FLAGOS_DCU_TORCH_LIB" \
  PYTHON="$VENV_PYTHON" bash scripts/bundle_dcu_libtorch.sh

# Wheel invariants for the decoupled default. A vendor core .so in lib_dcu means
# the wheel would shadow the official core through RPATH order, i.e. silently
# fall back to the old coupled layout; the shim's absence means the DTK device
# libraries have nothing to satisfy their DTK-private imports with.
if [[ "${FLAGOS_DCU_VENDOR_CORE:-0}" != "1" ]]; then
  python - <<'PY'
from pathlib import Path
import sys

bundle = Path("torch_fl/lib_dcu")
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
  #      mode-mismatch message, before mutating torch/lib.
  #   2. A full legacy bundle relinks, computes, and restores cleanly.
  #
  # Order matters: this runs *after* the decoupled gates because it rebuilds
  # lib_dcu. The last step restores the decoupled bundle, which is what the wheel
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
    DTK_ROOT="$DTK_ROOT" FLAGOS_DCU_TORCH_LIB="$FLAGOS_DCU_TORCH_LIB" \
      PYTHON="$VENV_PYTHON" FLAGOS_DCU_VENDOR_CORE=1 \
      bash scripts/bundle_dcu_libtorch.sh
    FLAGOS_DCU_VENDOR_CORE=1 python - <<'PY'
import torch_fl  # noqa: F401  (relinks torch/lib, then preloads DTK's core)
import torch

from torch_fl.accelerator.dcu._dcu_libtorch_link import restore_original_libtorch

try:
    assert torch.version.hip, f"torch.version.hip not restored: {torch.version.hip!r}"
    a = torch.randn(64, 64, device="flagos")
    b = torch.randn(64, 64, device="flagos")
    err = (a @ b).cpu() - (a.cpu() @ b.cpu())
    assert err.abs().max().item() < 1e-3, err.abs().max().item()
    print(
        f"Legacy DCU runtime OK: torch {torch.__version__} hip {torch.version.hip}, "
        f"mm err {err.abs().max().item():.3g}"
    )
finally:
    # Leave the venv's torch/lib as we found it: the symlinks point into a bundle
    # that is about to be replaced by the decoupled one.
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
print(f"Legacy smoke rolled back cleanly: no symlinks or backup in {lib}")
PY

    echo "Restoring the decoupled bundle for the wheel"
    DTK_ROOT="$DTK_ROOT" FLAGOS_DCU_TORCH_LIB="$FLAGOS_DCU_TORCH_LIB" \
      PYTHON="$VENV_PYTHON" bash scripts/bundle_dcu_libtorch.sh
    python - <<'PY'
from pathlib import Path
import sys

bundle = Path("torch_fl/lib_dcu")
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
print(f"DCU torch lib: {os.environ.get('FLAGOS_DCU_TORCH_LIB', '?')}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR DTK_ROOT ROCM_PATH \
    FLAGOS_DCU_TORCH_LIB FLAGGEMS_DIR FLAGCX_PATH TRITON_BACKENDS_IN_TREE \
    CMAKE_PREFIX_PATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
