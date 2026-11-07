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

# Kunlun P800 XPU environment for the common build/integration workflow.
#
# Kunlunxin is a native PrivateUse1 backend built over the vendor XPU SDK: the
# port's own runtime sources under csrc/runtime/accelerator/kunlunxin/ call the
# SDK's libxpurt runtime directly for device discovery, allocation, host/device
# copies and streams, and the compute kernels come from the vendor xdnn op
# library behind the native `kunlunxin` route key. There is no CUDA boxing, so
# nothing here needs a CUDA toolkit -- only the SDK's CUDA-compat header and
# libcudart shim, which back the pinned-pointer probe in csrc/runtime/hooks.h.
#
# The vendor torch package that may be present in the base image is never
# imported, linked or copied into the isolated environment: setup.py locates the
# two xdnn files (libxdnn_pytorch.so and include/xdnn_pytorch/xdnn_pytorch.h) by
# presence, so a stock CPU PyTorch wheel is enough to build and run the backend.
# See docs/vendors/kunlunxin/installation.md.
#
# The isolated venv needs a Triton build carrying the "xpu" backend (flagtree)
# for the XPU kernels. Kunlunxin is not yet in the FlagGems Python platform
# allowlist, so the generated configs/backends_kunlunxin.conf routes nothing to
# FlagGems; the FlagGems wheel is provisioned here the way every other vendor
# hook provisions it, so the fleet keeps one FlagGems release and a single
# --check-build-env contract across platforms.
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
PIP_INDEX_URL_ARG="${TORCH_FL_PIP_INDEX_URL:-$PIP_INDEX_URL_DEFAULT}"

# --- XPU SDK -----------------------------------------------------------------
# XPU_ROOT has to carry both header trees: xpu/runtime.h is the native libxpurt
# header the backend links, and cuda_runtime.h is what the pinned-pointer probe
# compiles against. Both are checked here, with the same pair of conditions
# setup.py::_kunlunxin_roots() applies, so a CUDA-facade-only tree fails at
# setup time rather than at cmake configure.
discover_xpu_root() {
  local candidate
  local -a candidates=("${XPU_ROOT:-}" "/usr/local/xpu")

  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate/include/xpu/runtime.h" &&
          -f "$candidate/include/cuda_runtime.h" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! XPU_ROOT="$(discover_xpu_root)"; then
  echo "::error::XPU SDK not found; expected include/xpu/runtime.h and include/cuda_runtime.h under \$XPU_ROOT or /usr/local/xpu" >&2
  exit 1
fi
export XPU_ROOT
echo "XPU_ROOT=$XPU_ROOT"

# The CUDA-compat shim's soname tracks the SDK's CUDA-compat version rather than
# a fixed one, and it sits either directly in XCUDART_ROOT (the SDK's own so/
# directory on a P800 install) or one level down in lib/, so match the family in
# both spots instead of pinning a filename. This is the only CUDA-shaped build
# input that remains: it carries no kernel and is not reachable from routing.
XCUDART_ROOT="${XCUDART_ROOT:-$XPU_ROOT/so}"
XCUDART_DIR=""
for candidate in "$XCUDART_ROOT" "$XCUDART_ROOT/lib"; do
  if compgen -G "$candidate/libcudart.so*" >/dev/null; then
    XCUDART_DIR="$candidate"
    break
  fi
done
if [[ -z "$XCUDART_DIR" ]]; then
  echo "::error::libcudart.so* was not found under $XCUDART_ROOT or $XCUDART_ROOT/lib" >&2
  exit 1
fi
export XCUDART_ROOT
echo "XCUDART_ROOT=$XCUDART_ROOT"

# The vendor xdnn op library does not ship in the XPU SDK root -- it is
# installed inside the vendor torch package (torch_xmlir/) -- so XDNN_ROOT has
# to be located rather than derived. Same precedence as
# setup.py::_kunlunxin_xdnn_root(): an explicit $XDNN_ROOT is validated and, if
# it does not hold the library, is an error rather than a reason to keep
# searching; then $FLAGOS_VENDOR_TORCH_LIB and its parent; then the vendor torch
# package in site-packages. A missing library is a hard error here because
# FLAGOS_BUILD_VENDOR is pinned on for this platform.
xdnn_root_is_complete() {
  local root="${1:-}"
  [[ -n "$root" ]] || return 1
  [[ -f "$root/include/xdnn_pytorch/xdnn_pytorch.h" ]] || return 1
  [[ -f "$root/libxdnn_pytorch.so" || -f "$root/lib/libxdnn_pytorch.so" ]]
}

if [[ -n "${XDNN_ROOT:-}" ]]; then
  if ! xdnn_root_is_complete "$XDNN_ROOT"; then
    echo "::error::XDNN_ROOT is set but does not contain libxdnn_pytorch.so and include/xdnn_pytorch/xdnn_pytorch.h: $XDNN_ROOT" >&2
    exit 1
  fi
else
  xdnn_candidates=(
    "${FLAGOS_VENDOR_TORCH_LIB:-}"
    "$(dirname "${FLAGOS_VENDOR_TORCH_LIB:-/nonexistent}")"
  )
  # torch_xmlir/ ships inside the image's own venv site-packages (VIRTUAL_ENV is
  # /flagos in the runtime image), not the isolated build venv this hook creates
  # below -- torch_xmlir is deliberately uninstalled from the build interpreter
  # (see the assertion further down). Mirror setup.py::_kunlunxin_xdnn_root()'s
  # site-packages fallback, retargeted at the image venv and so resolved here
  # (before the build venv replaces VIRTUAL_ENV). The trailing python3 probe is
  # the same site.getsitepackages() lookup setup.py performs, for an image that
  # leaves VIRTUAL_ENV unset.
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    for sp in "$VIRTUAL_ENV"/lib/python*/site-packages; do
      xdnn_candidates+=("$sp/torch_xmlir")
    done
  fi
  if sp_site="$(python3 -c 'import site; print("\n".join(p for p in site.getsitepackages() if p))' 2>/dev/null)"; then
    while IFS= read -r sp; do
      if [[ -n "$sp" ]]; then
        xdnn_candidates+=("$sp/torch_xmlir")
      fi
    done <<<"$sp_site"
  fi
  for candidate in "${xdnn_candidates[@]}"; do
    if xdnn_root_is_complete "$candidate"; then
      XDNN_ROOT="$candidate"
      break
    fi
  done
fi
if [[ -z "${XDNN_ROOT:-}" ]]; then
  echo "::error::The vendor xdnn op library was not found. Set XDNN_ROOT to the directory holding libxdnn_pytorch.so and include/xdnn_pytorch/xdnn_pytorch.h (the vendor torch package root torch_xmlir/ is one such directory), or FLAGOS_VENDOR_TORCH_LIB to that directory." >&2
  exit 1
fi
export XDNN_ROOT
echo "XDNN_ROOT=$XDNN_ROOT"

export FLAGOS_ACCELERATOR=kunlunxin
export FLAGOS_BUILD_VENDOR=1
# FLAGOS_BUILD_FLAGGEMS_CPP=0: the XPU SDK ships no FlagGems liboperators.so.
# FLAGOS_BUILD_FLAGGEMS=1 keeps the Python FlagGems dispatch slots compiled in,
# matching this platform's pinned kernel defaults; setup.py turns it on, but
# this variable is applied afterwards and would otherwise switch it back off.
export FLAGOS_BUILD_FLAGGEMS_CPP=0
export FLAGOS_BUILD_FLAGGEMS=1
export FLAGOS_DISABLE_CUDA_ASSETS=1
unset CUDA_HOME 2>/dev/null || true
unset CUDA_PATH 2>/dev/null || true

# The same paths setup.py::_setup_kunlunxin_build_env() prepends for the build
# subprocess, exported here so the wheel finds libxpurt and libxdnn_pytorch.so
# when it is imported and run. The SDK carries no kernel and neither does the
# shim, so neither can re-introduce a CUDA dispatch path.
export CPATH="$XPU_ROOT/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$XCUDART_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$XCUDART_DIR:$XDNN_ROOT:$XDNN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Isolated interpreter ----------------------------------------------------
# 3.10 specifically: the FlagTree XPU wheel is published for cp310 only, so a
# 3.12 venv would simply fail to resolve flagtree at pip time. Prefer the
# interpreter the wheel needs instead of discovering that later.
VENDOR_PYTHON="${TORCH_FL_VENDOR_PYTHON:-}"
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      VENDOR_PYTHON="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$VENDOR_PYTHON" || ! -x "$VENDOR_PYTHON" ]]; then
  echo "::error::Unable to find a Python interpreter to bootstrap the venv"
  exit 1
fi
if ! "$VENDOR_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
  echo "::error::Kunlunxin needs Python 3.10: the FlagTree xpu wheel is cp310-only, and $VENDOR_PYTHON is $("$VENDOR_PYTHON" -V 2>&1)"
  exit 1
fi

PREBUILT_VENV="${TORCH_FL_PREBUILT_KUNLUNXIN_VENV:-/opt/torch-fl-kunlunxin-venv}"
if [[ -z "${TORCH_FL_VENV_ROOT:-}" && -x "$PREBUILT_VENV/bin/python" ]]; then
  VENV_ROOT="$PREBUILT_VENV"
  echo "Using prebuilt Kunlunxin venv: $VENV_ROOT"
else
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-kunlunxin-${CI_STAGE}}"
  "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT" || true
fi

VENV_PYTHON="$VENV_ROOT/bin/python"
if ! venv_is_usable; then
  # Keep the dependency in the chip-specific setup path so the common workflow
  # stays image agnostic; a derived CI image should still bake python3.10-venv
  # in to avoid the apt fallback on every job.
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    python_mm="$($VENDOR_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "python${python_mm}-venv"
    "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"
  fi
fi

if ! venv_is_usable; then
  echo "::error::Isolated Python was not created at $VENV_ROOT; install python3.10-venv in the CI image" >&2
  exit 1
fi

if [[ "$VENV_ROOT" != "$PREBUILT_VENV" ]]; then
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel cmake ninja build
  install_cpu_torch
fi

export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import importlib.util
import os
import sys
import torch

torch_path = Path(torch.__file__).resolve()
assert torch.__version__.split("+", 1)[0] == "2.10.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/usr/local/lib/python3.10/dist-packages" not in str(torch_path), torch_path
assert importlib.util.find_spec("torch_xmlir") is None, (
    "torch_xmlir (the vendor torch package) leaked into the isolated venv; the "
    "de-boxed backend never imports or links it"
)
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"XPU SDK: {os.environ['XPU_ROOT']}")
print(f"xdnn op library root: {os.environ['XDNN_ROOT']}")
PY

# --- Kunlun Triton (flagtree) + FlagGems -------------------------------------
# Installed for both stages, not just integration: build and integration share
# one platform job, and restricting this to CI_STAGE=integration would leave the
# build job's venv without the Triton build that carries the xpu backend.
#
# --no-deps on both source packages so pip cannot replace the pinned CPU torch
# 2.10 with something a transitive requirement prefers.
# An NVIDIA triton wheel must not be present: it carries no "xpu" backend, so
# flag_gems' backend discovery fails and every FlagGems route raises at import.
# flagtree installs its own `triton` package under the same import name without
# registering a `triton` distribution, so remove any real one first --
# repeatedly, because a partially uninstalled triton leaves a dist record that
# keeps the stale files in place.
while "$VENV_PYTHON" -m pip show triton >/dev/null 2>&1; do
  "$VENV_PYTHON" -m pip uninstall -y triton || break
done
"$VENV_PYTHON" -m pip uninstall -y torch_xmlir 2>/dev/null || true

# flagtree is the Triton build carrying the "xpu" backend. The version is local
# to this hook rather than a FLAGTREE_VERSION_kunlunxin row in the pin table:
# only platforms with a wheel segment carry an interpreter pin beside their
# FlagTree pin, and kunlunxin publishes no wheel segment. Adding either row
# without the other breaks the pin table's invariants, and adding both would
# make setup.py declare a cp310-only FlagTree as a runtime requirement of a
# wheel that builds and runs on a stock CPU torch.
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-0.7.0+xpu3.6}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
install_flagtree

# FlagTree is published to flagos-pypi-hosted for every platform; FlagGems sits
# in the per-vendor lane. Select the Kunlunxin lane so this job never needs a
# GitHub source checkout for either.
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-kunlunxin/simple}"
install_flag_gems

# FlagGems' own runtime deps, installed one at a time for the same
# IncompleteRead reason pip_retry exists for. numpy is deliberately not listed:
# the CPU torch wheel does not pull it in, and nothing on this path imports it.
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'packaging>=26.0'
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'PyYAML==6.0.1'
pip_retry --index-url "$PIP_INDEX_URL_ARG" 'sqlalchemy==2.0.48'

# --- Verify the Triton stack imports -----------------------------------------
# flagtree must be the only provider of the `triton` package, and it must carry
# the "xpu" backend the XPU kernels are built for.
#
# flag_gems itself is not imported here. FlagGems 5.x resolves its vendor
# through the torch_fl device surface, which does not exist until the wheel is
# installed, so a bare `import flag_gems` raises before that point. The
# integration environment group does that import once torch_fl is importable.
"$VENV_PYTHON" - <<'PY'
import importlib.util

import triton

assert "xpu" in triton.backends.backends, sorted(triton.backends.backends)
assert importlib.util.find_spec("flag_gems") is not None, "flag_gems is not installed"
print(f"Triton: {triton.__version__} (backends: {sorted(triton.backends.backends)})")
print("flag_gems: installed")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  export_ci_env PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR FLAGOS_BUILD_VENDOR FLAGOS_BUILD_FLAGGEMS_CPP FLAGOS_BUILD_FLAGGEMS FLAGOS_DISABLE_CUDA_ASSETS XPU_ROOT XCUDART_ROOT XDNN_ROOT CPATH LIBRARY_PATH LD_LIBRARY_PATH
fi

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" ]]; then
  # Prebuild so package_data contains the generated libtorch_fl.so before the
  # common wheel workflow stages the final artifact.
  build_flagos_inplace

  python - <<'PY'
import torch_fl

assert torch_fl.flagos.is_available(), "flagos device is unavailable"
print(f"flagos devices: {torch_fl.flagos.device_count()}")
PY
fi
