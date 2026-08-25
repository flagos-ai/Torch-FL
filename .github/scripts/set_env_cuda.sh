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
CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-2.9.0}"
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
# External libtorch_cuda.so is sourced from a version-matched cu130 wheel
# (downloaded, NOT installed) so the CPU-only venv pairs with a 2.9.0 CUDA
# build instead of the image's torch. See
# docs/vendors/cuda/external-libtorch-cuda.md (constraint 3) and
# .claude/skills/cuda-op-integration/SKILL.md Step 1.
CUDA_TORCH_VERSION="${TORCH_FL_CUDA_TORCH_VERSION:-2.9.0}"
CUDA_TORCH_INDEX_URL="${TORCH_FL_CUDA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

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
    echo "::error::Vendor Python cannot import torch: $candidate" >&2
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
    "cuda": torch.version.cuda,
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
for key in ("version", "base_version", "cuda", "root", "site", "python"):
    print(info.get(key) or "")
PY
)
VENDOR_TORCH_VERSION="${VENDOR_FIELDS[0]}"
VENDOR_TORCH_BASE_VERSION="${VENDOR_FIELDS[1]}"
VENDOR_CUDA_VERSION="${VENDOR_FIELDS[2]}"
VENDOR_TORCH_ROOT="${VENDOR_FIELDS[3]}"
VENDOR_SITE="${VENDOR_FIELDS[4]}"
VENDOR_PYTHON_VERSION="${VENDOR_FIELDS[5]}"
VENDOR_TORCH_LIB="$VENDOR_TORCH_ROOT/lib"

VENDOR_NVIDIA_LIBS=""
for nvidia_lib in "$VENDOR_SITE"/nvidia/*/lib; do
  [[ -d "$nvidia_lib" ]] || continue
  VENDOR_NVIDIA_LIBS="${VENDOR_NVIDIA_LIBS:+$VENDOR_NVIDIA_LIBS:}$nvidia_lib"
done

# The image's vendor torch (2.10.0+cu130) is reused only for the base
# interpreter, nvidia runtime libs, and flag_gems/triton packages. Its torch
# minor version no longer needs to match CPU_TORCH_VERSION, because the
# matching libtorch_cuda.so is downloaded below from a 2.9.0+cu130 wheel.
# Require only that the image's CUDA runtime is 13.0: the nvidia-* libs it
# ships are cu130 and reusable across the 2.9-2.13 range.
if [[ "$VENDOR_CUDA_VERSION" != "13.0" ]]; then
  echo "::error::Vendor torch CUDA runtime is $VENDOR_CUDA_VERSION; expected 13.0"
  exit 1
fi

echo "Vendor Python: $VENDOR_PYTHON ($VENDOR_PYTHON_VERSION)"
echo "Vendor PyTorch: $VENDOR_TORCH_VERSION"
echo "Vendor torch root: $VENDOR_TORCH_ROOT"

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
if [[ -z "$VENDOR_FLAGGEMS_DIR" ]]; then
  echo "::error::FlagGems C++ package was not found in the vendor image"
  exit 1
fi
VENDOR_FLAGGEMS_LIB="$(cd "$VENDOR_FLAGGEMS_DIR/../.." && pwd)"
if ! compgen -G "$VENDOR_FLAGGEMS_LIB/liboperators.so*" >/dev/null; then
  echo "::error::FlagGems liboperators.so was not found under $VENDOR_FLAGGEMS_LIB"
  exit 1
fi

# Source accelerator-side PyTorch libraries from a version-matched cu130
# wheel (downloaded, NOT installed) rather than the image's torch/lib, so
# the CPU-only venv below pairs with a 2.9.0 libtorch_cuda.so. libc10.so,
# libtorch.so, libtorch_cpu.so and libtorch_python.so still come from the
# CPU wheel installed into the venv. setup.py copies these assets into
# torch_fl/lib, and torch_fl preloads them before importing torch.
# See docs/vendors/cuda/external-libtorch-cuda.md and
# .claude/skills/cuda-op-integration/SKILL.md Step 1.
CUDA_ASSETS_DIR="$REPO_ROOT/.libtorch_cuda_assets"
rm -rf "$CUDA_ASSETS_DIR"
mkdir -p "$CUDA_ASSETS_DIR"

CUDA_WHEEL_DIR="$(mktemp -d)"
echo "::group::Download torch==${CUDA_TORCH_VERSION}+cu130 (external libtorch_cuda.so)"
"$VENDOR_PYTHON" -m pip download "torch==${CUDA_TORCH_VERSION}+cu130" \
  --index-url "$CUDA_TORCH_INDEX_URL" --no-deps -d "$CUDA_WHEEL_DIR"
echo "::endgroup::"
CUDA_WHEEL="$(compgen -G "$CUDA_WHEEL_DIR/torch-${CUDA_TORCH_VERSION}+cu130-*.whl" | head -1)"
if [[ -z "$CUDA_WHEEL" ]]; then
  echo "::error::Expected wheel torch-${CUDA_TORCH_VERSION}+cu130-*.whl not found in $CUDA_WHEEL_DIR"
  ls -la "$CUDA_WHEEL_DIR"
  exit 1
fi
"$VENDOR_PYTHON" - "$CUDA_WHEEL" "$CUDA_ASSETS_DIR" <<'PY'
import shutil, stat, sys, zipfile
from pathlib import Path

wheel, out = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(wheel) as z:
    for info in z.infolist():
        if not info.filename.startswith("torch/lib/"):
            continue
        name = info.filename[len("torch/lib/"):]
        if not name or name.endswith("/"):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        # external_attr high 16 bits hold the Unix st_mode. S_ISLNK
        # (mode & 0o170000 == 0o120000) distinguishes symlinks from the
        # regular .so files that share the S_IFLNK bit under a naive
        # mask -- that misclassifies every .so as a symlink and then
        # utf-8-decodes binary contents, crashing with 0xd9.
        if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
            target.symlink_to(z.read(info).decode())
        else:
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
PY
rm -rf "$CUDA_WHEEL_DIR"

# The CUDA dispatcher library is the only mandatory accelerator asset. Some
# vendor Torch layouts (including cu130 images) do not ship a standalone
# libc10_cuda.so; torch_fl treats that library as optional and loads it when
# present. Keep the check layout-agnostic instead of requiring a fixed set of
# files on every CUDA image.
if [[ ! -e "$CUDA_ASSETS_DIR/libtorch_cuda.so" ]]; then
  echo "::error::Required CUDA asset was not found: $CUDA_ASSETS_DIR/libtorch_cuda.so"
  exit 1
fi
if [[ ! -e "$CUDA_ASSETS_DIR/libc10_cuda.so" ]]; then
  echo "::notice::Optional CUDA asset not present: $VENDOR_TORCH_LIB/libc10_cuda.so"
fi
echo "CUDA assets staged: $(find "$CUDA_ASSETS_DIR" -maxdepth 1 -type f -name '*.so*' | wc -l)"

VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-cuda-${CI_STAGE}}"
if ! "$VENDOR_PYTHON" -m venv --clear "$VENV_ROOT"; then
  echo "::warning::Vendor Python cannot create a venv; trying uv"
  if ! command -v uv >/dev/null 2>&1; then
    "$VENDOR_PYTHON" -m pip install --upgrade uv
  fi
  uv venv --clear --seed --python "$VENDOR_PYTHON" "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "::error::Isolated Python was not created at $VENV_ROOT"
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel cmake build
"$VENV_PYTHON" -m pip install \
  --index-url "$CPU_TORCH_INDEX_URL" \
  "torch==$CPU_TORCH_VERSION"
if [[ "$CI_STAGE" == "integration" ]]; then
  "$VENV_PYTHON" -m pip install pytest transformers
fi

# The container image exports LD_LIBRARY_PATH pointing at the vendor (torch
# 2.10) torch/lib. With it present, the 2.9 torch/_C*.so (which uses RUNPATH
# $ORIGIN/lib) loads the 2.10 libtorch_python.so instead -- LD_LIBRARY_PATH
# takes precedence over RUNPATH. The 2.10 C-side torch.device implementation
# then looks up sym_node.DynamicInt, which the 2.9 sym_node.py does not define,
# crashing import with AttributeError. strip_vendor_paths (defined below) does
# this same stripping at line ~287, but only AFTER the crash. Do it here,
# before any venv torch import, so the probe and the CPU_TORCH_ROOT block both
# run against a clean library path.
_vl=""
IFS=: read -ra _paths <<< "${LD_LIBRARY_PATH:-}"
for _p in "${_paths[@]}"; do
  [[ -z "$_p" ]] && continue
  case "$_p" in
    "$VENDOR_TORCH_ROOT"|"$VENDOR_TORCH_ROOT"/*) ;;   # drop vendor 2.10 torch lib paths
    *) _vl="${_vl:+$_vl:}$_p" ;;
  esac
done
export LD_LIBRARY_PATH="$_vl"
unset _vl _paths _p

# torch 2.9 auto-loads registered device backend extensions at import time
# (_import_device_backends). flagcx registers one, but flagcx._C is built
# against the vendor's full torch and needs libc10_cuda.so, which the CPU-only
# venv does not ship and the stripped LD_LIBRARY_PATH no longer provides. The
# probe and the CPU_TORCH_ROOT block only verify the CPU torch version -- they
# must not autoload any device backend. Disable it for the whole CI run.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# The vendor image's FlagGems C++ extensions (liboperators.so, libtriton_jit.so)
# are compiled against the vendor torch (2.10). csrc/CMakeLists.txt defaults
# FLAGGEMS_KERNEL=ON, which links FlagGems::operators into libtorch_fl.so and
# pulls libtriton_jit.so, whose c10::MessageLogger::stream symbol is absent from
# the 2.9 libc10.so preloaded above -> undefined symbol at import torch_fl._C.
# Every non-CUDA accelerator branch passes -DFLAGGEMS_KERNEL=OFF; CUDA omitted
# it only because the 2.10 baseline shared the vendor's ABI. Under the 2.9
# CPU-venv + external libtorch_cuda scheme the C++ FlagGems path is unusable,
# so force it off and rely on FLAGGEMS_PYTHON (CMake default ON) instead.
export FLAGGEMS_KERNEL=0

echo "::group::PROBE: import torch BEFORE copying vendor packages"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
"$VENV_PYTHON" -c "import torch; print('pre-cp OK', torch.__version__, torch.__file__)" || echo "pre-cp FAILED (see traceback above)"
echo "::endgroup::"

# Keep the vendor FlagGems/FlagCX Python packages available without copying the
# vendor torch package. They are pure-Python/extension packages used by the
# CUDA runtime path; the active torch package remains the CPU wheel below.
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

CPU_TORCH_ROOT="$("$VENV_PYTHON" - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent)
assert torch.__version__.split("+", 1)[0] == "2.9.0", torch.__version__
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
export ACCELERATOR=cuda
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export FLAGOS_CUDA_ASSETS_DIR="$CUDA_ASSETS_DIR"
export FLAGGEMS_DIR="$VENDOR_FLAGGEMS_DIR"
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"

CLEAN_CMAKE_PREFIX_PATH="$(strip_vendor_paths "${CMAKE_PREFIX_PATH:-}")"
CLEAN_LIBRARY_PATH="$(strip_vendor_paths "${LIBRARY_PATH:-}")"
CLEAN_LD_LIBRARY_PATH="$(strip_vendor_paths "${LD_LIBRARY_PATH:-}")"
export CMAKE_PREFIX_PATH="$CPU_TORCH_ROOT/share/cmake:$VENDOR_FLAGGEMS_DIR${CLEAN_CMAKE_PREFIX_PATH:+:$CLEAN_CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/lib64:$CUDA_ASSETS_DIR${CLEAN_LIBRARY_PATH:+:$CLEAN_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_ASSETS_DIR${VENDOR_NVIDIA_LIBS:+:$VENDOR_NVIDIA_LIBS}:$CPU_TORCH_ROOT/lib:$VENDOR_FLAGGEMS_LIB:$CUDA_HOME/lib64${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}"

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so and the bundled CUDA assets
  # before the common workflow invokes python -m build.
  python setup.py build_ext --inplace
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - <<'PY'
from pathlib import Path
import sys

# DynamicInt is a torch-2.10 API (PR #162194 landed after the v2.9.0 tag).
# A 2.9.0 torch package must not reference it. If `import torch` fails with
# AttributeError on sym_node.DynamicInt, the executing torch package is NOT
# the 2.9.0+cpu wheel installed into the venv -- either PATH resolved `python`
# to the vendor interpreter, or the venv torch package is mixed with 2.10
# files. Print the ground truth before re-raising so one CI run pinpoints it.
try:
    import torch
except Exception:
    import importlib.util
    import traceback

    spec = importlib.util.find_spec("torch")
    torch_dir = (
        Path(spec.origin).resolve().parent if spec and spec.origin else None
    )
    print("=== import torch FAILED (diagnostics) ===", flush=True)
    print(f"sys.executable: {sys.executable}", flush=True)
    print(f"torch dir: {torch_dir}", flush=True)
    if torch_dir is not None:
        vf = torch_dir / "version.py"
        print(
            f"version.py: "
            f"{vf.read_text(errors='replace').strip() if vf.is_file() else '(missing)'}",
            flush=True,
        )
        sn = torch_dir / "fx" / "experimental" / "sym_node.py"
        sn_txt = sn.read_text(errors="replace") if sn.is_file() else ""
        print(
            f"sym_node.py defines DynamicInt: "
            f"{'class DynamicInt' in sn_txt}",
            flush=True,
        )
        ft = torch_dir / "_subclasses" / "functional_tensor.py"
        if ft.is_file():
            ls = ft.read_text(errors="replace").splitlines()
            print("functional_tensor.py lines 274-284:", flush=True)
            for i in range(273, min(284, len(ls))):
                print(f"  {i + 1}: {ls[i]}", flush=True)
    traceback.print_exc()
    raise

torch_path = Path(torch.__file__).resolve()
assert sys.executable.startswith("/"), sys.executable
assert torch.__version__.split("+", 1)[0] == "2.9.0", torch.__version__
assert torch.version.cuda is None, torch.version.cuda
assert "/opt/conda/" not in str(torch_path), torch_path
assert Path(".libtorch_cuda_assets/libtorch_cuda.so").is_file()
print(f"Isolated Python: {sys.executable}")
print(f"CPU PyTorch: {torch.__version__}")
print(f"CPU torch path: {torch_path}")
print(f"CUDA assets: {Path('.libtorch_cuda_assets').resolve()}")
PY

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    PATH VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH ACCELERATOR CUDA_HOME CUDA_PATH \
    FLAGOS_CUDA_ASSETS_DIR FLAGGEMS_DIR FLAGCX_PATH \
    CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH \
    TORCH_DEVICE_BACKEND_AUTOLOAD FLAGGEMS_KERNEL; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
