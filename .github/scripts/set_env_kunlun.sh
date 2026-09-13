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

# Kunlun P800 CI environment setup. Unlike the CUDA line, the Kunlun vendor
# torch (2.9.0+cu129) is used directly: no CPU venv, no external
# libtorch_cuda.so extraction, no vendor-path stripping. setup.py forces
# -DFLAGGEMS_KERNEL=OFF for ACCELERATOR=kunlun because the image ships no
# liboperators.so, so this script does not set FLAGGEMS_KERNEL either.
# See docs/vendors/kunlun/installation.md.

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
VENDOR_TORCH_VERSION="$("$VENDOR_PYTHON" -c 'import torch; print(torch.__version__)')"
VENDOR_CUDA_VERSION="$("$VENDOR_PYTHON" -c 'import torch; print(torch.version.cuda)')"
echo "Vendor Python: $VENDOR_PYTHON"
echo "Vendor PyTorch: $VENDOR_TORCH_VERSION"
echo "Vendor torch CUDA runtime: $VENDOR_CUDA_VERSION"

# The image ships a system cmake at /usr/bin/cmake but not the cmake pip
# package. pyproject.toml lists cmake>=3.18 in [build-system] requires, and
# python -m build --no-isolation checks those requires at the pip-package
# level, so the system binary alone is not enough. Install only the cmake
# wheel -- the image already has pip/setuptools/wheel/build -- so the
# no-isolation build dependency check passes; setup.py still invokes the
# system cmake on PATH.
# NOTE: listing cmake in [build-system] requires is a cross-line smell -- it
# forces every platform to pip-install cmake even when a system cmake exists.
# Fixing it (dropping cmake from requires) touches all platforms and belongs
# in a separate PR; this line mirrors set_env_cuda.sh's approach.
python -m pip install cmake

export ACCELERATOR=kunlun
export XPU_ROOT="${XPU_ROOT:-/usr/local/xpu}"
export XCUDART_ROOT="${XCUDART_ROOT:-/usr/local/xcudart}"
# XPU-RT 5.37.1 requires this to be set before the first import of torch or
# torch_fl, otherwise import exits with "Runtime profiler is disabled".
export XPU_ENABLE_PROFILER_TRACING="${XPU_ENABLE_PROFILER_TRACING:-1}"
# XPU_CUPTI_ENABLE_DEVICE is intentionally NOT set. Binding the profiler to
# all 8 devices (0,1,...,7) causes CUPTI cuptiActivityEnable to fail with
# error 17 (etiEventSamplingSetMode) in every dispatch subprocess, aborting
# all of [6/6] main_ops. The runtime smoke harness no longer needs this env:
# it warms up the device context in-process (is_available -> init -> randn
# on device 0 and 1 via runpy) before its first empty, avoiding the XDNN
# fp64 min probe (min.cpp:41) without profiler binding. Whether the dispatch
# subprocesses (no init, first op is randn) also avoid the probe without
# this env is being verified by this change.
# export XPU_CUPTI_ENABLE_DEVICE="${XPU_CUPTI_ENABLE_DEVICE:-0,1,2,3,4,5,6,7}"

if [[ ! -f "$XPU_ROOT/include/cuda_runtime.h" ]]; then
  echo "::error::cuda_runtime.h not found at $XPU_ROOT/include"
  exit 1
fi
if [[ ! -f "$XCUDART_ROOT/lib/libcudart.so.12" ]]; then
  echo "::error::libcudart.so.12 not found at $XCUDART_ROOT/lib"
  exit 1
fi

cd "$REPO_ROOT"
if [[ "$CI_STAGE" == "build" || "$CI_STAGE" == "integration" ]]; then
  # Prebuild so package_data sees libtorch_fl.so before the common workflow
  # invokes python -m build.
  python setup.py build_ext --inplace
fi

if ! command -v xpu-smi >/dev/null 2>&1; then
  echo "::error::xpu-smi is unavailable"
  exit 1
fi
xpu-smi

python - <<'PY'
import importlib.util
import sys
import traceback
from pathlib import Path

try:
    import torch
except Exception:
    spec = importlib.util.find_spec("torch")
    torch_dir = Path(spec.origin).resolve().parent if spec and spec.origin else None
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
    traceback.print_exc()
    raise

assert sys.executable.startswith("/"), sys.executable
assert torch.__version__ == "2.9.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
print(f"Isolated Python: {sys.executable}")
print(f"Vendor PyTorch: {torch.__version__}")
print(f"torch path: {torch.__file__}")
PY

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    ACCELERATOR XPU_ROOT XCUDART_ROOT XPU_ENABLE_PROFILER_TRACING; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi
