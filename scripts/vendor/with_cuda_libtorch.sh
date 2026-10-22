#!/usr/bin/env bash
# Run an arbitrary command with a version-matched libtorch_cuda.so preloaded
# into the process.
#
# This lets the torch_fl CUDA backend (boxing mode) reuse the CUDA kernels
# already registered by PyTorch, without pip-installing the CUDA build of torch.
# See docs/vendors/cuda/external-libtorch-cuda.md.
#
# Hard constraint (docs §constraint 1): libtorch_cuda.so must be loaded before
# `import torch` (the CUDAHooks caching problem), so it is injected here through
# LD_PRELOAD rather than loaded later from __init__.py.
#
# Usage:
#   bash scripts/vendor/with_cuda_libtorch.sh pytest tests/integration/ops/test_add_dispatch.py -v
#   bash scripts/vendor/with_cuda_libtorch.sh python -c "import torch_fl, torch; ..."

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_ASSETS="${REPO_DIR}/.libtorch_cuda_assets"

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/vendor/with_cuda_libtorch.sh <command> [args...]" >&2
  exit 2
fi

for _so in libc10_cuda.so libtorch_cuda.so; do
  if [ ! -f "${CUDA_ASSETS}/${_so}" ]; then
    echo "error: missing ${CUDA_ASSETS}/${_so}" >&2
    echo "  (see docs/vendors/cuda/external-libtorch-cuda.md for how these assets are produced)" >&2
    exit 1
  fi
done

# 1) nvidia runtime library paths + pip torch's lib dir (libc10_cuda.so depends on
#    libc10.so) + CUDA_ASSETS itself (linalg and similar ops dlopen
#    libtorch_cuda_linalg.so by bare name; that .so lives in CUDA_ASSETS, so
#    CUDA_ASSETS must be on LD_LIBRARY_PATH for it to be found).
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="${CUDA_ASSETS}:$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')${TORCH_LIB}:${LD_LIBRARY_PATH}"

# 2) Hard constraint: load the CUDA .so into the process before `import torch`
#    -> LD_PRELOAD.
export LD_PRELOAD="${CUDA_ASSETS}/libc10_cuda.so:${CUDA_ASSETS}/libtorch_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"

exec "$@"
