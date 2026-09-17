#!/usr/bin/env bash
# Bundle the MetaX-forked libtorch C++ .so into torch_fl/lib_maca/ for a
# self-contained single wheel.
#
# The official torch+cpu wheel lacks the MetaX-forked libtorch (which exports
# at::maca::* / wcuda* symbols), and the boxing artifact libtorch_fl.so must
# link against those symbols. This script copies that batch of libtorch .so
# from the MetaX torch wheel into torch_fl/lib_maca/, and rewrites RPATH with
# patchelf:
#   - libtorch .so inside lib_maca  -> $ORIGIN:/opt/maca/lib:/opt/maca/lib64
#     (at runtime find mcblas/mcdnn and other maca runtime from /opt/maca on
#      the target machine; the runtime is not bundled)
#   - torch_fl/lib/libtorch_fl.so   -> $ORIGIN:$ORIGIN/../lib_maca
#     (find the forked libtorch from lib_maca inside the package, removing the
#      absolute path hard-coded on the build machine)
#
# maca runtime (libmcblas etc., ~4.9G) is not bundled: machines with MetaX
# cards installed must have the /opt/maca driver.
#
# Usage:
#   FLAGOS_VENDOR_TORCH_LIB=<maca torch/lib> bash scripts/vendor/bundle_maca_libtorch.sh
#   MACA_PATH=/opt/maca bash scripts/vendor/bundle_maca_libtorch.sh   # override maca path
#
# Should run after `python setup.py bdist_wheel` (FLAGOS_ACCELERATOR=metax) and before
# packing the wheel, or re-pack the wheel after running. Idempotent.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/vendor/bundle_common.sh
source "${REPO_DIR}/scripts/vendor/bundle_common.sh"

LIB_MACA="${REPO_DIR}/torch_fl/lib_maca"
TORCH_FL_LIB="${REPO_DIR}/torch_fl/lib"
# Vendor name only, matching setup.py / the metax CMake branch.
MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"

# MetaX torch/lib source: explicit env, or find +metax torch from conda.
SRC="${FLAGOS_VENDOR_TORCH_LIB:-}"
if [ -z "${SRC}" ]; then
  SRC="$(bundle_find_vendor_torch_lib libtorch_cuda.so metax maca || true)"
fi

if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
  echo "error: MetaX torch/lib not found. Set FLAGOS_VENDOR_TORCH_LIB=<maca torch/lib>" >&2
  exit 1
fi
if [ ! -f "${SRC}/libtorch_cuda.so" ]; then
  echo "error: ${SRC} does not contain libtorch_cuda.so, not a MetaX torch/lib" >&2
  exit 1
fi

bundle_require_patchelf

# Self-consistent set aligned with _metax_libtorch_link._CORE_SO + _CUDA_SO.
CORE_SO=(libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so libtorch_python.so)
CUDA_SO=(libc10_cuda.so libtorch_cuda.so libtorch_cuda_linalg.so libshm.so)

VENDOR_RPATH="${MACA_PATH}/lib:${MACA_PATH}/lib64"
# Same as DCU: libs inside the bundle must be openable from both lib_maca/
# and torch/lib/ symlink paths.
BUNDLE_ORIGIN="\$ORIGIN:\$ORIGIN/../../torch_fl/lib_maca"

echo "Source MetaX torch/lib : ${SRC}"
echo "Target lib_maca        : ${LIB_MACA}"
echo "maca runtime path      : ${MACA_PATH}/lib"

bundle_copy_so "${SRC}" "${LIB_MACA}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 \
    "${CORE_SO[@]}" "${CUDA_SO[@]}"

# Strip the build machine's hard-coded MetaX torch/lib absolute path from
# torch_fl.so, rewrite to find the forked libtorch from lib_maca inside the package.
bundle_rewrite_plugin_rpath "${TORCH_FL_LIB}" \
    "\$ORIGIN:\$ORIGIN/../lib_maca:${VENDOR_RPATH}"

bundle_summary "${LIB_MACA}"
bundle_check_needed "${LIB_MACA}" \
    "${MACA_PATH}/lib" "${MACA_PATH}/lib64" "/usr/lib64" "/usr/lib"
