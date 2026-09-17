#!/usr/bin/env bash
# Bundle the locally built PPU libtorch C++ .so into torch_fl/lib_ppu/ for a
# self-contained single wheel.
#
# PPU is compiled against PPU_SDK/CUDA_SDK with its own FLAGOS_ACCELERATOR=ppu value;
# the CUDA boxing kernels work as-is. The difference from a real NVIDIA machine lies in
# libtorch: it is a local USE_CUDA=1 source build, not an upstream wheel.
# Measured undefined symbols in libtorch_fl.so show that its libtorch_cpu.so
# provides 2092 of them, libtorch_cuda.so provides 0, and libc10_cuda.so
# provides 10 — so the core libs must be replaced, while libtorch_cuda.so just
# needs to be present (the CUDA dispatch key must have registered vendor kernels).
#
# That local build also links against system MKL in /usr/local/lib
# (libmkl_core / libmkl_gnu_thread / libmkl_intel_lp64, ~171 MB); the official
# torch+cpu wheel does not ship these files, so they are bundled into lib_ppu/
# together.
#
# Not bundled: PPU SDK runtime stays on the target machine at /usr/local/PPU_SDK.
#
# Usage:
#   FLAGOS_VENDOR_TORCH_LIB=<ppu torch/lib> bash scripts/vendor/bundle_ppu_libtorch.sh
#   PPU_SDK=/usr/local/PPU_SDK bash scripts/vendor/bundle_ppu_libtorch.sh
#   FLAGOS_PPU_MKL_DIR=/usr/local/lib bash scripts/vendor/bundle_ppu_libtorch.sh
#
# Should run after `python setup.py bdist_wheel` and before packing the wheel.
# Idempotent.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/vendor/bundle_common.sh
source "${REPO_DIR}/scripts/vendor/bundle_common.sh"

LIB_PPU="${REPO_DIR}/torch_fl/lib_ppu"
TORCH_FL_LIB="${REPO_DIR}/torch_fl/lib"
PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
MKL_DIR="${FLAGOS_PPU_MKL_DIR:-/usr/local/lib}"

SRC="${FLAGOS_VENDOR_TORCH_LIB:-}"
if [ -z "${SRC}" ]; then
  # PPU torch is locally built; version.py may not contain the "ppu" marker,
  # so first try finding it by marker, fall back to "any torch with
  # libtorch_cuda.so from the current interpreter" if not found.
  SRC="$(bundle_find_vendor_torch_lib libtorch_cuda.so ppu || true)"
  if [ -z "${SRC}" ]; then
    SRC="$(bundle_find_vendor_torch_lib libtorch_cuda.so || true)"
  fi
fi

if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
  echo "error: PPU torch/lib not found. Set FLAGOS_VENDOR_TORCH_LIB=<ppu torch/lib>" >&2
  exit 1
fi
if [ ! -f "${SRC}/libtorch_cuda.so" ]; then
  echo "error: ${SRC} does not contain libtorch_cuda.so, not a CUDA-built torch/lib" >&2
  exit 1
fi

bundle_require_patchelf

# Self-consistent set aligned with _ppu_libtorch_link._CORE_SO + _CUDA_SO.
CORE_SO=(libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so libtorch_python.so)
CUDA_SO=(libc10_cuda.so libtorch_cuda.so libtorch_cuda_linalg.so libshm.so)
MKL_SO=(libmkl_core.so.1 libmkl_gnu_thread.so.1 libmkl_intel_lp64.so.1)

VENDOR_RPATH="${PPU_SDK}/CUDA_SDK/lib64:${PPU_SDK}/lib:${PPU_SDK}/lib64"
# Same as DCU: libs inside the bundle must be openable from both lib_ppu/
# and torch/lib/ symlink paths.
BUNDLE_ORIGIN="\$ORIGIN:\$ORIGIN/../../torch_fl/lib_ppu"

echo "Source PPU torch/lib : ${SRC}"
echo "Target lib_ppu       : ${LIB_PPU}"
echo "PPU SDK path         : ${PPU_SDK}"

bundle_copy_so "${SRC}" "${LIB_PPU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 1 "${CORE_SO[@]}"
bundle_copy_so "${SRC}" "${LIB_PPU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${CUDA_SO[@]}"

# System MKL: a direct DT_NEEDED of libtorch_cpu.so, with no same-named file
# in the official wheel.
if [ -d "${MKL_DIR}" ]; then
  echo "Source MKL           : ${MKL_DIR}"
  bundle_copy_so "${MKL_DIR}" "${LIB_PPU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${MKL_SO[@]}"
else
  echo "warning: ${MKL_DIR} does not exist, skipping MKL; libs will be missing if the target machine has no MKL" >&2
fi

bundle_rewrite_plugin_rpath "${TORCH_FL_LIB}" \
    "\$ORIGIN:\$ORIGIN/../lib_ppu:${VENDOR_RPATH}"

bundle_summary "${LIB_PPU}"
bundle_check_needed "${LIB_PPU}" \
    "${PPU_SDK}/CUDA_SDK/lib64" "${PPU_SDK}/lib" "${PPU_SDK}/lib64" \
    "${MKL_DIR}" "/usr/lib64" "/usr/lib" "/usr/local/lib"
