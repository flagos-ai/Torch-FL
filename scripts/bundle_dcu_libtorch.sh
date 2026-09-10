#!/usr/bin/env bash
# Bundle DTK's libtorch device .so into torch_fl/lib_dcu/ for a self-contained
# single wheel.
#
# Default (decoupled) mode bundles only the *device* side of the DTK wheel:
#   libc10_hip.so libtorch_hip.so libmagma.so libcaffe2_nvrtc.so
# plus the auditwheel-mangled common libs they transitively need out of DTK's
# torch.libs/ (libgflags/libglog and the three MKL libs libmagma links against).
# The core runtime (libc10.so, libtorch_cpu.so, libtorch.so,
# libtorch_global_deps.so, libtorch_python.so, libshm.so) comes from the
# official torch wheel already installed on the target, which is never modified.
#
# Why that works: measured on DTK 2604 / torch 2.10.0, the official core
# satisfies every ordinary ABI requirement of the DTK device libs.
# libc10_hip.so has zero vendor-core-only imports, and libtorch_hip.so has only
# 16 DTK-private at::_ops::native_fuse_*::call imports -- exported by the
# compatibility shim libflagos_dtk_core_compat.so that cmake installs into
# lib_dcu. scripts/check_dcu_core_abi.py verifies that gap here and fails the
# bundle when a DTK release adds a new vendor-core-only import, instead of
# letting the loader fail (or crash) on the target. All other DTK kernels
# register under the CUDA dispatch key, which is what the PrivateUse1 -> CUDA
# boxing kernels dispatch into.
#
# FLAGOS_DCU_VENDOR_CORE=1 selects the legacy mode: bundle DTK's full core set
# plus all of torch.libs/, which torch_fl then symlinks over the official
# wheel's torch/lib at import (see torch_fl/accelerator/_vendor_libtorch.py).
# That is the rollback path and the only mode where the DTK-private fused
# schemas are usable, since their schema wrappers live in the core fork.
#
# Note on $ORIGIN semantics: glibc expands $ORIGIN from the path the object was
# *opened by*, not from the resolved real path. So opening a bundled .so through
# a torch/lib symlink gives $ORIGIN = torch/lib, where the hashed deps that sit
# alongside it in lib_dcu do not exist. Both the runtime preload and the legacy
# relink therefore walk the bundle's own paths, and the RUNPATH below names the
# bundle dir relatively as well.
#
# Not bundled in either mode: the DTK driver stack (libgalaxyhip.so.5
# libMIOpen.so.1 librocblas.so.4 librccl.so.1, ...) stays on the target under
# /opt/dtk. SDK version binding already exists and is stronger
# (libtorch_hip.so's DT_NEEDED hard-codes librocblas.so.4).
#
# Usage:
#   FLAGOS_DCU_TORCH_LIB=<dtk torch/lib> bash scripts/bundle_dcu_libtorch.sh
#   DTK_ROOT=/opt/dtk bash scripts/bundle_dcu_libtorch.sh
#   FLAGOS_DCU_VENDOR_CORE=1 bash scripts/bundle_dcu_libtorch.sh   # legacy
#
# Should run after `python setup.py build_ext --inplace` (ACCELERATOR=dcu) and
# before packing the wheel. Idempotent.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/bundle_common.sh
source "${REPO_DIR}/scripts/lib/bundle_common.sh"

LIB_DCU="${REPO_DIR}/torch_fl/lib_dcu"
TORCH_FL_LIB="${REPO_DIR}/torch_fl/lib"
DTK_ROOT="${DTK_ROOT:-${ROCM_PATH:-/opt/dtk}}"
COMPAT_SO="libflagos_dtk_core_compat.so"
COMPAT_MANIFEST="${REPO_DIR}/torch_fl/accelerator/dcu/dtk_core_compat_symbols.txt"

case "${FLAGOS_DCU_VENDOR_CORE:-0}" in
  1|on|ON|true|TRUE|yes|YES) VENDOR_CORE=1 ;;
  *) VENDOR_CORE=0 ;;
esac

SRC="${FLAGOS_DCU_TORCH_LIB:-}"
if [ -z "${SRC}" ]; then
  SRC="$(bundle_find_vendor_torch_lib libtorch_hip.so dtk hip das || true)"
fi

if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
  echo "error: DTK torch/lib not found. Set FLAGOS_DCU_TORCH_LIB=<dtk torch/lib>" >&2
  exit 1
fi
if [ ! -f "${SRC}/libtorch_hip.so" ]; then
  echo "error: ${SRC} does not contain libtorch_hip.so, not a DTK torch/lib" >&2
  exit 1
fi

bundle_require_patchelf

# Device/runtime .so the official +cpu wheel does not ship at all. libmagma.so
# is a DT_NEEDED of libtorch_hip.so and has no counterpart anywhere under
# /opt/dtk, so it must be bundled even though it is 360 MB. libcaffe2_nvrtc.so
# is dlopened by torch.cuda.init(), reached through GetFlagosDefaultCudaGenerator.
DEVICE_SO=(libc10_hip.so libtorch_hip.so libmagma.so libcaffe2_nvrtc.so)
# Core runtime, bundled in legacy mode only. libshm.so is a direct DT_NEEDED of
# libtorch_python.so (torch.multiprocessing's shared-memory manager) and is a
# different build in the DTK wheel, so it belongs to the same set.
CORE_SO=(libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so libtorch_python.so libshm.so)

# DTK driver stack measured layout (container LD_LIBRARY_PATH matches find results):
#   lib/          libhipnn librocfft.so.0 librocrand.so.1 librocsparse.so.1
#                 libMIOpen-recommend.so libunwind.so.8 libgalaxyhip.so.5
#   hip/lib/      hip runtime
#   aillvm/lib/   libomp.so (libmagma.so's DT_NEEDED)
#   .hyhal/rocm_smi/lib/  librocm_smi64.so.2
# All stay on the target machine (a box with DCU cards has /opt/dtk), but RPATH
# must cover them, else not-found when LD_LIBRARY_PATH is unset.
VENDOR_RPATH="${DTK_ROOT}/lib:${DTK_ROOT}/hip/lib:${DTK_ROOT}/lib64"
VENDOR_RPATH="${VENDOR_RPATH}:${DTK_ROOT}/aillvm/lib:${DTK_ROOT}/.hyhal/rocm_smi/lib"
VENDOR_RPATH="${VENDOR_RPATH}:${DTK_ROOT}/llvm/lib:/opt/hyhal/lib"
# DTK's auditwheel-mangled libmpi entry is a symlink into /opt/mpi. Legacy mode
# dereferences it, and its DT_NEEDED uses the real MPI/HCoLL sonames. They are
# part of the target's communication stack rather than torch assets, so keep
# them external and reachable through RUNPATH. Harmless in decoupled mode,
# where no DT_NEEDED references them.
VENDOR_RPATH="${VENDOR_RPATH}:/opt/mpi/lib:/opt/mellanox/hcoll/lib"

if [ "${VENDOR_CORE}" = "1" ]; then
  echo "Mode                 : legacy (vendor core libs bundled + relinked)"
else
  echo "Mode                 : decoupled (device libs only, official core kept)"
fi
echo "Source DTK torch/lib : ${SRC}"
echo "Target lib_dcu       : ${LIB_DCU}"
echo "DTK driver path      : ${DTK_ROOT}"

# Libs inside the bundle must be openable from two paths:
#   1. Directly from lib_dcu/ (the runtime preload walks this one, see
#      torch_fl/accelerator/dcu/_dcu_libtorch_link.py)
#   2. Through the torch/lib/ symlinks in legacy mode (when `import torch`
#      loads libtorch_global_deps.so itself)
# glibc expands $ORIGIN from the path the object was opened by, so path #2 gives
# $ORIGIN = torch/lib and cannot find libmpi-3fcb240d.so.40.40.3 and other
# auditwheel-mangled libs sitting in lib_dcu. Both dirs are siblings under
# site-packages (torch/lib -> ../../torch_fl/lib_dcu), so adding one more
# relative path covers both cases. Measured: without this, `import torch` before
# `import torch_fl` dies on libmpi not found.
BUNDLE_ORIGIN="\$ORIGIN:\$ORIGIN/../../torch_fl/lib_dcu"
BUNDLE_RPATH="${BUNDLE_ORIGIN}:${VENDOR_RPATH}"

# torch.libs/: auditwheel dir sibling to torch/, filenames carry hash suffixes.
TORCH_LIBS="$(cd "${SRC}/../.." && pwd)/torch.libs"

# Every .so in torch.libs/, sorted. Deliberately NOT `-type f`: DTK symlinks some
# entries out to system locations (libmpi-3fcb240d.so.40.40.3 -> /opt/mpi/...),
# and a -type f filter silently skipped those, leaving libtorch_cpu.so with an
# unresolvable DT_NEEDED on the target. bundle_copy_so's `cp -fL` dereferences.
_torch_libs_names() {
  find "${TORCH_LIBS}" -maxdepth 1 \( -type f -o -type l \) -name '*.so*' | sort
}

# Names that must never be taken from the bundle in decoupled mode: the process
# resolves them from the official torch wheel instead.
_is_core_so() {
  local name="$1" core
  for core in "${CORE_SO[@]}"; do
    [ "${name}" = "${core}" ] && return 0
  done
  return 1
}

# Drop everything in lib_dcu that the selected mode does not want, so a rerun
# after switching modes cannot leave a stale mix in the wheel (in particular a
# leftover vendor libtorch_cpu.so, which would silently re-couple the build).
# KEEP_NAMES is filled in by the mode branches.
KEEP_NAMES=("${COMPAT_SO}")
prune_unwanted() {
  local so name keep hit removed=0
  [ -d "${LIB_DCU}" ] || return 0
  for so in "${LIB_DCU}"/*.so*; do
    [ -f "${so}" ] || continue
    name="$(basename "${so}")"
    hit=0
    for keep in "${KEEP_NAMES[@]}"; do
      [ "${name}" = "${keep}" ] && { hit=1; break; }
    done
    [ "${hit}" = "1" ] && continue
    rm -f "${so}"
    echo "  pruned ${name}"
    removed=1
  done
  [ "${removed}" = "1" ] || echo "  nothing to prune"
}

# Transitively bundle the hashed common libs the given .so need out of
# torch.libs/. Names are hash-suffixed per DTK build, so resolve them from
# DT_NEEDED instead of hard-coding: a DTK respin changes the hashes but not the
# dependency graph. Appends every name it bundles to KEEP_NAMES.
bundle_needed_from_torch_libs() {
  local -a queue=("$@")
  local -a extra=()
  local so dep base
  [ -d "${TORCH_LIBS}" ] || return 0
  while [ ${#queue[@]} -gt 0 ]; do
    so="${queue[0]}"
    queue=("${queue[@]:1}")
    [ -f "${so}" ] || continue
    while read -r dep; do
      [ -n "${dep}" ] || continue
      _is_core_so "${dep}" && continue
      case " ${KEEP_NAMES[*]} " in *" ${dep} "*) continue ;; esac
      [ -f "${TORCH_LIBS}/${dep}" ] || continue
      bundle_copy_so "${TORCH_LIBS}" "${LIB_DCU}" "${BUNDLE_RPATH}" 1 "${dep}"
      base="${LIB_DCU}/${dep}"
      KEEP_NAMES+=("${dep}")
      extra+=("${dep}")
      queue+=("${base}")
    done < <(patchelf --print-needed "${so}" 2>/dev/null || true)
  done
  if [ ${#extra[@]} -eq 0 ]; then
    echo "  no additional torch.libs dependencies required"
  fi
}

if [ "${VENDOR_CORE}" = "1" ]; then
  # Legacy mode replaces the whole core, so every torch.libs entry can be a
  # transitive dependency of something in the bundle. Copy them all, as before.
  KEEP_NAMES+=("${CORE_SO[@]}" "${DEVICE_SO[@]}")
  if [ -d "${TORCH_LIBS}" ]; then
    while IFS= read -r f; do
      KEEP_NAMES+=("$(basename "${f}")")
    done < <(_torch_libs_names)
  fi
  prune_unwanted
  bundle_copy_so "${SRC}" "${LIB_DCU}" "${BUNDLE_RPATH}" 1 "${CORE_SO[@]}"
  bundle_copy_so "${SRC}" "${LIB_DCU}" "${BUNDLE_RPATH}" 1 "${DEVICE_SO[@]}"
  if [ -d "${TORCH_LIBS}" ]; then
    echo "Source torch.libs    : ${TORCH_LIBS}"
    _names=()
    while IFS= read -r f; do
      _names+=("$(basename "${f}")")
    done < <(_torch_libs_names)
    if [ ${#_names[@]} -gt 0 ]; then
      bundle_copy_so "${TORCH_LIBS}" "${LIB_DCU}" "${BUNDLE_RPATH}" 0 "${_names[@]}"
    else
      echo "warning: ${TORCH_LIBS} has no .so, skipping" >&2
    fi
  else
    echo "warning: ${TORCH_LIBS} not found; the target will lack hash-suffixed" >&2
    echo "         common libs (libglog-*.so.0 etc.) from libtorch_cpu.so's DT_NEEDED." >&2
  fi
else
  # The compatibility shim is what makes the official core sufficient, so the
  # bundle is invalid without it. cmake installs it into lib_dcu during
  # build_ext, i.e. it is already there before this script runs.
  if [ ! -f "${LIB_DCU}/${COMPAT_SO}" ]; then
    echo "error: ${LIB_DCU}/${COMPAT_SO} is missing. Build first with" >&2
    echo "       ACCELERATOR=dcu python setup.py build_ext --inplace" >&2
    exit 1
  fi

  KEEP_NAMES+=("${DEVICE_SO[@]}")
  # The SDK-native plugin is built into lib_dcu by CMake and is independent of
  # DTK's device-side libtorch. Keep it when refreshing the decoupled bundle;
  # otherwise this script's stale-file cleanup silently removes the artifact
  # that FLAGOS_DCU_SDK_OPS=1 is supposed to load.
  if [ -f "${LIB_DCU}/libdcu_aten_ops.so" ]; then
    KEEP_NAMES+=("libdcu_aten_ops.so")
  fi
  prune_unwanted
  bundle_copy_so "${SRC}" "${LIB_DCU}" "${BUNDLE_RPATH}" 1 "${DEVICE_SO[@]}"
  echo "Resolving hashed deps from ${TORCH_LIBS}"
  _device_paths=()
  for _so in "${DEVICE_SO[@]}"; do
    _device_paths+=("${LIB_DCU}/${_so}")
  done
  bundle_needed_from_torch_libs "${_device_paths[@]}"
  patchelf --set-rpath "${BUNDLE_RPATH}" "${LIB_DCU}/${COMPAT_SO}"

  # ABI guard: the DTK device libs must need nothing from the vendor core that
  # neither the official core nor the shim provides. The manifest is an allowed
  # superset because an official-header plugin imports fewer private wrappers than
  # a DTK-header plugin. --vendor-lib is the DTK source tree, not the bundle: the
  # check needs the vendor core exports too, and those are exactly what the
  # decoupled bundle no longer contains.
  OFFICIAL_TORCH_LIB="$("${PYTHON:-python}" - <<'PY'
import importlib.util
import os

spec = importlib.util.find_spec("torch")
root = spec.submodule_search_locations[0] if spec and spec.submodule_search_locations else ""
print(os.path.join(root, "lib") if root else "")
PY
)"
  if [ -z "${OFFICIAL_TORCH_LIB}" ] || [ ! -d "${OFFICIAL_TORCH_LIB}" ]; then
    echo "error: cannot locate the active torch wheel's lib dir for the ABI check" >&2
    exit 1
  fi
  if [ "$(cd "${OFFICIAL_TORCH_LIB}" && pwd -P)" = "$(cd "${SRC}" && pwd -P)" ]; then
    echo "error: the active torch IS the DTK wheel (${OFFICIAL_TORCH_LIB})." >&2
    echo "       Decoupled builds must run against the official torch+cpu wheel;" >&2
    echo "       use FLAGOS_DCU_VENDOR_CORE=1 to bundle the vendor core instead." >&2
    exit 1
  fi
  echo "Official torch/lib   : ${OFFICIAL_TORCH_LIB}"
  _ABI_ARGS=(
      --vendor-lib "${SRC}"
      --official-lib "${OFFICIAL_TORCH_LIB}"
      --shim "${LIB_DCU}/${COMPAT_SO}"
      --manifest "${COMPAT_MANIFEST}"
  )
  # A libtorch_fl.so compiled against DTK's patched headers imports its own
  # family of private schema wrappers, so fold it into the measured gap when built.
  # An official-header build has none of those imports; the guard accepts both.
  if [ -f "${TORCH_FL_LIB}/libtorch_fl.so" ]; then
    _ABI_ARGS+=(--plugin "${TORCH_FL_LIB}/libtorch_fl.so")
  fi
  "${PYTHON:-python}" "${REPO_DIR}/scripts/check_dcu_core_abi.py" "${_ABI_ARGS[@]}"

  # A decoupled wheel that still carries a vendor core lib would relink nothing
  # but shadow the official core through RPATH order; fail loudly instead.
  for _core in "${CORE_SO[@]}"; do
    if [ -e "${LIB_DCU}/${_core}" ]; then
      echo "error: ${LIB_DCU}/${_core} present in a decoupled bundle" >&2
      exit 1
    fi
  done
  echo "Verified: no vendor core lib in the bundle"
fi

# libtorch_fl.so / libflagos.so also need DTK's CUDA compatibility layer
# libcudart.so.12 (cuda_runtime_compat, see CMakeLists.txt's DCU_CUDA_ROOT).
# cmake already wrote it into RUNPATH; cannot drop it when rewriting here --
# else libcudart.so.12 becomes not-found in a clean environment.
_DCU_CUDA_LIB64=""
for _c in "${DTK_ROOT}"/cuda/cuda-*/lib64; do
  if [ -f "${_c}/libcudart.so.12" ]; then
    _DCU_CUDA_LIB64="${_c}"
    break
  fi
done
if [ -z "${_DCU_CUDA_LIB64}" ]; then
  echo "warning: ${DTK_ROOT}/cuda/cuda-*/lib64 has no libcudart.so.12" >&2
fi
PLUGIN_RPATH="\$ORIGIN:\$ORIGIN/../lib_dcu"
[ -n "${_DCU_CUDA_LIB64}" ] && PLUGIN_RPATH="${PLUGIN_RPATH}:${_DCU_CUDA_LIB64}"
bundle_rewrite_plugin_rpath "${TORCH_FL_LIB}" "${PLUGIN_RPATH}:${VENDOR_RPATH}"

# torch/version.py is pure Python generated at build time, so swapping .so
# cannot change it. In the self-contained DCU wheel the base is stock torch+cpu,
# whose torch.version.hip reports None, and triton's hcu backend gates
# is_active() on that value (backends/hcu/driver.py is_active():
# torch.cuda.is_available() and torch.version.hip is not None). None means never
# activate, and any flag_gems op dies in triton's driver factory: "0 active
# drivers ([])". Carry the vendor torch's own version.py; at import time
# _restore_dcu_hip_version() reads back the hip/rocm strings, and the version
# check in _dcu_libtorch_link.py uses its base version to reject a mismatched
# official wheel.
_VENDOR_VERSION_PY="$(cd "${SRC}/.." && pwd)/version.py"
if [ -f "${_VENDOR_VERSION_PY}" ]; then
  cp -fL "${_VENDOR_VERSION_PY}" "${LIB_DCU}/vendor_version.py"
  echo "Copied vendor version.py -> lib_dcu/vendor_version.py"
  grep -E "^\s*(hip|rocm|__version__)\s*(:|=)" "${LIB_DCU}/vendor_version.py" || true
else
  echo "warning: ${_VENDOR_VERSION_PY} not found, triton hcu backend may not activate" >&2
fi

bundle_summary "${LIB_DCU}"
bundle_check_needed "${LIB_DCU}" \
    "${DTK_ROOT}/lib" "${DTK_ROOT}/hip/lib" "${DTK_ROOT}/lib64" \
    "${DTK_ROOT}/aillvm/lib" "${DTK_ROOT}/llvm/lib" \
    "${DTK_ROOT}/.hyhal/rocm_smi/lib" "${_DCU_CUDA_LIB64:-/nonexistent}" \
    "/opt/hyhal/lib" "/opt/mpi/lib" "/opt/mellanox/hcoll/lib" \
    "/usr/lib64" "/usr/lib" "/usr/lib/x86_64-linux-gnu" \
    "${OFFICIAL_TORCH_LIB:-/nonexistent}"
