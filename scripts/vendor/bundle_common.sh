#!/usr/bin/env bash
# common parts for self-contained wheel bundling, sourced by scripts/vendor/bundle_{maca,dcu,ppu}_libtorch.sh.
#
# background: metax / dcu / ppu all run on a vendor fork of libtorch -- measured
# symbol attribution shows libtorch_cpu.so provides 2000+ undefined symbols for
# libtorch_fl.so, while the vendor lib (libtorch_hip.so / libtorch_cuda.so)
# provides 0. so a self-contained wheel must bundle the core libs together, not
# just the vendor lib. unified approach:
#   - .so inside the bundle dir      -> RPATH = $ORIGIN:<vendor driver dirs...>
#     (driver runtime stays on the target machine, not bundled: a box with the
#      card already has the driver)
#   - torch_fl/lib/libtorch_fl.so -> RPATH = $ORIGIN:$ORIGIN/../<bundle>:<driver...>
#     (removes the build machine's hard-coded absolute paths. cmake/FlagosRpath.cmake
#      already sets this; the patchelf here is fallback for existing .so / non-cmake
#      build artifacts, kept idempotent.)
#
# the runtime logic that symlinks the bundle's core libs into stock torch/lib is
# in torch_fl/accelerator/_vendor_libtorch.py; the .so list on both sides must align.
#
# provides:
#   bundle_require_patchelf
#   bundle_find_vendor_torch_lib <probe_so> <marker...>
#   bundle_copy_so <src_dir> <dst_dir> <rpath> <required:0|1> <so...>
#   bundle_rewrite_plugin_rpath <torch_fl_lib> <rpath>
#   bundle_summary <dst_dir>

set -euo pipefail

# patchelf is missing by default on all four nodes. on bw1000/810e `pip install patchelf`
# works directly; mc550's default index lacks the package and needs an explicit source.
bundle_require_patchelf() {
  if command -v patchelf >/dev/null 2>&1; then
    return 0
  fi
  cat >&2 <<'EOF'
error: patchelf not found, cannot rewrite RPATH. install with (any):
    pip install patchelf
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple patchelf   # when default index lacks it
    conda install -c conda-forge patchelf
EOF
  return 1
}

# locate the vendor torch/lib: prefer the current interpreter's torch (verify
# version.py contains a vendor marker and probe_so exists), return non-zero on
# failure. caller should check their own env override variable first.
# usage: bundle_find_vendor_torch_lib libtorch_hip.so dtk hip
bundle_find_vendor_torch_lib() {
  local probe="$1"
  shift
  local markers="$*"
  local py
  py="$(command -v python || command -v python3 || true)"
  [ -n "${py}" ] || return 1
  PROBE_SO="${probe}" VENDOR_MARKERS="${markers}" "${py}" - <<'PY' 2>/dev/null || return 1
import importlib.util, os

probe = os.environ["PROBE_SO"]
markers = os.environ["VENDOR_MARKERS"].split()
spec = importlib.util.find_spec("torch")
if spec and spec.submodule_search_locations:
    root = spec.submodule_search_locations[0]
    lib = os.path.join(root, "lib")
    ver = os.path.join(root, "version.py")
    txt = open(ver).read() if os.path.isfile(ver) else ""
    if (not markers or any(m in txt for m in markers)) and os.path.exists(
        os.path.join(lib, probe)
    ):
        print(lib)
PY
}

# copy .so and set RPATH. cp -fL dereferences symlinks, copies the real file.
# when required=1 a missing source fails; =0 skips it (the stock +cpu wheel does
# not ship vendor libs anyway).
bundle_copy_so() {
  local src_dir="$1" dst_dir="$2" rpath="$3" required="$4"
  shift 4
  local so src
  mkdir -p "${dst_dir}"
  for so in "$@"; do
    src="${src_dir}/${so}"
    if [ ! -f "${src}" ]; then
      if [ "${required}" = "1" ]; then
        echo "error: missing required ${src}" >&2
        return 1
      fi
      echo "warning: ${src} does not exist, not bundled" >&2
      continue
    fi
    cp -fL "${src}" "${dst_dir}/${so}"
    # non-ELF (rare .so that are actually linker scripts) will fail patchelf, not fatal.
    if ! patchelf --set-rpath "${rpath}" "${dst_dir}/${so}" 2>/dev/null; then
      echo "  ${so}: patchelf skipped (non-ELF?)"
    fi
    echo "  bundled ${so} ($(du -h "${dst_dir}/${so}" | cut -f1))"
  done
}

# rewrite RPATH for libtorch_fl.so / libtorch_bindings.so / libflagos.so.
bundle_rewrite_plugin_rpath() {
  local lib_dir="$1" rpath="$2" so target
  for so in libtorch_fl.so libtorch_bindings.so libflagos.so; do
    target="${lib_dir}/${so}"
    [ -f "${target}" ] || continue
    patchelf --set-rpath "${rpath}" "${target}"
    echo "  rewrote RPATH ${so} -> ${rpath}"
  done
}

bundle_summary() {
  local dst_dir="$1"
  echo "done. $(basename "${dst_dir}") total size: $(du -sh "${dst_dir}" | cut -f1) ($(find "${dst_dir}" -type f | wc -l | tr -d ' ') files)"
}

# baseline libs guaranteed present on the target machine, not reported during self-check.
_BUNDLE_BASELINE_RE='^(libc|libm|libdl|librt|libpthread|libstdc\+\+|libgcc_s|ld-linux.*|libutil|libresolv|libnsl|libcrypt|libatomic|libgomp|libz|libnuma)\.so'

# self-check: for each .so in the bundle, report every DT_NEEDED that is neither in
# the bundle, nor in the given target machine directories, nor a baseline system lib
# -- those are candidates for "not found" on the target machine.
# usage: bundle_check_needed <bundle_dir> <target_dir...>
bundle_check_needed() {
  local dst_dir="$1"
  shift
  local so dep d found
  echo "---- DT_NEEDED self-check (only reports items potentially missing on target) ----"
  local missing=0
  for so in "${dst_dir}"/*.so*; do
    [ -f "${so}" ] || continue
    while read -r dep; do
      [ -n "${dep}" ] || continue
      [[ "${dep}" =~ ${_BUNDLE_BASELINE_RE} ]] && continue
      [ -e "${dst_dir}/${dep}" ] && continue
      found=0
      for d in "$@"; do
        if [ -e "${d}/${dep}" ]; then found=1; break; fi
      done
      [ "${found}" = "1" ] && continue
      echo "  $(basename "${so}") -> ${dep}"
      missing=1
    done < <(patchelf --print-needed "${so}" 2>/dev/null || true)
  done
  if [ "${missing}" = "0" ]; then
    echo "  none (bundle + driver directories already cover all non-baseline deps)"
  fi
}
