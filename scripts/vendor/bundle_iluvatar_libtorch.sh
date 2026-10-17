#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Stage the matched CoreX binaries independently of the Python torch frontend.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_DIR}/scripts/vendor/bundle_common.sh"
SRC="${FLAGOS_VENDOR_TORCH_LIB:?Set FLAGOS_VENDOR_TORCH_LIB to the staged CoreX lib directory}"
DEST="${REPO_DIR}/torch_fl/lib_iluvatar"
COREX_PATH="${COREX_PATH:-/usr/local/corex}"
bundle_require_patchelf

for required in libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so \
    libtorch_python.so libc10_cuda.so libtorch_cuda.so libshm.so; do
  [[ -f "$SRC/$required" ]] || { echo "error: missing CoreX asset $SRC/$required" >&2; exit 1; }
done
[[ -f "$SRC/vendor_version.py" ]] || { echo "error: missing CoreX version metadata" >&2; exit 1; }

# Include auxiliary device libraries as well: libtorch_cuda loads some by soname.
libraries=()
for path in "$SRC"/*.so*; do libraries+=("$(basename "$path")"); done
bundle_copy_so "$SRC" "$DEST" "\$ORIGIN:$COREX_PATH/lib64:$COREX_PATH/lib" 1 "${libraries[@]}"
cp -f "$SRC/vendor_version.py" "$DEST/vendor_version.py"
bundle_rewrite_plugin_rpath "$REPO_DIR/torch_fl/lib" \
  "\$ORIGIN:\$ORIGIN/../lib_iluvatar:$COREX_PATH/lib64:$COREX_PATH/lib"
bundle_check_needed "$DEST" "$COREX_PATH/lib64" "$COREX_PATH/lib" /usr/local/lib
bundle_summary "$DEST"
