# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# The job container owns its venv; the image remains an immutable warm baseline.
# Caller supplies the requested versions, indexes and common install helpers.
prepare_iluvatar_packages() {
  if "$VENV_PYTHON" - "$FLAGTREE_VERSION" <<'PY'
import importlib.metadata
import sys

try:
    assert importlib.metadata.version("flagtree") == sys.argv[1]
except (AssertionError, importlib.metadata.PackageNotFoundError):
    raise SystemExit(1)
PY
  then
    echo "Using preinstalled FlagTree $FLAGTREE_VERSION"
  else
    # FlagTree owns triton's modules. Remove both metadata owners before
    # replacing the provider; installing stock Triton separately is invalid.
    "$VENV_PYTHON" -m pip uninstall -y flagtree triton triton_kernels
    prefer_direct_route "$FLAGTREE_INDEX_URL" "FlagTree wheel"
    install_flagtree
  fi

  if ! flag_gems_installed; then
    prefer_direct_route "$FLAGGEMS_INDEX_URL" "FlagGems wheel"
  fi
  install_flag_gems
  # New releases may require a different dependency cohort. Fail explicitly
  # instead of letting an incompatible install reach the build or hardware.
  "$VENV_PYTHON" -m pip check
}
