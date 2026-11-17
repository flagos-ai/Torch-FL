#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Run after the default wheel's suites. Keep its artifact and generated vendor
# libraries separate from the native build, then test the installed opt-in wheel.
set -euo pipefail
cd "${GITHUB_WORKSPACE:?}"
evidence="${RUNNER_TEMP:?}/torch-fl-native-host-evidence"
mkdir -p "$evidence"
build_source="$(mktemp -d "$RUNNER_TEMP/torch-fl-native-host-source.XXXXXX")"
# The container user can differ from the checkout owner; trust only this read.
git -c safe.directory="$GITHUB_WORKSPACE" archive HEAD | tar -x -C "$build_source"
cd "$build_source"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLAGOS_BUILD_NATIVE_HOST=1 FLAGOS_BUILD_BOXING=0
export FLAGOS_BUILD_FLAGGEMS=1 FLAGOS_BUILD_FLAGGEMS_CPP=0 FLAGOS_BUILD_TILEOPS=0
unset FLAGOS_FORCE_BACKEND FLAGOS_BACKEND_CONFIG FLAGOS_METAX_CUDART_SHIM

source "$GITHUB_WORKSPACE/.github/scripts/lib/set_env_common.sh"
PIP_RETRY_PYTHON="$(command -v python)" pip_retry 'Cython==3.2.4'
python setup.py bdist_wheel --dist-dir "$evidence" 2>&1 | tee "$evidence/build.log"
python scripts/tools/torch-fl-preflight --wheel "$evidence"/*.whl \
  --platform "$FLAGOS_ACCELERATOR" --check-build-env --check-native-host-backend \
  | tee "$evidence/preflight.log"
python -m pip install --force-reinstall --no-deps "$evidence"/*.whl

# Read the wheel without importing the source package. Reports are artifacts,
# not source-controlled policy or claims of measured operator correctness.
python - "$evidence" <<'PY'
from pathlib import Path
import sys
import zipfile

directory = Path(sys.argv[1])
with zipfile.ZipFile(next(directory.glob("*.whl"))) as wheel:
    names = ["torch_fl/_build_config.py", "torch_fl/compatibility.json"]
    names += [name for name in wheel.namelist()
              if name.startswith("torch_fl/native_host/_generated/")
              and name.endswith(("manifest.json", "coverage.json", "diagnostics.json"))]
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wheel.read(name))
PY
