#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Run after the default wheel's suites. Keep its artifact and generated vendor
# libraries separate from the native build, then test the installed opt-in wheel.
set -euo pipefail
case "${1:-}" in
  ""|--metax-precision|--test-install|--check-installed) ;;
  *) echo "Usage: $0 [--metax-precision|--test-install|--check-installed]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--metax-precision|--test-install|--check-installed]" >&2
  exit 2
fi

cd "${GITHUB_WORKSPACE:?}"
if [[ "${1:-}" == --metax-precision ]]; then
  if [[ "${FLAGOS_ACCELERATOR:?}" != metax ]]; then
    echo "--metax-precision requires FLAGOS_ACCELERATOR=metax" >&2
    exit 2
  fi
  : "${FLAGOS_VENDOR_TORCH_LIB:?MetaX precision requires the published vendor Torch libraries}"
  : "${FLAGOS_VENDOR_TORCH_VERSION:?Set the verified MetaX vendor Torch version}"
  evidence="$(mktemp -d "${RUNNER_TEMP:?}/torch-fl-metax-precision-evidence.XXXXXX")"
else
  evidence="${RUNNER_TEMP:?}/torch-fl-native-host-evidence"
  mkdir -p "$evidence"
fi
echo "Native host evidence: $evidence"
preflight="$GITHUB_WORKSPACE/scripts/tools/torch-fl-preflight"
record_wheel="$GITHUB_WORKSPACE/.github/scripts/record_native_host_wheel.py"

if [[ "${1:-}" == --check-installed ]]; then
  # The Metal runtime is shipped in the ordinary wheel installed by this job.
  python "$preflight" --wheel dist/*.whl --platform "$FLAGOS_ACCELERATOR" \
    --check-build-env | tee "$evidence/preflight.log"
  python "$preflight" --installed --platform "$FLAGOS_ACCELERATOR" \
    --check-build-env | tee "$evidence/installed-preflight.log"
  python "$record_wheel" --wheel dist/*.whl --evidence "$evidence"
  exit 0
fi

build_source="$(mktemp -d "$RUNNER_TEMP/torch-fl-native-host-source.XXXXXX")"
# The container user can differ from the checkout owner; trust only this read.
git -c safe.directory="$GITHUB_WORKSPACE" archive HEAD | tar -x -C "$build_source"
cd "$build_source"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLAGOS_BUILD_NATIVE_HOST=1
export FLAGOS_BUILD_FLAGGEMS=1 FLAGOS_BUILD_FLAGGEMS_CPP=0 FLAGOS_BUILD_TILEOPS=0
if [[ "$FLAGOS_ACCELERATOR" == metax ]]; then
  export FLAGOS_BUILD_VENDOR=1
  # Default CI stages link inputs without the optional precision cohort's
  # published package/headers. This lane tests the SDK-only native artifact.
  if [[ "${1:-}" != --metax-precision ]]; then
    unset FLAGOS_VENDOR_TORCH_LIB FLAGOS_VENDOR_TORCH_VERSION
  fi
elif [[ "$FLAGOS_ACCELERATOR" == hygon || "$FLAGOS_ACCELERATOR" == dcu ]]; then
  export FLAGOS_BUILD_VENDOR=1
  : "${FLAGOS_VENDOR_TORCH_LIB:?HCU native precision requires the published DTK torch libraries}"
fi
unset FLAGOS_FORCE_BACKEND FLAGOS_BACKEND_CONFIG FLAGOS_METAX_CUDART_SHIM

source "$GITHUB_WORKSPACE/.github/scripts/lib/set_env_common.sh"
PIP_RETRY_PYTHON="$(command -v python)" pip_retry 'Cython==3.2.4'
if [[ "${1:-}" == --test-install && "${TORCH_FL_CI_NATIVE_HOST_CACHE:-0}" == 1 &&
      ( "$FLAGOS_ACCELERATOR" == hygon || "$FLAGOS_ACCELERATOR" == dcu ) ]]; then
  python "$GITHUB_WORKSPACE/.github/scripts/hygon_native_host_cache.py" "$evidence" \
    2>&1 | tee "$evidence/build.log"
else
  python setup.py bdist_wheel --dist-dir "$evidence" 2>&1 | tee "$evidence/build.log"
fi
python "$preflight" --wheel "$evidence"/*.whl \
  --platform "$FLAGOS_ACCELERATOR" --check-build-env --check-native-host-backend \
  | tee "$evidence/preflight.log"
if [[ "${1:-}" == --test-install ]]; then
  # The common suite keeps its ordinary wheel. Native tests select this
  # verified package in a separate interpreter.
  install="$RUNNER_TEMP/torch-fl-native-host-install"
  python -m pip install --no-deps --upgrade --target "$install" "$evidence"/*.whl
  export PYTHONPATH="$install${PYTHONPATH:+:$PYTHONPATH}"
else
  python -m pip install --force-reinstall --no-deps "$evidence"/*.whl
fi

if [[ "${1:-}" == --test-install ]]; then
  python "$preflight" --installed --platform "$FLAGOS_ACCELERATOR" \
    --check-build-env --check-native-host-backend | tee "$evidence/installed-preflight.log"
fi
python "$record_wheel" --wheel "$evidence"/*.whl --evidence "$evidence"

if [[ "${1:-}" == --metax-precision ]]; then
  # Copy the tests into a wheel-only test root so pytest cannot import the
  # source package. Exercise the same native suites as CI and the vendor cohort.
  unset FLAGOS_VENDOR_TORCH_LIB FLAGOS_VENDOR_TORCH_VERSION
  test_root="$(mktemp -d "$RUNNER_TEMP/torch-fl-metax-precision-tests.XXXXXX")"
  cp -a tests pyproject.toml "$test_root/"
  cd "$test_root"
  torch-fl-preflight --installed --platform metax --check-build-env \
    --check-native-host-backend | tee "$evidence/installed-preflight.log"
  python -c 'from torch_fl import _build_config; assert _build_config.NATIVE_VENDOR'
  python -m pytest tests/integration/ops/test_native_metax_hosts.py \
    tests/integration/ops/test_native_metax_qwen_routes.py \
    tests/manual/test_native_metax_vendor_precision.py -v --tb=short \
    --junitxml="$evidence/conformance.xml" | tee "$evidence/conformance.log"
fi
