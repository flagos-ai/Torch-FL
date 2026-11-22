#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Sourced by set_env.sh inside the pinned FlagOS TsingMicro runtime image.
if [[ "${GITHUB_ACTIONS:-}" == true && "${RUNNER_NAME:-}" != tsm-249 ]]; then
  printf '::error::Expected runner tsm-249, received %s\n' "${RUNNER_NAME:-unknown}" >&2
  exit 1
fi

export FLAGOS_ACCELERATOR=tsingmicro
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export GEMS_VENDOR=tsingmicro
export FLAGOS_BUILD_JOBS="${FLAGOS_BUILD_JOBS:-8}"

if [[ ! -f /usr/local/kuiper/include/tx_runtime.h || ! -f /usr/local/kuiper/lib/libhpgr.so ]]; then
  echo "::error::The runner's Kuiper SDK must be mounted at /usr/local/kuiper" >&2
  exit 1
fi
if ! sdk_info="$(/usr/local/kuiper/bin/tsm_smi --version)"; then
  printf '::error::Kuiper SDK could not initialize the runner driver\n%s\n' "$sdk_info" >&2
  exit 1
fi
FLAGOS_SDK_VERSION="$(awk -F '|' '/[|][[:space:]]*Runtime Version[[:space:]]*[|]/ {
  gsub(/[[:space:]]/, "", $3)
  if ($3 ~ /^[0-9]+([.][0-9]+)*$/) print $3
  exit
}' <<< "$sdk_info")"
export FLAGOS_SDK_VERSION

# The image's app venv includes torch_txda and a newer CPU torch. Only the
# SDK's MLIR Python bindings are shared with the new venv.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$LLVM_SYSPATH/python_packages/mlir_core"
VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-tsingmicro-${CI_STAGE}}"
"python$FLAGTREE_PYTHON_VERSION_tsingmicro" -m venv --clear "$VENV_ROOT"
VENV_PYTHON="$VENV_ROOT/bin/python"
export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$PATH"

CPU_TORCH_VERSION="${TORCH_FL_CPU_TORCH_VERSION:-$CPU_TORCH_VERSION_DEFAULT}"
CPU_TORCH_INDEX_URL="${TORCH_FL_CPU_TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL_DEFAULT}"
FLAGTREE_VERSION="${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_tsingmicro}"
FLAGTREE_INDEX_URL="${TORCH_FL_FLAGTREE_INDEX_URL:-$FLAGTREE_INDEX_URL_DEFAULT}"
FLAGGEMS_VERSION="${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}"
FLAGGEMS_INDEX_URL="${TORCH_FL_FLAGGEMS_INDEX_URL:-$FLAGOS_WHEEL_ROOT_DEFAULT/flagos-pypi-tsingmicro/simple}"

install_with_pypi_fallback --upgrade setuptools wheel 'cmake>=3.18' ninja build
install_cpu_torch
install_flagtree
install_flag_gems
install_with_pypi_fallback 'packaging>=26.0' 'PyYAML==6.0.1' 'sqlalchemy>=1.4.31,<2.1' 'numpy<2'

if [[ "$CI_STAGE" == integration ]]; then
  install_with_pypi_fallback pytest 'transformers>=4.51,<5' safetensors sentencepiece tiktoken protobuf
fi

if [[ "$CI_STAGE" == "build" ]]; then
  build_flagos_inplace
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$VENV_ROOT/bin" >> "$GITHUB_PATH"
fi
export_ci_env VIRTUAL_ENV PYTHONNOUSERSITE PYTHONPATH FLAGOS_ACCELERATOR \
  TORCH_DEVICE_BACKEND_AUTOLOAD GEMS_VENDOR FLAGOS_BUILD_JOBS FLAGTREE_VERSION \
  FLAGOS_SDK_VERSION
