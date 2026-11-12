# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# The official runtime keeps vendor Torch in /flagos and FlagTree in
# /opt/flagtree. Reuse individual distributions, never either import path.
prepare_musa_runtime_environment() {
  local vendor_python="${TORCH_FL_VENDOR_PYTHON:-$(command -v python3.10 || true)}"
  export BASH_ENV=/dev/null
  export PYTHONPATH=""
  export PYTHONNOUSERSITE=1
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  if [[ ! -x "$vendor_python" ]] || ! "$vendor_python" -I -c \
      'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
    echo "::error::The MUSA runtime requires Python 3.10 for its FlagTree wheel" >&2
    return 1
  fi
  VENV_ROOT="${TORCH_FL_VENV_ROOT:-${RUNNER_TEMP:-$REPO_ROOT/.ci}/torch-fl-musa-${CI_STAGE}}"
  "$vendor_python" -I - "$VENV_ROOT" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

target, repo = (Path(p).resolve() for p in sys.argv[1:])
for prefix in (sys.prefix, sys.base_prefix):
    source = Path(prefix).resolve()
    if source.is_relative_to(target) or target.is_relative_to(source):
        raise SystemExit("CI venv must be separate from the runtime Python prefixes")
if repo.is_relative_to(target):
    raise SystemExit("CI venv must not contain the source checkout")
PY
  if ! "$vendor_python" -m venv --clear "$VENV_ROOT"; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "::error::Python venv creation failed and the runtime has no uv fallback" >&2
      return 1
    fi
    uv venv --clear --seed --python "$vendor_python" "$VENV_ROOT"
  fi
  VENV_PYTHON="$VENV_ROOT/bin/python"
  musa_venv_is_isolated
  "$vendor_python" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --target-python "$VENV_PYTHON" \
    --torch-version "${CPU_TORCH_VERSION%+cpu}" \
    --flagtree-version "${TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_mthreads}" \
    --flaggems-version "${TORCH_FL_FLAGGEMS_VERSION:-$FLAGGEMS_VERSION_DEFAULT}" \
    --requirements-file "$REPO_ROOT/.github/requirements-mthreads.txt"
  export VIRTUAL_ENV="$VENV_ROOT"
  export PATH="$VENV_ROOT/bin:$PATH"
  # A Python CMake distribution is required by pyproject.toml even when the
  # image supplies /usr/bin/cmake. Bootstrap pip before resolving CPU Torch.
  local requirement
  while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    [[ -z "$requirement" || "$requirement" == \#* ]] && continue
    musa_runtime_requirement "$requirement"
  done < "$REPO_ROOT/.github/requirements-mthreads.txt"
  if ! musa_cpu_torch_is_usable 2>/dev/null; then
    CPU_TORCH_VERSION="${CPU_TORCH_VERSION%+cpu}+cpu"
    install_cpu_torch
  fi
  musa_validate_build_tools
  if [[ "$CI_STAGE" == integration ]]; then
    for requirement in 'transformers>=4.51,<5' safetensors; do
      musa_runtime_requirement "$requirement"
    done
    if [[ "${TORCH_FL_INSTALL_QWEN_DEPS:-0}" == 1 ]]; then
      for requirement in sentencepiece tiktoken protobuf; do
        musa_runtime_requirement "$requirement"
      done
    fi
  fi
  export FLAGOS_SDK_VERSION=5.2.0
}

musa_runtime_requirement() {
  if "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
      --check-installed "$1"; then
    echo "Using prepared requirement: $1"
  else
    pip_retry --index-url "$PIP_INDEX_URL_ARG" "$1"
  fi
  "$VENV_PYTHON" "$REPO_ROOT/.github/scripts/seed_runtime.py" \
    --check-installed "$1"
}

musa_runtime_flagtree_is_usable() {
  "$VENV_PYTHON" - "$FLAGTREE_VERSION" <<'PY'
from importlib import metadata, util
from pathlib import Path
import sys

try:
    dist = metadata.distribution("flagtree")
    root = Path(dist.locate_file("triton")).resolve()
    spec = util.find_spec("triton")
    assert dist.version == sys.argv[1]
    assert root.is_relative_to(Path(sys.prefix).resolve())
    assert spec is not None and spec.origin
    assert Path(spec.origin).resolve().parent == root
    assert (root / "backends/mthreads").is_dir()
except (metadata.PackageNotFoundError, AssertionError):
    raise SystemExit(1)
PY
}
