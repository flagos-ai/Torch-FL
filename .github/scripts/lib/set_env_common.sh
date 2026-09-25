# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared helpers for the set_env_*.sh platform provisioning scripts.
#
# Sourced (not executed) by each script after REPO_ROOT and the version pins are
# set. These five functions were copied verbatim into every script -- `pip_retry`
# in five slightly different spellings, `flag_gems_installed`/`install_flag_gems`
# in six copies -- so a fix had to be applied seven times and was easy to miss.
#
# The interpreter is taken from `${VENV_PYTHON:-python}`: every script but MetaX
# sets VENV_PYTHON to its job-local venv; MetaX runs the image's /opt/venv via
# PATH and falls through to `python`. `pip_retry` additionally honours
# PIP_RETRY_PYTHON (a per-call override, used by the CUDA script's several
# interpreters), PIP_RETRY_TIMEOUT (default 300s) and PIP_RETRY_NO_CACHE
# (default 1; the CUDA script turns it off to reuse its build cache).
#
# The scripts must keep `set -euo pipefail` in effect; every variable read here
# is either set by the caller or has a default.

# pip install with retries for large wheels on unstable networks.
pip_retry() {
  local python_exe="${PIP_RETRY_PYTHON:-${VENV_PYTHON:-python}}"
  local timeout="${PIP_RETRY_TIMEOUT:-300}"
  local -a extra=()
  if [[ "${PIP_RETRY_NO_CACHE:-1}" == "1" ]]; then
    extra+=(--no-cache-dir)
  fi
  local attempt=1
  while true; do
    # Raise pip's own retry limit and timeout for large wheels on unstable
    # networks: the FlagTree wheel is hundreds of MB, and pip's default timeout
    # (15s) and retries (5) are not enough when the mirror link drops
    # mid-download.
    if "$python_exe" -m pip install --retries 10 --timeout "$timeout" "${extra[@]}" "$@"; then
      return 0
    fi
    if (( attempt >= 5 )); then
      echo "::error::pip install failed after $attempt attempts: $*"
      return 1
    fi
    echo "::warning::pip install attempt $attempt failed; retrying: $*"
    attempt=$((attempt + 1))
    sleep 10
  done
}

# A published wheel has a fixed version and an index-provided SHA-256. Check the
# prebuilt venv before downloading it, then verify the result so a stale image
# package cannot silently take precedence over the requested release.
flag_gems_installed() {
  "${VENV_PYTHON:-python}" - "$FLAGGEMS_VERSION" <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

try:
    assert metadata.version("flag_gems") == sys.argv[1]
    assert importlib.util.find_spec("flag_gems") is not None
except (AssertionError, metadata.PackageNotFoundError):
    raise SystemExit(1)
PY
}

# Install only the binary release, never a source distribution or Git checkout.
install_flag_gems() {
  if flag_gems_installed; then
    echo "Using preinstalled FlagGems $FLAGGEMS_VERSION"
    return 0
  fi
  pip_retry --no-deps --only-binary=:all: --index-url "$FLAGGEMS_INDEX_URL" \
    "flag-gems===$FLAGGEMS_VERSION"
  if ! flag_gems_installed; then
    echo "::error::FlagGems $FLAGGEMS_VERSION is not importable after wheel installation"
    return 1
  fi
}

# Drop a vendor-torch root from a colon-separated path list, so the isolated
# venv does not inherit the image's vendor torch.
strip_vendor_paths() {
  local value="${1:-}"
  local entry
  local -a entries=()
  local -a kept=()
  IFS=: read -ra entries <<< "$value"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      "$VENDOR_TORCH_ROOT"|"$VENDOR_TORCH_ROOT"/*) ;;
      *) kept+=("$entry") ;;
    esac
  done
  local joined=""
  for entry in "${kept[@]}"; do
    joined="${joined:+$joined:}$entry"
  done
  printf '%s' "$joined"
}

# True when the venv interpreter exists and has pip.
venv_is_usable() {
  [[ -x "$VENV_PYTHON" ]] || return 1
  "$VENV_PYTHON" -m pip --version >/dev/null 2>&1
}
