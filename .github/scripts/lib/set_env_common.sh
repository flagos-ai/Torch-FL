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

# The FlagGems install is a VCS install, and pip reports the exit status of its
# last step -- the wheel build of whichever tree it managed to fetch. A checkout
# the runner's proxy truncated therefore still ends in `Successfully installed`,
# and pip never notices. On 2026-09-18 the Ascend runner's clone spent ten
# minutes printing
#   fatal: unable to access 'https://github.com/flagos-ai/FlagGems.git/':
#   Proxy CONNECT aborted
# (364 times), alongside `error: unable to read sha1 file of ...` and
# `error: invalid object 100644 2e574121... for
# '.github/workflows/rule-check.yaml'` for the blobs it never received, and then
# reported
#   Successfully installed flaggems_setup-0.0.0
# -- a 2.1 MB stub named after the build scaffolding rather than the project,
# where the same revision produced the 10 MB
# flag_gems-5.4.0rc2.post1+g437ba3938 on every other platform that ran that
# morning. Nothing failed until the integration suite took its first FlagGems
# route, four minutes later.
#
# So check the install instead of trusting pip's status, and reinstall when it
# is wrong: the conf routes this platform's operators to flagos_python, so an
# unusable flag_gems is not a state a provisioning script may leave behind.
#
# Three attempts at most, and only for an install pip called successful: a pip
# failure has already been retried five times by pip_retry, and repeating that
# spends the job's budget on a link that is down rather than on a bad checkout.
flag_gems_installed() {
  "${VENV_PYTHON:-python}" - "${FLAGGEMS_REVISION:0:9}" <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

try:
    version = metadata.version("flag_gems")
except metadata.PackageNotFoundError:
    raise SystemExit("flag_gems is not installed")

# A distribution can be installed with no importable package behind it, which
# is what a truncated checkout produces.
if importlib.util.find_spec("flag_gems") is None:
    raise SystemExit(f"flag_gems {version} has no importable package")

print(f"flag_gems {version}")
if sys.argv[1] not in version:
    # Not a failure: a revision given as a branch name, or a tarball without
    # git metadata, lands on a version string that names neither. Only warn --
    # reinstalling cannot change how the version was written.
    print(
        f"::warning::flag_gems {version} does not name the requested revision "
        f"{sys.argv[1]}; the checkout it was built from may be incomplete",
        file=sys.stderr,
    )
PY
}

# Install FlagGems from the pinned revision, retrying until it is importable.
install_flag_gems() {
  local attempt=1
  while true; do
    if ! pip_retry --no-deps "git+${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"; then
      echo "::error::could not install FlagGems from ${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"
      return 1
    fi
    if flag_gems_installed; then
      return 0
    fi
    if (( attempt >= 3 )); then
      echo "::error::no usable flag_gems after $attempt installs of ${FLAGGEMS_REPO}@${FLAGGEMS_REVISION}"
      return 1
    fi
    echo "::warning::install attempt $attempt left no usable flag_gems; reinstalling"
    attempt=$((attempt + 1))
    # A truncated tree installs under the build scaffolding's name, so both
    # names have to go for the next attempt to be read as a fresh result.
    "${VENV_PYTHON:-python}" -m pip uninstall -y flag_gems flaggems_setup >/dev/null 2>&1 || true
    sleep 10
  done
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
