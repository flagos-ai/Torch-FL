# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Shared cache selection for Ascend, Iluvatar, Kunlunxin, MUSA and PPU. No package
# installation or device access belongs here. The optional key is supplied only
# by the integration workflow for its dedicated runner-cache directories.
pip_cache_try_directory() {
  local candidate="$1" key="${2:-}" probe="" key_file="" old_key=""
  [[ "$candidate" != *$'\n'* && "$candidate" != *$'\r'* ]] || return 1
  [[ ! -L "$candidate" ]] || return 1
  if [[ -e "$candidate" && ( ! -d "$candidate" || ! -O "$candidate" ) ]]; then
    return 1
  fi
  mkdir -p "$candidate" 2>/dev/null || return 1
  [[ -O "$candidate" && -w "$candidate" && -x "$candidate" ]] || return 1
  candidate="$(cd "$candidate" && pwd -P)" || return 1
  probe="$(mktemp "$candidate/.torch-fl-pip-probe.XXXXXX" 2>/dev/null)" || return 1
  rm -f "$probe" || return 1
  if [[ -n "$key" ]]; then
    key_file="$candidate/.torch-fl-ci-cache-key"
    [[ ! -L "$key_file" ]] || return 1
    if [[ -f "$key_file" ]]; then
      old_key="$(cat "$key_file")" || return 1
    elif [[ -f "$candidate.key" ]]; then
      old_key="$(cat "$candidate.key")" || return 1
    fi
    if [[ -n "$old_key" && "$old_key" != "$key" ]]; then
      echo "pip cache key changed; discarding previous downloads at $candidate"
      # Preserve the directory itself. Do not follow symlinks or change owners.
      # find includes hidden pip metadata as well as wheel/http directories.
      find "$candidate" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || return 1
    fi
    printf '%s\n' "$key" > "$key_file" || return 1
  fi
  export PIP_CACHE_DIR="$candidate"
}

prepare_pip_cache() {
  local platform="$1" key="${2:-}" requested="" candidate="" uid="" selected=0
  local job_root="${RUNNER_TEMP:-${REPO_ROOT:-$PWD}/.ci}"
  case "$platform" in
    ascend|iluvatar|kunlunxin|mthreads|thead) ;;
    *) echo "::error::Unsupported shared pip-cache platform: $platform" >&2; return 1 ;;
  esac
  export PIP_RETRY_NO_CACHE="${PIP_RETRY_NO_CACHE:-0}"
  # pip treats even PIP_NO_CACHE_DIR=0/false as disabling the cache.
  if [[ -n "${PIP_NO_CACHE_DIR:-}" ]]; then export PIP_RETRY_NO_CACHE=1; fi
  if [[ "$PIP_RETRY_NO_CACHE" != 1 ]]; then
    requested="${PIP_CACHE_DIR:-$job_root/pip-cache/$platform}"
    uid="$(id -u)"
    for candidate in "$requested" "$requested-uid-$uid" \
        "$job_root/pip-cache/$platform"; do
      if pip_cache_try_directory "$candidate" "$key"; then
        selected=1
        break
      fi
      echo "::warning::Cannot use pip cache at $candidate (uid=$uid); trying another directory" >&2
      if [[ -e "$candidate" ]]; then ls -ldn "$candidate" >&2 || true; fi
    done
    if [[ "$selected" == 0 ]]; then
      candidate="$(mktemp -d "${TMPDIR:-/tmp}/torch-fl-pip-$platform.XXXXXX" 2>/dev/null)" || candidate=""
      if [[ -n "$candidate" ]] && pip_cache_try_directory "$candidate" "$key"; then
        selected=1
      fi
    fi
    if [[ "$selected" == 0 ]]; then
      echo "::warning::No usable pip-cache directory; continuing without download caching" >&2
      export PIP_RETRY_NO_CACHE=1
    fi
  fi
  if [[ "$PIP_RETRY_NO_CACHE" == 1 ]]; then
    export PIP_CACHE_DIR=""
    export PIP_NO_CACHE_DIR=1
    echo "pip download cache disabled for $platform"
  else
    export PIP_NO_CACHE_DIR=""
    echo "pip download cache: $PIP_CACHE_DIR (uid=$(id -u))"
  fi
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf 'PIP_CACHE_DIR=%s\nPIP_RETRY_NO_CACHE=%s\nPIP_NO_CACHE_DIR=%s\n' \
      "$PIP_CACHE_DIR" "$PIP_RETRY_NO_CACHE" "$PIP_NO_CACHE_DIR" >> "$GITHUB_ENV"
  fi
}
