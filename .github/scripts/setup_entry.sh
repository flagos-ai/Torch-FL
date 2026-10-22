#!/usr/bin/env bash
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

# Source this file after selecting Bash. The profile can replace the platform
# entrypoint and is loaded last so its final exports reach the real setup.
create_setup_entry() {
  local setup_path="$1"
  local platform_workspace="$2"
  local setup_script="${SETUP_SCRIPT:?}"
  local entry="${RUNNER_TEMP:?}/torch-fl-platform-setup.sh"
  local setup_bash="${SETUP_BASH:?}"
  local stage_value="${CI_STAGE-}"
  local profile_value="${INTEGRATION_PROFILE-}"
  local path_value="${PATH-}"
  local path_defined="${PATH+x}"
  mkdir -p "$RUNNER_TEMP"
  if [[ "$platform_workspace" != /* || ! -d "$platform_workspace" ]]; then
    echo "::error::the integration workspace is not an absolute directory: $platform_workspace" >&2
    return 1
  fi

  if [[ "$setup_bash" == */* ]] &&
    { [[ "$setup_bash" != /* ]] || [[ ! -x "$setup_bash" ]]; }; then
    echo "::error::the selected Bash is not an absolute executable: $setup_bash" >&2
    return 1
  fi
  # Capture the profile's final exported state and remove BASH_ENV before
  # setup. Reloading the profile could repeat its side effects.
  printf 'set -euo pipefail\nunset BASH_ENV\n' > "$entry"
  if [[ -z "$stage_value" ]]; then
    echo "::error::the profile did not export a non-empty CI_STAGE" >&2
    return 1
  fi
  if [[ -z "$profile_value" ]]; then
    echo "::error::the profile did not export a non-empty INTEGRATION_PROFILE" >&2
    return 1
  fi
  if [[ -z "$path_defined" || -z "$path_value" ]]; then
    echo "::error::the profile did not export a non-empty PATH" >&2
    return 1
  fi
  local variable value
  for variable in CI_STAGE INTEGRATION_PROFILE PATH; do
    if [[ -z "${!variable:-}" ]]; then
      echo "::error::the profile did not export a non-empty $variable" >&2
      return 1
    fi
    printf -v value '%q' "${!variable}"
    printf 'export %s=%s\n' "$variable" "$value" >> "$entry"
  done
  printf 'export GITHUB_WORKSPACE=%q\n' "$platform_workspace" >> "$entry"
  printf 'cd %q\n' "$platform_workspace" >> "$entry"
  printf 'exec env -u BASH_ENV %q "$GITHUB_WORKSPACE/%s" --platform %q\n' \
    "$setup_bash" "$setup_script" "$SETUP_PLATFORM" >> "$entry"
}
