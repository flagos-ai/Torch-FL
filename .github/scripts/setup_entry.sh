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
  local name=torch-fl-platform-setup.sh
  local entry="${RUNNER_TEMP:?}/$name"
  local setup_bash="${SETUP_BASH:?}"
  [[ "$platform_workspace" = /* ]] && test -d "$platform_workspace"
  [[ "$setup_script" != */* ]] || { [[ "$setup_script" = /* ]] && test -f "$setup_script"; }
  if [[ "$setup_bash" == */* ]]; then
    [[ "$setup_bash" = /* ]] && test -x "$setup_bash"
  fi
  # Capture the profile's final exported state and remove BASH_ENV before
  # setup. Reloading the profile could repeat side effects recursively.
  printf 'set -euo pipefail\nunset BASH_ENV\n' > "$entry"
  printf '%s\n' "$(declare -p -x CI_STAGE INTEGRATION_PROFILE PATH)" \
    >> "$entry"
  printf 'export GITHUB_WORKSPACE=%q\n' "$platform_workspace" >> "$entry"
  printf 'cd %q\n' "$platform_workspace" >> "$entry"
  printf 'exec %q "$GITHUB_WORKSPACE/%s" --platform %q\n' \
    "$setup_bash" "$setup_script" "$SETUP_PLATFORM" >> "$entry"
}
