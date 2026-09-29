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

# The one parameterized entrypoint for platform environment provisioning:
#
#   bash .github/scripts/set_env.sh --platform <cuda|ascend|dcu|gcu|metax|musa|ppu>
#
# It owns the boilerplate every platform shares -- the CI_STAGE check, REPO_ROOT,
# the version pins and the shared library -- then sources the platform hook at
# hooks/set_env_<platform>.sh, which is the platform-specific part (SDK root
# discovery, vendor interpreter choice, device/toolkit setup, vendor assets).
# The hooks call the shared phases (install_cpu_torch, install_flagtree,
# install_flag_gems, build_flagos_inplace, export_ci_env, ...) defined in
# lib/set_env_common.sh.
#
# The old set_env_<platform>.sh files are thin wrappers around this entrypoint.

set -euo pipefail

PLATFORM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      PLATFORM="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "usage: set_env.sh --platform <cuda|ascend|dcu|gcu|metax|musa|ppu>"
      exit 0
      ;;
    *)
      echo "::error::set_env.sh: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PLATFORM" ]]; then
  echo "::error::set_env.sh: --platform is required" >&2
  exit 2
fi

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Shared version pins (torch, FlagTree, FlagGems); see .github/version-pins.env.
# shellcheck source=.github/version-pins.env
source "${REPO_ROOT}/.github/version-pins.env"

# Shared helpers and provisioning phases.
# shellcheck source=.github/scripts/lib/set_env_common.sh
source "${REPO_ROOT}/.github/scripts/lib/set_env_common.sh"

# Reaching a host around the runner's HTTP proxy (prefer_direct_route). Shared
# with the release upload step, which hits the same Nexus host from the same pod.
# shellcheck source=.github/scripts/lib/proxy_route.sh
source "${REPO_ROOT}/.github/scripts/lib/proxy_route.sh"

# pip constraints for the provisioning venvs: the transitive pulls the pin table
# cannot reach. Exported so every pip install in the platform hook honours it.
export PIP_CONSTRAINT="${REPO_ROOT}/.github/constraints.txt"

HOOK="${REPO_ROOT}/.github/scripts/hooks/set_env_${PLATFORM}.sh"
if [[ ! -f "$HOOK" ]]; then
  echo "::error::set_env.sh: unknown platform '${PLATFORM}'; expected one of: ascend cuda dcu gcu metax musa ppu" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$HOOK"
