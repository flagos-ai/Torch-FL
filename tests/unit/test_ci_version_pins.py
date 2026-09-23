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

"""The set_env_*.sh scripts must get their version pins from one file.

Each platform script used to hard-code the CPU-torch pin, the FlagTree index,
the FlagGems repo/revision and its FlagTree wheel -- so bumping a version meant
seven edits and it was easy to update six and miss one. Those values now live in
`.github/version-pins.env`, which every script sources. This checks the contract
holds: the pin file defines what the scripts read, and no script keeps a literal
copy of a shared pin as its fallback.

Pure text, no torch: the scripts cannot be executed here (they provision vendor
environments), but the pin wiring is fully visible in their source.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
PINS = REPO_ROOT / ".github" / "version-pins.env"

PLATFORMS = ["cuda", "ascend", "dcu", "gcu", "metax", "musa", "ppu"]

SHARED_PINS = [
    "CPU_TORCH_VERSION_DEFAULT",
    "CPU_TORCH_INDEX_URL_DEFAULT",
    "FLAGTREE_INDEX_URL_DEFAULT",
    "FLAGTREE_PYTHON_VERSION_DEFAULT",
    "FLAGTREE_MIN_GLIBC_DEFAULT",
    "FLAGGEMS_REPO_DEFAULT",
    "FLAGGEMS_REVISION_DEFAULT",
    "PIP_INDEX_URL_DEFAULT",
]

# Literal fallbacks that must no longer appear in a `TORCH_FL_*:-<literal>}`
# default; each must come from the pin file instead.
FORBIDDEN_LITERAL_DEFAULTS = [
    ":-2.10.0}",
    ":-https://download.pytorch.org",
    ":-https://resource.flagos.net",
    ":-437ba393",
    ":-https://github.com/flagos-ai/FlagGems",
    ":-https://pypi.org/simple",
]


def _pins() -> dict[str, str]:
    values = {}
    for line in PINS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def test_pin_file_defines_every_shared_pin():
    pins = _pins()
    for key in SHARED_PINS:
        assert key in pins and pins[key], key
    # One FlagTree wheel per platform script.
    for platform in PLATFORMS:
        assert pins.get(f"FLAGTREE_VERSION_{platform}"), platform


def test_every_script_sources_the_pin_file():
    for platform in PLATFORMS:
        text = (SCRIPTS_DIR / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        assert 'source "${REPO_ROOT}/.github/version-pins.env"' in text, platform


def test_scripts_read_their_platform_flagtree_pin():
    for platform in PLATFORMS:
        text = (SCRIPTS_DIR / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        assert (
            f'FLAGTREE_VERSION="${{TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_{platform}}}"'
            in text
        ), platform


def test_no_script_keeps_a_literal_copy_of_a_shared_pin():
    """The grep that proves a pin bump is one edit, not seven."""
    for platform in PLATFORMS:
        text = (SCRIPTS_DIR / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        for literal in FORBIDDEN_LITERAL_DEFAULTS:
            assert literal not in text, f"{platform}: still hard-codes {literal!r}"


def test_pin_values_are_not_empty_or_placeholders():
    pins = _pins()
    for key, value in pins.items():
        assert value, key
        assert not re.search(r"<[^>]+>", value), (
            f"{key}={value!r} looks like a placeholder"
        )
