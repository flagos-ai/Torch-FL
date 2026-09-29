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
the FlagGems release and its FlagTree wheel -- so bumping a version meant
seven edits and it was easy to update six and miss one. Those values now live in
`.github/version-pins.env`, which every script sources. This checks the contract
holds: the pin file defines what the scripts read, and no script keeps a literal
copy of a shared pin as its fallback.

Pure text, no torch: the scripts cannot be executed here (they provision vendor
environments), but the pin wiring is fully visible in their source.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
PINS = REPO_ROOT / ".github" / "version-pins.env"
PLATFORM_TABLE = REPO_ROOT / "cmake" / "flagos_platforms.json"

PLATFORMS = ["cuda", "ascend", "dcu", "gcu", "metax", "musa", "ppu"]
RUNNER = SCRIPTS_DIR / "set_env.sh"
HOOKS_DIR = SCRIPTS_DIR / "hooks"

SHARED_PINS = [
    "CPU_TORCH_VERSION_DEFAULT",
    "CPU_TORCH_INDEX_URL_DEFAULT",
    "FLAGTREE_INDEX_URL_DEFAULT",
    "FLAGTREE_MIN_GLIBC_DEFAULT",
    "FLAGGEMS_VERSION_DEFAULT",
    "FLAGOS_WHEEL_ROOT_DEFAULT",
    "PIP_INDEX_URL_DEFAULT",
]

# Literal fallbacks that must no longer appear in a `TORCH_FL_*:-<literal>}`
# default; each must come from the pin file instead.
FORBIDDEN_LITERAL_DEFAULTS = [
    ":-2.10.0}",
    ":-https://download.pytorch.org",
    ":-https://resource.flagos.net",
    ":-5.4.0}",
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


def test_the_runner_sources_the_pin_file():
    """One entrypoint sources the pins; the hooks inherit them."""
    text = RUNNER.read_text(encoding="utf-8")
    assert 'source "${REPO_ROOT}/.github/version-pins.env"' in text


def test_scripts_read_their_platform_flagtree_pin():
    for platform in PLATFORMS:
        text = (HOOKS_DIR / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        assert (
            f'FLAGTREE_VERSION="${{TORCH_FL_FLAGTREE_VERSION:-$FLAGTREE_VERSION_{platform}}}"'
            in text
        ), platform


def test_the_cuda_hook_reads_its_own_python_pin():
    """CUDA was the only hook with a Python pin, and it read a shared default.

    That default meant the same number in two places: here and the per-platform
    block, whose purpose is that each platform has exactly one interpreter. The
    hook now reads `_cuda`, like it already reads `_cuda` for FlagTree.
    """
    text = (HOOKS_DIR / "set_env_cuda.sh").read_text(encoding="utf-8")
    assert (
        'FLAGTREE_PYTHON_VERSION="${TORCH_FL_FLAGTREE_PYTHON_VERSION:-$FLAGTREE_PYTHON_VERSION_cuda}"'
        in text
    )


def test_no_script_keeps_a_literal_copy_of_a_shared_pin():
    """The grep that proves a pin bump is one edit, not seven."""
    for platform in PLATFORMS:
        text = (HOOKS_DIR / f"set_env_{platform}.sh").read_text(encoding="utf-8")
        for literal in FORBIDDEN_LITERAL_DEFAULTS:
            assert literal not in text, f"{platform}: still hard-codes {literal!r}"


def test_pin_values_are_not_empty_or_placeholders():
    pins = _pins()
    for key, value in pins.items():
        assert value, key
        assert not re.search(r"<[^>]+>", value), (
            f"{key}={value!r} looks like a placeholder"
        )


# --- The other half of the contract: what the wheel declares -----------------
#
# setup.py builds its FlagTree/FlagGems/FlagCX requirements from this same pin
# file, so a `pip install torch_fl` pulls the cohort CI measured instead of a
# second, hand-kept copy of the versions. The two halves have to agree: a pin
# with no declaration is an install that omits the operator source, and a
# declaration with no pin is one that cannot resolve. These read setup.py's own
# function rather than a copy of the rule.

TRITON_AND_FLAGTREE = (
    "FlagTree installs a `triton` distribution, so declaring both is the "
    "Ascend '0 active drivers' failure; declaring neither leaves Triton to "
    "chance"
)


def _platforms_in_the_table() -> list:
    table = json.loads(PLATFORM_TABLE.read_text(encoding="utf-8"))
    return sorted(table["accelerators"])


@lru_cache(maxsize=1)
def _setup_globals() -> dict:
    """setup.py's real module globals.

    ``runpy.run_path`` returns a copy of the namespace, so patching the dict it
    hands back does not reach the functions that look names up; their own
    ``__globals__`` is the mapping to patch. Nothing here needs torch: setup.py
    reads the pin file and the platform table at import, and calls setup() only
    on the last line.
    """
    import runpy

    import setuptools

    original = setuptools.setup
    setuptools.setup = lambda **values: None
    try:
        namespace = runpy.run_path(str(REPO_ROOT / "setup.py"))
    finally:
        setuptools.setup = original
    return namespace["_flagos_sibling_requires"].__globals__


def _declared(platform: str) -> list:
    globals_ = _setup_globals()
    globals_["FLAGOS_ACCELERATOR"] = platform
    return globals_["_flagos_sibling_requires"]()


def test_the_wheel_declares_the_flagtree_pin_its_platform_script_installs():
    pins = _pins()
    for platform in PLATFORMS:
        version = pins[f"FLAGTREE_VERSION_{platform}"]
        assert f"flagtree=={version}" in _declared(platform), platform


def test_flag_gems_is_declared_everywhere_at_the_pinned_version():
    pins = _pins()
    version = pins["FLAGGEMS_VERSION_DEFAULT"]
    for platform in _platforms_in_the_table():
        assert f"flag_gems=={version}" in _declared(platform), platform


def test_flagcx_is_declared_exactly_where_the_pin_file_has_a_version():
    pins = _pins()
    for platform in _platforms_in_the_table():
        declared = [r for r in _declared(platform) if r.startswith("flagcx")]
        version = pins.get(f"FLAGCX_VERSION_{platform}")
        if version:
            assert declared == [f"flagcx=={version}"], platform
        else:
            assert declared == [], platform


def test_triton_is_declared_only_where_no_flagtree_build_is_pinned():
    pins = _pins()
    for platform in _platforms_in_the_table():
        declared = _declared(platform)
        flagtree = any(r.startswith("flagtree==") for r in declared)
        assert flagtree == bool(pins.get(f"FLAGTREE_VERSION_{platform}")), platform
        assert flagtree != any(r.startswith("triton") for r in declared), (
            platform,
            TRITON_AND_FLAGTREE,
        )


def test_setup_refuses_to_guess_when_the_pin_file_is_missing(monkeypatch):
    """A floor was what this replaced, and a floor resolves to an untested cohort.

    ``flagtree`` and ``flagcx`` are not on PyPI and ``flag_gems`` there is an
    older cohort, so there is no range that could stand in for the pins.
    """
    globals_ = _setup_globals()
    monkeypatch.setitem(globals_, "VERSION_PINS", str(REPO_ROOT / "absent.env"))
    with pytest.raises(RuntimeError, match="FlagTree/FlagGems/FlagCX"):
        globals_["_flagos_sibling_requires"]()


# --- The interpreter, and the SDK in the filename ----------------------------


def _wheel_local(platform: str) -> str:
    table = json.loads(PLATFORM_TABLE.read_text(encoding="utf-8"))
    return table["accelerators"][platform]["wheel_local"]


def test_requires_python_is_the_platforms_pinned_interpreter():
    """The wheel asks for the platform's pin -- one interpreter, not a range.

    What this checks is the wiring: that `setup.py` reads
    `FLAGTREE_PYTHON_VERSION_<platform>` and emits it as `==<version>`, so a
    renamed key or a reinstated `>=3.8` default fails here. What it cannot check
    is that the pin is *correct*, because the cp tag is a fact about the index
    (FlagTree publishes one wheel per interpreter) and not derivable offline --
    editing both the pin and this expectation together would still pass. The cp
    tag is verified against the index where the wheel is built; the filename
    carries it either way, which is why a wrong pin is a loud failure at install
    and not a silent one.
    """
    pins = _pins()
    globals_ = _setup_globals()
    for platform in _platforms_in_the_table():
        pinned = pins.get(f"FLAGTREE_PYTHON_VERSION_{platform}")
        globals_["FLAGOS_ACCELERATOR"] = platform
        # Read it back off the kwargs setup() is called with, not off the helper:
        # a helper that computes the right string while setup() still hard-codes
        # `>=3.8` is exactly the regression this is here for.
        declared = globals_["_get_setup_kwargs"]()["python_requires"]
        if pinned:
            assert declared == f"=={pinned}", (platform, declared)
        else:
            # No FlagTree, so no lane and no artifact either.
            assert declared == ">=3.8", (platform, declared)


def test_every_platform_with_a_wheel_segment_has_an_interpreter_pin():
    """The two travel together: both describe the one published artifact."""
    pins = _pins()
    for platform in _platforms_in_the_table():
        assert bool(_wheel_local(platform)) == bool(
            pins.get(f"FLAGTREE_PYTHON_VERSION_{platform}")
        ), platform


def test_the_wheel_segment_is_the_flagcx_pin_where_flagcx_exists():
    """One SDK spelling, not two.

    The segment names the SDK the wheel was built against, and for every
    platform that has a FlagCX build that is the same token FlagCX is published
    under (`+cuda13.3`, `+dtk2604`). Ascend was aligned to CANN for this reason.
    PPU has no FlagCX build yet, so it has no counterpart to check against.
    """
    pins = _pins()
    for platform in _platforms_in_the_table():
        flagcx = pins.get(f"FLAGCX_VERSION_{platform}")
        if not flagcx:
            continue
        _, _, sdk = flagcx.partition("+")
        assert sdk, (platform, flagcx)
        assert _wheel_local(platform) == sdk, (platform, _wheel_local(platform), sdk)


def test_wheel_segments_are_unique_across_platforms():
    """Two platforms sharing a segment would be two wheels with one name."""
    segments = [s for s in map(_wheel_local, _platforms_in_the_table()) if s]
    assert len(segments) == len(set(segments)), sorted(segments)
