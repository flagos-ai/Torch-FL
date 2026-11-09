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
import os
import re
import subprocess
import sys
from email.parser import Parser
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / ".github" / "scripts"
PINS = REPO_ROOT / ".github" / "version-pins.env"
PLATFORM_TABLE = REPO_ROOT / "cmake" / "flagos_platforms.json"

# The lane names: a hook, a pin and a config are each named for the
# platform's vendor, which is also its flagos-pypi-<lane>.
PLATFORMS = [
    "nvidia",
    "ascend",
    "hygon",
    "enflame",
    "iluvatar",
    "metax",
    "mthreads",
    "thead",
]
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
    text = (HOOKS_DIR / "set_env_nvidia.sh").read_text(encoding="utf-8")
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

# The one platform whose wheel declares no operator cohort at all. Apple
# Silicon ships the Metal runtime floor and no operator kernels: there is no
# Metal FlagTree/FlagGems line, and PyPI publishes no `triton` wheel for macOS
# (zero macosx files across every release), so the rule enforced below --
# FlagTree or Triton, plus FlagGems -- cannot be met without an unsatisfiable
# requirement. setup.py's mps arm returns torch and packaging only, and
# backends_apple.conf names the native MPSGraph kernels in its full-coverage
# table, but its MPSGraph kernels need no FlagTree/FlagGems/Triton operator
# cohort; the native source is the design of the platform rather than the
# omission this check exists to catch.
NO_OPERATOR_COHORT = {"mps"}


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


def _table_vendor(platform: str) -> str:
    return json.loads(PLATFORM_TABLE.read_text())["accelerators"][platform]["vendor"]


def _declared(platform: str) -> list:
    globals_ = _setup_globals()
    globals_["FLAGOS_ACCELERATOR"] = globals_["_normalize_accelerator"](platform)
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
        if platform in NO_OPERATOR_COHORT:
            continue
        assert f"flag_gems=={version}" in _declared(platform), platform


def test_flagcx_is_required_on_pinned_vendors_except_cuda():
    pins = _pins()
    for platform in _platforms_in_the_table():
        declared = [r for r in _declared(platform) if r.startswith("flagcx")]
        version = pins.get(f"FLAGCX_VERSION_{platform}")
        if version and platform != "cuda":
            assert declared == [f"flagcx=={version}"], platform
        else:
            assert declared == [], platform


def test_cuda_communication_extra_uses_the_pinned_flagcx():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    extra = re.search(r'^cuda-comm = \["([^"]+)"\]', text, re.M)
    assert extra is not None
    assert extra.group(1) == f"flagcx=={_pins()['FLAGCX_VERSION_cuda']}"


@pytest.mark.parametrize("accelerator", ["cuda", "nvidia"])
def test_cuda_package_metadata_makes_flagcx_an_explicit_extra(tmp_path, accelerator):
    """Verify generated package metadata without simulating an installation."""
    from packaging.requirements import Requirement

    output = tmp_path / "metadata"
    output.mkdir()
    done = subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(output)],
        cwd=REPO_ROOT,
        env={**os.environ, "FLAGOS_ACCELERATOR": accelerator},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    metadata_path = next(output.glob("*.egg-info/PKG-INFO"))
    metadata_text = metadata_path.read_text()
    metadata = Parser().parsestr(metadata_text)
    requirements = [Requirement(raw) for raw in metadata.get_all("Requires-Dist", [])]
    flagcx = [req for req in requirements if req.name == "flagcx"]
    assert len(flagcx) == 1
    assert not flagcx[0].marker.evaluate({"extra": ""})
    assert flagcx[0].marker.evaluate({"extra": "cuda-comm"})
    assert str(flagcx[0].specifier) == f"=={_pins()['FLAGCX_VERSION_cuda']}"
    assert "cuda-comm" in metadata.get_all("Provides-Extra", [])


@pytest.mark.parametrize(
    ("enabled", "cuda_version", "expected_code"),
    [("0", "13.0.88", 0), ("0", "13.3.0", 0), ("1", "13.3.0", 0), ("1", "13.0.88", 1)],
)
def test_cuda_flagcx_provisioning_requires_explicit_compatible_opt_in(
    enabled, cuda_version, expected_code, tmp_path
):
    hook = (HOOKS_DIR / "set_env_nvidia.sh").read_text()
    branch = hook.split("# CUDA functional CI", 1)[1].split(
        "# Install the published FlagGems wheel", 1
    )[0]
    branch = "# CUDA functional CI" + branch
    script = (
        'set -euo pipefail\nsource "$1"\npip_retry() { printf "INSTALL %s\\n" "$*"; }\n'
        + branch
        + "\nexport_ci_env FLAGCX_TORCH_BACKEND\n"
        + '\nprintf "AUTOLOAD=%s TLE=%s\\n" "${TORCH_DEVICE_BACKEND_AUTOLOAD:-}" "${USE_TLE_DIST:-}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script, "cuda-test", str(SCRIPTS_DIR / "lib/set_env_common.sh")],
        env={
            "PATH": "/usr/bin:/bin",
            "TORCH_FL_CUDA_FLAGCX_WHEEL": enabled,
            "CUDA_VERSION": cuda_version,
            "VENV_PYTHON": sys.executable,
            "FLAGGEMS_INDEX_URL": "https://example.invalid/simple",
            "FLAGCX_VERSION": _pins()["FLAGCX_VERSION_cuda"],
            "GITHUB_ENV": str(tmp_path / "github-env"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    if enabled == "0":
        assert "INSTALL" not in result.stdout
        assert "FlagCX communication is not validated" in result.stdout
        assert "TLE=0" in result.stdout
        assert (tmp_path / "github-env").read_text() == "FLAGCX_TORCH_BACKEND=vendor\n"
    elif expected_code == 0:
        assert f"flagcx==={_pins()['FLAGCX_VERSION_cuda']}" in result.stdout
        assert "AUTOLOAD=0" in result.stdout
    else:
        assert "INSTALL" not in result.stdout
        assert "CUDA 13.3" in result.stderr


def test_triton_is_declared_only_where_no_flagtree_build_is_pinned():
    pins = _pins()
    for platform in _platforms_in_the_table():
        if platform in NO_OPERATOR_COHORT:
            continue
        declared = _declared(platform)
        flagtree = any(r.startswith("flagtree==") for r in declared)
        assert flagtree == bool(
            pins.get(f"FLAGTREE_VERSION_{_table_vendor(platform)}")
        ), platform
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
        globals_["FLAGOS_ACCELERATOR"] = globals_["_normalize_accelerator"](platform)
        # Read it back off the kwargs setup() is called with, not off the helper:
        # a helper that computes the right string while setup() still hard-codes
        # `>=3.8` is exactly the regression this is here for.
        declared = globals_["_get_setup_kwargs"]()["python_requires"]
        if pinned:
            # The minor line, not the exact version: `==3.12` does not match
            # 3.12.3, and every interpreter reports a patch release.
            major, minor = pinned.split(".")[:2]
            assert declared == f">={major}.{minor},<{major}.{int(minor) + 1}", (
                platform,
                declared,
            )
        else:
            # No FlagTree, so no lane and no artifact either.
            assert declared == ">=3.8", (platform, declared)


def test_setup_kwargs_survive_a_boxing_lane_walking_the_table(monkeypatch):
    """A boxing lane's FLAGOS_BUILD_BOXING=1 must not leak onto another platform.

    The DCU, MetaX and PPU lanes still export FLAGOS_BUILD_BOXING=1, because
    their wheels keep the CUDA boxing path; the CUDA lane deliberately does not,
    since it now tests the FlagGems-primary default. That value is inherited by
    the unit tests, including the loop above, which points FLAGOS_ACCELERATOR at
    every accelerator in the table. One of them -- iluvatar -- pins boxing off,
    so resolving the switch while computing another platform's requirements
    raises and fails the DCU lane. `_get_setup_kwargs()` describes the wheel for
    the platform it is asked about, so it must not reject an environment value
    that belongs to a different platform's build.
    """
    globals_ = _setup_globals()
    monkeypatch.setenv("FLAGOS_BUILD_BOXING", "1")
    for platform in _platforms_in_the_table():
        globals_["FLAGOS_ACCELERATOR"] = globals_["_normalize_accelerator"](platform)
        assert globals_["_get_setup_kwargs"]()["python_requires"], platform


def test_cuda_runtime_wheels_still_follow_boxing(monkeypatch):
    """The tolerant table walk must not drop the CUDA runtime-wheel policy.

    Scoping the switch lookup to CUDA is what keeps the walk above working; the
    cuda case itself has to keep pulling nvidia-*-cu12 in on the boxing opt-in
    and leaving them out for the FlagGems-primary default.
    """
    globals_ = _setup_globals()
    monkeypatch.setitem(globals_, "FLAGOS_ACCELERATOR", "cuda")
    monkeypatch.delenv("FLAGOS_BUILD_BOXING", raising=False)
    # A PPU-style environment exports this; it would hide the runtime wheels the
    # boxing opt-in is supposed to pull in, so clear it before the two halves.
    monkeypatch.delenv("FLAGOS_SKIP_CUDA_ASSETS", raising=False)
    assert not [r for r in globals_["_install_requires"]() if r.startswith("nvidia-")]
    monkeypatch.setenv("FLAGOS_BUILD_BOXING", "1")
    assert [r for r in globals_["_install_requires"]() if r.startswith("nvidia-")]


def test_only_an_sdk_segment_comes_with_an_interpreter_pin():
    """The two travel together: both describe the one published artifact.

    An SDK segment (``+cuda13.3``) names the FlagTree build the wheel links, so
    it must come with that build's interpreter. A platform-id segment (``+mps``)
    names the wheel's platform and nothing else -- MPS links no FlagTree -- so it
    must not borrow a pin it does not have; ``Requires-Python`` then falls back
    to the base builds' range.
    """
    pins = _pins()
    for platform in _platforms_in_the_table():
        segment = _wheel_local(platform)
        if segment == platform:
            assert f"FLAGTREE_PYTHON_VERSION_{platform}" not in pins, platform
        else:
            assert bool(segment) == bool(
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
