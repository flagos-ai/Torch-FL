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

"""A platform has two names: its vendor, and the id the build table keys on.

cmake/flagos_platforms.json carries both -- the accelerator key (cuda, dcu, ...)
that CMake, setup.py, the wheel filename and the CI lane files key on, and the
``vendor`` that names the company (nvidia, hygon, ...) and, where the platform
publishes, the ``<lane>`` of flagos-pypi-<lane>. A build accepts either.

The copies live in three places that cannot import each other (setup.py is a
script, ``torch_fl/_platform.py`` must stay stdlib-only for the test conftest),
so each copy is pinned to the JSON here instead of being derived at run time.
The chip layer (Ascend 310B/910B, ...) is source-only by design: these tests
also pin that no vendor or chip name reaches a conf or an accelerator id.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE = REPO_ROOT / "cmake" / "flagos_platforms.json"


def _platforms() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))["accelerators"]


def _load_platform_module():
    """torch_fl/_platform.py by path -- importing torch_fl would import torch."""
    path = REPO_ROOT / "torch_fl" / "_platform.py"
    spec = importlib.util.spec_from_file_location("_flagos_platform_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def platform_mod():
    return _load_platform_module()


def test_every_platform_declares_a_vendor():
    missing = [key for key, row in _platforms().items() if not row.get("vendor")]
    assert not missing, f"accelerators without a vendor: {missing}"


def test_vendor_names_are_identifiers_and_unique_per_platform():
    vendors = {key: row["vendor"] for key, row in _platforms().items()}
    bad = [v for v in vendors.values() if not re.fullmatch(r"[a-z][a-z0-9_]*", v)]
    assert not bad, (
        f"vendor names must be identifiers: {bad}. drobotics is the "
        "hyphen-free spelling of Nexus's d-robotics lane for this reason."
    )
    # One platform per vendor: the vendor is the lane, and a lane is one
    # publishing target. A second platform claiming it would mean two wheels
    # publishing over each other.
    seen = {}
    for key, vendor in vendors.items():
        assert vendor not in seen, f"{key} and {seen[vendor]} both claim {vendor}"
        seen[vendor] = key


def test_vendor_matches_the_publishing_lane_where_there_is_one():
    """The vendor IS the lane name, so a published platform cannot disagree."""
    mismatched = []
    for key, row in _platforms().items():
        lane = row.get("pypi_lane", "")
        if lane and lane != row["vendor"]:
            mismatched.append((key, row["vendor"], lane))
    assert not mismatched, (
        f"vendor and pypi_lane disagree for {mismatched}; the vendor is defined "
        "as the flagos-pypi-<lane> the wheel publishes to"
    )


def test_platform_module_vendor_table_matches_the_build_table(platform_mod):
    expected = {key: row["vendor"] for key, row in _platforms().items()}
    assert platform_mod.PLATFORM_VENDOR == expected


def test_platform_module_accepts_both_halves(platform_mod):
    for key, row in _platforms().items():
        assert platform_mod.platform_for(key) == key
        assert platform_mod.platform_for(row["vendor"]) == key
        assert platform_mod.vendor_for(key) == row["vendor"]
    # The older MetaX spelling stays accepted; a build record can carry it.
    assert platform_mod.platform_for("maca") == "metax"
    assert platform_mod.platform_for("nope") == ""


def test_setup_normalizes_the_same_way(platform_mod):
    """setup.py's own copy of the vendor map, read by AST -- setup.py is a script."""
    setup_src = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    body = re.search(r"_VENDOR_TO_PLATFORM = \{(.*?)\n\}", setup_src, re.S)
    assert body, "setup.py no longer defines _VENDOR_TO_PLATFORM"
    pairs = dict(re.findall(r'"([a-z0-9_-]+)":\s*"([a-z0-9_]+)"', body.group(1)))
    assert pairs == {
        vendor: key for key, vendor in platform_mod.PLATFORM_VENDOR.items()
    }
    # The Nexus spelling of D-Robotics is accepted as an alias.
    assert 'value == "d-robotics"' in setup_src


def test_no_chip_name_reaches_a_conf_or_an_accelerator_id():
    """The chip layer is source-only: an Ascend 310B is an `ascend` build."""
    platform_ids = set(_platforms()) | {row["vendor"] for row in _platforms().values()}
    chip_words = ("310b", "910b", "910_9382", "a100", "a800", "h100")
    offenders = sorted(pid for pid in platform_ids if pid in chip_words)
    assert not offenders, f"chip names in the platform/vendor tables: {offenders}"
    conf_dir = REPO_ROOT / "torch_fl" / "configs"
    for conf in conf_dir.glob("backends_*.conf"):
        # Route lines only: the header notes record which hardware a measurement
        # was taken on ("surveyed on 8x A100"), which is evidence, not a route.
        routes = [
            line.split("#", 1)[0].lower()
            for line in conf.read_text(encoding="utf-8").splitlines()
        ]
        hit = [w for w in chip_words if any(w in line for line in routes)]
        assert not hit, f"{conf.name} routes on a chip name: {hit}"
