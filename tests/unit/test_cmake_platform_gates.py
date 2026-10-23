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

"""Every CMake platform gate must hold for both spellings of the platform.

FLAGOS_ACCELERATOR carries whichever spelling the build was invoked with: the
table id (`cuda`, `mps`, `dcu`, ...) or its vendor (`nvidia`, `apple`,
`hygon`, ...), and both are advertised as legal. A gate written as
`FLAGOS_ACCELERATOR STREQUAL "mps"` therefore holds for one spelling only, and
the apple lane -- which exports `apple` -- has failed twice on that:

* its Objective-C++ sources were left out of the library, so the link stopped
  on `at::native::flagos::mps::CopyStridedToDense(...), referenced from ...`;
* the aten kernel filter compared the accelerator against the directory names,
  so no other vendor's tree was ever excluded and an apple build compiled every
  vendor's `.cc` into one library.

The platform is therefore resolved once (the `_flagos_is_<name>` flags in the
top-level CMakeLists, set from both the accelerator and its vendor), and no
file compares FLAGOS_ACCELERATOR itself except that resolver.

Pure text: no torch, no cmake.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CMAKE_FILES = (
    REPO_ROOT / "CMakeLists.txt",
    REPO_ROOT / "csrc" / "CMakeLists.txt",
    REPO_ROOT / "csrc" / "runtime" / "accelerator" / "CMakeLists.txt",
    REPO_ROOT / "cmake" / "FlagosPlatforms.cmake",
)

#: The one place FLAGOS_ACCELERATOR may be compared with a platform name.
RESOLVER = REPO_ROOT / "CMakeLists.txt"

_FLAG = re.compile(r"_flagos_is_([a-z0-9_]+)")


def _table() -> dict:
    return json.loads(
        (REPO_ROOT / "cmake" / "flagos_platforms.json").read_text(encoding="utf-8")
    )


def _vendors() -> set[str]:
    return {row["vendor"] for row in _table()["accelerators"].values()}


def _code(path: Path) -> str:
    """The file without its comments, so prose about the rule is not a gate."""
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_platform_table_publishes_the_vendor_to_cmake():
    """Every gate below reads it; an unset variable makes a gate never match."""
    text = _code(REPO_ROOT / "cmake" / "FlagosPlatforms.cmake")
    assert re.search(
        r"string\(JSON FLAGOS_PLATFORM_VENDOR GET\s+.*?accelerators "
        r'"\$\{FLAGOS_ACCELERATOR\}" vendor\)',
        text,
        re.S,
    ), "FlagosPlatforms.cmake must publish FLAGOS_PLATFORM_VENDOR from the table"


def test_cmake_resolves_the_platform_from_both_spellings():
    resolver = _code(RESOLVER)
    assert "macro(_flagos_resolve_platforms)" in resolver
    assert 'FLAGOS_ACCELERATOR STREQUAL "${_name}"' in resolver, (
        "the resolver must accept the platform id"
    )
    assert 'FLAGOS_PLATFORM_VENDOR STREQUAL "${_name_vendor}"' in resolver, (
        "the resolver must accept the vendor name"
    )
    # Every table row's id and vendor must be resolvable -- a name missing from
    # the list leaves that platform's gates never matching.
    for row, entry in _table()["accelerators"].items():
        for name in (row, entry["vendor"]):
            if name == "drobotics":
                # Nothing publishes to a drobotics row, so it is not a legal
                # vendor spelling (setup.py rejects it too).
                continue
            assert re.search(rf"\b{re.escape(name)}\b", resolver), (
                f"{name} is not in the resolver's name list"
            )


def test_no_cmake_file_compares_the_accelerator_to_a_platform():
    """The regression: an `mps`-only gate, and a filter that never matched."""
    names = set(_table()["accelerators"]) | _vendors()
    alternation = "|".join(sorted(map(re.escape, names)))
    pattern = re.compile(rf'FLAGOS_ACCELERATOR\s+STREQUAL\s+"?({alternation})"?\b')
    offenders = []
    for path in CMAKE_FILES:
        for number, line in enumerate(_code(path).splitlines(), start=1):
            if pattern.search(line):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
                )
    assert not offenders, (
        "these compare FLAGOS_ACCELERATOR with one spelling, so a build invoked "
        "with the other falls through; use the _flagos_is_<name> flag:\n"
        + "\n".join(offenders)
    )


def test_gates_name_platforms_that_exist_in_the_table():
    resolvable = set(_table()["accelerators"]) | _vendors()
    for path in CMAKE_FILES:
        used = set(_FLAG.findall(_code(path))) - {"is", "kernel", "accel", "platform"}
        unknown = used - resolvable
        assert not unknown, (
            f"{path.relative_to(REPO_ROOT)} gates on {sorted(unknown)}, which is "
            "neither a platform id nor a vendor in the table"
        )


def test_the_kernel_filter_reads_the_tree_the_table_names():
    """A row may keep kernels in a tree its own names do not spell.

    MetaX's kernels are the CUDA ones, while its vendor and its runtime dir are
    both `metax`. The filter keeps the platform's own trees and drops the rest,
    and the alternation it builds must exclude something: inside a quoted regex
    a `;`-joined list matches no name after the first.
    """
    platforms = _code(REPO_ROOT / "cmake" / "FlagosPlatforms.cmake")
    filter_ = _code(REPO_ROOT / "csrc" / "CMakeLists.txt")
    assert "FLAGOS_PLATFORM_ATEN_DIRS" in platforms, (
        "the filter needs the platform's own trees, resolved once"
    )
    assert "IN_LIST FLAGOS_PLATFORM_ATEN_DIRS" in filter_
    assert 'string(REPLACE ";" "|"' in filter_, (
        "the off arm must turn the list into an alternation"
    )
    for row in ("metax", "cuda"):
        assert row in _table()["accelerators"]["metax"].get("aten_backends", []), (
            f"the MetaX row must name {row} as one of its kernel trees"
        )


def test_a_build_without_a_marker_removes_an_inherited_one():
    """The install directory persists between builds, so a stale marker wins.

    torch_fl/lib/ is the source tree's own directory and is not cleaned between
    builds; a marker left there by another platform's build (or by the marker
    tests' copies) names a conf this build cannot route through. A MetaX build
    that inherited an "mps" marker read the empty backends_apple.conf and raised
    "routed to 'none'" on every op it does not list. The other arm of the same
    gate must therefore remove it at install time, when the prefix is known.
    """
    text = _code(REPO_ROOT / "csrc" / "CMakeLists.txt")
    assert "FLAGOS_PLATFORM_WRITES_MARKER" in text
    assert re.search(
        r"if\(FLAGOS_PLATFORM_WRITES_MARKER\).*?else\(\).*?"
        r"install\(CODE .*?file\(REMOVE .*?flagos_platform",
        text,
        re.S,
    ), "the no-marker arm must remove an inherited lib/flagos_platform"
