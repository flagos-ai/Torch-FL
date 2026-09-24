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

"""The device-tracer selection must cover every accelerator, and pick one.

``csrc/CMakeLists.txt`` selects exactly one ``csrc/profiler/*_device_tracer.cc``
per build: every one of them defines the same ``MakeDeviceTracer()`` factory, so
a build that keeps two fails to link, and a build that keeps the wrong one
silently loses device profiling. A CUDA-boxing accelerator routed to
``unavailable_device_tracer.cc`` is the worst case -- it links, it builds, and
``available()`` returning ``false`` makes the kineto adaptor skip registration,
so the trace carries no device timeline at all and nothing in the build log says
why. That is issue #411: ``ppu`` fell through to the stub while the comment
above the selection claimed it used the CUPTI tracer.

This is a pure text/parse check over ``csrc/CMakeLists.txt`` and
``cmake/flagos_platforms.json``. It needs no GPU, no build, and no ``torch_fl``,
so it runs in milliseconds and is wired into the platform-agnostic CI job.

Run: pytest tests/unit/test_device_tracer_selection.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE_LISTS = REPO_ROOT / "csrc" / "CMakeLists.txt"
PLATFORM_TABLE = REPO_ROOT / "cmake" / "flagos_platforms.json"
PROFILER_DIR = REPO_ROOT / "csrc" / "profiler"

# Anchor of the selection block in csrc/CMakeLists.txt. Kept as a marker rather
# than a line number so the check survives edits above it.
_BLOCK_MARKER = "# Select exactly one vendor-specific DeviceTracer implementation"

# Every tracer source that defines MakeDeviceTracer(), and the accelerator each
# is intended for. A tracer added to csrc/profiler/ without a row here fails
# test_every_tracer_source_is_reachable_from_the_ladder.
EXPECTED_TRACER = {
    "cuda": "cupti_device_tracer",
    "ppu": "cupti_device_tracer",
    "metax": "cupti_device_tracer",
    "dcu": "roctracer_device_tracer",
    "ascend": "cann_device_tracer",
    "gcu": "gcu_topspti_device_tracer",
    "musa": "musa_mupti_device_tracer",
    "tsingmicro": "unavailable_device_tracer",
    "bpu": "unavailable_device_tracer",
}


def _tracer_sources() -> set[str]:
    """Every ``*_device_tracer.cc`` basename present under ``csrc/profiler``."""
    return {path.stem for path in PROFILER_DIR.glob("*_device_tracer.cc")}


def _excluded_sources(body: str) -> set[str]:
    """Tracer basenames an arm excludes, read from its ``list(FILTER ...)`` lines.

    The CMake REGEX literals escape their dots (``\\.cc$``), so the backslashes
    are dropped before matching rather than doubled in the pattern.
    """
    return set(re.findall(r"profiler/(\w+)\.cc", body.replace("\\\\", "")))


def _selection_arms() -> list[tuple[str, str]]:
    """Return ``[(condition, body), ...]`` for the tracer-selection if/elseif/else.

    The arms are unindented and the bodies indented, which is what lets a line
    starting with ``if(``/``elseif(``/``else()`` be read as an arm boundary. A
    condition may span several lines, so it is accumulated until its parentheses
    balance.
    """
    text = CMAKE_LISTS.read_text(encoding="utf-8")
    block = text[text.index(_BLOCK_MARKER) :]
    block = block[: block.index("\nendif()\n")]

    arms: list[tuple[str, str]] = []
    condition: list[str] | None = None
    body: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if line == stripped and stripped.startswith(("if(", "elseif(", "else()")):
            if condition is not None:
                arms.append((" ".join(condition), "\n".join(body)))
            condition, body = [stripped], []
        elif condition is not None:
            if sum(part.count("(") for part in condition) == sum(
                part.count(")") for part in condition
            ):
                body.append(line)
            else:
                condition.append(stripped)
    if condition is not None:
        arms.append((" ".join(condition), "\n".join(body)))

    assert len(arms) >= 2, f"parsed {len(arms)} arm(s) out of the tracer selection"
    return arms


def _condition_terms(condition: str) -> list[str]:
    """The ``FLAGOS_ACCELERATOR`` values an ``if``/``elseif`` condition selects.

    The arms are a disjunction of ``FLAGOS_ACCELERATOR STREQUAL "<value>"``
    terms. Anything else would make this evaluator quietly wrong rather than
    merely incomplete, so the shape is asserted instead of guessed at. Splitting
    on a spaced ``OR`` matters: ``FLAGOS_ACCELERATOR`` itself contains those
    letters.
    """
    body = re.fullmatch(r"(?:if|elseif)\((.*)\)", condition, re.DOTALL)
    assert body, f"not a well-formed tracer-selection arm: {condition!r}"
    values = []
    for term in re.split(r"\s+OR\s+", body.group(1).strip()):
        match = re.fullmatch(r'FLAGOS_ACCELERATOR STREQUAL "([^"]+)"', term)
        assert match, f"unexpected tracer-selection condition term: {term!r}"
        values.append(match.group(1))
    return values


def _condition_matches(condition: str, accelerator: str) -> bool:
    return accelerator in _condition_terms(condition)


def _arm_for(accelerator: str) -> tuple[str, str]:
    """The arm of the selection chain ``accelerator`` resolves to."""
    for condition, body in _selection_arms():
        if condition == "else()" or _condition_matches(condition, accelerator):
            return condition, body
    raise AssertionError(f"no arm of the tracer selection matched {accelerator!r}")


def _compiled_tracers(accelerator: str) -> set[str]:
    """Tracer sources a build of ``accelerator`` keeps, per the CMake exclusions."""
    _, body = _arm_for(accelerator)
    return _tracer_sources() - _excluded_sources(body)


def _accelerators() -> list[str]:
    table = json.loads(PLATFORM_TABLE.read_text(encoding="utf-8"))
    return sorted(table["accelerators"])


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("accelerator", _accelerators())
def test_ladder_selects_the_expected_tracer(accelerator):
    """Each accelerator compiles the tracer it is supposed to.

    ``ppu`` is the regression under test: it is a CUDA-ABI boxing backend whose
    device activities come from the CUPTI tracer, and issue #411 was a build that
    gave it the unavailable stub instead.
    """
    assert _compiled_tracers(accelerator) == {EXPECTED_TRACER[accelerator]}, (
        f"{accelerator} resolves to {sorted(_compiled_tracers(accelerator))}, "
        f"expected exactly ['{EXPECTED_TRACER[accelerator]}']"
    )


@pytest.mark.parametrize("accelerator", _accelerators())
def test_exactly_one_tracer_is_compiled(accelerator):
    """Two tracer sources in one build is a duplicate-symbol link error.

    A simple statement, but it is the failure mode a hand-edited ladder produces
    when an exclusion is added to one arm and not to its siblings.
    """
    compiled = _compiled_tracers(accelerator)
    assert len(compiled) == 1, f"{accelerator} compiles {sorted(compiled)}"


def test_every_accelerator_in_the_table_has_an_expected_tracer():
    """A new platform in the table must be given a tracer deliberately.

    Without this, adding an accelerator to ``cmake/flagos_platforms.json`` would
    fall through to the ``else()`` arm -- which is exactly how ``ppu`` silently
    became an unavailable-tracer platform -- and the parametrized tests above
    would simply not cover it.
    """
    missing = sorted(set(_accelerators()) - set(EXPECTED_TRACER))
    assert not missing, (
        f"accelerators {missing} have no expected tracer; decide which tracer "
        "each should compile and add it to EXPECTED_TRACER"
    )


# ---------------------------------------------------------------------------
# Ladder shape
# ---------------------------------------------------------------------------


def test_last_arm_is_an_exhaustive_else():
    """The chain must end in ``else()``: an unmatched accelerator must still compile.

    Reaching the end of an if/elseif chain without a match would leave every
    tracer source in the build, and the duplicate ``MakeDeviceTracer()`` would be
    a link error. The ``else()`` arm is what makes an unknown or misspelled
    ``FLAGOS_ACCELERATOR`` fail at runtime (``available() == false``) instead of
    at link time.
    """
    condition, _ = _selection_arms()[-1]
    assert condition == "else()", f"the selection chain ends with {condition!r}"


def test_every_tracer_source_is_reachable_from_the_ladder():
    """Every tracer source on disk is excluded by every arm that should not keep it.

    A new ``*_device_tracer.cc`` that no arm excludes is compiled into every
    build, so every build fails to link. Naming it in the arms is part of adding
    it, and this check is what asks for that.
    """
    named: set[str] = set()
    for _, body in _selection_arms():
        named |= _excluded_sources(body)

    on_disk = _tracer_sources()
    assert named == on_disk, (
        f"tracer sources in no arm: {sorted(on_disk - named)}; "
        f"arm entries with no source on disk: {sorted(named - on_disk)}"
    )


def test_the_cupti_arm_names_its_platforms_explicitly():
    """The CUPTI arm lists its accelerators rather than matching a prefix.

    ``metax`` and ``ppu`` share no prefix, but a future ``metax2``-style name
    would be caught by a ``MATCHES`` on a stripped prefix and not by a list, so
    the explicit form is asserted. It is also the form that makes the mapping
    above readable.
    """
    condition, _ = _arm_for("cuda")
    assert condition.startswith('if(FLAGOS_ACCELERATOR STREQUAL "cuda"'), condition
    assert "MATCHES" not in condition and "STRLESS" not in condition
    assert sorted(re.findall(r'STREQUAL "([^"]+)"', condition)) == [
        "cuda",
        "metax",
        "ppu",
    ]
