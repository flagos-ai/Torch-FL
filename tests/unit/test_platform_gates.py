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

"""Hold the recorded per-platform gates to the manifest they describe.

Every manifest entry runs on every platform, so the only thing left that differs
is a pytest-level skip, and `tests/integration/platform_gates.py` is where those
skips are recorded -- the loader prints them for each platform before a runner
is queued, and the test files enforce them. A record nobody checks is how the
eight-way divergence grew back after the manifest removed it, so this module is
the check: a gate naming a step the manifest does not define, or a platform that
has no config, fails here instead of printing a line that reads as coverage.

The file is imported the way the composer imports it, by path, so a change that
makes it unimportable in the loader job (a torch import, a syntax change) fails
here first.

The same table is written out for readers in
`docs/reference/platform-capability-matrix.md`, and the last test here holds the
two to each other: a skip that is in one and not the other is a platform whose
documented coverage is not the coverage it runs.
"""

import re
from pathlib import Path

import pytest

from manifest_composer import composer, manifest, platform_names


_REPO = Path(__file__).resolve().parents[2]
_GATES_PATH = _REPO / "tests" / "integration" / "platform_gates.py"
_MATRIX_DOC = _REPO / "docs" / "reference" / "platform-capability-matrix.md"
# The eight platforms this layer has to keep in step. Written out rather than
# taken from the directory: a config that quietly disappeared would otherwise
# take its platform out of every check below and leave the suite green.
_PLATFORMS = ("ascend", "cuda", "dcu", "gcu", "kunlunxin", "metax", "musa", "ppu")
# A row of the matrix's integration-coverage whitelist. Deliberately spelled with
# the platform unquoted and the platform cell the only one on the line: the
# matrix's own tables start with "| `", and this must not be read as one of them.
# The second cell is a comma-separated list of step ids and nothing else, which
# is what keeps the table's header out of the reading.
_WHITELIST_ROW_RE = re.compile(
    r"^\| ([a-z][a-z0-9_]*) \| ([a-z_][a-z0-9_]*(?:, [a-z_][a-z0-9_]*)*) \|$", re.M
)


def _platform_gates():
    """The gate table as the composer loads it -- by path, stdlib only."""
    return composer._gates_module()


def _step_ids() -> list[str]:
    """Every id the manifest defines, in order."""
    return composer.validate_manifest(manifest())[1]


def test_the_gate_table_exists_and_imports_without_torch():
    """The loader job has no vendor wheel, so the table may not need one."""
    assert _GATES_PATH.is_file(), f"{_GATES_PATH} does not exist"
    module = _platform_gates()
    assert module is not None, "the composer could not load the gate table"
    assert module.all_gates(), "the gate table records no gates at all"


def test_every_gate_names_a_step_the_manifest_defines():
    """A gate on a step nobody runs is a line that reads as coverage."""
    known = set(_step_ids())
    unknown = sorted({gate.step for gate in _platform_gates().all_gates()} - known)
    assert not unknown, (
        f"{unknown} are gated in tests/integration/platform_gates.py but are not "
        f"manifest steps; the manifest's ids are {', '.join(known)}"
    )


def test_every_gate_names_platforms_that_have_configs():
    """A platform without a config is never scheduled, so it has no run."""
    known = set(platform_names())
    for gate in _platform_gates().all_gates():
        unknown = sorted(set(gate.platforms) - known)
        assert not unknown, (gate.step, unknown)


def test_no_gate_applies_to_every_platform_or_to_none():
    """Both extremes say the step does not belong in the shared manifest.

    A gate on all eight is a step no platform exercises, and one on no platform
    is not a gate; either should be a manifest edit or no entry at all, and both
    are caught here rather than by a reader noticing.
    """
    for gate in _platform_gates().all_gates():
        assert 0 < len(set(gate.platforms)) < len(_PLATFORMS), (
            f"{gate.step} applies to {len(set(gate.platforms))} platforms"
        )
        assert len(set(gate.platforms)) == len(gate.platforms), gate.step


def test_every_gate_states_a_reason():
    """The reason is the removal path; an empty one is a skip nobody can lift."""
    for gate in _platform_gates().all_gates():
        assert gate.reason.strip(), gate.step
        assert gate.target.strip(), gate.step


def test_every_gate_names_a_target_file_that_exists():
    """The target is a real file under tests/integration, not a stale path."""
    for gate in _platform_gates().all_gates():
        relative = re.split(r"::|,", gate.target)[0].strip()
        assert relative.endswith(".py"), (gate.step, gate.target)
        assert (_REPO / "tests" / "integration" / relative).is_file(), (
            f"{gate.step} points at {relative}, which does not exist"
        )


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_gates_for_selects_the_right_platform(platform):
    """`gates_for` is what the loader prints, so its selection is the answer."""
    module = _platform_gates()
    selected = module.gates_for(platform)
    expected = [gate for gate in module.all_gates() if platform in gate.platforms]
    assert list(selected) == expected
    # Every platform this layer runs has at least one recorded difference, and
    # none of them is full coverage with a gate table that forgot it.
    assert selected, (
        f"{platform} records no gate at all; if it really covers every step, "
        "the table has been wrong somewhere instead"
    )


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_the_loader_prints_the_gates_for_each_platform(platform):
    """The print the loader makes has to come from this table."""
    report = composer.gate_report(platform)
    assert report, f"the loader printed no gate for {platform}"
    for line in report:
        assert line.startswith("  "), line
        assert any(gate.step in line for gate in _platform_gates().all_gates()), line


def test_the_gated_steps_are_exactly_the_steps_with_gates():
    module = _platform_gates()
    assert module.GATED_STEPS == frozenset(gate.step for gate in module.all_gates())


def _documented_whitelist() -> dict[str, list[str]]:
    """The per-platform skip list the capability matrix states."""
    text = _MATRIX_DOC.read_text(encoding="utf-8")
    _, found, rest = text.partition("## Integration coverage")
    assert found, (
        "docs/reference/platform-capability-matrix.md has no Integration "
        "coverage section, so nothing records what a platform still skips"
    )
    section = rest.split("\n## ", 1)[0]
    return {
        match.group(1): [
            step.strip() for step in match.group(2).split(",") if step.strip()
        ]
        for match in _WHITELIST_ROW_RE.finditer(section)
    }


def test_the_matrix_whitelist_covers_every_platform_once():
    """A platform missing from the table would read as full coverage."""
    assert sorted(_documented_whitelist()) == sorted(_PLATFORMS)


@pytest.mark.parametrize("platform", _PLATFORMS)
def test_the_capability_matrix_states_the_same_gates(platform):
    """One answer in two places is two answers unless something compares them."""
    documented = _documented_whitelist()[platform]
    expected = [gate.step for gate in _platform_gates().gates_for(platform)]
    assert documented == expected, (
        f"docs/reference/platform-capability-matrix.md lists {documented} for "
        f"{platform} but tests/integration/platform_gates.py says {expected}"
    )
