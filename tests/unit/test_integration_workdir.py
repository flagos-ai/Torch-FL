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

"""Guards tying a platform's manifest to the workspace its workflow prepares.

The integration jobs do not run the manifest in the checkout. They build and
install the wheel, copy ``tests`` and ``pyproject.toml`` into a scratch
directory, and hand that directory to ``run_integration_tests.py`` as the cwd
for every configured command. A manifest entry that names a repository script
therefore has to execute somewhere that script exists -- and when it did not,
the group died on a bare ``can't open file`` after the whole provisioning and
build had already been paid for.

That is not hypothetical: ``Run unit tests (per file)`` was added to the DCU
manifest naming ``.github/scripts/run_unit_tests.py`` while the DCU workflow
copied only ``tests`` and ``pyproject.toml``. Job 104230731072 spent its
sixty-minute budget installing FlagTree and FlagGems, built and installed the
wheel, then reported:

    python: can't open file '<root>/.github/scripts/run_unit_tests.py':
    [Errno 2] No such file or directory
    Integration test 'Run unit tests (per file)' failed with exit code 2

with ten of the eleven groups never reached.

Copying the script in was the wrong answer for that entry, and the next run
showed why: ``tests/unit`` is a source-tree suite, and seven of its files read
``scripts/``, ``csrc/``, ``.github/workflows/`` and the rest of ``torch_fl/``
relative to the repository root, so they failed on ``FileNotFoundError``
against the scratch directory (job 104244996282) no matter what was copied
beside them. The fix is for that entry to change back to the checkout itself.
So a command that names a repository script now has two acceptable ways to
reach it -- be copied into the prepared directory, or ``cd`` to the checkout --
and this module holds each command to whichever one it chose.

These checks are static on purpose: they must run anywhere, without a device or
a network.

Nothing here imports a YAML parser, for the reason ``load-platform-tests.yml``
gives for running its own parsing off the vendor image: the accelerator
containers are not required to carry a YAML dependency, and this file runs
inside one of them.
"""

import re
import shlex
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_DIR = _REPO_ROOT / ".github/workflows"
_CONFIG_DIR = _REPO_ROOT / ".github/configs"

_PREPARE_STEP = "Prepare wheel-only test workspace"
# `platform: dcu`, the input the workflow passes to load-platform-tests.yml.
_PLATFORM_RE = re.compile(r"^\s*platform:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)
# A column-0 YAML key, used to delimit a top-level block.
_TOP_LEVEL_KEY_RE = re.compile(r"^[A-Za-z_][\w-]*:")
# A folded or literal scalar: `command: >-`, `command: |`.
_SCALAR_KEY_RE = re.compile(r"\s*command:\s*[>|][-+]?\s*")
# A repository script named by a manifest command.
_SCRIPT_RE = re.compile(r"\.github/scripts/[\w./-]+")
# A command that leaves the prepared workspace for the checkout.
_CHECKOUT_CD_RE = re.compile(r"""\bcd\s+["']?\$GITHUB_WORKSPACE["']?""")


def _platform_of(workflow: Path) -> str | None:
    """The manifest a workflow runs, or None if it declares none.

    The declared input has to agree with the filename, because together they
    are what connects a workflow to the config this module checks it against.
    """
    declared = _PLATFORM_RE.search(workflow.read_text(encoding="utf-8"))
    if declared is None:
        return None
    expected = workflow.stem.removeprefix("integration-test-")
    assert declared.group(1) == expected, (
        f"{workflow.name} declares platform '{declared.group(1)}' but is named "
        f"for '{expected}'"
    )
    return expected


def _indent(line: str) -> int:
    """Leading spaces of a line."""
    return len(line) - len(line.lstrip())


def _block_after(lines: list[str], start: int, indent: int) -> list[str]:
    """Lines after ``start`` that are indented deeper than ``indent``."""
    block = []
    for line in lines[start + 1 :]:
        if line.strip() and _indent(line) <= indent:
            break
        block.append(line)
    return block


def _run_script_of_step(text: str, step_name: str) -> str | None:
    """The ``run:`` script of the step named ``step_name``, dedented.

    A targeted read rather than a parse: find the step, take the lines nested
    under it, and inside those take the body of the ``run:`` scalar.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != f"- name: {step_name}":
            continue
        step = _block_after(lines, index, _indent(line))
        for offset, entry in enumerate(step):
            if not re.fullmatch(r"\s*run:\s*\|[-+]?\s*", entry):
                continue
            body = _block_after(step, offset, _indent(entry))
            margins = [_indent(row) for row in body if row.strip()]
            margin = min(margins) if margins else 0
            return "\n".join(row[margin:] for row in body)
    return None


def _prepare_steps() -> list[tuple[str, str, str]]:
    """Every prepare step: ``(workflow name, platform, run script)``."""
    found = []
    for workflow in sorted(_WORKFLOW_DIR.glob("integration-test-*.yml")):
        platform = _platform_of(workflow)
        if platform is None:
            continue
        script = _run_script_of_step(
            workflow.read_text(encoding="utf-8"), _PREPARE_STEP
        )
        if script is not None:
            found.append((workflow.name, platform, script))
    return found


def _logical_lines(script: str) -> list[str]:
    """Command lines, backslash continuations joined and comments dropped."""
    joined = []
    pending = ""
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1].strip() + " "
            continue
        joined.append(pending + stripped)
        pending = ""
    if pending:
        joined.append(pending)
    return joined


def _copied_paths(run_script: str) -> list[str]:
    """The sources of every ``cp`` in a prepare step, in the order written."""
    copied = []
    for line in _logical_lines(run_script):
        tokens = shlex.split(line)
        if len(tokens) < 3 or tokens[0] != "cp":
            continue
        # `cp -a <src>... <dst>`: everything but the flags and the destination.
        copied.extend(token for token in tokens[1:-1] if not token.startswith("-"))
    return copied


def _manifest_commands(platform: str) -> list[str]:
    """The command of every entry in one platform's ``integration_tests`` block.

    Only that block is read. The neighbouring top-level ``setup_script`` key
    names a script too, but that one is not executed from this workspace, so
    requiring it to be copied would be wrong.
    """
    config = _CONFIG_DIR / f"{platform}.yml"
    if not config.is_file():
        return []
    lines = config.read_text(encoding="utf-8").splitlines()
    commands = []
    in_tests = False
    for index, line in enumerate(lines):
        if _TOP_LEVEL_KEY_RE.match(line):
            in_tests = line.startswith("integration_tests:")
            continue
        if not in_tests or not _SCALAR_KEY_RE.fullmatch(line):
            continue
        body = _block_after(lines, index, _indent(line))
        commands.append(" ".join(row.strip() for row in body if row.strip()))
    return commands


def _script_references(platform: str) -> list[tuple[str, str]]:
    """``(command, repository script)`` for every script a manifest names."""
    return [
        (command, reference)
        for command in _manifest_commands(platform)
        for reference in _SCRIPT_RE.findall(command)
    ]


def _runs_in_the_checkout(command: str) -> bool:
    """Whether a command leaves the prepared workspace before running anything.

    A command that does is read against the repository, not against the
    prepared directory, because that is the tree its cwd resolves to.
    """
    return _CHECKOUT_CD_RE.search(command) is not None


def _is_covered(reference: str, copied: list[str]) -> bool:
    """Whether a copied source puts ``reference`` in the workspace.

    A copied directory covers everything below it, so the check is on path
    components rather than on an exact string: ``.github/scripts`` covers
    ``.github/scripts/run_unit_tests.py``.
    """
    return any(
        reference == source or reference.startswith(source.rstrip("/") + "/")
        for source in copied
    )


def test_there_is_at_least_one_prepare_step_to_check():
    """A silent read failure must not turn the checks below into no-ops."""
    assert _prepare_steps(), f"no '{_PREPARE_STEP}' step found under {_WORKFLOW_DIR}"


def test_dcu_manifest_still_runs_a_repository_script():
    """The case the checks below exist for, kept from going vacuous.

    The DCU manifest is the one place a group runs a repository script, and
    that group is also the one that has to leave the prepared workspace to
    reach it: ``tests/unit`` reads ``scripts/``, ``csrc/`` and
    ``.github/workflows/`` relative to the repository root. If either half of
    that stops being true, the guards have nothing to hold, and this says so
    rather than letting them pass by accident.
    """
    references = _script_references("dcu")
    assert references, "the reader found no repository script in the DCU manifest"
    assert all(_runs_in_the_checkout(command) for command, _ in references), (
        "the DCU manifest names a repository script from a command that stays in "
        "the prepared workspace, so the workspace would have to contain it"
    )


def test_the_manifest_reader_sees_only_the_integration_tests_block():
    """The block delimiter holds, so the check is not off by a section.

    ``setup_script`` sits at the same level as ``integration_tests`` and names
    a script the workspace is not expected to contain; if the reader leaked
    into it, the DCU case below would judge a command nothing runs.
    """
    commands = _manifest_commands("dcu")
    assert commands, "the reader found no commands at all, which cannot be right"
    assert not any("set_env_dcu.sh" in command for command in commands)


@pytest.mark.parametrize(
    "workflow,platform",
    [(name, platform) for name, platform, _ in _prepare_steps()],
)
def test_every_script_a_manifest_runs_exists_where_its_command_runs(workflow, platform):
    """A manifest command must find every repository script it names.

    The prepared directory is the cwd for the command unless the command
    changes directory first, so a script left in the checkout is not reachable
    from it -- and a script the command names after changing back to the
    checkout is only reachable if it is still there.
    """
    script = next(
        run
        for name, manifest_platform, run in _prepare_steps()
        if (name, manifest_platform) == (workflow, platform)
    )
    copied = _copied_paths(script)
    missing = []
    for command, reference in _script_references(platform):
        if _runs_in_the_checkout(command):
            if not (_REPO_ROOT / reference).is_file():
                missing.append(f"{reference} (the command cds to the checkout)")
        elif not _is_covered(reference, copied):
            missing.append(reference)
    assert not missing, (
        f"{workflow} prepares {copied} for the '{platform}' manifest, which runs "
        f"{missing}; each command starts in that workspace, so a script it names "
        "has to be copied into it or the command has to cd to the checkout first"
    )


def test_the_prepare_step_still_copies_the_tests_it_was_written_for():
    """The original purpose of the step, kept so a rewrite cannot drop it."""
    for workflow, _platform, script in _prepare_steps():
        assert "tests" in _copied_paths(script), workflow
