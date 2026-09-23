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

The second half of the module is the same kind of guard for a different
failure, recorded as issue #391: a manifest enumerates what it runs rather than
sweeping the tree, so an integration test file added at the top level of
``tests/integration/`` reached no CI job at all. Three multi-device contract
files sat that way for their whole existence, and the defect class they cover --
an op that reads or writes across devices -- faults the device or silently
returns its result on the wrong device rather than raising, so nothing went red.
The check below requires every integration test file to be reachable from at
least one manifest, against an explicit list of the ones that are known not to
be, so the next orphan fails instead of hiding.

Nothing here imports a YAML parser, for the reason ``load-platform-tests.yml``
gives for running its own parsing off the vendor image: the accelerator
containers are not required to carry a YAML dependency, and this file runs
inside one of them.
"""

import itertools
import re
import shlex
from pathlib import Path
from typing import NamedTuple

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_DIR = _REPO_ROOT / ".github/workflows"
_CONFIG_DIR = _REPO_ROOT / ".github/configs"
_INTEGRATION_DIR = _REPO_ROOT / "tests/integration"

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


# ---------------------------------------------------------------------------
# Manifest reachability (issue #391)
#
# A manifest enumerates the files it runs; only `tests/integration/ops/` is
# swept as a directory. A test file added at the top level of
# `tests/integration/` therefore runs nowhere until someone names it, and three
# multi-device contract files spent their whole existence that way. The check
# below requires every file under `tests/integration/` to be selected by at
# least one manifest command, against an explicit list of the ones that are
# known not to be.
#
# Reachability is modelled, not measured -- there is no pytest here. A command
# selects a file when its paths cover it and the file carries a mark its `-m`
# expression accepts. Both halves deliberately over-approximate: a false
# "reachable" is a gap left for later, while a false "unreachable" is a red
# build that says so. The failure mode this must never have is silence.
# ---------------------------------------------------------------------------

# `pytest.mark.<name>` in source form: the only mark spelling that can be read
# without importing the file.
_MARK_RE = re.compile(r"pytest\.mark\.([A-Za-z_]\w*)")
# Marks that configure a case rather than select it. `-m parametrize` would
# match nothing on purpose, so seeing one says nothing about whether a sweep's
# `-m` expression picks the file up.
_NON_SELECTING_MARKS = frozenset(
    {
        "parametrize",
        "skip",
        "skipif",
        "xfail",
        "usefixtures",
        "filterwarnings",
        "timeout",
        "tryfirst",
        "trylast",
    }
)
# A marker-expression token: a mark name, a parenthesis, or an operator.
_MARKER_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|[()]")
# pytest options whose value is a separate token. Without this the token after
# `--deselect tests/integration/ops/test_x.py::TestX` reads as a file the
# command runs, which would credit a file the command explicitly drops.
_OPTIONS_WITH_VALUES = frozenset(
    {
        "-m",
        "-k",
        "-c",
        "-p",
        "-n",
        "--deselect",
        "--ignore",
        "--ignore-glob",
        "--rootdir",
        "--junitxml",
    }
)


class _Selection(NamedTuple):
    """One manifest command, as what it hands to pytest."""

    paths: tuple[str, ...]
    marker: str | None
    ignored: tuple[str, ...]


def _pytest_arguments(command: str) -> list[str]:
    """The arguments after ``pytest`` in a manifest command, or none.

    The literal ``python -m pytest`` shape is required, so an environment probe
    that merely mentions pytest in a heredoc is not read as an invocation.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        # An unbalanced quote somewhere in a heredoc body, so not a pytest
        # invocation -- and not something to fail a reachability check over.
        return []
    for index, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1] != "pytest":
            continue
        if (
            index >= 2
            and tokens[index - 1] == "-m"
            and tokens[index - 2].endswith("python")
        ):
            return tokens[index + 1 :]
    return []


def _integration_target(token: str) -> str | None:
    """A pytest argument as a repository-relative path under ``tests/integration``.

    A node id targets the file before its first ``::``, so that is what is
    resolved. Anything landing outside ``tests/integration`` -- ``tests/unit/``,
    ``$MODEL_PATH`` -- is not this check's business and returns None.
    """
    candidate = token.split("::", 1)[0]
    if not candidate or candidate.startswith("-"):
        return None
    parts = Path(candidate).parts
    root = _INTEGRATION_DIR.relative_to(_REPO_ROOT).parts
    if parts[: len(root)] != root:
        return None
    return "/".join(parts)


def _selection_of(command: str) -> _Selection | None:
    """A manifest command split into the paths and filter it runs, or None."""
    arguments = _pytest_arguments(command)
    if not arguments:
        return None
    paths: list[str] = []
    ignored: list[str] = []
    marker: str | None = None
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token in _OPTIONS_WITH_VALUES:
            # Every one of these takes the next token, but only `-m` says
            # anything about which files the command runs.
            if token == "-m" and index + 1 < len(arguments):
                marker = arguments[index + 1]
            index += 2
            continue
        if token.startswith("--ignore="):
            ignored.append(token.split("=", 1)[1])
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        target = _integration_target(token)
        if target is not None:
            paths.append(target)
        index += 1
    return _Selection(tuple(paths), marker, tuple(ignored))


def _manifest_selections(platform: str) -> list[_Selection]:
    """What every pytest invocation in one platform's manifest runs."""
    selections = []
    for command in _manifest_commands(platform):
        selection = _selection_of(command)
        if selection is not None:
            selections.append(selection)
    return selections


def _manifest_platforms() -> list[str]:
    """Every platform manifest, by the name its file carries."""
    return sorted(path.stem for path in _CONFIG_DIR.glob("*.yml"))


def _marks_of(test_file: Path) -> set[str]:
    """The selecting marks a test file carries, read textually off the source.

    Over the whole file rather than per case: the question is whether *any*
    case in the file is collected, and answering it exactly would mean
    importing the file, which needs the device this check must not require.
    """
    return {
        name
        for name in _MARK_RE.findall(test_file.read_text(encoding="utf-8"))
        if name not in _NON_SELECTING_MARKS
    }


def _evaluate(expression: str, marks: set[str]) -> bool:
    """What pytest's ``-m`` would answer for a case carrying exactly ``marks``.

    Only the grammar the manifests use: mark names, ``and``, ``or``, ``not``,
    and parentheses. A name is true when the case carries it.
    """
    tokens = _MARKER_TOKEN_RE.findall(expression)
    position = 0

    def atom() -> bool:
        nonlocal position
        assert position < len(tokens), f"unbalanced marker expression: {expression!r}"
        token = tokens[position]
        position += 1
        if token == "(":
            value = disjunction()
            assert position < len(tokens) and tokens[position] == ")", (
                f"unbalanced marker expression: {expression!r}"
            )
            position += 1
            return value
        if token == "not":
            return not atom()
        assert token not in ("and", "or"), (
            f"malformed marker expression: {expression!r}"
        )
        return token in marks

    def conjunction() -> bool:
        nonlocal position
        value = atom()
        while position < len(tokens) and tokens[position] == "and":
            position += 1
            # Evaluated before the `and`, not after: Python would short-circuit
            # past the call and leave the cursor on the operand it never read.
            following = atom()
            value = value and following
        return value

    def disjunction() -> bool:
        nonlocal position
        value = conjunction()
        while position < len(tokens) and tokens[position] == "or":
            position += 1
            following = conjunction()
            value = value or following
        return value

    result = disjunction()
    assert position == len(tokens), (
        f"trailing tokens in marker expression: {expression!r}"
    )
    return result


def _selects(expression: str, marks: set[str]) -> bool:
    """Whether any case in a file carrying ``marks`` survives ``expression``.

    Asked of every subset of the file's marks rather than of their union, and
    that is not a detail: the union is wrong in the direction that matters.
    ``ops/test_acos_dispatch.py`` carries ``anyplatform`` on one class and
    ``flaggems_python`` on another, so the union makes ``(anyplatform or
    main_ops) and not flaggems_python`` false and the file reads as an orphan
    -- while pytest collects 8 of its 11 cases under exactly that filter.
    Every subset is a superset of the marks one case can carry, so this errs
    towards "reachable": that can leave a gap, which is the status quo, where
    the reversed error is a red build over a file that does run.
    """
    names = sorted(marks)
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            if _evaluate(expression, set(subset)):
                return True
    return False


def _names_target(relative: str, target: str) -> bool:
    """Whether a pytest target covers a file, by name or as a directory above it."""
    return relative == target or relative.startswith(target.rstrip("/") + "/")


def _test_file_paths() -> set[str]:
    """Every integration test file, relative to the repository root."""
    return {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _INTEGRATION_DIR.rglob("test_*.py")
    }


def _reachable_test_files() -> set[str]:
    """Every integration test file at least one manifest command would run."""
    files = {
        relative: _marks_of(_REPO_ROOT / relative)
        for relative in sorted(_test_file_paths())
    }
    reachable = set()
    for platform in _manifest_platforms():
        for selection in _manifest_selections(platform):
            for relative, marks in files.items():
                if any(_names_target(relative, skip) for skip in selection.ignored):
                    continue
                if not any(_names_target(relative, path) for path in selection.paths):
                    continue
                if selection.marker is not None and not _selects(
                    selection.marker, marks
                ):
                    continue
                reachable.add(relative)
    return reachable


# Files no manifest selects, each with the reason it is left that way. This is
# the ratchet: the check below is green while it matches reality exactly, and
# goes red the moment either half drifts -- a new orphan it has not been told
# about, or an entry here that has since been wired in.
_KNOWN_UNREACHABLE: dict[str, str] = {
    "tests/integration/test_apex_compat.py": (
        "needs NVIDIA apex and its amp_C extension; no platform image installs them"
    ),
    "tests/integration/test_clone_dispatch_case.py": (
        "runs clean on a 910 (3 passed) and is portable, so it should be wired "
        "into the manifests in a follow-up"
    ),
    "tests/integration/test_dtype_coverage.py": (
        "1 of 31 fails on a 910: a bool neg routed to FlagGems fails BiShengIR "
        "compilation; wire it in once that is fixed"
    ),
    "tests/integration/test_fallback_trace.py": (
        "needs --model and a mounted Qwen3 checkpoint"
    ),
    "tests/integration/test_fallback_trace_train.py": (
        "needs --model and a mounted Qwen3 checkpoint"
    ),
    "tests/integration/test_nonzero_device_dtype_cast.py": (
        "passes on a 910; should be wired into the manifests in a follow-up"
    ),
    "tests/integration/test_ops.py": (
        "2 of 58 fail on a 910 on an rtol/atol of 1e-4 for a float32 mm with "
        "K=128, which is tighter than the accumulation error; the file is "
        "otherwise superseded by the marker-selected ops/ suites"
    ),
    "tests/integration/test_profiler_qwen3_infer.py": (
        "needs --model and a mounted Qwen3 checkpoint"
    ),
    "tests/integration/ops/test_dcu_flaggems_sdpa.py": (
        "carries only the dcu mark, and both DCU sweeps require main_ops or flaggems"
    ),
    "tests/integration/ops/test_flaggems_cpp_dispatch.py": (
        "carries only flaggems_cpp, and every set_env_*.sh builds FlagGems "
        "without its C++ path (FLAGOS_BUILD_FLAGGEMS_CPP=0)"
    ),
    "tests/integration/ops/test_full_cuda_coverage.py": (
        "carries only the cuda mark, and both CUDA sweeps require main_ops or flaggems"
    ),
    "tests/integration/ops/test_gcu_sdpa_mask.py": (
        "carries only the gcu mark, and both GCU sweeps require anyplatform, "
        "main_ops or flaggems"
    ),
    "tests/integration/ops/test_musa_flaggems.py": (
        "documented in musa.yml as held back until the image ships the vendor "
        "triton stack; added now it would record a pass that measured nothing"
    ),
    "tests/integration/ops/test_tileops_generated.py": (
        "auto-generated with no marker at all, so no -m filter can select it; "
        "the fix is in scripts/codegen/codegen_tileops.py::render_test"
    ),
}


def test_the_reachability_reader_sees_what_the_manifests_actually_ask_for():
    """A reader that quietly found nothing would make the check below vacuous.

    Every manifest runs at least one sweep by marker, so a reading that missed
    them -- or one that stopped seeing ``-m`` -- has to fail here rather than
    report a clean tree.
    """
    selections = _manifest_selections("ascend")
    assert selections, "the reader found no pytest invocation in the Ascend manifest"
    assert any(selection.paths and selection.marker for selection in selections), (
        "the reader saw no marker-filtered sweep, so nothing below is being tested"
    )
    assert _reachable_test_files(), "the reader found no reachable file at all"


def test_no_integration_test_file_is_orphaned_from_every_manifest():
    """Every test file under tests/integration/ has to be run by some job.

    A manifest lists what it runs, so a file reaches CI only by being named --
    and the cases that go missing this way are the quiet ones. The three
    multi-device files behind issue #391 assert that an op reads and writes on
    the device it was handed; a violation faults the device or returns the
    result on the wrong one, so the files existed for months without a red
    build to say they never ran.
    """
    undeclared = sorted(
        path
        for path in _test_file_paths() - _reachable_test_files()
        if path not in _KNOWN_UNREACHABLE
    )
    assert not undeclared, (
        f"{undeclared} are selected by no manifest command, so no CI job runs "
        "them; wire each one into the platforms that have the hardware, or add "
        "it to _KNOWN_UNREACHABLE with the reason it cannot be"
    )


def test_the_known_unreachable_list_has_not_gone_stale():
    """The exclusions have to be re-earned, or they outlive their reasons.

    Both directions rot the same way: an entry whose file is now reachable
    documents a gap that no longer exists, and an entry naming a file that is
    gone is a stale path pretending to be a decision.
    """
    reachable = _reachable_test_files()
    wired_in = sorted(path for path in _KNOWN_UNREACHABLE if path in reachable)
    assert not wired_in, (
        f"{wired_in} are reachable now, so their entries in _KNOWN_UNREACHABLE "
        "describe a gap that has closed; drop them"
    )
    vanished = sorted(
        path for path in _KNOWN_UNREACHABLE if not (_REPO_ROOT / path).is_file()
    )
    assert not vanished, (
        f"{vanished} are listed in _KNOWN_UNREACHABLE but do not exist any more"
    )
