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

The commands are read through ``manifest_composer``, so this file and the
loader answer "what does this platform run?" the same way. That reader parses
YAML, which costs nothing here: this file also runs inside the DCU container, and
that environment already carries PyYAML for ``test_ci_pip_cache.py`` and for
FlagGems' own dependencies (``PyYAML==6.0.1`` in ``set_env_hygon.sh``).
"""

import ast
import itertools
import re
import shlex
from pathlib import Path
from typing import NamedTuple

import pytest
from manifest_composer import boxing_platforms, composed_steps, platform_names


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_DIR = _REPO_ROOT / ".github/workflows"
_INTEGRATION_DIR = _REPO_ROOT / "tests/integration"

_PREPARE_STEP = "Prepare wheel-only test workspace"
# A repository script named by a manifest command.
_SCRIPT_RE = re.compile(r"\.github/scripts/[\w./-]+")
# A command that leaves the prepared workspace for the checkout.
_CHECKOUT_CD_RE = re.compile(r"""\bcd\s+["']?\$GITHUB_WORKSPACE["']?""")


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


def _manifest_platforms() -> list[str]:
    """Every platform manifest, by the name its file carries."""
    return platform_names()


def _prepare_steps() -> list[tuple[str, str, str]]:
    """Every prepare step: ``(workflow name, platform, run script)``.

    There used to be a workflow per platform under ``integration-test-<p>.yml``,
    each with its own copy of the step, and the platform came off the file's
    name and its ``platform:`` input. The pipeline is one file now and the step
    body is identical for every platform, so each platform's manifest is paired
    with that one script.
    """
    workflow = _WORKFLOW_DIR / "integration-tests.yml"
    script = _run_script_of_step(workflow.read_text(encoding="utf-8"), _PREPARE_STEP)
    if script is None:
        return []
    return [(workflow.name, platform, script) for platform in _manifest_platforms()]


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


def _manifest_commands(platform: str, boxing_lane: bool = False) -> list[str]:
    """The command of every group one platform's build runs, in order.

    Composed from the shared manifest rather than read out of the platform's
    config, which is what the pipeline itself does: the loader job calls
    ``compose_integration_manifest.py`` and hands the result to the hardware
    job, so asking the same composer is what keeps this file about the commands
    that actually run. ``setup_script`` lives in the same config and names a
    script too, but the workflow runs that one outside this workspace, so it is
    not among these.
    """
    return [
        group["command"] for group in composed_steps(platform, boxing_lane=boxing_lane)
    ]


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
    references = _script_references("hygon")
    assert references, "the reader found no repository script in the DCU manifest"
    assert all(_runs_in_the_checkout(command) for command, _ in references), (
        "the DCU manifest names a repository script from a command that stays in "
        "the prepared workspace, so the workspace would have to contain it"
    )


def test_the_manifest_reader_reads_the_composed_groups():
    """The reader asks the composer, so it sees the pipeline's own commands.

    Two ways to get this wrong, and both are checked: a reader that silently
    found nothing would make the checks below vacuous, and one that swept the
    config file would collect the provisioning hook the workflow runs outside
    this workspace -- and then hold a command to a workspace requirement
    nothing has.
    """
    commands = _manifest_commands("hygon")
    assert commands, "the reader found no commands at all, which cannot be right"
    assert not any("set_env" in command for command in commands), commands
    # The environment check is the group that replaced the per-platform
    # heredocs, and it is composed rather than copied from the config.
    assert any(
        ".github/scripts/check_platform_environment.py" in command
        for command in commands
    ), commands


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
    keyword: str | None = None


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
    keyword: str | None = None
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token in _OPTIONS_WITH_VALUES:
            # Every one of these takes the next token. `-m` says which files
            # the command runs; `-k` says which cases inside them survive, and
            # the case-level check below is the only reader that needs it.
            if token == "-m" and index + 1 < len(arguments):
                marker = arguments[index + 1]
            if token == "-k" and index + 1 < len(arguments):
                keyword = arguments[index + 1]
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
    return _Selection(tuple(paths), marker, tuple(ignored), keyword)


def _manifest_selections(platform: str) -> list[_Selection]:
    """What every pytest invocation in one platform's manifest runs.

    A boxing platform runs two builds, and each composes its own selection of
    the same manifest; a file counts as reached when either build runs it.
    """
    selections = []
    builds = (False, True) if platform in boxing_platforms() else (False,)
    for boxing_lane in builds:
        for command in _manifest_commands(platform, boxing_lane=boxing_lane):
            selection = _selection_of(command)
            if selection is not None:
                selections.append(selection)
    return selections


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
#
# What is left after issue #409 worked the list #391 left behind is the
# environment group, and nothing else: files a job cannot run as the images and
# mounts stand. None of them is waiting on a code change, so the entry *is* the
# decision and is written as one, rather than as a task someone forgot.
_KNOWN_UNREACHABLE: dict[str, str] = {
    "tests/integration/test_apex_compat.py": (
        "needs NVIDIA apex and its amp_C extension, and no CI platform is "
        "NVIDIA: flagos boxes its device onto CUDA while carrying a CPU "
        "PyTorch, as cuda.yml's own version assertion states. Accepted "
        "exclusion unless a job that installs apex is added"
    ),
    "tests/integration/test_fallback_trace.py": (
        "needs --model and a mounted Qwen3 checkpoint, which only the CUDA "
        "runner mounts; wiring it in needs a step on that manifest and a "
        "decision about what the other six do with it"
    ),
    "tests/integration/test_fallback_trace_train.py": (
        "needs --model and a mounted Qwen3 checkpoint; same as test_fallback_trace.py"
    ),
    "tests/integration/test_profiler_qwen3_infer.py": (
        "needs --model and a mounted Qwen3 checkpoint; same as test_fallback_trace.py"
    ),
    "tests/integration/ops/test_flaggems_cpp_dispatch.py": (
        "carries only flaggems_cpp, and every set_env_*.sh builds FlagGems "
        "without its C++ path (FLAGOS_BUILD_FLAGGEMS_CPP=0). Accepted "
        "exclusion until one job opts into that build"
    ),
    "tests/integration/ops/test_full_cuda_coverage.py": (
        "exercises the CUDA boxing path: every case asserts that an op "
        "backends_nvidia.conf routes to `cuda` dispatches to the boxing kernel "
        "rather than reaching cpu_fallback, and the file's own docstring says "
        "its ops need the external libtorch_cuda.so. The CUDA lane was the only "
        "manifest that selected the `cuda` marker, and it now builds the "
        "default (FlagGems-primary) wheel, which has no boxing kernels for "
        "these cases to assert on. Accepted exclusion until the vendor-SDK "
        "native kernels that replace the boxing path land, or a boxing-on CUDA "
        "job is added"
    ),
    "tests/integration/ops/test_tileops_generated.py": (
        "carries `cuda` for the same reason -- its routes are bound on the CUDA "
        "dispatch key -- and the same CUDA lane step was the only manifest "
        "command that selected it; see test_full_cuda_coverage.py. Accepted "
        "exclusion on the same terms (it would also self-skip: no job builds "
        "the TileOPs kernel set)"
    ),
}


def test_the_reachability_reader_sees_what_the_manifests_actually_ask_for():
    """A reader that quietly found nothing would make the check below vacuous.

    Every manifest runs at least one sweep by marker, so a reading that missed
    them -- or one that stopped seeing ``-m`` -- has to fail here rather than
    report a clean tree.

    The Ascend manifest names ``test_factory_ops.py`` in a plain scalar entry
    and in no other, so a reader that lost that entry -- or one that stopped
    composing the manifest at all -- has to fail here rather than report a
    clean tree.
    """
    commands = _manifest_commands("ascend")
    assert commands, "the reader found no command at all in the Ascend manifest"
    plain = "python -m pytest tests/integration/test_factory_ops.py -v -s --tb=short"
    assert plain in commands, (
        "the plain-scalar `command:` form went unread; the reader sees "
        f"{[command for command in commands if 'test_factory_ops' in command]!r}"
    )
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


# ---------------------------------------------------------------------------
# Case-level reachability
#
# The check above is per file, and a file counts as reached when any one of its
# cases is. That is the weaker half of the same question, and the gap between
# the two halves is where coverage goes missing without a red build:
#
#   * `multi_device_contracts` deselects twenty case names with `-k`, and five
#     of them are every case in `test_nonzero_device_dtype_cast.py`. The file
#     is selected by that command, so the orphan check counts it -- and it has
#     no record on any of the fourteen lanes. The manifest comment says `-k`
#     was chosen over `--ignore` *because* `--ignore` trips the orphan check,
#     which makes the bypass a documented one.
#   * Fifty-four `cuda`- and `metax`-marked cases and twenty-seven
#     `flaggems_python` ones live in files a sweep does select, under marks no
#     sweep's `-m` accepts. Issue #555 registered five of them; the class is
#     larger.
#
# So this half asks the question per case: for each test function, is there a
# composed command whose paths cover its file, whose `-m` accepts its marks,
# and whose `-k` keeps its name? Marks are read per case here rather than per
# file, which needs the decorators a function and its enclosing class carry --
# read off the AST, so this check still runs with no device and no imports.
#
# Parametrization is not expanded: a case is the function, and `-k` matching
# against the id of one parameter set would need the parameters evaluated.
# Every `-k` in the manifest names functions and classes, never parameter ids,
# so the approximation costs nothing today and the check below says so if that
# changes.
# ---------------------------------------------------------------------------

# Marks a decorator can carry that select nothing, same list as the file-level
# reader uses.
_PYTEST_MARK_PREFIX = ("pytest", "mark")


def _decorator_marks(decorators: list[ast.expr]) -> set[str]:
    """The `pytest.mark.<name>` marks in a decorator list."""
    marks = set()
    for decorator in decorators:
        node = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        parts.reverse()
        if len(parts) >= 3 and tuple(parts[:2]) == _PYTEST_MARK_PREFIX:
            marks.add(parts[2])
    return marks - _NON_SELECTING_MARKS


def _module_marks(tree: ast.Module) -> set[str]:
    """The marks a module-level ``pytestmark`` applies to every case in it.

    ``pytestmark`` is sometimes assigned a list the module built under another
    name a few lines up -- ``ops/test_dtype_cast_copy.py`` does, and reading
    only the right-hand side made its eight cases look unmarked and so
    unreachable, when four lanes run all fifty-six of them. So module-level
    list and tuple bindings are collected first and a bare ``Name`` on the
    right is resolved through them.
    """
    bindings: dict[str, list[ast.expr]] = {}
    marks: set[str] = set()

    def items_of(value: ast.expr) -> list[ast.expr]:
        if isinstance(value, (ast.List, ast.Tuple)):
            return list(value.elts)
        if isinstance(value, ast.Name):
            return bindings.get(value.id, [])
        return [value]

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if isinstance(node.value, (ast.List, ast.Tuple)):
            for name in names:
                bindings[name] = list(node.value.elts)
        if "pytestmark" in names:
            marks |= _decorator_marks(items_of(node.value))
    return marks


class _Case(NamedTuple):
    """One test function: where it lives, its node id parts, and its marks."""

    file: str
    parts: tuple[str, ...]
    marks: frozenset[str]

    @property
    def node_id(self) -> str:
        return "::".join((self.file,) + self.parts)


def _cases_of(relative: str) -> list[_Case]:
    """Every test function in one file, with the marks it ends up carrying.

    A case carries its own decorators, those of every class enclosing it, and
    the module's ``pytestmark`` -- which is what pytest collects it under.
    """
    tree = ast.parse((_REPO_ROOT / relative).read_text(encoding="utf-8"))
    found: list[_Case] = []

    def walk(body, inherited: set[str], parts: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(
                    node.body,
                    inherited | _decorator_marks(node.decorator_list),
                    parts + (node.name,),
                )
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test"):
                found.append(
                    _Case(
                        relative,
                        parts + (node.name,),
                        frozenset(inherited | _decorator_marks(node.decorator_list)),
                    )
                )

    walk(tree.body, _module_marks(tree), ())
    return found


def _all_cases() -> list[_Case]:
    """Every test case under ``tests/integration/``."""
    return [
        case for relative in sorted(_test_file_paths()) for case in _cases_of(relative)
    ]


def _keeps(expression: str, parts: tuple[str, ...]) -> bool:
    """What pytest's ``-k`` would answer for a case whose id ends in ``parts``.

    Same grammar as ``-m``, with names matched as substrings of the node id's
    components rather than as marks, which is what ``-k`` does.
    """
    marks = {
        name
        for name in _MARKER_TOKEN_RE.findall(expression)
        if name not in ("and", "or", "not", "(", ")")
        and any(name in part for part in parts)
    }
    return _evaluate(expression, marks)


def _unselected_cases() -> list[_Case]:
    """Every case no composed command collects."""
    selections = {
        platform: _manifest_selections(platform) for platform in _manifest_platforms()
    }
    unselected = []
    for case in _all_cases():
        if any(
            not any(_names_target(case.file, skip) for skip in selection.ignored)
            and any(_names_target(case.file, path) for path in selection.paths)
            and (
                selection.marker is None or _evaluate(selection.marker, set(case.marks))
            )
            and (selection.keyword is None or _keeps(selection.keyword, case.parts))
            for platform_selections in selections.values()
            for selection in platform_selections
        ):
            continue
        unselected.append(case)
    return unselected


# Cases no composed command collects, grouped by the reason they are left that
# way. This is the ratchet at case granularity: it has to match reality
# exactly, so a `-k` that drops a case, or a mark no sweep accepts, is a
# visible edit here rather than a quiet hole behind a green lane.
#
# Every file-level exclusion in _KNOWN_UNREACHABLE implies its cases, so those
# files are not repeated below; what is left is the cases inside files that do
# reach a job.
_KNOWN_UNSELECTED_CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "cuda_reference": (
        (
            "asserts flagos against a live `cuda:0` reference, or overrides the "
            "op to the boxing route, and no lane has both: every platform's "
            "torch is CPU-only or vendor-CUDA-ABI, and `FLAGOS_OP_<op>=cuda` "
            "needs a boxing wheel. The cases self-skip on "
            "`torch.cuda.is_available()` where they can, so wiring them in on a "
            "boxing lane is the fix rather than deleting them. Tracked by issue "
            "#555"
        ),
        (
            "tests/integration/ops/test_abs_dispatch.py::TestAbsCorrectness::test_abs_matches_cuda",
            "tests/integration/ops/test_acos_dispatch.py::TestAcosCorrectness::test_acos_matches_cuda",
            "tests/integration/ops/test_acos_dispatch.py::TestAcosDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_add_dispatch.py::TestAddTensorCorrectness::test_add_matches_cuda",
            "tests/integration/ops/test_all_dispatch.py::TestAllCorrectness::test_all_matches_cuda",
            "tests/integration/ops/test_all_dispatch.py::TestAllDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_bitwise_and_dispatch.py::TestBitwiseAndDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_bmm_dispatch.py::TestBmmDispatch::test_bmm_matches_cuda_ref",
            "tests/integration/ops/test_bmm_dispatch.py::TestBmmDispatchLog::test_dispatch_log_bmm_out_cuda_override",
            "tests/integration/ops/test_cat_dispatch.py::TestCatDispatch::test_cat_matches_cuda_ref",
            "tests/integration/ops/test_constant_pad_nd_dispatch.py::TestConstantPadNdDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_cos_dispatch.py::TestCosCorrectness::test_cos_matches_cuda",
            "tests/integration/ops/test_cos_dispatch.py::TestCosDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_div_scalar_dispatch.py::TestDivScalarCorrectness::test_div_scalar_matches_cuda",
            "tests/integration/ops/test_embedding_dense_backward_dispatch.py::TestEmbeddingDenseBackwardDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_embedding_dispatch.py::TestEmbeddingDispatch::test_embedding_matches_cuda_ref",
            "tests/integration/ops/test_le_dispatch.py::TestLeDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_mean_dispatch.py::TestMeanDimCorrectness::test_mean_matches_cuda",
            "tests/integration/ops/test_mm_dispatch.py::TestMmDispatch::test_mm_matches_cuda_ref",
            "tests/integration/ops/test_mm_dispatch.py::TestMmDispatchLog::test_dispatch_log_mm_out_cuda_override",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorCorrectness::test_mul_matches_cuda",
            "tests/integration/ops/test_mul_scalar_dispatch.py::TestMulScalarCorrectness::test_mul_scalar_matches_cuda",
            "tests/integration/ops/test_neg_dispatch.py::TestNegCorrectness::test_neg_matches_cuda",
            "tests/integration/ops/test_new_ones_dispatch.py::TestNewOnesDispatch::test_dispatch_log_cuda",
            "tests/integration/ops/test_nll_loss_dispatch.py::TestNllLossDispatch::test_dispatch_log_backward_cuda",
            "tests/integration/ops/test_nll_loss_dispatch.py::TestNllLossDispatch::test_dispatch_log_forward_cuda",
            "tests/integration/ops/test_ones_like_dispatch.py::TestOnesLikeDispatch::test_dispatch_log_cuda",
            "tests/integration/ops/test_pow_dispatch.py::TestPowTensorScalarCorrectness::test_pow_matches_cuda",
            "tests/integration/ops/test_pow_dispatch.py::TestPowTensorScalarDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_rsqrt_dispatch.py::TestRsqrtCorrectness::test_rsqrt_matches_cuda",
            "tests/integration/ops/test_rsqrt_dispatch.py::TestRsqrtDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_scalar_tensor_dispatch.py::TestScalarTensorDispatch::test_dispatch_log_cuda",
            "tests/integration/ops/test_silu_backward_dispatch.py::TestSiluBackwardDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_silu_dispatch.py::TestSiluCorrectness::test_silu_matches_cuda",
            "tests/integration/ops/test_sin_dispatch.py::TestSinCorrectness::test_sin_matches_cuda",
            "tests/integration/ops/test_sin_dispatch.py::TestSinDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_slice_backward_dispatch.py::TestSliceBackwardDispatch::test_dispatch_log_cuda_override",
            "tests/integration/ops/test_sum_dispatch.py::TestSumDimCorrectness::test_sum_dim_matches_cuda",
            "tests/integration/ops/test_zeros_dispatch.py::TestZerosDispatch::test_dispatch_log_cuda",
        ),
    ),
    "flaggems_python_override": (
        (
            "asserts `FLAGOS_OP_<op>=flaggems_python` emits a dispatch line, "
            "which needs the op claimed on PrivateUse1; no lane's `-m` names "
            "the marker, and on a default wheel an op the conf routes elsewhere "
            "has no slot for the override to reach. This is issue #555's class "
            "exactly"
        ),
        (
            "tests/integration/ops/test_abs_dispatch.py::TestAbsDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_acos_dispatch.py::TestAcosDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_add_dispatch.py::TestAddTensorDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_all_dispatch.py::TestAllDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_bitwise_and_dispatch.py::TestBitwiseAndDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_bmm_dispatch.py::TestBmmDispatchLog::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_cat_dispatch.py::TestCatDispatchLog::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_constant_pad_nd_dispatch.py::TestConstantPadNdDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_cos_dispatch.py::TestCosDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_embedding_dense_backward_dispatch.py::TestEmbeddingDenseBackwardDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_embedding_dispatch.py::TestEmbeddingDispatchLog::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_le_dispatch.py::TestLeDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_mean_dispatch.py::TestMeanDimDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_mm_dispatch.py::TestMmDispatchLog::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_neg_dispatch.py::TestNegDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_nll_loss_dispatch.py::TestNllLossDispatch::test_dispatch_log_backward_flaggems_python",
            "tests/integration/ops/test_nll_loss_dispatch.py::TestNllLossDispatch::test_dispatch_log_forward_flaggems_python",
            "tests/integration/ops/test_pow_dispatch.py::TestPowTensorScalarDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_rsqrt_dispatch.py::TestRsqrtDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_silu_backward_dispatch.py::TestSiluBackwardDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_silu_dispatch.py::TestSiluDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_sin_dispatch.py::TestSinDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_slice_backward_dispatch.py::TestSliceBackwardDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_softmax_dispatch.py::TestSoftmaxDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_sum_dispatch.py::TestSumDimDispatch::test_dispatch_log_flaggems_python",
            "tests/integration/ops/test_where_dispatch.py::TestWhereDispatch::test_dispatch_log_flaggems_python",
        ),
    ),
    "multi_device_boxing_gaps": (
        (
            "deselected by name from `multi_device_contracts`, the one group "
            "that sweeps `-m multi_device`, because each reaches an op the "
            "boxing confs route to `none`. The `-k` was chosen over `--ignore` "
            "so the file-level check above keeps counting the files as "
            "selected, which is why the cases have to be named here. Five of "
            "them are every case in test_nonzero_device_dtype_cast.py, so that "
            "file runs nowhere at all"
        ),
        (
            "tests/integration/test_allocator.py::TestCachingAllocatorOps::test_matmul",
            "tests/integration/test_compute_device_index.py::test_binary_tensor_op",
            "tests/integration/test_compute_device_index.py::test_binary_tensor_scalar_op",
            "tests/integration/test_compute_device_index.py::test_cpu_scalar_operand_lands_on_operand_device",
            "tests/integration/test_compute_device_index.py::test_current_device_is_restored",
            "tests/integration/test_compute_device_index.py::test_kwarg_op",
            "tests/integration/test_compute_device_index.py::test_multinomial_on_nonzero_device",
            "tests/integration/test_compute_device_index.py::test_random_inplace_op",
            "tests/integration/test_compute_device_index.py::test_tensor_list_op",
            "tests/integration/test_compute_device_index.py::test_tensor_scalar_op",
            "tests/integration/test_compute_device_index.py::test_ternary_tensor_op",
            "tests/integration/test_compute_device_index.py::test_unary_inplace_op",
            "tests/integration/test_compute_device_index.py::test_unary_op",
            "tests/integration/test_factory_device_index.py::test_full_like_defaults_to_source_device",
            "tests/integration/test_factory_device_index.py::test_like_factory_defaults_to_source_device",
            "tests/integration/test_nonzero_device_dtype_cast.py::test_boxed_op_leaves_current_device_untouched",
            "tests/integration/test_nonzero_device_dtype_cast.py::test_cast_on_nonzero_device_leaves_current_device_alone",
            "tests/integration/test_nonzero_device_dtype_cast.py::test_cast_values_are_exact_on_nonzero_device",
            "tests/integration/test_nonzero_device_dtype_cast.py::test_int64_to_bool_cast_on_nonzero_device",
            "tests/integration/test_nonzero_device_dtype_cast.py::test_issue_313_repro_in_a_fresh_process",
        ),
    ),
    "unmarked_in_swept_dir": (
        (
            "carries no selecting mark, and every command covering its file "
            "filters by `-m`, so nothing collects the case. Giving it the mark "
            "its siblings carry is the fix; it is listed rather than fixed here "
            "so this ratchet lands without changing what any lane runs"
        ),
        (
            "tests/integration/ops/test_cat_dispatch.py::TestCatDispatch::test_cat_kv_shape_dtypes",
            "tests/integration/ops/test_cat_dispatch.py::TestCatKvCacheAppend::test_cat_kv_cache_append_preserves_prefill_prefix",
            "tests/integration/ops/test_cat_dispatch.py::TestCatKvCacheAppend::test_cat_kv_cache_various_head_dims",
            "tests/integration/ops/test_flaggems_conf_consistency.py::TestFlagGemsConfConsistency::test_python_kernels_use_runtime_resolved_names",
            "tests/integration/ops/test_index_dispatch.py::TestBaseTensorSequenceProtocol::test_base_tensor_is_not_a_sequence",
            "tests/integration/ops/test_index_dispatch.py::TestBaseTensorSequenceProtocol::test_cpu_scalar_list_after_device_init",
            "tests/integration/ops/test_index_dispatch.py::TestBaseTensorSequenceProtocol::test_device_scalar_list_after_device_init",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorComplex64::test_a_transposed_complex64_operand_declines_the_decomposition",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorComplex64::test_complex64_broadcast_matches_cpu",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorComplex64::test_complex64_is_bit_identical_to_the_real_view_product",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorComplex64::test_complex64_matches_cpu",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorComplex64::test_complex64_takes_the_real_views",
            "tests/integration/ops/test_mul_dispatch.py::TestMulTensorComplex64::test_real_times_complex64_promotes_and_matches_cpu",
        ),
    ),
    "metax_override": (
        (
            "asserts `FLAGOS_OP_<op>=metax` emits a dispatch line. "
            "backends_metax.conf has no vendor slot, so the MetaX lane's own "
            "sweep excludes the marker and no other platform can satisfy it"
        ),
        (
            "tests/integration/ops/test_bitwise_and_dispatch.py::TestBitwiseAndDispatch::test_dispatch_log_metax",
            "tests/integration/ops/test_cos_dispatch.py::TestCosDispatch::test_dispatch_log_metax",
            "tests/integration/ops/test_new_ones_dispatch.py::TestNewOnesDispatch::test_dispatch_log_metax",
            "tests/integration/ops/test_scalar_tensor_dispatch.py::TestScalarTensorDispatch::test_dispatch_log_metax",
            "tests/integration/ops/test_sin_dispatch.py::TestSinDispatch::test_dispatch_log_metax",
            "tests/integration/ops/test_where_dispatch.py::TestWhereDispatch::test_dispatch_log_metax",
        ),
    ),
    "flaggems_runtime_route": (
        (
            "carries `flaggems` alone while every sweep that sets `-m` pairs it "
            "with `main_ops`, so no lane collects it"
        ),
        (
            "tests/integration/ops/test_bmm_dispatch.py::TestBmmDispatchLog::test_dispatch_log_bmm_out_flaggems_runtime",
            "tests/integration/ops/test_mm_dispatch.py::TestMmDispatchLog::test_dispatch_log_mm_out_flaggems_runtime",
            "tests/integration/ops/test_rng_dispatch.py::TestRngDropout::test_dropout_reproducible_on_flaggems_path",
        ),
    ),
}


def test_the_case_reader_sees_the_cases_the_manifests_filter():
    """A reader that found no cases, or no `-k`, would make the check vacuous.

    Three things have to hold for the ratchet below to mean anything: cases are
    found at all, their marks are read per case rather than per file, and the
    `-k` filters are read. Each is asserted against a case that exists for its
    own reasons, so losing any one of them is red here rather than silent.
    """
    cases = _all_cases()
    assert len(cases) > 1000, f"the reader found only {len(cases)} cases"

    marks = {case.node_id: case.marks for case in cases}
    acos = "tests/integration/ops/test_acos_dispatch.py"
    assert marks[f"{acos}::TestAcosCorrectness::test_acos_shape"] == frozenset(
        {"anyplatform"}
    ), "class-level marks are not reaching the cases under them"
    assert marks[f"{acos}::TestAcosDispatch::test_dispatch_log_flaggems_python"] == (
        frozenset({"flaggems_python"})
    ), "a case is carrying its sibling class's marks, so marks are read per file"

    keywords = [
        selection.keyword
        for platform in _manifest_platforms()
        for selection in _manifest_selections(platform)
        if selection.keyword is not None
    ]
    assert keywords, "no `-k` filter was read, so every deselected case reads as run"
    assert _keeps("not test_matmul", ("TestCachingAllocatorOps", "test_allocator"))
    assert not _keeps("not test_matmul", ("TestCachingAllocatorOps", "test_matmul"))


def test_every_k_filter_still_names_something_that_exists():
    """A `-k` name that matches nothing is a deselection nobody is getting.

    The lists are maintained by hand against failures on one wheel, so a
    renamed or deleted case leaves a token behind that reads as a live
    exclusion while filtering nothing.
    """
    names = {path.rsplit("/", 1)[-1] for path in _test_file_paths()}
    for case in _all_cases():
        names.update(case.parts)
    stale = sorted(
        {
            token
            for platform in _manifest_platforms()
            for selection in _manifest_selections(platform)
            if selection.keyword is not None
            for token in _MARKER_TOKEN_RE.findall(selection.keyword)
            if token not in ("and", "or", "not")
            and not any(token in name for name in names)
        }
    )
    assert not stale, (
        f"{stale} appear in a manifest `-k` but match no file, class or case "
        "name; drop them from the filter they are in"
    )


def test_no_integration_case_is_collected_by_no_manifest():
    """Every case under tests/integration/ has to be collected by some job.

    The file-level check above passes as soon as one case in a file is
    selected, so a `-k` that drops the rest, or a mark no sweep's `-m`
    accepts, leaves cases that no lane runs and no check mentions. Five cases
    of test_nonzero_device_dtype_cast.py were in exactly that state: the file
    read as reachable and had no record on any of the fourteen lanes.
    """
    declared = {
        node_id
        for _reason, node_ids in _KNOWN_UNSELECTED_CASES.values()
        for node_id in node_ids
    }
    covered_by_file = set(_KNOWN_UNREACHABLE)
    undeclared = sorted(
        case.node_id
        for case in _unselected_cases()
        if case.node_id not in declared and case.file not in covered_by_file
    )
    assert not undeclared, (
        f"{undeclared} are collected by no manifest command, so no CI job runs "
        "them; give each the mark a sweep selects, or add it to "
        "_KNOWN_UNSELECTED_CASES under the reason it cannot be"
    )


def test_the_known_unselected_list_has_not_gone_stale():
    """The exclusions have to be re-earned, the same way the file list's are.

    An entry whose case a lane now collects documents a hole that has closed,
    an entry naming a case that is gone is a stale id pretending to be a
    decision, and an entry whose file is already excluded wholesale says
    nothing the file-level list does not.
    """
    unselected = {case.node_id for case in _unselected_cases()}
    existing = {case.node_id for case in _all_cases()}
    for reason, node_ids in _KNOWN_UNSELECTED_CASES.values():
        assert reason.strip(), "an entry carries no reason"
        wired_in = sorted(
            node_id
            for node_id in node_ids
            if node_id in existing and node_id not in unselected
        )
        assert not wired_in, (
            f"{wired_in} are collected now, so their entries in "
            "_KNOWN_UNSELECTED_CASES describe a hole that has closed; drop them"
        )
        vanished = sorted(node_id for node_id in node_ids if node_id not in existing)
        assert not vanished, (
            f"{vanished} are listed in _KNOWN_UNSELECTED_CASES but do not "
            "exist any more"
        )
        shadowed = sorted(
            node_id
            for node_id in node_ids
            if node_id.split("::", 1)[0] in _KNOWN_UNREACHABLE
        )
        assert not shadowed, (
            f"{shadowed} sit in files _KNOWN_UNREACHABLE already excludes "
            "whole; drop the case entries and leave the file entry"
        )
