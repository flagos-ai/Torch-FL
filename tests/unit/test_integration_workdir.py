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
FlagGems' own dependencies (``PyYAML==6.0.1`` in ``set_env_dcu.sh``).
"""

import io
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import tokenize
import types
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import pytest
import yaml
from manifest_composer import composed_steps, load_module, platform_names


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_DIR = _REPO_ROOT / ".github/workflows"
_INTEGRATION_DIR = _REPO_ROOT / "tests/integration"
_PLATFORM_CONFIG_DIR = _REPO_ROOT / ".github/configs"
_CONFTEST = load_module(_INTEGRATION_DIR / "conftest.py", "integration_conftest")
_OPS_CONFTEST = load_module(_INTEGRATION_DIR / "ops/conftest.py", "ops_conftest")

_PREPARE_STEP = "Prepare wheel-only test workspace"
_MODEL_MOUNT = "/var/lib/torch-fl-ci/models/Qwen3-0.6B"
_MODEL_CACHE = "qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca"
# A repository script named by a manifest command.
_SCRIPT_RE = re.compile(r"\.github/scripts/[\w./-]+")
# A repository script that owns a selection instead of naming every file.
# The FlagGems runner lives beside its cases and chooses the native or kernel
# pass through a required suite argument.
_TEST_RUNNER_RE = re.compile(r"\btests/integration/[\w./-]+/run\.py\b")
# A command that leaves the prepared workspace for the checkout.
_CHECKOUT_CD_RE = re.compile(r"""\bcd\s+["']?\$GITHUB_WORKSPACE["']?""")


def _model_test_module(relative: str, platform: str = "mps") -> types.ModuleType:
    """Load a model test without importing torch_fl's compiled extension."""
    torch = pytest.importorskip(
        "torch",
        reason="Fallback tracer tests require the platform's PyTorch runtime",
    )
    module = types.ModuleType(relative)
    module.__file__ = str(_REPO_ROOT / relative)
    init = types.ModuleType("integration_test_device_init")
    init.torch = torch
    init.torch_fl = types.SimpleNamespace(
        flagos=types.SimpleNamespace(synchronize=lambda: None)
    )
    capabilities = types.SimpleNamespace(platform=platform, device=True)
    support = types.SimpleNamespace(
        detect_platform=lambda: platform,
        capabilities_for_platform=lambda _: capabilities,
    )
    profiler = types.ModuleType("integration_test_profiler_support")
    profiler.capabilities_for_platform = support.capabilities_for_platform
    injected = {
        "device_init": init,
        "platform_support": support,
        "profiler_support": profiler,
    }
    with mock.patch.dict(sys.modules, injected):
        exec(
            compile(
                (_REPO_ROOT / relative).read_text(encoding="utf-8"), relative, "exec"
            ),
            module.__dict__,
        )
    return module


@pytest.mark.parametrize(
    "relative",
    (
        "tests/integration/test_fallback_trace.py",
        "tests/integration/test_fallback_trace_train.py",
    ),
)
def test_model_fallback_tracer_observes_calls_inside_kernel_bodies(relative):
    """A kernel body's CPU inputs are seen without a nested dispatch mode."""
    torch = pytest.importorskip("torch")
    module = _model_test_module(relative)
    tracer = module.FallbackTracer()
    tracer.flagos_device_types = ("cpu",)
    tracer._is_native = lambda _: True

    x = torch.ones(2)
    with tracer, torch.no_grad():
        imported = torch.mul(x, x)
        torch.reshape(imported, (2,))

    assert tracer.native_ops
    assert tracer.total_ops >= 2


@pytest.mark.parametrize(
    "relative",
    (
        "tests/integration/test_fallback_trace.py",
        "tests/integration/test_fallback_trace_train.py",
    ),
)
def test_model_fallback_tracer_records_cpu_input_dispatches(relative):
    """CPU kernel bodies are counted instead of losing a nested mode's calls."""
    torch = pytest.importorskip("torch")
    module = _model_test_module(relative)
    tracer = module.FallbackTracer()
    tracer.flagos_device_types = ("cpu",)

    with tracer:
        torch.add(torch.ones(2), torch.ones(2))

    assert tracer.cpu_input_calls


def test_model_profiler_recognizes_gpu_kernel_categories():
    """Device kernels are selected by name or category, not a text comparison."""
    pytest.importorskip("torch")
    module = _model_test_module("tests/integration/test_profiler_qwen3_infer.py")
    events = (
        {"name": "aten::add", "cat": "cpu_op"},
        {"name": "kernel", "cat": "gpu_kernel"},
        {"name": "Memcpy", "cat": "gpu_memcpy"},
    )
    assert module._kernel_events(events) == [events[1]]


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


def test_the_archive_fallback_replaces_a_stale_workspace():
    """Exercise the fallback that runs when both checkout attempts fail."""
    workflow = _run_script_of_step(
        (_WORKFLOW_DIR / "integration-tests.yml").read_text(encoding="utf-8"),
        "Fetch the source archive if checkout failed",
    )
    assert workflow is not None
    assert "api.github.com/repos/${SOURCE_REPOSITORY}/tarball/${SOURCE_REF}" in (
        workflow
    )
    script = workflow.replace(
        '"https://codeload.github.com/${SOURCE_REPOSITORY}/tar.gz/${SOURCE_REF}"',
        '"unused-codeload-endpoint"',
    ).replace(
        '"https://api.github.com/repos/${SOURCE_REPOSITORY}/tarball/${SOURCE_REF}"',
        '"source-archive-endpoint"',
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        (source / "setup.sh").write_text("echo verified\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "add", "setup.sh"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Torch-FL CI",
                "-c",
                "user.email=ci@example.com",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        archive = root / "source.tar.gz"
        outer = root / "Torch-FL-revision"
        outer.mkdir()
        inner = root / "source.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "archive",
                "--format=tar",
                f"--output={inner}",
                "HEAD",
            ],
            check=True,
        )
        subprocess.run(
            ["tar", "-xf", str(inner), "-C", str(outer)],
            check=True,
        )
        subprocess.run(
            ["tar", "-czf", str(archive), "-C", str(root), outer.name],
            check=True,
        )
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "stale.txt").write_text("stale\n", encoding="utf-8")
        commands = root / "bin"
        commands.mkdir()
        curl = commands / "curl"
        curl.write_text(
            "#!/bin/sh\n"
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "--output" ]; then\n'
            '    shift; cp "$SOURCE_ARCHIVE" "$1"; exit 0\n'
            "  fi\n"
            "  shift\n"
            "done\n"
            "exit 1\n",
            encoding="utf-8",
        )
        curl.chmod(0o755)
        runner_temp = root / "runner-temp"
        runner_temp.mkdir()
        environment = {
            **os.environ,
            "PATH": f"{commands}:{os.defpath}",
            "GIT_CEILING_DIRECTORIES": str(root),
            "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
            "GIT_DIR": "",
            "SOURCE_REPOSITORY": "flagos-ai/Torch-FL",
            "SOURCE_REF": revision,
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_WORKSPACE": str(workspace),
            "GH_TOKEN": "test-token",
            "SOURCE_ARCHIVE": str(archive),
        }
        result = subprocess.run(
            ["bash", "-xc", script],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (workspace / "setup.sh").read_text(encoding="utf-8") == (
            "echo verified\n"
        )
        assert not (workspace / "stale.txt").exists()


def _manifest_platforms() -> list[str]:
    """Every platform manifest, by the name its file carries."""
    return platform_names()


def _workflow_selections() -> list[tuple[str, "_Selection"]]:
    """Source-check pytest invocations outside the hardware manifests."""
    selections = []
    for workflow in _WORKFLOW_DIR.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        root = yaml.safe_load(text)
        jobs = root.get("jobs") or root.get(True, {}).get("jobs", {})
        for name, job in jobs.items():
            for step in job.get("steps", []):
                for command in _logical_lines(step.get("run", "")):
                    selection = _selection_of(command)
                    if selection is not None and selection.paths:
                        selections.append((f"{workflow.name}:{name}", selection))
    return selections


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


def test_the_model_mount_is_prepared_on_its_platform_host():
    """Restore the checkpoint on the runner that will start the vendor image."""
    workflow = yaml.safe_load(
        (_WORKFLOW_DIR / "integration-tests.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    integration = jobs["integration-tests"]
    assert "model-mount" not in jobs
    assert "container" not in integration
    assert integration["needs"] == "configuration"
    assert integration["runs-on"] == (
        "${{ fromJson(needs.configuration.outputs.runner_labels) }}"
    )
    steps = integration["steps"]
    locate = next(
        step
        for step in steps
        if step.get("name") == "Prepare the fixed model bind root"
    )
    assert "HOST_MODEL_ROOT" not in locate["env"]
    assert 'HOST_MODEL_ROOT="${RUNNER_TEMP:?}/torch-fl-models"' in locate["run"]
    assert 'root="$HOST_MODEL_ROOT/Qwen3-0.6B"' in locate["run"]
    assert "TORCH_FL_MODEL_ROOT=$HOST_MODEL_ROOT" in locate["run"]
    assert "root=$HOST_MODEL_ROOT" not in locate["run"]
    workspace = next(step for step in steps if step.get("id") == "workspace")
    assert workspace["env"]["HOST_MODEL_ROOT"] == ("${{ env.TORCH_FL_MODEL_ROOT }}")
    assert workspace["env"]["MODEL_PATH"] == (
        "${{ needs.configuration.outputs.model_path }}"
    )
    assert '--model-root "$HOST_MODEL_ROOT"' in workspace["run"]
    assert '--model-path "$MODEL_PATH"' in workspace["run"]
    restore = next(
        step for step in steps if step.get("uses") == "actions/cache/restore@v4"
    )
    assert restore["with"]["key"] == "${{ env.TORCH_FL_CHECKPOINT_CACHE_KEY }}"
    assert integration["env"]["TORCH_FL_CHECKPOINT_CACHE_KEY"] == _MODEL_CACHE
    assert restore["with"]["path"] == "${{ runner.temp }}/torch-fl-model-cache"
    verify = next(
        step
        for step in steps
        if step.get("name") == "Verify or download the pinned Qwen3 checkpoint"
    )
    assert verify["env"]["MODEL_SOURCE"] == "${{ steps.model-storage.outputs.root }}"
    assert "MODEL_COMMAND" not in verify["env"]
    assert 'MODEL_COMMAND=check MODEL_CACHE="$MODEL_CACHE"' in verify["run"]
    assert "stage-model --path" in verify["run"]
    assert "MODEL_COMMAND=stage" in verify["run"]
    assert "MODEL_COMMAND=verify" in verify["run"]
    assert 'str(config.get("_commit_hash", "")) == revision' in verify["run"]
    assert 'python3 - "$MODEL_CACHE"' in verify["run"]
    assert "hf_hub_download" not in verify["run"]
    setup = next(step for step in steps if step.get("id") == "setup")
    assert 'cd "$${GITHUB_WORKSPACE:?}"' in setup["run"]
    assert """$(printf '%q' "$GITHUB_WORKSPACE/$SETUP_SCRIPT")""" in setup["run"]
    fallback = next(step for step in steps if step.get("id") == "source-archive")
    assert (
        "api.github.com/repos/${SOURCE_REPOSITORY}/tarball/${SOURCE_REF}"
        in (fallback["run"])
    )
    assert 'archive="${RUNNER_TEMP:-/tmp}/torch-fl-source.tar.gz"' in fallback["run"]
    for step in steps:
        if step.get("id") in {"merge-report", "merge-coverage"}:
            assert step["env"]["SOURCE_REF"] == "${{ steps.source.outputs.ref }}"
    test_step = next(step for step in steps if step.get("id") == "integration-tests")
    container = next(step for step in steps if step.get("id") == "container")
    assert "container_workspace.py start" in container["run"]
    cleanup = next(
        step
        for step in steps
        if step.get("name") == "Close the platform command environment"
    )
    assert "!cancelled()" in cleanup["if"]
    assert cleanup["env"]["CONTAINER_NAME"] == "${{ steps.container.outputs.name }}"
    assert 'find "$GITHUB_WORKSPACE" "$RUNNER_TEMP"' in cleanup["run"]
    assert 'docker rm --force "$CONTAINER_NAME"' in cleanup["run"]
    names = [step.get("name", "") for step in steps]
    assert names.index("Close the platform command environment") > names.index(
        "Retain Python coverage"
    )
    assert steps[-1]["name"] == "Close the platform command environment"
    assert "docker run" not in test_step["run"]
    assert "docker exec" in test_step["run"]


def test_every_platform_restores_the_model_to_the_mount_path():
    """MPS uses the same cache and model path as every container lane."""
    workflow = yaml.safe_load(
        (_WORKFLOW_DIR / "integration-tests.yml").read_text(encoding="utf-8")
    )
    step = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("id") == "model-storage"
    )
    assert "HOST_MODEL_ROOT" not in step["env"]
    assert 'HOST_MODEL_ROOT="${RUNNER_TEMP:?}/torch-fl-models"' in step["run"]
    assert 'root="$HOST_MODEL_ROOT/Qwen3-0.6B"' in step["run"]
    assert "TORCH_FL_MODEL_ROOT=$HOST_MODEL_ROOT" in step["run"]
    assert "root=$HOST_MODEL_ROOT" not in step["run"]
    assert "NETS" not in step["run"]


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
    """The command of every group one platform runs, in the order it runs them.

    Composed from the shared manifest rather than read out of the platform's
    config, which is what the pipeline itself does: the loader job calls
    ``compose_integration_manifest.py`` and hands the result to the hardware
    job, so asking the same composer is what keeps this file about the commands
    that actually run. ``setup_script`` lives in the same config and names a
    script too, but the workflow runs that one outside this workspace, so it is
    not among these.
    """
    return [group["command"] for group in composed_steps(platform)]


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


def test_the_manifest_reader_reads_the_composed_groups():
    """The reader asks the composer, so it sees the pipeline's own commands.

    Two ways to get this wrong, and both are checked: a reader that silently
    found nothing would make the checks below vacuous, and one that swept the
    config file would collect the provisioning hook the workflow runs outside
    this workspace -- and then hold a command to a workspace requirement
    nothing has.
    """
    commands = _manifest_commands("dcu")
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
# selects a file when its paths cover it and a case from its source carries the
# marks its `-m` expression accepts. The ops conftest's platform and build gates
# then apply to each actual case. The model over-approximates runtime skips: a
# false "reachable" is a gap left for later, while a false "unreachable" is a red
# build that says so.
# ---------------------------------------------------------------------------

# `pytest.mark.<name>` in source form: the only mark spelling that can be read
# without importing the file.
_MARK_RE = re.compile(r"pytest\.mark\.([A-Za-z_]\w*)")
_NON_SELECTING_MARKS = {
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
    uses_ops_conftest: bool = True


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
            and re.fullmatch(r"(?:python|python\d+(?:\.\d+)*)", tokens[index - 2])
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
    uses_conftest = True
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
        if token == "--noconftest":
            uses_conftest = False
            index += 1
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
    return _Selection(tuple(paths), marker, tuple(ignored), uses_conftest)


def _manifest_selections(platform: str) -> list[_Selection]:
    """What every test command in one platform's manifest runs."""
    selections = []
    for command in _manifest_commands(platform):
        selection = _selection_of(command)
        if selection is not None:
            selections.append(selection)
            continue
        # A shared runner states its case files beside itself. Its suite
        # argument is required by its CLI, so the file path alone is enough
        # to credit the files in that directory; it cannot run an unknown set.
        for runner in _TEST_RUNNER_RE.findall(command):
            suite = Path(runner).parent.as_posix()
            if not (_REPO_ROOT / suite).is_dir():
                continue
            selections.append(_Selection((suite,), None, ()))
    return selections


def _source_mark_sets(test_file: Path) -> list[set[str]]:
    """The mark sets of test cases, read from source without importing them.

    Module, class and function marks form the set pytest applies to a case.
    A module alias declared as ``pytest.mark.skipif`` carries the mark it
    names; a file-level marker still over-approximates a gated top-level case,
    which the operator conftest check applies to its actual build.
    """
    source = test_file.read_text(encoding="utf-8")
    text = _mask_python(source)
    tokens = _python_tokens(text)
    module_skip = re.search(r"^pytest\.skip\s*\(", text, re.MULTILINE)
    if module_skip:
        module_call = _balanced_call(source, module_skip.start())
        if re.search(
            r"\ballow_module_level\s*=\s*True",
            source[module_skip.start() : module_skip.start() + len(module_call)],
        ):
            return []
    module_mark = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.type == tokenize.NAME
            and token.string == "pytestmark"
            and _top_level(tokens, index)
        ),
        None,
    )
    declaration = []
    if module_mark is not None:
        declaration = tokens[module_mark + 2 : _statement_end(tokens, module_mark + 3)]
    if declaration and any(
        "tileops_runtime.is_tileops_available("
        in re.sub(r"\s+", "", _balanced_arguments(source, tokens, opening, closing))
        for opening, closing in _skipif_calls(tokens, declaration)
    ):
        return []

    declarations = _module_declared_marks(tokens)
    module_marks = _declaration_marks(declaration) | declarations["pytestmark"]
    aliases = declarations["aliases"]
    classes: list[tuple[int, set[str]]] = []
    cases: list[set[str]] = []
    for kind, name, offset, marks, arguments in _function_scopes(text):
        indent = len(_line_start(source, offset).expandtabs(8)) - len(
            _line_start(source, offset).lstrip().expandtabs(8)
        )
        while classes and classes[-1][0] >= indent:
            classes.pop()
        if kind == "class":
            classes.append((indent, set(marks) | aliases.get(name, set())))
        elif name.startswith("test_"):
            enclosing = set(module_marks)
            parent = enclosing | aliases.get(name, set())
            for _, class_marks in classes:
                parent |= class_marks
            enclosing = parent | marks
            if "tileops_runtime.is_tileops_available(" in re.sub(
                r"\s+", "", arguments.get("skipif", "")
            ):
                return []
            if not enclosing.intersection({"skip", "skipif"}):
                cases.append(enclosing)
    if cases:
        return cases
    if module_marks:
        return [module_marks]
    if aliases or declarations["other"]:
        return [set().union(*aliases.values(), declarations["other"])]
    return [set()]


def _module_declared_marks(tokens) -> dict[str, dict[str, set[str]]]:
    """The marker aliases and file-level marks a module declares."""
    declared = {"pytestmark": set(), "aliases": {}, "other": set()}
    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME:
            continue
        if token.string == "pytestmark" and _top_level(tokens, index):
            end = _statement_end(tokens, index + 3)
            declared["pytestmark"] |= _declaration_marks(tokens[index + 2 : end])
            continue
        marks = _mark_names(tokens[index : index + 5])
        start = index + 5
        if (
            token.string == "pytestmark"
            or not marks
            or start >= len(tokens)
            or tokens[start].string != "="
        ):
            continue
        end = _statement_end(tokens, start + 1)
        value = tokens[start + 1 : end]
        names = {value[position].string for position in _mark_names(value)}
        names -= _NON_SELECTING_MARKS
        if _top_level(tokens, index):
            if names:
                declared["aliases"][token.string] = names
        else:
            declared["other"] |= names
    return declared


def _line_start(text: str, offset: int) -> str:
    """The text from the start of ``offset``'s line to ``offset``."""
    return text[text.rfind("\n", 0, offset) + 1 : offset]


def _function_scopes(
    text: str,
) -> list[tuple[str, str, int, set[str], dict[str, str]]]:
    """Every class and function with its decorator marks and arguments."""
    tokens = _python_tokens(text)
    scopes: list[tuple[str, str, int, set[str], dict[str, str]]] = []
    decorators: list[tuple[set[str], dict[str, str]]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.OP and token.string == "@":
            end = _statement_end(tokens, index + 1)
            declaration = tokens[index + 1 : end]
            marks = set()
            arguments = {}
            for position in _mark_names(declaration):
                mark = declaration[position].string
                marks.add(mark)
                if (
                    mark == "skipif"
                    and position + 1 < len(declaration)
                    and declaration[position + 1].string == "("
                ):
                    opening = index + position + 2
                    arguments["skipif"] = _balanced_arguments(
                        text, tokens, opening + 1, _matching_closing(tokens, opening)
                    )
            decorators.append((marks, arguments))
            index = end
            continue
        if token.type != tokenize.NAME or token.string not in {"class", "def"}:
            index += 1
            continue
        if index + 2 >= len(tokens):
            break
        name = tokens[index + 1]
        if name.type != tokenize.NAME or tokens[index + 2].string != "(":
            index += 1
            continue
        marks = set().union(*(marks for marks, _ in decorators))
        arguments = {}
        for _, decorator_arguments in decorators:
            arguments.update(decorator_arguments)
        scopes.append(
            (
                token.string,
                name.string,
                _offset(text, name.start),
                marks,
                arguments,
            )
        )
        decorators.clear()
        index += 3
    return scopes


def _balanced_arguments(text: str, tokens, position: int, end: int) -> str:
    """The source text inside a call's parentheses, with whitespace preserved."""
    if position == end:
        return ""
    return text[
        _offset(text, tokens[position].start) : _offset(text, tokens[end].start)
    ]


def _python_tokens(text: str):
    """Tokenize masked source into a complete module token sequence."""
    return [
        token
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.string
    ]


def _top_level(tokens, index: int) -> bool:
    """Whether a ``pytestmark`` statement starts at module indentation."""
    while index >= 0 and tokens[index].type != tokenize.NEWLINE:
        index -= 1
    while index + 1 < len(tokens) and tokens[index + 1].type in {
        tokenize.NL,
        tokenize.COMMENT,
    }:
        index += 1
    return (
        index + 1 < len(tokens)
        and tokens[index + 1].type == tokenize.NAME
        and tokens[index + 1].string == "pytestmark"
        and index + 2 < len(tokens)
        and tokens[index + 2].string == "="
    )


def _statement_end(tokens, start: int) -> int:
    """The end of a statement, including a bracketed multiline expression."""
    depth = 0
    for index in range(start, len(tokens)):
        token = tokens[index]
        depth += token.string in "([{"
        depth -= token.string in ")]}"
        if depth == 0 and token.type in {tokenize.NEWLINE, tokenize.ENDMARKER}:
            return index
        if (
            depth == 0
            and token.type == tokenize.NAME
            and token.string
            in {
                "async",
                "class",
                "def",
            }
        ):
            return index
        if token.type in {tokenize.INDENT, tokenize.DEDENT}:
            return index
    raise AssertionError("statement has no terminator")


def _skipif_calls(module_tokens, statement_tokens) -> list[tuple[int, int]]:
    """The opening and closing token positions of module ``skipif`` calls."""
    first = next(
        index
        for index, token in enumerate(module_tokens)
        if token == statement_tokens[0]
    )
    stop = first + len(statement_tokens)
    return [
        (index + 1, _matching_closing(module_tokens, index + 1))
        for index in range(first, stop)
        if module_tokens[index].string == "skipif"
        and index + 1 < stop
        and module_tokens[index + 1].string == "("
    ]


def _declaration_marks(tokens) -> set[str]:
    """Selecting marker names declared in a ``pytestmark`` assignment."""
    names = {tokens[index].string for index in _mark_names(tokens)}
    return names - _NON_SELECTING_MARKS


def _mark_names(tokens) -> list[int]:
    """The positions of mark names in a decorator or declaration."""
    marks = []
    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string != "pytest":
            continue
        if index + 4 < len(tokens) and (
            tokens[index + 1].string == "."
            and tokens[index + 2].string == "mark"
            and tokens[index + 3].string == "."
            and tokens[index + 4].type == tokenize.NAME
        ):
            marks.append(index + 4)
    return marks


def _matching_closing(tokens, opening: int) -> int:
    """The closing token paired with ``opening``."""
    depth = 0
    for index in range(opening, len(tokens)):
        depth += tokens[index].string in "([{"
        depth -= tokens[index].string in ")]}"
        if depth == 0:
            return index
    raise AssertionError("unmatched opening delimiter")


def _balanced_call(text: str, opening: int) -> str:
    """The text from ``opening`` through its matching closing parenthesis."""
    depth = 0
    for index, char in enumerate(text[opening:], opening):
        depth += char == "("
        depth -= char == ")"
        if depth == 0:
            return text[opening : index + 1]
    raise AssertionError("unterminated call")


def _mask_python(text: str) -> str:
    """Blank comments and string contents without changing offsets."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    masked = list(text)
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    for token in tokens:
        if token.type == tokenize.COMMENT:
            start, stop = _offset(text, token.start), _offset(text, token.end)
        elif token.type == tokenize.STRING:
            literal = token.string
            literal_offset = len(re.match(r"[A-Za-z]*", literal).group())
            opening = literal[literal_offset : literal_offset + 3]
            quote_length = 3 if opening in {chr(34) * 3, chr(39) * 3} else 1
            start = _offset(text, token.start) + literal_offset + quote_length
            stop = _offset(text, token.end) - quote_length
        else:
            continue
        for index in range(max(0, start), min(len(text), stop)):
            if masked[index] != "\n":
                masked[index] = " "
    return "".join(masked)


def _offset(text: str, position: tuple[int, int]) -> int:
    """The absolute offset of a tokenizer row and column."""
    lines = text.splitlines(keepends=True)
    return sum(map(len, lines[: position[0] - 1])) + position[1]


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
    """Whether a case carrying exactly ``marks`` survives ``-m expression``.

    A case with no marks is still collected by a plain sweep. Pytest's default
    marker expression is ``not skip and not skipif``, which filters such a case
    when an inherited module skip is present.
    """
    if "skip" in marks:
        return False
    if not expression:
        return True
    return _evaluate(expression, marks)


def _names_target(relative: str, target: str) -> bool:
    """Whether a pytest target covers a file, by name or as a directory above it."""
    relative_parts = Path(relative).parts
    target_parts = Path(target).parts
    if target.startswith("/") and not relative.startswith("/"):
        target_parts = Path(target).resolve().relative_to(_REPO_ROOT).parts
    return relative_parts[: len(target_parts)] == target_parts


def _test_file_paths() -> set[str]:
    """Every integration test file, relative to the repository root."""
    return {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _INTEGRATION_DIR.rglob("test_*.py")
    }


def _test_path(relative: str) -> Path:
    """Resolve a relative test name, allowing focused tests to supply a fixture."""
    path = Path(relative)
    return path if path.is_absolute() else _REPO_ROOT / path


def _is_operator_case(relative: str, uses_conftest: bool) -> bool:
    """Whether the operator conftest is loaded for this case."""
    path = _test_path(relative)
    return (
        uses_conftest
        and path.parent.name == "ops"
        and path.parent.parent.name == "integration"
    )


class _ConftestItem:
    """The part of a pytest item that ops/conftest.py's gates read."""

    def __init__(self, mark_names: set[str]):
        self.marks = {name: "mark" for name in mark_names}
        self._skip_reason = None

    def get_closest_marker(self, name: str):
        return self.marks.get(name)

    def add_marker(self, marker):
        if isinstance(marker, str) and _MARK_RE.fullmatch(marker):
            name = _MARK_RE.fullmatch(marker)["name"]
        elif isinstance(marker, dict):
            name = "skip"
            reason = marker.get("reason", "")
            match = re.search(r"requires @(\w+) backend", reason)
            if match:
                self._skip_reason = match.group(1)
            elif "TileOPs kernels" in reason:
                self._skip_reason = "tileops"
            elif "software low-precision" in reason:
                self._skip_reason = "soft_lowp"
            elif "FlagGems C++ kernels" in reason:
                self._skip_reason = "flaggems_cpp"
            else:
                self._skip_reason = "unconditional"
        else:
            reason = re.search(r'reason=(?:"([^"]*)"|\'([^\']*)\')', marker)
            reason = next((group for group in reason.groups() if group), "")
            match = re.search(r"requires @(\w+) backend", reason)
            if match:
                self._skip_reason = match.group(1)
            elif "TileOPs kernels" in reason:
                self._skip_reason = "tileops"
            elif "software low-precision" in reason:
                self._skip_reason = "soft_lowp"
            elif "FlagGems C++ kernels" in reason:
                self._skip_reason = "flaggems_cpp"
            else:
                self._skip_reason = "unconditional"
        self.marks[name] = marker


class _ConftestConfig:
    """The collector hook reads no configuration in the modeled path."""

    def addinivalue_line(self, name: str, value: str) -> None:
        """Accept the marker registrations without building a pytest session."""


def _conftest_outcome(relative: str, marks: set[str], platform: str) -> str | None:
    """The skip the operator conftest applies, or None when the case survives."""
    build = _lane_build(platform)
    item = _ConftestItem(marks)
    with mock.patch("pytest.mark.skip", side_effect=lambda **kwargs: kwargs):
        _OPS_CONFTEST.pytest_configure(_ConftestConfig())
        with mock.patch.object(
            _OPS_CONFTEST._platform, "detect_platform", return_value=platform
        ), mock.patch.object(
            _OPS_CONFTEST,
            "_flaggems_cpp_enabled",
            return_value="flaggems_cpp" in build,
        ), mock.patch.object(
            _OPS_CONFTEST, "_tileops_enabled", return_value="tileops" in build
        ), mock.patch.object(
            _OPS_CONFTEST, "_boxing_enabled", return_value="boxing" in build
        ):
            _OPS_CONFTEST.pytest_collection_modifyitems(_ConftestConfig(), [item])
    return item._skip_reason


def _conftest_would_skip(relative: str, marks: set[str], platform: str) -> bool:
    """Apply the operator conftest's build gates to one CI case."""
    return bool(_conftest_outcome(relative, marks, platform))


def _lane_build(platform: str) -> set[str]:
    """The kernel set a platform's CI setup hook builds."""
    config = yaml.safe_load(
        (_PLATFORM_CONFIG_DIR / f"{platform}.yml").read_text(encoding="utf-8")
    )["environment_check"]
    required = set(config.get("require_build_kernels") or [])
    forbidden = set(config.get("forbid_build_kernels") or [])
    hook = (_REPO_ROOT / ".github/scripts/hooks" / f"set_env_{platform}.sh").read_text()
    configured = set(required)
    for name, pattern in (
        ("boxing", r'^export FLAGOS_BUILD_BOXING=["\']?(?P<value>[01])["\']?\s*$'),
        (
            "flaggems_cpp",
            r'^export FLAGOS_BUILD_FLAGGEMS_CPP=["\']?(?P<value>[01])["\']?\s*$',
        ),
    ):
        match = re.search(pattern, hook, re.MULTILINE)
        if match:
            if match["value"] == "1" and name not in forbidden:
                configured.add(name)
            else:
                configured.discard(name)
    return configured - forbidden


def _reachable_test_files() -> set[str]:
    """Every integration test file remaining after selection and conftest gates.

    The operator conftest's actual hook applies the platform, boxing, FlagGems
    C++ and TileOPs skips to the marks the source declares.
    """
    cases_by_file = _module_cases()
    platform_selections = list(_platform_selections())
    workflow_selections = _workflow_selections()
    return {
        relative
        for relative, cases in cases_by_file.items()
        if any(
            _selection_can_run(relative, case_marks, selection, platform)
            for platform, selection in platform_selections
            if selection.paths
            for case_marks in cases
        )
        or any(
            _selection_can_run(relative, case_marks, selection, None)
            for _, selection in workflow_selections
            if selection.paths
            for case_marks in cases
        )
    }


def _platform_selections():
    """Every pytest invocation and the platform whose wheel it uses."""
    platform_selections = {
        platform: _manifest_selections(platform) for platform in _manifest_platforms()
    }
    for platform, selections in platform_selections.items():
        for selection in selections:
            yield platform, selection


def _module_cases() -> dict[str, list[set[str]]]:
    """The real test-case mark sets in every integration test file."""
    return {
        relative: _source_mark_sets(_REPO_ROOT / relative)
        for relative in _test_file_paths()
    }


def _selection_can_run(
    relative: str,
    case_marks: set[str],
    selection: _Selection,
    platform: str | None,
) -> bool:
    """Whether one source-declared case gets past pytest and the ops conftest.

    An operator case with no marker at all is still selected by a plain sweep.
    """
    if any(_names_target(relative, skip) for skip in selection.ignored):
        return False
    if not any(_names_target(relative, path) for path in selection.paths):
        return False
    if selection.marker is not None and not _selects(selection.marker, case_marks):
        return False
    if not selection.uses_ops_conftest:
        # A source-only job opts out of the integration conftests, including
        # module-level skips that they would otherwise apply.
        return selection.marker is None or _evaluate(selection.marker, case_marks)
    if not _is_operator_case(relative, selection.uses_ops_conftest):
        return True
    if platform is None:
        return False
    reason = _conftest_outcome(relative, set(case_marks), platform)
    return reason is None


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
    "tests/integration/ops/test_flaggems_cpp_dispatch.py": (
        "carries only flaggems_cpp, and every set_env_*.sh builds FlagGems "
        "without its C++ path (FLAGOS_BUILD_FLAGGEMS_CPP=0). Accepted "
        "exclusion until one job opts into that build"
    ),
    "tests/integration/ops/test_tileops_generated.py": (
        "no platform config requires the TileOPs kernel set, so its "
        "module-level gate skips every CI case; the file also requires SM90 "
        "at runtime. Accepted exclusion until a TileOPs build job is added"
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
    shared = {
        path
        for path in _test_file_paths()
        if path.startswith("tests/integration/flaggems/")
    }
    assert shared, "the shared FlagGems cases vanished"
    assert shared <= _reachable_test_files(), (
        "a shared FlagGems case is not credited to its common runner: "
        f"{sorted(shared - _reachable_test_files())!r}"
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


def test_boxing_cases_reach_a_vendor_sweep_and_skip_a_default_cuda_wheel():
    """The shared vendor sweep uses the CUDA route wherever a wheel builds it.

    The CUDA lane builds a FlagGems-primary wheel without boxing, so its
    conftest skips ``cuda``. DCU builds boxing and runs the same vendor sweep.
    The same file must therefore not appear in the exclusion list.
    """
    relative = "tests/integration/ops/test_full_cuda_coverage.py"
    marks = _source_mark_sets(_REPO_ROOT / relative)
    selected = [
        selection
        for selection in _manifest_selections("dcu")
        if any(_names_target(relative, path) for path in selection.paths)
    ]
    assert selected
    assert any(
        _selection_can_run(relative, case_marks, selection, "dcu")
        for case_marks in marks
        for selection in selected
    )
    assert _conftest_would_skip(relative, {"cuda"}, "cuda")
    assert relative not in _KNOWN_UNREACHABLE


@pytest.mark.parametrize(
    "relative,platform",
    (
        ("tests/integration/ops/test_metax_complex_fill.py", "metax"),
        ("tests/integration/ops/test_dcu_complex.py", "dcu"),
    ),
)
def test_vendor_ops_files_are_gated_by_their_source_marker(relative, platform):
    """A runtime platform check cannot replace the marker the shared sweep uses.

    Both files run in the shared main-ops group and check their platform inside
    each case. The collection hook only sees source-declared marks, so a missing
    vendor mark leaves the case selected on every other platform.
    """
    marks = _source_mark_sets(_REPO_ROOT / relative)
    assert marks and all(platform in case_marks for case_marks in marks)
    assert all("main_ops" in case_marks for case_marks in marks), (
        f"{relative} left the shared main-ops selection"
    )
    assert platform in _OPS_CONFTEST._PLATFORM_SKIP_MARKERS["kunlunxin"]
    assert all(
        _conftest_outcome(relative, case_marks, "kunlunxin") == platform
        for case_marks in marks
    )
    assert all(
        _conftest_outcome(relative, case_marks, platform) is None
        for case_marks in marks
    )
    other = "cuda" if platform == "metax" else "metax"
    assert all(
        _conftest_outcome(relative, case_marks, other) == platform
        for case_marks in marks
    )


def test_unmarked_operator_file_still_reaches_the_directory_sweep(tmp_path):
    """A bare helper must not make every ordinary operator file disappear."""
    relative = "tests/integration/ops/test_unmarked_example.py"
    source = _REPO_ROOT / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "import pytest\n\ndef test_ordinary_operator():\n    assert True\n"
    )
    selection = _selection_of("python -m pytest tests/integration/ops/")
    assert selection is not None
    try:
        marks = _source_mark_sets(source)
        assert marks == [set()]
        with mock.patch.object(
            _OPS_CONFTEST._platform, "detect_platform", return_value="dcu"
        ), mock.patch.object(
            _OPS_CONFTEST, "_flaggems_cpp_enabled", return_value=False
        ), mock.patch.object(
            _OPS_CONFTEST, "_tileops_enabled", return_value=False
        ), mock.patch.object(_OPS_CONFTEST, "_boxing_enabled", return_value=True):
            item = _ConftestItem(marks[0])
            with mock.patch("pytest.mark.skip", side_effect=lambda **kwargs: kwargs):
                _OPS_CONFTEST.pytest_configure(_ConftestConfig())
                _OPS_CONFTEST.pytest_collection_modifyitems(None, [item])
        assert selection.marker is None
        assert item._skip_reason is None
    finally:
        source.unlink()


def test_source_check_needs_a_real_test_case_for_its_marker(tmp_path):
    """A planted assertion must not credit the file's operator cases."""
    source = tmp_path / "test_planted_marker.py"
    source.write_text(
        textwrap.dedent("""\
            import pytest

            def assert_this_mentions_cuda():
                pass

            assert_this_mentions_cuda()
            pytest.mark.cuda

            @pytest.mark.anyplatform
            def test_operator():
                pass
        """)
    )
    relative = (tmp_path / source.name).as_posix()
    selection = _Selection((relative,), "cuda", (), uses_ops_conftest=False)
    assert not _selection_can_run(relative, set(), selection, "cuda")


def test_source_check_is_reachable_without_the_operator_conftest():
    """The text-only check runs in a job that has no flagos runtime."""
    relative = "tests/integration/ops/test_flaggems_conf_consistency.py"
    check, *_ = _source_mark_sets(_REPO_ROOT / relative)
    assert "anyplatform" in check
    source_check = _Selection((relative,), None, (), uses_ops_conftest=False)
    assert _selection_can_run(relative, check, source_check, None)
    operator_case = _Selection((relative,), "anyplatform", ())
    assert not _selection_can_run(relative, set(), operator_case, "cuda")
