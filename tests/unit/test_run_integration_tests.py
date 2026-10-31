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

"""Tests for the data-driven integration test runner."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / ".github/scripts/run_integration_tests.py"
)
_SPEC = importlib.util.spec_from_file_location("run_integration_tests", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


@pytest.mark.parametrize("mismatch", [None, "source_sha", "run_id"])
def test_plan_identity_with_different_checkout_owner(tmp_path, mismatch):
    source = tmp_path / "source"
    script = source / ".github/scripts/run_integration_tests.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(_RUNNER_PATH.read_bytes())
    global_config = tmp_path / "gitconfig"
    global_config.write_text("")
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(global_config),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    environment.pop("GIT_TEST_ASSUME_DIFFERENT_OWNER", None)
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Test checkout",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(source), *args],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    git_head = ["git", "-C", str(source), "rev-parse", "HEAD"]
    sha = subprocess.check_output(git_head, env=environment, text=True).strip()
    context = {"source_sha": sha, "run_id": "42"}
    if mismatch:
        context[mismatch] = "different"
    workdir = tmp_path / "work"
    workdir.mkdir()
    reports = tmp_path / "reports"
    environment.update(
        GIT_TEST_ASSUME_DIFFERENT_OWNER="1",
        CI_PLAN_CONTEXT=json.dumps(context),
        GITHUB_RUN_ID="42",
        GITHUB_RUN_ATTEMPT="1",
        REPORT_PLATFORM="nvidia",
        INTEGRATION_TESTS=json.dumps(
            [{"name": "probe", "command": "printf checked > executed"}]
        ),
        INTEGRATION_ENVIRONMENT="{}",
        INTEGRATION_WORKDIR=str(workdir),
        FLAGCICD_REPORT_DIR=str(reports),
    )
    rejected = subprocess.run(
        git_head, env=environment, capture_output=True, text=True, timeout=15
    )
    assert rejected.returncode == 128
    assert "dubious ownership" in rejected.stderr

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=workdir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if mismatch:
        assert result.returncode != 0
        assert "execution source or run differs from the plan" in result.stderr
        assert not (workdir / "executed").exists()
        assert not reports.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (workdir / "executed").read_text() == "checked"
        report = json.loads((reports / "groups.json").read_text())
        assert report["context"] == context
        assert report["groups"][0]["status"] == "passed"

    # The exception must not persist beyond the runner's single Git invocation.
    assert global_config.read_text() == ""
    rejected = subprocess.run(
        git_head, env=environment, capture_output=True, text=True, timeout=15
    )
    assert rejected.returncode == 128
    assert "dubious ownership" in rejected.stderr


def test_per_test_environment_is_expanded_and_scoped(monkeypatch, tmp_path):
    tests = [
        {
            "name": "profiled",
            "environment": {"LD_PRELOAD": "${ASCEND_MSPTI_PRELOAD}"},
            "command": (
                "printf '%s' \"${LD_PRELOAD-<unset>}\" > profiled.txt; "
                "printf '%s' \"$SHARED\" > shared.txt"
            ),
        },
        {
            "name": "ordinary",
            "command": "printf '%s' \"${LD_PRELOAD-<unset>}\" > ordinary.txt",
        },
    ]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", json.dumps({"SHARED": "value"}))
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setenv("ASCEND_MSPTI_PRELOAD", "/opt/cann/libmspti.so:/opt/base.so")
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    class Args:
        validate_only = False
        allow_empty = False

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())

    assert _RUNNER.main() == 0
    assert (tmp_path / "profiled.txt").read_text() == (
        "/opt/cann/libmspti.so:/opt/base.so"
    )
    assert (tmp_path / "ordinary.txt").read_text() == "<unset>"
    assert (tmp_path / "shared.txt").read_text() == "value"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ([], "must be an object"),
        ({"NOT-VALID": "value"}, "invalid environment variable name"),
        ({"VALID": "line one\nline two"}, "must be a single-line string"),
    ],
)
def test_per_test_environment_validation(monkeypatch, environment, message):
    tests = [{"name": "test", "command": "true", "environment": environment}]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")

    with pytest.raises(SystemExit, match=message):
        _RUNNER.load_configuration(allow_empty=False)


def _run_main(monkeypatch, tmp_path, tests):
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))

    class Args:
        validate_only = False
        allow_empty = False

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())
    return _RUNNER.main()


def test_unset_variable_is_named_instead_of_bash_unbound(monkeypatch, tmp_path):
    """A group whose command expands an unset variable says so, and stops.

    thead's Qwen3 groups run ``--model "$MODEL_PATH"`` while its
    integration_environment is empty. Under ``set -u`` that used to surface as
    ``bash: line 2: MODEL_PATH: unbound variable`` with no mention of the group
    or the configuration, which reads as a test failure.
    """
    tests = [
        {
            "name": "Run inference tests",
            "command": ('printf "$MODEL_PATH" > model.txt && touch should_not_run'),
        }
    ]
    monkeypatch.delenv("MODEL_PATH", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        _run_main(monkeypatch, tmp_path, tests)

    assert exit_info.value.code != 0
    assert not (tmp_path / "model.txt").exists()
    assert not (tmp_path / "should_not_run").exists()


def test_unset_variable_detection_ignores_supplied_and_builtin_parameters():
    assert _RUNNER._unset_variables(
        'python -m pytest x.py --model "$MODEL_PATH" -v', {}
    ) == ["MODEL_PATH"]
    assert _RUNNER._unset_variables(
        'cd "$GITHUB_WORKSPACE" && python ${HOME}/a.py $$ $?',
        {"GITHUB_WORKSPACE": "/w"},
    ) == ["HOME"]


CLEAN_SUMMARY_COMMAND = "printf '2 passed in 0.01s\\n'; kill -SEGV $$"
BANNERED_CLEAN_SUMMARY_COMMAND = (
    "printf '===== 2 passed, 1 skipped in 0.01s =====\\n'; kill -SEGV $$"
)
OPT_IN = {"INTEGRATION_ALLOW_TEARDOWN_CRASH": "1"}


def test_summary_line_is_recognised_with_a_banner_and_without_one():
    assert _RUNNER.clean_summary("2 passed in 0.01s") == "2 passed in 0.01s"
    assert (
        _RUNNER.clean_summary("===== 2 passed, 1 skipped in 0.01s =====")
        == "2 passed, 1 skipped in 0.01s"
    )
    assert _RUNNER.clean_summary("2 passed, 1 xfailed in 0.01s") is None


def test_missing_mpirun_is_caught_before_the_group_starts(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    tests = [
        {"name": "mpi", "command": "mpirun -np 2 python -m pytest"},
        {"name": "next", "command": "touch should_not_run"},
    ]

    with pytest.raises(SystemExit, match="mpirun is required"):
        _run_main(monkeypatch, tmp_path, tests)

    assert not (tmp_path / "should_not_run").exists()


def test_teardown_crash_after_clean_summary_is_tolerated_when_opted_in(
    monkeypatch, tmp_path
):
    """A fatal signal after pytest's clean summary passes with the opt-in.

    Mirrors the Ascend profiler group: the suite reports green, then a vendor
    destructor faults during interpreter teardown.
    """
    tests = [
        {
            "name": "teardown-crash",
            "command": CLEAN_SUMMARY_COMMAND,
            "environment": dict(OPT_IN),
        }
    ]
    assert _run_main(monkeypatch, tmp_path, tests) == 0


def test_teardown_crash_after_a_bannered_summary_is_tolerated(monkeypatch, tmp_path):
    tests = [
        {
            "name": "bannered-teardown-crash",
            "command": BANNERED_CLEAN_SUMMARY_COMMAND,
            "environment": dict(OPT_IN),
        }
    ]
    assert _run_main(monkeypatch, tmp_path, tests) == 0


def test_teardown_crash_without_opt_in_still_fails(monkeypatch, tmp_path):
    tests = [{"name": "teardown-crash", "command": CLEAN_SUMMARY_COMMAND}]
    with pytest.raises(SystemExit) as error:
        _run_main(monkeypatch, tmp_path, tests)
    assert error.value.code == -signal.SIGSEGV


def test_ordinary_failure_stops_the_run(monkeypatch, tmp_path):
    tests = [
        {"name": "first", "command": "exit 5"},
        {"name": "second", "command": "touch should_not_run"},
    ]

    with pytest.raises(SystemExit) as error:
        _run_main(monkeypatch, tmp_path, tests)

    assert error.value.code == 5
    assert not (tmp_path / "should_not_run").exists()


def test_failure_summary_is_not_tolerated_even_when_opted_in(monkeypatch, tmp_path):
    """A summary reporting failures fails with or without the opt-in: only a
    *clean* summary proves the tests passed before the signal."""
    tests = [
        {
            "name": "failed-suite",
            "command": "printf '1 failed, 1 passed in 0.02s\\n'; exit 1",
            "environment": dict(OPT_IN),
        }
    ]
    with pytest.raises(SystemExit) as error:
        _run_main(monkeypatch, tmp_path, tests)
    assert error.value.code == 1


@pytest.mark.parametrize(
    ("returncode", "output", "allow", "expected"),
    [
        (0, "", False, (True, False)),
        (-signal.SIGSEGV, "2 passed in 0.01s\n", True, (True, True)),
        (-signal.SIGSEGV, "2 passed in 0.01s\n", False, (False, False)),
        # Signal before any summary: nothing proved the tests passed.
        (-signal.SIGSEGV, "some traceback\n", True, (False, False)),
        # Plain nonzero exit, even with a clean summary, is a failure: only
        # fatal *signals* are tolerable, never exit statuses.
        (1, "2 passed in 0.01s\n", True, (False, False)),
        # A runner kill (SIGKILL, e.g. OOM or job timeout) must never be
        # forgiven: it is not a teardown destructor fault.
        (-signal.SIGKILL, "2 passed in 0.01s\n", True, (False, False)),
        # Shell-encoded signal (128+N) reads the same as negative-N.
        (128 + signal.SIGSEGV, "2 passed in 0.01s\n", True, (True, True)),
    ],
)
def test_classify_command_result(returncode, output, allow, expected):
    assert _RUNNER.classify_command_result(returncode, output, allow) == expected
