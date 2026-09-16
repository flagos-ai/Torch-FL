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
import signal
from pathlib import Path

import pytest


_RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / ".github/scripts/run_integration_tests.py"
)
_SPEC = importlib.util.spec_from_file_location("run_integration_tests", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


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


CLEAN_SUMMARY_COMMAND = "printf '2 passed in 0.01s\\n'; kill -SEGV $$"
OPT_IN = {"INTEGRATION_ALLOW_TEARDOWN_CRASH": "1"}


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


def test_teardown_crash_without_opt_in_still_fails(monkeypatch, tmp_path):
    tests = [{"name": "teardown-crash", "command": CLEAN_SUMMARY_COMMAND}]
    assert _run_main(monkeypatch, tmp_path, tests) == -signal.SIGSEGV


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
    assert _run_main(monkeypatch, tmp_path, tests) == 1


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
