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

"""Tests for the per-file unit test runner's result classification."""

import importlib.util
import signal
import sys
from pathlib import Path

import pytest


_RUNNER_PATH = Path(__file__).resolve().parents[2] / ".github/scripts/run_unit_tests.py"
_SPEC = importlib.util.spec_from_file_location("run_unit_tests", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


_CLEAN = "........................                                                 [100%]\n24 passed in 19.28s\n"
_WITH_FAILURE = "F.                                                                        [100%]\n1 failed, 1 passed in 3.10s\n"


@pytest.fixture(autouse=True)
def _default_teardown_policy(monkeypatch):
    """Pin the module's opt-in to its default for every test in this file.

    ``_ALLOW_TEARDOWN_CRASH`` is read from the environment at import time, and
    the CI step that runs this very file sets ``UNIT_TESTS_ALLOW_TEARDOWN_CRASH=1``
    (that is what makes the surrounding suite pass on containers whose RDMA stack
    faults during ``exit()``). Without this pin the assertions below would mean
    different things depending on how the runner was invoked -- which is how the
    signal-classification test failed in the first CI run of that step.
    """
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", False)


@pytest.fixture
def allow_teardown_crash(monkeypatch):
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", True)


def test_success_is_a_pass():
    assert _RUNNER.classify(0, _CLEAN) == ("PASS", False)


def test_pytest_reported_failure_is_a_failure():
    assert _RUNNER.classify(1, _WITH_FAILURE) == ("FAIL", False)


def test_summary_line_is_recognised_with_a_banner_and_without_one():
    # ``-q`` prints the bare counts; the default reporter wraps them in "=".
    assert _RUNNER.clean_summary("24 passed in 19.28s") == "24 passed in 19.28s"
    assert (
        _RUNNER.clean_summary("===== 24 passed, 1 skipped in 19.28s =====")
        == "24 passed, 1 skipped in 19.28s"
    )


def test_progress_lines_that_look_like_durations_are_not_the_summary():
    # A slow test prints its own "in Ns" only in the summary; a stack trace that
    # quotes "took 1.5s" must not be mistaken for a completed run.
    assert _RUNNER.clean_summary("Traceback: worker took 1.5s then died") is None


def test_crash_without_a_summary_is_never_tolerated(allow_teardown_crash):
    # SIGSEGV before pytest could report: nothing says the tests themselves ran,
    # so the opt-in for teardown crashes must not swallow it.
    assert (
        _RUNNER.classify(-signal.SIGSEGV, "Fatal Python error: Segmentation fault\n")[0]
        == "CRASH"
    )


def test_crash_after_a_clean_summary_is_tolerated_only_when_opted_in(
    monkeypatch, allow_teardown_crash
):
    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", False)
    assert _RUNNER.classify(-signal.SIGSEGV, _CLEAN)[0] == "CRASH"

    monkeypatch.setattr(_RUNNER, "_ALLOW_TEARDOWN_CRASH", True)
    assert _RUNNER.classify(-signal.SIGSEGV, _CLEAN) == (
        "PASS (crash after summary)",
        True,
    )


def test_subprocess_style_exit_codes_are_classified_like_signals():
    # subprocess reports a signal death as a negative return code, but a shell
    # wrapper reports 128 + signal; both spellings must reach the same verdict.
    assert _RUNNER.classify(128 + signal.SIGSEGV, _CLEAN)[0] == "CRASH"
    assert _RUNNER.classify(128 + signal.SIGABRT, _CLEAN)[0] == "CRASH"


def test_crash_after_a_failed_summary_stays_a_failure(allow_teardown_crash):
    # The tests did report, and what they reported was a failure; the crash that
    # followed must not launder it into a pass.
    assert _RUNNER.classify(-signal.SIGSEGV, _WITH_FAILURE)[0] == "CRASH"


def test_timeout_is_reported_as_such():
    assert _RUNNER.classify(124, _CLEAN) == ("TIMEOUT", False)


def test_tolerated_teardown_crash_does_not_fail_the_run(
    monkeypatch, tmp_path, allow_teardown_crash
):
    """A file forgiven under the opt-in must not then be counted as a failure.

    ``main()`` used to build its failure list from the status string alone, so
    every tolerated file was reported as a ``::warning::`` and the run still
    exited 1 -- an opt-in that bought nothing. The CI step exits on that code.
    """
    (tmp_path / "test_one.py").write_text("")
    monkeypatch.setattr(
        _RUNNER, "run_file", lambda *args, **kwargs: (-signal.SIGSEGV, _CLEAN, False)
    )
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 0


def test_a_real_failure_still_fails_the_run(monkeypatch, tmp_path):
    (tmp_path / "test_one.py").write_text("")
    monkeypatch.setattr(
        _RUNNER, "run_file", lambda *args, **kwargs: (1, _WITH_FAILURE, False)
    )
    monkeypatch.setattr(sys, "argv", ["run_unit_tests.py", "--test-dir", str(tmp_path)])
    assert _RUNNER.main() == 1
