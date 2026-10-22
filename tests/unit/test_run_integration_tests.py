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


def test_unit_group_runs_from_the_checkout_instead_of_the_wheel_workspace(
    monkeypatch, tmp_path
):
    """The source-tree suite must not resolve the checkout's package copy."""
    checkout = tmp_path / "checkout"
    workspace = tmp_path / "wheel-only"
    checkout.mkdir()
    workspace.mkdir()
    tests = [{"id": "unit_tests_per_file", "name": "unit", "command": "pwd -P"}]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(workspace))
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))
    monkeypatch.setattr(_RUNNER, "check_inventory_binding", lambda *_args: None)

    captured = []

    def record_run(command, _environment, workdir, _prefix=""):
        captured.append(workdir)
        return 0, "1 passed in 0.01s\n"

    monkeypatch.setattr(_RUNNER, "run_command", record_run)

    class Args:
        validate_only = False
        allow_empty = False
        inventory = []

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())

    assert _RUNNER.main() == 0
    assert [os.path.realpath(workdir) for workdir in captured] == [
        os.path.realpath(checkout)
    ]


def test_unit_group_fails_if_the_checkout_does_not_exist(monkeypatch, tmp_path):
    """A fallback to the wheel workspace would hide the source-tree failure."""
    missing = tmp_path / "missing-checkout"
    tests = [{"id": "unit_tests_per_file", "name": "unit", "command": "true"}]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_WORKSPACE", str(missing))

    class Args:
        validate_only = False
        allow_empty = False
        inventory = []

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())

    with pytest.raises(SystemExit, match="GITHUB_WORKSPACE does not exist"):
        _RUNNER.main()


def test_flaggems_group_receives_the_checkout_registration_source(
    monkeypatch, tmp_path
):
    """The routing contract must inspect the checkout, not the wheel workspace."""
    checkout = tmp_path / "checkout"
    workspace = tmp_path / "wheel-only"
    reports = tmp_path / "reports"
    for directory in (checkout, workspace, reports):
        directory.mkdir()
    python = tmp_path / "python"
    python.symlink_to(sys.executable)
    tests = [
        {
            "id": "flaggems_conformance",
            "name": "flaggems",
            "command": "true",
            "environment": {},
        }
    ]
    monkeypatch.setenv("INTEGRATION_TESTS", json.dumps(tests))
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(workspace))
    monkeypatch.setenv("INTEGRATION_INVENTORY_ROOT", str(checkout))
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(reports))
    monkeypatch.setenv("TORCH_FL_SOURCE_ROOT", str(workspace))
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(_RUNNER, "check_inventory_binding", lambda *_args: None)
    monkeypatch.setattr(
        _RUNNER,
        "load_inventory",
        lambda _paths: {"flaggems_conformance": {"cases": ["case"], "aliases": {}}},
    )
    monkeypatch.setattr(_RUNNER, "safe_report_directory", lambda _name: str(reports))
    monkeypatch.setattr(
        _RUNNER,
        "inventory_environment",
        lambda _test_id, _inventory: {"FLAGCICD_EXPECTED_CASES": "{}"},
    )
    monkeypatch.setattr(
        _RUNNER, "report_cases", lambda *_args, **_kwargs: {"case": "passed"}
    )
    monkeypatch.setattr(_RUNNER, "verify_inventory", lambda *_args: None)

    captured = []

    def record_run(command, environment, workdir, prefix=""):
        captured.append((command, environment.copy(), workdir, prefix))
        return 0, "1 passed in 0.01s\n"

    monkeypatch.setattr(_RUNNER, "run_command", record_run)

    class Args:
        validate_only = False
        allow_empty = False
        inventory = []

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())

    assert _RUNNER.main() == 0
    _, environment, workdir, prefix = captured[0]
    assert environment["TORCH_FL_SOURCE_ROOT"] == str(checkout)
    assert "export FLAGCICD_LAUNCHER=group" in prefix
    assert workdir == str(checkout)


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


def test_a_failed_environment_preflight_is_retained_as_incomplete(
    monkeypatch, tmp_path
):
    """The shared merger names an environment-preparation failure."""
    monkeypatch.setenv("PATH", str(tmp_path))
    python = tmp_path / "python"
    python.symlink_to(sys.executable)
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(reports))
    tests = [{"name": "mpi", "command": "mpirun -np 2 python -m pytest"}]

    with pytest.raises(SystemExit, match="mpirun is required"):
        _run_main(monkeypatch, tmp_path, tests)

    markers = list(reports.glob("attempt-*/environment-*.pending"))
    assert len(markers) == 1
    marker = json.loads(markers[0].read_text())
    assert marker["group"] == "mpi"
    assert "environment preparation failed" in marker["error"]


def test_a_group_command_error_preserves_earlier_sessions(monkeypatch, tmp_path):
    """A non-pytest setup failure keeps the groups that already ran."""
    reports = tmp_path / "reports"
    reports.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    session = json.dumps(
        {
            "exitcode": 0,
            "duration": 0.1,
            "summary": {"passed": 1},
            "tests": [
                {
                    "nodeid": "shared/test_first.py::test_pass",
                    "outcome": "passed",
                    "setup": {"outcome": "passed"},
                    "call": {"outcome": "passed"},
                    "teardown": {"outcome": "passed"},
                }
            ],
        }
    )
    monkeypatch.setenv(
        "INTEGRATION_TESTS",
        json.dumps(
            [
                {
                    "name": "first",
                    "command": (
                        f"cat > {reports}/pytest-first.json <<'JSON'\n"
                        + session
                        + "\nJSON\n"
                        f"printf '{{}}' > {reports}/pytest-first.pending"
                    ),
                },
                {"name": "second", "command": "exit -1"},
            ]
        ),
    )
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    monkeypatch.setenv("INTEGRATION_WORKDIR", str(workdir))
    monkeypatch.setenv("FLAGCICD_REPORT_DIR", str(reports))
    python = tmp_path / "python"
    python.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setattr(_RUNNER, "safe_report_directory", lambda _name: str(reports))

    class Args:
        validate_only = False
        allow_empty = False

    monkeypatch.setattr(_RUNNER, "parse_args", lambda: Args())

    with pytest.raises(SystemExit, match="255"):
        _RUNNER.main()
    assert (reports / "pytest-first.json").is_file()
    assert (reports / "pytest-first.pending").is_file()
    assert not list(reports.glob("environment-*.pending"))


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


def _inventory(tmp_path, cases):
    directory = tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "google-test.json").write_text(json.dumps(cases))
    return directory


def _args(inventory=()):
    class Args:
        validate_only = False
        allow_empty = False
        inventory = ()

    Args.inventory = list(inventory)
    return Args


def test_inventory_requires_every_named_case_to_pass_without_skips(
    monkeypatch,
):
    cases = {"tests/unit/test_file.py::test_first": "passed"}
    assert _RUNNER.verify_inventory("shared", cases, list(cases)) is None
    for actual, case in (
        ({}, "missing"),
        ({**cases, "tests/unit/test_file.py::test_extra": "passed"}, "extra"),
        ({**cases, "tests/unit/test_file.py::test_first": "skipped"}, "not passed"),
    ):
        with pytest.raises(SystemExit, match=case):
            _RUNNER.verify_inventory("shared", actual, list(cases))


def test_unit_file_report_reports_a_child_xpass_as_a_file_failure(tmp_path):
    """A non-strict xfail that starts passing must fail the unit inventory."""
    case = "tests/unit/test_one.py"
    marker = tmp_path / "unit-test_one.unit"
    marker.write_text(json.dumps({"unit_file": case}))
    (tmp_path / "unit-test_one.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": case + "::test_one",
                        "outcome": "xpassed",
                        "setup": {"outcome": "passed"},
                        "call": {"outcome": "passed"},
                        "teardown": {"outcome": "passed"},
                    }
                ],
                "summary": {"passed": 0, "xfailed": 0},
            }
        )
    )

    cases = _RUNNER.report_cases(
        tmp_path,
        allow_unit_files=True,
        expected={"cases": [case], "aliases": {}},
    )

    assert cases == {case: "failed"}


def test_inventory_rejects_duplicate_or_incomplete_sessions(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "first.pending").write_text(
        json.dumps({"group": "shared", "args": []})
    )
    (directory / "second.pending").write_text(
        json.dumps({"group": "shared", "args": []})
    )
    (directory / "first.json").write_text(
        json.dumps(
            {"exitcode": 0, "duration": 0.1, "summary": {"passed": 1}, "tests": []}
        )
    )
    with pytest.raises(SystemExit, match="incomplete"):
        _RUNNER.report_cases(directory)


def test_inventory_paths_are_loaded_by_group(tmp_path, monkeypatch):
    _inventory(
        tmp_path / "unit",
        {
            "cases": ["tests/unit/test_file.py"],
            "aliases": {},
            "group": "unit_tests_per_file",
        },
    )
    _inventory(
        tmp_path / "sdpa", {"cases": ["test_shared.py::test_one"], "aliases": {}}
    )
    monkeypatch.setenv("INTEGRATION_INVENTORY_ROOT", str(tmp_path))
    monkeypatch.setenv("INTEGRATION_SOURCE_ROOT", str(tmp_path))
    unit_file = tmp_path / "tests/unit/test_file.py"
    unit_file.parent.mkdir(parents=True)
    unit_file.write_text("")
    assert _RUNNER.load_inventory(["sdpa_lanes=sdpa", "unit_tests_per_file=unit"]) == {
        "sdpa_lanes": {
            "cases": ["test_shared.py::test_one"],
            "aliases": {},
        },
        "unit_tests_per_file": {
            "cases": ["tests/unit/test_file.py"],
            "aliases": {},
            "group": "unit_tests_per_file",
        },
    }


def test_source_directories_are_derived_through_the_runner_cli(monkeypatch):
    """The loader's paths are test trees; only materialized output is JSON."""
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("INTEGRATION_SOURCE_ROOT", str(root))
    tests = [
        {"id": group, "name": group, "command": "true"}
        for group in _RUNNER.INVENTORY_GROUPS
    ]
    assignments = [
        value
        for group, source in _RUNNER.INVENTORY_SOURCES.items()
        for value in ("--inventory", f"{group}={source}")
    ]
    result = subprocess.run(
        [sys.executable, str(_RUNNER_PATH), "--validate-only", *assignments],
        env={
            **os.environ,
            "INTEGRATION_TESTS": json.dumps(tests),
            "INTEGRATION_ENVIRONMENT": "{}",
            "INTEGRATION_SOURCE_ROOT": str(root),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Validated 4 configured integration test(s)" in result.stdout


def test_source_inventory_paths_are_limited_to_declared_directories(
    tmp_path, monkeypatch
):
    """A relative report path must not become a source-tree escape hatch."""
    report = tmp_path / "reports/google-test.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"cases": ["case"], "aliases": {}}))
    assert not _RUNNER.is_source_inventory("sdpa_lanes", "reports")
    monkeypatch.setenv("INTEGRATION_INVENTORY_ROOT", str(tmp_path))
    assert _RUNNER.load_inventory(["sdpa_lanes=reports"])["sdpa_lanes"] == {
        "cases": ["case"],
        "aliases": {},
    }


def test_native_pytest_alias_satisfies_an_explicit_inventory_case():
    explicit = "tests/test_identity.py::test_device[device]"
    native = "tests/test_identity.py::test_device[flagos]"
    document = {"cases": [explicit], "aliases": {explicit: [native]}}
    assert _RUNNER.verify_inventory("shared", {native: "passed"}, document) is None
    assert (
        _RUNNER.verify_inventory(
            "shared", {explicit: "passed", native: "passed"}, document
        )
        is None
    )
    for actual, message in (
        ({native: "skipped"}, "not passed"),
        ({native: "failed"}, "not passed"),
        ({native: "passed", explicit: "skipped"}, "not passed"),
        ({}, "missing"),
    ):
        with pytest.raises(SystemExit, match=message):
            _RUNNER.verify_inventory("shared", actual, document)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"cases": ["case"], "aliases": {"other": ["native"]}}, "undeclared"),
        ({"cases": ["case"], "aliases": {"case": []}}, "non-empty"),
        ({"cases": ["case"], "aliases": {"case": ["native", "native"]}}, "unique"),
        (
            {"cases": ["case", "native"], "aliases": {"case": ["native"]}},
            "collides",
        ),
        ({"cases": ["case"]}, "exactly"),
        ({"cases": ["case"], "aliases": {}, "extra": []}, "exactly"),
        ({"cases": [], "aliases": {}}, "non-empty"),
        ({"cases": ["case", "case"], "aliases": {}}, "unique"),
        ({"cases": ["case"], "aliases": []}, "object"),
        ("case", "list or an object"),
    ],
)
def test_inventory_documents_reject_ambiguous_aliases(tmp_path, document, message):
    path = tmp_path / "google-test.json"
    path.write_text(json.dumps(document))
    with pytest.raises(SystemExit, match=message):
        _RUNNER.load_inventory([f"sdpa_lanes={tmp_path}"])


def test_validation_refuses_fixed_case_counts(monkeypatch):
    monkeypatch.setenv(
        "INTEGRATION_TESTS",
        json.dumps(
            [
                {
                    "id": "shared",
                    "name": "shared",
                    "command": "true",
                    "requires": {"passed": 1, "skipped": 0},
                }
            ]
        ),
    )
    monkeypatch.setenv("INTEGRATION_ENVIRONMENT", "{}")
    with pytest.raises(SystemExit, match="fixed count"):
        _RUNNER.load_configuration(allow_empty=False)


def test_inventory_must_belong_to_a_group_in_the_plan():
    tests = [{"id": "unit_tests_per_file", "name": "unit", "command": "true"}]
    _RUNNER.check_inventory_binding(
        tests, {"unit_tests_per_file": ["tests/unit/test_file.py"]}
    )
    with pytest.raises(SystemExit, match="flaggems_conformance"):
        _RUNNER.check_inventory_binding(
            tests,
            {
                "unit_tests_per_file": ["tests/unit/test_file.py"],
                "flaggems_conformance": [
                    "tests/integration/flaggems/test_flaggems_route_table.py::test_route"
                ],
            },
        )
