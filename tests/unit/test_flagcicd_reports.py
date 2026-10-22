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

"""Exercise real pytest subprocesses, the wheel-only cwd, and report merging."""

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
SPEC = importlib.util.spec_from_file_location(
    "merge_reports", SCRIPTS / "merge_pytest_reports.py"
)
MERGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGER)
PYTHON = shlex.quote(sys.executable)


def _environment(directory):
    env = os.environ.copy()
    for key in ("PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_PLUGINS"):
        env.pop(key, None)
    for key in ("FLAGCICD_LAUNCHER", "FLAGCICD_REPORT_DIR", "FLAGCICD_TEST_GROUP"):
        env.pop(key, None)
    env.update(
        FLAGCICD_REPORT_DIR=str(directory),
        FLAGCICD_TEST_GROUP="probe",
        FLAGCICD_PYTHON_EXECUTABLE=sys.executable,
        PYTHONPATH=str(SCRIPTS / "pytest_reporting"),
        PYTEST_PLUGINS="flagcicd_report",
    )
    env["FLAGCICD_LAUNCHER"] = "group"
    return env


def _integration(tmp_path, groups, inventory=None, checkout=None):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir(exist_ok=True)
    reports = tmp_path / "reports"
    env = _environment(reports)
    env.update(
        INTEGRATION_TESTS=json.dumps(groups),
        INTEGRATION_ENVIRONMENT="{}",
        INTEGRATION_WORKDIR=str(workdir),
    )
    if inventory:
        env["INTEGRATION_INVENTORY_ROOT"] = str(tmp_path)
    if checkout:
        env.update(
            INTEGRATION_SOURCE_ROOT=str(checkout),
            GITHUB_WORKSPACE=str(checkout),
        )
    inventory_arguments = [
        argument
        for group, cases in (inventory or {}).items()
        for argument in ("--inventory", f"{group}={cases}")
    ]
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_integration_tests.py"),
            *inventory_arguments,
        ],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return process, reports


@pytest.mark.parametrize(
    ("case_source", "command_suffix", "expected_message"),
    [
        ("def test_shared(): pass\n", "", None),
        ("def test_other(): pass\n", "", "missing"),
        (
            "def test_shared(): pass\ndef test_extra(): pass\n",
            "",
            "extra",
        ),
        (
            "import pytest\n"
            "@pytest.mark.skip(reason='not available')\n"
            "def test_shared(): pass\n",
            "",
            "not passed",
        ),
    ],
    ids=["complete", "missing", "extra", "skipped"],
)
def test_inventory_verdict_uses_a_real_pytest_report(
    tmp_path, case_source, command_suffix, expected_message
):
    """Missing, extra and skipped cases make an otherwise clean run fail."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "test_cases.py").write_text(case_source)
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "google-test.json").write_text(
        json.dumps(["test_cases.py::test_shared"])
    )
    process, _ = _integration(
        tmp_path,
        [
            {
                "id": "sdpa_lanes",
                "name": "shared",
                "command": f"{PYTHON} -m pytest test_cases.py -q{command_suffix}",
            }
        ],
        inventory={"sdpa_lanes": "inventory"},
        checkout=checkout,
    )
    if expected_message:
        assert process.returncode != 0
        assert expected_message in process.stdout + process.stderr
    else:
        assert process.returncode == 0, process.stdout + process.stderr
        assert "Inventory met: 1 shared cases" in process.stdout


def test_complete_inventory_does_not_hide_a_command_failure(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "test_cases.py").write_text("def test_shared(): pass\n")
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "google-test.json").write_text(
        json.dumps(["test_cases.py::test_shared"])
    )
    command = f"{PYTHON} -m pytest test_cases.py -q; status=$?; exit $((status + 4))"
    process, _ = _integration(
        tmp_path,
        [{"id": "sdpa_lanes", "name": "shared", "command": command}],
        inventory={"sdpa_lanes": "inventory"},
        checkout=checkout,
    )
    assert process.returncode == 4, process.stdout + process.stderr
    assert "Inventory met: 1 shared cases" in process.stdout


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def test_one(): pass\ndef test_two(): pass\n", True),
        ("def test_one(): pass\ndef test_two(): assert False\n", False),
        (
            "import pytest\n"
            "def test_one(): pass\n"
            "@pytest.mark.skip(reason='not available')\n"
            "def test_two(): pass\n",
            False,
        ),
    ],
    ids=["pass", "failure", "skip"],
)
def test_per_file_unit_report_has_one_verdict(tmp_path, source, expected):
    """The source checkout's files map to the same case the loader validated."""
    checkout = tmp_path / "checkout"
    test_file = checkout / "tests/unit/test_cases.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(source)
    case = "tests/unit/test_cases.py"
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "google-test.json").write_text(
        json.dumps(
            {
                "cases": [case],
                "aliases": {},
                "group": "unit_tests_per_file",
            }
        )
    )
    run_dir = tmp_path / "unit-reports"
    run_dir.mkdir()
    report = run_dir / "unit-one.json"
    pytest_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "--json-report",
            f"--json-report-file={report}",
        ],
        env=os.environ
        | {
            "PYTHONPATH": str(SCRIPTS / "pytest_reporting"),
            "PYTEST_PLUGINS": "flagcicd_report",
        },
        cwd=checkout,
        text=True,
        capture_output=True,
        timeout=30,
    )
    # pytest itself succeeds when a file only contains a skip. The inventory
    # runner has to reject that report, so this check must not expect exit 1.
    if expected or source.endswith("test_two(): pass\n"):
        assert pytest_process.returncode == 0, pytest_process.stdout
    else:
        assert pytest_process.returncode != 0, pytest_process.stdout
    report.with_suffix(".unit").write_text(json.dumps({"args": [str(test_file)]}))

    inventory_run = tmp_path / "inventory-run"
    inventory_run.mkdir()
    (inventory_run / "google-test.json").write_text(
        (inventory / "google-test.json").read_text()
    )
    shutil.copytree(run_dir, inventory_run, dirs_exist_ok=True)
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    command = "true"
    tests = [
        {
            "id": "unit_tests_per_file",
            "name": "unit",
            "command": command,
            "environment": {"FLAGCICD_REPORT_DIR": str(inventory_run)},
        }
    ]
    reports = tmp_path / "reports"
    env = _environment(reports)
    env.update(
        INTEGRATION_TESTS=json.dumps(tests),
        INTEGRATION_ENVIRONMENT="{}",
        INTEGRATION_WORKDIR=str(workdir),
        GITHUB_WORKSPACE=str(checkout),
        INTEGRATION_SOURCE_ROOT=str(checkout),
    )
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_integration_tests.py"),
            "--inventory",
            f"unit_tests_per_file={inventory}",
        ],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if expected:
        assert process.returncode == 0, process.stdout + process.stderr
        assert "Inventory met: 1 shared cases" in process.stdout
    else:
        assert process.returncode != 0, process.stdout + process.stderr
        assert "not passed" in process.stdout + process.stderr


def test_inventory_requires_a_pytest_report(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "google-test.json").write_text(
        json.dumps(["test_cases.py::test_shared"])
    )
    process, _ = _integration(
        tmp_path,
        [{"id": "sdpa_lanes", "name": "shared", "command": "true"}],
        inventory={"sdpa_lanes": "inventory"},
    )
    assert process.returncode != 0
    assert "no pytest report was written" in process.stdout + process.stderr


def _merge(reports, outcome="success"):
    return MERGER.merge_reports(
        reports, "cuda", outcome, "flagos-ai/Torch-FL", "a" * 40
    )


def test_multiple_sessions_preserve_outcomes_and_group_identity(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_first.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "@pytest.mark.skip(reason='device unavailable')\n"
        "def test_skip(): pass\n"
        "@pytest.mark.xfail(reason='known gap')\n"
        "def test_xfail(): assert False\n"
        "@pytest.mark.xfail(reason='resolved gap')\n"
        "def test_xpass(): pass\n"
    )
    (workdir / "test_second.py").write_text("def test_second(): pass\n")
    groups = [
        {"name": "first", "command": f"{PYTHON} -m pytest test_first.py -q"},
        {"name": "second", "command": f"{PYTHON} -m pytest test_second.py -q"},
    ]
    process, reports = _integration(tmp_path, groups)
    assert process.returncode == 0, process.stdout + process.stderr
    report = _merge(reports)
    assert report["summary"] == {
        "total": 5,
        "collected": 5,
        "passed": 2,
        "skipped": 1,
        "xfailed": 1,
        "xpassed": 1,
    }
    assert len(report["tests"]) == 5
    assert len(report["flagcicd"]["sessions"]) == 2
    assert report["flagcicd"]["complete"]
    assert {test["metadata"]["flagcicd_group"] for test in report["tests"]} == {
        "first",
        "second",
    }


@pytest.mark.parametrize(
    "source,expected,exitcode",
    [
        ("def test_failure(): assert False\n", "failed", 1),
        (
            "import pytest\n@pytest.fixture\ndef broken(): raise RuntimeError('setup')\n"
            "def test_error(broken): pass\n",
            "error",
            1,
        ),
        ("raise RuntimeError('collection failed')\n", None, 2),
    ],
)
def test_failed_and_collection_error_sessions_survive_fail_fast(
    tmp_path, source, expected, exitcode
):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_broken.py").write_text(source)
    process, reports = _integration(
        tmp_path,
        [
            {"name": "broken", "command": f"{PYTHON} -m pytest test_broken.py -q"},
            {"name": "unreached", "command": "touch should-not-exist"},
        ],
    )
    assert process.returncode == exitcode, process.stdout + process.stderr
    report = _merge(reports, "failure")
    assert report["exitcode"] == 1
    assert not (workdir / "should-not-exist").exists()
    assert report["flagcicd"]["sessions"][0]["exitcode"] == exitcode
    if expected:
        assert report["tests"][0]["outcome"] == expected
    else:
        assert any(item["outcome"] == "failed" for item in report["collectors"])


def test_per_file_runner_reports_each_file_but_not_nested_fixture_pytests(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_a.py").write_text(
        "import subprocess, sys\n"
        "def test_child(tmp_path):\n"
        "    child = tmp_path / 'test_child.py'\n"
        "    child.write_text('def test_expected_failure(): assert False')\n"
        "    result = subprocess.run([sys.executable, '-m', 'pytest', str(child), '-q'])\n"
        "    assert result.returncode == 1\n"
    )
    (workdir / "test_b.py").write_text("def test_b(): pass\n")
    command = f"{PYTHON} {shlex.quote(str(SCRIPTS / 'run_unit_tests.py'))} --test-dir ."
    process, reports = _integration(
        tmp_path, [{"name": "per file", "command": command}]
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = _merge(reports)
    assert report["summary"]["passed"] == report["summary"]["total"] == 2
    assert len(report["flagcicd"]["sessions"]) == 2


def test_sample_session_stays_invisible_and_its_next_suite_reports(
    tmp_path,
):
    """A test's own failing probe must not suppress the next real suite."""
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_suite.py").write_text(
        "import os, subprocess, sys\n"
        "def test_a_suite_starts_pytest_after_its_probe(tmp_path):\n"
        "    probe = tmp_path / 'test_probe.py'\n"
        "    probe.write_text('def test_expected_failure(): assert False')\n"
        "    failed = subprocess.run([sys.executable, '-m', 'pytest', str(probe), '-q'])\n"
        "    assert failed.returncode == 1\n"
        "    assert 'FLAGCICD_REPORT_DIR' not in os.environ\n"
        "    suite = tmp_path / 'test_inner.py'\n"
        "    suite.write_text('def test_real_suite(): pass')\n"
        "    passed = subprocess.run([sys.executable, '-m', 'pytest', str(suite), '-q'])\n"
        "    assert passed.returncode == 0\n"
    )
    command = f"{PYTHON} -m pytest test_suite.py -q"
    process, reports = _integration(tmp_path, [{"name": "suite", "command": command}])
    assert process.returncode == 0, process.stdout + process.stderr
    report = _merge(reports)
    sessions = report["flagcicd"]["sessions"]
    assert len(sessions) == 1
    assert {session["group"] for session in sessions} == {"suite"}
    assert report["summary"]["passed"] == report["summary"]["total"] == 1


def test_active_group_runs_the_case_it_names_before_fail_fast(tmp_path):
    """Inherited plugin lists and shlex-like pytest arguments stay stable."""
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    suite = workdir / "test_suite.py"
    suite.write_text("def test_real_suite(): pass\n")
    process, reports = _integration(
        tmp_path,
        [
            {
                "name": "suite",
                "command": f"{PYTHON} -m pytest {shlex.quote(str(suite))} -q",
            },
        ],
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = _merge(reports)
    assert len(report["flagcicd"]["sessions"]) == 1
    assert report["flagcicd"]["sessions"][0]["group"] == "suite"
    assert report["tests"][0]["metadata"]["flagcicd_group"] == "suite"


def test_crashed_and_corrupt_sessions_remain_explicitly_incomplete(tmp_path):
    for name in ("crashed", "corrupt"):
        (tmp_path / f"{name}.pending").write_text(
            json.dumps({"group": name, "args": []})
        )
    (tmp_path / "corrupt.json").write_text("{truncated")
    report = _merge(tmp_path)
    assert report["exitcode"] == 1
    assert report["summary"]["total"] == 0
    assert not report["flagcicd"]["complete"]
    assert set(report["flagcicd"]["incomplete_sessions"]) == {
        "crashed.json",
        "corrupt.json",
    }


def test_abrupt_exit_keeps_completed_sessions_without_inventing_crash_results(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_good.py").write_text("def test_good(): pass\n")
    (workdir / "test_exit.py").write_text(
        "import os\ndef test_abrupt_exit(): os._exit(3)\n"
    )
    process, reports = _integration(
        tmp_path,
        [
            {"name": "completed", "command": f"{PYTHON} -m pytest test_good.py -q"},
            {"name": "crashed", "command": f"{PYTHON} -m pytest test_exit.py -q"},
        ],
    )
    assert process.returncode == 3
    report = _merge(reports, "failure")
    assert report["summary"]["total"] == report["summary"]["passed"] == 1
    assert len(report["flagcicd"]["incomplete_sessions"]) == 1
    assert report["exitcode"] == 1


def test_deselected_counts_and_later_environment_failure_are_preserved(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_cases.py").write_text(
        "def test_selected(): pass\ndef test_other(): pass\n"
    )
    process, reports = _integration(
        tmp_path,
        [
            {
                "name": "selected",
                "command": f"{PYTHON} -m pytest test_cases.py -k selected -q",
            },
            {"name": "environment check", "command": "exit 7"},
        ],
    )
    assert process.returncode == 7
    report = _merge(reports, "failure")
    assert report["summary"] == {
        "total": 1,
        "collected": 2,
        "passed": 1,
        "deselected": 1,
    }
    assert report["exitcode"] == 1


def test_merger_cli_writes_the_report_path_used_by_the_workflow(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_good.py").write_text("def test_good(): pass\n")
    process, reports = _integration(
        tmp_path,
        [
            {"name": "completed", "command": f"{PYTHON} -m pytest test_good.py -q"},
        ],
    )
    assert process.returncode == 0
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    steps = {
        step.get("id", step.get("name")): step
        for step in workflow["jobs"]["integration-tests"]["steps"]
    }
    container_workspace = tmp_path / "torch-fl-workspace"
    container_workspace.mkdir()
    host_workspace = tmp_path / "host-workspace"
    cache = tmp_path / "torch-fl-pip"
    cache.mkdir()
    result = subprocess.run(
        ["bash", "-e", "-c", steps["merge-report"]["run"]],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_WORKSPACE": str(container_workspace),
            "CONTAINER_CONFIG": "",
            "FLAGCICD_REPORT_DIR": str(reports),
            "PIP_CACHE_DIR": str(cache),
            "PYTEST_REPORT": str(container_workspace / "pytest-cuda.json"),
            "COVERAGE_REPORT": str(container_workspace / "coverage-cuda.json"),
            "REPORT_PLATFORM": "cuda",
            "TEST_OUTCOME": "success",
            "SOURCE_REPOSITORY": "flagos-ai/Torch-FL",
            "SOURCE_REF": "a" * 40,
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

    def container_path(value):
        # github.workspace can name a host-only path in a job container.
        value = value.replace("${{ github.workspace }}", str(host_workspace))
        value = value.replace("${{ env.PIP_CACHE_DIR }}", str(cache))
        value = value.replace("${{ env.ARTIFACT_ROOT }}", str(container_workspace))
        value = value.replace("${{ env.PYTEST_REPORT_ROOT }}", str(tmp_path / "pytest"))
        value = value.replace(
            "${{ env.PYTEST_REPORT }}", str(container_workspace / "pytest-cuda.json")
        )
        value = value.replace(
            "${{ env.COVERAGE_REPORT }}",
            str(container_workspace / "coverage-cuda.json"),
        )
        value = value.replace("${{ inputs.platform }}", "cuda")
        return container_workspace / value

    output = container_path(
        steps["Upload pytest report to FlagCICD"]["with"]["report_path"]
    )
    assert output.is_file(), (output, list(tmp_path.rglob("*.json")))
    report = json.loads(output.read_text())
    assert report["exitcode"] == 0
    assert report["flagcicd"]["source_ref"] == "a" * 40
    artifact_paths = steps["Retain pytest reports"]["with"]["path"].splitlines()
    assert any(
        container_path(path) == output
        for path in artifact_paths
        if "${{ env.FLAGCICD_REPORT_DIR }}" not in path
    )


def test_pre_pytest_failure_does_not_invent_an_empty_success_report(tmp_path):
    process, reports = _integration(
        tmp_path, [{"name": "environment", "command": "exit 7"}]
    )
    assert process.returncode == 7
    with pytest.raises(ValueError, match="No pytest sessions started"):
        _merge(reports, "failure")


def test_a_session_without_an_integration_group_cannot_complete_the_report(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "manual.pending").write_text(json.dumps({"args": []}))
    (reports / "manual.json").write_text(
        json.dumps(
            {
                "summary": {"total": 1, "passed": 1},
                "tests": [],
                "exitcode": 0,
                "duration": 0.1,
            }
        )
    )

    report = _merge(reports)

    assert not report["flagcicd"]["grouped"]
    assert report["exitcode"] == 1


def test_workflow_transfers_optional_token_and_uploads_the_generated_report():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert (
        ci["jobs"]["platform-pipeline"]["secrets"]["FLAGCICD_API_TOKEN"]
        == "${{ secrets.FLAGCICD_API_TOKEN }}"
    )
    assert (
        workflow[True]["workflow_call"]["secrets"]["FLAGCICD_API_TOKEN"]["required"]
        is False
    )
    job = workflow["jobs"]["integration-tests"]
    assert "container" not in job
    assert (
        "CONTAINER_OPTIONS"
        in next(step for step in job["steps"] if step.get("id") == "workspace")["env"]
    )
    steps = {step.get("id", step.get("name")): step for step in job["steps"]}
    assert "!cancelled()" in steps["merge-report"]["if"]
    assert "steps.integration-verdict" in steps["merge-report"]["if"]
    assert steps["merge-report"]["env"]["TEST_OUTCOME"] == (
        "${{ steps.integration-verdict.outputs.outcome }}"
    )
    assert steps["integration-verdict"]["if"].endswith(
        "steps.integration-tests.outcome != 'skipped' }}"
    )
    runner = steps["integration-tests"]
    workspace = next(step for step in job["steps"] if step.get("id") == "workspace")
    assert "python3 .github/scripts/container_workspace.py prepare" in workspace["run"]
    assert "RUNNER_TEMP:-/tmp" in workspace["run"]
    assert (
        "FLAGCICD_VERDICT_FILE"
        in (ROOT / ".github/scripts/container_workspace.py").read_text()
    )
    assert "PYTEST_PLUGINS" in runner["run"]
    assert "flagcicd_instance" not in runner["run"]
    auth = steps["flagcicd-auth"]
    assert "inputs.checkout_repository == github.repository" in auth["if"]
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in auth["if"]
    )
    upload = steps["Upload pytest report to FlagCICD"]
    assert upload["continue-on-error"] is True
    assert (
        upload["with"]["backend_url"]
        == "https://flagcicd.flagos.net/flagcicd/api/v1/cicd/metrics/"
    )
    assert upload["with"]["file_type"] == "pytest"
    assert upload["timeout-minutes"] == 2


@pytest.mark.parametrize(
    "checkout,event,head,cancelled,merged,covered,expected",
    [
        ("flagos-ai/Torch-FL", "push", "", False, "success", "skipped", True),
        ("", "workflow_dispatch", "", False, "success", "skipped", True),
        (
            "contributor/fork",
            "workflow_dispatch",
            "",
            False,
            "success",
            "skipped",
            False,
        ),
        (
            "flagos-ai/Torch-FL",
            "pull_request",
            "contributor/fork",
            False,
            "success",
            "success",
            False,
        ),
        (
            "flagos-ai/Torch-FL",
            "pull_request",
            "flagos-ai/Torch-FL",
            False,
            "success",
            "skipped",
            True,
        ),
        ("flagos-ai/Torch-FL", "push", "", True, "success", "success", False),
        ("flagos-ai/Torch-FL", "push", "", False, "failure", "skipped", False),
        ("flagos-ai/Torch-FL", "push", "", False, "failure", "success", True),
        (
            "contributor/fork",
            "workflow_dispatch",
            "",
            False,
            "failure",
            "success",
            False,
        ),
    ],
)
def test_workflow_upload_boundary(
    checkout, event, head, cancelled, merged, covered, expected
):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    auth = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("id") == "flagcicd-auth"
    )
    expression = auth["if"].removeprefix("${{").removesuffix("}}").strip()
    for key, value in {
        "github.event.pull_request.head.repo.full_name": head,
        "inputs.checkout_repository": checkout,
        "github.repository": "flagos-ai/Torch-FL",
        "github.event_name": event,
        "steps.merge-report.outcome": merged,
        "steps.merge-coverage.outcome": covered,
    }.items():
        expression = expression.replace(key, repr(value))
    expression = expression.replace("cancelled()", repr(cancelled))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression)
    assert eval(expression.strip(), {"__builtins__": {}}, {}) == expected


@pytest.mark.parametrize(
    "token,expected", [("", ""), ("test-placeholder", "available=true\n")]
)
def test_missing_token_skips_upload_without_logging_credentials(
    tmp_path, token, expected
):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    auth = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("id") == "flagcicd-auth"
    )
    output = tmp_path / "output"
    output.touch()
    result = subprocess.run(
        ["bash", "-e", "-c", auth["run"]],
        env={**os.environ, "FLAGCICD_API_TOKEN": token, "GITHUB_OUTPUT": str(output)},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == expected
    assert not token or token not in result.stdout + result.stderr
