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
    env.update(
        FLAGCICD_REPORT_DIR=str(directory),
        PYTHONPATH=str(SCRIPTS / "pytest_reporting"),
        PYTEST_PLUGINS="flagcicd_report",
    )
    return env


def _integration(tmp_path, groups):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir(exist_ok=True)
    reports = tmp_path / "reports"
    env = _environment(reports)
    env.update(
        INTEGRATION_TESTS=json.dumps(groups),
        INTEGRATION_ENVIRONMENT="{}",
        INTEGRATION_WORKDIR=str(workdir),
    )
    process = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_integration_tests.py")],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return process, reports


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
    output = tmp_path / "pytest-cuda.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "merge_pytest_reports.py"),
            "--report-dir",
            str(reports),
            "--output",
            str(output),
            "--platform",
            "cuda",
            "--outcome",
            "success",
            "--source-repository",
            "flagos-ai/Torch-FL",
            "--source-ref",
            "a" * 40,
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["exitcode"] == 0
    assert report["flagcicd"]["source_ref"] == "a" * 40


def test_pre_pytest_failure_does_not_invent_an_empty_success_report(tmp_path):
    process, reports = _integration(
        tmp_path, [{"name": "environment", "command": "exit 7"}]
    )
    assert process.returncode == 7
    with pytest.raises(ValueError, match="No pytest sessions started"):
        _merge(reports, "failure")


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
    container = job.get("container")
    if isinstance(container, dict):
        container_image = container["image"]
        volumes = container["volumes"]
        options = container["options"]
    else:
        # MPS runs directly on macOS because a macOS runner cannot start a
        # container. The job passes the configured values through one
        # expression, which it can turn into a mapping or null.
        if isinstance(container, str):
            assert "needs.configuration.outputs.ci_image" in container
            assert "needs.configuration.outputs.container_volumes" in container
            assert "needs.configuration.outputs.container_options" in container
            return
        container_image, volumes, options = yaml.safe_load(
            '{"image":"${{ needs.configuration.outputs.ci_image }}",'
            '"volumes":"${{ fromJson(needs.configuration.outputs.container_volumes) }}",'
            '"options":"${{ needs.configuration.outputs.container_options }}"}'
        ).values()
    assert container_image == "${{ needs.configuration.outputs.ci_image }}"
    assert volumes == "${{ fromJson(needs.configuration.outputs.container_volumes) }}"
    assert options == "${{ needs.configuration.outputs.container_options }}"
    steps = {step.get("id", step.get("name")): step for step in job["steps"]}
    assert "!cancelled()" in steps["merge-report"]["if"]
    assert "steps.integration-tests.outcome" in steps["merge-report"]["if"]
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
    assert (
        upload["with"]["report_path"]
        == "${{ github.workspace }}/pytest-${{ inputs.platform }}.json"
    )
    assert upload["timeout-minutes"] == 2


@pytest.mark.parametrize(
    "checkout,event,head,cancelled,merged,expected",
    [
        ("flagos-ai/Torch-FL", "push", "", False, "success", True),
        ("", "workflow_dispatch", "", False, "success", True),
        ("contributor/fork", "workflow_dispatch", "", False, "success", False),
        (
            "flagos-ai/Torch-FL",
            "pull_request",
            "contributor/fork",
            False,
            "success",
            False,
        ),
        (
            "flagos-ai/Torch-FL",
            "pull_request",
            "flagos-ai/Torch-FL",
            False,
            "success",
            True,
        ),
        ("flagos-ai/Torch-FL", "push", "", True, "success", False),
        ("flagos-ai/Torch-FL", "push", "", False, "failure", False),
    ],
)
def test_workflow_upload_boundary(checkout, event, head, cancelled, merged, expected):
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
