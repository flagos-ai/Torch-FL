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

"""Test real request planning, stage verdicts, missing reports and reruns."""

import copy
import importlib.util
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"


def load(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


PLANNER = load("flagcicd_plan")
RESOLVER = load("flagcicd_resolve")
REPORTER = load("flagcicd_report")
MERGER = load("merge_pytest_reports")
CONTRACT = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
SHA = "a" * 40


def request(**overrides):
    return {
        "source_repository": "flagos-ai/Torch-FL",
        "source_sha": SHA,
        "base_sha": "b" * 40,
        "requested_level": "auto",
        "platforms": "all",
        "task_id": 'task-"quoted"',
        "parent_run_id": "",
        **overrides,
    }


CONTROL_SHA = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
).strip()


def plan(level="L2", platforms=None):
    expected = platforms if platforms is not None else ["nvidia"]
    result = {
        **request(),
        "schema_version": "flagcicd.plan/v1",
        "entry": "flagcicd",
        "control_sha": CONTROL_SHA,
        "diff_head_sha": SHA,
        "profile": "pr",
        "test_level": level,
        "scope": CONTRACT["levels"][level]["scope"],
        "config_hash": "c" * 64,
        "risk_level": "R2",
        "risk_rule_version": "v1",
        "run_id": "42",
        "run_attempt": "1",
        "selection_reasons": [],
        "platforms": expected,
        "expected_platforms": expected,
        "expected_shared_checks": ["plan", "lint"]
        + ([] if level == "L0" else ["agnostic", "codegen"]),
        "execution_plan": [
            {
                "name": lane,
                "groups": [
                    {
                        "name": "smoke",
                        "group_id": "smoke",
                        "manifest_id": "smoke",
                        "command": "python -m pytest demo.py",
                        "environment": {},
                        "min_passed": 0,
                    }
                ],
            }
            for lane in expected
        ],
    }
    for lane in result["execution_plan"]:
        lane["groups"].append(
            {
                "name": "environment",
                "group_id": "environment",
                "manifest_id": "environment",
                "command": "true",
                "environment": {},
            }
        )
    result["plan_hash"] = PLANNER.plan_hash(result)
    return result


@pytest.mark.parametrize(
    "changed,level,risk",
    [
        (["README.md"], "L0", "R0"),
        (["docs/guide.rst"], "L0", "R0"),
        (["tests/unit/test_ci_matrix.py"], "L1", "R1"),
        (["torch_fl/__init__.py"], "L2", "R2"),
        (["csrc/runtime/test.cc"], "L2", "R3"),
        (["./.github/workflows/ci.yml"], "L2", "R2"),
        (["README.md", "unknown/file"], "L2", "R2"),
        (None, "L2", "R2"),
    ],
)
def test_risk_rules_come_from_the_capability_contract(changed, level, risk):
    actual = PLANNER.infer_level(changed, CONTRACT)
    assert actual[:2] == (level, risk)


@pytest.mark.parametrize(
    "changed",
    [
        ["requirements.txt"],
        ["requirements-dev.txt"],
        ["constraints.txt"],
        ["uv.lock"],
    ],
)
def test_dependency_pins_are_runtime_changes(changed):
    assert PLANNER.infer_level(changed, CONTRACT)[:2] == ("L2", "R2")


def test_unavailable_diff_risk_comes_from_the_contract():
    contract = json.loads(json.dumps(CONTRACT))
    contract["risk_rules"]["unknown_risk"] = "R1"
    assert PLANNER.infer_level(None, contract)[:2] == ("L2", "R1")


def test_plan_fetches_the_comparison_commit_for_fork_checkouts():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/flagcicd-entry.yml").read_text()
    )
    fetch = [
        step
        for step in workflow["jobs"]["plan"]["steps"]
        if "git -C source fetch" in step.get("run", "")
    ]
    assert len(fetch) == 1
    assert fetch[0]["env"]["BASE_SHA"] == "${{ steps.resolve.outputs.base_sha }}"
    # A shallow base has no parents, so merge-base cannot find the fork point.
    assert "--depth" not in fetch[0]["run"]
    assert "${{ github.repository }}" in fetch[0]["run"]


def test_contract_changes_change_the_actual_risk_decision():
    contract = json.loads(json.dumps(CONTRACT))
    contract["risk_rules"]["unknown_level"] = "L1"
    assert PLANNER.infer_level(["new-path"], contract)[0] == "L1"


@pytest.mark.parametrize("requested", ["L3", "L4", "invalid"])
def test_unsupported_levels_fail_before_platform_dispatch(requested):
    with pytest.raises(ValueError, match="unsupported"):
        PLANNER.build_plan(request(requested_level=requested), CONTRACT, ROOT, [])


def test_runtime_changes_cannot_be_lowered_to_l0():
    result = PLANNER.build_plan(
        request(requested_level="L0", platforms=["nvidia"]),
        CONTRACT,
        ROOT,
        ["csrc/aten/common.cc"],
    )
    assert result["test_level"] == "L2"
    # A boxing platform names both of its lanes; the plan runs each.
    assert result["expected_platforms"] == ["nvidia", "nvidia-boxing"]


def test_documentation_has_no_hardware_but_explicit_l2_is_nonempty():
    result = PLANNER.build_plan(request(), CONTRACT, ROOT, ["README.md"])
    assert result["test_level"] == "L0"
    assert result["expected_platforms"] == []
    result = PLANNER.build_plan(
        request(requested_level="L2"), CONTRACT, ROOT, ["README.md"]
    )
    assert result["expected_platforms"]


@pytest.mark.parametrize("platforms", [["../cuda"], [], [True], "cuda,wrong", {}])
def test_invalid_platforms_are_rejected(platforms):
    with pytest.raises(ValueError):
        PLANNER.build_plan(request(platforms=platforms), CONTRACT, ROOT, [])


def test_pr_resolution_pins_head_and_base_and_ignores_claimed_changed_files(
    monkeypatch,
):
    calls = []

    def api(path):
        calls.append(path)
        return {
            "state": "open",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": SHA},
            "base": {"sha": "b" * 40},
        }

    monkeypatch.setattr(RESOLVER, "api", api)
    result = RESOLVER.resolve(
        {
            "action": "flagcicd",
            "client_payload": {"pr_number": "12", "changed_files": []},
        },
        "repository_dispatch",
        "flagos-ai/Torch-FL",
        "c" * 40,
    )
    assert result["source_repository"] == "contributor/fork"
    assert result["source_sha"] == SHA
    assert result["base_sha"] == "b" * 40
    assert "changed_files" not in result
    assert calls == ["repos/flagos-ai/Torch-FL/pulls/12"]


@pytest.mark.parametrize(
    "payload",
    [
        {"pr_number": "$(whoami)"},
        {"pr_number": "12", "checkout_ref": "main"},
        {"task_id": "first\nplatforms=nvidia"},
        {"test_level": "L4"},
    ],
)
def test_bad_requests_fail(monkeypatch, payload):
    monkeypatch.setattr(RESOLVER, "commit", lambda *_: SHA)
    monkeypatch.setattr(RESOLVER, "api", lambda *_: {"default_branch": "main"})
    with pytest.raises(ValueError):
        RESOLVER.resolve({"inputs": payload}, "workflow_dispatch", "owner/repo", SHA)


def jobs(result="success", **stages):
    return {
        "configuration": {"result": "success"},
        "integration-tests": {
            "result": result,
            "outputs": {
                "stages": json.dumps(
                    {
                        **{
                            key: "success"
                            for key in (
                                "setup",
                                "build",
                                "validate",
                                "install",
                                "tests",
                            )
                        },
                        **stages,
                    }
                ),
                "run_attempt": "1",
            },
        },
    }


def raw_reports(tmp_path, source=SHA, complete=True, current=None):
    directory = tmp_path / "pytest"
    directory.mkdir(parents=True, exist_ok=True)
    current = current or plan()
    group = {
        **current["execution_plan"][0]["groups"][0],
        "status": "passed",
        "returncode": 0,
        "requires_pytest": True,
    }
    execution = {
        "expected": [g["name"] for g in current["execution_plan"][0]["groups"]],
        "groups": [
            group,
            {
                **current["execution_plan"][0]["groups"][1],
                "status": "passed",
                "returncode": 0,
                "requires_pytest": False,
            },
        ],
        "context": current,
        "environment": {},
        "platform": "nvidia",
        "run_attempt": 1,
    }
    sessions = [
        {
            "report": "session.json",
            "group": "smoke",
            "group_id": "smoke",
            "context": current,
            "platform": "nvidia",
            "run_attempt": 1,
            "selected_nodeids": ["demo.py::test_ok"],
            "observed_nodeids": ["demo.py::test_ok"],
            "exitcode": 0,
            "summary": {"total": 1, "passed": 1},
        }
    ]
    (directory / "pytest-nvidia.json").write_text(
        json.dumps(
            {
                "flagcicd": {
                    "platform": "nvidia",
                    "source_repository": "flagos-ai/Torch-FL",
                    "source_ref": source,
                    "complete": complete,
                    "execution": execution,
                    "sessions": sessions,
                },
                "exitcode": 0,
                "summary": {"total": 1, "passed": 1},
            }
        )
    )


@pytest.mark.parametrize(
    "result,stages,expected",
    [
        ("failure", {"build": "failure"}, "build_failed"),
        ("failure", {"validate": "failure"}, "build_failed"),
        ("failure", {"install": "failure"}, "build_failed"),
        ("failure", {"setup": "failure"}, "infra_error"),
        ("failure", {"tests": "failure"}, "test_failed"),
        ("cancelled", {}, "cancelled"),
        ("skipped", {}, "infra_error"),
    ],
)
def test_failure_before_pytest_still_produces_a_platform_outcome(
    tmp_path, result, stages, expected
):
    report = REPORTER.platform_report(
        plan(), "nvidia", jobs(result, **stages), tmp_path, 1
    )
    assert report["status"] == expected
    assert not report["metrics"]["coverage"]["available"]


def test_successful_job_requires_complete_pytest_data_but_coverage_is_optional(
    tmp_path,
):
    missing = REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 1)
    assert missing["status"] == "report_error"
    raw_reports(tmp_path)
    report = REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 1)
    assert report["status"] == "passed"
    assert report["metrics"]["tests"]["passed"] == 1
    assert report["coverage_warnings"]


def test_platform_report_validation_and_write_errors_cannot_succeed(
    tmp_path, monkeypatch
):
    plan_file = tmp_path / "plan.json"
    current = plan()
    plan_file.write_text(json.dumps(current))
    for key, value in {
        "FLAGCICD_PLAN": PLANNER.report_context(current),
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "JOB_RESULTS": json.dumps(jobs("failure", setup="failure")),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "platform",
            "--plan-file",
            str(plan_file),
            "--platform",
            "nvidia",
            "--directory",
            str(tmp_path),
            "--output",
            str(tmp_path / "missing" / "report.json"),
        ],
    )
    with pytest.raises(OSError):
        REPORTER.main()
    current["plan_hash"] = "invalid"
    plan_file.write_text(json.dumps(current))
    with pytest.raises(SystemExit, match="validated plan is unavailable"):
        REPORTER.main()


@pytest.mark.parametrize("source,complete", [("d" * 40, True), (SHA, False)])
def test_stale_or_partial_pytest_data_cannot_pass(tmp_path, source, complete):
    raw_reports(tmp_path, source, complete)
    assert (
        REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 1)["status"]
        == "report_error"
    )


def entry_jobs(level="L2"):
    return {
        name: {"result": "success" if expected else "skipped"}
        for name, expected in {
            "plan": True,
            "lint": True,
            "agnostic": level != "L0",
            "codegen": level != "L0",
            "platforms": level == "L2",
        }.items()
    }


def save_report(tmp_path, status="passed", attempt=1, **changes):
    raw = tmp_path / f"raw-{attempt}"
    raw_reports(raw)
    report = REPORTER.platform_report(plan(), "nvidia", jobs(), raw, attempt)
    report.update(status=status, **changes)
    path = tmp_path / f"flagcicd-platform-nvidia-{attempt}.json"
    path.write_text(json.dumps(report))
    return path


def test_missing_selected_platform_blocks_summary(tmp_path):
    summary = REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 1)
    assert summary["status"] != "passed"
    assert "missing platform report: nvidia" in summary["errors"]


@pytest.mark.parametrize("outcome", ["failure", "skipped", "cancelled"])
def test_job_result_cannot_be_hidden_by_a_passing_artifact(tmp_path, outcome):
    save_report(tmp_path)
    results = entry_jobs()
    results["platforms"]["result"] = outcome
    assert REPORTER.aggregate(plan(), results, tmp_path, 1)["status"] != "passed"


def test_l0_and_l1_accept_only_their_planned_skips(tmp_path):
    for level in ("L0", "L1"):
        summary = REPORTER.aggregate(plan(level, []), entry_jobs(level), tmp_path, 1)
        assert summary["status"] == "passed"
        assert summary["children"] == []
        results = entry_jobs(level)
        results["lint"]["result"] = "skipped"
        assert (
            REPORTER.aggregate(plan(level, []), results, tmp_path, 1)["status"]
            != "passed"
        )


def test_rerun_uses_latest_platform_report_and_keeps_other_successful_attempts(
    tmp_path,
):
    save_report(tmp_path, status="test_failed")
    save_report(tmp_path, attempt=2)
    summary = REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 2)
    assert summary["status"] == "passed"
    assert summary["children"][0]["run_attempt"] == 2
    assert REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 1)["status"] != "passed"


@pytest.mark.parametrize(
    "change",
    [{"source_sha": "d" * 40}, {"config_hash": "wrong"}, {"platform": {"name": "ppu"}}],
)
def test_reports_must_belong_to_the_plan(tmp_path, change):
    save_report(tmp_path, **change)
    assert REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 1)["status"] != "passed"


def test_diff_preserves_both_paths_of_a_vendor_rename(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    (tmp_path / "cuda.cc").write_text("native code\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "base",
    )
    base = git("rev-parse", "HEAD")
    git("mv", "cuda.cc", "ppu.cc")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "rename",
    )
    monkeypatch.chdir(tmp_path)
    assert set(PLANNER._changed_files("unused", base, "HEAD")) == {"cuda.cc", "ppu.cc"}


def test_unavailable_diff_is_distinct_from_an_empty_diff(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert PLANNER._changed_files("main", "a" * 40, "b" * 40) is None


def test_build_failure_keeps_its_stage_classification_in_summary(tmp_path):
    save_report(tmp_path, status="build_failed")
    results = entry_jobs()
    results["platforms"]["result"] = "failure"
    assert REPORTER.aggregate(plan(), results, tmp_path, 1)["status"] == "build_failed"


def test_matrix_failure_with_a_passing_report_is_infrastructure(tmp_path):
    # Every expected report exists and passes, so the report itself is not the
    # defect: the matrix job failed outside the stages it records.
    save_report(tmp_path)
    results = entry_jobs()
    results["platforms"]["result"] = "failure"
    assert REPORTER.aggregate(plan(), results, tmp_path, 1)["status"] == "infra_error"


@pytest.mark.parametrize("complete,available", [(True, True), (False, False)])
def test_coverage_is_per_platform_and_partial_data_is_unavailable(
    tmp_path, complete, available
):
    raw_reports(tmp_path)
    directory = tmp_path / "coverage"
    directory.mkdir()
    (directory / "coverage-nvidia.json").write_text(
        json.dumps(
            {
                "totals": {
                    "covered_lines": 5,
                    "num_statements": 20,
                    "percent_covered": 25.0,
                }
            }
        )
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "source_repository": "flagos-ai/Torch-FL",
                "source_ref": SHA,
                "platform": "nvidia",
                "complete": complete,
                "context": plan(),
                "run_attempt": 1,
            }
        )
    )
    report = REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 1)
    assert report["status"] == "passed"
    assert report["metrics"]["coverage"]["available"] is available


def test_duplicate_report_in_the_latest_attempt_blocks_success(tmp_path):
    save_report(tmp_path)
    latest = save_report(tmp_path, attempt=2)
    (tmp_path / "flagcicd-platform-duplicate.json").write_text(latest.read_text())
    assert REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 2)["status"] != "passed"


def test_entry_uses_trusted_control_code_and_pins_all_source_consumers():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/flagcicd-entry.yml").read_text()
    )
    assert set(workflow[True]) == {"workflow_dispatch", "repository_dispatch"}
    group = workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] is False
    # A run id is unique per run, so keying on it serializes nothing.
    assert "github.run_id" not in group
    for identity in ("pr_number", "checkout_ref", "task_id"):
        assert f"github.event.inputs.{identity}" in group
        assert f"github.event.client_payload.{identity}" in group
    control = workflow["jobs"]["plan"]["steps"][0]
    assert control["with"]["ref"] == "${{ github.workflow_sha }}"
    for name in ("lint", "agnostic", "codegen", "platforms"):
        inputs = workflow["jobs"][name]["with"]
        assert inputs["checkout_ref"] == "${{ needs.plan.outputs.source_sha }}"
        assert (
            inputs["checkout_repository"]
            == "${{ needs.plan.outputs.source_repository }}"
        )
    assert workflow["jobs"]["complete"]["if"] == "always()"


def test_plan_cli_preserves_json_identifiers_and_emits_a_real_plan(tmp_path):
    checkout = tmp_path / "source"
    shutil.copytree(ROOT / ".github/configs", checkout / ".github/configs")
    (checkout / ".github/scripts").mkdir()
    shutil.copyfile(
        ROOT / ".github/scripts/set_env.sh", checkout / ".github/scripts/set_env.sh"
    )
    shutil.copyfile(
        ROOT / ".github/integration-manifest.yml",
        checkout / ".github/integration-manifest.yml",
    )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args], text=True
        ).strip()

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "base",
    )
    sha = git("rev-parse", "HEAD")
    source = request(source_sha=sha, base_sha=sha, requested_level="L1")
    input_file = tmp_path / "request.json"
    input_file.write_text(json.dumps(source))
    output = tmp_path / "plan.json"
    github_output = tmp_path / "github-output"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(github_output),
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "flagcicd_plan.py"),
            "--request",
            str(input_file),
            "--source",
            str(checkout),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    actual = json.loads(output.read_text())
    assert actual["test_level"] == "L1"
    assert actual["expected_platforms"] == []
    assert actual["task_id"] == source["task_id"]
    lines = dict(line.split("=", 1) for line in github_output.read_text().splitlines())
    context = json.loads(lines["context"])
    assert "changed_files" not in context
    assert "matched_rules" not in context
    assert all(actual[key] == value for key, value in context.items())


@pytest.mark.parametrize(
    "result,stages,evidence,status,report_exit",
    [
        ("success", {}, True, "passed", 0),
        ("failure", {"build": "failure"}, False, "build_failed", 0),
        ("failure", {"setup": "failure"}, False, "infra_error", 0),
        ("failure", {"tests": "failure"}, True, "test_failed", 0),
        ("cancelled", {}, False, "cancelled", 0),
        ("success", {}, False, "report_error", 1),
    ],
)
def test_report_clis_retain_execution_and_summary(
    tmp_path, result, stages, evidence, status, report_exit
):
    raw = tmp_path / "raw"
    raw.mkdir()
    if evidence:
        raw_reports(raw)
    output = tmp_path / "flagcicd-platform-nvidia.json"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan()))
    env = {
        **os.environ,
        "CI_PLAN_FILE": str(plan_path),
        "FLAGCICD_PLAN": json.dumps(plan()),
        "JOB_RESULTS": json.dumps(jobs(result, **stages)),
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "flagcicd_report.py"),
            "platform",
            "--platform",
            "nvidia",
            "--directory",
            str(raw),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == report_exit, process.stderr
    assert json.loads(output.read_text())["status"] == status
    results = entry_jobs()
    results["platforms"]["result"] = (
        result if result != "success" else "failure" if report_exit else "success"
    )
    env["JOB_RESULTS"] = json.dumps(results)
    summary = tmp_path / "summary.json"
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "flagcicd_report.py"),
            "summary",
            "--directory",
            str(tmp_path),
            "--output",
            str(summary),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == (0 if status == "passed" else 1), process.stderr
    actual = json.loads(summary.read_text())
    assert actual["status"] == status
    assert actual["children"][0]["platform"] == {"name": "nvidia"}


@pytest.mark.parametrize(
    "event_name,event,expected",
    [
        (
            "workflow_dispatch",
            {"inputs": {"task_id": "invalid-level", "parent_run_id": "parent"}},
            {"task_id": "invalid-level", "parent_run_id": "parent"},
        ),
        (
            "repository_dispatch",
            {
                "action": "flagcicd",
                "client_payload": {"task_id": "invalid-platform", "parent_run_id": "p"},
            },
            {"task_id": "invalid-platform", "parent_run_id": "p"},
        ),
        (
            "workflow_dispatch",
            {"inputs": {"task_id": "unsafe\nvalue", "parent_run_id": "valid"}},
            {"task_id": "", "parent_run_id": "valid"},
        ),
        ("workflow_dispatch", {"inputs": ["invalid"]}, {"task_id": ""}),
    ],
)
def test_failed_plan_summary_keeps_only_safe_request_identity(
    tmp_path, monkeypatch, event_name, event, expected
):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    output = tmp_path / "summary.json"
    results = {name: {"result": "skipped"} for name in entry_jobs()}
    results["plan"]["result"] = "failure"
    for key, value in {
        "FLAGCICD_PLAN": "",
        "JOB_RESULTS": json.dumps(results),
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_REPOSITORY": "example/fork",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        sys,
        "argv",
        ["report", "summary", "--directory", str(tmp_path), "--output", str(output)],
    )
    assert REPORTER.main() == 1
    actual = json.loads(output.read_text())
    assert actual["status"] == "infra_error"
    assert actual["repository"] == "example/fork"
    assert actual["run_id"] == "42"
    assert actual["source_repository"] == actual["source_sha"] == ""
    assert all(actual[key] == value for key, value in expected.items())


@pytest.mark.parametrize("stage", ["setup", "build", "validate", "install"])
def test_passing_pytest_cannot_hide_a_skipped_required_stage(tmp_path, stage):
    raw_reports(tmp_path)
    result = REPORTER.platform_report(
        plan(), "nvidia", jobs(**{stage: "skipped"}), tmp_path, 1
    )
    assert result["status"] != "passed"


def test_l0_validates_all_configurations_during_planning():
    jobs = yaml.safe_load((ROOT / ".github/workflows/flagcicd-entry.yml").read_text())[
        "jobs"
    ]
    assert "preflight" not in jobs
    assert "plan" in jobs["complete"]["needs"]
    result = PLANNER.build_plan(request(), CONTRACT, ROOT, ["README.md"])
    assert result["test_level"] == "L0"
    assert result["validated_platforms"] == PLANNER._load_platforms(
        ROOT / ".github/configs"
    )
    assert result["expected_shared_checks"] == ["lint", "plan"]


def test_summary_artifact_name_agrees_with_the_contract():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/flagcicd-entry.yml").read_text()
    )
    upload = workflow["jobs"]["complete"]["steps"][-1]
    actual = upload["with"]["name"].replace("${{ github.run_attempt }}", "2")
    assert CONTRACT["report"]["summary_artifact"].format(run_attempt=2) == actual


def test_a_rerun_can_replace_a_report_from_a_previous_plan(tmp_path):
    save_report(tmp_path, config_hash="old-config", attempt=1)
    save_report(tmp_path, attempt=2)
    summary = REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 2)
    assert summary["status"] == "passed"
    assert summary["children"][0]["run_attempt"] == 2


def test_array_platforms_are_names_not_comma_separated_sublists():
    with pytest.raises(ValueError):
        PLANNER.build_plan(request(platforms=["cuda,ppu"]), CONTRACT, ROOT, [])


def test_large_diffs_remain_in_artifacts_and_context_can_start_a_process():
    changed = [
        f"csrc/runtime/generated/accelerator_implementation_{index}.cc"
        for index in range(2000)
    ]
    actual = PLANNER.build_plan(request(), CONTRACT, ROOT, changed)
    actual.update(run_id="42", run_attempt="1")
    assert len(json.dumps(actual).encode()) > 131072
    context = PLANNER.report_context(actual)
    assert len(context.encode()) < 16384
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,os; assert json.loads(os.environ['FLAGCICD_PLAN'])['test_level']=='L2'",
        ],
        env={**os.environ, "FLAGCICD_PLAN": context},
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert all(
        field not in json.loads(context) for field in ("changed_files", "matched_rules")
    )


@pytest.mark.parametrize(
    "job,level,status",
    [
        ("plan", "L0", "infra_error"),
        ("lint", "L0", "test_failed"),
        ("agnostic", "L1", "test_failed"),
        ("codegen", "L1", "test_failed"),
    ],
)
def test_shared_check_failures_keep_their_actual_failure_category(
    tmp_path, job, level, status
):
    results = entry_jobs(level)
    results[job]["result"] = "failure"
    assert REPORTER.aggregate(plan(level, []), results, tmp_path, 1)["status"] == status


@pytest.mark.parametrize("outcome", ["failure", "skipped", "cancelled"])
def test_configuration_plan_is_required_even_without_hardware(tmp_path, outcome):
    results = entry_jobs("L0")
    results["plan"]["result"] = outcome
    assert (
        REPORTER.aggregate(plan("L0", []), results, tmp_path, 1)["status"] != "passed"
    )


def test_failed_counts_cannot_hide_behind_a_zero_pytest_exit_code(tmp_path):
    raw_reports(tmp_path)
    path = tmp_path / "pytest/pytest-nvidia.json"
    data = json.loads(path.read_text())
    data["summary"] = {"total": 1, "failed": 1}
    path.write_text(json.dumps(data))
    assert (
        REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 1)["status"]
        == "report_error"
    )


def test_a_current_attempt_cannot_borrow_a_different_plans_success(tmp_path):
    save_report(tmp_path, attempt=2, config_hash="other-plan")
    summary = REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 2)
    assert summary["status"] == "report_error"


def test_rerun_does_not_reuse_stale_data_when_the_new_plan_never_ran(tmp_path):
    save_report(tmp_path, attempt=1, config_hash="old-config")
    summary = REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 2)
    assert summary["status"] != "passed"
    assert summary["ignored_reports"]
    assert not summary["children"]


def test_a_report_from_another_platform_cannot_supply_passing_metrics(tmp_path):
    raw_reports(tmp_path)
    path = tmp_path / "pytest/pytest-nvidia.json"
    data = json.loads(path.read_text())
    data["flagcicd"]["platform"] = "ppu"
    path.write_text(json.dumps(data))
    assert (
        REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 1)["status"]
        == "report_error"
    )


def test_cancelled_configuration_is_reported_as_cancelled(tmp_path):
    results = jobs("skipped")
    results["configuration"]["result"] = "cancelled"
    assert (
        REPORTER.platform_report(plan(), "nvidia", results, tmp_path, 1)["status"]
        == "cancelled"
    )


def test_workflow_stage_recorder_and_platform_report_use_the_same_contract(tmp_path):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    job = workflow["jobs"]["integration-tests"]
    record = next(step for step in job["steps"] if step.get("id") == "execution")
    output = tmp_path / "output"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_RUN_ATTEMPT": "3",
        **{key: "success" for key in record["env"]},
    }
    process = subprocess.run(
        ["bash", "-e", "-c", record["run"]], env=env, capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert set(json.loads(values["stages"])) == set(REPORTER.REQUIRED_STAGES)
    assert values["run_attempt"] == "3"
    raw_reports(tmp_path)
    path = tmp_path / "pytest/pytest-nvidia.json"
    payload = json.loads(path.read_text())
    payload["flagcicd"]["execution"]["run_attempt"] = 3
    for session in payload["flagcicd"]["sessions"]:
        session["run_attempt"] = 3
    path.write_text(json.dumps(payload))

    actual = jobs()
    actual["integration-tests"]["outputs"] = values
    report = REPORTER.platform_report(plan(), "nvidia", actual, tmp_path, 3)
    assert report["status"] == "passed"
    assert report["execution_attempt"] == "3"


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "extra",
        "order",
        "command",
        "environment",
        "min_passed",
        "requires_pytest",
        "not_run",
        "bad_exit",
        "wrong_sha",
        "wrong_control",
        "wrong_plan",
        "missing_session",
        "duplicate_session",
        "extra_session",
        "wrong_session_plan",
        "wrong_session_attempt",
        "missing_session_id",
        "shared_environment",
        "missing_inventory",
        "zero_collection",
        "bad_counts",
        "all_skipped",
    ],
)
def test_planned_group_evidence_is_checked_by_platform_and_summary(tmp_path, change):
    current = plan()
    raw_reports(tmp_path)
    good = REPORTER.platform_report(current, "nvidia", jobs(), tmp_path, 1)
    assert good["status"] == "passed"
    path = tmp_path / "pytest/pytest-nvidia.json"
    payload = json.loads(path.read_text())
    evidence = payload["flagcicd"]
    execution, sessions = evidence["execution"], evidence["sessions"]
    group = execution["groups"][0]
    if change == "missing":
        execution["groups"].pop()
    elif change == "duplicate":
        execution["groups"].append(copy.deepcopy(group))
    elif change == "extra":
        execution["groups"].append({**group, "name": "extra", "group_id": "extra"})
    elif change == "order":
        execution["groups"].reverse()
    elif change == "command":
        group["command"] = "true"
    elif change == "environment":
        group["environment"] = {"OTHER": "value"}
    elif change == "min_passed":
        group["min_passed"] = 1
    elif change == "requires_pytest":
        group["requires_pytest"] = False
    elif change == "not_run":
        group["status"] = "not_run"
    elif change == "bad_exit":
        group["returncode"] = 7
    elif change == "wrong_sha":
        execution["context"]["source_sha"] = "d" * 40
    elif change == "wrong_control":
        execution["context"]["control_sha"] = "d" * 40
    elif change == "wrong_plan":
        execution["context"]["plan_hash"] = "d" * 64
    elif change == "missing_session":
        sessions.clear()
    elif change == "duplicate_session":
        sessions.append(copy.deepcopy(sessions[0]))
    elif change == "extra_session":
        sessions.append({**sessions[0], "report": "extra.json", "group_id": "unknown"})
    elif change == "wrong_session_plan":
        sessions[0]["context"]["plan_hash"] = "d" * 64
    elif change == "wrong_session_attempt":
        sessions[0]["run_attempt"] = 2
    elif change == "missing_session_id":
        sessions[0].pop("group_id")
    elif change == "shared_environment":
        execution["environment"] = {"MODEL_PATH": "/other"}
    elif change == "missing_inventory":
        sessions[0].pop("selected_nodeids")
    elif change == "zero_collection":
        sessions[0]["selected_nodeids"] = sessions[0]["observed_nodeids"] = []
    elif change == "bad_counts":
        sessions[0]["summary"] = {"total": 1, "passed": 0}
    elif change == "all_skipped":
        sessions[0]["summary"] = {"total": 1, "skipped": 1}
    # Keep the self-reported inventory internally consistent; only the plan
    # supplies the independent expected group list.
    execution["expected"] = [g["name"] for g in execution["groups"]]
    path.write_text(json.dumps(payload))
    result = REPORTER.platform_report(current, "nvidia", jobs(), tmp_path, 1)
    assert result["status"] == "report_error", result
    good["evidence"] = {"execution": execution, "sessions": sessions}
    (tmp_path / "flagcicd-platform-tampered.json").write_text(json.dumps(good))
    summary = REPORTER.aggregate(current, entry_jobs(), tmp_path, 1)
    assert summary["status"] != "passed", summary


@pytest.mark.parametrize("declared,expected", [("1", "passed"), ("2", "report_error")])
def test_report_only_rerun_can_reuse_evidence_but_hardware_rerun_cannot(
    tmp_path, declared, expected
):
    raw_reports(tmp_path)
    results = jobs()
    results["integration-tests"]["outputs"]["run_attempt"] = declared
    report = REPORTER.platform_report(plan(), "nvidia", results, tmp_path, 2)
    assert report["status"] == expected


@pytest.mark.parametrize("entry", ["ci", "flagcicd"])
@pytest.mark.parametrize(
    "scenario", ["conditional_skip", "required_skip", "all_skipped"]
)
def test_real_execution_through_both_workflow_gates(tmp_path, entry, scenario):
    current = plan()
    current.update(source_sha=CONTROL_SHA, diff_head_sha=CONTROL_SHA, entry=entry)
    if entry == "ci":
        current["expected_shared_checks"] = [
            "resolve-checkout",
            "lint",
            "agnostic-checks",
            "codegen-checks",
            "discover-platforms",
        ]
        results = {
            name: {"result": "success"}
            for name in [*current["expected_shared_checks"], "platform-pipeline"]
        }
    else:
        results = entry_jobs()
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_required.py").write_text(
        "def test_required(): pass\n"
        if scenario == "conditional_skip"
        else "import pytest\n@pytest.mark.skip(reason='capability unavailable')\ndef test_required(): pass\n"
    )
    (suite / "test_conditional.py").write_text(
        "import pytest\n@pytest.mark.skip(reason='capability unavailable')\ndef test_conditional(): pass\n"
    )
    current["execution_plan"][0]["groups"] = [
        {
            "group_id": "required",
            "manifest_id": "required",
            "name": "required",
            "command": shlex.join(
                [sys.executable, "-m", "pytest", str(suite / "test_required.py"), "-q"]
            ),
            "min_passed": 0 if scenario == "all_skipped" else 1,
        },
        {
            "group_id": "conditional",
            "manifest_id": "conditional",
            "name": "conditional",
            "command": shlex.join(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(suite / "test_conditional.py"),
                    "-q",
                ]
            ),
        },
    ]
    current["plan_hash"] = PLANNER.plan_hash(current)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(current))
    raw = tmp_path / "raw"
    directory = raw / "pytest"
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "CI_PLAN_CONTEXT": PLANNER.report_context(current),
        "REPORT_PLATFORM": "nvidia",
        "FLAGCICD_REPORT_DIR": str(directory),
        "INTEGRATION_TESTS": json.dumps(current["execution_plan"][0]["groups"]),
        "INTEGRATION_ENVIRONMENT": "{}",
        "INTEGRATION_WORKDIR": str(suite),
        "PYTHONPATH": str(SCRIPTS / "pytest_reporting"),
        "PYTEST_PLUGINS": "flagcicd_report",
        "PYTEST_ADDOPTS": "",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GH_TOKEN": "",
    }
    run = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_integration_tests.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (run.returncode == 0) == (scenario != "required_skip"), (
        run.stdout + run.stderr
    )
    merged = MERGER.merge_reports(
        directory,
        "nvidia",
        "success" if run.returncode == 0 else "failure",
        current["source_repository"],
        current["source_sha"],
        True,
    )
    (directory / "pytest-nvidia.json").write_text(json.dumps(merged))
    platform_jobs = jobs() if run.returncode == 0 else jobs("failure", tests="failure")
    report = REPORTER.platform_report(current, "nvidia", platform_jobs, raw, 1)
    assert (report["status"] == "passed") == (scenario == "conditional_skip"), report
    if scenario == "conditional_skip":
        assert report["groups"][1]["summary"]["skipped"] == 1
        assert report["groups"][0]["summary"]["passed"] == 1
    elif scenario == "required_skip":
        assert [group["status"] for group in report["groups"]] == ["failed", "not_run"]
        results["platform-pipeline" if entry == "ci" else "platforms"]["result"] = (
            "failure"
        )
    else:
        results["platform-pipeline" if entry == "ci" else "platforms"]["result"] = (
            "failure"
        )
    reports = tmp_path / "platform-reports"
    reports.mkdir()
    (reports / "flagcicd-platform-nvidia.json").write_text(json.dumps(report))
    workflow = yaml.safe_load(
        (
            ROOT
            / ".github/workflows"
            / ("ci.yml" if entry == "ci" else "flagcicd-entry.yml")
        ).read_text()
    )
    gate = workflow["jobs"]["ci-complete" if entry == "ci" else "complete"]
    command = next(
        s["run"]
        for s in gate["steps"]
        if "flagcicd_report.py summary" in s.get("run", "")
    )
    summary_path = tmp_path / "summary.json"
    command = (
        command.replace(
            "--plan-file execution-plan/flagcicd-plan.json",
            "--plan-file " + shlex.quote(str(plan_path)),
        )
        .replace(
            "--directory platform-reports", "--directory " + shlex.quote(str(reports))
        )
        .replace(
            "--output flagcicd-run-summary.json",
            "--output " + shlex.quote(str(summary_path)),
        )
    )
    env.update(
        FLAGCICD_PLAN=PLANNER.report_context(current), JOB_RESULTS=json.dumps(results)
    )
    gate_run = subprocess.run(
        ["bash", "-e", "-c", command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (gate_run.returncode == 0) == (scenario == "conditional_skip"), (
        gate_run.stdout + gate_run.stderr
    )
    summary = json.loads(summary_path.read_text())
    assert summary["plan_hash"] == current["plan_hash"]
    assert summary["control_sha"] == current["control_sha"]
    assert (
        summary["children"][0]["evidence"]["execution"]["groups"][0]["group_id"]
        == "required"
    )


@pytest.mark.parametrize(
    "allowed,code,expected",
    [
        (True, 139, "passed"),
        (False, 139, "report_error"),
        (True, 7, "report_error"),
        (True, 0, "passed"),
    ],
)
def test_group_evidence_preserves_only_explicit_teardown_tolerance(
    tmp_path, allowed, code, expected
):
    current = plan()
    if allowed:
        current["execution_plan"][0]["groups"][0]["environment"] = {
            "INTEGRATION_ALLOW_TEARDOWN_CRASH": "1"
        }
    current["plan_hash"] = PLANNER.plan_hash(current)
    raw_reports(tmp_path, current=current)
    path = tmp_path / "pytest/pytest-nvidia.json"
    payload = json.loads(path.read_text())
    group = payload["flagcicd"]["execution"]["groups"][0]
    group.update(returncode=code, teardown_tolerated=code != 0)
    path.write_text(json.dumps(payload))
    report = REPORTER.platform_report(current, "nvidia", jobs(), tmp_path, 1)
    assert report["status"] == expected, report


def test_boxing_cannot_borrow_the_default_lanes_evidence(tmp_path):
    current = plan(platforms=["nvidia", "nvidia-boxing"])
    raw_reports(tmp_path, current=current)
    path = tmp_path / "pytest/pytest-nvidia.json"
    payload = json.loads(path.read_text())
    payload["flagcicd"]["platform"] = "nvidia-boxing"
    (path.parent / "pytest-nvidia-boxing.json").write_text(json.dumps(payload))
    report = REPORTER.platform_report(current, "nvidia-boxing", jobs(), tmp_path, 1)
    assert report["status"] == "report_error"
    assert any("lane" in error for error in report["errors"])


def test_passing_label_cannot_hide_report_errors(tmp_path):
    path = save_report(tmp_path)
    payload = json.loads(path.read_text())
    payload["errors"] = ["functional evidence is incomplete"]
    path.write_text(json.dumps(payload))
    assert (
        REPORTER.aggregate(plan(), entry_jobs(), tmp_path, 1)["status"]
        == "report_error"
    )


@pytest.mark.parametrize(
    "field,value", [("execution", []), ("sessions", [None]), ("sessions", {})]
)
def test_malformed_evidence_retains_a_failing_platform_outcome(tmp_path, field, value):
    raw_reports(tmp_path)
    path = tmp_path / "pytest/pytest-nvidia.json"
    payload = json.loads(path.read_text())
    payload["flagcicd"][field] = value
    path.write_text(json.dumps(payload))
    report = REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 1)
    assert report["status"] == "report_error"
    assert report["errors"]


def test_stale_coverage_is_optional_and_does_not_replace_functional_evidence(tmp_path):
    raw_reports(tmp_path)
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    (coverage / "coverage-nvidia.json").write_text(
        json.dumps(
            {
                "totals": {
                    "covered_lines": 1,
                    "num_statements": 2,
                    "percent_covered": 50.0,
                }
            }
        )
    )
    (coverage / "metadata.json").write_text(
        json.dumps(
            {
                "source_repository": plan()["source_repository"],
                "source_ref": SHA,
                "platform": "nvidia",
                "context": plan(),
                "complete": True,
                "run_attempt": 2,
            }
        )
    )
    report = REPORTER.platform_report(plan(), "nvidia", jobs(), tmp_path, 2)
    assert report["status"] == "passed"
    assert report["metrics"]["coverage"]["available"] is False
    assert report["coverage_warnings"]


@pytest.mark.parametrize(
    "name,status,expected",
    [
        ("lint", "failure", "test_failed"),
        ("agnostic-checks", "failure", "test_failed"),
        ("codegen-checks", "failure", "test_failed"),
        ("lint", "cancelled", "cancelled"),
        ("resolve-checkout", "failure", "infra_error"),
        ("discover-platforms", "failure", "infra_error"),
    ],
)
def test_pre_plan_failure_keeps_its_cause_without_inventing_planned_skips(
    tmp_path, name, status, expected
):
    results = {
        job: {"result": "skipped"}
        for job in (
            "resolve-checkout",
            "lint",
            "agnostic-checks",
            "codegen-checks",
            "discover-platforms",
            "platform-pipeline",
        )
    }
    if name != "resolve-checkout":
        results["resolve-checkout"]["result"] = "success"
    if name not in ("resolve-checkout", "lint"):
        for job in ("lint", "agnostic-checks", "codegen-checks"):
            results[job]["result"] = "success"
    results[name]["result"] = status
    summary = REPORTER.aggregate(None, results, tmp_path, 1, "ci")
    assert summary["status"] == expected
    assert "test plan is unavailable" in summary["errors"]
    assert f"{name}: {status}" in summary["errors"]
    assert not any(
        "expected skipped, received success" in error for error in summary["errors"]
    )
    assert "plan does not require its producer job" not in summary["errors"]


@pytest.mark.parametrize("failed", [False, True])
def test_platform_groups_are_summarized_once_on_success_and_failure(
    tmp_path, monkeypatch, failed
):
    raw_reports(tmp_path)
    if failed:
        path = tmp_path / "pytest/pytest-nvidia.json"
        payload = json.loads(path.read_text())
        payload["flagcicd"]["complete"] = False
        payload["flagcicd"]["execution"]["groups"][0]["status"] = "failed"
        path.write_text(json.dumps(payload))
    calls = {"diagnostic": 0, "validated": 0}
    diagnostic = REPORTER.summarize_groups
    validated = REPORTER.validate_execution.__globals__["summarize_groups"]

    def describe(*args, **kwargs):
        calls["diagnostic"] += 1
        return diagnostic(*args, **kwargs)

    def describe_validated(*args, **kwargs):
        calls["validated"] += 1
        return validated(*args, **kwargs)

    monkeypatch.setattr(REPORTER, "summarize_groups", describe)
    monkeypatch.setitem(
        REPORTER.validate_execution.__globals__, "summarize_groups", describe_validated
    )
    result = REPORTER.platform_report(
        plan(),
        "nvidia",
        jobs("failure", tests="failure") if failed else jobs(),
        tmp_path,
        1,
    )
    assert result["status"] == ("test_failed" if failed else "passed")
    assert calls == (
        {"diagnostic": 1, "validated": 0}
        if failed
        else {"diagnostic": 0, "validated": 1}
    )
    assert result["groups"][0]["status"] == ("failed" if failed else "passed")
