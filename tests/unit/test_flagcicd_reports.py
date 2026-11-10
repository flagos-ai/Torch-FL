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
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(MERGER)
finally:
    sys.path.pop(0)
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


def _skip_policy_session(nodeid, *, outcome="skipped", policies=None):
    return {
        "tests_details": [{"nodeid": nodeid, "outcome": outcome}],
        "skip_policy": {} if policies is None else policies,
    }


def _minimum_session(directory, *, group="current", group_id="current"):
    marker = directory / "b-current.pending"
    marker.write_text(
        json.dumps(
            {"group": group, "group_id": group_id, "selected_nodeids": ["test_ok"]}
        )
    )
    marker.with_suffix(".json").write_text(
        json.dumps(
            {
                "summary": {"passed": 1, "total": 1},
                "tests": [{"nodeid": "test_ok", "outcome": "passed"}],
                "exitcode": 0,
                "duration": 0.1,
            }
        )
    )


def test_partial_minimum_reports_actual_and_required_counts():
    with pytest.raises(ValueError, match="dtype: 8 passed, 9 required"):
        MERGER.validate_group_minimum({"name": "dtype", "min_passed": 9}, {"passed": 8})


@pytest.mark.parametrize("broken_json", [None, "invalid json"])
def test_minimum_ignores_other_group_but_final_merge_rejects_it(tmp_path, broken_json):
    marker = tmp_path / "a-earlier.pending"
    marker.write_text(json.dumps({"group": "earlier", "group_id": "earlier"}))
    if broken_json is not None:
        marker.with_suffix(".json").write_text(broken_json)
    _minimum_session(tmp_path)
    MERGER.check_group_minimum(
        tmp_path, {"name": "current", "group_id": "current", "min_passed": 1}
    )
    merged = _merge(tmp_path)
    assert merged["exitcode"] == 1
    assert merged["flagcicd"]["complete"] is False
    assert marker.with_suffix(".json").name in merged["flagcicd"]["incomplete_sessions"]


@pytest.mark.parametrize("group_id,complete", [("current", True), ("other", False)])
def test_final_minimum_matches_stable_id_instead_of_display_name(
    tmp_path, group_id, complete
):
    _minimum_session(tmp_path, group_id=group_id)
    group = {
        "name": "current",
        "group_id": "current",
        "command": "python -m pytest test_ok.py",
        "min_passed": 1,
        "requires_pytest": True,
        "status": "passed",
        "returncode": 0,
    }
    (tmp_path / "groups.json").write_text(
        json.dumps({"expected": ["current"], "groups": [group]})
    )
    merged = _merge(tmp_path)
    assert merged["flagcicd"]["complete"] is complete
    assert merged["exitcode"] == (0 if complete else 1)


def test_minimum_rejects_its_own_missing_report(tmp_path):
    _minimum_session(tmp_path)
    (tmp_path / "b-current.json").unlink()
    with pytest.raises(FileNotFoundError):
        MERGER.check_group_minimum(
            tmp_path, {"name": "current", "group_id": "current", "min_passed": 1}
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
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )
    steps = {
        step.get("id", step.get("name")): step
        for step in workflow["jobs"]["integration-tests"]["steps"]
    }
    container_workspace = tmp_path / "container-workspace"
    container_workspace.mkdir()
    host_workspace = tmp_path / "host-workspace"
    result = subprocess.run(
        ["bash", "-e", "-c", steps["merge-report"]["run"]],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_WORKSPACE": str(container_workspace),
            "FLAGCICD_REPORT_DIR": str(reports),
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
        value = value.replace("${{ inputs.platform }}", "cuda")
        return container_workspace / value

    output = container_path(
        steps["Upload pytest report to FlagCICD"]["with"]["report_path"]
    )
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


def test_swallowed_pytest_failure_cannot_pass_functional_evidence(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_mixed.py").write_text(
        "def test_pass(): pass\ndef test_failure(): assert False\n"
    )
    process, reports = _integration(
        tmp_path,
        [
            {
                "name": "wrapper",
                "command": f"{PYTHON} -m pytest test_mixed.py -q || true",
            }
        ],
    )
    assert process.returncode == 0
    report = _merge(reports)
    assert report["summary"]["failed"] == report["summary"]["passed"] == 1
    assert report["flagcicd"]["complete"] is False
    assert report["exitcode"] != 0
    assert report["flagcicd"]["execution_errors"]


@pytest.mark.parametrize(
    "summary", [{"passed": 1, "failed": 1}, {"passed": 1, "error": 1}, {"passed": True}]
)
def test_shared_evidence_validator_rejects_contradictory_counts(summary):
    with pytest.raises(ValueError):
        MERGER.validate_pytest_result(
            {"exitcode": 0, "summary": summary}, require_success=True
        )


def test_optional_skipped_session_does_not_invalidate_passing_evidence(tmp_path):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_pass.py").write_text("def test_ok(): pass\n")
    (workdir / "test_skip.py").write_text(
        "import pytest\n@pytest.mark.skip(reason='not applicable')\ndef test_skip(): pass\n"
    )
    process, reports = _integration(
        tmp_path,
        [
            {"name": "required", "command": f"{PYTHON} -m pytest test_pass.py -q"},
            {"name": "conditional", "command": f"{PYTHON} -m pytest test_skip.py -q"},
        ],
    )
    assert process.returncode == 0
    report = _merge(reports)
    assert report["flagcicd"]["complete"] is True
    assert report["exitcode"] == 0


def test_migrated_skip_requires_a_policy_classification():
    nodeid = (
        "tests/integration/ops/test_all_dispatch.py"
        "::TestAllCorrectness::test_all_empty_tensor"
    )
    errors = MERGER.validate_skip_policy(
        _skip_policy_session(nodeid),
        frozenset({"tests/integration/ops/test_all_dispatch.py"}),
    )
    assert errors == [f"{nodeid}: skipped in a migrated file has no skip-policy marker"]


@pytest.mark.parametrize(
    "policy",
    [
        {"category": "tracked_gap", "issue": "#710"},
        {
            "category": "platform_gate",
            "gate": "device_count",
            "detail": "requires two devices",
        },
        {
            "category": "platform_gate",
            "gate": "lane_config",
            "detail": "requires the shipped default conf",
        },
    ],
)
def test_valid_skip_policy_classifications_pass(policy):
    nodeid = "tests/integration/flaggems/test_contract.py::test_case"
    errors = MERGER.validate_skip_policy(
        _skip_policy_session(nodeid, policies={nodeid: policy}),
        frozenset({"tests/integration/flaggems/test_contract.py"}),
    )
    assert errors == []


@pytest.mark.parametrize(
    "policy,message",
    [
        ({"category": "other"}, "unknown skip-policy category"),
        ({"category": "tracked_gap"}, "unexpected fields"),
        ({"category": "tracked_gap", "issue": "710"}, "needs a numbered issue"),
        (
            {"category": "tracked_gap", "issue": "#710", "detail": "extra"},
            "unexpected fields",
        ),
        (
            {"category": "platform_gate", "detail": "missing gate"},
            "unexpected fields",
        ),
        (
            {
                "category": "platform_gate",
                "gate": "unsupported",
                "detail": "invalid gate",
            },
            "unknown platform gate",
        ),
        (
            {
                "category": "platform_gate",
                "gate": "build_record",
                "detail": "",
            },
            "non-empty single line",
        ),
        (
            {
                "category": "platform_gate",
                "gate": "build_record",
                "detail": "valid",
                "issue": "#710",
            },
            "unexpected fields",
        ),
        ({"category": "invalid", "reason": "conflicting markers"}, "conflicting"),
    ],
)
def test_invalid_skip_policy_classifications_fail(policy, message):
    nodeid = "tests/integration/ops/test_contract.py::test_case"
    errors = MERGER.validate_skip_policy(
        _skip_policy_session(nodeid, policies={nodeid: policy}),
        frozenset({"tests/integration/ops/test_contract.py"}),
    )
    assert len(errors) == 1
    assert errors[0].startswith(f"{nodeid}: invalid skip policy: ")
    assert message in errors[0]


def test_skip_policy_classification_outside_governed_roots_fails():
    nodeid = "tests/integration/test_contract.py::test_case"
    errors = MERGER.validate_skip_policy(
        _skip_policy_session(
            nodeid,
            policies={nodeid: {"category": "tracked_gap", "issue": "#710"}},
        ),
        frozenset(),
    )
    assert errors == [f"{nodeid}: skip policy recorded outside a governed file"]


def test_legacy_skip_in_unmigrated_governed_file_remains_allowed():
    nodeid = "tests/integration/ops/test_legacy.py::test_case"
    errors = MERGER.validate_skip_policy(
        _skip_policy_session(nodeid),
        frozenset({"tests/integration/ops/test_all_dispatch.py"}),
    )
    assert errors == []


def test_malformed_skip_policy_fails_merge_without_crashing(tmp_path):
    _minimum_session(tmp_path)
    marker = tmp_path / "b-current.pending"
    session = json.loads(marker.read_text())
    session["skip_policy"] = []
    marker.write_text(json.dumps(session))

    merged = _merge(tmp_path)

    assert merged["flagcicd"]["complete"] is False
    assert merged["exitcode"] == 1
    assert (
        "b-current.json: skip_policy must be an object"
        in merged["flagcicd"]["execution_errors"]
    )


def test_runtime_reports_policy_markers_and_merge_copies_the_classification(
    tmp_path,
):
    workdir = tmp_path / "wheel-only"
    test_file = workdir / "tests/integration/ops/test_policy.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "@pytest.mark.flagos_tracked_gap(issue='#710')\n"
        "@pytest.mark.skip(reason='tracked gap')\n"
        "def test_tracked(): pass\n"
        "@pytest.mark.flagos_platform_gate(\n"
        "    gate='device_count', detail='requires two devices'\n"
        ")\n"
        "@pytest.mark.skip(reason='platform gate')\n"
        "def test_gated(): pass\n"
    )
    process, reports = _integration(
        tmp_path,
        [
            {
                "name": "policy",
                "command": (
                    f"{PYTHON} -m pytest tests/integration/ops/test_policy.py -q"
                ),
            }
        ],
    )
    assert process.returncode == 0, process.stdout + process.stderr
    pending_paths = list(reports.glob("*.pending"))
    assert len(pending_paths) == 1
    pending = json.loads(pending_paths[0].read_text())
    assert pending["skip_policy"] == {
        "tests/integration/ops/test_policy.py::test_tracked": {
            "category": "tracked_gap",
            "issue": "#710",
        },
        "tests/integration/ops/test_policy.py::test_gated": {
            "category": "platform_gate",
            "gate": "device_count",
            "detail": "requires two devices",
        },
    }
    report = _merge(reports)
    assert report["flagcicd"]["complete"] is True
    by_nodeid = {test["nodeid"]: test for test in report["tests"]}
    assert (
        by_nodeid["tests/integration/ops/test_policy.py::test_tracked"]["metadata"][
            "skip_policy"
        ]
        == pending["skip_policy"]["tests/integration/ops/test_policy.py::test_tracked"]
    )
    assert (
        by_nodeid["tests/integration/ops/test_policy.py::test_gated"]["metadata"][
            "skip_policy"
        ]
        == pending["skip_policy"]["tests/integration/ops/test_policy.py::test_gated"]
    )


def test_platform_gate_replaces_a_tracked_gap_for_an_inapplicable_case(tmp_path):
    """A platform gate is the sole classification when it supersedes a gap.

    MUSA cannot serve ``flaggems_python`` at all. A test carrying both a
    tracked gap and that backend marker must therefore report a platform gate,
    rather than fail the report as having two classifications.
    """
    workdir = tmp_path / "wheel-only"
    test_dir = workdir / "tests/integration/ops"
    test_dir.mkdir(parents=True)
    policy_path = ROOT / "tests/integration/skip_policy.py"
    (test_dir / "conftest.py").write_text(
        "import importlib.util\n"
        "from pathlib import Path\n"
        "import pytest\n"
        f"policy_path = Path({str(policy_path)!r})\n"
        "spec = importlib.util.spec_from_file_location('skip_policy', policy_path)\n"
        "skip_policy = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(skip_policy)\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        "        if item.get_closest_marker('requires_missing_backend'):\n"
        "            skip_policy.add_platform_skip(\n"
        "                item,\n"
        "                gate='backend_marker',\n"
        "                detail='runtime is musa; case requires @flaggems_python',\n"
        "                reason='Skipped on musa runtime: requires @flaggems_python',\n"
        "            )\n"
    )
    (test_dir / "test_policy.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "@pytest.mark.requires_missing_backend\n"
        "@pytest.mark.flagos_tracked_gap(issue='#710')\n"
        "@pytest.mark.skipif(False, reason='tracked gap')\n"
        "def test_gated(): pass\n"
    )
    process, reports = _integration(
        tmp_path,
        [
            {
                "name": "platform gate",
                "command": (
                    f"{PYTHON} -m pytest tests/integration/ops/test_policy.py -q"
                ),
            }
        ],
    )
    assert process.returncode == 0, process.stdout + process.stderr
    pending_paths = list(reports.glob("*.pending"))
    assert len(pending_paths) == 1
    pending = json.loads(pending_paths[0].read_text())
    nodeid = "tests/integration/ops/test_policy.py::test_gated"
    assert pending["skip_policy"] == {
        nodeid: {
            "category": "platform_gate",
            "gate": "backend_marker",
            "detail": "runtime is musa; case requires @flaggems_python",
        }
    }
    report = _merge(reports)
    assert report["flagcicd"]["complete"] is True
    by_nodeid = {test["nodeid"]: test for test in report["tests"]}
    assert (
        by_nodeid[nodeid]["metadata"]["skip_policy"] == pending["skip_policy"][nodeid]
    )


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


@pytest.mark.parametrize(
    "case,expected",
    [("quiet", 0), ("skipped", 1), ("fake_stdout", 1), ("earlier_pass", 1)],
)
def test_early_minimum_uses_group_json_and_never_console_counts(
    tmp_path, case, expected
):
    workdir = tmp_path / "wheel-only"
    workdir.mkdir()
    (workdir / "test_pass.py").write_text("def test_ok(): pass\n")
    (workdir / "test_skip.py").write_text(
        "import pytest\n@pytest.mark.skip(reason='capability unavailable')\ndef test_skip(): pass\n"
    )
    marker = tmp_path / "must-not-run"
    if case == "fake_stdout":
        command = "printf '100 passed in 1.0s\\n'"
    else:
        command = f"{PYTHON} -m pytest {'test_pass.py' if case == 'quiet' else 'test_skip.py'} -qq"
    groups = [
        {"name": "minimum", "group_id": "minimum", "command": command, "min_passed": 1}
    ]
    if case == "earlier_pass":
        groups.insert(
            0,
            {
                "name": "earlier",
                "group_id": "earlier",
                "command": f"{PYTHON} -m pytest test_pass.py -qq",
            },
        )
    if expected:
        groups.append(
            {
                "name": "later",
                "group_id": "later",
                "command": "touch " + shlex.quote(str(marker)),
            }
        )
    run, reports = _integration(tmp_path, groups)
    assert run.returncode == expected, run.stdout + run.stderr
    execution = json.loads((reports / "groups.json").read_text())
    current = next(
        group for group in execution["groups"] if group["group_id"] == "minimum"
    )
    assert current["status"] == ("passed" if expected == 0 else "failed")
    if expected:
        assert current["error"]
        assert "evidence check failed" in run.stdout
        assert not marker.exists()
        assert execution["groups"][-1]["status"] == "not_run"
    if case == "quiet":
        merged = _merge(reports)
        assert merged["summary"]["passed"] == 1
        assert merged["flagcicd"]["complete"] is True
