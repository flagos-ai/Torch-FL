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

"""Keep matrix routing, checkout source and the required gate in agreement."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CONFIGS = ROOT / ".github" / "configs"
PLATFORMS = sorted(
    path.stem
    for path in CONFIGS.glob("*.yml")
    if path.stem not in {"template", "image_sources", "platforms"}
)


sys.path.insert(0, str(ROOT / ".github/scripts"))
try:
    import flagcicd_plan as PLANNER

    spec = importlib.util.spec_from_file_location(
        "routing_ci_gate", ROOT / ".github/scripts/flagcicd_report.py"
    )
    REPORTER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(REPORTER)
finally:
    sys.path.pop(0)


def gate_plan(selected=False):
    request = {
        "source_repository": "owner/fork",
        "source_sha": "a" * 40,
        "control_sha": "c" * 40,
        "base_sha": "b" * 40,
        "task_id": "",
        "parent_run_id": "",
        "run_id": "42",
        "run_attempt": "1",
    }
    contract = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())
    return PLANNER.build_ci_plan(
        request,
        contract,
        ROOT,
        ["README.md"],
        "workflow_dispatch" if selected else "pull_request",
        dispatch=PLATFORMS[0] if selected else "",
    )


def _shared_test_files(workflow):
    return [
        token
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        for token in shlex.split(step.get("run", ""), comments=True)
        if token.startswith("tests/") and token.endswith(".py")
    ]


def test_cpu_workflows_do_not_schedule_the_same_explicit_test_file_twice():
    agnostic = _shared_test_files(_workflow("agnostic-checks.yml"))
    codegen = _shared_test_files(_workflow("codegen-checks.yml"))
    assert len(agnostic) == len(set(agnostic))
    assert len(codegen) == len(set(codegen))
    assert not set(agnostic) & set(codegen)
    assert "tests/unit/test_gen_vendor_confs.py" in agnostic


def test_stdlib_vendor_conf_check_has_one_lightweight_owner():
    calls = []
    for name in ("agnostic-checks.yml", "codegen-checks.yml"):
        for job in _workflow(name)["jobs"].values():
            for step in job.get("steps", []):
                for line in step.get("run", "").replace("\\\n", " ").splitlines():
                    argv = shlex.split(line, comments=True)
                    if (
                        argv
                        and argv[0] in ("python", "python3")
                        and argv[1:]
                        == ["scripts/codegen/gen_vendor_confs.py", "--check"]
                    ):
                        calls.append(name)
    assert calls == ["agnostic-checks.yml"]


def test_both_entries_still_require_both_shared_cpu_jobs():
    ci = _workflow("ci.yml")["jobs"]
    entry = _workflow("flagcicd-entry.yml")["jobs"]
    assert {"agnostic-checks", "codegen-checks"} <= set(ci["ci-complete"]["needs"])
    assert {"agnostic", "codegen"} <= set(entry["complete"]["needs"])
    assert {"agnostic", "codegen"} <= set(entry["platforms"]["needs"])


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_required_gate_covers_matrix_pipeline():
    jobs = _workflow("ci.yml")["jobs"]
    gate = jobs["ci-complete"]
    assert set(gate["needs"]) == {
        "resolve-checkout",
        "lint",
        "agnostic-checks",
        "codegen-checks",
        "discover-platforms",
        "platform-pipeline",
    }
    assert (
        jobs["platform-pipeline"]["uses"] == "./.github/workflows/integration-tests.yml"
    )
    assert jobs["platform-pipeline"]["strategy"]["matrix"]["platform"] == (
        "${{ fromJson(needs.discover-platforms.outputs.platforms) }}"
    )
    assert not list(WORKFLOWS.glob("integration-test-*.yml"))
    assert not (WORKFLOWS / "all-tests-common.yml").exists()


def test_platform_tests_and_manifest_use_the_resolved_source():
    caller = _workflow("ci.yml")["jobs"]["platform-pipeline"]
    assert "resolve-checkout" in caller["needs"]
    assert caller["with"]["checkout_repository"] == (
        "${{ needs.resolve-checkout.outputs.repository }}"
    )
    assert caller["with"]["checkout_ref"] == (
        "${{ needs.resolve-checkout.outputs.ref }}"
    )

    workflow = _workflow("integration-tests.yml")
    inputs = workflow[True]["workflow_call"]["inputs"]
    assert {"checkout_repository", "checkout_ref"} <= inputs.keys()
    config_call = workflow["jobs"]["configuration"]["with"]
    assert config_call["checkout_repository"] == "${{ inputs.checkout_repository }}"
    assert config_call["checkout_ref"] == "${{ inputs.checkout_ref }}"
    checkout = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )["with"]
    assert checkout["repository"] == (
        "${{ inputs.checkout_repository || github.repository }}"
    )
    assert checkout["ref"] == "${{ inputs.checkout_ref || github.ref }}"
    assert checkout["persist-credentials"] is False

    config_workflow = _workflow("load-platform-config.yml")
    config_inputs = config_workflow[True]["workflow_call"]["inputs"]
    assert {"checkout_repository", "checkout_ref"} <= config_inputs.keys()
    config_checkout = next(
        step
        for step in config_workflow["jobs"]["configuration"]["steps"]
        if step.get("name") == "Checkout source data"
    )["with"]
    assert config_checkout["repository"] == (
        "${{ inputs.checkout_repository || github.repository }}"
    )
    assert config_checkout["ref"] == "${{ inputs.checkout_ref || github.ref }}"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_dispatch_can_select_each_platform(platform):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github" / "scripts" / "ci" / "generate_matrix.py"),
            "--event-name",
            "workflow_dispatch",
            "--configs-dir",
            str(CONFIGS),
            "--dispatch-platforms",
            platform,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    # A boxing platform is one wheel built two ways, so naming it dispatches
    # both of its lanes and never only one.
    scheduled = [platform]
    config = yaml.safe_load((CONFIGS / f"{platform}.yml").read_text())
    if config.get("boxing_lane") is True:
        scheduled = [platform, f"{platform}-boxing"]
    assert json.loads(result.stdout.removeprefix("platforms=")) == scheduled


@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("result", ["success", "skipped", "failure", "cancelled"])
def test_required_gate_checks_platform_result(tmp_path, selected, result):
    gate = _workflow("ci.yml")["jobs"]["ci-complete"]
    results = {name: {"result": "success"} for name in gate["needs"]}
    results["platform-pipeline"]["result"] = result
    report = REPORTER.aggregate(gate_plan(selected), results, tmp_path, 1)
    assert (report["status"] == "passed") == (
        not selected and result in ("success", "skipped")
    )
    if selected and result == "success":
        assert any("missing platform report" in e for e in report["errors"])
    elif result not in ("success", "skipped") or (selected and result == "skipped"):
        assert any("platform-pipeline:" in e for e in report["errors"])


@pytest.mark.parametrize(
    "name",
    [
        "resolve-checkout",
        "lint",
        "agnostic-checks",
        "codegen-checks",
        "discover-platforms",
    ],
)
def test_required_gate_rejects_skipped_shared_checks(tmp_path, name):
    gate = _workflow("ci.yml")["jobs"]["ci-complete"]
    results = {job: {"result": "success"} for job in gate["needs"]}
    results[name]["result"] = "skipped"
    report = REPORTER.aggregate(gate_plan(), results, tmp_path, 1)
    assert report["status"] != "passed"
    assert any(name + ":" in e and "skipped" in e for e in report["errors"])


def _container_expression(name):
    job = next(j for j in _workflow(name)["jobs"].values() if "steps" in j)
    return job["container"]


@pytest.mark.parametrize("name", ["build-wheel-common.yml", "integration-tests.yml"])
def test_the_container_is_empty_when_the_config_names_no_image(name):
    """A platform that uses the runner's own accelerator needs no container.

    `container:` is a job field, not a step, and it cannot use `if:`. The
    expression supplies the vendor image when the config has one and `null`
    otherwise, which omits the field on a macOS runner.
    """
    expression = _container_expression(name)
    assert isinstance(expression, str), expression
    assert expression.startswith("${{")
    assert "fromJson(format(" in expression
    assert " && " in expression and " || null }}" in expression
    assert expression.count("container_volumes") == 1
    assert expression.count("container_options") == 1


def test_the_container_expression_uses_empty_config_fields_as_placeholders():
    """Every input still has a value even though its resulting object is null."""
    for config in CONFIGS.glob("*.yml"):
        if config.stem in {"template", "image_sources", "platforms"}:
            continue
        values = yaml.safe_load(config.read_text(encoding="utf-8"))
        assert isinstance(values.get("container_options"), str), config
        assert isinstance(values.get("ci_image"), str), config


def test_the_container_expression_uses_json_values_for_yaml_fields():
    """Keep the generated image and options as strings when they contain YAML syntax."""
    for name, job in (
        ("build-wheel-common.yml", "build"),
        ("integration-tests.yml", "integration-tests"),
    ):
        expression = _workflow(name)["jobs"][job]["container"]
        assert "toJson(" in expression
        assert expression.count("toJson(") == 2
        assert '"image":"{0}"' not in expression
        assert '"options":"{2}"' not in expression


def test_the_one_platform_without_a_container_is_apple():
    """An empty image is allowed only for MPS.

    A vendor config that lost its image would otherwise run outside its prepared
    container and fail on missing runtime dependencies.
    """
    empty = []
    for platform in PLATFORMS:
        config = yaml.safe_load((CONFIGS / f"{platform}.yml").read_text())
        assert "ci_image" in config, platform
        assert isinstance(config["ci_image"], str), platform
        if config["ci_image"] == "":
            empty.append(platform)
    assert empty == ["apple"]


@pytest.mark.parametrize(
    "image,accepted", [("", True), ("vendor/image", True), (None, False), (42, False)]
)
def test_the_loader_accepts_an_empty_image_but_rejects_null_or_non_strings(
    tmp_path, image, accepted
):
    import shutil

    shutil.copytree(ROOT / ".github/configs", tmp_path / ".github/configs")
    (tmp_path / ".github/scripts").mkdir()
    shutil.copyfile(
        ROOT / ".github/scripts/set_env.sh", tmp_path / ".github/scripts/set_env.sh"
    )
    shutil.copyfile(
        ROOT / ".github/integration-manifest.yml",
        tmp_path / ".github/integration-manifest.yml",
    )
    config_path = tmp_path / ".github/configs/nvidia.yml"
    config = yaml.safe_load(config_path.read_text())
    config["ci_image"] = image
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/scripts/platform_config.py"),
            "--source",
            str(tmp_path),
            "--lane",
            "nvidia",
            "--output",
            str(tmp_path / "outputs"),
        ],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is accepted, result.stderr
    if not accepted:
        assert "ci_image" in result.stderr
        assert not (tmp_path / "outputs").exists()


def test_the_apple_lane_is_locked_to_the_standard_hosted_apple_silicon_runner():
    """Pin the label with an Apple Silicon host without a paid larger runner."""
    config = yaml.safe_load((CONFIGS / "apple.yml").read_text(encoding="utf-8"))
    assert config["runner_labels"] == ["macos-15"]
    assert config["ci_image"] == ""
    assert config["environment_check"]["flagos_devices"] == {"at_least": 1}


def test_gate_import_remains_isolated_when_hardware_reporting_plugin_is_loaded(
    tmp_path,
):
    env = os.environ.copy()
    for key in (
        "CI_PLAN_CONTEXT",
        "FLAGCICD_TEST_GROUP",
        "FLAGCICD_GROUP_ID",
        "FLAGCICD_COVERAGE_DIR",
        "PYTEST_ADDOPTS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    ):
        env.pop(key, None)
    env.update(
        PYTHONPATH=str(ROOT / ".github/scripts/pytest_reporting"),
        PYTEST_PLUGINS="flagcicd_report",
        FLAGCICD_REPORT_DIR=str(tmp_path / "reports"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__)),
            "-k",
            "required_gate_rejects_skipped_shared_checks",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 passed" in result.stdout
