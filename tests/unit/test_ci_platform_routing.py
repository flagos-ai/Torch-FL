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

import json
import os
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


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_required_gate_covers_matrix_pipeline():
    jobs = _workflow("ci.yml")["jobs"]
    gate = jobs["ci-complete"]
    assert set(gate["needs"]) == {
        "resolve-checkout",
        "ci-lifecycle",
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
        if step.get("uses", "").startswith("actions/checkout@")
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
    assert json.loads(result.stdout.removeprefix("platforms=")) == [platform]


@pytest.mark.parametrize("selected", [True, False])
@pytest.mark.parametrize("result", ["success", "skipped", "failure", "cancelled"])
def test_required_gate_checks_platform_result(selected, result):
    gate = _workflow("ci.yml")["jobs"]["ci-complete"]
    results = {name: {"result": "success"} for name in gate["needs"]}
    results["discover-platforms"]["outputs"] = {
        "platforms": json.dumps(PLATFORMS[:1] if selected else [])
    }
    results["platform-pipeline"]["result"] = result
    process = subprocess.run(
        ["bash", "-e", "-c", gate["steps"][0]["run"]],
        env={**os.environ, "RESULTS": json.dumps(results)},
        capture_output=True,
        text=True,
    )
    assert (process.returncode == 0) == (
        result == "success" or (not selected and result == "skipped")
    )
    if process.returncode:
        assert f"platform-pipeline: {result}" in process.stdout


@pytest.mark.parametrize(
    "name",
    [
        "resolve-checkout",
        "ci-lifecycle",
        "lint",
        "agnostic-checks",
        "codegen-checks",
        "discover-platforms",
    ],
)
def test_required_gate_rejects_skipped_shared_checks(name):
    gate = _workflow("ci.yml")["jobs"]["ci-complete"]
    results = {job: {"result": "success"} for job in gate["needs"]}
    results["discover-platforms"]["outputs"] = {"platforms": "[]"}
    results[name]["result"] = "skipped"
    process = subprocess.run(
        ["bash", "-e", "-c", gate["steps"][0]["run"]],
        env={**os.environ, "RESULTS": json.dumps(results)},
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert f"{name}: skipped" in process.stdout
