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


@pytest.mark.parametrize(
    "job",
    [
        "lint",
        "agnostic-checks",
        "codegen-checks",
        "discover-platforms",
        "platform-pipeline",
    ],
)
def test_shared_checks_and_platforms_follow_lifecycle_cleanup(job):
    jobs = _workflow("ci.yml")["jobs"]
    assert "ci-lifecycle" in jobs[job]["needs"]


def test_shared_checks_and_platform_queues_are_independent():
    jobs = _workflow("ci.yml")["jobs"]
    groups = {
        jobs[name]["concurrency"]["group"].split("${{")[0]
        for name in (
            "agnostic-checks",
            "codegen-checks",
            "discover-platforms",
            "platform-pipeline",
        )
    }
    assert len(groups) == 4
    assert (
        "-${{ matrix.platform }}" in jobs["platform-pipeline"]["concurrency"]["group"]
    )
    assert "max-parallel" not in jobs["platform-pipeline"]["strategy"]


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
    # A boxing platform is one wheel built two ways, so naming it dispatches
    # both of its lanes and never only one.
    scheduled = [platform]
    config = yaml.safe_load((CONFIGS / f"{platform}.yml").read_text())
    if config.get("boxing_lane") is True:
        scheduled = [platform, f"{platform}-boxing"]
    assert json.loads(result.stdout.removeprefix("platforms=")) == scheduled


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


def test_the_loader_accepts_an_empty_image_but_rejects_null_or_non_strings(tmp_path):
    """The empty string passes through; null and non-strings do not."""
    text = (WORKFLOWS / "load-platform-config.yml").read_text(encoding="utf-8")
    assert '-z "$CI_IMAGE"' not in text
    assert ".ci_image | type" in text and "!!str" in text
    condition = next(
        line.strip()
        for line in text.splitlines()
        if line.lstrip().startswith('if [[ "$CI_IMAGE" == "null"')
    )
    script = "\n".join(
        [
            "CI_IMAGE=$(yq -r '.ci_image' \"$CONFIG_FILE\")",
            f"{condition}",
            "  exit 1",
            "fi",
        ]
    )
    script_directory = tmp_path / "stub-bin"
    script_directory.mkdir()
    stub_yq = script_directory / "yq"
    stub_yq.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, yaml\n"
        "expression = sys.argv[-2]\n"
        "key = expression.split()[0].removeprefix('.')\n"
        "path = os.environ['CONFIG_FILE']\n"
        "value = yaml.safe_load(open(path))[key]\n"
        "if expression.endswith('| type'):\n"
        "    print({str: '!!str', int: '!!int'}.get(type(value), '!!null' if value is None else '!!other'))\n"
        "elif expression == '.ci_image':\n"
        "    print('null' if value is None else value)\n"
        "else:\n"
        "    raise SystemExit(f'unsupported expression: {expression}')\n",
        encoding="utf-8",
    )
    stub_yq.chmod(0o755)
    for image, accepted in (
        ('""', True),
        ('"vendor/image"', True),
        ("null", False),
        ("42", False),
    ):
        config = tmp_path / "platform.yml"
        config.write_text(f"ci_image: {image}\n", encoding="utf-8")
        env = dict(
            os.environ,
            CONFIG_FILE=str(config),
            PATH=f"{script_directory}:{os.environ['PATH']}",
        )
        run = subprocess.run(
            ["bash", "-e", "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )
        assert (run.returncode == 0) is accepted, (image, run.stdout, run.stderr)


def test_the_apple_lane_is_locked_to_the_standard_hosted_apple_silicon_runner():
    """Pin the label with an Apple Silicon host without a paid larger runner."""
    config = yaml.safe_load((CONFIGS / "apple.yml").read_text(encoding="utf-8"))
    assert config["runner_labels"] == ["macos-15"]
    assert config["ci_image"] == ""
    assert config["environment_check"]["flagos_devices"] == {"at_least": 1}
