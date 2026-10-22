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

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest import mock

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
    for job_name in ("model-mount", "integration-tests"):
        steps = workflow["jobs"][job_name]["steps"]
        checkout = next(
            step
            for step in steps
            if step.get("id") == "checkout"
            and step.get("uses", "").startswith("actions/checkout@")
        )
        retries = [
            step
            for step in steps
            if step.get("name") == "Retry checkout source"
            and step.get("uses", "").startswith("actions/checkout@")
        ]
        assert len(retries) == 1
        retry = retries[0]
        assert checkout["continue-on-error"] is True
        retry_outcome = (
            steps[steps.index(retry) + 1]["if"]
            if job_name == "integration-tests"
            else retry["if"]
        )
        assert retry_outcome == "${{ steps.checkout.outcome == 'failure' }}"
        for step in (checkout, retry):
            assert step["with"]["repository"] == (
                "${{ inputs.checkout_repository || github.repository }}"
            )
            assert step["with"]["ref"] == "${{ inputs.checkout_ref || github.ref }}"
            assert step["with"]["persist-credentials"] is False

        if job_name == "integration-tests":
            verify = steps[steps.index(retry) + 1]
            assert "git rev-parse HEAD" in verify["run"]
            assert verify["env"]["EXPECTED_REF"] == (
                "${{ inputs.checkout_ref || github.sha }}"
            )

    outputs = workflow["jobs"]["model-mount"]["outputs"]
    assert set(outputs) == {"host_model_root", "model_path"}
    assert outputs["host_model_root"] == (
        "${{ steps.configure-model.outputs.host_model_root }}"
    )
    assert outputs["model_path"] == "${{ steps.configure-model.outputs.model_path }}"

    config_workflow = _workflow("load-platform-config.yml")
    config_inputs = config_workflow[True]["workflow_call"]["inputs"]
    assert {"checkout_repository", "checkout_ref"} <= config_inputs.keys()
    config_steps = config_workflow["jobs"]["configuration"]["steps"]
    config_checkout = next(
        step
        for step in config_steps
        if step.get("id") == "checkout"
        and step.get("uses", "").startswith("actions/checkout@")
    )
    config_retry = next(
        step
        for step in config_steps
        if step.get("name") == "Retry checkout source"
        and step.get("uses", "").startswith("actions/checkout@")
    )
    assert config_checkout["continue-on-error"] is True
    assert config_retry["if"] == "${{ steps.checkout.outcome == 'failure' }}"
    for step in (config_checkout, config_retry):
        assert step["with"]["repository"] == (
            "${{ inputs.checkout_repository || github.repository }}"
        )
        assert step["with"]["ref"] == "${{ inputs.checkout_ref || github.ref }}"


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


def _integration_jobs():
    return _workflow("integration-tests.yml")["jobs"]


def _integration_steps(name):
    jobs = _workflow(name)["jobs"]
    return {
        step.get("id", step.get("name")): step
        for step in jobs["integration-tests"]["steps"]
    }


@pytest.mark.parametrize("name", ["build-wheel-common.yml", "integration-tests.yml"])
def test_the_container_is_empty_when_the_config_names_no_image(name):
    """MPS runs natively; Linux starts exactly one vendor container."""
    if name == "integration-tests.yml":
        steps = _integration_steps(name)
        assert "docker run" not in steps["integration-tests"]["run"]
        assert "docker run" not in steps["workspace"]["run"]
        assert (
            ".github/scripts/container_workspace.py start" in steps["container"]["run"]
        )
        assert steps["container"]["if"] == (
            "${{ needs.configuration.outputs.ci_image != '' }}"
        )
        return
    expression = _workflow(name)["jobs"]["build"]["container"]
    assert isinstance(expression, str), expression
    assert expression.startswith("${{")
    assert "fromJson(format(" in expression
    assert " && " in expression and " || null }}" in expression
    assert expression.count("container_volumes") == 1
    assert expression.count("container_options") == 1


def test_the_container_expression_uses_json_values_for_yaml_fields():
    """Keep the generated image and options as strings when they contain YAML syntax."""
    expression = _workflow("build-wheel-common.yml")["jobs"]["build"]["container"]
    assert "toJson(" in expression
    assert expression.count("toJson(") == 2
    assert '"image":"{0}"' not in expression
    assert '"options":"{2}"' not in expression
    workspace = _integration_steps("integration-tests.yml")["workspace"]
    assert "docker image inspect --format" in workspace["run"]
    assert "CONTAINER_VOLUMES" in workspace["env"]


def test_every_platform_uses_the_same_workspace_and_container_boundary():
    """The model and checkout are prepared on the host before Docker starts."""
    jobs = _integration_jobs()
    integration = jobs["integration-tests"]
    steps = integration["steps"]
    names = [step.get("name", "") for step in steps]
    assert "container" not in integration
    assert integration["needs"] == ["configuration", "model-mount"]
    assert "container" not in jobs["model-mount"]
    assert names.index("Prepare the integration workspace") < names.index(
        "Start the platform command environment"
    )
    workspace = _integration_steps("integration-tests.yml")["workspace"]
    assert "container_workspace.py prepare" in workspace["run"]
    assert "always()" not in str(workspace.get("if", ""))
    for platform in PLATFORMS:
        config = yaml.safe_load((CONFIGS / f"{platform}.yml").read_text())
        assert isinstance(config.get("container_volumes"), list), platform
        if platform == "mps":
            assert config["ci_image"] == ""
        else:
            assert config["ci_image"]
    for name in (
        "build",
        "validate",
        "install",
        "integration-tests",
    ):
        step = _integration_steps("integration-tests.yml")[name]
        assert 'docker exec "$name" --env HOME=/tmp bash -e -c' in step["run"]
        assert 'source "$INTEGRATION_PROFILE"' in step["run"]


def test_the_container_expression_uses_empty_config_fields_as_placeholders():
    """Every input still has a value even though its resulting object is null."""
    for config in CONFIGS.glob("*.yml"):
        if config.stem in {"template", "image_sources", "platforms"}:
            continue
        values = yaml.safe_load(config.read_text(encoding="utf-8"))
        assert isinstance(values.get("container_options"), str), config
        assert isinstance(values.get("ci_image"), str), config


def test_the_one_platform_without_a_container_is_mps():
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
    assert empty == ["mps"]


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


def test_mps_is_locked_to_the_standard_hosted_apple_silicon_runner():
    """Pin the label with an Apple Silicon host without a paid larger runner."""
    config = yaml.safe_load((CONFIGS / "mps.yml").read_text(encoding="utf-8"))
    assert config["runner_labels"] == ["macos-15"]
    assert config["ci_image"] == ""
    assert config["environment_check"]["flagos_devices"] == {"at_least": 1}


def _load_ci_script(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workspace_copies_the_checkout_and_replaces_the_model_mount(
    tmp_path, monkeypatch
):
    host_env = tmp_path / "host-env"
    monkeypatch.setenv("INTEGRATION_PLATFORM", "cuda")
    monkeypatch.setenv("GITHUB_ENV", str(host_env))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "host-output"))
    monkeypatch.setenv("GITHUB_PATH", str(tmp_path / "host-path"))
    monkeypatch.delenv("INTEGRATION_PARENT_PATH", raising=False)
    monkeypatch.setenv("CHECKOUT_REPOSITORY", "flagos-ai/Torch-FL")
    monkeypatch.setenv("CHECKOUT_REF", "a" * 40)
    host_env.write_text("BASE=1\n")
    (tmp_path / "host-output").touch()
    (tmp_path / "host-path").write_text("/base/bin\n")
    workspace_script = _load_ci_script(
        ".github/scripts/container_workspace.py", "ci_container_workspace"
    )
    source = tmp_path / "checkout"
    (source / "nested").mkdir(parents=True)
    (source / "nested/source.py").write_text("source = True\n")
    model_root = tmp_path / "models"
    (model_root / "Qwen3-0.6B").mkdir(parents=True)
    settings = workspace_script.prepare(
        argparse.Namespace(
            checkout=str(source),
            runner_temp=str(tmp_path / "runner"),
            model_root=str(model_root),
            model_path="/var/lib/torch-fl-ci/models/Qwen3-0.6B",
            image="vendor/image:tag",
            digest="vendor/image@sha256:" + "a" * 64,
            volumes='["/models:/models:ro", "/runtime:/runtime", "/var/lib/torch-fl-ci/models:/old/models:ro"]',
            options="",
        )
    )
    config = json.loads(Path(settings["state"], "config.json").read_text())
    assert config["cache"] == str(tmp_path / "runner/torch-fl-pip")
    assert config["platform"] == "cuda"
    assert Path(settings["workspace"], "nested/source.py").is_file()
    environment = Path(config["environment"]).read_text()
    model_path = shlex.split(environment.split("MODEL_PATH=", 1)[1].splitlines()[0])[0]
    assert model_path == str(model_root / "Qwen3-0.6B")
    for value in config["inline_environment"]:
        assert f"{value}=" in Path(config["environment"]).read_text(), value
        assert f"{value}=" in host_env.read_text(), value
    inline = {
        line.split("=", 1)[0]
        for line in host_env.read_text().splitlines()
        if "=" in line
    }
    assert workspace_script.INLINE_ENVIRONMENT <= inline
    assert (
        f"RUNNER_TEMP={tmp_path / 'runner'}" in Path(config["environment"]).read_text()
    )
    volumes = workspace_script._container_volumes(config["volumes"], config)
    assert "/models:/models:ro" not in volumes
    assert f"{model_root}:/var/lib/torch-fl-ci/models:ro" in volumes
    expected = {
        "INTEGRATION_MODEL": "/var/lib/torch-fl-ci/models/Qwen3-0.6B",
        "MODEL_PATH": str(model_root / "Qwen3-0.6B"),
    }
    for name, value in expected.items():
        line = next(item for item in environment.splitlines() if name + "=" in item)
        assert shlex.split(line)[0] == f"{name}={value}"
    native = workspace_script.prepare(
        argparse.Namespace(
            checkout=str(source),
            runner_temp=str(tmp_path / "native"),
            model_root=str(model_root),
            model_path="/var/lib/torch-fl-ci/models/Qwen3-0.6B",
            image="",
            digest="",
            volumes="[]",
            options="",
        )
    )
    assert native["host_model_path"] == str(model_root / "Qwen3-0.6B")
    missing = tmp_path / "missing-model"
    missing.mkdir()
    with pytest.raises(workspace_script.WorkspaceError, match="prepared checkpoint"):
        workspace_script.prepare(
            argparse.Namespace(
                checkout=str(source),
                runner_temp=str(tmp_path / "missing-runner"),
                model_root=str(missing),
                model_path="/var/lib/torch-fl-ci/models/Qwen3-0.6B",
                image="",
                digest="",
                volumes="[]",
                options="",
            )
        )
    with pytest.raises(workspace_script.WorkspaceError, match="inside the checkout"):
        workspace_script.prepare(
            argparse.Namespace(
                checkout=str(source),
                runner_temp=str(source),
                model_root=str(model_root),
                model_path="/var/lib/torch-fl-ci/models/Qwen3-0.6B",
                image="",
                digest="",
                volumes="[]",
                options="",
            )
        )


def test_environment_hooks_keep_each_commands_additions_once_and_failures(tmp_path):
    """Replay two container hooks with one inherited environment each."""
    hook_module = _load_ci_script(
        ".github/scripts/publish_container_environment.py", "ci_container_hook"
    )
    root = tmp_path / "hook"
    state, workspace, runtime = (
        root / name for name in ("state", "workspace", "runtime")
    )
    for directory in (state, workspace, runtime):
        directory.mkdir(parents=True)
    host_env, host_path = root / "host-env", root / "host-path"
    host_env.write_text("BASE=1\n")
    host_path.write_text("/base/bin\n")
    container_env, container_path = root / "container-environment", root / "path"
    container_env.write_text("BASE=1\n")
    container_path.write_text("/base/bin\n")
    settings = {
        "state": str(state),
        "workspace": str(workspace),
        "runtime": str(runtime),
        "environment": str(container_env),
        "path": str(container_path),
        "host_files": {"GITHUB_ENV": str(host_env), "GITHUB_PATH": str(host_path)},
        "image": "vendor/image",
        "container": "container",
        "shell": str(state / "shell.sh"),
    }
    shell = state / "shell.sh"
    shell.write_text(
        "#!/bin/sh\n"
        "set -euo pipefail\n"
        'test "$INTEGRATION_PROFILE" = ' + str(state / "profile.sh") + "\n"
        "export GITHUB_ENV=" + str(state / "command-env") + " "
        "GITHUB_PATH=" + str(state / "command-path") + "\n"
        'exec bash "$@"\n'
    )
    shell.chmod(0o755)
    (state / "profile.sh").write_text(
        'if [[ -n "${INTEGRATION_PARENT_PATH:-}" ]]; then\n'
        '  PATH="$INTEGRATION_PARENT_PATH"\n'
        "fi\n"
        'if [[ -s "$GITHUB_ENV" ]]; then\n'
        "  set -a\n"
        '  . "$GITHUB_ENV"\n'
        "  set +a\n"
        "fi\n"
        'if [[ -s "$GITHUB_PATH" ]]; then\n'
        "  while IFS= read -r entry; do\n"
        '    [[ -n "$entry" ]] && export PATH="$entry:$PATH"\n'
        '  done < "$GITHUB_PATH"\n'
        "fi\n"
    )
    (workspace / "second.sh").write_text(
        "#!/usr/bin/env bash\n"
        'test "${HOOKED:-}" = 1\n'
        'test "${FIRST:-}" = 1\n'
        'case ":$PATH:" in *:/vendor/bin:*) ;; *) exit 8 ;; esac\n'
        "printf 'SECOND=1\\n' >> \"$GITHUB_ENV\"\n"
    )
    (workspace / "hook.sh").write_text(
        "#!/bin/sh\n"
        "set -euo pipefail\n"
        'test "$GITHUB_WORKSPACE" = ' + shlex.quote(str(workspace)) + "\n"
        'test "$INTEGRATION_PROFILE" = ' + shlex.quote(str(state / "profile.sh")) + "\n"
        "printf 'HOOKED=1\\nFIRST=1\\n' >> \"$GITHUB_ENV\"\n"
        "printf '/vendor/bin\\n' >> \"$GITHUB_PATH\"\n"
        "exit 7\n"
    )
    environment = {
        "HOME": "/tmp",
        "GITHUB_WORKSPACE": str(workspace),
        "GITHUB_ENV": str(container_env),
        "GITHUB_PATH": str(container_path),
        "INTEGRATION_PROFILE": str(state / "profile.sh"),
        "INTEGRATION_RUNTIME": str(runtime),
        "PATH": os.environ["PATH"],
    }
    command_environment = {
        **environment,
        "GITHUB_ENV": str(state / "command-env"),
        "GITHUB_PATH": str(state / "command-path"),
    }
    for path in (state / "command-env", state / "command-path"):
        path.write_bytes(
            (
                container_env if path.name == "command-env" else container_path
            ).read_bytes()
        )
    commands = []
    original_run = subprocess.run

    def execute(command, **kwargs):
        commands.append(command)
        if command[:3] == ["bash", "-E", "-e"]:
            return original_run(command, **kwargs)

        assert command[:2] == ["docker", "exec"]
        # Run the mapped command files exactly as the container shell would.
        entrypoint = [command[command.index("bash")], "--norc"]
        arguments = command[command.index("bash") + 2 :]
        return original_run(
            [
                *entrypoint,
                "-E",
                "-e",
                "-u",
                "-o",
                "pipefail",
                *arguments,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=command_environment,
            check=False,
        )

    with mock.patch.object(hook_module.subprocess, "run", side_effect=execute):
        assert hook_module.hook(settings, "hook.sh") == 7
        assert hook_module.hook(settings, "second.sh") == 0
    assert len(commands) == 2
    assert all(command[:2] == ["docker", "exec"] for command in commands)
    assert host_env.read_text().splitlines() == [
        "BASE=1",
        "HOOKED=1",
        "FIRST=1",
        "SECOND=1",
    ]
    assert container_env.read_text() == host_env.read_text()
    assert host_path.read_text().splitlines() == ["/base/bin", "/vendor/bin"]
    assert (state / "env.command").read_text().splitlines() == [
        "BASE=1",
        "HOOKED=1",
        "FIRST=1",
        "SECOND=1",
    ]
    assert (state / "path.command").read_text() == host_path.read_text()


def test_unqualified_image_tags_resolve_to_their_repository_digest():
    """Docker reports an unqualified pull by repository, not by tag.

    The PPU image is configured as ``registry/image:tag`` yet Docker's
    RepoDigests entry omits the tag. Requiring a literal match stopped the job
    before its device was inspected; the resolver must accept that repository
    digest without accepting a different repository.
    """
    workflow = _workflow("integration-tests.yml")
    run = next(
        step
        for step in workflow["jobs"]["integration-tests"]["steps"]
        if step.get("id") == "workspace"
    )["run"]
    resolver = run.split("python3 -c '", 1)[1].split('\' "$image")', 1)[0]
    resolver = "\n".join(
        line[10:] if line.startswith("          ") else line
        for line in resolver.splitlines()
    )
    digest = "sha256:" + "a" * 64
    image = "harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260811-ppu-dev"
    repository = image.rsplit(":", 1)[0]
    misleading = "harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl-misleading@" + digest
    result = subprocess.run(
        [sys.executable, "-c", resolver, image],
        input=json.dumps([misleading, f"{repository}@{digest}"]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{repository}@{digest}"
