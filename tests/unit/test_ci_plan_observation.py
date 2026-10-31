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

"""Verify final plans against routing, composition and real execution boundaries."""

import copy
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "observed_plan", SCRIPTS / "flagcicd_plan.py"
    )
    PLANNER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PLANNER)
finally:
    sys.path.pop(0)
CONSUMER = sys.modules["platform_config"]
CONTRACT = yaml.safe_load((ROOT / ".flagcicd/project.yaml").read_text())


def request(**updates):
    return {
        "source_repository": "owner/fork",
        "source_sha": "a" * 40,
        "control_sha": "c" * 40,
        "base_sha": "b" * 40,
        "requested_level": "auto",
        "platforms": "all",
        "task_id": "",
        "parent_run_id": "",
        **updates,
    }


@pytest.mark.parametrize("profile", ["pr", "nightly"])
def test_every_planned_lane_matches_the_existing_loader_cli(tmp_path, profile):
    plan = PLANNER.build_plan(request(profile=profile), CONTRACT, ROOT, None)
    assert [lane["name"] for lane in plan["execution_plan"]] == plan[
        "expected_platforms"
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            yaml.safe_load((ROOT / ".github/integration-manifest.yml").read_text())
        )
    )
    config_file = tmp_path / "config.json"
    for lane in plan["execution_plan"]:
        config = yaml.safe_load(
            (ROOT / f".github/configs/{lane['platform']}.yml").read_text()
        )
        boxing = lane["lane"] == "boxing"
        if boxing:
            config["display_name"] += " (CUDA boxing)"
        config_file.write_text(json.dumps(config))
        # These are the exact CLI inputs supplied by load-platform-config.yml.
        command = [
            sys.executable,
            str(SCRIPTS / "compose_integration_manifest.py"),
            "--platform",
            lane["platform"],
            "--profile",
            profile,
            "--manifest",
            str(manifest),
            "--platform-config",
            str(config_file),
            "--config-dir",
            str(ROOT / ".github/configs"),
            "--environment-check",
            json.dumps(
                config.get(
                    "boxing_environment_check" if boxing else "environment_check"
                )
            ),
        ]
        if boxing:
            command.append("--boxing-lane")
        expected = json.loads(
            subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
        )
        assert [
            {k: v for k, v in g.items() if k != "group_id"} for g in lane["groups"]
        ] == expected
        ids = [group["group_id"] for group in lane["groups"]]
        assert ids == [group["manifest_id"] for group in expected]
        assert len(ids) == len(set(ids))
        assert lane["environment"] == config.get("integration_environment", {})


@pytest.mark.parametrize(
    "event,changed,labels,dispatch,reason",
    [
        ("pull_request", ["README.md"], [], "", "documentation_or_empty_diff"),
        ("pull_request", None, [], "", "diff_unavailable"),
        ("pull_request", ["unknown/runtime.cc"], [], "", "unmatched_path"),
        (
            "pull_request",
            ["csrc/aten/backends/mthreads/mudnn_copy.cc"],
            [],
            "",
            "watch_paths",
        ),
        ("pull_request", ["README.md"], ["platform:nvidia"], "", "labels"),
        ("pull_request", ["README.md"], ["platform:unknown"], "", "labels"),
        ("workflow_dispatch", ["README.md"], [], "hygon-boxing", "dispatch"),
        ("push", [], [], "", "full_event"),
        ("schedule", [], [], "", "full_event"),
    ],
)
def test_observation_preserves_the_router_matrix(
    event, changed, labels, dispatch, reason
):
    command = [
        sys.executable,
        str(SCRIPTS / "ci/generate_matrix.py"),
        "--event-name",
        event,
        "--configs-dir",
        str(ROOT / ".github/configs"),
        "--pr-labels",
        ",".join(labels),
        "--dispatch-platforms",
        dispatch,
    ]
    if changed is not None:
        command += ["--changed-files", "\n".join(changed)]
    else:
        command += ["--base-sha", "missing-comparison-commit"]
    expected = json.loads(
        subprocess.check_output(
            command, text=True, stderr=subprocess.PIPE
        ).removeprefix("platforms=")
    )
    plan = PLANNER.build_ci_plan(
        request(), CONTRACT, ROOT, changed, event, labels, dispatch
    )
    assert plan["expected_platforms"] == expected
    assert plan["selection_reasons"][-1]["kind"] == reason
    assert plan["expected_shared_checks"] == [
        "resolve-checkout",
        "lint",
        "agnostic-checks",
        "codegen-checks",
        "discover-platforms",
    ]


def test_ci_documentation_does_not_inherit_external_l2_fallback():
    req = request(requested_level="L2")
    observed = PLANNER.build_ci_plan(req, CONTRACT, ROOT, ["README.md"], "pull_request")
    assert observed["expected_platforms"] == observed["execution_plan"] == []
    assert observed["test_level"] == "L1"
    assert PLANNER.build_plan(req, CONTRACT, ROOT, ["README.md"])["expected_platforms"]


def test_plan_hash_is_deterministic_and_ignores_run_and_task_identity():
    first = PLANNER.build_plan(request(), CONTRACT, ROOT, ["setup.py"])
    second = PLANNER.build_plan(
        request(
            run_id="99", run_attempt="3", task_id="another-task", parent_run_id="88"
        ),
        CONTRACT,
        ROOT,
        ["setup.py"],
    )
    assert first["plan_hash"] == second["plan_hash"]
    first["execution_plan"][0]["environment"] = {"A": "1", "B": "2"}
    second["execution_plan"][0]["environment"] = {"B": "2", "A": "1"}
    assert PLANNER.plan_hash(second) == PLANNER.plan_hash(first)


@pytest.mark.parametrize(
    "field",
    ["source_sha", "control_sha", "base_sha", "diff_head_sha", "source_repository"],
)
def test_source_and_control_identity_change_the_hash(field):
    first = PLANNER.build_plan(request(), CONTRACT, ROOT, ["setup.py"])
    second = PLANNER.build_plan(
        request(**{field: "d" * 40}), CONTRACT, ROOT, ["setup.py"]
    )
    assert first["plan_hash"] != second["plan_hash"]


@pytest.mark.parametrize(
    "field", ["command", "environment", "min_passed", "order", "shared_environment"]
)
def test_final_execution_content_is_part_of_the_hash(field):
    plan = PLANNER.build_plan(
        request(platforms="nvidia", requested_level="L2"), CONTRACT, ROOT, []
    )
    altered = copy.deepcopy(plan)
    lane = altered["execution_plan"][0]
    if field == "order":
        lane["groups"].reverse()
    elif field == "shared_environment":
        lane["environment"]["OBSERVED_VALUE"] = "changed"
    elif field == "min_passed":
        lane["groups"][0][field] = 100
    else:
        lane["groups"][0][field] = (
            {"OBSERVED_VALUE": "changed"} if field == "environment" else "changed"
        )
    assert PLANNER.plan_hash(altered) != plan["plan_hash"]


def test_large_execution_plan_stays_out_of_the_dispatch_context():
    plan = PLANNER.build_plan(
        request(run_id="42", run_attempt="1"), CONTRACT, ROOT, None
    )
    context = json.loads(PLANNER.report_context(plan))
    assert "execution_plan" not in context
    assert "selection_reasons" not in context
    assert context["plan_hash"] == plan["plan_hash"]


@pytest.mark.parametrize(
    "labels,selected", [([], []), (["platform:nvidia"], ["nvidia", "nvidia-boxing"])]
)
def test_real_plan_cli_records_identity_and_supplies_the_matrix(
    tmp_path, labels, selected
):
    sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    output = tmp_path / "plan.json"
    env = {
        **os.environ,
        "SOURCE_REPOSITORY": "owner/fork",
        "EVENT_NAME": "pull_request",
        "BASE_SHA": sha,
        "HEAD_SHA": sha,
        "PR_LABELS_JSON": json.dumps(labels),
        "TEST_PROFILE": "pr",
        "DISPATCH_PLATFORMS": "",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "flagcicd_plan.py"),
            "--ci",
            "--source",
            str(ROOT),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(output.read_text())
    assert plan["source_sha"] == plan["control_sha"] == sha
    assert not plan.get("observation_only", False)
    assert [lane["name"] for lane in plan["execution_plan"]] == selected
    outputs = dict(
        line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines()
    )
    assert json.loads(outputs["platforms"]) == selected
    assert outputs["artifact_name"] == "flagcicd-plan-1"
    assert "validated_platforms" not in outputs


def test_one_required_plan_supplies_the_matrix_and_execution():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    assert "observe-plan" not in jobs
    steps = jobs["discover-platforms"]["steps"]
    plan = next(s for s in steps if s.get("id") == "detect")
    assert "flagcicd_plan.py" in plan["run"]
    assert "generate_matrix.py" not in plan["run"]
    assert "SELECTED_PLATFORMS" not in plan["env"]
    assert "#" not in plan["run"]
    assert "continue-on-error" not in jobs["discover-platforms"]
    for consumer in ("platform-pipeline", "ci-complete"):
        assert "discover-platforms" in jobs[consumer]["needs"]
    inputs = jobs["platform-pipeline"]["with"]
    assert (
        inputs["execution_plan_artifact"]
        == "${{ needs.discover-platforms.outputs.plan_artifact || 'missing-required-execution-plan' }}"
    )
    assert (
        inputs["execution_plan_context"]
        == "${{ needs.discover-platforms.outputs.context }}"
    )
    assert "flagcicd_plan" not in inputs


def test_observer_tests_are_conditional_for_older_source_checkouts():
    steps = yaml.safe_load(
        (ROOT / ".github/workflows/agnostic-checks.yml").read_text()
    )["jobs"]["agnostic-checks"]["steps"]
    calls = [
        s for s in steps if "tests/unit/test_ci_plan_observation.py" in s.get("run", "")
    ]
    assert len(calls) == 1
    assert calls[0]["if"] == "hashFiles('tests/unit/test_ci_plan_observation.py') != ''"


def test_observer_diffs_the_pr_head_but_records_the_tested_merge_commit(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(ROOT / ".github/configs", source / ".github/configs")
    shutil.copyfile(
        ROOT / ".github/integration-manifest.yml",
        source / ".github/integration-manifest.yml",
    )

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                *args,
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("checkout", "-qb", "pr")
    (source / "README.md").write_text("documentation")
    git("add", ".")
    git("commit", "-qm", "documentation PR")
    head = git("rev-parse", "HEAD")
    git("checkout", "-q", "main")
    (source / "runtime.cc").write_text("upstream runtime")
    git("add", ".")
    git("commit", "-qm", "upstream runtime")
    git("merge", "--no-ff", "-qm", "merge PR", "pr")
    merged = git("rev-parse", "HEAD")
    output = tmp_path / "plan.json"
    env = {
        **os.environ,
        "SOURCE_REPOSITORY": "owner/base",
        "EVENT_NAME": "pull_request",
        "BASE_SHA": base,
        "HEAD_SHA": head,
        "PR_LABELS_JSON": "[]",
        "TEST_PROFILE": "pr",
        "DISPATCH_PLATFORMS": "",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "flagcicd_plan.py"),
            "--ci",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(output.read_text())
    assert plan["source_sha"] == merged != head
    assert plan["diff_head_sha"] == head
    assert plan["changed_files"] == ["README.md"]
    assert plan["execution_plan"] == []


def test_ci_nightly_is_full_and_weekly_stays_reserved():
    plan = PLANNER.build_ci_plan(
        request(profile="nightly"), CONTRACT, ROOT, [], "schedule"
    )
    configs = ROOT / ".github/configs"
    known = PLANNER._load_platforms(configs)
    assert len(plan["execution_plan"]) == len(known) + len(
        PLANNER.boxing_platforms(configs, known)
    )
    with pytest.raises(ValueError, match="weekly is reserved"):
        PLANNER.build_ci_plan(request(profile="weekly"), CONTRACT, ROOT, [], "schedule")
    with pytest.raises(ValueError, match="nightly requires platforms=all"):
        PLANNER.build_ci_plan(
            request(profile="nightly"),
            CONTRACT,
            ROOT,
            [],
            "workflow_dispatch",
            dispatch="nvidia",
        )


def executable_plan(monkeypatch, **updates):
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    return PLANNER.build_plan(
        request(
            platforms="nvidia",
            requested_level="L2",
            run_id="42",
            run_attempt="1",
            **updates,
        ),
        CONTRACT,
        ROOT,
        [],
    )


@pytest.mark.parametrize("profile", ["pr", "nightly"])
def test_all_snapshot_configuration_outputs_match_platform_data(
    monkeypatch, tmp_path, profile
):
    plan = PLANNER.build_plan(
        request(profile=profile, run_id="42", run_attempt="1"), CONTRACT, ROOT, None
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    context = json.loads(PLANNER.report_context(plan))
    for lane in plan["execution_plan"]:
        config = yaml.safe_load(
            (ROOT / f".github/configs/{lane['platform']}.yml").read_text()
        )
        boxing = lane["lane"] == "boxing"
        consumed = CONSUMER.consume_plan(
            plan, context, lane["name"], profile, plan["source_sha"]
        )
        outputs = consumed["configuration"]
        assert outputs == {
            "platform": lane["platform"],
            "boxing_lane": "1" if boxing else "",
            "display_name": config["display_name"]
            + (" (CUDA boxing)" if boxing else ""),
            "setup_script": config["setup_script"],
            "setup_platform": config["setup_platform"],
            "setup_stage": config.get("setup_stage", "integration"),
            "ci_image": config["ci_image"],
            "runner_labels": config["runner_labels"],
            "container_volumes": config["container_volumes"],
            "container_options": config["container_options"],
            "timeout_minutes": str(config["timeout_minutes"]),
            "integration_timeout_minutes": str(
                config.get("integration_timeout_minutes", config["timeout_minutes"])
            ),
            "wheel_artifact_name": config["wheel_artifact_name"]
            + ("-boxing" if boxing else ""),
            "integration_build_tools": "\n".join(
                config.get("integration_build_tools", [])
            ),
            "integration_test_dependencies": "\n".join(
                config.get("integration_test_dependencies", [])
            ),
        }
        path = tmp_path / "outputs"
        path.unlink(missing_ok=True)
        CONSUMER.emit_outputs(consumed, path)
        emitted = parse_outputs(path)
        assert json.loads(emitted["integration_tests"]) == lane["groups"]
        assert json.loads(emitted["integration_environment"]) == lane["environment"]
        assert json.loads(emitted["runner_labels"]) == outputs["runner_labels"]
        assert emitted["integration_build_tools"] == outputs["integration_build_tools"]
        assert emitted["setup_platform"] == outputs["setup_platform"]
        standalone = tmp_path / "standalone-outputs"
        standalone.unlink(missing_ok=True)
        run = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "platform_config.py"),
                "--source",
                str(ROOT),
                "--lane",
                lane["name"],
                "--profile",
                profile,
                "--output",
                str(standalone),
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        assert parse_outputs(standalone) == emitted


def parse_outputs(path):
    lines = iter(path.read_text().splitlines())
    values = {}
    for line in lines:
        key, delimiter = line.split("<<", 1)
        value = []
        for line in lines:
            if line == delimiter:
                break
            value.append(line)
        values[key] = "\n".join(value)
    return values


@pytest.mark.parametrize(
    "mutation",
    [
        "command",
        "order",
        "environment",
        "runner",
        "profile",
        "sha",
        "hash",
        "control",
        "run",
        "attempt",
        "unplanned",
        "observation",
        "duplicate",
    ],
)
def test_invalid_plan_never_falls_back_to_composition(monkeypatch, mutation):
    plan = executable_plan(monkeypatch)
    context = json.loads(PLANNER.report_context(plan))
    lane, profile, sha = "nvidia", "pr", plan["source_sha"]
    if mutation == "command":
        plan["execution_plan"][0]["groups"][0]["command"] = "exit 0"
    elif mutation == "order":
        plan["execution_plan"][0]["groups"].reverse()
    elif mutation == "environment":
        plan["execution_plan"][0]["environment"]["MODEL_PATH"] = "/other"
    elif mutation == "runner":
        plan["execution_plan"][0]["configuration"]["runner_labels"] = ["other"]
    elif mutation == "profile":
        profile = "nightly"
    elif mutation == "sha":
        sha = "d" * 40
    elif mutation == "hash":
        context["plan_hash"] = "0" * 64
    elif mutation == "control":
        context["control_sha"] = "d" * 40
    elif mutation == "run":
        monkeypatch.setenv("GITHUB_RUN_ID", "43")
    elif mutation == "attempt":
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "0")
    elif mutation == "unplanned":
        lane = "apple"
    elif mutation == "observation":
        plan["observation_only"] = True
    elif mutation == "duplicate":
        plan["execution_plan"].append(copy.deepcopy(plan["execution_plan"][0]))
        plan["plan_hash"] = context["plan_hash"] = PLANNER.plan_hash(plan)
    monkeypatch.setattr(
        CONSUMER,
        "compose",
        lambda *a, **k: pytest.fail("must not recompose a planned lane"),
    )
    with pytest.raises(ValueError):
        CONSUMER.consume_plan(plan, context, lane, profile, sha)


def test_failed_job_rerun_can_consume_the_exact_original_plan(monkeypatch):
    plan = executable_plan(monkeypatch)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    context = json.loads(PLANNER.report_context(plan))
    assert (
        CONSUMER.consume_plan(plan, context, "nvidia", "pr", plan["source_sha"])
        == plan["execution_plan"][0]
    )


def test_real_consumer_cli_executes_ordered_groups_without_yaml_or_composition(
    tmp_path, monkeypatch
):
    sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    plan = executable_plan(monkeypatch, source_sha=sha, control_sha=sha)
    lane = plan["execution_plan"][0]
    lane["environment"] = {"SHARED": "from-plan"}
    lane["groups"] = [
        {
            "name": "first",
            "manifest_id": "first",
            "group_id": "first",
            "command": 'printf "%s:%s\\n" "$SHARED" "$GROUP" >> "$TRACE"',
            "environment": {"GROUP": "one"},
        },
        {
            "name": "second",
            "manifest_id": "second",
            "group_id": "second",
            "command": 'printf "%s:%s\\n" "$SHARED" "$GROUP" >> "$TRACE"',
            "environment": {"GROUP": "two"},
        },
    ]
    plan["plan_hash"] = PLANNER.plan_hash(plan)
    artifact = tmp_path / "plan.json"
    artifact.write_text(json.dumps(plan))
    output = tmp_path / "outputs"
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "yaml.py").write_text('raise RuntimeError("consumer must not read YAML")')
    env = {
        **os.environ,
        "PLAN_CONTEXT": PLANNER.report_context(plan),
        "PYTHONPATH": str(shadow),
    }
    command = [
        sys.executable,
        str(SCRIPTS / "platform_config.py"),
        "--source",
        str(ROOT),
        "--lane",
        "nvidia",
        "--plan",
        str(artifact),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = parse_outputs(output)
    trace = tmp_path / "trace"
    run = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_integration_tests.py")],
        env={
            **env,
            "INTEGRATION_TESTS": values["integration_tests"],
            "INTEGRATION_ENVIRONMENT": values["integration_environment"],
            "TRACE": str(trace),
        },
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert trace.read_text().splitlines() == ["from-plan:one", "from-plan:two"]
    output.unlink()
    plan["control_sha"] = "d" * 40
    plan["plan_hash"] = PLANNER.plan_hash(plan)
    artifact.write_text(json.dumps(plan))
    env["PLAN_CONTEXT"] = PLANNER.report_context(plan)
    wrong_control = subprocess.run(command, env=env, capture_output=True, text=True)
    assert wrong_control.returncode != 0
    assert "control_sha differs" in wrong_control.stderr
    assert not output.exists()
    artifact.unlink()
    missing = subprocess.run(command, env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("runner_labels", []),
        ("runner_labels", [42]),
        ("runner_labels", [""]),
        ("runner_labels", [" \t"]),
        ("runner_labels", ["torch-fl-nv", ""]),
        ("container_volumes", None),
        ("integration_build_tools", [42]),
        ("timeout_minutes", True),
        ("integration_timeout_minutes", 0),
        ("setup_stage", "other"),
        ("setup_script", "missing.sh"),
        ("integration_tests", []),
        ("ci_image", None),
    ],
)
def test_l0_configuration_errors_fail_during_planning(tmp_path, key, value):
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
    config[key] = value
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises((ValueError, SystemExit)):
        PLANNER.build_plan(request(), CONTRACT, tmp_path, ["README.md"])


def test_both_entries_forward_one_plan_to_the_shared_configuration_consumer():
    entry = yaml.safe_load((ROOT / ".github/workflows/flagcicd-entry.yml").read_text())[
        "jobs"
    ]
    assert "preflight" not in entry
    assert "validated_platforms" not in entry["plan"]["outputs"]
    inputs = entry["platforms"]["with"]
    assert (
        inputs["execution_plan_artifact"]
        == "${{ needs.plan.outputs.plan_artifact || 'missing-required-execution-plan' }}"
    )
    assert inputs["execution_plan_context"] == "${{ needs.plan.outputs.context }}"
    assert "flagcicd_plan" not in inputs
    integration = yaml.safe_load(
        (ROOT / ".github/workflows/integration-tests.yml").read_text()
    )["jobs"]
    for key in ("execution_plan_artifact", "execution_plan_context"):
        assert integration["configuration"]["with"][key] == "${{ inputs." + key + " }}"
    loader = yaml.safe_load(
        (ROOT / ".github/workflows/load-platform-config.yml").read_text()
    )["jobs"]["configuration"]
    download = next(
        s for s in loader["steps"] if s.get("name") == "Download execution plan"
    )
    assert "continue-on-error" not in download
    assert download["with"]["name"] == "${{ inputs.execution_plan_artifact }}"
    script = next(s["run"] for s in loader["steps"] if s.get("id") == "config")
    assert "platform_config.py" in script
    assert "yq" not in script and "compose_integration_manifest.py" not in script
    install = next(
        s for s in loader["steps"] if s.get("name") == "Install configuration parser"
    )
    assert install["if"] == "inputs.execution_plan_artifact == ''"


def test_null_optional_configuration_fields_keep_the_loader_defaults(tmp_path):
    shutil.copytree(ROOT / ".github/configs", tmp_path / ".github/configs")
    (tmp_path / ".github/scripts").mkdir()
    shutil.copyfile(
        ROOT / ".github/scripts/set_env.sh", tmp_path / ".github/scripts/set_env.sh"
    )
    shutil.copyfile(
        ROOT / ".github/integration-manifest.yml",
        tmp_path / ".github/integration-manifest.yml",
    )
    path = tmp_path / ".github/configs/nvidia.yml"
    config = yaml.safe_load(path.read_text())
    for key in (
        "setup_stage",
        "integration_timeout_minutes",
        "integration_build_tools",
        "integration_test_dependencies",
        "integration_environment",
    ):
        config[key] = None
    path.write_text(yaml.safe_dump(config))
    plan = PLANNER.build_plan(
        request(platforms="nvidia", requested_level="L2"), CONTRACT, tmp_path, []
    )
    lane = plan["execution_plan"][0]
    assert lane["environment"] == {}
    assert lane["configuration"]["setup_stage"] == "integration"
    assert lane["configuration"]["integration_timeout_minutes"] == str(
        config["timeout_minutes"]
    )
    assert lane["configuration"]["integration_build_tools"] == ""
    assert lane["configuration"]["integration_test_dependencies"] == ""


@pytest.mark.parametrize(
    "entry,event,changed",
    [
        ("ci", "push", None),
        ("ci", "pull_request", ["csrc/aten/backends/mthreads/mudnn_copy.cc"]),
        ("ci", "pull_request", ["README.md"]),
        ("external", "", ["csrc/aten/backends/mthreads/mudnn_copy.cc"]),
    ],
)
def test_each_platform_file_is_read_once_per_plan(monkeypatch, entry, event, changed):
    counts = Counter()
    original = Path.open

    def counted(path, *args, **kwargs):
        if path.parent == ROOT / ".github/configs" and path.suffix == ".yml":
            counts[path.name] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    if entry == "ci":
        PLANNER.build_ci_plan(request(), CONTRACT, ROOT, changed, event)
    else:
        PLANNER.build_plan(request(), CONTRACT, ROOT, changed)
    assert counts == {
        name + ".yml": 1 for name in PLANNER._load_platforms(ROOT / ".github/configs")
    }


@pytest.mark.parametrize(
    "name,job_name",
    [
        ("ci.yml", "discover-platforms"),
        ("load-platform-config.yml", "configuration"),
        ("flagcicd-entry.yml", "plan"),
    ],
)
def test_control_checkout_precedes_source_in_separate_sibling_directories(
    name, job_name
):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    steps = workflow["jobs"][job_name]["steps"]
    checkouts = [
        s["with"] for s in steps if s.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2
    control, source = checkouts
    assert control["ref"] == "${{ github.workflow_sha }}"
    assert source["ref"] != control["ref"]
    control_path, source_path = Path(control["path"]), Path(source["path"])
    assert not control_path.is_absolute() and not source_path.is_absolute()
    assert control_path.parent == source_path.parent == Path(".")
    assert control_path != source_path
    assert control_path.name not in (".", "..") and source_path.name not in (".", "..")
    assert all(c["persist-credentials"] is False for c in checkouts)


@pytest.mark.parametrize("planned", [True, False])
def test_real_workflow_commands_keep_control_isolated_from_source_symlinks(
    tmp_path, planned
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    control, source = workspace / "control", workspace / "source"
    shutil.copytree(
        SCRIPTS,
        control / ".github/scripts",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (control / ".flagcicd").mkdir()
    shutil.copyfile(ROOT / ".flagcicd/project.yaml", control / ".flagcicd/project.yaml")
    shutil.copytree(ROOT / ".github/configs", source / ".github/configs")
    (source / ".github/scripts").mkdir()
    shutil.copyfile(
        ROOT / ".github/scripts/set_env.sh", source / ".github/scripts/set_env.sh"
    )
    shutil.copyfile(
        ROOT / ".github/integration-manifest.yml",
        source / ".github/integration-manifest.yml",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("untouched")
    (source / "_ci_control").symlink_to(outside, target_is_directory=True)
    (source / "control").symlink_to(control, target_is_directory=True)

    def git(directory, *args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(directory),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                *args,
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    for directory in (control, source):
        shutil.copyfile(ROOT / ".gitignore", directory / ".gitignore")
        git(directory, "init", "-q")
        git(directory, "add", ".")
        git(directory, "commit", "-qm", "fixture")
    source_sha, control_sha = (
        git(source, "rev-parse", "HEAD"),
        git(control, "rev-parse", "HEAD"),
    )
    assert source_sha != control_sha
    temporary = tmp_path / "runner-temp"
    temporary.mkdir()
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "SOURCE_REPOSITORY": "owner/fork",
        "EVENT_NAME": "push",
        "TEST_PROFILE": "pr",
        "BASE_SHA": source_sha,
        "HEAD_SHA": source_sha,
        "DISPATCH_PLATFORMS": "",
        "PR_LABELS_JSON": "[]",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_OUTPUT": str(tmp_path / "plan-outputs"),
        "RUNNER_TEMP": str(temporary),
        "PLATFORM": "nvidia",
        "PLAN_ARTIFACT": "",
        "PLAN_CONTEXT": "",
    }
    # Exercise the normal CI interpreter, even when the outer test run disables
    # bytecode writes or redirects its cache. The fixture uses the real ignores.
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("PYTHONPYCACHEPREFIX", None)
    if planned:
        jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
        command = next(
            s["run"]
            for s in jobs["discover-platforms"]["steps"]
            if s.get("id") == "detect"
        )
        producer = subprocess.run(
            ["bash", "-e", "-c", command],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
        )
        assert producer.returncode == 0, producer.stderr
        plan = json.loads((workspace / "flagcicd-plan.json").read_text())
        assert plan["source_sha"] == source_sha
        assert plan["control_sha"] == control_sha
        outputs = dict(
            line.split("=", 1)
            for line in (tmp_path / "plan-outputs").read_text().splitlines()
        )
        env.update(
            PLAN_ARTIFACT=outputs["artifact_name"], PLAN_CONTEXT=outputs["context"]
        )
        artifact_dir = temporary / "ci-execution-plan"
        artifact_dir.mkdir()
        shutil.copyfile(
            workspace / "flagcicd-plan.json", artifact_dir / "flagcicd-plan.json"
        )
    env["GITHUB_OUTPUT"] = str(tmp_path / "configuration-outputs")
    loader = yaml.safe_load(
        (ROOT / ".github/workflows/load-platform-config.yml").read_text()
    )
    command = next(
        s["run"]
        for s in loader["jobs"]["configuration"]["steps"]
        if s.get("id") == "config"
    )
    consumer = subprocess.run(
        ["bash", "-e", "-c", command],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
    )
    assert consumer.returncode == 0, consumer.stderr
    outputs = parse_outputs(tmp_path / "configuration-outputs")
    assert outputs["platform"] == "nvidia"
    assert outputs["runner_labels"] == '["torch-fl-nv"]'
    assert git(source, "rev-parse", "HEAD") == source_sha
    assert git(control, "rev-parse", "HEAD") == control_sha
    assert sentinel.read_text() == "untouched"
    assert list((control / ".github/scripts").rglob("*.pyc"))
    assert not git(control, "status", "--porcelain")
    assert not git(source, "status", "--porcelain")
