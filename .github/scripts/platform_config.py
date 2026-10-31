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

"""Compile platform data once, or consume its immutable planned snapshot."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from compose_integration_manifest import compose
from run_integration_tests import validate_configuration


def load_platform_data(source):
    """Read each platform once for routing, compilation and fingerprinting."""
    import yaml

    from ci.generate_matrix import _load_platforms

    configs, raw = {}, {}
    for platform in _load_platforms(source / ".github/configs"):
        try:
            raw[platform] = (source / f".github/configs/{platform}.yml").read_bytes()
            configs[platform] = yaml.safe_load(raw[platform])
        except (OSError, yaml.YAMLError):
            configs[platform] = None
    return configs, raw


def compile_platform(source, lane, profile, manifest, platform_configs, boxing):
    """Compile a lane from the same data used by routing and hashing."""
    known = list(platform_configs)
    platform = lane.removesuffix("-boxing")
    is_boxing = lane != platform
    if platform not in known or (is_boxing and platform not in boxing):
        raise ValueError(f"unsupported platform lane: {lane}")
    config = copy.deepcopy(platform_configs[platform])
    if not isinstance(config, dict):
        raise ValueError(f"{platform} configuration must be an object")
    if config.get("integration_tests") is not None:
        raise ValueError("integration_tests belong in the shared manifest")

    def string(key, default=None, empty=False):
        value = config.get(key)
        if value is None:
            value = default
        if (
            not isinstance(value, str)
            or "\0" in value
            or (not empty and not value.strip())
        ):
            raise ValueError(
                f"{key} must be a {'possibly empty' if empty else 'non-empty'} string"
            )
        return value

    def strings(key, default=None, nonempty=False):
        value = config.get(key)
        if value is None:
            value = default
        if (
            not isinstance(value, list)
            or not all(isinstance(v, str) and "\0" not in v for v in value)
            or (nonempty and (not value or any(not v.strip() for v in value)))
        ):
            raise ValueError(
                f"{key} must be a {'non-empty ' if nonempty else ''}list of strings"
            )
        return value

    def timeout(key, default=None):
        value = config.get(key)
        if value is None:
            value = default
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return str(value)

    setup_script = string("setup_script")
    script_path = (source / setup_script).resolve()
    if not script_path.is_relative_to(source.resolve()) or not script_path.is_file():
        raise ValueError(f"Invalid setup_script: {setup_script}")
    stage = string("setup_stage", "integration")
    if stage not in ("build", "integration"):
        raise ValueError("setup_stage must be build or integration")
    artifact = string("wheel_artifact_name") + ("-boxing" if is_boxing else "")
    if not re.fullmatch(r"wheel-[a-z0-9]+(-boxing)?", artifact):
        raise ValueError(
            "wheel_artifact_name must be wheel-<platform> (optionally -boxing)"
        )
    config["display_name"] = string("display_name") + (
        " (CUDA boxing)" if is_boxing else ""
    )
    outputs = {
        "platform": platform,
        "boxing_lane": "1" if is_boxing else "",
        "display_name": config["display_name"],
        "setup_script": setup_script,
        "setup_platform": string("setup_platform"),
        "setup_stage": stage,
        "ci_image": string("ci_image", empty=True),
        "runner_labels": strings("runner_labels", nonempty=True),
        "container_volumes": strings("container_volumes"),
        "container_options": string("container_options", empty=True),
        "timeout_minutes": timeout("timeout_minutes"),
        "integration_timeout_minutes": timeout(
            "integration_timeout_minutes", config.get("timeout_minutes")
        ),
        "wheel_artifact_name": artifact,
        "integration_build_tools": "\n".join(strings("integration_build_tools", [])),
        "integration_test_dependencies": "\n".join(
            strings("integration_test_dependencies", [])
        ),
    }
    check = config.get("boxing_environment_check" if is_boxing else "environment_check")
    if is_boxing and not check:
        raise ValueError(f"{lane} has no boxing_environment_check")
    groups = compose(
        manifest,
        config,
        platform,
        known,
        boxing_lane=is_boxing,
        environment_check=check,
        boxing=boxing,
        profile=profile,
    )
    environment = config.get("integration_environment")
    if environment is None:
        environment = {}
    validate_configuration(groups, environment)
    return {
        "name": lane,
        "platform": platform,
        "lane": "boxing" if is_boxing else "default",
        "configuration": outputs,
        "environment": environment,
        "groups": [{"group_id": group["manifest_id"], **group} for group in groups],
    }


def plan_hash(plan):
    """Hash deterministic identity and final content, excluding run/task IDs."""
    fields = (
        "schema_version",
        "entry",
        "source_repository",
        "source_sha",
        "control_sha",
        "diff_head_sha",
        "base_sha",
        "profile",
        "test_level",
        "scope",
        "config_hash",
        "expected_platforms",
        "expected_shared_checks",
        "selection_reasons",
        "execution_plan",
    )
    payload = json.dumps(
        {key: plan[key] for key in fields}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_plan(plan, context):
    """Bind an artifact to its producer context and the current workflow run."""
    if not isinstance(plan, dict) or not isinstance(context, dict):
        raise ValueError("plan and context must be objects")
    if plan.get("schema_version") != "flagcicd.plan/v1" or plan.get("observation_only"):
        raise ValueError("an executable plan is required")
    if plan_hash(plan) != plan.get("plan_hash"):
        raise ValueError("plan_hash does not match plan content")
    for key in (
        "plan_hash",
        "source_repository",
        "source_sha",
        "control_sha",
        "profile",
        "run_id",
        "run_attempt",
    ):
        if not context.get(key) or str(context[key]) != str(plan.get(key)):
            raise ValueError(f"plan context mismatch: {key}")
    if any(key not in plan or value != plan[key] for key, value in context.items()):
        raise ValueError("plan context content differs from the artifact")
    if str(plan["run_id"]) != os.environ.get("GITHUB_RUN_ID"):
        raise ValueError("plan does not belong to current GITHUB_RUN_ID")
    if not 1 <= int(plan["run_attempt"]) <= int(os.environ["GITHUB_RUN_ATTEMPT"]):
        raise ValueError("plan attempt is not available in this run")


def consume_plan(plan, context, lane, profile, source_sha):
    """Never recompose a planned lane or fall back after invalid evidence."""
    verify_plan(plan, context)
    if profile != plan["profile"] or source_sha != plan["source_sha"]:
        raise ValueError("plan profile or checked-out source SHA differs")
    lanes = plan["execution_plan"]
    names = [item["name"] for item in lanes]
    if names != plan["expected_platforms"] or len(names) != len(set(names)):
        raise ValueError("execution lanes differ from expected_platforms")
    selected = [item for item in lanes if item["name"] == lane]
    if len(selected) != 1:
        raise ValueError(f"lane {lane} is not uniquely planned")
    result = selected[0]
    ids = [group["group_id"] for group in result["groups"]]
    if len(ids) != len(set(ids)) or any(
        g["group_id"] != g["manifest_id"] for g in result["groups"]
    ):
        raise ValueError("planned group identities are invalid")
    validate_configuration(result["groups"], result["environment"])
    return result


def emit_outputs(lane, output):
    values = {
        **lane["configuration"],
        "integration_tests": lane["groups"],
        "integration_environment": lane["environment"],
    }
    with output.open("a") as stream:
        for key, value in values.items():
            encoded = (
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            delimiter = "ci_" + uuid.uuid4().hex
            while delimiter in encoded.splitlines():
                delimiter = "ci_" + uuid.uuid4().hex
            stream.write(f"{key}<<{delimiter}\n{encoded}\n{delimiter}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--lane", required=True)
    parser.add_argument("--profile", default="pr")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()
    source = args.source.resolve()
    if args.plan:
        sha = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        plan = json.loads(args.plan.read_text())
        control_sha = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
        if control_sha != plan["control_sha"]:
            raise ValueError("plan control_sha differs from checked-out control code")
        lane = consume_plan(
            plan, json.loads(os.environ["PLAN_CONTEXT"]), args.lane, args.profile, sha
        )
    else:
        import yaml

        from ci.generate_matrix import boxing_platforms

        configs, _raw = load_platform_data(source)
        lane = compile_platform(
            source,
            args.lane,
            args.profile,
            yaml.safe_load((source / ".github/integration-manifest.yml").read_text()),
            configs,
            boxing_platforms(source / ".github/configs", list(configs), configs),
        )
    emit_outputs(lane, args.output)


if __name__ == "__main__":
    main()
