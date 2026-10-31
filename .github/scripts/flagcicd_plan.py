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

"""Build an auditable test plan from the source diff and one capability contract."""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess

import yaml

from ci.generate_matrix import (
    _changed_files,
    _load_platforms as _load_platforms,
    _platforms_from_dispatch,
    _platforms_from_files,
    boxing_platforms,
    select_platforms,
)
from platform_config import compile_platform, load_platform_data, plan_hash


LEVELS = {"L0": 0, "L1": 1, "L2": 2}


def infer_level(changed, contract):
    policy = contract["risk_rules"]
    if changed is None:
        return policy["no_diff_level"], policy["unknown_risk"], [], "unavailable"
    level, risk, matched = "L0", "R0", []
    for path in changed:
        path = path[2:] if path.startswith("./") else path
        rules = [
            rule
            for rule in policy["rules"]
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule["paths"])
        ]
        if not rules:
            rules = [
                {
                    "id": "unknown",
                    "minimum_level": policy["unknown_level"],
                    "risk": policy["unknown_risk"],
                }
            ]
        for rule in rules:
            level = max((level, rule["minimum_level"]), key=LEVELS.__getitem__)
            risk = max(risk, rule["risk"])
            matched.append({"path": path, "rule": rule["id"]})
    return level, risk, matched, "available"


def build_plan(request, contract, source, changed):
    requested = request["requested_level"]
    if requested not in ("auto", *LEVELS):
        raise ValueError("unsupported test level")
    configs = source / ".github/configs"
    platform_configs, raw_configs = load_platform_data(source)
    known = list(platform_configs)
    if not known:
        raise ValueError("no platform configurations found")
    selection = request["platforms"]
    if isinstance(selection, list):
        if not selection or not all(
            isinstance(p, str) and p in known for p in selection
        ):
            raise ValueError("platforms must be a non-empty array of platform names")
        selection = ",".join(selection)
    if not isinstance(selection, str) or not selection.strip():
        raise ValueError(
            "platforms must be all, a comma-separated list, or a JSON array"
        )
    profile = request.get("profile", "pr")
    if profile not in ("pr", "nightly"):
        raise ValueError("weekly is reserved; supported profiles are pr and nightly")
    if profile == "nightly" and selection.strip() != "all":
        raise ValueError("nightly requires platforms=all")
    boxing = boxing_platforms(configs, known, platform_configs)
    reasons = [{"kind": "dispatch", "requested": selection}]
    selected = _platforms_from_dispatch(selection, known, boxing)
    if profile == "pr" and selection.strip() == "all" and changed is not None:
        reasons = []
        selected = _platforms_from_files(
            changed, known, configs, boxing, reasons, platform_configs
        )
    elif changed is None and profile == "pr" and selection.strip() == "all":
        reasons = [{"kind": "diff_unavailable"}]
    if profile == "nightly":
        reasons = [{"kind": "nightly_full_matrix"}]
    inferred, risk, matched, diff_status = infer_level(changed, contract)
    level = max(
        (inferred, "L0" if requested == "auto" else requested), key=LEVELS.__getitem__
    )
    if profile == "nightly":
        level = "L2"
    if contract["levels"][level]["support"] != "supported":
        raise ValueError(f"test level {level} is not supported")
    # An explicit hardware request must not become an empty, passing matrix.
    if level == "L2" and not selected:
        selected = _platforms_from_dispatch("all", known, boxing)
        reasons.append({"kind": "l2_requires_hardware"})
    plan = {
        "schema_version": "flagcicd.plan/v1",
        **request,
        "entry": "flagcicd",
        "profile": profile,
        "test_level": level,
        "scope": contract["levels"][level]["scope"],
        "risk_level": risk,
        "risk_rule_version": contract["risk_rules"]["version"],
        "diff_status": diff_status,
        "changed_files": changed,
        "matched_rules": matched,
        "platforms": selected,
        "validated_platforms": selected or known,
        "expected_platforms": selected if level == "L2" else [],
        "expected_shared_checks": ["lint", "plan"]
        + ([] if level == "L0" else ["agnostic", "codegen"]),
        "selection_reasons": reasons,
    }
    return complete_plan(plan, contract, source, platform_configs, raw_configs, boxing)


def build_ci_plan(
    request, contract, source, changed, event_name, labels=(), dispatch=""
):
    """Plan automatic CI without applying the external entry's level policy."""
    profile = request.get("profile", "pr")
    if profile not in ("pr", "nightly"):
        raise ValueError("weekly is reserved; supported profiles are pr and nightly")
    if profile == "nightly" and dispatch not in ("", "all"):
        raise ValueError("nightly requires platforms=all")
    platform_configs, raw_configs = load_platform_data(source)
    selected, reasons = select_platforms(
        event_name,
        source / ".github/configs",
        changed,
        labels,
        dispatch,
        platform_configs=platform_configs,
    )
    boxing = boxing_platforms(
        source / ".github/configs", list(platform_configs), platform_configs
    )
    _inferred, risk, matched, diff_status = infer_level(changed, contract)
    # Ordinary CI always runs all three shared checks, including documentation
    # PRs. An empty hardware matrix is valid here, even when risk inference
    # would raise an explicit external request to L2.
    level = "L2" if selected else "L1"
    return complete_plan(
        {
            "schema_version": "flagcicd.plan/v1",
            **request,
            "profile": profile,
            "entry": "ci",
            "event_name": event_name,
            "test_level": level,
            "scope": contract["levels"][level]["scope"],
            "risk_level": risk,
            "risk_rule_version": contract["risk_rules"]["version"],
            "diff_status": diff_status,
            "changed_files": changed,
            "matched_rules": matched,
            "platforms": selected,
            "expected_platforms": selected,
            "validated_platforms": selected,
            "expected_shared_checks": [
                "resolve-checkout",
                "lint",
                "agnostic-checks",
                "codegen-checks",
                "discover-platforms",
            ],
            "selection_reasons": reasons,
        },
        contract,
        source,
        platform_configs,
        raw_configs,
        boxing,
    )


def complete_plan(plan, contract, source, platform_configs, raw_configs, boxing):
    """Validate configuration and snapshot final executable lanes once."""
    manifest_bytes = (source / ".github/integration-manifest.yml").read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    fingerprint = hashlib.sha256(manifest_bytes)
    execution = []
    for lane in plan["validated_platforms"]:
        compiled = compile_platform(
            source, lane, plan["profile"], manifest, platform_configs, boxing
        )
        platform = compiled["platform"]
        fingerprint.update(lane.encode())
        fingerprint.update(raw_configs[platform])
        if lane in plan["expected_platforms"]:
            execution.append(compiled)
    fingerprint.update(json.dumps(contract, sort_keys=True).encode())
    fingerprint.update(plan["profile"].encode())
    plan.update(
        config_hash=fingerprint.hexdigest(),
        control_sha=plan.get("control_sha", ""),
        diff_head_sha=plan.get("diff_head_sha", plan["source_sha"]),
        execution_plan=execution,
    )
    plan["plan_hash"] = plan_hash(plan)
    return plan


def report_context(plan):
    """Keep potentially large diffs in artifacts, outside process environments."""
    fields = (
        "schema_version",
        "source_repository",
        "source_sha",
        "base_sha",
        "test_level",
        "scope",
        "profile",
        "risk_level",
        "risk_rule_version",
        "config_hash",
        "control_sha",
        "plan_hash",
        "platforms",
        "expected_platforms",
        "expected_shared_checks",
        "run_id",
        "run_attempt",
        "task_id",
        "parent_run_id",
    )
    encoded = json.dumps({key: plan[key] for key in fields}, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16384:
        raise ValueError("report context exceeds the 16 KiB dispatch limit")
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--ci", action="store_true", help="Plan ordinary CI routing")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source = args.source.resolve()
    control = Path(__file__).resolve().parents[2]
    control_sha = subprocess.check_output(
        ["git", "-C", str(control), "rev-parse", "HEAD"], text=True
    ).strip()
    contract = yaml.safe_load((control / ".flagcicd/project.yaml").read_text())
    if args.ci:
        source_sha = subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        request = {
            "source_repository": os.environ["SOURCE_REPOSITORY"],
            "source_sha": source_sha,
            "base_sha": os.environ.get("BASE_SHA", ""),
            "diff_head_sha": os.environ.get("HEAD_SHA") or source_sha,
            "profile": os.environ.get("TEST_PROFILE", "pr"),
            "task_id": "",
            "parent_run_id": "",
        }
    else:
        request = json.loads(args.request.read_text())
    request["control_sha"] = control_sha
    previous = Path.cwd()
    try:
        os.chdir(args.source)
        changed = (
            _changed_files(
                os.environ.get("BASE_REF", "main"),
                request["base_sha"],
                request.get("diff_head_sha", request["source_sha"]),
            )
            if not args.ci or os.environ["EVENT_NAME"] == "pull_request"
            else None
        )
    finally:
        os.chdir(previous)
    if args.ci:
        plan = build_ci_plan(
            request,
            contract,
            args.source,
            changed,
            os.environ["EVENT_NAME"],
            json.loads(os.environ.get("PR_LABELS_JSON") or "[]") or [],
            os.environ.get("DISPATCH_PLATFORMS", ""),
        )
    else:
        plan = build_plan(request, contract, args.source, changed)
    plan.update(
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
    )
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("context=" + report_context(plan) + "\n")
        output.write(f"artifact_name=flagcicd-plan-{plan['run_attempt']}\n")
        output.write(f"test_level={plan['test_level']}\n")
        output.write(f"profile={plan['profile']}\n")
        output.write("platforms=" + json.dumps(plan["expected_platforms"]) + "\n")
    print(
        f"Plan: {plan['test_level']} ({plan['risk_level']}), platforms={plan['expected_platforms']}"
    )


if __name__ == "__main__":
    main()
