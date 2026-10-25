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

import yaml

from ci.generate_matrix import (
    _changed_files,
    _load_platforms,
    _platforms_from_dispatch,
    _platform_of,
    _platforms_from_files,
    boxing_platforms,
)
from compose_integration_manifest import compose


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
    known = _load_platforms(configs)
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
    boxing = boxing_platforms(configs, known)
    selected = _platforms_from_dispatch(selection, known, boxing)
    if selection.strip() == "all" and changed is not None:
        selected = _platforms_from_files(changed, known, configs, boxing)
    inferred, risk, matched, diff_status = infer_level(changed, contract)
    level = max(
        (inferred, "L0" if requested == "auto" else requested), key=LEVELS.__getitem__
    )
    if contract["levels"][level]["support"] != "supported":
        raise ValueError(f"test level {level} is not supported")
    # An explicit hardware request must not become an empty, passing matrix.
    if level == "L2" and not selected:
        selected = known
    manifest_path = source / ".github/integration-manifest.yml"
    manifest = yaml.safe_load(manifest_path.read_text())
    fingerprint = hashlib.sha256(manifest_path.read_bytes())
    # `selected` names lanes, so a boxing platform arrives as both
    # `<platform>` and `<platform>-boxing`; both read the platform's one config
    # and compose a build of it, and the config counts once in the fingerprint.
    for lane in selected or known:
        platform = _platform_of(lane, boxing)
        boxing_lane = lane != platform
        path = configs / f"{platform}.yml"
        config = yaml.safe_load(path.read_text())
        compose(
            manifest, config, platform, known, boxing_lane=boxing_lane, boxing=boxing
        )
        fingerprint.update(lane.encode())
        fingerprint.update(path.read_bytes())
    fingerprint.update(json.dumps(contract, sort_keys=True).encode())
    return {
        "schema_version": "flagcicd.plan/v1",
        **request,
        "test_level": level,
        "scope": contract["levels"][level]["scope"],
        "risk_level": risk,
        "risk_rule_version": contract["risk_rules"]["version"],
        "diff_status": diff_status,
        "changed_files": changed,
        "matched_rules": matched,
        "config_hash": fingerprint.hexdigest(),
        "platforms": selected,
        "validated_platforms": selected or known,
        "expected_platforms": selected if level == "L2" else [],
    }


def report_context(plan):
    """Keep potentially large diffs in artifacts, outside process environments."""
    fields = (
        "schema_version",
        "source_repository",
        "source_sha",
        "base_sha",
        "test_level",
        "scope",
        "risk_level",
        "risk_rule_version",
        "config_hash",
        "platforms",
        "expected_platforms",
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
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    contract = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".flagcicd/project.yaml").read_text()
    )
    previous = Path.cwd()
    try:
        os.chdir(args.source)
        changed = _changed_files("main", request["base_sha"], request["source_sha"])
    finally:
        os.chdir(previous)
    plan = build_plan(request, contract, args.source, changed)
    plan.update(
        run_id=os.environ["GITHUB_RUN_ID"],
        run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
    )
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write("context=" + report_context(plan) + "\n")
        output.write(f"test_level={plan['test_level']}\n")
        output.write("platforms=" + json.dumps(plan["expected_platforms"]) + "\n")
        output.write(
            "validated_platforms=" + json.dumps(plan["validated_platforms"]) + "\n"
        )
    print(
        f"Plan: {plan['test_level']} ({plan['risk_level']}), platforms={plan['expected_platforms']}"
    )


if __name__ == "__main__":
    main()
