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

"""Merge pytest-json-report sessions without turning missing results into passes."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def merge_reports(
    directory, platform, outcome, source_repository, source_ref, require_execution=False
):
    summary = Counter(total=0, collected=0)
    tests, collectors, warnings, sessions, incomplete = [], [], [], [], []
    duration = 0.0
    for marker in sorted(Path(directory).glob("*.pending")):
        session = {"report": marker.with_suffix(".json").name}
        try:
            session.update(json.loads(marker.read_text(encoding="utf-8")))
            report = json.loads(marker.with_suffix(".json").read_text(encoding="utf-8"))
            if (
                not isinstance(report, dict)
                or not isinstance(report.get("summary"), dict)
                or not isinstance(report.get("tests"), list)
                or not isinstance(report.get("exitcode"), int)
                or not isinstance(report.get("duration"), (int, float))
                or not all(
                    isinstance(value, int) and value >= 0
                    for value in report["summary"].values()
                )
            ):
                raise ValueError("missing pytest summary or test details")
        except (OSError, ValueError, TypeError) as error:
            session["error"] = str(error)
            incomplete.append(session["report"])
            sessions.append(session)
            continue
        selected = session.get("selected_nodeids")
        # A group name is the marker that this session came from a FlagCICD
        # invocation.  The integration runner writes groups.json, but the
        # acceptance and wheel-only entry points can run pytest directly and
        # still use the reporting plugin.  Requiring groups.json here would
        # make their missing collection inventory silently pass.
        if session.get("group") and selected is None:
            incomplete.append(session["report"])
            session["collection_error"] = "selected test inventory is missing"
        if selected is not None:
            observed = [test["nodeid"] for test in report["tests"]]
            if (
                not selected
                or len(observed) != len(selected)
                or set(selected) != set(observed)
            ):
                incomplete.append(session["report"])
                session["collection_error"] = (
                    "selected test inventory does not match executed tests"
                )
        session["exitcode"] = report["exitcode"]
        session["summary"] = report["summary"]
        sessions.append(session)
        summary.update(report["summary"])
        duration += report["duration"]
        for test in report["tests"]:
            test.setdefault("metadata", {}).update(
                flagcicd_group=session.get("group", ""),
                flagcicd_session=session["report"],
            )
            tests.append(test)
        collectors.extend(report.get("collectors", []))
        warnings.extend(report.get("warnings", []))

    if not sessions:
        raise ValueError("No pytest sessions started; no test report to upload")
    execution = None
    execution_errors = []
    execution_path = Path(directory) / "groups.json"
    if execution_path.is_file():
        execution = json.loads(execution_path.read_text())
        groups = execution.get("groups", [])
        if (
            not groups
            or execution.get("expected") != [g.get("name") for g in groups]
            or any(g.get("status") != "passed" for g in groups)
        ):
            execution_errors.append(
                "integration groups did not all complete successfully"
            )
        for group in groups:
            matched = [
                session for session in sessions if session.get("group") == group["name"]
            ]
            if group.get("requires_pytest") and not matched:
                execution_errors.append("missing pytest session: " + group["name"])
            if group.get("min_passed", 0) > sum(
                session.get("summary", {}).get("passed", 0) for session in matched
            ):
                execution_errors.append(
                    "required group has no passing tests: " + group["name"]
                )
    elif require_execution:
        execution_errors.append("integration execution record is missing")
    if not summary.get("passed", 0):
        execution_errors.append("no tests passed")
    complete = not incomplete and not execution_errors
    # The integration runner owns the verdict, including its existing explicit
    # teardown-crash policy and failures in non-pytest environment checks.
    return {
        "created": time.time(),
        "duration": duration,
        "exitcode": 0 if outcome == "success" and complete else 1,
        "root": str(Path.cwd()),
        "environment": {"platform": platform},
        "summary": dict(summary),
        "tests": tests,
        "collectors": collectors,
        "warnings": warnings,
        "flagcicd": {
            "platform": platform,
            "outcome": outcome,
            "source_repository": source_repository,
            "source_ref": source_ref,
            "complete": complete,
            "execution": execution,
            "execution_errors": execution_errors,
            "incomplete_sessions": incomplete,
            "sessions": sessions,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--require-execution", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument(
        "--outcome", choices=["success", "failure", "cancelled"], required=True
    )
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    report = merge_reports(
        args.report_dir,
        args.platform,
        args.outcome,
        args.source_repository,
        args.source_ref,
        args.require_execution,
    )
    Path(args.output).write_text(json.dumps(report), encoding="utf-8")
    print(
        f"Merged {len(report['flagcicd']['sessions'])} pytest sessions into {args.output}"
    )
    if not report["flagcicd"]["complete"]:
        print(
            "::warning::Some pytest sessions did not produce valid reports; see flagcicd metadata."
        )


if __name__ == "__main__":
    main()
