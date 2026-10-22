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


def merge_reports(directory, platform, outcome, source_repository, source_ref):
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
    complete = not incomplete
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
            "incomplete_sessions": incomplete,
            "sessions": sessions,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
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
