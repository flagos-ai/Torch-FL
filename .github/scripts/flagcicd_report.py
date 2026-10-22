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

"""Describe platform execution and aggregate a plan without combining coverage."""

import argparse
import json
import math
import os
from pathlib import Path

from flagcicd_resolve import text as request_text


SCHEMA = "flagcicd.report/v1"
FAILURES = {"build_failed", "test_failed", "infra_error", "cancelled", "report_error"}
IDENTITY = ("source_repository", "source_sha", "run_id", "test_level", "config_hash")
REQUIRED_STAGES = ("setup", "build", "validate", "install", "tests")


def identity(plan):
    return {
        "repository": plan["source_repository"],
        **{
            key: plan[key]
            for key in (*IDENTITY, "scope", "risk_level", "task_id", "parent_run_id")
        },
    }


def failed_request_identity():
    """Retain safe correlation IDs when no validated source plan is available."""
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            request = event.get("inputs")
        elif (
            os.environ.get("GITHUB_EVENT_NAME") == "repository_dispatch"
            and event.get("action") == "flagcicd"
        ):
            request = event.get("client_payload")
        else:
            return {}
    except (KeyError, OSError, ValueError, AttributeError):
        return {}
    if not isinstance(request, dict):
        return {}
    fields = {}
    for key in ("task_id", "parent_run_id"):
        try:
            fields[key] = request_text(request.get(key, ""), key)
        except ValueError:
            continue
    return fields


def one_report(directory, filename):
    paths = list(directory.rglob(filename))
    if len(paths) != 1:
        raise ValueError(f"expected one {filename}, found {len(paths)}")
    payload = json.loads(paths[0].read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{filename} must contain an object")
    return payload


def platform_report(plan, platform, jobs, raw, attempt):
    if platform not in plan["expected_platforms"]:
        raise ValueError("platform is not expected by the plan")
    job = jobs["integration-tests"]
    stages = json.loads(job.get("outputs", {}).get("stages") or "{}")
    result = job["result"]
    errors = []
    if jobs["configuration"]["result"] == "cancelled" or result == "cancelled":
        status = "cancelled"
    elif jobs["configuration"]["result"] != "success":
        status = "infra_error"
    elif any(stages.get(key) == "failure" for key in ("build", "validate", "install")):
        status = "build_failed"
    elif stages.get("setup") == "failure":
        status = "infra_error"
    elif stages.get("tests") == "failure":
        status = "test_failed"
    elif result == "success" and all(
        stages.get(key) == "success" for key in REQUIRED_STAGES
    ):
        status = "passed"
    else:
        status = "infra_error"
        errors.append("the platform did not successfully complete all required stages")
    metrics = {"coverage": {"available": False}}
    try:
        pytest = one_report(raw / "pytest", f"pytest-{platform}.json")
        metadata = pytest["flagcicd"]
        if (
            metadata["source_repository"] != plan["source_repository"]
            or metadata["source_ref"] != plan["source_sha"]
            or metadata["platform"] != platform
        ):
            raise ValueError(
                "pytest report does not match the planned source and platform"
            )
        if metadata["complete"] is not True:
            raise ValueError("pytest report is incomplete")
        counts = pytest["summary"]
        if not isinstance(counts, dict) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts.values()
        ):
            raise ValueError("invalid pytest summary")
        if status == "passed" and pytest["exitcode"] != 0:
            raise ValueError("pytest report contradicts the successful job result")
        if status == "passed" and any(
            counts.get(key, 0) for key in ("failed", "error")
        ):
            raise ValueError(
                "pytest failure counts contradict the successful job result"
            )
        metrics["tests"] = counts
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
        if status == "passed":
            status = "report_error"
    coverage_errors = []
    try:
        coverage = one_report(raw / "coverage", f"coverage-{platform}.json")
        metadata = one_report(raw / "coverage", "metadata.json")
        if (
            metadata["source_repository"] != plan["source_repository"]
            or metadata["source_ref"] != plan["source_sha"]
            or metadata["platform"] != platform
            or metadata["complete"] is not True
        ):
            raise ValueError("coverage is incomplete or belongs to another source")
        totals = coverage["totals"]
        covered, total = totals["covered_lines"], totals["num_statements"]
        if (
            not all(type(value) is int for value in (covered, total))
            or not 0 <= covered <= total
        ):
            raise ValueError("invalid coverage line counts")
        percent = totals["percent_covered"]
        if (
            not isinstance(percent, (int, float))
            or not math.isfinite(percent)
            or not 0 <= percent <= 100
        ):
            raise ValueError("invalid coverage percentage")
        metrics["coverage"] = {
            "available": True,
            "scope": "installed torch_fl Python files",
            "metric_type": "line",
            "covered": covered,
            "total": total,
            "percent": percent,
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        coverage_errors.append(str(error))
    return {
        "schema_version": SCHEMA,
        "report_type": "platform",
        **identity(plan),
        "run_attempt": attempt,
        "platform": {"name": platform},
        "status": status,
        "configuration_result": jobs["configuration"]["result"],
        "execution_result": result,
        "execution_attempt": job.get("outputs", {}).get("run_attempt", ""),
        "stages": stages,
        "metrics": metrics,
        "errors": errors,
        "coverage_warnings": coverage_errors,
    }


def aggregate(plan, jobs, directory, attempt):
    errors, latest, seen, duplicates, ignored = [], {}, set(), set(), []
    if plan is None:
        errors.append("test plan is unavailable")
        plan = {"expected_platforms": []}
    level = plan.get("test_level", "L0")
    expected_jobs = {"plan", "lint", "preflight"}
    if level in ("L1", "L2"):
        expected_jobs.update(("agnostic", "codegen"))
    if level == "L2":
        expected_jobs.add("platforms")
    for name in ("plan", "lint", "preflight", "agnostic", "codegen", "platforms"):
        result = jobs.get(name, {}).get("result", "missing")
        expected = "success" if name in expected_jobs else "skipped"
        if result != expected:
            errors.append(f"{name}: expected {expected}, received {result}")
    for path in sorted(directory.rglob("flagcicd-platform-*.json")):
        try:
            report = json.loads(path.read_text())
            if (
                report["schema_version"] != SCHEMA
                or report["report_type"] != "platform"
            ):
                raise ValueError("unsupported platform report schema")
            report_attempt = report["run_attempt"]
            if type(report_attempt) is not int or not 1 <= report_attempt <= attempt:
                raise ValueError("invalid report attempt")
            matches = all(report[key] == plan[key] for key in IDENTITY)
            if not matches:
                if report_attempt < attempt and report["run_id"] == plan["run_id"]:
                    ignored.append(
                        {
                            "path": path.name,
                            "reason": "previous attempt used a different plan",
                        }
                    )
                    continue
                raise ValueError("report identity does not match the plan")
            platform = report["platform"]["name"]
            if platform not in plan["expected_platforms"]:
                raise ValueError("report platform is not expected")
            if report["status"] not in FAILURES | {"passed"}:
                raise ValueError("invalid report status")
            key = (platform, report_attempt)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
            previous = latest.get(platform)
            if previous is None or previous["run_attempt"] < report_attempt:
                latest[platform] = report
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{path.name}: {error}")
    for platform in plan["expected_platforms"]:
        if platform not in latest:
            errors.append(f"missing platform report: {platform}")
        elif (platform, latest[platform]["run_attempt"]) in duplicates:
            errors.append(f"duplicate latest platform report: {platform}")
    statuses = [report["status"] for report in latest.values()]
    status = "passed"
    for candidate in (
        "cancelled",
        "test_failed",
        "build_failed",
        "infra_error",
        "report_error",
    ):
        if candidate in statuses:
            status = candidate
    for name in ("plan", "preflight", "lint", "agnostic", "codegen", "platforms"):
        if name not in expected_jobs:
            continue
        result = jobs.get(name, {}).get("result", "missing")
        if result == "failure":
            if name in ("plan", "preflight"):
                status = "infra_error"
            elif name != "platforms":
                status = "test_failed"
            elif status == "passed":
                # All expected reports are present and passing, yet the matrix
                # job failed: the failure sits outside the stages the report
                # covers (a child workflow's own failure, a setup step before
                # the stages, an unusable runner). That is infra, not a report
                # defect; a missing or contradictory report is caught above.
                status = "infra_error"
        elif result == "cancelled" and status == "passed":
            status = "cancelled"
        elif result in ("missing", "skipped") and status == "passed":
            status = "infra_error"
    if errors and status == "passed":
        status = (
            "report_error"
            if jobs.get("plan", {}).get("result") == "success"
            else "infra_error"
        )
    return {
        "schema_version": SCHEMA,
        "report_type": "run_summary",
        "platform": {"name": "multi"},
        "repository": plan.get("source_repository", ""),
        **{
            key: plan.get(key, "")
            for key in (*IDENTITY, "scope", "risk_level", "task_id", "parent_run_id")
        },
        "run_attempt": attempt,
        "status": status,
        "expected_platforms": plan["expected_platforms"],
        "jobs": {name: job["result"] for name, job in jobs.items()},
        "children": [latest[key] for key in sorted(latest)],
        "coverage_aggregation": "per_platform_only",
        "errors": errors,
        "ignored_reports": ignored,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("platform", "summary"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="")
    args = parser.parse_args()
    raw_plan = os.environ.get("FLAGCICD_PLAN", "")
    plan = json.loads(raw_plan) if raw_plan else None
    jobs = json.loads(os.environ["JOB_RESULTS"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    if args.mode == "platform":
        report = platform_report(plan, args.platform, jobs, args.directory, attempt)
    else:
        report = aggregate(plan, jobs, args.directory, attempt)
        report["run_id"] = os.environ["GITHUB_RUN_ID"]
        if not plan:
            report["repository"] = os.environ.get("GITHUB_REPOSITORY", "")
            report.update(failed_request_identity())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"FlagCICD {report['report_type']}: {report['status']}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
