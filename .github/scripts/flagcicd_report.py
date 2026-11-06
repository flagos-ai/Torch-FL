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
from collections import Counter
import json
import math
import os
import subprocess
from pathlib import Path

from flagcicd_resolve import text as request_text
from merge_pytest_reports import (
    summarize_groups,
    validate_execution,
    validate_pytest_result,
)
from platform_config import plan_hash, verify_plan


SCHEMA = "flagcicd.report/v1"
FAILURES = {"build_failed", "test_failed", "infra_error", "cancelled", "report_error"}
IDENTITY = (
    "source_repository",
    "source_sha",
    "control_sha",
    "plan_hash",
    "run_id",
    "test_level",
    "config_hash",
    "profile",
)
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


def planned_evidence(plan, platform, evidence, attempt, execution_attempt):
    lanes = [lane for lane in plan["execution_plan"] if lane["name"] == platform]
    if len(lanes) != 1:
        raise ValueError("platform is not uniquely described by the plan")
    if evidence["execution"].get("environment") != lanes[0].get("environment", {}):
        raise ValueError("shared execution environment differs from the plan")
    actual_attempt = evidence["execution"].get("run_attempt")
    if type(execution_attempt) not in (str, int) or str(actual_attempt) != str(
        execution_attempt
    ):
        raise ValueError("execution attempt differs from the hardware job output")
    return validate_execution(
        lanes[0]["groups"],
        evidence["execution"],
        evidence["sessions"],
        identity={key: plan[key] for key in IDENTITY},
        platform=platform,
        attempt=attempt,
    )


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
    evidence, group_results = {}, []
    try:
        metrics["hardware"] = one_report(raw / "pytest", f"hardware-{platform}.json")
    except (OSError, ValueError, TypeError):
        metrics["hardware"] = {"available": False}
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
        evidence = {
            "execution": metadata["execution"],
            "sessions": metadata["sessions"],
        }
        validate_pytest_result(
            pytest,
            require_success=status == "passed",
            require_passed=status == "passed",
        )
        metrics["tests"] = pytest["summary"]
        if metadata["complete"] is not True:
            raise ValueError(
                "pytest report is incomplete: "
                + "; ".join(str(e) for e in metadata.get("execution_errors", []))
            )
        if status == "passed":
            group_results = planned_evidence(
                plan,
                platform,
                evidence,
                attempt,
                job.get("outputs", {}).get("run_attempt"),
            )
            counts = Counter()
            for session in evidence["sessions"]:
                counts.update(session["summary"])
            if dict(counts) != pytest["summary"]:
                raise ValueError("pytest totals differ from group session evidence")
        else:
            group_results = summarize_groups(
                evidence["execution"], evidence["sessions"]
            )
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        errors.append(str(error))
        if evidence and not group_results:
            try:
                group_results = summarize_groups(
                    evidence["execution"], evidence["sessions"]
                )
            except (ValueError, TypeError, KeyError, AttributeError):
                pass
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
            or str(metadata.get("run_attempt"))
            != str(job.get("outputs", {}).get("run_attempt"))
            or not isinstance(metadata.get("context"), dict)
            or any(metadata["context"].get(k) != plan[k] for k in IDENTITY)
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
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
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
        "groups": group_results,
        "evidence": evidence,
        "errors": errors,
        "coverage_warnings": coverage_errors,
    }


def aggregate(plan, jobs, directory, attempt, entry="flagcicd"):
    errors, latest, seen, duplicates, ignored = [], {}, set(), set(), []
    has_plan = plan is not None
    if not has_plan:
        errors.append("test plan is unavailable")
        plan = {"expected_platforms": [], "entry": entry, "expected_shared_checks": []}
    else:
        try:
            if plan_hash(plan) != plan["plan_hash"]:
                raise ValueError("plan content does not match plan_hash")
            if [lane["name"] for lane in plan["execution_plan"]] != plan[
                "expected_platforms"
            ]:
                raise ValueError("execution lanes differ from expected_platforms")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    ordinary = plan.get("entry") == "ci"
    producer, matrix = (
        ("discover-platforms", "platform-pipeline")
        if ordinary
        else ("plan", "platforms")
    )
    expected_jobs = set(plan["expected_shared_checks"])
    if has_plan and producer not in expected_jobs:
        errors.append("plan does not require its producer job")
    expected_jobs.add(producer)
    if plan["expected_platforms"]:
        expected_jobs.add(matrix)
    for name in sorted(expected_jobs | set(jobs)):
        result = jobs.get(name, {}).get("result", "missing")
        if not has_plan:
            if result in ("failure", "cancelled"):
                errors.append(f"{name}: {result}")
            continue
        expected = "success" if name in expected_jobs else "skipped"
        if (
            name == matrix
            and not plan["expected_platforms"]
            and ordinary
            and result in ("success", "skipped")
        ):
            continue
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
            if report["status"] == "passed":
                if (
                    report.get("configuration_result") != "success"
                    or report.get("errors")
                    or report.get("execution_result") != "success"
                    or any(
                        report.get("stages", {}).get(k) != "success"
                        for k in REQUIRED_STAGES
                    )
                ):
                    raise ValueError(
                        "passing report has errors or incomplete execution stages"
                    )
                groups = planned_evidence(
                    plan,
                    platform,
                    report["evidence"],
                    report_attempt,
                    report.get("execution_attempt"),
                )
                if report.get("groups") != groups:
                    raise ValueError("group report differs from execution evidence")
                counts = Counter()
                for session in report["evidence"]["sessions"]:
                    counts.update(session["summary"])
                if report.get("metrics", {}).get("tests") != dict(counts):
                    raise ValueError(
                        "platform test totals differ from session evidence"
                    )
            key = (platform, report_attempt)
            if key in seen:
                duplicates.add(key)
            seen.add(key)
            previous = latest.get(platform)
            if previous is None or previous["run_attempt"] < report_attempt:
                latest[platform] = report
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
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
    for name in sorted(expected_jobs if has_plan else set(jobs)):
        result = jobs.get(name, {}).get("result", "missing")
        if result == "failure":
            if name in (producer, "resolve-checkout"):
                status = "infra_error"
            elif name != matrix:
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
        elif has_plan and result in ("missing", "skipped") and status == "passed":
            status = "infra_error"
    if errors and status == "passed":
        status = (
            "report_error"
            if jobs.get(producer, {}).get("result") == "success"
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
        "expected_shared_checks": plan["expected_shared_checks"],
        "jobs": {name: job["result"] for name, job in jobs.items()},
        "children": [latest[key] for key in sorted(latest)],
        "coverage_aggregation": "per_platform_only",
        "errors": errors,
        "ignored_reports": ignored,
    }


def diagnose_jobs():
    """Keep optional paginated job diagnostics independent of the gate verdict."""
    try:
        run_id, repo = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_REPOSITORY"]
        if not os.environ.get("GH_TOKEN"):
            raise ValueError("not in a GitHub run with a token")
        listing = subprocess.run(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{repo}/actions/runs/{run_id}/jobs",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        listed = [job for page in json.loads(listing) for job in page.get("jobs", [])]
    except (
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        AttributeError,
    ) as error:
        print(f"::warning::could not list job conclusions: {error}")
        return
    for job in listed:
        name, conclusion = job.get("name", ""), job.get("conclusion")
        if conclusion in ("failure", "timed_out"):
            print(f"::error::{name}: {conclusion}")
        elif conclusion == "cancelled" and name.startswith("Platform pipeline ("):
            print(
                f"::notice::{name} was cancelled before it finished, so it measured nothing; re-run that lane to measure it"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("platform", "summary"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="")
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=os.environ.get("CI_PLAN_FILE", "flagcicd-plan.json"),
    )
    parser.add_argument("--entry", choices=("ci", "flagcicd"), default="flagcicd")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    raw_plan = os.environ.get("FLAGCICD_PLAN", "")
    plan, plan_error = None, ""
    try:
        plan = json.loads(args.plan_file.read_text())
        verify_plan(plan, json.loads(raw_plan))
        control = subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
        if control != plan["control_sha"]:
            raise ValueError("report control_sha differs from the plan")
    except (OSError, KeyError, TypeError, ValueError) as error:
        plan_error, plan = str(error), None
    jobs = json.loads(os.environ["JOB_RESULTS"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    if args.mode == "platform":
        if plan is None:
            raise SystemExit("validated plan is unavailable: " + plan_error)
        report = platform_report(plan, args.platform, jobs, args.directory, attempt)
    else:
        report = aggregate(plan, jobs, args.directory, attempt, args.entry)
        report["run_id"] = os.environ["GITHUB_RUN_ID"]
        if not plan:
            report["repository"] = os.environ.get("GITHUB_REPOSITORY", "")
            report.update(failed_request_identity())
    if plan_error:
        report["errors"].append(plan_error)
    if args.diagnostics:
        diagnose_jobs()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"FlagCICD {report['report_type']}: {report['status']}")
    # A platform outcome describes its upstream job; generating that outcome
    # must not add a false root failure for the bounded network retry policy.
    # The summary remains the gate, and real reporting failures still fail.
    if args.mode == "platform":
        return int(report["status"] == "report_error")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
