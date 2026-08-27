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

"""Run a platform's integration manifest and judge the result.

An exit code of zero from pytest is not sufficient evidence that a test group
ran. A group whose tests all self-skip, or that collects nothing because a path
or marker stopped matching, also exits zero. This runner therefore reads the
JUnit XML of every pytest group and requires that the group actually executed
tests, with no skipped and no xfailed outcome:

  * zero tests collected            -> the group failed
  * any skipped or xfailed outcome  -> the group failed
  * missing JUnit XML               -> the group failed

A capability the platform does not have must surface as a real test failure that
the platform owners can act on. A skip erases that evidence while reporting
success, so it is treated as a failure of the group rather than a pass.

Manifest entries declare id, phase and failure_policy explicitly; nothing is
inferred from the entry name. Phases run in order (every preflight entry before
every functional entry), so an absent model mount or device fails before the
functional groups spend the runner's time.

  phase: preflight   environment facts (device nodes, model mount, Torch/wheel
                     identity). Default policy fail-fast: continuing past a
                     broken environment produces misleading functional results.
  phase: functional  test groups. Default policy continue-after-failure, so one
                     failing group does not hide the others; the job still exits
                     non-zero when any group failed.
"""

import argparse
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PHASES = ("preflight", "functional")
POLICIES = ("fail-fast", "continue-after-failure")

# Directory (under the integration workdir) that receives generated reports.
REPORT_DIR = "integration-reports"

# pytest's own exit code for "no tests were collected".
PYTEST_EXIT_NO_TESTS_COLLECTED = 5

ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def _require_field(entry, index, field, allowed):
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(
            f"integration_tests[{index}].{field} is required and must be a "
            f"non-empty string (one of: {', '.join(allowed)})"
        )
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise SystemExit(
            f"integration_tests[{index}].{field} must be one of "
            f"{', '.join(allowed)}, got: {value}"
        )
    return normalized


def _entry_meta(entry, index):
    """Return (id, phase, failure_policy) from explicit manifest fields."""
    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not ID_PATTERN.fullmatch(entry_id.strip()):
        raise SystemExit(
            f"integration_tests[{index}].id is required and must match "
            f"[a-z0-9][a-z0-9-]* (lowercase slug), got: {entry_id!r}"
        )
    phase = _require_field(entry, index, "phase", PHASES)
    policy = _require_field(entry, index, "failure_policy", POLICIES)

    return entry_id.strip(), phase, policy


def _validate_entries(tests):
    """Validate every entry, plus the ordering and policy invariants."""
    seen_ids = set()
    metas = []
    for index, test in enumerate(tests, start=1):
        if not isinstance(test, dict):
            raise SystemExit(f"integration_tests[{index}] must be an object")
        name = test.get("name")
        command = test.get("command")
        if (
            not isinstance(name, str)
            or not name.strip()
            or "\n" in name
            or "\r" in name
        ):
            raise SystemExit(
                f"integration_tests[{index}].name must be a non-empty single-line string"
            )
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise SystemExit(
                f"integration_tests[{index}].command must be a non-empty string"
            )
        entry_id, phase, policy = _entry_meta(test, index)
        if entry_id in seen_ids:
            raise SystemExit(
                f"integration_tests[{index}].id is a duplicate: {entry_id!r}. "
                "Report file names are derived from the id, so it must be unique."
            )
        seen_ids.add(entry_id)
        metas.append((entry_id, phase, policy))

    # Preflight before functional. A functional group that runs ahead of the
    # model or device check reports a failure whose real cause is the missing
    # environment, which is the confusion this ordering rule prevents.
    seen_functional = False
    for index, (entry_id, phase, _policy) in enumerate(metas, start=1):
        if phase == "functional":
            seen_functional = True
        elif seen_functional:
            raise SystemExit(
                f"integration_tests[{index}] ({entry_id}) has phase 'preflight' "
                "after a functional entry; every preflight entry must come first"
            )

    # At least one fail-fast entry, so a broken environment stops the run instead
    # of producing a full functional report of environment-induced failures.
    if metas and all(policy == "continue-after-failure" for _i, _p, policy in metas):
        raise SystemExit(
            "integration_tests must have at least one fail-fast entry "
            "(preflight checks should abort on failure)"
        )
    return metas


def load_configuration(allow_empty):
    try:
        tests = json.loads(os.environ.get("INTEGRATION_TESTS", "[]"))
        environment = json.loads(os.environ.get("INTEGRATION_ENVIRONMENT", "{}"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid integration configuration JSON: {error}") from error

    if not isinstance(tests, list):
        raise SystemExit("integration_tests must be an array")
    if not allow_empty and not tests:
        raise SystemExit("integration_tests must not be empty")

    if not isinstance(environment, dict):
        raise SystemExit("integration_environment must be an object")
    for key, value in environment.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f"invalid environment variable name: {key}")
        if (
            not isinstance(value, str)
            or "\n" in value
            or "\r" in value
            or "\0" in value
        ):
            raise SystemExit(
                f"integration_environment[{key}] must be a single-line string"
            )

    _validate_entries(tests)
    return tests, environment


def _command_is_pytest(command):
    # Detect a pytest invocation so per-group JUnit can be requested without the
    # manifest having to repeat --junitxml. shlex.split is deliberately not used:
    # the manifest also carries heredocs and multi-line shell, which it mangles.
    if "<<" in command or "\n" in command:
        return False
    return bool(re.search(r"(?:^|\s)(?:python3?\s+-m\s+)?pytest(?:\s|$)", command))


def _has_junitxml(command):
    return "--junitxml" in command or "--junit-xml" in command


def _parse_junit(junit_path):
    """Return per-outcome counts from a pytest JUnit XML file.

    pytest records an xfailed test as <skipped type="pytest.xfail">, so the
    skipped total already covers xfail; the split is kept for the report.
    """
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "xfailed": 0}
    for suite in suites:
        for case in suite.findall("testcase"):
            counts["tests"] += 1
            if case.find("failure") is not None:
                counts["failures"] += 1
            if case.find("error") is not None:
                counts["errors"] += 1
            skipped = case.find("skipped")
            if skipped is not None:
                counts["skipped"] += 1
                if (skipped.get("type") or "").endswith("xfail"):
                    counts["xfailed"] += 1
    counts["executed"] = counts["tests"] - counts["skipped"]
    return counts


def _judge_pytest_group(exit_code, junit_path):
    """Return (violations, counts) for a pytest group.

    A zero exit code is not proof the group ran. Require real executed tests.
    Any skip or xfail is a failure: a capability the platform does not have must
    surface as a real test failure, not a skip that reports success while erasing
    the evidence.

    Exit code 5 (no tests collected) and zero-execution groups always fail.
    Real test failures and errors always fail.
    """
    if exit_code == PYTEST_EXIT_NO_TESTS_COLLECTED:
        return (
            [
                "pytest collected no tests (exit code 5); the path or marker "
                "selection no longer matches any test"
            ],
            None,
        )
    if not junit_path.is_file():
        return (
            [f"no JUnit report was produced at {junit_path.name}"],
            None,
        )
    try:
        counts = _parse_junit(junit_path)
    except ET.ParseError as error:
        return ([f"JUnit report {junit_path.name} is not parseable: {error}"], None)

    violations = []
    if counts["tests"] == 0:
        violations.append("the JUnit report contains no test cases")
    elif counts["executed"] == 0:
        violations.append(
            f"every collected test was skipped ({counts['skipped']} skipped); "
            "a group with zero executed tests is not evidence of support"
        )
    elif counts["skipped"]:
        violations.append(
            f"{counts['skipped']} test(s) were skipped "
            f"({counts['xfailed']} of them xfailed); an unsupported capability "
            "must fail rather than skip"
        )
    return violations, counts


def run_entry(test, meta, command_environment, workdir, report_root, index, total):
    entry_id, phase, policy = meta
    name = test["name"]
    command = test["command"]

    log_path = report_root / "integration.log"
    junit_path = report_root / f"junit-{entry_id}.xml"

    header = f"===== [{index}/{total}] {name} ({phase}/{entry_id}) ====="
    print(header, flush=True)

    run_command = command
    is_pytest = _command_is_pytest(command)
    if is_pytest and not _has_junitxml(command):
        run_command = f"{command} --junitxml={junit_path}"

    start = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n{header}\n")
        log_fh.flush()
        # Streamed line by line rather than captured and printed at the end, so a
        # group that hangs still shows how far it got before the job timeout.
        process = subprocess.Popen(
            ["bash", "-c", f"set -euo pipefail\n{run_command}"],
            env=command_environment,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout:
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                print(line, end="", flush=True)
                log_fh.write(line)
        exit_code = process.wait()

        violations = []
        counts = None
        if is_pytest:
            violations, counts = _judge_pytest_group(exit_code, junit_path)

        status = "passed" if exit_code == 0 and not violations else "failed"
        if status == "failed":
            signal_desc = ""
            if exit_code < 0:
                signal_desc = f" (signal {-exit_code})"
            elif exit_code == 134:
                signal_desc = " (SIGABRT)"
            log_fh.write(
                f"\n--- {name} FAILED (exit code {exit_code}{signal_desc}) ---\n"
            )
        for violation in violations:
            message = f"::error::{name}: {violation}"
            print(message, flush=True)
            log_fh.write(f"{violation}\n")

    duration = time.monotonic() - start
    result = {
        "id": entry_id,
        "name": name,
        "phase": phase,
        "failure_policy": policy,
        "command": command,
        "exit_code": exit_code,
        "duration_seconds": round(duration, 3),
        "status": status,
        "violations": violations,
    }
    if counts is not None:
        result["counts"] = counts
    return result


def _skipped_result(test, meta):
    entry_id, phase, policy = meta
    return {
        "id": entry_id,
        "name": test["name"],
        "phase": phase,
        "failure_policy": policy,
        "command": test["command"],
        "exit_code": None,
        "duration_seconds": 0.0,
        "status": "not-run",
        "violations": [],
    }


def write_summary(results, success, out_dir):
    summary = {
        "success": success,
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "not_run": sum(1 for r in results if r["status"] == "not-run"),
        "entries": results,
    }
    (out_dir / "integration-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    gha = os.environ.get("GITHUB_STEP_SUMMARY")
    if not gha:
        return
    with open(gha, "a", encoding="utf-8") as fh:
        fh.write("## Integration test results\n\n")
        fh.write(f"- Total: {summary['total']}\n")
        fh.write(f"- Passed: {summary['passed']}\n")
        fh.write(f"- Failed: {summary['failed']}\n")
        fh.write(f"- Not run (aborted): {summary['not_run']}\n\n")
        if summary["failed"]:
            fh.write("| Entry | Phase | Exit | Tests | Reason |\n")
            fh.write("| --- | --- | ---: | ---: | --- |\n")
            for r in results:
                if r["status"] != "failed":
                    continue
                counts = r.get("counts") or {}
                executed = counts.get("executed", "")
                reason = "; ".join(r["violations"]) or "non-zero exit code"
                fh.write(
                    f"| {r['name']} | {r['phase']} | {r['exit_code']} "
                    f"| {executed} | {reason} |\n"
                )
            fh.write("\n")


def _print_final_report(results):
    print("\n===== Integration summary =====", flush=True)
    for r in results:
        counts = r.get("counts") or {}
        detail = ""
        if counts:
            detail = (
                f" [{counts['executed']} executed, {counts['failures']} failed, "
                f"{counts['skipped']} skipped]"
            )
        print(f"  {r['status']:<8} {r['name']}{detail}", flush=True)
    failures = [r for r in results if r["status"] == "failed"]
    if not failures:
        return
    # Every failing group is listed, not just the first: the point of
    # continue-after-failure is a complete picture of what the platform fails.
    print("\nFailed groups:", flush=True)
    for r in failures:
        reason = "; ".join(r["violations"]) or f"exit code {r['exit_code']}"
        print(f"  - {r['name']} ({r['id']}): {reason}", flush=True)


def main():
    args = parse_args()
    tests, environment = load_configuration(args.allow_empty)
    if args.validate_only:
        print(f"Validated {len(tests)} configured integration test(s)")
        return 0

    command_environment = os.environ.copy()
    command_environment.update(environment)
    workdir = os.environ.get("INTEGRATION_WORKDIR")
    if not workdir:
        raise SystemExit("INTEGRATION_WORKDIR is not set")
    if not os.path.isdir(workdir):
        raise SystemExit(f"INTEGRATION_WORKDIR does not exist: {workdir}")

    report_root = Path(workdir) / REPORT_DIR
    report_root.mkdir(parents=True, exist_ok=True)

    metas = _validate_entries(tests)
    results = []
    aborted_at = None
    for index, (test, meta) in enumerate(zip(tests, metas), start=1):
        entry = run_entry(
            test, meta, command_environment, workdir, report_root, index, len(tests)
        )
        results.append(entry)
        if entry["status"] != "failed":
            continue
        if entry["failure_policy"] == "fail-fast":
            print(
                f"'{entry['name']}' failed and its policy is fail-fast; "
                "aborting the remaining entries",
                flush=True,
            )
            aborted_at = index
            break
        print(
            f"'{entry['name']}' failed; policy is continue-after-failure, "
            "continuing to collect the remaining groups",
            flush=True,
        )

    if aborted_at is not None:
        for test, meta in zip(tests[aborted_at:], metas[aborted_at:]):
            results.append(_skipped_result(test, meta))

    success = all(r["status"] == "passed" for r in results)
    _print_final_report(results)
    write_summary(results, success, report_root)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
