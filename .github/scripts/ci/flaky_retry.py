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

"""Retry a failed CI attempt once, only when every root failure is network-related."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from github_api import GitHub, GitHubError

# Device faults, hangs, OOM and plain timeouts may be deterministic bugs. They
# require investigation rather than being silently labelled as flaky.
BUG = re.compile(
    r"AssertionError|\bFAILED\s+.*\.py|=+ FAILURES =+|"
    r"ModuleNotFoundError|ImportError|undefined symbol|CMake Error|fatal error:|"
    r"SyntaxError|NameError|TypeError|KeyError|ValueError|"
    r"\.(?:cc|cpp|c|h|cu):\d+(?::\d+)?:\s*(?:fatal\s+)?error:|"
    r"ninja: build stopped|collect2: error:|"
    r"Segmentation fault|device-side assert|out of memory|OutOfMemoryError|"
    r"No matching distribution found|Could not find a version that satisfies|"
    r"(?:^|\n)\s*E\s+assert\b",
    re.IGNORECASE,
)
AMBIGUOUS = re.compile(
    r"(?:watchdog|collective).*(?:timeout|timed out)|"
    r"(?:NCCL|HCCL|XCCL) error|EE9999|ERR02005|"
    r"killed by (?:signal|OOM)|SIGKILL",
    re.IGNORECASE,
)
NETWORK = re.compile(
    r"ReadTimeoutError|requests\.exceptions\.(?:ReadTimeout|ConnectionError)|"
    r"Could not resolve host|Temporary failure in name resolution|"
    r"Connection reset by peer|Connection timed out|RemoteDisconnected|"
    r"Failed to connect to .+ port \d+|SSL(?:Error|: EOF)|"
    r"TLS (?:handshake|connection).*(?:failed|timeout)|"
    r"HTTP(?: Error| status(?: code)?|/\d(?:\.\d)?)?[: ]+(?:502|503|504)\b|"
    r"(?:502 Bad Gateway|503 Service Unavailable|504 Gateway Timeout)|"
    r"curl: \(?(?:6|7|18|28|35|52|56)\)?",
    re.IGNORECASE,
)


def classify(log):
    if not log.strip():
        return "unknown"
    if BUG.search(log):
        return "bug"
    if AMBIGUOUS.search(log):
        return "unknown"
    if NETWORK.search(log):
        return "network"
    return "unknown"


def retry(event, github):
    trigger = event.get("workflow_run", {})
    if trigger.get("name") != "CI" or trigger.get("conclusion") != "failure":
        return "No retry: this is not a failed CI run."
    run_id = trigger.get("id")
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("Invalid workflow run ID.")
    run = github.repo_api(f"actions/runs/{run_id}")
    if run.get("path", "").split("@")[0] != ".github/workflows/ci.yml":
        return "No retry: the run is not the configured CI workflow."
    if run.get("status") != "completed" or run.get("conclusion") != "failure":
        return "No retry: the run is no longer a completed failure."
    if run.get("run_attempt") != 1:
        return "No retry: the one-retry budget is already used."
    if run.get("head_sha") != trigger.get("head_sha"):
        return "No retry: the run commit differs from the triggering event."
    pages = github.repo_api(
        f"actions/runs/{run_id}/attempts/1/jobs?per_page=100", paginate=True
    )
    jobs = [job for page in pages for job in page["jobs"]]
    if any(job.get("conclusion") == "cancelled" for job in jobs):
        return "No retry: at least one job was cancelled."
    failed = [job for job in jobs if job.get("conclusion") in {"failure", "timed_out"}]
    # CI complete is a dependent summary, not an independent root cause.
    roots = [job for job in failed if job.get("name") != "CI complete"]
    if not roots:
        return "No retry: no failed root job is available for classification."
    for job in roots:
        log = github.command(
            "run",
            "view",
            str(run_id),
            "--repo",
            github.repository,
            "--job",
            str(job["id"]),
            "--log-failed",
        )
        verdict = classify(log)
        if verdict != "network":
            return f"No retry: a root failure is {verdict}; the original CI failure is retained."
    # Recheck the attempt immediately before the write; duplicate queued events
    # or a maintainer's manual rerun must not spend a second retry budget.
    current = github.repo_api(f"actions/runs/{run_id}")
    if (
        current.get("run_attempt") != 1
        or current.get("status") != "completed"
        or current.get("conclusion") != "failure"
    ):
        return "No retry: another rerun or state change already occurred."
    github.command("run", "rerun", str(run_id), "--repo", github.repository, "--failed")
    return f"Retry requested for CI run {run_id}: all {len(roots)} root failures matched network errors."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-file", default=os.environ.get("GITHUB_EVENT_PATH"))
    args = parser.parse_args()
    event = json.loads(Path(args.event_file).read_text())
    try:
        message = retry(event, GitHub(event["repository"]["full_name"]))
        print(message)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
                summary.write(message + "\n")
    except (GitHubError, ValueError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
