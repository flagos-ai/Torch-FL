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

"""Cancel active CI runs replaced by a newer PR head or branch push."""

import argparse
import json
import os
import re
import sys

from github_api import GitHub, GitHubError

SHA = re.compile(r"[0-9a-f]{40}")
ACTIVE_STATES = {"queued", "in_progress", "waiting", "requested", "pending"}
CI_WORKFLOW = ".github/workflows/ci.yml"


def _revision(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("revision must be a full lowercase commit SHA")
    return value


def _pull_request(value):
    number = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("pull request must be a positive integer")
    return int(number)


def _branch(value):
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n"):
        raise ValueError("branch must be a nonempty single line")
    return value


def _records(github):
    """Read every active ci.yml run, including those without PR metadata."""
    output = github.command(
        "api",
        f"repos/{github.repository}/actions/workflows/ci.yml/runs"
        "?per_page=100&exclude_pull_requests=true",
        "--paginate",
        "--slurp",
    )
    try:
        pages = json.loads(output) if output.strip() else []
    except json.JSONDecodeError as exc:
        raise ValueError("GitHub returned an invalid workflow run listing") from exc
    if not isinstance(pages, list) or not all(isinstance(page, dict) for page in pages):
        raise ValueError("GitHub returned an invalid workflow run listing")
    runs = []
    for page in pages:
        records = page.get("workflow_runs")
        if not isinstance(records, list):
            raise ValueError("GitHub returned an invalid workflow run listing")
        runs.extend(records)
    if not all(
        isinstance(run, dict)
        and isinstance(run.get("id"), int)
        and run["id"] > 0
        and isinstance(run.get("status"), str)
        and isinstance(run.get("event"), str)
        and isinstance(run.get("workflow_id"), int)
        and is_ci_run(run)
        for run in runs
    ):
        raise ValueError("GitHub returned an invalid workflow run listing")
    return runs


def _active(runs):
    return [run for run in runs if run["status"] in ACTIVE_STATES]


def is_ci_run(run):
    """Reject records GitHub attributes to another workflow."""
    return run.get("path") in (None, CI_WORKFLOW)


def _cancel(github, runs, current_run):
    cancelled = []
    for run in sorted(runs, key=lambda item: item["id"]):
        if run["id"] == current_run:
            continue
        try:
            github.repo_api(f"actions/runs/{run['id']}/cancel", method="POST")
            cancelled.append(run["id"])
        except GitHubError as exc:
            print(
                f"::warning::Could not cancel CI run {run['id']}: {exc}",
                file=sys.stderr,
            )
    return cancelled


def cancel_pull_request(github, revision, number, current_run=None):
    """Cancel active runs from an older commit of the same pull request.

    Query the live open PR after source resolution. Matching both its source
    repository and branch prevents a same-named branch elsewhere being cancelled.
    """
    revision = _revision(revision)
    number = _pull_request(number)
    pull = github.repo_api(f"pulls/{number}")
    if not isinstance(pull, dict) or pull.get("state") != "open":
        raise ValueError(f"PR #{number} is not open")
    head = pull.get("head") if isinstance(pull.get("head"), dict) else {}
    _revision(head.get("sha"))
    source = head.get("repo") if isinstance(head.get("repo"), dict) else {}
    repository = source.get("full_name")
    branch = _branch(head.get("ref"))
    if not (
        isinstance(repository, str)
        and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        and repository
    ):
        raise ValueError(f"PR #{number} has no valid source repository")

    candidates = []
    for run in _active(_records(github)):
        # A queued run may update this PR only after its API lookup. A queued
        # run for the current head may have started during an older revision's
        # API call, so cancel every active PR run whose source still matches.
        if run.get("event") != "pull_request":
            continue
        run_source = run.get("head_repository")
        run_repository = (
            run_source.get("full_name") if isinstance(run_source, dict) else None
        )
        if run.get("head_branch") == branch and run_repository == repository:
            candidates.append(run)
    return _cancel(github, candidates, current_run)


def cancel_push(github, revision, branch, current_run=None):
    """Cancel active runs for older pushes to one repository branch."""
    revision = _revision(revision)
    branch = _branch(branch)
    candidates = []
    for run in _active(_records(github)):
        head_repository = run.get("head_repository")
        if (
            run.get("event") == "push"
            and run.get("head_branch") == branch
            and run.get("head_sha") != revision
            and (
                head_repository is None
                or (
                    isinstance(head_repository, dict)
                    and head_repository.get("full_name") == github.repository
                )
            )
        ):
            candidates.append(run)
    return _cancel(github, candidates, current_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--revision")
    parser.add_argument("--pull-request")
    parser.add_argument("--branch")
    parser.add_argument("--current-run")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        print("::error::repository must be owner/name", file=sys.stderr)
        return 2
    if (args.pull_request is None) == (args.branch is None):
        print("::error::select one of --pull-request or --branch", file=sys.stderr)
        return 2
    try:
        current_run = int(args.current_run) if args.current_run else None
        if current_run is not None and current_run <= 0:
            raise ValueError("current run must be a positive integer")
        if args.pull_request is None and args.revision is None:
            raise ValueError("--revision is required for branch cleanup")
        github = GitHub(args.repository)
        if args.pull_request is not None:
            cancelled = cancel_pull_request(
                github, args.revision, args.pull_request, current_run
            )
        else:
            cancelled = cancel_push(github, args.revision, args.branch, current_run)
    except (GitHubError, ValueError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(f"Cancelled {len(cancelled)} superseded CI run(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
