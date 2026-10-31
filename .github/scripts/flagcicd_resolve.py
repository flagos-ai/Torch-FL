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

"""Resolve dispatch requests through GitHub before checking out source data."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote


def text(value, field):
    if (
        not isinstance(value, str)
        or len(value) > 256
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(
            f"{field} must be a single-line string of at most 256 characters"
        )
    return value


def api(path):
    result = subprocess.run(
        ["gh", "api", path], check=True, capture_output=True, text=True, timeout=60
    )
    return json.loads(result.stdout)


def commit(repository, ref):
    sha = api(f"repos/{repository}/commits/{quote(ref, safe='')}")["sha"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GitHub did not return a full commit SHA")
    return sha


def resolve(event, event_name, repository, event_sha):
    if event_name == "workflow_dispatch":
        request = event.get("inputs") or {}
    elif event_name == "repository_dispatch" and event.get("action") == "flagcicd":
        request = event.get("client_payload") or {}
    else:
        raise ValueError(
            "only workflow_dispatch or repository_dispatch:flagcicd is supported"
        )
    if not isinstance(request, dict):
        raise ValueError("dispatch request must be an object")
    profile = request.get("profile", "pr")
    if profile not in ("pr", "nightly"):
        raise ValueError("weekly is reserved; supported profiles are pr and nightly")
    number = str(request.get("pr_number", ""))
    checkout_ref = text(request.get("checkout_ref", ""), "checkout_ref")
    base_ref = text(request.get("base_ref", ""), "base_ref")
    if number and (checkout_ref or base_ref):
        raise ValueError("pr_number cannot be combined with checkout_ref or base_ref")
    source_repository = repository
    if number:
        if not re.fullmatch(r"[1-9][0-9]*", number):
            raise ValueError("pr_number must be a positive integer")
        pr = api(f"repos/{repository}/pulls/{number}")
        if pr.get("state") != "open":
            raise ValueError("the requested PR is not open")
        source_repository = (pr["head"].get("repo") or {}).get("full_name", "")
        source_sha = pr["head"]["sha"]
        base_sha = pr["base"]["sha"]
    else:
        source_sha = commit(repository, checkout_ref or event_sha)
        if not base_ref:
            base_ref = api(f"repos/{repository}")["default_branch"]
        base_sha = commit(repository, base_ref)
    if not isinstance(source_repository, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source_repository
    ):
        raise ValueError("source repository is unavailable")
    for sha in (source_sha, base_sha):
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("source and base must be immutable commit SHAs")
    requested_level = text(request.get("test_level", "auto"), "test_level")
    if requested_level not in ("auto", "L0", "L1", "L2"):
        raise ValueError("supported test levels are auto, L0, L1 and L2")
    return {
        "source_repository": source_repository,
        "source_sha": source_sha,
        "base_sha": base_sha,
        "platforms": request.get("platforms", "all"),
        "requested_level": requested_level,
        "profile": profile,
        "task_id": text(request.get("task_id", ""), "task_id"),
        "parent_run_id": text(request.get("parent_run_id", ""), "parent_run_id"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = resolve(
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_SHA"],
    )
    args.output.write_text(json.dumps(request, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key in ("source_repository", "source_sha", "base_sha"):
            output.write(f"{key}={request[key]}\n")


if __name__ == "__main__":
    main()
