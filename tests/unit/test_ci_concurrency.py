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

"""Cancel only CI runs replaced by a newer PR head or branch push."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"
sys.path.insert(0, str(SCRIPTS))

import ci_concurrency  # noqa: E402

OLD_SHA = "a" * 40
SHA = "b" * 40
REPOSITORY = "flagos-ai/Torch-FL"


class FakeGitHub:
    def __init__(self, runs=(), pull_request=None):
        self.repository = REPOSITORY
        self.runs = list(runs)
        self.pull_request = pull_request or {
            "state": "open",
            "head": {
                "sha": SHA,
                "ref": "feature",
                "repo": {"full_name": "contributor/fork"},
            },
        }
        self.calls = []

    def command(self, *args, **kwargs):
        self.calls.append(("command", args, kwargs))
        return json.dumps([{"workflow_runs": self.runs}])

    def repo_api(self, suffix, **kwargs):
        self.calls.append(("repo_api", suffix, kwargs))
        if suffix.startswith("pulls/"):
            return self.pull_request
        if suffix.endswith("/cancel"):
            return None
        raise AssertionError(suffix)

    def cancelled(self):
        return [
            int(suffix.split("/")[2])
            for kind, suffix, kwargs in self.calls
            if kind == "repo_api"
            and suffix.endswith("/cancel")
            and kwargs["method"] == "POST"
        ]


def _run(
    run_id,
    sha=OLD_SHA,
    *,
    event="pull_request",
    status="queued",
    branch="feature",
    repository="contributor/fork",
    workflow_id=7,
):
    record = {
        "id": run_id,
        "status": status,
        "event": event,
        "head_sha": sha,
        "head_branch": branch,
        "head_repository": {"full_name": repository} if repository else None,
        "path": ".github/workflows/ci.yml",
    }
    record["workflow_id"] = workflow_id
    return record


def test_pr_update_cancels_only_active_runs_from_its_source_branch():
    github = FakeGitHub(
        [
            _run(1),
            _run(2, status="in_progress"),
            _run(3, sha=SHA),
            _run(4, status="completed"),
            _run(5, repository="someone/other"),
            _run(6, branch="other"),
            _run(
                7,
                event="push",
                repository=REPOSITORY,
            ),
            _run(8, repository=None),
        ]
    )
    assert ci_concurrency.cancel_pull_request(github, SHA, "42", current_run=2) == [
        1,
        3,
    ]
    assert github.cancelled() == [1, 3]
    listing = next(call for call in github.calls if call[0] == "command")[1]
    assert any("exclude_pull_requests=true" in arg for arg in listing)
    assert "--slurp" in listing


def test_push_replaces_only_older_pushes_to_the_same_branch():
    github = FakeGitHub(
        [
            _run(
                1,
                event="push",
                branch="main",
                repository=REPOSITORY,
            ),
            _run(2, sha=SHA, event="push", branch="main", repository=REPOSITORY),
            _run(
                3,
                event="push",
                branch="main",
                repository="contributor/fork",
            ),
            _run(4),
            _run(5, event="push", branch="dev", repository=REPOSITORY),
        ]
    )
    assert ci_concurrency.cancel_push(github, SHA, "main") == [1]
    assert github.cancelled() == [1]


def test_listing_without_run_path_still_matches_the_ci_workflow_id():
    run = _run(
        1,
        event="push",
        branch="main",
        repository=REPOSITORY,
    )
    run.pop("path")
    github = FakeGitHub([run])
    assert ci_concurrency.cancel_push(github, SHA, "main") == [1]


def test_listing_rejects_a_different_workflow():
    github = FakeGitHub([{**_run(1), "path": ".github/workflows/other.yml"}])
    with pytest.raises(ValueError, match="invalid workflow run listing"):
        ci_concurrency.cancel_push(github, SHA, "feature")
    assert not github.cancelled()


def test_workflow_top_level_concurrency_is_rejected():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert "concurrency" not in workflow


def test_closed_or_invalid_pr_does_not_cancel_anything():
    closed = FakeGitHub([_run(1)], pull_request={"state": "closed", "head": {}})
    with pytest.raises(ValueError, match="not open"):
        ci_concurrency.cancel_pull_request(closed, SHA, "42")
    missing_source = FakeGitHub(
        [_run(1)],
        pull_request={
            "state": "open",
            "head": {
                "sha": OLD_SHA,
                "ref": "feature",
                "repo": None,
            },
        },
    )
    with pytest.raises(ValueError, match="no valid source"):
        ci_concurrency.cancel_pull_request(missing_source, SHA, "42")
    assert not closed.cancelled()
    assert not missing_source.cancelled()


def test_pr_cleanup_uses_the_live_head_and_cancels_a_run_that_started_late():
    newer_head = "c" * 40
    github = FakeGitHub(
        [
            _run(1, sha=newer_head),
        ],
        pull_request={
            "state": "open",
            "head": {
                "sha": newer_head,
                "ref": "feature",
                "repo": {"full_name": "contributor/fork"},
            },
        },
    )
    assert ci_concurrency.cancel_pull_request(github, SHA, "42") == [1]


def test_cancel_failure_is_a_warning_and_other_runs_are_still_processed():
    github = FakeGitHub([_run(1), _run(2)])
    original = github.repo_api

    def partially_failing(suffix, **kwargs):
        if suffix == "actions/runs/1/cancel":
            raise ci_concurrency.GitHubError("permission denied")
        return original(suffix, **kwargs)

    github.repo_api = partially_failing
    assert ci_concurrency.cancel_pull_request(github, SHA, "42") == [2]


def test_rejects_unsafe_command_line_values():
    with pytest.raises(ValueError):
        ci_concurrency.cancel_push(FakeGitHub(), "main; echo unsafe", "main")
    with pytest.raises(ValueError):
        ci_concurrency.cancel_push(FakeGitHub(), SHA, "main\n::error::injected")
    with pytest.raises(ValueError):
        ci_concurrency.cancel_pull_request(FakeGitHub(), SHA, "0")
