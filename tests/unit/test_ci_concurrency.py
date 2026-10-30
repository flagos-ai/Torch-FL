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
import os
import subprocess
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


def test_pr_cleanup_uses_the_live_branch_after_a_concurrent_update():
    newer_head = "c" * 40
    github = FakeGitHub(
        [
            _run(1),
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


def test_read_only_fork_token_skips_cleanup_without_failing_the_lifecycle(monkeypatch):
    github = FakeGitHub([_run(1)])

    def read_only(*args, **kwargs):
        raise ci_concurrency.GitHubError(
            "HTTP 403: Resource not accessible by integration"
        )

    github.repo_api = read_only
    monkeypatch.setattr(ci_concurrency, "GitHub", lambda repository: github)
    assert (
        ci_concurrency.main(
            [
                "--repository",
                REPOSITORY,
                "--revision",
                SHA,
                "--pull-request=42",
                "--current-run=2",
            ]
        )
        == 0
    )


def test_other_cleanup_failures_are_errors(monkeypatch):
    github = FakeGitHub()

    def unavailable(*args, **kwargs):
        raise ci_concurrency.GitHubError("GitHub API unavailable")

    github.command = unavailable
    monkeypatch.setattr(ci_concurrency, "GitHub", lambda repository: github)
    assert (
        ci_concurrency.main(
            [
                "--repository",
                REPOSITORY,
                "--revision",
                SHA,
                "--branch=main",
            ]
        )
        == 1
    )


def test_rejects_unsafe_command_line_values():
    with pytest.raises(ValueError):
        ci_concurrency.cancel_push(FakeGitHub(), "main; echo unsafe", "main")
    with pytest.raises(ValueError):
        ci_concurrency.cancel_push(FakeGitHub(), SHA, "main\n::error::injected")
    with pytest.raises(ValueError):
        ci_concurrency.cancel_pull_request(FakeGitHub(), SHA, "0")


@pytest.mark.parametrize(
    ("source", "selectors"),
    [
        (
            "pr",
            ["--revision", SHA, "--pull-request=42"],
        ),
        (
            "push",
            ["--revision", SHA, "--branch=main"],
        ),
    ],
)
def test_lifecycle_step_runs_the_cleanup_script(tmp_path, source, selectors):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["ci-lifecycle"]["steps"]
    assert steps[0]["uses"] == "actions/checkout@v4"
    assert steps[0]["with"] == {
        "repository": "${{ needs.resolve-checkout.outputs.repository }}",
        "ref": "${{ needs.resolve-checkout.outputs.ref }}",
        "persist-credentials": False,
    }
    run = steps[1]["run"]
    output = tmp_path / "record"

    stub = tmp_path / "python3"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "pathlib.Path(os.environ['RECORD_PATH']).write_text(\n"
        "    os.environ['PYTHONPATH'] + '\\n' + '\\n'.join(sys.argv[1:])\n"
        ")\n"
    )
    stub.chmod(0o755)
    process = subprocess.run(
        ["bash", "-e", "-c", run],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "EVENT_NAME": "pull_request" if source == "pr" else "push",
            "SOURCE": source,
            "GITHUB_REPOSITORY": REPOSITORY,
            "REVISION": SHA,
            "PR_NUMBER": "42",
            "BRANCH": "main",
            "DISPATCH_PR": "",
            "GITHUB_RUN_ID": "123",
            "RECORD_PATH": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    recorded = output.read_text().splitlines()
    assert recorded[0] == ".github/scripts/ci"
    assert recorded[1] == ".github/scripts/ci/ci_concurrency.py"
    assert recorded[2:4] == ["--repository", REPOSITORY]
    assert all(selector in recorded for selector in selectors)
    assert recorded[-2:] == ["--current-run", "123"]


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_lifecycle_step_is_a_no_op_where_nothing_is_superseded(tmp_path, event):
    """An event with no earlier run has to leave the step green, not red.

    The nightly schedule has no newer head of a PR and no older push to the
    same branch, and a workflow_dispatch without `pr_number` is a platforms-only
    rerun of whatever it was dispatched on. Neither supersedes anything, and
    neither may fail: every job in ci.yml `needs` this one and the required gate
    reads a non-success dependency as an error, so a failure here is a run that
    tests nothing and reports red. Run against the step's real body, with a
    `python3` that fails if it is called at all.
    """
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    run = next(
        step["run"]
        for step in workflow["jobs"]["ci-lifecycle"]["steps"]
        if "ci_concurrency.py" in step.get("run", "")
    )

    stub = tmp_path / "python3"
    stub.write_text("#!/bin/sh\necho 'cleanup ran' >&2\nexit 9\n")
    stub.chmod(0o755)
    process = subprocess.run(
        ["bash", "-e", "-c", run],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "EVENT_NAME": event,
            # What resolve-checkout emits for these events: no pull request, so
            # the selector would fall through to the push branch and use the
            # ref it happens to be on.
            "SOURCE": "push",
            "GITHUB_REPOSITORY": REPOSITORY,
            "REVISION": SHA,
            "PR_NUMBER": "",
            "BRANCH": "main",
            "DISPATCH_PR": "",
            "GITHUB_RUN_ID": "123",
        },
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "cleanup ran" not in process.stderr, (
        f"the cleanup script ran on {event}, which supersedes no earlier run"
    )


def test_lifecycle_step_still_cleans_up_a_dispatched_pull_request(tmp_path):
    """A workflow_dispatch carrying `pr_number` does supersede the PR's runs.

    The same event name is both a no-op and a cleanup depending on that input,
    so the early exit has to read it rather than the event name alone.
    """
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    run = next(
        step["run"]
        for step in workflow["jobs"]["ci-lifecycle"]["steps"]
        if "ci_concurrency.py" in step.get("run", "")
    )
    output = tmp_path / "record"
    stub = tmp_path / "python3"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "pathlib.Path(os.environ['RECORD_PATH']).write_text('\\n'.join(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)
    process = subprocess.run(
        ["bash", "-e", "-c", run],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "EVENT_NAME": "workflow_dispatch",
            "SOURCE": "pr",
            "GITHUB_REPOSITORY": REPOSITORY,
            "REVISION": SHA,
            "PR_NUMBER": "42",
            "BRANCH": "main",
            "DISPATCH_PR": "42",
            "GITHUB_RUN_ID": "123",
            "RECORD_PATH": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    assert "--pull-request=42" in output.read_text().splitlines()
