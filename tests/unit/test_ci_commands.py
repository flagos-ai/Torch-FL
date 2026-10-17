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

"""Run comment dispatch and retry policy against simulated GitHub responses."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


API = _load("github_api")
_load("generate_matrix")
SLASH = _load("slash_command")
FLAKY = _load("flaky_retry")
PLATFORMS = [
    "ascend",
    "cuda",
    "dcu",
    "gcu",
    "iluvatar",
    "kunlunxin",
    "metax",
    "musa",
    "ppu",
]
REPOSITORY = "flagos-ai/Torch-FL"
SHA = "a" * 40
COMMENT_EVENT = {
    "repository": {"full_name": REPOSITORY},
    "issue": {"number": 42, "pull_request": {"url": "pr"}},
    "comment": {"id": 9, "body": "/ci run ascend", "user": {"login": "reviewer"}},
}
RUN_EVENT = {
    "workflow_run": {"name": "CI", "id": 7, "conclusion": "failure", "head_sha": SHA}
}


class FakeGitHub:
    repository = REPOSITORY

    def __init__(self):
        self.calls = []
        self.permission = "write"
        self.pr = {
            "state": "open",
            "head": {"sha": SHA, "repo": {"full_name": "contributor/fork"}},
        }
        self.run = {
            "path": ".github/workflows/ci.yml",
            "status": "completed",
            "conclusion": "failure",
            "run_attempt": 1,
            "head_sha": SHA,
        }
        self.current = None
        self.run_reads = 0
        self.jobs = [
            {"id": 11, "name": "Build and test (CUDA)", "conclusion": "failure"},
            {"id": 12, "name": "CI complete", "conclusion": "failure"},
        ]
        self.log = "pip download: ReadTimeoutError: server timed out"
        self.logs = {}
        self.fail_dispatch = False
        self.fail_logs = False

    def api(self, path, **kwargs):
        self.calls.append(("api", path, kwargs))
        return {"default_branch": "main"}

    def repo_api(self, path, **kwargs):
        self.calls.append(("api", path, kwargs))
        if kwargs.get("method", "GET") != "GET":
            return None
        if path.startswith("collaborators/"):
            return {"permission": self.permission}
        if path.startswith("pulls/"):
            return self.pr
        if path.startswith("issues/"):
            return [[{"name": "platform:cuda"}, {"name": "bug"}]]
        if path.startswith("labels"):
            return [[{"name": "platform:cuda"}]]
        if path.endswith("/jobs?per_page=100"):
            return [{"jobs": self.jobs}]
        if path == "actions/runs/7":
            self.run_reads += 1
            return self.current if self.run_reads > 1 and self.current else self.run
        raise AssertionError(f"Unexpected API path: {path}")

    def command(self, *args):
        self.calls.append(("command", args, {}))
        if args[:2] == ("workflow", "run") and self.fail_dispatch:
            raise API.GitHubError("dispatch rejected")
        if args[:2] == ("run", "view"):
            if self.fail_logs:
                raise API.GitHubError("log unavailable")
            return self.logs.get(args[args.index("--job") + 1], self.log)
        return ""

    def writes(self):
        return [
            call
            for call in self.calls
            if call[0] == "api" and call[2].get("method", "GET") != "GET"
        ]

    def reruns(self):
        return [
            call
            for call in self.calls
            if call[0] == "command" and call[1][:2] == ("run", "rerun")
        ]


@pytest.mark.parametrize(
    "body, expected",
    [
        ("/ci run", PLATFORMS),
        ("/ci run all", PLATFORMS),
        ("/ci run musa ascend musa", ["ascend", "musa"]),
        ("/ci run cuda,gcu", ["cuda", "gcu"]),
        ("/ci run ascend\n/ci run cuda", ["ascend"]),
        ("ordinary comment", None),
    ],
)
def test_command_parser(body, expected):
    assert SLASH.parse_command(body, PLATFORMS) == expected


@pytest.mark.parametrize(
    "body",
    [
        "/ci",
        "/ci help",
        "/ci run all ascend",
        "/ci run missing",
        "/ci run $(touch /tmp/oops)",
        "/ci run cuda; curl example.invalid",
        "/ci run 'ascend'",
    ],
)
def test_invalid_commands_cannot_be_dispatched(body):
    with pytest.raises(ValueError):
        SLASH.parse_command(body, PLATFORMS)


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_unprivileged_users_cannot_run_hardware_even_if_they_authored_the_pr(
    permission,
):
    github = FakeGitHub()
    github.permission = permission
    event = copy.deepcopy(COMMENT_EVENT)
    event["issue"]["user"] = event["comment"]["user"]
    assert "write access" in SLASH.handle(event, PLATFORMS, github)
    assert not github.writes()
    assert not any(call[0] == "command" for call in github.calls)


@pytest.mark.parametrize("permission", ["write", "maintain", "admin"])
def test_trusted_fork_dispatch_uses_base_workflow_and_keeps_nonrouting_labels(
    permission,
):
    github = FakeGitHub()
    github.permission = permission
    assert "dispatched" in SLASH.handle(COMMENT_EVENT, PLATFORMS, github)
    commands = [call[1] for call in github.calls if call[0] == "command"]
    assert commands == [
        (
            "workflow",
            "run",
            "ci.yml",
            "--repo",
            REPOSITORY,
            "--ref",
            "main",
            "--field",
            "platforms=ascend",
            "--field",
            "pr_number=42",
        )
    ]
    deleted = [call[1] for call in github.writes() if call[2].get("method") == "DELETE"]
    assert deleted == ["issues/42/labels/platform%3Acuda"]
    assert any(call[1] == "issues/42/comments" for call in github.writes())
    assert any(call[1] == "issues/comments/9/reactions" for call in github.writes())


def test_all_command_labels_and_dispatches_full_matrix():
    github = FakeGitHub()
    event = copy.deepcopy(COMMENT_EVENT)
    event["comment"]["body"] = "/ci run all"
    assert "dispatched: all" in SLASH.handle(event, PLATFORMS, github)
    assert any(
        call[2].get("payload") == {"labels": ["platform:all"]}
        for call in github.writes()
    )


def test_invalid_command_is_explained_without_dispatch_or_label_mutation():
    github = FakeGitHub()
    event = copy.deepcopy(COMMENT_EVENT)
    event["comment"]["body"] = "/ci run unknown"
    assert "rejected" in SLASH.handle(event, PLATFORMS, github)
    assert len(github.writes()) == 1
    assert github.writes()[0][1] == "issues/42/comments"
    assert "/ci run all" in github.writes()[0][2]["payload"]["body"]


@pytest.mark.parametrize("state", ["closed", "deleted-fork"])
def test_unavailable_pr_cannot_dispatch(state):
    github = FakeGitHub()
    if state == "closed":
        github.pr["state"] = "closed"
        assert "closed" in SLASH.handle(COMMENT_EVENT, PLATFORMS, github)
    else:
        github.pr["head"]["repo"] = None
        with pytest.raises(ValueError):
            SLASH.handle(COMMENT_EVENT, PLATFORMS, github)
    assert not github.writes()


def test_failed_dispatch_does_not_claim_success():
    github = FakeGitHub()
    github.fail_dispatch = True
    with pytest.raises(API.GitHubError):
        SLASH.handle(COMMENT_EVENT, PLATFORMS, github)
    assert not any(call[1].startswith("issues/42/labels") for call in github.writes())
    assert not any(
        call[1].endswith(("/comments", "/reactions")) for call in github.writes()
    )


@pytest.mark.parametrize(
    "log",
    [
        "ReadTimeoutError: download.pytorch.org",
        "curl: (28) operation timed out",
        "Could not resolve host: pypi.org",
        "HTTP Error 503: Service Unavailable",
        "Connection reset by peer",
        "SSL: EOF occurred in violation of protocol",
    ],
)
def test_recognized_network_failures(log):
    assert FLAKY.classify(log) == "network"


@pytest.mark.parametrize(
    "log",
    [
        "",
        "TimeoutError: kernel never finished",
        "watchdog collective timed out",
        "driver error EE9999",
        "process killed by signal 9",
        "ReadTimeoutError\nwatchdog caught collective timeout",
        "Connection timed out\ndriver error EE9999",
    ],
)
def test_ambiguous_failures_are_not_automatically_retried(log):
    assert FLAKY.classify(log) == "unknown"


@pytest.mark.parametrize(
    "log",
    [
        "AssertionError: tensor values differ",
        "ModuleNotFoundError: torch_fl",
        "undefined symbol: c10::bad",
        "CUDA out of memory",
        "CMake Error: ABI mismatch",
        "fatal error: header.h: No such file",
        "Segmentation fault",
        "No matching distribution found for torch==unknown",
        "ReadTimeoutError followed by AssertionError: deterministic failure",
        "ReadTimeoutError\n/tmp/op.cc:45:3: error: invalid conversion",
        "ReadTimeoutError\nTypeError: unsupported tensor operation",
        "job\tstep\t2026-10-14T00:00:00Z FAILED tests/test_op.py::test_values",
    ],
)
def test_real_failures_override_network_noise(log):
    assert FLAKY.classify(log) == "bug"


def test_one_network_failure_retries_failed_jobs_and_dependents():
    github = FakeGitHub()
    assert "Retry requested" in FLAKY.retry(RUN_EVENT, github)
    assert len(github.reruns()) == 1
    assert github.reruns()[0][1] == (
        "run",
        "rerun",
        "7",
        "--repo",
        REPOSITORY,
        "--failed",
    )
    assert all("--repo" in call[1] for call in github.calls if call[0] == "command")


@pytest.mark.parametrize(
    "key, value",
    [
        ("run_attempt", 2),
        ("status", "in_progress"),
        ("conclusion", "success"),
        ("path", ".github/workflows/other.yml"),
        ("head_sha", "b" * 40),
    ],
)
def test_ineligible_run_never_retries(key, value):
    github = FakeGitHub()
    github.run[key] = value
    assert "No retry" in FLAKY.retry(RUN_EVENT, github)
    assert not github.reruns()


@pytest.mark.parametrize("log", ["AssertionError", "driver error", ""])
def test_one_nonnetwork_root_blocks_retry_even_if_another_root_is_network(log):
    github = FakeGitHub()
    github.jobs.append(
        {"id": 13, "name": "Build and test (Ascend)", "conclusion": "failure"}
    )
    github.logs["13"] = log
    assert "No retry" in FLAKY.retry(RUN_EVENT, github)
    assert not github.reruns()


def test_cancelled_job_blocks_retry():
    github = FakeGitHub()
    github.jobs.append({"id": 13, "name": "another job", "conclusion": "cancelled"})
    assert "cancelled" in FLAKY.retry(RUN_EVENT, github)
    assert not github.reruns()


def test_duplicate_event_after_manual_rerun_cannot_retry_twice():
    github = FakeGitHub()
    github.current = {**github.run, "run_attempt": 2}
    assert "another rerun" in FLAKY.retry(RUN_EVENT, github)
    assert not github.reruns()


def test_missing_logs_fail_without_requesting_retry():
    github = FakeGitHub()
    github.fail_logs = True
    with pytest.raises(API.GitHubError):
        FLAKY.retry(RUN_EVENT, github)
    assert not github.reruns()


def test_github_write_payload_is_passed_on_stdin_without_a_shell(monkeypatch):
    captured = {}

    def run(args, **kwargs):
        captured.update(args=args, **kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(API.subprocess, "run", run)
    API.GitHub(REPOSITORY).repo_api(
        "issues/42/comments",
        method="POST",
        payload={"body": "$(not executed)\n`also literal`"},
    )
    assert captured["args"] == [
        "gh",
        "api",
        f"repos/{REPOSITORY}/issues/42/comments",
        "--method",
        "POST",
        "--input",
        "-",
    ]
    assert json.loads(captured["input"])["body"] == "$(not executed)\n`also literal`"
    assert not captured.get("shell")
