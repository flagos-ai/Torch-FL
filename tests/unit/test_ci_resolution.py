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

"""Execute trusted checkout resolution with a fake, local-only GitHub CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
RUN = WORKFLOW["jobs"]["resolve-checkout"]["steps"][0]["run"]
SHA = "a" * 40


def _resolve(tmp_path, number="", pr=None, base_sha="", main_sha=None):
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if '--jq' in args:\n"
        "    args = args[:args.index('--jq')]\n"
        "if 'commits/main' in args[-1]:\n"
        "    print(os.environ['FAKE_MAIN_SHA'])\n"
        "else:\n"
        "    print(os.environ['FAKE_PR'])\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "output"
    output.touch()
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "EVENT_REPOSITORY": "flagos-ai/Torch-FL",
        "EVENT_REF": SHA,
        "PR_NUMBER": number,
        "PR_BASE_SHA": base_sha,
        "GITHUB_OUTPUT": str(output),
        "FAKE_PR": json.dumps(pr or {}),
        "FAKE_MAIN_SHA": main_sha or SHA,
    }
    process = subprocess.run(
        ["bash", "-e", "-c", RUN], env=env, capture_output=True, text=True
    )
    return process, output.read_text()


def test_normal_run_pins_the_event_commit(tmp_path):
    process, output = _resolve(tmp_path)
    assert process.returncode == 0, process.stderr
    assert output == (
        f"repository=flagos-ai/Torch-FL\n"
        f"ref={SHA}\n"
        "base_repository=flagos-ai/Torch-FL\n"
        f"base_ref={SHA}\n"
    )


def test_fork_run_resolves_both_repository_and_immutable_head(tmp_path):
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "contributor/fork"}, "sha": "b" * 40},
    }
    process, output = _resolve(tmp_path, "42", pr)
    assert process.returncode == 0, process.stderr
    assert output == (
        f"repository=contributor/fork\n"
        f"ref={'b' * 40}\n"
        "base_repository=flagos-ai/Torch-FL\n"
        f"base_ref={SHA}\n"
    )


def test_pull_request_uses_the_base_sha_for_the_issue_query(tmp_path):
    base_sha = "c" * 40
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "contributor/fork"}, "sha": "b" * 40},
    }
    process, output = _resolve(tmp_path, "42", pr, base_sha)
    assert process.returncode == 0, process.stderr
    assert "base_repository=flagos-ai/Torch-FL\n" in output
    assert f"base_ref={base_sha}\n" in output


def test_invalid_pull_request_base_fails_closed(tmp_path):
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "contributor/fork"}, "sha": SHA},
    }
    process, output = _resolve(tmp_path, "42", pr, "main")
    assert process.returncode == 1
    assert output == ""


@pytest.mark.parametrize("number", ["0", "-1", "01", "42; echo unsafe", "$(whoami)"])
def test_invalid_pr_numbers_fail_before_any_checkout(tmp_path, number):
    process, output = _resolve(tmp_path, number)
    assert process.returncode == 1
    assert output == ""


@pytest.mark.parametrize(
    "pr",
    [
        {
            "state": "closed",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": SHA},
        },
        {"state": "open", "head": {"repo": None, "sha": SHA}},
        {
            "state": "open",
            "head": {"repo": {"full_name": "contributor/fork"}, "sha": "main"},
        },
        {
            "state": "open",
            "head": {"repo": {"full_name": "bad/repo/extra"}, "sha": SHA},
        },
    ],
)
def test_unavailable_or_invalid_heads_fail_before_checkout(tmp_path, pr):
    process, output = _resolve(tmp_path, "42", pr)
    assert process.returncode == 1
    assert output == ""


def test_label_events_and_dispatches_do_not_cancel_each_other_or_main_runs():
    events = WORKFLOW[True]
    assert {"labeled", "unlabeled"} <= set(events["pull_request"]["types"])
    assert "paths-ignore" not in events["push"]
    group = WORKFLOW["concurrency"]["group"]
    assert "github.event_name" in group
    assert "github.event.pull_request.number" in group
    assert "inputs.pr_number" in group


def test_privileged_control_workflows_only_checkout_the_trusted_event_commit():
    for filename in ("slash-command.yml", "flaky-retry.yml"):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
        for job in workflow["jobs"].values():
            checkouts = [
                step
                for step in job["steps"]
                if step.get("uses", "").startswith("actions/checkout@")
            ]
            assert len(checkouts) == 1
            assert checkouts[0]["with"]["ref"] == "${{ github.sha }}"
            assert checkouts[0]["with"]["persist-credentials"] is False
        assert "contents" in workflow["permissions"]


def test_case_gap_controller_and_data_come_from_the_same_submitted_revision():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/case-gap-issues.yml").read_text()
    )
    checkouts = [
        step
        for step in workflow["jobs"]["verify-gate"]["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2
    registry, trusted = (step["with"] for step in checkouts)
    assert (
        trusted["repository"]
        == "${{ inputs.registry_repository || github.repository }}"
    )
    assert trusted["ref"] == "${{ inputs.registry_ref || github.sha }}"
    assert registry["repository"].startswith("${{ inputs.registry_repository || ")
    assert registry["ref"].startswith("${{ inputs.registry_ref || ")
    assert all(step["with"]["persist-credentials"] is False for step in checkouts)
