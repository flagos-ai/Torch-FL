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


def _resolve(tmp_path, number="", pr=None):
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text("#!/usr/bin/env python3\nimport os\nprint(os.environ['FAKE_PR'])\n")
    gh.chmod(0o755)
    output = tmp_path / "output"
    output.touch()
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "EVENT_REPOSITORY": "flagos-ai/Torch-FL",
        "EVENT_REF": SHA,
        "PR_NUMBER": number,
        "GITHUB_OUTPUT": str(output),
        "FAKE_PR": json.dumps(pr or {}),
    }
    process = subprocess.run(
        ["bash", "-e", "-c", RUN], env=env, capture_output=True, text=True
    )
    return process, output.read_text()


def test_normal_run_pins_the_event_commit(tmp_path):
    process, output = _resolve(tmp_path)
    assert process.returncode == 0, process.stderr
    assert output == f"repository=flagos-ai/Torch-FL\nref={SHA}\n"


def test_fork_run_resolves_both_repository_and_immutable_head(tmp_path):
    pr = {
        "state": "open",
        "head": {"repo": {"full_name": "contributor/fork"}, "sha": "b" * 40},
    }
    process, output = _resolve(tmp_path, "42", pr)
    assert process.returncode == 0, process.stderr
    assert output == f"repository=contributor/fork\nref={'b' * 40}\n"


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
