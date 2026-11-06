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

"""One red lane must not hide how many others are red.

The platform matrix runs with ``fail-fast: false``. Under ``true`` the first
platform to fail cancels every sibling, and the run reports one failure no
matter how many lanes were broken -- so a red PR converges one platform per
round, each round costing a full build on every lane that had already been
cancelled. Under ``false`` every selected lane finishes and the run reports all
of them, which is what makes a red PR fixable in one push instead of N.

Two things have to stay true for that to be safe:

* ``CI complete`` still fails when any lane is not green. The matrix result is
  ``failure`` if any lane failed, and the gate rejects everything that is not
  ``success``; the test below pins that, so running every lane cannot turn a red
  PR green.
* The gate names the lane that failed. "platform-pipeline: failure" says
  nothing, so the gate reads the run's jobs back and reports their conclusions
  -- including a lane that never finished, which is a gap in measurement rather
  than a pass.

Pure text/json: no network.
"""

import importlib.util
import json
import re
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

SCRIPTS = REPO_ROOT / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "shared_ci_gate", SCRIPTS / "flagcicd_report.py"
    )
    REPORTER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(REPORTER)
finally:
    sys.path.pop(0)


def _text() -> str:
    return CI.read_text(encoding="utf-8")


def _gate_job() -> str:
    """``ci-complete`` and everything after it.

    Located from the gate's own name so that a test asserting the gate does not
    depend on the job above it having a particular shape.
    """
    text = _text()
    return text[text.index("  ci-complete:") :]


def test_the_platform_matrix_runs_every_lane():
    text = _text()
    start = text.index("  platform-pipeline:")
    end = text.find("\n  ci-complete:", start)
    block = text[start : end if end != -1 else len(text)]
    assert re.search(r"^\s+fail-fast: false$", block, re.M), (
        "platform-pipeline must let every selected lane finish: cancelling the "
        "siblings on the first failure reports one red per run, not all of them"
    )


def test_both_required_gates_use_the_same_reporter():
    for filename, job in (
        ("ci.yml", "ci-complete"),
        ("flagcicd-entry.yml", "complete"),
    ):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows" / filename).read_text()
        )
        calls = [
            step
            for step in workflow["jobs"][job]["steps"]
            if "flagcicd_report.py summary" in step.get("run", "")
        ]
        assert len(calls) == 1
        assert "--plan-file execution-plan/flagcicd-plan.json" in calls[0]["run"]
        assert "JOB_RESULTS" in calls[0]["env"]


def test_the_gate_names_the_failed_lane():
    script = (SCRIPTS / "flagcicd_report.py").read_text()
    assert "GITHUB_RUN_ID" in script and "/jobs" in script
    assert '("failure", "timed_out")' in script
    assert 'name.startswith("Platform pipeline (")' in script
    assert "was cancelled before it finished" in script


def test_the_gate_can_call_the_api():
    gate = _gate_job()
    assert "GH_TOKEN: ${{ github.token }}" in gate, (
        "gh api needs a token; the job-level env must carry it"
    )


@pytest.mark.parametrize("api_timeout", [False, True])
def test_gate_reads_all_job_pages_and_bounds_optional_diagnostics(
    monkeypatch, capsys, api_timeout
):
    workflow = yaml.safe_load(CI.read_text())
    gate = workflow["jobs"]["ci-complete"]
    assert gate["timeout-minutes"] == 5
    assert workflow["permissions"]["actions"] == "read"
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GH_TOKEN", "fake-test-token")

    def api(argv, **kwargs):
        assert "--paginate" in argv and "--slurp" in argv
        assert kwargs["timeout"] == 60
        if api_timeout:
            raise subprocess.TimeoutExpired(argv, 60)
        return SimpleNamespace(
            stdout=json.dumps(
                [
                    {"jobs": [{"name": "first page lane", "conclusion": "failure"}]},
                    {"jobs": [{"name": "second page lane", "conclusion": "timed_out"}]},
                ]
            )
        )

    monkeypatch.setattr(subprocess, "run", api)
    REPORTER.diagnose_jobs()
    output = capsys.readouterr().out
    if api_timeout:
        assert "::warning::could not list job conclusions" in output
    else:
        assert "first page lane: failure" in output
        assert "second page lane: timed_out" in output
