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

"""The platform-config job bootstraps `yq` from github.com before it can read YAML.

That download is the first thing a lane does, so its failure mode is expensive:
on run 42465366556, 11 of the 14 lanes died on it with wget's exit 4 after ~160
seconds each, and the pipeline showed as failed for a reader that never moved.
The step has to survive a download that does not answer, and it must not hand a
truncated binary to the next lane.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/load-platform-config.yml"
STEP_NAME = "Load and validate platform configuration"


def _step_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == STEP_NAME:
                return step["run"]
    raise AssertionError(f"{STEP_NAME} not found in {WORKFLOW.name}")


def test_the_yq_download_is_retried():
    """A single wget is what turned a network blip into a failed pipeline.

    wget's own --tries does not cover this: the observed failure was DNS never
    resolving, which wget does not retry at all.
    """
    script = _step_script()
    fetch = script[script.index("YQ_URL=") :]
    assert re.search(r"for attempt in 1 2 3 4 5; do", fetch)
    assert "wget" in fetch
    assert re.search(r"sleep \$\(\(attempt", fetch)


def test_a_failed_attempt_cannot_leave_a_half_written_yq():
    """Writing straight to /usr/local/bin would strand a truncated binary.

    The next lane's `yq --version` guard would then read the leftover as a
    working install and fail later, with a parse error instead of a download
    error -- a much harder failure to attribute.
    """
    script = _step_script()
    assert "/tmp/yq.download" in script
    assert "install -m 0755 /tmp/yq.download /usr/local/bin/yq" in script
    # Nothing writes the final path directly.
    assert not re.search(r"wget[^\n]*-qO\s+/usr/local/bin/yq", script)
    # A failed attempt removes its partial file before retrying.
    assert re.search(r"rm -f /tmp/yq\.download", script)


def test_an_existing_yq_is_reused():
    """Runners are long-lived; re-downloading a fixed 5 MB asset per lane is the
    failure this guard removes when the binary is already there."""
    script = _step_script()
    assert "if ! yq --version >/dev/null 2>&1; then" in script


def test_a_missing_yq_fails_with_a_readable_error():
    """The step cannot proceed without it, so it must say so rather than let the
    next `yq` call produce bash's "command not found" under `set -e`."""
    script = _step_script()
    assert "::error::yq is unavailable" in script
    assert re.search(r"::error::yq is unavailable[^\n]*\n\s*exit 1", script)
