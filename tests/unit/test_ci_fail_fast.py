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

"""A platform lane that is already red must not keep the other chips busy.

The platform matrix runs with ``fail-fast: true``: once one platform's build or
tests fail, the PR is red whatever the rest do, and each remaining lane would
hold an accelerator for minutes to confirm it. GitHub cancels the siblings.

Two things have to stay true for that to be safe:

* ``CI complete`` still fails on a fast-failed matrix. A cancelled sibling makes
  the matrix result ``failure``, which the gate already rejects; the test below
  pins that it keeps doing so, so fail-fast cannot turn a red PR green.
* The gate names the lane that failed. With fail-fast the first failure is the
  only one reported, and "platform-pipeline: failure" says nothing; the gate
  reads the run's jobs back and reports their conclusions.

Pure text/json: no network.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _text() -> str:
    return CI.read_text(encoding="utf-8")


def _gate_job() -> str:
    """``ci-complete`` and everything after it.

    Located from the gate's own name so that a test asserting the gate does not
    depend on the job above it having a particular shape.
    """
    text = _text()
    return text[text.index("  ci-complete:") :]


def _gate_script() -> str:
    block = _gate_job()
    return block[block.index("python3 - <<'PY'") : block.index("\n          PY\n")]


def test_the_platform_matrix_is_fail_fast():
    text = _text()
    start = text.index("  platform-pipeline:")
    end = text.find("\n  ci-complete:", start)
    block = text[start : end if end != -1 else len(text)]
    assert re.search(r"^\s+fail-fast: true$", block, re.M), (
        "platform-pipeline must cancel its siblings on the first failure"
    )


def test_a_fast_failed_matrix_still_fails_ci_complete():
    script = _gate_script()
    assert 'platform_result = jobs["platform-pipeline"]["result"]' in script
    assert 'if selected and platform_result != "success":' in script, (
        "a cancelled sibling makes this 'failure'; the gate must reject anything "
        "that is not success"
    )
    assert "sys.exit(1)" in script


def test_the_gate_names_the_failed_lane():
    script = _gate_script()
    assert "GITHUB_RUN_ID" in script and "/jobs" in script, (
        "the gate must list the run's jobs: with fail-fast, the matrix result "
        "alone does not say which platform broke"
    )
    assert '("failure", "timed_out")' in script
    assert '"cancelled"' in script, (
        "a cancelled sibling must be reported as fail-fast, not as a finding"
    )
    assert 'name.startswith("Platform pipeline (")' in script


def test_the_gate_can_call_the_api():
    gate = _gate_job()
    assert "GH_TOKEN: ${{ github.token }}" in gate, (
        "gh api needs a token; the job-level env must carry it"
    )
