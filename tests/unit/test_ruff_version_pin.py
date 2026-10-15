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

"""The ruff version is stated in three places and every one of them must agree.

CI installs one version (``.github/workflows/lint.yml``), the ``dev`` extra
installs the same one for local work, and ``[tool.ruff] required-version`` is
what makes a *different* local version fail loudly instead of quietly reporting
a tree that only looks dirty. That last one is the whole point: without it an
unpinned local ruff disagrees with the gate about what needs formatting (0.17.0
formats code fences inside Markdown, so it calls 18 files here unformatted on a
tree the gate reports clean, and ``ruff format .`` under it rewrites them).

Reading the pins out of the files rather than repeating them is what keeps a
future bump to one edit: change lint.yml's ``pip install`` line and this test
names the two files that are now out of step.

Text only, no ruff needed: the platform pipelines run this suite in
environments that do not have the linter installed.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LINT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "lint.yml"


def _ruff_pin_from_workflow() -> str:
    """The version the lint job's `pip install` line asks for."""
    text = LINT_WORKFLOW.read_text(encoding="utf-8")
    matches = re.findall(r"^\s*(?:run:\s*)?pip install ruff==(\S+)\s*$", text, re.M)
    assert len(matches) == 1, (
        f"expected exactly one ruff install in the lint job: {matches}"
    )
    return matches[0]


def _dev_extra_pin() -> str:
    """The version `pip install -e .[dev]` resolves."""
    text = PYPROJECT.read_text(encoding="utf-8")
    dev_extra = re.search(r"^dev = \[(.*?)\]", text, re.M | re.S)
    assert dev_extra, "pyproject.toml has no dev extra"
    matches = re.findall(r'"ruff==([^"]+)"', dev_extra.group(1))
    assert len(matches) == 1, (
        f"expected exactly one ruff pin in the dev extra: {matches}"
    )
    return matches[0]


def _required_version() -> str:
    """What `[tool.ruff]` lets the tool run as."""
    text = PYPROJECT.read_text(encoding="utf-8")
    matches = re.findall(r'^required-version\s*=\s*"([^"]*)"', text, re.M)
    assert len(matches) == 1, (
        "[tool.ruff] must set required-version exactly once; without it a local "
        f"ruff of any version runs and can disagree with CI: {matches}"
    )
    return matches[0]


def test_the_three_ruff_pins_agree():
    """CI's install, the dev extra and the enforced version are one number."""
    pins = {
        ".github/workflows/lint.yml": _ruff_pin_from_workflow(),
        "pyproject.toml [dev]": _dev_extra_pin(),
        "pyproject.toml required-version": _required_version(),
    }
    assert len(set(pins.values())) == 1, pins


def test_required_version_is_an_exact_pin():
    """A range here would accept the versions this guard exists to reject.

    ruff takes `>=0.15.12` as readily as it takes `0.15.12`, and the caret form
    would let 0.17 in -- the release whose formatting differs from CI's.
    """
    pin = _required_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", pin), pin
