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

"""Capability gaps temporarily made explicit by the shared-case migration.

An expected gap receives a stable ID from a migration group. The same ID may be
used by every chip affected by one defect, so a fix closes one issue for all of
them. Every issue still blocks the next migration group until its skipped cases
pass. Closing it removes the skip and its row in the same change.

This registry is the offline source of truth for the source contract. The
separate case-gap workflow verifies that its issues are still open on GitHub.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest


MIGRATED_CASE_FILES = frozenset(
    {
        "tests/integration/ops/test_all_dispatch.py",
    }
)


@dataclasses.dataclass(frozen=True)
class CaseGapIssue:
    """One tracked capability gap and the cases it currently owns."""

    number: str
    cases: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    upstream: str = ""

    def __post_init__(self):
        if re.fullmatch(r"#[1-9][0-9]*", self.number) is None:
            raise ValueError(f"case gap needs a numbered issue, got {self.number!r}")
        if not self.cases and not self.routes:
            raise ValueError(f"case gap {self.number} must own cases or routes")
        if any(value.startswith("#") for value in self.routes):
            raise ValueError("case IDs must be listed in cases, not routes")
        if not self.upstream:
            raise ValueError(f"case gap {self.number} needs an upstream issue")

        if (
            re.fullmatch(
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[1-9][0-9]*",
                self.upstream,
            )
            is None
        ):
            raise ValueError("upstream must be owner/repository/issues/number")


_ISSUES = (
    CaseGapIssue(
        number="#710",
        cases=(
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_empty_dimensions",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_empty_tensor",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_empty_dim_reduction",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_empty_results_keep_their_shape",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_empty_full_reduction",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_dim_empty_reduction_reaches_its_route",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllDispatch::test_dispatch_log_flaggems_python",
        ),
        upstream="flagos-ai/FlagGems/issues/8411",
    ),
    CaseGapIssue(
        number="#710",
        routes=("all", "all:flaggems_python"),
        upstream="flagos-ai/FlagGems/issues/8411",
    ),
)


def open_case_gap(key: str) -> CaseGapIssue:
    """Return the open issue that owns a case ID or route key."""
    matches = {}
    for entry in _ISSUES:
        issue_key = key.startswith("#")
        owned = entry.number == key if issue_key else key in entry.routes
        if owned:
            matches.setdefault(entry.number, entry)
    if len(matches) != 1:
        raise LookupError(f"{key} is not owned by one open registered case gap")
    return next(iter(matches.values()))


def tracked_skip(condition: bool, *, issue: str) -> Any:
    """Attach a skip to a registered gap without hiding why it remains."""
    return pytest.mark.skipif(condition, reason=_skip_reason(issue))


def _skip_reason(issue: str) -> str:
    """Name both issues, but leave their live state to the trusted CI query."""
    entry = open_case_gap(issue)
    number = entry.number.removeprefix("#")
    return (
        f"{entry.number} https://github.com/flagos-ai/Torch-FL/issues/{number}; "
        f"upstream https://github.com/{entry.upstream}"
    )
