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
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CaseGapIssue:
    """One tracked capability gap and the cases it currently owns."""

    number: str
    state: str
    cases: tuple[str, ...]
    routes: tuple[str, ...] = ()


_ISSUES = (
    CaseGapIssue(
        "#710",
        "open",
        (
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_empty_dimensions",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllCorrectness::test_all_empty_tensor",
            "tests/integration/ops/test_all_dispatch.py"
            "::TestAllDispatch::test_dispatch_log_flaggems_python",
        ),
        ("all", "all:flaggems_python"),
    ),
)


def case_gap_reason(key: str) -> str:
    """Return a skip reason whose route or case is owned by an open issue."""
    entry = open_case_gap(key)
    if key.startswith("#"):
        route = "<case>"
    else:
        route = key
    return f"expected case gap {entry.number} ({route})"


def open_case_gap(key: str) -> CaseGapIssue:
    """Return the open issue that owns a case ID or route key."""
    matches = [entry for entry in _ISSUES if key == entry.number or key in entry.routes]
    if len(matches) != 1 or matches[0].state != "open":
        raise LookupError(f"{key} is not owned by one open registered case gap")
    return matches[0]
