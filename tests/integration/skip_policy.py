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

"""Machine-readable categories for skips in the shared integration suites.

A skip is either a platform/lane applicability gate or an issue-backed
capability gap. Legacy skip forms are frozen by
``skip_policy_baseline.json`` and may only shrink. Keeping the category in a
marker means the report merger never has to infer policy from a skip reason.
"""

from __future__ import annotations

import re
from typing import Any

import pytest


TRACKED_GAP_MARKER = "flagos_tracked_gap"
PLATFORM_GATE_MARKER = "flagos_platform_gate"
TRACKED_GAP_CATEGORY = "tracked_gap"
PLATFORM_GATE_CATEGORY = "platform_gate"
SKIP_OUTCOMES = frozenset({"skipped", "xfailed", "xpassed"})
POLICY_ROOTS = (
    "tests/integration/flaggems/",
    "tests/integration/ops/",
)
PLATFORM_GATES = frozenset(
    {
        "backend_marker",
        "build_record",
        "device_count",
        "platform_support",
    }
)


def register_skip_policy_markers(config: pytest.Config) -> None:
    """Register the markers used to classify skips at collection time."""
    config.addinivalue_line(
        "markers",
        f"{TRACKED_GAP_MARKER}(issue): a skipped capability gap owned by an "
        "open Torch-FL issue",
    )
    config.addinivalue_line(
        "markers",
        f"{PLATFORM_GATE_MARKER}(gate, detail): a case is inapplicable to this "
        "platform/lane/build; gates are backend_marker, build_record, "
        "device_count, and platform_support",
    )


def tracked_gap_marker(issue: str) -> Any:
    """Return the marker paired with an issue-backed ``tracked_skip``."""
    if re.fullmatch(r"#[1-9][0-9]*", issue) is None:
        raise ValueError(f"tracked gap needs a numbered issue, got {issue!r}")
    return pytest.mark.flagos_tracked_gap(issue=issue)


def platform_gate_marker(*, gate: str, detail: str) -> Any:
    """Return a validated platform/lane applicability marker."""
    if gate not in PLATFORM_GATES:
        raise ValueError(f"unknown platform gate {gate!r}")
    if not detail or "\n" in detail or "\r" in detail:
        raise ValueError("platform gate detail must be a non-empty single line")
    return pytest.mark.flagos_platform_gate(gate=gate, detail=detail)


def add_platform_skip(
    item: pytest.Item,
    *,
    gate: str,
    detail: str,
    reason: str,
) -> None:
    """Mark one collected case inapplicable through a centralized gate.

    Platform applicability is determined during collection and takes precedence
    over a capability gap written on the test. Keep exactly one policy marker so
    the report can explain the actual reason the case could not run.
    """
    if not reason or "\n" in reason or "\r" in reason:
        raise ValueError("platform skip reason must be a non-empty single line")
    policy_markers = {TRACKED_GAP_MARKER, PLATFORM_GATE_MARKER}
    item.own_markers[:] = [
        marker
        for marker in item.own_markers
        if getattr(marker, "name", None) not in policy_markers
    ]
    item.add_marker(platform_gate_marker(gate=gate, detail=detail))
    item.add_marker(pytest.mark.skip(reason=reason))


def is_skip_policy_nodeid(nodeid: str) -> bool:
    """Whether a node belongs to the integration scope governed by this policy."""
    path = nodeid.partition("::")[0].replace("\\", "/").removeprefix("./")
    return path.startswith(POLICY_ROOTS)
