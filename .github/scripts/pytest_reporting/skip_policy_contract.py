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

"""Stable skip-policy data shared by pytest collection and report merging."""

from __future__ import annotations

import ast
import re
from pathlib import Path


TRACKED_GAP_MARKER = "flagos_tracked_gap"
PLATFORM_GATE_MARKER = "flagos_platform_gate"
TRACKED_GAP_CATEGORY = "tracked_gap"
PLATFORM_GATE_CATEGORY = "platform_gate"
PLATFORM_GATES = frozenset(
    {
        "backend_marker",
        "build_record",
        "device_count",
        "platform_support",
    }
)
SKIP_OUTCOMES = frozenset({"skipped", "xfailed", "xpassed"})
POLICY_ROOTS = (
    "tests/integration/flaggems/",
    "tests/integration/ops/",
)
_ISSUE = re.compile(r"#[1-9][0-9]*\Z")


def nodeid_path(nodeid: str) -> str:
    """Return the normalized source path from a pytest node id."""
    return nodeid.partition("::")[0].replace("\\", "/").removeprefix("./")


def is_governed_nodeid(nodeid: str) -> bool:
    """Whether a collected node belongs to a cross-chip policy root."""
    return nodeid_path(nodeid).startswith(POLICY_ROOTS)


def read_migrated_case_files(path: Path) -> frozenset[str]:
    """Read the migration scope without importing submitted test code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "MIGRATED_CASE_FILES"
    ]
    if len(declarations) != 1:
        raise ValueError("MIGRATED_CASE_FILES must be declared exactly once")
    declaration = declarations[0].value
    if (
        not isinstance(declaration, ast.Call)
        or not isinstance(declaration.func, ast.Name)
        or declaration.func.id != "frozenset"
        or len(declaration.args) != 1
        or declaration.keywords
        or not isinstance(declaration.args[0], (ast.Set, ast.Tuple, ast.List))
    ):
        raise ValueError("MIGRATED_CASE_FILES must be a literal frozenset")
    sources = []
    for value in declaration.args[0].elts:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ValueError("MIGRATED_CASE_FILES entries must be strings")
        if not is_governed_nodeid(value.value) or not value.value.endswith(".py"):
            raise ValueError(f"invalid migrated source {value.value!r}")
        sources.append(value.value)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("MIGRATED_CASE_FILES must list unique integration files")
    return frozenset(sources)


def describe_policy_markers(markers) -> dict | None:
    """Normalize the policy markers attached to one collected pytest item.

    ``markers`` is an iterable of ``(name, args, kwargs)`` tuples. An invalid
    or conflicting declaration is retained as an explicit invalid record so
    the report merger can reject it instead of silently treating the test as
    unclassified.
    """
    relevant = [
        (name, args, kwargs)
        for name, args, kwargs in markers
        if name in (TRACKED_GAP_MARKER, PLATFORM_GATE_MARKER)
    ]
    if not relevant:
        return None
    if len(relevant) != 1:
        return {
            "category": "invalid",
            "reason": "a test must have exactly one skip-policy marker",
        }

    name, args, kwargs = relevant[0]
    if name == TRACKED_GAP_MARKER:
        issue = kwargs.get("issue")
        if args or set(kwargs) != {"issue"}:
            return {
                "category": "invalid",
                "reason": "flagos_tracked_gap accepts only issue=",
            }
        if not isinstance(issue, str) or _ISSUE.fullmatch(issue) is None:
            return {
                "category": "invalid",
                "reason": "flagos_tracked_gap needs a numbered issue",
            }
        return {"category": TRACKED_GAP_CATEGORY, "issue": issue}

    gate = kwargs.get("gate")
    detail = kwargs.get("detail")
    if args or set(kwargs) != {"gate", "detail"}:
        return {
            "category": "invalid",
            "reason": "flagos_platform_gate accepts only gate= and detail=",
        }
    if gate not in PLATFORM_GATES:
        return {
            "category": "invalid",
            "reason": f"unknown platform gate {gate!r}",
        }
    if not isinstance(detail, str) or not detail or "\n" in detail or "\r" in detail:
        return {
            "category": "invalid",
            "reason": "platform gate detail must be a non-empty single line",
        }
    return {
        "category": PLATFORM_GATE_CATEGORY,
        "gate": gate,
        "detail": detail,
    }


def validate_policy_record(record) -> str:
    """Return why a reported skip-policy record is invalid, or an empty string."""
    if not isinstance(record, dict):
        return "skip policy must be an object"
    category = record.get("category")
    if category == TRACKED_GAP_CATEGORY:
        issue = record.get("issue")
        if set(record) != {"category", "issue"}:
            return "tracked gap skip policy has unexpected fields"
        if not isinstance(issue, str) or _ISSUE.fullmatch(issue) is None:
            return "tracked gap needs a numbered issue"
        return ""
    if category == PLATFORM_GATE_CATEGORY:
        gate, detail = record.get("gate"), record.get("detail")
        if set(record) != {"category", "gate", "detail"}:
            return "platform gate skip policy has unexpected fields"
        if gate not in PLATFORM_GATES:
            return f"unknown platform gate {gate!r}"
        if (
            not isinstance(detail, str)
            or not detail
            or "\n" in detail
            or "\r" in detail
        ):
            return "platform gate detail must be a non-empty single line"
        return ""
    if category == "invalid":
        return str(record.get("reason") or "invalid skip-policy marker")
    return f"unknown skip-policy category {category!r}"
