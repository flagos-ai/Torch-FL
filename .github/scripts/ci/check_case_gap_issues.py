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

"""Require every registered test gap to have an open tracking issue.

The source contract checks that a skip belongs to the registry without
importing torch. This check reads that registry and asks GitHub for the current
issue state. A static ``state`` field cannot notice when an issue is closed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCRIPT_ROOT = REPO_ROOT / ".github/scripts/ci"
DEFAULT_REGISTRY = REPO_ROOT / "tests/integration/case_gap_issues.py"
ISSUE_REFERENCE = re.compile(
    r"(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/issues/"
    r"(?P<number>[1-9][0-9]*)"
)
PROJECT_REPOSITORY = "flagos-ai/Torch-FL"


@dataclass(frozen=True)
class RegisteredGap:
    """Issue locations and owned references aggregated from registry rows."""

    number: str
    cases: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    repository: str = PROJECT_REPOSITORY
    upstream: str = ""


def _literal_string(node: ast.AST | None, field: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError(f"{field} must be a string")
    return node.value


def _string_tuple(node: ast.AST | None, field: str) -> tuple[str, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise ValueError(f"{field} must be a tuple of strings")
    values = tuple(_literal_string(value, field) for value in node.elts)
    if len(values) != len(set(values)) or any(not value for value in values):
        raise ValueError(f"{field} must contain unique non-empty strings")
    return values


def parse_registered_gaps(tree: ast.Module) -> tuple[RegisteredGap, ...]:
    """Read issue references and their ownership without importing test code."""
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_ISSUES"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("case gap registry must define _ISSUES exactly once")
    assignment = assignments[0]
    if not isinstance(assignment.value, (ast.Tuple, ast.List)):
        raise ValueError("case gap registry must define a tuple named _ISSUES")

    grouped = {}
    grouped_cases = {}
    grouped_routes = {}
    owned_cases = set()
    owned_routes = set()
    for row in assignment.value.elts:
        if not isinstance(row, ast.Call) or not (
            isinstance(row.func, ast.Name) and row.func.id == "CaseGapIssue"
        ):
            raise ValueError("every case gap must use CaseGapIssue(...)")
        if any(keyword.arg is None for keyword in row.keywords):
            raise ValueError("CaseGapIssue has unsupported fields")
        fields = {keyword.arg: keyword.value for keyword in row.keywords}
        if row.args or not fields.keys() <= {
            "number",
            "cases",
            "routes",
            "upstream",
        }:
            raise ValueError("CaseGapIssue has unsupported fields")
        number = _literal_string(fields.get("number"), "issue number")
        if re.fullmatch(r"#[1-9][0-9]*", number) is None:
            raise ValueError("case gap number must look like #710")
        cases = _string_tuple(fields.get("cases") or ast.Tuple(elts=[]), "cases")
        routes = _string_tuple(fields.get("routes") or ast.Tuple(elts=[]), "routes")
        if not cases and not routes:
            raise ValueError(f"{number} must own cases or routes")
        if not cases and any(value.startswith("#") for value in routes):
            raise ValueError(f"{number} must put case IDs in cases")
        upstream = _literal_string(
            fields.get("upstream") or ast.Constant(value=""), "upstream"
        )
        if upstream and ISSUE_REFERENCE.fullmatch(upstream) is None:
            raise ValueError("upstream must be owner/repository/issues/number")
        previous_upstream = grouped.setdefault(number, upstream)
        if previous_upstream != upstream:
            raise ValueError(f"{number} must use one upstream issue")
        grouped_cases.setdefault(number, []).extend(cases)
        grouped_routes.setdefault(number, []).extend(routes)
        if (owned_cases & set(cases)) or (owned_routes & set(routes)):
            raise ValueError("a case or route must have exactly one issue owner")
        owned_cases.update(cases)
        owned_routes.update(routes)
    result = tuple(
        RegisteredGap(
            number=number,
            cases=tuple(grouped_cases[number]),
            routes=tuple(grouped_routes[number]),
            upstream=upstream,
        )
        for number, upstream in grouped.items()
    )
    return result


def registered_gaps(path: Path = DEFAULT_REGISTRY) -> tuple[RegisteredGap, ...]:
    """Parse a registry without importing the test helper."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return parse_registered_gaps(tree)


def _api_issue(repository: str, number: str) -> str:
    """Read an issue's state through the GitHub CLI."""
    result = subprocess.run(
        ["gh", "api", f"repos/{repository}/issues/{number}"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("GitHub issue query returned an invalid payload")
    if "pull_request" in payload:
        raise ValueError("a case-gap reference must name an issue, not a pull request")
    return str(payload.get("state", "")).lower()


def _verify_issue(reference: str, lookup, label: str) -> str:
    match = ISSUE_REFERENCE.fullmatch(reference)
    if match is None:
        return f"could not verify {label} {reference}: issue reference is malformed"
    try:
        state = lookup(match.group("repository"), match.group("number"))
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
        return f"could not verify {label} {reference}: {exc}"
    if str(state).lower() != "open":
        return f"{label} {reference} is {state or 'missing'}"
    return ""


def validate_gaps(gaps, lookup=None) -> list[str]:
    """Return all reasons that the issue-backed skip gate is not satisfied."""
    lookup = lookup or _api_issue
    errors = []
    for gap in gaps:
        issue_reference = f"{PROJECT_REPOSITORY}/issues/{gap.number.removeprefix('#')}"
        error = _verify_issue(issue_reference, lookup, "issue")
        if error:
            errors.append(
                f"{gap.number}: {error}; remove its source skips and registry "
                "row in the same change"
            )
        if gap.upstream:
            error = _verify_issue(gap.upstream, lookup, "upstream")
            if error:
                errors.append(
                    f"{gap.number}: {error}; the tracking issue must be open too"
                )
    return errors


def main() -> int:
    """Verify all registry entries and report each failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-root", type=Path, default=DEFAULT_SCRIPT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    trusted_script = args.script_root / "check_case_gap_issues.py"
    if Path(sys.argv[0]).resolve() != trusted_script.resolve():
        print(
            "::error::the issue query must run from the trusted base script: "
            f"{trusted_script}"
        )
        return 1
    try:
        gaps = registered_gaps(args.registry)
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"::error::invalid case-gap registry: {exc}")
        return 1
    if not gaps:
        print("WARNING: no case gaps are registered")
        return 0
    errors = validate_gaps(gaps)
    failed_numbers = {error.split(":", 1)[0] for error in errors}
    for gap in gaps:
        print(f"{gap.number}: {'failed' if gap.number in failed_numbers else 'open'}")
    for error in errors:
        print(f"::error::{error}")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
