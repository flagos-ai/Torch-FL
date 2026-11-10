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

"""Require every registered test gap to have open tracking issues.

The registry is parsed without importing submitted source. A temporary skip may
be added only with a ``tracked_skip`` decorator in a migrated cross-chip
integration group, and every registered case must have that exact decorator.
Every Torch-FL and upstream issue is queried live. A static ``state`` field
cannot notice when an issue is closed.
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
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCRIPT_ROOT = REPO_ROOT / ".github/scripts/ci"
DEFAULT_REGISTRY = REPO_ROOT / "tests/integration/case_gap_issues.py"
DEFAULT_BASELINE = REPO_ROOT / "tests/integration/skip_policy_baseline.json"
DEFAULT_SOURCE_CONTRACT = REPO_ROOT / "tests/unit/test_integration_source_contract.py"
POLICY_ROOTS = (
    "tests/integration/flaggems/",
    "tests/integration/ops/",
)
LEGACY_SKIP_CALLS = frozenset(
    {
        "pytest.skip",
        "pytest.importorskip",
        "pytest.xfail",
        "pytest.mark.skip",
        "pytest.mark.skipif",
        "pytest.mark.xfail",
        "pytest.mark.skip_class",
        "mark.skip",
        "mark.skipif",
        "mark.xfail",
        "skip",
        "skipif",
        "skipIf",
        "skipUnless",
        "skipTest",
        "expectedFailure",
        "importorskip",
        "xfail",
        "unittest.skip",
        "unittest.skipIf",
        "unittest.skipUnless",
        "unittest.expectedFailure",
    }
)
DIRECT_POLICY_MARKERS = frozenset(
    {
        "flagos_platform_gate",
        "flagos_tracked_gap",
    }
)
ISSUE_REFERENCE = re.compile(
    r"(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/issues/"
    r"(?P<number>[1-9][0-9]*)"
)
PROJECT_REPOSITORY = "flagos-ai/Torch-FL"
UPSTREAM_REPOSITORY = "flagos-ai/FlagGems"


@dataclass(frozen=True)
class CaseGapContract:
    """The submitted source contract and the integration files it governs."""

    contract_source: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class SkipPolicyBaseline:
    """The exact legacy skip inventory allowed by the source ratchet."""

    roots: tuple[str, ...]
    files: dict[str, dict[str, int]]
    kinds: dict[str, int]


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


def parse_registered_gaps(
    tree: ast.Module, *, require_upstream: bool = False
) -> tuple[RegisteredGap, ...]:
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
        if any(value.startswith("#") for value in routes):
            raise ValueError(f"{number} must put case IDs in cases")
        upstream = _literal_string(
            fields.get("upstream") or ast.Constant(value=""), "upstream"
        )
        if not upstream:
            if require_upstream:
                raise ValueError(f"{number} must have an upstream issue")
        else:
            match = ISSUE_REFERENCE.fullmatch(upstream)
            if match is None:
                raise ValueError("upstream must be owner/repository/issues/number")
            if match.group("repository") != UPSTREAM_REPOSITORY and match.group(
                "number"
            ) == number.removeprefix("#"):
                raise ValueError(
                    "upstream must not be the same issue as its Torch-FL gap"
                )
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
    return parse_registered_gaps(tree, require_upstream=True)


def case_gap_contract(
    registry_source: str,
    contract_source: str,
) -> CaseGapContract:
    """Read the migrated source scope as data without importing it."""
    try:
        tree = ast.parse(registry_source, filename="case_gap_issues.py")
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
        source_node = declarations[0].value
        if (
            not isinstance(source_node, ast.Call)
            or not isinstance(source_node.func, ast.Name)
            or source_node.func.id != "frozenset"
            or len(source_node.args) != 1
            or source_node.keywords
            or not isinstance(source_node.args[0], (ast.Set, ast.Tuple, ast.List))
        ):
            raise ValueError("MIGRATED_CASE_FILES must be a literal frozenset")
        sources = tuple(
            _literal_string(value, "migrated source")
            for value in source_node.args[0].elts
        )
    except SyntaxError as exc:
        raise ValueError(f"invalid declared source scope: {exc}") from exc
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("MIGRATED_CASE_FILES must list unique integration files")
    invalid = [
        source
        for source in sources
        if not any(source.startswith(root) for root in POLICY_ROOTS)
        or not source.endswith(".py")
        or source == "tests/integration/case_gap_issues.py"
    ]
    if invalid:
        raise ValueError(
            f"migrated source must be under a governed integration root: {invalid}"
        )
    return CaseGapContract(contract_source, sources)


def _callee_name(function: ast.AST) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _direct_callee_name(function: ast.AST) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        return f"{function.value.id}.{function.attr}"
    return ""


def _dotted_name(node: ast.AST | None) -> str:
    """Return a dotted call target without evaluating the source."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _legacy_skip_kind(node: ast.AST) -> str:
    """Classify a source AST call that can produce a skip-like outcome."""
    if not isinstance(node, ast.Call):
        return ""
    name = _dotted_name(node.func)
    if name in LEGACY_SKIP_CALLS:
        return name
    if name.endswith(".skipTest"):
        return "skipTest"
    return ""


def skip_call_counts(source: str, *, filename: str) -> dict[str, int]:
    """Count legacy skip forms in one source file without importing it."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ValueError(f"{filename}:{exc.lineno}: invalid Python: {exc.msg}") from exc
    counts = {}
    for node in ast.walk(tree):
        kind = _legacy_skip_kind(node)
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def direct_policy_marker_violations(source: str, *, filename: str) -> list[str]:
    """Reject source that fabricates the markers reserved for shared helpers."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ValueError(f"{filename}:{exc.lineno}: invalid Python: {exc.msg}") from exc
    return [
        f"{filename}:{node.lineno}: skip-policy markers must come from the "
        "shared helper"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in DIRECT_POLICY_MARKERS
    ]


def load_skip_policy_baseline(
    path: Path = DEFAULT_BASELINE,
    *,
    roots: tuple[str, ...] = POLICY_ROOTS,
) -> SkipPolicyBaseline:
    """Read and validate the exact legacy skip ratchet baseline."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read skip policy baseline {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("skip policy baseline must be an object")
    if payload.get("version") != 1:
        raise ValueError("skip policy baseline version must be 1")
    declared_roots = payload.get("roots")
    if declared_roots != list(roots):
        raise ValueError(f"skip policy baseline roots must be {list(roots)}")
    files = payload.get("files")
    kinds = payload.get("kinds")
    if not isinstance(files, dict) or not isinstance(kinds, dict):
        raise ValueError("skip policy baseline needs files and kinds objects")

    normalized_files = {}
    aggregate = {}
    for relative, row in files.items():
        if not isinstance(relative, str):
            raise ValueError("skip policy baseline file names must be strings")
        if (
            relative.startswith("/")
            or "\\" in relative
            or ".." in relative.split("/")
            or not relative.endswith(".py")
            or not any(relative.startswith(root) for root in roots)
        ):
            raise ValueError(f"invalid skip policy baseline file: {relative}")
        if not isinstance(row, dict):
            raise ValueError(
                f"skip policy baseline entry must be an object: {relative}"
            )
        normalized_row = {}
        for kind, count in row.items():
            if kind not in LEGACY_SKIP_CALLS and kind != "skipTest":
                raise ValueError(f"unknown legacy skip kind: {kind}")
            if type(count) is not int or count <= 0:
                raise ValueError(
                    f"legacy skip count must be a positive integer: {relative}:{kind}"
                )
            normalized_row[kind] = count
            aggregate[kind] = aggregate.get(kind, 0) + count
        if normalized_row:
            normalized_files[relative] = normalized_row

    if kinds != aggregate:
        raise ValueError("skip policy baseline kinds do not match its file rows")
    return SkipPolicyBaseline(
        roots=tuple(roots),
        files=dict(sorted(normalized_files.items())),
        kinds=dict(sorted(aggregate.items())),
    )


def build_skip_policy_baseline(
    files: dict[str, dict[str, int]],
    *,
    roots: tuple[str, ...] = POLICY_ROOTS,
) -> dict:
    """Build the canonical JSON object for a generated baseline."""
    normalized_files = {}
    kinds = {}
    for relative in sorted(files):
        row = {kind: count for kind, count in sorted(files[relative].items()) if count}
        if row:
            normalized_files[relative] = row
            for kind, count in row.items():
                kinds[kind] = kinds.get(kind, 0) + count
    return {
        "version": 1,
        "roots": list(roots),
        "files": normalized_files,
        "kinds": dict(sorted(kinds.items())),
    }


def read_policy_sources(
    root: Path,
    *,
    roots: tuple[str, ...] = POLICY_ROOTS,
) -> dict[str, str]:
    """Read every submitted Python file under the governed integration roots."""
    root = root.resolve()
    sources = {}
    for relative_root in roots:
        directory = (root / relative_root).resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"policy root leaves its checkout: {relative_root}"
            ) from exc
        if not directory.is_dir():
            raise ValueError(f"policy root is missing: {relative_root}")
        for path in sorted(directory.rglob("*.py")):
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f"policy source leaves its checkout: {path}") from exc
            sources[relative] = resolved.read_text(encoding="utf-8")
    return sources


def skip_policy_violations(
    sources: dict[str, str],
    baseline: SkipPolicyBaseline,
    *,
    migrated: frozenset[str] = frozenset(),
) -> list[str]:
    """Check that legacy skips are frozen by file and call kind.

    A decrease is an error too: otherwise the baseline can hide a temporary
    skip long after it was removed. A newly added file never inherits an
    allowance from another file.
    """
    current = {}
    errors = []
    for relative in sorted(sources):
        try:
            counts = skip_call_counts(sources[relative], filename=relative)
            errors.extend(
                direct_policy_marker_violations(sources[relative], filename=relative)
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if counts:
            current[relative] = counts
        if relative in migrated and counts:
            errors.append(
                f"{relative}: migrated cross-chip files may not contain legacy "
                "skip calls; use tracked_skip or a platform gate"
            )

    for relative, counts in current.items():
        expected = baseline.files.get(relative)
        if expected is None:
            errors.append(
                f"{relative}: new legacy skip calls are forbidden; migrate them "
                "to tracked_skip or a platform gate"
            )
            continue
        for kind, count in sorted(counts.items()):
            allowed = expected.get(kind, 0)
            if count > allowed:
                errors.append(
                    f"{relative}:{kind}: {count} legacy skip calls, baseline "
                    f"allows {allowed}"
                )
            elif count < allowed:
                errors.append(
                    f"{relative}:{kind}: {count} legacy skip calls, baseline "
                    f"allows {allowed}; shrink skip_policy_baseline.json in the "
                    "same change"
                )

    for relative, expected in baseline.files.items():
        observed = current.get(relative, {})
        if not observed:
            errors.append(
                f"{relative}: baseline file has no legacy skip calls anymore; "
                "remove its baseline row"
            )
        for kind, count in sorted(expected.items()):
            if observed.get(kind, 0) == 0:
                errors.append(f"{relative}:{kind}: baseline entry is stale; remove it")
    return errors


def _is_tracked_skip(node: ast.AST) -> bool:
    """Recognize only the shared decorator, including its qualified spelling."""
    return isinstance(node, ast.Call) and _direct_callee_name(node.func) in {
        "tracked_skip",
        "case_gap_issues.tracked_skip",
    }


def _test_functions(
    tree: ast.Module,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        node
        for node in _walk_complete(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def _walk_complete(tree: ast.AST):
    """Yield every node, including those in arguments and decorators."""
    yield tree
    for child in ast.iter_child_nodes(tree):
        yield from _walk_complete(child)


def _qualname(
    function: ast.FunctionDef | ast.AsyncFunctionDef, parents: dict[int, ast.AST]
) -> str:
    names = [function.name]
    current = parents.get(id(function))
    while current is not None:
        if isinstance(current, ast.ClassDef):
            names.append(current.name)
        current = parents.get(id(current))
    return "::".join(reversed(names))


def _source_violations(
    relative: str,
    source: str,
    gaps: tuple[RegisteredGap, ...],
    migrated: frozenset[str],
    *,
    enforce_legacy: bool = True,
) -> tuple[list[str], dict[str, int]]:
    """Check one source before any test module is imported."""
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        return [f"{relative}:{exc.lineno}: invalid Python: {exc.msg}"], {}
    parents = {
        id(child): parent
        for parent in _walk_complete(tree)
        for child in ast.iter_child_nodes(parent)
    }
    functions = _test_functions(tree)
    function_ids = {id(function) for function in functions}
    errors = []
    imports = set()
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            if statement.module == "tests.integration.case_gap_issues":
                imports.update(
                    "tracked_skip"
                    for alias in statement.names
                    if alias.name == "tracked_skip" and alias.asname is None
                )
            elif statement.module == "tests.integration":
                imports.update(
                    "tracked_skip"
                    for alias in statement.names
                    if alias.name == "case_gap_issues" and alias.asname is None
                )
        elif isinstance(statement, ast.Import):
            imports.update(
                "tracked_skip"
                for alias in statement.names
                if alias.name == "tests.integration.case_gap_issues"
                and alias.asname is None
            )
        else:
            continue
        if any(
            alias.name == "tracked_skip" and alias.asname is not None
            for alias in statement.names
        ):
            errors.append(
                f"{relative}:{statement.lineno}: tracked_skip must not be aliased"
            )
    uses_tracked_skip = any(
        _callee_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
        == "tracked_skip"
        for function in functions
        for decorator in function.decorator_list
    ) or any(
        isinstance(node, ast.Call) and _is_tracked_skip(node)
        for node in _walk_complete(tree)
    )
    if (imports or uses_tracked_skip) and imports != {"tracked_skip"}:
        errors.append(
            f"{relative}: tracked skip must be imported from the shared registry"
        )
    if enforce_legacy:
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            else:
                continue
            if any(
                isinstance(node, ast.Name) and node.id == "pytestmark"
                for target in targets
                for node in _walk_complete(target)
            ):
                for node in _walk_complete(statement.value):
                    if isinstance(node, ast.Call) and _callee_name(node.func) in {
                        "skip",
                        "skipif",
                        "xfail",
                    }:
                        errors.append(
                            f"{relative}:{node.lineno}: module-wide skips are forbidden"
                        )
    for node in _walk_complete(tree):
        if isinstance(node, ast.Assign):
            assignments = node.targets
        elif isinstance(node, ast.AnnAssign):
            assignments = [node.target]
        else:
            assignments = []
        for target in assignments:
            for name in _walk_complete(target):
                if isinstance(name, ast.Name) and name.id == "tracked_skip":
                    errors.append(
                        f"{relative}:{node.lineno}: tracked_skip cannot be rebound"
                    )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "tracked_skip":
                errors.append(
                    f"{relative}:{node.lineno}: local tracked_skip definitions are forbidden"
                )
            if enforce_legacy and node.name.startswith("test_"):
                for decorator in _walk_complete(node):
                    if isinstance(decorator, ast.Call) and _callee_name(
                        decorator.func
                    ) in {"skip", "skipif", "xfail"}:
                        errors.append(
                            f"{relative}:{decorator.lineno}: test skips must use the tracked_skip decorator"
                        )
            for decorator in node.decorator_list:
                callee = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if _callee_name(callee) == "tracked_skip":
                    if id(node) not in function_ids or not node.name.startswith(
                        "test_"
                    ):
                        errors.append(
                            f"{relative}:{node.lineno}: tracked_skip may only decorate tests"
                        )
                elif (
                    enforce_legacy
                    and id(node) in function_ids
                    and _callee_name(callee)
                    in {
                        "skip",
                        "skipif",
                        "xfail",
                    }
                ):
                    errors.append(
                        f"{relative}:{node.lineno}: test skips must use the tracked_skip decorator"
                    )
        if not isinstance(node, ast.Call):
            continue
        callee = _direct_callee_name(node.func)
        if (
            enforce_legacy
            and callee
            in {
                "pytest.skip",
                "pytest.skipif",
                "pytest.xfail",
                "pytest.importorskip",
                "unittest.skip",
                "unittest.skipIf",
                "unittest.skipUnless",
            }
            and id(node)
            not in {
                id(decorator)
                for function in functions
                for decorator in function.decorator_list
            }
        ):
            errors.append(
                f"{relative}:{node.lineno}: test skips must use the tracked_skip decorator"
            )
        if enforce_legacy and callee in {"pytest.importorskip", "importorskip"}:
            errors.append(
                f"{relative}:{node.lineno}: test skipping must use the tracked_skip decorator"
            )
        if not _is_tracked_skip(node):
            continue
        owner = id(node) in {
            id(decorator)
            for function in functions
            for decorator in function.decorator_list
        }
        if not owner:
            errors.append(
                f"{relative}:{node.lineno}: tracked_skip must be called directly on a test"
            )
        fields = {keyword.arg: keyword.value for keyword in node.keywords}
        if (
            node.args
            or any(keyword.arg is None for keyword in node.keywords)
            or set(fields) != {"condition", "issue"}
        ):
            errors.append(
                f"{relative}:{node.lineno}: tracked_skip accepts only condition and issue"
            )
    skipped = {}
    for function in functions:
        decorators = [
            decorator
            for decorator in function.decorator_list
            if _is_tracked_skip(decorator)
        ]
        if not decorators:
            continue
        if len(decorators) != 1 or not isinstance(decorators[0], ast.Call):
            errors.append(
                f"{relative}:{function.lineno}: each test needs at most one tracked_skip"
            )
            continue
        decorator = decorators[0]
        fields = {keyword.arg: keyword.value for keyword in decorator.keywords}
        issue = fields.get("issue")
        if not isinstance(issue, ast.Constant) or not isinstance(issue.value, str):
            errors.append(
                f"{relative}:{decorator.lineno}: tracked_skip needs a literal issue"
            )
            continue
        gaps_for_issue = [gap for gap in gaps if gap.number == issue.value]
        if not gaps_for_issue:
            errors.append(
                f"{relative}:{decorator.lineno}: {issue.value} has no registry row"
            )
            continue
        base = f"{relative}::{_qualname(function, parents)}"
        matched = [
            case
            for gap in gaps_for_issue
            for case in gap.cases
            if case == base or case.startswith(f"{base}[")
        ]
        if not matched:
            errors.append(
                f"{relative}:{decorator.lineno}: {issue.value} has no cases owned by {base}"
            )
        for case in matched:
            if case in skipped:
                errors.append(f"{case} has more than one tracked skip")
            skipped[case] = decorator.lineno
    return errors, skipped


def source_violations(
    sources: dict[str, str],
    gaps: tuple[RegisteredGap, ...],
    migrated: frozenset[str],
) -> list[str]:
    """Check the declared migrated files and every registered case."""
    missing = sorted(
        {case.partition("::")[0] for gap in gaps for case in gap.cases} - sources.keys()
    )
    errors = [f"registered case source is missing: {source}" for source in missing]
    skipped = {}
    for relative in sorted(sources):
        found, owned = _source_violations(
            relative,
            sources[relative],
            gaps,
            migrated,
            enforce_legacy=relative in migrated,
        )
        errors.extend(found)
        skipped.update(owned)
    errors.extend(
        f"{gap.number}: registered case {case} has no tracked skip"
        for gap in gaps
        for case in gap.cases
        if case not in skipped
    )
    return errors


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


def validate_gaps(
    gaps: tuple[RegisteredGap, ...],
    lookup: Callable[[str, str], str] | None = None,
    *,
    sources: dict[str, str] | None = None,
    migrated_case_files: frozenset[str] | None = None,
) -> list[str]:
    """Return all reasons that the issue-backed skip gate is not satisfied."""
    lookup = lookup or _api_issue
    errors = (
        source_violations(sources, gaps, migrated_case_files or frozenset())
        if sources is not None
        else []
    )
    for gap in gaps:
        issue_reference = f"{PROJECT_REPOSITORY}/issues/{gap.number.removeprefix('#')}"
        error = _verify_issue(issue_reference, lookup, "issue")
        if error:
            errors.append(
                f"{gap.number}: {error}; remove its source skips and registry "
                "row in the same change"
            )
        error = _verify_issue(gap.upstream, lookup, "upstream")
        if error:
            errors.append(
                f"{gap.number}: {error}; remove its source skips and registry "
                "row in the same change"
            )
    return errors


def _submitted_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"submitted path leaves its checkout: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"submitted file is missing: {relative}")
    return path


def _read_contract_sources(root: Path, contract: CaseGapContract) -> dict[str, str]:
    return {
        relative: _submitted_file(root, relative).read_text(encoding="utf-8")
        for relative in contract.sources
    }


def main() -> int:
    """Verify all registry entries and report each failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-root", type=Path, default=DEFAULT_SCRIPT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source-contract", type=Path, default=DEFAULT_SOURCE_CONTRACT)
    args = parser.parse_args()
    trusted_script = args.script_root / "check_case_gap_issues.py"
    if Path(sys.argv[0]).resolve() != trusted_script.resolve():
        print(
            "::error::the issue query must run from the trusted base script: "
            f"{trusted_script}"
        )
        return 1
    try:
        repository_root = args.registry.parents[2].resolve()
        gaps = registered_gaps(args.registry)
        contract = case_gap_contract(
            args.registry.read_text(encoding="utf-8"),
            args.source_contract.read_text(encoding="utf-8"),
        )
        baseline = load_skip_policy_baseline(args.baseline)
        policy_sources = read_policy_sources(repository_root)
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"::error::invalid case-gap contract: {exc}")
        return 1
    errors = skip_policy_violations(
        policy_sources,
        baseline,
        migrated=frozenset(contract.sources),
    )
    errors.extend(
        validate_gaps(
            gaps,
            sources=policy_sources,
            migrated_case_files=frozenset(contract.sources),
        )
    )
    failed_numbers = {error.split(":", 1)[0] for error in errors}
    for gap in gaps:
        print(f"{gap.number}: {'failed' if gap.number in failed_numbers else 'open'}")
    for error in errors:
        print(f"::error::{error}")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
