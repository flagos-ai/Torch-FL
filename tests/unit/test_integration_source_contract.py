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

"""The operator conformance suite has one shape on every accelerator.

Changing chips must not require editing test source. Accelerator-specific facts
belong in the wheel's routing table and build record, or in a capability probe
run inside a case. A probe must not decide which test functions are collected,
which parameters exist, or whether a module can be imported at all.

Every expected skip in a migrated file belongs to one tracked issue in
``tests/integration/case_gap_issues.py``. Previously migrated files stay in
``MIGRATED_CASE_FILES``, so adding a skip to them requires a registered gap.
An issue stays open, with its skip in place, until the gap is fixed; only then
can the next shared case group be migrated. The CPU-only lane checks both
collection shape and skip ownership.

Only the Python standard library is imported here. These are source-tree checks
and must not need torch or a device.
"""

import ast
import dataclasses
import json
import re

import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_RELATIVE_PATHS = ("tests/integration/flaggems", "tests/integration/ops")
SUITE_ROOTS = tuple(REPO_ROOT / relative for relative in SUITE_RELATIVE_PATHS)
AGNOSTIC_WORKFLOW = REPO_ROOT / ".github/workflows/agnostic-checks.yml"
CASE_GAP_ISSUES = REPO_ROOT / "tests/integration/case_gap_issues.py"

#: The generated op suites with unconditional MPS expectations that the user
#: approved for the first cross-chip migration. Until a new migration step is
#: reviewed, no other file can introduce a skip without an issue.
MIGRATED_CASE_FILES = frozenset(
    {
        "tests/integration/ops/test_all_dispatch.py",
        "tests/integration/ops/test_mps_all.py",
        "tests/integration/ops/test_mps_argmax.py",
    }
)


def _suite_sources():
    for root in SUITE_ROOTS:
        yield from sorted(root.glob("*.py"))


def _function_name(function):
    """The final name in an expression such as ``platform_support.detect_platform``."""
    if isinstance(function, ast.Constant) and isinstance(function.value, str):
        return function.value
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _case_gap_issues():
    """Read registered issue rows without importing the test helper."""
    tree = ast.parse(_read(CASE_GAP_ISSUES))
    table = next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            and (
                statement.target.id
                if isinstance(statement, ast.AnnAssign)
                else statement.targets[0].id
            )
            == "_ISSUES"
        ),
        None,
    )
    value = table.value if table is not None else None
    if isinstance(value, (ast.Tuple, ast.List)):
        args = value.elts
    elif isinstance(value, ast.Call):
        args = value.args
    else:
        args = ()
    if any(not isinstance(arg, ast.Call) for arg in args):
        raise AssertionError("case gaps may contain only CaseGapIssue(...) rows")
    entries = [arg.args for arg in args]
    issues = {}
    route_owners = {}
    for entry in entries:
        if len(entry) not in {3, 4}:
            raise AssertionError("each gap names an issue, state, cases, and routes")
        issue, state, cases, *routes = entry
        if not (
            isinstance(issue, ast.Constant)
            and isinstance(issue.value, str)
            and re.fullmatch(r"#[1-9][0-9]*", issue.value)
            and isinstance(state, ast.Constant)
            and state.value in {"open", "fixed"}
            and isinstance(cases, (ast.Tuple, ast.List))
            and cases.elts
            and all(
                isinstance(case, ast.Constant)
                and isinstance(case.value, str)
                and "::" in case.value
                for case in cases.elts
            )
            and (
                not routes
                or isinstance(routes[0], (ast.Tuple, ast.List))
                and all(
                    isinstance(route, ast.Constant)
                    and isinstance(route.value, str)
                    and route.value
                    for route in routes[0].elts
                )
            )
        ):
            raise AssertionError("invalid case-gap issue row")
        if issue.value in issues:
            raise AssertionError(f"{issue.value} is registered more than once")
        issues[issue.value] = (
            state.value,
            tuple(case.value for case in cases.elts),
            tuple(route.value for route in routes[0].elts) if routes else (),
        )
        for route in routes[0].elts if routes else ():
            if route.value in route_owners:
                raise AssertionError(f"{route.value} is registered more than once")
            route_owners[route.value] = issue.value
    return issues


def _scope_name(function, parents):
    scope = function
    while id(scope) in parents:
        scope = parents[id(scope)]
        if isinstance(scope, ast.ClassDef):
            return f"{scope.name}::{function.name}"
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return function.name
    return function.name


def _case_marker_names(tree, parents):
    """Yield each collected case and the backend markers it carries."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = set()
        for decorator in node.decorator_list:
            mark = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _function_name(mark) == "mark":
                if not isinstance(decorator, ast.Call) or not decorator.args:
                    continue
                mark = decorator.args[0]
            name = _function_name(mark)
            if name not in {"skip", "skipif"}:
                names.add(name)
        if names:
            yield _scope_name(node, parents), names


def _module_mark_owner(mark, tree):
    """Find a module mark through its bare or list-valued ``pytestmark``."""
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in statement.targets
        ):
            value = statement.value
            items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            if any(item is mark for item in items):
                return "<module>"
    return None


def _skip_marks(tree):
    """Return every skip call and the collected node whose outcome it owns."""
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    marks = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and _function_name(node.func) in {"skip", "skipif"}
        ):
            continue
        component = _module_mark_owner(node, tree)
        current = node
        while component is None and id(current) in parents:
            current = parents[id(current)]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                component = _scope_name(current, parents)
        if component is None and (
            isinstance(node.func.value, ast.Attribute)
            and _function_name(node.func.value.value) == "pytest"
        ):
            component = "<module>"
        marks.append((node, component))
    return marks


def _issue_mentioned(reason):
    """Return every issue key this skip names."""
    mentioned = set()
    for node in ast.walk(reason):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            mentioned.update(re.findall(r"#[1-9][0-9]*", node.value))
        if (
            isinstance(node, ast.Call)
            and _function_name(node.func) == "case_gap_reason"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            mentioned.add(node.args[0].value)
    return mentioned


def _skip_ownership(tree, relative, issues):
    """Associate each skip in a migrated file with its registry row."""
    ownership = []
    route_entries = {
        route: number for number, (_, _, routes) in issues.items() for route in routes
    }
    for call, case in _skip_marks(tree):
        reason = (
            call.args[0]
            if call.args
            else next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg in {"reason", "condition"}
                ),
                None,
            )
        )
        mentioned = _issue_mentioned(reason) if reason is not None else set()
        keys = mentioned | {
            route
            for _, _, routes in issues.values()
            for route in routes
            if route in mentioned
        }
        if not keys:
            ownership.append((call.lineno, None, "skip has no reason", case, relative))
            continue
        for key in keys:
            if key in route_entries:
                key = route_entries[key]
            if key not in issues:
                ownership.append(
                    (call.lineno, key, "skip has no open issue", case, relative)
                )
                continue
            state, cases, routes = issues[key]
            if state != "open":
                ownership.append(
                    (call.lineno, key, "issue is not open", case, relative)
                )
                continue
            owned_by_route = any(route in mentioned for route in routes)
            if owned_by_route:
                ownership.append((call.lineno, key, "tracked", case, relative))
            elif case is None or not any(
                case == registered.removeprefix(f"{relative}::") for registered in cases
            ):
                ownership.append(
                    (call.lineno, key, "skip owner is not registered", case, relative)
                )
            else:
                ownership.append((call.lineno, key, "tracked", case, relative))
    return ownership


def _registered_gap_evidence(source, relative, issues):
    """Return the registered cases and routes that have source evidence."""
    tree = ast.parse(source, filename=relative)
    parents = {
        id(child): node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    cases = {
        f"{relative}::{name}": markers
        for name, markers in _case_marker_names(tree, parents)
    }
    registered = {
        case: issue
        for issue, (_, owned_cases, _) in issues.items()
        for case in owned_cases
    }
    registered_routes = {
        route: issue
        for issue, (_, _, owned_routes) in issues.items()
        for route in owned_routes
    }
    ownership = _skip_ownership(tree, relative, issues)
    observed = set()
    observed_routes = set()
    for line, issue, message, case, _ in ownership:
        if message == "skip has no reason":
            continue
        if issue not in issues or message != "tracked":
            raise AssertionError(
                "the registered skip does not name its owner: "
                f"{relative}:{line}: {issue}: {message}"
            )
        node = f"{relative}::{case}" if case else ""
        if node in registered:
            observed.add(node)
    for route, issue in registered_routes.items():
        if not route.startswith("@"):
            route_cases = {
                f"{relative}::{case}"
                for _, owner, message, case, _ in ownership
                if owner == issue and message == "tracked" and case
            }
            in_source = any(
                isinstance(child, ast.Constant) and child.value == route
                for child in ast.walk(tree)
            )
            found = in_source and bool(route_cases)
            if found:
                observed.update(route_cases & registered.keys())
                observed_routes.add(route)
            continue
        for case, markers in cases.items():
            if registered.get(case) == issue and any(
                route[1:] in marker for marker in markers
            ):
                observed.add(case)
                observed_routes.add(route)
    return observed, observed_routes


def _unattributed_case_skips(migrated=MIGRATED_CASE_FILES):
    """Skips in migrated files without an open, owning issue."""
    findings = []
    for path in _suite_sources():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative not in migrated:
            continue
        for line, issue, message, _, _ in _skip_ownership(
            ast.parse(_read(path), filename=str(path)),
            relative,
            _case_gap_issues(),
        ):
            if message not in {"tracked", "route"}:
                findings.append((relative, line, issue, message))
    return findings


def _read(path):
    return path.read_text(encoding="utf-8")


def _skip_findings(source, issues, relative="test_case.py"):
    tree = ast.parse(source)
    if issues and isinstance(next(iter(issues.values())), _SyntheticIssue):
        issues = {
            issue: (entry.state, entry.cases, entry.routes)
            for issue, entry in issues.items()
        }
    return [
        message
        for _, issue, message, _, _ in _skip_ownership(tree, relative, issues)
        if message not in {"tracked", "route"}
    ]


@dataclasses.dataclass(frozen=True)
class _SyntheticIssue:
    state: str
    cases: tuple[str, ...]
    routes: tuple[str, ...] = ()


def _platform_probe_calls(tree):
    """Locations that read the accelerator identity.

    The build accelerator and detector call sites are explicit so an unrelated
    helper sharing a name such as ``build`` does not produce a false failure.
    The ``platform`` argument on a parameterized case names a *source* arm to
    inspect (for example each CUDA-ABI implementation), not the chip the test is
    running on, so it remains data for the whole suite.
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = _function_name(node.func)
            if callee in {"detect_platform", "platform_for"} or (
                callee in {"build_accelerator", "_build_accelerator"}
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"torch_fl", "_platform"}
            ):
                found.append(node)
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "ACCELERATOR"
            and isinstance(node.value, ast.Name)
            and node.value.id == "_build_config"
        ):
            found.append(node)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if (
                node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "FLAGOS_ACCELERATOR"
            ):
                found.append(node)
    return found


def _module_statements(tree):
    """Statements at module scope, including those inside ``if`` or ``try``."""
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield statement
        if isinstance(statement, (ast.If, ast.Try)):
            yield from _nested_module_statements(statement)


def _nested_module_statements(statement):
    for body in (
        getattr(statement, "body", ()),
        getattr(statement, "orelse", ()),
        getattr(statement, "finalbody", ()),
    ):
        for nested in body:
            if isinstance(
                nested, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            yield nested
            if isinstance(nested, (ast.If, ast.Try)):
                yield from _nested_module_statements(nested)


def _probe_uses(tree):
    """Accelerator probes that decide the suite's collection shape.

    A capability probe inside a test body is allowed to skip that call. What
    this rejects is a probe used at import or decoration time to choose the test
    functions, parameter list, or skip condition.
    """
    probes = {id(probe): probe for probe in _platform_probe_calls(tree)}
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    function_parents = {}
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        function_parents[function] = {
            id(child): parent_node
            for parent_node in ast.walk(function)
            for child in ast.iter_child_nodes(parent_node)
        }

    declared_probes = {}
    declaration_names = {}
    for statement in ast.walk(tree):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = getattr(statement, "value", None)
        if not isinstance(value, ast.expr):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            name = targets[0].id
            declaration_names[name] = value
            declared_probes[name] = [
                probe for probe in _platform_probe_calls(value) if id(probe) in probes
            ]

    def source_probes(expression):
        found = []
        todo = list(expression) if isinstance(expression, list) else [expression]
        seen = set()
        while todo:
            current = todo.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if id(current) in probes:
                found.append(current)
            elif isinstance(current, ast.expr):
                todo.extend(ast.iter_child_nodes(current))
                if isinstance(current, ast.Name) and current.id in declaration_names:
                    todo.append(declaration_names[current.id])
        return found

    def shape_uses(root, parents):
        for node in ast.walk(root):
            shape_probes = source_probes(node)
            if not shape_probes:
                continue
            if any(
                id(probe) in parents
                and isinstance(parents[id(probe)], ast.expr)
                and isinstance(probe, ast.expr)
                for probe in shape_probes
            ):
                yield from shape_probes
                continue
            current = node
            while id(current) in parents:
                current = parents[id(current)]
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If)):
                    break
                if isinstance(current, ast.Call):
                    if isinstance(current.func, ast.Attribute) and (
                        current.func.attr in {"parametrize", "skipif"}
                    ):
                        yield from shape_probes
                        break
                    if any(
                        keyword.arg in {"condition", "skip", "param", "argvalues"}
                        for keyword in current.keywords
                    ):
                        yield from shape_probes
                        break

    uses = {
        probe
        for statement in _module_statements(tree)
        for probe in shape_uses(statement, parent)
    }
    for declared_name in declaration_names:
        current = declaration_names[declared_name]
        while id(current) in parent:
            current = parent[id(current)]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If)):
                break
            if isinstance(current, ast.Call) and any(
                keyword.arg in {"condition", "skip", "param", "argvalues"}
                for keyword in current.keywords
            ):
                uses.update(
                    source_probes(
                        [
                            name
                            for name in ast.walk(current)
                            if isinstance(name, ast.Name) and name.id == declared_name
                        ]
                    )
                )
                break
    for function, parents in function_parents.items():
        if any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr in {"parametrize", "skipif"}
            for decorator in function.decorator_list
        ):
            for decorator in function.decorator_list:
                uses.update(shape_uses(decorator, parents))
    return uses


def _collection_violations():
    """Return ``(relative path, line, node)`` for each illegal probe use."""
    violations = []
    for path in _suite_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(REPO_ROOT).as_posix()
        for probe in _probe_uses(tree):
            violations.append((relative, probe.lineno, ast.dump(probe)))
    return violations


#: The platform-shaped test files this first contract step found. They predate
#: the shared-case migration: most carry a module-level MPS skip and their own
#: vocabulary for the same operator. Every entry must still contain a violation;
#: fixing one requires deleting it here, so the list cannot hide the next
#: branch after that file has moved to the shared shape.
_MIGRATION_EXEMPTIONS = frozenset(
    {
        *(
            f"tests/integration/ops/test_mps_{name}.py"
            for name in (
                "batch_norm",
                "binary",
                "cat",
                "contiguous",
                "conv",
                "cumsum",
                "div",
                "embedding",
                "erf",
                "exp",
                "gather",
                "gemm",
                "index",
                "index_select",
                "layer_norm",
                "log",
                "masked_fill",
                "pow",
                "qwen35",
                "reduce",
                "rotary_and_compare",
                "rsqrt",
                "sigmoid",
                "silu",
                "softmax",
                "softplus",
                "sqrt",
                "take",
                "tanh",
                "triu",
                "unary",
                "views",
                "vision_interpolation",
                "where",
            )
        ),
    }
)


def test_platform_detector_covers_every_vendor_conf():
    """A new vendor is a detector/conf naming change, not a suite branch."""
    confs = {
        path.stem.removeprefix("backends_")
        for path in (REPO_ROOT / "torch_fl/configs").glob("backends_*.conf")
        if not path.stem.endswith("_boxing")
    }
    platform_source = ast.parse(
        (REPO_ROOT / "torch_fl/_platform.py").read_text(encoding="utf-8")
    )
    cmake_platforms = json.loads(
        (REPO_ROOT / "cmake/flagos_platforms.json").read_text(encoding="utf-8")
    )
    cmake_vendors = {row["vendor"] for row in cmake_platforms["accelerators"].values()}
    platform_mapping = next(
        node.value
        for node in platform_source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PLATFORM_VENDOR"
            for target in node.targets
        )
    )
    vendors = {
        item.value for item in platform_mapping.keys if isinstance(item, ast.Constant)
    }
    assert confs
    for vendor in confs:
        assert vendor in vendors or vendor in cmake_vendors, vendor


def test_probe_recognizes_imported_and_qualified_detectors():
    tree = ast.parse(
        "from platform_support import detect_platform\n"
        "from torch_fl._platform import platform_for\n"
        "detect_platform()\n"
        "platform_for('nvidia')\n"
        "torch_fl._build_accelerator()\n"
    )
    assert len(_platform_probe_calls(tree)) == 3


def test_unrelated_build_accelerator_does_not_match_the_platform_probe():
    tree = ast.parse("my_build_accelerator()\nother.build_accelerator()\n")
    assert not _platform_probe_calls(tree)


def test_new_platform_shape_does_not_alter_collection():
    tree = ast.parse(
        "import pytest\n"
        "from platform_support import detect_platform\n"
        "pytestmark = pytest.mark.skipif(detect_platform() != 'mps')\n"
    )
    assert len(_probe_uses(tree)) == 1


def test_inside_case_capability_probe_is_not_collection_control():
    tree = ast.parse(
        "import pytest\n"
        "def test_case():\n"
        "    try:\n"
        "        run()\n"
        "    except NotImplementedError:\n"
        "        pytest.skip('route absent')\n"
    )
    assert not _platform_probe_calls(tree)


def test_inside_case_platform_probe_is_not_collection_control():
    tree = ast.parse(
        "import pytest\n"
        "def test_case():\n"
        "    if detect_platform() == 'gcu':\n"
        "        pytest.skip('native gate is absent')\n"
    )
    assert not _probe_uses(tree)


def test_named_test_with_a_skip_probe_does_not_change_the_parameter_set():
    tree = ast.parse("import pytest\ndef test_case(dtype):\n    run('kernel gate')\n")
    assert not _probe_uses(tree)


def test_parameter_list_from_a_probe_is_checked():
    tree = ast.parse(
        "import pytest\n"
        "marks = [pytest.mark.skipif(False)]\n"
        "def test_case():\n"
        "    pass\n"
        "marks.append(pytest.mark.parametrize('chip', [detect_platform()]))\n"
    )
    assert len(_probe_uses(tree)) == 1


def test_indirect_parameter_probe_is_checked():
    tree = ast.parse(
        "import pytest\n"
        "PLATFORM = detect_platform()\n"
        "@pytest.mark.parametrize('case', [] if PLATFORM == 'mps' else [1])\n"
        "def test_case(case):\n"
        "    pass\n"
    )
    assert len(_probe_uses(tree)) == 1


def test_platform_guarded_import_is_checked():
    tree = ast.parse("if detect_platform() == 'mps':\n    import mps_helper\n")
    assert len(_probe_uses(tree)) == 1


def test_platform_defined_only_test_is_checked():
    tree = ast.parse(
        "if detect_platform() == 'mps':\n    def test_mps_only():\n        pass\n"
    )
    assert len(_probe_uses(tree)) == 1


def test_no_new_platform_branch_changes_the_integration_suite_shape():
    violations = _collection_violations()
    unexpected = sorted(
        (path, line, expression)
        for path, line, expression in violations
        if path not in _MIGRATION_EXEMPTIONS
    )
    assert not unexpected, "move the capability check inside the test case: " + repr(
        unexpected
    )


def test_migration_exemptions_are_exact_and_still_necessary():
    violations = _collection_violations()
    observed = {path for path, _, _ in violations}
    assert observed == _MIGRATION_EXEMPTIONS, (
        "remove fixed files from _MIGRATION_EXEMPTIONS without adding new ones: "
        f"fixed={sorted(_MIGRATION_EXEMPTIONS - observed)}, "
        f"new={sorted(observed - _MIGRATION_EXEMPTIONS)}"
    )


def test_every_expected_skip_names_one_open_case_issue():
    findings = _unattributed_case_skips()
    assert not findings, (
        "register the gap in integration/case_gap_issues.py and name its issue "
        "in the skip: " + repr(findings)
    )


def test_migrated_skips_reject_unregistered_and_closed_issues():
    cases = ("tests/integration/ops/test_case.py::test_case",)
    rows = {
        "#1": _SyntheticIssue("open", cases),
        "#2": _SyntheticIssue("fixed", cases),
    }
    source = (
        "import pytest\ndef test_case():\n    pytest.skip(case_gap_reason('{issue}'))\n"
    )
    relative = "tests/integration/ops/test_case.py"
    assert not _skip_findings(source.format(issue="#1"), rows, relative)
    assert _skip_findings(source.format(issue="#2"), rows, relative) == [
        "issue is not open"
    ]
    assert _skip_findings(source.format(issue="#3"), rows, relative) == [
        "skip has no open issue"
    ]
    assert _skip_findings(
        "import pytest\ndef test_case():\n    pytest.skip('x')\n", rows, relative
    ) == ["skip has no reason"]


def test_case_gap_evidence_rejects_a_missing_skip_or_wrong_owner():
    issues = {
        "#1": (
            "open",
            ("tests/integration/ops/test_case.py::test_skip",),
            ("all:none",),
        ),
        "#2": (
            "open",
            ("tests/integration/ops/test_case.py::TestGated::test_gated",),
            ("@cuda",),
        ),
    }
    source = """\
import pytest

def test_skip():
    pytest.skip(case_gap_reason("#1"))

class TestGated:
    @pytest.mark.cuda
    def test_gated(self):
        assert routed_backend_or_none("all") is None
"""
    relative = "tests/integration/ops/test_case.py"
    assert _registered_gap_evidence(source, relative, issues) == (
        {
            f"{relative}::{case.split('::', 1)[1]}"
            for _, cases, _ in issues.values()
            for case in cases
        },
        {"@cuda"},
    )
    with pytest.raises(AssertionError, match="registered skip"):
        wrong_owner = {
            "#2": issues["#2"],
            "#1": (issues["#1"][0], issues["#1"][1], ()),
        }
        _registered_gap_evidence(
            source.replace('case_gap_reason("#1")', '"expected case gap #2"'),
            relative,
            wrong_owner,
        )
    for missing_route in ("all:none", "@cuda"):
        incomplete = {
            issue: (
                state,
                cases,
                tuple(route for route in routes if route != missing_route),
            )
            for issue, (state, cases, routes) in issues.items()
        }
        observed, routes = _registered_gap_evidence(source, relative, incomplete)
        assert missing_route not in routes
    no_skip, _ = _registered_gap_evidence(
        source.replace('pytest.skip(case_gap_reason("#1"))', "pass"),
        relative,
        issues,
    )
    assert f"{relative}::test_skip" not in no_skip
    no_gate, no_route = _registered_gap_evidence(
        source.replace("@pytest.mark.cuda", ""), relative, issues
    )
    assert f"{relative}::TestGated::test_gated" not in no_gate
    assert "@cuda" not in no_route


def _registered_evidence_for_suite():
    issues = _case_gap_issues()
    observed = set()
    routes = set()
    for path in _suite_sources():
        relative = path.relative_to(REPO_ROOT).as_posix()
        found, found_routes = _registered_gap_evidence(_read(path), relative, issues)
        observed.update(found)
        routes.update(found_routes)
    return observed, routes


def test_every_registered_case_and_route_has_its_own_gap_evidence():
    issues = _case_gap_issues()
    expected = {
        (issue, case) for issue, (_, cases, _) in issues.items() for case in cases
    }
    evidence = {
        (issue, case)
        for issue, (_, cases, _) in issues.items()
        for case in cases
        if case
        in _registered_gap_evidence(
            _read(REPO_ROOT / case.split("::", 1)[0]),
            case.split("::", 1)[0],
            issues,
        )[0]
    }
    assert expected == evidence, (
        "register every expected skip against its actual case: "
        f"missing={sorted(expected - evidence)}"
    )
    expected_routes = {route for _, _, routes in issues.values() for route in routes}
    found, observed_routes = _registered_evidence_for_suite()
    assert expected_routes == observed_routes, (
        "register a route only when it gates one of its issue's recorded cases: "
        f"missing={sorted(expected_routes - observed_routes)}"
    )
    assert found == {case for _, cases, _ in issues.values() for case in cases}, (
        "each registered case needs its owning skip or route gate"
    )


def test_gap_registry_skips_are_enforced_by_the_cpu_lane():
    workflow = AGNOSTIC_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/unit/test_integration_source_contract.py" in workflow


def test_platform_shape_contract_runs_in_cpu_ci():
    workflow = AGNOSTIC_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/unit/test_integration_source_contract.py" in workflow
    assert workflow.index("tests/unit/test_integration_source_contract.py") < (
        workflow.index("tests/unit/test_integration_workdir.py")
    ), "check the CPU workflow reader still reaches the integration tests"
    assert "tests/unit/test_integration_manifest.py" in workflow
    assert "Run platform-agnostic unit tests" in workflow
