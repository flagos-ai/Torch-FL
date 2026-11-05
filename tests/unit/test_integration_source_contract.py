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

This module checks that rule statically, so the CPU-only lane catches a new
platform branch before any accelerator job is spent on it. The migration list
below records the existing branches that still need to move. It may shrink as
those files gain shared cases, but adding an entry for a new violation is not a
supported way to make this check pass.

Only the Python standard library is imported here. These are source-tree checks
and must not need torch or a device.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOTS = tuple(
    REPO_ROOT / relative
    for relative in ("tests/integration/flaggems", "tests/integration/ops")
)
AGNOSTIC_WORKFLOW = REPO_ROOT / ".github/workflows/agnostic-checks.yml"
MPS_SUITE = REPO_ROOT / "tests/integration/ops"
MIGRATED_CASE_FILES = frozenset(
    {
        "tests/integration/ops/test_all_dispatch.py",
        "tests/integration/ops/test_mps_all.py",
        "tests/integration/ops/test_mps_argmax.py",
    }
)
CASE_GAP_REGISTRY = REPO_ROOT / "tests/integration/case_gap_issues.py"
CASE_GAP_CHECKER = REPO_ROOT / ".github/scripts/ci/check_case_gap_issues.py"


def _suite_sources():
    for root in SUITE_ROOTS:
        if root.is_dir():
            yield from sorted(root.glob("*.py"))


def _case_gap_checker():
    spec = importlib.util.spec_from_file_location("case_gap_contract", CASE_GAP_CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _function_name(function):
    """The final name in an expression such as ``platform_support.detect_platform``."""
    if isinstance(function, ast.Constant) and isinstance(function.value, str):
        return function.value
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _call_keywords(call):
    if not isinstance(call, ast.Call):
        return {}
    return {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


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
#: vocabulary for the same operator. Every entry must still exist; a missing one
#: requires deleting it here, so a renamed or split file cannot escape review.
#: A migrated file leaves this list. A newly added MPS-only file must be migrated
#: before it lands -- adding it here requires a deliberate source change and is
#: reported as a migration gap, not accepted as a valid way to add coverage.
_MIGRATION_EXEMPTIONS = frozenset(
    {
        *(
            f"tests/integration/ops/test_mps_{name}.py"
            for name in (
                "activations",
                "addmm",
                "all",
                "argmax",
                "argmax_nan",
                "batch_norm",
                "binary",
                "binary_compare",
                "cat",
                "clamp_integral",
                "contiguous",
                "conv",
                "conv_backward",
                "cumprod",
                "cumsum",
                "div",
                "differential_sweep",
                "embedding",
                "erf",
                "exp",
                "exp_log_bases",
                "extrema_product",
                "gather",
                "gemm",
                "index",
                "index_select",
                "inverse_trig",
                "layer_norm",
                "log",
                "masked_fill",
                "not_tan",
                "pow",
                "qwen35",
                "reduce",
                "rotary_and_compare",
                "rounding_predicates",
                "rsqrt",
                "sigmoid",
                "silu",
                "softmax",
                "softplus",
                "sqrt",
                "take",
                "tanh",
                "tril",
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
    string_literals = {
        node.value
        for node in ast.walk(platform_source)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert confs
    for vendor in confs:
        assert vendor in string_literals, vendor


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


def test_every_platform_specific_mps_file_is_migrated_or_listed():
    """A missing entry has to be migrated or receive a deliberate exemption."""
    platform_files = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in MPS_SUITE.glob("test_mps_*.py")
    }
    unhandled = platform_files - MIGRATED_CASE_FILES - _MIGRATION_EXEMPTIONS
    unhandled -= {path for path, _, _ in _collection_violations()}
    assert not unhandled, (
        f"platform-specific files without a collection violation: {sorted(unhandled)}"
    )


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
        if path not in MIGRATED_CASE_FILES and path not in _MIGRATION_EXEMPTIONS
    )
    assert not unexpected, "move the capability check inside the test case: " + repr(
        unexpected
    )


def test_migration_exemptions_are_exact_and_still_necessary():
    violations = _collection_violations()
    observed = {path for path, _, _ in violations}
    migrated_files = {
        "tests/integration/ops/test_mps_all.py",
        "tests/integration/ops/test_mps_argmax.py",
    }
    assert observed & _MIGRATION_EXEMPTIONS
    assert observed == _MIGRATION_EXEMPTIONS - migrated_files, (
        "platform-specific files must be migrated or stay on the frozen exemption list: "
        f"unmigrated={sorted(_MIGRATION_EXEMPTIONS - observed)}, "
        f"new={sorted(observed - _MIGRATION_EXEMPTIONS)}"
    )


def test_platform_shape_contract_runs_in_cpu_ci():
    workflow = AGNOSTIC_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/unit/test_integration_source_contract.py" in workflow
    assert workflow.index("tests/unit/test_integration_source_contract.py") < (
        workflow.index("tests/unit/test_integration_workdir.py")
    ), "check the CPU workflow reader still reaches the integration tests"
    assert "tests/unit/test_integration_manifest.py" in workflow
    assert "Run platform-agnostic unit tests" in workflow


def _string_tuple(node):
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise ValueError("case gap cases and routes must be tuples")
    values = []
    for value in node.elts:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ValueError("case gap IDs must be strings")
        values.append(value.value)
    for value in values:
        if (
            not value
            or "/" in value
            and value.split("/")[-1].isdigit()
            or value.startswith("#")
        ):
            raise ValueError(f"invalid case gap reference {value!r}")
    return tuple(values)


def _skip_violations(tree, *, brand_new=False):
    """Return bare skips and issue-tag/API mismatches in the source."""
    decorated_tests = {
        id(decorator): node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
    }

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _function_name(node.func)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"skip", "skip_class"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
        ):
            found.append((node.lineno, "bare pytest.skip"))
        if isinstance(node.func, ast.Name) and node.func.id == "skip":
            found.append((node.lineno, "bare skip is forbidden"))
        if brand_new and (
            (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"skip", "skip_class", "skipif", "xfail"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "mark"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"skip", "skipif", "xfail"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "mark"
            )
        ):
            found.append((node.lineno, "skip aliases are forbidden"))
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"skip", "skipif"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "mark"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "pytest"
        ):
            found.append((node.lineno, "pytest.mark.skip and skipif are forbidden"))
        if callee != "tracked_skip":
            continue
        if not isinstance(node.func, ast.Name):
            found.append((node.lineno, "tracked_skip must be imported"))
        keywords = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }
        if {"condition", "issue"} - set(keywords):
            found.append((node.lineno, "tracked_skip needs keyword arguments"))
        if set(keywords) != {"condition", "issue"} or any(
            keyword.arg is None for keyword in node.keywords
        ):
            found.append((node.lineno, "tracked_skip has an unsupported field"))
        condition = keywords.get("condition")
        if condition is None:
            found.append((node.lineno, "tracked_skip needs a direct condition"))
        issue = keywords.get("issue")
        if not (
            isinstance(issue, ast.Constant)
            and isinstance(issue.value, str)
            and issue.value.startswith("#")
            and issue.value[1:].isdigit()
            and issue.value[1:] != "0"
        ):
            found.append((node.lineno, "tracked_skip needs a literal issue number"))
        owner = decorated_tests.get(id(node))
        if owner is None or not owner.name.startswith("test_"):
            found.append((node.lineno, "tracked_skip must decorate a test"))
    return found


def _registered_cases(path, rows, tree):
    relative = path.relative_to(REPO_ROOT).as_posix()
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def components(node):
        path = [node.name]
        current = parents.get(id(node))
        while current is not None:
            if isinstance(
                current, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                path.append(current.name)
            current = parents.get(id(current))
        return tuple(reversed(path))

    functions = [
        node
        for statement in tree.body
        for node in (
            [statement]
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            else list(ast.walk(statement))
            if isinstance(statement, ast.ClassDef)
            else ()
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    all_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    skipped = {
        id(node): node
        for node in functions
        if any(
            _function_name(
                decorator.func if isinstance(decorator, ast.Call) else decorator
            )
            == "tracked_skip"
            for decorator in node.decorator_list
        )
    }

    cases = []
    registered = set()
    for number, owner_cases, _ in rows:
        for case in owner_cases:
            source, _, qualname = case.partition("::")
            if source != relative:
                continue
            wanted = tuple(
                value.split("[", 1)[0] for value in qualname.split("::") if value
            )
            matches = [node for node in functions if components(node) == wanted]
            if not matches:
                uncollected = [
                    node
                    for node in all_functions
                    if components(node)[-1:] == wanted[-1:]
                ]
                if uncollected:
                    raise ValueError(
                        f"{relative}: no test function for {case}; "
                        "the registered case is not collected"
                    )
                raise ValueError(f"{relative}: no test function for {case}")
            if len(matches) != 1:
                raise ValueError(f"{relative}: duplicate test functions for {case}")
            function = matches[0]
            if id(function) not in skipped:
                raise ValueError(
                    f"{relative}: {case} is not registered to a tracked skip"
                )
            issues = [
                _call_keywords(decorator).get("issue")
                for decorator in function.decorator_list
                if _function_name(
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                == "tracked_skip"
            ]
            if len(issues) != 1 or not any(
                isinstance(issue, ast.Constant) and issue.value == number
                for issue in issues
            ):
                if len(issues) != 1:
                    raise ValueError(
                        f"{relative}: {case} must have exactly one tracked skip"
                    )
                raise ValueError(
                    f"{relative}: tracked skip is not registered to {case} under {number}"
                )
            cases.append(case)
            registered.add(id(function))
    extra = set(skipped) - registered
    if extra:
        line_numbers = sorted(skipped[node].lineno for node in extra)
        raise ValueError(
            f"{relative}: tracked skips not registered at lines {line_numbers}"
        )
    return cases


def _verify_case_gaps(registry=None):
    registry = registry or CASE_GAP_REGISTRY
    registry_source = registry.read_text(encoding="utf-8")
    if "state=" in registry_source:
        raise ValueError("issue state must be read from GitHub, not the registry")
    checker = _case_gap_checker()
    registered = checker.registered_gaps(registry)
    rows = tuple((gap.number, gap.cases, gap.routes) for gap in registered)
    all_cases = [case for _, cases, _ in rows for case in cases]
    all_routes = [route for _, _, routes in rows for route in routes]
    if len(all_cases) != len(set(all_cases)):
        raise ValueError("a skipped case must belong to exactly one issue")
    if len(all_routes) != len(set(all_routes)):
        raise ValueError("a route must belong to exactly one issue")
    migrated = MIGRATED_CASE_FILES | {
        case.split("::", 1)[0] for _, cases, _ in rows for case in cases
    }
    source_paths = list(_suite_sources())
    source_paths.extend(
        REPO_ROOT / case.split("::", 1)[0]
        for _, cases, _ in rows
        for case in cases
        if (REPO_ROOT / case.split("::", 1)[0]).resolve()
        not in {path.resolve() for path in source_paths}
        and (REPO_ROOT / case.split("::", 1)[0]).is_file()
    )
    sources = {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in source_paths
    }
    source_names = {path.relative_to(REPO_ROOT).as_posix() for path in source_paths}
    violations = []
    for path, tree in sources.items():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative not in migrated:
            continue
        for line, reason in _skip_violations(tree, brand_new=True):
            violations.append(f"{relative}:{line}: {reason}")
    for source in sorted(
        {case.split("::", 1)[0] for _, cases, _ in rows for case in cases}
        - source_names
    ):
        violations.append(f"registered case source is missing: {source}")
    if violations:
        raise ValueError("; ".join(violations))
    observed = [
        case
        for path, tree in sources.items()
        if path.relative_to(REPO_ROOT).as_posix() in migrated
        for case in _registered_cases(path, rows, tree)
    ]
    missing_cases = set(all_cases) - set(observed)
    if missing_cases:
        raise ValueError(
            f"registered cases have no tracked skip: {sorted(missing_cases)}"
        )
    return rows


def test_case_gap_skips_are_tracked_and_owned():
    rows = _verify_case_gaps()
    assert {number for number, _, _ in rows} == {"#710"}


@pytest.mark.parametrize(
    "source, message",
    [
        (
            "import pytest\ndef test_x():\n    pytest.skip('gap')\n",
            "bare pytest.skip",
        ),
        (
            "from inspect import skip\ndef test_x():\n    skip('gap')\n",
            "bare skip",
        ),
        (
            "import pytest\ndef test_x():\n"
            "    pytest.skip(require_case_gap(condition(), 'gap'))\n",
            "bare pytest.skip",
        ),
        (
            "import pytest\n@pytest.mark.skipif(lambda: True, reason='gap')\n"
            "def test_x():\n    pass\n",
            "skipif",
        ),
        (
            "from x import tracked_skip\n"
            "@tracked_skip(condition=True, issue='#1')\n"
            "def test_x():\n    pass\n",
            "",
        ),
    ],
)
def test_tracked_skip_source_contract_rejects_bare_skips(source, message):
    violations = _skip_violations(ast.parse(source))
    if message:
        assert any(message in reason for _, reason in violations)
    else:
        assert violations == []


def test_tracked_skip_cannot_be_used_inside_a_case():
    tree = ast.parse(
        "from x import tracked_skip\n"
        "def test_x():\n"
        "    tracked_skip(condition=True, issue='#1')\n"
    )
    assert any("must decorate a test" in reason for _, reason in _skip_violations(tree))


def test_tracked_skip_cannot_accept_a_handwritten_reason():
    tree = ast.parse(
        "from gaps import tracked_skip\n"
        "@tracked_skip(condition=True, issue='#1', reason='gap')\n"
        "def test_x():\n"
        "    pass\n"
    )
    assert any(
        "has an unsupported field" in reason for _, reason in _skip_violations(tree)
    )


@pytest.mark.parametrize(
    "row",
    [
        'CaseGapIssue("#1", cases=(), routes=(), upstream="a/b/issues/1")',
        'CaseGapIssue("#1", cases=("x::test_x",), routes=("x",), upstream="a/b/issues/1")',
        'CaseGapIssue("#1", cases=(), routes=("#1",), upstream="a/b/issues/1")',
        'CaseGapIssue(number="1", cases=("x",), routes=(), upstream="a/b/issues/1")',
        'CaseGapIssue(number="#1", cases=("x",), routes=(), upstream="a/b/pulls/1")',
        'CaseGapIssue(number="#1", cases=("x",), routes=(), upstream="a/b/issues/1", state="open")',
    ],
)
def test_malformed_or_untracked_gap_fails(row):
    with pytest.raises(ValueError):
        _case_gap_checker().parse_registered_gaps(ast.parse(f"_ISSUES = ({row},)\n"))


def _write_gap_fixture(root, source):
    ops = root / "tests/integration/ops"
    ops.mkdir(parents=True)
    registry = ops.parent / "case_gap_issues.py"
    registry.write_text(source, encoding="utf-8")
    return registry


def _write_gap_suite(root, registry_source, test_source):
    registry = _write_gap_fixture(root, registry_source)
    (root / "tests/integration/ops/test_x.py").write_text(test_source, encoding="utf-8")
    return registry


def _use_gap_suite(monkeypatch, root):
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", root)
    monkeypatch.setattr(
        sys.modules[__name__], "SUITE_ROOTS", (root / "tests/integration/ops",)
    )


_VALID_GAP_REGISTRY = (
    "CaseGapIssue(number='#1', "
    "cases=('tests/integration/ops/test_x.py::test_x',), routes=(), "
    "upstream='a/b/issues/2')\n"
)
_VALID_GAP_SOURCE = (
    "from gaps import tracked_skip\n"
    "@tracked_skip(condition=True, issue='#1')\n"
    "def test_x():\n"
    "    pass\n"
)


def test_case_gap_fixture_passes_the_shared_contract(monkeypatch, tmp_path):
    registry = _write_gap_suite(
        tmp_path, f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n", _VALID_GAP_SOURCE
    )
    _use_gap_suite(monkeypatch, tmp_path)
    assert _verify_case_gaps(registry)[0][0] == "#1"


def test_optional_upstream_does_not_require_a_second_issue():
    tree = ast.parse(
        '_ISSUES = (CaseGapIssue(number="#1", cases=("x",), routes=(), upstream=""),)\n'
    )
    assert _case_gap_checker().parse_registered_gaps(tree)[0].number == "#1"


@pytest.mark.parametrize(
    "registry_source, test_source, message",
    [
        (
            f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n",
            "import pytest\ndef test_x():\n    pytest.skip('gap')\n",
            "bare pytest.skip",
        ),
        (
            "_ISSUES = (CaseGapIssue(number='#1', cases=(), "
            "routes=(), upstream='a/b/issues/2'),)\n",
            _VALID_GAP_SOURCE,
            "must own cases or routes",
        ),
        (f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n", "", "no test function"),
        (
            f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n",
            _VALID_GAP_SOURCE.replace("issue='#1'", "issue='#2'"),
            "is not registered",
        ),
        (
            f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n",
            _VALID_GAP_SOURCE.replace("@tracked_skip", "@other_skip"),
            "is not registered to a tracked skip",
        ),
        (
            f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n",
            "try:\n"
            "    import unavailable\n"
            "    def test_x():\n"
            "        pass\n"
            "except ImportError:\n"
            "    pass\n",
            "no test function",
        ),
    ],
)
def test_case_gap_contract_rejects_incomplete_registration(
    monkeypatch, tmp_path, registry_source, test_source, message
):
    registry = _write_gap_suite(tmp_path, registry_source, test_source)
    _use_gap_suite(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=message):
        _verify_case_gaps(registry)


def test_duplicate_issue_reference_fails_the_contract():
    source = f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n_ISSUES = ()\n"
    with pytest.raises(ValueError, match="exactly once"):
        _case_gap_checker().parse_registered_gaps(ast.parse(source))


@pytest.mark.parametrize("issue", ["#", "#abc", "#0", "#-1", "#1x"])
def test_tracked_skip_rejects_malformed_issue(issue):
    source = _VALID_GAP_SOURCE.replace("issue='#1'", f"issue={issue!r}")
    assert any(
        "literal issue number" in reason
        for _, reason in _skip_violations(ast.parse(source))
    )


def test_registered_case_cannot_be_inside_a_helper(monkeypatch, tmp_path):
    row = _VALID_GAP_REGISTRY.replace("test_x'", "helper::test_x'")
    registry = _write_gap_suite(
        tmp_path,
        f"_ISSUES = ({row},)\n",
        "def helper():\n"
        "    @tracked_skip(condition=True, issue='#1')\n"
        "    def test_x():\n"
        "        pass\n",
    )
    _use_gap_suite(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no test function"):
        _verify_case_gaps(registry)


def test_registered_case_uses_its_outer_skip_decorator(monkeypatch, tmp_path):
    row = _VALID_GAP_REGISTRY.replace("test_x'", "TestOuter::test_x[1]'")
    registry = _write_gap_suite(
        tmp_path,
        f"_ISSUES = ({row},)\n",
        "import pytest\n"
        "from gaps import tracked_skip\n"
        "class TestOuter:\n"
        "    @tracked_skip(condition=True, issue='#1')\n"
        "    @pytest.mark.parametrize('value', [1, 2])\n"
        "    def test_x(self, value):\n"
        "        pass\n",
    )
    _use_gap_suite(monkeypatch, tmp_path)
    case = "tests/integration/ops/test_x.py::TestOuter::test_x[1]"
    assert _verify_case_gaps(registry)[0][1] == (case,)


def test_conditional_registered_case_is_uncollected(monkeypatch, tmp_path):
    row = _VALID_GAP_REGISTRY.replace("test_x'", "TestConditional::test_x'")
    registry = _write_gap_suite(
        tmp_path,
        f"_ISSUES = ({row},)\n",
        "from gaps import tracked_skip\n"
        "try:\n"
        "    import unavailable\n"
        "    @tracked_skip(condition=True, issue='#1')\n"
        "    def test_x():\n"
        "        pass\n"
        "except ImportError:\n"
        "    pass\n",
    )
    _use_gap_suite(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no test function"):
        _verify_case_gaps(registry)


def test_registry_matches_full_test_node_id(monkeypatch, tmp_path):
    row = _VALID_GAP_REGISTRY.replace("test_x'", "test_module'")
    registry = _write_gap_suite(
        tmp_path,
        f"_ISSUES = ({row},)\n",
        _VALID_GAP_SOURCE.replace("def test_x", "def test_module")
        + "def test_x():\n"
        + "    pass\n"
        + "class TestOther:\n"
        + "    @other_skip(condition=True)\n"
        + "    def test_x(self):\n"
        + "        pass\n",
    )
    _use_gap_suite(monkeypatch, tmp_path)
    assert _verify_case_gaps(registry)[0][0] == "#1"


def test_duplicate_function_names_are_not_registered_by_their_class(
    monkeypatch, tmp_path
):
    row = _VALID_GAP_REGISTRY.replace("test_x'", "TestOwned::test_x'")
    registry = _write_gap_suite(
        tmp_path,
        f"_ISSUES = ({row},)\n",
        "class TestOther:\n"
        "    @tracked_skip(condition=True, issue='#1')\n"
        "    def test_x(self):\n"
        "        pass\n"
        "class TestOwned:\n"
        "    def test_x(self):\n"
        "        pass\n",
    )
    _use_gap_suite(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="not registered to a tracked skip"):
        _verify_case_gaps(registry)


def test_registered_case_rejects_two_skip_decorators(monkeypatch, tmp_path):
    registry = _write_gap_suite(
        tmp_path,
        f"_ISSUES = ({_VALID_GAP_REGISTRY},)\n",
        "from gaps import tracked_skip\n"
        "@tracked_skip(condition=True, issue='#1')\n"
        "@tracked_skip(condition=True, issue='#1')\n"
        "def test_x():\n"
        "    pass\n",
    )
    _use_gap_suite(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="exactly one tracked skip"):
        _verify_case_gaps(registry)
