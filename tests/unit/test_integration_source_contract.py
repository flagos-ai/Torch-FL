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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOTS = tuple(
    REPO_ROOT / relative
    for relative in ("tests/integration/flaggems", "tests/integration/ops")
)
AGNOSTIC_WORKFLOW = REPO_ROOT / ".github/workflows/agnostic-checks.yml"


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
                "all",
                "argmax",
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


def test_platform_shape_contract_runs_in_cpu_ci():
    workflow = AGNOSTIC_WORKFLOW.read_text(encoding="utf-8")
    assert "tests/unit/test_integration_source_contract.py" in workflow
    assert workflow.index("tests/unit/test_integration_source_contract.py") < (
        workflow.index("tests/unit/test_integration_workdir.py")
    ), "check the CPU workflow reader still reaches the integration tests"
    assert "tests/unit/test_integration_manifest.py" in workflow
    assert "Run platform-agnostic unit tests" in workflow
