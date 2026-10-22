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

"""Static rules for integration tests, usable without importing torch."""

from __future__ import annotations

import ast
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
# These modules defer imports to functions by design. Their callers initialize
# the device at module scope before invoking the helper.
LAZY_IMPORT_ALLOWLIST = frozenset(
    {
        "conftest.py",
        "flaggems/flaggems_caps.py",
        "flaggems/run.py",
        "profiler_support.py",
        "runtime/runtime_caps.py",
        "test_musa_runtime.py",
        "vendor_lanes/vendor_caps.py",
    }
)
DEVICE_IMPORT = "from device_init import torch\n"
DEVICE_IMPORT_WITH_FL = "from device_init import torch, torch_fl\n"
DEVICE_NAMES = ("torch", "torch_fl")
_TORCH_PACKAGES = frozenset(DEVICE_NAMES)


def package_imports(node: ast.AST) -> list[tuple[str, int]]:
    """Return each torch package this statement actually loads."""
    if isinstance(node, ast.Import):
        return [
            (alias.name.split(".", 1)[0], node.lineno)
            for alias in node.names
            if alias.name.split(".", 1)[0] in _TORCH_PACKAGES
        ]
    if (
        isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".", 1)[0] in _TORCH_PACKAGES
        and any(alias.name == "*" for alias in node.names)
    ):
        return [((node.module or "").split(".", 1)[0], node.lineno)]
    return []


def is_initializer(node: ast.AST) -> bool:
    """Whether an import loads the selected torch build through the helper."""
    return is_plain_device_import(node) and any(
        alias.name == "torch" for alias in node.names
    )


def is_plain_device_import(node: ast.AST) -> bool:
    """Whether an import binds only unaliased ``torch`` and ``torch_fl``."""
    return (
        isinstance(node, ast.ImportFrom)
        and node.module == "device_init"
        and not node.level
        and bool(node.names)
        and all(
            alias.name in DEVICE_NAMES and alias.asname is None for alias in node.names
        )
    )


def initializers(tree: ast.AST) -> list[ast.ImportFrom]:
    """Every initializer, including one hidden in a function."""
    return [node for node in ast.walk(tree) if is_initializer(node)]


def _direct_bindings(node: ast.AST) -> list[int]:
    """Assignments that replace the initializer's torch or torch_fl binding."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return []
    return [
        node.lineno
        for target in targets
        if isinstance(target, ast.Name) and target.id in DEVICE_NAMES
    ]


def _check_nested_imports(node: ast.AST, initializer: ast.ImportFrom) -> None:
    """Reject a package import inside a function after module initialization."""
    for nested in ast.walk(node):
        if not isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        for item in ast.walk(nested):
            if item is nested or not isinstance(item, (ast.Import, ast.ImportFrom)):
                continue
            if package_imports(item):
                raise AssertionError(
                    "a function-local package import cannot see the shared "
                    f"initializer; found an import at line {item.lineno}"
                )


def _check_statement(
    statement: ast.AST,
    initializer: ast.ImportFrom | None,
    *,
    function_local: bool = False,
) -> None:
    """Check one statement before entering any function or class body."""
    if statement is initializer:
        return
    imports = package_imports(statement)
    if imports and function_local:
        raise AssertionError(
            "a function-local package import cannot see the shared "
            f"initializer; found an import at line {statement.lineno}"
        )
    if imports:
        location = ", ".join(f"{package} at line {line}" for package, line in imports)
        if initializer is None:
            raise AssertionError(
                "initialize the device with "
                f"`{DEVICE_IMPORT_WITH_FL.strip()}`, or `{DEVICE_IMPORT.strip()}` "
                f"when torch_fl is unused, before torch loads; found {location}"
            )
        if statement.lineno < initializer.lineno:
            raise AssertionError(
                "torch imports must follow the shared initializer at "
                f"line {initializer.lineno}; found {location}"
            )
        if isinstance(statement, (ast.Import, ast.ImportFrom)) and any(
            alias.name in DEVICE_NAMES for alias in statement.names
        ):
            raise AssertionError(
                "the shared device import is rebound directly at "
                f"line {statement.lineno}"
            )
    if function_local:
        return
    for line in _direct_bindings(statement):
        raise AssertionError(
            "the shared device import is rebound directly at "
            f"line {line}; use the initializer instead"
        )


def _check_scope(
    statements: list[ast.AST],
    initializer: ast.ImportFrom | None,
    *,
    function_local: bool = False,
) -> None:
    for statement in statements:
        _check_statement(statement, initializer, function_local=function_local)
        if function_local:
            continue
        if initializer is not None:
            _check_nested_imports(statement, initializer)
        for child in ast.iter_child_nodes(statement):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _check_scope(child.body, initializer, function_local=True)


def checked_source(source: str) -> ast.Module:
    """Parse source and require one module initializer before all torch imports."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AssertionError(
            f"cannot parse source at line {error.lineno}: {error.msg}"
        ) from error
    found = [node for node in tree.body if is_plain_device_import(node)]
    if len(found) > 1:
        lines = ", ".join(str(node.lineno) for node in found)
        raise AssertionError(
            f"initialize the device once; found imports at lines {lines}"
        )
    initializer = found[0] if found else None
    _check_scope(tree.body, initializer)
    return tree


def check_source(path: str | Path) -> None:
    target = Path(path).resolve()
    if target.name in ("device_init.py", "integration_rules.py"):
        return
    try:
        checked_source(target.read_text(encoding="utf-8"))
    except AssertionError as error:
        raise AssertionError(f"{target}: {error}") from error


def violations() -> dict[Path, str]:
    """Return each integration file that bypasses the shared initializer."""
    errors: dict[Path, str] = {}
    for path in sorted(TEST_ROOT.rglob("*.py")):
        if path.relative_to(TEST_ROOT).as_posix() in LAZY_IMPORT_ALLOWLIST:
            continue
        try:
            check_source(path)
        except AssertionError as error:
            errors[path] = str(error)
    return errors


def check_source_files() -> None:
    """Fail collection when integration files bypass shared initialization."""
    errors = violations()
    if errors:
        raise AssertionError(
            "integration tests must import the shared device initializer "
            "before torch:\n" + "\n".join(errors.values())
        )
