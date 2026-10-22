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

"""Keep the shared MPS sweep importable when its environment has no NumPy.

The GCU and Kunlunxin lanes intentionally install no NumPy, but they select
``tests/integration/ops/`` and therefore import every MPS-only module there.
An MPS module with a module-level ``import numpy`` turned those sweeps into
collection errors before any operator ran. The platform-agnostic lane does not
install torch either, so this test checks the module sources rather than
importing the full integration package, which requires a built device runtime.
The same guard applies to optional PyTorch imports in unit tests that also run
static workflow and manifest checks in the no-torch lane.
"""

import ast
import io
import tokenize
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPS_DIR = _REPO_ROOT / "tests/integration/ops"
_UNIT_DIR = _REPO_ROOT / "tests/unit"
_SHARED_MPS_MODULES = ("test_mps_erf.py", "test_mps_sqrt.py", "test_mps_tanh.py")


def test_shared_mps_modules_do_not_import_numpy_without_guards():
    """All NumPy uses in these modules must be scoped to guarded test bodies."""
    failures = []
    for name in _SHARED_MPS_MODULES:
        path = _OPS_DIR / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if _is_numpy_import(node):
                failures.append(f"{name}:{node.lineno}: module-level NumPy import")
                continue
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            numpy_imports = [
                inner
                for inner in _walk_without_nested_callables(node)
                if _is_numpy_import(inner)
            ]
            for inner in numpy_imports:
                function_name = node.name
                if function_name.startswith("test_"):
                    continue
                skips = [
                    caller
                    for caller in ast.walk(node)
                    if isinstance(caller, ast.Name)
                    and caller.id in {"skip", "skipif", "importorskip"}
                ]
                if not skips:
                    failures.append(
                        f"{name}:{inner.lineno}: unguarded NumPy import in {function_name}"
                    )
    assert not failures, "\n".join(failures)


def _is_numpy_import(node):
    """Whether an AST node imports NumPy rather than a similarly named module."""
    if isinstance(node, ast.Import):
        return any(alias.name == "numpy" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and node.module == "numpy"
    return False


def _walk_without_nested_callables(function):
    """Yield nodes from one function body, excluding definitions inside it."""
    pending = list(function.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _is_torch_import(node):
    """Whether an AST node imports PyTorch rather than a similarly named module."""
    if isinstance(node, ast.Import):
        return any(alias.name == "torch" for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and node.module == "torch"
    return False


def _is_importorskip(node):
    """Whether an AST node is an ``importorskip('torch')`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "importorskip"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "torch"
    )


def _torch_import_guards(name):
    """Report ``import torch`` uses not preceded by ``importorskip``."""
    path = _UNIT_DIR / name
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    guarded_values = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _is_importorskip(node.value)
    }
    guarding_lines = {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME and token.string == "importorskip"
    }
    failures = []
    for node in ast.walk(tree):
        if not _is_torch_import(node) or id(node) in guarded_values:
            continue
        # Function-local imports run only after a guard appears in the file.
        if any(line < node.lineno for line in guarding_lines):
            continue
        failures.append(f"{name}:{node.lineno}: unguarded torch import")
    return failures


def test_integration_workdir_tests_import_torch_only_after_a_guard():
    """The static checks stay collectable in the no-torch lane."""
    name = "test_integration_workdir.py"
    failures = _torch_import_guards(name)
    assert not failures, "\n".join(failures)
    assert any(
        "importorskip" in line for line in (_UNIT_DIR / name).read_text().splitlines()
    ), "the optional PyTorch imports disappeared, making this guard vacuous"
