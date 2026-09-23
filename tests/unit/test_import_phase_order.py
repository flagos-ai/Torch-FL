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

"""Static check of the torch_fl import-time phase pipeline.

``torch_fl/__init__.py`` used to run its import side effects as statements
scattered through the module, with the order enforced only by where each one sat
and by comments. They now live in five named phases called from one runner. This
pins that runner: the phases, their order, and that nothing else executes at
module import.

Parsed from source rather than imported: ``torch_fl/__init__.py`` imports torch
and loads ``torch_fl._C``, which needs a built wheel and hardware, so this check
must not execute it. The order itself is what matters and is fully visible in
the AST.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INIT = REPO_ROOT / "torch_fl" / "__init__.py"

#: The pipeline, in the order the runner must call it.
PHASES = [
    "_phase_conf",
    "_phase_preload",
    "_phase_claim",
    "_phase_vendor_compat",
    "_phase_ecosystem",
]


def _tree() -> ast.Module:
    return ast.parse(INIT.read_text(encoding="utf-8"))


def test_runner_calls_the_phases_in_order():
    """The only module-level calls are the phases, in the documented order.

    Any other top-level call would be an import side effect that escaped the
    pipeline -- exactly the drift this refactor removes -- so the equality is
    deliberate, not a subset check.
    """
    tree = _tree()
    calls = [
        ast.unparse(node.value.func)
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert calls == PHASES


def test_every_phase_is_defined_before_the_runner():
    tree = _tree()
    defined = {
        node.name: node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in PHASES:
        assert name in defined, name
    runner_line = min(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) in PHASES
    )
    for name in PHASES:
        assert defined[name] < runner_line, name


def test_no_other_module_level_execution():
    """Everything that runs at import is an import, a constant, a def, or a phase.

    A bare statement (a loop, a try, a with) at module level would execute outside
    the pipeline; none should remain.
    """
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.ClassDef)
    for node in _tree().body:
        if isinstance(node, allowed):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            assert ast.unparse(node.value.func) in PHASES, ast.dump(node)
            continue
        raise AssertionError(
            f"unexpected module-level {type(node).__name__} at line {node.lineno}"
        )
