# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Differential checks for binding normalization without importing Torch."""

import ast
import importlib.util
from pathlib import Path
from typing import Any, Optional

import pytest

PATH = Path(__file__).resolve().parents[2] / "torch_fl/native_host/compiler/bindings.py"
SPEC = importlib.util.spec_from_file_location("scalar_bindings", PATH)
bindings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bindings)


def programs(source, **globals_):
    scope = {"Optional": Optional, "Any": Any, **globals_}
    original = ast.parse(source).body[0]
    transformed = bindings.preserve_scalar_bindings(original, scope)
    before, after = dict(scope), dict(scope)
    exec(compile(ast.Module([original], []), "before.py", "exec"), before)
    exec(compile(ast.Module([transformed], []), "after.py", "exec"), after)
    return before[original.name], after[original.name], transformed


@pytest.mark.parametrize("value", [0.0, -0.0, 2.5, float("inf"), float("nan")])
@pytest.mark.parametrize("flag", [True, False])
def test_rebinding_keeps_value_type_aliases_and_single_execution(value, flag):
    source = """
def f(value: float, flag: bool, effects) -> Any:
    prior = value
    if flag:
        value = bool(value)
    effects.append(type(value).__name__)
    return prior, value
"""
    original, lowered, _ = programs(source)
    left, right = [], []
    expected, actual = original(value, flag, left), lowered(value, flag, right)
    assert left == right == ["bool" if flag else "float"]
    assert tuple(map(type, actual)) == tuple(map(type, expected))
    assert repr(actual) == repr(expected)


@pytest.mark.parametrize("value", [None, 0, 7, 2**63 - 1])
def test_optional_default_does_not_widen_present_integers(value):
    original, lowered, _ = programs("""
def f(value: Optional[int]) -> Any:
    if value is None:
        value = 1.0
    return value
""")
    assert type(lowered(value)) is type(original(value))
    assert lowered(value) == original(value)


@pytest.mark.parametrize(
    "first,second", [(True, True), (True, False), (False, True), (False, False)]
)
def test_nested_branches_keep_order_and_early_returns(first, second):
    original, lowered, _ = programs("""
def f(value: float, first: bool, second: bool, effects) -> Any:
    effects.append("before")
    if first:
        value = bool(value)
        if second:
            effects.append("early")
            return value
    else:
        if second:
            value = 3
    effects.append(type(value).__name__)
    return value
""")
    left, right = [], []
    expected = original(2.5, first, second, left)
    actual = lowered(2.5, first, second, right)
    assert type(actual) is type(expected) and actual == expected
    assert left == right


def test_fresh_names_do_not_shadow_user_locals():
    original, lowered, _ = programs("""
def f(value: float, flag: bool) -> Any:
    _host_binding_0 = 17
    if flag:
        value = bool(value)
    return _host_binding_0, value
""")
    assert lowered(2.0, True) == original(2.0, True)


def test_branch_local_numeric_types_do_not_merge():
    original, lowered, _ = programs("""
def f(flag: bool) -> Any:
    if flag:
        value = True
    else:
        value = 2.5
    return value
""")
    for flag in (True, False):
        assert type(lowered(flag)) is type(original(flag))
        assert lowered(flag) == original(flag)


@pytest.mark.parametrize(
    "statement",
    [
        "import math as _host_binding_0",
        "global _host_binding_0",
        "try:\n    raise ValueError()\nexcept ValueError as _host_binding_0:\n    pass",
    ],
)
def test_fresh_names_respect_non_name_bindings(statement):
    source = "def f(value: float) -> Any:\n    value = bool(value)\n"
    source += "\n".join("    " + line for line in statement.splitlines())
    source += "\n    return value\n"
    original, lowered, _ = programs(source)
    assert lowered(2.5) is original(2.5) is True
    assert "_host_binding_0" not in lowered.__globals__


@pytest.mark.parametrize(
    "body,reason",
    [
        ("for _ in range(2):\n    value = bool(value)", "loop"),
        ("def read():\n    return value\nvalue = bool(value)", "nested scope"),
        ("read = (value for _ in range(1))\nvalue = bool(value)", "nested scope"),
        (
            "value = bool(value)\nitems = [value for value in range(3)]",
            "comprehension scope",
        ),
        ("value = bool(value)\nreturn locals()", "namespace introspection"),
        ("value = bool(value) if value else value", "conditional expression"),
        ("try:\n    value = bool(value)\nfinally:\n    pass", "protected scope"),
        ("value = int(value)", "integer conversion"),
        ("value = bool(value)\nvalue = unknown()", "unknown value type"),
        ("value = bool(value)\nimport math as value", "import"),
        ("value = bool(value)\nfrom math import pi as value", "import"),
        ("value = bool(value)\ndef value():\n    return 1", "definition"),
        ("value = bool(value)\nclass value:\n    pass", "definition"),
    ],
)
def test_unsupported_binding_scopes_decline_without_running_source(body, reason):
    source = "def f(value: float):\n" + "\n".join(
        "    " + line for line in body.splitlines()
    )
    with pytest.raises(NotImplementedError, match=reason):
        bindings.preserve_scalar_bindings(ast.parse(source).body[0], {})


def test_continuation_budget_declines_expansion():
    source = "def f(value: float, flag: bool):\n"
    source += (
        "    if flag:\n        value = bool(value)\n    else:\n        value = 2.5\n"
        * 12
    )
    source += "    return value\n"
    with pytest.raises(NotImplementedError, match="source budget"):
        bindings.preserve_scalar_bindings(ast.parse(source).body[0], {})
