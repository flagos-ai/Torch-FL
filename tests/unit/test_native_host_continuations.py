# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Source continuations keep the original exception and return control flow."""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "continuation_source_build", ROOT / "torch_fl/native_host/build.py"
)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def programs(source):
    original = ast.parse(source).body[0]
    regions = build.continuation_regions(original)
    assert len(regions) == 1
    helper, record = regions[0]
    changed = copy.deepcopy(original)
    index = next(i for i, node in enumerate(changed.body) if isinstance(node, ast.Try))
    changed.body[index:] = [
        ast.Return(
            ast.Call(
                ast.Name(helper.name, ast.Load()),
                [ast.Name(name, ast.Load()) for name in record["inputs"]],
                [],
            )
        )
    ]
    before, after = {}, {}
    exec(compile(ast.Module([original], []), "original.py", "exec"), before)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module([helper, changed], [])),
            "continuation.py",
            "exec",
        ),
        after,
    )
    return before[original.name], after[original.name], record


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_guard_exception_finally_and_effect_order(enabled, fail):
    original, lowered, record = programs("""
def host(enabled, fail, effects):
    effects.append('prefix')
    if not enabled:
        return False
    value = 3
    try:
        effects.append('try')
        if fail:
            raise ValueError('failure')
        value = value + 4
    except ValueError:
        effects.append('except')
        return False
    finally:
        effects.append('finally')
    effects.append(value)
    return True
""")
    left, right = [], []
    assert lowered(enabled, fail, right) is original(enabled, fail, left)
    assert left == right
    assert record["inputs"] == ["effects", "fail", "value"]


def test_rebound_argument_and_return_alias_are_current():
    original, lowered, _ = programs("""
def host(value, enabled, effects):
    if not enabled:
        return value
    value = value + 7
    effects.append(value)
    try:
        effects.append(value + 1)
    except ValueError:
        raise
    return effects
""")
    left, right = [], []
    assert original(3, True, left) is left
    assert lowered(3, True, right) is right
    assert left == right == [10, 11]


def test_exception_propagates_without_repeating_the_prefix():
    original, lowered, _ = programs("""
def host(enabled, effects):
    effects.append('prefix')
    if not enabled:
        return False
    try:
        effects.append('try')
        raise KeyError('missing')
    finally:
        effects.append('finally')
    return True
""")
    for function in (original, lowered):
        effects = []
        with pytest.raises(KeyError, match="missing"):
            function(True, effects)
        assert effects == ["prefix", "try", "finally"]


@pytest.mark.parametrize(
    "body",
    [
        "if flag:\n    value = 3\ntry:\n    return value if flag else False\nexcept ValueError:\n    return False",
        "def read():\n    return flag\ntry:\n    return read()\nexcept ValueError:\n    return False",
        "try:\n    return locals()\nexcept ValueError:\n    return False",
        "global state\ntry:\n    state = flag\nexcept ValueError:\n    return False",
        "try:\n    return [v for v in (flag,)]\nexcept ValueError:\n    return False",
        "del flag\ntry:\n    return flag\nexcept ValueError:\n    return False",
    ],
)
def test_unsafe_scope_is_not_split(body):
    source = "def host(flag):\n    if flag is None:\n        return False\n"
    source += "\n".join("    " + line for line in body.splitlines())
    assert build.continuation_regions(ast.parse(source).body[0]) == []


def test_exception_only_root_is_not_an_artificial_native_facade():
    tree = ast.parse("""
def host(value):
    try:
        return bool(value)
    except ValueError:
        return False
""")
    assert build.continuation_regions(tree.body[0]) == []


def test_existing_metadata_region_keeps_the_remainder_native():
    tree = ast.parse("""
def host(value, enabled):
    if not enabled:
        return False
    try:
        count = value.numel()
    except RuntimeError as error:
        raise RuntimeError(str(error)) from error
    value.add_(count)
    return True
""")
    assert len(build.metadata_regions(tree.body[0])) == 1
    assert build.continuation_regions(tree.body[0]) == []
