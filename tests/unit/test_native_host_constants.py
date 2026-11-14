# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Constant folding may inspect pinned metadata, never execute host side effects."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "torch_fl/native_host/compiler/constants.py"
)
SPEC = importlib.util.spec_from_file_location("native_host_constants", SOURCE)
constants = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(constants)

pinned = types.ModuleType("pinned")
pinned.__version__ = "3.5.0+vendor"
runtime_flag = True
called = 0


def version_before(major, minor):
    version = pinned.__version__.split("+", 1)[0]
    parts = version.split(".")
    try:
        current = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return False
    return current < (major, minor)


def changing_global():
    return runtime_flag


def side_effect():
    global called
    called += 1
    return True


def catches_unknown():
    try:
        return side_effect()
    except Exception:
        return False


def module_api_exists():
    import pinned.core as core

    frontend_tuple = getattr(core, "tuple", None)
    return frontend_tuple is not None and hasattr(frontend_tuple, "__getitem__")


def evaluator(**modules):
    return constants.Evaluator(
        {__name__: sys.modules[__name__], "pinned": pinned, **modules}
    )


@pytest.mark.parametrize(
    "version,pair,result",
    [
        ("3.5.0+vendor", (3, 3), False),
        ("3.2.7", (3, 3), True),
        ("invalid", (3, 3), False),
    ],
)
def test_pinned_version(version, pair, result, monkeypatch):
    monkeypatch.setattr(pinned, "__version__", version)
    program = evaluator()
    assert program.call(version_before, pair, {}) == result
    assert program.dependencies == {__name__, "pinned"}


@pytest.mark.parametrize("function", [changing_global, side_effect, catches_unknown])
def test_runtime_state_is_not_executed(function):
    with pytest.raises(constants.NotConstant):
        evaluator().call(function, (), {})
    assert called == 0


def test_metadata_module_does_not_require_a_compiled_host():
    core = types.ModuleType("pinned.core")
    core.tuple = type(
        "tuple", (), {"__module__": "pinned.core", "__getitem__": object()}
    )
    program = evaluator(**{"pinned.core": core})
    assert program.call(module_api_exists, (), {}) is True
    assert "pinned.core" in program.dependencies
    del core.tuple
    assert program.call(module_api_exists, (), {}) is False
