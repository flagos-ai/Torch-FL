# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
import types

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "torch_fl/native_host/build.py"
SPEC = importlib.util.spec_from_file_location("native_host_build", SOURCE)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


@pytest.fixture
def source_module(tmp_path, monkeypatch):
    def load(source, name="sample_host"):
        path = tmp_path / "sample_host.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    package = types.ModuleType("fg_aot")
    package.runtime = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "fg_aot", package)
    return load


def test_source_generation_keeps_unexecuted_branches_and_loops(source_module):
    original = source_module("""
calls = []
def host(size, strides):
    calls.append(size)
    if size == 0:
        return []
    if size < 0:
        raise ValueError('negative size')
    result = []
    for index in range(size):
        if index % 2:
            result.append(index * strides[0])
        else:
            result.append(index + strides[1])
    return result
""")
    source, record = build.source_module(original, "fg_aot")
    assert original.calls == []
    generated = {}
    exec(source, generated)
    generated["_bind"](lambda value: value)
    assert generated["host"](0, (2, 3)) == []
    assert generated["host"](5, (2, 3)) == [3, 2, 5, 6, 7]
    with pytest.raises(ValueError, match="negative size"):
        generated["host"](-1, (2, 3))
    assert len(record["sha256"]) == 64


def test_closure_source_preserves_wrapper_and_free_variables(source_module):
    original = source_module("""
import functools
def target(value):
    return value * 3
def decorate(function):
    cache = {}
    @functools.wraps(function)
    def wrapper(value):
        if value not in cache:
            cache[value] = function(value) + 7
        return cache[value]
    return wrapper
wrapped = decorate(target)
""")
    source = build.closure_source(original.wrapped, "sample_host", "wrapped")
    assert build.function_identity(original.wrapped) != build.function_identity(
        original.target
    )
    generated = {}
    exec(source, generated)
    generated["_bind"](lambda value: value)
    closure = generated["make"](
        {
            name: cell.cell_contents
            for name, cell in zip(
                original.wrapped.__code__.co_freevars, original.wrapped.__closure__
            )
        }
    )
    assert closure(5) == 22
    assert closure(8) == 31


def test_identical_source_does_not_touch_build_artifact(tmp_path):
    path = tmp_path / "host.pyx"
    build.write_changed(path, "def f(): return 1\n")
    before = path.stat().st_mtime_ns
    build.write_changed(path, "def f(): return 1\n")
    assert path.stat().st_mtime_ns == before
    build.write_changed(path, "def f(): return 2\n")
    assert path.read_text() == "def f(): return 2\n"


def test_dataclass_constructor_preserves_enum_default_and_factory(source_module):
    original = source_module("""
from dataclasses import dataclass, field
from enum import Enum
class Mode(Enum):
    DEFAULT = 1
@dataclass(frozen=True)
class Descriptor:
    shape: tuple
    mode: Mode = Mode.DEFAULT
    values: list = field(default_factory=list)
    def __post_init__(self):
        if len(self.shape) == 0:
            raise ValueError('empty shape')
""")
    source, _ = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    a, b = generated["Descriptor"]((3, 5)), generated["Descriptor"]((7,))
    assert isinstance(a, original.Descriptor)
    assert a.mode is original.Mode.DEFAULT
    a.values.append(9)
    assert b.values == []
    with pytest.raises(ValueError, match="empty shape"):
        generated["Descriptor"](())


def test_function_local_import_uses_prebuilt_binding(source_module):
    original = source_module("""
def host(value):
    from math import ceil
    return ceil(value)
""")
    source, _ = build.source_module(original, "fg_aot", {"math"})
    runtime = sys.modules["fg_aot"].runtime
    calls = []

    def binding(module, name):
        calls.append((module, name))
        return lambda value: 42

    runtime.import_attribute = binding
    generated = {}
    exec(source, generated)
    assert generated["host"](1.5) == 42
    assert calls == [("math", "ceil")]


def test_shared_host_class_keeps_dynamic_control_flow(source_module):
    original = source_module(
        """
class Schedule:
    def __init__(self, size):
        self.size = size
    @property
    def work(self):
        return self.size * 3
    def run(self):
        result = 0
        for i in range(self.work):
            result += i
        return result
def host(size):
    return Schedule(size).run()
""",
        "flag_gems.testing_source",
    )
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    assert "Schedule" in record["functions"]
    assert generated["host"](0) == 0
    assert generated["host"](5) == sum(range(15))
