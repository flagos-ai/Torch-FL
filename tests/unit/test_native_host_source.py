# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import importlib.util
import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import types
from contextlib import nullcontext

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "torch_fl/native_host/build.py"
SPEC = importlib.util.spec_from_file_location("native_host_build", SOURCE)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


@pytest.mark.parametrize("grid_kind", ["tuple", "callable"])
@pytest.mark.parametrize("entry_method", ["prepare", "run"])
def test_native_warmup_passes_serializable_grid_metadata(
    monkeypatch, grid_kind, entry_method
):
    expected = (3, 1, 1)
    launches = []
    prepared = False
    result = object()

    def prepare(*_):
        nonlocal prepared
        prepared = True

    def launch(*_):
        if not prepared:
            return b"compile"
        launches.append(result)
        return result

    cache = types.SimpleNamespace(
        missing_key=lambda *_: b"compile", prepare=prepare, run=launch
    )
    launcher = types.SimpleNamespace(KernelCache=lambda *_: cache)
    core = types.SimpleNamespace(_get_device=lambda: 0, _get_device_count=lambda: 1)
    monkeypatch.setitem(
        sys.modules, "fg_aot", types.SimpleNamespace(_launcher=launcher)
    )
    monkeypatch.setitem(sys.modules, "torch_fl", types.SimpleNamespace(_C=core))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(
        sys.modules,
        "triton.runtime",
        types.SimpleNamespace(
            Autotuner=type("Autotuner", (), {}), Heuristics=type("Heuristics", (), {})
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "native_runtime_test", SOURCE.with_name("runtime.py")
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    calls = []

    def compile_only(**options):
        assert options == {"grid": expected, "warmup": True}
        calls.append(json.dumps(options, default=vars))
        return types.SimpleNamespace(run=None, function=None, packed_metadata=None)

    jit = types.SimpleNamespace(
        arg_names=("GRID",),
        run=compile_only,
        params=(
            types.SimpleNamespace(
                name="GRID", default=expected, has_default=True, is_constexpr=True
            ),
        ),
    )
    entry = types.SimpleNamespace(
        arg_names=jit.arg_names, jit_function=jit, fn=jit, lock=nullcontext()
    )
    kernel = runtime.KernelEntry(entry, entry, compile_before_enqueue=True)
    grid = operator.itemgetter("GRID") if grid_kind == "callable" else expected
    if entry_method == "run":
        assert kernel.run(grid=grid) is result
        assert launches == [result]
    else:
        assert kernel.prepare_native(grid=grid)
        assert not launches
    assert len(calls) == 1


@pytest.mark.parametrize("accelerator", ["metax", "cuda", "ascend"])
def test_builder_selects_vendor_core_before_importing_torch(tmp_path, accelerator):
    """Exercise the real CLI entry point without loading accelerator libraries."""
    package = tmp_path / "torch_fl"
    backend = package / "native_host"
    backend.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import sys\n"
        "if 'torch' in sys.modules:\n"
        "    raise RuntimeError('vendor core must be selected before torch')\n"
    )
    (package / "_build_config.py").write_text(
        f"ACCELERATOR = {accelerator!r}\nKERNELS = ('flaggems',)\n"
    )
    (backend / "__init__.py").write_text("")
    (backend / "backend.py").write_text(
        "from pathlib import Path\n"
        "WHEEL_PACKAGE = 'torch_fl.native_host._generated'\n"
        "def build_extensions(*args):\n"
        "    raise AssertionError('startup fixture must not compile')\n"
        "stage_native_sources = build_extensions\n"
        "def build_record(*args, **kwargs):\n"
        "    raise AssertionError('startup fixture must not compile')\n"
        "def get_backend(accelerator):\n"
        "    (Path(__file__).parents[2] / 'selected-backend').write_text(accelerator)\n"
        "    raise SystemExit(0)\n"
    )
    (tmp_path / "torch.py").write_text(
        "import sys\n"
        "if 'torch_fl' not in sys.modules:\n"
        "    raise RuntimeError('torch loaded before the vendor bootstrap')\n"
    )
    (tmp_path / "triton.py").write_text("")
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE),
            "--output",
            str(tmp_path / "out"),
            "--launcher-only",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "selected-backend").read_text() == accelerator


@pytest.mark.parametrize("accelerator", ["cuda", "ascend", "metax"])
def test_builder_initializes_only_platforms_that_require_early_setup(
    tmp_path, accelerator
):
    package = tmp_path / "torch_fl"
    backend = package / "native_host"
    backend.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "def initialize():\n"
        "    (Path(__file__).parent / 'initialized').touch()\n"
        "flagos = SimpleNamespace(init=initialize)\n"
    )
    (package / "_build_config.py").write_text(
        f"ACCELERATOR = {accelerator!r}\nKERNELS = ('flaggems',)\n"
    )
    (backend / "__init__.py").write_text("")
    (backend / "vendor_cohort.py").write_text(
        (SOURCE.parent / "vendor_cohort.py").read_text()
    )
    (backend / "backend.py").write_text(
        "from types import SimpleNamespace\n"
        "WHEEL_PACKAGE = 'torch_fl.native_host._generated'\n"
        "def get_backend(accelerator):\n"
        "    import torch_fl\n"
        "    config = SimpleNamespace(vendor='nvidia' if accelerator == 'cuda' else accelerator)\n"
        "    result = SimpleNamespace(config=config, initialize_runtime=lambda: None)\n"
        "    if accelerator == 'ascend':\n"
        "        result.initialize_runtime = torch_fl.flagos.init\n"
        "    return result\n"
        "def build_record(*args, **kwargs):\n"
        f"    return dict(vendor={accelerator!r})\n"
        "def build_extensions(*args):\n"
        "    return []\n"
        "stage_native_sources = build_extensions\n"
    )
    (tmp_path / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__ = '2.10.0+cpu'\n"
        "version = SimpleNamespace(git_version='test-revision')\n"
        "def compiled_with_cxx11_abi(): return True\n"
    )
    (tmp_path / "triton.py").write_text("__version__ = '3.5.0'\n")
    (tmp_path / "flag_gems.py").write_text(
        "from pathlib import Path\n"
        "assert (Path(__file__).parent / 'torch_fl/initialized').exists() "
        f"== {accelerator == 'ascend'}, "
        "'platform initialization changed outside its backend hook'\n"
        "raise SystemExit(0)\n"
    )
    result = subprocess.run(
        [sys.executable, str(SOURCE), "--output", str(tmp_path / "out")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (package / "initialized").exists() == (accelerator == "ascend")


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


def test_source_manifest_records_published_line_endings(source_module):
    original = source_module("def host(value):\r\n    return value + 1\r\n")
    _, record = build.source_module(original, "fg_aot")
    published = Path(original.__file__).read_bytes()
    assert b"\r\n" in published
    assert record["sha256"] == hashlib.sha256(published).hexdigest()


def test_collect_modules_keeps_device_metadata_as_a_source_less_import(
    source_module, monkeypatch
):
    name = "flag_gems.runtime.backend._cambricon.utils"
    metadata = types.ModuleType(name)
    metadata.TOTAL_CORE_NUM = 48
    monkeypatch.setitem(sys.modules, name, metadata)
    flag_gems = types.ModuleType("flag_gems")
    flag_gems.runtime = types.SimpleNamespace(
        device=types.SimpleNamespace(vendor_name="cambricon")
    )
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    for module in ("flag_gems.utils.pointwise_dynamic", "triton"):
        monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    provider = source_module(f"""
def host():
    from {name} import TOTAL_CORE_NUM
    return TOTAL_CORE_NUM
""")
    modules = build.collect_modules([provider.host])
    assert modules == {provider.__name__: provider}
    compiled, _ = build.source_module(provider, "fg_aot", modules)
    assert f"from {name} import TOTAL_CORE_NUM" in compiled


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


def test_source_generation_keeps_provider_behind_a_compatibility_wrapper(source_module):
    provider = source_module("def host(value):\n    return value * 3\n")
    original = provider.host

    def wrapper(value):
        return original(value) + 7

    wrapper.__wrapped__ = original
    provider.host = wrapper
    source, record = build.source_module(provider, "fg_aot")
    generated = {}
    exec(source, generated)

    assert generated["host"](5) == 15
    assert provider.host(5) == 22
    assert record["functions"] == ["host"]


def test_identical_source_does_not_touch_build_artifact(tmp_path):
    path = tmp_path / "host.pyx"
    build.write_changed(path, "def f(): return 1\n")
    before = path.stat().st_mtime_ns
    build.write_changed(path, "def f(): return 1\n")
    assert path.stat().st_mtime_ns == before
    build.write_changed(path, "def f(): return 2\n")
    assert path.read_text() == "def f(): return 2\n"


def test_lambda_closure_retains_free_variables_and_defaults(source_module):
    original = source_module("""
def make(scale):
    return lambda value, offset=7: value * scale + offset
wrapped = make(3)
""")
    original.wrapped.__name__ = "wrapped"
    source = build.closure_source(original.wrapped, "sample_host", "wrapped")
    generated = {}
    exec(source, generated)
    generated["_bind"](lambda value: value)
    closure = generated["make"]({"scale": 3})
    assert closure(5) == 22
    assert closure(8, offset=2) == 26


def test_shadowed_device_definition_keeps_the_original_jit_alias(source_module):
    original = source_module("""
decorations = []
def device(function):
    decorations.append(function)
    return object()
@device
def copy_(value):
    return value
device_copy = copy_
def copy_(value):
    return device_copy
""")
    source, _ = build.source_module(original, "fg_aot")
    assert "@device" not in source
    assert source.count("def copy_(") == 1
    generated = {}
    exec(source, generated)
    generated["_bind"](lambda value: value)
    assert generated["copy_"](None) is original.device_copy
    assert len(original.decorations) == 1


def test_vendor_pointwise_cache_accepts_the_published_config_without_balance_grid():
    from types import SimpleNamespace

    path = SOURCE.with_name("runtime.py")
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if getattr(node, "name", None) == "pointwise_instantiate"
    )
    calls = []

    def adapter(owner, rank, kernel):
        calls.append((rank, kernel))
        return object()

    namespace = {
        "EnflamePointwiseOverload": adapter,
        "PointwiseOverload": adapter,
        "_kernel_factory": lambda owner, rank: rank,
    }
    exec(compile(ast.Module([function], []), str(path), "exec"), namespace)
    owner = SimpleNamespace(
        config=SimpleNamespace(prefer_block_pointer=True),
        overloads={},
        _native_pointwise_module="published.vendor.pointwise",
    )
    first = namespace["pointwise_instantiate"](owner, 1)
    assert namespace["pointwise_instantiate"](owner, 1) is first
    assert namespace["pointwise_instantiate"](owner, 3) is not first
    assert calls == [(1, 1), (3, 3)]


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


def test_metadata_exception_region_keeps_conversion_and_error(source_module):
    original = source_module("""
def host(dim):
    try:
        dim = list(dim)
    except TypeError:
        raise TypeError(f'invalid dimensions: {type(dim)}')
    return dim
""")
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    (region,) = record["regions"]
    helper = generated[region["name"]]
    assert region["inputs"] == region["outputs"] == ["dim"]
    dimensions = [1, 2]
    assert helper(dimensions) == dimensions
    assert helper(dimensions) is not dimensions
    with pytest.raises(TypeError, match="invalid dimensions"):
        helper(None)


@pytest.mark.parametrize(
    "expression,values,expected,effects",
    [
        ("any(check(x) for x in values)", [1, 0], True, [1]),
        ("all(check(x) for x in values)", [0, 1], False, [0]),
        ("next(x for x in values if check(x))", [0, 2, 3], 2, [0, 2]),
        ("sum(check(x) for x in values)", [0, 2, 3], 2, [0, 2, 3]),
        ("max(check(x) for x in values)", [0, 2, 3], True, [0, 2, 3]),
    ],
)
def test_iterator_regions_preserve_lazy_effect_order(
    source_module, expression, values, expected, effects
):
    original = source_module(
        "effects = []\n"
        "def check(value):\n    effects.append(value)\n    return bool(value)\n"
        f"def host(values):\n    return {expression}\n"
    )
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    generated["_bind"](lambda value: value)
    (region,) = [r for r in record["regions"] if r.get("kind") == "expression"]
    assert region["inputs"] == ["values"]
    assert generated[region["name"]](values) == expected
    assert original.effects == effects


@pytest.mark.parametrize(
    "consumer,error", [("next", StopIteration), ("max", ValueError)]
)
def test_iterator_regions_preserve_empty_errors(source_module, consumer, error):
    original = source_module(
        f"def host(values):\n    return {consumer}(x for x in values)\n"
    )
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    (region,) = record["regions"]
    with pytest.raises(error):
        generated[region["name"]]([])


def test_iterator_scope_keeps_outer_iterable_and_nested_captures(source_module):
    original = source_module("""
def host(values, offset):
    return sum(values + inner + offset for values in values for inner in [values])
""")
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    (region,) = record["regions"]
    assert region["inputs"] == ["offset", "values"]
    assert generated[region["name"]](3, [1, 2]) == original.host([1, 2], 3) == 12


def test_object_loop_keeps_live_inputs_and_original_exception(source_module):
    original = source_module("""
class Buffer:
    def __init__(self, value):
        if value < 0:
            raise ValueError('negative buffer')
        self.value = value
def host(values, output):
    for value in values:
        buffer = Buffer(value)
        output.append(buffer.value)
    return output
""")
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    generated["_bind"](lambda value: value)
    (region,) = [r for r in record["regions"] if r.get("kind") == "object_loop"]
    assert region["inputs"] == ["output", "values"]
    output = []
    assert generated[region["name"]](output, [2, 3]) is None
    assert output == [2, 3]
    with pytest.raises(ValueError, match="negative buffer"):
        generated[region["name"]](output, [4, -1, 5])
    assert output == [2, 3, 4]


@pytest.mark.parametrize(
    "body",
    [
        "for value in values:\n    result = value\nreturn result",
        "for value in values:\n    output[0] = value\nreturn output",
        "for value in values:\n    if value:\n        break\nreturn output",
        "for value in values:\n    return value\nreturn output",
        "for value in values:\n    output.append(locals())\nreturn output",
    ],
)
def test_object_loop_refuses_escaping_bindings_or_control(source_module, body):
    source = "def host(values, output):\n" + "\n".join(
        "    " + s for s in body.splitlines()
    )
    original = source_module(source)
    _, record = build.source_module(original, "fg_aot")
    assert not [r for r in record["regions"] if r.get("kind") == "object_loop"]


def test_typed_iterator_regions_lower_without_root_fallback(source_module, tmp_path):
    torch = pytest.importorskip("torch")
    original = source_module("""
import itertools
effects = []
class MetadataView:
    def __init__(self, value):
        self.rank = value.ndim
def host(values, dim):
    shapes = [list(value.shape) for value in values]
    rank = max(value.ndim for value in values)
    mismatch = any(value.ndim != rank for value in values)
    uniform = all(value.ndim == rank for value in values)
    ref = next(value for value in values if value.ndim == rank)
    smallest = min(value.shape[0] for value in values if value.ndim == rank)
    total = sum(shape[dim] for shape in shapes)
    offsets = list(itertools.accumulate([shape[dim] for shape in shapes], initial=0))
    for value in values:
        view = MetadataView(value)
        effects.append(view.rank)
    return rank, mismatch, ref, total, offsets, uniform, smallest
""")
    original.MetadataView.__module__ = "flag_gems.metadata_fixture"
    source, record = build.source_module(original, "fg_aot")
    compiled = {}
    exec(source, compiled)
    spec = importlib.util.spec_from_file_location(
        "test_iterator_compiler",
        SOURCE.parent / "compiler/__init__.py",
        submodule_search_locations=[str(SOURCE.parent / "compiler")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    frontend = __import__(spec.name + ".frontend", fromlist=["Frontend"])
    hosts = types.SimpleNamespace(
        originals={original.__name__: original},
        manifest={"modules": {original.__name__: record}},
        modules={original.__name__: types.SimpleNamespace(**compiled)},
        resolve=lambda fn: fn,
    )
    helpers = []

    def register(function, prefix):
        helpers.append(function)
        return len(helpers) - 1

    compiler = frontend.Frontend(
        hosts, types.SimpleNamespace(register_foreign=register)
    )
    generated = compiler.source(
        original.host,
        "def f(values: List[torch.Tensor], dim: int) -> Tuple[int, bool, torch.Tensor, int, List[int], bool, int]: pass",
    )
    assert not compiler.fallback_functions, compiler.fallback_reasons
    assert len(helpers) == 7
    schema = torch.library.Library("flagos_source_native", "FRAGMENT")
    schema.define(
        "foreign_value(int slot, Any[] args, Dict(str, Any) kwargs) -> Any",
        alias_analysis="CONSERVATIVE",
    )
    executed = []

    def invoke(slot, args, kwargs):
        executed.append(slot)
        return helpers[slot](*args, **kwargs)

    schema.impl(
        "foreign_value",
        invoke,
        "CompositeExplicitAutograd",
    )
    path = tmp_path / "iterator_host.py"
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location("test_iterator_host", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    scripted = torch.jit.script(module.Host())
    for values in (
        [torch.empty(2, 3), torch.empty(4, 3)],
        [torch.empty(2, 1, 3)],
        [torch.empty(2, 3), torch.empty(0, 3), torch.empty(4)],
    ):
        original.effects.clear()
        executed.clear()
        actual = scripted(values, 0)
        # Only the opaque class loop needs Cython; scalar generator reductions
        # and cumulative offsets execute directly in the generated typed host.
        assert executed == [6]
        assert original.effects == [value.ndim for value in values]
        expected = original.host(values, 0)
        assert actual[:2] == expected[:2]
        assert actual[2] is expected[2]
        assert actual[3:] == expected[3:]
    with pytest.raises(Exception, match="empty"):
        scripted([], 0)


def test_local_callback_region_retains_each_calls_capture(source_module):
    original = source_module("""
callbacks = []
def install(callback):
    callbacks.append(callback)
def host(value):
    if value is not None:
        def callback(scale, value=value):
            return value * scale
        install(callback)
    return value
""")
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    generated["_bind"](lambda value: value)
    (region,) = [r for r in record["regions"] if r.get("kind") == "closure"]
    assert region["inputs"] == ["value"]
    helper = generated[region["name"]]
    helper(3)
    helper(7)
    assert [callback(5) for callback in original.callbacks] == [15, 35]


def test_local_callback_region_refuses_nonlocal_mutation(source_module):
    original = source_module("""
def install(callback):
    callback()
def host(value):
    def callback():
        nonlocal value
        value += 1
    install(callback)
    return value
""")
    _, record = build.source_module(original, "fg_aot")
    assert not [r for r in record["regions"] if r.get("kind") == "closure"]


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


@pytest.mark.parametrize("wheel_origin", [False, True])
def test_vendor_alias_class_binds_original_kernel_from_its_wheel(
    source_module, tmp_path, wheel_origin
):
    original = source_module(
        """
from functools import partial
from operator import mul
kernel = partial(mul, 3)
class Helper:
    @staticmethod
    def forward(size):
        return kernel(size)
def host(size):
    return Helper.forward(size)
""",
        "_vendor.ops.example",
    )
    if wheel_origin:
        path = tmp_path / "flag_gems/runtime/backend/_vendor/ops/example.py"
        path.parent.mkdir(parents=True)
        path.write_text(Path(original.__file__).read_text())
        original.__file__ = str(path)
    source, record = build.source_module(original, "fg_aot")
    generated = {}
    exec(source, generated)
    calls = []

    def resolved_kernel(size):
        calls.append(size)
        return original.kernel(size)

    generated["_bind"](
        lambda value: resolved_kernel if value is original.kernel else value
    )
    assert generated["host"](0) == 0
    assert generated["host"](7) == 21
    assert ("Helper" in record["functions"]) == wheel_origin
    assert calls == ([0, 7] if wheel_origin else [])
