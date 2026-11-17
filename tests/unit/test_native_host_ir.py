# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile metadata-only host programs and compare their Python semantics."""

import importlib.util
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FLAG = True
INITIALIZED_WIDTH = int("11")


def set_runtime_flag(value):
    global RUNTIME_FLAG
    RUNTIME_FLAG = value


def runtime_global_host(value):
    if RUNTIME_FLAG:
        return value
    return value.view_as(value)


def initialized_global_host(value):
    return torch.empty([INITIALIZED_WIDTH], dtype=value.dtype, device=value.device)


def conditional_helper_host(value, flag):
    if flag:

        def width():
            return 2
    else:

        def width():
            return 3

    return torch.empty([width()], dtype=value.dtype)


def mutable_default_host(value):
    def width(items=[]):
        items.append(1)
        return len(items)

    return torch.empty([width()], dtype=value.dtype)


class ReadOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        copied = shape[:]
        return x.reshape(copied)


class Aliased(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [2, 3]
        alias = shape
        alias.append(2)
        return x.reshape(shape)


class Nested(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [2, 3]
        nested = [shape]
        nested[0].append(2)
        return x.reshape(shape)


class Copied(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [2, 3]
        copied = list(shape)
        shape.append(2)
        copied[0] = 4
        return x.reshape(copied)


class TupleIndex(torch.nn.Module):
    def forward(self, x: torch.Tensor, index: int) -> int:
        shape = (x.size(0), x.size(1))
        return shape[index]


class Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        return x is y


class Nonfinite(torch.nn.Module):
    def forward(self, index: int) -> float:
        if index == 0:
            return float("inf")
        elif index == 1:
            return -float("inf")
        return float("nan")


class MutatingCopy(torch.nn.Module):
    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return target.copy_(source)


class TensorAlpha(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y, alpha=3)


class ChannelsLast(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> bool:
        return x.is_contiguous(memory_format=torch.channels_last)


class TensorTruth(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> bool:
        return bool(x)


def appended_shape(value, width):
    return torch.empty((*value.shape, width), device=value.device, dtype=value.dtype)


def nested_shape(value, width=None):
    def normalize(item, default):
        if item is None:
            return default
        if isinstance(item, torch.Tensor):
            item = item.item()
        return int(item)

    width = normalize(width, value.numel())
    return torch.empty([width], device=value.device, dtype=value.dtype)


def scalar_rebind_host(value, flag):
    if flag:
        value = bool(value)
    return value


def optional_default_host(value):
    if value is None:
        value = 1.0
    return value


def optional_comparison_host(left, right):
    return left == right, left != right


def optional_comparison_effects(left, right):
    order = [0]

    def first(value, order):
        order.append(1)
        return value

    def second(value, order):
        order.append(2)
        return value

    equal = first(left, order) == second(right, order)
    return equal, order


def unsupported_dtype_helper(value, dtype):
    value = int(value)
    return torch.empty([value], dtype=dtype)


def optional_dtype_fallback_host(value, dtype):
    return unsupported_dtype_helper(value, dtype)


PROGRAMS = (
    ReadOnly,
    Aliased,
    Nested,
    Copied,
    TupleIndex,
    Identity,
    Nonfinite,
    MutatingCopy,
    TensorAlpha,
    ChannelsLast,
    TensorTruth,
)


@pytest.fixture(scope="module")
def compiled_hosts(tmp_path_factory):
    if not shutil.which("c++"):
        pytest.skip("metadata-host compiler validation requires a C++ compiler")
    from setuptools import Distribution, Extension

    directory = tmp_path_factory.mktemp("native-host-ir")
    package = "test_native_host_compiler"
    spec = importlib.util.spec_from_file_location(
        package,
        ROOT / "torch_fl/native_host/compiler/__init__.py",
        submodule_search_locations=[str(ROOT / "torch_fl/native_host/compiler")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    ir = importlib.import_module(package + ".ir")
    frontend = importlib.import_module(package + ".frontend")
    schema = torch.library.Library("flagos_script_host", "FRAGMENT")
    schema.define("global_int(int slot) -> int")
    pieces = [ir.HEADER]
    for program in PROGRAMS:
        graph = torch.jit.script(program()).forward.graph.copy()
        torch._C._jit_pass_inline(graph)
        pieces.append(ir.emit(graph, function_name=program.__name__, binding=False))
    hosts = SimpleNamespace(
        originals={__name__: sys.modules[__name__]}, resolve=lambda fn: fn
    )
    generated_names = []
    for name, function, signature in (
        (
            "AppendedShape",
            appended_shape,
            "def f(value: torch.Tensor, width: int) -> torch.Tensor: pass",
        ),
        (
            "NestedShape",
            nested_shape,
            "def f(value: torch.Tensor, width: Optional[int]=None) -> torch.Tensor: pass",
        ),
        (
            "InitializedShape",
            initialized_global_host,
            "def f(value: torch.Tensor) -> torch.Tensor: pass",
        ),
        (
            "ScalarRebind",
            scalar_rebind_host,
            "def f(value: float, flag: bool) -> Any: pass",
        ),
        (
            "OptionalDefault",
            optional_default_host,
            "def f(value: Optional[int]) -> Any: pass",
        ),
        (
            "OptionalIntCompare",
            optional_comparison_host,
            "def f(left: Optional[int], right: Optional[int]) -> Tuple[bool, bool]: pass",
        ),
        (
            "OptionalFloatCompare",
            optional_comparison_host,
            "def f(left: Optional[float], right: float) -> Tuple[bool, bool]: pass",
        ),
        (
            "OptionalStrCompare",
            optional_comparison_host,
            "def f(left: str, right: Optional[str]) -> Tuple[bool, bool]: pass",
        ),
        (
            "OptionalCompareEffects",
            optional_comparison_effects,
            "def f(left: Optional[int], right: int) -> Tuple[bool, List[int]]: pass",
        ),
    ):
        recorder = SimpleNamespace(
            register_global=lambda module, name, kind: 0,
            register_foreign=lambda *args: 0,
        )
        compiler = frontend.Frontend(hosts, recorder)
        generated = compiler.source(function, signature)
        assert not compiler.fallback_functions, compiler.fallback_reasons
        path = directory / (name + ".py")
        path.write_text(generated)
        spec = importlib.util.spec_from_file_location("native_shape_test_" + name, path)
        shape_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = shape_module
        spec.loader.exec_module(shape_module)
        graph = torch.jit.script(shape_module.Host()).forward.graph.copy()
        torch._C._jit_pass_inline(graph)
        pieces.append(ir.emit(graph, function_name=name, binding=False))
        generated_names.append(name)
    pieces.append("""
#include <torch/csrc/jit/python/pybind_utils.h>
template <typename R, typename... A>
void BindMetadataHost(py::module_& module, const char* name, R (*function)(A...)) {
  module.def(name, [function](A... args) {
    return torch::jit::toPyObject(c10::IValue(function(args...)));
  });
}
""")
    pieces.append(
        "PYBIND11_MODULE(native_host_ir_test, m) {"
        + "".join(f'm.def("{p.__name__}", &{p.__name__});' for p in PROGRAMS)
        + "".join(f'BindMetadataHost(m,"{name}",&{name});' for name in generated_names)
        + 'm.def("bind_global", [](int64_t value){module_globals={c10::IValue(value)};});}'
    )
    source = directory / "host.cpp"
    source.write_text("\n".join(pieces))
    torch_dir = Path(torch.__file__).parent
    extension = Extension(
        "native_host_ir_test",
        [str(source)],
        include_dirs=[
            str(torch_dir / "include"),
            str(torch_dir / "include/torch/csrc/api/include"),
            str(ROOT / "torch_fl/native_host/csrc"),
        ],
        library_dirs=[str(torch_dir / "lib")],
        libraries=["torch_python", "torch_cpu", "c10"],
        language="c++",
        extra_compile_args=[
            "-O1",
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        ],
    )
    distribution = Distribution({"ext_modules": [extension]})
    command = distribution.get_command_obj("build_ext")
    command.build_lib = str(directory)
    command.build_temp = str(directory / "objects")
    command.ensure_finalized()
    command.run()
    spec = importlib.util.spec_from_file_location(
        "native_host_ir_test", command.get_ext_fullpath(extension.name)
    )
    compiled = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled)
    return compiled


@pytest.mark.parametrize("program", PROGRAMS[:4])
def test_shape_aliases_and_fresh_copies(compiled_hosts, program):
    value = torch.arange(12.0)
    actual = getattr(compiled_hosts, program.__name__)(value)
    expected = program()(value)
    assert actual.shape == expected.shape
    assert actual.data_ptr() == value.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tuple_runtime_indices_and_bounds(compiled_hosts):
    value = torch.empty(3, 5)
    for index in (0, 1, -1, -2):
        assert compiled_hosts.TupleIndex(value, index) == TupleIndex()(value, index)
    for index in (2, -3):
        with pytest.raises(IndexError, match="tuple index out of range"):
            compiled_hosts.TupleIndex(value, index)


def test_tensor_identity_is_not_storage_or_value_equality(compiled_hosts):
    value = torch.arange(12.0)
    for other in (value, value.view_as(value), value.clone()):
        assert compiled_hosts.Identity(value, other) is (value is other)


def test_nonfinite_literals(compiled_hosts):
    import math

    assert compiled_hosts.Nonfinite(0) == float("inf")
    assert compiled_hosts.Nonfinite(1) == -float("inf")
    assert math.isnan(compiled_hosts.Nonfinite(2))


def test_inplace_typed_call_keeps_mutation_and_storage(compiled_hosts):
    target = torch.empty(3, 5)
    source = torch.arange(15.0).reshape(3, 5)
    result = compiled_hosts.MutatingCopy(target, source)
    assert result.data_ptr() == target.data_ptr()
    torch.testing.assert_close(target, source, rtol=0, atol=0)


def test_tensor_schema_keeps_nondefault_alpha(compiled_hosts):
    x, y = torch.arange(15.0), torch.ones(15)
    torch.testing.assert_close(
        compiled_hosts.TensorAlpha(x, y), x + 3 * y, rtol=0, atol=0
    )


def test_contiguity_keeps_the_requested_memory_format(compiled_hosts):
    value = torch.empty(2, 3, 5, 7)
    assert compiled_hosts.ChannelsLast(value) is False
    assert (
        compiled_hosts.ChannelsLast(value.contiguous(memory_format=torch.channels_last))
        is True
    )


def test_tensor_truth_keeps_scalar_validation(compiled_hosts):
    assert compiled_hosts.TensorTruth(torch.tensor(0)) is False
    assert compiled_hosts.TensorTruth(torch.tensor(2)) is True
    with pytest.raises(RuntimeError, match="more than one value"):
        compiled_hosts.TensorTruth(torch.ones(2))


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (0, 3), (2, 3, 5, 7)])
def test_starred_shapes_preserve_unseen_rank(compiled_hosts, shape):
    value = torch.empty(shape, dtype=torch.float64)
    actual = compiled_hosts.AppendedShape(value, 11)
    assert actual.shape == (*shape, 11)
    assert actual.dtype == value.dtype


@pytest.mark.parametrize("width", [None, 0, 4])
def test_local_helper_keeps_optional_refinement(compiled_hosts, width):
    value = torch.empty(2, 6)
    assert compiled_hosts.NestedShape(value, width).shape == (
        12 if width is None else width,
    )


@pytest.mark.parametrize(
    "function,signature",
    [
        (runtime_global_host, "def f(value: torch.Tensor) -> torch.Tensor: pass"),
        (
            conditional_helper_host,
            "def f(value: torch.Tensor, flag: bool) -> torch.Tensor: pass",
        ),
        (mutable_default_host, "def f(value: torch.Tensor) -> torch.Tensor: pass"),
        (
            optional_dtype_fallback_host,
            "def f(value: float, dtype: Optional[torch.dtype]) -> torch.Tensor: pass",
        ),
    ],
)
def test_runtime_state_retains_compiled_object_semantics(
    compiled_hosts, function, signature
):
    frontend = importlib.import_module("test_native_host_compiler.frontend")
    hosts = SimpleNamespace(
        originals={__name__: sys.modules[__name__]}, resolve=lambda fn: fn
    )
    recorder = SimpleNamespace(register_foreign=lambda fn, prefix: 0)
    compiler = frontend.Frontend(hosts, recorder)
    compiler.source(function, signature)
    assert function in compiler.fallback_functions


def test_initialized_globals_bind_without_recompilation(compiled_hosts):
    value = torch.empty(1)
    for width in (11, 23):
        compiled_hosts.bind_global(width)
        assert compiled_hosts.InitializedShape(value).shape == (width,)


@pytest.mark.parametrize("value", [0.0, -0.0, 3.5, float("inf"), float("nan")])
@pytest.mark.parametrize("flag", [False, True])
def test_compiled_scalar_binding_keeps_python_type(compiled_hosts, value, flag):
    actual = compiled_hosts.ScalarRebind(value, flag)
    expected = scalar_rebind_host(value, flag)
    assert type(actual) is type(expected)
    assert repr(actual) == repr(expected)


@pytest.mark.parametrize("value", [None, 0, 7, 2**63 - 1, -(2**63)])
def test_compiled_optional_default_does_not_widen_ints(compiled_hosts, value):
    actual = compiled_hosts.OptionalDefault(value)
    expected = optional_default_host(value)
    assert type(actual) is type(expected)
    assert actual == expected


@pytest.mark.parametrize(
    "left,right", [(None, None), (None, 3), (3, None), (3, 3), (2**63 - 1, 2**63 - 2)]
)
def test_compiled_optional_integer_comparison(compiled_hosts, left, right):
    assert compiled_hosts.OptionalIntCompare(left, right) == optional_comparison_host(
        left, right
    )


@pytest.mark.parametrize(
    "left,right",
    [
        (None, 0.0),
        (0.0, -0.0),
        (float("nan"), float("nan")),
        (float("inf"), float("inf")),
    ],
)
def test_compiled_optional_float_comparison(compiled_hosts, left, right):
    assert compiled_hosts.OptionalFloatCompare(left, right) == optional_comparison_host(
        left, right
    )


@pytest.mark.parametrize(
    "left,right", [("", None), ("", ""), ("a", "b"), ("\u03bb", "\u03bb")]
)
def test_compiled_optional_string_comparison(compiled_hosts, left, right):
    assert compiled_hosts.OptionalStrCompare(left, right) == optional_comparison_host(
        left, right
    )


@pytest.mark.parametrize("left,right", [(None, 3), (3, 3), (2, 3)])
def test_optional_comparison_evaluates_both_operands_once(compiled_hosts, left, right):
    assert compiled_hosts.OptionalCompareEffects(
        left, right
    ) == optional_comparison_effects(left, right)
