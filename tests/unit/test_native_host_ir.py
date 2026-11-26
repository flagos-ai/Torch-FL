# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile metadata-only host programs and compare their Python semantics."""

import builtins
import importlib.util
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FLAG = True
INITIALIZED_WIDTH = int("11")
RUNTIME_DEVICE = SimpleNamespace(name="vendor_device", vendor_name="test_vendor")
PUBLISHED_GEMS = SimpleNamespace(device=RUNTIME_DEVICE.name)
runtime_device = RUNTIME_DEVICE
_DEVICE_NAME = runtime_device.name


def published_device_alias(tensor):
    return tensor.device.type == PUBLISHED_GEMS.device


def published_mul_device_alias(tensor):
    return tensor.device.type == _DEVICE_NAME


ALLOWED_DTYPES = {torch.bfloat16, torch.float32, torch.int64}
_ordered_datatypes = [torch.float32, torch.bfloat16, torch.float16]
TRITON_LANGUAGE = None
DTYPE_KERNEL = None


class StridedBuffer:
    """Metadata interface of the published wrapper, without a device kernel."""

    def __init__(self, base, shape=None, strides=None, dtype=None, offset=0):
        self.base = base
        self.shape = tuple(base.shape if shape is None else shape)
        self.strides = tuple(base.stride() if strides is None else strides)
        self.offset = offset

    def numel(self):
        import math

        return math.prod(self.shape)

    def stride(self):
        return self.strides

    def data_ptr(self):
        return self.base.data_ptr() + self.offset * self.base.element_size()


StridedBuffer.__module__ = "flag_gems.utils.tensor_wrapper"


def tensor_buffer_metadata(value, offset):
    view = StridedBuffer(value, value.shape, value.stride(), offset=offset)
    return view.numel(), list(view.stride()), view.data_ptr()


def indexed_metadata_sort(values, lookup, reverse):
    return sorted(values, key=lambda value: lookup[value], reverse=reverse)


def compress_metadata_dimensions(value, dims):
    rank = value.ndim
    strides = value.stride()
    batch = [axis for axis in range(rank) if axis not in dims]
    reduction = sorted(dims, key=lambda axis: strides[axis], reverse=True)
    return value.permute(batch + reduction).contiguous()


def filtered_metadata_range(start, stop, step, excluded):
    axis = "outer binding"
    selected = [
        axis
        for axis in range(start, stop, step)
        if axis not in excluded
        if axis % 2 == 1
    ]
    return selected, axis


def set_runtime_flag(value):
    global RUNTIME_FLAG
    RUNTIME_FLAG = value


def runtime_global_host(value):
    if RUNTIME_FLAG:
        return value
    return value.view_as(value)


def initialized_global_host(value):
    return torch.empty([INITIALIZED_WIDTH], dtype=value.dtype, device=value.device)


def arange_start(device=None):
    if device is None:
        device = device.name
    return torch.empty([1], device=device)


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


class DeviceIndex(torch.nn.Module):
    def forward(self, device: torch.device) -> Optional[int]:
        return device.index


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


def variadic_dtype_predicate(*values):
    return any(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float64
        and value.ndim == 0
        for value in values
    )


def variadic_dtype_host(first, second, scalar):
    return variadic_dtype_predicate(first, second, scalar)


def qualified_extrema_host(first, second, third):
    return builtins.min(first, second, third), builtins.max(first, second, third)


def optional_device_index_host(requested):
    if requested is None:
        requested = torch.device("cuda:2")
    else:
        requested = torch.device(requested)
    return requested.index


def runtime_factory_device_host(requested):
    if requested is None:
        requested = torch.device(RUNTIME_DEVICE.name)
    return torch.empty([1], device=requested)


def dtype_membership_host(value):
    return value.dtype in ALLOWED_DTYPES, value.dtype not in ALLOWED_DTYPES


def dtype_itemsize_host(value):
    return torch.float32.itemsize, value.dtype.itemsize


def integer_dtype_metadata_host(value):
    return (
        value.numel() <= torch.iinfo(torch.int32).max,
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).bits,
    )


def triton_dtype_launch_host(value):
    output_dtype = TRITON_LANGUAGE.float32
    DTYPE_KERNEL[(1,)](value, OUTPUT_DTYPE=output_dtype)
    return value


def initialized_global_grid_host(value):
    elements = value.numel()
    DTYPE_KERNEL[
        lambda meta: (min(INITIALIZED_WIDTH, elements // meta["BLOCK_SIZE"]),)
    ](value)
    return value


def ordered_dtype_host(left, right):
    for dtype in _ordered_datatypes:
        if left.dtype == dtype:
            return True
        if right.dtype == dtype:
            return False
    raise ValueError("Unsupported dtype")


def get_sm_count():
    return 32


def get_device_capability():
    return (9, 0)


def device_query_host():
    return get_sm_count(), get_device_capability()


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
    DeviceIndex,
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
    schema.define("data_ptr(Tensor value) -> int")
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
        (
            "TensorBufferMetadata",
            tensor_buffer_metadata,
            "def f(value: torch.Tensor, offset: int) -> Tuple[int, List[int], int]: pass",
        ),
        (
            "IndexedMetadataSort",
            indexed_metadata_sort,
            "def f(values: List[int], lookup: List[int], reverse: bool) -> List[int]: pass",
        ),
        (
            "CompressMetadataDimensions",
            compress_metadata_dimensions,
            "def f(value: torch.Tensor, dims: List[int]) -> torch.Tensor: pass",
        ),
        (
            "FilteredMetadataRange",
            filtered_metadata_range,
            "def f(start: int, stop: int, step: int, excluded: List[int]) -> Tuple[List[int], str]: pass",
        ),
        (
            "VariadicDtype",
            variadic_dtype_host,
            "def f(first: torch.Tensor, second: torch.Tensor, scalar: float) -> bool: pass",
        ),
        (
            "QualifiedExtrema",
            qualified_extrema_host,
            "def f(first: int, second: int, third: int) -> Tuple[int, int]: pass",
        ),
        (
            "OptionalDeviceIndex",
            optional_device_index_host,
            "def f(requested: Optional[torch.device]) -> Optional[int]: pass",
        ),
        (
            "DtypeMembership",
            dtype_membership_host,
            "def f(value: torch.Tensor) -> Tuple[bool, bool]: pass",
        ),
        (
            "DtypeItemsize",
            dtype_itemsize_host,
            "def f(value: torch.Tensor) -> Tuple[int, int]: pass",
        ),
        (
            "IntegerDtypeMetadata",
            integer_dtype_metadata_host,
            "def f(value: torch.Tensor) -> Tuple[bool, int, int]: pass",
        ),
        (
            "OrderedDtype",
            ordered_dtype_host,
            "def f(left: torch.Tensor, right: torch.Tensor) -> bool: pass",
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
    grid_lookup = torch.jit.CompilationUnit("""
def lookup(meta: Dict[str, int], key: str, default: int) -> Tuple[Optional[int], int]:
    return meta.get(key), meta.get(key, default)
""").lookup
    pieces.append(ir.emit(grid_lookup.graph, function_name="GridLookup", binding=False))
    pieces.append("""
#include <torch/csrc/jit/python/pybind_utils.h>
namespace pybind11::detail {
template <> struct type_caster<HostIntList> : list_caster<HostIntList, int64_t> {};
}
template <typename R, typename... A>
void BindMetadataHost(py::module_& module, const char* name, R (*function)(A...)) {
  module.def(name, [function](A... args) {
    return torch::jit::toPyObject(c10::IValue(function(args...)));
  });
}
void BindGridLookup(py::module_& module) {
  module.def("grid_lookup", [](const std::string& key, int64_t fallback) {
    c10::Dict<std::string, int64_t> constants;
    constants.insert("BLOCK_M", 32);
    const std::vector<std::string> names{"BLOCK_M", "SPLIT_K", "STRIDE"};
    const c10::IValue args[]{16, 2, 5};
    const flagos::native_host::Keyword kwargs[]{{"BLOCK_M", 64}, {"SPLIT_K", false}};
    return GridLookup({&constants, &names, {args}, {kwargs}}, key, fallback);
  });
}
""")
    pieces.append(
        "PYBIND11_MODULE(native_host_ir_test, m) {"
        + "".join(f'm.def("{p.__name__}", &{p.__name__});' for p in PROGRAMS)
        + "".join(f'BindMetadataHost(m,"{name}",&{name});' for name in generated_names)
        + "BindGridLookup(m);"
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
            str(ROOT / "csrc/native_host"),
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


@pytest.mark.parametrize(
    "first,second",
    [
        (torch.ones(3), torch.tensor(1.0)),
        (torch.ones(3, dtype=torch.float64), torch.tensor(1.0)),
        (torch.ones(3), torch.tensor(1.0, dtype=torch.float64)),
        (torch.tensor(1.0, dtype=torch.float64), torch.ones(3)),
    ],
)
def test_variadic_metadata_predicate_keeps_all_arguments(compiled_hosts, first, second):
    assert compiled_hosts.VariadicDtype(first, second, 0.5) == variadic_dtype_host(
        first, second, 0.5
    )


def test_qualified_extrema_keeps_all_arguments(compiled_hosts):
    assert compiled_hosts.QualifiedExtrema(7, 3, 11) == (3, 11)


@pytest.mark.parametrize(
    "requested", [None, torch.device("cuda:3"), torch.device("cpu")]
)
def test_optional_device_refines_after_default_assignment(compiled_hosts, requested):
    assert compiled_hosts.OptionalDeviceIndex(requested) == optional_device_index_host(
        requested
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_dtype_membership_in_published_set(compiled_hosts, dtype):
    value = torch.empty(0, dtype=dtype)
    assert compiled_hosts.DtypeMembership(value) == dtype_membership_host(value)


def test_dtype_sequence_preserves_provider_order(compiled_hosts):
    high = torch.empty(0, dtype=torch.float32)
    low = torch.empty(0, dtype=torch.bfloat16)
    assert compiled_hosts.OrderedDtype(high, low) is True
    assert compiled_hosts.OrderedDtype(low, high) is False


@pytest.mark.parametrize("program", PROGRAMS[:4])
def test_shape_aliases_and_fresh_copies(compiled_hosts, program):
    value = torch.arange(12.0)
    actual = getattr(compiled_hosts, program.__name__)(value)
    expected = program()(value)
    assert actual.shape == expected.shape
    assert actual.data_ptr() == value.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("offset", [0, 1, -1])
def test_tensor_backed_buffer_keeps_relative_pointer_and_metadata(
    compiled_hosts, transpose, offset
):
    base = torch.arange(120.0).reshape(10, 12)
    value = base[1:5, 2:8]
    if transpose:
        value = value.t()
    actual = compiled_hosts.TensorBufferMetadata(value, offset)
    expected = tensor_buffer_metadata(value, offset)
    assert actual == expected
    assert actual[2] == value.data_ptr() + offset * value.element_size()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("values", [[], [3, 1, 2, 0], [0, -1, 1, 2], [2, 2, 0]])
def test_indexed_metadata_sort_preserves_ties_and_input(
    compiled_hosts, values, reverse
):
    lookup = [2, 1, 2, 1]
    original = values.copy()
    actual = compiled_hosts.IndexedMetadataSort(values, lookup, reverse)
    assert actual == indexed_metadata_sort(values, lookup, reverse)
    actual.append(17)
    assert values == original


@pytest.mark.parametrize("dims", [[2], [1, 2], [0, 2], [2, 0]])
@pytest.mark.parametrize("transpose", [False, True])
def test_dimension_compression_keeps_values_and_stride_order(
    compiled_hosts, dims, transpose
):
    value = torch.arange(3 * 5 * 7).reshape(3, 5, 7)
    if transpose:
        value = value.transpose(0, 2)
    actual = compiled_hosts.CompressMetadataDimensions(value, dims)
    expected = compress_metadata_dimensions(value, dims)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride() == expected.stride()


@pytest.mark.parametrize(
    "args",
    [(0, 7, 1, [1, 4]), (7, -4, -2, [3]), (0, 0, 1, []), (2, 9, 2, [4])],
)
def test_filtered_metadata_range_keeps_scope_and_filters(compiled_hosts, args):
    assert compiled_hosts.FilteredMetadataRange(*args) == filtered_metadata_range(*args)


def test_filtered_metadata_range_preserves_zero_step_error(compiled_hosts):
    with pytest.raises(ValueError):
        filtered_metadata_range(0, 7, 0, [])
    with pytest.raises(RuntimeError, match="zero"):
        compiled_hosts.FilteredMetadataRange(0, 7, 0, [])


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


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:3"])
def test_device_index_preserves_unspecified_and_zero_indices(compiled_hosts, device):
    device = torch.device(device)
    assert compiled_hosts.DeviceIndex(device) == device.index


def test_metax_arange_defaults_to_the_native_factory_device(
    compiled_hosts, monkeypatch
):
    package = sys.modules["test_native_host_compiler.frontend"]
    provider = "_metax.ops.arange"
    monkeypatch.setattr(arange_start, "__module__", provider)
    hosts = SimpleNamespace(
        originals={provider: sys.modules[__name__]}, resolve=lambda value: value
    )
    compiler = package.Frontend(hosts, SimpleNamespace())
    source = compiler.source(
        arange_start,
        "def f(device: Optional[torch.device]=None) -> torch.Tensor: pass",
    )
    assert not compiler.fallback_functions, compiler.fallback_reasons
    assert "device = torch.ops.flagos_source_native.factory_device()" in source


def test_factory_uses_the_bound_runtime_device_across_vendors(
    compiled_hosts, monkeypatch
):
    module = sys.modules["test_native_host_compiler.frontend"]
    monkeypatch.setitem(
        sys.modules, "flag_gems.runtime", SimpleNamespace(device=RUNTIME_DEVICE)
    )
    hosts = SimpleNamespace(
        manifest={"backend": {"vendor": RUNTIME_DEVICE.vendor_name}},
        originals={__name__: sys.modules[__name__]},
        resolve=lambda value: value,
    )
    compiler = module.Frontend(hosts, SimpleNamespace())
    source = compiler.source(
        runtime_factory_device_host,
        "def f(requested: Optional[torch.device]) -> torch.Tensor: pass",
    )
    assert not compiler.fallback_functions, compiler.fallback_reasons
    assert "requested = torch.ops.flagos_source_native.factory_device()" in source


@pytest.mark.parametrize(
    "function,module_name",
    [
        (published_device_alias, __name__),
        (published_mul_device_alias, "flag_gems.ops.mul"),
    ],
)
def test_published_device_alias_uses_the_matching_native_device(
    compiled_hosts, monkeypatch, function, module_name
):
    module = sys.modules["test_native_host_compiler.frontend"]
    monkeypatch.setitem(sys.modules, "flag_gems", PUBLISHED_GEMS)
    monkeypatch.setitem(
        sys.modules, "flag_gems.runtime", SimpleNamespace(device=RUNTIME_DEVICE)
    )
    monkeypatch.setattr(function, "__module__", module_name)
    hosts = SimpleNamespace(
        manifest={"backend": {"vendor": RUNTIME_DEVICE.vendor_name}},
        originals={module_name: sys.modules[__name__]},
        resolve=lambda value: value,
    )
    signature = "def f(tensor: torch.Tensor) -> bool: pass"
    launcher = SimpleNamespace(register_global=lambda *_args: 0)
    source = module.Frontend(hosts, launcher).source(function, signature)
    assert "tensor.device.type == 'flagos'" in source

    hosts.manifest["backend"]["vendor"] = "other_vendor"
    source = module.Frontend(hosts, launcher).source(function, signature)
    assert "tensor.device.type == 'flagos'" not in source


def test_device_queries_use_the_installed_adapter_and_published_binding(
    compiled_hosts, monkeypatch
):
    module = sys.modules["test_native_host_compiler.frontend"]
    device_info = SimpleNamespace(
        get_sm_count=get_sm_count, get_device_capability=get_device_capability
    )
    monkeypatch.setitem(sys.modules, "flag_gems.utils.device_info", device_info)
    ops = SimpleNamespace(get_sm_count=object(), compute_capability_major=object())
    monkeypatch.setattr(torch.ops, "flagos_source_native", ops)
    hosts = SimpleNamespace(
        originals={__name__: sys.modules[__name__]}, resolve=lambda value: value
    )
    signature = "def f() -> Tuple[int, Tuple[int, int]]: pass"

    compiler = module.Frontend(hosts, SimpleNamespace())
    source = compiler.source(device_query_host, signature)
    assert "torch.ops.flagos_source_native.get_sm_count()" in source
    assert "torch.ops.flagos_source_native.compute_capability_major()" not in source

    ops.compute_capability_minor = object()
    device_info.get_sm_count = lambda: 64
    compiler = module.Frontend(hosts, SimpleNamespace())
    source = compiler.source(device_query_host, signature)
    assert "torch.ops.flagos_source_native.get_sm_count()" not in source
    assert (
        "(torch.ops.flagos_source_native.compute_capability_major(), "
        "torch.ops.flagos_source_native.compute_capability_minor())" in source
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_dtype_itemsize_preserves_static_and_tensor_dtypes(compiled_hosts, dtype):
    value = torch.empty(1, dtype=dtype)
    assert compiled_hosts.DtypeItemsize(value) == dtype_itemsize_host(value)


def test_static_integer_dtype_metadata_matches_torch(compiled_hosts):
    value = torch.empty(1)
    assert compiled_hosts.IntegerDtypeMetadata(value) == integer_dtype_metadata_host(
        value
    )


def test_triton_dtype_lowers_as_a_kernel_specialization(compiled_hosts, monkeypatch):
    module = sys.modules["test_native_host_compiler.frontend"]
    language = ModuleType("triton.language")
    language.dtype = type("dtype", (), {})
    language.float32 = language.dtype()
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(globals(), "TRITON_LANGUAGE", language)
    kernel = SimpleNamespace(
        params=(
            SimpleNamespace(name="value", is_constexpr=False),
            SimpleNamespace(name="OUTPUT_DTYPE", is_constexpr=True),
        )
    )
    monkeypatch.setitem(globals(), "DTYPE_KERNEL", kernel)
    hosts = SimpleNamespace(
        originals={__name__: sys.modules[__name__]}, resolve=lambda value: value
    )
    launcher = SimpleNamespace(register_script_kernel=lambda value: 0)
    source = module.Frontend(hosts, launcher).source(
        triton_dtype_launch_host,
        "def f(value: torch.Tensor) -> torch.Tensor: pass",
    )
    assert "'OUTPUT_DTYPE': '__flagos_triton_dtype__:float32'" in source
    assert "output_dtype = " not in source


def test_grid_captures_bind_initialized_globals(compiled_hosts, monkeypatch):
    frontend = sys.modules["test_native_host_compiler.frontend"]
    kernel = SimpleNamespace(params=(SimpleNamespace(is_constexpr=False),))
    monkeypatch.setitem(globals(), "DTYPE_KERNEL", kernel)
    hosts = SimpleNamespace(
        originals={__name__: sys.modules[__name__]}, resolve=lambda value: value
    )
    globals_bound = []

    def register_global(module, name, kind):
        globals_bound.append((module, name, kind))
        return 0

    launcher = SimpleNamespace(
        register_script_kernel=lambda value: 0, register_global=register_global
    )
    compiler = frontend.Frontend(hosts, launcher)
    source = compiler.source(
        initialized_global_grid_host,
        "def f(value: torch.Tensor) -> torch.Tensor: pass",
    )
    assert not compiler.fallback_functions
    assert globals_bound == [(__name__, "INITIALIZED_WIDTH", "int")]
    assert "[torch.ops.flagos_script_host.global_int(0), elements]" in source


@pytest.mark.parametrize(
    "key,expected",
    [
        ("BLOCK_M", (32, 32)),
        ("SPLIT_K", (0, 0)),
        ("STRIDE", (5, 5)),
        ("absent", (None, 7)),
    ],
)
def test_grid_metadata_get_preserves_integer_values_and_default(
    compiled_hosts, key, expected
):
    assert compiled_hosts.grid_lookup(key, 7) == expected


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
