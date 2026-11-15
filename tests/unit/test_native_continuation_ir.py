# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute native prefixes and original exception continuations together."""

import importlib.util
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def continuation_host(tmp_path_factory):
    if not shutil.which("c++"):
        pytest.skip("native continuation validation requires a C++ compiler")
    from setuptools import Distribution, Extension

    directory = tmp_path_factory.mktemp("native-continuation")

    def load(name, path, package=False):
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[str(path.parent)] if package else None,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    build = load("continuation_ir_build", ROOT / "torch_fl/native_host/build.py")
    package = load(
        "continuation_compiler",
        ROOT / "torch_fl/native_host/compiler/__init__.py",
        True,
    )
    from importlib import import_module

    frontend = import_module(package.__name__ + ".frontend")
    ir = import_module(package.__name__ + ".ir")
    path = directory / "original.py"
    path.write_text("""import torch
import sys
api = sys.modules[__name__]

class Descriptor:
    @staticmethod
    def from_tensor(value):
        return value

def host(value, enabled, fail):
    descriptor = getattr(api, "Descriptor", None)
    enabled = enabled and descriptor is not None and hasattr(descriptor, "from_tensor")
    value.add_(1)
    if not enabled:
        return False
    try:
        if fail:
            raise ValueError("caught")
        if value.item() < 0:
            raise KeyError("uncaught")
    except ValueError:
        return False
    finally:
        value.add_(2)
    return True

def dtype_host(value, dtype):
    if value.numel() == 0:
        return value
    try:
        return torch.empty_like(value, dtype=dtype)
    except ValueError:
        return value

def metadata_host(value, enabled):
    if not enabled:
        return False
    try:
        count = value.numel()
    except RuntimeError as error:
        raise RuntimeError(str(error)) from error
    value.add_(count)
    return True
""")
    original = load("original_continuation_host", path)
    runtime = ModuleType("continuation_runtime")
    runtime.runtime = SimpleNamespace()
    sys.modules[runtime.__name__] = runtime
    generated_source, record = build.source_module(original, runtime.__name__)
    generated = ModuleType("compiled_continuation_host")
    exec(generated_source, generated.__dict__)
    generated._bind(lambda value: value)
    hosts = SimpleNamespace(
        originals={original.__name__: original},
        modules={original.__name__: generated},
        manifest={"modules": {original.__name__: record}},
        resolve=lambda value: value,
    )
    callbacks = []

    def register(function, prefix):
        assert not prefix
        callbacks.append(function)
        return len(callbacks) - 1

    compiler = frontend.Frontend(hosts, SimpleNamespace(register_foreign=register))
    source = compiler.source(
        original.host,
        "def f(value: torch.Tensor, enabled: bool, fail: bool) -> bool: pass",
    )
    assert not compiler.fallback_functions
    assert "getattr(" not in source and "hasattr(" not in source
    assert original.__name__ in compiler.constants.dependencies
    path = directory / "host.py"
    path.write_text(source)
    host = load("script_continuation_host", path)
    library = torch.library.Library("flagos_source_native", "FRAGMENT")
    library.define(
        "foreign_value(int slot, Any[] args, Dict(str, Any) kwargs) -> Any",
        alias_analysis="CONSERVATIVE",
    )
    graph = torch.jit.script(host.Host()).forward.graph.copy()
    torch._C._jit_pass_inline(graph)
    cpp = ir.HEADER + ir.emit(graph, function_name="Continuation", binding=False)
    cpp += """
#include <torch/csrc/jit/python/pybind_utils.h>
py::object& test_callbacks = *new py::object();
c10::IValue TestForeign(int64_t slot, flagos::native_host::Arguments args,
                       flagos::native_host::Keywords kwargs) {
  py::gil_scoped_acquire gil;
  TORCH_CHECK(kwargs.size()==0);
  py::tuple values(args.size());
  for(size_t i=0;i<args.size();++i) values[i]=torch::jit::toPyObject(args.get(i));
  auto result=test_callbacks[py::int_(slot)](*values);
  return torch::jit::toIValue(result,c10::BoolType::get());
}
PYBIND11_MODULE(continuation_ir_test, m) {
  m.def("call",&Continuation);
  m.def("bind",[](py::object values){test_callbacks=values;foreign_api=&TestForeign;});
}
"""
    path = directory / "host.cpp"
    path.write_text(cpp)
    torch_dir = Path(torch.__file__).parent
    extension = Extension(
        "continuation_ir_test",
        [str(path)],
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
    native = load(
        "continuation_ir_test", Path(command.get_ext_fullpath(extension.name))
    )
    yield native, callbacks, original, hosts
    del library


@pytest.mark.parametrize(
    "enabled,fail", [(False, False), (False, True), (True, False), (True, True)]
)
def test_native_guard_calls_only_the_reached_tail(continuation_host, enabled, fail):
    native, callbacks, original, _ = continuation_host
    called = []

    def callback(*args):
        called.append(True)
        return callbacks[0](*args)

    native.bind([callback])
    left, right = torch.tensor(0), torch.tensor(0)
    assert native.call(right, enabled, fail) is original.host(left, enabled, fail)
    assert called == ([True] if enabled else [])
    assert right.item() == left.item() == (3 if enabled else 1)


def test_tail_exception_keeps_effects_without_retry(continuation_host):
    native, callbacks, original, _ = continuation_host
    native.bind(callbacks)
    for call in (native.call, original.host):
        value = torch.tensor(-5)
        with pytest.raises(KeyError, match="uncaught"):
            call(value, True, False)
        assert value.item() == -2


def test_optional_dtype_cannot_cross_a_continuation(continuation_host):
    _, _, original, hosts = continuation_host
    from continuation_compiler.frontend import Frontend

    compiler = Frontend(hosts, SimpleNamespace(register_foreign=lambda *args: 0))
    compiler.source(
        original.dtype_host,
        "def f(value: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor: pass",
    )
    assert original.dtype_host in compiler.fallback_functions


def test_metadata_region_does_not_export_a_whole_tail(continuation_host, tmp_path):
    _, _, original, hosts = continuation_host
    from continuation_compiler.frontend import Frontend

    def unexpected(*args):
        pytest.fail("an already supported metadata region became an object callback")

    compiler = Frontend(hosts, SimpleNamespace(register_foreign=unexpected))
    source = compiler.source(
        original.metadata_host,
        "def f(value: torch.Tensor, enabled: bool) -> bool: pass",
    )
    path = tmp_path / "metadata_host.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("native_metadata_region", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scripted = torch.jit.script(module.Host())
    for enabled in (False, True):
        value, expected = torch.ones(3), torch.ones(3)
        assert scripted(value, enabled) is original.metadata_host(expected, enabled)
        torch.testing.assert_close(value, expected)
    assert not compiler.fallback_functions
