# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend contracts are inspectable without importing Torch or any SDK."""

from functools import cache
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from native_host_test_utils import import_native_host


ROOT = Path(__file__).resolve().parents[2]
NATIVE_SOURCE = ROOT / "csrc/native_host"
SOURCE = ROOT / "torch_fl/native_host/backend.py"
backend = import_native_host("backend")
driver_binding = import_native_host("backends.driver")


@pytest.fixture(params=sorted(backend._ADAPTERS))
def contract(tmp_path, monkeypatch, request):
    adapter = backend.get_backend(request.param)
    root = tmp_path / "native_host"
    shutil.copytree(SOURCE.parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    backend.stage_native_sources(adapter, NATIVE_SOURCE, root / "csrc")
    triton = tmp_path / "triton"
    for path in backend.driver_sources(adapter, triton).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// published backend source\n")
    monkeypatch.setattr(backend, "_ROOT", root)
    monkeypatch.setattr(backend.importlib.metadata, "version", lambda name: "0.7.0")
    record = backend.build_record(adapter, triton)
    compiled = {name: record[name] for name in backend.COMPILED_FIELDS}
    return adapter, triton, record, compiled


def test_backend_record_and_compiled_extension_agree(contract):
    adapter, root, record, compiled = contract
    backend.validate_record(record, adapter, root, compiled)
    assert isinstance(adapter, backend.BackendInterface)
    assert record["vendor"] == adapter.config.vendor
    assert record["compiler_backend"] == adapter.config.compiler_backend
    assert record["abi_version"] == 2
    assert record == backend.build_record(adapter, root, native_dir=NATIVE_SOURCE)
    selected = f"csrc/backends/{adapter.config.vendor}"
    assert f"{selected}/launch.cc" in record["native_sources"]
    assert f"{selected}/precision.cc" not in record["native_sources"]
    assert set(record["native_sources"]) == {
        "csrc/" + path.relative_to(backend._ROOT / "csrc").as_posix()
        for path in (backend._ROOT / "csrc").rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("vendor", ["unregistered", "gcu", "cuda", "../nvidia"])
def test_unvalidated_backend_never_inherits_a_cuda_shaped_abi(vendor):
    with pytest.raises(RuntimeError, match="no validated native-host launch adapter"):
        backend.get_backend(vendor)


def test_registered_backend_must_implement_the_adapter_interface(monkeypatch):
    name = backend.__package__ + ".backends.metax"
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(Backend=SimpleNamespace))
    backend.get_backend.cache_clear()
    with pytest.raises(
        RuntimeError, match="invalid native-host launch adapter for metax"
    ):
        backend.get_backend("metax")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("abi_version", 0),
        ("vendor", "unregistered"),
        ("compiler_backend", "foreign"),
        ("capabilities", 0),
        ("compiler_distribution", {"name": "flagtree", "version": "0.8.0"}),
    ],
)
def test_changed_contract_is_rejected(contract, field, value):
    adapter, root, record, compiled = contract
    record[field] = value
    with pytest.raises(RuntimeError, match="contract|ABI"):
        backend.validate_record(record, adapter, root, compiled)


@pytest.mark.parametrize("kind", ["driver", "adapter", "shared_binding", "native"])
def test_patched_loader_or_native_source_is_rejected(contract, kind):
    adapter, root, record, compiled = contract
    vendor = adapter.config.vendor
    paths = {
        "driver": next(iter(backend.driver_sources(adapter, root).values())),
        "adapter": backend._ROOT / "backends" / (vendor + ".py"),
        "shared_binding": backend._ROOT / "backends/driver.py",
        "native": backend._ROOT / "csrc/backends" / vendor / "launch.cc",
    }
    with paths[kind].open("a") as output:
        output.write("\n// changed\n")
    with pytest.raises(RuntimeError, match="ABI changed"):
        backend.validate_record(record, adapter, root, compiled)


def test_stale_compiled_backend_is_rejected_even_with_current_sources(contract):
    adapter, root, record, compiled = contract
    compiled["native_source_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="precompiled native-host backend ABI"):
        backend.validate_record(record, adapter, root, compiled)


def test_build_reuses_the_published_loader_and_tracks_header_dependencies(
    contract, monkeypatch, tmp_path
):
    adapter, root, record, _ = contract
    vendor = adapter.config.vendor
    if vendor == "hygon":
        sdk = tmp_path / "dtk/lib"
        sdk.mkdir(parents=True)
        (sdk / "libgalaxyhip.so").touch()
        monkeypatch.setenv("ROCM_PATH", str(sdk.parent))
        (root / "backends/hcu/driver.c").write_text(
            "typedef struct { unsigned words[12]; } TDMDescriptor;\n"
            "static bool encodeTDMDescriptor() { return true; }\n"
            "/*py_libhip_search_path*/\n/*py_hip_get_device_properties_sym*/ NULL\n"
        )
    torch = SimpleNamespace(
        __file__="/cpu-torch/torch/__init__.py",
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    source_dir = tmp_path / "build-sources"
    launcher, utilities = backend.build_extensions(
        adapter, "test_hosts", NATIVE_SOURCE, torch, record, root, source_dir
    )
    assert launcher.sources == [
        str(NATIVE_SOURCE / "launcher.cc"),
        str(NATIVE_SOURCE / "backends" / vendor / "launch.cc"),
    ]
    assert str(NATIVE_SOURCE) in launcher.include_dirs
    assert str(NATIVE_SOURCE / "backend_api.h") in launcher.depends
    for header in adapter.config.shared_headers:
        assert str(NATIVE_SOURCE / header) in launcher.depends
        assert f"csrc/{header}" in record["native_sources"]
    assert launcher.extra_compile_args == [
        "-O3",
        "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ]
    assert dict(launcher.define_macros)["FLAGOS_NATIVE_HOST_BACKEND_SHA256"] == (
        '"' + record["native_source_sha256"] + '"'
    )
    if vendor == "hygon":
        utility_path = Path(utilities.sources[0])
        assert utility_path.is_relative_to(source_dir)
        assert str(utility_path.with_name("hcu_descriptor.h")) in launcher.depends
        utility = utility_path.read_text()
        assert '"hipGetDeviceProperties_v2"' in utility
        assert "/*py_libhip_search_path*/" not in utility
        assert "cuda" not in launcher.libraries
        assert adapter.config.capabilities == 187
    elif vendor == "kunlunxin":
        driver = root / "backends/xpu"
        utility = Path(utilities.sources[0])
        assert utility.suffix == ".cc"
        assert utility.read_bytes() == (driver / "driver.c").read_bytes()
        assert launcher.extra_objects == [str(driver / "xpu3/lib/liblaunch.a")]
        assert {"launch_shared", "xpurt", "cudart"} <= set(launcher.libraries)
        assert "xpurt" in utilities.libraries
        assert adapter.config.capabilities == 129
    else:
        assert utilities.sources == [
            str(path)
            for path in backend.driver_sources(adapter, root).values()
            if path.suffix in {".c", ".cpp"}
        ]


@pytest.mark.parametrize("contract", ["thead"], indirect=True)
def test_ppu_extensions_use_the_published_hggc_driver(monkeypatch, contract, tmp_path):
    adapter, root, record, compiled = contract
    monkeypatch.setenv("PPU_SDK", "/published-ppu")
    monkeypatch.setenv("CUDA_HOME", "/unrelated-cuda")
    torch = SimpleNamespace(
        __file__="/cpu-torch/torch/__init__.py",
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    launcher, utilities = backend.build_extensions(
        adapter, "test_hosts", NATIVE_SOURCE, torch, record, root, tmp_path
    )
    assert "hggc" in launcher.libraries and "cuda" not in launcher.libraries
    assert utilities.libraries == ["hggc"]
    assert utilities.name == "test_hosts.hggc_utils"
    assert "/published-ppu/include" in launcher.include_dirs
    assert "/published-ppu/lib" in launcher.library_dirs
    assert record["compiler_backend"] == "ppu"
    assert compiled["capabilities"] == 253


@pytest.mark.parametrize("contract", ["nvidia", "thead"], indirect=True)
def test_shared_launch_parameters_are_part_of_the_backend_contract(contract):
    adapter, root, record, compiled = contract
    (backend._ROOT / "csrc/launch_params.h").write_text("// changed launch layout\n")
    with pytest.raises(RuntimeError, match="ABI changed"):
        backend.validate_record(record, adapter, root, compiled)


@pytest.mark.parametrize("contract", ["metax"], indirect=True)
def test_metax_metadata_packing_is_part_of_the_backend_contract(contract):
    adapter, root, record, compiled = contract
    (root / "backends/metax/compiler.py").write_text("# changed metadata ABI\n")
    with pytest.raises(RuntimeError, match="ABI changed"):
        backend.validate_record(record, adapter, root, compiled)


@pytest.mark.parametrize("contract", ["metax"], indirect=True)
def test_metax_extensions_use_the_native_runtime_without_cuda(
    monkeypatch, contract, tmp_path
):
    adapter, root, record, _ = contract
    monkeypatch.setenv("MACA_PATH", "/published-maca")
    monkeypatch.setenv("CUDA_HOME", "/unrelated-cuda")
    torch = SimpleNamespace(
        __file__="/cpu-torch/torch/__init__.py",
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    launcher, utilities = backend.build_extensions(
        adapter, "test_hosts", NATIVE_SOURCE, torch, record, root, tmp_path
    )
    assert "mcruntime" in launcher.libraries and "cuda" not in launcher.libraries
    assert "/published-maca/include" in launcher.include_dirs
    assert not any("/unrelated-cuda" in path for path in launcher.include_dirs)
    assert utilities.name == "test_hosts.maca_utils"
    assert adapter.config.capabilities == 141


@pytest.mark.parametrize(
    "contract", ["nvidia", "metax", "hygon", "thead"], indirect=True
)
def test_reinstallation_updates_existing_and_future_driver_streams(
    monkeypatch, contract
):
    adapter, _, _, _ = contract
    config = adapter.config
    driver = SimpleNamespace()
    driver_name, utils_name, launcher_name = config.driver_symbols

    class Driver:
        def __init__(self):
            self.utils = getattr(driver, utils_name)()
            self.launcher_cls = getattr(driver, launcher_name)
            self.get_current_stream = lambda device: -1

    setattr(driver, driver_name, Driver)
    active = object.__new__(Driver)
    default = object.__new__(Driver)
    monkeypatch.setitem(
        sys.modules,
        "triton.runtime",
        SimpleNamespace(driver=SimpleNamespace(_active=active, _default=default)),
    )
    exports = config.utility_exports
    modules = {f"triton.backends.{config.driver_directory}.driver": driver}
    for package in ("first", "second"):
        modules[f"{package}.{config.utility_module}"] = SimpleNamespace(
            **{name: object() for name in exports}
        )
    monkeypatch.setattr(
        driver_binding,
        "importlib",
        SimpleNamespace(import_module=modules.__getitem__),
    )
    for offset, package in ((100, "first"), (200, "second")):

        def provider(device, offset=offset):
            return device + offset

        kernel = object()
        driver_binding.install_driver(adapter, package, provider, kernel)
        utilities = modules[f"{package}.{config.utility_module}"]
        for instance in (active, default, Driver()):
            assert instance.utils is utilities
            assert instance.launcher_cls is kernel
            assert instance.get_current_stream(3) == offset + 3
        for name in exports:
            assert getattr(driver, name) is getattr(utilities, name)
        with pytest.raises(RuntimeError, match="compilation is forbidden"):
            driver.compile_module_from_src("unused")


def test_xpubin_uses_the_prebuilt_utility_and_normalized_launch_layout(monkeypatch):
    adapter = backend.get_backend("kunlunxin")
    driver = SimpleNamespace(
        XPUUtils=object,
        XPULauncher=object,
        get_xpu_spec=lambda arch, sdnn: (12, 8 if sdnn else 64),
        ty_to_cpp=lambda kind: (
            "void *"
            if kind.startswith("*")
            else {"i1": "int32_t", "i32": "int32_t", "i64": "int64_t", "bf16": "float"}[
                kind
            ]
        ),
    )

    @cache
    def utils():
        return driver.XPUUtils()

    class Driver:
        def __init__(self):
            self.utils = self.xpu_utils = utils()
            self.launcher_cls = self.launcher_cls_xpu = driver.XPULauncher

    driver.XPUDriver = Driver
    driver._xpu_utils_singleton = utils
    existing = Driver()
    monkeypatch.setitem(
        sys.modules,
        "triton.runtime",
        SimpleNamespace(driver=SimpleNamespace(_active=existing)),
    )
    utility = object()
    adapter.bind_driver(
        driver, utility, lambda src, meta: (src, meta), lambda d: d + 10
    )
    source = SimpleNamespace(
        fn=object(),
        signature={
            "value": "tensordesc<fp32[32,64]>",
            "scale": "bf16",
            "C": "constexpr",
        },
    )
    metadata = SimpleNamespace(xpu_arch=3, tensor_args=(), ewtable="", is_sdnn=False)
    for instance in (existing, Driver()):
        assert instance.xpu_utils is instance.utils is utility
        assert instance.launcher_cls_xpu is instance.launcher_cls
        assert instance.get_current_stream(2) == 12
        src, launch = instance.launcher_cls_xpu(source, metadata)
        assert src.fn is source.fn
        assert src.signature == {**source.signature, "scale": "fp32"}
        assert launch.tensordesc_meta == [2]
        assert launch._flagos_xpu_geometry == (12, 64)
        assert launch._flagos_xpu_parameters == [
            *((8, 0, False),) * 5,
            (4, 0, True),
            *((4, 0, False),) * 2,
            *((8, 0, False),) * 2,
            (4, 0, False),
        ]
    with pytest.raises(RuntimeError, match="launch-time quantization"):
        existing.launcher_cls_xpu(
            source, SimpleNamespace(**{**vars(metadata), "tensor_args": (0,)})
        )


def test_enflame_build_fingerprints_and_links_its_own_published_loader(
    tmp_path, monkeypatch
):
    adapter = backend.get_backend("enflame")
    assert (adapter.config.vendor, adapter.config.compiler_backend) == (
        "enflame",
        "gcu",
    )
    for path in backend.driver_sources(adapter, tmp_path).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("published backend source\n")
    monkeypatch.setattr(
        backend.importlib.metadata, "version", lambda name: "0.7.0+enflame3.6"
    )
    monkeypatch.setenv("TOPS_HOME", "/test/tops")
    record = backend.build_record(adapter, tmp_path)
    torch = SimpleNamespace(
        __file__="/cpu/torch/__init__.py",
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    launcher, utilities = backend.build_extensions(
        adapter, "hosts", NATIVE_SOURCE, torch, record, tmp_path, tmp_path / "sources"
    )
    assert "topsrt" in launcher.libraries
    assert "cuda" not in launcher.libraries
    assert "/test/tops/include" in launcher.include_dirs
    assert utilities.sources == [str(tmp_path / "backends/enflame/utils/gcu.cpp")]
    assert "backends/enflame/VERSION" in record["driver_sources"]
    backend.validate_record(record, adapter, tmp_path)
    (tmp_path / "backends/enflame/backend.py").write_text("changed ABI")
    with pytest.raises(RuntimeError, match="ABI changed"):
        backend.validate_record(record, adapter, tmp_path)


@pytest.mark.parametrize("pointers", [(0,), (2,), (-1,), (True,), (1,)])
def test_enflame_validates_integer_pointer_slots(pointers):
    module = import_native_host("backends.enflame")
    driver = SimpleNamespace(
        _triton_version=lambda: (3, 6), _expand_signature=lambda values: list(values)
    )
    source = SimpleNamespace(signature={0: "*fp32", 1: "i64"})
    metadata = SimpleNamespace(ptr_int_args=pointers, name="original_kernel")
    if pointers == (1,) and type(pointers[0]) is int:
        result = module.prepare(source, metadata, driver)
        assert result.name == "original_kernel" and result.ptr_int_args == (1,)
    else:
        with pytest.raises(ValueError, match="integer pointer"):
            module.prepare(source, metadata, driver)


def test_enflame_installer_preserves_current_stream_after_reinstallation(monkeypatch):
    import ctypes

    def enable(abi):
        assert abi == 2
        return 0

    monkeypatch.setattr(
        ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(FlagosEnflameEnableInterop=enable),
    )

    class Driver:
        def __init__(self):
            self.utils = module.GCUUtils()
            self.launcher_cls = module.GcuLauncher
            self.get_current_stream = lambda device: 0

    module = SimpleNamespace(GCUDriver=Driver)
    wrapped = SimpleNamespace(_driver=Driver.__new__(Driver))
    wrapper = SimpleNamespace(_GCUDriver=SimpleNamespace(instance=wrapped))
    active = Driver.__new__(Driver)
    runtime = SimpleNamespace(_active=active, _default=None)
    utilities = SimpleNamespace(load_binary=object(), get_device_properties=object())
    modules = {
        "torch_fl": SimpleNamespace(__file__="/installed/torch_fl/__init__.py"),
        "triton.backends.enflame.backend": module,
        "triton.backends.enflame.driver": wrapper,
        "hosts.gcu_utils": utilities,
    }
    adapter = backend.get_backend("enflame")
    loader = SimpleNamespace(import_module=modules.__getitem__)
    monkeypatch.setattr(driver_binding, "importlib", loader)
    monkeypatch.setattr(import_native_host("backends.enflame"), "importlib", loader)
    monkeypatch.setitem(sys.modules, "triton.runtime", SimpleNamespace(driver=runtime))
    driver_binding.install_driver(
        adapter,
        "hosts",
        lambda device: device + 100,
        object(),
    )
    first = Driver()
    assert first.get_current_stream(3) == active.get_current_stream(3) == 103
    assert first.get_current_stream() == 100
    driver_binding.install_driver(
        adapter,
        "hosts",
        lambda device: device + 200,
        object(),
    )
    assert Driver().get_current_stream(3) == active.get_current_stream(3) == 203
    assert first.get_current_stream(3) == 203
    assert wrapped.get_current_stream(3) == wrapped._driver.get_current_stream(3) == 203
    assert wrapped.utils is wrapped._driver.utils is utilities
    assert first.utils is utilities
    for function in (module.build_gcu_ext, module.compile_module_from_src):
        with pytest.raises(RuntimeError, match="compilation is forbidden"):
            function()


def test_backend_api_compiles_without_torch_or_a_vendor_sdk(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ conformance test requires a host compiler")
    source = tmp_path / "conformance.cc"
    source.write_text(r"""
#include "csrc/native_host/backend_api.h"
#include <cassert>
using namespace flagos::native_host::backend;
struct TestKernel : Kernel {
  void Enqueue(const Launch& value) const override {
    assert(value.grid[0] == 17 && value.stream == 123 && value.function == 456);
  }
};
std::unique_ptr<Kernel> KernelFactory(_object*) { return std::make_unique<TestKernel>(); }
bool Capturing(uint64_t stream) { return stream == 123; }
int main() {
  Api api{kAbiVersion, sizeof(Api), "conformance", "test", StreamCaptureQuery,
          64, &KernelFactory, nullptr, &Capturing};
  assert(Validate(api).is_capturing(123));
  auto kernel = api.kernel(nullptr);
  kernel->Enqueue({{17, 1, 1}, 123, 456, nullptr});
  Slot slots[2];
  slots[0].Set(uint64_t(0xffffffffffffffff));
  slots[1].Set(double(-1.5));
  uint64_t integer;
  double real;
  std::memcpy(&integer, slots[0].bytes.data(), sizeof(integer));
  std::memcpy(&real, slots[1].bytes.data(), sizeof(real));
  assert(integer == 0xffffffffffffffff && real == -1.5);
  assert(reinterpret_cast<uintptr_t>(&slots[0]) % kSlotAlignment == 0);
  bool rejected = false;
  api.abi_version = 0;
  try { Validate(api); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  api.abi_version = kAbiVersion;
  api.is_capturing = nullptr;
  rejected = false;
  try { Validate(api); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { CheckedProduct(UINT64_MAX, 2); } catch (const std::overflow_error&) { rejected = true; }
  assert(rejected);
}
""")
    binary = tmp_path / "conformance"
    result = subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True, timeout=10)
