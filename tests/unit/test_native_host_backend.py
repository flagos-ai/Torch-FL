# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend contracts are inspectable without importing Torch or any SDK."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "torch_fl/native_host/backend.py"
SPEC = importlib.util.spec_from_file_location("native_backend_under_test", SOURCE)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


@pytest.fixture(params=tuple(backend._ADAPTERS))
def contract(tmp_path, monkeypatch, request):
    adapter = backend.get_backend(request.param)
    root = tmp_path / "native_host"
    shutil.copytree(SOURCE.parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    driver = tmp_path / "triton/backends" / adapter.vendor
    for path in adapter.driver_sources(driver.parents[1]).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// published backend source\n")
    monkeypatch.setattr(backend, "_ROOT", root)
    monkeypatch.setattr(backend.importlib.metadata, "version", lambda name: "0.7.0")
    record = backend.build_record(adapter, driver.parents[1])
    compiled = {name: record[name] for name in backend.COMPILED_FIELDS}
    return adapter, driver.parents[1], record, compiled


def test_backend_record_and_compiled_extension_agree(contract):
    adapter, root, record, compiled = contract
    backend.validate_record(record, adapter, root, compiled)
    assert record["vendor"] == adapter.vendor
    assert record["compiler_backend"] == adapter.compiler_backend


@pytest.mark.parametrize("vendor", ["unregistered", "cuda", "../nvidia"])
def test_unvalidated_backend_never_inherits_a_cuda_shaped_abi(vendor):
    with pytest.raises(RuntimeError, match="no validated native-host launch adapter"):
        backend.get_backend(vendor)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("abi_version", 0),
        ("vendor", "unregistered"),
        ("compiler_backend", "hip"),
        ("capabilities", 0),
        ("compiler_distribution", {"name": "flagtree", "version": "0.8.0"}),
    ],
)
def test_changed_contract_is_rejected(contract, field, value):
    adapter, root, record, compiled = contract
    record[field] = value
    with pytest.raises(RuntimeError, match="contract|ABI"):
        backend.validate_record(record, adapter, root, compiled)


@pytest.mark.parametrize("kind", ["driver", "adapter", "native"])
def test_patched_loader_or_native_source_is_rejected(contract, kind):
    adapter, root, record, compiled = contract
    paths = {
        "driver": root / "backends" / adapter.vendor / "driver.py",
        "adapter": backend._ROOT / "backends" / (adapter.vendor + ".py"),
        "native": backend._ROOT / adapter.sources[0],
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


def test_build_reuses_the_published_loader_and_tracks_header_dependencies(contract):
    adapter, root, record, _ = contract
    torch = SimpleNamespace(
        __file__="/cpu-torch/torch/__init__.py",
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    launcher, utilities = adapter.extensions(
        "test_hosts", backend._ROOT, torch, record, root
    )
    assert str(backend._ROOT / adapter.sources[0]) in launcher.sources
    assert str(backend._ROOT / "csrc/backend_api.h") in launcher.depends
    assert utilities.sources == [
        str(path)
        for path in adapter.driver_sources(root).values()
        if path.suffix in {".c", ".cpp"}
    ]


@pytest.mark.parametrize("contract", ["metax"], indirect=True)
def test_metax_metadata_packing_is_part_of_the_backend_contract(contract):
    adapter, root, record, compiled = contract
    (root / "backends/metax/compiler.py").write_text("# changed metadata ABI\n")
    with pytest.raises(RuntimeError, match="ABI changed"):
        backend.validate_record(record, adapter, root, compiled)


@pytest.mark.parametrize("contract", ["metax"], indirect=True)
def test_metax_extensions_use_the_native_runtime_without_cuda(monkeypatch, contract):
    adapter, root, record, _ = contract
    monkeypatch.setenv("MACA_PATH", "/published-maca")
    monkeypatch.setenv("CUDA_HOME", "/unrelated-cuda")
    torch = SimpleNamespace(
        __file__="/cpu-torch/torch/__init__.py",
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    launcher, utilities = adapter.extensions(
        "test_hosts", backend._ROOT, torch, record, root
    )
    assert "mcruntime" in launcher.libraries and "cuda" not in launcher.libraries
    assert "/published-maca/include" in launcher.include_dirs
    assert not any("/unrelated-cuda" in path for path in launcher.include_dirs)
    assert utilities.name == "test_hosts.maca_utils"
    assert adapter.capabilities == 141


@pytest.mark.parametrize("contract", ["metax"], indirect=True)
def test_metax_reinstallation_updates_existing_and_future_driver_streams(
    monkeypatch, contract
):
    adapter, _, _, _ = contract
    driver = SimpleNamespace()

    class Driver:
        def __init__(self):
            self.utils = driver.MacaUtils()
            self.launcher_cls = driver.MacaLauncher
            self.get_current_stream = lambda device: -1

    driver.MacaDriver = Driver
    active = object.__new__(Driver)
    default = object.__new__(Driver)
    monkeypatch.setitem(
        sys.modules,
        "triton.runtime",
        SimpleNamespace(driver=SimpleNamespace(_active=active, _default=default)),
    )
    modules = {
        "triton.backends.metax.driver": driver,
        "first.maca_utils": object(),
        "second.maca_utils": object(),
    }
    monkeypatch.setitem(
        adapter.install.__globals__,
        "importlib",
        SimpleNamespace(import_module=modules.__getitem__),
    )
    for offset, package in ((100, "first"), (200, "second")):

        def provider(device, offset=offset):
            return device + offset

        kernel = object()
        adapter.install(package, SimpleNamespace(current_stream=provider), kernel)
        for instance in (active, default, Driver()):
            assert instance.utils is modules[package + ".maca_utils"]
            assert instance.launcher_cls is kernel
            assert instance.get_current_stream(3) == offset + 3
        with pytest.raises(RuntimeError, match="compilation is forbidden"):
            driver.compile_module_from_src("unused")


def test_backend_api_compiles_without_torch_or_a_vendor_sdk(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ conformance test requires a host compiler")
    source = tmp_path / "conformance.cc"
    source.write_text(r"""
#include "torch_fl/native_host/csrc/backend_api.h"
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
