# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Admission tests for original MetaX GPU stubs on the official CPU core."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def published(tmp_path):
    cohort = load("metax_vendor_cohort_test", "torch_fl/native_host/vendor_cohort.py")
    policy = cohort.load_build_policy(ROOT).precision_policy("metax")
    library = tmp_path / "torch/lib"
    library.mkdir(parents=True)
    (library.parent / "version.py").write_text(
        f"__version__ = {policy['vendor_torch']!r}\n"
        f"git_version = {policy['torch_git_version']!r}\ncuda = '11.6'\n"
    )
    for name in cohort.METAX_LIBRARIES:
        (library / name).write_bytes(name.encode())
    for relative in policy["host_headers"]:
        header = library.parent / "include" / relative
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("original published environment implementation")
        policy["host_headers"][relative] = cohort.digest(header)
    info = tmp_path / "torch-metax.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: torch\nVersion: {policy['vendor_torch']}\n")
    (info / "licenses").mkdir()
    (info / "licenses/LICENSE").write_text("published license")
    return cohort, policy, library


def test_metax_bundle_contains_only_gpu_libraries_and_license(published, tmp_path):
    cohort, policy, library = published
    record = cohort.describe(library, policy)
    destination = tmp_path / "package"
    hashes = cohort.bundle(record, destination)
    assert set(hashes) == {"libc10_cuda.so", "libtorch_cuda.so"}
    assert record["requirements"] == []
    root = destination / "native_host/_generated/metax_vendor"
    assert {path.name for path in root.iterdir()} == set(hashes)
    assert not (destination / "lib_metax").exists()
    assert not (destination / "lib").exists()
    for name, checksum in hashes.items():
        assert cohort.digest(root / name) == checksum


@pytest.mark.parametrize(
    "mutation", ["header", "version", "missing_gpu", "missing_header"]
)
def test_metax_unmeasured_vendor_assets_are_rejected(published, mutation):
    cohort, policy, library = published
    if mutation == "header":
        relative = next(iter(policy["host_headers"]))
        (library.parent / "include" / relative).write_text("changed flag semantics")
    elif mutation == "version":
        (library.parent / "version.py").write_text(
            "__version__ = '2.10.0+metax.other'\n"
        )
    elif mutation == "missing_gpu":
        (library / "libtorch_cuda.so").unlink()
    else:
        relative = next(iter(policy["host_headers"]))
        (library.parent / "include" / relative).unlink()
    with pytest.raises(ValueError):
        cohort.describe(library, policy)


def test_generated_metax_vendor_abi_and_dtype_are_immutable():
    pytest.importorskip("torchgen")
    generator = load(
        "metax_vendor_codegen_test", "scripts/codegen/codegen_metax_native_vendor.py"
    )
    outputs, catalog = generator.render()
    assert (outputs, catalog) == generator.render()
    for path, text in outputs.items():
        assert path.read_text() == text
    for field, value in (
        ("dtypes", ["Float"]),
        ("stub", "other_stub"),
        ("family", "other"),
    ):
        policy = generator.precision_policy("metax")
        policy["operators"][0][field] = value
        with pytest.raises(ValueError, match="unmeasured MetaX"):
            generator.render(policy)
    policy = generator.precision_policy("metax")
    changed = copy.deepcopy(policy)
    changed["retirement"] += " Updated evidence."
    assert generator.render(changed)[1][0]["abi"] != catalog[0]["abi"]


def test_silu_exception_is_only_in_explicit_native_configuration():
    native = ROOT / "torch_fl/configs/backends_metax_native_host.conf"
    ordinary = ROOT / "torch_fl/configs/backends_metax.conf"
    clause = "silu = flaggems; BFloat16:metax\n"
    assert clause in native.read_text()
    assert clause not in ordinary.read_text()


@pytest.mark.parametrize(
    "mutation", [None, "vendor_core", "missing_core", "cuda_torch"]
)
def test_runtime_rejects_replaced_cpu_core(tmp_path, monkeypatch, mutation):
    package = ModuleType("metax_vendor_runtime_fixture")
    package.__path__ = []
    cohort = ModuleType(package.__name__ + ".vendor_cohort")
    cohort.digest = lambda path: "unused"
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, cohort.__name__, cohort)
    module = load(package.__name__ + ".vendor", "torch_fl/native_host/vendor.py")
    torch = ModuleType("torch")
    torch.__file__ = str(tmp_path / "torch/__init__.py")
    torch.__version__ = "2.10.0+cpu"
    torch.version = SimpleNamespace(cuda=None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    names = ["libc10.so", "libtorch_cpu.so", "libtorch_python.so"]
    maps = [str(tmp_path / "torch/lib" / name) for name in names]
    if mutation == "vendor_core":
        maps[1] = str(tmp_path / "vendor/libtorch_cpu.so")
    elif mutation == "missing_core":
        maps.pop()
    elif mutation == "cuda_torch":
        torch.__version__ = "2.10.0+metax3.8.1.0"
        torch.version.cuda = "11.6"
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "\n".join(maps)
            if str(path) == "/proc/self/maps"
            else original(path, *args, **kwargs)
        ),
    )
    if mutation is None:
        module._verify_cpu_core()
    else:
        with pytest.raises(RuntimeError, match="CPU"):
            module._verify_cpu_core()
