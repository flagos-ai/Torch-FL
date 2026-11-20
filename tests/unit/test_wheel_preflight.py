# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wheel preflight must work without importing torch_fl or accelerator runtimes."""

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from native_host_test_utils import import_native_host

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = SOURCE_ROOT / "scripts" / "tools" / "torch-fl-preflight"
loader = importlib.machinery.SourceFileLoader(
    "torch_fl_preflight_test", str(PREFLIGHT_PATH)
)
spec = importlib.util.spec_from_loader(loader.name, loader)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.mark.parametrize(
    "mutation", [None, "binary", "cpu_library", "version", "location"]
)
def test_metax_native_precision_bundle_is_verified_without_imports(mutation):
    prefix = "native_host/_generated/"
    assets = {
        prefix + "_metax_vendor.so": b"typed host glue",
        prefix + "metax_vendor/libc10_cuda.so": b"original vendor runtime",
        prefix + "metax_vendor/libtorch_cuda.so": b"original vendor GPU kernels",
    }
    record = {
        "vendor": "metax",
        "vendor_torch": "2.10.0+metax3.8.1.0",
        "artifacts": {
            path: hashlib.sha256(data).hexdigest() for path, data in assets.items()
        },
    }
    manifest = {
        "build": {
            "vendor_torch_libraries": True,
            "vendor_torch_version": record["vendor_torch"],
            "bundle_libdir": prefix + "metax_vendor",
        }
    }
    if mutation == "binary":
        assets[prefix + "metax_vendor/libtorch_cuda.so"] = b"changed numerical source"
    elif mutation == "cpu_library":
        record["artifacts"][prefix + "metax_vendor/libtorch_cpu.so"] = "0" * 64
    elif mutation == "version":
        manifest["build"]["vendor_torch_version"] = "unmeasured"
    elif mutation == "location":
        manifest["build"]["bundle_libdir"] = "lib_metax"
    files = {"torch_fl/" + path: data for path, data in assets.items()}
    files["torch_fl/" + prefix + "vendor_manifest.json"] = json.dumps(record).encode()
    if mutation is None:
        preflight.check_native_vendor_bundle(
            manifest, files.__getitem__, files.__contains__, "metax"
        )
    else:
        with pytest.raises(ValueError, match="vendor|artifact"):
            preflight.check_native_vendor_bundle(
                manifest, files.__getitem__, files.__contains__, "metax"
            )


def test_manifest_builder_works_without_source_root_on_import_path(
    tmp_path, monkeypatch
):
    """PEP 517 runs setup.py with a sys.path that cannot import sibling files."""
    source_root = SOURCE_ROOT
    source = ast.parse((source_root / "setup.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_write_compatibility_manifest"
    )
    namespace = {
        "os": os,
        "importlib": importlib,
        "json": json,
        "SOURCE_DIR": str(source_root),
        "BASE_DIR": str(tmp_path),
        "FLAGOS_ACCELERATOR": "cuda",
        "_kernel_switches": lambda _: {"VENDOR": True},
        "KERNEL_SWITCHES": ("VENDOR",),
        "KERNEL_SET_NAME": {"VENDOR": "vendor"},
        "_BUNDLE_LIBDIR": "lib",
        "_platform_entry": lambda _: {
            "vendor": "nvidia",
            "vendor_torch_libraries": True,
        },
        "_install_requires": lambda: ["torch>=2.10,<2.11"],
        "_native_vendor_cohort": lambda: None,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"),
        namespace,
    )
    fake_torch = types.ModuleType("torch")
    fake_torch._C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "2.10.0")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "path",
        [
            entry
            for entry in sys.path
            if Path(entry or os.getcwd()).resolve() != source_root
        ],
    )
    (tmp_path / "torch_fl").mkdir()

    namespace["_write_compatibility_manifest"]("0.1.0")

    manifest = json.loads((tmp_path / "torch_fl" / "compatibility.json").read_text())
    assert manifest["platform"] == "cuda"
    assert manifest["build"]["distributions"]["torch"] == "2.10.0"


@pytest.fixture
def manifest(monkeypatch):
    versions = {
        "torch": "2.10.0+cpu",
        "flagtree": "0.7.0rc2+cuda13.3",
        "flag_gems": "5.4.0",
        "flagcx": "0.14.0rc2.post2+cuda13.3",
    }
    monkeypatch.setattr(preflight, "installed_version", versions.get)
    return preflight.make_manifest(
        platform="cuda",
        wheel_version="0.1.0+cuda13.3",
        kernels=["vendor", "boxing"],
        bundle_libdir="lib",
        vendor_torch_libraries=True,
        requirements=["torch>=2.10,<2.11", "flag_gems>=5.0.2"],
        torch_abi=False,
        sdk_version="13.3",
        vendor_torch_version="2.10.0+cu130",
    )


def _wheel(tmp_path, manifest, *, version="0.1.0+cuda13.3", requirements=None):
    path = tmp_path / "torch_fl.whl"
    requirements = requirements or ["torch<2.11,>=2.10", "flag_gems>=5.0.2"]
    wheel_metadata = (
        "Metadata-Version: 2.3\n"
        "Name: torch-fl\n"
        f"Version: {version}\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requirements)
        + 'Requires-Dist: flagcx; extra == "flagcx"\n'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(preflight.MANIFEST_PATH, json.dumps(manifest))
        archive.writestr("torch_fl-0.1.0.dist-info/METADATA", wheel_metadata)
    return path


def _snapshot_modules():
    """The names in ``sys.modules`` now, to diff across a call under test.

    Outright membership is not assertable from a test in this file. pytest
    imports every test module of the session during collection, and twenty-five
    files under tests/unit import torch_fl at module level, so by the time any
    test here runs the name is already present whatever order the command line
    gave -- ``pytest tests/unit/`` cannot pass a bare ``"torch_fl" not in
    sys.modules``. The diff across the call is the property this file is about,
    and under the per-file CI runner (``run_unit_tests.py``), where torch_fl is
    absent, that diff is the outright membership test.

    An import already satisfied from ``sys.modules`` leaves no trace in a diff,
    so it cannot be read as proof that the tool never asks for the package:
    ``test_script_cli_does_not_import_torch_fl`` covers that case by running the
    CLI in a subprocess with a poisoned ``torch_fl`` on ``PYTHONPATH``.
    """
    return set(sys.modules)


def _assert_imported_no_torch_fl(since):
    added = {
        name
        for name in set(sys.modules) - since
        if name == "torch_fl" or name.startswith("torch_fl.")
    }
    assert not added, f"the preflight imported {sorted(added)}"


def test_wheel_manifest_agrees_with_metadata_without_importing_package(
    tmp_path, manifest
):
    before = _snapshot_modules()
    loaded = preflight.load_wheel(_wheel(tmp_path, manifest))
    assert loaded == manifest
    _assert_imported_no_torch_fl(before)


def test_wheel_rejects_missing_or_stale_manifest(tmp_path, manifest):
    with pytest.raises(ValueError, match="version differs"):
        preflight.load_wheel(_wheel(tmp_path, manifest, version="0.1.1"))
    with pytest.raises(ValueError, match="requirements differ"):
        preflight.load_wheel(
            _wheel(tmp_path, manifest, requirements=["torch>=2.10,<2.11"])
        )
    with zipfile.ZipFile(tmp_path / "missing.whl", "w") as archive:
        archive.writestr("torch_fl-0.1.0.dist-info/METADATA", "Name: torch-fl\n")
    with pytest.raises(ValueError, match="compatibility.json"):
        preflight.load_wheel(tmp_path / "missing.whl")
    manifest["build"]["distributions"]["torch"] = "2.11.0"
    with pytest.raises(ValueError, match="outside the wheel's declared range"):
        preflight.load_wheel(_wheel(tmp_path, manifest))


def test_a_wheel_is_accepted_under_its_vendor_name(manifest):
    """CI lanes export the vendor name; a wheel records the table id.

    The apple lane runs `--platform apple` against a wheel whose manifest says
    `mps`, which failed closed with "Wheel platform is mps; requested apple".
    """
    errors, _ = preflight.check_environment(manifest, platform="cuda")
    assert errors == []
    for vendor in ("nvidia", "cuda"):
        errors, _ = preflight.check_environment(manifest, platform=vendor)
        assert errors == [], (vendor, errors)


def test_the_vendor_equivalence_follows_the_platform_table(manifest):
    """The row under test is accepted under both names; no other row is.

    Keyed off the wheel's own row (`cuda`) so the test does not depend on which
    platform the fixture happens to build.
    """
    table = json.loads(
        (SOURCE_ROOT / "cmake" / "flagos_platforms.json").read_text(encoding="utf-8")
    )["accelerators"]
    row = manifest["platform"]
    vendor = table[row]["vendor"]
    for name in (row, vendor):
        errors, _ = preflight.check_environment(manifest, platform=name)
        assert not any("Wheel platform is" in item for item in errors), name

    others = [name for name in table if name != row]
    for name in others:
        errors, _ = preflight.check_environment(manifest, platform=name)
        assert f"Wheel platform is {row}; requested {name}" in errors, name


def test_installed_preflight_reads_the_packaged_table(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "PLATFORM_TABLE", tmp_path / "missing.json")
    table = SOURCE_ROOT / "cmake" / "flagos_platforms.json"
    distribution = types.SimpleNamespace(locate_file=lambda _: table)
    monkeypatch.setattr(preflight.metadata, "distribution", lambda _: distribution)
    assert preflight._same_platform("apple", "mps")
    assert preflight._same_platform("hygon", "dcu")
    assert not preflight._same_platform("nvidia", "ppu")


def test_without_the_table_the_comparison_stays_exact(monkeypatch, tmp_path):
    """Older wheel installs may lack the platform table."""
    monkeypatch.setattr(preflight, "PLATFORM_TABLE", tmp_path / "missing.json")
    distribution = types.SimpleNamespace(locate_file=lambda name: tmp_path / name)
    monkeypatch.setattr(preflight.metadata, "distribution", lambda _: distribution)
    assert preflight.platform_vendor("apple") is None
    assert preflight._same_platform("apple", "mps") is False
    assert preflight._same_platform("mps", "mps") is True


def test_sdk_and_platform_mismatches_fail_closed(manifest):
    errors, _ = preflight.check_environment(
        manifest, platform="musa", sdk_version="13.0"
    )
    assert "Wheel platform is cuda; requested musa" in errors
    assert "Wheel SDK is 13.3; requested 13.0" in errors

    manifest["build"]["sdk_version"] = None
    errors, _ = preflight.check_environment(manifest, require_sdk=True)
    assert "FLAGOS_SDK_VERSION" in errors[0]
    errors, _ = preflight.check_environment(manifest, sdk_version="13.3")
    assert "unknown" in errors[0]


def test_build_env_checks_actual_distributions(manifest, monkeypatch):
    versions = {
        "torch": "2.10.0+cpu",
        "flagtree": "0.7.0rc2+cuda13.3",
        "flag_gems": "5.3.0",
        "flagcx": None,
    }
    monkeypatch.setattr(preflight, "installed_version", versions.get)
    errors, _ = preflight.check_environment(manifest, check_build_env=True)
    assert any("flag_gems build version is 5.4.0" in error for error in errors)
    assert any("flagcx build version is" in error for error in errors)


def test_runtime_uses_declared_ranges_and_warns_on_unverified_abi(
    manifest, monkeypatch
):
    versions = {
        "torch": "2.10.1+cpu",
        "flagtree": "0.7.0rc2+cuda13.3",
        "flag_gems": "5.5.0",
        "flagcx": "0.14.0rc2.post2+cuda13.3",
    }
    monkeypatch.setattr(preflight, "installed_version", versions.get)
    errors, warnings = preflight.check_environment(manifest, check_installed=True)
    assert errors == []
    assert any("native ABI compatibility is not proven" in item for item in warnings)
    assert any("flag_gems was built/tested with 5.4.0" in item for item in warnings)

    versions["torch"] = "2.11.0"
    errors, _ = preflight.check_environment(manifest, check_installed=True)
    assert any("Required torch>=2.10,<2.11" in item for item in errors)

    versions["torch"] = "2.10.0+cpu"
    versions["flag_gems"] = None
    errors, _ = preflight.check_environment(manifest, check_installed=True)
    assert any("Required flag_gems>=5.0.2" in item for item in errors)


def test_strict_tested_rejects_optional_build_version_changes(manifest, monkeypatch):
    versions = {
        "torch": "2.10.0+cpu",
        "flagtree": "0.7.0rc3+cuda13.3",
        "flag_gems": "5.4.0",
        "flagcx": "0.14.0rc2.post2+cuda13.3",
    }
    monkeypatch.setattr(preflight, "installed_version", versions.get)
    errors, _ = preflight.check_environment(
        manifest, check_installed=True, strict_tested=True
    )
    assert any("flagtree was built/tested with" in item for item in errors)


def test_wheel_name_names_the_platform_id(tmp_path, manifest, capsys):
    """`--wheel-name` ties --platform to the SDK segment in the filename.

    The artifact names its build variant (`torch_fl-2.10.0+cuda13.3`), and the
    MPS build's segment is `+mps` -- not the vendor `apple`. The check is
    therefore against the table id, which is what a wheel's filename carries.
    """
    wheel = tmp_path / "torch_fl-2.10.0+cuda13.3-cp312-cp312-linux_x86_64.whl"
    wheel.write_bytes(_wheel(tmp_path, manifest).read_bytes())
    assert (
        preflight.main(["--wheel", str(wheel), "--platform", "cuda", "--wheel-name"])
        == 0
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as raised:
        preflight.main(["--wheel", str(wheel), "--platform", "mps", "--wheel-name"])
    assert raised.value.code == 2
    capsys.readouterr()
    # The MPS build's version carries no SDK segment, so the platform id is the
    # segment (`+mps`), as a base build's `+cpu` would be.
    mps_wheel = tmp_path / "torch_fl-2.10.0+mps-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(mps_wheel, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.endswith("/METADATA"):
                data = data.replace(b"Version: 0.1.0+cuda13.3", b"Version: 0.1.0+mps")
            elif item.filename == preflight.MANIFEST_PATH:
                data = json.dumps(
                    {**manifest, "platform": "mps", "wheel_version": "0.1.0+mps"}
                ).encode()
            target.writestr(item, data)
    assert (
        preflight.main(["--wheel", str(mps_wheel), "--platform", "mps", "--wheel-name"])
        == 0
    )
    capsys.readouterr()
    # A lane that names its platform by the vendor spelling still finds the
    # segment through the table, and another platform's still does not.
    assert (
        preflight.main(
            ["--wheel", str(mps_wheel), "--platform", "apple", "--wheel-name"]
        )
        == 0
    )
    capsys.readouterr()
    with pytest.raises(SystemExit) as raised:
        preflight.main(
            ["--wheel", str(mps_wheel), "--platform", "nvidia", "--wheel-name"]
        )
    assert raised.value.code == 2


def test_cli_reports_mismatch_before_import(tmp_path, manifest, capsys):
    wheel = _wheel(tmp_path, manifest)
    before = _snapshot_modules()
    assert preflight.main(["--wheel", str(wheel), "--platform", "mthreads"]) == 1
    assert "Wheel platform is cuda" in capsys.readouterr().err
    _assert_imported_no_torch_fl(before)


def test_script_cli_does_not_import_torch_fl(tmp_path, manifest):
    wheel = _wheel(tmp_path, manifest)
    (tmp_path / "torch_fl").mkdir()
    (tmp_path / "torch_fl" / "__init__.py").write_text(
        'raise RuntimeError("torch_fl was imported")\n', encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT_PATH),
            "--wheel",
            str(wheel),
            "--platform",
            "musa",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Wheel platform is cuda; requested musa" in result.stderr
    assert "torch_fl was imported" not in result.stderr


def test_release_table_comes_from_wheel_and_requires_known_sdk(
    tmp_path, manifest, capsys
):
    wheel = _wheel(tmp_path, manifest)
    assert (
        preflight.main(["--wheel", str(wheel), "--markdown-table", "--require-sdk"])
        == 0
    )
    assert (
        "| cuda | 0.1.0+cuda13.3 | 2.10.0+cpu | 2.10.0+cu130 | 13.3 |"
        in capsys.readouterr().out
    )
    manifest["build"]["sdk_version"] = None
    wheel = _wheel(tmp_path, manifest)
    assert (
        preflight.main(["--wheel", str(wheel), "--markdown-table", "--require-sdk"])
        == 1
    )
    assert "FLAGOS_SDK_VERSION" in capsys.readouterr().err


def test_release_rejects_missing_vendor_torch_provenance(manifest):
    manifest["build"]["vendor_torch_version"] = None
    errors, _ = preflight.check_environment(manifest, release=True)
    assert any("FLAGOS_VENDOR_TORCH_VERSION" in item for item in errors)
    manifest["build"]["vendor_torch_libraries"] = False
    errors, _ = preflight.check_environment(manifest, release=True)
    assert errors == []


def test_placeholder_sdk_cannot_be_declared(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "installed_version",
        lambda name: "2.10.0" if name == "torch" else None,
    )
    with pytest.raises(ValueError, match="verified version"):
        preflight.make_manifest(
            platform="cuda",
            wheel_version="0.1.0",
            kernels=[],
            bundle_libdir="lib",
            vendor_torch_libraries=True,
            requirements=["torch>=2.10,<2.11"],
            torch_abi=False,
            sdk_version="unknown",
        )


def test_every_ci_wheel_build_checks_the_embedded_manifest():
    """New platform workflows must not publish an unchecked compatibility record."""
    workflows = Path(__file__).resolve().parents[2] / ".github" / "workflows"
    for path in workflows.glob("*.yml"):
        source = path.read_text(encoding="utf-8")
        if "python -m build --wheel --no-isolation" not in source:
            continue
        build_at = source.index("python -m build --wheel --no-isolation")
        verify_at = source.index(
            "python scripts/tools/torch-fl-preflight --wheel dist/*.whl"
        )
        assert verify_at > build_at, path
        assert "--check-build-env" in source[verify_at : verify_at + 160], path
        if path.name in ("build-wheel-common.yml", "integration-tests.yml"):
            assert (
                '--platform "$FLAGOS_ACCELERATOR"'
                in source[verify_at : verify_at + 180]
            ), path
        install_at = source.find("python -m pip install", build_at)
        if install_at >= 0:
            assert verify_at < install_at, path


@pytest.fixture(params=["nvidia", "metax", "hygon", "ascend"])
def native_wheel(tmp_path, manifest, monkeypatch, request):
    backend = import_native_host("backend")
    driver_root = tmp_path / "compiler"
    vendor = request.param
    adapter = backend.get_backend(vendor)
    for path in backend.driver_sources(adapter, driver_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// published backend source\n")
    monkeypatch.setattr(
        backend.importlib.metadata,
        "version",
        lambda name: manifest["build"]["distributions"][name],
    )
    record = backend.build_record(adapter, driver_root)
    staged_native = tmp_path / "staged_native"
    backend.stage_native_sources(
        adapter, SOURCE_ROOT / "csrc/native_host", staged_native / "csrc"
    )
    manifest["platform"] = {"nvidia": "cuda", "hygon": "dcu"}.get(vendor, vendor)
    manifest["build"]["kernels"] = ["vendor", "flaggems"]
    path = _wheel(tmp_path, manifest)
    prefix = "torch_fl/native_host/"
    with zipfile.ZipFile(path, "a") as wheel:
        wheel.writestr(
            prefix + "_generated/manifest.json", json.dumps({"backend": record})
        )
        wheel.writestr(
            "torch_fl/_build_config.py",
            f'ACCELERATOR = "{vendor}"\nNATIVE_HOSTS = True\n',
        )
        for category in ("native_sources", "adapter_sources"):
            for name in record[category]:
                root = (
                    staged_native
                    if category == "native_sources"
                    else SOURCE_ROOT / prefix
                )
                wheel.writestr(prefix + name, (root / name).read_bytes())
        wheel.writestr(
            prefix + "_generated/_launcher.cpython-312-test.so", b"not loaded"
        )
        wheel.writestr(
            prefix + "_generated/" + record["utility_modules"][0] + ".so", b"not loaded"
        )
    return path, record, driver_root


def _replace_wheel_entry(path, name, value):
    with zipfile.ZipFile(path) as wheel:
        contents = {entry: wheel.read(entry) for entry in wheel.namelist()}
    if value is None:
        del contents[name]
    else:
        contents[name] = value
    with zipfile.ZipFile(path, "w") as wheel:
        for entry, content in contents.items():
            wheel.writestr(entry, content)


def test_versioned_native_backend_is_verified_without_imports(native_wheel):
    path, _, _ = native_wheel
    before = _snapshot_modules()
    preflight.load_wheel(path, native_backend=True)
    _assert_imported_no_torch_fl(before)


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "version",
        "source",
        "missing_source",
        "driver_binding",
        "utility",
        "compiler",
    ],
)
def test_native_backend_rejects_mismatched_artifacts(native_wheel, change):
    path, record, _ = native_wheel
    prefix = "torch_fl/native_host/"
    if change in {"source", "missing_source"}:
        _replace_wheel_entry(
            path,
            f"{prefix}csrc/backends/{record['vendor']}/launch.cc",
            b"changed source" if change == "source" else None,
        )
    elif change == "driver_binding":
        _replace_wheel_entry(path, prefix + "backends/driver.py", b"changed binding")
    elif change == "utility":
        _replace_wheel_entry(
            path, prefix + "_generated/" + record["utility_modules"][0] + ".so", None
        )
    else:
        if change == "target":
            record["vendor"] = "unregistered"
        elif change == "version":
            record["abi_version"] = 99
        else:
            record["compiler_distribution"]["version"] = "0.8.0"
        _replace_wheel_entry(
            path, prefix + "_generated/manifest.json", json.dumps({"backend": record})
        )
    with pytest.raises(ValueError, match="backend|Native-host|native-host"):
        preflight.load_wheel(path, native_backend=True)


def test_native_backend_verifies_installed_driver_sources(native_wheel, monkeypatch):
    path, record, root = native_wheel
    monkeypatch.setattr(
        preflight.metadata,
        "distribution",
        lambda name: types.SimpleNamespace(locate_file=lambda name: root.parent / name),
    )
    # The distribution root contains triton/, as in the actual published wheel.
    (root.parent / "triton").symlink_to(root, target_is_directory=True)
    preflight.load_wheel(path, native_backend=True, check_driver=True)
    (root / next(iter(record["driver_sources"]))).write_text("# changed ABI\n")
    with pytest.raises(ValueError, match="driver ABI differs"):
        preflight.load_wheel(path, native_backend=True, check_driver=True)
    assert record["compiler_distribution"]["name"] == "flagtree"


def test_native_backend_requirement_is_explicit_for_historical_wheels(
    tmp_path, manifest
):
    path = _wheel(tmp_path, manifest)
    preflight.load_wheel(path)
    with pytest.raises(ValueError, match="no native-host backend manifest"):
        preflight.load_wheel(path, native_backend=True)


def test_native_backend_cli_does_not_import_package(native_wheel, tmp_path):
    path, _, _ = native_wheel
    poisoned = tmp_path / "poison/torch_fl"
    poisoned.mkdir(parents=True)
    (poisoned / "__init__.py").write_text(
        'raise AssertionError("preflight imported torch_fl")\n'
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT_PATH),
            "--wheel",
            str(path),
            "--check-native-host-backend",
        ],
        env={**os.environ, "PYTHONPATH": str(poisoned.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
