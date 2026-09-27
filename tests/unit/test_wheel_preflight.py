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

import json
import sys
import zipfile
from pathlib import Path

import pytest

import torch_fl_preflight as preflight


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


def test_wheel_manifest_agrees_with_metadata_without_importing_package(
    tmp_path, manifest
):
    assert "torch_fl" not in sys.modules
    loaded = preflight.load_wheel(_wheel(tmp_path, manifest))
    assert loaded == manifest
    assert "torch_fl" not in sys.modules


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


def test_cli_reports_mismatch_before_import(tmp_path, manifest, capsys):
    wheel = _wheel(tmp_path, manifest)
    assert preflight.main(["--wheel", str(wheel), "--platform", "musa"]) == 1
    assert "Wheel platform is cuda" in capsys.readouterr().err
    assert "torch_fl" not in sys.modules


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
        verify_at = source.index("python -m torch_fl_preflight --wheel dist/*.whl")
        assert verify_at > build_at, path
        assert "--check-build-env" in source[verify_at : verify_at + 160], path
        install_at = source.find("python -m pip install", build_at)
        if install_at >= 0:
            assert verify_at < install_at, path
