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

"""Unit coverage for torch_fl/_platform.py, the shared platform detector.

This is the one detector the package (torch_fl/__init__.py::_build_accelerator)
and both integration test trees (ops/conftest.py, platform_support.py) read. The
sibling test_platform_support_detect.py exercises it through the test-tree
re-export; this file covers the module directly, including the pieces only it
resolves (lib_ppu/ bundle, the conf fallback, and the documented default).

Loaded by path, not imported: importing torch_fl executes torch_fl/__init__.py,
whose device-claiming side effects must not run in a unit test.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLATFORM_PATH = _REPO_ROOT / "torch_fl" / "_platform.py"


def _load_platform():
    loader = importlib.machinery.SourceFileLoader(
        "torch_fl_platform_under_test", str(_PLATFORM_PATH)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


platform = _load_platform()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.delenv("FLAGOS_ACCELERATOR", raising=False)


@pytest.fixture
def isolated_torch_fl(monkeypatch):
    """Hide any real torch_fl so find_spec() resolves a stub wheel instead."""
    real = sys.modules.pop("torch_fl", None)
    try:
        yield
    finally:
        sys.modules.pop("torch_fl", None)
        if real is not None:
            sys.modules["torch_fl"] = real


def _install_wheel(tmp_path, accelerator=None, marker=None, lib_ppu=False):
    """Put a stub torch_fl on sys.path describing a wheel; caller removes it."""
    package = tmp_path / "torch_fl"
    package.mkdir()
    (package / "__init__.py").write_text("")
    if accelerator is not None:
        (package / "_build_config.py").write_text(
            f'ACCELERATOR = "{accelerator}"\nKERNELS = ()\n'
        )
    if marker is not None:
        (package / "lib").mkdir()
        (package / "lib" / "flagos_platform").write_text(f"{marker}\n")
    if lib_ppu:
        (package / "lib_ppu").mkdir()
    sys.path.insert(0, str(tmp_path))
    return tmp_path


def test_default_platform_is_cuda():
    """The documented default: an unidentified wheel behaves as cuda."""
    assert platform.DEFAULT_PLATFORM == "cuda"


def test_build_accelerator_reads_the_record(isolated_torch_fl, tmp_path):
    _install_wheel(tmp_path, accelerator="ascend")
    try:
        assert platform.build_accelerator() == "ascend"
    finally:
        sys.path.remove(str(tmp_path))


def test_build_accelerator_blank_without_a_record(isolated_torch_fl, tmp_path):
    _install_wheel(tmp_path)
    try:
        assert platform.build_accelerator() == ""
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.parametrize(
    "accelerator, expected",
    [
        ("ascend", "ascend"),
        ("dcu", "dcu"),
        ("gcu", "gcu"),
        ("metax", "metax"),
        ("maca", "metax"),
        ("musa", "musa"),
        ("ppu", "ppu"),
    ],
)
def test_record_maps_to_platform(isolated_torch_fl, tmp_path, accelerator, expected):
    _install_wheel(tmp_path, accelerator=accelerator)
    try:
        assert platform.detect_platform() == expected
    finally:
        sys.path.remove(str(tmp_path))


def test_marker_identifies_without_a_record(isolated_torch_fl, tmp_path):
    _install_wheel(tmp_path, marker="ascend")
    try:
        assert platform.detect_platform() == "ascend"
    finally:
        sys.path.remove(str(tmp_path))


def test_lib_ppu_bundle_identifies_ppu(isolated_torch_fl, tmp_path):
    """PPU wheels predating the ppu record value ship a lib_ppu/ bundle instead."""
    _install_wheel(tmp_path, lib_ppu=True)
    try:
        assert platform.detect_platform() == "ppu"
    finally:
        sys.path.remove(str(tmp_path))


def test_ppu_sdk_env_wins_over_the_marker(isolated_torch_fl, tmp_path, monkeypatch):
    _install_wheel(tmp_path, marker="ascend")
    monkeypatch.setenv("PPU_SDK", "/opt/ppu")
    try:
        assert platform.detect_platform() == "ppu"
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.parametrize(
    "conf, expected",
    [
        ("/x/backends_gcu.conf", "gcu"),
        ("/x/backends_dcu.conf", "dcu"),
        ("/x/backends_cuda.conf", "cuda"),
    ],
)
def test_conf_fallback(isolated_torch_fl, tmp_path, monkeypatch, conf, expected):
    _install_wheel(tmp_path)
    monkeypatch.setenv("FLAGOS_BACKEND_CONFIG", conf)
    try:
        assert platform.detect_platform() == expected
    finally:
        sys.path.remove(str(tmp_path))


def test_default_when_nothing_identifies_the_build(isolated_torch_fl, tmp_path):
    _install_wheel(tmp_path)
    try:
        assert platform.detect_platform() == "cuda"
    finally:
        sys.path.remove(str(tmp_path))
