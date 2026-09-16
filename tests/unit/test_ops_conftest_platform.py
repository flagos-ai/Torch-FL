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

"""Unit coverage for tests/integration/ops/conftest.py platform detection.

The ops gate skips backend-specific tests per detected platform
(_PLATFORM_SKIP_MARKERS). PPU is a CUDA-ABI boxing backend whose build and CI
deliberately report ACCELERATOR=cuda, so the ACCELERATOR check cannot tell it
apart -- before it was named here, PPU fell into the "default" bucket and was
correct only by accident of that bucket's skip set. These tests pin the three
PPU signals (PPU_SDK / PPU_HOME env, the lib_ppu/ bundle dir, the resolved
backends_ppu.conf name -- the same signals torch_fl._is_ppu_build() uses) and
the skip-set parity with "default" that keeps this change behavior-neutral.

Run: pytest tests/unit/test_ops_conftest_platform.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPS_CONFTEST_PATH = _REPO_ROOT / "tests" / "integration" / "ops" / "conftest.py"


def _load_ops_conftest():
    """Import ops/conftest.py without going through the tests/integration
    pytest-plugin machinery (which requires the full integration conftest)."""
    spec = importlib.util.spec_from_file_location(
        "ops_conftest_under_test", _OPS_CONFTEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ops_conftest = _load_ops_conftest()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ACCELERATOR", raising=False)
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.delenv("PPU_HOME", raising=False)


@pytest.fixture
def isolated_torch_fl_import():
    """Temporarily hide the real torch_fl module so _detect_platform()'s
    ``import torch_fl`` resolves a throwaway stub instead.

    Same pattern as test_platform_support_detect.py: the real torch_fl.__init__
    has import-time side effects, and swapping sys.modules out and back in
    avoids re-triggering that import entirely.
    """
    real_torch_fl = sys.modules.pop("torch_fl", None)
    try:
        yield
    finally:
        sys.modules.pop("torch_fl", None)
        if real_torch_fl is not None:
            sys.modules["torch_fl"] = real_torch_fl


def _fake_torch_fl(tmp_path, *, marker=None, bundle_lib_ppu=False):
    """Point ``import torch_fl`` at a stub package with the given layout."""
    package = tmp_path / "torch_fl"
    package.mkdir()
    (package / "__init__.py").write_text("")
    if marker is not None:
        lib_dir = package / "lib"
        lib_dir.mkdir()
        (lib_dir / "flagos_platform").write_text(marker + "\n")
    if bundle_lib_ppu:
        (package / "lib_ppu").mkdir()
    sys.path.insert(0, str(tmp_path))
    try:
        yield package
    finally:
        sys.path.remove(str(tmp_path))


@pytest.fixture
def fake_torch_fl(tmp_path):
    yield from _fake_torch_fl(tmp_path)


def test_accelerator_names_still_map(monkeypatch):
    for accelerator, expected in [
        ("ascend", "ascend"),
        ("metax", "metax"),
        ("maca", "metax"),
        ("musa", "musa"),
        ("dcu", "dcu"),
    ]:
        monkeypatch.setenv("ACCELERATOR", accelerator)
        assert ops_conftest._detect_platform() == expected


def test_ppu_sdk_env_identifies_ppu_despite_cuda_accelerator(monkeypatch):
    """PPU CI exports ACCELERATOR=cuda (load-bearing for the build); the env
    signal is what tells the gate apart."""
    monkeypatch.setenv("ACCELERATOR", "cuda")
    monkeypatch.setenv("PPU_SDK", "/usr/local/PPU_SDK")
    assert ops_conftest._detect_platform() == "ppu"


def test_ppu_home_env_identifies_ppu(monkeypatch):
    monkeypatch.setenv("PPU_HOME", "/opt/ppu")
    assert ops_conftest._detect_platform() == "ppu"


def test_cuda_without_ppu_signals_stays_default(
    monkeypatch, isolated_torch_fl_import, fake_torch_fl
):
    """No PPU signal anywhere: same bucket as before this change."""
    monkeypatch.setenv("ACCELERATOR", "cuda")
    assert ops_conftest._detect_platform() == "default"


def test_lib_ppu_bundle_dir_identifies_ppu_without_env(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """An installed PPU wheel with no PPU_SDK/PPU_HOME in the environment
    (dev pod with the baked SDK) is still recognized by its bundle dir."""
    monkeypatch.setenv("ACCELERATOR", "cuda")
    for _ in _fake_torch_fl(tmp_path, bundle_lib_ppu=True):
        assert ops_conftest._detect_platform() == "ppu"


def test_marker_stays_authoritative_over_bundle_dir(
    monkeypatch, isolated_torch_fl_import, tmp_path
):
    """A flagos_platform marker wins even if lib_ppu/ is also present, so the
    native-kernel platforms keep their existing precedence."""
    monkeypatch.setenv("ACCELERATOR", "cuda")
    for _ in _fake_torch_fl(tmp_path, marker="gcu", bundle_lib_ppu=True):
        assert ops_conftest._detect_platform() == "gcu"


def test_resolved_ppu_backend_config_identifies_ppu(
    monkeypatch, isolated_torch_fl_import, fake_torch_fl
):
    """Importing torch_fl resolves backends_ppu.conf for a PPU wheel; the
    config-name fallback catches PPU even when neither env var is set."""
    monkeypatch.setenv(
        "FLAGOS_BACKEND_CONFIG", "/opt/venv/lib/torch_fl/configs/backends_ppu.conf"
    )
    assert ops_conftest._detect_platform() == "ppu"


def test_ppu_skip_set_matches_default():
    """Behavior-neutrality pin: PPU skips exactly what the old implicit
    "default" bucket skipped. Any future divergence must be a deliberate,
    separately reviewed edit to this tuple."""
    assert (
        ops_conftest._PLATFORM_SKIP_MARKERS["ppu"]
        == ops_conftest._PLATFORM_SKIP_MARKERS["default"]
    )
    assert "cuda" not in ops_conftest._PLATFORM_SKIP_MARKERS["ppu"]
