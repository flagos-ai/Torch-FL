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

"""Unit coverage for tests/integration/platform_support.py::detect_platform().

detect_platform() is the shared platform-name resolver behind the profiler and
AMP cross-backend contracts (tests/integration/profiler_support.py,
tests/integration/amp_support.py). Its marker check
(torch_fl/lib/flagos_platform) previously had no Ascend-writing counterpart in
csrc/CMakeLists.txt, so it could only identify Ascend via the final
FLAGOS_BACKEND_CONFIG substring match -- which itself depends on torch_fl's
own /dev/davinci* runtime probe having succeeded. See issue #192: that probe
is not guaranteed (e.g. /dev is not enumerable), and its fallback path is
"cuda". These tests confirm the marker now lets detect_platform() identify
Ascend without depending on that runtime probe at all.

Run: pytest tests/unit/test_platform_support_detect.py -v
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLATFORM_SUPPORT_PATH = _REPO_ROOT / "tests" / "integration" / "platform_support.py"


def _load_platform_support():
    """Import platform_support.py without going through the tests/integration
    pytest-plugin machinery (which requires the full integration conftest)."""
    spec = importlib.util.spec_from_file_location(
        "platform_support_under_test", _PLATFORM_SUPPORT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


platform_support = _load_platform_support()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ACCELERATOR", raising=False)
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("PPU_SDK", raising=False)
    monkeypatch.delenv("PPU_HOME", raising=False)


def test_accelerator_env_wins_outright():
    os.environ["ACCELERATOR"] = "ascend"
    assert platform_support.detect_platform() == "ascend"


@pytest.fixture
def isolated_torch_fl_import(monkeypatch):
    """Temporarily hide the real torch_fl module so detect_platform()'s
    ``import torch_fl`` resolves a throwaway stub instead.

    The real torch_fl.__init__ has import-time side effects (it registers the
    "flagos" device module with torch._register_device_module()), which
    raises RuntimeError on a second real import and corrupts state for any
    test running afterwards. Swapping sys.modules["torch_fl"] out and back in
    avoids re-triggering that import entirely.
    """
    real_torch_fl = sys.modules.pop("torch_fl", None)
    try:
        yield
    finally:
        sys.modules.pop("torch_fl", None)
        if real_torch_fl is not None:
            sys.modules["torch_fl"] = real_torch_fl


def test_marker_identifies_ascend_without_any_env_or_config(
    isolated_torch_fl_import, tmp_path
):
    """The scenario the issue's reproduction targets: no ACCELERATOR, no
    FLAGOS_BACKEND_CONFIG. Previously this fell through to the "cuda" default
    unless torch_fl's own /dev/davinci* probe had already set
    FLAGOS_BACKEND_CONFIG as a side effect of import. The marker makes this
    deterministic instead of accidental."""
    fake_torch_fl_dir = tmp_path / "torch_fl"
    lib_dir = fake_torch_fl_dir / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "flagos_platform").write_text("ascend\n")
    (fake_torch_fl_dir / "__init__.py").write_text("")

    sys.path.insert(0, str(tmp_path))
    try:
        assert platform_support.detect_platform() == "ascend"
    finally:
        sys.path.remove(str(tmp_path))


def test_no_marker_and_no_config_falls_back_to_cuda(isolated_torch_fl_import, tmp_path):
    """Documents the residual gap: a torch_fl install with no marker and no
    FLAGOS_BACKEND_CONFIG set still can't be identified as anything but cuda.
    This is expected for CUDA-compatible platforms; for Ascend it is now
    avoided by the marker (see the sibling test above), not by this
    function's own logic."""
    fake_torch_fl_dir = tmp_path / "torch_fl"
    fake_torch_fl_dir.mkdir()
    (fake_torch_fl_dir / "__init__.py").write_text("")

    sys.path.insert(0, str(tmp_path))
    try:
        assert platform_support.detect_platform() == "cuda"
    finally:
        sys.path.remove(str(tmp_path))


def test_ppu_sdk_env_detected_before_marker_check():
    os.environ["PPU_SDK"] = "/opt/ppu"
    assert platform_support.detect_platform() == "ppu"
