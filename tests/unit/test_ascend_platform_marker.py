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

"""Unit coverage for the Ascend lib/flagos_platform marker.

csrc/CMakeLists.txt now writes lib/flagos_platform for Ascend builds too (it
previously only did so for gcu/musa/bpu), so
tests/integration/platform_support.py::detect_platform() can identify Ascend
from the installed marker instead of relying solely on the /dev/davinci*
runtime probe in torch_fl/__init__.py -- a probe that silently falls back to
the CUDA config if /dev enumeration fails for any reason (permissions, a
sandboxed container, etc). See issue #192.

These tests exercise torch_fl._select_backend_config() directly against a
fake install tree (no real ACL device needed), and pin the exact regression
this fix must not reintroduce: the marker branch runs *before* the
/dev/davinci* branch, so it must special-case the Ascend FlagGems opt-in
(FLAGOS_USE_FLAGGEMS=1 -> backends_ascend_flagos_py.conf) itself, the same way
it already does for MUSA. Without that, installing the marker would silently
shadow the FlagGems opt-in on real Ascend hardware.
"""

import os

import pytest

import torch_fl


@pytest.fixture
def fake_ascend_install(tmp_path, monkeypatch):
    """Point torch_fl.__file__ at a scratch tree with an Ascend marker + confs."""
    lib_dir = tmp_path / "lib"
    conf_dir = tmp_path / "configs"
    lib_dir.mkdir()
    conf_dir.mkdir()
    (lib_dir / "flagos_platform").write_text("ascend\n")
    (conf_dir / "backends_ascend.conf").write_text("")
    (conf_dir / "backends_ascend_flagos_py.conf").write_text("")

    monkeypatch.setattr(torch_fl, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("FLAGOS_USE_FLAGGEMS", raising=False)
    monkeypatch.delenv("FLAGOS_USE_FLAGGEMS_CPP", raising=False)
    monkeypatch.delenv("FLAGOS_USE_TILEOPS", raising=False)
    monkeypatch.delenv("FLAGOS_METAX_BOXING", raising=False)
    return conf_dir


def test_ascend_marker_selects_native_conf_by_default(fake_ascend_install):
    conf_dir = fake_ascend_install
    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(conf_dir / "backends_ascend.conf")


def test_ascend_marker_honors_flaggems_opt_in(monkeypatch, fake_ascend_install):
    """Regression guard: the marker branch must not shadow the Ascend FlagGems
    conf that the /dev/davinci* branch would otherwise pick."""
    conf_dir = fake_ascend_install
    monkeypatch.setenv("FLAGOS_USE_FLAGGEMS", "1")
    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(
        conf_dir / "backends_ascend_flagos_py.conf"
    )


def test_ascend_marker_falls_back_to_native_conf_if_flaggems_conf_missing(
    monkeypatch, tmp_path
):
    """If a wheel ships the marker but not the flagos_py conf (old install
    layout), the native conf must still be usable rather than raising."""
    lib_dir = tmp_path / "lib"
    conf_dir = tmp_path / "configs"
    lib_dir.mkdir()
    conf_dir.mkdir()
    (lib_dir / "flagos_platform").write_text("ascend\n")
    (conf_dir / "backends_ascend.conf").write_text("")

    monkeypatch.setattr(torch_fl, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("FLAGOS_USE_FLAGGEMS", "1")

    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(conf_dir / "backends_ascend.conf")


def test_explicit_backend_config_overrides_the_marker(monkeypatch, fake_ascend_install):
    """FLAGOS_BACKEND_CONFIG is documented as always winning (advanced/testing
    use); the marker must not override an explicit choice."""
    monkeypatch.setenv("FLAGOS_BACKEND_CONFIG", "/tmp/explicit.conf")
    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == "/tmp/explicit.conf"
