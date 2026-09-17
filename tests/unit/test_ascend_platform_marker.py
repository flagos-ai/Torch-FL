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
fake install tree (no real ACL device needed), and pin the regression this fix
must not reintroduce: the marker branch runs *before* the /dev/davinci* branch,
so whatever conf the /dev probe would have chosen, the marker must choose too.

Retired opt-in variables must not divert the selection to a second conf:
backends_ascend.conf is now the only Ascend conf and already states the
FlagGems-first routing. The shadowing hazard remains worth pinning because the
branch order that caused it is unchanged.
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

    monkeypatch.setattr(torch_fl, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("FLAGOS_USE_FLAGGEMS_CPP", raising=False)
    monkeypatch.delenv("FLAGOS_USE_TILEOPS", raising=False)
    return conf_dir


def test_ascend_marker_selects_native_conf_by_default(fake_ascend_install):
    conf_dir = fake_ascend_install
    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(conf_dir / "backends_ascend.conf")


def test_ascend_marker_ignores_retired_opt_in_var(monkeypatch, fake_ascend_install):
    """The old per-platform opt-in var is retained as a no-op for backward
    compat. Setting it must not divert the selection to a second conf, since
    backends_ascend.conf is now the only Ascend conf and already states the
    FlagGems-first routing it used to switch between."""
    conf_dir = fake_ascend_install
    monkeypatch.setenv("FLAGOS_USE_VENDOR_OPS", "1")
    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(conf_dir / "backends_ascend.conf")


def test_ascend_marker_selects_conf_that_is_the_only_one_shipped(monkeypatch, tmp_path):
    """The install tree carries exactly one conf per platform; the marker must
    resolve against it without depending on a second hybrid file existing."""
    lib_dir = tmp_path / "lib"
    conf_dir = tmp_path / "configs"
    lib_dir.mkdir()
    conf_dir.mkdir()
    (lib_dir / "flagos_platform").write_text("ascend\n")
    (conf_dir / "backends_ascend.conf").write_text("")

    monkeypatch.setattr(torch_fl, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)

    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(conf_dir / "backends_ascend.conf")


def test_explicit_backend_config_overrides_the_marker(monkeypatch, fake_ascend_install):
    """FLAGOS_BACKEND_CONFIG is documented as always winning (advanced/testing
    use); the marker must not override an explicit choice."""
    monkeypatch.setenv("FLAGOS_BACKEND_CONFIG", "/tmp/explicit.conf")
    torch_fl._select_backend_config()
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == "/tmp/explicit.conf"
