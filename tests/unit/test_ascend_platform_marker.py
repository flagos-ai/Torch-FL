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

The selection is read back through torch_fl.backend_config_path(), not through
os.environ. _select_backend_config() used to write FLAGOS_BACKEND_CONFIG so the
C++ reader would find it, which made the wheel's own choice indistinguishable
from a user's for the rest of the process; it now hands the path over through
_C._set_backend_config_path() instead. These tests assert that directly: after a
selection the variable is still absent from the environment, and the accessor
reports the path.

Retired opt-in variables must not divert the selection to a second conf:
backends_ascend.conf is now the only Ascend conf and already states the
FlagGems-first routing. The shadowing hazard remains worth pinning because the
branch order that caused it is unchanged.

Driving the selector directly rewrites the process-global
torch_fl._BACKEND_CONFIG_PATH, which monkeypatch does not own -- see the
restoring fixture in tests/unit/conftest.py and the leak check at the foot of
this file.
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
    monkeypatch.delenv("FLAGOS_FORCE_BACKEND", raising=False)
    return conf_dir


def test_ascend_marker_selects_native_conf_by_default(fake_ascend_install):
    conf_dir = fake_ascend_install
    torch_fl._select_backend_config()
    assert torch_fl.backend_config_path() == str(conf_dir / "backends_ascend.conf")
    assert "FLAGOS_BACKEND_CONFIG" not in os.environ


def test_ascend_marker_ignores_retired_opt_in_var(monkeypatch, fake_ascend_install):
    """The old per-platform opt-in var is retained as a no-op for backward
    compat. Setting it must not divert the selection to a second conf, since
    backends_ascend.conf is now the only Ascend conf and already states the
    FlagGems-first routing it used to switch between."""
    conf_dir = fake_ascend_install
    monkeypatch.setenv("FLAGOS_USE_VENDOR_OPS", "1")
    torch_fl._select_backend_config()
    assert torch_fl.backend_config_path() == str(conf_dir / "backends_ascend.conf")


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
    assert torch_fl.backend_config_path() == str(conf_dir / "backends_ascend.conf")


def test_explicit_backend_config_overrides_the_marker(
    monkeypatch, fake_ascend_install, tmp_path
):
    """FLAGOS_BACKEND_CONFIG is documented as always winning (advanced/testing
    use); the marker must not override an explicit choice, and the user's value
        must be left exactly as they set it.

        The override names a real file because an unreadable one is now an error
        rather than an override -- see
        test_explicit_backend_config_must_be_readable.
    """
    explicit = tmp_path / "explicit.conf"
    explicit.write_text("mm = cuda\n")
    monkeypatch.setenv("FLAGOS_BACKEND_CONFIG", str(explicit))
    torch_fl._select_backend_config()
    assert torch_fl.backend_config_path() == str(explicit)
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(explicit)


def test_explicit_backend_config_must_be_readable(
    monkeypatch, fake_ascend_install, tmp_path
):
    """An override that will not open is fatal, not a silent fallback.

    csrc/aten/common.cc leaves its routing table empty when the conf file will
    not open, and an empty table resolves every op to Backend::kFlagGems -- so
    a typo used to send the entire workload down the slowest route behind a
    single stderr line, and to switch off the FlagGems process-level setup that
    reads the same file. The marker branch must not quietly fall through to
    backends_ascend.conf either.
    """
    missing = tmp_path / "does-not-exist.conf"
    monkeypatch.setenv("FLAGOS_BACKEND_CONFIG", str(missing))
    with pytest.raises(RuntimeError, match="cannot be read"):
        torch_fl._select_backend_config()
    # Nothing may have been selected behind the failure: the environment still
    # names what the user set, and nothing was handed to the C++ reader.
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == str(missing)
    assert torch_fl.backend_config_path() == str(missing)


def test_explicit_relative_backend_config_is_resolved_to_an_absolute_path(
    monkeypatch, tmp_path
):
    """A relative override is resolved at import, not at first dispatch.

    The C++ loader reads the variable lazily, on the first op dispatch, so a
    relative path would otherwise be resolved against whatever the cwd happens
    to be at that moment rather than against the one the caller had. The
    resolved path is handed to the C++ reader through _BACKEND_CONFIG_PATH, not
    written back into os.environ; the environment keeps the user's own value.
    """
    (tmp_path / "custom.conf").write_text("mm = cuda\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLAGOS_BACKEND_CONFIG", "custom.conf")
    torch_fl._select_backend_config()
    assert torch_fl.backend_config_path() == str(tmp_path / "custom.conf")
    assert os.environ["FLAGOS_BACKEND_CONFIG"] == "custom.conf"


def test_selection_does_not_leak_into_the_next_test():
    """The selector must leave nothing behind for the rest of the process.

    Every test above calls _select_backend_config(), which rewrites the
    module-global torch_fl._BACKEND_CONFIG_PATH -- the last call here even clears
    it, because the FLAGOS_BACKEND_CONFIG early return in
    test_explicit_backend_config_overrides_the_marker runs after the reset at the
    top of the function. Nothing in these tests owns that global, so without the
    restoring fixture in tests/unit/conftest.py it stays empty for every test that
    runs afterwards.

    That is not academic: tests/unit/test_musa_rng_bridge.py reads the global
    through _conf_routes_to_flaggems(), and an empty path closes the FlagGems
    Philox gate, so its three symbol-gated tests fail in a whole-suite run while
    passing on their own. Assert the resolved conf is still there, which holds
    only because the fixture restores the import-time selection at teardown.
    """
    assert torch_fl.backend_config_path(), (
        "a scratch-tree selection leaked into the process global; "
        "the restoring fixture in tests/unit/conftest.py is missing or broken"
    )
