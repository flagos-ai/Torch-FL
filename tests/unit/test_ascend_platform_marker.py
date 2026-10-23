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

"""Unit coverage for the conf the Ascend build record selects.

_torch_fl_select_backend_config() used to read the lib/flagos_platform marker
first and name the conf after it. It now reads the build record setup.py writes
(_build_config.py) and only falls back to the hardware probe, the bundle dir and
the record a wheel predates. These tests exercise torch_fl._select_backend_config()
directly against a fake install tree (no real ACL device needed), and pin the
regression the change exists for: a marker left in torch_fl/lib/ by another
platform's build must not decide the conf.

The selection is read back through torch_fl.backend_config_path(), not through
os.environ. _select_backend_config() used to write FLAGOS_BACKEND_CONFIG so the
C++ reader would find it, which made the wheel's own choice indistinguishable
from a user's for the rest of the process; it now hands the path over through
_C._set_backend_config_path() instead. These tests assert that directly: after a
selection the variable is still absent from the environment, and the accessor
reports the path.

Retired opt-in variables must not divert the selection to a second conf:
backends_ascend.conf is now the only Ascend conf and already states the
FlagGems-first routing.

Driving the selector directly rewrites the process-global
torch_fl._BACKEND_CONFIG_PATH, which monkeypatch does not own -- see the
restoring fixture in tests/unit/conftest.py and the leak check at the foot of
this file.
"""

import os

import pytest

import torch_fl


def _fake_tree(tmp_path, monkeypatch, accelerator, marker=""):
    """Point torch_fl.__file__ at a scratch tree with a build record and confs."""
    # _env and _platform each load the record beside their own module, so the
    # scratch tree carries both files.
    tree = tmp_path / "torch_fl"
    lib_dir = tree / "lib"
    conf_dir = tree / "configs"
    lib_dir.mkdir(parents=True)
    conf_dir.mkdir()
    record = f'ACCELERATOR = "{accelerator}"\nKERNELS = ()\n'
    (tree / "_build_config.py").write_text(record)
    (tree / "_platform_alias.py").write_text(
        f'ACCELERATOR = "{accelerator}"\nALIASES = ["{accelerator}"]\n'
    )
    if marker:
        (lib_dir / "flagos_platform").write_text(f"{marker}\n")
    (conf_dir / "backends_ascend.conf").write_text("")
    (conf_dir / "backends_apple.conf").write_text("")
    (conf_dir / "backends_metax.conf").write_text("")
    # The build record answers from _env.py's directory and is cached, so it is
    # read from the scratch tree here. conftest's _restore_build_record fixture
    # clears that cache after every test, so the real record comes back.
    env_module = torch_fl._env
    cached_lookup = env_module._build_record
    monkeypatch.setattr(env_module, "__file__", str(tree / "_env.py"))
    cached_lookup.cache_clear()
    cached_lookup()
    del cached_lookup
    # _select_backend_config() answers from _build_config.py beside the package
    # (torch_fl.__file__), and _platform.build_accelerator() resolves the
    # package with find_spec() -- which reads sys.modules for an imported
    # package, not sys.path -- so the record is pinned there for this test too.
    monkeypatch.setattr(torch_fl, "__file__", str(tree / "__init__.py"))
    monkeypatch.setattr(
        torch_fl._build_accelerator.__globals__["_platform"],
        "build_accelerator",
        lambda: accelerator,
    )
    monkeypatch.delenv("FLAGOS_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("FLAGOS_FORCE_BACKEND", raising=False)
    return conf_dir


@pytest.fixture
def fake_ascend_install(tmp_path, monkeypatch):
    """An Ascend build record that names no marker of its own."""
    return _fake_tree(tmp_path, monkeypatch, "ascend")


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
    """The install tree carries exactly one conf per platform; the record must
    resolve against it without depending on a second hybrid file existing."""
    conf_dir = _fake_tree(tmp_path, monkeypatch, "ascend")
    torch_fl._select_backend_config()
    assert torch_fl.backend_config_path() == str(conf_dir / "backends_ascend.conf")


def test_another_platforms_marker_does_not_decide_the_conf(monkeypatch, tmp_path):
    """The regression: a MetaX build inherited the "mps" marker the marker tests'
    scratch trees leave in torch_fl/lib/, read the empty backends_apple.conf, and
    raised "routed to 'none'" for every op that conf does not list -- 12 failures
    in test_allocator and 47 in the multi_device group on a real C550."""
    conf_dir = _fake_tree(tmp_path, monkeypatch, "metax", marker="mps")
    torch_fl._select_backend_config()
    assert torch_fl.backend_config_path() == str(conf_dir / "backends_metax.conf")


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
