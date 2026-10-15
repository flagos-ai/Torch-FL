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

"""Unit coverage for when flagos decides it may install the FlagGems patches.

Every patch in ``torch_fl/flagos/__init__.py`` has to be installed before the
first routed op, because the C++ bridge resolves ``flag_gems.<name>`` once and
caches the callable. They are also installed from a window that can open *while
flag_gems is importing*: GCU's runtime shims and MUSA's Philox bridge both
import flag_gems during ``import torch_fl``, and triton asks for the active
device on the way through, which runs ``torch::utils::device_lazy_init`` ->
``torch_flagos`` -> ``_lazy_init`` from inside ``flag_gems/__init__.py``.

At that moment the op names the patches look for do not exist yet, so every
patch's ``getattr(flag_gems, name, None) is None`` guard reads "this build never
had the op". The distinction between a missing name and an unfinished import is
what ``_flaggems_op_names_are_bound`` exists to draw, and the install has to
decline to latch until it can see a finished module -- otherwise ``slice_backward``
would stay unpatched on the platforms that import flag_gems eagerly, while
passing on every platform that imports it later.

Both decisions are pure Python and both fail silently, so they are covered here
rather than left to the hardware lanes.
"""

import sys
import types

import pytest

from torch_fl import flagos


PATCH_FUNCTIONS = (
    "_align_flaggems_device_identity",
    "_patch_flaggems_vector_norm",
    "_patch_flaggems_pow_scalar_square",
    "_patch_flaggems_mean_last_dim",
    "_patch_flaggems_index_put",
    "_patch_flaggems_pointwise_dispatch",
    "_patch_flaggems_tune_space",
    "_patch_flaggems_where_out",
    "_patch_flaggems_slice_backward",
)


def _fake_flag_gems(initializing: bool):
    """A stand-in for ``flag_gems`` that says how far its import got."""
    module = types.ModuleType("flag_gems")
    module.__spec__ = types.SimpleNamespace(_initializing=initializing)
    return module


@pytest.fixture
def unlatched(monkeypatch):
    monkeypatch.setattr(flagos, "_flaggems_patches_installed", False)


@pytest.fixture
def recorded(monkeypatch, unlatched):
    """Replace every patch with a recorder, and return the call log."""
    calls = []
    for name in PATCH_FUNCTIONS:
        monkeypatch.setattr(flagos, name, lambda _name=name: calls.append(_name))
    return calls


class TestOpNamesAreBound:
    def test_module_absent_is_not_ready(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "flag_gems", raising=False)
        assert flagos._flaggems_op_names_are_bound() is False

    def test_module_still_importing_is_not_ready(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flag_gems", _fake_flag_gems(True))
        assert flagos._flaggems_op_names_are_bound() is False

    def test_finished_import_is_ready(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flag_gems", _fake_flag_gems(False))
        assert flagos._flaggems_op_names_are_bound() is True


class TestInstallLatchesOnlyAgainstAFinishedImport:
    def test_a_half_imported_module_installs_nothing_and_latches_nothing(
        self, monkeypatch, recorded
    ):
        monkeypatch.setitem(sys.modules, "flag_gems", _fake_flag_gems(True))

        assert flagos._install_flaggems_patches() is False
        assert recorded == []
        assert flagos._flaggems_patches_installed is False

    def test_the_next_call_after_the_import_installs_every_patch(
        self, monkeypatch, recorded
    ):
        module = _fake_flag_gems(True)
        monkeypatch.setitem(sys.modules, "flag_gems", module)
        assert flagos._install_flaggems_patches() is False

        module.__spec__._initializing = False

        assert flagos._install_flaggems_patches() is True
        assert recorded == list(PATCH_FUNCTIONS)
        assert flagos._flaggems_patches_installed is True

    def test_an_installed_set_is_not_installed_again(self, monkeypatch, recorded):
        monkeypatch.setitem(sys.modules, "flag_gems", _fake_flag_gems(False))
        assert flagos._install_flaggems_patches() is True

        recorded.clear()

        assert flagos._install_flaggems_patches() is True
        assert recorded == []


class TestEveryCallerRetries:
    def test_lazy_init_retries_on_the_already_initialized_path(
        self, monkeypatch, recorded
    ):
        monkeypatch.setitem(sys.modules, "flag_gems", _fake_flag_gems(False))
        monkeypatch.setattr(flagos, "_initialized", True)
        monkeypatch.setattr(
            flagos._C, "_init", lambda: pytest.fail("device init ran a second time")
        )

        flagos._lazy_init()

        assert recorded == list(PATCH_FUNCTIONS)

    def test_init_retries_after_lazy_init(self, monkeypatch, recorded):
        monkeypatch.setitem(sys.modules, "flag_gems", _fake_flag_gems(False))
        monkeypatch.setattr(flagos, "_lazy_init", lambda: None)

        flagos.init()

        assert recorded == list(PATCH_FUNCTIONS)
