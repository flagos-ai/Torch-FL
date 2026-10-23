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

"""The autoload guard's decision, for every spelling an install can answer in.

``test_startup_subprocess._require_declared_entry_point`` skips a wheel that
omits the ``torch.backends`` hook by design and fails any other wheel that
lacks it. It compared the accelerator it resolved -- the vendor spelling when
the build record carries one -- with ``_NO_AUTOLOAD``, which is keyed by table
id, so an MPS wheel was failed as "apple" instead of skipped as "mps". Every
lane whose install records its own vendor (iluvatar, tsingmicro) still passed,
which is why the iluvatar lane, not the apple one, reported it first.

The readers that answer with a spelling are replaced here, so the decision is
checked without a built wheel: only the table lookup, the skip and the
failure are under test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "test_startup_subprocess_under_test",
    Path(__file__).with_name("test_startup_subprocess.py"),
)
_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_test)

#: The platform table, read the way the install's `_platform.py` reads it.
_detector = _test._installed_platform_module()


def _unset(monkeypatch, name):
    """``FLAGOS_ACCELERATOR`` set, or absent -- the second is a source tree."""
    if name:
        monkeypatch.setenv("FLAGOS_ACCELERATOR", name)
    else:
        monkeypatch.delenv("FLAGOS_ACCELERATOR", raising=False)


@pytest.mark.parametrize(
    ("recorded", "environment"),
    [
        ("apple", ""),  # the MPS wheel: build record in the vendor spelling
        ("mps", ""),  # and in the table id, should setup.py ever write that
        ("", "apple"),  # a source tree built with FLAGOS_ACCELERATOR=apple
        ("", "mps"),
    ],
)
def test_a_wheel_that_omits_the_hook_by_design_is_skipped(
    monkeypatch, recorded, environment
):
    monkeypatch.setattr(_test, "_entry_point_hooks", lambda: False)
    monkeypatch.setattr(_test, "_build_accelerator", lambda: recorded)
    _unset(monkeypatch, environment)
    with pytest.raises(pytest.skip.Exception):
        _test._require_declared_entry_point()


@pytest.mark.parametrize(
    ("recorded", "environment"),
    [
        ("iluvatar", ""),  # a vendor that is also its table id
        ("nvidia", ""),  # a vendor whose table id is `cuda`
        ("cuda", ""),
        ("", "hygon"),
        ("", "dcu"),
    ],
)
def test_a_wheel_that_should_declare_it_but_does_not_fails(
    monkeypatch, recorded, environment
):
    monkeypatch.setattr(_test, "_entry_point_hooks", lambda: False)
    monkeypatch.setattr(_test, "_build_accelerator", lambda: recorded)
    _unset(monkeypatch, environment)
    with pytest.raises(pytest.fail.Exception):
        _test._require_declared_entry_point()


def test_an_install_that_names_no_accelerator_is_skipped(monkeypatch):
    monkeypatch.setattr(_test, "_entry_point_hooks", lambda: False)
    monkeypatch.setattr(_test, "_build_accelerator", lambda: "")
    monkeypatch.delenv("FLAGOS_ACCELERATOR", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _test._require_declared_entry_point()


def test_a_declared_hook_runs_the_test(monkeypatch):
    monkeypatch.setattr(_test, "_entry_point_hooks", lambda: True)
    monkeypatch.setattr(_test, "_build_accelerator", lambda: "nvidia")
    _test._require_declared_entry_point()


def test_the_omission_list_is_keyed_by_table_id():
    """A vendor-spelled key would make the skip miss the wheel it is about."""
    for accelerator in _test._NO_AUTOLOAD:
        assert _detector["platform_for"](accelerator) == accelerator, accelerator
