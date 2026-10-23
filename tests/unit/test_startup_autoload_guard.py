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

``test_startup_subprocess._require_declared_entry_point`` fails a wheel that
lacks the ``torch.backends`` hook and skips a checkout that names no
accelerator. The accelerator it reports comes from the build record or
``FLAGOS_ACCELERATOR``, either of which may hold a vendor spelling;
``platform_for()`` turns the vendor spelling into a table id, so the failure
says `dcu` rather than `hygon`.

The readers that answer with a spelling are replaced here, so the decision is
checked without a built wheel: only the skip, the failure and the message are
under test.
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


def _unset(monkeypatch, name):
    """``FLAGOS_ACCELERATOR`` set, or absent -- the second is a source tree."""
    if name:
        monkeypatch.setenv("FLAGOS_ACCELERATOR", name)
    else:
        monkeypatch.delenv("FLAGOS_ACCELERATOR", raising=False)


@pytest.mark.parametrize(
    ("recorded", "environment", "reported"),
    [
        ("apple", "", "mps"),  # the build record's vendor spelling
        ("", "apple", "mps"),  # FLAGOS_ACCELERATOR, on a source tree
        ("iluvatar", "", "iluvatar"),  # a vendor that is also its table id
        ("", "hygon", "dcu"),
        ("cuda", "", "cuda"),
    ],
)
def test_a_wheel_without_the_hook_fails_in_the_table_spelling(
    monkeypatch, recorded, environment, reported
):
    monkeypatch.setattr(_test, "_entry_point_hooks", lambda: False)
    monkeypatch.setattr(_test, "_build_accelerator", lambda: recorded)
    _unset(monkeypatch, environment)
    with pytest.raises(pytest.fail.Exception, match=f"the {reported} wheel"):
        _test._require_declared_entry_point()


def test_an_install_that_names_no_accelerator_is_skipped(monkeypatch):
    monkeypatch.setattr(_test, "_entry_point_hooks", lambda: False)
    monkeypatch.setattr(_test, "_build_accelerator", lambda: "")
    monkeypatch.delenv("FLAGOS_ACCELERATOR", raising=False)
    with pytest.raises(pytest.skip.Exception):
        _test._require_declared_entry_point()


def test_a_declared_hook_runs_the_test(monkeypatch):
    monkeypatch.setattr(_test, "_entry_point_hooks", lambda: True)
    monkeypatch.setattr(_test, "_build_accelerator", lambda: "apple")
    _test._require_declared_entry_point()
