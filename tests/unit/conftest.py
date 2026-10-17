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

"""Fixtures shared by the whole unit suite."""

import sys

import pytest

try:
    # Imported for its side effect, and here rather than in each test module
    # because the order is the whole point. torch's device-backend autoload
    # imports torch_fl from *inside* ``import torch``, and that arrives too late
    # on PPU: torch_fl has to relink the vendor libtorch core before torch
    # starts, so a module that reaches ``import torch`` first fails the entire
    # file -- every test in it -- at collection. Every unit test module does
    # that at module scope, and this conftest is imported before any of them.
    import torch_fl  # noqa: F401
except ImportError:
    # No built wheel in this checkout (there is no ``torch_fl._C``). This
    # import exists for ordering, not for coverage, and the tests that need a
    # wheel skip themselves through ``importorskip``.
    pass


@pytest.fixture(autouse=True)
def _restore_backend_config_path():
    """Put the conf torch_fl resolved at import time back after every test.

    torch_fl._select_backend_config() records its pick in the module-global
    torch_fl._BACKEND_CONFIG_PATH, and clears that global at the top of each call.
    A test that drives the function directly against a scratch tree
    (tests/unit/test_ascend_platform_marker.py) therefore overwrites the value the
    import-time call left behind, and -- when its last call early-returns on
    FLAGOS_BACKEND_CONFIG -- leaves an empty string in its place. monkeypatch does
    not own that global, so nothing else puts it back.

    Everything that reads the global afterwards then sees a scratch path pytest
    deleted with tmp_path, or nothing at all. tests/unit/test_musa_rng_bridge.py
    reads it indirectly: _conf_routes_to_flaggems() answers False, the FlagGems
    Philox bridge is never patched, and its three symbol-gated tests fail -- but
    only when the whole suite runs, never when the file is run on its own, which
    is why the leak went unnoticed.

    Only act when torch_fl is already imported. Several unit tests deliberately
    observe the module without it (tests/unit/test_import_phase_order.py parses
    the source in the AST precisely so it does not have to execute it, and the
    tests here that build a stub torch_fl pop the real one first).
    """
    torch_fl = sys.modules.get("torch_fl")
    if torch_fl is None:
        yield
        return

    saved = getattr(torch_fl, "_BACKEND_CONFIG_PATH", None)
    try:
        yield
    finally:
        if saved is not None:
            torch_fl._BACKEND_CONFIG_PATH = saved
