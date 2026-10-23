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
    never import it (tests/unit/test_import_phase_order.py parses the source in
    the AST precisely so it can avoid the import), and importing it here would
    change what they observe.
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


@pytest.fixture(autouse=True)
def _restore_build_record():
    """Clear the cached build record after every test that loaded a scratch one.

    torch_fl._env._build_record() is an lru_cache over the _build_config.py that
    sits beside _env.py. A test that points _env.__file__ at a scratch tree
    (tests/unit/test_ascend_platform_marker.py) loads that tree's record into the
    cache, and monkeypatch cannot put a cached_value back -- so the real record
    is re-read here instead. Without it, the next test in the process sees the
    scratch platform as this wheel's build.
    """
    torch_fl = sys.modules.get("torch_fl")
    env_module = getattr(torch_fl, "_env", None) if torch_fl else None
    try:
        yield
    finally:
        if env_module is not None and hasattr(env_module._build_record, "cache_clear"):
            env_module._build_record.cache_clear()
