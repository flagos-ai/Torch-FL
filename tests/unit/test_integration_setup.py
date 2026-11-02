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

"""Cover accelerator setup in tests/integration/conftest.py without torch."""

import importlib.util
import types
from pathlib import Path

import pytest


_INTEGRATION_CONFTEST = (
    Path(__file__).resolve().parents[1] / "integration" / "conftest.py"
)


def _load_integration_conftest():
    spec = importlib.util.spec_from_file_location(
        "integration_conftest_under_test", _INTEGRATION_CONFTEST
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integration_conftest = _load_integration_conftest()


def test_device_setup_rejects_unavailable_runtime_before_init():
    initialized = []

    with pytest.raises(
        RuntimeError, match=r"torch_fl\.flagos\.is_available\(\) returned False"
    ):
        integration_conftest._require_flagos_device(
            lambda: False, lambda: initialized.append(True)
        )

    assert initialized == []


def test_device_setup_propagates_initialization_failure():
    def fail_init():
        raise RuntimeError("driver initialization failed")

    with pytest.raises(RuntimeError, match="driver initialization failed"):
        integration_conftest._require_flagos_device(lambda: True, fail_init)


def test_device_setup_initializes_once_when_ready():
    initialized = []

    integration_conftest._require_flagos_device(
        lambda: True, lambda: initialized.append(True)
    )

    assert initialized == [True]


def test_configure_reports_setup_failure_as_usage_error(monkeypatch):
    def unavailable():
        raise RuntimeError("device initialization failed")

    monkeypatch.setattr(integration_conftest, "_initialize_flagos_device", unavailable)
    config = types.SimpleNamespace(addinivalue_line=lambda *_: None)

    with pytest.raises(
        pytest.UsageError, match="device initialization failed"
    ) as error:
        integration_conftest.pytest_configure(config)

    assert "RuntimeError: device initialization failed" in str(error.value)
    for issue in ("#710", "#711"):
        assert issue not in str(error.value)
    assert not isinstance(error.value, pytest.skip.Exception)
