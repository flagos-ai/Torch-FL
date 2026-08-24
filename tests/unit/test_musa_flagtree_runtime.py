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

"""Unit coverage for the MUSA FlagTree runtime binding.

These run on any host: the point is the *wiring* -- that FlagTree's MThreads
driver ends up calling torch_fl instead of the torch_musa plugin. The
on-hardware behaviour is covered by tests/integration/test_compile.py.
"""

import sys
import types

import pytest

from torch_fl.compile import musa_runtime


def _fake_mthreads_backend(monkeypatch):
    """A stand-in for triton.backends with an MThreads-shaped driver.

    Mirrors the vendor driver's structure: ``__init__`` copies the class-level
    getters onto the instance, which is why the binding has to happen before any
    driver is constructed.
    """

    class FakeDriver:
        @staticmethod
        def is_active():
            raise AssertionError("vendor is_active() must have been rebound")

        def __init__(self):
            self.get_device_capability = self._get_device_capability
            self.get_current_stream = self._get_current_stream
            self.get_current_device = self._get_current_device
            self.set_current_device = self._set_current_device

        @staticmethod
        def _get_device_capability(device):
            raise AssertionError("vendor capability getter must have been rebound")

        @staticmethod
        def _get_current_stream(idx):
            raise AssertionError("vendor stream getter must have been rebound")

        @staticmethod
        def _get_current_device():
            raise AssertionError("vendor device getter must have been rebound")

        @staticmethod
        def _set_current_device(device):
            raise AssertionError("vendor set_device must have been rebound")

    backends = types.ModuleType("triton.backends")
    backends.backends = {"mthreads": types.SimpleNamespace(driver=FakeDriver)}
    triton = types.ModuleType("triton")
    triton.backends = backends
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.backends", backends)
    monkeypatch.setattr(musa_runtime, "_bound", False)
    return FakeDriver


def test_binding_replaces_every_torch_musa_lookup(monkeypatch):
    """All four vendor runtime lookups must come from torch_fl after binding."""
    driver_cls = _fake_mthreads_backend(monkeypatch)
    monkeypatch.setattr(musa_runtime.flagos, "device_count", lambda: 1)
    monkeypatch.setattr(musa_runtime.flagos, "current_device", lambda: 0)

    assert musa_runtime.bind_flagtree_musa_driver()

    driver = driver_cls()
    assert driver_cls.is_active is musa_runtime.is_available
    assert driver.get_device_capability is musa_runtime.get_device_capability
    assert driver.get_current_stream is musa_runtime.get_current_raw_stream
    assert driver.get_current_device is musa_runtime.current_device
    assert driver.set_current_device is musa_runtime.set_device


def test_binding_is_idempotent(monkeypatch):
    driver_cls = _fake_mthreads_backend(monkeypatch)
    assert musa_runtime.bind_flagtree_musa_driver()
    assert musa_runtime.bind_flagtree_musa_driver()
    assert driver_cls.is_active is musa_runtime.is_available


def test_binding_is_a_noop_without_the_mthreads_backend(monkeypatch):
    """A stock-Triton environment has nothing to bind, and that is not an error."""
    backends = types.ModuleType("triton.backends")
    backends.backends = {"nvidia": types.SimpleNamespace(driver=object)}
    triton = types.ModuleType("triton")
    triton.backends = backends
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.backends", backends)
    monkeypatch.setattr(musa_runtime, "_bound", False)

    assert musa_runtime.bind_flagtree_musa_driver() is False


def test_capability_comes_from_the_device(monkeypatch):
    """FlagTree derives its target arch and warp size from these two numbers."""
    monkeypatch.setattr(
        musa_runtime.flagos,
        "get_device_properties",
        lambda idx: types.SimpleNamespace(major=3, minor=1),
    )
    monkeypatch.setattr(musa_runtime.flagos, "current_device", lambda: 0)

    assert musa_runtime.get_device_capability() == (3, 1)


@pytest.mark.parametrize(
    "given, expected",
    [(None, 2), (1, 1), ("flagos:1", 1), ("flagos", 2)],
)
def test_device_index_normalization(monkeypatch, given, expected):
    """The vendor runtime takes ints; Inductor passes str/torch.device/None."""
    monkeypatch.setattr(musa_runtime.flagos, "current_device", lambda: 2)

    assert musa_runtime._device_index(given) == expected


def test_no_torch_musa_module_is_created():
    """The whole point: torch_fl must never publish a torch_musa stand-in.

    A fabricated module would let the vendor driver "work" while reading from
    something other than the runtime that owns PrivateUse1.
    """
    import torch_fl  # noqa: F401

    assert "torch_musa" not in sys.modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
