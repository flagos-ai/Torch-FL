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

"""Unit coverage for the MUSA FlagTree driver binding in ``flagtree_shim``.

These run on any host: the point is the *wiring* -- that FlagTree's MThreads
driver ends up calling torch_fl instead of importing the torch_musa plugin. The
on-hardware behaviour is covered by tests/integration/test_compile.py.
"""

import sys
import types

import pytest

from torch_fl import flagos
from torch_fl.compile import flagtree_shim


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
    monkeypatch.setattr(flagtree_shim, "_musa_driver_bound", False)
    return FakeDriver


def test_binding_replaces_every_torch_musa_lookup(monkeypatch):
    """All four vendor runtime lookups must come from torch_fl after binding."""
    driver_cls = _fake_mthreads_backend(monkeypatch)

    assert flagtree_shim.bind_flagtree_musa_driver()

    driver = driver_cls()
    assert driver_cls.is_active is flagos.is_available
    assert driver.get_device_capability is flagtree_shim.get_musa_device_capability
    assert driver.get_current_stream is flagtree_shim.get_musa_current_raw_stream
    assert driver.get_current_device is flagos.current_device
    assert driver.set_current_device is flagos.set_device


def test_binding_is_idempotent(monkeypatch):
    driver_cls = _fake_mthreads_backend(monkeypatch)
    assert flagtree_shim.bind_flagtree_musa_driver()
    assert flagtree_shim.bind_flagtree_musa_driver()
    assert driver_cls.is_active is flagos.is_available


def test_binding_is_a_noop_without_the_mthreads_backend(monkeypatch):
    """A stock-Triton environment has nothing to bind, and that is not an error."""
    backends = types.ModuleType("triton.backends")
    backends.backends = {"nvidia": types.SimpleNamespace(driver=object)}
    triton = types.ModuleType("triton")
    triton.backends = backends
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.backends", backends)
    monkeypatch.setattr(flagtree_shim, "_musa_driver_bound", False)

    assert flagtree_shim.bind_flagtree_musa_driver() is False


def test_capability_comes_from_the_device(monkeypatch):
    """FlagTree derives its target arch and warp size from these two numbers."""
    monkeypatch.setattr(
        flagos,
        "get_device_properties",
        lambda idx: types.SimpleNamespace(major=3, minor=1),
    )
    monkeypatch.setattr(flagos, "current_device", lambda: 0)

    assert flagtree_shim.get_musa_device_capability() == (3, 1)
    assert flagtree_shim.get_musa_device_capability(1) == (3, 1)


def test_compile_path_does_not_import_the_torch_musa_plugin(monkeypatch):
    """The binding must not reach the plugin, even as a fabricated module.

    torch_fl does publish a small ``torch_musa`` compatibility surface for
    FlagGems discovery (``_install_musa_flaggems_compat``), so the claim here is
    narrower than "no such module ever exists": the FlagTree binding resolves
    every lookup through ``torch_fl.flagos`` and never consults that name.
    """
    driver_cls = _fake_mthreads_backend(monkeypatch)
    monkeypatch.delitem(sys.modules, "torch_musa", raising=False)

    assert flagtree_shim.bind_flagtree_musa_driver()
    driver_cls()

    assert "torch_musa" not in sys.modules
    bound = (
        driver_cls.is_active,
        driver_cls._get_device_capability,
        driver_cls._get_current_stream,
        driver_cls._get_current_device,
        driver_cls._set_current_device,
    )
    for func in bound:
        assert func.__module__.startswith("torch_fl"), func


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
