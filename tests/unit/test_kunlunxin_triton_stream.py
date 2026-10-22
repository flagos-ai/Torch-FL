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

"""Kunlunxin's Triton stream binding, exercised against a stand-in driver.

The defect these tests pin is a property of Triton's own class hierarchy:
``GPUDriver.__init__`` installs

    self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream

when ``torch._C._cuda_getCurrentRawStream`` is absent -- which it is on the
CPU-torch wheel Kunlunxin runs on -- and FlagTree's ``XPUDriver`` overrides the
device pair but never the stream. The first kernel launch therefore dies in
``torch.cuda._lazy_init``. Reproducing that needs no accelerator, so the checks
below drive ``_bind_stream_on_driver`` and ``_bind_kunlunxin_triton_stream``
against synthetic driver objects with the same shape.

Measured on a P800 (torch 2.10.0+cpu, triton 3.6.0): with the parent's lambda a
plain ``tl.load``/``tl.store`` kernel raises ``AssertionError: Torch not compiled
with CUDA enabled``; with the binding in place it launches, synchronizes and
reads back ``min == max == 2.0`` for ``ones + ones``.

The functions are loaded out of ``torch_fl/flagos/__init__.py`` by AST extraction
rather than by importing torch_fl: the unit environment has no torch, and
``import torch_fl`` claims the PrivateUse1 dispatch key. Extracting only the two
functions under test keeps this file runnable everywhere.

Run: pytest tests/unit/test_kunlunxin_triton_stream.py -v
"""

import ast
import sys
import types
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "torch_fl" / "flagos" / "__init__.py"
_UNDER_TEST = ("_bind_stream_on_driver", "_bind_kunlunxin_triton_stream")


def _load_functions():
    """Compile just the two functions under test into a throwaway module."""
    tree = ast.parse(_SOURCE.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _UNDER_TEST
    ]
    assert {node.name for node in wanted} == set(_UNDER_TEST), (
        f"{_SOURCE} no longer defines {_UNDER_TEST}"
    )
    module = types.ModuleType("kunlunxin_stream_under_test")
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(_SOURCE), "exec"),
        module.__dict__,
    )
    # ``_bind_stream_on_driver`` resolves ``current_stream`` through
    # ``sys.modules[__name__]``, so the module has to be registered under the
    # name its own ``__name__`` reports.
    sys.modules[module.__name__] = module
    return module


@pytest.fixture()
def flagos():
    """The extracted module, with the collaborators the functions reach for."""
    module = _load_functions()
    module._platform = lambda: "kunlunxin"
    module.current_stream = lambda index: types.SimpleNamespace(handle=0)
    return module


class _FallbackDriver:
    """A driver carrying the parent class's fallback lambda, as XPUDriver does."""

    def __init__(self):
        self.get_current_stream = _fallback_lambda()


def _fallback_lambda():
    """A lambda whose ``__qualname__`` is exactly Triton's fallback's.

    Triton defines it inside ``GPUDriver.__init__``, so the qualname is
    ``GPUDriver.__init__.<locals>.<lambda>`` and that string is what the binder
    keys on. A plain module-level lambda cannot carry that name, so the value is
    built with the real class body below and then relabelled.
    """

    class GPUDriver:
        def __init__(self):
            def torch_cuda_current_stream(idx):
                raise AssertionError("torch.cuda.current_stream must never be called")

            self.get_current_stream = lambda idx: (
                torch_cuda_current_stream(idx).cuda_stream
            )

    fn = GPUDriver().get_current_stream
    # The real qualname is produced by the class body; rename to the string the
    # binder compares against so a Triton upgrade that renames it fails here.
    fn.__qualname__ = "GPUDriver.__init__.<locals>.<lambda>"
    return fn


class _OwnStreamDriver:
    """A driver that grew a real implementation; the binder must leave it."""

    def __init__(self):
        self.get_current_stream = lambda idx: 4242


class _DriverConfig:
    """Mirror of triton.runtime.driver.DriverConfig's lazy ``active``.

    ``_flagos_stream_bound`` is set on the class by the binder, so each instance
    carries its own class and cannot leak the marker into the next test.
    """

    def __init__(self, factory):
        self._factory = factory
        self._active = None
        self.created = 0

    @property
    def default(self):
        self.created += 1
        return self._factory()

    @property
    def active(self):
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver):
        self._active = driver

    def reset_active(self):
        self._active = self.default


def _assert_bound(driver):
    """The accessor must no longer be the parent's fallback lambda."""
    owner = getattr(driver.get_current_stream, "__qualname__", "")
    assert "<lambda>" not in owner, owner
    assert "_current_raw_stream" in owner, owner


def test_the_fallback_lambda_is_what_we_key_on():
    """Guards the discriminator: Triton's fallback must match the string used."""
    driver = _FallbackDriver()
    assert "GPUDriver.__init__" in driver.get_current_stream.__qualname__


def test_bind_replaces_the_parent_fallback(flagos):
    driver = _FallbackDriver()
    bound = flagos._bind_stream_on_driver(driver)
    assert bound is driver
    _assert_bound(bound)


def test_bind_is_idempotent(flagos):
    driver = _FallbackDriver()
    first = flagos._bind_stream_on_driver(driver).get_current_stream
    second = flagos._bind_stream_on_driver(driver).get_current_stream
    assert first is second


def test_bind_leaves_a_real_implementation_alone(flagos):
    driver = _OwnStreamDriver()
    flagos._bind_stream_on_driver(driver)
    assert driver.get_current_stream(0) == 4242


def test_bound_accessor_reads_torch_fl_current_stream(flagos):
    """The replacement must answer from torch_fl's handle, never from torch.cuda."""
    flagos.current_stream = lambda index: types.SimpleNamespace(handle=7)
    driver = flagos._bind_stream_on_driver(_FallbackDriver())
    assert driver.get_current_stream(0) == 7
    assert driver.get_current_stream(3) == 7


def test_bound_accessor_returns_an_int(flagos):
    flagos.current_stream = lambda index: types.SimpleNamespace(handle="0")
    driver = flagos._bind_stream_on_driver(_FallbackDriver())
    assert driver.get_current_stream(0) == 0
    assert isinstance(driver.get_current_stream(0), int)


def test_bind_handles_none_driver(flagos):
    assert flagos._bind_stream_on_driver(None) is None


@pytest.fixture()
def fake_triton_driver(monkeypatch):
    """Install a synthetic ``triton.runtime.driver`` for the module-level binder.

    ``import triton.runtime.driver as X`` normally binds the *submodule*, but on
    the real wheel ``triton/runtime/__init__.py`` rebinds that name to the
    ``DriverConfig`` instance, so X is the instance. Reproducing that here is the
    point of the fixture: no ``triton.runtime.driver`` submodule is registered,
    only the package attribute.
    """
    config = _DriverConfig(_FallbackDriver)
    instance = config
    instance.__class__ = type("TestDriverConfig", (_DriverConfig,), {})

    runtime = types.ModuleType("triton.runtime")
    runtime.driver = instance
    triton = types.ModuleType("triton")
    triton.runtime = runtime

    # A real submodule has to exist for ``import triton.runtime.driver as X`` to
    # succeed; on the wheel X then comes back as the package attribute, which is
    # the DriverConfig. Both are pointed at the same instance here.
    submodule = types.ModuleType("triton.runtime.driver")
    submodule.driver = instance

    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.driver", submodule)
    return instance


def test_lazy_active_is_bound_on_first_read(flagos, monkeypatch, fake_triton_driver):
    """The driver is built lazily, so the patch has to be applied lazily too."""
    config = fake_triton_driver
    assert not getattr(type(config), "_flagos_stream_bound", False)

    flagos._bind_kunlunxin_triton_stream()

    assert config.created == 0, "the binder must not force driver creation"
    driver = config.active
    assert config.created == 1
    _assert_bound(driver)


def test_set_active_binds_a_substituted_driver(flagos, fake_triton_driver):
    flagos._bind_kunlunxin_triton_stream()

    replacement = _FallbackDriver()
    fake_triton_driver.set_active(replacement)

    _assert_bound(fake_triton_driver.active)


def test_noop_off_kunlunxin(flagos, fake_triton_driver):
    flagos._platform = lambda: "ascend"
    flagos._bind_kunlunxin_triton_stream()

    assert not getattr(type(fake_triton_driver), "_flagos_stream_bound", False)
    driver = fake_triton_driver.active
    assert "GPUDriver.__init__" in driver.get_current_stream.__qualname__


def test_idempotent_class_patch(flagos, fake_triton_driver):
    flagos._bind_kunlunxin_triton_stream()
    cls = type(fake_triton_driver)
    patched = cls.active
    flagos._bind_kunlunxin_triton_stream()
    assert cls.active is patched
