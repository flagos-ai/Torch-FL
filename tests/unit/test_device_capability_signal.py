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

"""The capability predicate dispatch code gates on instead of ``tensor.is_cuda``.

Issue #515: ``is_cuda`` is False on flagos, so a caller choosing between
``torch.compile(flex_attention)`` and eager ``flex_attention`` by hand takes the
eager branch -- correct, and several times slower, with only a ``UserWarning``
to say so. ``fused_kernels_available()`` answers the question that gate means.

The property worth pinning is *which* signal it trusts. The pre-existing
``optional_integration_status()["compile"]`` records only that the
backend-registration imports succeeded; nothing on that path probes Triton, so
it reports ``active`` on a build whose vendor Triton backend is absent. These
tests therefore drive a case where registration succeeded but Triton is
unavailable, and require the answer to be False.

The functions are loaded out of ``torch_fl/__init__.py`` by AST extraction rather
than by importing torch_fl: the unit environment has no native ``_C``, and
``import torch_fl`` claims the PrivateUse1 dispatch key. Extracting only the
functions under test keeps this file runnable everywhere.

Run: pytest tests/unit/test_device_capability_signal.py -v
"""

import ast
import sys
import types
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "torch_fl" / "__init__.py"
_UNDER_TEST = ("fused_kernels_available", "device_capabilities", "is_flaggems_enabled")


def _load_functions():
    """Compile just the functions under test into a throwaway module."""
    tree = ast.parse(_SOURCE.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _UNDER_TEST
    ]
    assert {node.name for node in wanted} == set(_UNDER_TEST), (
        f"{_SOURCE} no longer defines {_UNDER_TEST}"
    )
    module = types.ModuleType("flagos_capability_under_test")
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), str(_SOURCE), "exec"),
        module.__dict__,
    )
    return module


class _Interface:
    """Stand-in for ``FlagOSDeviceInterface``, with its two relevant members."""

    def __init__(self, available=True, triton_error=None):
        self._available = available
        self._triton_error = triton_error
        self.asked_for = []

    def is_available(self):
        return self._available

    def raise_if_triton_unavailable(self, device=None):
        self.asked_for.append(device)
        if self._triton_error is not None:
            raise self._triton_error


@pytest.fixture()
def capability(monkeypatch):
    """The extracted module, with a stub ``torch_fl.compile.device_interface``.

    ``fused_kernels_available`` imports the interface inside the function body,
    so the stub is installed in ``sys.modules`` and torn down per test.
    """
    module = _load_functions()
    module._registered_ops = []
    module._optional_integration_state = {}

    def install(interface):
        pkg = types.ModuleType("torch_fl.compile")
        di = types.ModuleType("torch_fl.compile.device_interface")
        di.FlagOSDeviceInterface = interface
        monkeypatch.setitem(sys.modules, "torch_fl.compile", pkg)
        monkeypatch.setitem(sys.modules, "torch_fl.compile.device_interface", di)
        return interface

    module.install_interface = install
    return module


# ---------------------------------------------------------------------------
# fused_kernels_available: the signal it trusts
# ---------------------------------------------------------------------------


def test_true_when_compile_is_active_and_triton_answers(capability):
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface())
    assert capability.fused_kernels_available() is True


def test_false_when_compile_was_never_activated(capability):
    capability.install_interface(_Interface())
    assert capability.fused_kernels_available() is False


def test_false_when_compile_failed(capability):
    capability._optional_integration_state["compile"] = "failed: ImportError: triton"
    capability.install_interface(_Interface())
    assert capability.fused_kernels_available() is False


def test_false_when_triton_backend_is_missing_though_compile_is_active(capability):
    """The case ``optional_integration_status()`` gets wrong.

    Registration succeeded, so the pre-existing state says ``active``; the
    vendor Triton backend is absent, so no fused kernel can be built. The
    predicate has to disagree with the state it started from, or it is just a
    second spelling of it.
    """
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(
        _Interface(
            triton_error=RuntimeError(
                "triton not built with the 'maca' backend; install the vendor "
                "Triton runtime"
            )
        )
    )
    assert capability.fused_kernels_available() is False


def test_false_when_the_gpu_is_too_old_for_triton(capability):
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface(triton_error=RuntimeError("too old")))
    assert capability.fused_kernels_available() is False


def test_false_when_no_device_is_present(capability):
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface(available=False))
    assert capability.fused_kernels_available() is False


def test_no_device_short_circuits_before_the_triton_probe(capability):
    """With no device there is nothing to ask about, so do not ask."""
    capability._optional_integration_state["compile"] = "active"
    interface = capability.install_interface(_Interface(available=False))
    capability.fused_kernels_available()
    assert interface.asked_for == []


def test_false_when_the_interface_cannot_be_imported(capability):
    """A build with no compile subpackage answers False rather than raising."""
    capability._optional_integration_state["compile"] = "active"
    # No stub installed, so the in-body import fails.
    assert capability.fused_kernels_available() is False


def test_an_unexpected_error_is_not_propagated(capability):
    """A branch must never be turned into a crash by its own gate."""
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface(triton_error=ValueError("surprise")))
    assert capability.fused_kernels_available() is False


def test_the_device_argument_reaches_the_probe(capability):
    """Capability is per-device, so the argument must not be dropped."""
    capability._optional_integration_state["compile"] = "active"
    interface = capability.install_interface(_Interface())
    capability.fused_kernels_available("flagos:3")
    assert interface.asked_for == ["flagos:3"]


def test_default_device_is_passed_through_as_none(capability):
    capability._optional_integration_state["compile"] = "active"
    interface = capability.install_interface(_Interface())
    capability.fused_kernels_available()
    assert interface.asked_for == [None]


def test_returns_a_real_bool_not_a_truthy_object(capability):
    """Callers may compare with ``is True``; keep the type exact."""
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface())
    assert type(capability.fused_kernels_available()) is bool


# ---------------------------------------------------------------------------
# device_capabilities: the one-call dict
# ---------------------------------------------------------------------------


def test_capabilities_reports_every_documented_key(capability):
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface())
    caps = capability.device_capabilities()
    assert set(caps) >= {"compile", "fused_kernels", "fused_attention", "flaggems"}


def test_fused_attention_needs_both_compile_and_the_flex_hook(capability):
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface())
    assert capability.device_capabilities()["fused_attention"] is False
    capability._optional_integration_state["flex_attention"] = "active"
    assert capability.device_capabilities()["fused_attention"] is True


def test_fused_attention_is_false_when_triton_is_missing(capability):
    """The flex hook being installed does not make a kernel compilable."""
    capability._optional_integration_state["compile"] = "active"
    capability._optional_integration_state["flex_attention"] = "active"
    capability.install_interface(_Interface(triton_error=RuntimeError("no backend")))
    assert capability.device_capabilities()["fused_attention"] is False


def test_compile_key_mirrors_the_integration_state(capability):
    capability.install_interface(_Interface())
    assert capability.device_capabilities()["compile"] is False
    capability._optional_integration_state["compile"] = "active"
    assert capability.device_capabilities()["compile"] is True


def test_compile_true_with_fused_kernels_false_is_representable(capability):
    """The distinction the whole fix rests on has to survive the dict.

    ``compile`` says the backend registered; ``fused_kernels`` says a kernel can
    actually be built. Collapsing them would reintroduce #515's wrong answer.
    """
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface(triton_error=RuntimeError("no backend")))
    caps = capability.device_capabilities()
    assert caps["compile"] is True
    assert caps["fused_kernels"] is False


def test_flaggems_key_follows_the_registered_ops(capability):
    capability.install_interface(_Interface())
    assert capability.device_capabilities()["flaggems"] is False
    capability._registered_ops.append("aten::mm")
    assert capability.device_capabilities()["flaggems"] is True


def test_capabilities_values_are_all_bools(capability):
    capability._optional_integration_state["compile"] = "active"
    capability.install_interface(_Interface())
    caps = capability.device_capabilities()
    assert all(type(value) is bool for value in caps.values()), caps


def test_capabilities_forwards_the_device(capability):
    capability._optional_integration_state["compile"] = "active"
    interface = capability.install_interface(_Interface())
    capability.device_capabilities("flagos:2")
    assert interface.asked_for and set(interface.asked_for) == {"flagos:2"}


# ---------------------------------------------------------------------------
# The structural claim: the predicate must not be a synonym for the state
# ---------------------------------------------------------------------------


def test_predicate_consults_the_device_interface_not_only_the_state():
    """Pin the mechanism, so a later simplification cannot quietly undo the fix.

    If ``fused_kernels_available`` is ever reduced to reading
    ``_optional_integration_state`` alone, it becomes the signal issue #515
    showed to be wrong, and every test above that stubs a failing Triton probe
    would start passing for the wrong reason.
    """
    tree = ast.parse(_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fused_kernels_available"
    )
    called = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "raise_if_triton_unavailable" in called, (
        "fused_kernels_available no longer probes Triton; it would report a "
        "fusable device on a build with no vendor Triton backend (issue #515)"
    )
