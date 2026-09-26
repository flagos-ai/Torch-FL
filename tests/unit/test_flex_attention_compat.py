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

"""Unit tests for the flex-attention device-gate compatibility shim.

The patch is exercised against a stand-in module that carries upstream's own
gate, so the assertions are about the shim rather than about whichever torch the
runner happens to have installed. The real module is touched twice: once as a
drift guard on the assumption the shim is built on (upstream still refuses a
device outside its set), and once for idempotency. Neither mutates anything a
later test reads -- the shim is already installed by the time this file imports
``torch_fl``, which is the state the second test pins.
"""

from types import ModuleType, SimpleNamespace

import pytest

import torch_fl.compat.flex_attention as compat


#: Upstream's set, spelled out here rather than imported so the shim is tested
#: against the contract and not against the module it patches.
_UPSTREAM_DEVICES = ("cuda", "cpu", "xpu", "hpu")


def _fake_tensor(device_type):
    return SimpleNamespace(device=SimpleNamespace(type=device_type, index=0))


def _fake_flex_module():
    """A stand-in for ``torch.nn.attention.flex_attention`` with its own gate."""
    module = ModuleType(compat._MODULE)
    calls = {"validated": []}

    def _validate_device(query, key, value):
        calls["validated"].append(query.device.type)
        if query.device.type not in _UPSTREAM_DEVICES:
            raise ValueError(
                "FlexAttention is only supported on CUDA, CPU or HPU devices. "
                f"Found input tensors on {query.device.type} device."
            )

    module._validate_device = _validate_device
    return module, calls


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    """The disable switch is off unless a test asks otherwise."""
    monkeypatch.delenv(compat._DISABLE_ENV, raising=False)


# ---------------------------------------------------------------------------
# The patch itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_type", ["flagos", "privateuseone"])
def test_the_flagos_device_passes_the_gate(device_type):
    """It passes by not reaching upstream, which would refuse it.

    `calls` staying empty is the assertion: a shim that forwarded and swallowed
    the `ValueError` would look identical from the outside and would keep
    upstream's gate in the path for every other device.
    """
    module, calls = _fake_flex_module()
    assert compat._patch_validate_device(module) is True

    module._validate_device(_fake_tensor(device_type), None, None)
    assert calls["validated"] == []


@pytest.mark.parametrize("device_type", _UPSTREAM_DEVICES)
def test_a_device_upstream_admits_still_reaches_upstream(device_type):
    """The shim delegates rather than swallowing the check.

    A wrapper that returned early for everything would pass the test above and
    silently stop enforcing the gate for every other device.
    """
    module, calls = _fake_flex_module()
    compat._patch_validate_device(module)

    module._validate_device(_fake_tensor(device_type), None, None)
    assert calls["validated"] == [device_type]


def test_a_device_outside_the_set_is_still_refused():
    module, _ = _fake_flex_module()
    compat._patch_validate_device(module)

    with pytest.raises(ValueError, match="only supported on"):
        module._validate_device(_fake_tensor("mtia"), None, None)


def test_patching_twice_wraps_once():
    """The second call is a no-op, and the original stays reachable.

    A second wrap would still pass the gate, so the observable behaviour cannot
    tell the two apart -- the recorded original is what pins it, and it is what
    a future unpatch would need.
    """
    module, _ = _fake_flex_module()
    original = module._validate_device

    assert compat._patch_validate_device(module) is True
    first = module._validate_device
    assert compat._patch_validate_device(module) is True
    assert module._validate_device is first
    assert getattr(first, compat._ORIGINAL_ATTR) is original
    assert not getattr(original, compat._PATCHED_ATTR, False)


def test_a_module_without_the_gate_is_reported_not_raised():
    """Upstream removing ``_validate_device`` is a result, not a crash."""
    module = ModuleType(compat._MODULE)
    assert compat._patch_validate_device(module) is False


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def test_install_patches_an_importable_module(monkeypatch):
    module, _ = _fake_flex_module()
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: module)

    assert compat.install_flex_attention_compat() is True
    assert getattr(module._validate_device, compat._PATCHED_ATTR, False)


def test_install_is_a_noop_when_the_module_is_absent(monkeypatch):
    def _missing(name):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(compat.importlib, "import_module", _missing)
    assert compat.patch_flex_attention() is False


def test_the_disable_switch_installs_nothing(monkeypatch):
    module, _ = _fake_flex_module()
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: module)
    monkeypatch.setenv(compat._DISABLE_ENV, "1")

    assert compat.is_flex_attention_compat_available() is False
    assert compat.install_flex_attention_compat() is False
    assert not getattr(module._validate_device, compat._PATCHED_ATTR, False)


def test_the_disable_switch_is_read_at_call_time(monkeypatch):
    """Checking availability after a late export is still effective."""
    monkeypatch.delenv(compat._DISABLE_ENV, raising=False)
    assert compat.is_flex_attention_compat_available() is True
    monkeypatch.setenv(compat._DISABLE_ENV, "1")
    assert compat.is_flex_attention_compat_available() is False


# ---------------------------------------------------------------------------
# The upstream assumption
# ---------------------------------------------------------------------------


def test_upstream_still_refuses_a_device_outside_its_set():
    """Drift guard on the assumption this shim exists for.

    Reads the recorded original when the shim is already installed in this
    process, so the assertion is about upstream's function either way.
    """
    flex = pytest.importorskip("torch.nn.attention.flex_attention")
    validate = getattr(flex, "_validate_device", None)
    if validate is None:
        pytest.skip("upstream has dropped the device gate; this shim is obsolete")

    original = getattr(validate, compat._ORIGINAL_ATTR, validate)
    with pytest.raises(ValueError, match="only supported on"):
        original(_fake_tensor("flagos"), None, None)


def test_the_real_module_ends_up_patched_and_patched_once():
    """``import torch_fl`` installs the shim, and a second call changes nothing."""
    flex = pytest.importorskip("torch.nn.attention.flex_attention")
    validate = flex._validate_device
    assert getattr(validate, compat._PATCHED_ATTR, False)

    fresh = compat.install_flex_attention_compat()
    if fresh is True:
        assert flex._validate_device is validate
