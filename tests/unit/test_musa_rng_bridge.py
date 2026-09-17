"""Unit coverage for the MUSA FlagGems RNG reservation bridge."""

import sys
import types

import pytest

import torch_fl


def _fallback_seed_offset(increment, generator=None):
    """Stand-in for the real FlagGems ``philox_backend_seed_offset``."""
    return 999, increment


def _install_fake_flag_gems(monkeypatch, original=_fallback_seed_offset):
    """Stand up the slice of ``flag_gems`` that the patch under test imports.

    ``original`` stands in for the real FlagGems ``philox_backend_seed_offset``.
    Returns the fake ``flag_gems.utils.random_utils`` module holding it.
    """
    random_utils = types.ModuleType("flag_gems.utils.random_utils")
    random_utils.torch_device_fn = types.SimpleNamespace(current_device=lambda: 3)
    random_utils.philox_backend_seed_offset = original
    utils = types.ModuleType("flag_gems.utils")
    utils.random_utils = random_utils
    flag_gems = types.ModuleType("flag_gems")
    flag_gems.__path__ = []
    flag_gems.utils = utils

    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setitem(sys.modules, "flag_gems.utils", utils)
    monkeypatch.setitem(sys.modules, "flag_gems.utils.random_utils", random_utils)
    monkeypatch.setattr(torch_fl, "_build_accelerator", lambda: "musa")
    return random_utils


@pytest.mark.skipif(
    not hasattr(torch_fl.flagos._C, "_reserve_rng_seed"),
    reason="requires the MUSA RNG bridge symbol",
)
def test_flaggems_philox_uses_flagos_reservations(monkeypatch):
    calls = []

    def reserve_seed(device, generator=None):
        calls.append((device, generator))
        return (1 << 63) + len(calls)

    random_utils = _install_fake_flag_gems(monkeypatch)
    monkeypatch.setattr(torch_fl.flagos._C, "_reserve_rng_seed", reserve_seed)

    torch_fl._patch_flaggems_philox()

    assert random_utils.philox_backend_seed_offset(128) == (-(1 << 63) + 1, 0)
    generator = types.SimpleNamespace(
        device=types.SimpleNamespace(type="flagos", index=2)
    )
    assert random_utils.philox_backend_seed_offset(256, generator) == (
        -(1 << 63) + 2,
        0,
    )
    assert calls == [(3, None), (2, generator)]


def test_flaggems_philox_preserves_non_flagos_generators(monkeypatch):
    sentinel = object()
    random_utils = _install_fake_flag_gems(
        monkeypatch,
        original=lambda increment, generator=None: (sentinel, increment),
    )

    torch_fl._patch_flaggems_philox()

    generator = types.SimpleNamespace(
        device=types.SimpleNamespace(type="cpu", index=None)
    )
    assert random_utils.philox_backend_seed_offset(17, generator) == (sentinel, 17)


@pytest.mark.skipif(
    not hasattr(torch_fl.flagos._C, "_reserve_rng_seed"),
    reason="requires the MUSA RNG bridge symbol",
)
def test_flaggems_philox_reaches_vendor_backend_modules(monkeypatch):
    """The module behind ``flag_gems.randn`` is named after the vendor, not flag_gems.

    FlagGems puts each backend's directory on ``sys.path`` and imports
    ``_<vendor>.ops`` from it, so the MUSA op modules are ``_mthreads.ops.*``.
    ``SpecOpRegistrar`` then republishes their entries as package-level
    attributes -- ``flag_gems.randn`` is ``_mthreads.ops.randn.randn`` -- and
    that is the module the generated kernel resolves to. A patch that filters on
    a ``flag_gems`` module-name prefix never reaches it.
    """
    random_utils = _install_fake_flag_gems(monkeypatch)
    vendor_randn = types.ModuleType("_mthreads.ops.randn")
    vendor_randn.philox_backend_seed_offset = random_utils.philox_backend_seed_offset
    monkeypatch.setitem(sys.modules, "_mthreads.ops.randn", vendor_randn)
    monkeypatch.setattr(
        torch_fl.flagos._C,
        "_reserve_rng_seed",
        lambda device, generator=None: (1 << 63) + 5,
    )

    torch_fl._patch_flaggems_philox()

    assert vendor_randn.philox_backend_seed_offset(128) == (-(1 << 63) + 5, 0)
