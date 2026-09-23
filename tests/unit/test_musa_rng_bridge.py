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

    install_modules(
        monkeypatch,
        {
            "flag_gems": flag_gems,
            "flag_gems.utils": utils,
            "flag_gems.utils.random_utils": random_utils,
        },
    )
    monkeypatch.setattr(torch_fl, "_build_accelerator", lambda: "musa")
    return random_utils


def install_modules(monkeypatch, modules):
    """Insert ``modules`` at the tail of ``sys.modules``, in the given order.

    A plain ``monkeypatch.setitem`` overwrites in place, so a key that some
    earlier import already claimed keeps that import's position in
    ``sys.modules`` iteration order. The bridge sweeps that order, and the
    defect under test is positional: entries appended *after* the flag_gems
    modules cost nothing, because the sweep has already rebound them by the time
    it reaches one. The harness reaches the defect through the opposite order --
    it imports transformers first and torch_fl last, so flag_gems' modules are
    only loaded once the lazy modules that raise are already in the table.
    Deleting first reproduces that: every name passed here is swept after
    everything inserted before this call.
    """
    for name in modules:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


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
    install_modules(monkeypatch, {"_mthreads.ops.randn": vendor_randn})
    monkeypatch.setattr(
        torch_fl.flagos._C,
        "_reserve_rng_seed",
        lambda device, generator=None: (1 << 63) + 5,
    )

    torch_fl._patch_flaggems_philox()

    assert vendor_randn.philox_backend_seed_offset(128) == (-(1 << 63) + 5, 0)


class _LazyModule(types.ModuleType):
    """A module whose module-level ``__getattr__`` runs a failing import.

    transformers generates one of these per fast image processor. Looking up any
    name the module does not define runs ``import torchvision`` and propagates
    the resulting ``ModuleNotFoundError``; a ``getattr(obj, name, default)``
    default does not suppress it, because only ``AttributeError`` is.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise ModuleNotFoundError("No module named 'torchvision'")


def test_flaggems_philox_survives_modules_that_raise_on_getattr(monkeypatch):
    """A hostile ``sys.modules`` entry must cost one module, not the whole bridge.

    Whether such an entry is present when torch_fl's import runs depends on the
    host: a bare interpreter has no transformers lazy modules loaded and the
    sweep completes, while a process that imported transformers first does and
    the sweep used to die at the first of them — leaving the canonical module
    bound to the unpatched function, whose ``state_copy.view(torch.int64)``
    unpacks the flagos generator's MT19937 state into two variables and raises
    ``ValueError: too many values to unpack (expected 2)`` on every ``randn``.
    """
    # Insertion order is sweep order, so the hostile entries go in first: a
    # sweep that dies on either of them never reaches the flag_gems modules
    # below it, which is exactly the shape that left the bridge uninstalled.
    install_modules(
        monkeypatch,
        {
            "transformers.models.aria.image_processing_aria_fast": _LazyModule(
                "transformers.models.aria.image_processing_aria_fast"
            ),
            # A non-module entry has no ``__dict__`` to read at all.
            "not_a_module": object(),
        },
    )
    random_utils = _install_fake_flag_gems(monkeypatch)
    vendor_randn = types.ModuleType("_mthreads.ops.randn")
    vendor_randn.philox_backend_seed_offset = random_utils.philox_backend_seed_offset
    install_modules(monkeypatch, {"_mthreads.ops.randn": vendor_randn})

    torch_fl._patch_flaggems_philox()

    assert random_utils.philox_backend_seed_offset is not _fallback_seed_offset
    assert vendor_randn.philox_backend_seed_offset is not _fallback_seed_offset
    assert (
        vendor_randn.philox_backend_seed_offset
        is random_utils.philox_backend_seed_offset
    )
