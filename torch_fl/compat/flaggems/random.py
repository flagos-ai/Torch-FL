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

"""Bridge FlagGems Philox reservations to the flagos generator."""

import sys
import types

from . import runtime


def _is_philox_binding(func) -> bool:
    """Is ``func`` the bridge's patched seed function, from any torch_fl copy?"""
    handle = getattr(func, "__self__", None)
    return isinstance(handle, types.SimpleNamespace) and hasattr(handle, "philox")


def _philox_rebind(mod, patched, orig) -> None:
    """Point one module's import-time copy at `patched`; never raise."""
    try:
        namespace = vars(mod)
        if namespace.get("philox_backend_seed_offset") is orig:
            namespace["philox_backend_seed_offset"] = patched
    except Exception:
        pass


_PHILOX_REBIND = None


def patch_philox(accelerator: str, flagos) -> None:
    """Route MUSA/Cambricon FlagGems RNG reservations through the flagos generator.

    Native muRAND/mudnn kernels and this bridge both consume one 64-bit seed per
    stochastic operation from the selected PrivateUse1 generator. FlagGems then
    starts its per-operation Philox stream at offset zero. This keeps
    manual_seed/get_rng_state/set_rng_state and mixed native/FlagGems call order
    deterministic without maintaining a second CUDA-style generator state.
    """
    if (
        accelerator not in ("mthreads", "tsingmicro", "cambricon")
        or not runtime.routes_to_flaggems()
    ):
        return

    # The bridge as installed in this interpreter, whichever `torch_fl` copy
    # installed it: the handle below, which carries the patched seed function
    # and the sweep that binds it. Reset here so a previous call cannot be
    # adopted by this one.
    global _PHILOX_BINDING, _PHILOX_REBIND
    _PHILOX_BINDING = None
    _PHILOX_REBIND = None

    try:
        # Import the *package* first, then the submodule. ``flag_gems.utils.
        # random_utils`` does ``import flag_gems`` at module scope, and
        # ``flag_gems/__init__.py`` imports ``flag_gems.ops``, which imports
        # ``random_utils`` back -- so entering through the submodule runs that
        # cycle with ``random_utils`` half-built and raises ImportError. The
        # package entry point has ``random_utils`` already loaded by then and
        # completes.
        #
        # This is not merely "the bridge fails to install". A failed import
        # leaves the submodules that *did* load in ``sys.modules`` while
        # removing ``flag_gems`` itself, so every one of them keeps a reference
        # to the abandoned half-initialised package object. The next
        # ``import flag_gems`` (the dispatcher's, on first op) builds a fresh
        # module with ``vendor_name`` set, but those stale bindings still read
        # the orphan -- and FlagGems' own ``tensor_wrapper``/``pointwise_dynamic``
        # read ``flag_gems.vendor_name``, so ops that go through them (``cat``
        # among them) die with ``module 'flag_gems' has no attribute
        # 'vendor_name'``.
        import flag_gems  # noqa: F401
        from flag_gems.utils import random_utils

        _orig = random_utils.philox_backend_seed_offset

        _rebind = _philox_rebind

        def _patched(increment, generator=None):
            if generator is None:
                device = random_utils.torch_device_fn.current_device()
                seed = flagos._C._reserve_rng_seed(device)
            else:
                generator_device = getattr(generator, "device", None)
                if getattr(generator_device, "type", None) not in (
                    "flagos",
                    "privateuseone",
                ):
                    return _orig(increment, generator=generator)
                device = generator_device.index
                if device is None:
                    device = random_utils.torch_device_fn.current_device()
                seed = flagos._C._reserve_rng_seed(device, generator)

            # FlagGems obtains CUDA generator seeds through an int64 state tensor.
            # Preserve those signed bit semantics when the reserved uint64 has its
            # high bit set.
            if seed >= 1 << 63:
                seed -= 1 << 64
            return seed, 0

        # Bind the canonical module first. The sweep below is best-effort; every
        # module it fails to reach still falls back to this one through
        # `random_utils`, so the bridge must never be skipped because the sweep
        # was interrupted.
        #
        # Another copy of torch_fl in this interpreter (the installed package
        # beside a tree on PYTHONPATH — `tests/unit/conftest.py` restores the
        # backend config path for the same split) has already bound the bridge
        # here. Adopt its function rather than bind this copy's: it closes over
        # that copy's `_reserve_rng_seed`, and binding it costs the first copy
        # its sweep targets. Either way the module must end up bound to the
        # bridge, since a previous test may have put its fake back.
        handle = None
        current = random_utils.philox_backend_seed_offset
        if current is not _orig:
            if not _is_philox_binding(current):
                return
            handle = current.__self__
        elif handle is None:
            handle = types.SimpleNamespace(philox=_patched, rebind=_rebind)
            _PHILOX_BINDING = handle
        random_utils.philox_backend_seed_offset = handle.philox

        # RNG modules bind this function with ``from ... import`` at import time,
        # so update every already-loaded copy as well as the canonical module.
        # Match on the bound object rather than on the module name: FlagGems
        # republishes each vendor backend's op tree under the package-level names
        # (`flag_gems.randn` is `_mthreads.ops.randn.randn` on MUSA), and those
        # modules are named `_mthreads.ops.*`, not `flag_gems.*`. A name filter
        # silently skips them and the vendor kernel reaches the unpatched
        # function, whose `state_copy.view(torch.int64)` unpacks the flagos
        # generator's MT19937 state into two variables and raises ValueError.
        #
        # Nothing about foreign modules may abort the sweep. ``sys.modules``
        # holds modules with a module-level ``__getattr__`` that runs arbitrary
        # code and raises on a missing optional dependency — transformers'
        # lazy image-processor modules raise ``ModuleNotFoundError: No module
        # named 'torchvision'`` — and it also holds non-module entries such as
        # ``torch.ops``. The namespace is therefore read from ``__dict__``,
        # which neither consults ``__getattr__`` nor depends on the entry being
        # a module, and every step is isolated so one bad entry costs one module
        # rather than the whole bridge. Which modules are loaded when torch_fl's
        # import runs depends on the host's installed packages: a bare
        # interpreter has no lazy transformers modules loaded and the sweep
        # completes, while a pytest process that imported transformers first does
        # and the sweep used to die at the first of them, leaving every
        # ``randn``/``rand``/``randperm`` on the failure path above.
        #
        #
        # FlagGems re-imports op modules under the dispatcher's name during
        # `prepare_dispatch()`, after this runs: `_mthreads.ops.
        # randn` and `flag_gems.runtime.backend._mthreads.ops.randn` are two
        # module objects with their own `from ... import`. Those are swept by
        # `refresh_philox_bindings()` at the end of that phase.
        for mod in list(sys.modules.values()):
            _rebind(mod, handle.philox, _orig)
        # Read back by `refresh_philox_bindings`, once registration has
        # imported the dispatcher's copies of the op modules.
        _PHILOX_REBIND = (handle.philox, _orig)
    except Exception:
        # FlagGems remains optional; native MUSA kernels stay available.
        pass


def refresh_philox_bindings() -> None:
    """Re-bind the RNG seed function in modules registration imported late.

    `patch_philox` replaces `philox_backend_seed_offset` in every
    module loaded at that point. `prepare_dispatch` then imports
    each vendor op module under the dispatcher's dotted name, which creates
    modules that bind the original function through their own `from ...
    import` -- on MUSA, `flag_gems.runtime.backend._mthreads.ops.randn` beside
    the `_mthreads.ops.randn` FlagGems had loaded. An unpatched binding dies
    unpacking the flagos generator's MT19937 state (`ValueError: too many
    values to unpack`), so every such module is swept once more here.
    """
    if _PHILOX_REBIND is None:
        return
    patched, orig = _PHILOX_REBIND
    rebind = _PHILOX_BINDING.rebind if _PHILOX_BINDING else _philox_rebind
    for mod in list(sys.modules.values()):
        rebind(mod, patched, orig)
