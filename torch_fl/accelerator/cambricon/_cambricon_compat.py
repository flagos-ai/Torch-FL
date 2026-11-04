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

"""The ``torch.mlu`` surface FlagGems' CAMBRICON backend needs, without torch_mlu.

A flagos-on-MLU wheel runs on the stock ``torch==X.Y+cpu`` wheel. The vendor
plugin is deliberately not installed: it hard-links its own ``libtorch.so``,
is ABI-bound to the vendor's torch minor, and claims the process-global
PrivateUse1 key that torch_fl owns.

FlagGems and the Triton mlu driver it launches through still reach for the
vendor names, in four places, all at import time or on the launch path:

* ``flag_gems/runtime/backend/_cambricon/__init__.py`` -- ``import torch_mlu``
  (and the same line in ``ops/dropout.py``, ``ops/alpha_dropout.py``,
  ``ops/feature_dropout.py``);
* ``_cambricon/utils/__init__.py`` -- ``torch.mlu.current_device()`` and
  ``torch.mlu.device_count()`` read at import, through
  ``triton.backends.mlu.driver.BangUtils()``;
* ``triton/backends/mlu/driver.py`` ``BangDriver.__init__`` -- binds
  ``torch.mlu.current_device``/``set_device``/``current_stream`` and hands
  ``torch.mlu`` back as the device interface, and ``is_active()`` gates on
  ``import torch_mlu`` then ``torch.mlu.is_available()``;
* ``torch.backends.mlu`` -- a real subpackage of torch_mlu, imported by name.

Synthesize exactly those attributes from ``torch_fl.flagos`` -- the same
surface, backed by the cnrt runtime torch_fl owns -- and publish a stub module
under ``torch_mlu`` so the imports succeed. The TsingMicro shim
(``_install_tsingmicro_flaggems_compat`` in ``torch_fl/__init__.py``) solves the
identical problem for its vendor.

Measured on an MLU590 (neuware 4.6.7, torch 2.10.0+cpu, flag_gems 5.4.0,
triton 3.4.0+mlu2.1.1): ``import flag_gems`` completes and
``BangDriver.is_active()`` returns True, with
``BangUtils().get_device_properties(0)`` reporting the real card
(cluster_num 12, core_num_per_cluster 4, max_nram_size 524288, isa_version 592).
"""

from __future__ import annotations

import ctypes
import sys
import types

#: Resolved by soname, so the neuware install on the loader path supplies it.
_CNRT_SONAME = "libcnrt.so"


def _cnrt():
    """The process' ``libcnrt.so``, or None when the MLU runtime is absent."""
    try:
        return ctypes.CDLL(_CNRT_SONAME)
    except OSError:
        return None


def _device_count() -> int:
    runtime = _cnrt()
    if runtime is None:
        return 0
    count = ctypes.c_uint(0)
    status = runtime.cnrtGetDeviceCount(ctypes.byref(count))
    return int(count.value) if status == 0 else 0


def _current_device() -> int:
    """cnrt's current device, without going through torch's lazy init.

    FlagGems' ``_cambricon/utils`` calls ``torch.mlu.current_device()`` at module
    scope, i.e. from inside the ``import flag_gems`` that ``flagos._lazy_init``
    itself performs. flagos' own ``current_device`` routes through
    ``_C._get_device()``, which calls ``device_lazy_init(PrivateUse1)`` first;
    that re-enters torch's device initialisation while ``torch_fl`` is halfway
    through it, and the process dies in ``cnrtGetDevice`` instead of raising.

    cnrt itself needs no initialisation to answer this -- measured on an MLU590
    with a cold process, ``cnrtGetDevice`` returns 0 -- so the shim reads it
    directly. That keeps the import-time call free of torch's initialisation
    path while still reporting the real device.
    """
    runtime = _cnrt()
    if runtime is None:
        return 0
    ordinal = ctypes.c_int(-1)
    status = runtime.cnrtGetDevice(ctypes.byref(ordinal))
    return int(ordinal.value) if status == 0 else 0


def _build_torch_mlu_surface(flagos) -> types.ModuleType:
    """The ``torch.mlu`` module FlagGems and Triton's mlu driver read."""
    mlu = types.ModuleType("torch.mlu")
    mlu.__doc__ = "Synthesized by torch_fl: the MLU surface FlagGems reads."

    # Straight proxies: torch_fl.flagos already exposes the CUDA-shaped names.
    for name in (
        "device",
        "current_device",
        "set_device",
        "synchronize",
        "is_available",
        "get_device_properties",
        "current_stream",
        "stream",
        "Stream",
        "Event",
        "default_generators",
        "empty_cache",
        "mem_get_info",
        "memory_allocated",
        "memory_reserved",
        "manual_seed",
        "manual_seed_all",
        "initial_seed",
        "get_rng_state",
        "set_rng_state",
    ):
        value = getattr(flagos, name, None)
        if value is not None:
            setattr(mlu, name, value)

    mlu.device_count = _device_count
    # Replaces the straight proxy above: see `_current_device` for why the
    # import-time read must not enter torch's lazy initialisation.
    mlu.current_device = _current_device

    def _get_device_capability(device=None):
        if device is None:
            device = flagos.current_device()
        props = flagos.get_device_properties(device)
        return props.major, props.minor

    mlu.get_device_capability = _get_device_capability
    mlu.get_device_name = lambda device=None: (
        flagos.get_device_properties(
            flagos.current_device() if device is None else device
        ).name
    )
    return mlu


def _publish_stub(name: str, surface: types.ModuleType) -> types.ModuleType:
    """A ``torch_mlu``-shaped module that forwards every attribute to ``surface``.

    ``torch_mlu`` is imported by FlagGems and by Triton's driver for its side
    effects only; what they then read is ``torch.mlu``. Forwarding instead of
    re-exporting keeps one implementation, and ``module.__getattr__`` covers
    whatever name a future FlagGems release starts reaching for.
    """
    import importlib.machinery

    stub = types.ModuleType(name)
    stub.__doc__ = "Synthesized by torch_fl; forwards to torch.mlu."
    stub.__getattr__ = lambda attribute: getattr(surface, attribute)
    stub.__spec__ = importlib.machinery.ModuleSpec(
        name=name, loader=None, origin="torch_fl_shim"
    )
    sys.modules[name] = stub
    return stub


def install_torch_mlu_shim(flagos) -> None:
    """Publish ``torch.mlu``, ``torch.backends.mlu`` and ``torch_mlu``.

    Idempotent. Must run after ``import torch`` (the attributes land on the
    torch module object) and before ``flag_gems`` is imported -- the FlagGems
    backend and Triton's driver both read ``torch.mlu`` at import time.
    """
    import torch

    if getattr(torch, "mlu", None) is None:
        torch.mlu = _build_torch_mlu_surface(flagos)

    # torch_mlu ships ``torch.backends.mlu`` as a real subpackage. A stock torch
    # has no slot for it and the FlagGems cambricon backend imports it by name,
    # so bind it on the namespace as well as in sys.modules.
    if getattr(getattr(torch, "backends", None), "mlu", None) is None:
        backends_mlu = types.ModuleType("torch.backends.mlu")
        backends_mlu.__doc__ = "Synthesized by torch_fl."
        torch.backends.__dict__["mlu"] = backends_mlu
        sys.modules["torch.backends.mlu"] = backends_mlu

    if "torch_mlu" not in sys.modules:
        stub = _publish_stub("torch_mlu", torch.mlu)
        # torch_mlu.utils.model_transfer.transfer is imported by the FlagGems
        # backend inside a try/except ImportError; these stubs keep the parent
        # package lookup consistent for callers that do reach for it.
        stub.utils = types.ModuleType("torch_mlu.utils")
        stub.utils.model_transfer = types.ModuleType("torch_mlu.utils.model_transfer")
        stub.utils.model_transfer.transfer = lambda model, *args, **kwargs: model
        sys.modules["torch_mlu.utils"] = stub.utils
        sys.modules["torch_mlu.utils.model_transfer"] = stub.utils.model_transfer


def install_mlu_driver_device_strings() -> None:
    """Point Triton's mlu driver at the flagos device name.

    ``triton/backends/mlu/driver.py`` names the device by the vendor string in
    two places:

    * ``get_empty_cache_for_benchmark`` -- ``torch.empty(..., device='mlu')``,
      the 256 MB L2 flush buffer built once per autotune run;
    * ``get_active_torch_device`` -- ``torch.device("mlu", ...)``.

    Both are handed to torch. torch_mlu would have registered ``"mlu"`` as the
    PrivateUse1 name; this build claims that name as ``flagos`` instead (torch
    allows exactly one), and a ``str`` device argument is resolved in C++ before
    any Python alias sees it -- which is why the device-alias entry in
    ``torch_fl._VENDOR_DEVICE_ALIASES`` helps for direct ``torch.empty`` calls
    but not once the call is made from inside FlagGems' dispatcher.

    Rebinding the two methods is the whole fix: the device object they should
    have built is the one flagos builds, so the replacement asks
    ``torch._C._get_privateuse1_backend_name()`` for the registered name rather
    than hardcoding either spelling.

    Skipped when Triton is absent (a FlagGems-less build) or when its mlu backend
    is not importable -- this platform does not install FlagTree, and without the
    mlu backend there is nothing here to patch.
    """
    try:
        from triton.backends.mlu.driver import BangDriver
    except ImportError:
        return

    import torch

    name = torch._C._get_privateuse1_backend_name()

    def get_active_torch_device(self):
        return torch.device(name, self.get_current_device())

    def get_empty_cache_for_benchmark(self):
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device=name)

    # Both replacements are attached only when the upstream method still hardcodes
    # the vendor spelling, so a future Triton release that learns the registered
    # name is left alone instead of being shadowed.
    if "mlu" in (BangDriver.get_active_torch_device.__code__.co_consts or ()):
        BangDriver.get_active_torch_device = get_active_torch_device
    if "mlu" in (BangDriver.get_empty_cache_for_benchmark.__code__.co_consts or ()):
        BangDriver.get_empty_cache_for_benchmark = get_empty_cache_for_benchmark


def install_cambricon_compat() -> None:
    """Vendor-side setup a Cambricon build does before FlagGems is imported.

    The torch.mlu shim and the Triton driver's device strings -- the two things
    this platform needs that its vendor runtime does not provide, both of them
    consequences of running on torch+cpu without torch_mlu. A no-op without an
    MLU runtime.
    """
    if _device_count() <= 0:
        return

    from torch_fl import flagos

    install_torch_mlu_shim(flagos)
    install_mlu_driver_device_strings()

    # The cnrt context has to exist before the properties are read, and on this
    # platform nothing else does it: torch_mlu's `import torch_mlu` normally
    # stands the runtime up, and torch_mlu is not installed here. `_C._init` is
    # the same call flagos._lazy_init makes first, minus the FlagGems import that
    # would re-enter what the next line is preparing for.
    try:
        flagos._C._init()
        flagos._initialized = True
    except Exception:  # noqa: BLE001 - device bring-up is best-effort here
        return
    _preload_flaggems_cambricon_utils()


def _preload_flaggems_cambricon_utils() -> None:
    """Break the import cycle FlagGems' ``_cambricon`` backend contains.

    ``_cambricon/__init__.py`` starts with ``from .utils import DEVICE_COUNT``,
    which runs ``utils`` -- and ``utils`` reads the device at module scope
    (``BangUtils().get_device_properties(torch.mlu.current_device())``). Its
    sibling ``ops/sort.py`` imports the same module by *absolute* path
    (deliberately: "so that this file can also be loaded as a standalone module
    ... without the surrounding package context"), so loading ``_cambricon.ops``
    first re-enters the still-executing ``_cambricon`` package and runs ``utils``
    a second time. ``BangUtils`` is a singleton whose ``__init__`` recompiles and
    reloads the ``bang_utils`` C module on every call, so that second run tears
    down the cnrt context the first one created and the process dies inside
    ``cnrtGetDevice`` instead of raising.

    Which order ``flag_gems`` walks that cycle in depends on what is already in
    ``sys.modules``. On this platform the device is not up yet when the import
    starts (there is no vendor package doing it in ``import torch_mlu``), so the
    re-entrant path is the one it takes.

    Seeding the module with the values ``utils`` would have computed removes the
    re-entry: the real import then finds it present and keeps the values. The
    properties come from the same ``BangUtils`` call ``utils`` makes, so this is
    not a stand-in for the measurement -- it is the measurement, taken once, on
    a device that is up. Skipped whenever anything is missing; FlagGems is
    optional everywhere.
    """
    import sys
    import types

    name = "flag_gems.runtime.backend._cambricon.utils"
    if name in sys.modules:
        return
    try:
        from triton.backends.mlu.driver import BangUtils

        props = BangUtils().get_device_properties(_current_device())
    except Exception:  # noqa: BLE001 - FlagGems/Triton absent or undetectable
        return

    int_max = sys.maxsize
    utils = types.ModuleType(name)
    utils.__doc__ = "Seeded by torch_fl; see _preload_flaggems_cambricon_utils."
    cluster_num = props.get("cluster_num", 0)
    utils.TOTAL_CLUSTER_NUM = cluster_num
    utils.TOTAL_CORE_NUM = cluster_num * props.get("core_num_per_cluster", 0)
    utils.MAX_NRAM_SIZE = props.get("max_nram_size", 0)
    utils.DEVICE_COUNT = _device_count()
    utils.MAX_GRID_SIZES = [
        props.get(f"max_block_task_dim_{axis}", int_max) for axis in "xyz"
    ]
    utils.MAX_GRID_SIZE_X, utils.MAX_GRID_SIZE_Y, utils.MAX_GRID_SIZE_Z = (
        utils.MAX_GRID_SIZES
    )
    sys.modules[name] = utils
