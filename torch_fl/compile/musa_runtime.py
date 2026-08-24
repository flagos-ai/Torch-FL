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

"""
torch_fl's own MUSA runtime, exposed to the MThreads FlagTree compiler.

FlagTree's MThreads backend needs four runtime facts to compile and launch a
kernel: whether a MUSA device is usable, the current device index, its compute
capability, and the raw ``musaStream_t`` kernels must be submitted to. The
vendor driver reads all four from the separate ``torch_musa`` plugin.

That plugin is not part of this build and cannot be part of this process:
PrivateUse1 has exactly one owner and torch_fl claims it (see
docs/vendors/musa/installation.md). Rather than answer for another plugin's
name, this module answers those four questions from torch_fl's MUSA runtime and
binds the answers onto the vendor driver class. FlagTree then talks to torch_fl
directly, and the tensors it launches against are the same ``flagos`` tensors
the native mudnn kernels use -- one device, one stream, one runtime owner.
"""

from typing import Any, Optional, Tuple

from torch_fl import flagos
from torch_fl import _C  # type: ignore[misc]


def is_available() -> bool:
    """Whether torch_fl has a usable MUSA device."""
    return flagos.device_count() > 0


def device_count() -> int:
    return flagos.device_count()


def current_device() -> int:
    return flagos.current_device()


def set_device(device: Any) -> None:
    flagos.set_device(_device_index(device))


def get_device_properties(device: Any = None) -> Any:
    return flagos.get_device_properties(_device_index(device))


def get_device_capability(device: Any = None) -> Tuple[int, int]:
    """``(major, minor)`` as FlagTree's ``GPUTarget`` expects it.

    FlagTree turns this into ``major * 10 + minor`` and picks the warp size from
    it, so the values must come from the real device, not a constant.
    """
    props = get_device_properties(device)
    return props.major, props.minor


def get_current_raw_stream(device: Any = None) -> int:
    """The ``musaStream_t`` handle kernels must be launched on.

    This is the same handle the native mudnn kernels submit to, which is what
    makes a compiled kernel ordered against eager MUSA work without an explicit
    synchronize between them.
    """
    return _C._get_musa_current_raw_stream(_device_index(device))


def current_stream(device: Any = None) -> Any:
    return flagos.current_stream(_device_index(device))


def synchronize(device: Any = None) -> None:
    flagos.synchronize(device)


def _device_index(device: Any = None) -> int:
    """Normalize None / str / torch.device / int into a device index."""
    if device is None:
        return flagos.current_device()
    if isinstance(device, str):
        import torch

        device = torch.device(device)
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    if hasattr(device, "type"):  # torch.device("flagos") -- no explicit index
        return flagos.current_device()
    return int(device)


_bound = False


def bind_flagtree_musa_driver() -> bool:
    """Point FlagTree's MThreads driver at this module. Idempotent.

    Returns False when no MThreads FlagTree backend is installed, so callers can
    treat a stock-Triton environment as "nothing to bind" rather than an error.

    The vendor driver resolves its runtime through ``torch_musa`` in three
    places: ``is_active()`` (which is what makes Triton select the backend at
    all), the four runtime getters, and a module-level ``import torch_musa``
    guarded by ``is_active()``. Rebinding the class attributes covers all of
    them -- the driver's ``__init__`` copies these into the instance, so this
    must happen before the first driver instantiation, which is why the compile
    backend calls it ahead of any Triton work.
    """
    global _bound
    if _bound:
        return True

    try:
        import triton.backends
    except ImportError:
        return False

    backend = triton.backends.backends.get("mthreads")
    if backend is None:
        return False

    driver_cls = backend.driver
    driver_cls.is_active = staticmethod(is_available)
    driver_cls._get_device_capability = staticmethod(get_device_capability)
    driver_cls._get_current_stream = staticmethod(get_current_raw_stream)
    driver_cls._get_current_device = staticmethod(current_device)
    driver_cls._set_current_device = staticmethod(set_device)

    _bound = True
    return True


def flagtree_musa_driver_target() -> Optional[Tuple[str, int, int]]:
    """``(backend, capability, warp_size)`` FlagTree will compile for, or None.

    Useful as a single check that the binding took effect: it exercises the
    driver's own target resolution, which now runs entirely through torch_fl.
    """
    if not bind_flagtree_musa_driver():
        return None
    from triton.runtime import driver

    target = driver.active.get_current_target()
    return target.backend, target.arch, target.warp_size
