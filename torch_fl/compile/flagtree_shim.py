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
FlagTree detection and vendor-runtime wiring for torch.compile.

FlagTree is a Triton fork that substitutes itself for Triton *at install time*:
its wheel is named ``flagtree``, but the module it installs is ``triton``, and
installing it uninstalls the official ``triton``. So inductor's own
``import triton`` already resolves to FlagTree once it is installed, and nothing
here needs to patch ``sys.modules`` -- there is no ``flagtree`` module to import.

Most of this module therefore only *reports* which Triton is active, so that
FLAGOS_USE_FLAGTREE=1 can assert FlagTree is really in use instead of silently
compiling with stock Triton.

The MThreads backend needs one thing beyond detection. Its driver reads device
availability, current device, compute capability and the raw ``musaStream_t``
from the ``torch_musa`` plugin, whose ``__init__`` claims the process-global
PrivateUse1 hooks that torch_fl must own. ``bind_flagtree_musa_driver`` points
those lookups at torch_fl's own MUSA runtime instead, so FlagTree launches onto
the same stream the native mudnn kernels use.
"""

import importlib.metadata
import importlib.util
from typing import Any, Optional, Tuple


# FlagTree-only module. Present regardless of which vendor backend was built,
# unlike triton._flagtree_backend.FLAGTREE_BACKEND, which is the empty string on
# nvidia/amd because upstream tells you not to set FLAGTREE_BACKEND for those.
_FLAGTREE_MARKER = "triton._flagtree_spec"


def is_flagtree_active() -> bool:
    """Whether the importable ``triton`` is FlagTree rather than stock Triton.

    Newer FlagTree releases carry ``triton._flagtree_spec``. The MThreads 3.1
    wheel predates that marker but still publishes the ``flagtree`` distribution,
    so use package metadata as a compatibility fallback and verify that the
    distribution owns the active ``triton`` package path.
    """
    try:
        if importlib.util.find_spec(_FLAGTREE_MARKER) is not None:
            return True
    except (ImportError, ValueError):
        pass

    try:
        import triton

        distribution = importlib.metadata.distribution("flagtree")
        root = str(distribution.locate_file("triton"))
        return any(str(path).startswith(root) for path in triton.__path__)
    except (ImportError, importlib.metadata.PackageNotFoundError, ValueError):
        return False


def flagtree_backend() -> Optional[str]:
    """The vendor backend FlagTree was built for, if it recorded one.

    Empty on nvidia/amd (upstream builds those without FLAGTREE_BACKEND set), so
    an empty result means "FlagTree, vendor unrecorded", not "not FlagTree".
    Returns None when FlagTree is not active.
    """
    if not is_flagtree_active():
        return None
    try:
        from triton._flagtree_backend import FLAGTREE_BACKEND

        if FLAGTREE_BACKEND:
            return FLAGTREE_BACKEND
    except ImportError:
        pass

    # The MThreads Triton 3.1 wheel predates _flagtree_backend but exposes a
    # single discoverable backend package. Infer only the unambiguous vendor
    # names needed by runtime workarounds; an unknown build remains empty.
    try:
        import triton.backends

        names = set(triton.backends.backends)
        if names == {"mthreads"}:
            return "mthreads"
    except (ImportError, AttributeError):
        pass
    return ""


def require_flagtree() -> None:
    """Fail unless the active Triton is FlagTree.

    Called for FLAGOS_USE_FLAGTREE=1. FlagTree cannot be enabled from here -- it
    is chosen when the environment is built -- so the only useful thing this can
    do is refuse to pretend.
    """
    if is_flagtree_active():
        return

    raise RuntimeError(
        "FLAGOS_USE_FLAGTREE=1 but the active 'triton' module is stock Triton, "
        "not FlagTree. FlagTree replaces triton at install time; it is not "
        "something this process can switch on. Build it from source "
        "(https://github.com/flagos-ai/FlagTree) -- there is no 'flagtree' "
        "package on PyPI, and 'import flagtree' never works because the module "
        "it installs is named 'triton'. Unset FLAGOS_USE_FLAGTREE to compile "
        "with stock Triton instead."
    )


def _musa_device_index(device: Any = None) -> int:
    """Resolve any device spelling to the plain index the MUSA runtime takes.

    Inductor hands these entry points ``None``, an ``int``, or a ``torch.device``
    depending on the call site, while ``torch_fl._C`` accepts an ``int`` only.
    Delegates to the device interface's existing normalizer rather than carrying
    another copy of it.
    """
    from torch_fl.compile.device_interface import FlagOSDeviceInterface

    return int(FlagOSDeviceInterface._vendor_device(device))


def get_musa_device_capability(device: Any = None) -> Tuple[int, int]:
    """``(major, minor)`` for FlagTree's ``GPUTarget``, read from the device.

    FlagTree turns this into ``major * 10 + minor`` and derives the warp size
    from it, so the values must come from the real device, not a constant.
    """
    from torch_fl import flagos

    props = flagos.get_device_properties(_musa_device_index(device))
    return props.major, props.minor


def get_musa_current_raw_stream(device: Any = None) -> int:
    """The ``musaStream_t`` handle kernels must be launched on.

    This is the same handle the native mudnn kernels submit to, which is what
    makes a compiled kernel ordered against eager MUSA work without an explicit
    synchronize between them. Also the import target of the raw-stream line
    inductor writes into generated code (see ``inductor_codegen.py``).
    """
    from torch_fl import _C

    return _C._get_musa_current_raw_stream(_musa_device_index(device))


_musa_driver_bound = False


def bind_flagtree_musa_driver() -> bool:
    """Point FlagTree's MThreads driver at torch_fl's MUSA runtime. Idempotent.

    Returns False when no MThreads FlagTree backend is installed, so callers can
    treat a stock-Triton environment as "nothing to bind" rather than an error.

    The vendor driver reaches ``torch_musa`` in three places: ``is_active()``
    (which is what makes Triton select the backend at all), the four runtime
    getters, and a module-level import guarded by ``is_active()``. Rebinding the
    class attributes covers all of them -- the driver's ``__init__`` copies these
    onto the instance, so this must run before the first driver instantiation,
    which is why the compile backend calls it ahead of any Triton work.
    """
    global _musa_driver_bound
    if _musa_driver_bound:
        return True

    try:
        import triton.backends
    except ImportError:
        return False

    backend = triton.backends.backends.get("mthreads")
    if backend is None:
        return False

    from torch_fl import flagos

    driver_cls = backend.driver
    driver_cls.is_active = staticmethod(flagos.is_available)
    driver_cls._get_device_capability = staticmethod(get_musa_device_capability)
    driver_cls._get_current_stream = staticmethod(get_musa_current_raw_stream)
    driver_cls._get_current_device = staticmethod(flagos.current_device)
    driver_cls._set_current_device = staticmethod(flagos.set_device)

    _musa_driver_bound = True
    return True


def flagtree_musa_driver_target() -> Optional[Tuple[str, int, int]]:
    """``(backend, capability, warp_size)`` FlagTree will compile for, or None.

    A single check that the binding took effect: it exercises the driver's own
    target resolution, which now runs entirely through torch_fl.
    """
    if not bind_flagtree_musa_driver():
        return None
    from triton.runtime import driver

    target = driver.active.get_current_target()
    return target.backend, target.arch, target.warp_size
