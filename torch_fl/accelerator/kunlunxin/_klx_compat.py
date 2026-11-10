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

"""FlagGems glue for the Kunlunxin (XPU) backend.

Two reach-throughs, both structural, both measured on a P800:

1. ``DeviceDetector.name``. FlagGems' ``_kunlunxin`` descriptor declares
   ``device_name="cuda"``, and the op modules capture that name into a module
   global and compare it against the input's device type. A flagos tensor
   reports ``"flagos"``, so the comparison never matches and the op takes
   FlagGems' fallback branch instead of running its Triton kernel. Reading the
   installed FlagGems on a P800 gives exactly ``vendor='kunlunxin',
   name='cuda'``; realigning it is what CUDA, DCU and Iluvatar already do.

2. ``torch.cuda`` itself. ``flag_gems.runtime.torch_device_fn`` is
   ``torch.cuda`` for every descriptor that names its device ``cuda`` -- the
   kunlunxin one included -- and FlagGems' op wrappers use it for ordinary work:
   ``with torch_device_fn._DeviceGuard(idx)`` around a launch, ``synchronize()``,
   ``current_device()``, and the capability Triton turns into its ``arch``. This
   build links the **CPU** torch wheel (the XPU runtime lives in torch_fl's own
   ``_C``, not in a vendor libtorch), so every one of those raises
   ``RuntimeError: PyTorch was compiled without CUDA support`` and the op never
   launches. That is what ``install_klx_cuda_shim`` routes to ``torch.flagos``.

The shim is required, not an optimization: with the routes wired and the device
name realigned but no shim, ``torch.div`` on a flagos tensor reached
``flag_gems/runtime/backend/_kunlunxin/ops/div.py`` and then died in
``torch/cuda/__init__.py:_exchange_device``. With it, 22 of 22 elementwise and
reduction ops ran on device and matched a CPU reference (max|d| <= 2.4e-07).

Iluvatar carries the same two pieces (``_ilu_compat``) for the same reason: it is
the other FlagGems-first platform whose build links a CPU torch.
"""

import sys

import torch

_cuda_shim_installed = False
_device_name_patched = False

# FlagGems' own name for this accelerator once it is pointed at torch_fl's
# device, and the name its kunlunxin descriptor declares today. See
# patch_klx_flag_gems_device_name for why they have to agree.
_FLAGGEMS_DEVICE_NAME = "flagos"
_VENDOR_DEVICE_NAME = "cuda"


def _device_index(device):
    """An int device index from None / int / "flagos[:N]" / torch.device."""
    if device is None:
        return torch.flagos.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, (str, torch.device)):
        d = torch.device(device) if isinstance(device, str) else device
        return d.index if d.index is not None else torch.flagos.current_device()
    return torch.flagos.current_device()


def install_klx_cuda_shim():
    """Make the ``torch.cuda`` calls FlagGems makes answer for flagos. Idempotent.

    Only the handful FlagGems reaches for are replaced; everything else on
    ``torch.cuda`` is left alone, so a genuine "no CUDA here" answer stays
    visible to everybody else. Same shape as
    ``torch_fl.accelerator.iluvatar._ilu_compat.install_ilu_cuda_shim``.

    Replacing the entry points is preferable to pointing FlagGems at
    ``torch.flagos`` because ``flag_gems.runtime.torch_device_fn`` is bound at
    import time, and several op modules reach it as a module attribute.
    """
    global _cuda_shim_installed
    if _cuda_shim_installed:
        return
    _cuda_shim_installed = True

    flagos = torch.flagos
    cuda = torch.cuda

    cuda.current_device = lambda: flagos.current_device()
    cuda.set_device = lambda device: flagos.set_device(_device_index(device))
    cuda.device_count = lambda: flagos.device_count()
    cuda.is_available = lambda: True
    cuda.synchronize = lambda device=None: flagos.synchronize()
    cuda.get_device_properties = lambda device=None: flagos.get_device_properties(
        _device_index(device)
    )

    def _capability(device=None):
        """The real (major, minor), which Triton turns into its ``arch``.

        Triton asks the device interface for this and compiles for
        ``major * 10 + minor``. A stand-in value does not fail loudly -- the
        kernel loads and then hangs the device on launch, which is why this
        reads the properties rather than answering a constant.
        """
        props = flagos.get_device_properties(_device_index(device))
        return int(props.major), int(props.minor)

    cuda.get_device_capability = _capability
    cuda.get_device_name = lambda device=None: (
        flagos.get_device_properties(_device_index(device)).name
    )
    cuda.current_stream = lambda device=None: flagos.current_stream(
        _device_index(device)
    )
    cuda.default_stream = lambda device=None: flagos.current_stream(
        _device_index(device)
    )

    # CPU torch's DeviceContext calls torch.cuda._exchange_device on entry, which
    # is exactly the call that raises. Swapping the device and reporting the
    # previous one is the whole of what it is asked for.
    def _exchange_device(idx):
        if idx is None or int(idx) < 0:
            return -1
        previous = flagos.current_device()
        flagos.set_device(int(idx))
        return previous

    cuda._exchange_device = _exchange_device
    cuda._maybe_exchange_device = _exchange_device
    cuda._lazy_init = lambda: None

    # flagos owns real event and stream objects over the same runtime.
    cuda.Event = flagos.Event
    cuda.Stream = flagos.Stream


def patch_klx_flag_gems_device_name():
    """Point FlagGems' device identity at the name torch_fl registered.

    ``DeviceDetector`` copies ``device_name`` out of the vendor descriptor at
    construction, and the op modules capture it into a module global as they are
    imported (``device`` in the generic-path modules, ``_DEVICE_NAME`` in
    ``ops/mul.py``), so both are rewritten. Correcting the singleton alone would
    leave each of those literals stale.

    Must run after ``import flag_gems``. Idempotent. A no-op when the resolved
    descriptor is not the kunlunxin one, so an import from a platform whose
    descriptor names its device something else leaves that descriptor alone.
    """
    global _device_name_patched
    if _device_name_patched:
        return

    try:
        from flag_gems.runtime.backend.device_finder import DeviceDetector
    except ImportError:
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
        except ImportError:
            return

    detector = DeviceDetector()
    if getattr(detector, "vendor_name", None) != "kunlunxin":
        return
    if detector.name == _FLAGGEMS_DEVICE_NAME:
        _device_name_patched = True
        return
    if detector.name != _VENDOR_DEVICE_NAME:
        # Another descriptor is in charge; not ours to rewrite.
        return

    detector.name = _FLAGGEMS_DEVICE_NAME
    _device_name_patched = True

    for module in list(sys.modules.values()):
        if module is None:
            continue
        name = getattr(module, "__name__", "") or ""
        if not (name == "flag_gems" or name.startswith("flag_gems.")):
            continue
        for attr in ("device", "_DEVICE_NAME"):
            if getattr(module, attr, None) == _VENDOR_DEVICE_NAME:
                setattr(module, attr, _FLAGGEMS_DEVICE_NAME)
