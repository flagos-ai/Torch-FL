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

"""Optional NVIDIA Apex compatibility for CUDA-ABI flagos backends.

Apex's multi-tensor extension calls ``amp_C`` directly instead of going through
ATen. Consequently, Torch-FL's dispatcher boxing guard cannot reinterpret a
flagos tensor as the CUDA tensor that Apex expects. This compatibility layer
wraps Apex's common ``MultiTensorApply`` entry point and uses the existing
zero-copy views in ``torch_fl._C`` at that boundary.

The patch is deliberately limited to vendors whose flagos storage is a CUDA
ABI alias. Reinterpreting a native non-CUDA tensor as CUDA would be unsafe, so
those platforms are left untouched.
"""

from __future__ import annotations

import builtins
import functools
import importlib
import importlib.util
import os
import sys
import warnings
from typing import Any

import torch

from torch_fl.comm.process_group import is_cuda_alias_vendor


_DISABLE_ENV = "FLAGOS_DISABLE_APEX_COMPAT"
_PATCHED_ATTR = "_flagos_apex_compat_patched"
_ORIGINAL_ATTR = "_flagos_apex_compat_original_call"
_IMPORT_HOOK_ATTR = "_flagos_apex_compat_import_hook"

_original_import = None
_import_hook = None
_importing_apex = False


# ---------------------------------------------------------------------------
# Capability and recursive conversion helpers
# ---------------------------------------------------------------------------


def _is_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _build_accelerator() -> str:
    """Return the build accelerator without importing torch_fl.__init__."""
    value = os.environ.get("ACCELERATOR", "").strip().lower()
    if value:
        return value
    try:
        from torch_fl._build_config import ACCELERATOR
    except ImportError:
        return ""
    return str(ACCELERATOR).strip().lower()


def _active_vendor() -> str:
    """Resolve the vendor name used by the shared CUDA-alias capability table."""
    configured = os.environ.get("GEMS_VENDOR", "").strip().lower()
    if configured:
        # FlagGems has used both spellings over time; the shared table uses the
        # canonical profile name for the generic CUDA target.
        return "nvidia" if configured == "cuda" else configured

    # GEMS_VENDOR is normally initialized by torch_fl.__init__ before this
    # module is installed. The fallback keeps direct use of this module
    # deterministic in tests and in applications that call the installer early.
    accelerator = _build_accelerator()
    return {
        "dcu": "hygon",
        "metax": "metax",
        "cuda": "nvidia",
        "": "nvidia",
    }.get(accelerator, accelerator)


def is_apex_compat_available() -> bool:
    """Return whether the safe zero-copy Apex bridge may be installed."""
    return not _is_disabled() and is_cuda_alias_vendor(_active_vendor())


def _is_flagos_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type in (
        "flagos",
        "privateuseone",
    )


def _contains_flagos(value: Any) -> bool:
    if _is_flagos_tensor(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_flagos(item) for item in value)
    return False


def _map_inputs(value: Any, view_fn) -> Any:
    """Recursively convert flagos tensors while preserving container types."""
    if _is_flagos_tensor(value):
        return view_fn(value)
    if isinstance(value, list):
        return [_map_inputs(item, view_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_inputs(item, view_fn) for item in value)
    return value


def _map_outputs(value: Any, view_fn) -> Any:
    """Recursively convert CUDA results back to flagos views."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            return view_fn(value)
        return value
    if isinstance(value, list):
        return [_map_outputs(item, view_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_outputs(item, view_fn) for item in value)
    return value


def _get_view_functions():
    try:
        import torch_fl._C as extension
    except (ImportError, AttributeError):
        return None, None

    to_cuda = getattr(extension, "_flagos_to_cuda_view", None)
    to_flagos = getattr(extension, "_cuda_to_flagos_view", None)
    if to_cuda is None or to_flagos is None:
        return None, None
    return to_cuda, to_flagos


# ---------------------------------------------------------------------------
# Apex patch
# ---------------------------------------------------------------------------


def _patch_multi_tensor_apply(module) -> bool:
    """Patch the class shared by Apex's multi-tensor appliers."""
    cls = getattr(module, "MultiTensorApply", None)
    original = getattr(cls, "__call__", None) if cls is not None else None
    if original is None:
        return False
    if getattr(original, _PATCHED_ATTR, False):
        return True

    @functools.wraps(original)
    def patched(self, op, noop_flag_buffer, tensor_lists, *args):
        # The environment switch is checked at call time as well as install
        # time, so setting it before a later optimizer call is still effective.
        if _is_disabled() or not is_apex_compat_available():
            return original(self, op, noop_flag_buffer, tensor_lists, *args)

        to_cuda, to_flagos = _get_view_functions()
        if to_cuda is None or to_flagos is None:
            return original(self, op, noop_flag_buffer, tensor_lists, *args)

        has_flagos = _contains_flagos((noop_flag_buffer, tensor_lists, args))
        if not has_flagos:
            return original(self, op, noop_flag_buffer, tensor_lists, *args)

        converted_noop = _map_inputs(noop_flag_buffer, to_cuda)
        converted_lists = _map_inputs(tensor_lists, to_cuda)
        converted_args = tuple(_map_inputs(arg, to_cuda) for arg in args)
        result = original(
            self,
            op,
            converted_noop,
            converted_lists,
            *converted_args,
        )
        return _map_outputs(result, to_flagos)

    setattr(patched, _PATCHED_ATTR, True)
    setattr(patched, _ORIGINAL_ATTR, original)
    cls.__call__ = patched
    return True


def patch_apex() -> bool:
    """Patch Apex if its multi-tensor Python module is available.

    Apex is optional. A missing package, missing extension, or an older Apex
    layout simply leaves the normal Torch-FL path unchanged.
    """
    if not is_apex_compat_available():
        return False

    try:
        module = importlib.import_module("apex.multi_tensor_apply.multi_tensor_apply")
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception as exc:  # noqa: BLE001 - optional third-party package
        warnings.warn(
            f"[torch_fl] Apex compatibility patch was skipped: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    try:
        return _patch_multi_tensor_apply(module)
    except (AttributeError, TypeError) as exc:
        warnings.warn(
            f"[torch_fl] Apex compatibility patch was skipped: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


# ---------------------------------------------------------------------------
# Import-order handling
# ---------------------------------------------------------------------------


def _remove_import_hook() -> None:
    global _original_import, _import_hook
    if _import_hook is not None and builtins.__import__ is _import_hook:
        builtins.__import__ = _original_import
    _original_import = None
    _import_hook = None


def _apex_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _importing_apex

    original = _original_import
    if original is None:
        return builtins.__import__(name, globals, locals, fromlist, level)
    if level != 0 or name.split(".", 1)[0] != "apex":
        return original(name, globals, locals, fromlist, level)

    if _is_disabled() or not is_apex_compat_available():
        _remove_import_hook()
        return original(name, globals, locals, fromlist, level)

    if _importing_apex:
        return original(name, globals, locals, fromlist, level)

    _importing_apex = True
    try:
        result = original(name, globals, locals, fromlist, level)
        patch_apex()
        _remove_import_hook()
        return result
    finally:
        _importing_apex = False


def _install_import_hook() -> None:
    global _original_import, _import_hook
    if _import_hook is not None:
        return
    _original_import = builtins.__import__
    _import_hook = _apex_import
    setattr(_import_hook, _IMPORT_HOOK_ATTR, True)
    builtins.__import__ = _import_hook


def install_apex_compat() -> bool:
    """Install the optional Apex patch and import-order compatibility hook."""
    if not is_apex_compat_available():
        _remove_import_hook()
        return False

    if "apex" in sys.modules:
        # Apex was imported before torch_fl. Importing this one small Python
        # module is safe and ensures that `import apex` alone is enough to patch
        # the common class before the first optimizer step.
        patch_apex()
        return True

    # Avoid installing a process-wide import wrapper when Apex is not installed.
    # `find_spec` inspects package metadata without importing Apex or amp_C.
    try:
        if importlib.util.find_spec("apex") is None:
            return False
    except (ImportError, AttributeError, ValueError):
        return False

    _install_import_hook()
    return True


__all__ = [
    "install_apex_compat",
    "is_apex_compat_available",
    "patch_apex",
]
