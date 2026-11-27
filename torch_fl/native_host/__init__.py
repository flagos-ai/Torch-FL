# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build-time compiled FlagGems hosts, enabled by the wheel's build record."""

import os
from pathlib import Path

_hosts = None
_prepared = False


def _bootstrap_launcher():
    """Install the prebuilt backend before FlagGems initializes its compiler driver."""
    global _prepared
    from torch_fl import _build_config

    if os.environ.get("_TORCH_FL_BUILDING_NATIVE_HOST") == "1":
        return
    if not getattr(_build_config, "NATIVE_HOSTS", False):
        return
    if "boxing" in _build_config.KERNELS:
        raise RuntimeError("native hosts require a non-boxing build")
    from .backend import get_backend

    backend = get_backend(_build_config.ACCELERATOR)
    manifest = Path(__file__).parent / "_generated/manifest.json"
    if not manifest.is_file():
        raise RuntimeError("this wheel's precompiled FlagGems hosts are missing")
    backend.prepare_runtime()
    from .load import install_launcher

    install_launcher("torch_fl.native_host._generated")
    _prepared = True


def _activate():
    global _hosts
    if not _prepared:
        return
    from torch_fl import _build_config
    from .backend import get_backend
    from .load import Hosts

    get_backend(_build_config.ACCELERATOR).initialize_runtime()
    _hosts = Hosts(Path(__file__).parent / "_generated")
    _hosts.activate()
    from . import vendor

    vendor.activate()


def audit():
    """Return source and artifact evidence for the active precompiled hosts."""
    if _hosts is None:
        raise RuntimeError("the wheel was built without native FlagGems hosts")
    return _hosts.audit()


def execution_scope(device=None):
    """Bind the current device-library stream while ordinary eager code runs."""
    from .vendor import stream_scope

    return stream_scope(device)


def _model_runtime():
    """Return the installed runtime for precompiled operator hosts."""
    from torch_fl import _C

    if _hosts is not None and hasattr(_hosts, "typed"):
        from .load import _launcher_module

        return _launcher_module
    return getattr(_C, "_native_host", None)


def compile(model, *, dynamic=True, fullgraph=True):
    """Compile an inference host graph over the wheel's precompiled operators."""
    from .model import compile_model

    return compile_model(model, dynamic=dynamic, fullgraph=fullgraph)
