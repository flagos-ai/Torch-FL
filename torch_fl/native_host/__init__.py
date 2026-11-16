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
    prepare = getattr(backend, "prepare_runtime", None)
    if prepare is not None:
        prepare()
    from .load import install_launcher

    install_launcher("torch_fl.native_host._generated")
    _prepared = True


def _activate():
    global _hosts
    if not _prepared:
        return
    from torch_fl import flagos
    from .load import Hosts

    flagos.init()
    _hosts = Hosts(Path(__file__).parent / "_generated")
    _hosts.activate()
    from . import vendor

    vendor.activate()


def audit():
    """Return source and artifact evidence for the active precompiled hosts."""
    if _hosts is None:
        raise RuntimeError("the wheel was built without native FlagGems hosts")
    return _hosts.audit()


def compile(model, *, dynamic=True, fullgraph=True):
    """Compile an inference model's host graph; numerical kernels stay FlagGems."""
    from .model import compile_model

    return compile_model(model, dynamic=dynamic, fullgraph=fullgraph)
