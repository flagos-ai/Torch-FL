# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build-time compiled FlagGems hosts, enabled by the wheel's build record."""

import os
from pathlib import Path

_hosts = None
_prepared = False


def _bootstrap_launcher():
    """Install prebuilt CUDA utilities before FlagGems initializes Triton."""
    global _prepared
    from torch_fl import _build_config

    if os.environ.get("_TORCH_FL_BUILDING_NATIVE_HOST") == "1":
        return
    if not getattr(_build_config, "NATIVE_HOSTS", False):
        return
    if _build_config.ACCELERATOR != "nvidia" or "boxing" in _build_config.KERNELS:
        raise RuntimeError("native hosts require the NVIDIA vendor build")
    manifest = Path(__file__).parent / "_generated/manifest.json"
    if not manifest.is_file():
        raise RuntimeError("this wheel's precompiled FlagGems hosts are missing")
    from .load import install_launcher

    install_launcher("torch_fl.native_host._generated")
    _prepared = True


def _activate():
    global _hosts
    if not _prepared:
        return
    from .load import Hosts

    _hosts = Hosts(Path(__file__).parent / "_generated")
    _hosts.activate()


def audit():
    """Return source and artifact evidence for the active precompiled hosts."""
    if _hosts is None:
        raise RuntimeError("the wheel was built without native FlagGems hosts")
    return _hosts.audit()
