# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Install and audit schema-typed precision exceptions from a prebuilt module."""

import importlib
import importlib.metadata
import json
from pathlib import Path

from .vendor_cohort import digest

_module = None
_manifest = None


def _verify_libraries(module, manifest):
    libraries = module.libraries()
    if set(libraries) != set(manifest["numerical_libraries"]):
        raise RuntimeError("native vendor numerical library set changed")
    for name, path in libraries.items():
        if digest(path) != manifest["numerical_libraries"][name]:
            raise RuntimeError(f"native vendor numerical library changed: {name}")


def activate():
    global _module, _manifest
    import torch
    from torch_fl import _build_config

    if not getattr(_build_config, "NATIVE_VENDOR", False):
        return
    if _build_config.ACCELERATOR != "nvidia" or "boxing" in _build_config.KERNELS:
        raise RuntimeError(
            "native vendor precision requires the NVIDIA non-boxing build"
        )
    if "vendor" not in _build_config.KERNELS:
        raise RuntimeError(
            "native vendor precision was built without the vendor kernel set"
        )
    directory = Path(__file__).parent / "_generated"
    _manifest = json.loads((directory / "vendor_manifest.json").read_text())
    if (
        str(torch.__version__) != _manifest["torch"]
        or torch.version.git_version != _manifest["torch_git_version"]
        or importlib.metadata.version("flag_gems") != _manifest["flag_gems"]
    ):
        raise RuntimeError("native vendor precision dependency cohort changed")
    package = Path(__file__).parents[1]
    for relative, expected in _manifest["artifacts"].items():
        if digest(package / relative) != expected:
            raise RuntimeError(f"native vendor artifact changed: {relative}")
    expected = _manifest["catalog"]
    actual = json.loads(torch.ops.flagos.native_vendor_catalog())
    if expected != actual:
        raise RuntimeError(
            "native vendor schema/policy catalog differs from the core library"
        )
    _module = importlib.import_module("torch_fl.native_host._generated._vendor")
    _verify_libraries(_module, _manifest)
    bindings = _module.bindings()
    if (
        {(row[0], row[1]) for row in bindings}
        != {(row["op"], row["abi"]) for row in expected}
        or len(bindings) != len(expected)
        or any(not row[2] for row in bindings)
    ):
        raise RuntimeError(
            "native vendor module has missing or unexpected typed entries"
        )
    for name, abi, address, _ in bindings:
        torch.ops.flagos.native_vendor_register("nvidia", name, abi, address)


def audit():
    if _module is None:
        return None
    _verify_libraries(_module, _manifest)
    package = Path(__file__).parents[1]
    for relative, expected in _manifest["artifacts"].items():
        if digest(package / relative) != expected:
            raise RuntimeError(
                f"native vendor artifact changed during execution: {relative}"
            )
    return {
        "vendor": "nvidia",
        "vendor_torch": _manifest["vendor_torch"],
        "artifacts": _manifest["artifacts"],
        "numerical_libraries": _manifest["numerical_libraries"],
        "catalog": _manifest["catalog"],
        "hits": {name: hits for name, _, _, hits in _module.bindings()},
    }
