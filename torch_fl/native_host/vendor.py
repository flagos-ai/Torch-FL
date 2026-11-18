# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Install and audit schema-typed precision exceptions from a prebuilt module."""

import importlib
import importlib.metadata
import ctypes
import json
import os
from pathlib import Path
import sys

from .vendor_cohort import digest

_module = None
_manifest = None
_sdk_manifest = None


def _verify_cpu_core():
    import torch

    if not str(torch.__version__).endswith("+cpu") or torch.version.cuda is not None:
        raise RuntimeError("MetaX native vendor precision requires official CPU Torch")
    root = Path(torch.__file__).resolve().parent / "lib"
    maps = Path("/proc/self/maps").read_text().splitlines()
    for name in ("libc10.so", "libtorch_cpu.so", "libtorch_python.so"):
        mapped = {
            Path(line.split()[-1]).resolve()
            for line in maps
            if line.split()[-1].endswith("/" + name)
        }
        if mapped != {(root / name).resolve()}:
            raise RuntimeError(
                f"MetaX native vendor entry changed the CPU core: {name}"
            )


def _load_metax_vendor(package):
    _verify_cpu_core()
    directory = package / "native_host/_generated/metax_vendor"
    for name in ("libc10_cuda.so", "libtorch_cuda.so"):
        ctypes.CDLL(str(directory / name), mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
    try:
        module = importlib.import_module(
            "torch_fl.native_host._generated._metax_vendor"
        )
    finally:
        sys.setdlopenflags(old_flags)
    _verify_cpu_core()
    return module


def _verify_sdk(manifest):
    import torch

    catalog = json.loads(torch.ops.flagos.native_metax_sdk_catalog())
    if catalog != {name: manifest[name] for name in ("policy", "policy_sha256")}:
        raise RuntimeError("native MetaX SDK precision policy differs from the core")
    for name in ("flag_gems", "flagtree"):
        if importlib.metadata.version(name) != manifest[name]:
            raise RuntimeError("native MetaX SDK precision dependency cohort changed")
    libraries = dict(torch.ops.flagos.native_metax_sdk_libraries())
    if libraries.keys() != manifest["numerical_libraries"].keys():
        raise RuntimeError("native MetaX SDK numerical library set changed")
    for name, path in libraries.items():
        record = manifest["numerical_libraries"][name]
        if digest(path) != record["sha256"]:
            raise RuntimeError(f"native MetaX SDK numerical library changed: {name}")


def _verify_libraries(module, manifest):
    libraries = module.libraries()
    if set(libraries) != set(manifest["numerical_libraries"]):
        raise RuntimeError("native vendor numerical library set changed")
    for name, path in libraries.items():
        if digest(path) != manifest["numerical_libraries"][name]:
            raise RuntimeError(f"native vendor numerical library changed: {name}")


def activate():
    global _module, _manifest, _sdk_manifest
    import torch
    from torch_fl import _build_config

    if _build_config.ACCELERATOR == "metax" and "vendor" in _build_config.KERNELS:
        if "boxing" in _build_config.KERNELS:
            raise RuntimeError("native MetaX SDK precision requires a non-boxing build")
        directory = Path(__file__).parent / "_generated"
        _sdk_manifest = json.loads((directory / "manifest.json").read_text()).get(
            "sdk_precision"
        )
        if _sdk_manifest is None:
            raise RuntimeError("native MetaX SDK precision manifest is missing")
        _verify_sdk(_sdk_manifest)
        if not getattr(_build_config, "NATIVE_VENDOR", False):
            return
    if not getattr(_build_config, "NATIVE_VENDOR", False):
        return
    vendor = _build_config.ACCELERATOR
    if vendor not in ("nvidia", "metax") or "boxing" in _build_config.KERNELS:
        raise RuntimeError(
            "native vendor precision requires a measured non-boxing build"
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
    _module = (
        _load_metax_vendor(package)
        if vendor == "metax"
        else importlib.import_module("torch_fl.native_host._generated._vendor")
    )
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
        torch.ops.flagos.native_vendor_register(vendor, name, abi, address)


def audit():
    sdk = None
    if _sdk_manifest is not None:
        import torch

        _verify_sdk(_sdk_manifest)
        sdk = {
            "vendor": "metax",
            "vendor_torch": _sdk_manifest["policy"]["vendor_torch"],
            "policy": _sdk_manifest["policy"],
            "catalog": _sdk_manifest["policy"]["operators"],
            "numerical_libraries": _sdk_manifest["numerical_libraries"],
            "hits": dict(torch.ops.flagos.native_metax_sdk_hits()),
        }
    if _module is None:
        return sdk
    _verify_libraries(_module, _manifest)
    package = Path(__file__).parents[1]
    for relative, expected in _manifest["artifacts"].items():
        if digest(package / relative) != expected:
            raise RuntimeError(
                f"native vendor artifact changed during execution: {relative}"
            )
    vendor = _manifest.get("vendor", "nvidia")
    if vendor == "metax":
        _verify_cpu_core()
    result = {
        "vendor": vendor,
        "vendor_torch": _manifest["vendor_torch"],
        "artifacts": _manifest["artifacts"],
        "numerical_libraries": _manifest["numerical_libraries"],
        "catalog": _manifest["catalog"],
        "hits": {name: hits for name, _, _, hits in _module.bindings()},
    }
    if sdk is not None:
        result["sdk_precision"] = sdk
        result["catalog"] = sdk["catalog"] + result["catalog"]
        result["hits"] = {**sdk["hits"], **result["hits"]}
    return result
