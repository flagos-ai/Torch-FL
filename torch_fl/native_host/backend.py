# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Versioned build/runtime contract for native-host launch adapters.

This module also loads by filename from setup.py without importing torch_fl.
An adapter is admitted only after its real backend conformance is validated.
"""

from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil


ABI_VERSION = 2
WHEEL_PACKAGE = "torch_fl.native_host._generated"
COMPILED_FIELDS = (
    "abi_version",
    "vendor",
    "compiler_backend",
    "capabilities",
    "scratch_alignment",
    "slot_bytes",
    "slot_alignment",
    "native_source_sha256",
)
_ADAPTERS = {
    "nvidia": "nvidia.py",
    "hygon": "hygon.py",
    "metax": "metax.py",
    "ascend": "ascend.py",
    "enflame": "enflame.py",
}
_ROOT = Path(__file__).resolve().parent
_LAUNCHER_SOURCE = "launcher.cc"
_LAUNCHER_HEADERS = ("backend_api.h", "host_abi.h")


@lru_cache
def get_backend(vendor):
    if vendor not in _ADAPTERS:
        raise RuntimeError(f"no validated native-host launch adapter for {vendor}")
    path = _ROOT / "backends" / _ADAPTERS[vendor]
    spec = importlib.util.spec_from_file_location(f"native_host_backend_{vendor}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Backend()


def driver_sources(adapter, triton_root):
    directory = Path("backends") / adapter.driver_directory
    return {
        (directory / name).as_posix(): Path(triton_root) / directory / name
        for name in adapter.driver_files
    }


def _launcher_extension(
    name,
    root,
    torch,
    native_sources,
    fingerprint,
    *,
    include_dirs=(),
    library_dirs=(),
    libraries=(),
    system_libraries=(),
    define_macros=(),
    depends=(),
):
    from setuptools import Extension

    torch_dir = Path(torch.__file__).parent
    return Extension(
        name,
        [str(root / path) for path in native_sources if not path.endswith(".h")],
        depends=[
            *(str(root / path) for path in native_sources if path.endswith(".h")),
            *depends,
        ],
        include_dirs=[str(root), str(torch_dir / "include"), *include_dirs],
        library_dirs=[*library_dirs, str(torch_dir / "lib")],
        libraries=[*libraries, "torch_python", "torch_cpu", "c10", *system_libraries],
        define_macros=[
            *define_macros,
            ("FLAGOS_NATIVE_HOST_BACKEND_SHA256", f'"{fingerprint}"'),
        ],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        ],
    )


def _native_sources(adapter):
    directory = f"backends/{adapter.vendor}"
    return (
        _LAUNCHER_SOURCE,
        *_LAUNCHER_HEADERS,
        f"{directory}/launch.cc",
        *(f"{directory}/{name}" for name in getattr(adapter, "launcher_headers", ())),
    )


def _native_source_dir():
    packaged = _ROOT / "csrc"
    if packaged.is_dir():
        return packaged
    return _ROOT.parents[1] / "csrc/native_host"


def stage_native_sources(adapter, native_dir, destination):
    """Copy the selected launch ABI inputs into the bundle for verification."""
    for name in _native_sources(adapter):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(native_dir / name, target)


def build_extensions(
    adapter, namespace, native_dir, torch, record, triton_root, source_dir
):
    """Build one CPU-Torch launcher and its published FlagTree utility."""
    from setuptools import Extension

    driver = triton_root / "backends" / adapter.driver_directory
    sdk, utility = adapter.extension_options(driver, source_dir)
    utility = dict(utility)
    utility["extra_compile_args"] = ["-O3", *utility.get("extra_compile_args", ())]
    return [
        _launcher_extension(
            namespace + "._launcher",
            native_dir,
            torch,
            _native_sources(adapter),
            record["native_source_sha256"],
            **sdk,
        ),
        Extension(namespace + "." + adapter.utility_module, **utility),
    ]


def _hashes(paths):
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def build_record(adapter, triton_root, *, native_dir=None):
    """Fingerprint the selected adapter and the published backend loader ABI."""
    if native_dir is None:
        native_dir = _native_source_dir()
    sources = {f"csrc/{name}": native_dir / name for name in _native_sources(adapter)}
    contract = {
        "abi_version": ABI_VERSION,
        "vendor": adapter.vendor,
        "compiler_backend": adapter.compiler_backend,
        "capabilities": adapter.capabilities,
        "scratch_alignment": adapter.scratch_alignment,
        "slot_bytes": 128,
        "slot_alignment": 128,
    }
    native_sources = _hashes(sources)
    contract["native_source_sha256"] = hashlib.sha256(
        json.dumps(
            {"contract": contract, "sources": native_sources}, sort_keys=True
        ).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        **contract,
        "compiler_distribution": {
            "name": adapter.compiler_distribution,
            "version": importlib.metadata.version(adapter.compiler_distribution),
        },
        "utility_modules": [adapter.utility_module],
        "driver_sources": _hashes(driver_sources(adapter, triton_root)),
        "adapter_sources": _hashes(
            {
                name: _ROOT / name
                for name in (
                    "backend.py",
                    "backends/driver.py",
                    "backends/" + _ADAPTERS[adapter.vendor],
                )
            }
        ),
        "native_sources": native_sources,
    }


def validate_record(record, adapter, triton_root, compiled=None, *, native_dir=None):
    """Reject stale/foreign adapters before changing a driver or launching work."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise RuntimeError(
            "native-host backend contract is missing or unsupported; rebuild the wheel"
        )
    if record != build_record(adapter, triton_root, native_dir=native_dir):
        raise RuntimeError(
            "native-host backend source, target or compiler ABI changed; rebuild the wheel"
        )
    if compiled is not None and compiled != {
        name: record[name] for name in COMPILED_FIELDS
    }:
        raise RuntimeError(
            "precompiled native-host backend ABI does not match its manifest"
        )
