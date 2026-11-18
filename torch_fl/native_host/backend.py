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


ABI_VERSION = 1
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
_ADAPTERS = {"nvidia": "nvidia.py", "metax": "metax.py", "ascend": "ascend.py"}
_ROOT = Path(__file__).resolve().parent


@lru_cache
def get_backend(vendor):
    if vendor not in _ADAPTERS:
        raise RuntimeError(f"no validated native-host launch adapter for {vendor}")
    path = _ROOT / "backends" / _ADAPTERS[vendor]
    spec = importlib.util.spec_from_file_location(f"native_host_backend_{vendor}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Backend()


def _hashes(paths):
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def build_record(adapter, triton_root):
    """Fingerprint the selected adapter and the published backend loader ABI."""
    sources = {
        name: _ROOT / name
        for name in (
            "csrc/launcher.cc",
            "csrc/backend_api.h",
            "csrc/host_abi.h",
            *adapter.sources,
        )
    }
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
        "utility_modules": list(adapter.utility_modules),
        "driver_sources": _hashes(adapter.driver_sources(Path(triton_root))),
        "adapter_sources": _hashes(
            {
                name: _ROOT / name
                for name in ("backend.py", "backends/" + _ADAPTERS[adapter.vendor])
            }
        ),
        "native_sources": native_sources,
    }


def validate_record(record, adapter, triton_root, compiled=None):
    """Reject stale/foreign adapters before changing a driver or launching work."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise RuntimeError(
            "native-host backend contract is missing or unsupported; rebuild the wheel"
        )
    if record != build_record(adapter, triton_root):
        raise RuntimeError(
            "native-host backend source, target or compiler ABI changed; rebuild the wheel"
        )
    if compiled is not None and compiled != {
        name: record[name] for name in COMPILED_FIELDS
    }:
        raise RuntimeError(
            "precompiled native-host backend ABI does not match its manifest"
        )
