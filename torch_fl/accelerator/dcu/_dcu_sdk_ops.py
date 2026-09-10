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

"""Load and initialize Torch-FL's DTK SDK-native DCU operator plugin.

This module is intentionally independent from vendor-torch discovery. The
plugin is built against the official CPU torch C++ ABI and DTK's torch-free
SDK libraries (currently rocBLAS), and is loaded only after ``torch_fl._C`` has
put the core-side registration bridge into the process.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path


_PLUGIN = "libdcu_aten_ops.so"
_MANIFEST = "dcu_sdk_manifest.json"
_INIT = "FlagosDcuSdkPluginInit"
_STATUS = "FlagosDcuSdkStatusString"
_OK = 0
_TRUTHY = {"1", "on", "true", "yes"}
_HANDLES: list[ctypes.CDLL] = []
_CORE_HANDLES: list[ctypes.CDLL] = []
_LOADED = False
_INITIALIZED = False


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUTHY


def enabled() -> bool:
    """Whether the SDK-native route was requested, by either switch.

    ``FLAGOS_DCU_SDK_ONLY=1`` implies it: SDK-only means "run the SDK cohort with
    no vendor libtorch behind it", which is meaningless unless the plugin is
    loaded. Reading only FLAGOS_DCU_SDK_OPS here would select the SDK config,
    skip the vendor preload, and then leave every routed GEMM without a kernel.
    """
    return _flag("FLAGOS_DCU_SDK_OPS") or sdk_only()


def sdk_only() -> bool:
    """Whether SDK mode must prove no DTK libtorch device library is mapped."""
    return _flag("FLAGOS_DCU_SDK_ONLY")


def _package_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "lib_dcu"


def _sdk_root() -> str:
    for key in ("FLAGOS_DCU_SDK_ROOT", "DTK_ROOT", "ROCM_PATH"):
        value = os.environ.get(key, "").strip()
        if value and os.path.isdir(value):
            return value
    return "/opt/dtk"


def discover_plugin() -> str | None:
    """Find the plugin, preferring the installed package over SDK directories."""
    override = os.environ.get("FLAGOS_DCU_SDK_OPS_LIB", "").strip()
    candidates = [override] if override else []
    package = _package_dir()
    candidates.append(str(package / _PLUGIN))
    root = _sdk_root()
    candidates.extend(str(Path(root) / subdir / _PLUGIN) for subdir in ("lib", "lib64"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def discover_manifest(plugin: str | None = None) -> str | None:
    override = os.environ.get("FLAGOS_DCU_SDK_MANIFEST", "").strip()
    candidates = [override] if override else []
    if plugin:
        candidates.append(str(Path(plugin).with_name(_MANIFEST)))
    candidates.append(str(_package_dir() / _MANIFEST))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def load_manifest(plugin: str | None = None) -> dict:
    path = discover_manifest(plugin)
    if not path:
        raise RuntimeError(
            "DCU SDK-native mode requires dcu_sdk_manifest.json next to "
            f"{_PLUGIN}. Set FLAGOS_DCU_SDK_MANIFEST or rebuild torch-fl."
        )
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read DCU SDK manifest {path}: {exc}") from exc

    required = {
        "schema_version",
        "torch_base",
        "torch_abi",
        "dispatch_key",
        "library",
        "operators",
        "dtypes",
        "layouts",
        "stream_behavior",
        "fallback",
        "sdk_only",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise RuntimeError(
            f"DCU SDK manifest {path} is missing required fields: {', '.join(missing)}"
        )
    if manifest["library"] != _PLUGIN:
        raise RuntimeError(
            f"DCU SDK manifest names {manifest['library']!r}, expected {_PLUGIN!r}"
        )
    if manifest["dispatch_key"] != "PrivateUse1":
        raise RuntimeError(
            "DCU SDK manifest has an unexpected dispatch key: "
            f"{manifest['dispatch_key']!r}"
        )
    if manifest["schema_version"] != 2:
        raise RuntimeError(
            f"Unsupported DCU SDK manifest schema {manifest['schema_version']!r}; "
            "upgrade torch-fl"
        )
    for field in ("torch_version", "sdk_version", "registration_abi", "route_count"):
        if field not in manifest:
            raise RuntimeError(
                f"DCU SDK manifest {path} is missing required schema-2 field: {field}"
            )
    version = manifest["torch_version"]
    if not isinstance(version, dict) or any(
        key not in version for key in ("major", "minor", "patch")
    ):
        raise RuntimeError(
            f"DCU SDK manifest {path} has an invalid torch_version object"
        )
    if manifest["registration_abi"] != 2:
        raise RuntimeError(
            "Unsupported DCU SDK registration ABI "
            f"{manifest['registration_abi']!r}; upgrade torch-fl"
        )
    if manifest["route_count"] <= 0:
        raise RuntimeError(
            f"DCU SDK manifest {path} must contain a positive route_count"
        )
    if not isinstance(manifest["operators"], list) or not manifest["operators"]:
        raise RuntimeError(
            f"DCU SDK manifest {path} must contain a non-empty operators list"
        )
    return manifest


def _active_torch_base() -> str:
    import torch

    return torch.__version__.split("+", 1)[0]


def _check_sdk_available() -> None:
    root = _sdk_root()
    if not os.path.isdir(root):
        raise RuntimeError(
            "DCU SDK-native mode could not find the DTK SDK. Set "
            "FLAGOS_DCU_SDK_ROOT, DTK_ROOT, or ROCM_PATH to the SDK root."
        )
    lib_candidates = [
        os.path.join(root, subdir, "librocblas.so.4") for subdir in ("lib", "lib64")
    ]
    header = os.path.join(root, "include", "rocblas", "rocblas.h")
    if not any(os.path.exists(path) for path in lib_candidates) or not os.path.exists(
        header
    ):
        missing = [path for path in lib_candidates if not os.path.exists(path)]
        raise RuntimeError(
            "DCU SDK-native mode found an incomplete SDK under "
            f"{root}; expected rocBLAS under {', '.join(missing)} and header "
            f"{header}. Install the DTK rocBLAS component or correct "
            "FLAGOS_DCU_SDK_ROOT."
        )


def mapped_core_libraries(maps: str) -> list[str]:
    """Paths of every libtorch_fl.so already mapped into the process."""
    found = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if path.rsplit("/", 1)[-1] == "libtorch_fl.so" and path not in found:
            found.append(path)
    return found


def _core_candidates() -> list[str]:
    """Where to look for the core bridge, most authoritative first.

    The already-mapped path wins. ``torch_fl._C`` reaches libtorch_fl through a
    DT_NEEDED edge resolved by the dynamic loader, which honors LD_LIBRARY_PATH
    ahead of the extension's RUNPATH -- so the live library is not necessarily
    the copy inside the package directory. Opening the package copy in that case
    creates a *second* libtorch_fl in the process, and its static initializers
    re-register the AutocastPrivateUse1 fallback, which aborts with "Tried to
    register multiple backend fallbacks for the same dispatch key". Reopening the
    mapped path instead just bumps its dlopen refcount.
    """
    candidates = []
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    except OSError:
        maps = ""
    candidates.extend(mapped_core_libraries(maps))
    import torch_fl

    package = Path(torch_fl.__file__).resolve().parent
    for path in (package / "lib" / "libtorch_fl.so", package / "libtorch_fl.so"):
        if str(path) not in candidates:
            candidates.append(str(path))
    return candidates


def _core_handle():
    """Return libtorch_fl.so with global symbol visibility.

    Python extension modules are commonly dlopened with RTLD_LOCAL, so
    ``ctypes.CDLL(None)`` may not see the bridge even though the library is
    already mapped. Reopening the live path with RTLD_GLOBAL promotes the
    existing object's symbols without loading a second copy.
    """
    if _CORE_HANDLES:
        return _CORE_HANDLES[0]
    for path in _core_candidates():
        if not os.path.isfile(path):
            continue
        try:
            handle = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        _CORE_HANDLES.append(handle)
        return handle
    return ctypes.CDLL(None)


def _registration_bridge():
    """Resolve the core-side C ABI from the already-loaded torch_fl library."""
    process = _core_handle()
    try:
        register = getattr(process, "FlagosDcuSdkRegisterKernels")
        status_string = getattr(process, _STATUS)
    except AttributeError as exc:
        raise RuntimeError(
            "DCU SDK-native plugin bridge is absent from libtorch_fl.so. Rebuild "
            "torch-fl with the SDK-native registration bridge enabled."
        ) from exc
    register.restype = ctypes.c_int
    register.argtypes = [ctypes.c_void_p]
    status_string.restype = ctypes.c_char_p
    status_string.argtypes = [ctypes.c_int]
    return register, status_string


def _registration_query():
    process = _core_handle()
    try:
        query = getattr(process, "FlagosDcuSdkKernelsRegistered")
    except AttributeError:
        return None
    query.restype = ctypes.c_int
    query.argtypes = []
    return query


def _load_plugin(plugin: str):
    """Load the plugin after promoting the core bridge to RTLD_GLOBAL."""
    _registration_bridge()
    try:
        return ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load DCU SDK plugin {plugin}: {exc}. Check DTK SDK "
            "libraries and the plugin's RUNPATH."
        ) from exc


def load_and_register() -> dict:
    """Load the plugin once, validate its manifest, and register its kernels."""
    global _LOADED, _INITIALIZED
    if _INITIALIZED:
        plugin = discover_plugin()
        return load_manifest(plugin)

    _check_sdk_available()
    plugin = discover_plugin()
    if not plugin:
        raise RuntimeError(
            "SDK-native GEMM kernels were requested (FLAGOS_DCU_SDK_OPS or "
            f"FLAGOS_DCU_SDK_ONLY), but {_PLUGIN} was not found. Build/install "
            "the DCU SDK plugin, or unset both to use the existing libtorch_hip "
            "route."
        )
    manifest = load_manifest(plugin)
    expected = _active_torch_base()
    if manifest["torch_base"] != expected:
        raise RuntimeError(
            "DCU SDK plugin Torch ABI mismatch: manifest targets "
            f"torch {manifest['torch_base']}, active torch is {expected}. "
            "Install the matching official CPU torch or rebuild torch-fl."
        )

    if not _LOADED:
        _HANDLES.append(_load_plugin(plugin))
        _LOADED = True

    try:
        init = getattr(_HANDLES[0], _INIT)
    except AttributeError as exc:
        raise RuntimeError(
            f"{plugin} does not export {_INIT}; rebuild the SDK-native plugin."
        ) from exc
    init.restype = ctypes.c_int
    init.argtypes = []
    status = int(init())
    if status != _OK:
        _, status_string = _registration_bridge()
        rendered = status_string(status)
        message = rendered.decode("utf-8", "replace") if rendered else "unknown"
        raise RuntimeError(
            f"DCU SDK plugin registration failed ({status}): {message}. "
            "Use the exact official torch 2.10.x/C++ ABI used to build torch-fl."
        )
    _INITIALIZED = True
    return manifest


# DTK's torch fork, by library name. libtorch_python.so is included because the
# official +cpu wheel does ship one, but under torch/lib; a vendor copy under the
# bundle or a DTK tree invalidates SDK-only measurements.
_VENDOR_ONLY_LIBS = ("libtorch_hip.so", "libc10_hip.so")
_VENDOR_PATH_LIBS = ("libtorch_cpu.so", "libc10.so", "libtorch_python.so")
_VENDOR_PATH_MARKERS = ("/lib_dcu/", "/dtk/", "/dtk-")


def _vendor_mappings(maps: str) -> list[str]:
    """Vendor-torch libraries mapped into this process, as ``name (path)``."""
    found = {}
    for line in maps.splitlines():
        # <range> <perms> <offset> <dev> <inode> <path>
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if not path.startswith("/"):
            continue
        name = path.rsplit("/", 1)[-1]
        base = name.split(".so", 1)[0] + ".so"
        if base in _VENDOR_ONLY_LIBS:
            found[name] = path
        elif base in _VENDOR_PATH_LIBS and any(
            marker in path for marker in _VENDOR_PATH_MARKERS
        ):
            found[name] = path
    return [f"{name} ({path})" for name, path in sorted(found.items())]


def assert_sdk_only_process() -> None:
    """Fail if SDK-only mode accidentally mapped DTK's torch libraries."""
    if not sdk_only():
        return
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    except OSError:
        return
    found = _vendor_mappings(maps)
    if found:
        raise RuntimeError(
            "DCU SDK-only mode unexpectedly loaded DTK's torch libraries: "
            f"{', '.join(found)}. Unset FLAGOS_DCU_SDK_ONLY, or clear the vendor "
            "preload first: FLAGOS_DCU_VENDOR_CORE must be unset and any legacy "
            "symlinks under torch/lib restored."
        )


def kernels_registered() -> bool:
    """Whether the core bridge accepted a registration in this process."""
    query = _registration_query()
    return bool(query()) if query is not None else False


def supported_operator_names(manifest: dict) -> set[str]:
    return set(manifest.get("operators", ()))
