# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Versioned build/runtime contract for native-host launch adapters.

Setup reads the admission registry without importing torch_fl. Adapters load
on demand when building or activating a native-host wheel.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
from types import ModuleType, SimpleNamespace
from typing import TypedDict


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
# Admit a backend after it passes installed-wheel conformance on hardware.
_ADAPTERS = frozenset(
    {
        "ascend",
        "cambricon",
        "enflame",
        "hygon",
        "iluvatar",
        "kunlunxin",
        "metax",
        "mthreads",
        "nvidia",
        "thead",
        "tsingmicro",
    }
)
_ROOT = Path(__file__).resolve().parent
_LAUNCHER_SOURCE = "launcher.cc"
_LAUNCHER_HEADERS = ("backend_api.h", "enqueue_trace.h", "host_abi.h")


@dataclass(frozen=True)
class BackendConfig:
    vendor: str
    compiler_backend: str
    driver_directory: str
    driver_files: tuple[str, ...]
    utility_module: str
    capabilities: int
    scratch_alignment: int
    compiler_distribution: str = "flagtree"
    driver_entry: str = "driver"
    driver_symbols: tuple[str, str, str] | None = None
    utility_exports: tuple[str, ...] = ()
    launcher_headers: tuple[str, ...] = ()
    shared_headers: tuple[str, ...] = ()


class _ExtensionOptions(TypedDict, total=False):
    include_dirs: list[str]
    library_dirs: list[str]
    libraries: list[str]
    define_macros: list[tuple[str, str]]


class LauncherOptions(_ExtensionOptions, total=False):
    system_libraries: list[str]
    depends: list[str]
    extra_objects: list[str]


class UtilityOptions(_ExtensionOptions, total=False):
    sources: list[str]
    language: str
    extra_compile_args: list[str]


BuildOptions = tuple[LauncherOptions, UtilityOptions]


def align_published_embedding_device(device: str) -> None:
    flag_gems = importlib.import_module("flag_gems")
    provider = getattr(flag_gems, "embedding_dense_backward", None)
    if provider is None:
        return
    module = importlib.import_module(provider.__module__)
    source = Path(module.__file__)
    # This provider uses the binding only to validate its input tensors.
    if getattr(
        getattr(module, "flag_gems", None), "device", None
    ) == device and hashlib.sha256(source.read_bytes()).hexdigest() in {
        "7f28f60d77ef087943f43da45ccf8ad77f411d4b1ff7cbc61f2f91c578da98ef",
        "829984a49951853946c732d9dc049c88fd909e5c76f5954cec3e799ce4f657cc",
    }:
        module.flag_gems = SimpleNamespace(device="flagos")


def align_published_mul_device(device: str) -> None:
    module = importlib.import_module("flag_gems.ops.mul")
    source = Path(module.__file__)
    # This provider caches the accelerator name before PrivateUse1 is renamed.
    if (
        getattr(module, "_DEVICE_NAME", None) == device
        and hashlib.sha256(source.read_bytes()).hexdigest()
        == "df1d9068fbe35596b74bad406446b39901deb6777e2439cc09bb2b6d1de0f348"
    ):
        module._DEVICE_NAME = "flagos"


class BackendInterface(ABC):
    """Configuration and hooks for one native-host launch backend."""

    @property
    @abstractmethod
    def config(self) -> BackendConfig:
        """Describe the published driver and launch ABI."""

    @abstractmethod
    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        """Return options for the launcher and the published driver utility."""

    def prepare_runtime(self) -> None:
        """Prepare vendor compatibility before installing the prebuilt driver."""

    def initialize_runtime(self) -> None:
        """Initialize the device before loading FlagGems host functions."""

    def bind_driver(
        self,
        module: ModuleType,
        utilities: ModuleType,
        create_launcher: Callable[..., object],
        current_stream: Callable[[int], int],
    ) -> None:
        """Install prebuilt factories on the published driver."""
        from .backends.driver import bind_driver

        symbols = self.config.driver_symbols
        if symbols is None:
            raise NotImplementedError(
                f"{self.config.vendor} must implement bind_driver"
            )
        bind_driver(
            module,
            symbols,
            utilities,
            create_launcher,
            current_stream,
            utility_exports=self.config.utility_exports,
        )


def require_backend(vendor: str) -> None:
    if vendor not in _ADAPTERS:
        raise RuntimeError(f"no validated native-host launch adapter for {vendor}")


@lru_cache
def get_backend(vendor: str) -> BackendInterface:
    require_backend(vendor)
    module = importlib.import_module(f".backends.{vendor}", __package__)
    adapter = module.Backend()
    if not isinstance(adapter, BackendInterface) or adapter.config.vendor != vendor:
        raise RuntimeError(f"invalid native-host launch adapter for {vendor}")
    return adapter


def driver_sources(adapter: BackendInterface, triton_root: Path):
    config = adapter.config
    directory = Path("backends") / config.driver_directory
    return {
        (directory / name).as_posix(): Path(triton_root) / directory / name
        for name in config.driver_files
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
    extra_objects=(),
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
        extra_objects=list(extra_objects),
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


def _native_sources(config: BackendConfig):
    directory = f"backends/{config.vendor}"
    return (
        _LAUNCHER_SOURCE,
        *_LAUNCHER_HEADERS,
        *config.shared_headers,
        f"{directory}/launch.cc",
        *(f"{directory}/{name}" for name in config.launcher_headers),
    )


def _native_source_dir():
    packaged = _ROOT / "csrc"
    if packaged.is_dir():
        return packaged
    return _ROOT.parents[1] / "csrc/native_host"


def stage_native_sources(
    adapter: BackendInterface, native_dir: Path, destination: Path
):
    """Copy the selected launch ABI inputs into the bundle for verification."""
    for name in _native_sources(adapter.config):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(native_dir / name, target)


def build_extensions(
    adapter: BackendInterface,
    namespace,
    native_dir,
    torch,
    record,
    triton_root,
    source_dir,
):
    """Build one CPU-Torch launcher and its published FlagTree utility."""
    from setuptools import Extension

    config = adapter.config
    driver = triton_root / "backends" / config.driver_directory
    sdk, utility = adapter.extension_options(driver, source_dir)
    fingerprint = record["native_source_sha256"]
    source_dir.mkdir(parents=True, exist_ok=True)
    stamp = source_dir / "launcher.sha256"
    content = f"{fingerprint}\n"
    if not stamp.is_file() or stamp.read_text() != content:
        stamp.write_text(content)
    sdk = {**sdk, "depends": [*sdk.get("depends", ()), str(stamp)]}
    utility = utility.copy()
    utility["extra_compile_args"] = ["-O3", *utility.get("extra_compile_args", ())]
    return [
        _launcher_extension(
            namespace + "._launcher",
            native_dir,
            torch,
            _native_sources(config),
            fingerprint,
            **sdk,
        ),
        Extension(namespace + "." + config.utility_module, **utility),
    ]


def _hashes(paths):
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def build_record(adapter: BackendInterface, triton_root: Path, *, native_dir=None):
    """Fingerprint the selected adapter and the published backend loader ABI."""
    if native_dir is None:
        native_dir = _native_source_dir()
    config = adapter.config
    sources = {f"csrc/{name}": native_dir / name for name in _native_sources(config)}
    contract = {
        "abi_version": ABI_VERSION,
        "vendor": config.vendor,
        "compiler_backend": config.compiler_backend,
        "capabilities": config.capabilities,
        "scratch_alignment": config.scratch_alignment,
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
            "name": config.compiler_distribution,
            "version": importlib.metadata.version(config.compiler_distribution),
        },
        "utility_modules": [config.utility_module],
        "driver_sources": _hashes(driver_sources(adapter, triton_root)),
        "adapter_sources": _hashes(
            {
                name: _ROOT / name
                for name in (
                    "backend.py",
                    "backends/driver.py",
                    f"backends/{config.vendor}.py",
                )
            }
        ),
        "native_sources": native_sources,
    }


def validate_record(
    record,
    adapter: BackendInterface,
    triton_root: Path,
    compiled=None,
    *,
    native_dir=None,
):
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
