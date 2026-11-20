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

"""Platform names and the installed build record, readable before PyTorch loads."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path
from typing import NamedTuple

DEFAULT_PLATFORM = "cuda"
_MODULE_DIR = Path(__file__).resolve().parent


def _platform_table():
    table = _MODULE_DIR / "_platforms.json"
    if not table.is_file():
        table = _MODULE_DIR.parent / "cmake" / "flagos_platforms.json"
    return json.loads(table.read_text(encoding="utf-8"))["accelerators"]


_PLATFORMS = _platform_table()
PLATFORM_VENDOR = {platform: row["vendor"] for platform, row in _PLATFORMS.items()}
_VENDOR_TO_PLATFORM = {vendor: platform for platform, vendor in PLATFORM_VENDOR.items()}
_ACCELERATOR_TO_PLATFORM = {
    **{platform: platform for platform in PLATFORM_VENDOR},
    **_VENDOR_TO_PLATFORM,
    "maca": "metax",
}

# Check other vendor names first if a legacy conf path contains both.
_CONF_VENDORS = tuple(
    vendor for vendor in _VENDOR_TO_PLATFORM if vendor != "nvidia"
) + ("nvidia",)


def platform_for(name: str) -> str:
    """Resolve a platform ID, vendor name or legacy accelerator spelling."""
    return _ACCELERATOR_TO_PLATFORM.get(name.strip().lower(), "")


def vendor_for(platform: str) -> str:
    """Return the vendor for a platform ID, or an empty string."""
    return PLATFORM_VENDOR.get(platform, "")


class BuildInfo(NamedTuple):
    """Facts about the installed build, including the raw legacy accelerator."""

    accelerator: str = ""
    platform: str = ""
    kernels: frozenset[str] = frozenset()
    native_hosts: bool = False
    platform_recorded: bool = False

    @property
    def vendor(self) -> str:
        return vendor_for(self.platform) or self.accelerator

    @property
    def has_native_host_config(self) -> bool:
        # Codegen emits a registration fragment for each native-host conf.
        registration = _PLATFORMS.get(self.platform, {}).get("registration", {})
        return self.native_hosts and bool(registration.get("native_host"))


def _torch_fl_package_dir() -> Path | None:
    # Regular imports already know which wheel owns the module. Tools that load
    # this file by path may be inspecting a different wheel on sys.path.
    if __name__ == "torch_fl._platform":
        return _MODULE_DIR
    try:
        spec = importlib.util.find_spec("torch_fl")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _load_record(path: Path):
    loader = importlib.machinery.SourceFileLoader(path.stem, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    record = importlib.util.module_from_spec(spec)
    try:
        loader.exec_module(record)
    except (OSError, ImportError, SyntaxError):
        return None
    return record


def _legacy_platform(package: Path) -> str:
    alias = _load_record(package / "_platform_alias.py")
    platform = platform_for(str(getattr(alias, "ACCELERATOR", "")))
    if platform:
        return platform
    try:
        marker = (package / "lib" / "flagos_platform").read_text().strip().lower()
    except OSError:
        marker = ""
    if marker:
        return platform_for(marker) or marker
    return "ppu" if (package / "lib_thead").is_dir() else ""


@cache
def _read_build_info(package: Path | None) -> BuildInfo:
    if package is None:
        return BuildInfo()
    record = _load_record(package / "_build_config.py")
    accelerator = str(getattr(record, "ACCELERATOR", "")).strip().lower()
    platform = platform_for(str(getattr(record, "PLATFORM", "")))
    if not platform and accelerator != "cuda":
        platform = platform_for(accelerator)
    return BuildInfo(
        accelerator=accelerator,
        platform=platform or _legacy_platform(package) or platform_for(accelerator),
        kernels=frozenset(
            str(kernel).strip().lower() for kernel in getattr(record, "KERNELS", ())
        ),
        native_hosts=bool(getattr(record, "NATIVE_HOSTS", False)),
        platform_recorded=bool(platform),
    )


def build_info(package: Path | None = None) -> BuildInfo:
    """Read the wheel identity once per package directory."""
    return _read_build_info(package if package is not None else _torch_fl_package_dir())


def build_accelerator() -> str:
    """Return the accelerator as recorded by the build, or an empty string."""
    return build_info().accelerator


def detect_platform() -> str:
    """Return the platform, with SDK and conf fallbacks for legacy builds."""
    build = build_info()
    if build.platform_recorded:
        return build.platform
    if os.environ.get("PPU_SDK") or Path("/usr/local/PPU_SDK").is_dir():
        return "ppu"
    if build.platform and build.platform != DEFAULT_PLATFORM:
        return build.platform

    package = sys.modules.get("torch_fl")
    resolve = getattr(package, "backend_config_path", None)
    config = (
        resolve() if resolve else os.environ.get("FLAGOS_BACKEND_CONFIG", "")
    ).lower()
    for vendor in _CONF_VENDORS:
        if vendor in config:
            return _VENDOR_TO_PLATFORM[vendor]
    return DEFAULT_PLATFORM
