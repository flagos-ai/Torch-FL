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

"""The one platform/build detector: build record, PPU signals, marker, conf.

Platform detection used to exist in three copies that disagreed:
``torch_fl/__init__.py`` read only the build record, ``tests/integration/ops/
conftest.py`` fell back to ``"default"``, and ``tests/integration/
platform_support.py`` fell back to ``"cuda"``. A test could therefore think it
was on a platform the wheel was not built for. This module is the single source
for both the package and the test trees.

Stdlib-only by design: ``tests/integration`` loads it by path (the ``_env.py``
precedent) because importing ``torch_fl`` executes ``torch_fl/__init__.py``,
which imports torch and claims the PrivateUse1 key -- side effects the test
conftest must run before torch is imported.

Two functions:

``build_accelerator()``
    What the installed wheel was built for, from the ``_build_config.py`` the
    build writes. ``""`` for a source checkout with no build. The environment is
    deliberately ignored: ``FLAGOS_ACCELERATOR`` is a build input, and a stale
    export from another build must not steer detection.

``detect_platform()``
    The platform *name* the integration contracts use, resolving the build
    record first, then PPU's markers (older PPU wheels report cuda), then the
    ``lib/flagos_platform`` marker and the ``lib_thead`` bundle dir, then the
    selected conf, and finally ``DEFAULT_PLATFORM``.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path

#: The name returned when nothing identifies the build. "cuda" is the safe
#: default: the CUDA-ABI platforms are the base every vendor conf inherits, so an
#: unidentified wheel behaves as one. A caller that wants a distinct "unknown"
#: bucket must name it itself rather than rely on this falling through.
DEFAULT_PLATFORM = "cuda"

#: Platform name -> vendor layer, mirroring ``vendor`` in
#: cmake/flagos_platforms.json. The vendor is the stable, public half of a
#: platform's name: it is what FLAGOS_ACCELERATOR accepts and, where the platform
#: publishes, exactly the ``<lane>`` of flagos-pypi-<lane>. The platform name
#: stays the build-table id (cuda, dcu, ...), because CMake, setup.py, the wheel
#: filename and the CI lane files all key on it; a rename of those is its own
#: change. The chip layer (Ascend 310B/910B, ...) is source-only and never
#: appears here. A test keeps this table and the JSON equal.
PLATFORM_VENDOR = {
    "ascend": "ascend",
    "mps": "apple",
    "bpu": "drobotics",
    "dcu": "hygon",
    "gcu": "enflame",
    "iluvatar": "iluvatar",
    "kunlunxin": "kunlunxin",
    "metax": "metax",
    "musa": "mthreads",
    "cuda": "nvidia",
    "ppu": "thead",
    "tsingmicro": "tsingmicro",
}

#: Vendor layer -> platform name. A FLAGOS_ACCELERATOR may name either half.
_VENDOR_TO_PLATFORM = {vendor: platform for platform, vendor in PLATFORM_VENDOR.items()}

#: Build-record accelerator -> platform name. "maca" is the older MetaX spelling,
#: and older records carry the platform id itself (cuda, dcu, ...); a record
#: written by a newer build may carry the vendor name. Both resolve here.
_ACCELERATOR_TO_PLATFORM = {
    **{platform: platform for platform in PLATFORM_VENDOR},
    **_VENDOR_TO_PLATFORM,
    "maca": "metax",
}
# A record that reads "cuda" is ambiguous: it is both NVIDIA's platform id and
# the value older builds wrote for a PPU wheel that predated its own id. Those
# wheels are identified by PPU_SDK / the lib_thead bundle, so the record alone must
# NOT answer for them -- only a non-identity record (a vendor name, or a platform
# id that is not also a fallback value) short-circuits detect_platform().
_AMBIGUOUS_ACCELERATORS = frozenset({"cuda"})

#: Conf-name fragments, matched in order, for the last-resort case where the
#: marker and the record are both absent. "nvidia" is last so a vendor conf that
#: happens to mention it cannot win over its own name. These are the conf file
#: names (backends_<vendor>.conf), so they are vendor names; each maps back to its
#: platform through _VENDOR_TO_PLATFORM, because detect_platform() answers in
#: platform ids.
_CONF_VENDOR_FRAGMENTS = (
    "ascend",
    "metax",
    "mthreads",
    "enflame",
    "hygon",
    "thead",
    "kunlunxin",
    "iluvatar",
    "apple",
    "tsingmicro",
    "nvidia",
)


def _torch_fl_package_dir() -> Path | None:
    """Locate the torch_fl package without executing it (find_spec)."""
    try:
        spec = importlib.util.find_spec("torch_fl")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def build_accelerator() -> str:
    """The accelerator the installed wheel was built for ("" if unknown).

    Reads ``torch_fl/_build_config.py`` -- the record setup.py writes -- by path
    so a caller that must not execute ``torch_fl/__init__.py`` still gets an
    answer. find_spec() finds the package without importing it; the record
    itself imports nothing.
    """
    package = _torch_fl_package_dir()
    if package is None:
        return ""
    path = package / "_build_config.py"
    try:
        loader = importlib.machinery.SourceFileLoader("_flagos_build_record", str(path))
        module = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(loader.name, loader)
        )
        loader.exec_module(module)
    except (OSError, ImportError, SyntaxError):
        # A source checkout with no build yet; the marker/conf below identify it.
        return ""
    return str(getattr(module, "ACCELERATOR", "")).strip().lower()


def _alias_record(package: Path) -> str:
    """``torch_fl/_platform_alias.py``'s platform id ("" when unreadable)."""
    path = package / "_platform_alias.py"
    try:
        loader = importlib.machinery.SourceFileLoader(
            "_flagos_platform_alias", str(path)
        )
        module = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(loader.name, loader)
        )
        loader.exec_module(module)
    except (OSError, ImportError, SyntaxError):
        return ""
    return str(getattr(module, "ACCELERATOR", "")).strip().lower()


def platform_for(name: str) -> str:
    """The platform a ``FLAGOS_ACCELERATOR`` (or build record) value names.

    Accepts both halves of the name: the vendor (``nvidia``, ``hygon``,
    ``mthreads``, ``enflame``, ``thead``, ``drobotics``, ``apple``) and the
    platform id it belongs to (``cuda``, ``dcu``, ``musa``, ``gcu``, ``ppu``,
    ``bpu``, ``mps``). ``""`` for an unknown name, which callers report as the
    error they already did -- a second, differently-worded "unknown accelerator"
    message here would be the two-copies problem this module exists to avoid.
    """
    return _ACCELERATOR_TO_PLATFORM.get(name.strip().lower(), "")


def vendor_for(platform: str) -> str:
    """The vendor layer of a platform name ("" if the platform is unknown)."""
    return PLATFORM_VENDOR.get(platform, "")


def _resolved_conf() -> str:
    """The conf torch_fl resolved, or the FLAGOS_BACKEND_CONFIG the user set.

    Asked of torch_fl when it is importable -- its ``backend_config_path()``
    answers from the wheel's own choice, which is no longer written to the
    environment. The variable is the fallback for a source checkout, where
    importing torch_fl would run the device-claiming side effects.
    """
    try:
        import torch_fl
    except ImportError:
        return os.environ.get("FLAGOS_BACKEND_CONFIG", "")
    resolve = getattr(torch_fl, "backend_config_path", None)
    return resolve() if resolve else os.environ.get("FLAGOS_BACKEND_CONFIG", "")


def detect_platform() -> str:
    """The active platform name, resolved as documented in the module docstring."""
    accelerator = build_accelerator()
    if accelerator not in _AMBIGUOUS_ACCELERATORS:
        platform = _ACCELERATOR_TO_PLATFORM.get(accelerator)
        if platform:
            return platform

    # PPU signals come before the marker: a PPU wheel predating its own build-record
    # value reports cuda, whose row writes no lib/flagos_platform marker, so
    # PPU_SDK is what identifies it.
    if os.environ.get("PPU_SDK") or Path("/usr/local/PPU_SDK").is_dir():
        return "ppu"

    package = _torch_fl_package_dir()
    if package is not None:
        # The build record's twin: the row this wheel was built for, in the
        # table's spelling whatever name the build was told (`metax` is both a
        # row and its vendor, and the row is `metax`).
        if (package / "_platform_alias.py").is_file():
            return _alias_record(package) or DEFAULT_PLATFORM
        try:
            marker = (package / "lib" / "flagos_platform").read_text().strip().lower()
        except OSError:
            marker = ""
        if marker:
            return marker
        if (package / "lib_thead").is_dir():
            return "ppu"

    config = _resolved_conf().lower()
    for fragment in _CONF_VENDOR_FRAGMENTS:
        if fragment in config:
            return _VENDOR_TO_PLATFORM[fragment]
    return DEFAULT_PLATFORM
