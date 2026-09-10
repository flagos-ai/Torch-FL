#!/usr/bin/env python3
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

"""Prove libdcu_aten_ops.so links no part of DTK's forked PyTorch.

The SDK-native plugin exists to run DCU matmuls on the official PyTorch core, so
"does not depend on the vendor torch" is its defining property -- and one that a
single stray ``target_link_libraries`` entry silently breaks, because DTK's
libraries are present on any machine that can build this at all. The check is
therefore mechanical and runs in CI:

1. ``DT_NEEDED`` contains none of DTK's torch libraries.
2. ``DT_NEEDED`` does contain the official core (libc10 / libtorch_cpu) and
   rocBLAS, so we know the plugin really is bound to both sides.
3. The registration entry point is exported, since the loader finds it by dlsym
   and an accidental ``-fvisibility=hidden`` on it would only fail at runtime.
4. Its ``RUNPATH`` names no build-machine torch directory, which would make the
   installed wheel load a different interpreter's libtorch.

Usage:
    python scripts/check_dcu_sdk_abi.py torch_fl/lib_dcu/libdcu_aten_ops.so
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

# DTK's fork of the PyTorch runtime, in any of the shapes it ships. The plugin
# must reference none of these: libtorch_hip/libc10_hip are the device halves the
# whole SDK-native effort replaces, and the others would drag in the forked core.
_FORBIDDEN = (
    "libtorch_hip.so",
    "libc10_hip.so",
    "libtorch.so",
    "libtorch_python.so",
    "libtorch_global_deps.so",
    "libmagma.so",
    "libcaffe2_nvrtc.so",
)

# The official core the plugin's at::Tensor/at::Scalar handling resolves against,
# plus the SDK library that actually does the maths.
_REQUIRED = ("libc10.so", "librocblas.so.4")

_ENTRY_POINTS = ("FlagosDcuSdkPluginInit", "FlagosDcuSdkPluginAbiVersion")

_NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]")
_RUNPATH_RE = re.compile(
    r"\((?:RUNPATH|RPATH)\)\s+Library r(?:unpath|path): \[([^\]]*)\]"
)


def _readelf(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["readelf", "-d", str(path)], text=True, stderr=subprocess.STDOUT
        )
    except FileNotFoundError as exc:
        raise RuntimeError("readelf is required for the DCU SDK ABI check") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"readelf failed for {path}:\n{exc.output}") from exc


def needed(path: Path) -> list[str]:
    return _NEEDED_RE.findall(_readelf(path))


def runpath(path: Path) -> list[str]:
    entries: list[str] = []
    for value in _RUNPATH_RE.findall(_readelf(path)):
        entries.extend(part for part in value.split(":") if part)
    return entries


def exported(path: Path) -> set[str]:
    try:
        output = subprocess.check_output(
            ["nm", "-D", "--defined-only", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nm is required for the DCU SDK ABI check") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nm failed for {path}:\n{exc.output}") from exc
    symbols = set()
    for line in output.splitlines():
        fields = line.split()
        if fields:
            symbols.add(fields[-1].split("@", 1)[0])
    return symbols


def verify_plugin(path: Path) -> list[str]:
    """Raise on any violation; return the plugin's DT_NEEDED list on success."""
    if not path.is_file():
        raise FileNotFoundError(f"DCU SDK plugin is missing: {path}")

    deps = needed(path)
    # Compare on the soname prefix: DT_NEEDED carries versions
    # (librocblas.so.4), and DTK's auditwheel-mangled names carry a hash
    # (libtorch_hip-1234abcd.so), so exact equality would miss both.
    vendor = sorted(
        {
            dep
            for dep in deps
            for bad in _FORBIDDEN
            if dep.split(".so", 1)[0] == bad.split(".so", 1)[0]
        }
    )
    if vendor:
        raise RuntimeError(
            f"{path.name} depends on DTK's forked PyTorch runtime: "
            + ", ".join(vendor)
            + ". The SDK-native plugin must link only the official core plus the "
            "DTK SDK; check target_link_libraries(libdcu_aten_ops ...) in "
            "csrc/CMakeLists.txt."
        )

    absent = [
        want
        for want in _REQUIRED
        if not any(dep.split(".so", 1)[0] == want.split(".so", 1)[0] for dep in deps)
    ]
    if absent:
        raise RuntimeError(
            f"{path.name} is missing expected dependencies: {', '.join(absent)}. "
            "Without them the plugin is not bound to the official core and the "
            "SDK at all, so the check cannot confirm what it links."
        )

    symbols = exported(path)
    missing = [name for name in _ENTRY_POINTS if name not in symbols]
    if missing:
        raise RuntimeError(
            f"{path.name} does not export {', '.join(missing)}; the Python loader "
            "resolves these by dlsym and would fail at import."
        )

    leaked = [
        entry
        for entry in runpath(path)
        if "site-packages/torch/lib" in entry or "dist-packages/torch/lib" in entry
    ]
    if leaked:
        raise RuntimeError(
            f"{path.name} has a build-machine torch directory in its RUNPATH: "
            + ", ".join(leaked)
            + ". flagos_set_portable_rpath() should have stripped it."
        )

    return deps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin", type=Path, help="path to libdcu_aten_ops.so")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        deps = verify_plugin(args.plugin)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"DCU SDK plugin ABI check passed: {args.plugin.name} links "
        f"{len(deps)} libraries, none of them DTK's forked PyTorch"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
