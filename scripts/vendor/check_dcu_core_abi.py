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

"""Verify that DTK device libraries need no unaccounted vendor-core symbols.

A decoupled DCU wheel keeps DTK's device libraries but drops DTK's fork of the
PyTorch core. That only works while every symbol those libraries import from the
fork also exists in the official core -- plus a small, known set supplied by
``libflagos_dtk_core_compat.so``. This script measures that set with ``nm`` and
fails when it changes, so a DTK upgrade is caught at build time instead of as an
undefined-symbol ImportError (or a much later dispatch crash) on the target.

Two consumers are checked: DTK's own device libraries, and optionally
``libtorch_fl.so`` via ``--plugin``. A plugin built against DTK's patched ATen
headers imports a second family of private schema wrappers; the official-header
CI build does not. The manifest is therefore an allowed compatibility superset,
not an exact snapshot of every individual build's imports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable


_DEVICE_LIBS = ("libc10_hip.so", "libtorch_hip.so")
_CORE_LIBS = ("libc10.so", "libtorch_cpu.so")


def _nm_symbols(path: Path, *, defined: bool) -> set[str]:
    option = "--defined-only" if defined else "--undefined-only"
    try:
        output = subprocess.check_output(
            ["nm", "-D", option, str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nm is required for the DCU core ABI check") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nm failed for {path}:\n{exc.output}") from exc

    symbols = set()
    for line in output.splitlines():
        fields = line.split()
        if fields:
            symbols.add(fields[-1].split("@", 1)[0])
    return symbols


def _symbols(paths: Iterable[Path], *, defined: bool) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"DCU ABI input is missing: {path}")
        result.update(_nm_symbols(path, defined=defined))
    return result


def read_manifest(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"DCU compatibility manifest is missing: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def vendor_core_gap(
    vendor_lib: Path, official_lib: Path, plugin: Path | None = None
) -> set[str]:
    """Symbols imported from DTK's core fork that the official core lacks.

    ``plugin`` (``libtorch_fl.so``) is folded into the same set: its DTK-private
    imports have to be satisfied by the same shim, and grouping them keeps one
    manifest as the single source of truth.
    """
    consumers = [vendor_lib / name for name in _DEVICE_LIBS]
    if plugin is not None:
        consumers.append(plugin)
    undefined = _symbols(consumers, defined=False)
    vendor_core_exports = _symbols(
        (vendor_lib / name for name in _CORE_LIBS), defined=True
    )
    official_core_exports = _symbols(
        (official_lib / name for name in _CORE_LIBS), defined=True
    )
    return (undefined & vendor_core_exports) - official_core_exports


def verify_core_abi(
    vendor_lib: Path,
    official_lib: Path,
    shim: Path,
    manifest: Path,
    plugin: Path | None = None,
) -> set[str]:
    expected = read_manifest(manifest)
    actual = vendor_core_gap(vendor_lib, official_lib, plugin)
    missing_from_manifest = actual - expected
    if missing_from_manifest:
        raise RuntimeError(
            "new vendor-core-only imports appeared outside the compatibility superset; "
            "update and review the compatibility shim before bundling:\n  "
            + "\n  ".join(sorted(missing_from_manifest))
        )

    shim_exports = _nm_symbols(shim, defined=True)
    missing_from_shim = expected - shim_exports
    if missing_from_shim:
        raise RuntimeError(
            f"{shim} does not export every manifest symbol:\n  "
            + "\n  ".join(sorted(missing_from_shim))
        )
    return actual


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-lib", required=True, type=Path)
    parser.add_argument("--official-lib", required=True, type=Path)
    parser.add_argument("--shim", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--plugin",
        type=Path,
        default=None,
        help="libtorch_fl.so, checked alongside the DTK device libraries",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        gap = verify_core_abi(
            args.vendor_lib,
            args.official_lib,
            args.shim,
            args.manifest,
            args.plugin,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        "DCU core ABI check passed: "
        f"{len(gap)} vendor-private imports are covered by {args.shim.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
