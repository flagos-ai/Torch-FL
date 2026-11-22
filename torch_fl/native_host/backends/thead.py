# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HGGC launch adapter for the published PPU FlagTree backend."""

import os
from pathlib import Path

from ..backend import BackendConfig, BackendInterface, BuildOptions, LauncherOptions


class Backend(BackendInterface):
    config = BackendConfig(
        vendor="thead",
        compiler_backend="ppu",
        driver_directory="ppu",
        driver_files=("driver.py", "driver.c", "compiler.py"),
        driver_symbols=("PPUDriver", "HggcUtils", "HggcLauncher"),
        utility_module="hggc_utils",
        utility_exports=("PyHGtensorMap",),
        capabilities=253,
        scratch_alignment=256,
        shared_headers=("launch_params.h",),
    )

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        sdk = Path(
            os.environ.get("PPU_SDK")
            or os.environ.get("PPU_HOME")
            or "/usr/local/PPU_SDK"
        )
        toolkit: LauncherOptions = {
            "include_dirs": [str(sdk / "include")],
            "library_dirs": [str(sdk / "lib"), str(sdk / "lib64")],
            "libraries": ["hggc"],
        }
        return toolkit, {**toolkit, "sources": [str(driver / "driver.c")]}
