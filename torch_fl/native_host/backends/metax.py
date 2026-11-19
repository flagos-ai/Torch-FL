# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MACA launch adapter for the published MetaX FlagTree backend."""

import os
from pathlib import Path

from ..backend import BackendConfig, BackendInterface, BuildOptions, LauncherOptions


class Backend(BackendInterface):
    config = BackendConfig(
        vendor="metax",
        compiler_backend="maca",
        driver_directory="metax",
        driver_files=("driver.py", "driver.c", "compiler.py"),
        driver_symbols=("MacaDriver", "MacaUtils", "MacaLauncher"),
        utility_module="maca_utils",
        capabilities=141,
        scratch_alignment=256,
    )

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        maca = Path(
            os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME") or "/opt/maca"
        )
        toolkit: LauncherOptions = {
            "include_dirs": [str(maca / "include")],
            "library_dirs": [str(maca / "lib"), str(maca / "lib64")],
            "libraries": ["mcruntime"],
        }
        return toolkit, {
            **toolkit,
            "sources": [str(driver / "driver.c")],
            "define_macros": [("__MACA__", "1")],
        }
