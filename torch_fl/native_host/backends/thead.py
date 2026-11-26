# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HGGC launch adapter for the published PPU FlagTree backend."""

import os
from pathlib import Path

from ..backend import (
    BackendConfig,
    BackendInterface,
    BuildOptions,
    LauncherOptions,
    align_published_embedding_device,
    align_published_mul_device,
)


# These are entries from the published PPU tuning domains. Resolve the
# configurations against the live wrapper before using them for eager launch.
_EAGER_CONFIGS = {
    (
        "mm_kernel_general",
        "d37eb3ca6511b79340f2c199428873baf7d1e704f34146e154c61c1801aa8d89",
    ): dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4, num_ctas=1),
}


class Backend(BackendInterface):
    eager_kernel_configs = _EAGER_CONFIGS

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

    def initialize_runtime(self) -> None:
        align_published_embedding_device("cuda")
        align_published_mul_device("cuda")

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
