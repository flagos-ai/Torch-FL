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
_MUL_SOURCE = "df1d9068fbe35596b74bad406446b39901deb6777e2439cc09bb2b6d1de0f348"
_MUL_CONFIG = dict(BLOCK_SIZE=1024, num_warps=4, num_stages=3, num_ctas=1)
_EAGER_CONFIGS = {
    (
        "mm_kernel_general",
        "d37eb3ca6511b79340f2c199428873baf7d1e704f34146e154c61c1801aa8d89",
    ): dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4, num_ctas=1),
    (
        "bmm_kernel",
        "e07964330ae4b2a4fffea5144d5382e275bc5f4b11deab9cea714705b5e7e509",
    ): dict(
        TILE_M=64,
        TILE_N=64,
        TILE_K=32,
        GROUP_M=2,
        num_warps=4,
        num_stages=2,
        num_ctas=1,
    ),
    ("mul_scalar_kernel", _MUL_SOURCE): _MUL_CONFIG,
    ("mul_generic_nd_kernel", _MUL_SOURCE): _MUL_CONFIG,
    (
        "sum_dim_kernel",
        "5259b9f60445ef4a99e85387bc1e4c7a2264681170c695c6fc9b94378fa6348b",
    ): dict(BLOCK_M=4, BLOCK_N=1024, num_warps=4, num_stages=2, num_ctas=1),
    (
        "softmax_backward_kernel_inner",
        "d7c6eef628fba956403c1f43f4a2c94dea1142ec8fa532359de5016e3bf9488d",
    ): dict(TILE_N=128, num_warps=4, num_stages=2, num_ctas=1),
    (
        "log_softmax_backward_kernel",
        "eb890f9b6d9caec2555d9d26a1c1b2ab091f65f3edc16f7c566e10a9c224d948",
    ): dict(BLOCK_M=16, BLOCK_N=512, num_warps=8, num_stages=2, num_ctas=1),
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
