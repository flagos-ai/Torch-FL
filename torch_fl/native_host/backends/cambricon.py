# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch ABI of the published Cambricon Triton backend."""

import os
from pathlib import Path
from sys import maxsize

from ..backend import BackendConfig, BackendInterface, BuildOptions, LauncherOptions


class Backend(BackendInterface):
    config = BackendConfig(
        vendor="cambricon",
        compiler_backend="mlu",
        compiler_distribution="triton",
        driver_directory="mlu",
        driver_files=("driver.py", "driver.c", "compiler.py"),
        driver_symbols=("BangDriver", "BangUtils", "BangLauncher"),
        utility_module="bang_utils",
        capabilities=129,
        scratch_alignment=256,
        shared_headers=("launch_params.h",),
    )

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        sdk = Path(os.environ.get("NEUWARE_HOME") or "/usr/local/neuware")
        toolkit: LauncherOptions = {
            "include_dirs": [str(sdk / "include")],
            "library_dirs": [str(sdk / "lib64")],
            "libraries": ["cndrv", "cnrt"],
        }
        return toolkit, {**toolkit, "sources": [str(driver / "driver.c")]}

    def bind_driver(self, module, utilities, create_launcher, current_stream):
        def max_grid_size(device):
            properties = utilities.get_device_properties(device)
            return tuple(
                properties.get(f"max_block_task_dim_{axis}", maxsize) for axis in "xyz"
            )

        utilities.get_max_grid_size = max_grid_size
        super().bind_driver(module, utilities, create_launcher, current_stream)
