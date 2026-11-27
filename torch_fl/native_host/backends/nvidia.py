# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA adapter for the pinned published FlagTree driver."""

import os
from pathlib import Path

from ..backend import BackendConfig, BackendInterface, BuildOptions, LauncherOptions


class Backend(BackendInterface):
    config = BackendConfig(
        vendor="nvidia",
        compiler_backend="cuda",
        driver_directory="nvidia",
        driver_files=("driver.py", "driver.c"),
        driver_symbols=("CudaDriver", "CudaUtils", "CudaLauncher"),
        utility_module="cuda_utils",
        utility_exports=("PyCUtensorMap",),
        capabilities=255,
        scratch_alignment=256,
        shared_headers=("launch_params.h",),
    )

    def initialize_runtime(self) -> None:
        import torch

        activate = getattr(torch.ops.flagos, "_activate_native_nvidia", None)
        if activate is not None:
            activate()

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        cuda = Path(
            os.environ.get("CUDA_HOME")
            or os.environ.get("CUDA_PATH")
            or "/usr/local/cuda"
        )
        toolkit: LauncherOptions = {
            "include_dirs": [str(cuda / "include")],
            "library_dirs": [str(cuda / "lib64/stubs"), str(cuda / "lib/stubs")],
            "libraries": ["cuda"],
        }
        return toolkit, {**toolkit, "sources": [str(driver / "driver.c")]}
