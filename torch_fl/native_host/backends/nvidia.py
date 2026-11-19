# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA adapter for the pinned published FlagTree driver."""

import os
from pathlib import Path


class Backend:
    vendor = "nvidia"
    compiler_backend = "cuda"
    compiler_distribution = "flagtree"
    driver_directory = "nvidia"
    driver_files = ("driver.py", "driver.c")
    driver_symbols = ("CudaDriver", "CudaUtils", "CudaLauncher")
    utility_module = "cuda_utils"
    utility_exports = ("PyCUtensorMap",)
    capabilities = 255
    scratch_alignment = 256

    def extension_options(self, driver, _source_dir):
        cuda = Path(
            os.environ.get("CUDA_HOME")
            or os.environ.get("CUDA_PATH")
            or "/usr/local/cuda"
        )
        toolkit = {
            "include_dirs": [str(cuda / "include")],
            "library_dirs": [str(cuda / "lib64/stubs"), str(cuda / "lib/stubs")],
            "libraries": ["cuda"],
        }
        return toolkit, {**toolkit, "sources": [str(driver / "driver.c")]}
