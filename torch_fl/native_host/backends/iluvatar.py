# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch ABI of the published FlagTree CoreX backend."""

import os
from pathlib import Path

from ..backend import BackendConfig, BackendInterface, BuildOptions, LauncherOptions


class Backend(BackendInterface):
    config = BackendConfig(
        vendor="iluvatar",
        compiler_backend="corex",
        driver_directory="iluvatar",
        driver_files=("driver.py", "driver.c", "compiler.py"),
        driver_symbols=("CorexDriver", "CorexUtils", "CorexLauncher"),
        utility_module="cuda_utils",
        capabilities=253,
        scratch_alignment=256,
        shared_headers=("launch_params.h",),
    )

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        sdk = Path(os.environ.get("COREX_PATH") or "/usr/local/corex")
        toolkit: LauncherOptions = {
            "include_dirs": [str(sdk / "include")],
            "library_dirs": [str(sdk / "lib64"), str(sdk / "lib")],
            "libraries": ["cuda"],
        }
        return toolkit, {**toolkit, "sources": [str(driver / "driver.c")]}

    def bind_driver(self, module, utilities, create_launcher, current_stream):
        from triton.runtime._distributed import DistributedRtContext

        def launcher(source, metadata):
            if getattr(metadata, "debug_enabled", False) or getattr(
                metadata, "debug_launch_hidden_arg", False
            ):
                raise RuntimeError(
                    "native CoreX hosts do not support debugger arguments"
                )
            if DistributedRtContext().is_lite_mode:
                raise RuntimeError(
                    "native CoreX hosts do not support distributed arguments"
                )
            return create_launcher(source, metadata)

        super().bind_driver(module, utilities, launcher, current_stream)
