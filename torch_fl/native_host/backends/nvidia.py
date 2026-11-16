# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA adapter for the pinned published FlagTree driver."""

import importlib
import os
from pathlib import Path


class Backend:
    vendor = "nvidia"
    compiler_backend = "cuda"
    compiler_distribution = "flagtree"
    utility_modules = ("cuda_utils",)
    capabilities = 255
    scratch_alignment = 256
    sources = ("csrc/backends/nvidia.cc",)

    def driver_sources(self, triton_root):
        return {
            name: triton_root / name
            for name in ("backends/nvidia/driver.py", "backends/nvidia/driver.c")
        }

    def extensions(self, namespace, root, torch, record, triton_root):
        from setuptools import Extension

        cuda = Path(
            os.environ.get("CUDA_HOME")
            or os.environ.get("CUDA_PATH")
            or "/usr/local/cuda"
        )
        libraries = [str(cuda / "lib64/stubs"), str(cuda / "lib/stubs")]
        return [
            Extension(
                namespace + "._launcher",
                [
                    str(root / "csrc/launcher.cc"),
                    *(str(root / path) for path in self.sources),
                ],
                depends=[
                    str(root / "csrc/backend_api.h"),
                    str(root / "csrc/host_abi.h"),
                ],
                include_dirs=[
                    str(Path(torch.__file__).parent / "include"),
                    str(cuda / "include"),
                ],
                library_dirs=[*libraries, str(Path(torch.__file__).parent / "lib")],
                libraries=["cuda", "torch_python", "torch_cpu", "c10"],
                define_macros=[
                    (
                        "FLAGOS_NATIVE_HOST_BACKEND_SHA256",
                        '"' + record["native_source_sha256"] + '"',
                    )
                ],
                language="c++",
                extra_compile_args=[
                    "-O3",
                    "-std=c++17",
                    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                ],
            ),
            Extension(
                namespace + ".cuda_utils",
                [str(self.driver_sources(triton_root)["backends/nvidia/driver.c"])],
                include_dirs=[str(cuda / "include")],
                library_dirs=libraries,
                libraries=["cuda"],
                extra_compile_args=["-O3"],
            ),
        ]

    def install(self, package, launcher, kernel_launcher):
        driver = importlib.import_module("triton.backends.nvidia.driver")
        utilities = importlib.import_module(package + ".cuda_utils")
        driver.CudaUtils = lambda: utilities
        driver.PyCUtensorMap = utilities.PyCUtensorMap
        driver.CudaLauncher = kernel_launcher

        def reject(*args, **kwargs):
            raise RuntimeError("runtime C/C++ launcher compilation is forbidden")

        driver.compile_module_from_src = reject
        from triton.runtime import driver as runtime_driver

        if not getattr(driver.CudaDriver.__init__, "_flagos_stream_provider", False):
            original_init = driver.CudaDriver.__init__

            def initialize(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                instance.get_current_stream = launcher.current_stream

            initialize._flagos_stream_provider = True
            driver.CudaDriver.__init__ = initialize
        for active in (
            getattr(runtime_driver, "_active", None),
            getattr(runtime_driver, "_default", None),
        ):
            if active is not None:
                active.utils = utilities
                active.launcher_cls = kernel_launcher
                active.get_current_stream = launcher.current_stream
