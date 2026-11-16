# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MACA launch adapter for the published MetaX FlagTree backend."""

import importlib
import os
from pathlib import Path


class Backend:
    vendor = "metax"
    compiler_backend = "maca"
    compiler_distribution = "flagtree"
    utility_modules = ("maca_utils",)
    capabilities = 141
    scratch_alignment = 256
    sources = ("csrc/backends/metax.cc",)

    def driver_sources(self, triton_root):
        return {
            name: triton_root / name
            for name in (
                "backends/metax/driver.py",
                "backends/metax/driver.c",
                "backends/metax/compiler.py",
            )
        }

    def extensions(self, namespace, root, torch, record, triton_root):
        from setuptools import Extension

        maca = Path(
            os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME") or "/opt/maca"
        )
        library_dirs = [str(maca / "lib"), str(maca / "lib64")]
        return [
            Extension(
                namespace + "._launcher",
                [
                    str(root / "csrc/launcher.cc"),
                    *(str(root / p) for p in self.sources),
                ],
                depends=[
                    str(root / "csrc/backend_api.h"),
                    str(root / "csrc/host_abi.h"),
                ],
                include_dirs=[
                    str(Path(torch.__file__).parent / "include"),
                    str(maca / "include"),
                ],
                library_dirs=[*library_dirs, str(Path(torch.__file__).parent / "lib")],
                libraries=["mcruntime", "torch_python", "torch_cpu", "c10"],
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
                namespace + ".maca_utils",
                [str(self.driver_sources(triton_root)["backends/metax/driver.c"])],
                include_dirs=[str(maca / "include")],
                library_dirs=library_dirs,
                libraries=["mcruntime"],
                define_macros=[("__MACA__", "1")],
                extra_compile_args=["-O3"],
            ),
        ]

    def install(self, package, launcher, kernel_launcher):
        driver = importlib.import_module("triton.backends.metax.driver")
        utilities = importlib.import_module(package + ".maca_utils")
        driver.MacaUtils = lambda: utilities
        driver.MacaLauncher = kernel_launcher

        def reject(*args, **kwargs):
            raise RuntimeError("runtime C/C++ launcher compilation is forbidden")

        driver.compile_module_from_src = reject
        from triton.runtime import driver as runtime_driver

        if not getattr(driver.MacaDriver.__init__, "_flagos_stream_provider", False):
            original_init = driver.MacaDriver.__init__

            def initialize(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                instance.get_current_stream = launcher.current_stream

            initialize._flagos_stream_provider = True
            driver.MacaDriver.__init__ = initialize
        for active in (
            getattr(runtime_driver, "_active", None),
            getattr(runtime_driver, "_default", None),
        ):
            if active is not None:
                active.utils = utilities
                active.launcher_cls = kernel_launcher
                active.get_current_stream = launcher.current_stream
