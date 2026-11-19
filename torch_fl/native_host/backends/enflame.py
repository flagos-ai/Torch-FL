# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""TOPS launch adapter for the published Enflame FlagTree backend."""

import ctypes
import importlib
import os
from pathlib import Path
from types import SimpleNamespace


def prepare(source, metadata, driver):
    """Resolve the published scalar-pointer ABI once per device specialization."""
    # FlagTree 3.7 adds a descriptor flag absent from the shared descriptor value.
    # Reject that ABI rather than silently packing the 3.6 layout for it.
    if driver._triton_version() != (3, 6):
        raise RuntimeError("Enflame native hosts require the published Triton 3.6 ABI")
    signature = driver._expand_signature(source.signature.values())
    runtime = [kind for kind in signature if kind != "constexpr"]
    pointers = tuple(getattr(metadata, "ptr_int_args", ()))
    for position in pointers:
        if (
            type(position) is not int
            or position < 0
            or position >= len(runtime)
            or runtime[position] not in ("i64", "u64")
        ):
            raise ValueError("invalid Enflame integer pointer argument metadata")
    fields = {
        name: getattr(metadata, name)
        for name in (
            "name",
            "num_warps",
            "shared",
            "arch",
            "num_ctas",
            "cluster_dims",
            "launch_cooperative_grid",
            "redundant_sip",
            "global_scratch_size",
            "profile_scratch_size",
            "launch_pdl",
            "tensordesc_meta",
        )
        if hasattr(metadata, name)
    }
    fields["ptr_int_args"] = pointers
    return SimpleNamespace(**fields)


class Backend:
    vendor = "enflame"
    compiler_backend = "gcu"
    compiler_distribution = "flagtree"
    utility_modules = ("gcu_utils",)
    capabilities = 177
    scratch_alignment = 256
    sources = ("csrc/backends/enflame.cc",)

    def driver_sources(self, triton_root):
        return {
            name: triton_root / name
            for name in (
                "backends/enflame/backend.py",
                "backends/enflame/driver.py",
                "backends/enflame/utils/gcu.cpp",
                "backends/enflame/compiler.py",
                "backends/enflame/VERSION",
            )
        }

    def extensions(self, namespace, root, torch, record, triton_root):
        from setuptools import Extension

        tops = Path(os.environ.get("TOPS_HOME") or "/opt/tops")
        libraries = [str(tops / "lib"), str(tops / "lib64")]
        return [
            Extension(
                namespace + "._launcher",
                [str(root / "csrc/launcher.cc"), str(root / self.sources[0])],
                depends=[
                    str(root / "csrc/backend_api.h"),
                    str(root / "csrc/host_abi.h"),
                ],
                include_dirs=[
                    str(Path(torch.__file__).parent / "include"),
                    str(tops / "include"),
                ],
                library_dirs=[*libraries, str(Path(torch.__file__).parent / "lib")],
                libraries=["topsrt", "torch_python", "torch_cpu", "c10"],
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
                namespace + ".gcu_utils",
                [str(triton_root / "backends/enflame/utils/gcu.cpp")],
                include_dirs=[str(tops / "include")],
                library_dirs=libraries,
                libraries=["topsrt"],
                language="c++",
                extra_compile_args=["-O3", "-std=c++17"],
            ),
        ]

    def install(self, package, launcher, kernel_launcher):
        driver = importlib.import_module("triton.backends.enflame.backend")
        wrapper = importlib.import_module("triton.backends.enflame.driver")
        utilities = importlib.import_module(package + ".gcu_utils")

        runtime = (
            Path(importlib.import_module("torch_fl").__file__).parent
            / "lib/libflagos.so"
        )
        enable = ctypes.CDLL(
            str(runtime), mode=ctypes.RTLD_GLOBAL
        ).FlagosEnflameEnableInterop
        enable.argtypes = [ctypes.c_uint32]
        enable.restype = ctypes.c_int
        if enable(2):
            raise RuntimeError("Enflame native-host stream interop ABI was rejected")

        def create(source, metadata):
            return kernel_launcher(source, prepare(source, metadata, driver))

        def reject(*args, **kwargs):
            raise RuntimeError("runtime C/C++ launcher compilation is forbidden")

        driver.GCUUtils = lambda: utilities
        driver.GcuLauncher = create
        driver.build_gcu_ext = reject
        driver.compile_module_from_src = reject
        driver._flagos_native_stream = launcher.current_stream
        # The existing compatibility layer wraps both constructors to restore
        # current-device/stream binding. Keep that wrapper and apply ours last.
        if not getattr(driver.GCUDriver.__init__, "_flagos_native_host", False):
            original = driver.GCUDriver.__init__

            def initialize(instance, *args, **kwargs):
                original(instance, *args, **kwargs)
                instance.get_current_stream = lambda device=0: (
                    driver._flagos_native_stream(device)
                )

            initialize._flagos_native_host = True
            driver.GCUDriver.__init__ = initialize
        from triton.runtime import driver as runtime_driver

        for active in (
            getattr(runtime_driver, "_active", None),
            getattr(runtime_driver, "_default", None),
            getattr(wrapper._GCUDriver, "instance", None),
        ):
            if active is not None:
                for instance in (active, getattr(active, "_driver", None)):
                    if instance is not None:
                        instance.utils = utilities
                        instance.launcher_cls = create
                        instance.get_current_stream = driver._flagos_native_stream
