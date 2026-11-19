# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""HCU launch mechanics for the published DTK FlagTree backend."""

import importlib
import os
from pathlib import Path


def _published_codec(source):
    """Reuse the pinned driver's descriptor layout and encoder unchanged."""
    start = source.index("typedef struct {")
    end = source.index("} TDMDescriptor;", start) + len("} TDMDescriptor;")
    function = source.index("static bool encodeTDMDescriptor(")
    body = source.index("{", function)
    depth = 1
    stop = body + 1
    while depth:
        depth += (source[stop] == "{") - (source[stop] == "}")
        stop += 1
    return source[start:end] + "\n" + source[function:stop] + "\n"


def _substitute(source, marker, value):
    if source.count(marker) != 1:
        raise RuntimeError("unrecognized published HCU utility source: " + marker)
    return source.replace(marker, value, 1)


class Backend:
    vendor = "hygon"
    compiler_backend = "hip"
    compiler_distribution = "flagtree"
    utility_modules = ("hip_utils",)
    # Software/TDM descriptors, profile scratch, cooperative/cluster launch,
    # stream-capture query. The published HCU ABI always passes null global scratch.
    capabilities = 187
    scratch_alignment = 256
    sources = ("csrc/backends/hygon.cc",)

    def driver_sources(self, triton_root):
        return {
            "backends/hcu/" + name: triton_root / "backends/hcu" / name
            for name in ("driver.py", "driver.c", "compiler.py", "compiler_hcu.py")
        }

    def extensions(self, namespace, root, torch, record, triton_root):
        from setuptools import Extension

        sdk = Path(os.environ.get("ROCM_PATH") or "/opt/dtk")
        runtime = sdk / "lib/libgalaxyhip.so"
        if not runtime.is_file():
            raise RuntimeError(
                "HCU native hosts require DTK libgalaxyhip.so: " + str(runtime)
            )
        published = triton_root / "backends/hcu"
        source = (published / "driver.c").read_text()
        generated = root / "_generated/hcu"
        generated.mkdir(parents=True, exist_ok=True)
        (generated / "hcu_descriptor.h").write_text(_published_codec(source))
        source = _substitute(source, "/*py_libhip_search_path*/", str(runtime))
        source = _substitute(
            source,
            "/*py_hip_get_device_properties_sym*/ NULL",
            '"hipGetDeviceProperties_v2"',
        )
        utility = generated / "driver.c"
        utility.write_text(source)
        include = str(Path(torch.__file__).parent / "include")
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
                    str(generated / "hcu_descriptor.h"),
                ],
                include_dirs=[include, str(published / "include"), str(generated)],
                library_dirs=[str(Path(torch.__file__).parent / "lib")],
                libraries=["torch_python", "torch_cpu", "c10", "dl"],
                define_macros=[
                    ("__HIP_PLATFORM_AMD__", "1"),
                    ("FLAGOS_HCU_RUNTIME_LIBRARY", '"' + str(runtime) + '"'),
                    (
                        "FLAGOS_NATIVE_HOST_BACKEND_SHA256",
                        '"' + record["native_source_sha256"] + '"',
                    ),
                ],
                language="c++",
                extra_compile_args=[
                    "-O3",
                    "-std=c++17",
                    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                ],
            ),
            Extension(
                namespace + ".hip_utils",
                [str(utility)],
                include_dirs=[str(published / "include")],
                libraries=["dl", "m"],
                extra_compile_args=["-O3"],
            ),
        ]

    def install(self, package, launcher, kernel_launcher):
        driver = importlib.import_module("triton.backends.hcu.driver")
        utilities = importlib.import_module(package + ".hip_utils")
        driver.HIPUtils = lambda: utilities
        driver.PyTDMDescriptor = utilities.PyTDMDescriptor
        driver.HIPLauncher = kernel_launcher

        def reject(*args, **kwargs):
            raise RuntimeError("runtime C/C++ launcher compilation is forbidden")

        driver.compile_module_from_src = reject
        from triton.runtime import driver as runtime_driver

        if not getattr(driver.HIPDriver.__init__, "_flagos_stream_provider", False):
            original_init = driver.HIPDriver.__init__

            def initialize(instance, *args, **kwargs):
                original_init(instance, *args, **kwargs)
                instance.get_current_stream = launcher.current_stream

            initialize._flagos_stream_provider = True
            driver.HIPDriver.__init__ = initialize
        for active in (
            getattr(runtime_driver, "_active", None),
            getattr(runtime_driver, "_default", None),
        ):
            if active is not None:
                active.utils = utilities
                active.launcher_cls = kernel_launcher
                active.get_current_stream = launcher.current_stream
