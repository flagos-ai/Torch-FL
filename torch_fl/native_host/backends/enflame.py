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
    driver_directory = "enflame"
    driver_entry = "backend"
    driver_files = (
        "backend.py",
        "driver.py",
        "utils/gcu.cpp",
        "compiler.py",
        "VERSION",
    )
    utility_module = "gcu_utils"
    capabilities = 177
    scratch_alignment = 256

    def extension_options(self, driver, _source_dir):
        tops = Path(os.environ.get("TOPS_HOME") or "/opt/tops")
        toolkit = {
            "include_dirs": [str(tops / "include")],
            "library_dirs": [str(tops / "lib"), str(tops / "lib64")],
            "libraries": ["topsrt"],
        }
        return toolkit, {
            **toolkit,
            "sources": [str(driver / "utils/gcu.cpp")],
            "language": "c++",
            "extra_compile_args": ["-std=c++17"],
        }

    def bind_driver(self, driver, utilities, kernel_launcher, current_stream):
        from torch_fl.native_host.backends.driver import (
            active_drivers,
            bind_instances,
            reject_compilation,
        )

        wrapper = importlib.import_module("triton.backends.enflame.driver")

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

        driver.GCUUtils = lambda: utilities
        driver.GcuLauncher = create
        driver.build_gcu_ext = reject_compilation
        driver.compile_module_from_src = reject_compilation
        driver._flagos_native_stream = current_stream
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
        wrapped = getattr(wrapper._GCUDriver, "instance", None)
        for active in (*active_drivers(), wrapped):
            bind_instances(
                (active, getattr(active, "_driver", None)),
                utilities,
                create,
                current_stream,
            )
