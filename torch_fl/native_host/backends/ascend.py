# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch ABI of the published FlagTree Ascend backend, without torch_npu."""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace


def _signature(value):
    if isinstance(value, tuple):
        return tuple(_signature(child) for child in value)
    # The released Ascend launcher passes these scalar parameters as float,
    # rather than the 16-bit payload used by the CUDA-shaped launchers.
    return "fp32" if value in {"fp16", "bf16"} else value


def _leaves(value):
    if isinstance(value, tuple):
        for child in value:
            yield from _leaves(child)
    elif value != "constexpr":
        yield value


def prepare(source, metadata, utils, runtime):
    """Describe the compiled kernel ABI; retain no runtime tensor or grid."""
    for name in (
        "TRITON_COMPILE_ONLY",
        "TRITON_DEVICE_PRINT",
        "TRITON_REGISTER_TENSOR_MSPROF",
        "TRITON_ENABLE_TASKQUEUE",
    ):
        if os.environ.get(name, "false").lower() in {"1", "true"}:
            raise RuntimeError(f"native Ascend hosts do not support {name}")
    if getattr(metadata, "debug_enabled", False) or getattr(
        metadata, "debug_launch_hidden_arg", False
    ):
        raise RuntimeError("native Ascend hosts do not support debugger arguments")
    signature = {key: _signature(value) for key, value in source.signature.items()}
    values = metadata._asdict() if hasattr(metadata, "_asdict") else vars(metadata)
    values = dict(values)
    arch = utils.get_ascend_arch_from_env()
    auto_blocks = getattr(metadata, "enable_auto_blockify", None)
    if auto_blocks is None:
        auto_blocks = utils._is_auto_map_parallel_blocks_enabled()
    blocks = 1
    if auto_blocks:
        blocks = (
            runtime.get_aivector_core_num()
            if metadata.mix_mode == "aiv"
            else runtime.get_aicore_num()
        )
    values.update(
        _flagos_ascend_types=tuple(
            leaf for value in signature.values() for leaf in _leaves(value)
        ),
        _flagos_ascend_ffts=utils.is_ffts_supported(arch)
        and not utils.force_disable_ffts(),
        _flagos_ascend_auto_blocks=bool(auto_blocks),
        _flagos_ascend_blocks=blocks,
    )
    return SimpleNamespace(fn=source.fn, signature=signature), SimpleNamespace(**values)


class Backend:
    vendor = "ascend"
    compiler_backend = "npu"
    compiler_distribution = "flagtree"
    driver_directory = "ascend"
    driver_files = ("driver.py", "npu_utils.cpp", "compiler.py", "utils.py")
    utility_module = "npu_utils"
    capabilities = 132
    scratch_alignment = 512
    sources = ("backends/ascend.cc", "ascend_abi.h")

    def prepare_runtime(self):
        from torch_fl import _patch_flaggems_codegen_config

        _patch_flaggems_codegen_config()

    def initialize_runtime(self):
        from torch_fl import flagos

        flagos.init()

    def extension_options(self, driver, _source_dir):
        cann = Path(
            os.environ.get("ASCEND_HOME")
            or os.environ.get("ASCEND_HOME_PATH")
            or "/usr/local/Ascend/ascend-toolkit/latest"
        )
        includes = [
            str(cann / name)
            for name in (
                "include",
                "include/experiment",
                "include/experiment/msprof",
            )
        ]
        if not (cann / "include/experiment/runtime/runtime/rt.h").is_file():
            includes.extend(
                str(cann / name) for name in ("pkg_inc", "pkg_inc/profiling")
            )
        toolkit = {
            "include_dirs": includes,
            "library_dirs": [str(cann / "lib64")],
            "libraries": ["runtime", "ascendcl"],
        }
        return toolkit, {
            **toolkit,
            "sources": [str(driver / "npu_utils.cpp")],
            "language": "c++",
            "extra_compile_args": ["-std=c++17"],
        }

    def bind_driver(self, driver, utilities, kernel_launcher, stream_for_device):
        from torch_fl.native_host.backends.driver import (
            active_drivers,
            bind_instances,
            reject_compilation,
        )

        utils = importlib.import_module("triton.backends.ascend.utils")
        # Bypass the original singleton constructor, which compiles C++ at
        # runtime. Keep its published property/loader methods unchanged.
        original = getattr(driver, "_flagos_npu_utils_class", driver.NPUUtils)
        driver._flagos_npu_utils_class = original
        runtime = object.__new__(original)
        runtime.npu_utils_mod = utilities
        # compiler.py imports this class before installation. Preserve its
        # identity so those aliases also use the prebuilt utility singleton.
        original.instance = runtime
        original.__new__ = staticmethod(lambda cls: cls.instance)
        original.__init__ = lambda self: None
        driver.NPUUtils = original

        def create(source, metadata):
            source, metadata = prepare(source, metadata, utils, runtime)
            return kernel_launcher(source, metadata)

        def current_stream(instance, device=None):
            if device is None:
                import torch

                device = torch.flagos.current_device()
            return stream_for_device(device)

        def active_device(instance):
            import torch

            return torch.device("flagos", instance.get_current_device())

        driver.NPULauncher = create
        driver.make_npu_launcher_stub = reject_compilation
        driver._build_npu_ext = reject_compilation
        driver.NPUDriver.get_current_stream = current_stream
        driver.NPUDriver.get_active_torch_device = active_device
        bind_instances(active_drivers(), runtime, create)
