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
    utility_modules = ("npu_utils",)
    capabilities = 132
    scratch_alignment = 512
    sources = ("csrc/backends/ascend.cc", "csrc/ascend_abi.h")

    def driver_sources(self, triton_root):
        return {
            name: triton_root / name
            for name in (
                "backends/ascend/driver.py",
                "backends/ascend/npu_utils.cpp",
                "backends/ascend/compiler.py",
                "backends/ascend/utils.py",
            )
        }

    def extensions(self, namespace, root, torch, record, triton_root):
        from setuptools import Extension

        cann = Path(
            os.environ.get("ASCEND_HOME")
            or os.environ.get("ASCEND_HOME_PATH")
            or "/usr/local/Ascend/ascend-toolkit/latest"
        )
        includes = [str(cann / name) for name in ("include", "include/experiment")]
        if not (cann / "include/experiment/runtime/runtime/rt.h").is_file():
            includes.append(str(cann / "pkg_inc"))
        libraries = [str(cann / "lib64")]
        return [
            Extension(
                namespace + "._launcher",
                [str(root / "csrc/launcher.cc"), str(root / self.sources[0])],
                depends=[
                    str(root / "csrc/backend_api.h"),
                    str(root / "csrc/host_abi.h"),
                    str(root / "csrc/ascend_abi.h"),
                ],
                include_dirs=[str(Path(torch.__file__).parent / "include"), *includes],
                library_dirs=[*libraries, str(Path(torch.__file__).parent / "lib")],
                libraries=["runtime", "ascendcl", "torch_python", "torch_cpu", "c10"],
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
                namespace + ".npu_utils",
                [
                    str(
                        self.driver_sources(triton_root)[
                            "backends/ascend/npu_utils.cpp"
                        ]
                    )
                ],
                include_dirs=includes,
                library_dirs=libraries,
                libraries=["runtime", "ascendcl"],
                language="c++",
                extra_compile_args=["-O3", "-std=c++17"],
            ),
        ]

    def install(self, package, launcher, kernel_launcher):
        driver = importlib.import_module("triton.backends.ascend.driver")
        utils = importlib.import_module("triton.backends.ascend.utils")
        utilities = importlib.import_module(package + ".npu_utils")
        # Bypass the original singleton constructor, which compiles C++ at
        # runtime. Keep its published property/loader methods unchanged.
        original = getattr(driver, "_flagos_npu_utils_class", driver.NPUUtils)
        driver._flagos_npu_utils_class = original
        runtime = object.__new__(original)
        runtime.npu_utils_mod = utilities
        driver.NPUUtils = lambda: runtime

        def create(source, metadata):
            source, metadata = prepare(source, metadata, utils, runtime)
            return kernel_launcher(source, metadata)

        def reject(*args, **kwargs):
            raise RuntimeError("runtime C/C++ launcher compilation is forbidden")

        def current_stream(instance, device=None):
            if device is None:
                import torch

                device = torch.flagos.current_device()
            return launcher.current_stream(device)

        def active_device(instance):
            import torch

            return torch.device("flagos", instance.get_current_device())

        driver.NPULauncher = create
        driver.make_npu_launcher_stub = reject
        driver._build_npu_ext = reject
        driver.NPUDriver.get_current_stream = current_stream
        driver.NPUDriver.get_active_torch_device = active_device
        from triton.runtime import driver as runtime_driver

        for active in (
            getattr(runtime_driver, "_active", None),
            getattr(runtime_driver, "_default", None),
        ):
            if active is not None:
                active.utils = runtime
                active.launcher_cls = create
