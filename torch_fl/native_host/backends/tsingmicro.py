# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch ABI of the published TsingMicro FlagTree backend."""

import os
from pathlib import Path
from types import SimpleNamespace

from ..backend import (
    BackendConfig,
    BackendInterface,
    BuildOptions,
    LauncherOptions,
    align_published_embedding_device,
)
from .driver import bind_driver


_EAGER_CONFIGS = {
    (
        "zeros_kernel",
        "37db4658618b65b598fc9ae43a2366c61fb83ace411a0c6cb3a25c67666f6caf",
    ): dict(BLOCK_SIZE=1024, num_warps=1, num_stages=1, num_ctas=1),
    (
        "fill_scalar_kernel",
        "b15dd66bff906ac8157ae0fb849faa1e6447e369593567069f3796945715cee6",
    ): dict(BLOCK_SIZE=1024, num_warps=1, num_stages=1, num_ctas=1),
    (
        "argmax_kernel",
        "4a9697a4bed8d70d4a754d8ea705adf6e4bdc8be6654dd7205aa47e06ae835a2",
    ): dict(BLOCK_M=8, BLOCK_N=16384, num_warps=1, num_stages=1, num_ctas=1),
    (
        "mm_kernel",
        "a2a0b05d840b360bd9c6dcc6bd40d7103223a5ae8796959df8c5fff2b44a0dba",
    ): dict(
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=64,
        SPLIT_K=1,
        num_warps=1,
        num_stages=1,
        num_ctas=1,
    ),
    (
        "sum_dim_kernel",
        "5259b9f60445ef4a99e85387bc1e4c7a2264681170c695c6fc9b94378fa6348b",
    ): dict(BLOCK_M=4, BLOCK_N=1024, num_warps=1, num_stages=1, num_ctas=1),
}


_SCALAR_BYTES = {
    "i1": 1,
    "u1": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
    "i64": 8,
    "u64": 8,
    "fp32": 4,
    "f32": 4,
    "fp64": 8,
}


def _launcher_signature(signature):
    widths = []

    def normalize(kind):
        if isinstance(kind, tuple):
            return tuple(normalize(child) for child in kind)
        if kind == "constexpr":
            return kind
        if kind.startswith("*"):
            widths.append(0)
            return kind
        if kind in ("fp16", "bf16"):
            kind = "fp32"
        if kind not in _SCALAR_BYTES:
            raise ValueError(f"unsupported TsingMicro Triton argument {kind!r}")
        widths.append(_SCALAR_BYTES[kind])
        return kind

    return {key: normalize(kind) for key, kind in signature.items()}, widths


class Backend(BackendInterface):
    eager_kernel_configs = _EAGER_CONFIGS

    config = BackendConfig(
        vendor="tsingmicro",
        compiler_backend="txda",
        driver_directory="tsingmicro",
        driver_files=("driver.py", "driver.cpp", "compiler.py"),
        driver_symbols=("TXDADriver", "TXDAUtils", "TXDALauncher"),
        utility_module="tx81_utils",
        capabilities=128,
        scratch_alignment=256,
        shared_headers=("launch_params.h",),
    )

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        sdk = Path(os.environ.get("KUIPER_ROOT") or "/usr/local/kuiper")
        toolkit: LauncherOptions = {
            "include_dirs": [str(sdk / "include")],
            "library_dirs": [str(sdk / "lib")],
            "libraries": ["hpgr"],
        }
        return toolkit, {
            **toolkit,
            "sources": [str(driver / "driver.cpp")],
            "language": "c++",
            "extra_compile_args": ["-std=c++17"],
        }

    def prepare_runtime(self) -> None:
        from torch_fl import flagos
        from torch_fl.compat.flaggems.device import install_tsingmicro_runtime_alias

        # FlagTree imports torch_txda while discovering its native driver.
        install_tsingmicro_runtime_alias(flagos)

    def initialize_runtime(self) -> None:
        align_published_embedding_device("txda")

    def bind_driver(self, module, utilities, create_launcher, current_stream):
        if any(
            os.environ.get(key, "").lower() in {"1", "true", "yes"}
            for key in ("USE_SIM_MODE", "TXDA_LAUNCH_VIA_GGL")
        ):
            raise RuntimeError(
                "TsingMicro Native Host requires loaded function handles"
            )

        def create(source, metadata):
            signature, widths = _launcher_signature(source.signature)
            fields = metadata._asdict()
            fields["argument_widths"] = widths
            return create_launcher(
                SimpleNamespace(fn=source.fn, signature=signature),
                SimpleNamespace(**fields),
            )

        bind_driver(
            module,
            self.config.driver_symbols,
            utilities,
            create,
            current_stream,
            compiler_hooks=("compile_native",),
        )
