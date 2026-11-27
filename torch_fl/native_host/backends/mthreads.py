# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch ABI of the published MUSA FlagTree backend."""

import os
from pathlib import Path
from types import SimpleNamespace

from ..backend import (
    BackendConfig,
    BackendInterface,
    BuildOptions,
    LauncherOptions,
    align_published_embedding_device,
    align_published_mul_device,
)


def _leaves(signature):
    for kind in signature:
        if isinstance(kind, tuple):
            yield from _leaves(kind)
        else:
            yield kind


def _descriptors(signature, entries, driver):
    if not entries:
        return None
    types = [
        kind
        for leaf in _leaves(signature.values())
        if (kind := driver._parse_tensordesc_type(leaf)) is not None
    ]
    if len(entries) > len(types):
        raise ValueError("MUSA tensor descriptor metadata arity mismatch")
    descriptors = [None] * len(types)
    for index, entry in enumerate(entries):
        if entry is not None:
            dtype, rank = types[index]
            if len(entry["block_size"]) != rank:
                raise ValueError("MUSA tensor descriptor block rank mismatch")
            descriptors[index] = {
                **entry,
                "elem_type": driver._musa_tme_dtype_enum(dtype),
                "nan_fill": driver._MUSA_TME_NAN_FILL.get(dtype),
            }
            if "elem_size" not in entry:
                # The compiler may leave element width out of TME metadata.
                # The descriptor signature still carries the element dtype.
                from triton.language import dtype as triton_dtype

                descriptors[index]["elem_size"] = (
                    triton_dtype(dtype).primitive_bitwidth // 8
                )
    return tuple(descriptors)


def _prepare(source, metadata, driver):
    if getattr(metadata, "debug_launch_hidden_arg", False):
        raise RuntimeError("native MUSA hosts do not support debugger arguments")

    def position(key):
        if isinstance(key, str):
            return source.fn.arg_names.index(key)
        if isinstance(key, tuple) and len(key) == 1:
            return key[0]
        return key

    signature = dict(
        sorted((position(key), value) for key, value in source.signature.items())
    )
    values = metadata._asdict() if hasattr(metadata, "_asdict") else vars(metadata)
    values = dict(values)
    values.update(
        warp_size=getattr(getattr(metadata, "target", None), "warp_size", None) or 32,
        global_scratch_size=getattr(metadata, "global_scratch_size", None) or 0,
        global_scratch_align=getattr(metadata, "global_scratch_align", None) or 1,
        profile_scratch_size=0,
        profile_scratch_align=1,
        tensordesc_meta=_descriptors(signature, values.get("tensordesc_meta"), driver),
    )
    return SimpleNamespace(fn=source.fn, signature=signature), SimpleNamespace(**values)


class Backend(BackendInterface):
    config = BackendConfig(
        vendor="mthreads",
        compiler_backend="musa",
        driver_directory="mthreads",
        driver_files=("driver.py", "driver.c", "compiler.py", "_musa_arch.py"),
        driver_symbols=("MusaDriver", "MusaUtils", "MusaLauncher"),
        utility_module="musa_utils",
        capabilities=167,
        scratch_alignment=256,
        shared_headers=("launch_params.h",),
    )

    def initialize_runtime(self) -> None:
        align_published_embedding_device("musa")
        align_published_mul_device("musa")

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        sdk = Path(os.environ.get("MUSA_HOME") or "/usr/local/musa")
        toolkit: LauncherOptions = {
            "include_dirs": [str(sdk / "include")],
            "library_dirs": [str(sdk / name) for name in ("lib", "lib64", "lib/stubs")],
            "libraries": ["musa"],
        }
        return toolkit, {**toolkit, "sources": [str(driver / "driver.c")]}

    def bind_driver(self, module, utilities, create_launcher, current_stream):
        def launcher(source, metadata):
            return create_launcher(*_prepare(source, metadata, module))

        super().bind_driver(module, utilities, launcher, current_stream)
