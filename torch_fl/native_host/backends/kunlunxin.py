# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch adapter for the published FlagTree XPU3 backend."""

import ctypes
import importlib
from pathlib import Path
from types import SimpleNamespace

from ..backend import BackendConfig, BackendInterface, BuildOptions
from .driver import active_drivers, bind_driver


# Parameter widths and type tags follow the published XPU launch table.
_WIDTHS = {
    "void *": 8,
    "float": 4,
    "double": 8,
    "int8_t": 1,
    "int16_t": 2,
    "int32_t": 4,
    "int64_t": 8,
    "uint8_t": 1,
    "uint16_t": 2,
    "uint32_t": 4,
    "uint64_t": 8,
}
_TYPE_TAGS = {
    "*fp32": 1,
    "*f64": 2,
    "*int32": 3,
    "*int64": 4,
    "bool": 5,
    "*fp16": 6,
    "*bf16": 7,
    "int32": 8,
    "int64": 9,
    "int8": 10,
    "int16": 11,
}


def _flatten(kind: str, ranks: list[int]) -> tuple[str, ...]:
    if not kind.startswith("tensordesc"):
        return (kind,)
    _, start, extents = kind.partition("[")
    dimensions, end, _ = extents.partition("]")
    if not start or not end or not dimensions:
        raise ValueError(f"invalid XPU tensor descriptor signature: {kind}")
    rank = len(dimensions.split(","))
    if rank > 5:
        raise ValueError("XPU tensor descriptors support up to five dimensions")
    ranks.append(rank)
    return (
        "*" + kind,
        *("i64",) * (2 * rank),
        "i1",
        *("i32",) * rank,
        *("i64",) * rank,
    )


def prepare(source, metadata, driver):
    if metadata.xpu_arch != 3:
        raise RuntimeError("native XPU hosts require the published XPU3 launch ABI")
    if metadata.tensor_args:
        raise RuntimeError("native XPU hosts do not support launch-time quantization")

    ranks: list[int] = []
    parameters = [
        (_WIDTHS[driver.ty_to_cpp(kind)], _TYPE_TAGS.get(kind, 0), kind in ("i1", "u1"))
        for parameter in source.signature.values()
        if parameter != "constexpr"
        for kind in _flatten(parameter, ranks)
    ]
    # FlagTree's XPU host parses the two low-precision scalar types as float.
    signature = {
        name: "fp32" if kind in ("fp16", "bf16") else kind
        for name, kind in source.signature.items()
    }
    values = dict(
        metadata._asdict() if hasattr(metadata, "_asdict") else vars(metadata)
    )
    values.update(
        _flagos_xpu_parameters=parameters,
        _flagos_xpu_geometry=driver.get_xpu_spec(metadata.xpu_arch, metadata.is_sdnn),
        _flagos_xpu_ewtable=Path(metadata.ewtable).read_bytes()
        if metadata.ewtable
        else b"",
        tensordesc_meta=ranks or None,
    )
    return SimpleNamespace(fn=source.fn, signature=signature), SimpleNamespace(**values)


class Backend(BackendInterface):
    config = BackendConfig(
        vendor="kunlunxin",
        compiler_backend="xpu",
        driver_directory="xpu",
        driver_files=(
            "driver.py",
            "driver.c",
            "compiler.py",
            "xpu3/include/cuda_runtime_api.h",
            "xpu3/include/xpu/runtime.h",
            "xpu3/lib/liblaunch.a",
        ),
        driver_symbols=("XPUDriver", "XPUUtils", "XPULauncher"),
        utility_module="xpu_utils",
        capabilities=129,
        scratch_alignment=128,
    )

    def extension_options(self, driver: Path, source_dir: Path) -> BuildOptions:
        kit = driver / "xpu3"
        includes = [str(kit / "include")]
        libraries = [str(kit / "so")]
        # The published `driver.c` uses C++ even though its name ends in .c.
        utility = source_dir / "xpu_utils.cc"
        utility.parent.mkdir(parents=True, exist_ok=True)
        utility.write_bytes((driver / "driver.c").read_bytes())
        return {
            "include_dirs": includes,
            "library_dirs": libraries,
            "libraries": ["launch_shared", "xpurt", "cudart"],
            "extra_objects": [str(kit / "lib/liblaunch.a")],
        }, {
            "sources": [str(utility)],
            "include_dirs": includes,
            "library_dirs": libraries,
            "libraries": ["xpurt"],
            "language": "c++",
        }

    def prepare_runtime(self) -> None:
        driver = importlib.import_module("triton.backends.xpu.driver")
        driver._preload_libstdcxx()
        # Resolve the installed dependency before loading the extension. CI
        # installs the AOT wheel in a separate import root from FlagTree.
        kit = Path(driver.__file__).resolve().parent / "xpu3"
        ctypes.CDLL(str(kit / "so/liblaunch_shared.so"))

    def bind_driver(self, module, utilities, create_launcher, current_stream) -> None:
        def create(source, metadata):
            source, metadata = prepare(source, metadata, module)
            return create_launcher(source, metadata)

        module._xpu_utils_singleton.cache_clear()
        bind_driver(
            module, self.config.driver_symbols, utilities, create, current_stream
        )
        # FlagTree's CompiledKernel selects these aliases for xpubin binaries.
        # New XPUDriver instances capture the patched factories directly.
        for driver in active_drivers():
            if isinstance(driver, module.XPUDriver):
                driver.xpu_utils = utilities
                driver.launcher_cls_xpu = create
