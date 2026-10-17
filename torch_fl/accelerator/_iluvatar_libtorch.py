# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select a bundled CoreX runtime alongside an official CPU torch frontend.

CoreX 4.5.0 uses the old C++ ABI. Its device libraries cannot bind to the
official 2.10 CPU core, so select the matched binary set through the existing
private facade mechanism, without modifying the installed torch package.
"""

from torch_fl.accelerator._vendor_libtorch import ensure_vendor_libtorch_links

CORE_SO = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
)
DEVICE_SO = ("libc10_cuda.so", "libtorch_cuda.so", "libshm.so")
LOAD_ORDER = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libc10_cuda.so",
    "libtorch_cuda.so",
    "libshm.so",
    "libtorch_python.so",
)


def ensure_iluvatar_libtorch():
    return ensure_vendor_libtorch_links(
        "lib_iluvatar",
        CORE_SO,
        extra_so=DEVICE_SO,
        env_override="FLAGOS_VENDOR_TORCH_LIB",
        vendor_markers=("corex",),
        probe_so="libtorch_cuda.so",
        vendor="Iluvatar",
        load_order=LOAD_ORDER,
    )


def restore_iluvatar_cuda_version():
    """Read the loaded runtime's version; the CPU frontend reports None."""
    import torch

    if torch.version.cuda is None:
        getter = getattr(torch._C, "_cuda_getCompiledVersion", None)
        if not callable(getter):
            raise RuntimeError("The CoreX CUDA runtime was not selected before torch")
        compiled = int(getter())
        if compiled < 1000 or compiled % 10:
            raise RuntimeError(f"Invalid CoreX CUDA ABI version: {compiled}")
        torch.version.cuda = f"{compiled // 1000}.{(compiled % 1000) // 10}"
    return torch.version.cuda
