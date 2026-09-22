# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Symlink the PPU libtorch .so into the active (official) torch wheel's lib dir.

PPU builds against ``PPU_SDK/CUDA_SDK`` with its own ``FLAGOS_ACCELERATOR=ppu`` value;
the CUDA boxing kernels apply unchanged.  What differs from a real NVIDIA box is the
libtorch: it is a local ``USE_CUDA=1`` source build, not the upstream wheel, and
it resolves 2092 of ``libtorch_fl.so``'s undefined symbols (``libtorch_cuda.so``
resolves 0, ``libc10_cuda.so`` 10) -- so the core libs must be swapped in, and
``libtorch_cuda.so`` must merely be present for the CUDA dispatch keys.

That local build also links the system MKL from ``/usr/local/lib``
(``libmkl_core``/``libmkl_gnu_thread``/``libmkl_intel_lp64``), which the bundling
script copies into ``lib_ppu/`` alongside the core libs.

See ``torch_fl.accelerator._vendor_libtorch`` for the symlink mechanism and why a
ctypes preload cannot replace core libs.  No env gate: the relink is skipped when
``torch_fl/lib_ppu/`` was not bundled or torch already IS the PPU build.  The PPU
SDK runtime stays on the target under ``/usr/local/PPU_SDK``.
"""

from torch_fl.accelerator._vendor_libtorch import (
    bundled_lib_dir,
    discover_vendor_torch_lib,
    ensure_vendor_libtorch_links,
)
from torch_fl.accelerator._vendor_libtorch import (
    restore_original_libtorch as _restore,
)

_BUNDLE_DIR = "lib_ppu"

# Core C++ .so that must come from the PPU build as a self-consistent set.
_CORE_SO = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
)
# CUDA-side .so the stock +cpu wheel does not ship at all.
# libshm.so is a DT_NEEDED of PPU's libtorch_python.so (torch.multiprocessing's
# shared-memory manager) and is a *different* build in the PPU wheel, so it has
# to come from the same set.
_CUDA_SO = (
    "libc10_cuda.so",
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
    "libshm.so",
)

# Dependency order for the RTLD_GLOBAL preload: core (CPU) first, then the CUDA
# side, then libshm and libtorch_python. Same reasoning as MetaX, including why
# libshm.so has to be listed explicitly ahead of libtorch_python.so.
_LOAD_ORDER = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libc10_cuda.so",
    "libtorch_cuda_linalg.so",
    "libtorch_cuda.so",
    "libshm.so",
    "libtorch_python.so",
)

_MARKERS = ("ppu",)


def _bundled_ppu_lib():
    """PPU libtorch bundled inside this wheel (self-contained PPU build)."""
    return bundled_lib_dir(_BUNDLE_DIR, "libtorch_cuda.so")


def _discover_ppu_torch_lib():
    """Locate the PPU libtorch .so dir.

    Priority: bundled lib_ppu/, then FLAGOS_VENDOR_TORCH_LIB, then sibling conda
    envs whose torch is a PPU build.
    """
    return discover_vendor_torch_lib(
        _BUNDLE_DIR,
        "libtorch_cuda.so",
        env_override="FLAGOS_VENDOR_TORCH_LIB",
        vendor_markers=_MARKERS,
    )


def ensure_ppu_libtorch_links():
    """Symlink the active torch wheel's core .so to the PPU build's copies.

    Idempotent; reversible via ``torch/lib/_orig_backup/``.  Returns True if
    links are in place (or already were), False if there was nothing to do.
    """
    return ensure_vendor_libtorch_links(
        _BUNDLE_DIR,
        _CORE_SO,
        extra_so=_CUDA_SO,
        env_override="FLAGOS_VENDOR_TORCH_LIB",
        vendor_markers=_MARKERS,
        probe_so="libtorch_cuda.so",
        vendor="PPU",
        load_order=_LOAD_ORDER,
    )


def _cuda_version_string(compiled_version):
    """Convert PyTorch's integer CUDA ABI code (for example 13000) to 13.0."""
    try:
        value = int(compiled_version)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"PPU libtorch reported an invalid compiled CUDA version: "
            f"{compiled_version!r}"
        ) from exc
    if value < 1000 or value % 10:
        raise RuntimeError(
            f"PPU libtorch reported an invalid compiled CUDA version: {value}"
        )
    major = value // 1000
    minor = (value % 1000) // 10
    return f"{major}.{minor}"


def restore_ppu_cuda_version():
    """Reconcile stock ``torch+cpu`` metadata with the loaded PPU runtime.

    ``torch.version.cuda`` comes from the Python wheel's generated version.py.
    Relinking that wheel to PPU's CUDA-enabled libtorch changes the actual C++
    runtime but cannot update the Python constant, leaving third-party packages
    such as bitsandbytes to mis-detect the live PPU as a CPU-only runtime.

    Query the version from the loaded C++ runtime instead of hard-coding an SDK
    release or carrying a second version file. A real vendor Torch already has
    the correct value and is left unchanged.
    """
    import torch

    current = getattr(torch.version, "cuda", None)
    if current:
        return current
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PPU libtorch is active but torch.cuda.is_available() is false; "
            "the vendor runtime was not loaded before importing torch"
        )
    getter = getattr(torch._C, "_cuda_getCompiledVersion", None)
    if not callable(getter):
        raise RuntimeError(
            "PPU libtorch does not expose torch._C._cuda_getCompiledVersion; "
            "cannot reconcile the CPU wheel's CUDA metadata"
        )
    version = _cuda_version_string(getter())
    torch.version.cuda = version
    return version


def restore_original_libtorch():
    """Undo ensure_ppu_libtorch_links(): remove links, restore backups."""
    _restore(_CORE_SO, _CUDA_SO, bundle_dirname=_BUNDLE_DIR)
