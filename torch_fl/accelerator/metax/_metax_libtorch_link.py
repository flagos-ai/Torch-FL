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

"""Symlink MetaX libtorch .so into the active (official) torch wheel's lib dir.

On MetaX we reuse PyTorch's CUDA boxing kernels by running
the *MetaX* C++ runtime (libtorch_cpu.so / libtorch_cuda.so / libc10.so ...),
which is a hard fork exporting ``at::maca::*`` symbols.  With a stock
``torch==X.Y.Z+cpu`` front-end, the process must load that fork instead of the
upstream .so shipped in ``torch/lib``.

The mechanism -- symlink replacement, ``_orig_backup/``, the RTLD_GLOBAL preload,
and why a pure ctypes preload cannot do this on its own -- lives in
``torch_fl.accelerator._vendor_libtorch``.  This module is just the MetaX .so
lists; whether to relink at all is decided by
``torch_fl.__init__._relink_vendor_libtorch`` (unconditional on a MetaX build --
it is boxing-only, reaching the vendor torch through FLAGOS_MACA_TORCH_LIB or a
self-contained lib_maca/ bundle).
"""

from torch_fl.accelerator._vendor_libtorch import (
    bundled_lib_dir,
    discover_vendor_torch_lib,
    ensure_vendor_libtorch_links,
)
from torch_fl.accelerator._vendor_libtorch import (
    restore_original_libtorch as _restore,
)

_BUNDLE_DIR = "lib_maca"

# Core C++ .so that must come from the MetaX wheel as a self-consistent set.
# libtorch_python.so is included because the stock one references symbols
# (e.g. torch::jit::fuser::onednn::fuseGraph) absent from the MetaX libtorch_cpu.so.
_CORE_SO = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
)
# CUDA .so the stock +cpu wheel does not ship at all; symlinked in fresh.
# libshm.so is a DT_NEEDED of MetaX's libtorch_python.so (torch.multiprocessing's
# shared-memory manager) and is a *different* build in the MetaX wheel, so it has
# to come from the same set.
_CUDA_SO = (
    "libc10_cuda.so",
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
    "libshm.so",
)

# Dependency order for the RTLD_GLOBAL preload: core (CPU) first, then the CUDA
# side, then libtorch_python.  GetFlagosDefaultCudaGenerator lives in the forked
# ATen CPU runtime, not in libtorch_cuda.so, so loading only the CUDA lib would
# leave that symbol unresolvable from libtorch_fl.so.
#
# libshm.so must precede libtorch_python.so: it is a direct DT_NEEDED of it, and
# dlopen resolves that by soname through the *loader's* search path, not through
# this list.  In a fresh install nothing has put libshm.so anywhere the loader
# looks yet, so omitting it here fails with "libshm.so: cannot open shared object
# file" even though the file is sitting in the bundle dir.
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

_MARKERS = ("metax", "maca")


def _bundled_maca_lib():
    """Forked libtorch bundled inside this wheel (self-contained MetaX build)."""
    return bundled_lib_dir(_BUNDLE_DIR, "libtorch_cuda.so")


def _discover_maca_torch_lib():
    """Locate the MetaX libtorch .so dir.

    Priority: bundled lib_maca/, then FLAGOS_MACA_TORCH_LIB, then sibling conda
    envs whose torch is a ``+metax``/``+maca`` build.
    """
    return discover_vendor_torch_lib(
        _BUNDLE_DIR,
        "libtorch_cuda.so",
        env_override="FLAGOS_MACA_TORCH_LIB",
        vendor_markers=_MARKERS,
    )


def ensure_maca_libtorch_links():
    """Symlink the active torch wheel's core .so to the MetaX wheel's copies.

    Idempotent; reversible via ``torch/lib/_orig_backup/``.  Returns True if
    links are in place (or already were), False if there was nothing to do (no
    bundle, no MetaX torch found, or torch already IS the MetaX wheel).

    Deciding *whether* to relink is the caller's job -- see
    ``torch_fl.__init__._relink_vendor_libtorch``, which relinks on every MetaX
    build (a self-contained wheel finds lib_maca/, an in-place build finds the
    vendor torch through FLAGOS_MACA_TORCH_LIB).  This used to self-gate on a
    mode variable, which made the self-contained path a silent no-op:
    the stock libtorch_cpu.so stayed in place and libtorch_cuda.so then failed
    to resolve at::maca symbols that only the forked CPU runtime defines.
    """
    return ensure_vendor_libtorch_links(
        _BUNDLE_DIR,
        _CORE_SO,
        extra_so=_CUDA_SO,
        env_override="FLAGOS_MACA_TORCH_LIB",
        vendor_markers=_MARKERS,
        probe_so="libtorch_cuda.so",
        vendor="MetaX",
        load_order=_LOAD_ORDER,
    )


def restore_original_libtorch():
    """Undo ensure_maca_libtorch_links(): remove links, restore backups."""
    _restore(_CORE_SO, _CUDA_SO, bundle_dirname=_BUNDLE_DIR)
