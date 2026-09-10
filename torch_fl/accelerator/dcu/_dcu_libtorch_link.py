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

"""Load DTK's libtorch *device* libraries on top of an official torch wheel.

DCU runs the CUDA boxing kernels on DTK's hipified libtorch: its HIP kernels are
registered under the CUDA dispatch key, so PrivateUse1 -> CUDA re-dispatch works
unchanged.  The kernels themselves only exist as machine code inside
``libtorch_hip.so``, so that library is always required; what is *not* required
is DTK's fork of the core runtime.

Default (decoupled) mode
    dlopen ``libc10_hip.so`` and ``libtorch_hip.so`` RTLD_GLOBAL *before*
    ``import torch``, against the official wheel's core.  Measured on DTK 2604 /
    torch 2.10.0: ``libc10_hip.so`` has zero vendor-core-only imports,
    ``libtorch_hip.so`` has 16 (``at::_ops::native_fuse_*::call``) and
    ``libtorch_fl.so`` itself has 16 more (``at::_ops::fuse_*::call``, picked up
    from compiling against DTK's patched ``ATen/autocast_mode.h``).  All 32 are
    supplied by ``libflagos_dtk_core_compat.so``, which is loaded first and
    throws a descriptive error if a DTK-private op is ever actually called.
    ``scripts/check_dcu_core_abi.py`` re-proves that gap at build time.

    Nothing under the official ``torch/lib`` is touched, so the wheel is
    installable next to a stock torch and uninstalling torch_fl leaves no trace.

Legacy mode (``FLAGOS_DCU_VENDOR_CORE=1``)
    Symlink DTK's whole core set over the official ``torch/lib`` and preload it,
    the pre-decoupling behaviour.  This is the rollback path, and the only mode
    where DTK-private schemas such as ``aten::native_fuse_rmsnorm`` work, since
    their schema wrappers and autograd registrations live in the core fork.  It
    mutates the torch installation in place (reversible via
    ``torch/lib/_orig_backup/``) and needs the matching full bundle from
    ``FLAGOS_DCU_VENDOR_CORE=1 bash scripts/bundle_dcu_libtorch.sh``.

Why the preload must happen before ``import torch``: PyTorch caches its
CUDAHooks on first import (docs/vendors/cuda/external-libtorch-cuda.md,
constraint 1).  Loading the device libraries afterwards leaves device init
failing with "Cannot initialize CUDA without ATen_cuda library" even though the
kernels did register.

The DTK driver stack (``libgalaxyhip.so.5``, ``libMIOpen.so.1``,
``librocblas.so.4``, ``librccl.so.1``, ...) stays on the target under
``/opt/dtk`` and is reached through the RUNPATH baked in by
``cmake/FlagosRpath.cmake`` / ``scripts/bundle_dcu_libtorch.sh``.
"""

import ctypes
import os

from torch_fl.accelerator._vendor_libtorch import (
    active_torch_lib,
    bundled_lib_dir,
    discover_vendor_torch_lib,
    ensure_vendor_libtorch_links,
)
from torch_fl.accelerator._vendor_libtorch import (
    restore_original_libtorch as _restore,
)

_BUNDLE_DIR = "lib_dcu"

# Core C++ .so that legacy mode takes from the DTK wheel as a self-consistent set.
_CORE_SO = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
)
# HIP-side .so the stock +cpu wheel does not ship at all. libmagma.so is a
# DT_NEEDED of DTK's libtorch_hip.so; libshm.so is one of libtorch_python.so's
# (torch.multiprocessing's shared-memory manager) and is a *different* build in
# the DTK wheel, so in legacy mode it has to come from the same set.
_HIP_SO = (
    "libc10_hip.so",
    "libtorch_hip.so",
    "libmagma.so",
    "libshm.so",
)

# Dependency order for legacy mode's RTLD_GLOBAL preload: core (CPU) first, then
# the HIP side, then libshm and libtorch_python. Symbols the plugin needs live in
# the forked CPU runtime, and loading only the HIP lib can leave its CPU
# dependency RTLD_LOCAL -- which is also why libshm.so is listed explicitly ahead
# of libtorch_python.so. libtorch.so has a transitive DT_NEEDED on libtorch_hip.so,
# which in turn needs libmagma.so. Load libmagma from the bundle before libtorch.so
# so its auditwheel-mangled MKL dependencies resolve relative to the bundle rather
# than through a torch/lib symlink selected by LD_LIBRARY_PATH in a source checkout.
_LOAD_ORDER = (
    "libc10.so",
    "libtorch_cpu.so",
    "libmagma.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libc10_hip.so",
    "libtorch_hip.so",
    "libshm.so",
    "libtorch_python.so",
)

# Decoupled mode. The official core has to be dlopened RTLD_GLOBAL first: the
# bundle's RUNPATH is $ORIGIN plus the DTK driver dirs and deliberately does NOT
# name the official torch/lib, so libtorch_hip.so's hard DT_NEEDED on libc10.so /
# libtorch_cpu.so would otherwise be unresolvable. Loading them by path up front
# satisfies those entries by soname against the already-mapped objects, and it is
# the same set `import torch` maps a moment later (same inode, no second copy).
_OFFICIAL_CORE_PRELOAD = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
)
# The compatibility shim must precede libtorch_hip.so so its 32 exports are
# already in the global scope when the loader binds it. libmagma.so is pulled in
# by libtorch_hip.so's DT_NEEDED through the bundle RUNPATH; naming it here just
# makes the failure mode obvious if it went missing.
# libcaffe2_nvrtc.so is reached by *soname* through a dlopen inside ATen's lazy
# NVRTC stub, not through any DT_NEEDED, so RUNPATH does not help: without it
# already mapped, constructing a CUDA generator fails with "Error in dlopen:
# libcaffe2_nvrtc.so: cannot open shared object file". Preloading it by path
# satisfies that dlopen from the process image. Measured: with it in place,
# torch.Generator(device="cuda").get_state() returns the 16-byte philox state
# FlagGems expects.
_COMPAT_SO = "libflagos_dtk_core_compat.so"
_DEVICE_LOAD_ORDER = (
    _COMPAT_SO,
    "libcaffe2_nvrtc.so",
    "libc10_hip.so",
    "libmagma.so",
    "libtorch_hip.so",
)

_MARKERS = ("dtk", "hip", "das")

# dlopen handles kept alive for the process lifetime.
_device_handles = []
_preloaded = False


def vendor_core_mode():
    """True when the legacy vendor-core (symlink) path is requested."""
    return os.environ.get("FLAGOS_DCU_VENDOR_CORE", "0").lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _bundled_dcu_lib():
    """DTK device libraries bundled inside this wheel (self-contained build)."""
    return bundled_lib_dir(_BUNDLE_DIR, "libtorch_hip.so")


def _discover_dcu_torch_lib():
    """Locate the DTK libtorch .so dir.

    Priority: bundled lib_dcu/, then FLAGOS_DCU_TORCH_LIB, then sibling conda
    envs whose torch is a DTK build.
    """
    return discover_vendor_torch_lib(
        _BUNDLE_DIR,
        "libtorch_hip.so",
        env_override="FLAGOS_DCU_TORCH_LIB",
        vendor_markers=_MARKERS,
    )


def _compat_shim_path(lib_dir):
    """The ABI shim, from the bundle or from the build tree next to it."""
    for cand in (
        os.path.join(lib_dir, _COMPAT_SO),
        # Non-bundled in-place build: cmake installs into torch_fl/lib_dcu, but a
        # dev may point FLAGOS_DCU_TORCH_LIB straight at the DTK wheel.
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            _BUNDLE_DIR,
            _COMPAT_SO,
        ),
    ):
        if os.path.exists(cand):
            return cand
    return None


def preload_dcu_device_libs():
    """dlopen the DTK device libraries RTLD_GLOBAL, before ``import torch``.

    Returns True when the device set is loaded (or already was), False when there
    is nothing to load (no bundle and no discoverable DTK torch, i.e. a plain
    non-DCU build).  Raises with an actionable message on a real failure: a
    partially loaded process would otherwise fail much later, inside a dispatch,
    with an undefined-symbol error that says nothing about the cause.
    """
    global _preloaded
    if _preloaded:
        return True

    lib_dir = _discover_dcu_torch_lib()
    if not lib_dir:
        return False

    shim = _compat_shim_path(lib_dir)
    if shim is None:
        raise RuntimeError(
            f"{_COMPAT_SO} not found next to the DTK device libraries in "
            f"{lib_dir}. It supplies the DTK-private ATen symbols that "
            "libtorch_hip.so imports from DTK's forked core, so the official "
            "PyTorch core cannot be used without it. Rebuild with "
            "ACCELERATOR=dcu python setup.py build_ext --inplace, or set "
            "FLAGOS_DCU_VENDOR_CORE=1 to use DTK's own core libraries."
        )

    fallback = active_torch_lib()
    if not fallback:
        raise RuntimeError(
            "DCU: no importable torch installation found; the DTK device "
            "libraries need an official PyTorch core to bind against."
        )
    for name in _OFFICIAL_CORE_PRELOAD:
        path = os.path.join(fallback, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DCU: official PyTorch core library missing: {path}"
            )
        _device_handles.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))

    for name in _DEVICE_LOAD_ORDER:
        path = shim if name == _COMPAT_SO else os.path.join(lib_dir, name)
        if not os.path.exists(path):
            # libmagma.so may legitimately live in the vendor wheel's own dir only.
            alt = os.path.join(fallback, name) if fallback else ""
            if alt and os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(
                    f"DCU/DTK device library missing: {os.path.join(lib_dir, name)}. "
                    "Run scripts/bundle_dcu_libtorch.sh, or point "
                    "FLAGOS_DCU_TORCH_LIB at a DTK torch/lib."
                )
        try:
            _device_handles.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load DCU/DTK device library {path}: {exc}\n"
                "Common causes: the DTK driver stack is not installed under "
                "/opt/dtk (set DTK_ROOT/ROCM_PATH and source its env.sh), or the "
                "installed torch wheel is ABI-incompatible with this DTK build "
                "(see docs/vendors/dcu/vendor-free-core-libs.md)."
            ) from exc

    _preloaded = True
    return True


def ensure_dcu_libtorch_links():
    """Legacy mode: symlink the active torch wheel's core .so to DTK's copies.

    Idempotent; reversible via ``torch/lib/_orig_backup/``.  Returns True if
    links are in place (or already were), False if there was nothing to do.
    """
    # Pre-flight: a bundle built in decoupled mode has no core .so, and the
    # generic linker would fail mid-way on the first missing one with a message
    # that does not explain the real cause (mode / bundle mismatch). Nothing is
    # mutated before this point, so bailing out here leaves torch/lib untouched.
    bundle = _bundled_dcu_lib()
    if bundle:
        absent = [so for so in _CORE_SO if not os.path.exists(os.path.join(bundle, so))]
        if absent:
            raise RuntimeError(
                "FLAGOS_DCU_VENDOR_CORE=1 needs DTK's core libraries in the "
                f"bundle, but {bundle} has none ({', '.join(absent)} missing). "
                "This wheel was bundled in decoupled mode. Either unset "
                "FLAGOS_DCU_VENDOR_CORE, or rebuild the bundle with "
                "FLAGOS_DCU_VENDOR_CORE=1 bash scripts/bundle_dcu_libtorch.sh."
            )
    return ensure_vendor_libtorch_links(
        _BUNDLE_DIR,
        _CORE_SO,
        extra_so=_HIP_SO,
        env_override="FLAGOS_DCU_TORCH_LIB",
        vendor_markers=_MARKERS,
        probe_so="libtorch_hip.so",
        vendor="DCU/DTK",
        load_order=_LOAD_ORDER,
    )


def setup_dcu_runtime():
    """Make DTK's kernels available in this process, before ``import torch``.

    Dispatches on ``FLAGOS_DCU_VENDOR_CORE``: decoupled preload by default,
    legacy relink when the env var is set. SDK-native mode deliberately skips
    the DTK torch-device preload when SDK-only operation was requested; the
    plugin is loaded later, after ``torch_fl._C`` has installed the core-side
    registration bridge.

    ``FLAGOS_DCU_SDK_ONLY=1`` and ``FLAGOS_DCU_VENDOR_CORE=1`` are mutually
    exclusive and rejected together rather than silently ordered: the first says
    "map no part of DTK's torch", the second replaces the whole core with it. A
    precedence rule either way would leave the user with a process that quietly
    contradicts one of the two switches they set.
    """
    from torch_fl.accelerator.dcu._dcu_sdk_ops import sdk_only

    if sdk_only() and vendor_core_mode():
        raise RuntimeError(
            "FLAGOS_DCU_SDK_ONLY=1 and FLAGOS_DCU_VENDOR_CORE=1 are mutually "
            "exclusive: SDK-only mode runs the DTK SDK on the official PyTorch "
            "core and maps no vendor torch, while vendor-core mode replaces that "
            "core with DTK's fork. Unset one of them."
        )
    if vendor_core_mode():
        return ensure_dcu_libtorch_links()
    if sdk_only():
        return False
    return preload_dcu_device_libs()


def restore_original_libtorch():
    """Undo ensure_dcu_libtorch_links(): remove links, restore backups.

    A no-op for decoupled mode, which never modified ``torch/lib``.
    """
    _restore(_CORE_SO, _HIP_SO, bundle_dirname=_BUNDLE_DIR)
