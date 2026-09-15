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

"""Point the active (stock) torch wheel's ``torch/lib`` at a bundled vendor libtorch.

Why this exists
---------------
Several backends run on a *forked* libtorch: MetaX (``at::maca::*``), Hygon DCU
(DTK's hipified build -- its ``libtorch_cpu.so`` carries hip symbols and needs
``libgalaxyhip.so.5``), and PPU (a local ``USE_CUDA=1`` build).  Measured symbol
attribution says the fork lives in the *core* libs: for both DCU and PPU,
``libtorch_cpu.so`` resolves ~2100 of ``libtorch_fl.so``'s undefined symbols while
the vendor lib (``libtorch_hip.so`` / ``libtorch_cuda.so``) resolves 0.  So a
self-contained wheel must ship the core libs, not just the vendor one.

Pure ``ctypes`` preloading does NOT work for core libs.  The stock wheel's
``_C.so`` / ``libtorch_python.so`` carry an ``$ORIGIN`` RUNPATH that pulls the
upstream ``libc10.so`` back in *by full path*, so the process ends up with two
libc10 and dies in duplicate static init (``Key already registered ...
caffe2_report_cpu_memory_usage``).  The robust fix is to make the physical files
that RUNPATH resolves to *be* the vendor ones: replace the stock wheel's
``torch/lib/<so>`` with symlinks into the bundle dir.  Originals move to
``torch/lib/_orig_backup/``, so the operation is fully reversible.

The CUDA backend is the one exception and does not use this module: the official
``+cpu`` wheel's core libs *are* the upstream ones, so only the extra CUDA libs
are missing and a ctypes preload (``torch_fl.__init__._preload_cuda_assets``)
suffices.

Callers must invoke this from ``torch_fl/__init__.py`` BEFORE ``import torch``
(afterwards libc10 is already mapped and relinking is too late).  Every entry
point here is idempotent.

Note on ``$ORIGIN`` and symlinks: glibc expands ``$ORIGIN`` from the path the
object was *loaded by*, NOT from its resolved target.  Opening
``torch/lib/libtorch_cpu.so`` (a symlink) therefore gives ``$ORIGIN`` =
``torch/lib``, where a bundle-internal dependency does not exist -- measured on
DCU, whose ``libc10.so`` needs the auditwheel-mangled ``libgflags-8aee0f6c.so``
that ships in the bundle dir:

    ctypes.CDLL(".../torch/lib/libc10.so")     -> libgflags-...so: not found
    ctypes.CDLL(".../torch_fl/lib_dcu/libc10.so") -> OK

So ``_preload_global`` dlopens the *bundle* paths, not the symlinks.  That also
covers the symlinks: glibc keys loaded objects by (device, inode), and a symlink
shares both with its target, so a later lookup that resolves through
``torch/lib`` finds the object already mapped instead of re-opening it.
"""

import ctypes
import importlib.util
import os
import sys

# One flag per bundle dir: a process only ever relinks for its own backend, but
# keying by name keeps the module reentrant and makes the no-op cheap.
_done = set()
# dlopen handles kept alive for the process lifetime (see _preload_global).
_runtime_handles = []


def active_torch_lib():
    """``torch/lib`` of the importable torch, WITHOUT importing torch."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    lib = os.path.join(spec.submodule_search_locations[0], "lib")
    return lib if os.path.isdir(lib) else None


def bundled_lib_dir(bundle_dirname, probe_so):
    """The bundle dir inside this wheel, if the bundling step actually ran.

    ``scripts/vendor/bundle_<vendor>_libtorch.sh`` copies the vendor libtorch .so into
    ``torch_fl/<bundle_dirname>/``.  When present this is the preferred source:
    the target machine then needs only the official ``torch+cpu`` wheel plus the
    vendor driver runtime, no vendor torch wheel at all.  Absent (a plain
    in-place/dev build) every entry point here becomes a no-op.
    """
    # this file: torch_fl/accelerator/_vendor_libtorch.py -> torch_fl/
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libdir = os.path.join(pkg_root, bundle_dirname)
    if os.path.isdir(libdir) and os.path.exists(os.path.join(libdir, probe_so)):
        return libdir
    return None


def _scan_sibling_envs(probe_so, vendor_markers):
    """Fallback for multi-env dev setups: a sibling conda env's vendor torch.

    Matches on ``torch/version.py`` containing one of ``vendor_markers`` (e.g.
    "metax"/"maca", "dtk"/"hip", "ppu") so we never pick up a stock wheel.
    """
    if not vendor_markers:
        return None
    prefix = os.environ.get("CONDA_PREFIX") or os.path.dirname(
        os.path.dirname(os.__file__)
    )
    envs_root = os.path.dirname(prefix)  # .../envs
    if not os.path.isdir(envs_root):
        return None
    py = "python{}.{}".format(*sys.version_info[:2])
    try:
        names = sorted(os.listdir(envs_root))
    except OSError:
        return None
    for name in names:
        cand = os.path.join(envs_root, name, "lib", py, "site-packages", "torch")
        libdir = os.path.join(cand, "lib")
        ver_file = os.path.join(cand, "version.py")
        if not os.path.isfile(ver_file) or not os.path.isdir(libdir):
            continue
        try:
            with open(ver_file) as f:
                txt = f.read()
        except OSError:
            continue
        if any(m in txt for m in vendor_markers) and os.path.exists(
            os.path.join(libdir, probe_so)
        ):
            return libdir
    return None


def discover_vendor_torch_lib(
    bundle_dirname, probe_so, env_override=None, vendor_markers=()
):
    """Locate the vendor libtorch .so dir.

    Priority: bundled in this wheel, then ``env_override``, then sibling conda
    envs whose torch is a vendor build.
    """
    bundled = bundled_lib_dir(bundle_dirname, probe_so)
    if bundled:
        return bundled
    if env_override:
        env = os.environ.get(env_override)
        if env and os.path.isdir(env):
            return env
    return _scan_sibling_envs(probe_so, vendor_markers)


def _link_one(dst_dir, backup_dir, name, target, required, vendor):
    """Idempotently point ``dst_dir/name`` at ``target`` (a vendor .so)."""
    dst = os.path.join(dst_dir, name)
    if not os.path.exists(target):
        if required:
            raise FileNotFoundError(f"{vendor} so missing: {target}")
        return
    # Already correctly linked?
    if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(target):
        return
    # Back up a real (non-symlink) original once.
    if os.path.exists(dst) and not os.path.islink(dst):
        os.makedirs(backup_dir, exist_ok=True)
        bak = os.path.join(backup_dir, name)
        if not os.path.exists(bak):
            os.replace(dst, bak)
        else:
            os.remove(dst)
    elif os.path.islink(dst):
        os.remove(dst)  # stale/incorrect link
    os.symlink(target, dst)


def _preload_global(lib_dir, load_order, core_so, vendor, fallback_dir=None):
    """dlopen the vendor set RTLD_GLOBAL, in dependency order.

    A CPU-only torch wheel never loads the forked runtime itself.  Symlinking the
    files is not sufficient on its own: symbols the plugin needs may live in the
    forked *CPU* runtime (``GetFlagosDefaultCudaGenerator`` is the measured case
    on MetaX) and loading only the vendor library can leave its CPU dependency
    RTLD_LOCAL, after which ``libtorch_fl.so`` cannot resolve that symbol.
    Loading the whole set globally, core first, avoids that.

    ``lib_dir`` must be the *source* dir (the bundle), never the ``torch/lib``
    symlink dir -- see the ``$ORIGIN`` note in the module docstring.

    ``fallback_dir`` (the stock wheel's ``torch/lib``) covers a non-core .so the
    vendor image simply does not ship.  Measured: the MetaX CI's
    ``/opt/vendor-libtorch/lib`` has no ``libshm.so``, so the bundle has none
    either, yet ``libtorch_python.so`` carries a hard ``DT_NEEDED: libshm.so``.
    dlopening the stock copy RTLD_GLOBAL *before* ``libtorch_python.so``
    satisfies that DT_NEEDED by soname against the already-loaded object; without
    it the loader only searches the bundle's RUNPATH and dies with "libshm.so:
    cannot open shared object file".  Its own deps (libc10, libtorch_cpu) resolve
    through ``torch/lib``, where they are symlinks into the bundle, so they share
    an inode with what is already mapped and no second copy appears.
    """
    handles = []
    for name in load_order:
        path = os.path.join(lib_dir, name)
        if not os.path.exists(path):
            if name in core_so:
                raise FileNotFoundError(f"{vendor} libtorch runtime missing: {path}")
            path = os.path.join(fallback_dir, name) if fallback_dir else ""
            if not path or not os.path.exists(path):
                continue
        try:
            handles.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load {vendor} libtorch runtime: {path}"
            ) from exc
    return handles


def ensure_vendor_libtorch_links(
    bundle_dirname,
    core_so,
    extra_so=(),
    env_override=None,
    vendor_markers=(),
    probe_so=None,
    vendor=None,
    load_order=None,
):
    """Symlink the active torch wheel's core .so to the vendor wheel's copies.

    Args:
        bundle_dirname: dir inside the wheel holding the bundle ("lib_maca", ...).
        core_so: .so that MUST be present; a missing one raises.
        extra_so: .so the stock ``+cpu`` wheel may not ship at all (the vendor
            libs); linked in fresh when available, skipped silently otherwise.
        env_override: env var naming an explicit source dir.
        vendor_markers: substrings identifying a vendor torch in ``version.py``.
        probe_so: file whose presence proves a dir is a real vendor libtorch dir.
            Defaults to the first entry of ``extra_so``, else of ``core_so``.
        vendor: label used in error messages.
        load_order: when given, dlopen these RTLD_GLOBAL after linking, in this
            order (see ``_preload_global``). Names absent from ``core_so`` may be
            missing; a missing core .so raises.

    Returns True if links are in place (or already were), False if there was
    nothing to do (no bundle, no vendor torch found, or already running on it).
    """
    if bundle_dirname in _done:
        return True
    probe = probe_so or (extra_so[0] if extra_so else core_so[0])
    label = vendor or bundle_dirname

    active = active_torch_lib()
    src = discover_vendor_torch_lib(
        bundle_dirname, probe, env_override=env_override, vendor_markers=vendor_markers
    )
    if active is None or src is None:
        return False
    # Already running on the vendor wheel itself -> nothing to do.
    if os.path.realpath(active) == os.path.realpath(src):
        _done.add(bundle_dirname)
        return True

    backup = os.path.join(active, "_orig_backup")
    for name in core_so:
        _link_one(active, backup, name, os.path.join(src, name), True, label)
    for name in extra_so:
        _link_one(active, backup, name, os.path.join(src, name), False, label)

    if load_order:
        # dlopen from `src` (the bundle), not from `active`: loading through the
        # torch/lib symlinks would expand $ORIGIN to torch/lib and lose the
        # bundle-internal deps. `active` is only the fallback for a non-core .so
        # the vendor image does not ship. Keep the handles alive for the process
        # lifetime.
        _runtime_handles.extend(
            _preload_global(src, load_order, core_so, label, fallback_dir=active)
        )

    _done.add(bundle_dirname)
    return True


def restore_original_libtorch(core_so, extra_so=(), bundle_dirname=None):
    """Undo ensure_vendor_libtorch_links(): drop links, restore the backups."""
    active = active_torch_lib()
    if active is None:
        return
    backup = os.path.join(active, "_orig_backup")
    for name in tuple(core_so) + tuple(extra_so):
        dst = os.path.join(active, name)
        if os.path.islink(dst):
            os.remove(dst)
    if os.path.isdir(backup):
        for name in os.listdir(backup):
            os.replace(os.path.join(backup, name), os.path.join(active, name))
        try:
            os.rmdir(backup)
        except OSError:
            pass
    if bundle_dirname:
        _done.discard(bundle_dirname)
