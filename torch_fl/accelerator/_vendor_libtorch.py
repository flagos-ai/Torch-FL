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

"""Load a bundled vendor libtorch through a private torch package overlay.

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
upstream ``libc10.so`` back in, so the process ends up with two libc10 and dies
in duplicate static init.  Instead, create a private ``torch/`` facade under a
user-owned cache.  Its top-level files point at the installed torch package;
its ``lib/`` points at the bundled vendor libraries, falling back to stock
files only when the bundle does not provide one.  Put that facade first on
``sys.path`` (and ``PYTHONPATH`` for child interpreters) before importing torch.
The installed torch wheel is never written to, including on interrupted startup.

The CUDA backend is the one exception and does not use this module: the official
``+cpu`` wheel's core libs *are* the upstream ones, so only the extra CUDA libs
are missing and a ctypes preload (``torch_fl.__init__._preload_cuda_assets``)
suffices.

Callers must invoke this from ``torch_fl/__init__.py`` BEFORE ``import torch``
(afterwards libc10 is already mapped and selecting another ABI is too late).
Every entry point here is idempotent. A vendor-core bundle must contain the
required core libraries and a ``vendor_version.py`` with the same three-part
PyTorch version as the installed front-end. An unpackaged vendor torch can use
its adjacent ``version.py``. A legacy library-only image may declare that
version with ``TORCH_FL_VENDOR_TORCH_VERSION`` instead.

Note on ``$ORIGIN`` and symlinks: glibc expands ``$ORIGIN`` from the path the
object was loaded by, not from its resolved target.  ``_preload_global`` opens
vendor libraries by their bundle paths so their bundled dependencies resolve;
the facade includes those dependencies as well for later RUNPATH lookups.
"""

import ast
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile

# One flag per bundle dir: a process only ever activates its own backend, but
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


def _same_core_files(active, src, core_so):
    """Only skip the overlay when every active core library is byte-identical."""
    for name in core_so:
        current = os.path.join(active, name)
        vendor = os.path.join(src, name)
        if not os.path.isfile(current) or not os.path.isfile(vendor):
            return False
        if os.path.samefile(current, vendor):
            continue
        if os.path.getsize(current) != os.path.getsize(vendor):
            return False
        digests = []
        for path in (current, vendor):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests.append(digest.digest())
        if digests[0] != digests[1]:
            return False
    return True


def _torch_base_version(path):
    """Read the generated torch version without importing (or executing) torch."""
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"Cannot read PyTorch ABI version from {path}") from exc
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise RuntimeError(f"Invalid PyTorch ABI version in {path}") from exc
        match = (
            re.match(r"^(\d+)\.(\d+)\.(\d+)", value) if isinstance(value, str) else None
        )
        if match:
            return match.group(0)
    raise RuntimeError(f"Missing PyTorch ABI version in {path}")


def _check_abi_version(active, src, vendor):
    installed = os.path.join(os.path.dirname(active), "version.py")
    bundled = os.path.join(src, "vendor_version.py")
    source = (
        bundled
        if os.path.isfile(bundled)
        else os.path.join(os.path.dirname(src), "version.py")
    )
    front_version = _torch_base_version(installed)
    if os.path.isfile(source):
        vendor_version = _torch_base_version(source)
    else:
        # Older MetaX images expose only a staged torch/lib directory. The
        # image owner can declare its ABI explicitly; an undeclared ABI fails.
        vendor_version = os.environ.get("TORCH_FL_VENDOR_TORCH_VERSION", "")
        if not re.fullmatch(r"\d+\.\d+\.\d+", vendor_version):
            raise RuntimeError(
                f"{vendor} libtorch ABI version metadata missing: {source}; "
                "supply torch/version.py or TORCH_FL_VENDOR_TORCH_VERSION"
            )
    if front_version != vendor_version:
        raise RuntimeError(
            f"{vendor} libtorch {vendor_version} is incompatible with the "
            f"installed PyTorch Python package {front_version}; install a matching "
            "torch wheel or rebuild the vendor bundle"
        )


def _overlay_cache_root():
    """Use a private cache so concurrent interpreters share only complete overlays."""
    root = os.path.join(tempfile.gettempdir(), f"torch-fl-libtorch-{os.getuid()}")
    os.makedirs(root, mode=0o700, exist_ok=True)
    info = os.lstat(root)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise RuntimeError(f"Unsafe vendor libtorch overlay cache: {root}")
    return root


def _overlay_key(active, src):
    paths = []
    for path in (os.path.dirname(active), active, src):
        info = os.stat(path)
        paths.append(
            (os.path.realpath(path), info.st_dev, info.st_ino, info.st_mtime_ns)
        )
    return hashlib.sha256(repr((sys.version_info[:2], paths)).encode()).hexdigest()[:24]


def _overlay_marker(active):
    """Return the vendor source if active torch/lib already belongs to an overlay."""
    marker = os.path.join(os.path.dirname(os.path.dirname(active)), ".ready")
    try:
        with open(marker, encoding="utf-8") as handle:
            return json.load(handle).get("vendor_source")
    except (OSError, ValueError, AttributeError):
        return None


def _link_entries(src, dst, *, skip=()):
    for entry in os.scandir(src):
        if entry.name in skip:
            continue
        os.symlink(entry.path, os.path.join(dst, entry.name))


def _prepare_overlay(active, src):
    """Atomically publish a private torch facade, leaving the installed wheel intact."""
    torch_root = os.path.dirname(active)
    if not os.path.isfile(os.path.join(torch_root, "__init__.py")):
        raise RuntimeError(f"No torch package found beside {active}")
    if os.path.exists(os.path.join(active, "_orig_backup")):
        raise RuntimeError(
            f"{active} contains a legacy _orig_backup; restore the installed "
            "PyTorch wheel before using the immutable vendor runtime"
        )
    cache = _overlay_cache_root()
    key = _overlay_key(active, src)
    overlay = os.path.join(cache, key)
    ready = os.path.join(overlay, ".ready")
    if os.path.isfile(ready):
        return overlay
    stage = tempfile.mkdtemp(prefix=f"{key}.", dir=cache)
    try:
        facade = os.path.join(stage, "torch")
        os.mkdir(facade)
        facade_lib = os.path.join(facade, "lib")
        os.mkdir(facade_lib)
        _link_entries(torch_root, facade, skip=("lib",))
        # Vendor files take precedence.  Including the whole bundle is required
        # for auditwheel-mangled and other $ORIGIN-relative dependencies.
        vendor_names = {entry.name for entry in os.scandir(src)}
        _link_entries(src, facade_lib)
        _link_entries(active, facade_lib, skip=vendor_names | {"_orig_backup"})
        with open(os.path.join(stage, ".ready"), "w", encoding="utf-8") as handle:
            json.dump({"vendor_source": os.path.realpath(src)}, handle)
        try:
            os.replace(stage, overlay)
        except OSError:
            # Another process may have published the same complete overlay.
            if not os.path.isfile(ready):
                raise
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage)
    return overlay


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

    ``fallback_dir`` (the private facade's ``torch/lib``) covers a non-core .so the
    vendor image simply does not ship.  Measured: the MetaX CI's
    ``/opt/vendor-libtorch/lib`` has no ``libshm.so``, so the bundle has none
    either, yet ``libtorch_python.so`` carries a hard ``DT_NEEDED: libshm.so``.
    dlopening the stock copy RTLD_GLOBAL *before* ``libtorch_python.so``
    satisfies that DT_NEEDED by soname against the already-loaded object; without
    it the loader only searches the bundle's RUNPATH and dies with "libshm.so:
    cannot open shared object file".  Its own deps (libc10, libtorch_cpu) resolve
    through the facade's ``torch/lib``, where they are symlinks into the bundle, so they share
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
    """Select the vendor core through a private torch facade before importing torch.

    Args:
        bundle_dirname: dir inside the wheel holding the bundle ("lib_maca", ...).
        core_so: .so that MUST be present; a missing one raises.
        extra_so: vendor libraries expected by the caller; retained for API
            compatibility. The facade contains every file in the vendor bundle.
        env_override: env var naming an explicit source dir.
        vendor_markers: substrings identifying a vendor torch in ``version.py``.
        probe_so: file whose presence proves a dir is a real vendor libtorch dir.
            Defaults to the first entry of ``extra_so``, else of ``core_so``.
        vendor: label used in error messages.
        load_order: when given, dlopen these RTLD_GLOBAL after creating the facade, in this
            order (see ``_preload_global``). Names absent from ``core_so`` may be
            missing; a missing core .so raises.

    Returns True when vendor libraries are selected (or already active), False
    when no bundle or vendor torch is available. The installed wheel is immutable.
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
    _check_abi_version(active, src, label)
    # An inherited PYTHONPATH can select an already published facade before this
    # module is imported in a subprocess. Its core libraries are already correct.
    selected_source = _overlay_marker(active)
    if selected_source and selected_source != os.path.realpath(src):
        raise RuntimeError(
            f"{label} cannot use torch from a different vendor facade "
            f"({selected_source}); clear the inherited PYTHONPATH before startup"
        )
    if selected_source:
        if load_order:
            _runtime_handles.extend(
                _preload_global(src, load_order, core_so, label, fallback_dir=active)
            )
        _done.add(bundle_dirname)
        return True
    if os.path.realpath(active) == os.path.realpath(src) or _same_core_files(
        active, src, tuple(core_so) + tuple(extra_so)
    ):
        _done.add(bundle_dirname)
        return True
    if "torch" in sys.modules:
        raise RuntimeError(
            f"{label} requires a different libtorch core; import torch_fl before "
            "torch so the vendor runtime can be selected without changing the "
            "installed PyTorch wheel"
        )
    for name in core_so:
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} libtorch runtime missing: {path}")

    overlay = _prepare_overlay(active, src)
    facade_lib = os.path.join(overlay, "torch", "lib")

    if load_order:
        # Load bundle dependencies through their real source paths. The facade
        # provides a safe fallback for libraries missing from the bundle.
        _runtime_handles.extend(
            _preload_global(src, load_order, core_so, label, fallback_dir=facade_lib)
        )

    sys.path.insert(0, overlay)
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        os.pathsep.join((overlay, inherited)) if inherited else overlay
    )
    importlib.invalidate_caches()
    _done.add(bundle_dirname)
    return True


def restore_original_libtorch(core_so, extra_so=(), bundle_dirname=None):
    """Compatibility no-op: the installed torch wheel no longer needs restoring.

    A loaded libtorch cannot be switched back inside the same interpreter. The
    private facade and its paths therefore remain active until process exit.
    """
