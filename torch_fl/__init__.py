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

import ctypes
import functools
import os
import sys

# Single access point for every FLAGOS_* variable (see torch_fl/_env.py). Imported
# for its side effect as well: it scans the environment once and warns about a
# FLAGOS_* name torch_fl does not recognise, which is the only way a typo like
# FLAGOS_LOG_DISPACH is ever noticed -- the misspelled variable is simply never
# read and the setting silently does nothing.
from torch_fl import _env  # noqa: F401
from torch_fl import _platform


# The answer is fixed for the life of the process: a different wheel is a
# different process, and nothing in this one can swap _build_config.py out from
# under it. Reading it is not cheap, though -- find_spec, a Path.resolve (a
# realpath syscall) and an exec_module -- and _remap asks for it on every call
# that carries a device argument, which is every factory call in a model.
# Measured on an MLU590 with no cache: torch.device("flagos:0") took 197us and
# torch.empty(4, device=...) 207.5us, against 2us and 6.8us with it.
#
# The memo lives here rather than in _platform.build_accelerator() on purpose.
# The integration test trees load _platform.py by path -- under their own module
# names, and not through sys.modules -- to stand in for "a process running a
# different wheel", and a memo inside _platform would follow them across the
# swap and answer with the previous test's platform. Here, only the alias the
# runtime itself calls is memoised and those callers keep reading through.
_ACCELERATOR = (None, "")


def _build_accelerator() -> str:
    """Accelerator this wheel was built for, lowercased ("" if unknown).

    Thin alias for torch_fl._platform.build_accelerator(), which the integration
    test trees read too (loaded by path). Read from the _build_config.py setup.py
    writes at build time, and never from the environment. The generated file is
    what makes a DCU wheel self-describing: _select_backend_config() runs before
    `import torch`, so it cannot inspect torch.version.hip to detect DCU on its
    own. A stale FLAGOS_ACCELERATOR from another build must not select a conf the
    wheel was not built for; to route through a different conf on purpose,
    FLAGOS_BACKEND_CONFIG names the file directly.

    Memoised after the first call; see _ACCELERATOR.
    """
    global _ACCELERATOR
    # Keyed on the delegate's identity, not on nothing. In a real process
    # _platform.build_accelerator is the same function object for the life of the
    # runtime and this is a hit every time. A test that pins a different record
    # replaces that attribute on the module (tests/unit/test_ascend_platform_marker.py
    # does exactly this), which makes the identity differ and re-reads -- so the
    # memo cannot shadow a monkeypatched source.
    source = _platform.build_accelerator
    if _ACCELERATOR[0] is source:
        return _ACCELERATOR[1]
    value = source()
    _ACCELERATOR = (source, value)
    return value


def _is_ppu_build() -> bool:
    """True for a PPU wheel, which needs backends_thead.conf rather than CUDA's.

    PPU has its own value in the build record's ACCELERATOR now (it is a CUDA-ABI
    boxing vendor like MetaX/DCU), so the record already answers this -- no
    environment probe. The lib_thead/ bundle directory stays as a fallback so a
    wheel built before PPU had an accelerator value of its own is still
    recognised.

    Without this, PPU read backends_nvidia.conf and inherited its FlagGems routes.
    That is what sent mm/bmm through the _hygon kernel PPU's triton cannot
    compile: 28 minutes inside a single test_mm and a SIGSEGV at interpreter
    exit, on a runner where every assertion still passed.
    """
    if _build_accelerator() == "thead":
        return True
    return os.path.isdir(os.path.join(os.path.dirname(__file__), "lib_thead"))


# The conf _select_backend_config() resolved, or "" when the user named one
# through FLAGOS_BACKEND_CONFIG (nothing to resolve) or none was found. Handed to
# the C++ routing table right after _C loads; read through backend_config_path().
_BACKEND_CONFIG_PATH = ""
_STARTUP_PROFILE = "full"


def _mps_shutdown_allocator() -> None:
    """Drain Metal and free its cached blocks before interpreter teardown."""
    try:
        if "torch_fl._C" in sys.modules:
            torch_fl._C._synchronize()
            torch_fl._C._shutdown_allocator()
    except (NameError, AttributeError):
        # Python may clear this module before atexit runs. The runtime's own
        # final-buffer hook still releases the cache.
        pass


def _select_backend_config() -> None:
    """Pick the op-routing config file for this build.

    Ordinary wheels select backends_<vendor>.conf. Experimental MetaX native
    hosts select backends_metax_native_host.conf with their additional claims.
    Selection is a property of the build, not of a runtime mode variable.
    The platform is read from the build record setup.py writes
    (torch_fl/_build_config.py): the wheel records the platform it resolved
    FLAGOS_ACCELERATOR to, and that record, not the environment and not a file
    in the install directory, is what this reads.

    It used to read lib/flagos_platform first. That file is an install()
    artifact in torch_fl/lib/, a directory no build cleans in a source tree
    (and one the marker tests write "ascend"/"mps" into), so whichever platform
    built there last decided the conf for every later build: a MetaX build that
    inherited an "mps" marker read the empty backends_apple.conf and raised
    "routed to 'none'" for every op that conf does not list. It is no longer
    consulted, and builds that write no marker now remove a stale one.

    Within that build family, numerical kernel switches decide which entry
    point an op takes once dispatched: flaggems_cpp is Backend::kFlagGemsCpp and
    needs FLAGOS_BUILD_FLAGGEMS_CPP=ON (liboperators.so built for the vendor);
    tileops is Backend::kTileOps and needs FLAGOS_BUILD_TILEOPS=ON plus an SM90
    device. Where the slot is empty Dispatcher::GetFn degrades to the boxing
    kernel rather than raising, so one file stays correct across both builds.
    FLAGOS_FORCE_BACKEND survives as a table transform in common.cc
    (ApplyForcedBackend): the tileops mode repins every op the conf annotates
    `# tileops` onto that path, the same op set backends_tileops.conf used to
    carry.

    Which entry point an op actually takes is then decided by what got compiled
    in, not by which file was read. flaggems_cpp is Backend::kFlagGemsCpp and needs
    FLAGOS_BUILD_FLAGGEMS_CPP=ON (liboperators.so built for the vendor); tileops
    needs FLAGOS_BUILD_TILEOPS=ON and a provider for the actual device. Where the
    slot is empty Dispatcher::GetFn degrades to the boxing kernel rather than
    raising, so one file stays correct across both builds. Every conf states all
    five keys per op directly, and a route a build cannot serve degrades at
    dispatch rather than being swapped for another file.

    An explicit FLAGOS_BACKEND_CONFIG always wins (advanced/testing use), and
    the per-op FLAGOS_OP_<name> overrides in common.cc still apply on top. This
    must run before the first op dispatch triggers BackendTable() init; setting
    it at import time (before any flagos tensor op) is well before that.

    The choice is handed to the C++ reader through
    _C._set_backend_config_path(), not written to os.environ. It used to be an
    environment write, which made the wheel's own selection indistinguishable
    from a user's for the rest of the process -- see backend_config_path().
    """
    global _BACKEND_CONFIG_PATH

    # Recorded per call, not accumulated. backend_config_path() reports what this
    # resolution picked *first*, mirroring the C++ reader's setter-before-env
    # order, so a path left over from an earlier call would outrank the user's
    # FLAGOS_BACKEND_CONFIG and the early return below would leave it in place.
    # At import time the function runs once and the reset is a no-op; it is what
    # makes a direct second call -- which tests/unit/test_ascend_platform_marker.py
    # makes, against a different fake tree each time -- answer for the
    # environment in front of it rather than for the previous test's.
    _BACKEND_CONFIG_PATH = ""

    if _env.value("FLAGOS_BACKEND_CONFIG"):
        # An override that cannot be read is fatal, not a fallback. The C++
        # loader leaves its table empty when the file will not open, and an empty
        # table resolves every op to Dispatcher::GetFn's default
        # (Backend::kFlagGems) -- so a typo here silently routes the whole
        # workload down the slowest path behind a single stderr line, and it also
        # switches off the FlagGems process-level setup that reads this same file
        # (see compat.flaggems.runtime.routes_to_flaggems). Measured on A100: a nonexistent
        # path sends every dispatch to flagos_python.
        #
        # Resolved to an absolute path because the C++ side reads the variable
        # lazily, at the first op dispatch, which a caller may reach after a
        # chdir -- and a relative path is resolved against the cwd at that
        # moment, not at import.
        #
        # Recorded through _BACKEND_CONFIG_PATH (which _set_backend_config_path
        # hands to the C++ reader), not written back to os.environ: an
        # environment write makes the wheel's own selection indistinguishable
        # from the user's for the rest of the process -- see backend_config_path().
        override = _env.value("FLAGOS_BACKEND_CONFIG")
        resolved = os.path.abspath(override)
        try:
            with open(resolved):
                pass
        except OSError as err:
            raise RuntimeError(
                f"FLAGOS_BACKEND_CONFIG={override!r} cannot be read "
                f"(resolved to {resolved!r}): {err}. Unset it to let torch_fl "
                f"select the conf matching this build, or point it at an "
                f"existing conf file."
            ) from err
        _BACKEND_CONFIG_PATH = resolved
        return

    # The conf follows from the build itself, not from a runtime mode switch: the
    # recorded accelerator says what the wheel was built for, and the conf is
    # named after the vendor it resolves to. The record is the answer; the two
    # wheels that predate their record value still resolve: an older PPU wheel
    # reports cuda and is recognised by lib_thead/, an older MetaX wheel
    # reports maca.
    platform = _platform.platform_for(_build_accelerator())
    if _is_ppu_build():
        platform = "ppu"
    vendor = _platform.vendor_for(platform) or "nvidia"
    conf_name = f"backends_{vendor}.conf"
    if vendor == "metax":
        from torch_fl import _build_config

        if getattr(_build_config, "NATIVE_HOSTS", False):
            conf_name = f"backends_{vendor}_native_host.conf"
    conf_path = os.path.join(os.path.dirname(__file__), "configs", conf_name)
    if os.path.exists(conf_path):
        _BACKEND_CONFIG_PATH = conf_path
        return
    if conf_name.endswith("_native_host.conf"):
        raise RuntimeError(
            f"native-host wheel is missing its generated config: {conf_path}"
        )
    # A vendor conf named for the build's own accelerator but shipped under the
    # other of its names: `musa` and `mthreads` are the same vendor (the record,
    # the CI platform and the conf generator each use one of them). Without the
    # fallback no conf is selected at all, which leaves every op unclaimed while
    # the build still has its vendor and FlagGems kernels -- the call reaches
    # the dispatcher from the other side. The `[flagos] loading backend config`
    # line names the file that was taken.
    alias_path = os.path.join(
        os.path.dirname(__file__),
        "configs",
        f"backends_{_platform.vendor_alias(_build_accelerator())}.conf",
    )
    if alias_path != conf_path and os.path.exists(alias_path):
        _BACKEND_CONFIG_PATH = alias_path


def backend_config_path() -> str:
    """Path of the conf the op-routing table is read from ("" if none).

    Answers in the same order the C++ reader resolves it
    (csrc/aten/common.cc:ResolveBackendConfigPath): what _select_backend_config()
    picked and handed over, then the user's FLAGOS_BACKEND_CONFIG, then nothing --
    the last case being a path C++ derives from its own library location and
    Python has no reason to duplicate.

    It used to be answered by reading os.environ["FLAGOS_BACKEND_CONFIG"], which
    torch_fl wrote at import time. That made "the user overrode the conf" and
    "the wheel chose its own conf" the same observable state for the rest of the
    process, and it leaked the path into every child process the wheel spawns.
    """
    return _BACKEND_CONFIG_PATH or _env.value("FLAGOS_BACKEND_CONFIG") or ""


def _relink_vendor_libtorch() -> None:
    """Select bundled vendor libtorch before importing torch.

    MetaX, DCU and PPU all run on a *forked* libtorch whose core .so
    (libc10/libtorch_cpu/libtorch_python/...) differ from the upstream ones a
    stock ``torch==X.Y.Z+cpu`` wheel ships.  A self-contained wheel bundles them
    under torch_fl/lib_{maca,dcu,ppu}/ and expose them through a private torch
    package facade without modifying the installed PyTorch wheel;
    see torch_fl.accelerator._vendor_libtorch for why a ctypes preload alone is
    not enough there.

    This MUST run before `import torch` -- afterwards libc10 is already mapped
    and selecting another core is too late. Every backend's entry point is idempotent and a
    no-op when its bundle dir is absent (a plain in-place build, where torch
    already IS the vendor wheel), so this is safe to call unconditionally.

    MetaX native-host wheels retain the official CPU core. Other MetaX builds
    select the vendor core (an in-place build reaches it through
    FLAGOS_VENDOR_TORCH_LIB, a self-contained wheel through lib_metax/).
    DCU and PPU have no native-kernel
    mode, so bundle-dir presence alone decides.  The CUDA backend is not here:
    the official +cpu wheel's core .so ARE the upstream ones, so only the extra
    CUDA libs are missing and _preload_cuda_assets() below handles those with
    ctypes.
    """
    accel = _build_accelerator()

    if accel == "metax":
        from torch_fl import _build_config

        # Native hosts link the official CPU core and submit through MACA's
        # runtime. Selecting a discoverable vendor core here would make that
        # contract depend on which package the application imports first.
        if getattr(_build_config, "NATIVE_HOSTS", False):
            return
        from torch_fl.accelerator.metax._metax_libtorch_link import (
            ensure_maca_libtorch_links,
        )

        ensure_maca_libtorch_links()
        return

    if accel == "hygon":
        # Decoupled by default: preload only DTK's device libraries on top of the
        # official core, leaving torch/lib untouched. FLAGOS_DCU_VENDOR_CORE=1
        # selects the full vendor core through a private facade. See
        # torch_fl/accelerator/hygon/_dcu_libtorch_link.py.
        from torch_fl.accelerator.hygon._dcu_libtorch_link import setup_dcu_runtime

        setup_dcu_runtime()
        return

    # PPU is its own accelerator now; an older PPU wheel reports "cuda" and is
    # recognised by its lib_thead/ bundle instead.
    if accel in ("nvidia", "", "thead"):
        from torch_fl.accelerator._vendor_libtorch import bundled_lib_dir

        if bundled_lib_dir("lib_thead", "libtorch_cuda.so"):
            from torch_fl.accelerator.thead._ppu_libtorch_link import (
                ensure_ppu_libtorch_links,
            )

            ensure_ppu_libtorch_links()


def _preload_cuda_assets() -> None:
    """Load the bundled CUDA .so into this process BEFORE `import torch`.

    Hard constraint (docs/vendors/cuda/external-libtorch-cuda.md, constraint 1):
    PyTorch caches its CUDAHooks on first `import torch`. If libtorch_cuda.so is
    loaded afterwards, device init fails with "Cannot initialize CUDA without
    ATen_cuda library" even though the kernels register. So we ctypes-dlopen it
    here, at the very top of torch_fl, before torch is imported.

    libtorch_cuda.so has unresolved deps on the NVIDIA runtime libs (libcudart,
    libcublas, libcudnn, libnvshmem_host, ...) shipped by the pip nvidia-*-cu12
    wheels. Since the process is already running, LD_LIBRARY_PATH cannot help;
    we must explicitly dlopen those deps (RTLD_GLOBAL) in dependency order first,
    then torch's own libc10/libtorch_cpu, then the CUDA libs.

    Skipped when:
      * FLAGOS_DISABLE_CUDA_ASSETS=1 (Ascend/MetaX/pure-CPU, or external preload)
      * the bundled libtorch_cuda.so is absent (e.g. slim build)
    """
    import ctypes
    import glob
    import importlib.util

    if _env.flag("FLAGOS_DISABLE_CUDA_ASSETS"):
        return

    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    main_cuda = os.path.join(lib_dir, "libtorch_cuda.so")
    if not os.path.exists(main_cuda):
        # No bundled assets; rely on an out-of-band preload (e.g. LD_PRELOAD via
        # scripts/vendor/with_nvidia_libtorch.sh) if the user set one up.
        return

    def _try(path, mode=ctypes.RTLD_GLOBAL):
        try:
            ctypes.CDLL(path, mode=mode)
            return True
        except OSError:
            return False

    # 1) NVIDIA runtime deps from pip nvidia-*-cu12 wheels. Locate their lib dirs
    #    via the installed `nvidia` namespace package (no torch import needed).
    nvidia_lib_dirs = []
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        for base in spec.submodule_search_locations:
            nvidia_lib_dirs.extend(sorted(glob.glob(os.path.join(base, "*", "lib"))))
    # Dependency order: cudart first (everything needs it), then the math/comm
    # libs, then nvshmem. Load by soname glob; ignore any that are absent.
    _dep_order = [
        "libcudart.so*",
        "libnvrtc.so*",
        "libnvjitlink.so*",
        "libnvJitLink.so*",
        "libcublasLt.so*",
        "libcublas.so*",
        "libcudnn*.so*",
        "libcufft.so*",
        "libcurand.so*",
        "libcusparse.so*",
        "libcusparseLt.so*",
        "libcusolver.so*",
        "libnccl.so*",
        "libcufile.so*",
        "libnvshmem_host.so*",
        "libnvToolsExt.so*",
        "libcupti.so*",
    ]
    for pattern in _dep_order:
        for d in nvidia_lib_dirs:
            for so in sorted(glob.glob(os.path.join(d, pattern))):
                _try(so)

    # 2) torch's own CPU libs (libtorch_cuda depends on libc10 / libtorch_cpu).
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and torch_spec.submodule_search_locations:
        torch_lib = os.path.join(list(torch_spec.submodule_search_locations)[0], "lib")
        for name in ("libc10.so", "libtorch_cpu.so"):
            _try(os.path.join(torch_lib, name))

    # 3) Bundled CUDA libs. Order: nvshmem/nvrtc helpers, libc10_cuda, then the
    #    big libtorch_cuda.so (which pulls linalg on demand via bare dlopen, so
    #    its dir must be resolvable -- it is, since we load from lib_dir).
    for name in (
        "libtorch_nvshmem.so",
        "libcaffe2_nvrtc.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
        # linalg ops dlopen this by bare soname on demand; preloading makes the
        # loaded copy satisfy that later bare-name dlopen.
        "libtorch_cuda_linalg.so",
    ):
        p = os.path.join(lib_dir, name)
        if os.path.exists(p):
            _try(p)


def _disable_vendor_backend_autoload() -> None:
    """Stop a vendor PrivateUse1 backend from claiming the key before flagos.

    torch_musa ships a `torch.backends` entry point, so a bare `import torch`
    autoloads it, and it calls rename_privateuse1_backend("musa") + registers the
    PrivateUse1 hooks/allocator. flagos wants that same single key, and PyTorch
    allows exactly one owner: our later rename raises "already been set!
    Current backend: musa".

    torch.__init__ honours TORCH_DEVICE_BACKEND_AUTOLOAD=0 to skip entry-point
    autoloading, so set it before `import torch`. Nothing of torch_musa is
    needed either way: the MUSA operator route calls mudnn, which is part of the
    MUSA toolkit and independent of the vendor's torch plugin.

    An explicit user setting always wins, so exporting
    TORCH_DEVICE_BACKEND_AUTOLOAD=1 restores stock torch_musa behaviour (useful
    for A/B testing against the vendor plugin, with torch_fl not imported).
    """
    if _build_accelerator() != "mthreads":
        return
    _env.set_foreign("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _validate_dcu_decoupled_runtime() -> None:
    """Prove the DTK device libs really bound to the official core.

    Only runs for a decoupled DCU build (a DCU wheel without
    FLAGOS_DCU_VENDOR_CORE=1), right after `import torch`; the checks themselves
    live in accelerator/hygon/_dcu_runtime_check.py. Set
    FLAGOS_DCU_SKIP_RUNTIME_CHECK=1 to bypass (e.g. when deliberately testing a
    non-matching wheel pair).
    """
    if _build_accelerator() != "hygon":
        return
    if _env.flag("FLAGOS_DCU_SKIP_RUNTIME_CHECK"):
        return

    from torch_fl.accelerator.hygon._dcu_libtorch_link import vendor_core_mode

    if vendor_core_mode():
        return  # legacy mode replaces the core wholesale; nothing to reconcile.

    from torch_fl.accelerator.hygon._dcu_runtime_check import validate_decoupled_runtime

    validate_decoupled_runtime(os.path.join(os.path.dirname(__file__), "lib_hygon"))


def _check_privateuse1_unclaimed() -> None:
    """Fail with an actionable message if a vendor plugin already took the key.

    PrivateUse1 admits exactly one backend name. `import torch` autoloads any
    `torch.backends` entry point -- torch_musa has one -- so when torch is
    imported before torch_fl, the name is already "musa" and our rename raises a
    bare "already been set!". _disable_vendor_backend_autoload only covers the
    torch_fl-first order, since by the time we run in the other order torch has
    already been initialised.
    """
    current = torch._C._get_privateuse1_backend_name()
    if current in ("privateuseone", "flagos"):
        return
    raise RuntimeError(
        f"PrivateUse1 is already claimed by the '{current}' backend, so torch_fl "
        "cannot register 'flagos'. A vendor plugin was autoloaded by `import "
        "torch` before torch_fl. Either import torch_fl first, or export "
        "TORCH_DEVICE_BACKEND_AUTOLOAD=0 before starting Python."
    )


def _keep_device_identity_checks_working(real_device, shim):
    """Repair the two torch registries that compare against ``torch.device``.

    Rebinding ``torch.device`` to a Python class (above) is invisible to almost
    everything -- ``isinstance`` still works, and every consumer that only
    *calls* it gets a genuine device back. Two places compare the attribute by
    identity instead, and both silently change behaviour:

    ``torch.fx.graph.add_global`` carves out ``obj != torch.device`` so that a
    device constant is emitted as the bare name ``device(type='cpu')`` and
    resolved through the ``device`` custom builtin. With the attribute swapped,
    the real device class no longer equals it, so the branch falls through to
    the qualified-name HACK path meant for custom ops -- the name is never added
    to the module's globals, and the generated forward dies with
    ``NameError: name 'device' is not defined`` the moment it runs. It is
    ``torch.arange(..., device=x.device)`` that emits such a constant, which is
    how every HF model builds its position ids, so this took out essentially
    every transformer under ``torch.compile``.

    ``torch._dynamo.utils.common_constant_types`` holds the types Dynamo may
    wrap as a ``ConstantVariable``, and it is tested with ``type(obj) in ...``.
    Reading ``tensor.device`` while tracing then asserts with "Cannot construct
    ``ConstantVariable`` for value of type ``torch.device``".

    Both registries are keyed on objects rather than rebuilt per call, so they
    can be corrected in place once, here. Re-running this is harmless.
    """
    from torch.fx.graph import _register_custom_builtin

    # Point the builtin at the shim: `add_global` skips the qualified-name path
    # for anything not defined under `torch`, so the shim reaches the normal
    # branch and the name lands in the generated module's globals. Calling it
    # still yields a real device, which is all the generated code needs.
    _register_custom_builtin("device", "from torch import device", shim)

    # Dynamo compares `type(obj)`, and a constructed device is always the real
    # C type whatever `torch.device` currently names -- so it is the real class
    # that has to be in the set.
    from torch._dynamo.utils import common_constant_types

    common_constant_types.add(real_device)


# Whether _alias_cuda_to_flagos has taken over torch.cuda, i.e. whether every
# "cuda" this process reports is the flagos device wearing a CUDA name. Read by
# torch_fl.compat.triton.stand_down_foreign_drivers. Stays False on a build
# with real CUDA, where cuda means cuda.
_cuda_alias_active = False


# Device strings that have to mean the flagos device on this build, keyed by the
# accelerator the build record holds -- the vendor, which is also what the
# lookup below passes. "cuda" is on every row: it is the name the ecosystem
# hardcodes.
# The second entry on a row is the vendor's *own* torch device name, which the
# vendor's Triton/FlagGems integration hands to torch.device but which is not
# registered here -- torch_fl claims PrivateUse1 as "flagos" and never imports
# the vendor package (torch_musa / torch_txda) that would register it.
#
#   mthreads   torch_musa's device, and FlagGems' own device.name.
#   tsingmicro FlagTree's TsingMicro Triton driver resolves its active device as
#              torch.device("txda", ...) (triton/backends/tsingmicro/driver.py),
#              and FlagGems' tsingmicro VendorDescriptor names the device "txda"
#              too. That call happens at *import* of flag_gems, from Triton's
#              hint manager, so leaving "txda" out does not merely mistype a
#              tensor: `import flag_gems` raises and every flaggems-routed op
#              on TsingMicro is unreachable.
#   cambricon  Triton's mlu driver builds its autotune scratch as
#              torch.empty(..., device='mlu') and reports the device interface
#              as torch.device("mlu", torch.mlu.current_device())
#              (triton/backends/mlu/driver.py). torch_mlu would have registered
#              the name; this build does not install torch_mlu, so the alias is
#              what keeps the benchmark step of every autotuned FlagGems kernel
#              from dying with "Expected one of cpu, cuda, ... device type at
#              start of device string: mlu". Deliberately NOT "cuda": MLU has no
#              CUDA runtime, and a flagos tensor here is cnrt memory, so
#              reinterpreting a device string as CUDA would be wrong rather than
#              merely unhelpful.
_VENDOR_DEVICE_ALIASES = {
    "mthreads": ("cuda", "musa"),
    "tsingmicro": ("cuda", "txda"),
    "cambricon": ("mlu",),
}

# The CUDA probe as it stands when the alias decision is made: torch's own
# answer, or the constant-True probe a CUDA-ABI vendor compat layer installed
# (`patch_torch_cuda_for_flagos` and the MetaX/DCU equivalents), and before
# `_alias_cuda_to_flagos` itself replaces the attribute. `_real_cuda_is_available`
# answers from this; see the note at the assignment in `_phase_ecosystem` for
# why neither the live attribute nor the `_flagos_original_is_available` stash
# can be read there.
_TORCH_CUDA_PROBE = None


def _real_cuda_is_available() -> bool:
    """Whether a genuine CUDA runtime is reachable through ``torch.cuda``.

    That is torch's own answer on a CUDA build, and the CUDA-ABI vendor compat
    layers' constant-True probe on a CPU torch wheel paired with the vendor's
    externally loaded ``libtorch_cuda.so``. It is *not* the answer once torch_fl
    itself owns the attribute: `_alias_cuda_to_flagos` writes
    ``lambda: flagos.device_count() > 0`` and `_patch_native_cuda_probe` writes
    ``torch.flagos.is_available``, both of which answer True on any build with a
    flagos device, whether or not a CUDA runtime exists.

    Reading the alias's own probe instead answers yes on exactly the builds the
    alias exists for. Measured on MUSA with the redirect in place and this guard
    reading it: nothing past the guard ran, ``torch.cuda.current_device()``,
    ``synchronize``, ``device_count`` and ``get_device_properties`` all stayed
    torch's, and dynamo's ``cuda_extra_check`` -- which calls
    ``torch.cuda.current_device()`` -- landed in ``torch.cuda._lazy_init`` and
    raised ``AssertionError: Torch not compiled with CUDA enabled``, so
    ``torch.compile(model, fullgraph=True)`` could not compile at all.

    The answer is captured while it still means that, not read live: by the time
    anything else asks, torch_fl may own the attribute.
    """
    return bool(_TORCH_CUDA_PROBE())


# DIAGNOSTIC hook, temporary: see `_capability_diagnostic`. Read once, so one
# process prints one stack.
_CAPABILITY_DIAGNOSED = False


def _capability_diagnostic(original):
    """Wrap `torch.cuda.get_device_capability` so a raise explains itself.

    Enflame GCU is the lane that fails here: `torch.compile` dies in
    `torch.utils._triton._device_supports_tma` with "Torch not compiled with CUDA
    enabled", and neither the log nor a simulated `gcu` build on another machine
    says which binding of the getter was reached. The wrapper answers exactly as
    the original does; on the first raise it prints the call stack and the state
    of the probes that can produce this error, then re-raises.

    Remove once the GCU lane is green.
    """

    @functools.wraps(original)
    def capability_with_evidence(device=None):
        global _CAPABILITY_DIAGNOSED
        try:
            return original(device)
        except AssertionError as exc:
            if not _CAPABILITY_DIAGNOSED:
                _CAPABILITY_DIAGNOSED = True
                import sys
                import traceback

                print("[gcu-diag] get_device_capability raised:", exc, file=sys.stderr)
                traceback.print_stack(file=sys.stderr)
                print(
                    "[gcu-diag] alias=%s gcu_attr=%s lazy_init=%s props=%s "
                    "is_available=%s torch.module=%s torch.cuda.module=%s "
                    "torch_fl._C=%s"
                    % (
                        _cuda_alias_active,
                        hasattr(torch, "gcu"),
                        getattr(torch.cuda._lazy_init, "__name__", "?"),
                        getattr(torch.cuda.get_device_properties, "__name__", "?"),
                        torch.cuda.is_available(),
                        getattr(sys.modules.get("torch"), "__file__", None),
                        getattr(sys.modules.get("torch.cuda"), "__file__", None),
                        getattr(sys.modules.get("torch_fl._C"), "__name__", None),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            raise

    capability_with_evidence._flagos_diagnostic = True
    return capability_with_evidence


def _install_capability_diagnostic() -> None:
    """Install the wrapper once, before anything can call the stock getter.

    Called at the end of `_phase_claim`, which is before `_phase_vendor_compat`
    and before the first `import flag_gems`; the alias runs later and would be
    too late if FlagGems were imported during the vendor phase. Independent of
    the alias's own early returns, so `FLAGOS_ALIAS_CUDA=0` still diagnoses.
    """
    if torch.cuda.get_device_capability is _STOCK_DEVICE_CAPABILITY:
        torch.cuda.get_device_capability = _capability_diagnostic(
            _STOCK_DEVICE_CAPABILITY
        )


def _alias_cuda_to_flagos():
    """Make ``device="cuda"`` mean the flagos device when there is no real CUDA.

    Most of the PyTorch ecosystem hardcodes ``"cuda"``: ``model.cuda()``,
    ``device_map="cuda"``, ``torch.device("cuda")`` in example scripts, and
    ``torch.cuda.is_available()`` as the "do I have an accelerator" test. On a
    vendor backend built without CUDA (Ascend), every one of those raises
    ``AssertionError: Torch not compiled with CUDA enabled`` -- so code that runs
    unmodified elsewhere has to be edited to say ``"flagos"``.

    This rewrites ``cuda`` device *arguments* to the flagos device, so that
    hardcoded-``cuda`` code lands on the accelerator that is actually present.

    Deliberately a no-op when *torch* reports a CUDA device: on the CUDA and
    boxing backends ``cuda`` already means a real device, and hijacking it there
    would break the boxing path, which submits genuine CUDA work. Which is not
    the same question as whether ``torch.cuda.is_available()`` says so --
    torch_fl redirects that probe on a build without a CUDA runtime, and reading
    the redirect here is what kept this alias uninstalled; see
    ``_real_cuda_is_available``.

    Opt out with ``FLAGOS_ALIAS_CUDA=0`` -- worth doing if you need
    ``device="cuda"`` to keep failing loudly rather than silently redirecting.
    """
    if _real_cuda_is_available():
        return
    if not _env.flag("FLAGOS_ALIAS_CUDA", True):
        return

    # From here on, every "cuda" this process sees is this alias. Anything that
    # treats torch.cuda as a hardware probe has to be told, because it can no
    # longer distinguish the accelerator from a CUDA device that does not exist;
    # see torch_fl.compat.triton.stand_down_foreign_drivers.
    global _cuda_alias_active
    _cuda_alias_active = True

    from torch.overrides import TorchFunctionMode

    _flagos_type = torch._C._get_privateuse1_backend_name()  # "flagos"
    _orig_device = torch.device

    def _remap(dev):
        """Vendor accelerator aliases -> flagos; everything else untouched."""
        if isinstance(dev, str):
            aliases = _VENDOR_DEVICE_ALIASES.get(_build_accelerator(), ("cuda",))
            for alias in aliases:
                if dev == alias:
                    return _flagos_type
                if dev.startswith(f"{alias}:"):
                    return f"{_flagos_type}:{dev[len(alias) + 1 :]}"
            return dev
        if isinstance(dev, _orig_device) and dev.type == "cuda":
            # `dev.index` is passed through even when it is None, which is the
            # whole point: an index-less target means "the current device", and
            # torch resolves that through the privateuse1 device guard, honouring
            # `set_device`. Forcing 0 here instead sent every rank of a
            # multi-rank job to flagos:0 -- and only through the `torch.device`
            # object, because the str branch above already leaves the index to
            # torch. `nn.Module.to("cuda")` reaches this branch, since
            # `_parse_to` turns its argument into a device object first.
            return _orig_device(_flagos_type, dev.index)
        return dev

    # A TorchFunctionMode, not a wrapper around torch.device.
    #
    # Wrapping torch.device is not enough and was tried first: factory functions
    # parse their `device=` argument in C++ (THPDevice / the argument parser), so
    # `torch.randn(4, device="cuda")` never reaches a Python torch.device call and
    # still dies in torch.cuda._lazy_init. A torch-function mode sits above the
    # C++ parser and sees the keyword before it is resolved, which catches every
    # factory uniformly -- randn, zeros, empty, tensor, arange, and `.to()`.
    #
    # Pushed permanently onto the mode stack at import. That is unusual but
    # intended: the alias has to hold for the whole process, not a `with` block.
    # `__torch_function__` runs on every op, so the body is kept to a dict lookup
    # and a `str.startswith` on the miss path.
    class _CudaAliasMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            dev = kwargs.get("device")
            if dev is not None:
                remapped = _remap(dev)
                if remapped is not dev:
                    kwargs = {**kwargs, "device": remapped}
            # Positional device, as in `Tensor.to("cuda")`.
            elif args and func is torch.Tensor.to:
                remapped = _remap(args[1]) if len(args) > 1 else None
                if remapped is not None and len(args) > 1 and remapped is not args[1]:
                    args = (args[0], remapped) + args[2:]
            return func(*args, **kwargs)

    torch._C._push_on_torch_function_stack(_CudaAliasMode())

    # torch.device("cuda") itself, for code that builds the device object first
    # and only later passes it to a factory (transformers' device_map does this).
    # torch.device is a C type and cannot be subclassed, so wrap the constructor;
    # isinstance(x, torch.device) must keep working, hence __instancecheck__.
    class _DeviceMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, _orig_device)

    class device(metaclass=_DeviceMeta):  # noqa: N801  (mirrors torch.device)
        def __new__(cls, *args, **kwargs):
            if args:
                args = (_remap(args[0]),) + args[1:]
            elif "device" in kwargs:
                kwargs = {**kwargs, "device": _remap(kwargs["device"])}
            return _orig_device(*args, **kwargs)

    # Pickle saves a *class* by reference, looking it up as
    # ``__module__.__qualname__``. Left as defined, that is
    # ``torch_fl._alias_cuda_to_flagos.<locals>.device``, which is unreachable,
    # so ``pickle.dumps(torch.device)`` fails with "Can't get local object".
    # That costs more than hand-pickling a device: Inductor pickles the graph
    # module to build its FX graph cache key, and a device constant in the graph
    # brings this class along, so every compile hits BypassFxGraphCache ("Failed
    # to pickle cache key") and recompiles from scratch. Naming it ``torch.device``
    # -- which is exactly what the next line makes it -- makes that lookup find
    # this same object, so the reference round-trips.
    device.__module__ = "torch"
    device.__qualname__ = "device"
    device.__name__ = "device"

    torch.device = device
    _keep_device_identity_checks_working(_orig_device, device)

    # Tensor.cuda() / Module.cuda() -> the flagos device.
    def _tensor_cuda(self, device=None, non_blocking=False, **kwargs):
        # `Tensor.cuda()` with no argument means the current device on CUDA, so
        # it has to mean the current device here too: an index-less target is
        # left index-less and resolved by torch, the same way the `_remap` str
        # branch does it. Pinning 0 instead put every rank of a multi-rank job
        # on flagos:0 and NCCL rejected the duplicate device.
        target = _flagos_type
        if device is not None:
            index = _orig_device(_remap(device)).index
            if index is not None:
                target = f"{_flagos_type}:{index}"
        return self.to(target, non_blocking=non_blocking)

    torch.Tensor.cuda = _tensor_cuda

    # `torch.cuda.is_available()` is the ecosystem's "have I got an accelerator"
    # probe, and gating on it is what sends code down the CPU path. Report the
    # flagos device count so that probe finds the accelerator that is there.
    #
    # Left alone: is_bf16_supported, get_device_capability, and the rest of
    # torch.cuda -- the vendor shim already owns those, and overriding them here
    # would fight it.
    torch.cuda.is_available = lambda: flagos.device_count() > 0
    torch.cuda.device_count = flagos.device_count
    torch.cuda.current_device = flagos.current_device
    torch.cuda.set_device = flagos.set_device
    torch.cuda.synchronize = flagos.synchronize
    # Once is_available() says yes, dynamo's CudaInterface.get_device_properties
    # is called for every device while building compilation metrics, and the
    # stock one goes through torch.cuda._lazy_init -> "Torch not compiled with
    # CUDA enabled". Point it at the flagos properties instead.
    torch.cuda.get_device_properties = flagos.get_device_properties


# Python registration is empty because operators use the C++ dispatcher.
_registered_ops = []


def get_registered_ops():
    """Return list of registered FlagGems operators for flagos device."""
    return list(_registered_ops)


def is_flaggems_enabled():
    """Check if FlagGems operators are registered for flagos device."""
    return len(_registered_ops) > 0


def fused_kernels_available(device=None) -> bool:
    """Whether a fused Triton kernel can be compiled for this device.

    The predicate dispatch code should gate on instead of ``tensor.is_cuda``.
    ``is_cuda`` is False on flagos, so a caller choosing between
    ``torch.compile(flex_attention)`` and eager ``flex_attention`` by hand takes
    the eager branch and silently materializes the full scores matrix -- correct,
    and several times slower (issue #515).

    This asks the question the caller means: can inductor generate a Triton
    kernel for a tensor on this device, here, now. It is deliberately *not*
    ``optional_integration_status()["compile"] == "active"``: that state records
    only that the backend-registration imports succeeded, and nothing on that
    path probes Triton, so it reports ``active`` on a build whose vendor Triton
    backend is missing. ``raise_if_triton_unavailable`` is the honest check --
    it applies the CUDA compute-capability floor where one applies and otherwise
    lets the vendor backend decide -- so that is what runs here, with its
    exception turned into the boolean a branch needs.

    A False answer is a statement about this build and device, not about the
    operator: the eager path still computes the right result.
    """
    if _optional_integration_state.get("compile") != "active":
        return False
    try:
        from torch_fl.compile.device_interface import FlagOSDeviceInterface

        if not FlagOSDeviceInterface.is_available():
            return False
        FlagOSDeviceInterface.raise_if_triton_unavailable(device)
    except Exception:  # noqa: BLE001 - any failure to prove it means "no"
        return False
    return True


def device_capabilities(device=None) -> dict:
    """Capability flags for this build, for code that cannot import torch_fl APIs.

    One dict so a downstream framework can read every gate it needs in a single
    call, rather than discovering one predicate at a time. Keys are stable; new
    ones may be added, so read by key and do not unpack.
    """
    return {
        "compile": _optional_integration_state.get("compile") == "active",
        "fused_kernels": fused_kernels_available(device),
        "fused_attention": (
            fused_kernels_available(device)
            and _optional_integration_state.get("flex_attention") == "active"
        ),
        "flaggems": is_flaggems_enabled(),
    }


# ---------------------------------------------------------------------------
# Distributed: register "flagos" ProcessGroup backend for privateuseone
# ---------------------------------------------------------------------------


def _register_distributed_backend():
    """Register ProcessGroupFlagOS as the 'flagos' torch.distributed backend.

    After this call:
      - ``torch.distributed.init_process_group("flagos")`` works
      - ``torch.distributed.init_process_group(
            device_id=torch.device("privateuseone:0"))`` auto-selects "flagos"
      - All ``torch.distributed.*`` collectives work on flagos tensors without
        any monkeypatching — the ProcessGroup itself does the view conversion.
    """
    from torch_fl.comm import register_flagos_backend

    register_flagos_backend()


# ---------------------------------------------------------------------------
# DDP auto-patch: torch.nn.parallel.DistributedDataParallel
# ---------------------------------------------------------------------------


def _patch_ddp_for_flagos():
    """Patch DDP to transparently support flagos (privateuseone) models.

    PyTorch's C++ Reducer has CUDA-specific assertions that fail for
    privateuseone tensors. This patch detects when the wrapped module lives on
    a flagos device and transparently:
      1. Forces ``python_reducer`` mode (bypasses the C++ Reducer).
      2. Replaces default accum-grad hooks with flagos-compatible ones that
         call ``dist.all_reduce`` (routed through ProcessGroupFlagOS).

    Users call standard DDP — no ``flagos_dist.DistributedDataParallel`` needed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    """
    import functools
    import torch.distributed as _dist
    from torch.nn.parallel import DistributedDataParallel as _DDP

    _orig_init = _DDP.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, module, **kwargs):
        # torch_fl renames PrivateUse1 to "flagos", so parameter tensors report
        # device.type == "flagos" (not the raw "privateuseone").
        device_types = {p.device.type for p in module.parameters()}
        if not device_types & {"flagos", "privateuseone"}:
            return _orig_init(self, module, **kwargs)

        # Force python_reducer to avoid C++ Reducer CUDA assertions
        import torch._dynamo.utils

        _orig_mode = torch._dynamo.utils.get_optimize_ddp_mode
        torch._dynamo.utils.get_optimize_ddp_mode = lambda: "python_reducer"
        try:
            kwargs.setdefault("gradient_as_bucket_view", True)
            kwargs.setdefault("broadcast_buffers", False)
            _orig_init(self, module, **kwargs)
        finally:
            torch._dynamo.utils.get_optimize_ddp_mode = _orig_mode

        # Replace DDP's python_reducer accum hooks. The stock hooks use
        # torch.distributed._functional_collectives (torch.ops._c10d_functional.*),
        # whose dispatcher path is not registered for privateuseone. We instead
        # go through dist.all_reduce on the group, which routes to
        # ProcessGroupFlagOS and does the privateuseone->cuda view conversion.
        # Mirrors DDP.compiled_accum_grad_hook, including _comm_hooks support.
        for h in self._accum_grad_hooks:
            h.remove()
        self._accum_grad_hooks.clear()

        def _accum_grad_hook(param, *, ddp_model=self):
            if not ddp_model.require_backward_grad_sync:
                return
            if param.grad is None:
                return
            pg = ddp_model.process_group
            if ddp_model._comm_hooks:
                for hook, state in ddp_model._comm_hooks:
                    hook(state, (param.grad, param))
            else:
                param.grad.div_(pg.size())
                _dist.all_reduce(param.grad, op=_dist.ReduceOp.SUM, group=pg)

        for param in self._module_parameters:
            if param.requires_grad:
                self._accum_grad_hooks.append(
                    param.register_post_accumulate_grad_hook(
                        functools.partial(_accum_grad_hook, ddp_model=self)
                    )
                )

    _DDP.__init__ = _patched_init


# ---------------------------------------------------------------------------
# DataParallel auto-patch: torch.nn.parallel.DataParallel and its comm layer
# ---------------------------------------------------------------------------

# Both spellings name the same device: the claim phase renames PrivateUse1 to
# "flagos", so tensors created afterwards report "flagos", while
# "privateuseone" is the raw name a tensor can still carry from before the
# rename (or from a wheel whose rename never ran).
_FLAGOS_DEVICE_TYPES = ("flagos", "privateuseone")


def _flagos_device_type_of_tensors(tensors):
    """Device type of the first flagos device among ``tensors``, else None."""
    for tensor in tensors:
        if tensor.device.type in _FLAGOS_DEVICE_TYPES:
            return tensor.device.type
    return None


def _flagos_module_device_type(module):
    """Device type a module is placed on if it is a flagos one, else None.

    Parameters and buffers both: DataParallel's own device guard walks the two
    together, and a module can hold buffers with no parameters.
    """
    for group in (module.parameters(), module.buffers()):
        device_type = _flagos_device_type_of_tensors(group)
        if device_type is not None:
            return device_type
    return None


def _patch_comm_for_flagos():
    """Hand ``torch.nn.parallel.comm``'s device moves to the flagos ops.

    ``comm.scatter``, ``comm.gather`` and ``comm.broadcast_coalesced`` -- and
    the ``_out`` forms DataParallel reaches through their ``out=`` argument --
    are thin validators over seven CUDA-only ops on ``torch._C`` (``_scatter``,
    ``_scatter_out``, ``_gather``, ``_gather_out``, ``_broadcast``,
    ``_broadcast_out``, ``_broadcast_coalesced``). Each reads the caller's
    device list as a list of *CUDA* indices whatever the tensors are, so on a
    flagos build it either mislabels the result or refuses outright -- measured
    on PPU:

        torch._C._scatter(flagos:0 tensor, [0, 1], ...) -> [flagos:0, cuda:1]
        torch._C._gather([flagos:0, flagos:1], 0, 0)    -> RuntimeError:
            "Expected all input tensors to be CUDA tensors, but tensor at
             index 0 has device flagos:0"

    The flagos implementations live in the extension
    (csrc/bindings/dataparallel.cc) and are published by rebinding those
    seven attributes on ``torch._C``, which is how torch_npu reaches the same
    symbols (its ``initCommMethods()``). They read and write flagos tensors the
    way the CUDA ones read and write CUDA ones, and each delegates to the
    original it replaced as soon as no flagos tensor is involved, so CUDA and
    CPU keep the stock code path. Nothing in comm's Python is replaced: the
    validation, the ``_handle_complex`` handling and the device index
    resolution DataParallel already relies on all stay torch's.

    Rebinding is idempotent on the extension side, so calling this more than
    once is harmless.
    """
    from torch_fl import _C

    _C._init_dataparallel_comm()


def _patch_dataparallel_for_flagos():
    """Make ``torch.nn.DataParallel`` place its replicas on flagos devices.

    Three device decisions in DataParallel are CUDA-only, and a flagos-placed
    module trips all three:

      * the device *type* comes from ``torch._utils._get_available_device_type()``,
        which answers "cuda" before it ever asks privateuse1. So on PPU and MetaX
        (where ``torch.cuda.is_available()`` is shimmed True) a module on
        ``flagos:0`` is given ``src_device_obj = torch.device("cuda", 0)`` and
        then fails DataParallel's own guard in forward with "module must have its
        parameters and buffers on device cuda:0 (device_ids[0]) but found one of
        them on device: flagos:0";
      * the default device list is ``_get_all_device_indices()``, i.e. torch.cuda's
        device count, which on a stock +cpu torch is 0;
      * ``_check_balance`` is CUDA-only, and on MetaX it is also what moves the
        current device: ``_query_metax_device_properties`` calls ``mcSetDevice``
        per device and never restores, so after the probe loop every later
        operation meant for device 0 runs on the last device probed.

    A flagos-placed module therefore builds its own state: devices counted and
    typed from the flagos device module, no balance probe. Everything after
    construction is DataParallel's own code -- scatter, replicate,
    parallel_apply, gather -- and works once ``torch.nn.parallel.comm`` has
    flagos branches, which is what ``_patch_comm_for_flagos`` installs.

    A module that is not on a flagos device goes to the original ``__init__``
    unchanged, so CUDA and CPU behavior is untouched.
    """
    import functools

    from torch._utils import _get_device_index
    from torch.nn.parallel import DataParallel as _DataParallel

    _orig_init = _DataParallel.__init__
    _orig_scatter = _DataParallel.scatter

    @functools.wraps(_orig_init)
    def _patched_init(self, module, device_ids=None, output_device=None, dim=0):
        device_type = _flagos_module_device_type(module)
        if device_type is None:
            return _orig_init(self, module, device_ids, output_device, dim)

        # Module.__init__ has to run before anything is assigned to self: it
        # installs the __setattr__ machinery that every `self.x = ...` below goes
        # through. The original __init__ is not called, so this replaces the
        # super().__init__() at its top.
        torch.nn.Module.__init__(self)
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")

        if device_ids is None:
            # The flagos device module rather than torch.cuda: whatever
            # torch.cuda reports on this host is unrelated to the devices the
            # module's parameters live on. torch.flagos is the fallback for the
            # raw "privateuseone" spelling, whose own attribute does not exist.
            device_ids = range(getattr(torch, device_type, torch.flagos).device_count())
        if len(device_ids) == 0:
            raise RuntimeError("no available devices were found")
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])
        # The marker DataParallel.scatter reads below to decide whether the
        # device ids it hands to comm.scatter are flagos ones.
        self._flagos_device_type = device_type

        # No _check_balance: it is the CUDA memory/cores warning, and on MetaX
        # running it is a side effect on the current device.
        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    @functools.wraps(_orig_scatter)
    def _patched_scatter(self, inputs, kwargs, device_ids):
        if getattr(self, "_flagos_device_type", None) is None:
            return _orig_scatter(self, inputs, kwargs, device_ids)
        # Tells torch._C._scatter that the integer device ids it is about to
        # receive name flagos devices. Only load-bearing for a CPU input, which
        # carries no device type of its own; see SetScatterScope in
        # csrc/bindings/dataparallel.h for why the ids alone are ambiguous
        # here.
        from torch_fl import _C

        previous = _C._set_scatter_scope(True)
        try:
            return _orig_scatter(self, inputs, kwargs, device_ids)
        finally:
            _C._set_scatter_scope(previous)

    _DataParallel.__init__ = _patched_init
    _DataParallel.scatter = _patched_scatter


def _patch_data_parallel_for_flagos():
    """Give the functional ``torch.nn.parallel.data_parallel()`` flagos support.

    It is the class's forward path with the device resolution restated inline,
    so it fails a flagos-placed module the same way (and additionally raises
    "device type could not be determined" on a build where nothing reports
    availability). For a flagos-placed module it is delegated to the patched
    class instead of restating torch's logic a second time, which keeps the two
    from drifting. Two consequences of that delegation, both deliberate:
    ``_check_balance`` is skipped where the class would also skip it, and a
    single-device call now ends with ``module.to(device_ids[0])``, which the
    class does in ``__init__`` and the function does not -- a no-op for a module
    already on the device it is being run on.
    """
    import functools

    from torch.nn.parallel import DataParallel as _DataParallel
    from torch.nn.parallel.data_parallel import data_parallel as _data_parallel

    @functools.wraps(_data_parallel)
    def _patched_data_parallel(
        module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None
    ):
        if _flagos_module_device_type(module) is None:
            return _data_parallel(
                module, inputs, device_ids, output_device, dim, module_kwargs
            )
        if not isinstance(inputs, tuple):
            inputs = (inputs,) if inputs is not None else ()
        return _DataParallel(module, device_ids, output_device, dim)(
            *inputs, **(module_kwargs or {})
        )

    # Both spellings of the name: torch.nn.parallel re-exports the function, and
    # the module that defines it is still reachable by path.
    torch.nn.parallel.data_parallel = _patched_data_parallel
    sys.modules[
        "torch.nn.parallel.data_parallel"
    ].data_parallel = _patched_data_parallel


# Register torch.compile backend for flagos device (torch 2.0+)
def _register_compile_backend():
    """Register the 'flagos' backend with torch._dynamo if available."""
    try:
        from torch_fl.compile.inductor_backend import register_backend

        register_backend()

        # Also wire flagos into inductor *eagerly*, so the default
        # `torch.compile` backend (`backend="inductor"`) works on flagos too.
        # transformers' `CompileConfig` compiles with `backend="inductor"`
        # rather than our `"flagos"` backend; without this, `backend="inductor"`
        # lowers e.g. RMSNorm (`aten.mean.dim`) and trips
        # `get_backend_features("flagos") -> assert scheduling_ctor`
        # (torch/_inductor/codegen/common.py:460) because flagos was never
        # registered in inductor's codegen table. These patches are idempotent
        # and mirror what `flagos_compile_backend` runs before every
        # compile_fx.
        from torch_fl.compile.device_interface import register_flagos_device_interface
        from torch_fl.compile.inductor_codegen import (
            publish_codegen_on_device_module,
            register_flagos_codegen,
        )
        from torch_fl.compile.triton_libdevice import (
            patch_triton_libdevice_module_map,
        )
        from torch_fl.compile.triton_byte_loads import (
            patch_triton_byte_load_workarounds,
        )
        from torch_fl.compile.triton_resource_limits import (
            patch_triton_resource_limit_errors,
        )
        from torch_fl.compile.triton_64bit_guard import patch_triton_64bit_guard

        register_flagos_device_interface()
        publish_codegen_on_device_module()
        register_flagos_codegen()
        patch_triton_libdevice_module_map()
        patch_triton_resource_limit_errors()
        patch_triton_byte_load_workarounds()

        # The default `backend="inductor"` path bypasses `flagos_compile_backend`
        # entirely, and a GCU300 64-bit kernel kills the process from inside the
        # vendor pass manager where no Python handler can reach it. Install the
        # guard here, not only on the flagos backend.
        patch_triton_64bit_guard()
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("torch.compile integration is unavailable") from exc


def _register_bpu_compile_backend() -> None:
    """Register torch.compile(backend="bpu") on a BPU build.

    The RDK BPU executes whole compiled graphs (a .hbm produced by hbdk4), not
    individual operators, so it has no per-op kernels: eager ops reach
    cpu_fallback and all acceleration comes through this backend. That is the
    opposite of every other platform here, where the compile path is incidental
    and the kernels do the work.

    The optional integration runner reports import failures independently. A
    board without the ONNX toolchain can still use its eager fallback.
    """
    if _build_accelerator() != "bpu":
        raise RuntimeError("BPU compile backend requires a BPU build")
    from torch_fl.accelerator import bpu

    bpu.register()


_OPTIONAL_INTEGRATIONS = (
    "apex",
    "ddp",
    "parallel_comm",
    "dataparallel",
    "data_parallel",
    "compile",
    "flex_attention",
    "bpu",
)
_optional_integration_state = {}


def _install_apex_compat() -> bool:
    from torch_fl.compat.apex import install_apex_compat

    return install_apex_compat()


def _install_flex_attention_compat() -> bool:
    from torch_fl.compat.flex_attention import install_flex_attention_compat

    return install_flex_attention_compat()


def optional_integration_status() -> dict:
    """Report each optional hook as inactive, active, or failed with a reason."""
    return {
        name: _optional_integration_state.get(name, "inactive")
        for name in _OPTIONAL_INTEGRATIONS
    }


def activate_optional_integrations(*names: str, strict: bool = True) -> dict:
    """Install optional framework hooks after the mandatory device bootstrap.

    Repeated calls do not wrap PyTorch objects twice. A failed hook is not
    retried in the same process because it may have partly changed global
    state. ``strict=False`` reports failures and continues with other hooks;
    the import-time compatibility profile uses that mode. Explicit calls raise
    so applications know when a requested integration was unavailable.
    """
    selected = names or tuple(
        name
        for name in _OPTIONAL_INTEGRATIONS
        if name != "bpu" or _build_accelerator() == "bpu"
    )
    unknown = sorted(set(selected) - set(_OPTIONAL_INTEGRATIONS))
    if unknown:
        raise ValueError(f"Unknown torch_fl integration(s): {', '.join(unknown)}")

    installers = {
        "apex": _install_apex_compat,
        "ddp": _patch_ddp_for_flagos,
        "parallel_comm": _patch_comm_for_flagos,
        "dataparallel": _patch_dataparallel_for_flagos,
        "data_parallel": _patch_data_parallel_for_flagos,
        "compile": _register_compile_backend,
        "flex_attention": _install_flex_attention_compat,
        "bpu": _register_bpu_compile_backend,
    }
    prerequisites = {
        "dataparallel": ("parallel_comm",),
        "data_parallel": ("parallel_comm", "dataparallel"),
        "flex_attention": ("compile",),
    }

    def activate(name: str) -> None:
        status = _optional_integration_state.get(name)
        if status == "active":
            return
        if status is not None:
            if strict:
                raise RuntimeError(f"torch_fl integration {name} previously {status}")
            return
        for prerequisite in prerequisites.get(name, ()):
            activate(prerequisite)
            if _optional_integration_state.get(prerequisite) != "active":
                message = f"failed: prerequisite {prerequisite} is unavailable"
                _optional_integration_state[name] = message
                if strict:
                    raise RuntimeError(f"torch_fl integration {name} {message}")
                _env.warn(f"Optional integration {name} {message}")
                return
        try:
            installed = installers[name]()
        except Exception as exc:  # noqa: BLE001 - independent optional hooks
            message = f"failed: {type(exc).__name__}: {exc}"
            _optional_integration_state[name] = message
            if strict:
                raise RuntimeError(f"torch_fl integration {name} {message}") from exc
            _env.warn(f"Optional integration {name} {message}")
        else:
            if installed is False:
                if strict:
                    raise RuntimeError(f"torch_fl integration {name} is unavailable")
            else:
                _optional_integration_state[name] = "active"

    for name in selected:
        activate(name)
    return optional_integration_status()


def _phase_conf() -> None:
    """Pick the op-routing conf and stage the MetaX cudart shim.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    global _STARTUP_PROFILE
    _STARTUP_PROFILE = _env.choice(
        "FLAGOS_STARTUP_PROFILE", ("full", "minimal"), "full"
    )
    _select_backend_config()

    # Optional: PyTorch wheels may require libcudart.so.12 version tags on MetaX.
    if _env.flag("FLAGOS_METAX_CUDART_SHIM"):
        from torch_fl.accelerator.metax._metax_cudart_shim import ensure_cudart_shim

        ensure_cudart_shim()


def _phase_preload() -> None:
    """Select/preload the vendor libtorch and CUDA assets before `import torch`.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    _relink_vendor_libtorch()

    _preload_cuda_assets()
    _disable_vendor_backend_autoload()


# The stock torch.cuda.get_device_capability, captured before anything wraps it;
# the diagnostic in `_install_capability_diagnostic` keys on this identity.
_STOCK_DEVICE_CAPABILITY = None


def _phase_claim() -> None:
    """Import torch, claim PrivateUse1, load _C and install the device module.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    global torch, flagos
    import torch  # noqa: E402

    # Immediately after `import torch`, and before anything relies on CUDA dispatch:
    # confirm the DTK device libraries actually bound to the official core.
    _validate_dcu_decoupled_runtime()

    # A self-contained PPU build may front its bundled CUDA-enabled libtorch with
    # the official torch+cpu Python wheel. The actual runtime then supports CUDA
    # dispatch while torch/version.py still reports cuda=None. Restore that build
    # metadata before optional packages inspect it and select a native library.
    if _is_ppu_build():
        from torch_fl.accelerator.thead._ppu_libtorch_link import (  # noqa: E402
            restore_ppu_cuda_version,
        )

        restore_ppu_cuda_version()

    if sys.platform == "win32":
        from ._utils import _load_dll_libraries

        _load_dll_libraries()
        del _load_dll_libraries

    # Optional FlagGems-on-MetaX compat (does not patch torch.cuda unless enabled).
    if _env.flag("FLAGOS_METAX_COMPAT"):
        from .compat.flaggems.device import install_metax_cuda_shim

        install_metax_cuda_shim()

    # Expose libtorch symbols globally so the Ascend Triton backend's JIT-compiled
    # launcher .so can resolve c10/ATen symbols (it links implicitly, not via
    # DT_NEEDED). Applies to both FlagTree and the legacy triton-ascend toolchain.
    import os as _os  # noqa: E402

    _torch_lib = _os.path.join(_os.path.dirname(torch.__file__), "lib")
    for _lib in ("libc10.so", "libtorch.so", "libtorch_cpu.so"):
        _p = _os.path.join(_torch_lib, _lib)
        if _os.path.exists(_p):
            ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

    # Load libstream_api.so with RTLD_GLOBAL so that liboperators.so (FlagGems)
    # can resolve GetCurrentStream at runtime.
    _stream_api_path = _os.path.join(
        _os.path.dirname(__file__), "lib", "libstream_api.so"
    )
    if _os.path.exists(_stream_api_path):
        ctypes.CDLL(_stream_api_path, mode=ctypes.RTLD_GLOBAL)

    # Checked *before* loading _C, not just before the rename: libtorch_fl.so
    # registers the AutogradPrivateUse1 fallback at dlopen time, and a vendor plugin
    # that already registered one makes that a std::terminate ("Tried to register
    # multiple backend fallbacks for the same dispatch key") -- an abort we cannot
    # catch or report. Running the check first turns that into the actionable
    # message below.
    _check_privateuse1_unclaimed()

    import torch_fl._C  # type: ignore[misc]  # noqa: E402, F401

    # Hand the conf _select_backend_config() resolved to the C++ reader, which builds
    # the routing table on the first op dispatch -- still well after this point.
    # This is a call rather than an os.environ write so the wheel's own choice stays
    # distinguishable from the user's FLAGOS_BACKEND_CONFIG; see
    # backend_config_path(). Nothing to hand over when the user set the variable or
    # nothing was found, in which case the C++ reader resolves it itself.
    if _BACKEND_CONFIG_PATH:
        torch_fl._C._set_backend_config_path(_BACKEND_CONFIG_PATH)

    from . import flagos  # noqa: E402

    if _build_accelerator() == "apple":
        # Finish the allocator before torch's module teardown clears Storage
        # objects. A DataPtr can then be deleted without the caching allocator
        # reinitialising during static destruction.
        import atexit

        atexit.register(_mps_shutdown_allocator)

    torch.utils.rename_privateuse1_backend("flagos")
    torch._register_device_module("flagos", flagos)
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

    if _build_accelerator() == "enflame":
        from .compat.flaggems.device import install_gcu_device_probes

        install_gcu_device_probes(flagos)
        # `has_triton` has to answer for the FlagTree enflame backend before
        # inductor's `create_backend` asks (it raises `TritonMissing` otherwise,
        # measured on this lane). `_prime_has_triton` is the routine that makes
        # it answer -- it borrows a table row and memoizes True -- and it is
        # normally first reached from `register_flagos_device_interface`, which
        # only runs when the compile backend starts, i.e. too late here.
        from torch_fl.compile.device_interface import _prime_has_triton

        _prime_has_triton()

    # DIAGNOSTIC, temporary: see `_capability_diagnostic`. Installed here rather
    # than in the alias because `_phase_vendor_compat` and the first
    # `import flag_gems` can both reach `torch.cuda.get_device_capability`
    # before `_phase_ecosystem` runs the alias.
    global _STOCK_DEVICE_CAPABILITY
    _STOCK_DEVICE_CAPABILITY = torch.cuda.get_device_capability
    _install_capability_diagnostic()


def _phase_vendor_compat() -> None:
    """Install the vendor runtime shims and resolve GEMS_VENDOR (fail-loud).

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    from . import native_host
    from .compat.flaggems import device, runtime

    native_host._bootstrap_launcher()
    device.install_device_shims(_build_accelerator(), flagos)

    # torch::utils::device_lazy_init(PrivateUse1) imports the module named
    # `torch_<backend_name>` and calls its _lazy_init(). It only does so once some
    # library has called set_requires_device_init(PrivateUse1, true) -- which
    # some vendor libraries do, so the very first flagos factory call can raise
    # "No module named 'torch_flagos'". Publishing the device module under that name
    # satisfies the lookup; flagos._lazy_init is the real initializer, so this is a
    # rename, not a stub. Harmless on backends that never trigger lazy init.
    sys.modules.setdefault("torch_flagos", flagos)

    # Apex's amp_C extension bypasses the ATen dispatcher and therefore cannot use
    # the normal DeviceBoxingGuard. Install an optional, CUDA-alias-only shim at the
    # common MultiTensorApply boundary; it remains lazy when Apex is not installed
    # and supports applications that import Apex either before or after torch_fl.
    if _STARTUP_PROFILE == "full":
        activate_optional_integrations("apex", strict=False)

    # Patch FlagGems codegen config before any FlagGems code is imported
    runtime.configure_vendor(_build_accelerator(), flagos)
    # patch_philox() is deliberately NOT called here: it imports
    # flag_gems, and on a build where the vendor's own device name has to be
    # aliased first (TsingMicro) that import needs _alias_cuda_to_flagos to have
    # run. See its call site in _phase_ecosystem.


def _phase_ecosystem() -> None:
    """Mandatory FlagGems/distributed setup, then profile-selected framework hooks.

    One phase of the import-time pipeline below; the order is
    load-bearing, so the constraints are documented at the runner.
    """
    global _TORCH_CUDA_PROBE

    from .compat import triton
    from .compat.flaggems import device, ops, random, runtime

    accelerator = _build_accelerator()
    device.prepare_cuda(accelerator)

    # Capture the CUDA probe at the only point that answers the guard's
    # question. Every vendor compat layer has run by now: `_phase_vendor_compat`
    # -> `configure_vendor` installs the CUDA, MetaX and DCU
    # facades, each of which sets `is_available = lambda: True` because the CPU
    # torch wheel is paired with the vendor's externally loaded libtorch_cuda.so
    # and CUDA genuinely works (the optional MetaX branch in `_phase_claim` does
    # the same, earlier). The alias below has not yet replaced the attribute
    # with its own lambda.
    #
    # Not at `_phase_claim`: there the capture sees the pristine CPU wheel's
    # `False`, the alias installs on a build that has real CUDA, and
    # `stand_down_foreign_drivers` then disables the nvidia Triton
    # backend -- every `torch.compile` dies in `set_driver_to_gpu()` with
    # "Could not find an active GPU backend".
    #
    # Not after this point either: the alias writes its own lambda, and
    # `_patch_native_cuda_probe` overwrites it with `torch.flagos.is_available`
    # later in this phase; both answer True on every build with a flagos
    # device, whether or not a CUDA runtime exists.
    _TORCH_CUDA_PROBE = torch.cuda.is_available

    _alias_cuda_to_flagos()

    # Both of these have to precede the flag_gems import below, which happens
    # here rather than in _phase_vendor_compat for exactly that reason. That
    # import reaches Triton's hint manager, which asks the driver factory to
    # resolve -- and the factory refuses while two backends claim to be active.
    # On a FlagTree wheel the alias installed just above is what makes the stock
    # nvidia backend claim that (its probe is `torch.cuda.is_available()`), so
    # the foreign driver has to be stood down first; `flagos._lazy_init` does
    # this too, but only on the first device operation, which is after this.
    # FlagTree's TsingMicro driver then answers `torch.device("txda", ...)`,
    # which resolves only once the alias has mapped "txda" to flagos.
    #
    # Getting either wrong makes the import raise, and a failed import leaves
    # every flag_gems submodule that *did* load in sys.modules bound to the
    # abandoned half-initialised package. FlagGems' own tensor_wrapper and
    # pointwise_dynamic read `flag_gems.vendor_name` off that binding, so ops
    # going through them (`cat` among them) then fail with `module 'flag_gems'
    # has no attribute 'vendor_name'` for the life of the process.
    triton.stand_down_foreign_drivers(_cuda_alias_active)
    random.patch_philox(accelerator, flagos)

    # Correct flag_gems' layer-norm backward for 3-D+ inputs; must precede the
    # first op call, which is where the generated kernel resolves the callable.
    ops.patch_layer_norm_backward(accelerator)

    runtime.prepare_dispatch(accelerator)
    # Dispatch preparation can import op modules with their own copy of the
    # seed function the bridge replaced above.
    random.refresh_philox_bindings()

    from . import quantization  # noqa: E402, F401

    _register_distributed_backend()

    if _STARTUP_PROFILE == "full":
        activate_optional_integrations(
            "ddp",
            "parallel_comm",
            "dataparallel",
            "data_parallel",
            "compile",
            "flex_attention",
            strict=False,
        )
        if _build_accelerator() == "bpu":
            activate_optional_integrations("bpu", strict=False)

    from . import native_host

    native_host._activate()


# ===========================================================================
# Import-time phase pipeline.
#
# torch_fl runs a fixed sequence of side effects at import. The order is
# load-bearing -- a wrong order produces a dlopen abort or a wrong-vendor build,
# not a Python exception -- so it lives in exactly one place here instead of
# being implied by where each statement happens to sit in the file.
#
#   1. conf          pick the op-routing config and stage the MetaX shim
#   2. preload       select/preload the vendor libtorch and CUDA assets
#   3. claim         import torch, free PrivateUse1, load _C, install the device
#   4. vendor_compat install the vendor runtime shims and resolve GEMS_VENDOR
#   5. ecosystem     mandatory FlagGems/distributed setup, then optional hooks
#
# Constraints, each next to the phase it constrains:
#   * preload before claim: the vendor libtorch has to be in place before
#     `import torch`, and the CUDA assets before _C is dlopened.
#   * within claim, the PrivateUse1 check precedes `import torch_fl._C`: the .so
#     registers the AutogradPrivateUse1 fallback at dlopen time, and a vendor
#     plugin that already claimed the key turns that into an uncatchable abort.
#   * claim before vendor_compat: FlagGems reads GEMS_VENDOR at its import, so
#     the shims and the vendor resolution must precede the ecosystem hooks.
#   * vendor_compat before ecosystem: the FlagGems/dist/compile hooks act on the
#     vendor surface the previous phase installed.
# ===========================================================================
_phase_conf()
_phase_preload()
_phase_claim()
_phase_vendor_compat()
_phase_ecosystem()


__all__ = [
    "flagos",
    "distributed",
    "get_registered_ops",
    "is_flaggems_enabled",
    "fused_kernels_available",
    "device_capabilities",
    "activate_optional_integrations",
    "optional_integration_status",
    "quantization",
]
