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

import os
import re
import sys


def _build_accelerator() -> str:
    """Accelerator this wheel was built for, lowercased ("" if unknown).

    Reads the ACCELERATOR env var first, then the _build_config.py that setup.py
    writes at build time. The generated file is what makes a DCU wheel
    self-describing: _select_backend_config() runs before `import torch`, so it
    cannot inspect torch.version.hip to detect DCU on its own.
    """
    env = os.environ.get("ACCELERATOR", "").strip().lower()
    if env:
        return env
    try:
        from torch_fl._build_config import ACCELERATOR as built
    except ImportError:
        return ""
    return str(built).strip().lower()


def _select_backend_config() -> None:
    """Pick the op-routing config file based on the FLAGOS_USE_FLAGGEMS switch.

    The C++ dispatcher (csrc/aten/common.cc) reads FLAGOS_BACKEND_CONFIG to
    decide, per op, whether to run the CUDA boxing kernel or the FlagGems
    Python-path kernel. Both kernel sets are compiled into the wheel, so the
    choice is purely runtime:

      * FLAGOS_USE_TILEOPS=1                   -> backends_tileops.conf
      * FLAGOS_USE_FLAGGEMS_CPP=1              -> backends_flaggems_cpp.conf
      * FLAGOS_USE_FLAGGEMS_CPP=1 + METAX_BOXING=1
                                               -> backends_metax_flaggems_cpp.conf
      * FLAGOS_USE_FLAGGEMS=1                  -> backends_flaggems.conf
      * FLAGOS_USE_FLAGGEMS=1 + METAX_BOXING=1 -> backends_metax_flaggems.conf
      * FLAGOS_DCU_SDK_OPS=1                  -> backends_dcu_sdk.conf
      * FLAGOS_USE_FLAGGEMS=1 + ACCELERATOR=dcu -> backends_dcu_flaggems.conf
      * unset / 0                              -> backends_cuda.conf (pure boxing)

    FLAGOS_USE_FLAGGEMS_CPP=1 activates the C++ FlagGems path (kFlagOs,
    backends_flaggems_cpp.conf): 18 ops route to the flag_gems C++ runtime
    (liboperators.so, no GIL), the remainder fall back to flagos_python.
    Only valid when torch_fl was built with FLAGGEMS_KERNEL=ON.

    On MetaX the C++ path uses backends_metax_flaggems_cpp.conf: 17 of those 18
    ops are verified on-device, but mm/mm.out go to the cuda boxing kernel
    because flag_gems' C++ mm_kernel_general requests 98304 bytes of shared
    memory and MetaX C550 provides 65536 (mcErrorInvalidValue at launch). Its
    non-C++ ops inherit the backends_metax_flaggems.conf routing, so the
    triton-metax fallbacks documented there still apply. This needs a FlagGems
    built for MACA (cpp/ -DFLAGGEMS_BACKEND=MACA) linked in at build time.

    On an Ascend NPU box (detected via /dev/davinci*), the ACL C++ backend is the
    only usable one, so the choice is instead:

      * FLAGOS_USE_FLAGGEMS=1 -> backends_ascend_flagos_py.conf (FlagGems Triton
                                 where triton-ascend can run, else ascend aclnn)
      * unset / 0             -> backends_ascend.conf (pure aclnn C++)

    The MetaX flaggems conf mirrors backends_flaggems.conf but routes the ops
    triton-metax cannot run (mm/bmm/mean.dim) back to the cuda boxing kernel
    (maca libtorch_cuda) instead of flagos_python. The DCU one does the same for
    the ops DTK's triton (hcu backend) cannot run -- slice_backward (hardware
    VMFault) and silu_backward (missing div_rn lowering). An explicit
    FLAGOS_BACKEND_CONFIG always wins (advanced/testing use), and the per-op
    FLAGOS_OP_<name> overrides in common.cc still apply on top. This must run
    before the first op dispatch triggers BackendTable() init; setting it at
    import time (before any flagos tensor op) is well before that.
    """
    if os.environ.get("FLAGOS_BACKEND_CONFIG"):
        return
    use_tileops = os.environ.get("FLAGOS_USE_TILEOPS", "0") not in (
        "0",
        "",
        "off",
        "OFF",
        "false",
        "FALSE",
    )
    use_flaggems_cpp = os.environ.get("FLAGOS_USE_FLAGGEMS_CPP", "0") not in (
        "0",
        "",
        "off",
        "OFF",
        "false",
        "FALSE",
    )
    use_flaggems = os.environ.get("FLAGOS_USE_FLAGGEMS", "0") not in (
        "0",
        "",
        "off",
        "OFF",
        "false",
        "FALSE",
    )
    use_dcu_sdk = os.environ.get("FLAGOS_DCU_SDK_OPS", "0").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    ) or os.environ.get("FLAGOS_DCU_SDK_ONLY", "0").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )
    metax_boxing = os.environ.get("FLAGOS_METAX_BOXING", "0") == "1"

    # A vendor build whose kernels are native (no CUDA boxing) records its
    # platform in lib/flagos_platform. MUSA and Ascend each have an explicit
    # opt-in hybrid config; all other native platforms retain their
    # native-only config. This branch runs before the /dev/davinci* runtime
    # check below, so it must reproduce that check's FlagGems opt-in itself --
    # otherwise an Ascend build with the marker installed would silently drop
    # FLAGOS_USE_FLAGGEMS=1 back to the native-only conf.
    marker = os.path.join(os.path.dirname(__file__), "lib", "flagos_platform")
    if os.path.exists(marker):
        with open(marker) as f:
            platform = f.read().strip().lower()
        platform_name = f"backends_{platform}"
        if platform in ("musa", "ascend") and use_flaggems:
            platform_name = f"backends_{platform}_flagos_py"
        platform_conf = os.path.join(
            os.path.dirname(__file__), "configs", f"{platform_name}.conf"
        )
        if os.path.exists(platform_conf):
            os.environ["FLAGOS_BACKEND_CONFIG"] = platform_conf
            return

    conf_dir = os.path.join(os.path.dirname(__file__), "configs")

    # Ascend builds compile the ACL C++ backend (Backend::kAscend), not the CUDA
    # boxing kernels, so the cuda/flaggems confs (which route ops to `cuda`) can
    # never apply. Since every wheel ships all backends*.conf files, the conf set
    # can't distinguish the build; use the runtime hardware signal instead. An
    # Ascend NPU exposes /dev/davinci* device nodes -- their presence means this
    # is an Ascend box, where the only usable routing is the ascend conf. (A CUDA
    # build could not run here anyway, so this never mis-fires on a CUDA host.)
    ascend_default = os.path.join(conf_dir, "backends_ascend.conf")
    ascend_flaggems = os.path.join(conf_dir, "backends_ascend_flagos_py.conf")
    try:
        is_ascend_build = os.path.exists(ascend_default) and any(
            name.startswith("davinci") for name in os.listdir("/dev")
        )
    except OSError:
        is_ascend_build = False

    if is_ascend_build:
        conf_path = (
            ascend_flaggems
            if (use_flaggems and os.path.exists(ascend_flaggems))
            else ascend_default
        )
        if os.path.exists(conf_path):
            os.environ["FLAGOS_BACKEND_CONFIG"] = conf_path
        return

    if use_dcu_sdk and _build_accelerator() == "dcu":
        if os.environ.get("FLAGOS_DCU_SDK_ONLY", "0").strip().lower() in (
            "1",
            "on",
            "true",
            "yes",
        ):
            conf_name = "backends_dcu_sdk_only.conf"
        else:
            conf_name = "backends_dcu_sdk.conf"
    elif use_tileops:
        conf_name = "backends_tileops.conf"
    elif use_flaggems_cpp and metax_boxing:
        conf_name = "backends_metax_flaggems_cpp.conf"
    elif use_flaggems_cpp:
        conf_name = "backends_flaggems_cpp.conf"
    elif use_flaggems and metax_boxing:
        conf_name = "backends_metax_flaggems.conf"
    elif use_flaggems and _build_accelerator() == "dcu":
        conf_name = "backends_dcu_flaggems.conf"
    elif use_flaggems:
        conf_name = "backends_flaggems.conf"
    else:
        conf_name = "backends_cuda.conf"
    conf_path = os.path.join(os.path.dirname(__file__), "configs", conf_name)
    if os.path.exists(conf_path):
        os.environ["FLAGOS_BACKEND_CONFIG"] = conf_path


_select_backend_config()

# Optional: PyTorch wheels may require libcudart.so.12 version tags on MetaX.
if os.environ.get("FLAGOS_METAX_CUDART_SHIM", "0") == "1":
    from torch_fl.accelerator.metax._metax_cudart_shim import ensure_cudart_shim

    ensure_cudart_shim()


def _relink_vendor_libtorch() -> None:
    """Point the active torch wheel's torch/lib at this wheel's bundled libtorch.

    MetaX, DCU and PPU all run on a *forked* libtorch whose core .so
    (libc10/libtorch_cpu/libtorch_python/...) differ from the upstream ones a
    stock ``torch==X.Y.Z+cpu`` wheel ships.  A self-contained wheel bundles them
    under torch_fl/lib_{maca,dcu,ppu}/ and symlinks them over the stock files;
    see torch_fl.accelerator._vendor_libtorch for why a ctypes preload alone is
    not enough there.

    This MUST run before `import torch` -- afterwards libc10 is already mapped
    and relinking is too late.  Every backend's entry point is idempotent and a
    no-op when its bundle dir is absent (a plain in-place build, where torch
    already IS the vendor wheel), so this is safe to call unconditionally.

    MetaX: FLAGOS_METAX_BOXING=1 triggers relink unconditionally (for in-place
    MetaX builds that want to test boxing).  When accel=="metax" and the bundle
    dir exists, relink regardless of the env var (self-contained wheel).  DCU
    and PPU have no native-kernel mode, so bundle-dir presence alone decides.
    The CUDA backend is not here: the official +cpu wheel's core .so ARE the
    upstream ones, so only the extra CUDA libs are missing and
    _preload_cuda_assets() below handles those with ctypes.
    """
    accel = _build_accelerator()

    if os.environ.get("FLAGOS_METAX_BOXING", "0") == "1":
        from torch_fl.accelerator.metax._metax_libtorch_link import (
            ensure_maca_libtorch_links,
        )

        ensure_maca_libtorch_links()
        return

    if accel == "metax":
        from torch_fl.accelerator._vendor_libtorch import bundled_lib_dir

        if bundled_lib_dir("lib_maca", "libtorch_cuda.so"):
            from torch_fl.accelerator.metax._metax_libtorch_link import (
                ensure_maca_libtorch_links,
            )

            ensure_maca_libtorch_links()
            return

    if accel == "dcu":
        # Decoupled by default: preload only DTK's device libraries on top of the
        # official core, leaving torch/lib untouched. FLAGOS_DCU_VENDOR_CORE=1
        # selects the legacy full-core relink. See
        # torch_fl/accelerator/dcu/_dcu_libtorch_link.py.
        from torch_fl.accelerator.dcu._dcu_libtorch_link import setup_dcu_runtime

        setup_dcu_runtime()
        return

    # PPU builds as ACCELERATOR=cuda (it targets PPU_SDK/CUDA_SDK), so the only
    # distinguishing signal at import time is its own bundle dir.
    if accel in ("cuda", ""):
        from torch_fl.accelerator._vendor_libtorch import bundled_lib_dir

        if bundled_lib_dir("lib_ppu", "libtorch_cuda.so"):
            from torch_fl.accelerator.ppu._ppu_libtorch_link import (
                ensure_ppu_libtorch_links,
            )

            ensure_ppu_libtorch_links()


_relink_vendor_libtorch()


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

    if os.environ.get("FLAGOS_DISABLE_CUDA_ASSETS", "0") == "1":
        return

    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    main_cuda = os.path.join(lib_dir, "libtorch_cuda.so")
    if not os.path.exists(main_cuda):
        # No bundled assets; rely on an out-of-band preload (e.g. LD_PRELOAD via
        # scripts/with_cuda_libtorch.sh) if the user set one up.
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
        "libcublasLt.so*",
        "libcublas.so*",
        "libcudnn*.so*",
        "libcufft.so*",
        "libcurand.so*",
        "libcusparse.so*",
        "libcusparseLt.so*",
        "libcusolver.so*",
        "libnccl.so*",
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
    if _build_accelerator() != "musa":
        return
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _validate_dcu_decoupled_runtime() -> None:
    """Prove the DTK device libs really bound to the official core.

    Only runs for a decoupled DCU build (ACCELERATOR=dcu without
    FLAGOS_DCU_VENDOR_CORE=1), right after `import torch`; the checks themselves
    live in accelerator/dcu/_dcu_runtime_check.py. Set
    FLAGOS_DCU_SKIP_RUNTIME_CHECK=1 to bypass (e.g. when deliberately testing a
    non-matching wheel pair).
    """
    if _build_accelerator() != "dcu":
        return
    if os.environ.get("FLAGOS_DCU_SKIP_RUNTIME_CHECK", "0") == "1":
        return

    from torch_fl.accelerator.dcu._dcu_libtorch_link import vendor_core_mode

    if vendor_core_mode():
        return  # legacy mode replaces the core wholesale; nothing to reconcile.

    from torch_fl.accelerator.dcu._dcu_sdk_ops import sdk_only

    if sdk_only():
        # No DTK libtorch is mapped in this mode, so the decoupled checks would
        # fail by design. Its own validation needs the plugin loaded, which only
        # happens after torch_fl._C, so it runs from there instead.
        return

    from torch_fl.accelerator.dcu._dcu_runtime_check import validate_decoupled_runtime

    validate_decoupled_runtime(os.path.join(os.path.dirname(__file__), "lib_dcu"))


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


_preload_cuda_assets()
_disable_vendor_backend_autoload()

import torch  # noqa: E402

# Immediately after `import torch`, and before anything relies on CUDA dispatch:
# confirm the DTK device libraries actually bound to the official core.
_validate_dcu_decoupled_runtime()

if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries


# Optional FlagGems-on-MetaX compat (does not patch torch.cuda unless enabled).
if os.environ.get("FLAGOS_METAX_COMPAT", "0") == "1":
    from torch_fl.accelerator.metax._metax_compat import (  # noqa: E402
        is_metax_available,
        patch_torch_cuda_for_metax,
    )

    if is_metax_available():
        patch_torch_cuda_for_metax()


# Expose libtorch symbols globally so triton-ascend's JIT-compiled launcher .so
# can resolve c10/ATen symbols (it links implicitly, not via DT_NEEDED).
import ctypes  # noqa: E402
import os as _os  # noqa: E402

_torch_lib = _os.path.join(_os.path.dirname(torch.__file__), "lib")
for _lib in ("libc10.so", "libtorch.so", "libtorch_cpu.so"):
    _p = _os.path.join(_torch_lib, _lib)
    if _os.path.exists(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

# Load libstream_api.so with RTLD_GLOBAL so that liboperators.so (FlagGems)
# can resolve GetCurrentStream at runtime.
_stream_api_path = _os.path.join(_os.path.dirname(__file__), "lib", "libstream_api.so")
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


# SDK-native DCU operators are registered only after the core extension is loaded:
# the plugin calls an explicit ABI bridge exported by libtorch_fl.so, and must not
# create a competing PrivateUse1 registration. SDK-only mode also checks the
# process map here, before any user operation can accidentally load libtorch_hip.
if _build_accelerator() == "dcu":
    from torch_fl.accelerator.dcu._dcu_sdk_ops import (
        assert_sdk_only_process as _assert_dcu_sdk_only_process,
        enabled as _dcu_sdk_ops_enabled,
        load_and_register as _load_dcu_sdk_ops,
    )

    if _dcu_sdk_ops_enabled():
        _dcu_sdk_manifest = _load_dcu_sdk_ops()
        _assert_dcu_sdk_only_process()
        if os.environ.get("FLAGOS_DCU_SDK_ONLY", "0").strip().lower() in (
            "1",
            "on",
            "true",
            "yes",
        ):
            from torch_fl.accelerator.dcu._dcu_runtime_check import (
                validate_sdk_only_runtime,
            )

            validate_sdk_only_runtime()
            del validate_sdk_only_runtime
        del _dcu_sdk_manifest

    del _assert_dcu_sdk_only_process, _dcu_sdk_ops_enabled, _load_dcu_sdk_ops


from . import flagos  # noqa: E402

torch.utils.rename_privateuse1_backend("flagos")
torch._register_device_module("flagos", flagos)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)


def _install_musa_flaggems_compat() -> None:
    """Expose the MUSA surface expected by FlagGems on top of flagos.

    FlagGems 5.x selects its MThreads backend from ``torch.musa`` and imports
    ``current_device``/``get_device_capability`` from ``torch_musa``. The
    native torch_musa plugin cannot claim PrivateUse1 in the same process, so
    provide only the small compatibility surface required during FlagGems
    discovery. The actual tensor device remains ``flagos``.
    """
    if _build_accelerator() != "musa" or os.environ.get(
        "FLAGOS_USE_FLAGGEMS", "0"
    ).lower() in ("0", "", "off", "false"):
        return

    import importlib.machinery
    import importlib.util
    import types

    musa = types.ModuleType("torch.musa")
    for name in (
        "device",
        "device_count",
        "current_device",
        "set_device",
        "synchronize",
        "is_available",
        "get_device_properties",
        "current_stream",
        "stream",
        "Event",
        "default_generators",
    ):
        setattr(musa, name, getattr(flagos, name))

    def _get_device_capability(device=None):
        if device is None:
            device = flagos.current_device()
        props = flagos.get_device_properties(device)
        return props.major, props.minor

    musa.get_device_capability = _get_device_capability
    musa.get_device_name = lambda device=None: (
        flagos.get_device_properties(
            flagos.current_device() if device is None else device
        ).name
    )
    musa.__spec__ = importlib.machinery.ModuleSpec(
        name="torch.musa", loader=None, origin="torch_fl_shim"
    )
    torch.musa = musa

    # FlagTree's MThreads benchmark helper allocates its cache with the literal
    # device name "musa" while running under a context that bypasses
    # TorchFunctionMode. Stock PyTorch cannot parse that vendor device name, so
    # translate this one factory call to the renamed PrivateUse1 backend.
    original_empty = torch.empty

    def _musa_empty(*args, **kwargs):
        device = kwargs.get("device")
        if isinstance(device, str) and (device == "musa" or device.startswith("musa:")):
            suffix = device[4:]
            kwargs = {**kwargs, "device": f"flagos{suffix}"}
        return original_empty(*args, **kwargs)

    torch.empty = _musa_empty

    if "torch_musa" not in sys.modules:
        # Preserve access to the installed package's distributed submodule for
        # MCCL fallback without importing torch_musa.__init__, which would claim
        # the process-global PrivateUse1 hooks before torch_fl can own them.
        search_locations = None
        try:
            spec = importlib.util.find_spec("torch_musa")
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            search_locations = spec.submodule_search_locations
        torch_musa = types.ModuleType("torch_musa")
        torch_musa.current_device = flagos.current_device
        torch_musa.set_device = flagos.set_device
        torch_musa.current_stream = flagos.current_stream
        torch_musa.get_device_properties = flagos.get_device_properties
        torch_musa.get_device_name = musa.get_device_name
        torch_musa.get_device_capability = musa.get_device_capability
        torch_musa._MUSAC = types.SimpleNamespace(
            _musa_getCurrentRawStream=flagos._C._get_musa_current_raw_stream
        )
        torch_musa.musa = musa
        torch_musa.__path__ = list(search_locations or ())
        torch_musa.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_musa",
            loader=None,
            origin="torch_fl_shim",
            is_package=True,
        )
        torch_musa.__spec__.submodule_search_locations = torch_musa.__path__
        sys.modules["torch_musa"] = torch_musa

    # ProcessGroupMCCL may already be registered by a linked vendor library.
    # Publishing it through the shim keeps process_group.py's fallback lookup
    # useful even when importing torch_musa.distributed itself is unavailable.
    mccl_cls = getattr(torch.distributed, "ProcessGroupMCCL", None)
    if mccl_cls is not None and "torch_musa.distributed" not in sys.modules:
        distributed = types.ModuleType("torch_musa.distributed")
        distributed.ProcessGroupMCCL = mccl_cls
        distributed.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_musa.distributed", loader=None, origin="torch_fl_shim"
        )
        sys.modules["torch_musa.distributed"] = distributed
        sys.modules["torch_musa"].distributed = distributed


_install_musa_flaggems_compat()

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
try:
    from torch_fl.compat.apex import install_apex_compat

    install_apex_compat()
except Exception as exc:  # noqa: BLE001 - Apex compatibility is optional
    import warnings

    warnings.warn(
        f"[torch_fl] Apex compatibility setup was skipped: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


# Global library instance to keep registrations alive
_flaggems_lib = None
_autograd_lib = None
_registered_ops = []


def _patch_flaggems_philox():
    """Route MUSA FlagGems RNG reservations through the flagos generator.

    Native muRAND/mudnn kernels and this bridge both consume one 64-bit seed per
    stochastic operation from the selected PrivateUse1 generator. FlagGems then
    starts its per-operation Philox stream at offset zero. This keeps
    manual_seed/get_rng_state/set_rng_state and mixed native/FlagGems call order
    deterministic without maintaining a second CUDA-style generator state.
    """
    if _build_accelerator() != "musa" or os.environ.get(
        "FLAGOS_USE_FLAGGEMS", "0"
    ).lower() in ("0", "", "off", "false"):
        return

    try:
        from flag_gems.utils import random_utils

        _orig = random_utils.philox_backend_seed_offset

        def _patched(increment, generator=None):
            if generator is None:
                device = random_utils.torch_device_fn.current_device()
                seed = flagos._C._reserve_rng_seed(device)
            else:
                generator_device = getattr(generator, "device", None)
                if getattr(generator_device, "type", None) not in (
                    "flagos",
                    "privateuseone",
                ):
                    return _orig(increment, generator=generator)
                device = generator_device.index
                if device is None:
                    device = random_utils.torch_device_fn.current_device()
                seed = flagos._C._reserve_rng_seed(device, generator)

            # FlagGems obtains CUDA generator seeds through an int64 state tensor.
            # Preserve those signed bit semantics when the reserved uint64 has its
            # high bit set.
            if seed >= 1 << 63:
                seed -= 1 << 64
            return seed, 0

        # RNG modules bind this function with ``from ... import`` at import time,
        # so update every already-loaded copy as well as the canonical module.
        for mod in list(sys.modules.values()):
            name = getattr(mod, "__name__", "")
            if name.startswith("flag_gems") and hasattr(
                mod, "philox_backend_seed_offset"
            ):
                mod.philox_backend_seed_offset = _patched
        random_utils.philox_backend_seed_offset = _patched
    except Exception:
        # FlagGems remains optional; native MUSA kernels stay available.
        pass


def _restore_dcu_hip_version() -> None:
    """Set torch.version.hip/rocm for a self-contained DCU wheel.

    See the DCU branch of _patch_flaggems_codegen_config() for why this matters:
    the bundled libtorch is DTK's HIP build, but torch/version.py comes from the
    stock torch+cpu wheel in front and reports hip=None, which switches triton's
    hcu backend off. scripts/bundle_dcu_libtorch.sh copies the vendor torch's own
    version.py next to the bundled .so as vendor_version.py; read the strings
    back from there. No-op when a real vendor torch is in front.
    """
    import torch

    if getattr(torch.version, "hip", None):
        return  # a real DTK torch is in front; leave its values alone.

    hip_ver = os.environ.get("FLAGOS_DCU_HIP_VERSION", "").strip()
    rocm_ver = ""
    if not hip_ver:
        ver_py = os.path.join(os.path.dirname(__file__), "lib_dcu", "vendor_version.py")
        try:
            with open(ver_py, encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"\s*hip\s*(?::[^=]*)?=\s*'([^']+)'", line)
                    if m:
                        hip_ver = m.group(1)
                        continue
                    m = re.match(r"\s*rocm\s*(?::[^=]*)?=\s*'([^']+)'", line)
                    if m:
                        rocm_ver = m.group(1)
        except OSError:
            return  # not a bundled build (source checkout); nothing to restore.
    if hip_ver:
        torch.version.hip = hip_ver
        if rocm_ver:
            torch.version.rocm = rocm_ver


def _patch_flaggems_codegen_config():
    """
    Configure FlagGems' vendor + torch.cuda shim for the flagos device.

    FlagGems uses GEMS_VENDOR env var to detect the hardware vendor.

    - Generic NVIDIA CUDA (default when a real NVIDIA GPU is reachable via
      libcuda.so and MetaX compat is not requested): set GEMS_VENDOR=nvidia and
      shim torch.cuda so FlagGems' Triton kernels can compile/run under CPU
      torch + external libtorch_cuda.so. GEMS_VENDOR=nvidia is REQUIRED so
      FlagGems' tl_extra_shim resolves triton.language.extra.cuda.libdevice
      (which has `pow`); otherwise it falls back to tl.math (no `pow`).
      Disable with FLAGOS_DISABLE_CUDA_SHIM=1.

    - MetaX (boxing + FlagGems): set GEMS_VENDOR=metax so FlagGems uses the
      MetaX codegen config (triton-metax backend, prefer_block_pointer=False to
      avoid the triton-metax block-pointer bug), and patch torch.cuda (device
      props + stream/availability) so FlagGems' Triton kernels run on the
      CPU-frozen torch wheel against maca's libtorch_cuda.so. Auto-selected when
      FLAGOS_METAX_BOXING=1 (or FLAGOS_METAX_COMPAT=1) and a MetaX card is present.

    - Hygon DCU (DTK): set GEMS_VENDOR=hygon so FlagGems uses its _hygon
      codegen config (triton hcu backend, triton_extra_name="hip"). No
      torch.cuda shim is needed -- DTK ships a hipified torch with a real,
      working torch.cuda. This branch must precede the generic-NVIDIA one:
      is_nvidia_cuda_available() is False on DTK (there is no libcuda.so, only
      libgalaxyhip), so without it DCU would reach the ascend fallback and get
      GEMS_VENDOR=ascend -- which also breaks the comm layer, since that vendor
      selects the HCCL profile (see comm/process_group.py _VENDOR_PROFILES).

    - Ascend (fallback): set GEMS_VENDOR=ascend so FlagGems uses the ASCEND
      codegen config (prefer_block_pointer=False, avoiding a triton-ascend
      tl.make_block_ptr bug), and register torch.flagos as a torch.npu shim so
      FlagGems' gen_torch_device_object('ascend') resolves correctly.
    """
    import os
    import sys

    # --- Moore Threads MUSA branch ---
    if _build_accelerator() == "musa":
        os.environ.setdefault("GEMS_VENDOR", "mthreads")
        return

    # --- MetaX branch (boxing + FlagGems) ---
    # Auto-detects MetaX hardware (like DCU does) or triggered by explicit
    # FLAGOS_METAX_COMPAT=1 or FLAGOS_METAX_BOXING=1. Must come before the ascend
    # fallback so MetaX never wrongly gets GEMS_VENDOR=ascend.
    _metax_requested = (
        _build_accelerator() == "metax"
        or os.environ.get("FLAGOS_METAX_COMPAT", "0") == "1"
        or os.environ.get("FLAGOS_METAX_BOXING", "0") == "1"
    )
    if _metax_requested and os.environ.get("GEMS_VENDOR") not in ("nvidia", "ascend"):
        from torch_fl.accelerator.metax._metax_compat import (
            is_metax_available,
            patch_torch_cuda_for_metax,
        )

        if is_metax_available():
            os.environ.setdefault("GEMS_VENDOR", "metax")
            patch_torch_cuda_for_metax()
            return

    # --- Hygon DCU branch (DTK) ---
    # Keyed on the build accelerator rather than probing the runtime: DTK's torch
    # is hipified, so torch.cuda/torch.version.hip look "cuda-ish" and no
    # libcuda.so probe can tell the two apart. Must come before both the generic
    # NVIDIA branch (which no-ops here anyway -- no libcuda.so) and the ascend
    # fallback. setdefault so an explicit GEMS_VENDOR still wins.
    if _build_accelerator() == "dcu" and os.environ.get("GEMS_VENDOR") != "ascend":
        os.environ.setdefault("GEMS_VENDOR", "hygon")
        # torch.version is pure Python (torch/version.py), generated when the
        # wheel is built -- swapping the bundled DTK .so files cannot change it.
        # A self-contained DCU wheel therefore front-ends a stock torch+cpu whose
        # torch.version.hip is None, while the DTK torch it replaces reports
        # e.g. "6.3.26113". triton's hcu backend gates on exactly that value
        # (backends/hcu/driver.py is_active(): torch.cuda.is_available() and
        # torch.version.hip is not None), so with None the driver never activates
        # and any flag_gems op dies in triton's driver factory with
        # "0 active drivers ([]). There should only be one." Restore the attribute
        # from the bundled libtorch's own version so triton sees a HIP torch,
        # matching what the vendor wheel reported.
        _restore_dcu_hip_version()
        from torch_fl.accelerator.dcu._dcu_compat import (
            install_dcu_rng_bridge,
            patch_torch_cuda_for_dcu,
        )

        # Order matters. A decoupled DCU wheel runs on the official torch+cpu
        # wheel, whose torch.cuda reports is_available()=False and raises from
        # _lazy_init() ("Torch not compiled with CUDA enabled") -- the other half
        # of triton's hcu gate, alongside the torch.version.hip restored above. So
        # patch torch.cuda first (a no-op when a real DTK torch is in front), then
        # bridge the flagos RNG onto the CUDA generators it now exposes.
        patch_torch_cuda_for_dcu()
        install_dcu_rng_bridge()
        return

    # --- Enflame GCU branch ---
    # Keyed on the build accelerator for the same reason as DCU: no runtime probe
    # distinguishes GCU here, and the tops stack has no libcuda.so, so without
    # this branch GCU would reach the ascend fallback and get GEMS_VENDOR=ascend
    # (which also picks the wrong comm profile). FlagGems' Triton kernels need
    # Enflame's triton_gcu plugin plus its /opt/triton_gcu compiler toolchain; if
    # either is missing, patch_triton_gcu_for_flagos() returns False and we leave
    # GEMS_VENDOR unset so the topsaten kernels and cpu_fallback stay in charge.
    if _build_accelerator() == "gcu" and os.environ.get("GEMS_VENDOR") != "ascend":
        from torch_fl.accelerator.gcu._gcu_compat import (
            install_gcu_rng_generators,
            patch_triton_gcu_for_flagos,
        )

        install_gcu_rng_generators()
        if patch_triton_gcu_for_flagos():
            os.environ.setdefault("GEMS_VENDOR", "enflame")
        return

    # --- Generic NVIDIA CUDA branch (default) ---
    if (
        os.environ.get("FLAGOS_DISABLE_CUDA_SHIM", "0") != "1"
        and os.environ.get("FLAGOS_METAX_COMPAT", "0") != "1"
        and os.environ.get("FLAGOS_METAX_BOXING", "0") != "1"
        and os.environ.get("GEMS_VENDOR") != "ascend"
    ):
        from torch_fl.accelerator.cuda._cuda_compat import (
            is_nvidia_cuda_available,
            patch_torch_cuda_for_flagos,
        )

        if is_nvidia_cuda_available():
            os.environ.setdefault("GEMS_VENDOR", "nvidia")
            # patch_torch_cuda_for_flagos installs per-device CUDA generators as
            # torch.cuda.default_generators and routes cuda seeding to them, so
            # flag_gems' philox_backend_seed_offset finds a real, seedable
            # generator on its own -- no philox monkeypatch needed.
            patch_torch_cuda_for_flagos()
            return

    # --- Ascend fallback branch ---
    # Set vendor before FlagGems runtime initializes
    if "GEMS_VENDOR" not in os.environ:
        os.environ["GEMS_VENDOR"] = "ascend"

    # FlagGems' ASCEND backend expects torch.npu to exist (device_name="npu").
    # Provide torch.flagos as a shim so gen_torch_device_object() succeeds.
    # Mark is_available()=False so transformers/accelerate don't think real
    # NPU hardware is present and try to import npu_fusion_attention etc.
    if not hasattr(torch, "npu"):
        import types

        _npu_device_shim = types.ModuleType("torch.npu")
        _npu_device_shim.is_available = lambda: False
        _npu_device_shim.device_count = flagos.device_count
        _npu_device_shim.current_device = flagos.current_device
        _npu_device_shim.set_device = flagos.set_device
        _npu_device_shim.synchronize = flagos.synchronize
        _npu_device_shim.device = flagos.device
        _npu_device_shim.Stream = flagos.Stream
        _npu_device_shim.Event = flagos.Event
        _npu_device_shim.current_stream = flagos.current_stream
        _npu_device_shim.default_generators = flagos.default_generators
        # FlagGems' utils/triton_driver_helper.py captures
        # torch_device_fn.get_device_properties at import time and falls back to
        # triton's driver on AttributeError -- and triton-ascend's version returns
        # a plain dict, so gems' `get_device_properties(idx).multi_processor_count`
        # raises AttributeError deep inside a kernel launch. cumsum hit this on the
        # (1, 151936) logits of Qwen3's sampler, failing every generate() on the
        # gems path while smaller shapes took a branch that never queried it.
        _npu_device_shim.get_device_properties = flagos.get_device_properties
        torch.npu = _npu_device_shim

    # FlagGems' ASCEND backend imports torch_npu in _get_vendor_from_quick_cmd.
    # Provide a minimal shim module so the import doesn't fail.
    # Also set __spec__ to satisfy importlib.util.find_spec() checks (used by
    # accelerate.utils.imports.is_npu_available).
    if "torch_npu" not in sys.modules:
        import types
        import importlib.machinery

        _npu_shim = types.ModuleType("torch_npu")
        _npu_shim.npu = _npu_device_shim
        _npu_shim.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_npu",
            loader=None,
            origin="torch_fl_shim",
        )
        sys.modules["torch_npu"] = _npu_shim


# Patch FlagGems codegen config before any FlagGems code is imported
_patch_flaggems_codegen_config()
_patch_flaggems_philox()


def _patch_cuda_device_context():
    """
    Monkey-patch torch.cuda.device to handle flagos devices.

    FlagGems internally calls torch_device_fn.device(tensor.device), but when
    tensor.device is 'flagos:0', torch.cuda.device() fails because it expects
    a CUDA device. This patch wraps torch.cuda.device.__init__ to extract just
    the device index from flagos/privateuseone devices.
    """
    _original_cuda_device_init = torch.cuda.device.__init__

    def _patched_cuda_device_init(self, device):
        # Handle flagos/privateuseone devices by extracting just the index
        if hasattr(device, "type") and hasattr(device, "index"):
            if device.type in ("privateuseone", "flagos"):
                device = device.index if device.index is not None else 0
        return _original_cuda_device_init(self, device)

    torch.cuda.device.__init__ = _patched_cuda_device_init


# Patch torch.cuda.device before FlagGems is used
_patch_cuda_device_context()

# Initialize CUDA runtime only when FlagGems Python path needs it (CUDA backend ops).
# The check must be against the *build* backend, not torch.cuda.is_available():
# a DCU/PPU self-contained wheel relinks a hipified/cuda libtorch into a stock
# +cpu torch, which makes is_available() return True even though the CUDA runtime
# libs are absent, and torch.cuda.init() would fail with "libcaffe2_nvrtc.so: not
# found". Only actual CUDA-backend builds need this init.
if (
    os.environ.get("FLAGOS_DISABLE_FLAGGEMS_PY", "0") != "1"
    and _build_accelerator() in ("cuda", "")
    and torch.cuda.is_available()
):
    torch.cuda.init()


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

    Deliberately a no-op when ``torch.cuda.is_available()``: on the CUDA and
    boxing backends ``cuda`` already means a real device, and hijacking it there
    would break the boxing path, which submits genuine CUDA work.

    Opt out with ``FLAGOS_ALIAS_CUDA=0`` -- worth doing if you need
    ``device="cuda"`` to keep failing loudly rather than silently redirecting.
    """
    if torch.cuda.is_available():
        return
    if os.environ.get("FLAGOS_ALIAS_CUDA", "1").lower() in ("0", "off", "false"):
        return

    from torch.overrides import TorchFunctionMode

    _flagos_type = torch._C._get_privateuse1_backend_name()  # "flagos"
    _orig_device = torch.device

    def _remap(dev):
        """Vendor accelerator aliases -> flagos; everything else untouched."""
        if isinstance(dev, str):
            aliases = ("cuda", "musa") if _build_accelerator() == "musa" else ("cuda",)
            for alias in aliases:
                if dev == alias:
                    return _flagos_type
                if dev.startswith(f"{alias}:"):
                    return f"{_flagos_type}:{dev[len(alias) + 1 :]}"
            return dev
        if isinstance(dev, _orig_device) and dev.type == "cuda":
            return _orig_device(_flagos_type, dev.index if dev.index is not None else 0)
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
        idx = 0
        if device is not None:
            d = _orig_device(_remap(device))
            idx = d.index if d.index is not None else 0
        return self.to(f"{_flagos_type}:{idx}", non_blocking=non_blocking)

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


_alias_cuda_to_flagos()


# Ops that use torch_device_fn.device(device) with explicit device parameter
# These don't work with flagos device and should use cpu_fallback instead
_EXCLUDED_OPS = {
    # Factory functions that take device parameter
    "randn",
    "randn_like",
    "rand",
    "rand_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "arange",
    "arange.start",
    "arange.start_step",
    "linspace",
    "logspace",
    "eye",
    "eye.m",
    "randperm",
    "empty.memory_format",  # Already registered in C++
    "empty_strided",  # Already registered in C++
    # FlagGems registers the *bare* aten name "empty" as well, implemented by a
    # function also called "empty" -- so "empty.memory_format" above does not
    # filter it (the registrar matches on the function name). Left registered, it
    # is the one factory op FlagGems takes over, and it ignores the requested
    # device: an `empty(..., device="flagos:1")` while flagos:0 is current
    # allocates on 0, and the first write through that pointer faults the driver
    # ("IoctlCmdWriteRead errno[14]. The device is out of service", then a SIP
    # exception that aborts the process). The C++ empty is already correct here.
    "empty",
    # Random ops that use device context.
    #
    # These also cover FlagGems' philox path, which reads the generator state as
    # exactly two int64s (the CUDA layout: seed + offset). torch_fl's flagos
    # generators are CPU Mersenne-Twister generators whose state unpacks to 632
    # int64s, so philox_backend_seed_offset raises "too many values to unpack".
    # Every name FlagGems registers for an RNG op has to appear here or the op
    # reaches that unpacking; the factory/RNG ops are correct on the topsaten and
    # CPU paths anyway, so nothing is lost by keeping them off FlagGems.
    "uniform_",
    "normal_",
    "normal.float_Tensor",
    "normal.Tensor_float",
    "normal.Tensor_tensor",
    "normal.Tensor_Tensor",
    "log_normal_",
    "randint",
    "randint_like",
    "exponential_",
    "multinomial",
    # The rest of the philox users, found by running tests/integration/ops/
    # test_rng_dispatch.py: each of these reaches philox_backend_seed_offset and
    # fails the same way. cauchy_/dropout/poisson are the *vendor's* gcu300
    # overrides, so they are not discoverable from flag_gems/ops/ alone.
    "bernoulli",
    "bernoulli.p",
    "bernoulli_.float",
    "bernoulli_.Tensor",
    "cauchy_",
    "dropout",
    "native_dropout",
    "poisson",
    # Copy ops - already registered in C++, skip to avoid duplicate registration
    "copy_",
    "_to_copy",
    "contiguous",
    "clone",
    # log_softmax - registered in C++ with CUDA structured kernels
    "_log_softmax",
    "_log_softmax_backward_data",
    "_softmax_backward_data",
    "div.Scalar",
    # Ops dispatched by C++ stub (DispatchStub) which reads backends.conf
    # at load time to route to flaggems or cuda per-op.
    "mm",
    "mm.out",
    "bmm",
    "bmm.out",
    "cat",
    "embedding",
    "add.Tensor",
    "mul.Tensor",
    "silu",
    "rsqrt",
    "mean.dim",
    "cos",
    "sin",
    "neg",
    "pow.Tensor_Scalar",
    "all",
    "_softmax",
    "bitwise_and.Tensor",
    "le.Tensor",
    "where.self",
    "index.Tensor",
    "new_ones",
    "scalar_tensor",
    "ones_like",
    "zeros",
    "silu_backward",
    "sum.dim_IntList",
    "slice_backward",
    "constant_pad_nd",
    "embedding_dense_backward",
    "nll_loss_forward",
    "nll_loss_backward",
}


# Ops excluded from FlagGems on Enflame GCU only, on top of _EXCLUDED_OPS.
#
# These have a FlagGems kernel that its Triton backend cannot compile for the
# GCU, so they must stay unregistered to keep reaching the topsaten kernels or
# cpu_fallback. Verified individually on hardware -- only the listed overload
# fails, e.g. var.dim, std.correction and var_mean.correction all work.
_GCU_EXCLUDED_OPS = {
    # NOTE: FlagGems matches this list against the *implementing function* name
    # (op_registrar.config_filter compares item[1].__name__), not the aten op
    # name -- those merely coincide for most ops. aten::var.correction is
    # implemented by var_correction, so that is the name to list.
    #
    # The full-reduction path (var_kernel_1, var.py:88) emits an int64 widening
    # that the GCU backend marks illegal: "failed to legalize operation
    # 'arith.extsi'" -- consistent with the tops stack having no int64 kernels.
    "var",
    "var_correction",
    "var_dim",
    "var_mean",
    # Both are built on Triton's float `%`, which on the GCU returns x rather
    # than 0 when y divides x exactly (the internal division lands just below
    # the integer, so the floor is one too low). torch.remainder(2*y, y) then
    # gives y instead of 0 -- for ~10% of random lanes, silently. Non-multiple
    # operands are correct, which is why this needs an exact-multiple probe to
    # see. Verified on gcu300 with the vendor rem_tt/fmod kernels.
    "remainder",
    "remainder_",
    "fmod_scalar",
    "fmod_tensor",
    "fmod_scalar_",
    "fmod_tensor_",
    "fmod_",
    # Same rounding defect seen through the quotient instead of the remainder:
    # floor_divide(y, y) yields 0 rather than 1 on those same lanes.
    "floor_divide",
    "floor_divide_",
    # No GCU kernel to link against: the vendor linker rejects the relocation
    # for tops_nv_nextafterf_v4_fp32 ("R_DTU_ADDR16_LO_ICALL cannot be used
    # against symbol"), so the op cannot be compiled at all.
    "nextafter",
    "nextafter_",
    # The sort kernels emit the same illegal int64 widening as var_kernel_1
    # ("failed to legalize operation 'arith.extsi'"). msort is sort's caller, so
    # it fails identically, and sort.stable (function sort_stable) shares the
    # kernel -- it fails as "Pipeline run failed: PassManager execution failed".
    # topk/argmax use different kernels and are fine.
    "sort",
    "sort_stable",
    "msort",
    # stack.py:65 builds its offsets in int64 and hits that same legalization
    # failure -- and the backend then core-dumps while reporting the error, so
    # this one cannot be left to raise. cat/vstack/hstack are unaffected and
    # verified correct.
    "stack",
    # The conv VJP (flag_gems/ops/conv2d.py, which conv1d unsqueezes into)
    # passes a stride of 0 as a runtime argument, and the GCU asserts on that
    # inside the kernel: "Not Support dynamic stride is 0, please add
    # tl.constexpr to stride arg in kernel args list". That is a SIP assert, so
    # it aborts the process instead of raising -- it cannot be caught and fallen
    # back from, which is why the whole family stays off FlagGems even though
    # every forward is numerically correct.
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "_conv_depthwise2d",
    "cudnn_convolution",
    # The vendor's own gcu300 layernorm.py:326 hits the int64 widening in its
    # backward kernel, which breaks any .backward() through a LayerNorm. The
    # forward (layer_norm) compiles and is verified correct, so only the backward
    # is excluded -- the gradient falls back to the topsaten/CPU path.
    "native_layer_norm_backward",
    "layer_norm_backward",
    # int64 again, from the other direction: these two are asked for int64
    # *output* rather than int64 indices. `zeros(dtype=torch.int64)` goes through
    # the vendor's gcu300 zeros.py zero_ and fails to compile; `scalar_tensor`
    # with dtype=int64 compiles but returns garbage (42 came back as
    # 4441830098096545884). Both are correct on the C++/topsaten path.
    "zero_",
    "scalar_tensor",
    # flag_gems/ops/diff.py builds its offsets in int64 -- same legalization
    # failure as stack. torch.diff decomposes to narrow+sub on the fallback path.
    "diff",
    # Same story as layernorm: the vendor's own gcu300 embedding.py:98 backward
    # kernel does not legalize, while its forward (embedding) is correct and
    # stays on FlagGems.
    "embedding_dense_backward",
    "embedding_backward",
    # fill_ silently corrupts int64 tensors -- fill_(42) on an int64 tensor comes
    # back as -4846589848703729622, at every rank, with no error. The int64
    # diversion in device_guarded_config cannot rescue it: fill_ writes through
    # its operand, so computing on a CPU copy would discard the result. Excluding
    # it also fixes torch.scalar_tensor(dtype=torch.int64), which is an
    # empty+fill_ underneath and returned garbage even though scalar_tensor
    # itself was already excluded.
    "fill_.Scalar",
    "fill_.Tensor",
    "fill.Scalar",
    "fill.Tensor",
}


# Cache for CUDA runtime library
_cudart_lib = None
_cudaMemcpy = None


def _get_cudaMemcpy():
    """Get cudaMemcpy function from CUDA runtime library (cached)."""
    global _cudart_lib, _cudaMemcpy
    if _cudaMemcpy is not None:
        return _cudaMemcpy

    import ctypes

    # Try to load CUDA runtime library
    try:
        _cudart_lib = ctypes.CDLL("libcudart.so")
    except OSError:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _cudart_lib = ctypes.CDLL(f"{cuda_home}/lib64/libcudart.so")

    _cudaMemcpy = _cudart_lib.cudaMemcpy
    _cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _cudaMemcpy.restype = ctypes.c_int

    return _cudaMemcpy


def _flaggems_exclusion_names(flag_gems, aten_names):
    """Translate aten op names into the function names FlagGems excludes on.

    ``flag_gems.enable(unused=...)`` looks like it takes aten op names, but its
    registrar filters on the *implementing function* name
    (``op_registrar.config_filter`` compares ``item[1].__name__``). Those
    coincide for plain ops -- "randn", "mm" -- and diverge for overloads and
    private ops: ``normal.Tensor_float`` is implemented by
    ``normal_tensor_float``, ``_softmax`` by ``softmax``, ``div.Scalar`` by
    ``true_divide``. Passing an aten name that diverges silently excludes
    nothing, and the op gets registered after all.

    Both spellings are returned: the function name is what actually filters,
    while keeping the original is harmless and covers ops whose two names agree.
    Names absent from FlagGems' config pass through unchanged.
    """
    op_to_func = {}
    for item in getattr(flag_gems, "_FULL_CONFIG", ()):
        if len(item) < 2:
            continue
        func_name = getattr(item[1], "__name__", None)
        if func_name:
            op_to_func.setdefault(item[0], func_name)

    names = set(aten_names)
    for aten_name in aten_names:
        func_name = op_to_func.get(aten_name)
        if func_name:
            names.add(func_name)
    return sorted(names)


def _register_flaggems_operators():
    """
    Register FlagGems operators with the PrivateUse1 (flagos) dispatch key.

    Disabled: Python-layer FlagGems registration is not used.
    All ops are dispatched through the C++ stub path instead.
    """
    global _flaggems_lib, _autograd_lib, _registered_ops

    if os.environ.get("FLAGOS_DISABLE_FLAGGEMS_PY", "0") == "1":
        _registered_ops = []
        return 0

    import importlib.util

    if importlib.util.find_spec("flag_gems") is None:
        # flag_gems not installed, will use cpu_fallback
        return 0

    # GCU FlagGems uses the C++ dispatcher path. Its generated kernels are
    # compiled when FLAGGEMS_PYTHON=ON and selected per overload by
    # FLAGOS_BACKEND_CONFIG. Calling flag_gems.enable() here would register a
    # competing PrivateUse1 implementation and bypass the shared dispatcher.
    if _build_accelerator() == "gcu":
        from torch_fl.accelerator.gcu._gcu_compat import is_triton_gcu_available

        if not is_triton_gcu_available():
            _registered_ops = []
            return 0
        try:
            import flag_gems

            from torch_fl.accelerator.gcu._gcu_compat import (
                bind_vendor_ops_in_generic_modules,
                patch_flaggems_device_name,
            )

            patch_flaggems_device_name()
            bind_vendor_ops_in_generic_modules(flag_gems)
            _registered_ops = []
            return 0
        except Exception as exc:
            import warnings

            warnings.warn(
                f"FlagGems runtime preparation for GCU failed ({type(exc).__name__}: "
                f"{exc}); flagos_python routes may be unavailable.",
                stacklevel=2,
            )
            _registered_ops = []
            return 0

    _flaggems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = []
    return 0


def _register_composite_ops():
    """
    Register CompositeExplicitAutograd ops that cause cpu_fallback segfault.

    Some PyTorch ops are CompositeExplicitAutograd (not CompositeImplicitAutograd),
    meaning they don't auto-decompose for PrivateUse1. They fall through to
    cpu_fallback which segfaults when handling privateuseone tensors.

    Previously _log_softmax and _log_softmax_backward_data were registered here
    as Python decompositions. They are now registered in C++ with proper CUDA
    structured kernels for full performance.
    """
    lib = torch.library.Library("aten", "IMPL")

    # No Python-registered ops remaining; keep the library alive for future use.

    return lib  # prevent GC


# Hold reference to prevent garbage collection of the library
_composite_ops_lib = None


def get_registered_ops():
    """Return list of registered FlagGems operators for flagos device."""
    return list(_registered_ops)


def is_flaggems_enabled():
    """Check if FlagGems operators are registered for flagos device."""
    return len(_registered_ops) > 0


# Auto-register FlagGems operators on import
_register_flaggems_operators()
_composite_ops_lib = _register_composite_ops()

# Re-export integration utilities
from . import quantization  # noqa: E402

from torch_fl.integration import (  # noqa: E402
    is_flaggems_available,
    enable_flaggems_for_flagos,
    use_flaggems,
)

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
    try:
        from torch_fl.comm import register_flagos_backend

        register_flagos_backend()
    except Exception as e:
        import warnings

        warnings.warn(f"[torch_fl] Failed to register 'flagos' dist backend: {e}")


_register_distributed_backend()


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


_patch_ddp_for_flagos()


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
        # registered in inductor's codegen table. The three functions are
        # idempotent and mirror what `flagos_compile_backend` runs before every
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

        register_flagos_device_interface()
        publish_codegen_on_device_module()
        register_flagos_codegen()
        patch_triton_libdevice_module_map()
        patch_triton_resource_limit_errors()
        patch_triton_byte_load_workarounds()
    except (ImportError, AttributeError):
        # torch._dynamo not available (torch < 2.0) or inductor missing
        pass


_register_compile_backend()


def _register_bpu_compile_backend() -> None:
    """Register torch.compile(backend="bpu") on a BPU build.

    The RDK BPU executes whole compiled graphs (a .hbm produced by hbdk4), not
    individual operators, so it has no per-op kernels: eager ops reach
    cpu_fallback and all acceleration comes through this backend. That is the
    opposite of every other platform here, where the compile path is incidental
    and the kernels do the work.

    Import failures are swallowed deliberately. The backend pulls in onnx and
    (optionally) hbdk4, so on a board that has the runtime but not the
    toolchain, raising here would make `import torch_fl` fail outright and take
    the working eager path down with it.
    """
    if _build_accelerator() != "bpu":
        return
    try:
        from torch_fl.accelerator import bpu

        bpu.register()
    except Exception as exc:  # noqa: BLE001
        import warnings

        warnings.warn(
            f'torch.compile(backend="bpu") is unavailable: {exc}. '
            "Eager ops still work (they run on the CPU); the BPU offload path "
            "needs onnx installed.",
            RuntimeWarning,
            stacklevel=2,
        )


_register_bpu_compile_backend()


__all__ = [
    "flagos",
    "distributed",
    "get_registered_ops",
    "is_flaggems_enabled",
    "is_flaggems_available",
    "enable_flaggems_for_flagos",
    "use_flaggems",
    "quantization",
]
