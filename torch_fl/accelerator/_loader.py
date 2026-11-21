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

"""Library loading for the PyTorch and native extension startup phases."""

import ctypes
import os

from torch_fl import _env, _platform

_PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))


def _relink_vendor_libtorch() -> None:
    """Select the vendor's libtorch before PyTorch maps its core libraries.

    MetaX native-host builds use the official CPU core. Hygon normally
    layers its device libraries over that core. Older PPU wheels identify
    their runtime through the bundled libraries.
    """
    build = _platform.build_info()

    if build.platform == "metax":
        # Native hosts link the official CPU core and submit through MACA's
        # runtime. Selecting a discoverable vendor core here would make that
        # contract depend on which package the application imports first.
        if build.native_hosts:
            return
        from torch_fl.accelerator.metax._metax_libtorch_link import (
            ensure_maca_libtorch_links,
        )

        ensure_maca_libtorch_links()
        return

    if build.platform == "dcu":
        # Decoupled by default: preload only DTK's device libraries on top of the
        # official core, leaving torch/lib untouched. FLAGOS_DCU_VENDOR_CORE=1
        # selects the full vendor core through a private facade. See
        # torch_fl/accelerator/hygon/_dcu_libtorch_link.py.
        from torch_fl.accelerator.hygon._dcu_libtorch_link import setup_dcu_runtime

        setup_dcu_runtime()
        return

    # PPU is its own accelerator now; an older PPU wheel reports "cuda" and is
    # recognised by its lib_thead/ bundle instead.
    if build.platform == "ppu":
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
    before PyTorch imports its core libraries.

    libtorch_cuda.so has unresolved deps on the NVIDIA runtime libs (libcudart,
    libcublas, libcudnn, libnvshmem_host, ...) shipped by the pip nvidia-*-cu12
    wheels. Since the process is already running, LD_LIBRARY_PATH cannot help;
    we must explicitly dlopen those deps (RTLD_GLOBAL) in dependency order first,
    then torch's own libc10/libtorch_cpu, then the CUDA libs.

    Skipped when:
      * FLAGOS_DISABLE_CUDA_ASSETS=1 (Ascend/MetaX/pure-CPU, or external preload)
      * the bundled libtorch_cuda.so is absent (e.g. slim build)
    """
    import glob
    import importlib.util

    if _env.flag("FLAGOS_DISABLE_CUDA_ASSETS"):
        return

    lib_dir = os.path.join(_PACKAGE_DIR, "lib")
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
    if _platform.build_info().platform != "musa":
        return
    _env.set_foreign("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def validate_dcu_runtime() -> None:
    """Prove the DTK device libs really bound to the official core.

    Only runs for a decoupled DCU build (a DCU wheel without
    FLAGOS_DCU_VENDOR_CORE=1), right after `import torch`; the checks themselves
    live in accelerator/hygon/_dcu_runtime_check.py. Set
    FLAGOS_DCU_SKIP_RUNTIME_CHECK=1 to bypass (e.g. when deliberately testing a
    non-matching wheel pair).
    """
    if _platform.build_info().platform != "dcu":
        return
    if _env.flag("FLAGOS_DCU_SKIP_RUNTIME_CHECK"):
        return

    from torch_fl.accelerator.hygon._dcu_libtorch_link import vendor_core_mode

    if vendor_core_mode():
        return  # legacy mode replaces the core wholesale; nothing to reconcile.

    from torch_fl.accelerator.hygon._dcu_runtime_check import validate_decoupled_runtime

    validate_decoupled_runtime(os.path.join(_PACKAGE_DIR, "lib_hygon"))


def preload() -> None:
    """Select the runtime, load CUDA assets, then configure backend autoload."""
    _relink_vendor_libtorch()
    _preload_cuda_assets()
    _disable_vendor_backend_autoload()


def load_extension_dependencies(torch_file: str) -> None:
    """Expose the symbols used by the extension and external launchers."""
    # Ascend's Triton launcher resolves c10/ATen from the global namespace.
    torch_lib = os.path.join(os.path.dirname(torch_file), "lib")
    for name in ("libc10.so", "libtorch.so", "libtorch_cpu.so"):
        path = os.path.join(torch_lib, name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    # FlagGems' liboperators.so resolves GetCurrentStream from this library.
    stream_api = os.path.join(_PACKAGE_DIR, "lib", "libstream_api.so")
    if os.path.exists(stream_api):
        ctypes.CDLL(stream_api, mode=ctypes.RTLD_GLOBAL)
