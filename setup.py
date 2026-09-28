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

import glob
import importlib.machinery
import importlib.util
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# Accelerator platform: "cuda" (default), "ppu", "metax", "ascend",
# "tsingmicro", "dcu", "gcu", "musa", or "bpu"
FLAGOS_ACCELERATOR = os.environ.get("FLAGOS_ACCELERATOR", "cuda").lower()

# Where the sources live. Distinct from BASE_DIR, which is repointed at a
# scratch tree when the generated files are redirected somewhere else -- the
# files this build reads back (torch_fl/_env.py, cmake/flagos_platforms.json) do
# not move with them.
SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))

BASE_DIR = SOURCE_DIR

_PLATFORM_TABLE = None


def _platform_table() -> dict:
    """cmake/flagos_platforms.json: the platform/build matrix shared with CMake.

    CMakeLists.txt reads the same file, so the kernel switches, the
    bundled-libtorch dir, the platform list and the pins have one definition.
    """
    global _PLATFORM_TABLE
    if _PLATFORM_TABLE is None:
        path = os.path.join(SOURCE_DIR, "cmake", "flagos_platforms.json")
        with open(path, encoding="utf-8") as handle:
            _PLATFORM_TABLE = json.load(handle)
    return _PLATFORM_TABLE


def _platform_entry(accelerator: str) -> dict:
    """One accelerator's row from the shared table (raises on an unknown id)."""
    table = _platform_table()
    try:
        return table["accelerators"][accelerator]
    except KeyError:
        known = ", ".join(sorted(table["accelerators"]))
        raise ValueError(
            f"Unknown FLAGOS_ACCELERATOR '{accelerator}'. Expected one of: {known}"
        ) from None


# The kernel sets a build can compile in, and their build-record names, from the
# shared table. Each name is simultaneously an environment variable, a CMake
# option() and one entry of the generated build_config.KERNELS, so no
# env-name -> -D-name -> record-name translation exists to fall out of sync.
KERNEL_SWITCHES = tuple(_platform_table()["kernel_switches"])
KERNEL_SET_NAME = dict(_platform_table()["kernel_set_names"])

# Directory inside the wheel holding a bundled forked libtorch, for the backends
# that ship one (see scripts/vendor/bundle_*_libtorch.sh). "lib" means "no separate
# bundle dir": the CUDA backend drops its extra .so straight into torch_fl/lib/.
# From the table -- the same value CMake's FLAGOS_BUNDLE_LIBDIR gets. _C.so's
# RUNPATH has to reach the bundle or its auditwheel-mangled deps
# (libglog-*.so.0) go missing.
_BUNDLE_LIBDIR = _platform_entry(FLAGOS_ACCELERATOR)["bundle_libdir"]

# Only run cmake build for actual build commands, not metadata collection
BUILD_COMMANDS = {
    "build",
    "build_ext",
    "install",
    "develop",
    "bdist_wheel",
    "bdist_egg",
    "editable_wheel",
}
RUN_BUILD_DEPS = any(arg in BUILD_COMMANDS for arg in sys.argv)


def _ensure_metax_cudart_shim():
    """On MetaX, compile and load a complete cudart shim before importing torch.

    MetaX's libsymbol_cu.so provides CUDA runtime symbols but without the
    @@libcudart.so.12 version tags that PyTorch's .so files require.
    We build a single shared library (csrc/runtime/accelerator/metax/cudart_shim.c) that:
      1. Forwards ~79 symbols to libsymbol_cu.so via dlsym
      2. Stubs ~11 symbols for APIs missing from MetaX entirely
      3. Tags ALL exported symbols with @@libcudart.so.12 via a version script
    """
    import ctypes

    csrc = os.path.join(BASE_DIR, "csrc", "runtime", "accelerator", "metax")
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    shim_so = os.path.join(build_dir, "libcudart_shim.so")
    shim_src = os.path.join(csrc, "cudart_shim.c")
    version_script = os.path.join(csrc, "libcudart.version")

    inputs = [shim_src, version_script]
    if not os.path.exists(shim_so) or any(
        os.path.exists(s) and os.path.getmtime(s) > os.path.getmtime(shim_so)
        for s in inputs
    ):
        subprocess.check_call(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-o",
                shim_so,
                shim_src,
                f"-Wl,--version-script={version_script}",
                "-Wl,-soname,libcudart.so.12",
                "-ldl",
            ]
        )

    ctypes.CDLL(shim_so, mode=ctypes.RTLD_GLOBAL)


if FLAGOS_ACCELERATOR == "metax":
    _ensure_metax_cudart_shim()


def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


def get_pytorch_dir():
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def _cuda_toolkit_root() -> str | None:
    """Locate CUDA toolkit root (directory containing include/cuda_runtime.h)."""
    candidates: list[str] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(key)
        if val:
            candidates.append(val)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [
                os.path.join(conda_prefix, "targets", "x86_64-linux"),
                conda_prefix,
            ]
        )
    candidates.append("/usr/local/cuda")

    seen: set[str] = set()
    for root in candidates:
        root = os.path.realpath(root)
        if root in seen:
            continue
        seen.add(root)
        if os.path.isfile(os.path.join(root, "include", "cuda_runtime.h")):
            return root
    return None


def _find_nvcc(cuda_root: str) -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    for candidate in (
        os.path.join(cuda_root, "bin", "nvcc"),
        os.path.join(conda_prefix, "bin", "nvcc") if conda_prefix else None,
        shutil.which("nvcc"),
    ):
        if candidate and os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return None


def _prepend_env_path(env: dict, key: str, *paths: str) -> None:
    parts = [p for p in paths if p and os.path.isdir(p)]
    existing = env.get(key, "")
    if existing:
        parts.append(existing)
    if parts:
        env[key] = os.pathsep.join(parts)


def _pip_nvidia_include_dirs() -> list[str]:
    """Headers from pip nvidia-* wheels when conda toolkit is minimal."""
    import pathlib
    import site

    dirs: list[str] = []
    for sp in site.getsitepackages():
        nvidia = pathlib.Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            inc = pkg / "include"
            if inc.is_dir():
                dirs.append(str(inc))
    return dirs


def _setup_cuda_build_env(env: dict) -> str | None:
    """Export CUDA paths for cmake/nvcc (incl. conda pip wheel layout)."""
    cuda_root = _cuda_toolkit_root()
    if not cuda_root:
        return None

    env.setdefault("CUDA_HOME", cuda_root)
    env.setdefault("CUDA_PATH", cuda_root)
    _prepend_env_path(env, "CPATH", os.path.join(cuda_root, "include"))
    _prepend_env_path(env, "CPATH", *_pip_nvidia_include_dirs())
    _prepend_env_path(env, "LIBRARY_PATH", os.path.join(cuda_root, "lib"))
    _prepend_env_path(env, "LD_LIBRARY_PATH", os.path.join(cuda_root, "lib"))
    _prepend_env_path(env, "CMAKE_PREFIX_PATH", cuda_root)
    return cuda_root


def _find_nvrtc_library() -> str | None:
    try:
        import importlib.util
        import pathlib

        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if spec is None or not spec.origin:
            return None
        lib = pathlib.Path(spec.origin).resolve().parent / "lib" / "libnvrtc.so.12"
        return str(lib) if lib.is_file() else None
    except Exception:
        return None


def _append_cuda_cmake_args(cmake_args: list[str], cuda_root: str) -> None:
    nvcc = _find_nvcc(cuda_root)
    if nvcc:
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")
    cmake_args.append(f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_root}")
    nvrtc = _find_nvrtc_library()
    if nvrtc:
        cmake_args.append(f"-DCUDA_nvrtc_LIBRARY={nvrtc}")


def _find_flaggems_dir() -> str | None:
    env_dir = os.environ.get("FLAGGEMS_DIR")
    if env_dir and os.path.isfile(os.path.join(env_dir, "FlagGemsConfig.cmake")):
        return env_dir

    import site

    search_roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        search_roots.append(user_site)
    for sp in search_roots:
        cand = os.path.join(sp, "flag_gems", "lib", "cmake", "FlagGems")
        if os.path.isfile(os.path.join(cand, "FlagGemsConfig.cmake")):
            return cand
    return None


def _metax_path_from_env() -> str:
    return os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME") or "/opt/maca"


def _setup_metax_build_env(env: dict) -> str:
    """PATH/LD_LIBRARY_PATH for the MetaX SDK. Returns MACA_PATH.

    MetaX is a CUDA-boxing build: host g++ compiles the generated boxing kernels
    against maca's cu-bridge headers, so the SDK's include/lib directories have
    to be reachable. The mxcc/cucc device compiler belonged to the retired
    native-kernel path, so its absence is no longer an error here.
    """
    metax_path = _metax_path_from_env()
    cu_bridge = os.path.join(metax_path, "tools", "cu-bridge")
    if not os.path.isdir(os.path.join(cu_bridge, "include")):
        raise RuntimeError(f"MetaX cu-bridge headers not found: {cu_bridge}/include")

    env.setdefault("MACA_PATH", metax_path)
    env["PATH"] = os.pathsep.join(
        p
        for p in (
            os.path.join(cu_bridge, "bin"),
            os.path.join(metax_path, "bin"),
            os.path.join(metax_path, "mxgpu_llvm", "bin"),
            env.get("PATH", ""),
        )
        if p
    )
    ld_parts = [
        os.path.join(metax_path, "lib"),
        os.path.join(cu_bridge, "lib"),
        os.path.join(metax_path, "mxgpu_llvm", "lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in ld_parts if p)
    return metax_path


def _dtk_root() -> str:
    """Hygon DTK install root, from ROCM_PATH (what DTK's env.sh exports), else
    the default install location."""
    path = os.environ.get("ROCM_PATH")
    if path and os.path.isdir(path):
        return path
    default = "/opt/dtk"
    if not os.path.isdir(default):
        raise RuntimeError(
            "FLAGOS_ACCELERATOR=dcu selected, but no DTK installation was found. "
            "Source DTK's env.sh (which sets ROCM_PATH) or install DTK at /opt/dtk."
        )
    return default


# The values an explicit environment variable may not contradict, from the
# shared table's per-platform "pins". Each pin is a set(... CACHE BOOL ... FORCE)
# in CMakeLists.txt: the FORCE would win over the -D silently, so a build record
# derived from the requested value would describe a wheel that was never built.
# Asking for the other value is an error rather than a silent override, on both
# sides (here and in CMakeLists.txt).
#
# Only genuinely impossible combinations are pinned. metax, gcu, ppu and
# tsingmicro are not, because FLAGOS_BUILD_FLAGGEMS_CPP=OFF there is a default an
# explicit =1 (with a vendor-built liboperators.so) may replace -- MetaX's MACA
# build is the documented case. DCU, MUSA and BPU pin it off for the opposite
# reason: no FlagGems C++ build exists for DTK, the MUSA toolkit or the BPU at
# all, so there is nothing an explicit 1 could link against. Ascend/musa
# FLAGOS_BUILD_VENDOR is pinned because their whole kernel story is the vendor
# library.
def _pinned_kernel_switches(accelerator: str) -> dict[str, bool]:
    pins = _platform_entry(accelerator).get("pins", {})
    return {name: spec["value"] for name, spec in pins.items()}


_ENV_MODULE = None


def _env_module():
    """torch_fl/_env.py, loaded by path so the build side shares its truth table.

    setup.py cannot ``import torch_fl._env``: that would execute
    torch_fl/__init__.py, which imports torch. Loading the file directly costs
    nothing (the module is stdlib-only by design) and means a build with
    FLAGOS_BUILD_VENDOR=2 rejects it exactly the way the run time would, instead of
    the build keeping a second, more permissive copy of the parser.
    """
    global _ENV_MODULE
    if _ENV_MODULE is None:
        path = os.path.join(SOURCE_DIR, "torch_fl", "_env.py")
        loader = importlib.machinery.SourceFileLoader("torch_fl._env", path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        _ENV_MODULE = module
    return _ENV_MODULE


def _vendor_kernel_switches(accelerator: str) -> dict[str, bool]:
    """What one accelerator builds by default, before any explicit environment value.

    The middle of the three layers in _kernel_switches, and the only one that
    differs per vendor. The values -- and the per-platform rationale (why MetaX
    has no native kernels, why Ascend/GCU/MUSA turn boxing off, why BPU drops
    FlagGems, ...) -- live in cmake/flagos_platforms.json, next to the CMake side
    that consumes the same table.
    """
    return dict(_platform_entry(accelerator)["kernel_defaults"])


def _kernel_switches(accelerator: str) -> dict[str, bool]:
    """Resolve the five kernel-set switches for one accelerator.

    The single place that decides what a build compiles in. Both the -D list
    handed to CMake and the KERNELS tuple written into the build record come
    from here, so the wheel cannot be described as something it is not.

    Order: the CMake option() default, then what this accelerator builds by
    default, then an explicit environment value, then a pin. An explicit value
    that contradicts a pin raises rather than being dropped, because CMake would
    ignore it and the resulting record would be wrong.
    """
    resolved = dict.fromkeys(KERNEL_SWITCHES, True)
    resolved.update(_vendor_kernel_switches(accelerator))

    # The runtime's own accessor, not a second copy of its truth table: an
    # unparseable value (FLAGOS_BUILD_VENDOR=2) warns and keeps the vendor default
    # here for the same reason it does at run time, instead of the build
    # reading it as "on" and compiling a set the user never asked for.
    env = _env_module()
    pinned = _pinned_kernel_switches(accelerator)
    for name in KERNEL_SWITCHES:
        requested = env.flag(name, resolved[name])
        if name in pinned and requested != pinned[name]:
            raise ValueError(
                f"{name}={os.environ.get(name)} is not possible with "
                f"FLAGOS_ACCELERATOR={accelerator}: this platform always builds with "
                f"{name}={'ON' if pinned[name] else 'OFF'}"
            )
        resolved[name] = requested
    return resolved


def _cmake_build_jobs() -> int:
    """Parallel compile jobs for cmake/ninja. Set FLAGOS_BUILD_JOBS=1 to stay serial."""
    for key in ("FLAGOS_BUILD_JOBS", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            jobs = int(raw)
            if jobs < 1:
                raise ValueError(f"{key} must be >= 1, got {raw!r}")
            return jobs
    return multiprocessing.cpu_count()


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(os.path.join(BASE_DIR, "torch_fl")),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
        # CMake probes the environment for optional packages (torch_musa,
        # flag_gems). It must use *this* interpreter, not whatever python is
        # first on PATH, or the probe reads a different site-packages than the
        # one we are building against.
        "-DPYTHON_EXECUTABLE=" + sys.executable,
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
    ]

    cmake_args.append(f"-DFLAGOS_ACCELERATOR={FLAGOS_ACCELERATOR}")

    # What gets compiled in, resolved in one place so the wheel can describe
    # itself accurately. _kernel_switches() is also what _write_build_config()
    # reads, so the -D list and the KERNELS record cannot disagree.
    #
    # The per-vendor rationale (why MetaX has no native kernels, why Ascend
    # turns boxing off, ...) lives in cmake/flagos_platforms.json.
    kernels = _kernel_switches(FLAGOS_ACCELERATOR)
    for switch in KERNEL_SWITCHES:
        cmake_args.append(f"-D{switch}={'ON' if kernels[switch] else 'OFF'}")

    build_env = os.environ.copy()
    build_jobs = _cmake_build_jobs()
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_jobs)
    cmake = "cmake"

    # FlagGems C++ library path (optional, enables low-overhead C++ dispatch)
    flaggems_dir = os.environ.get("FLAGGEMS_DIR")
    if flaggems_dir:
        cmake_args.append(f"-DFlagGems_DIR={flaggems_dir}")
    flaggems_source_dir = os.environ.get("FLAGGEMS_SOURCE_DIR")
    if flaggems_source_dir:
        cmake_args.append(f"-DFLAGGEMS_SOURCE_DIR={flaggems_source_dir}")

    if FLAGOS_ACCELERATOR == "metax":
        metax_path = _setup_metax_build_env(build_env)
        cmake_args.append(f"-DMACA_PATH={metax_path}")
        cmake_args.append("-G")
        cmake_args.append("Ninja")
    elif FLAGOS_ACCELERATOR == "cuda":
        cuda_root = _setup_cuda_build_env(build_env)
        if cuda_root:
            _append_cuda_cmake_args(cmake_args, cuda_root)
        flaggems_dir = _find_flaggems_dir()
        if flaggems_dir:
            cmake_args.append(f"-DFLAGGEMS_DIR={flaggems_dir}")
    elif FLAGOS_ACCELERATOR == "dcu":
        cmake_args.append(f"-DROCM_PATH={_dtk_root()}")

    subprocess.check_call([cmake, BASE_DIR] + cmake_args, cwd=build_dir, env=build_env)

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--config",  # For multi-config generators
        "Release",
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(build_jobs)]
    else:
        build_args += ["-j", str(build_jobs)]

    subprocess.check_call([cmake] + build_args, cwd=build_dir, env=build_env)
    _verify_built_native_libs()
    _bundle_cuda_assets()
    _write_build_config()


def _write_build_config() -> None:
    """Record what this wheel was built for.

    torch_fl._select_backend_config() runs at import time, before `import torch`,
    so it cannot sniff torch.version.hip to tell a DCU build apart. Recording the
    accelerator as ACCELERATOR here lets it pick backends_dcu.conf without the
    user having to keep FLAGOS_ACCELERATOR set at run time.

    The two attribute names are the record's own vocabulary, not the switches':
    ACCELERATOR holds a platform value and KERNELS holds kernel-set values, while
    the build inputs a user sets are FLAGOS_ACCELERATOR and the five
    FLAGOS_BUILD_* flags.

    KERNELS records the compiled kernel sets, because the accelerator alone does
    not describe a wheel: two MetaX wheels built with different kernel-set flags
    share a FLAGOS_ACCELERATOR and are not the same wheel. Run-time code that
    used to infer the kernel set from an environment variable reads this
    instead, so there is nothing left to keep in sync by hand.

    Both values come from _kernel_switches(), the same function whose output
    becomes the -D list handed to CMake, so this file cannot describe a build
    other than the one that produced it.
    """
    kernels = _kernel_switches(FLAGOS_ACCELERATOR)
    compiled = tuple(sorted(KERNEL_SET_NAME[s] for s in KERNEL_SWITCHES if kernels[s]))
    body = ", ".join(f'"{name}"' for name in compiled)
    if len(compiled) == 1:
        body += ","
    path = os.path.join(BASE_DIR, "torch_fl", "_build_config.py")
    content = (
        "# AUTO-GENERATED by setup.py at build time. Do not edit.\n"
        f'ACCELERATOR = "{FLAGOS_ACCELERATOR}"\n'
        f"KERNELS = ({body})\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_compatibility_manifest(wheel_version: str) -> None:
    """Write artifact facts after the native build, before wheel staging."""
    import torch

    # setuptools' PEP 517 backend executes setup.py with the source root absent
    # from sys.path, even for --no-isolation builds. Load this source file by
    # absolute path instead of relying on an import that only works in a shell.
    preflight_path = os.path.join(SOURCE_DIR, "scripts", "tools", "torch-fl-preflight")
    loader = importlib.machinery.SourceFileLoader("torch_fl_preflight", preflight_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load wheel preflight from {preflight_path}")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    kernels = _kernel_switches(FLAGOS_ACCELERATOR)
    compiled = [KERNEL_SET_NAME[name] for name in KERNEL_SWITCHES if kernels[name]]
    manifest = preflight.make_manifest(
        platform=FLAGOS_ACCELERATOR,
        wheel_version=wheel_version,
        kernels=compiled,
        bundle_libdir=_BUNDLE_LIBDIR,
        vendor_torch_libraries=_platform_entry(FLAGOS_ACCELERATOR)[
            "vendor_torch_libraries"
        ],
        requirements=_install_requires(),
        torch_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        sdk_version=os.environ.get("FLAGOS_SDK_VERSION"),
        vendor_torch_version=os.environ.get("FLAGOS_VENDOR_TORCH_VERSION"),
    )
    if manifest["build"]["distributions"]["torch"] is None:
        raise RuntimeError("Build-time PyTorch distribution is missing")
    path = os.path.join(BASE_DIR, "torch_fl", "compatibility.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _bundle_cuda_assets() -> None:
    """Copy the external CUDA .so assets into torch_fl/lib so the wheel is
    self-contained.

    torch_fl's CUDA backend reuses PyTorch's registered CUDA kernels via an
    externally-supplied libtorch_cuda.so (CPU-only pip torch does not ship it).
    Historically this was LD_PRELOAD-ed by scripts/vendor/with_cuda_libtorch.sh; for a
    single self-contained wheel we bundle the assets and preload them from
    torch_fl/__init__.py before `import torch` (see that doc, constraint 1).
    CUDA only.

    Set FLAGOS_SKIP_CUDA_ASSETS=1 to skip (e.g. a slim build for a machine that
    supplies libtorch_cuda.so out-of-band).
    """
    if FLAGOS_ACCELERATOR != "cuda":
        return
    if os.environ.get("FLAGOS_SKIP_CUDA_ASSETS", "0") == "1":
        return
    assets_dir = os.environ.get(
        "FLAGOS_CUDA_ASSETS_DIR",
        os.path.join(BASE_DIR, ".libtorch_cuda_assets"),
    )
    if not os.path.isdir(assets_dir):
        print(
            f"[setup] warning: CUDA assets dir {assets_dir} not found; wheel "
            "will require an externally-supplied libtorch_cuda.so at runtime."
        )
        return
    dst_dir = os.path.join(BASE_DIR, "torch_fl", "lib")
    os.makedirs(dst_dir, exist_ok=True)
    import glob

    copied = []
    for src in sorted(glob.glob(os.path.join(assets_dir, "*.so*"))):
        dst = os.path.join(dst_dir, os.path.basename(src))
        # Skip if already present and identical size (avoid re-copying ~1GB).
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            copied.append(os.path.basename(src))
            continue
        shutil.copy2(src, dst)
        copied.append(os.path.basename(src))
    if copied:
        print(f"[setup] bundled CUDA assets into torch_fl/lib: {', '.join(copied)}")


_NCCL_EXT_BUILD_PATH = os.path.join(
    SOURCE_DIR, "torch_fl", "comm", "_nccl_ext", "build.py"
)


def _load_nccl_ext_builder():
    """Import torch_fl/comm/_nccl_ext/build.py without importing torch_fl.

    Loaded from its path rather than as ``torch_fl.comm._nccl_ext.build``: that
    form imports torch_fl, which imports torch, which -- with device-backend
    autoloading on -- imports the half-built native package this build is in the
    middle of producing. build.py is stdlib-only at import time for exactly this
    reason; tests/unit/test_nccl_ext_build.py loads it the same way.
    """
    spec = importlib.util.spec_from_file_location(
        "torch_fl_nccl_ext_build", _NCCL_EXT_BUILD_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load the _flagos_nccl builder at {_NCCL_EXT_BUILD_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _nccl_ext_extension():
    """The ``_flagos_nccl`` Extension for this accelerator, or None.

    None means the accelerator does not wire the native comm bridge into its
    wheel (the builder's docstring names the DCU-only scope, and why); the
    standalone builder still covers that platform. A configuration error on an
    accelerator that *is* wired is raised instead of skipped: a released DCU
    wheel without this extension is the defect, so an unresolvable DTK link set
    has to stop the build rather than quietly ship a wheel whose
    ProcessGroupFlagOS has no native RCCL fallback.

    Called from _get_setup_kwargs() under RUN_BUILD_DEPS, so a real build pass
    always reaches it. That is deliberate -- the DTK link set is resolvable
    exactly when a DCU build is possible at all, and setup.py must not be able
    to describe a DCU wheel it cannot build.
    """
    if not os.path.isfile(_NCCL_EXT_BUILD_PATH):
        return None
    builder = _load_nccl_ext_builder()
    try:
        extension = builder.wheel_extension()
    except RuntimeError as exc:
        raise RuntimeError(
            f"FLAGOS_ACCELERATOR={FLAGOS_ACCELERATOR} selected, but the "
            f"_flagos_nccl native comm bridge cannot be configured: {exc}"
        ) from exc
    if extension is not None:
        print(f"[setup] native comm bridge target: {extension.name}")
    return extension


def _verify_built_native_libs() -> None:
    lib = os.path.join(BASE_DIR, "torch_fl", "lib", "libtorch_fl.so")
    if not os.path.isfile(lib):
        raise RuntimeError(
            f"Native build finished but {lib} is missing. "
            "Check cmake/ninja output above."
        )
    if FLAGOS_ACCELERATOR == "dcu":
        shim = os.path.join(
            BASE_DIR, "torch_fl", "lib_dcu", "libflagos_dtk_core_compat.so"
        )
        if not os.path.isfile(shim):
            raise RuntimeError(
                f"DCU native build finished but {shim} is missing. The shim is "
                "required when DTK's device libraries run on the official "
                "PyTorch core. Check the flagos_dtk_core_compat cmake target."
            )
        return
    if FLAGOS_ACCELERATOR != "metax":
        return
    try:
        undef = subprocess.check_output(
            ["nm", "-u", lib], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return
    if "get_maca_enable_elementwise_kernel_info" in undef:
        raise RuntimeError(
            f"{lib} still references at::maca::* (mcPytorch). Remove build/ and "
            "torch_fl/lib/*.so, then rebuild with FLAGOS_ACCELERATOR=metax."
        )


class BuildExtWithCmake(_build_ext):
    """Run cmake before setuptools builds torch_fl._C."""

    def _prepare_nccl_extension(self):
        """Finish the torch_fl.comm._nccl_ext._flagos_nccl target in place.

        The target itself is appended in _get_setup_kwargs() so setuptools'
        bookkeeping (finalize_options' ext_map, _needs_stub, get_outputs) sees
        it. What can only be added here is torch's own include and library
        paths, which belong to the interpreter build_deps() has just run with.
        """
        builder = _load_nccl_ext_builder()
        for extension in self.extensions:
            if extension.name != builder.WHEEL_EXTENSION_NAME:
                continue
            from torch.utils.cpp_extension import include_paths, library_paths

            # Torch's paths go last so DTK's cuda.h/nccl.h win over any
            # torch-shipped copy, the ordering build.py's CppExtension gives.
            extension.include_dirs += include_paths()
            extension.library_dirs += library_paths()
            # The c10/torch/torch_cpu the factory's pybind module links against,
            # mirroring CppExtension in torch_fl/comm/_nccl_ext/build.py.
            extension.libraries += ["c10", "torch", "torch_cpu"]
            print(
                "[setup] _flagos_nccl link set: "
                + ", ".join(extension.libraries)
                + " | "
                + ", ".join(extension.extra_link_args)
            )
            return

    def run(self):
        build_deps()
        _write_compatibility_manifest(self.distribution.get_version())
        self._prepare_nccl_extension()
        # ``build`` runs build_py before build_ext, but CMake installs package
        # data into torch_fl/ during build_ext. Setuptools caches build_py's file
        # list, so copy late-generated files explicitly into wheel staging.
        relative_paths = [
            "_build_config.py",
            "compatibility.json",
            "lib/flagos_platform",
            "include/flagos.h",
        ]
        patterns = ["lib/*.so*", "lib/*.dylib*", "lib/*.dll", "lib/*.lib"]
        if FLAGOS_ACCELERATOR == "metax":
            patterns.extend(("lib_maca/*.so*", "lib_maca/vendor_version.py"))
        if FLAGOS_ACCELERATOR == "ppu":
            patterns.extend(("lib_ppu/*.so*", "lib_ppu/vendor_version.py"))
        if FLAGOS_ACCELERATOR == "dcu":
            # cmake installs the core-ABI shim straight into lib_dcu during
            # build_ext. build_py has already cached its file list by then, so
            # copy it (and any pre-bundled device assets) into wheel staging just
            # like the native libs under lib/.
            patterns.extend(("lib_dcu/*.so*", "lib_dcu/vendor_version.py"))
        for pattern in patterns:
            relative_paths.extend(
                os.path.relpath(path, os.path.join(BASE_DIR, "torch_fl"))
                for path in glob.glob(os.path.join(BASE_DIR, "torch_fl", pattern))
            )
        for relative_path in relative_paths:
            source = os.path.join(BASE_DIR, "torch_fl", relative_path)
            if os.path.isfile(source):
                destination = os.path.join(self.build_lib, "torch_fl", relative_path)
                self.mkpath(os.path.dirname(destination))
                self.copy_file(source, destination)
        super().run()


class EditableWheelWithCmake(_editable_wheel):
    """PEP 660 editable installs must build native libs (pip often skips build_ext)."""

    def run(self):
        self.run_command("build_ext")
        super().run()


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_fl/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_fl")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def _extension_rpath_args():
    """RUNPATH for torch_fl._C: torch_fl/lib plus the bundle dir when separate.

    _C.so links libtorch_bindings.so out of torch_fl/lib, which in turn pulls the
    bundled vendor libtorch and its auditwheel-mangled deps out of the bundle dir.
    Without the second entry a self-contained wheel fails at import with e.g.
    "libglog.so.0: cannot open shared object file".
    """
    args = make_relative_rpath_args("lib")
    if _BUNDLE_LIBDIR != "lib":
        args += make_relative_rpath_args(_BUNDLE_LIBDIR)
    return args


def _extension_compile_args():
    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = [
            "/NODEFAULTLIB:LIBCMT.LIB"
        ] + _extension_rpath_args()
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = _extension_rpath_args()
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            "-fno-strict-aliasing",
        ]
    return extra_link_args, extra_compile_args


def _get_setup_kwargs():
    extra_link_args, extra_compile_args = _extension_compile_args()
    ext_modules = [
        Extension(
            name="torch_fl._C",
            sources=["torch_fl/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_fl/lib")],
            extra_link_args=extra_link_args,
        )
    ]
    # torch_fl.comm._nccl_ext._flagos_nccl: the factory that exposes the
    # vendor's ProcessGroupNCCL to a CPU-only torch wheel. Added to ext_modules
    # here rather than at build_ext time so setuptools' own bookkeeping --
    # finalize_options' ext_map/_needs_stub, get_outputs, and the inplace copy
    # back into torch_fl/comm/_nccl_ext/ -- sees the target. torch's include and
    # library paths are added later, in BuildExtWithCmake (they need the
    # interpreter build_deps() runs with).
    #
    # Gated on RUN_BUILD_DEPS: resolving the DCU link set reads DTK paths that
    # only a build environment has, so a metadata-only pass (egg_info, sdist,
    # --version) must not need them.
    if RUN_BUILD_DEPS:
        nccl_ext = _nccl_ext_extension()
        if nccl_ext is not None:
            ext_modules.append(nccl_ext)

    package_data = {
        "torch_fl": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
            # Self-contained wheels: the vendor's forked libtorch C++ .so bundled
            # here so the process loads that C++ runtime without a separate
            # vendor torch wheel (see scripts/vendor/bundle_*_libtorch.sh, and
            # torch_fl/accelerator/_vendor_libtorch.py for private selection).
            # The trailing * matters for lib_dcu: DTK's auditwheel-mangled
            # torch.libs deps end in a version suffix (libglog-6ed04f2c.so.0.0.0).
            "lib_maca/*.so*",
            "lib_maca/vendor_version.py",
            "lib_dcu/*.so*",
            # DTK torch's own version.py, carried so _restore_dcu_hip_version()
            # can hand triton's hcu backend the hip/rocm strings the stock +cpu
            # torch in front does not have. Needed explicitly: the globs above
            # only match *.so*.
            "lib_dcu/vendor_version.py",
            "lib_ppu/*.so*",
            "lib_ppu/vendor_version.py",
            "include/*.h",
            # The native comm bridge's source, so the standalone builder
            # (comm/_nccl_ext/build.py) can still rebuild the extension from an
            # installed wheel. build.py itself ships with the package; the .cpp
            # next to it does not, and a release that omitted it left users
            # unable to rebuild _flagos_nccl in place.
            "comm/_nccl_ext/*.cpp",
            # The DTK-private symbol manifest that libflagos_dtk_core_compat.so
            # must export, shipped so an installed wheel can be re-audited with
            # scripts/vendor/check_dcu_core_abi.py against a different DTK release.
            "accelerator/dcu/dtk_core_compat_symbols.txt",
            # All backend configs, not just the default: per-op routing tables
            # read by torch_fl at import (backends_cuda.conf /
            # backends_metax.conf, ...). Now consolidated under configs/.
            "configs/backends*.conf",
            "codegen_skip_ops.txt",
            "compatibility.json",
        ]
    }

    # The project version names the PyTorch minor line the wheel binds to, so
    # torch_fl 2.10.0 is the release built against torch 2.10.x. The generated
    # ATen bindings are tied to that same line (TORCH_PIN below), which makes a
    # wheel unusable on any other one; carrying the line in the version keeps
    # that visible from the filename alone. Bumping it is a deliberate act that
    # belongs with a codegen regeneration, not a routine edit.
    version = "2.10.0"
    # A local version segment tags which vendor a self-contained wheel bundles a
    # forked libtorch for. That bundle is SDK-version-bound whether we say so or
    # not -- DTK's libtorch_hip.so has librocblas.so.4 written into its
    # DT_NEEDED -- so making the binding visible in the filename is strictly
    # better than leaving two incompatible wheels both called 2.10.0. Override
    # with FLAGOS_WHEEL_LOCAL to pin the exact SDK, e.g.
    # FLAGOS_WHEEL_LOCAL=metax3.8.1 / FLAGOS_WHEEL_LOCAL=dtk2604.
    _default_local = _platform_entry(FLAGOS_ACCELERATOR)["wheel_local"] or None
    local = os.environ.get("FLAGOS_WHEEL_LOCAL", _default_local)
    if local:
        version = f"{version}+{local}"

    return dict(
        name="torch_fl",
        version=version,
        description="FlagGems operators as a custom PyTorch device (flagos)",
        author="FlagGems Team",
        packages=find_packages(
            include=["torch_fl*", "accelerator*", "csrc.runtime.accelerator*"]
        ),
        scripts=["scripts/tools/torch-fl-preflight"],
        package_dir={"": "."},
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": BuildExtWithCmake,
            "editable_wheel": EditableWheelWithCmake,
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
        python_requires=">=3.8",
        install_requires=_install_requires(),
        # No extras_require here: pyproject.toml's
        # [project.optional-dependencies] owns that table and setuptools reports
        # `extras_require` overwritten in `pyproject.toml` when both are given,
        # so a `cuda` extra declared here was never in any artifact. The CUDA
        # runtime dependencies are hard requirements on that platform anyway
        # (_install_requires), and `pip install torch_fl[cuda]` is unaffected --
        # it never resolved to anything.
    )


# NVIDIA CUDA runtime libs that the bundled libtorch_cuda.so (cu12.x) links
# against. Pinned to the cu12 major sonames it needs (libcudart.so.12,
# libcublas.so.12, libcudnn.so.9, libnvshmem_host.so.3, ...). Lower bounds keep
# pip free to resolve a compatible patch; the bundled .so was built against the
# cu12.8 wheels present in the build env.
_CUDA_RUNTIME_DEPS = [
    "nvidia-cuda-runtime-cu12>=12.8",
    "nvidia-cublas-cu12>=12.8",
    "nvidia-cudnn-cu12>=9.0",
    "nvidia-cuda-nvrtc-cu12>=12.8",
    "nvidia-cufft-cu12>=11.0",
    "nvidia-curand-cu12>=10.0",
    "nvidia-cusolver-cu12>=11.0",
    "nvidia-cusparse-cu12>=12.0",
    "nvidia-cusparselt-cu12>=0.7",
    "nvidia-nccl-cu12>=2.20",
    "nvidia-nvtx-cu12>=12.8",
    "nvidia-cuda-cupti-cu12>=12.8",
    "nvidia-nvjitlink-cu12>=12.8",
    "nvidia-nvshmem-cu12>=3.0",
]


# torch_fl is a thin build on three packages that live outside this repository:
# FlagTree (the Triton build carrying the vendor backend), FlagGems (the operator
# source) and FlagCX (the distributed backend). Omitting them is not a crash --
# every import in the Python layer is guarded -- but the result is a flagos device
# that routes no operator and a distributed path staged through the host, so the
# wheel declares the exact versions it was built and measured against.
#
# The versions come from .github/version-pins.env, the file the CI setup scripts
# source, rather than a second copy here. Two things that file settles and a
# guessed range does not:
#
#   * flagtree and flagcx are not on PyPI at all, and flag_gems on a generic
#     index resolves to an older cohort (5.0.x) than the one the per-op routing
#     tables in torch_fl/configs/backends_*.conf were generated against, so a
#     floor like `flag_gems>=5.0.2` is satisfied by a build this wheel was never
#     measured on;
#   * FlagTree is per-platform -- the same package name carries the vendor's own
#     Triton backend (0.7.0rc2+hcu3.6 for DCU, 0.7.0rc3+metax3.6 for MetaX) -- so
#     a plain `triton` requirement is the wrong shape. Declaring triton next to a
#     vendor FlagTree is what produces Ascend's "0 active drivers" failure.
#
# Resolution needs an index that carries both locations: FlagTree is published to
# flagos-pypi-hosted while flag_gems and flagcx sit in the per-vendor lane
# (flagos-pypi-<vendor>), which is exactly how the CI scripts are configured
# (FLAGTREE_INDEX_URL and FLAGGEMS_INDEX_URL in .github/scripts/hooks/set_env_*.sh).
# A single --index-url therefore has to name a group repository containing both,
# plus a proxy for PyPI and one for download.pytorch.org/whl/cpu.
VERSION_PINS = os.path.join(SOURCE_DIR, ".github", "version-pins.env")


def _version_pins() -> dict:
    """The KEY=VALUE assignments of .github/version-pins.env.

    Read as data rather than sourced. The file says of itself that it is "a plain
    assignment file meant to be sourced" -- no `export`, no quoting and no value
    that expands a variable -- so a parser here is the same read the CI scripts
    do, and the two cannot drift into disagreeing about a pin.
    """
    if not os.path.isfile(VERSION_PINS):
        raise RuntimeError(
            f"{VERSION_PINS} is missing, so the FlagTree/FlagGems/FlagCX versions "
            "this wheel must declare are unknown. They cannot be guessed: the "
            "package names are vendor- and cohort-specific. Build from a checkout "
            "of the repository."
        )
    pins = {}
    with open(VERSION_PINS, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            pins[key.strip()] = value.strip().strip("\"'")
    return pins


def _flagos_sibling_requires() -> list:
    """FlagTree, FlagGems and FlagCX at the versions this wheel was built against."""
    pins = _version_pins()
    reqs = []
    flagtree = pins.get(f"FLAGTREE_VERSION_{FLAGOS_ACCELERATOR}")
    if flagtree:
        # FlagTree *is* Triton -- it installs a `triton` distribution -- so
        # pinning it is what keeps a stock NVIDIA triton wheel out of the
        # environment on every platform that has a FlagTree build.
        reqs.append(f"flagtree=={flagtree}")
    else:
        # No FlagTree build is published for this platform, so Triton has to
        # come from somewhere else. tsingmicro and bpu are the two.
        reqs.append("triton>=3.5.1")
    flag_gems = pins.get("FLAGGEMS_VERSION_DEFAULT")
    if not flag_gems:
        raise RuntimeError(f"{VERSION_PINS} defines no FLAGGEMS_VERSION_DEFAULT")
    reqs.append(f"flag_gems=={flag_gems}")
    flagcx = pins.get(f"FLAGCX_VERSION_{FLAGOS_ACCELERATOR}")
    if flagcx:
        # Only the platforms whose vendor runtime FlagCX has a build for. The
        # others fall back to the NCCL-shaped path, so there is nothing to pin.
        reqs.append(f"flagcx=={flagcx}")
    return reqs


# The checked-in csrc/aten/generated/* bindings are generated against a
# specific ATen surface, so torch is pinned to the 2.10 series rather than left
# open. Newer torch drifts from those bindings, and a mismatch shows up as a
# wall of compile errors at build time rather than a clean resolver failure --
# the pin is what turns that into an install-time message. Moving to a newer
# torch is a deliberate act: re-run scripts/codegen/codegen_ops.py, do not hand-edit
# the generated files.
TORCH_PIN = "torch>=2.10,<2.11"


def _install_requires():
    reqs = [TORCH_PIN, "packaging>=23"]
    reqs += _flagos_sibling_requires()
    # For a CUDA wheel we bundle libtorch_cuda.so and preload it at import; it
    # needs the NVIDIA runtime libs present, so make them hard deps. Ascend/MetaX
    # builds do not (they supply their own runtime), so keep it CUDA-only.
    #
    # FLAGOS_SKIP_CUDA_ASSETS=1 means we do NOT bundle libtorch_cuda.so (the same
    # switch _bundle_cuda_assets() honors). That is the PPU case: the active torch
    # is already a CUDA-enabled build (CUDA 13, PPU_SDK/CUDA_SDK supplies the
    # runtime), so the pinned nvidia-*-cu12 wheels are both mismatched and
    # unnecessary. Skip them so `pip install` does not drag in cu12 packages.
    skip_assets = os.environ.get("FLAGOS_SKIP_CUDA_ASSETS", "0") == "1"
    if FLAGOS_ACCELERATOR == "cuda" and not skip_assets:
        reqs += _CUDA_RUNTIME_DEPS
    return reqs


# PEP 517 / pip install -e loads setup.py as a script; setup() must run at import time
# so cmdclass (build_ext / editable_wheel) is registered. Do not hide setup() in main().
setup(**_get_setup_kwargs())
