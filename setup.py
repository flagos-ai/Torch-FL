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

# Directory inside the wheel holding a bundled forked libtorch, for the backends
# that ship one (see scripts/vendor/bundle_*_libtorch.sh). "lib" means "no separate
# bundle dir": the CUDA backend drops its extra .so straight into torch_fl/lib/.
# Must match FLAGOS_BUNDLE_LIBDIR in CMakeLists.txt -- _C.so's RUNPATH has to
# reach the bundle or its auditwheel-mangled deps (libglog-*.so.0) go missing.
_BUNDLE_LIBDIR = {"metax": "lib_maca", "dcu": "lib_dcu", "ppu": "lib_ppu"}.get(
    FLAGOS_ACCELERATOR, "lib"
)

# Where the sources live. Distinct from BASE_DIR, which is repointed at a
# scratch tree when the generated files are redirected somewhere else -- the
# files this build reads back (torch_fl/_env.py) do not move with them.
SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))

BASE_DIR = SOURCE_DIR

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


# The kernel sets a build can compile in. Each name is simultaneously an
# environment variable, a CMake option() and one entry of the generated
# build_config.KERNELS, so no env-name -> -D-name -> record-name translation
# exists to fall out of sync. csrc/CMakeLists.txt declares the same five.
KERNEL_SWITCHES = (
    "FLAGOS_BUILD_VENDOR",
    "FLAGOS_BUILD_FLAGGEMS",
    "FLAGOS_BUILD_BOXING",
    "FLAGOS_BUILD_FLAGGEMS_CPP",
    "FLAGOS_BUILD_TILEOPS",
)

# What the build record calls the same five sets. The accelerator alone cannot
# answer "was the FlagGems C++ wrapper compiled into this wheel?" -- two MetaX
# wheels built with different kernel-set flags share a FLAGOS_ACCELERATOR and
# are not the same wheel.
KERNEL_SET_NAME = {
    "FLAGOS_BUILD_VENDOR": "vendor",
    "FLAGOS_BUILD_FLAGGEMS": "flaggems",
    "FLAGOS_BUILD_BOXING": "boxing",
    "FLAGOS_BUILD_FLAGGEMS_CPP": "flaggems_cpp",
    "FLAGOS_BUILD_TILEOPS": "tileops",
}

# The values an explicit environment variable may not contradict, because the
# platform cannot produce the other answer at all. Each entry mirrors a
# set(... CACHE BOOL ... FORCE) in CMakeLists.txt: the FORCE would win over the
# -D silently, so a build record derived from the requested value would describe
# a wheel that was never built. Asking for one is an error rather than a silent
# override, on both sides (here and in CMakeLists.txt).
#
# Only genuinely impossible combinations are listed. metax, gcu, ppu and
# tsingmicro are absent because FLAGOS_BUILD_FLAGGEMS_CPP=OFF there is a default
# that an explicit FLAGOS_BUILD_FLAGGEMS_CPP=1 (with a vendor-built
# liboperators.so) may replace -- MetaX's MACA build is the documented case.
# DCU, MUSA and BPU pin it off for the opposite reason: no FlagGems C++ build
# exists for DTK, for the MUSA toolkit or for the BPU at all, so there is nothing
# an explicit 1 could link against. Ascend/musa FLAGOS_BUILD_VENDOR is pinned
# because their whole kernel story is the vendor library: Ascend has no CUDA
# boxing runtime to fall back to, and MUSA's toolbox ships no CUDA runtime.
_PINNED_KERNEL_SWITCHES: dict[str, dict[str, bool]] = {
    "ascend": {"FLAGOS_BUILD_VENDOR": True},
    "musa": {"FLAGOS_BUILD_VENDOR": True, "FLAGOS_BUILD_FLAGGEMS_CPP": False},
    "dcu": {"FLAGOS_BUILD_FLAGGEMS_CPP": False},
    "bpu": {"FLAGOS_BUILD_FLAGGEMS_CPP": False},
}

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
    differs per vendor. The comments here are the whole build-side policy: there
    is no mode variable, because what a wheel can run follows from what it
    compiled in.
    """
    vendor: dict[str, bool] = {}

    if accelerator != "cuda":
        # TileOPs is TileLang on SM90 (Hopper) NVIDIA parts only. Its stubs are
        # harmless on other vendors -- the shims fall back to aten -- but there
        # is nothing for them to reach, so keep them out of vendor wheels. An
        # explicit FLAGOS_BUILD_TILEOPS=1 still wins.
        vendor["FLAGOS_BUILD_TILEOPS"] = False

    if accelerator not in ("cuda", "tsingmicro"):
        # The FlagGems C++ wrappers link a liboperators.so built for the vendor;
        # upstream's CUDA/ROCm build is the only one that exists by default, so
        # every other platform keeps them off and links cleanly without it. An
        # explicit FLAGOS_BUILD_FLAGGEMS_CPP=1 with FLAGGEMS_DIR=<vendor build>
        # still wins -- MetaX's MACA build is the documented case. TsingMicro is
        # exempt because it has no vendor branch at all and has always taken the
        # CUDA default here.
        vendor["FLAGOS_BUILD_FLAGGEMS_CPP"] = False

    if accelerator == "metax":
        # MetaX is a CUDA-boxing build: no native mxcc kernels compile in, the
        # generated boxing kernels provide acceleration, and the FlagGems C++
        # wrappers stay off (no MACA-built liboperators). The hand-written
        # kernels under backends/metax/ are not built -- the dead native path is
        # retired.
        #
        # FLAGOS_BUILD_FLAGGEMS stays at the CUDA default: the boxing wheel also
        # compiles the FlagGems Python-path kernels (flagos_python backend) so
        # the conf can route to them. python_op_caller links torch_python_library
        # (already in the metax link set) and adds nothing to the bundled wheel
        # size. FLAGOS_BUILD_FLAGGEMS=0 gives a slim pure-boxing build.
        #
        # FLAGOS_BUILD_FLAGGEMS_CPP needs a FlagGems built for MACA, which is a separate
        # build:
        #     cd FlagGems/cpp && cmake -B build-maca -DFLAGGEMS_BUILD_C_EXTENSIONS=ON \
        #         -DFLAGGEMS_BACKEND=MACA -DMACA_PATH=/opt/maca
        # Opt in with FLAGOS_BUILD_FLAGGEMS_CPP=1 FLAGGEMS_DIR=<that build dir>. The C++
        # kernels reach the device via the same DeviceBoxingGuard as the boxing
        # path, so they need the boxing kernels.
        vendor["FLAGOS_BUILD_VENDOR"] = False
    elif accelerator == "ascend":
        # Ascend uses ACLNN as the native fallback, with the FlagGems Python path
        # enabled by default. FlagGems runs on FlagTree (the vendor's Triton 3.5
        # build), not on triton-ascend: torch_fl imports before triton and carries
        # a torch_npu-free backend policy for it, so nothing here links torch_npu.
        # The generated Ascend conf is FlagGems-first for measured routes, while
        # unsupported or unregistered operators remain on ACLNN/CPU fallback.
        #
        # FLAGOS_BUILD_BOXING=OFF: Ascend is not a CUDA-compatible vendor, and its conf
        # routes nothing to the boxing kernel (0 `= cuda` entries), so compiling
        # the generated CUDA wrappers was pure dead weight -- the same reason
        # gcu/musa exclude them.
        vendor.update(
            {
                "FLAGOS_BUILD_BOXING": False,
                "FLAGOS_BUILD_FLAGGEMS": True,
                "FLAGOS_BUILD_VENDOR": True,
            }
        )
    # tsingmicro needs no branch: no vendor backend directory exists, so
    # FLAGOS_BUILD_VENDOR is a no-op, and boxing plus FlagGems Python stay on by
    # default.
    elif accelerator == "dcu":
        # Boxing build. The DCU torch wheel is a hipified build whose HIP kernels
        # are registered under the CUDA dispatch key, so the generated
        # PrivateUse1 -> CUDA boxing kernels reach them with no hand-written
        # kernels of our own.
        #
        # FLAGOS_BUILD_FLAGGEMS stays on, same as metax/cuda: DTK ships a working
        # triton (hcu backend) that flag_gems runs on, so the choice becomes a
        # runtime one (backends_dcu.conf). python_op_caller links
        # torch_python_library, already in the link set, so this adds nothing to
        # the wheel size. FLAGOS_BUILD_FLAGGEMS=0 gives a slim pure-boxing build.
        pass
    elif accelerator == "ppu":
        # PPU (T-Head) builds against PPU_SDK/CUDA_SDK and reuses the CUDA
        # boxing kernels, but its libtorch is a local PPU build bundled into
        # lib_ppu/. FLAGOS_BUILD_TILEOPS is already off by the non-cuda rule above
        # (TileOps is SM90).
        pass
    elif accelerator == "bpu":
        # D-Robotics RDK BPU. The BPU's unit of execution is a whole compiled
        # graph (a .hbm produced by hbdk4), not an individual operator, so there
        # are no FlagGems kernels to build: eager ops reach cpu_fallback, and
        # acceleration comes from the torch.compile backend in
        # torch_fl/accelerator/bpu/. Only the runtime layer (UCP allocator,
        # device/stream stubs) is native.
        #
        # FLAGOS_BUILD_VENDOR stays at the ON default, as it always has for this
        # platform, but compiles nothing: csrc/aten/backends/bpu/ does not
        # exist, so the vendor block at the top of csrc/CMakeLists.txt finds no
        # directory to add. The resulting "vendor" entry in the build record is
        # therefore a switch that is on, not a kernel set that is present.
        vendor["FLAGOS_BUILD_FLAGGEMS"] = False
    elif accelerator == "gcu":
        # Enflame GCU has no CUDA runtime: the tops runtime provides the device
        # layer and libtopsaten the operators, so CUDA/vendor kernel sets stay
        # off and FLAGOS_BUILD_VENDOR (topsaten) provides the native compute ops. Ops
        # without a topsaten kernel fall back to CPU.
        #
        # Keep the FlagGems Python kernels in the same C++ dispatcher as the
        # topsaten kernels. This mirrors the CUDA unified-RNG design: one
        # PrivateUse1 wrapper owns an exact ATen overload, while the backend
        # config chooses kGcu or kFlagOsPython at runtime. GCU initialization
        # prepares triton_gcu but does not call flag_gems.enable(), so the
        # Python layer cannot register a second PrivateUse1 implementation.
        vendor.update(
            {
                "FLAGOS_BUILD_BOXING": False,
                "FLAGOS_BUILD_FLAGGEMS": True,
                "FLAGOS_BUILD_VENDOR": True,
            }
        )
    elif accelerator == "musa":
        # Moore Threads MUSA has no CUDA runtime: the musa* API provides the
        # device layer, and mudnn provides the native operators (FLAGOS_BUILD_VENDOR).
        # Compile the FlagGems Python callers into the same wheel so the conf
        # can route the hybrid path at runtime. Kernel execution still requires a
        # compatible MUSA Triton backend; without one, native routing remains the
        # default and unaffected.
        vendor.update(
            {
                "FLAGOS_BUILD_BOXING": False,
                "FLAGOS_BUILD_FLAGGEMS": True,
                "FLAGOS_BUILD_VENDOR": True,
            }
        )

    return vendor


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
    pinned = _PINNED_KERNEL_SWITCHES.get(accelerator, {})
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
    # turns boxing off, ...) lives in _vendor_kernel_switches().
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

    def run(self):
        build_deps()
        # ``build`` runs build_py before build_ext, but CMake installs package
        # data into torch_fl/ during build_ext. Setuptools caches build_py's file
        # list, so copy late-generated files explicitly into wheel staging.
        relative_paths = [
            "_build_config.py",
            "lib/flagos_platform",
            "include/flagos.h",
        ]
        patterns = ["lib/*.so*", "lib/*.dylib*", "lib/*.dll", "lib/*.lib"]
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

    package_data = {
        "torch_fl": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
            # Self-contained wheels: the vendor's forked libtorch C++ .so bundled
            # here so the process loads that C++ runtime without a separate
            # vendor torch wheel (see scripts/vendor/bundle_*_libtorch.sh, and
            # torch_fl/accelerator/_vendor_libtorch.py for the relink at import).
            # The trailing * matters for lib_dcu: DTK's auditwheel-mangled
            # torch.libs deps end in a version suffix (libglog-6ed04f2c.so.0.0.0).
            "lib_maca/*.so*",
            "lib_dcu/*.so*",
            # DTK torch's own version.py, carried so _restore_dcu_hip_version()
            # can hand triton's hcu backend the hip/rocm strings the stock +cpu
            # torch in front does not have. Needed explicitly: the globs above
            # only match *.so*.
            "lib_dcu/vendor_version.py",
            "lib_ppu/*.so*",
            "include/*.h",
            # The DTK-private symbol manifest that libflagos_dtk_core_compat.so
            # must export, shipped so an installed wheel can be re-audited with
            # scripts/vendor/check_dcu_core_abi.py against a different DTK release.
            "accelerator/dcu/dtk_core_compat_symbols.txt",
            # All backend configs, not just the default: per-op routing tables
            # read by torch_fl at import (backends_cuda.conf /
            # backends_metax.conf, ...). Now consolidated under configs/.
            "configs/backends*.conf",
            "codegen_skip_ops.txt",
        ]
    }

    version = "0.1.0"
    # A local version segment tags which vendor a self-contained wheel bundles a
    # forked libtorch for. That bundle is SDK-version-bound whether we say so or
    # not -- DTK's libtorch_hip.so has librocblas.so.4 written into its
    # DT_NEEDED -- so making the binding visible in the filename is strictly
    # better than leaving two incompatible wheels both called 0.1.0. Override
    # with FLAGOS_WHEEL_LOCAL to pin the exact SDK, e.g.
    # FLAGOS_WHEEL_LOCAL=metax3.8.1 / FLAGOS_WHEEL_LOCAL=dtk2604.
    _default_local = {"metax": "metax", "dcu": "dtk", "ppu": "ppu"}.get(
        FLAGOS_ACCELERATOR
    )
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
        extras_require={"cuda": _cuda_runtime_requires()},
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


def _cuda_runtime_requires():
    return list(_CUDA_RUNTIME_DEPS)


def _vendor_supplies_triton() -> bool:
    """True when the target platform ships its own Triton, so PyPI's
    NVIDIA-targeted wheel must not be pulled in as a dependency.

    - FLAGOS_ACCELERATOR=dcu: DTK ships its own Triton (and builds pure-boxing).
    - FLAGOS_ACCELERATOR=ascend: `triton` is provided by FlagTree, installed out of
      band (it has no PyPI release satisfying `triton>=3.5.1`). Declaring the
      dep makes pip install stock triton over FlagTree, after which any
      Triton entry point dies with "0 active drivers".
    - PPU (FLAGOS_ACCELERATOR=ppu): the vendor Triton lives on a private index and is
      versioned 3.x+<sdk> (e.g. 3.5.0+v0.2.0.ppu2.1.0), which does not satisfy
      a `triton>=3.5.1` pin; its sdist is also a download shim that pip cannot
      always build. Install it manually, then `pip install --no-deps` this
      package. See "Build from Source (PPU Platform)" in the README.
    """
    return FLAGOS_ACCELERATOR in ("dcu", "ascend", "ppu")


# The checked-in csrc/aten/generated/* bindings are generated against a
# specific ATen surface, so torch is pinned to the 2.10 series rather than left
# open. Newer torch drifts from those bindings, and a mismatch shows up as a
# wall of compile errors at build time rather than a clean resolver failure --
# the pin is what turns that into an install-time message. Moving to a newer
# torch is a deliberate act: re-run scripts/codegen/codegen_ops.py, do not hand-edit
# the generated files.
TORCH_PIN = "torch>=2.10,<2.11"


def _install_requires():
    reqs = [TORCH_PIN]
    # FlagGems (and its Triton) is the default operator source, so it is a hard
    # runtime dep everywhere it can actually run. Platforms that ship their own
    # Triton are the exception: pulling PyPI's NVIDIA-targeted triton wheel would
    # install ~200 MB of the wrong artifact (or fail to resolve outright). All
    # flag_gems imports in the Python layer are ImportError-guarded, so omitting
    # it is safe.
    if not _vendor_supplies_triton():
        reqs += ["flag_gems>=5.0.2", "triton>=3.5.1"]
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
