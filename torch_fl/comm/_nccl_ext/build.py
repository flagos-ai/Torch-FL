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

"""Build the flagos NCCL/RCCL backend extension (``_flagos_nccl``).

This standalone builder is for CPU-only torch installations where
``torch.distributed`` does not expose ``ProcessGroupNCCL``, while an external
vendor libtorch provides the implementation. It supports NVIDIA CUDA/NCCL and
Hygon DCU/RCCL; wheel integration is handled separately.

CUDA example::

    SP=<env>/site-packages
    export LIBRARY_PATH=".libtorch_cuda_assets:$SP/torch/lib:$(ls -d $SP/nvidia/*/lib | tr '\n' ':')"
    export LD_LIBRARY_PATH="$LIBRARY_PATH"
    python torch_fl/comm/_nccl_ext/build.py

DCU example::

    source <DTK>/env.sh
    export FLAGOS_ACCELERATOR=dcu
    export FLAGOS_VENDOR_TORCH_LIB=<directory-containing-libtorch_hip.so>
    export CXX=<gcc-9-or-newer>/bin/g++
    python torch_fl/comm/_nccl_ext/build.py

The extension is written next to this file. On DCU, ``ROCM_PATH`` (or
``DTK_ROOT``) locates DTK, ``CUDA_HOME`` may select its CUDA compatibility
headers, and ``FLAGOS_VENDOR_TORCH_LIB`` may select external ``c10_hip`` and
``torch_hip`` libraries. Activated ``LIBRARY_PATH``/``LD_LIBRARY_PATH`` entries
are also considered.
"""

import glob
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_COMMON_MACROS = (
    ("USE_C10D_NCCL", None),
    ("C10_CUDA_NO_CMAKE_CONFIGURE_FILE", None),
)


@dataclass(frozen=True)
class _BuildConfig:
    accelerator: str
    include_dirs: Tuple[str, ...]
    library_dirs: Tuple[str, ...]
    libraries: Tuple[str, ...]
    rpaths: Tuple[str, ...]


def _normalize_dir(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _dedupe_existing_dirs(paths: Sequence[str]) -> Tuple[str, ...]:
    result = []
    seen = set()
    for path in paths:
        if not path:
            continue
        normalized = _normalize_dir(path)
        key = os.path.normcase(os.path.realpath(normalized))
        if key in seen or not os.path.isdir(normalized):
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)


def _env_paths(env: Mapping[str, str], key: str) -> Tuple[str, ...]:
    value = env.get(key, "")
    if not value:
        return ()
    return tuple(part for part in value.split(os.pathsep) if part)


def _selected_accelerator(env: Mapping[str, str]) -> str:
    accelerator = (
        env.get("FLAGOS_ACCELERATOR") or env.get("ACCELERATOR") or "cuda"
    ).lower()
    aliases = {
        "hygon": "dcu",
        "hip": "dcu",
        "rocm": "dcu",
        "nvidia": "cuda",
    }
    accelerator = aliases.get(accelerator, accelerator)
    if accelerator not in ("cuda", "dcu"):
        raise RuntimeError(
            "_flagos_nccl standalone build supports only "
            f"FLAGOS_ACCELERATOR=cuda or dcu, got {accelerator!r}."
        )
    return accelerator


def _cuda_library_dirs(repo_root: str) -> Tuple[str, ...]:
    candidates = []
    # External CUDA assets: libc10_cuda.so / libtorch_cuda.so live here.
    candidates.append(os.path.join(repo_root, ".libtorch_cuda_assets"))
    # Also the installed torch_fl/lib (build output) as a fallback.
    candidates.append(os.path.join(repo_root, "torch_fl", "lib"))
    # pip nvidia-*-cu12 wheels provide libnccl.so and related libraries.
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        for base in spec.submodule_search_locations:
            candidates.extend(sorted(glob.glob(os.path.join(base, "*", "lib"))))
    return _dedupe_existing_dirs(candidates)


def _cuda_config(env: Mapping[str, str], repo_root: str) -> _BuildConfig:
    library_dirs = _cuda_library_dirs(repo_root)
    include_dirs = [os.path.join(env.get("CUDA_HOME", "/usr/local/cuda"), "include")]
    include_dirs.extend(
        os.path.join(os.path.dirname(directory), "include")
        for directory in library_dirs
        if os.path.basename(os.path.dirname(directory)) == "nccl"
    )
    return _BuildConfig(
        accelerator="cuda",
        include_dirs=tuple(include_dirs),
        library_dirs=library_dirs,
        libraries=("c10_cuda", "torch_cuda", "nccl"),
        rpaths=library_dirs,
    )


def _dcu_root(env: Mapping[str, str]) -> str:
    for key in ("ROCM_PATH", "DTK_ROOT"):
        value = env.get(key)
        if not value:
            continue
        root = _normalize_dir(value)
        if not os.path.isdir(root):
            raise RuntimeError(
                f"{key}={value!r} does not name a DTK directory. "
                "Source DTK's env.sh or correct the path."
            )
        return root

    default = "/opt/dtk"
    if os.path.isdir(default):
        return default
    raise RuntimeError(
        "FLAGOS_ACCELERATOR=dcu requires DTK. Source DTK's env.sh so "
        "ROCM_PATH is set, set DTK_ROOT, or install DTK at /opt/dtk."
    )


def _dcu_cuda_root(env: Mapping[str, str], dtk_root: str) -> str:
    explicit = env.get("CUDA_HOME")
    if explicit:
        root = _normalize_dir(explicit)
        if os.path.isfile(os.path.join(root, "include", "cuda.h")):
            return root
        raise RuntimeError(
            f"CUDA_HOME={explicit!r} does not contain include/cuda.h. "
            "Point it at DTK's CUDA compatibility toolkit."
        )

    candidates = sorted(glob.glob(os.path.join(dtk_root, "cuda", "cuda-*")))
    candidates.append(os.path.join(dtk_root, "cuda"))
    for root in reversed(candidates):
        if os.path.isfile(os.path.join(root, "include", "cuda.h")):
            return _normalize_dir(root)
    raise RuntimeError(
        f"No DTK CUDA compatibility headers were found under {dtk_root!r}. "
        "Set CUDA_HOME to the directory containing include/cuda.h."
    )


def _has_link_library(directory: str, name: str) -> bool:
    return any(
        os.path.isfile(os.path.join(directory, f"lib{name}{suffix}"))
        for suffix in (".so", ".a", ".dylib")
    )


def _dcu_library_dirs(
    env: Mapping[str, str], repo_root: str, dtk_root: str
) -> Tuple[str, ...]:
    candidates = []
    vendor_torch_lib = env.get("FLAGOS_VENDOR_TORCH_LIB")
    if vendor_torch_lib:
        candidates.append(vendor_torch_lib)
    candidates.append(os.path.join(repo_root, "torch_fl", "lib_dcu"))
    candidates.extend([os.path.join(dtk_root, "lib"), os.path.join(dtk_root, "lib64")])
    candidates.extend(_env_paths(env, "LIBRARY_PATH"))
    candidates.extend(_env_paths(env, "LD_LIBRARY_PATH"))

    required = ("c10_hip", "torch_hip", "rccl")
    directories = tuple(
        directory
        for directory in _dedupe_existing_dirs(candidates)
        if any(_has_link_library(directory, name) for name in required)
    )
    missing = [
        name
        for name in required
        if not any(_has_link_library(directory, name) for directory in directories)
    ]
    if missing:
        raise RuntimeError(
            "DCU _flagos_nccl build cannot find linkable "
            f"{', '.join('lib' + name for name in missing)}. "
            "Set FLAGOS_VENDOR_TORCH_LIB to the DTK/DAS torch lib directory "
            "and source DTK's env.sh so RCCL is on LIBRARY_PATH."
        )
    return directories


def _dcu_config(env: Mapping[str, str], repo_root: str) -> _BuildConfig:
    dtk_root = _dcu_root(env)
    cuda_root = _dcu_cuda_root(env, dtk_root)
    include_dirs = _dedupe_existing_dirs(
        (
            os.path.join(cuda_root, "include"),
            os.path.join(dtk_root, "include"),
            os.path.join(dtk_root, "include", "rccl"),
        )
    )
    if not any(
        os.path.isfile(os.path.join(directory, "cuda.h")) for directory in include_dirs
    ):
        raise RuntimeError("DCU _flagos_nccl build cannot find cuda.h.")
    if not any(
        os.path.isfile(os.path.join(directory, "nccl.h")) for directory in include_dirs
    ):
        raise RuntimeError(
            "DCU _flagos_nccl build cannot find nccl.h. Check ROCM_PATH/include/rccl."
        )

    library_dirs = _dcu_library_dirs(env, repo_root, dtk_root)
    bundled_lib_dir = _normalize_dir(os.path.join(repo_root, "torch_fl", "lib_dcu"))
    rpaths = tuple(
        "$ORIGIN/../../lib_dcu"
        if os.path.normcase(directory) == os.path.normcase(bundled_lib_dir)
        else directory
        for directory in library_dirs
    )
    return _BuildConfig(
        accelerator="dcu",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=("c10_hip", "torch_hip", "rccl"),
        rpaths=rpaths,
    )


def _build_config(env: Mapping[str, str] = None, repo_root: str = None) -> _BuildConfig:
    selected_env = os.environ if env is None else env
    selected_repo = _REPO if repo_root is None else _normalize_dir(repo_root)
    accelerator = _selected_accelerator(selected_env)
    if accelerator == "dcu":
        return _dcu_config(selected_env, selected_repo)
    return _cuda_config(selected_env, selected_repo)


def _compiler_command(env: Mapping[str, str]) -> Tuple[str, ...]:
    configured = env.get("CXX")
    command = tuple(shlex.split(configured)) if configured else ()
    if not command:
        discovered = shutil.which("c++") or shutil.which("g++")
        if not discovered:
            raise RuntimeError(
                "FLAGOS_ACCELERATOR=dcu requires a C++17 compiler. Set CXX to "
                "GCC 9 or newer for PyTorch 2.10."
            )
        command = (discovered,)
    executable = shutil.which(command[0])
    if executable is None and not os.path.isfile(command[0]):
        raise RuntimeError(f"CXX compiler {command[0]!r} was not found.")
    return command


def _validate_dcu_compiler(env: Mapping[str, str] = None) -> int:
    selected_env = os.environ if env is None else env
    command = _compiler_command(selected_env)
    try:
        completed = subprocess.run(
            [*command, "-dumpfullversion", "-dumpversion"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"Failed to inspect CXX compiler {command[0]!r}: {exc}"
        ) from exc

    match = re.search(r"(?:^|\s)(\d+)(?:\.\d+)*", completed.stdout or "")
    if completed.returncode != 0 or match is None:
        warnings.warn(
            f"Could not determine the version of CXX={command[0]!r}; "
            "PyTorch 2.10 requires GCC 9 or newer."
        )
        return 0

    major = int(match.group(1))
    if major < 9:
        raise RuntimeError(
            f"CXX={command[0]!r} reports compiler version {major}; "
            "PyTorch 2.10 requires GCC 9 or newer. Set CC/CXX to a supported "
            "toolchain before building _flagos_nccl."
        )
    return major


def main():
    config = _build_config()
    if config.accelerator == "dcu":
        _validate_dcu_compiler()

    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    extension = CppExtension(
        name="_flagos_nccl",
        sources=[os.path.join(_HERE, "nccl_backend.cpp")],
        # USE_C10D_NCCL unlocks ProcessGroupNCCL.hpp in a CPU-only torch wheel.
        define_macros=list(_COMMON_MACROS),
        include_dirs=list(config.include_dirs),
        library_dirs=list(config.library_dirs),
        libraries=list(config.libraries),
        extra_compile_args=["-std=c++17"],
        extra_link_args=[f"-Wl,-rpath,{path}" for path in config.rpaths],
    )

    original_argv = sys.argv[:]
    original_cwd = os.getcwd()
    try:
        os.chdir(_HERE)
        sys.argv = [sys.argv[0], "build_ext", "--inplace"]
        setup(
            name="_flagos_nccl",
            ext_modules=[extension],
            cmdclass={"build_ext": BuildExtension},
        )
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
