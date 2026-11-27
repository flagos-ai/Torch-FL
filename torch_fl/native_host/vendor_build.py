# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the optional typed vendor precision module during wheel packaging."""

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
import ctypes
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from vendor_cohort import bundle, bundle_directory, describe, digest, load_build_policy


@dataclass(frozen=True)
class _LinkOptions:
    includes: tuple[Path, ...]
    libraries: tuple[Path, ...]
    rpaths: tuple[str, ...]
    cflags: tuple[str, ...] = ()


def _nvidia_links(cohort, bundled, _build_temp):
    vendor_root = Path(cohort["library_dir"]).parent
    matches = list(vendor_root.parent.glob("nvidia/*/lib/libcublas.so.13"))
    if len(matches) != 1:
        raise RuntimeError(
            "cannot uniquely resolve the validated vendor cuBLAS library"
        )
    return _LinkOptions(
        includes=(vendor_root / "include", Path(os.environ["CUDA_HOME"]) / "include"),
        libraries=(
            bundled / "libtorch_cuda.so",
            bundled / "libc10_cuda.so",
            matches[0],
        ),
        rpaths=("$ORIGIN/../../lib",),
    )


def _hygon_links(cohort, bundled, _build_temp):
    vendor_root = Path(cohort["library_dir"]).parent
    toolkit = Path(os.environ.get("DTK_ROOT", "/opt/dtk"))
    return _LinkOptions(
        includes=(vendor_root / "include", toolkit / "include"),
        libraries=(bundled / "libtorch_hip.so", bundled / "libc10_hip.so"),
        rpaths=("$ORIGIN/../../lib_hygon", "$ORIGIN/../../lib"),
        cflags=("-DUSE_ROCM", "-D__HIP_PLATFORM_AMD__"),
    )


def _stage_published_headers(cohort, build_temp):
    # Expose only verified device headers to the official CPU Torch frontend.
    headers = build_temp / "published_headers"
    for relative, record in cohort["host_headers"].items():
        target = headers / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["path"], target)
        if digest(target) != record["sha256"]:
            raise RuntimeError(f"published device header copy changed: {relative}")
    return headers


def _metax_links(cohort, bundled, build_temp):
    maca = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    return _LinkOptions(
        includes=(
            _stage_published_headers(cohort, build_temp),
            maca / "tools/cu-bridge/include",
            maca / "include",
            maca / "include/mcr",
        ),
        libraries=tuple(bundled / name for name in cohort["libraries"]),
        rpaths=("$ORIGIN/metax_vendor",),
        cflags=("-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",),
    )


def _thead_links(cohort, bundled, build_temp):
    sdk = Path(
        os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME") or "/usr/local/PPU_SDK"
    )
    cuda = Path(os.environ.get("CUDA_HOME") or sdk / "CUDA_SDK")
    return _LinkOptions(
        includes=(
            _stage_published_headers(cohort, build_temp),
            cuda / "include",
            sdk / "include",
        ),
        libraries=(
            *(bundled / name for name in cohort["libraries"]),
            cuda / "lib64/libcudart.so.13",
        ),
        rpaths=("$ORIGIN/thead_vendor",),
    )


@contextmanager
def _preload_libraries(libraries):
    flags = os.RTLD_LAZY | os.RTLD_GLOBAL
    for path in libraries:
        ctypes.CDLL(str(path), mode=flags)
    previous = sys.getdlopenflags()
    sys.setdlopenflags(flags)
    try:
        yield
    finally:
        sys.setdlopenflags(previous)


@dataclass(frozen=True)
class _BuildSpec:
    links: Callable[[dict, Path, Path], _LinkOptions]
    catalog: tuple[str, ...]
    module: str = "_vendor"
    header: str = "precision_runtime.h"
    official_cpu_first: bool = False
    explicit_cxx11_abi: bool = True
    load: Callable[[tuple[Path, ...]], AbstractContextManager] = nullcontext


_BUILD_SPECS = {
    "nvidia": _BuildSpec(
        _nvidia_links, ("codegen_native_vendor.py", "--vendor", "nvidia")
    ),
    "hygon": _BuildSpec(
        _hygon_links, ("codegen_native_vendor.py", "--vendor", "hygon")
    ),
    "metax": _BuildSpec(
        _metax_links,
        ("codegen_metax_native_vendor.py",),
        module="_metax_vendor",
        header="backends/metax/precision.h",
        official_cpu_first=True,
        explicit_cxx11_abi=False,
        load=_preload_libraries,
    ),
    "thead": _BuildSpec(
        _thead_links,
        ("codegen_native_vendor.py", "--vendor", "thead"),
        module="_thead_vendor",
        official_cpu_first=True,
        load=_preload_libraries,
    ),
}


def dispatch_definitions(torch_library):
    # DispatchStub's public C++ signature depends on the CPU build capabilities.
    # Derive them from the exact linked library rather than the build machine's ISA.
    symbols = subprocess.check_output(["nm", "-D", "-C", str(torch_library)], text=True)
    definitions = []
    for capability in ("AVX512", "AVX2", "VSX", "ZVECTOR", "SVE256"):
        if re.search(r"DispatchStub<[^\n]+>::" + capability + r"\s*$", symbols, re.M):
            definitions.append("HAVE_" + capability + "_CPU_DEFINITION")
    return definitions


def generate_catalog(source_root, vendor="nvidia"):
    script, *selection = _BUILD_SPECS[vendor].catalog
    return json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(source_root) / "scripts/codegen" / script),
                "--check",
                "--print-catalog",
                *selection,
            ],
            text=True,
        )
    )


def _load_cpu_torch(policy, cohort, *, exact_cpu=False):
    import torch

    version = str(torch.__version__)
    expected = cohort["version"].split("+", 1)[0]
    version_matches = (
        version == expected + "+cpu"
        if exact_cpu
        else version.split("+", 1)[0] == expected
    )
    if (
        not version_matches
        or torch.version.git_version
        != policy.get("cpu_torch_git_version", cohort["git_version"])
        or torch.version.cuda is not None
        or importlib.metadata.version("flag_gems") != policy["flag_gems"]
    ):
        raise RuntimeError(
            "native vendor build must match the CPU Torch and published FlagGems cohort"
        )
    return torch


def build(source_root, output, build_temp, vendor_library, vendor="nvidia"):
    if os.environ.get("_TORCH_FL_BUILDING_NATIVE_HOST") != "1":
        raise RuntimeError("native vendor compilation is a wheel-build operation")
    source_root, output, build_temp = (
        Path(path).resolve() for path in (source_root, output, build_temp)
    )
    spec = _BUILD_SPECS[vendor]
    policy = load_build_policy(source_root).precision_policy(vendor)
    cohort = describe(vendor_library, policy)
    # Device-only builds keep the official CPU Torch frontend. The other builds
    # let torch_fl select the platform libraries before importing Torch.
    torch = (
        _load_cpu_torch(policy, cohort, exact_cpu=True)
        if spec.official_cpu_first
        else None
    )
    package = output / "torch_fl"
    bundle_dir = bundle_directory(vendor)
    artifacts = {
        (bundle_dir / name).as_posix(): checksum
        for name, checksum in bundle(cohort, package, source_root=source_root).items()
    }

    import torch_fl
    from torch.utils.cpp_extension import load

    if torch_fl._build_config.ACCELERATOR != vendor:
        raise RuntimeError("native vendor selection differs from the build record")
    if torch is None:
        torch = _load_cpu_torch(policy, cohort)
    if "boxing" in torch_fl._build_config.KERNELS:
        raise RuntimeError("native vendor precision cannot be built with boxing")
    catalog = generate_catalog(source_root, vendor)
    if json.loads(torch.ops.flagos.native_vendor_catalog(vendor)) != catalog:
        raise RuntimeError(
            "rebuild the core library after generating native vendor entries"
        )

    native_dir = Path("csrc/native_host")
    source = native_dir / "backends" / vendor / "precision.cc"
    header = native_dir / spec.header
    definitions = dispatch_definitions(
        Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    )
    build_temp.mkdir(parents=True, exist_ok=True)
    links = spec.links(cohort, package / bundle_dir, build_temp)
    includes = [source_root / "csrc/include", *links.includes]
    if header.parent != source.parent:
        includes.insert(0, source_root / header.parent)
    cflags = ["-O3"]
    if spec.explicit_cxx11_abi:
        cflags.append(
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
        )
    with spec.load(links.libraries):
        module = load(
            name=spec.module,
            sources=[str(source_root / source)],
            extra_include_paths=[str(path) for path in includes],
            extra_cflags=[
                *cflags,
                *("-D" + value for value in definitions),
                *links.cflags,
            ],
            extra_ldflags=[
                str(package / "lib/libflagos.so"),
                *(str(path) for path in links.libraries),
                *(f"-Wl,-rpath,'${path}'" for path in links.rpaths),
            ],
            build_directory=str(build_temp),
            with_cuda=False,
            verbose=True,
        )
    dynamic = subprocess.check_output(["readelf", "-d", module.__file__], text=True)
    if any(path not in dynamic for path in links.rpaths):
        raise RuntimeError(
            "native vendor module is missing its relocatable library path"
        )

    target = package / "native_host/_generated" / (spec.module + ".so")
    shutil.copy2(module.__file__, target)
    artifacts[str(target.relative_to(package))] = digest(target)
    manifest = {
        "schema_version": 1,
        "vendor": vendor,
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "flag_gems": policy["flag_gems"],
        "vendor_torch": cohort["version"],
        "vendor_metadata_sha256": cohort["metadata_sha256"],
        "cpu_dispatch_definitions": definitions,
        "catalog": catalog,
        "policy": policy,
        "artifacts": artifacts,
        "numerical_libraries": {
            name: digest(path) for name, path in module.libraries().items()
        },
        "sources": {
            path.as_posix(): digest(source_root / path)
            for path in (
                Path("scripts/codegen/native_host_policy.py"),
                Path("scripts/codegen") / spec.catalog[0],
                source,
                header,
            )
        },
    }
    if cohort["host_headers"]:
        manifest["published_host_headers"] = {
            name: record["sha256"] for name, record in cohort["host_headers"].items()
        }
    (target.parent / "vendor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Built {len(catalog)} typed vendor precision entries for {vendor}", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=tuple(_BUILD_SPECS), default="nvidia")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-temp", type=Path, required=True)
    parser.add_argument("--vendor-library", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.source_root, args.output, args.build_temp, args.vendor_library, args.vendor
    )
