# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the optional typed vendor module; never imported by the runtime."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from vendor_cohort import bundle, describe, digest, load_build_policy


def dispatch_definitions(torch_library):
    # DispatchStub's public C++ signature depends on the CPU build capabilities.
    # Derive them from the exact linked library rather than the build machine's ISA.
    symbols = subprocess.check_output(["nm", "-D", "-C", str(torch_library)], text=True)
    definitions = []
    for capability in ("AVX512", "AVX2", "VSX", "ZVECTOR", "SVE256"):
        if re.search(r"DispatchStub<[^\n]+>::" + capability + r"\s*$", symbols, re.M):
            definitions.append("HAVE_" + capability + "_CPU_DEFINITION")
    return definitions


def generate_catalog(source_root):
    return json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(Path(source_root) / "scripts/codegen/codegen_native_vendor.py"),
                "--check",
                "--print-catalog",
            ],
            text=True,
        )
    )


def build(source_root, output, build_temp, vendor_library):
    if os.environ.get("_TORCH_FL_BUILDING_NATIVE_HOST") != "1":
        raise RuntimeError("native vendor compilation is a wheel-build operation")
    import torch
    from torch.utils.cpp_extension import load

    source_root, output, build_temp = map(Path, (source_root, output, build_temp))
    package = output / "torch_fl"
    sources = source_root / "torch_fl/native_host"
    policy = load_build_policy(source_root).precision_policy("nvidia")
    cohort = describe(vendor_library, policy)
    if (
        str(torch.__version__).split("+", 1)[0] != cohort["version"].split("+", 1)[0]
        or torch.version.git_version != cohort["git_version"]
        or torch.version.cuda is not None
        or importlib.metadata.version("flag_gems") != policy["flag_gems"]
    ):
        raise RuntimeError(
            "native vendor build must match the CPU Torch and published FlagGems cohort"
        )
    artifacts = {
        "lib/" + name: value for name, value in bundle(cohort, package).items()
    }
    # Loading torch_fl supplies libflagos and the existing NVIDIA dependency
    # loader. The build environment disables native-host activation explicitly.
    import torch_fl

    if "boxing" in torch_fl._build_config.KERNELS:
        raise RuntimeError("native vendor precision cannot be built with boxing")
    catalog = generate_catalog(source_root)
    if json.loads(torch.ops.flagos.native_vendor_catalog()) != catalog:
        raise RuntimeError(
            "rebuild the core library after generating native vendor entries"
        )
    vendor_root = Path(cohort["library_dir"]).parent
    matches = list(vendor_root.parent.glob("nvidia/*/lib/libcublas.so.13"))
    if len(matches) != 1:
        raise RuntimeError(
            "cannot uniquely resolve the validated vendor cuBLAS library"
        )
    cuda_root = Path(os.environ["CUDA_HOME"])
    torch_library = Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    definitions = dispatch_definitions(torch_library)
    build_temp.mkdir(parents=True, exist_ok=True)
    module = load(
        name="_vendor",
        sources=[str(sources / "csrc/vendor_generated.cc")],
        extra_include_paths=[
            str(sources / "csrc"),
            str(source_root / "csrc/include"),
            str(vendor_root / "include"),
            str(cuda_root / "include"),
        ],
        extra_cflags=[
            "-O3",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *("-D" + value for value in definitions),
        ],
        extra_ldflags=[
            str(package / "lib/libflagos.so"),
            str(package / "lib/libtorch_cuda.so"),
            str(package / "lib/libc10_cuda.so"),
            str(matches[0]),
            "-Wl,-rpath,'$$ORIGIN/../../lib'",
        ],
        build_directory=str(build_temp),
        with_cuda=False,
        verbose=True,
    )
    dynamic = subprocess.check_output(["readelf", "-d", module.__file__], text=True)
    if "$ORIGIN/../../lib" not in dynamic:
        raise RuntimeError(
            "native vendor module is missing its relocatable library path"
        )
    target = package / "native_host/_generated/_vendor.so"
    shutil.copy2(module.__file__, target)
    artifacts[str(target.relative_to(package))] = digest(target)
    libraries = module.libraries()
    manifest = {
        "schema_version": 1,
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "flag_gems": policy["flag_gems"],
        "vendor_torch": cohort["version"],
        "vendor_metadata_sha256": cohort["metadata_sha256"],
        "cpu_dispatch_definitions": definitions,
        "catalog": catalog,
        "policy": policy,
        "artifacts": artifacts,
        "numerical_libraries": {name: digest(path) for name, path in libraries.items()},
        "sources": {
            name: digest(source_root / name)
            for name in (
                "scripts/codegen/native_host_policy.py",
                "scripts/codegen/codegen_native_vendor.py",
                "torch_fl/native_host/csrc/vendor_generated.cc",
                "torch_fl/native_host/csrc/vendor_runtime.h",
            )
        },
    }
    (target.parent / "vendor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Built {len(catalog)} typed vendor precision entries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-temp", type=Path, required=True)
    parser.add_argument("--vendor-library", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root, args.output, args.build_temp, args.vendor_library)
