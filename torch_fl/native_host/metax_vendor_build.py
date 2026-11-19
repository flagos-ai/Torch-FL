# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Build CPU host glue for the measured, original MetaX vendor GPU stub."""

import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from vendor_build import dispatch_definitions
from vendor_cohort import bundle, describe, digest, load_build_policy


def build(source_root, output, build_temp, vendor_library):
    if os.environ.get("_TORCH_FL_BUILDING_NATIVE_HOST") != "1":
        raise RuntimeError("native vendor compilation is a wheel-build operation")
    import torch
    from torch.utils.cpp_extension import load

    source_root, output, build_temp = map(Path, (source_root, output, build_temp))
    policy = load_build_policy(source_root).precision_policy("metax")
    cohort = describe(vendor_library, policy)
    if (
        str(torch.__version__) != policy["vendor_torch"].split("+", 1)[0] + "+cpu"
        or torch.version.git_version != policy["cpu_torch_git_version"]
        or torch.version.cuda is not None
        or importlib.metadata.version("flag_gems") != policy["flag_gems"]
    ):
        raise RuntimeError(
            "MetaX vendor stub requires the measured official CPU cohort"
        )
    package = output / "torch_fl"
    prefix = Path("native_host/_generated/metax_vendor")
    artifacts = {
        str(prefix / name): value for name, value in bundle(cohort, package).items()
    }
    build_temp.mkdir(parents=True, exist_ok=True)
    headers = build_temp / "published_headers"
    for relative, record in cohort["host_headers"].items():
        target = headers / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["path"], target)
        if digest(target) != record["sha256"]:
            raise RuntimeError("published MetaX host header copy changed")
    # Only the two verified published headers enter this include root. Vendor
    # ATen/core headers must never shadow the official CPU Torch build headers.
    import torch_fl

    if "boxing" in torch_fl._build_config.KERNELS:
        raise RuntimeError("native MetaX vendor stub cannot be built with boxing")
    catalog = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(source_root / "scripts/codegen/codegen_metax_native_vendor.py"),
                "--check",
                "--print-catalog",
            ],
            text=True,
        )
    )
    if json.loads(torch.ops.flagos.native_vendor_catalog()) != catalog:
        raise RuntimeError("rebuild the core after generating MetaX vendor entries")
    libraries = package / prefix
    for name in cohort["libraries"]:
        ctypes.CDLL(str(libraries / name), mode=os.RTLD_LAZY | os.RTLD_GLOBAL)
    definitions = dispatch_definitions(
        Path(torch.__file__).parent / "lib/libtorch_cpu.so"
    )
    maca = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
    try:
        module = load(
            name="_metax_vendor",
            sources=[str(source_root / "csrc/native_host/metax_ops.cc")],
            extra_include_paths=[
                str(source_root / "csrc/include"),
                str(headers),
                str(maca / "tools/cu-bridge/include"),
                str(maca / "include"),
                str(maca / "include/mcr"),
            ],
            extra_cflags=[
                "-O3",
                "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
                *("-D" + value for value in definitions),
            ],
            extra_ldflags=[
                str(package / "lib/libflagos.so"),
                *(str(libraries / name) for name in cohort["libraries"]),
                "-Wl,-rpath,'$$ORIGIN/metax_vendor'",
            ],
            build_directory=str(build_temp),
            with_cuda=False,
            verbose=True,
        )
    finally:
        sys.setdlopenflags(old_flags)
    target = package / "native_host/_generated/_metax_vendor.so"
    shutil.copy2(module.__file__, target)
    artifacts[str(target.relative_to(package))] = digest(target)
    numerical_libraries = {
        name: digest(path) for name, path in module.libraries().items()
    }
    manifest = {
        "schema_version": 1,
        "vendor": "metax",
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "flag_gems": policy["flag_gems"],
        "vendor_torch": cohort["version"],
        "vendor_metadata_sha256": cohort["metadata_sha256"],
        "cpu_dispatch_definitions": definitions,
        "catalog": catalog,
        "policy": policy,
        "artifacts": artifacts,
        "numerical_libraries": numerical_libraries,
        "published_host_headers": {
            name: record["sha256"] for name, record in cohort["host_headers"].items()
        },
        "sources": {
            name: digest(source_root / name)
            for name in (
                "scripts/codegen/native_host_policy.py",
                "scripts/codegen/codegen_metax_native_vendor.py",
                "csrc/native_host/metax_ops.cc",
                "csrc/native_host/metax_ops.h",
            )
        },
    }
    (target.parent / "vendor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Built {len(catalog)} measured MetaX vendor host entries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-temp", type=Path, required=True)
    parser.add_argument("--vendor-library", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root, args.output, args.build_temp, args.vendor_library)
