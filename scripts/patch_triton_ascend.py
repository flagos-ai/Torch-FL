#!/usr/bin/env python3

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

"""
Patch triton-ascend to work with torch_fl (without torch_npu dependency).

Original triton-ascend is designed to work with torch_npu. This script
patches it to use the torch_fl (flagos) device interface instead,
removing all hard dependencies on libtorch_npu.so.

Usage:
    python scripts/patch_triton_ascend.py [--triton-path /path/to/triton]

    If --triton-path is not given, auto-detects from `import triton`.
"""

import argparse
import os
import sys


def find_triton_path():
    """Auto-detect triton installation path without importing its backend."""
    try:
        import triton

        return os.path.dirname(triton.__file__)
    except (ImportError, OSError, RuntimeError):
        # Importing triton-ascend can fail before this patch is applied because
        # its Ascend backend imports torch_npu. Locate the package on disk so
        # the patch can remove that dependency before triton is imported again.
        for site_packages in sys.path:
            candidate = os.path.join(site_packages, "triton")
            if os.path.isdir(os.path.join(candidate, "backends", "ascend")):
                return candidate

        print(
            "ERROR: triton-ascend package not found. Specify --triton-path explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)


def patch_file(filepath, replacements):
    """Apply (old, new) replacements to a file. Returns True if changed."""
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return False

    with open(filepath, "r") as f:
        content = f.read()

    original = content
    applied = 0

    for old, new in replacements:
        if old not in content:
            if new in content:
                continue  # already patched
            print("  WARNING: pattern not found:")
            print(f"    {repr(old[:80])}")
            continue
        content = content.replace(old, new)
        applied += 1

    if content == original:
        print(f"  OK (already patched): {filepath}")
        return False

    with open(filepath, "w") as f:
        f.write(content)
    print(f"  PATCHED ({applied} changes): {filepath}")
    return True


def get_torch_npu_path(triton_path):
    """Determine torch_npu path for include headers."""
    site_packages = os.path.dirname(triton_path)
    return os.path.join(site_packages, "torch_npu")


STREAM_HELPER = '''

_flagos_stream_fn = None


def _flagos_raw_stream(device):
    """Return torch_fl's acl stream for `device` as an int, for rtKernelLaunch.

    Resolved by ctypes against the already-loaded libflagos.so rather than through
    torch_fl._C: `GetCurrentStream` is exported from stream_api.cc with default
    visibility expressly so FlagGems/triton can reach the aclrtStream, and no _C
    binding surfaces it. torch_fl has necessarily dlopened that library before any
    kernel can launch, so CDLL returns a handle to the same instance -- this cannot
    create a second stream.

    Cached: get_current_stream sits on the launch path of every kernel.

    Falls back to 0 only if the symbol cannot be resolved. That fallback has no
    ordering against torch_fl's aclnn ops, so it warns rather than silently
    reintroducing the race it exists to fix.
    """
    global _flagos_stream_fn
    if _flagos_stream_fn is None:
        import ctypes

        try:
            lib = ctypes.CDLL("libflagos.so")
            fn = lib.GetCurrentStream
        except (OSError, AttributeError) as e:
            import warnings

            warnings.warn(
                "triton-ascend could not resolve GetCurrentStream from "
                f"libflagos.so ({e}); falling back to rt stream 0, which is NOT "
                "ordered against torch_fl's aclnn ops and can silently corrupt "
                "results."
            )
            _flagos_stream_fn = lambda _d: 0  # noqa: E731
        else:
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_int]
            _flagos_stream_fn = fn
    return _flagos_stream_fn(int(device)) or 0
'''


def _inject_stream_helper(fp):
    """Insert _flagos_raw_stream into driver.py, before `class NPUUtils`.

    Kept out of the `replacements` list because those are plain str.replace calls:
    any anchor that survives into the replacement text would match again on a
    re-run and inject a second copy. Guarding on the function name makes this
    idempotent, which matters because a `pip install --force-reinstall` of
    triton-ascend wipes the patch and the script gets re-run.
    """
    if not os.path.exists(fp):
        return False
    with open(fp, "r") as f:
        content = f.read()
    if "_flagos_raw_stream" in content:
        return False
    anchor = "\nclass NPUUtils(object):"
    if anchor not in content:
        print(f"  WARNING: anchor 'class NPUUtils' not found in {fp}")
        return False
    content = content.replace(anchor, STREAM_HELPER + anchor, 1)
    with open(fp, "w") as f:
        f.write(content)
    print(f"  PATCHED (stream helper): {fp}")
    return True


def patch_driver(triton_path):
    """Patch backends/ascend/driver.py"""
    fp = os.path.join(triton_path, "backends", "ascend", "driver.py")

    _inject_stream_helper(fp)

    replacements = [
        # --- Python API patches ---
        # get_current_device
        (
            "        import torch_npu\n        return torch.npu.current_device()",
            "        # import torch_npu  # patched for torch_fl\n"
            "        return torch.flagos.current_device()",
        ),
        # set_current_device
        (
            "        import torch_npu\n        return torch.npu.set_device(device)",
            "        # import torch_npu  # patched for torch_fl\n"
            "        return torch.flagos.set_device(device)",
        ),
        # get_current_stream: remove native _C import
        (
            "        import torch_npu\n"
            "        from torch_npu._C import"
            " _npu_getCurrentRawStream",
            "        # import torch_npu  # patched for torch_fl\n"
            "        # from torch_npu._C import"
            " _npu_getCurrentRawStream  # patched",
        ),
        # Return torch_fl's OWN acl stream, not 0.
        #
        # Returning 0 (the NULL/default rt stream) is not merely a simplification:
        # torch_fl runs every aclnn op on a stream it creates itself
        # (GetDefaultAclStream -> aclrtCreateStream), so a gems kernel launched on
        # stream 0 has NO ordering against the aclnn ops producing its inputs or
        # consuming its outputs. The generated launcher ends with
        # rtStreamSynchronize(stream), which then drains the wrong stream and
        # returns while the real work is still outstanding.
        #
        # Symptom this caused: Qwen3 training loss came out `nan` on the gems path
        # but was finite (matching aclnn to 3 decimals) the moment a
        # TorchDispatchMode inserted a device synchronize after every op. It only
        # appeared after a warmup forward+backward had freed blocks back to the
        # caching allocator, i.e. once block reuse could hand a still-in-flight
        # buffer to the next kernel. Ordinary short runs looked fine.
        (
            "        return _npu_getCurrentRawStream(device)",
            "        return _flagos_raw_stream(device)  # patched for torch_fl",
        ),
        # Migration: an earlier version of this script wrote `return 0`, so an
        # environment patched by it has no _npu_getCurrentRawStream line left to
        # match. Rewrite that too, otherwise re-running the script leaves the
        # stream race in place and reports success.
        (
            "        return 0  # default stream for torch_fl",
            "        return _flagos_raw_stream(device)  # patched for torch_fl",
        ),
        # get_device_interface
        (
            "        return torch.npu\n",
            "        return torch.flagos\n",
        ),
        # get_empty_cache_for_benchmark
        (
            "device='npu')",
            "device='flagos')",
        ),
        # --- C++ template patches ---
        # Remove NPUWorkspaceAllocator.h
        (
            "#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>",
            "// torch_npu removed: workspace allocated via rtMalloc",
        ),
        # Replace at_npu allocator with rtMalloc
        (
            "    at::Tensor syncBlockLock_tensor = "
            "at_npu::native::allocate_workspace"
            "(syncBlockLockSize, stream);\n"
            "    syncBlockLock_ptr = const_cast<void *>"
            "(syncBlockLock_tensor.storage().data());",
            "    ret = rtMalloc(&syncBlockLock_ptr,"
            " syncBlockLockSize, RT_MEMORY_HBM);\n"
            "    ",
        ),
        # TRITON_ENABLE_TASKQUEUE default: true -> false
        # (taskqueue requires torch_npu OpCommand.h)
        (
            "\"TRITON_ENABLE_TASKQUEUE\", 'true'",
            "\"TRITON_ENABLE_TASKQUEUE\", 'false'",
        ),
    ]

    return patch_file(fp, replacements)


def patch_utils(triton_path):
    """Patch backends/ascend/utils.py"""
    fp = os.path.join(triton_path, "backends", "ascend", "utils.py")
    npu_path = get_torch_npu_path(triton_path)

    replacements = [
        # Remove -ltorch_npu linker flag
        (
            '        "-ltorch_npu",',
            '        # "-ltorch_npu",  # removed for torch_fl',
        ),
        # _npu_version_hash: hardcode version
        # (import torch_npu is not adjacent to torch_npu_version)
        (
            "    torch_npu_version = torch_npu.version.git_version",
            '    torch_npu_version = "torch_fl_shim"',
        ),
        # Replace dynamic torch_npu path with hardcoded
        # (import torch_npu is not adjacent to torch_npu_path)
        (
            "    torch_npu_path = os.path.dirname("
            "os.path.realpath(torch_npu.__file__))",
            '    torch_npu_path = "' + npu_path + '"',
        ),
    ]

    # Comment out all bare `import torch_npu` lines
    # (they appear after `import torch` in several functions)
    replacements.append(
        (
            "    import torch_npu\n",
            "    # import torch_npu  # patched for torch_fl\n",
        )
    )

    return patch_file(fp, replacements)


def patch_backend_register(triton_path):
    """Patch compiler flags that otherwise require libtorch_npu."""
    fp = os.path.join(triton_path, "backends", "ascend", "backend_register.py")
    replacements = [
        (
            "    import torch_npu\n"
            "    torch_path = os.path.dirname(os.path.realpath(torch.__file__))\n"
            "    torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))",
            "    # torch_npu is a compatibility shim supplied by torch_fl.\n"
            "    torch_path = os.path.dirname(os.path.realpath(torch.__file__))\n"
            "    torch_npu_path = torch_path",
        ),
        (
            "        f\"-I{os.path.join(torch_npu_path, 'include')}\",",
            "        # torch_npu headers are not needed by the FLAGOS backend,",
        ),
        (
            "            f\"-L{os.path.join(torch_npu_path, 'lib')}\",\n"
            '            f"-ltorch_npu",',
            "            # The FLAGOS backend links against torch_fl/ATen, not torch_npu.",
        ),
    ]
    return patch_file(fp, replacements)


def patch_npu_utils(triton_path):
    """Patch backends/ascend/npu_utils.cpp for CANN 9.0.0 enum names.

    triton-ascend 3.2.0's npu_utils.cpp references rtLimitType_t enumerators
    from a newer CANN release. CANN 9.0.0 (rt_external_base.h) names the SIMT
    per-warp stack limit RT_LIMIT_TYPE_SIMT_STACK_SIZE, not the newer
    RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE, so the JIT compile of npu_utils.cpp
    fails with "could not convert brace-enclosed initializer list". Map the
    "WARP_STACK_SIZE" key onto the enumerator that CANN 9.0.0 actually
    provides. Idempotent: the newer name only ever appears here.
    """
    fp = os.path.join(triton_path, "backends", "ascend", "npu_utils.cpp")

    replacements = [
        (
            "rtLimitType_t::RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE",
            "rtLimitType_t::RT_LIMIT_TYPE_SIMT_STACK_SIZE",
        ),
    ]

    return patch_file(fp, replacements)


def main():
    parser = argparse.ArgumentParser(
        description="Patch triton-ascend for torch_fl compatibility"
    )
    parser.add_argument(
        "--triton-path",
        default=None,
        help="Path to triton package directory. Auto-detected if not specified.",
    )
    args = parser.parse_args()

    triton_path = args.triton_path or find_triton_path()
    print(f"Triton path: {triton_path}")

    if not os.path.isdir(os.path.join(triton_path, "backends", "ascend")):
        print(
            "ERROR: backends/ascend/ not found. Is this triton-ascend?",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[1/4] Patching backends/ascend/driver.py ...")
    patch_driver(triton_path)

    print("\n[2/4] Patching backends/ascend/utils.py ...")
    patch_utils(triton_path)

    print("\n[3/4] Patching backends/ascend/backend_register.py ...")
    patch_backend_register(triton_path)

    print("\n[4/4] Patching backends/ascend/npu_utils.cpp (CANN 9.0.0 enum) ...")
    patch_npu_utils(triton_path)

    print("\nDone. triton-ascend is now compatible with torch_fl.")
    print("NOTE: Clear triton kernel cache if you had previously compiled kernels:")
    print("  rm -rf ~/.triton/cache/")


if __name__ == "__main__":
    main()
