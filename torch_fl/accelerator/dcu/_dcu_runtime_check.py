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

"""Post-import checks that the DTK device libs really bound to the official core.

Runs for a decoupled DCU build right after ``import torch``.  Two checks, both
cheap, both catching failures that would otherwise surface much later as a
wrong-looking numerical result or a bare undefined-symbol abort:

1. Base version alignment.  The bundled DTK ``libtorch_hip.so`` was compiled
   against DTK's fork of a specific torch minor; a mismatched official wheel in
   front is an ABI mismatch that ``dlopen`` does *not* reject, because the symbols
   it needs happen to exist with the same names.
2. CUDA dispatch presence.  ``libtorch_hip.so`` registers its kernels under the
   CUDA dispatch key, which is what the PrivateUse1 boxing kernels re-dispatch
   into.  If it did not actually load (or loaded RTLD_LOCAL), the dispatcher has
   no CUDA kernel for these ops and every boxed op raises "Could not run
   'aten::mm' with arguments from the 'CUDA' backend".

Both raise; a partially wired process is not worth continuing with.

SDK-only mode (``FLAGOS_DCU_SDK_ONLY=1``) loads no DTK libtorch at all, so
neither check applies -- they would fail on exactly the property that mode
establishes.  ``validate_sdk_only_runtime()`` is its equivalent: the SDK plugin
registered its kernels, and no vendor torch is mapped.
"""

import os
import re

# A small, load-bearing sample: one blas op, one elementwise, one reduction-ish,
# one batched blas. All four come from libtorch_hip.so, none from the CPU core, so
# a missing one means the device library is not in the dispatcher.
REQUIRED_CUDA_OPS = ("aten::mm", "aten::add.Tensor", "aten::_softmax", "aten::bmm")

_VERSION_RE = re.compile(r"\s*__version__\s*(?::[^=]*)?=\s*'([^']+)'")


def base_version(version):
    """``"2.10.0+das.opt1.dtk2604"`` / ``"2.10.0+cpu"`` -> ``"2.10.0"``."""
    return version.split("+", 1)[0]


def bundled_vendor_version(bundle_dir):
    """The torch version the bundled DTK libtorch was built from, or None.

    Reads ``vendor_version.py``, a verbatim copy of the DTK wheel's own
    ``torch/version.py`` placed there by ``scripts/bundle_dcu_libtorch.sh``.
    None means a source checkout with no bundle, where the preload was a no-op
    too, so there is nothing to reconcile.
    """
    path = os.path.join(bundle_dir, "vendor_version.py")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                match = _VERSION_RE.match(line)
                if match:
                    return match.group(1)
    except OSError:
        return None
    return None


def check_version_alignment(torch_version, vendor_version):
    """Raise when the installed torch and the bundled DTK libtorch disagree."""
    if not vendor_version:
        return
    vendor_base = base_version(vendor_version)
    if base_version(torch_version) == vendor_base:
        return
    raise RuntimeError(
        f"DCU: the installed torch {torch_version} does not match the bundled "
        f"DTK libtorch, which was built against torch {vendor_base}. Install "
        f"torch=={vendor_base} (the CPU wheel is enough), or set "
        "FLAGOS_DCU_SKIP_RUNTIME_CHECK=1 to proceed at your own risk."
    )


def check_cuda_dispatch(has_kernel):
    """Raise when DTK's CUDA-key kernels are absent from the dispatcher.

    ``has_kernel(op)`` is normally
    ``torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA")``.
    """
    missing = [op for op in REQUIRED_CUDA_OPS if not has_kernel(op)]
    if not missing:
        return
    raise RuntimeError(
        "DCU: DTK's libtorch_hip.so did not register CUDA kernels for "
        f"{', '.join(missing)}, so PrivateUse1 -> CUDA boxing cannot work. The "
        "device libraries were not loaded before `import torch` (import torch_fl "
        "first, never torch_fl after torch in a fresh process), or the bundle is "
        "incomplete -- rerun scripts/bundle_dcu_libtorch.sh."
    )


def validate_decoupled_runtime(bundle_dir):
    """Run both checks against the torch that is now imported."""
    import torch

    check_version_alignment(torch.__version__, bundled_vendor_version(bundle_dir))
    check_cuda_dispatch(
        lambda op: torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA")
    )


def validate_sdk_only_runtime():
    """The SDK-only counterpart: the plugin registered, and no vendor torch.

    Both checks above are about DTK's ``libtorch_hip.so`` binding correctly to the
    official core, which SDK-only mode deliberately never loads -- running them
    there would fail on the very property the mode exists to establish. What is
    load-bearing instead is that the SDK plugin's kernels really are in the
    dispatcher (an empty ``Backend::kDcuSdk`` slot means every routed GEMM raises)
    and that no vendor torch crept into the process.

    Version alignment needs no separate check here: the plugin manifest's
    ``torch_base`` is compared against the active torch during
    ``load_and_register()``, before any kernel pointer is installed.
    """
    from torch_fl.accelerator.dcu._dcu_sdk_ops import (
        assert_sdk_only_process,
        kernels_registered,
    )

    if not kernels_registered():
        raise RuntimeError(
            "DCU SDK-only mode is active but libdcu_aten_ops.so registered no "
            "kernels, so the SDK-native routes have nothing behind them. This "
            "means load_and_register() did not run -- import torch_fl before "
            "torch in a fresh process."
        )
    assert_sdk_only_process()
