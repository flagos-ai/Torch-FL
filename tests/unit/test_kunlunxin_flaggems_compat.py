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

"""The FlagGems glue a Kunlunxin build needs, pinned as a source contract.

Everything here was measured on a P800 and can only be measured there: the
repository's unit environment has no XPU and no torch_fl._C. What it can pin is
the shape that produced the measurement, because the failure mode of getting it
wrong is silent -- a `flaggems` route in `backends_kunlunxin.conf` that reaches
an op and then dies inside FlagGems' own launch, several frames away from the
routing decision that allowed it.

Two facts this file exists for:

  * Kunlunxin's `_patch_flaggems_codegen_config` branch used to label the vendor
    and return *before any shim ran* (`... only labels the vendor and returns
    before any shim runs`). That is correct about the generic-NVIDIA shim -- it
    installs CUDA-backed generators this CPU wheel cannot construct -- but it
    also meant no shim at all, so `torch_device_fn._DeviceGuard(idx)` inside
    every FlagGems op wrapper hit CPU torch's
    `RuntimeError: PyTorch was compiled without CUDA support` and the kernel
    never launched. The branch must now call the two entry points below and
    still not reach the NVIDIA one.

  * The device-name realignment has to happen after `import flag_gems`, because
    `DeviceDetector` copies `device_name` out of the vendor descriptor at
    construction and the op modules capture it into module globals as they are
    imported. Calling it before the import would rewrite nothing.

Measured on 10.21.1.13 (P800 96G), torch 2.10.0+cpu, FlagGems 5.4.0, FlagTree
xpu: with both pieces in place 22 of 22 elementwise and reduction ops ran on
device and matched a CPU reference (max|d| <= 2.4e-07); `DeviceDetector`
reported `vendor='kunlunxin' name='flagos'`.
"""

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INIT = REPO_ROOT / "torch_fl" / "__init__.py"
COMPAT = REPO_ROOT / "torch_fl" / "accelerator" / "kunlunxin" / "_klx_compat.py"


def _kunlunxin_branch() -> str:
    """The body of the kunlunxin arm of ``_patch_flaggems_codegen_config``.

    Every branch in that chain opens with a ``# --- Name branch ---`` comment, so
    the arm ends at the next such comment. Cutting at the next same-indent `if`
    would instead run into the next branch's prologue -- MetaX's assigns
    ``_metax_requested`` before its ``if``.
    """
    text = INIT.read_text()
    start = text.index('if _build_accelerator() == "kunlunxin":')
    rest = text[start:]
    end = rest.index("\n    # --- ", 1)
    return rest[:end]


def test_compat_module_defines_the_two_entry_points():
    """Both pieces FlagGems needs exist and are exported by name."""
    tree = ast.parse(COMPAT.read_text())
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "install_klx_cuda_shim" in names
    assert "patch_klx_flag_gems_device_name" in names


def test_the_branch_installs_the_shim_and_a_realigns_the_name():
    """The kunlunxin arm calls both, rather than only labelling the vendor.

    This is the regression that matters: the arm used to be
    ``set_foreign(...); return`` and a route would then launch nothing.
    """
    body = _kunlunxin_branch()
    assert "install_klx_cuda_shim()" in body
    assert "patch_klx_flag_gems_device_name()" in body


def test_the_branch_does_not_reach_the_generic_nvidia_shim():
    """It must keep returning before the NVIDIA branch, for the stated reason.

    That shim binds the flagos RNG onto ``torch.Generator(device="cuda")``,
    which a CPU wheel cannot construct, so reaching it would break the native
    PrivateUse1 generator instead of fixing anything.
    """
    body = _kunlunxin_branch()
    assert "patch_torch_cuda_for_flagos" not in body
    assert "install_klx_rng" not in body
    assert body.rstrip().endswith("return")


def test_the_realignment_runs_only_after_importing_flag_gems():
    """``patch_klx_flag_gems_device_name()`` sits in an ``else:`` of the import.

    The detector's name is copied at construction and the op modules cache it at
    import time, so realigning before ``import flag_gems`` would be a no-op.
    """
    body = _kunlunxin_branch()
    imported = body.index("import flag_gems")
    called = body.index("patch_klx_flag_gems_device_name()")
    assert imported < called
    assert "else:" in body[imported:called]


def test_the_shim_forwards_only_the_entry_points_flaggems_calls():
    """A narrow shim: the ones FlagGems reaches for, and no generators.

    ``install_klx_cuda_shim`` must not define ``default_generators`` -- that is
    the NVIDIA shim's job and is deliberately not done here.
    """
    text = COMPAT.read_text()
    for name in (
        "current_device",
        "set_device",
        "device_count",
        "is_available",
        "synchronize",
        "get_device_properties",
        "get_device_capability",
        "current_stream",
        "_exchange_device",
        "_maybe_exchange_device",
        "_lazy_init",
    ):
        assert re.search(rf"cuda\.{re.escape(name)} = ", text), name
    assert "default_generators" not in text


def test_both_installers_are_idempotent():
    """Each is guarded by a module-level flag, so two imports cannot double-apply."""
    text = COMPAT.read_text()
    assert "_cuda_shim_installed = False" in text
    assert "_device_name_patched = False" in text
    assert "if _cuda_shim_installed:" in text
    assert "if _device_name_patched:" in text


def test_the_documented_mismatch_matches_the_measured_values():
    """The two names the realignment bridges, as measured on the chip."""
    text = COMPAT.read_text()
    assert '_FLAGGEMS_DEVICE_NAME = "flagos"' in text
    assert '_VENDOR_DEVICE_NAME = "cuda"' in text
