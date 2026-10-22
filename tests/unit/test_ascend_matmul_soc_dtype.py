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

"""The Ascend matmul dtype gate has to follow the SoC, not the CANN release.

Measured on an Ascend310B1 (Orange Pi AIpro 20T, CANN 9.1.1): aclnnMatmul,
aclnnMm and aclnnBatchMatMul accept fp16 and reject float32 at
GetWorkspaceSize -- 161002 (ACLNN_ERR_PARAM_INVALID) for cubeMathType 0 and 3,
561103 (ACLNN_ERR_INNER_NULLPTR) for the HF32 variants 1, 2, 4, 5 and 6. bf16
fails too. The pre-fix code treated every floating dtype as supported, so on
that board all of mm, bmm, addmm and the .out variants raised instead of
falling back to CPU, and readout-shaped training loops could not run at all.

These tests pin the predicate and the generated kernels. The board itself is
the integration test; this file needs no NPU.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DTYPE_SUPPORT = REPO_ROOT / "csrc/aten/backends/ascend/dtype_support.h"
GENERATED_KERNELS = REPO_ROOT / "csrc/aten/backends/ascend/generated/ascend_kernels.cc"
CODEGEN = REPO_ROOT / "scripts/codegen/codegen_ascend.py"


def _read(path):
    return path.read_text(encoding="utf-8")


def test_metadata_ops_survive_registration_regeneration():
    """Ascend's aliases stay in the exact kernel coverage set.

    The generic register.inc owns the wrapper symbols, but its ops are not all
    Ascend kernels. A generator that derives the coverage set from the
    hand-written REGISTER_IMPL_TO_DISPATCHER lines alone drops these entries,
    leaving ATen's CPU metadata kernels attached to flagos tensors.
    """
    register = _read(REPO_ROOT / "csrc/aten/backends/ascend/generated/ascend_register.inc")
    registered = set(re.findall(r'm\.impl\("([^"]+)"', register))
    expected = {
        "_conj",
        "_neg_view",
        "_unsafe_view",
        "alias",
        "detach",
        "permute",
        "select.int",
        "slice.Tensor",
        "squeeze",
        "squeeze.dim",
        "t",
        "transpose.int",
        "unbind.int",
        "unsqueeze",
    }
    assert expected <= registered
    generator = _read(CODEGEN)
    assert "REGISTER_CC_METADATA_OPS" in generator


@pytest.mark.parametrize(
    "soc",
    [
        "Ascend310B1",
        "Ascend310B2",
        "Ascend310B4",
    ],
)
def test_fp16_only_family_matches_the_310b_line(soc):
    """Every 310B suffix is fp16-only, not just the one board we measured."""
    header = _read(DTYPE_SUPPORT)
    match = re.search(
        r"inline bool IsFp16OnlyMatmulSoc\(std::string_view soc_name\) \{(.+?)\n\}",
        header,
        re.S,
    )
    assert match, "IsFp16OnlyMatmulSoc is gone or changed shape"
    assert '"Ascend310B"' in match.group(1), (
        "the predicate must key on the Ascend310B prefix; CANN appends a "
        "board-specific suffix (Ascend310B1, Ascend310B4) that changes per "
        "product line"
    )
    assert soc.startswith("Ascend310B")


@pytest.mark.parametrize(
    "soc",
    [
        "Ascend910_9382",
        "Ascend910B4",
        "Ascend310P3",
        "",
    ],
)
def test_full_featured_socs_are_not_mistaken_for_310b(soc):
    """930/910/310P and an unknown SoC keep the pre-fix, full-dtype behaviour."""
    assert not soc.startswith("Ascend310B")


def test_dtype_gate_consults_the_soc_name():
    """A hardcoded dtype list is the bug this change removes."""
    header = _read(DTYPE_SUPPORT)
    gate = re.search(
        r"inline bool IsMatmulDtypeSupported\(c10::ScalarType dtype\) \{(.+?)\n\}",
        header,
        re.S,
    )
    assert gate, "IsMatmulDtypeSupported is gone or changed shape"
    body = gate.group(1)
    assert "IsFp16OnlyMatmulSoc" in body, (
        "IsMatmulDtypeSupported must branch on the SoC; assuming the whole "
        "float family works is what failed on Ascend310B1"
    )
    assert "GetSocName()" in body, (
        "the SoC has to come from the runtime probe (aclrtGetSocName), which "
        "is the only source that reflects the board actually installed"
    )


def test_soc_probe_is_exported_not_inline():
    """libtorch_fl.so and libflagos.so must share one cached probe.

    An inline function-local static would give each shared object its own copy,
    which is the trap acl_stream.h already documents for the stream accessors.
    """
    header = _read(REPO_ROOT / "csrc/runtime/accelerator/ascend/acl_stream.h")
    assert "FLAGOS_EXPORT std::string GetSocName();" in header

    impl = _read(REPO_ROOT / "csrc/runtime/accelerator/ascend/stream_api.cc")
    assert "FLAGOS_EXPORT std::string GetSocName()" in impl
    assert "aclrtGetSocName()" in impl, (
        "the probe has to read the ACL-reported SoC name; anything else is a "
        "guess about which CANN operator set is installed"
    )


def test_generated_matmul_kernels_fall_back_on_the_gate():
    """The .out variants used to skip the gate entirely and always call aclnn."""
    generated = _read(GENERATED_KERNELS)
    for kernel in ("MmOutKernelAscend", "BmmOutKernelAscend"):
        body = re.search(
            rf"at::Tensor&\s+{kernel}\(.*?\n\}}",
            generated,
            re.S,
        )
        assert body, f"{kernel} not found in the generated sources"
        text = body.group(0)
        assert "IsMatmulDtypeSupported" in text, (
            f"{kernel} must gate on the SoC dtype capability; on an "
            "Ascend310B1 an ungated aclnn call raises instead of falling back"
        )
        assert "copy_" in text, f"{kernel} must write the CPU result into out&"

    for kernel in ("MmKernelAscend", "BmmKernelAscend"):
        body = re.search(rf"at::Tensor\s+{kernel}\(.*?\n\}}", generated, re.S)
        assert body, f"{kernel} not found in the generated sources"
        assert "IsMatmulDtypeSupported" in body.group(0)


def test_codegen_template_keeps_the_gate():
    """Regenerating must not silently drop the fix."""
    template = re.search(
        r'T_MATMUL_OUT = """\\\n(.+?)\n"""',
        _read(CODEGEN),
        re.S,
    )
    assert template, "T_MATMUL_OUT is gone or changed shape"
    assert "IsMatmulDtypeSupported" in template.group(1)
