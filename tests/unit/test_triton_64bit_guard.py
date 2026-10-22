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

"""The 64-bit predicate that keeps a GCU300 kernel away from the vendor compiler.

The predicate is a pure function of what Inductor generated, so it is tested
here against the generated sources this repository actually observed, without
needing the vendor Triton stack. The compile-time behaviour is covered by the
GCU tests in tests/integration/test_compile.py.
"""

from types import SimpleNamespace

import pytest

from torch_fl.compile.triton_64bit_guard import wide_uses


# The reduction kernel from flex attention's create_block_mask: int32 operands,
# int32 signature, and an int64 index dtype Inductor promoted into the kernel.
# Legalizing arith.extsi on GCU300 segfaults the process. Trimmed to the lines
# that matter, so line numbers below are this copy's, not the generated file's.
_CRASHING_SRC = """\
def triton_red_fused__to_copy_arange_bitwise_and_constant_pad_nd_1(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK, R0_BLOCK):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (r0_index), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    tmp1 = tl.full([1, 1], 7, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp16 = tmp15.to(tl.int64)
"""

# An int64 pointwise kernel: the 64-bit type is in the signature, and the
# generated source never names it. The compiler rejects it cleanly today.
_INT64_SIGNATURE = {
    "in_ptr0": "*i64",
    "out_ptr0": "*i64",
    "xnumel": "i32",
    "XBLOCK": "constexpr",
}

_BENIGN_SRC = """\
tmp0 = tl.load(in_ptr0 + (x0), xmask)
tmp1 = tl.full([XBLOCK], 1, tl.int32)
tmp2 = tmp0 + tmp1
tl.store(out_ptr0 + (x0), tmp2, xmask)
"""


@pytest.mark.anyplatform
def test_promoted_index_is_found_by_line():
    """The crashing kernel is recognised, and the report points at the line."""
    found = wide_uses(_CRASHING_SRC, {"in_ptr0": "*i32", "xnumel": "i32"})
    assert found, "promoted int64 index not detected"
    lines = [where for where, _ in found]
    # Every tl.full(..., tl.int64) and the .to(tl.int64), but not the kernel
    # definition or the int32 loads around them.
    assert lines == ["line 6", "line 7", "line 9"], lines


@pytest.mark.anyplatform
def test_int64_operand_is_found_in_the_signature():
    """An int64 tensor is a 64-bit kernel even when its source says nothing."""
    found = wide_uses(_BENIGN_SRC, _INT64_SIGNATURE)
    assert [where for where, _ in found] == ["signature", "signature"]
    assert {what for _, what in found} == {"in_ptr0: *i64", "out_ptr0: *i64"}


@pytest.mark.anyplatform
def test_float64_operand_is_found():
    """float64 is the same limitation, and the compiler says so itself."""
    found = wide_uses(_BENIGN_SRC, {"in_ptr0": "*fp64", "out_ptr0": "*fp64"})
    assert len(found) == 2


@pytest.mark.parametrize(
    "signature",
    [
        {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32"},
        {"in_ptr0": "*i32", "xnumel": "i32", "XBLOCK": "constexpr"},
        None,
    ],
)
@pytest.mark.anyplatform
def test_thirty_two_bit_kernels_are_untouched(signature):
    """The guard must not refuse a kernel the compiler handles.

    Every kernel in the passing integration cohort is in this shape, so a false
    positive here is a compile that used to work and now raises.
    """
    assert wide_uses(_BENIGN_SRC, signature) == []


@pytest.mark.anyplatform
def test_a_longer_identifier_is_not_a_64_bit_type():
    """Whole-word matching, so a name that merely contains 'int64' is ignored."""
    src = "tmp0 = tl.load(in_ptr0 + x_int64_stride, xmask)\n"
    assert wide_uses(src, {"in_ptr0": "*fp32"}) == []


@pytest.mark.anyplatform
def test_signature_accepts_a_name_value_sequence():
    """Inductor hands over a dict, but a sequence of pairs must work too."""
    found = wide_uses(None, [("in_ptr0", "*i64")])
    assert [what for _, what in found] == ["in_ptr0: *i64"]


class _FakeAutotuner:
    """The two attributes the guard reads off a CachingAutotuner."""

    def __init__(self, src, signature, kernel_name="triton_fake_0"):
        self.fn = SimpleNamespace(src=src)
        self.inductor_meta = {"signature": signature, "kernel_name": kernel_name}


@pytest.mark.anyplatform
def test_patch_refuses_a_wide_kernel_before_compiling_it(monkeypatch):
    """The refusal has to happen *before* triton.compile is entered.

    That is the whole point of the patch: the vendor pass manager aborts the
    process on a 64-bit kernel, and the abort is below the try/except Inductor
    wraps around triton.compile, so a check placed any later never runs.
    """
    th = pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    from torch._inductor.exc import InductorError
    from torch_fl.compile import triton_64bit_guard as g

    monkeypatch.setattr(g, "gcu_64bit_unsupported", lambda: True)
    monkeypatch.setattr(th, g._PATCH_FLAG, False, raising=False)
    compiled = []
    monkeypatch.setattr(
        th.CachingAutotuner,
        "_precompile_config",
        lambda self, cfg: compiled.append(cfg) or "compiled",
    )

    g.patch_triton_64bit_guard()

    autotuner = _FakeAutotuner(_CRASHING_SRC, {"in_ptr0": "*i32"})
    with pytest.raises(InductorError, match="no 64-bit support"):
        th.CachingAutotuner._precompile_config(autotuner, cfg=None)
    assert compiled == [], "a 64-bit kernel reached the compiler anyway"

    benign = _FakeAutotuner(_BENIGN_SRC, {"in_ptr0": "*fp32"})
    assert th.CachingAutotuner._precompile_config(benign, cfg=7) == "compiled"
    assert compiled == [7], "a 32-bit kernel must still compile"


@pytest.mark.anyplatform
def test_the_error_names_the_kernel_and_the_offending_lines(monkeypatch):
    """A refusal a user cannot act on is not worth the raised exception."""
    pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    from torch_fl.compile import triton_64bit_guard as g

    error = g._unsupported_error(
        "triton_poi_fused_add_0", wide_uses(_CRASHING_SRC, None)
    )
    message = str(error)

    assert "triton_poi_fused_add_0" in message
    assert "line 6" in message and "tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)" in message
    assert "FLAGOS_COMPILE_FALLBACK_EAGER" in message
    assert "create_block_mask" in message


@pytest.mark.anyplatform
def test_patch_is_skipped_off_gcu(monkeypatch):
    """64-bit kernels compile on every other target, so nothing may be wrapped."""
    th = pytest.importorskip("torch._inductor.runtime.triton_heuristics")
    from torch_fl.compile import triton_64bit_guard as g

    monkeypatch.setattr(g, "gcu_64bit_unsupported", lambda: False)
    before = th.CachingAutotuner._precompile_config

    g.patch_triton_64bit_guard()

    assert th.CachingAutotuner._precompile_config is before
