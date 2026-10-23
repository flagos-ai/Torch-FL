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

"""Stage-4 Apple Silicon (Metal / MPSGraph) division and reciprocal.

``add``/``sub``/``mul`` have been served since stage 2; division is the fourth
and last of the elementwise arithmetic ops, and the one the transformers reach
for at two places that matter -- the ``1 / sqrt(d)`` attention scale and the
``x / sum(exp(x))`` tail of a hand-written softmax. ``reciprocal`` rides with it,
because ``1 / x`` is division with a constant numerator and the caller who wants
a reciprocal *writes* that.

Three properties are pinned here, and the second is the reason this file is not
just a copy of ``test_mps_binary.py``.

* **The tensor and scalar spellings are one node family.** MPSGraph offers both
  ``divisionWithPrimaryTensor:secondaryTensor:`` and ``reciprocalWithTensor:``,
  and a port that reached for a reciprocal-multiply -- the usual trick on
  hardware where division is slow -- would be building a second node. Measured
  against ``at::div`` over the cases that would separate the readings (a zero
  denominator of either sign, a signed-zero numerator, the inf/nan pairs, and
  the denormals), the two readings are bit-identical to each other *and* to the
  reference, so the kernel takes the one node and compares exact below rather
  than with a tolerance.
* **A subnormal *result* is flushed, on the operand-array path only.** This is
  the one measured difference from the reference and the one place the tests
  assert *against* the CPU rather than with it. ``1e-38 / 2.0`` returns
  ``5e-39`` and ``torch.tensor(1e-38, device=...) / torch.tensor(2.0, ...)``
  returns ``0`` -- same values, different divisor spelling -- and ``mul.Tensor``
  has behaved identically since stage 2. The boundary is the result, at the
  smallest normal float; the tests below walk both sides of it so the property
  is met as a documented edge rather than bisected out of a model.
* **``reciprocal`` is gated by the division predicate, not the unary one.** The
  unary gate is wider (it serves the integer widths). It must not be used here:
  ``x / 2`` and ``1 / x`` are one node family, and two ops built on it that
  disagree about which dtypes they serve would be a divergence a reader cannot
  see. The dtype cases hold the gate the two share.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import routed_backend_or_none
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}

# float16 and bfloat16 carry ~3 decimal digits, so a graph that rounds once and a
# reference that rounds once can differ by an ulp or two of the narrow type.
LOW_PRECISION_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    return TOLERANCE if dtype == torch.float32 else LOW_PRECISION_TOLERANCE


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(
        out.cpu().float(), expected.float(), **_tolerance(expected.dtype)
    )


def _tensor_div(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """``aten::div.Tensor`` on device tensors, the two-operand overload.

    The opaque aten spelling rather than ``a / b``: both reach this kernel, but
    the tensor overload is the one whose gate is over two dtypes and is
    therefore the one the dtype cases have to name.
    """
    return torch.ops.aten.div.Tensor(self, other)


def _scalar_div(self: torch.Tensor, other) -> torch.Tensor:
    """``aten::div.Scalar`` -- the host-scalar overload.

    ``a / 2.0`` decomposes to the *Tensor* overload before PrivateUse1 dispatch
    -- measured, not assumed: on this device ``FLAGOS_LOG=dispatch`` prints
    ``div.Tensor -> mps`` for ``a / 2.0`` and ``div.Scalar -> mps`` only for the
    opaque spelling here. The two overloads are distinct kernels with distinct
    gates (the scalar one folds the divisor into a constant), so a test about the
    scalar kernel has to spell it out; the dispatch test names both, so the
    decomposition itself is pinned there.
    """
    return torch.ops.aten.div.Scalar(self, other)


def _graph_flushes_subnormals() -> bool:
    """Whether this GPU's MPSGraph flushes a subnormal *result* to zero.

    The boundary ``TestSubnormalBoundaryIsDocumented`` walks is real, but which
    side of it the graph lands on is a property of the Metal the process is
    running on rather than of this backend. Measured on the two lanes this
    suite runs on: an **M5** (macOS 26 / Metal 4) returns ``0`` for
    ``1e-38 / 2.0``, and an **M1** (macOS 14 / Metal 3, the earlier hosted
    ``macos-14`` runner) returns ``5e-39`` -- the subnormal itself. The same
    split shows through ``mul.Tensor``, which has shared the behaviour since
    stage 2, so it is the graph's arithmetic rather than this op's.

    The cases below therefore assert *the documented outcome for the policy the
    machine actually has*, and this probe is what tells them which one that is.
    It is measured once at import rather than assumed from the OS version: the
    version string is a proxy for the Metal feature set, and the feature set is
    what the assertion is about. A machine whose graph flushes and one whose
    graph keeps the subnormal are both pinned -- a machine that does something
    in between still fails, because the probe compares against zero exactly.
    """
    operand = torch.tensor([1e-38]).to(DEVICE)
    divisor = torch.tensor([2.0]).to(DEVICE)
    return torch.ops.aten.div.Tensor(operand, divisor).cpu().item() == 0.0


# Measured once, at import, on whatever GPU this process is running on.
GRAPH_FLUSHES_SUBNORMALS = _graph_flushes_subnormals()


def _assert_subnormal_policy(got: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert a subnormal *result* against whichever policy this GPU has.

    ``expected`` is the CPU reference and is the subnormal itself, so the two
    documented outcomes are ``0`` (this graph flushes) and ``expected`` (this
    graph represents it). Anything else is neither policy and fails here rather
    than in a model -- which is what keeps this a boundary rather than a
    tolerance.
    """
    assert expected.item() < torch.finfo(expected.dtype).tiny
    assert expected.item() != 0.0
    if GRAPH_FLUSHES_SUBNORMALS:
        assert got.item() == 0.0
    else:
        assert got.item() == expected.item()


class TestDivTensorCorrectness:
    """``div.Tensor`` over two flagos tensors equals the CPU reference."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = (torch.randn(4, 64) * 3.0).to(dtype)
        # A divisor bounded away from zero and of one sign, so the case is about
        # the arithmetic rather than about the edge cases the classes below own.
        divisor = (torch.rand(4, 64) + 0.5).to(dtype)
        _assert_matches_cpu(
            _tensor_div(_to_device(host), _to_device(divisor)), host / divisor
        )

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((4, 5), (5,)),
            ((4, 5), (4, 1)),
            ((2, 3, 4), (4,)),
            ((2, 3, 4), (3, 1)),
            ((2, 1, 3), (1, 4, 1)),
            ((4, 5), ()),
        ],
    )
    @pytest.mark.anyplatform
    def test_broadcast_pairs(self, shape_a, shape_b):
        """MPSGraph's division broadcasts exactly as aten does.

        Each operand is described to the graph in its own shape and the result
        is sized by the reference's own ``at::infer_size``, so the pairs walk
        the broadcast cases aten distinguishes: a trailing dimension against a
        shorter operand, a size-1 axis on either side, and a 0-dim denominator
        (the ``a / 2.0``-as-a-tensor shape).
        """
        a = torch.randn(shape_a) * 2.0
        b = (torch.rand(shape_b) + 0.5) * 2.0
        out = _tensor_div(_to_device(a), _to_device(b))
        expected = a / b
        assert out.shape == expected.shape
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), expected, **TOLERANCE)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows, not a multiple of 16
            (3, 7),
            (5, 4),  # controls: rows already a multiple of 16 bytes
            (4, 8, 16),
            (2, 3, 5),
            (128, 3),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_at_row_widths_that_are_not_a_16_byte_multiple(self, shape):
        """The shapes MPSNDArray's row pitch would silently corrupt.

        See the row-pitch note in ``mps_common.h``: a (4, 5) float32 operand has
        20-byte rows that MPSNDArray reads as 32, so an aten-shaped descriptor
        would take every row after the first from the wrong offset. Both
        operands and the result go through the same 1-D-then-reshape
        marshalling; the aligned shapes are what show the padding was the
        variable rather than the shape.
        """
        a = torch.randn(*shape) * 3.0
        b = (torch.rand(*shape) + 0.5) * 3.0
        out = _tensor_div(_to_device(a), _to_device(b))
        torch.testing.assert_close(out.cpu(), a / b, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_attention_scale_and_softmax_tail(self):
        """The two shapes division is on the hot path for.

        The ``1 / sqrt(d)`` scale is a *scalar* divisor on a (batch, heads, seq,
        seq) score tensor, and the softmax tail is a tensor-over-tensor division
        of the same tensor by a (batch, heads, seq, 1) normaliser -- the pair
        that exercises broadcasting on a four-dimensional operand, which is
        where a marshalling mistake would show up as a wrong shape rather than a
        wrong value.
        """
        scores = torch.randn(1, 4, 64, 64) * 8.0
        scale = 1.0 / (64**0.5)
        _assert_matches_cpu(_scalar_div(_to_device(scores), scale), scores / scale)

        exp = torch.exp(scores)
        normaliser = exp.sum(dim=-1, keepdim=True)
        _assert_matches_cpu(
            _tensor_div(_to_device(exp), _to_device(normaliser)), exp / normaliser
        )

    @pytest.mark.anyplatform
    def test_strided_operands_are_materialised(self):
        """A non-contiguous operand takes the same answer as the CPU.

        MPSGraph reads dense memory, so the kernel materialises each operand
        first. Both sides are strided here, and the narrowing leaves 5 columns
        after the stride, so the materialised operands have the unaligned rows
        too.
        """
        a = _to_device(torch.randn(8, 10))[:, ::2]
        b = _to_device(torch.rand(8, 10) + 0.5)[:, ::2]
        assert not a.is_contiguous()
        assert not b.is_contiguous()
        out = _tensor_div(a, b)
        torch.testing.assert_close(out.cpu(), a.cpu() / b.cpu(), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_transposed_operands(self):
        """A transposed operand is materialised, not reinterpreted."""
        host = torch.randn(6, 4)
        divisor = torch.rand(4, 6) + 0.5
        a = _to_device(host).t()
        b = _to_device(divisor)
        out = _tensor_div(a, b)
        torch.testing.assert_close(out.cpu(), host.t() / divisor, **TOLERANCE)

    @pytest.mark.parametrize("shape", [(0, 4), (0,), (3, 0, 5)])
    @pytest.mark.anyplatform
    def test_empty_tensor(self, shape):
        """No elements is an empty result, not a graph run.

        MPSGraph rejects an empty tensor, so the kernel returns the correctly
        shaped empty output before it builds anything.
        """
        a = _to_device(torch.empty(*shape))
        b = _to_device(torch.empty(*shape))
        out = _tensor_div(a, b)
        assert out.shape == shape
        assert out.numel() == 0
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero(self):
        host = torch.tensor(3.0)
        out = _tensor_div(_to_device(host), _to_device(torch.tensor(2.0)))
        assert out.shape == ()
        torch.testing.assert_close(out.cpu(), torch.tensor(1.5), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_broadcast_failure_raises_exactly_like_cpu(self):
        """A pair that cannot broadcast raises the reference's own error.

        ``at::infer_size`` is the kernel's own check: when it throws, the kernel
        takes the round-trip and the reference raises -- so a bad broadcast is a
        ``RuntimeError`` about the mismatched dimensions rather than a truncated
        result or a graph that read past an operand.
        """
        a = _to_device(torch.randn(2, 3))
        b = _to_device(torch.randn(4))
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            _tensor_div(a, b)


class TestDivScalarCorrectness:
    """``div.Scalar`` over a flagos tensor and a host scalar."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("value", [2.0, -1.5, 0.25, 8.0])
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype, value):
        """The divisor becomes a one-element MPSGraph constant of the dtype.

        One element is what MPSGraph broadcasts against an operand of any shape.
        The negative divisor is the case that would expose a constant built at
        the wrong element type or a sign dropped on the way in.
        """
        host = (torch.randn(4, 6) * 3.0).to(dtype)
        _assert_matches_cpu(_scalar_div(_to_device(host), value), host / value)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows
            (3, 7),
            (5, 4),  # controls
            (4, 8, 16),
            (2, 3, 5),
            (128, 3),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_at_row_widths_that_are_not_a_16_byte_multiple(self, shape):
        """The row-pitch rule, on the operand and the result of a scalar op."""
        host = torch.randn(*shape) * 3.0
        _assert_matches_cpu(_scalar_div(_to_device(host), 1.5), host / 1.5)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_integral_scalar_on_a_float_tensor_keeps_the_dtype(self, dtype):
        """An integer divisor on a float operand stays in the float width."""
        host = (torch.randn(4, 6) * 3.0).to(dtype)
        out = _scalar_div(_to_device(host), 2)
        assert out.dtype == dtype
        torch.testing.assert_close(
            out.cpu().float(), (host / 2).float(), **_tolerance(dtype)
        )

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.randn(8, 10)
        operand = _to_device(host)[:, ::2]
        assert not operand.is_contiguous()
        out = _scalar_div(operand, 2.0)
        torch.testing.assert_close(out.cpu(), host[:, ::2] / 2.0, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        operand = _to_device(torch.empty(0, 4))
        out = _scalar_div(operand, 2.0)
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"


class TestReciprocalCorrectness:
    """``reciprocal`` is ``1 / x`` on the same node family as ``div``."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = (torch.randn(4, 64) * 2.0 + 3.0).to(dtype)
        _assert_matches_cpu(torch.reciprocal(_to_device(host)), torch.reciprocal(host))

    @pytest.mark.parametrize(
        "shape",
        [(4, 5), (3, 7), (5, 4), (2, 3, 5, 7), (4096,), (1, 1)],
    )
    @pytest.mark.anyplatform
    def test_shape_independence(self, shape):
        host = torch.randn(*shape) + torch.sign(torch.randn(*shape)) + 0.5
        _assert_matches_cpu(torch.reciprocal(_to_device(host)), torch.reciprocal(host))

    @pytest.mark.anyplatform
    def test_agrees_with_division_by_the_same_operand(self):
        """``reciprocal(x)`` and ``1 / x`` build the same arithmetic.

        The two spellings are the reason the two ops share a gate, so they are
        held against each other as well as against the reference: a divergence
        between them is what a future change to either body would produce, and it
        would be invisible to a case that only compared one of them to the CPU.
        """
        host = torch.randn(4, 32) * 4.0 + torch.sign(torch.randn(4, 32)) * 1.5
        on_device = _to_device(host)
        one = _to_device(torch.ones_like(host))
        torch.testing.assert_close(
            torch.reciprocal(on_device).cpu(),
            _tensor_div(one, on_device).cpu(),
            **TOLERANCE,
        )

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.randn(8, 10) + 2.0
        operand = _to_device(host)[:, ::2]
        assert not operand.is_contiguous()
        out = torch.reciprocal(operand)
        torch.testing.assert_close(
            out.cpu(), torch.reciprocal(host[:, ::2]), **TOLERANCE
        )

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        operand = _to_device(torch.empty(0, 4))
        out = torch.reciprocal(operand)
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero(self):
        host = torch.tensor(4.0)
        out = torch.reciprocal(_to_device(host))
        assert out.shape == ()
        torch.testing.assert_close(out.cpu(), torch.tensor(0.25), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_aten_spelling_matches_the_public_one(self):
        """``torch.reciprocal`` and ``aten.reciprocal`` are one kernel."""
        host = torch.randn(4, 16) + 2.0
        torch.testing.assert_close(
            torch.reciprocal(_to_device(host)).cpu(),
            torch.ops.aten.reciprocal(_to_device(host)).cpu(),
            **TOLERANCE,
        )


class TestDivisionMatchesTheReferenceOnItsEdges:
    """The values where a wrong selector or a wrong constant would show up."""

    @pytest.mark.anyplatform
    def test_zero_and_signed_zero_denominators_are_bit_exact(self):
        """``x/0`` and ``x/-0`` keep the reference's result and sign.

        This is the case that separates ``a / b`` from ``a * reciprocal(b)`` if
        either reading is wrong, and the case a naive implementation gets wrong
        by folding the denominator into a constant. Compared bit for bit rather
        than with a tolerance: ``-0.0`` and ``0.0`` are different results and
        ``assert_close`` does not distinguish them at this tolerance.
        """
        numerators = [0.0, -0.0, 1.0, -1.0, 2.5]
        denominators = [0.0, -0.0]
        a = torch.tensor(numerators)
        b = torch.tensor(denominators)
        # Outer product of the pairs, so every combination is one case.
        grid_a = a.repeat_interleave(len(denominators))
        grid_b = b.repeat(len(numerators))
        got = _tensor_div(_to_device(grid_a), _to_device(grid_b)).cpu()
        expected = grid_a / grid_b
        # nan != nan, so the nan positions are checked by classification and the
        # rest bit for bit -- which is what distinguishes -inf from inf and -0.0
        # from 0.0 here.
        both_nan = torch.isnan(got) & torch.isnan(expected)
        assert torch.equal(torch.isnan(got), torch.isnan(expected))
        finite = ~torch.isnan(expected)
        assert torch.equal(
            got[finite].view(torch.int32), expected[finite].view(torch.int32)
        )
        assert both_nan.any()  # the sweep really did produce 0/0

    @pytest.mark.anyplatform
    def test_infinities_and_nans(self):
        """The inf/nan quadrants produce the reference's own answers."""
        a = torch.tensor(
            [float("inf"), float("-inf"), 1.0, float("nan"), 0.0, float("inf")]
        )
        b = torch.tensor([2.0, 2.0, float("inf"), 1.0, float("inf"), float("inf")])
        got = _tensor_div(_to_device(a), _to_device(b)).cpu()
        expected = a / b
        assert torch.equal(torch.isnan(got), torch.isnan(expected))
        finite = ~torch.isnan(expected)
        assert torch.equal(got[finite], expected[finite])
        # The quadrant this case is for: inf / inf is nan on both sides, and a
        # finite numerator over inf is the 0 on both sides.
        assert torch.isnan(expected[5]) and torch.isnan(got[5])
        assert expected[2].item() == 0.0 and got[2].item() == 0.0
        assert expected[4].item() == 0.0 and got[4].item() == 0.0

    @pytest.mark.anyplatform
    def test_an_exactly_representable_quotient_is_exact(self):
        """A quotient with no rounding to do is returned unchanged.

        Powers of two and a divisor of 1 are the cases where the arithmetic is
        exact, so any difference from the reference here is a marshalling or
        constant bug rather than the floating point of the operation.
        """
        a = torch.tensor([[1.0, 2.0, 4.0, -8.0, 0.5]] * 3)
        b = torch.tensor([[1.0, 2.0, 2.0, 4.0, 0.25]] * 3)
        got = _tensor_div(_to_device(a), _to_device(b)).cpu()
        assert torch.equal(got, a / b)


class TestSubnormalBoundaryIsDocumented:
    """The one measured difference from the reference, pinned as intentional.

    A subnormal *result* lands on a boundary between this backend and the CPU
    reference, and which side of it the graph falls on is a property of the GPU
    the process is running on rather than of division: the M5 (macOS 26 /
    Metal 4) flushes it to zero, and the M1 (macOS 14 / Metal 3, the earlier
    hosted ``macos-14`` runner) represents it. Both are the same arithmetic
    seen through different Metal feature sets, and the case is shared with the
    ops already routed to this backend since stage 2, so it is recorded here --
    as a boundary a reader meets deliberately -- rather than quietly tolerated
    by a loose tolerance.

    ``GRAPH_FLUSHES_SUBNORMALS`` is measured once at import and the cases below
    assert the documented outcome for whichever policy is present, so each
    machine's behaviour is pinned rather than one machine's being hard-coded.
    The boundary is the result rather than either operand, and it holds through
    both bodies, which is what the cases below walk.
    """

    @pytest.mark.anyplatform
    def test_a_subnormal_tensor_quotient_flushes_to_zero(self):
        """``tensor(1e-38) / tensor(2.0)`` is ``0`` while the CPU says ``5e-39``.

        On its own this looks like a bug, which is why the cases beside it exist:
        the same operand over a *normal* quotient is returned unchanged, and the
        already-shipped ``mul.Tensor`` flushes the identical way. Together they
        say the flush is a representation boundary in the graph's arithmetic
        rather than a wrong answer from this kernel.
        """
        tiny_operand = torch.tensor([1e-38])
        divisor = torch.tensor([2.0])
        got = _tensor_div(_to_device(tiny_operand), _to_device(divisor)).cpu()
        expected = tiny_operand / divisor
        _assert_subnormal_policy(got, expected)

    @pytest.mark.anyplatform
    def test_the_scalar_body_flushes_the_same_way(self):
        """The folded-constant body is not an exception to the flush.

        ``div.Scalar`` builds ``x / constant`` rather than ``x / operand``, so it
        would be reasonable to expect the subnormal to survive -- the reference's
        does. It does not: the graph flushes a subnormal result on both bodies,
        which is what makes this a property of the graph's arithmetic rather than
        of one overload. Held so a future change to the constant path is noticed
        here rather than in a model.
        """
        tiny_operand = torch.tensor([1e-38])
        expected = tiny_operand / 2.0
        got = _scalar_div(_to_device(tiny_operand), 2.0).cpu()
        _assert_subnormal_policy(got, expected)

    @pytest.mark.anyplatform
    def test_the_already_routed_mul_tensor_flushes_the_same_way(self):
        """``mul.Tensor`` has behaved identically since stage 2.

        If this case ever fails, the subnormal policy of the two ops has
        diverged and the note in ``mps_common.h`` -- which says ``div`` inherits
        ``mul``'s policy rather than inventing a second one -- is stale.
        """
        operand = torch.tensor([1e-38])
        factor = torch.tensor([0.5])
        got = (operand.to(DEVICE) * factor.to(DEVICE)).cpu()
        expected = operand * factor
        _assert_subnormal_policy(got, expected)

        # Same shape as the division cases: on a flushing graph the opaque
        # spelling reaches the kernel and flushes, while the Python operator
        # resolves to a composite that never builds the graph and keeps the
        # subnormal. That pair is what says the value belongs to the graph
        # rather than to the dtype. On a graph that keeps the subnormal there is
        # no difference to see, and both spellings agree with the reference.
        opaque = torch.ops.aten.mul.Scalar(operand.to(DEVICE), 0.5).cpu()
        if GRAPH_FLUSHES_SUBNORMALS:
            assert opaque.item() == 0.0
        else:
            assert opaque.item() == expected.item()
        assert (operand.to(DEVICE) * 0.5).cpu().item() == expected.item()

    @pytest.mark.anyplatform
    def test_the_python_operators_are_not_a_counterexample(self):
        """``x / 2.0`` keeps the subnormal because it never enters the graph.

        On a flushing graph this looks like a contradiction and is not one: the
        Python scalar operators are resolved by PyTorch's own composite wrapper
        before PrivateUse1 dispatch, so the value is computed by the wrapper over
        the parenthesised operation and no graph node is built for the constant.
        The opaque overloads -- which the wrapper is *not* in front of -- do build
        the node and do flush. Both spellings are pinned here, because a future
        reader who measured only the Python form would conclude the graph keeps
        subnormals. On a graph that keeps the subnormal the two spellings agree,
        and the case still holds that the operator form reaches the reference.
        """
        tiny_operand = torch.tensor([1e-38])
        expected = (tiny_operand / 2.0).item()
        assert expected < torch.finfo(torch.float32).tiny

        on_device = _to_device(tiny_operand)
        # The Python operator and the opaque tensor overload with a Python float
        # both resolve inside the composite, so both agree with the reference
        # regardless of the graph's own policy.
        assert (on_device / 2.0).cpu().item() == expected
        assert torch.ops.aten.div.Tensor(on_device, 2.0).cpu().item() == expected
        # The kernel-level spellings build the graph, so they follow its policy.
        graph_expected = 0.0 if GRAPH_FLUSHES_SUBNORMALS else expected
        assert torch.ops.aten.div.Scalar(on_device, 2.0).cpu().item() == graph_expected
        assert (
            torch.ops.aten.div.Tensor(on_device, torch.tensor([2.0]).to(DEVICE))
            .cpu()
            .item()
            == graph_expected
        )

    @pytest.mark.anyplatform
    def test_a_normal_quotient_beside_the_boundary_is_untouched(self):
        """The flush begins at the *result*, at the smallest normal float.

        A quotient that lands on or above the smallest normal is returned
        unchanged, so the difference is a representation boundary rather than a
        magnitude threshold the kernel approximates with.
        """
        largest_subnormal = torch.finfo(torch.float32).tiny * 2.0
        got = _tensor_div(
            _to_device(torch.tensor([largest_subnormal])),
            _to_device(torch.tensor([2.0])),
        ).cpu()
        expected = largest_subnormal / 2.0
        assert expected >= torch.finfo(torch.float32).tiny  # is normal
        assert got.item() == expected

        # And one step below it, the same operands do part.
        just_below = torch.finfo(torch.float32).tiny * 1.5
        assert just_below / 2.0 < torch.finfo(torch.float32).tiny
        below = _tensor_div(
            _to_device(torch.tensor([just_below])),
            _to_device(torch.tensor([2.0])),
        ).cpu()
        _assert_subnormal_policy(below, torch.tensor(just_below / 2.0))

    @pytest.mark.anyplatform
    def test_a_reciprocal_that_overflows_from_a_subnormal_operand(self):
        """``1 / 1e-38`` returns ``inf`` where the reference returns ``1e38``.

        The reciprocal of a value below the smallest normal is the other side of
        the same boundary, and it lands on the operand rather than the result: a
        graph that flushes treats the subnormal operand as the zero it flushes
        to and reaches ``inf`` where the reference reaches a finite result, and
        a graph that keeps the operand reaches the reference's own value. Both
        are pinned here.
        """
        operand = torch.tensor([1e-38])
        assert operand.item() < torch.finfo(torch.float32).tiny
        got = torch.reciprocal(_to_device(operand)).cpu()
        expected = torch.reciprocal(operand)
        assert expected.item() != float("inf")
        if GRAPH_FLUSHES_SUBNORMALS:
            assert got.item() == float("inf")
        else:
            assert got.item() == expected.item()
        # The same overflow through the tensor spelling.
        via_div = _tensor_div(
            _to_device(torch.tensor([1.0])), _to_device(operand)
        ).cpu()
        if GRAPH_FLUSHES_SUBNORMALS:
            assert via_div.item() == float("inf")
        else:
            assert via_div.item() == expected.item()

    @pytest.mark.anyplatform
    def test_the_narrow_widths_share_the_boundary(self):
        """bfloat16 flushes at its own smallest normal, not at float32's.

        The boundary is a property of the element type the graph is running, so
        a bfloat16 case below its own ``tiny`` flushes as well. Held so the
        property is not mistaken for a float32 special case.
        """
        tiny_bf16 = torch.finfo(torch.bfloat16).tiny
        operand = torch.tensor([tiny_bf16], dtype=torch.bfloat16)
        divisor = torch.tensor([2.0], dtype=torch.bfloat16)
        expected = operand / divisor
        assert expected.item() < tiny_bf16
        got = _tensor_div(_to_device(operand), _to_device(divisor)).cpu()
        _assert_subnormal_policy(got, expected)


class TestDivDeclinedDtypes:
    """The widths the graph does not serve stay correct through the round-trip."""

    @pytest.mark.anyplatform
    def test_float64_round_trips(self):
        """float64 has no MPSGraph element type; the value is still right."""
        a = torch.tensor([[1.5, -2.25]], dtype=torch.float64)
        b = torch.tensor([[0.5, 4.0]], dtype=torch.float64)
        out = _tensor_div(_to_device(a), _to_device(b))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), a / b, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_float64_reciprocal_round_trips(self):
        a = torch.tensor([[1.5, -2.25]], dtype=torch.float64)
        out = torch.reciprocal(_to_device(a))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), torch.reciprocal(a), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_integral_division_takes_the_reference_result(self):
        """``div(int, int)`` is a *float* tensor on the reference.

        This is the case that made the gate more than a formality. The graph's
        integral division truncates in place, so a gate borrowed from ``mul``
        (which *is* dtype-preserving) returned the int64 ``-3`` where the CPU
        returns the float ``-3.5`` -- a wrong answer and a wrong dtype, silently.
        The pair now declines and the reference answers; this holds both halves.
        """
        a = torch.tensor([[-7, -1, 0, 3, 11]])
        b = torch.tensor([[2, 2, 3, -2, 4]])
        out = _tensor_div(_to_device(a), _to_device(b))
        expected = torch.ops.aten.div.Tensor(a, b)
        assert expected.dtype == torch.float32
        assert out.dtype == expected.dtype
        torch.testing.assert_close(out.cpu(), expected, **TOLERANCE)
        assert out.cpu().flatten()[0].item() == -3.5  # not the truncated -3

    @pytest.mark.anyplatform
    def test_fractional_scalar_on_an_integral_tensor_promotes_like_cpu(self):
        """``int_tensor / 2.0`` is a float tensor on both paths.

        A fractional scalar moves the aten result off the operand's dtype, which
        a constant of the operand's own element type cannot express, so the
        kernel declines and the reference computes it. The dtype is the point.
        """
        host = torch.tensor([1, 2, 3])
        out = _scalar_div(_to_device(host), 2.0)
        expected = torch.ops.aten.div.Scalar(host, 2.0)
        assert out.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(out.cpu(), expected, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_integral_scalar_on_an_integral_tensor_is_float_like_cpu(self):
        """An integral divisor still yields a float tensor on the reference.

        The scalar spelling of the case above, and it needs its own gate for the
        same reason: the shared scalar predicate admits an integral operand with
        an integral divisor, which is right for ``mul`` and wrong here. The
        divisor is then free on a float operand, which the second half holds.
        """
        host = torch.tensor([[-7, -1, 0, 3, 11]])
        out = _scalar_div(_to_device(host), 2)
        expected = torch.ops.aten.div.Scalar(host, 2)
        assert expected.dtype == torch.float32
        assert out.dtype == expected.dtype
        torch.testing.assert_close(out.cpu(), expected, **TOLERANCE)

        # On a float operand both the integral and the fractional divisor are
        # served by the graph, so neither is a round-trip.
        float_host = host.float()
        for divisor in (2, 2.5):
            got = _scalar_div(_to_device(float_host), divisor)
            assert got.dtype == torch.float32
            torch.testing.assert_close(got.cpu(), float_host / divisor, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_mixed_float_widths_take_the_reference_result(self):
        """A mixed pair has no MPSGraph answer, so it round-trips.

        MPSGraph's arithmetic does not promote and aten's does
        (``float16 / float32 -> float32``), so a mismatched pair is declined.
        What has to hold is the result's dtype, which is the part a graph built
        from the operands' own element types would get wrong.
        """
        a = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
        b = torch.tensor([[0.5, 4.0]], dtype=torch.float32)
        out = _tensor_div(_to_device(a), _to_device(b))
        expected = a / b
        assert out.dtype == expected.dtype
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), expected, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_complex_round_trips(self):
        """A complex operand has no MPSGraph element type."""
        a = torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex64)
        b = torch.tensor([2.0 - 1.0j, 0.5 + 0.5j], dtype=torch.complex64)
        out = _tensor_div(_to_device(a), _to_device(b))
        assert out.dtype == torch.complex64
        torch.testing.assert_close(out.cpu(), a / b, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_bool_division_promotes_to_float_like_cpu(self):
        """``bool / bool`` is a float division rather than an error here.

        Unlike ``silu(bool)`` -- which has no definition and raises on every
        backend -- division *does* have one, so the gate declines the pair on its
        dtype and the reference answers with a float tensor. ``True / False`` is
        ``inf``, which is the case that shows the operand was read as a value
        rather than coerced.
        """
        a = _to_device(torch.tensor([True, False, True]))
        b = _to_device(torch.tensor([True, True, False]))
        out = _tensor_div(a, b)
        expected = torch.ops.aten.div.Tensor(
            torch.tensor([True, False, True]), torch.tensor([True, True, False])
        )
        assert expected.dtype == torch.float32
        assert out.dtype == expected.dtype
        assert out.cpu()[0].item() == 1.0
        assert out.cpu()[2].item() == float("inf")
        torch.testing.assert_close(out.cpu()[:2], expected[:2], **TOLERANCE)


class TestDivScheduleAndPlanReuse:
    """The marshalling contract every MPS kernel shares (mps_common.mm)."""

    @pytest.mark.anyplatform
    def test_interleaved_dtypes_and_shapes_never_share_a_plan(self):
        """A dtype or shape change on a warm cache still gets its own graph.

        The tensor body's key is the op name alone -- its operands' signatures
        are folded in by ``MpsRunGraph`` -- while the scalar body names the host
        divisor, so both kinds of key are on one cache here. A key that dropped
        either half would let a float32 (4, 6) graph answer a bfloat16 (6, 4)
        request, or a ``/ 2.0`` graph answer a ``/ 3.0`` one.
        """
        cases = [
            torch.randn(4, 6),
            torch.randn(4, 6).half(),
            torch.randn(6, 4),
            torch.randn(4, 6).bfloat16(),
            torch.randn(2, 3, 5),
            torch.randn(4, 6),
            torch.randn(6, 4).half(),
        ]
        divisors = [2.0, 3.0, 0.25, 2.0, 8.0, 3.0, 2.0]
        on_device = [_to_device(case) for case in cases]
        wanted_tensor = [
            case / _to_device(torch.full_like(case, 2.0)) for case in cases
        ]
        wanted_scalar = [case / divisor for case, divisor in zip(cases, divisors)]

        for round_ in range(3):
            for offset in range(len(cases)):
                index = (offset + round_) % len(cases)
                case = on_device[index]
                dtype = cases[index].dtype
                out = _tensor_div(case, _to_device(torch.full_like(case, 2.0)))
                torch.testing.assert_close(
                    out.cpu().float(),
                    wanted_tensor[index].cpu().float(),
                    **_tolerance(dtype),
                )
                out = _scalar_div(case, divisors[index])
                torch.testing.assert_close(
                    out.cpu().float(),
                    wanted_scalar[index].cpu().float(),
                    **_tolerance(dtype),
                )

    @pytest.mark.anyplatform
    def test_results_survive_block_reuse_and_immediate_readback(self):
        """Nothing waits for the GPU, so both ends of the call are checked here.

        An operand's block goes back to the pool as soon as the caller drops it,
        while the graph reading it has only been scheduled; and the value the
        caller reads back is produced by work the runtime has not awaited. Both
        rest on the sentinel command buffer the runtime commits behind the graph
        (see ``mps_common.h``).
        """
        held = []
        for _ in range(32):
            src = torch.randn(1 << 12, device=DEVICE) + 2.0
            divisor = torch.rand(1 << 12, device=DEVICE) + 0.5
            wanted = src.cpu() / divisor.cpu()
            out = _tensor_div(src, divisor)
            del src, divisor  # recycled while the graph above may still read them
            for _ in range(4):  # blocks the allocator is free to hand back out
                _tensor_div(
                    torch.randn(1 << 12, device=DEVICE) + 2.0,
                    torch.rand(1 << 12, device=DEVICE) + 0.5,
                )
            held.append((out.cpu(), wanted))  # read back without a synchronize

        for got, wanted in held:
            torch.testing.assert_close(got, wanted, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_repeated_calls_on_one_operand_are_stable(self):
        """The compiled plan is reused across calls and keeps answering."""
        src = torch.randn(64, 64, device=DEVICE) + 2.0
        wanted = torch.reciprocal(src.cpu())
        first = torch.reciprocal(src).cpu()
        for _ in range(16):
            torch.testing.assert_close(torch.reciprocal(src).cpu(), first, **TOLERANCE)
        torch.testing.assert_close(first, wanted, **TOLERANCE)


class TestDivDispatch:
    """The three overloads are claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    @pytest.mark.parametrize("op", ["div.Tensor", "div.Scalar", "reciprocal"])
    def test_conf_routes_op_to_mps(self, op):
        assert routed_backend_or_none(op) == "mps"

    @pytest.mark.main_ops
    @pytest.mark.parametrize(
        "op",
        [
            "div.Tensor_mode",
            "div.Scalar_mode",
            "div.out",
            "div.out_mode",
            "div.Scalar_out",
            "div.Scalar_mode_out",
            "div_.Tensor",
            "div_.Scalar",
            "reciprocal.out",
            "reciprocal_",
        ],
    )
    def test_conf_leaves_the_other_spellings_on_the_fallback(self, op):
        """The `mode` overloads and every `.out`/in-place form are unregistered.

        ``div.Tensor_mode`` / ``div.Scalar_mode`` carry a ``rounding_mode`` that
        would need a ``floor`` node *after* an exact division -- a composition
        rather than a keyword -- and the ``.out`` forms write into a tensor the
        caller owns, like every other ``.out`` here.
        """
        assert routed_backend_or_none(op) is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernels ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or the op reached
        ``cpu_fallback``, because both compute the same answer. The log line is
        emitted by the C++ dispatcher only on the ``mps`` slot. Run in a
        subprocess because ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, device='flagos:0'); "
                "b = torch.rand(4, 4, device='flagos:0') + 0.5; "
                "torch.ops.aten.div.Tensor(a, b); "
                "torch.ops.aten.div.Scalar(a, 2.0); "
                "torch.ops.aten.reciprocal(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        for line in (
            "[flagos dispatch] div.Tensor -> mps",
            "[flagos dispatch] div.Scalar -> mps",
            "[flagos dispatch] reciprocal -> mps",
        ):
            assert line in result.stderr, result.stderr

    @pytest.mark.main_ops
    def test_the_python_operators_reach_the_same_kernels(self):
        """The Python spellings decompose onto these kernels, and the log says so.

        Measured on this device rather than assumed: ``a / b`` reaches
        ``div.Tensor``, ``a / 2.0`` decomposes to ``div.Tensor`` as well (not to
        ``div.Scalar`` -- PyTorch wraps the Python number as a scalar inside the
        Tensor overload's own wrapper), and ``2.0 / a`` decomposes the *other*
        way, to ``reciprocal`` then ``mul``. All three land on this increment,
        which is why they are held together: a change that moved any of them off
        the kernel would otherwise show up as a correct answer computed on the
        fallback.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, device='flagos:0'); "
                "b = torch.rand(4, 4, device='flagos:0') + 0.5; "
                "a / b; a / 2.0; 2.0 / a; torch.reciprocal(a); a.reciprocal()",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # `a / b` and `a / 2.0` are the two Tensor dispatches; the reciprocal is
        # reached three times (`2.0 / a`, `torch.reciprocal`, `.reciprocal()`).
        assert result.stderr.count("[flagos dispatch] div.Tensor -> mps") == 2
        assert result.stderr.count("[flagos dispatch] reciprocal -> mps") == 3

    @pytest.mark.main_ops
    def test_a_declined_dtype_keeps_the_reference_dtype(self):
        """A float64 call declines the graph and returns float64.

        The decline is what keeps the reference's own dtype and error, so a case
        that ran the graph anyway would be a wrong answer rather than a slow one.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, dtype=torch.float64, device='flagos:0'); "
                "b = torch.rand(4, 4, dtype=torch.float64, device='flagos:0'); "
                "print(torch.ops.aten.div.Tensor(a, b).dtype); "
                "print(torch.ops.aten.reciprocal(a).dtype)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("torch.float64") == 2
