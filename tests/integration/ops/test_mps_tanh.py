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

"""Stage-4 Apple Silicon (Metal / MPSGraph) tanh.

``tanh`` is the activation a transformer block reaches for at both ends -- the
tanh approximation inside a GELU and the one inside a recurrent or attention
gate -- and MPSGraph names it directly as ``tanhWithTensor:``. The body is the
shared one-node unary template plus one guard, and both the guard and the gate
are the increment's findings:

* **The node saturates a NaN operand, so the body guards it.** Measured at all
  three float widths, ``tanh(nan)`` on the bare node is ``1.0`` where the
  reference answers ``nan``. That is the one deviation in this body that is a
  wrong *answer* rather than a rounding, and it is what separates a tanh from a
  saturating approximation -- so ``isNaNWithTensor:`` selects a NaN back over the
  node's answer, and all three widths then propagate nan. The case below is the
  regression test for that guard.
* **float16 and bfloat16 are served, and the node is the reference's own
  function at those widths.** Both are bit-identical to the reference over a
  400001-sample sweep of ``[-12, 12]`` except for the last-ulp roundings and the
  subnormal band recorded below, and both are *exactly* odd over the same sweep
  -- ``max |g(x) + g(-x)|`` is ``0.0``, the reference's own value.
* **float32 is declined, and this is the first gate here to decline the
  *widest* float.** The node is a coarser approximation of a tanh at this width
  rather than a differently-rounded exact one: it breaks odd symmetry by
  1.788e-07 where the reference is exactly symmetric, its error against the exact
  answer is ~100x the reference's at every scale of the curve, and 41% of a
  400001-sample sweep of ``[-10, 10]`` comes back a different float32 value --
  up to a visibly different number inside the saturation band rather than a
  neighbouring ulp. Serving that would replace the reference's answer with a
  coarser one on a majority of the values the most-called activation in a
  transformer block is handed, so the float32 caller keeps the ``cpu_fallback``
  round-trip. See ``MpsSupportsTanhDtype`` for the numbers and for the ~2x the
  decline leaves on the table.

Two edges are host-dependent rather than this body's, and both were measured
*both* ways -- on an M5 with a current SDK and on the ``macos-14`` runner used
by the initial CI lane:

* the sign of a **signed zero**. On the M5, ``tanh(-0.0)`` comes back ``+0.0``
  where the reference answers ``-0.0`` at float16 and bfloat16; the CI runner
  answers the reference's ``-0.0``. The value is identical either way and the
  curve is odd everywhere else.
* the **bfloat16 subnormal band**. On the M5 all 127 subnormal bfloat16 operands
  (``bits 0x0001`` to ``0x007f``) come back ``0``; the CI runner reproduces the
  reference's subnormal answers exactly. This is the flush the sibling
  ``sigmoid`` recorded at the result side, read at the operand, and float16 does
  not flush at all on either host (all 1023 of its positive subnormals reproduce
  the reference bit for bit).

Because neither reading is stable across hosts, the cases below assert what *is*:
a zero for the signed-zero operand with the sign preserved everywhere the operand
is not a zero, a subnormal operand answering the reference or a flushed zero, and
the band's far edge -- the smallest normal and up -- reproduced bit for bit. The
deviations themselves are recorded in ``MpsSupportsTanhDtype`` and here, not
pinned as an assertion that one of the two machines would fail.

The tests below hold what the numbers alone cannot: that the two narrow widths
are served and agree with ``at::tanh``, that nan propagates at every width through
the guard, that the curve is odd at the served widths -- the property that
separates a tanh from the saturating node float32 was declined for -- that the
edges (``+-inf`` -> ``+-1``) are the ones the reference answers, that neither
host-dependent edge leaves its band, and that float32, float64 and the
integral/bool operands take the round-trip.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401

from backend_conf import dispatched_to_mps, routed_backend_or_none
from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

# The widths the graph serves: the two narrow ones, where the node is the
# reference's own function. float32 is *not* here and is the increment's finding;
# see MpsSupportsTanhDtype.
SERVED_DTYPES = [torch.float16, torch.bfloat16]

# The widths the reference has a kernel for and this body declines. float64 has
# no MPSGraph element type; float32 is declined on node fidelity. The round-trip
# answers both, so the assertion is CPU equality at the operand's own dtype
# rather than an error.
DECLINED_FLOAT_DTYPES = [torch.float32, torch.float64]

# The curve's edges at the served widths. ``tanh`` is bounded, so both infinities
# answer a *finite* +-1 -- the edge a body built on ``exp`` or on a reciprocal
# gets wrong -- and the guard is what answers the nan. The signed zero is *not*
# here: it is the one edge the served widths miss, and it has its own case so
# that the miss is asserted rather than averaged into this list.
DOMAIN_EDGES = [
    0.0,  # 0.0
    float("inf"),  # 1.0
    float("-inf"),  # -1.0
    float("nan"),  # nan, and the guard is the assertion
    1.0,  # 0.7615941559557649
    -1.0,  # -0.7615941559557649
    -7.0,  # -0.9999983369439446, near the bound but not saturated
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::tanh`` on the host, in the operand's own dtype.

    The op and not a ``(exp(2x) - 1) / (exp(2x) + 1)`` composition: the
    composition's extra roundings are measurable against it -- measured over
    [-10, 10] at float32 it differs from the reference on 151496 of 400001
    samples, more than the node it would replace -- and the reference is what
    the kernel is held to.
    """
    return torch.ops.aten.tanh(host)


def _assert_matches_cpu(
    out: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-2
) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(out.cpu().float(), expected.float(), rtol=rtol, atol=0)


def _same(x: float, y: float) -> bool:
    """Equality that treats two nans as the same answer."""
    return (x == y) or (x != x and y != y)


class TestTanhCorrectness:
    """``torch.tanh`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = torch.randn(4, 64).to(dtype)
        _assert_matches_cpu(torch.tanh(_to_device(host)), _reference(host))

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_curve_is_the_reference_to_within_one_ulp(self, dtype):
        """The increment's finding, and the reason the width can be served.

        Measured, the two served widths reproduce ``at::tanh`` over a 400001
        sample sweep of ``[-12, 12]`` almost everywhere: float16 differs on 5
        samples, bfloat16 on 1, and every one of them is within a *few* of the
        width's own representable steps from the reference's answer. That is the
        bar the node is held to here, and it is the bar the same node fails at
        float32, where 41% of the same sweep is off by up to 1597438 steps inside
        the saturation band.

        The samples are counted as the width's own ulps rather than in float
        distance, because at a bfloat16 or float16 magnitude a relative float
        bound says nothing about a neighbouring representable value -- and the
        assertion is a *count* rather than an exact value, so the case does not
        demand that another machine's SDK round the same six samples the same
        way. What it demands is what a coarser node could not deliver: a handful
        of samples rather than a share of the curve.
        """
        host = torch.linspace(-12.0, 12.0, 400001).to(dtype)
        out = torch.tanh(_to_device(host)).cpu()
        expected = _reference(host)
        assert out.dtype == dtype
        # The width's integer view, so a difference is counted in that width's
        # own representable steps rather than in floats.
        got = out.view(torch.int16).to(torch.int32)
        want = expected.view(torch.int16).to(torch.int32)
        differing = (got != want) & ~(torch.isnan(out) & torch.isnan(expected))
        # A handful out of 400001, and never on a meaningful share of the curve:
        # the same measurement at float32 is 41% of the sweep, which this bound
        # would reject by four orders of magnitude.
        assert int(differing.sum()) <= 16, (
            f"{dtype}: {int(differing.sum())} of 400001 samples differ"
        )
        assert int(differing.sum()) < out.numel() // 1000, dtype
        # And the distance, measured in the width's own steps: a neighbouring
        # representable answer rather than a coarser value. The bfloat16 sample
        # at the top of this sweep is 3 steps -- it sits where the reference's
        # answer is itself a subnormal, the band the case below is about.
        assert int((got - want).abs().max()) <= 4, (
            f"{dtype}: max {int((got - want).abs().max())} steps"
        )

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_result_is_odd(self, dtype):
        """``g(x) + g(-x)`` is zero -- the property float32 was declined for.

        This is the sharpest stable invariant in the file. Measured, the
        reference is *exactly* odd at every width, and so is the node at
        float16 and bfloat16; at float32 the same measurement on the node gives
        1.788e-07, which is why that width keeps the round-trip. A body that
        evaluated ``1 - 2 / (exp(2x) + 1)`` would fail this too: measured, that
        composition's own asymmetry is what a reader would be trading the
        reference for.
        """
        host = torch.linspace(0.0, 12.0, 20001).to(dtype)
        out = torch.tanh(_to_device(host)).cpu()
        neg = torch.tanh(_to_device(-host)).cpu()
        expected = _reference(host)
        # The reference's own answer, so the bar is not this body's.
        assert (expected + _reference(-host)).abs().max().item() == 0.0
        assert (out + neg).abs().max().item() == 0.0, (out + neg).abs().max().item()

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_nan_propagates_at_every_served_width(self, dtype):
        """The guard, and the one deviation that would be a wrong answer.

        Measured, the bare ``tanhWithTensor:`` node answers ``1.0`` for a NaN
        operand at every width -- it saturates rather than propagates -- where
        the reference answers ``nan``. This is the case that fails if the
        ``isNaNWithTensor:`` select is ever dropped from the body, and it is
        also what makes the body a tanh rather than a saturating approximation
        at the widths the node is otherwise exact.
        """
        host = torch.tensor([float("nan"), 0.0, float("nan"), 0.0]).to(dtype)
        out = torch.tanh(_to_device(host)).cpu()
        expected = _reference(host)
        assert torch.isnan(out[0]).item(), out
        assert out[1].item() == 0.0
        assert torch.isnan(out[2]).item(), out
        assert out[3].item() == 0.0
        # The reference's own answer on the same operand, so the bar is the
        # reference's and not this body's: nan in, nan out; a zero in, the same
        # zero out. The comparison is per element rather than ``torch.equal``
        # because ``equal`` is False for two nan tensors however well they agree.
        assert expected[1].item() == 0.0 and expected[3].item() == 0.0
        assert expected[0] != expected[0] and expected[2] != expected[2]
        for i in (0, 1, 2, 3):
            assert _same(out[i].item(), expected[i].item()), (i, out, expected)

    @pytest.mark.anyplatform
    def test_the_saturation_edges_are_exact(self):
        """``tanh(+-inf)`` is exactly ``+-1`` -- a bounded op's edge.

        The edge a composed body gets wrong: ``(exp(2x) - 1) / (exp(2x) + 1)``
        at ``x = inf`` is ``inf / inf``, a nan, where the reference answers a
        finite ``1.0``. Measured, the node reproduces it exactly at both served
        widths, and the bound also holds at ordinary large operands.
        """
        for dtype in SERVED_DTYPES:
            host = torch.tensor(
                [float("inf"), float("-inf"), 20.0, -20.0, 1e4, -1e4],
                dtype=dtype,
            )
            out = torch.tanh(_to_device(host)).cpu()
            assert out[0].item() == 1.0, dtype
            assert out[1].item() == -1.0, dtype
            assert out[2].item() == 1.0 and out[3].item() == -1.0, dtype
            assert out[4].item() == 1.0 and out[5].item() == -1.0, dtype
            # The bound itself, which no answer inside the graph's range may
            # leave: a saturating approximation stays inside it and this is what
            # makes the float32 decline a question of fidelity rather than of
            # range.
            assert (out.abs() <= 1.0).all()

    @pytest.mark.anyplatform
    def test_the_domain_edges_are_the_references(self):
        """Every edge through one node, valued against the reference.

        Measured, all of these match the reference at both served widths --
        including ``-7.0``, where the answer is within 2e-06 of the bound, so a
        body that saturated a step early would still be caught where a test that
        only looked at ``+-inf`` would not.
        """
        for dtype in SERVED_DTYPES:
            host = torch.tensor(DOMAIN_EDGES, dtype=dtype)
            out = torch.tanh(_to_device(host)).cpu()
            expected = _reference(host)
            assert out[3] != out[3], "nan must propagate"
            assert out[1].item() == 1.0 and out[2].item() == -1.0
            for i, x in enumerate(DOMAIN_EDGES):
                got, want = out[i].item(), expected[i].item()
                assert _same(got, want), (
                    f"{dtype} tanh({x!r}) = {got!r}, expected {want!r}"
                )

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_signed_zero_is_a_zero_at_the_served_widths(self, dtype):
        """The one edge where the graph may differ from the reference, held to
        the part of it that is stable.

        The sign of ``tanh(-0.0)`` is a property of the *host* and not of this
        body, and it was measured both ways: on an M5 with a current SDK the
        served widths answer ``+0.0`` where the reference answers ``-0.0``,
        while the ``macos-14`` CI runner answers the reference's ``-0.0``. So the
        case asserts the reading that is the same on both -- the answer is a
        *zero*, and the sign is preserved everywhere the operand is not one --
        and the deviation itself is recorded in ``MpsSupportsTanhDtype`` and in
        this module's docstring rather than pinned here, because a case that
        pinned it would fail on one of the two machines.

        The float32 round-trip keeps the sign on both, which is the contrast
        that makes this a property of the served widths rather than of the op.
        """
        host = torch.tensor([-0.0, 0.0, -1e-3, -1.0, -7.0], dtype=dtype)
        out = torch.tanh(_to_device(host)).cpu()
        expected = _reference(host)
        assert expected[0].signbit().item(), "the reference keeps -0.0"
        # The value is the reference's -- a zero either way -- and the bits may
        # differ only in that sign, which is what the magnitude check says.
        assert out[0].item() == 0.0 and expected[0].item() == 0.0
        assert abs(out[0].item()) == abs(expected[0].item())
        # The sign is preserved everywhere the operand is not a zero, so what
        # would be a regression is the ambiguity widening past the zero.
        assert out[2].signbit().item() and out[3].signbit().item(), out
        assert out[4].signbit().item(), out
        assert torch.equal(out[1:], expected[1:]), out

    @pytest.mark.anyplatform
    def test_the_ordinary_middle_agrees(self):
        """The values a model actually evaluates, compared to the reference.

        A body that answered the saturating node's value at float32 would still
        pass the edges above; what separates the two widths is the middle of the
        curve, which is where this case looks -- and it holds the two served
        widths to the reference's own answer rather than to a bound.
        """
        host = torch.tensor([-3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0])
        for dtype in SERVED_DTYPES:
            narrow = host.to(dtype)
            out = torch.tanh(_to_device(narrow)).cpu()
            expected = _reference(narrow)
            assert torch.equal(out, expected), dtype
            # The sign, which the curve's shape carries and a scale would not.
            assert out[0].item() < out[1].item() < 0.0
            assert out[4].item() == 0.0
            assert 0.0 < out[5].item() < out[8].item() < 1.0

    @pytest.mark.anyplatform
    def test_the_bfloat16_subnormal_band_stays_inside_the_band(self):
        """The band a sibling lost this width to, read at the *operand* here.

        ``sigmoid`` recorded that MPSGraph flushes a subnormal result at
        bfloat16, and the same flush is visible here on the operand side -- but
        like the signed zero above, whether it happens is a property of the host
        and it was measured both ways: on an M5 with a current SDK all 127
        subnormal bfloat16 operands (``bits 0x0001`` through ``0x007f``) come
        back ``0``, while the ``macos-14`` CI runner reproduces the reference's
        subnormal answers exactly.

        So the case asserts what holds on both: a subnormal operand answers the
        reference's answer *or* a zero, never a third value, and the band's far
        edge -- the smallest normal and up -- is the reference bit for bit on
        both machines. That second half is the half that would break if the
        flush widened, which is the reading worth having a case for.
        """
        subnormal_bits = torch.arange(1, 0x0080, dtype=torch.int32) << 16
        host = subnormal_bits.view(torch.float32).to(torch.bfloat16)
        tiny = torch.finfo(torch.bfloat16).tiny
        assert (host > 0).all() and (host < tiny).all(), "the band is covered"
        out = torch.tanh(_to_device(host)).cpu().float()
        expected = _reference(host).float()
        assert (expected > 0).all(), "the reference answers a subnormal here"
        for x, got, want in zip(host.tolist(), out.tolist(), expected.tolist()):
            assert got in (want, 0.0), (
                f"tanh({x!r}) = {got!r}, expected the reference's {want!r} or a "
                "flushed zero"
            )
        # The band's far edge, which is the half that would break if the flush
        # widened: the smallest normal and up are the reference exactly, on any
        # host.
        normals = torch.tensor([tiny, tiny * 2.0, 1e-20, 1e-3, 1.0])
        out_n = torch.tanh(_to_device(normals)).cpu()
        assert torch.equal(out_n, _reference(normals)), out_n
        assert (out_n > 0).all()

    @pytest.mark.anyplatform
    def test_float16_does_not_flush_its_subnormal_operands(self):
        """The contrast that keeps the flush above from being a blanket rule.

        Measured, all 1023 positive float16 subnormals reproduce the reference
        bit for bit, so the flush is a bfloat16 property rather than a platform
        one. A body that generalised the bfloat16 band to every width would fail
        here with a zero where the reference has a small nonzero number.
        """
        subnormals = torch.arange(1, 0x0400, dtype=torch.int32)
        host = subnormals.to(torch.int16).view(torch.float16)
        assert (host > 0).all() and (host < torch.finfo(torch.float16).tiny).all()
        out = torch.tanh(_to_device(host)).cpu()
        expected = _reference(host)
        assert (out > 0).all(), "a positive float16 subnormal must not flush"
        assert torch.equal(out, expected)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows of float16
            (3, 7),
            (5, 4),  # rows already a multiple of 16 bytes
            (2, 3, 5, 7),
            (4096,),
            (1, 1),
        ],
    )
    @pytest.mark.anyplatform
    def test_shape_independence(self, shape):
        host = torch.randn(*shape).to(torch.bfloat16)
        _assert_matches_cpu(torch.tanh(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        host = torch.empty(0)
        out = torch.tanh(_to_device(host))
        assert out.shape == (0,)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        host = torch.tensor(0.5, dtype=torch.bfloat16)
        _assert_matches_cpu(torch.tanh(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.randn(8, 8).t().to(torch.bfloat16)
        assert not host.is_contiguous()
        _assert_matches_cpu(torch.tanh(_to_device(host)), _reference(host))


class TestTanhDtypeGate:
    """The declined widths are declined, and the round-trip answers them exactly."""

    @pytest.mark.parametrize("dtype", DECLINED_FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_declined_widths_round_trip_at_their_own_dtype(self, dtype):
        host = torch.randn(4, 32).to(dtype)
        out = torch.tanh(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        # A graph at the wrong width would come back float32, so the dtype is
        # the assertion that the round-trip is what answered.
        assert out.cpu().dtype == dtype

    @pytest.mark.anyplatform
    def test_float32_matches_the_reference_bit_for_bit(self):
        """The decline buys the reference's answer, not an approximation of it.

        This is the case that would fail if the gate were widened to float32:
        the *node* differs from ``at::tanh`` on 41% of these samples, so a
        bit-exact comparison is what the round-trip is for and what a graph at
        this width could not pass.
        """
        host = torch.linspace(-10.0, 10.0, 40001)
        out = torch.tanh(_to_device(host))
        assert out.cpu().dtype == torch.float32
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.anyplatform
    def test_float32_keeps_the_edges_the_sign_and_the_guard(self):
        """The declined width gets the reference's answer at every edge.

        Including the signed zero the two served widths answer ``+0.0`` for: the
        round-trip *is* the reference, so the one edge the graph loses is the one
        the decline buys back. The comparison is on the bits rather than a plain
        inequality, because ``-0.0 == 0.0`` hides exactly the deviation this case
        is about.
        """
        host = torch.tensor(
            [-0.0, 0.0, float("inf"), float("-inf"), float("nan"), 1.0, -7.0]
        )
        out = torch.tanh(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[2].item() == 1.0 and out[3].item() == -1.0
        assert out[0].signbit().item(), out
        assert torch.isnan(out[4]).item()
        assert torch.equal(out.view(torch.int32), expected.view(torch.int32)), (
            out,
            expected,
        )

    @pytest.mark.anyplatform
    def test_float32_odd_symmetry_is_the_references(self):
        """The property the node broke at this width, restored by the decline.

        Measured, the node's ``max |g(x) + g(-x)|`` at float32 is 1.788e-07
        where the reference is exactly 0.0. The round-trip is what holds this
        case to 0.0 -- the same invariant the served widths are held to above,
        which is why the gate is stated as fidelity rather than as a tolerance.
        """
        host = torch.linspace(0.0, 3.0, 20001)
        out = torch.tanh(_to_device(host)).cpu()
        neg = torch.tanh(_to_device(-host)).cpu()
        assert (out + neg).abs().max().item() == 0.0

    @pytest.mark.anyplatform
    def test_float64_matches_the_reference_bit_for_bit(self):
        """The round-trip's answer is the reference, not a 1e-7 approximation."""
        host = torch.linspace(-20.0, 20.0, 513, dtype=torch.float64)
        out = torch.tanh(_to_device(host))
        assert out.cpu().dtype == torch.float64
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.anyplatform
    def test_float64_keeps_the_edges(self):
        """The edges belong to the reference, so a declined width gets them too.

        The signed zero is the one the served widths answer ``+0.0`` for, and
        this is the case that says the difference is the graph's rather than the
        op's.
        """
        host = torch.tensor(
            [-0.0, 0.0, float("inf"), float("-inf"), float("nan"), 1.0, -7.0]
        )
        out = torch.tanh(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[0].item() == 0.0 and out[0].signbit().item()
        assert out[2].item() == 1.0 and out[3].item() == -1.0
        assert torch.isnan(out[4]).item()
        # The sign of the zero is the assertion the value cannot make, so it is
        # read from the item rather than from an equality that would hide it.
        for i in (0, 1, 5, 6):
            assert out[i].item() == expected[i].item(), (i, out, expected)

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8]
    )
    @pytest.mark.anyplatform
    def test_integral_operand_promotes_exactly_like_cpu(self, dtype):
        """An integral operand is promoted to float32, and the body declines it.

        ``at::tanh`` has no integral kernel and does not raise: measured, it
        promotes to float32, and a *large* integral operand saturates to ``+-1``
        rather than leaving the range. The body declines the integral width, so
        the round-trip is what performs the promotion -- and the assertion is the
        promoted dtype and the values, because a graph that had served it would
        have had to allocate from the operand's own options.
        """
        host = torch.arange(-3, 4).to(dtype)
        out = torch.tanh(_to_device(host))
        expected = _reference(host)
        assert expected.dtype == torch.float32
        assert out.cpu().dtype == torch.float32
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-6, atol=0)
        assert out.cpu()[0].item() < 0.0 < out.cpu()[-1].item()
        assert out.cpu()[3].item() == 0.0

    @pytest.mark.anyplatform
    def test_bool_operand_promotes_exactly_like_cpu(self):
        """The promoted path keeps the bounds an integral operand reaches."""
        host = torch.tensor([True, False])
        out = torch.tanh(_to_device(host)).cpu()
        expected = _reference(host)
        assert expected.dtype == torch.float32
        assert out[1].item() == 0.0 == expected[1].item()
        assert 0.0 < out[0].item() < 1.0
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=0)


class TestTanhPlanReuse:
    """A plan is per (dtype, shape), and interleaving them shares nothing."""

    @pytest.mark.anyplatform
    def test_interleaved_signatures_never_share_a_plan(self):
        cases = [
            torch.randn(4, 64).to(torch.bfloat16),
            torch.randn(64, 4).to(torch.bfloat16),
            torch.randn(4096).to(torch.float16),
            torch.randn(4, 64).to(torch.float16),
        ]
        held = []
        for host in cases:
            out = torch.tanh(_to_device(host))
            held.append((out.cpu(), _reference(host), host.dtype))
        for got, wanted, dtype in held:
            assert got.dtype == dtype
            # One ulp at worst, the same bar the curve case measured: what this
            # case is about is the *plan*, so the assertion is that no case
            # picked up another's answer.
            assert (
                int(((got.view(torch.int16) - wanted.view(torch.int16)).abs()).max())
                <= 1
            ), dtype
            assert int((got != wanted).sum()) <= 8, dtype

    @pytest.mark.anyplatform
    def test_repeated_calls_reuse_one_plan_and_stay_correct(self):
        host = torch.randn(8, 16).to(torch.bfloat16)
        wanted = _reference(host)
        for _ in range(8):
            _assert_matches_cpu(torch.tanh(_to_device(host)), wanted)


class TestTanhDispatch:
    """``tanh`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_tanh_to_mps(self):
        assert routed_backend_or_none("tanh") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_tanh_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``tanh`` reached
        ``cpu_fallback``, because for the served widths both compute the same
        answer. The log line is emitted by the C++ dispatcher only on the ``mps``
        slot, so it is what separates the two.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, dtype=torch.bfloat16, device='flagos:0'); "
                "torch.tanh(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] tanh -> apple" in result.stderr

    @pytest.mark.main_ops
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_a_declined_width_stays_served(self, dtype):
        """A declined width is the body's round-trip, not a ``none`` route.

        float32 is the increment's decline, and it is the one worth naming: the
        node is not the reference's function at that width, so the body hands it
        to the reference -- inside the ``mps`` slot rather than by narrowing the
        conf. This holds that the decline is the body's, and that the conf still
        routes the op for every width.

        The operand is materialised on the device from a host tensor rather than
        through an arithmetic op, so the only dispatch the census can see is
        ``tanh``'s own: a served or round-tripped ``tanh`` logs once either way,
        while a ``none`` route would log it against a different backend.
        """
        source = (
            "import torch, torch_fl; "
            f"a = torch.randn(4, 4, dtype=torch.{dtype}).to('flagos:0'); "
            "torch.tanh(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
