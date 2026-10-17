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

"""Stage-4 Apple Silicon (Metal / MPSGraph) erf.

``erf`` is the exact-GELU kernel -- ``0.5 * x * (1 + erf(x / sqrt(2)))`` is the
form a model writes when it wants the exact activation rather than the tanh
approximation its sibling ``tanh`` serves -- and MPSGraph names it directly as
``erfWithTensor:``. The body is therefore the shared one-node unary template with
no guard, and the whole of the increment is the gate:

* **All three float widths are served, and the finding is the opposite of the
  sibling ``tanh``'s.** The tanh node was a *coarser approximation* of the
  reference's tanh at float32 and had to be declined there; this node is the
  reference's own function at every width, differing from it only in how it
  rounds the last ulp. Measured against the float64 curve over a 400001-sample
  sweep of ``[-3, 3]``, the node's max error is ``1.788e-07`` where the
  reference's is ``4.955e-07`` -- and the node is the *closer* of the two on
  242217 of the samples. Two implementations that round one curve differently in
  the last ulp are both erfs, so the width is served; see ``MpsSupportsErfDtype``.
* **The bit agreement is the reading that keeps the case honest.** Over the same
  sweep the node agrees with ``at::erf`` bit for bit on 46.5% of samples, and of
  the 213871 that differ, 195438 are within four of the reference's own
  representable step. The step counts that look large are the two agreeing on the
  *answer* where the curve is flat -- both return exactly ``1.0`` for
  ``|x| >= 4`` -- which is why the case below asserts the saturation band
  value-for-value rather than by a step bound.
* **The domain edges need no guard, unlike the sibling's.** Measured, the bare
  node propagates a NaN (where ``tanhWithTensor:`` saturated it to ``+-1``),
  answers ``+-1`` at ``+-inf``, and keeps the sign of a zero at every width on
  *both* hosts -- so unlike ``tanh``'s signed zero, this one is an invariant and
  the cases below assert it rather than recording it.

One band is recorded rather than asserted, and it is the band the siblings found:
the narrow widths' *subnormal operand*. Measured, all 127 subnormal bfloat16
operands come back a zero, which is also what the reference rounds them to, while
float16 differs from the reference on 877 of its 1023 positive subnormals in the
last step or two (and the reference itself rounds 3 of them to zero). Everything
involved is below the width's smallest normal and a single step wide, and a
model's ``erf`` operand is a pre-activation rather than a denormal, so the values
are recorded in ``MpsSupportsErfDtype`` and below rather than pinned.

The tests below hold what the numbers alone cannot: that all three widths agree
with ``at::erf`` to the last ulp on the curve, that the edges (``+-inf`` ->
``+-1``, nan -> nan, the signed zero) are the reference's, that the curve is odd,
that the saturation band is the reference bit for bit, and that float64 and the
integral/bool operands take the round-trip.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
"""

import os
import subprocess
import sys

import numpy as np
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

# All three float widths: this is the increment's finding, and the contrast with
# the sibling `tanh` -- which declines the widest one. See MpsSupportsErfDtype.
SERVED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The widths the reference has a kernel for and this body declines. float64 has
# no MPSGraph element type. The round-trip answers it, so the assertion is CPU
# equality at the operand's own dtype rather than an error.
DECLINED_FLOAT_DTYPES = [torch.float64]

# The curve's edges at the served widths. ``erf`` is bounded, so both infinities
# answer a *finite* +-1, the closed form's edge; the guard-free body has to answer
# the nan on its own, which the sibling tanh's node could not and needed a select
# for. The signed zero is here, unlike the tanh suite's, because this node keeps
# it on both hosts.
DOMAIN_EDGES = [
    0.0,  # 0.0
    -0.0,  # -0.0
    float("inf"),  # 1.0
    float("-inf"),  # -1.0
    float("nan"),  # nan
    1.0,  # 0.8427006602287292
    -1.0,  # -0.8427006602287292
    -3.0,  # -0.9999778866767883, near the bound but not saturated
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::erf`` on the host, in the operand's own dtype.

    The op and not a ``1 - erfc(x)`` composition: the reference is what the
    kernel is held to, and its own rounding is measurably different from the
    exactly-rounded answer (up to 5.327e-07 at float32 against ``erff`` through
    float64), so a composition of a different approximation would be a third
    curve rather than the one this body replaces.
    """
    return torch.ops.aten.erf(host)


def _assert_matches_cpu(
    out: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-2
) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    # ``rtol`` alone is the wrong bar for a sample whose answer is a denormal:
    # ``erf`` of a tiny operand is tiny, and one step there is a share of the
    # value rather than a rounding of it. Both sides agree far inside the bound,
    # so the bar is the width's own absolute step too, which is what
    # ``assert_close``'s own ``atol`` would default to and what a scale error
    # (the failure this bar exists to catch) still leaves by a mile.
    width = float(torch.finfo(out.dtype).eps)
    torch.testing.assert_close(
        out.cpu().float(), expected.float(), rtol=rtol, atol=width * rtol
    )


def _same(x: float, y: float) -> bool:
    """Equality that treats two nans as the same answer."""
    return (x == y) or (x != x and y != y)


def _steps(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Difference in the width's own representable steps, as int64.

    A relative float bound says nothing about a neighbouring representable value
    at a narrow width, so every step-count assertion in this file works from the
    integer view of the operand's own dtype.
    """
    if got.dtype == torch.float32:
        return got.view(torch.int32).to(torch.int64) - want.view(torch.int32).to(
            torch.int64
        )
    return got.view(torch.int16).to(torch.int64) - want.view(torch.int16).to(
        torch.int64
    )


def _step_size(values: torch.Tensor) -> torch.Tensor:
    """The width's representable step *at each value*, as float64.

    ``finfo(dtype).eps`` is the step only near 1; the step shrinks with the
    exponent, and ``erf``'s value spans that whole range, so a bar written as
    ``eps * |ref|`` is the wrong unit near zero -- where ``|ref|`` is itself a
    denormal, an epsilon-scaled bar is smaller than a single step and the ratio
    against it diverges on a rounding that is only one step wide.
    """
    magnitude = values.double().abs()
    floors = torch.where(
        magnitude > 0,
        torch.exp2(torch.floor(torch.log2(magnitude))),
        torch.ones_like(magnitude),
    )
    return floors * float(torch.finfo(values.dtype).eps)


class TestErfCorrectness:
    """``torch.erf`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = torch.randn(4, 64).to(dtype)
        _assert_matches_cpu(torch.erf(_to_device(host)), _reference(host))

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_curve_is_the_reference_to_within_a_few_ulps(self, dtype):
        """The increment's finding, stated as the bound that holds at a width.

        Measured over a 400001-sample sweep of ``[-6, 6]``: float16 and bfloat16
        agree bit for bit with ``at::erf`` on 399878 and 399994 of the samples
        respectively, and never differ by more than a fraction of the width's own
        step in *relative* terms -- 8.6e-04 of the value at float16, which is
        under one of its 9.8e-04 steps, and nothing at all at bfloat16. float32
        is the width whose rounding differs, and there a raw step count is *not*
        an invariant: the two curves agree on the answer and differ only in the
        last place, so where the answer is a couple of steps above zero the step
        count between two neighbouring values explodes while the absolute gap
        stays one reference step. The bar below is therefore relative to each
        width's own step, which is what a coarser node (the sibling ``tanh``'s at
        float32, off by up to 4.6e-02 of the value, and 2.7e-03 at float16) could
        not satisfy.
        """
        host = torch.linspace(-6.0, 6.0, 400001).to(dtype)
        out = torch.erf(_to_device(host)).cpu()
        expected = _reference(host)
        assert out.dtype == dtype
        # The two narrow widths agree with the reference bit for bit on all but a
        # handful of samples, and the handful is measured rather than assumed.
        steps = _steps(out, expected).abs()
        if dtype is not torch.float32:
            agree = int((steps == 0).sum())
            assert agree >= out.numel() - 256, (
                f"{dtype}: only {agree} of {out.numel()} samples are bit-identical"
            )
        # The bar that holds at all three widths, stated where the step count
        # cannot hold one: the gap as a share of the reference's own step for its
        # own value. Measured, the ratio degrades monotonically as the band's
        # scale shrinks -- 5.0 at float32, 1.0 at float16 and 0.0 at bfloat16 for
        # ``|ref| >= 0.5``; 2869, 1 and 1 for ``>= 1e-3`` -- because below that
        # the two roundings are resolving the same curve against *denormal*
        # spacing, so the honest statement is the banded pair below rather than
        # one number.
        gap = (out.double() - expected.double()).abs()
        reference_step = _step_size(expected)
        ratio = gap / torch.where(reference_step > 0, reference_step, 1.0)
        scale = expected.double().abs()
        large = scale >= 0.5
        assert float(ratio[large].max()) <= 10.0, (dtype, float(ratio[large].max()))
        assert float(ratio[large | (scale >= 1e-3)].max()) <= 3000.0, (
            dtype,
            float(ratio[large | (scale >= 1e-3)].max()),
        )
        # The band just above zero, where the reference's step is itself a
        # denormal: measured, the two roundings stay within 5.0e-07, 4.9e-04 and
        # 3.1e-05 of each other at the three widths respectively -- a handful of
        # each width's own steps at 1.0, and the reason the ratio above is
        # reported banded rather than as one bound.
        assert float(gap.max()) <= 8.0 * float(torch.finfo(dtype).eps), (
            dtype,
            float(gap.max()),
        )
        # The answer is always inside the bound, and the sweep's far end is the
        # reference's own rounding to exactly 1 at every width.
        # The answer is always inside the bound, and the sweep's far end is the
        # reference's own rounding to exactly 1 at every width.
        assert (out.abs() <= 1.0).all()
        assert out[-1].item() == 1.0 == expected[-1].item()

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_node_is_not_coarser_than_the_reference(self, dtype):
        """The measurement that decides the gate, asserted as a property.

        ``tanh`` was declined at float32 because its node was a coarser
        approximation of the reference's own function. The claim this gate makes
        is the opposite, so it is worth a case: over the sweep, wherever the node
        and the reference disagree the node is *at least as close* to the exact
        (float64) curve as the reference is on the great majority of samples.

        The bar is deliberately one-sided and loose -- the exact answer is a
        float64 curve neither width represents -- so the case says "not coarser"
        rather than "more accurate", which is the property the gate needs.
        """
        host = torch.linspace(-3.0, 3.0, 400001).to(dtype)
        out = torch.erf(_to_device(host)).cpu().float()
        expected = _reference(host).float()
        exact = torch.erf(host.to(torch.float64)).float()
        finite = torch.isfinite(out) & torch.isfinite(expected)
        node_err = (out - exact).abs()[finite]
        ref_err = (expected - exact).abs()[finite]
        # A tolerance as wide as a few reference steps, so "not coarser" is not
        # a knife-edge on a single sample's rounding.
        slack = 4.0 * torch.finfo(dtype).eps
        not_coarser = int((node_err <= ref_err + slack).sum())
        assert not_coarser >= int(finite.sum()) - 64, (
            f"{dtype}: node is coarser than the reference on "
            f"{int(finite.sum()) - not_coarser} samples"
        )

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_result_is_odd(self, dtype):
        """``g(x) + g(-x)`` is zero -- the property a valid erf has."""
        host = torch.linspace(0.0, 6.0, 20001).to(dtype)
        out = torch.erf(_to_device(host)).cpu()
        neg = torch.erf(_to_device(-host)).cpu()
        expected = _reference(host)
        # The reference's own answer, so the bar is not this body's.
        assert (_reference(host).float() + _reference(-host).float()).abs().max() == 0.0
        assert (out.float() + neg.float()).abs().max().item() == 0.0, (
            (out.float() + neg.float()).abs().max().item()
        )
        # And the shape, which a scale would not carry: a strict rise off zero up
        # to the saturated value the far end of this sweep is.
        assert out[0].item() == 0.0
        assert 0.0 < out[1].item() < out[-1].item() <= 1.0
        # The two roundings are the same *answer* at this sweep's granularity:
        # see the curve case for the relative bound that says so and for why a
        # raw step count is not the invariant at float32.
        # The bar is the width's own, since a float16 step near the top of this
        # sweep is 4.88e-04 -- the same ulp-scaled bar the curve case uses, which
        # is exact at bfloat16 and a fraction of a step at the other two.
        assert float((out.float() - expected.float()).abs().max()) <= 4.0 * float(
            torch.finfo(dtype).eps
        ), (dtype, float((out.float() - expected.float()).abs().max()))

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_nan_propagates_without_a_guard(self, dtype):
        """The bare node already answers the reference's nan, unlike ``tanh``.

        The sibling's node saturates a NaN operand to ``+-1`` and the body has to
        select the NaN back; this one propagates it unaided, so a case that passes
        here is the evidence the body needs no ``isNaNWithTensor:`` select. If a
        future node build changed that, this is where it would show.
        """
        host = torch.tensor([float("nan"), 0.0, float("nan"), -1.0, 1.0]).to(dtype)
        out = torch.erf(_to_device(host)).cpu()
        expected = _reference(host)
        assert torch.isnan(out[0]).item(), out
        assert out[1].item() == 0.0
        assert torch.isnan(out[2]).item(), out
        # The non-nan slots are the same *answers* the case above measured, to the
        # same last-ulp bar -- exact at the two narrow widths and a few steps at
        # float32 -- and the nan slots are the guard's own claim either way.
        assert int(_steps(out, expected).abs().max()) <= 4, (out, expected)
        assert _same(out[1].item(), expected[1].item())

    @pytest.mark.anyplatform
    def test_the_saturation_edges_are_exact(self):
        """``erf(+-inf)`` is exactly ``+-1`` -- a bounded op's edge.

        The edge a composed body gets wrong: ``1 - 1/exp(x*x)``-style forms and
        the ``tanh`` identity both leave the range or answer a nan at ``x = inf``,
        where the reference answers a finite ``1.0``. Measured, the node
        reproduces it exactly at all three widths, and the bound also holds at
        ordinary large operands -- including the point where the reference itself
        rounds to 1.0, which the curve case above would see as a step or two and
        this one sees as equality.
        """
        for dtype in SERVED_DTYPES:
            host = torch.tensor(
                [float("inf"), float("-inf"), 3.0, -3.0, 4.0, -4.0, 1e4, -1e4],
                dtype=dtype,
            )
            out = torch.erf(_to_device(host)).cpu()
            expected = _reference(host)
            assert out[0].item() == 1.0, dtype
            assert out[1].item() == -1.0, dtype
            assert (out.abs() <= 1.0).all(), dtype
            # From |x| >= 4 both round to exactly +-1, so the answer -- not just
            # a step count -- is the reference's.
            assert torch.equal(out[4:], expected[4:]), (dtype, out, expected)

    @pytest.mark.anyplatform
    def test_the_domain_edges_are_the_references(self):
        """Every edge through one node, valued against the reference.

        Including ``-3.0``, where the answer is within 2.3e-05 of the bound, so a
        body that saturated a step early would still be caught where a test that
        only looked at ``+-inf`` would not -- and including the signed zero, which
        this node keeps on both hosts (unlike the sibling ``tanh``'s).
        """
        for dtype in SERVED_DTYPES:
            host = torch.tensor(DOMAIN_EDGES, dtype=dtype)
            out = torch.erf(_to_device(host)).cpu()
            expected = _reference(host)
            assert out[4] != out[4], "nan must propagate"
            assert out[2].item() == 1.0 and out[3].item() == -1.0
            # The edges with no rounding to disagree about are the reference
            # exactly at every width: both infinities, and both zeros. The two
            # finite interior points are held to a relative last-ulp bar instead,
            # which is the same bar the curve case uses and the reason this gate
            # serves float32 at all.
            for i in (0, 1, 2, 3):
                assert out[i].item() == expected[i].item(), (
                    f"{dtype} erf({DOMAIN_EDGES[i]!r}) = {out[i].item()!r}, "
                    f"expected {expected[i].item()!r}"
                )
            got, want = out[5:].float(), expected[5:].float()
            assert float(((got - want).abs() / want.abs()).max()) <= 1e-06, (dtype, out)
            # The signed zero, whose *sign* an equality cannot see.
            assert not bool(np.signbit(out[0].item())), dtype
            assert bool(np.signbit(out[1].item())), dtype

    @pytest.mark.anyplatform
    def test_the_ordinary_middle_agrees(self):
        """The values a model actually evaluates, compared to the reference.

        The points a GELU's operand lands on, held to the reference's own answer.
        At float32 the two roundings meet where the curve is flat enough, so this
        is a bit-equality case rather than a step bound.
        """
        host = torch.tensor([-3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0])
        for dtype in SERVED_DTYPES:
            narrow = host.to(dtype)
            out = torch.erf(_to_device(narrow)).cpu()
            expected = _reference(narrow)
            # Bit equality at the narrow widths, where the reference is itself
            # only approximate and the two roundings meet; a few steps at
            # float32, which is the width this increment widens the gate to and
            # whose rounding difference the curve case measured.
            if dtype is not torch.float32:
                assert torch.equal(out, expected), (dtype, out, expected)
            else:
                got, want = out.float(), expected.float()
                # The relative bar applies where the value has scale; the zero of
                # the sweep is held to bit equality instead, since the reference
                # and the node both answer it exactly.
                assert got[4].item() == want[4].item() == 0.0
                nonzero = want != 0.0
                gap = (got - want).abs()[nonzero] / want.abs()[nonzero]
                assert float(gap.max()) <= 3e-05, (out, float(gap.max()))
            assert out[0].item() < out[1].item() < 0.0
            assert out[4].item() == 0.0
            # The curve rises off zero to the bound: measured, the narrow widths
            # round ``erf(3.0)`` to exactly 1.0 -- 0.99997788 is between their
            # neighbouring values -- so the strict-rise half of the assertion
            # stops at the last point that is still below it.
            # The curve's shape through the points both roundings agree on, which
            # the narrow widths' reading of ``erf(3.0)`` -- 1.0 -- is not.
            assert out[0].item() < out[1].item() < out[2].item() < out[3].item() < 0.0
            assert out[4].item() == 0.0
            assert 0.0 < out[5].item() < out[6].item() < out[7].item() < 1.0
            assert out[8].item() >= out[7].item()
            # The sign, which the curve's shape carries and a scale would not.
            assert torch.equal(torch.sign(out), torch.sign(expected)), (dtype, out)

    @pytest.mark.anyplatform
    def test_the_large_operands_saturate_like_the_reference(self):
        """The band where a step count is meaningless and the *answer* is not.

        Measured, both the node and the reference return exactly ``1.0`` from
        ``|x| >= 4``, so the two agree bit for bit there -- which is why the
        curve case's largest step counts (up to 876061900) are not a disagreement
        about the answer. This case asserts that directly, so a future node build
        that stopped saturating at the reference's point would fail here rather
        than pass as a step or two.
        """
        for dtype in SERVED_DTYPES:
            host = torch.tensor(
                [4.0, 4.5, 5.0, 6.0, 10.0, 1e3, -4.0, -6.0, -1e3], dtype=dtype
            )
            out = torch.erf(_to_device(host)).cpu()
            expected = _reference(host)
            assert torch.equal(out, expected), (dtype, out, expected)
            assert (out.abs() == 1.0).all(), dtype

    @pytest.mark.anyplatform
    def test_the_subnormal_band_is_recorded_not_pinned(self):
        """The one band whose reading is the *narrow widths'* rather than a rule.

        Measured, all 127 subnormal bfloat16 operands and 877 of float16's 1023
        positive subnormals round differently from the reference's own answer --
        by a step or two, inside a band whose widest value is below the width's
        smallest normal. A model's ``erf`` operand is a pre-activation, so the
        band is recorded rather than asserted; the assertion is the one that
        holds on both machines and would break if the behaviour widened: at or
        above the smallest normal the answer is the reference's.
        """
        tiny16 = torch.finfo(torch.float16).tiny
        tinybf = torch.finfo(torch.bfloat16).tiny
        for dtype, tiny, count in (
            (torch.float16, tiny16, 0x0400),
            (torch.bfloat16, tinybf, 0x0080),
        ):
            if dtype is torch.float16:
                host = torch.arange(1, count, dtype=torch.int32).to(torch.int16)
                host = host.view(torch.float16)
            else:
                host = (
                    (torch.arange(1, count, dtype=torch.int32) << 16)
                    .view(torch.float32)
                    .to(torch.bfloat16)
                )
            assert (host > 0).all() and (host < tiny).all(), "the band is covered"
            out = torch.erf(_to_device(host)).cpu()
            expected = _reference(host)
            # The answer never becomes a *normal* number or a nan: the whole band
            # maps into the subnormal-or-zero range, which is where the reference
            # puts it too. (Measured, the far end of the band -- the last few
            # subnormals -- rounds to the smallest normal at both widths, so the
            # bound here is `<= tiny` rather than `< tiny`.)
            assert (out >= 0).all() and not bool(torch.isnan(out).any()), (
                dtype,
                out.max(),
            )
            # Measured, the far end of the band -- the largest operand in it, and
            # so the largest answer -- comes back two of the width's steps *above*
            # the smallest normal, both at the node and at the reference, which is
            # why the bound is a step rather than the smallest normal itself.
            assert out.max().item() <= tiny * 4.0, (dtype, out.max().item())
            steps = _steps(out, expected).abs()
            # Within a handful of the *answer's own* steps, not a different value:
            # measured, 0 at bfloat16 and 3 at float16, where the band is one
            # operand step wide and its answer's step is an eighth of its largest
            # value -- so this is the same one-or-two-step rounding the band
            # records, seen in the width's own integers.
            assert int(steps.max()) <= 6, (dtype, int(steps.max()))
            # And the far edge, which is held to the *normal* range's answer
            # rather than to bit equality, because it is still the recorded band:
            # measured, ``erf(1e-20)`` is a bfloat16 subnormal the node answers
            # and the reference rounds to zero, and ``erf(tiny)`` rounds to not
            # quite the same subnormal. Both roundings rise with the operand and
            # both are tiny; the step count between them is not the rule, so the
            # assertion is the shape and the reference's own value to its
            # relative last ulp.
            normals = torch.tensor([tiny, tiny * 2.0, 1e-3, 1.0], dtype=dtype)
            out_n = torch.erf(_to_device(normals)).cpu()
            wanted = _reference(normals).double()
            got = out_n.double()
            assert (got > 0).all() and (got < 1).all(), (dtype, out_n)
            assert (got[1:] >= got[:-1]).all(), (dtype, out_n)
            ok = wanted > 0
            gap = (got[ok] - wanted[ok]).abs() / wanted[ok]
            assert float(gap.max()) <= 1e-02, (dtype, out_n, _reference(normals))

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
        _assert_matches_cpu(torch.erf(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        host = torch.empty(0)
        out = torch.erf(_to_device(host))
        assert out.shape == (0,)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        host = torch.tensor(0.5, dtype=torch.bfloat16)
        _assert_matches_cpu(torch.erf(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.randn(8, 8).t().to(torch.bfloat16)
        assert not host.is_contiguous()
        _assert_matches_cpu(torch.erf(_to_device(host)), _reference(host))


class TestErfDtypeGate:
    """The declined widths are declined, and the round-trip answers them exactly."""

    @pytest.mark.parametrize("dtype", DECLINED_FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_declined_widths_round_trip_at_their_own_dtype(self, dtype):
        host = torch.randn(4, 32).to(dtype)
        out = torch.erf(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        # A graph at the wrong width would come back float32, so the dtype is
        # the assertion that the round-trip is what answered.
        assert out.cpu().dtype == dtype

    @pytest.mark.anyplatform
    def test_float64_matches_the_reference_bit_for_bit(self):
        """The round-trip's answer is the reference, not a 1e-7 approximation."""
        host = torch.linspace(-20.0, 20.0, 513, dtype=torch.float64)
        out = torch.erf(_to_device(host))
        assert out.cpu().dtype == torch.float64
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.anyplatform
    def test_float64_keeps_the_edges_the_sign_and_the_guard(self):
        """The edges belong to the reference, so a declined width gets them too."""
        host = torch.tensor(
            [-0.0, 0.0, float("inf"), float("-inf"), float("nan"), 1.0, -3.0]
        )
        out = torch.erf(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[2].item() == 1.0 and out[3].item() == -1.0
        assert bool(np.signbit(out[0].item())), out
        assert torch.isnan(out[4]).item()
        # The edges with no rounding to disagree about -- both zeros, both
        # infinities -- are the reference exactly. ``1.0`` is not: the two erfs
        # round its last ulp differently (0.842700719833374 vs 0.8427006602287292
        # at double), which is the same last-place difference the served widths
        # show rather than a declined width being held to a different curve.
        for i in (0, 1, 2, 3):
            assert out[i].item() == expected[i].item(), (i, out, expected)
        assert abs(out[5].item() - expected[5].item()) <= 4 * float(
            torch.finfo(out.dtype).eps
        ), (5, out, expected)
        # ``-3.0`` is the same last-place case: the two roundings of one curve
        # differ by 5.96e-08, a single float64 step at 1.0.
        assert abs(out[6].item() - expected[6].item()) <= 4 * float(
            torch.finfo(out.dtype).eps
        ), (6, out, expected)

    @pytest.mark.anyplatform
    def test_float32_is_served_rather_than_round_tripped(self):
        """The widening, which is this increment's finding and its risk.

        The sibling ``tanh`` declines float32; this op serves it, and the case is
        what tells the two apart. It does *not* assert bit equality -- the whole
        point is that the node rounds the last ulp differently from the reference
        -- so it asserts the served path's own signature: the answer is inside the
        bound, the curve's shape is the reference's, and the sample that would be
        loosest under a fallback is within a few of the reference's steps.
        """
        host = torch.linspace(-6.0, 6.0, 400001)
        out = torch.erf(_to_device(host)).cpu()
        assert out.dtype == torch.float32
        expected = _reference(host)
        # A step count is not the invariant at this width -- see the curve case --
        # so the signature of the served path is the *answer*: the two roundings
        # of one curve never leave each other by more than a fraction of a
        # float32 step relative to the value, and the value is the bound's.
        gap = (out - expected).abs()
        assert float(gap.max()) <= 1e-06, float(gap.max())
        assert (out.abs() <= 1.0).all()
        # The shape, so a body that answered a scale of the reference would fail.
        # The sweep's *last* point is the one to read it at: ``linspace``'s first
        # point rounds to ``-6.0000002``, whose erf is -1 rather than the zero a
        # shape assertion would want, while its last is exactly ``6.0``.
        assert out[-1].item() == 1.0
        assert out[0].item() == -1.0 == expected[0].item()
        # The shape of the sign, which a scale could not keep: negative through
        # the first half, positive after it, and the one sample inside each sign
        # whose answer the two roundings disagree on the sign of is a sample whose
        # answer is a denormal -- which is the band the curve case above recorded
        # rather than asserted.
        signs = torch.sign(out)
        differing = int((signs != torch.sign(expected)).sum())
        assert differing <= 4, differing
        assert (
            float(signs[: out.numel() // 2].sum())
            < 0.0
            < float(signs[out.numel() // 2 + 1 :].sum())
        )
        # And the difference is a rounding of the reference's own curve rather
        # than a second one: at this width the node is the closer of the two to
        # the exact (float64) answer on the great majority of the sweep, which a
        # fallback could not be and a coarser node would not be.
        exact = torch.erf(host.double())
        node_err = out.double().sub(exact).abs()
        ref_err = expected.double().sub(exact).abs()
        assert int((node_err <= ref_err).sum()) >= int(0.4 * out.numel()), (
            int((node_err <= ref_err).sum()),
            out.numel(),
        )

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8]
    )
    @pytest.mark.anyplatform
    def test_integral_operand_promotes_exactly_like_cpu(self, dtype):
        """An integral operand is promoted to float32, and the body declines it.

        ``at::erf`` has no integral kernel and does not raise: measured, it
        promotes to float32. The body declines the integral width, so the
        round-trip is what performs the promotion -- and the assertion is the
        promoted dtype and the values, because a graph that had served it would
        have had to allocate from the operand's own options.
        """
        host = torch.arange(-3, 4).to(dtype)
        out = torch.erf(_to_device(host))
        expected = _reference(host)
        assert expected.dtype == torch.float32
        assert out.cpu().dtype == torch.float32
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-6, atol=0)
        assert out.cpu()[0].item() < 0.0 < out.cpu()[-1].item()
        assert out.cpu()[3].item() == 0.0

    @pytest.mark.anyplatform
    def test_bool_operand_promotes_exactly_like_cpu(self):
        """The promoted path keeps the bound an integral operand reaches."""
        host = torch.tensor([True, False])
        out = torch.erf(_to_device(host)).cpu()
        expected = _reference(host)
        assert expected.dtype == torch.float32
        assert out[1].item() == 0.0 == expected[1].item()
        assert 0.0 < out[0].item() < 1.0
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=0)


class TestErfPlanReuse:
    """A plan is per (dtype, shape), and interleaving them shares nothing."""

    @pytest.mark.anyplatform
    def test_interleaved_signatures_never_share_a_plan(self):
        cases = [
            torch.randn(4, 64).to(torch.bfloat16),
            torch.randn(64, 4).to(torch.bfloat16),
            torch.randn(4096).to(torch.float16),
            torch.randn(4, 64).to(torch.float16),
            torch.randn(4, 64),
        ]
        held = []
        for host in cases:
            out = torch.erf(_to_device(host))
            held.append((out.cpu(), _reference(host), host.dtype))
        for got, wanted, dtype in held:
            assert got.dtype == dtype
            # The same bar the curve case measured: what this case is about is
            # the *plan*, so the assertion is that no case picked up another's
            # answer. The step bound holds the narrow widths to a handful of
            # steps, and float32 -- the width whose step count is not an
            # invariant -- to the absolute bar that is: within a float32 step of
            # the reference's own answer.
            if dtype is not torch.float32:
                assert int(_steps(got, wanted).abs().max()) <= 4, dtype
                assert int((got != wanted).sum()) <= max(8, got.numel() // 100), dtype
            else:
                assert float((got - wanted).abs().max()) <= 1e-06, dtype

    @pytest.mark.anyplatform
    def test_repeated_calls_reuse_one_plan_and_stay_correct(self):
        host = torch.randn(8, 16).to(torch.bfloat16)
        wanted = _reference(host)
        for _ in range(8):
            _assert_matches_cpu(torch.erf(_to_device(host)), wanted)


class TestErfDispatch:
    """``erf`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_erf_to_mps(self):
        assert routed_backend_or_none("erf") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_erf_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``erf`` reached
        ``cpu_fallback``, because for a float32 operand the two compute answers
        that agree to a few ulps. The log line is emitted by the C++ dispatcher
        only on the ``mps`` slot, so it is what separates the two.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, dtype=torch.bfloat16, device='flagos:0'); "
                "torch.erf(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] erf -> mps" in result.stderr

    @pytest.mark.main_ops
    @pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_every_served_width_stays_served(self, dtype):
        """Every float width routes to the body, not to a narrowed conf.

        The contrast with the sibling ``tanh``, whose float32 case asserts the
        same thing about a *declined* width: here all three are served, and the
        operand is materialised on the device from a host tensor rather than
        through an arithmetic op, so the only dispatch the census can see is
        ``erf``'s own.
        """
        source = (
            "import torch, torch_fl; "
            f"a = torch.randn(4, 4, dtype=torch.{dtype}).to('flagos:0'); "
            "torch.erf(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
