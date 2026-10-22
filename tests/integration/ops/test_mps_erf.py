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

``erf`` is the kernel in the explicit GELU form
``0.5 * x * (1 + erf(x / sqrt(2)))`` -- rather than the tanh approximation its
sibling ``tanh`` serves -- and MPSGraph names it directly as ``erfWithTensor:``.
The body is therefore the shared one-node unary template with no guard, and the
whole of the increment is the gate:

* **All three float widths are served on bounds close to the reference's
  own error.** The sibling ``tanh`` node was a *coarser approximation* of the
  reference's tanh at float32 and had to be declined there; this node follows the
  reference's own curve. Measured on an M5 against the float64 curve on a
  498299-point probe grid, the node's float32 error reaches ``1.64`` steps at 1.0
  where the reference's reaches ``4.14``, and the node is closer on ``63.3%`` of
  the points where they differ. On the same host, where the reference has an
  error above half a step, the node's error can exceed it by at most ``0.98`` of
  a step. The node's float32 gap from ``at::erf`` is at most ``4.19`` steps at
  1.0 -- a few float32 steps across the curve, visible because the probe samples
  fall between grid points. The earlier ``macos-14`` runner used an older
  MPSGraph and reported a larger absolute node error (``4.14`` steps); both
  hosts stay within ``5.0`` steps of the reference. See
  ``MpsSupportsErfDtype``.
* **GELU needs its own gate.** ``F.gelu(x)`` and
  ``F.gelu(x, approximate="tanh")`` gave the same output on this host, and
  neither named this ``erf`` kernel. The explicit form
  ``0.5 * x * (1 + erf(x / sqrt(2)))`` reaches it; its float32 result differs
  from ``F.gelu``'s default by as much as ``4.74e-04`` over ``[-8, 8]``.
  Any later ``gelu`` work is a separate gate.
* **The domain edges need no guard, unlike the sibling's.** Measured, the bare
  node propagates a NaN (where ``tanhWithTensor:`` saturated it to ``+-1``),
  answers ``+-1`` at ``+-inf``, and keeps the sign of a zero at every width on
  *both* hosts -- so unlike ``tanh``'s signed zero, this one is an invariant and
  the cases below assert it rather than recording it.

One band is host-dependent, and it is the band the siblings found: the narrow
widths' *subnormal operand*. Measured on this host, all 127 subnormal bfloat16
operands come back a zero, and float16 differs from the reference on 877 of its
1023 positive subnormals. The M5 also keeps some of those results as subnormals;
the ``macos-14`` CI runner flushed two of them to zero. The suite therefore
asserts the monotonic shape, the reference's answer where it is representable,
and no flush in the normal range. The full readings are in
``MpsSupportsErfDtype``; a model's ``erf`` operand is a pre-activation rather
than a denormal, so the host split is not pinned.

The tests below hold what the numbers alone cannot: that each width stays within
the measured gap from ``at::erf``, that the float32 node's error against the
exact curve stays bounded relative to the reference's, that the edges
(``+-inf`` -> ``+-1``, nan -> nan, the signed zero) are the reference's, that
the curve is odd, that the saturation band is the reference bit for bit, and
that float64 and the integral/bool operands take the round-trip.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
"""

import os
import subprocess
import sys

import pytest
from device_init import torch

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


# One step of each served width at 1.0: 2**-23 at float32, 2**-10 at float16 and
# 2**-8 at bfloat16. ``finfo(dtype).eps`` is the same quantity; the table is
# spelled out so a bound reads as "n steps at 1.0" without a call at each use.
ULP_AT_ONE = {
    torch.float32: 2.0**-23,
    torch.float16: 2.0**-10,
    torch.bfloat16: 2.0**-8,
}


def _probe_grid(dtype: torch.dtype) -> torch.Tensor:
    """Operands whose answers are the reference's *and* the exact curve's.

    The measurement cases below are about the reference's own rounding, so the
    dense sweep alone is the wrong operand set: ``linspace(-6, 6, 400001)``
    lands on float32 values whose ``erf`` the reference rounds exactly, and a
    bound read off it alone is smaller than the one the curve actually holds.
    The samples where the reference has an error to compare are the operands
    *between* the width's representable ones. This grid therefore takes the
    dense sweep plus three points inside a 16384-part grid over ``[0, 9]`` --
    every grid point is a float32 operand, and most of the float16 and bfloat16
    spread -- and its negation, so the grid is symmetric and the oddness and the
    sign of each answer are both exercised.
    """
    sweep = torch.linspace(-6.0, 6.0, 400001).to(dtype)
    edges = torch.linspace(0.0, 9.0, 1 << 14, dtype=torch.float64)
    low = edges[:-1].to(dtype).to(torch.float64)
    high = edges[1:].to(dtype).to(torch.float64)
    probes = [(low + (high - low) * f).to(dtype) for f in (0.21, 0.5123456789, 0.81)]
    midpoints = torch.cat(probes)
    return torch.cat([sweep, midpoints, -midpoints])


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

        The bound is measured on ``_probe_grid``'s operands rather than on the
        dense sweep alone, because the dense sweep's float32 answers are the
        ones the reference rounds *exactly*: its measured gap there is
        ``2.5e-10``, which would describe a body that nothing exercises. On the
        probe grid, measured:

        * float16 and bfloat16 agree with ``at::erf`` bit for bit on 399878 and
          399994 of the dense sweep and on 98282 and 98296 of the 98298 probes,
          and their gap never leaves half a step at 1.0 -- at bfloat16 it is
          `0.0039` of one, at the widest;
        * float32 agrees bit for bit on 186130 of the 400001 sweep samples and
          on 249310 of all 498299 probe-grid points, and its gap reaches
          ``4.19`` steps at 1.0 on the full grid. The reference has an error of
          its own at this width, and the node follows different grid points
          from it; see the case below.

        The unit is the width's step *at 1.0* and not at the reference's own
        value: near zero the reference's spacing is smaller than the two curves'
        agreement (a denormal at the smallest inputs, ``3.6e-09`` at
        ``|ref| ~ 0.03``), so a step count at the reference's value reads in the
        thousands there.
        """
        host = _probe_grid(dtype)
        out = torch.erf(_to_device(host)).cpu()
        expected = _reference(host)
        assert out.dtype == dtype
        unit = ULP_AT_ONE[dtype]
        # The dense sweep is the first 400001 operands; its agreement is the
        # reading the increment's first measurement took.
        dense = 400001
        steps = _steps(out[:dense], expected[:dense]).abs()
        if dtype is not torch.float32:
            agree = int((steps == 0).sum())
            assert agree >= dense - 256, f"{dtype}: {agree} of {dense} bit-identical"
            probes = int((out[dense:] == expected[dense:]).sum())
            assert probes >= out.numel() - dense - 16, (dtype, probes)
        gap = (out.double() - expected.double()).abs()
        bound = (0.5 if dtype is not torch.float32 else 4.5) * unit
        assert float(gap.max()) <= bound, (dtype, float(gap.max()))
        # The answer is always inside the bound, and the dense sweep's far end is
        # the reference's own rounding to exactly 1 at every width.
        assert (out.abs() <= 1.0).all()
        assert out[dense - 1].item() == 1.0 == expected[dense - 1].item()

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_node_tracks_the_reference(self, dtype):
        """The native node follows the reference at every served width.

        ``tanh`` was declined at float32 because its node was a coarser
        approximation of the reference's own function -- ``~100x`` its error at
        every scale of the curve. This gate is based on a measurement against
        the exact (float64) erf on the probe grid, where the reference's error
        is its own:

        * at both narrow widths the node is closer to exact on every sampled
          point where the two round differently;
        * at float32 both hosts stay within five steps of the reference. The M5
          measures ``4.19``; the ``macos-14`` runner's older MPSGraph measures a
          larger gap that remains inside the same bound. The M5's exact-curve
          ranking -- ``63.3%`` closer, ``0.98``-step excess -- is recorded in
          ``MpsSupportsErfDtype`` and is not asserted, because the runner orders
          the rounding differently.
        """
        host = _probe_grid(dtype)
        out = torch.erf(_to_device(host)).cpu()
        expected = _reference(host)
        node_err = (out.double() - expected.double()).abs()
        ref_err = (expected.double() - torch.erf(host.double())).abs()
        unit = ULP_AT_ONE[dtype]
        # The narrow widths: the node is closer wherever they disagree.
        if dtype is not torch.float32:
            differ = out != expected
            exact = torch.erf(host.double())
            node_exact = (out.double() - exact).abs()
            closer = int((node_exact[differ] < ref_err[differ]).sum())
            assert closer == int(differ.sum()), (dtype, closer, int(differ.sum()))
            assert float(node_exact.max()) > 0.0, dtype
            return
        # float32: both implementations have their own last-place rounding. Hold
        # the native path to the gap measured across both hosts, and prove this
        # call was served rather than answered by a CPU round trip.
        assert float(node_err.max()) <= 5.0 * unit, float(node_err.max())
        assert int((node_err == 0.0).sum()) > 100000, int((node_err == 0.0).sum())
        assert int((ref_err > 0.5 * unit).sum()) > 1000, int(
            (ref_err > 0.5 * unit).sum()
        )
        source = (
            "from device_init import torch, torch_fl; "
            "a = torch.randn(4, 4, device='flagos:0'); "
            "torch.erf(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1

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
        # Compare in float64 so a narrow result is not promoted to float32
        # before it reaches the bound. These operands follow a different grid
        # from the probe cases; float16's measured gaps reach four steps at 1.0.
        gap = (out.double() - expected.double()).abs().max().item()
        steps = 4.0 if dtype is torch.float16 else 4.5
        assert gap <= steps * ULP_AT_ONE[dtype], (dtype, gap)

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
        # same few-step bar -- exact at the two narrow widths and a few steps at
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
            # finite interior points are held to the curve case's step bound.
            for i in (0, 1, 2, 3):
                assert out[i].item() == expected[i].item(), (
                    f"{dtype} erf({DOMAIN_EDGES[i]!r}) = {out[i].item()!r}, "
                    f"expected {expected[i].item()!r}"
                )
            gap = (out[5:].double() - expected[5:].double()).abs()
            assert float(gap.max()) <= 5.0 * ULP_AT_ONE[dtype], (dtype, out, gap)
            # The signed zero, whose *sign* an equality cannot see.
            assert not out[0].signbit().item(), dtype
            assert out[1].signbit().item(), dtype

    @pytest.mark.anyplatform
    def test_the_ordinary_middle_agrees(self):
        """The values a model actually evaluates, compared to the reference.

        The operand values around an activation's usual range, held to the
        reference's own answer. At float32 several round differently, so the
        bound is the measured error rather than bit equality.
        """
        host = torch.tensor([-3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0])
        for dtype in SERVED_DTYPES:
            narrow = host.to(dtype)
            out = torch.erf(_to_device(narrow)).cpu()
            expected = _reference(narrow)
            # Bit equality at the narrow widths, where the reference is itself
            # only approximate and the two roundings meet; a small bound at
            # float32, which is the width this increment widens the gate to and
            # whose rounding difference the curve case measured.
            if dtype is not torch.float32:
                assert torch.equal(out, expected), (dtype, out, expected)
            else:
                got, want = out, expected
                # The zero is exact; the reference's own nearby answers are up to
                # 2.06 float32 steps at 1.0 from exact on these operands.
                assert got[4].item() == want[4].item() == 0.0
                nonzero = want != 0.0
                gap = (got.double() - want.double()).abs()[nonzero]
                assert float(gap.max()) <= 2.5 * ULP_AT_ONE[torch.float32], (
                    out,
                    float(gap.max()),
                )
            assert out[0].item() < out[1].item() < 0.0
            assert out[4].item() == 0.0
            # The curve rises off zero to the bound. The exact ``erf(3.0)`` is
            # 0.99997788: below 1 at float32 and rounded to 1 at both narrow
            # widths, where the final comparison is the non-strict ``<=``.
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
        ``|x| >= 4``, so the two agree bit for bit there. This case asserts that
        directly, so a future node build that stopped saturating at the
        reference's point would fail here rather than pass as a step or two.
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

        Measured on this host, all 127 subnormal bfloat16 operands and 877 of
        float16's 1023 positive subnormals round differently from the reference's
        own answer -- by a step or two, inside a band whose widest value is below
        the width's smallest normal. The CI runner flushes some bfloat16 results
        in that band to zero. A model's ``erf`` operand is a pre-activation, so
        these readings are recorded rather than pinned; the assertion is what
        holds on both machines and would break if the behaviour widened: nothing
        in the normal range is flushed.
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
            # measured, 0 at bfloat16 and 3 at float16 on this host.
            assert int(steps.max()) <= 6, (dtype, int(steps.max()))
            # And the far edge, which is held to the *normal* range's answer
            # rather than to bit equality, because it is still the recorded band:
            # measured, ``erf(1e-20)`` is a bfloat16 subnormal the node answers
            # and the reference rounds to zero, and ``erf(tiny)`` rounds to not
            # quite the same subnormal. That subnormal is itself flushed by the
            # historical `macos-14` CI runner, so this is the *other*
            # host-dependent reading the band records: the node answers
            # ``erf(tiny)`` as a subnormal or as a zero, where the reference
            # always answers a subnormal. Both readings rise with the operand
            # (the zeros sit at the low end), so the assertions are the shape
            # plus the observation that a flushed answer only ever replaces a
            # reference that is itself subnormal; nothing at or above the normal
            # range is held to a flush.
            normals = torch.tensor([tiny, tiny * 2.0, 1e-3, 1.0], dtype=dtype)
            out_n = torch.erf(_to_device(normals)).cpu()
            wanted = _reference(normals).double()
            got = out_n.double()
            assert (got >= 0).all() and (got < 1).all(), (dtype, out_n)
            assert got[0].item() <= tiny * 4.0, (dtype, out_n)
            flushed = got == 0
            assert not bool(flushed[2:].any()), (dtype, out_n)
            assert bool((wanted[flushed] < tiny).all()), (dtype, out_n)
            rising = torch.ones_like(flushed)
            rising[1:] = ~(flushed[:-1] & ~flushed[1:])
            assert bool((got[1:][rising[1:]] >= got[:-1][rising[1:]]).all()), (
                dtype,
                out_n,
            )
            kept = (wanted > 0) & ~flushed
            gap = (got[kept] - wanted[kept]).abs() / wanted[kept]
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
        assert out[0].signbit().item(), out
        assert torch.isnan(out[4]).item()
        # The edges with no rounding to disagree about -- both zeros, both
        # infinities -- are the reference exactly. The two interior values can
        # differ because the reference keeps its own rounding at float64.
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
        what tells the two apart. It does *not* assert bit equality: the
        reference and the node round different points of the curve differently.
        It asserts the served path's own signature: the answer is inside the
        bound, the curve's shape is the reference's, and the node's error stays
        close to the reference's own.
        """
        host = torch.linspace(-6.0, 6.0, 400001)
        out = torch.erf(_to_device(host)).cpu()
        assert out.dtype == torch.float32
        expected = _reference(host)
        # On this regular sweep the two roundings stay within a few float32
        # steps; the curve case measures the gap on the harder probe grid.
        gap = (out - expected).abs()
        assert float(gap.max()) <= 5.0 * ULP_AT_ONE[torch.float32], float(gap.max())
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
        # The node is not a coarser curve: it is at least as close to the exact
        # float64 answer as the reference on the great majority of this sweep.
        exact = torch.erf(host.double())
        node_err = out.double().sub(exact).abs()
        ref_err = expected.double().sub(exact).abs()
        assert int((node_err <= ref_err).sum()) >= int(0.75 * out.numel()), (
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
            # What this case is about is the *plan*, so the assertion is that no
            # case picked up another's answer. The step bound holds the narrow
            # widths to a handful of steps and float32 to a few float32 steps at
            # 1.0.
            if dtype is not torch.float32:
                assert int(_steps(got, wanted).abs().max()) <= 4, dtype
                assert int((got != wanted).sum()) <= max(8, got.numel() // 100), dtype
            else:
                assert float((got.double() - wanted.double()).abs().max()) <= (
                    5.0 * ULP_AT_ONE[torch.float32]
                ), dtype

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
                "from device_init import torch, torch_fl; "
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
            "from device_init import torch, torch_fl; "
            f"a = torch.randn(4, 4, dtype=torch.{dtype}).to('flagos:0'); "
            "torch.erf(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
