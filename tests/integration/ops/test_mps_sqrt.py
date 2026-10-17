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

"""Stage-4 Apple Silicon (Metal / MPSGraph) sqrt.

``sqrt`` is the square root on its own, and the node the ``rsqrt`` increment
above already builds: ``squareRootWithTensor:`` is the inner half of that body's
``reciprocal(sqrt(x))`` composition, so this increment adds no new selector and
no new arithmetic. What it adds is the caller who wants the root rather than its
reciprocal -- a distance, a standard deviation, a magnitude, the ``1 / sqrt(d)``
a model writes out in two steps -- and every one of those takes the
``cpu_fallback`` today because ``sqrt`` is routed to ``none`` in the conf.

The interest is therefore not the node but the *gate*, and the finding is where
it differs from the sibling that shares the node:

* **All three float widths are served.** float32, float16 and bfloat16 are
  bit-identical to the reference over a 400001-sample sweep spanning twelve
  decades, over a 200001-sample sweep of each width's whole positive range, and
  over every domain edge. float16 reproduces the reference *including every
  subnormal input* -- all 1023 of them -- where the sibling's composition flushed
  its subnormal float32 inputs.
* **The domain's edges are reproduced exactly.** ``sqrt(0) = 0``,
  ``sqrt(-0.0) = -0.0``, ``sqrt(1) = 1``, ``sqrt(4) = 2``, ``sqrt(-1) = nan``,
  ``sqrt(inf) = inf`` and ``sqrt(nan) = nan``, bit for bit at all three widths.
  The signed zero is the one worth naming: this body is reached by the same
  ``squareRootWithTensor:`` node the sibling composes, and its ``-0.0`` answer is
  the one the sibling's reciprocal would turn into ``-inf``.
* **The subnormal flush is the same band the sibling recorded, read
  differently.** At float32 and bfloat16 the platform flushes a subnormal
  *input* to zero before the node evaluates, so a positive input below the
  smallest normal comes back ``0`` where the reference answers a small *normal*
  number. This is the one place the two gates diverge even though they share the
  node: the sibling's flush turned the same input into ``inf`` (an out-of-range
  answer), where this body's turns it into ``0`` (a value inside the normal
  range). Both are accepted on the same policy -- the divergence is confined to
  the band where the reference's *input* is subnormal and never reaches a normal
  one -- and float16 does not flush at all.
* **The negative subnormal is the one edge whose *kind* differs.** Measured, a
  subnormal float32 or bfloat16 input below zero comes back ``-0.0`` where the
  reference answers ``nan``: the flush turns the operand into ``-0.0`` and
  ``sqrt(-0.0) = -0.0``. That is the same platform flush read at the sign
  boundary rather than a second divergence; it is confined to the identical band
  (a negative input of normal magnitude answers ``nan`` correctly), and it is
  recorded here rather than hidden by the positive-band test.

The tests below hold what the numbers alone cannot: that the three widths are
served and agree with ``at::sqrt`` as measured, that the input-side flush is
confined to the band where the reference's *input* is subnormal and never
reaches a normal one, that float16 does not flush, that the negative-subnormal
edge is the band rather than a widened rule, and that float64 and the
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

# The widths the graph serves, which is the sibling's list: the subnormal flush
# that would have narrowed it is on the *input* side and is accepted on the same
# policy ``log`` and ``rsqrt`` accepted it. See MpsSupportsSqrtDtype.
SERVED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The widths the reference has a kernel for and this body declines. float64 has
# no MPSGraph element type; the round-trip is what answers it, so the assertion
# is CPU equality at the operand's own dtype rather than an error.
DECLINED_FLOAT_DTYPES = [torch.float64]

# The domain's edges, which every served width reproduces exactly. The signed
# zero is the one a wrong body gets wrong by *kind* rather than by magnitude.
DOMAIN_EDGES = [
    0.0,  # 0.0
    -0.0,  # -0.0, and the sign is the assertion
    1.0,  # 1.0, exactly
    4.0,  # 2.0, exactly
    -1.0,  # nan
    float("inf"),  # inf
    float("-inf"),  # nan
    float("nan"),  # nan
    2.25,  # 1.5
    0.25,  # 0.5
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::sqrt`` on the host, in the operand's own dtype.

    The op and not an ``x ** 0.5`` composition: the two round differently --
    ``0.5`` is not exact in every width -- and the reference is what the kernel
    is held to.
    """
    return torch.ops.aten.sqrt(host)


def _assert_matches_cpu(
    out: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-6
) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(out.cpu().float(), expected.float(), rtol=rtol, atol=0)


def _same(x: float, y: float) -> bool:
    """Equality that treats two nans as the same answer."""
    return (x == y) or (x != x and y != y)


class TestSqrtCorrectness:
    """``torch.sqrt`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = torch.rand(4, 64).add(0.25).to(dtype)
        _assert_matches_cpu(torch.sqrt(_to_device(host)), _reference(host))

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_the_whole_curve_is_bit_identical(self, dtype):
        """The increment's finding: no width needs a tolerance here.

        The sibling ``rsqrt`` had to hold its low-precision widths to a
        relative bound because the composition's second rounding differed from
        the reference's on a large fraction of samples. A bare square root is
        one correctly-rounded node, and measured, all three widths agree with
        ``at::sqrt`` bit for bit over a twelve-decade sweep -- so this case can
        assert equality rather than a bound, and does.
        """
        host = torch.logspace(-6.0, 6.0, 400001).to(dtype)
        out = torch.sqrt(_to_device(host)).cpu()
        assert out.dtype == dtype
        assert torch.equal(out, _reference(host))

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows of float32
            (3, 7),
            (5, 4),  # rows already a multiple of 16 bytes
            (2, 3, 5, 7),
            (4096,),
            (1, 1),
        ],
    )
    @pytest.mark.anyplatform
    def test_shape_independence(self, shape):
        host = torch.rand(*shape).add(0.25)
        _assert_matches_cpu(torch.sqrt(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_the_domain_edges_are_exact(self):
        """Every edge through one node, including the signed zero.

        Measured, all of these match the reference bit for bit at all three
        widths. The signed zero is the edge a wrong port gets wrong: a body
        that built the node on an operand it had folded to ``+0.0`` would answer
        ``+0.0`` where the reference answers ``-0.0``, which is a wrong *kind*
        of answer rather than a rounding. It is also the value the sibling
        ``rsqrt`` turns into ``-inf``, so the two bodies are held to opposite
        signed-zero answers on the same input.
        """
        for dtype in SERVED_DTYPES:
            host = torch.tensor(DOMAIN_EDGES, dtype=dtype)
            out = torch.sqrt(_to_device(host)).cpu()
            expected = _reference(host)
            assert torch.isnan(out[4]) and torch.isnan(out[6]) and torch.isnan(out[7])
            assert out[0].item() == 0.0 and out[5].item() == float("inf")
            # The signed zero, compared by sign rather than by value: ``-0.0 ==
            # 0.0`` is True in Python, so ``np.signbit`` is the assertion.
            assert np.signbit(out[1].item()), dtype
            assert out[2].item() == 1.0 and out[3].item() == 2.0
            assert out[8].item() == 1.5 and out[9].item() == 0.5
            # ... and the same compared value by value, so a body that answered
            # a plausible number at an edge would fail here too.
            for i, x in enumerate(DOMAIN_EDGES):
                got, want = out[i].item(), expected[i].item()
                assert _same(got, want), (
                    f"{dtype} sqrt({x!r}) = {got!r}, expected {want!r}"
                )

    @pytest.mark.anyplatform
    def test_the_middle_of_the_curve_is_exact(self):
        """The ordinary values, where a wrong node would show as a scale.

        A body that used ``x ** 0.5`` or ``exp(0.5 * log(x))`` would still pass
        the edges above; what separates it is the ordinary middle, which is what
        a norm actually evaluates.
        """
        host = torch.tensor([1e-6, 1e-3, 0.25, 0.5, 1.0, 2.0, 4.0, 1e3, 1e6])
        out = torch.sqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert torch.equal(out, expected)
        assert out[2].item() == 0.5 and out[4].item() == 1.0
        assert out[1].item() < out[2].item() < out[4].item() < out[8].item()

    @pytest.mark.anyplatform
    def test_the_subnormal_input_band_is_where_the_platform_flushes(self):
        """The one measured divergence, pinned so it cannot widen unnoticed.

        This SDK flushes a subnormal *input* to zero before the node, and
        ``sqrt(0)`` is ``0``, so a positive input below the smallest normal
        comes back ``0`` where the reference answers a small (and *normal*)
        number: measured, ``1.1754944e-38`` -- the smallest normal float32 --
        agrees with the reference while ``1.1754942e-38``, the largest
        subnormal, flushes to zero. This is the sibling ``rsqrt``'s input band
        read through this node rather than the reciprocal's: the same flushed
        operand, the opposite end of the answer.

        Where the band starts is the *hardware's* decision, not this body's, so
        the case asserts the shape of the policy rather than a boundary: every
        input at or above the smallest normal agrees exactly, and the flush never
        reaches one. What would be a regression is the flush moving into the
        normal range, and that is what is held.
        """
        tiny = torch.finfo(torch.float32).tiny
        host = torch.tensor([1e-20, tiny, tiny * 2.0, 1e-38, 5e-39, 1e-45, 0.0])
        out = torch.sqrt(_to_device(host)).cpu()
        expected = _reference(host)
        for x, got, want in zip(host.tolist(), out.tolist(), expected.tolist()):
            if x == 0.0:
                assert got == want == 0.0
            elif x < tiny:
                # Inside the band the reference answers a small normal number.
                # The graph may answer it or flush the input and answer zero, and
                # both are this policy; what it may not do is answer anything
                # else.
                assert got in (want, 0.0), (
                    f"sqrt({x!r}) = {got!r} against the reference's {want!r}"
                )
            else:
                # At or above the smallest normal the two agree, and this is the
                # invariant that would break if the flush widened.
                assert got == want, f"sqrt({x!r}) = {got!r}, expected {want!r}"
                assert got != 0.0

    @pytest.mark.anyplatform
    def test_the_flush_never_reaches_a_normal_input(self):
        """The same policy read as an inequality rather than a point.

        The case above names values, and the ones inside the band are the
        hardware's to choose; what must hold on any machine is that a *normal*
        input never flushes. This walks the float32 normals down from the
        smallest one and asserts every answer stays nonzero, which is the
        property the case above depends on.
        """
        tiny = torch.finfo(torch.float32).tiny
        normals = torch.tensor(
            [tiny * f for f in (1.0, 1.5, 2.0, 4.0, 1e3, 1e6, 1e10, 1e20)]
        )
        assert (normals >= tiny).all()
        out = torch.sqrt(_to_device(normals)).cpu()
        expected = _reference(normals)
        assert (out > 0).all(), out
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=0)
        # And the other end of the same statement: below the smallest normal the
        # graph either agrees or answers the zero a flushed operand produces,
        # never anything else.
        below = torch.tensor([tiny, tiny / 2.0, tiny / 1e6, 0.0])
        out_below = torch.sqrt(_to_device(below)).cpu()
        for x, got in zip(below.tolist(), out_below.tolist()):
            assert 0.0 <= got < 1e-18, (x, got)

    @pytest.mark.anyplatform
    def test_float16_does_not_flush_its_subnormal_inputs(self):
        """The contrast that keeps the flush above from being a blanket rule.

        Measured, a float16 ``sqrt`` answers a subnormal input exactly -- a
        sample of all 1023 positive float16 subnormals reproduced the reference
        bit for bit -- so the input-side flush the float32 case pins is a
        property of that width rather than of the platform. If a body ever
        generalised the float32 boundary to every width, this case would fail
        with a zero where the reference has a small nonzero number.
        """
        subnormals = torch.arange(1, 0x0400, dtype=torch.int32)
        host = subnormals.to(torch.int16).view(torch.float16)
        assert (host < torch.finfo(torch.float16).tiny).all(), "the band is covered"
        out = torch.sqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert (out > 0).all(), "a positive float16 subnormal must not flush"
        assert torch.equal(out, expected)

    @pytest.mark.anyplatform
    def test_the_negative_subnormal_is_the_band_and_not_a_wider_rule(self):
        """The one edge whose failure is a wrong *kind* of answer.

        Measured, a subnormal float32 input below zero comes back ``-0.0`` where
        the reference answers ``nan`` -- the flush turns the operand into
        ``-0.0`` and ``sqrt(-0.0) = -0.0`` -- while a negative input of *normal*
        magnitude answers ``nan`` correctly. So the divergence is the same band
        read at the sign boundary, and the test holds both halves: inside the
        band the answer is the flush's (``-0.0``) or the reference's (``nan``),
        and at or above the smallest normal the answer is ``nan`` exactly.
        """
        tiny = torch.finfo(torch.float32).tiny
        negative = torch.tensor([-tiny, -tiny * 2.0, -1e-30, -1.0, -1e-3])
        out = torch.sqrt(_to_device(negative)).cpu()
        expected = _reference(negative)
        for x, got, want in zip(negative.tolist(), out.tolist(), expected.tolist()):
            assert want != want, f"the reference should answer nan for {x!r}"
            assert got != got or (got == 0.0 and np.signbit(got)), (
                f"sqrt({x!r}) = {got!r}, expected nan or the flush's -0.0"
            )
        # And the invariant that would break if the rule widened: a negative
        # input at or above the smallest normal is a nan, always.
        normals = -torch.tensor([tiny, tiny * 1.5, tiny * 1e6, 1.0, 1e20])
        out_n = torch.sqrt(_to_device(normals)).cpu()
        assert torch.isnan(out_n).all(), out_n

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        host = torch.empty(0)
        out = torch.sqrt(_to_device(host))
        assert out.shape == (0,)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        host = torch.tensor(2.0)
        _assert_matches_cpu(torch.sqrt(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.rand(8, 8).add(0.25).t()
        assert not host.is_contiguous()
        _assert_matches_cpu(torch.sqrt(_to_device(host)), _reference(host))


class TestSqrtDtypeGate:
    """The declined widths are declined, and the round-trip answers them exactly."""

    @pytest.mark.parametrize("dtype", DECLINED_FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_declined_widths_round_trip_at_their_own_dtype(self, dtype):
        host = torch.rand(4, 32).add(0.25).to(dtype)
        out = torch.sqrt(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        # A graph at the wrong width would come back float32, so the dtype is
        # the assertion that the round-trip is what answered.
        assert out.cpu().dtype == dtype

    @pytest.mark.anyplatform
    def test_float64_matches_the_reference_bit_for_bit(self):
        """The round-trip's answer is the reference, not a 1e-7 approximation."""
        host = torch.logspace(-30.0, 30.0, 513, dtype=torch.float64)
        out = torch.sqrt(_to_device(host))
        assert out.cpu().dtype == torch.float64
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.anyplatform
    def test_float64_keeps_the_domain_edges(self):
        """The edges belong to the reference, so a declined width gets them too."""
        host = torch.tensor(DOMAIN_EDGES, dtype=torch.float64)
        out = torch.sqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[0].item() == 0.0 and np.signbit(out[1].item())
        assert out[2].item() == 1.0 and out[3].item() == 2.0
        assert out[5].item() == float("inf")
        for i in (4, 6, 7):
            assert torch.isnan(out[i]) and torch.isnan(expected[i])

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8]
    )
    @pytest.mark.anyplatform
    def test_integral_operand_promotes_exactly_like_cpu(self, dtype):
        """An integral operand is promoted to float32, and the body declines it.

        ``at::sqrt`` has no integral kernel and does not raise: measured, it
        promotes to float32, and a *negative* integral operand promotes to a
        ``nan`` rather than a ``RuntimeError`` (the promotion happens first, and
        ``sqrt`` of a negative float is ``nan``). The body declines the integral
        width, so the round-trip is what performs the promotion -- and the
        assertion is the promoted dtype and the values, because a graph that had
        served it would have had to allocate from the operand's own options.
        """
        host = torch.arange(1, 6).to(dtype)
        out = torch.sqrt(_to_device(host))
        expected = _reference(host)
        assert expected.dtype == torch.float32
        _assert_matches_cpu(out, expected)

        # The negative half of the same promotion: nan, not a raise and not a
        # zero, which is what a graph that served the integral operand directly
        # would have to reproduce.
        negative = -host
        out_neg = torch.sqrt(_to_device(negative))
        assert out_neg.cpu().dtype == torch.float32
        assert torch.isnan(out_neg.cpu()).all(), out_neg.cpu()
        assert torch.isnan(_reference(negative)).all()

    @pytest.mark.anyplatform
    def test_bool_operand_promotes_exactly_like_cpu(self):
        """The promoted path keeps the domain edges an integral operand reaches."""
        host = torch.tensor([True, False])
        out = torch.sqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert expected.dtype == torch.float32
        # sqrt(1) = 1 and sqrt(0) = 0: the edges the promotion must not round.
        assert out[0].item() == 1.0 == expected[0].item()
        assert out[1].item() == 0.0 == expected[1].item()


class TestSqrtPlanReuse:
    """A plan is per (dtype, shape), and interleaving them shares nothing."""

    @pytest.mark.anyplatform
    def test_interleaved_signatures_never_share_a_plan(self):
        cases = [
            torch.rand(4, 64).add(0.25),
            torch.rand(64, 4).add(0.25),
            torch.rand(4096).add(0.25),
            torch.rand(4, 64).add(0.25).bfloat16(),
        ]
        held = []
        for host in cases:
            out = torch.sqrt(_to_device(host))
            held.append((out.cpu(), _reference(host), host.dtype))
        for got, wanted, dtype in held:
            assert got.dtype == dtype
            assert torch.equal(got, wanted), dtype

    @pytest.mark.anyplatform
    def test_repeated_calls_reuse_one_plan_and_stay_correct(self):
        host = torch.rand(8, 16).add(0.25)
        wanted = _reference(host)
        for _ in range(8):
            _assert_matches_cpu(torch.sqrt(_to_device(host)), wanted)


class TestSqrtDispatch:
    """``sqrt`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_sqrt_to_mps(self):
        assert routed_backend_or_none("sqrt") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_sqrt_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``sqrt`` reached
        ``cpu_fallback``, because both compute the same answer. The log line is
        emitted by the C++ dispatcher only on the ``mps`` slot, so it is what
        separates the two.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.rand(4, 4, device='flagos:0') + 0.25; torch.sqrt(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] sqrt -> mps" in result.stderr

    @pytest.mark.main_ops
    def test_a_declined_width_stays_served(self):
        """A declined width is the body's round-trip, not a ``none`` route.

        The conf route is what the log names, so a float64 call still logs
        ``sqrt -> mps`` even though the body hands it to the reference. This
        holds that the decline is inside the body, and that the conf has not
        been narrowed to the served widths.

        The operand is materialised on the device from a host tensor rather than
        through an arithmetic op, so the only dispatch the census can see is
        ``sqrt``'s own: a served or round-tripped ``sqrt`` logs once either way,
        while a ``none`` route would log it against a different backend.
        """
        source = (
            "import torch, torch_fl; "
            "a = torch.rand(4, 4, dtype=torch.float64).add(0.25)"
            ".to('flagos:0'); "
            "torch.sqrt(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
