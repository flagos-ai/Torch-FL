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

"""Stage-4 Apple Silicon (Metal / MPSGraph) rsqrt.

``rsqrt`` is ``1 / sqrt(x)``, the tail every normalisation is written in terms
of -- ``x * rsqrt(v + eps)`` is a layer norm, an RMS norm and a batch norm, and
the ``1 / sqrt(d)`` attention scale is the same shape. It is the first body here
built from *two* MPSGraph nodes rather than one: MPSGraph names the fused
``reciprocalSquareRootWithTensor:``, but that selector is macOS 15+ and the
hosted MPS lane is a ``macos-14`` image, so the fused node aborts the process.
The body composes ``reciprocal(sqrt(x))`` from the two selectors the SDK does
have, and the finding is what the *composition* does at the widths:

* **All three float widths are served.** float32 is bit-identical to the
  reference over 400001 samples and over a sweep of the top of the range.
  float16 and bfloat16 differ on a large fraction of samples, but every
  difference is a last-ulp rounding in which the *graph* is closer to the exact
  answer than the CPU reference is -- measured, the graph is farther from exact
  on **zero** samples at either width. That is the same "both are correct"
  reading ``log``'s float16 recorded rather than a graph error.
* **The domain's edges are reproduced exactly.** ``rsqrt(0) = inf``,
  ``rsqrt(-0.0) = -inf``, ``rsqrt(1) = 1``, ``rsqrt(4) = 0.5``,
  ``rsqrt(-1) = nan``, ``rsqrt(inf) = 0`` and ``rsqrt(nan) = nan``, bit for bit
  at all three widths. The signed zero is the one worth naming: the composition
  reaches ``-inf`` through ``sqrt(-0.0) = -0.0`` and ``reciprocal(-0.0) = -inf``
  rather than through a fused node.
* **The half a composition was expected to risk cost nothing.** A reciprocal
  whose *result* is subnormal is the band ``sigmoid`` declined float16 for; here
  it is unreachable, because ``rsqrt`` is subnormal only for an input beyond the
  largest representable float32, and at float16 the curve saturates at
  ``rsqrt(inf) = 0`` before it goes subnormal. So no width is declined for the
  result half, and the input half is ``log``'s finding unchanged.

The tests below hold what the numbers alone cannot: that the three widths are
served and agree with ``at::rsqrt`` as measured, that the input-side flush is
confined to the band where the reference's *input* is subnormal and never
reaches a normal one, that the result is never farther from exact than the
reference, and that float64 and the integral/bool operands take the round-trip.

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

# The widths the graph serves. Unlike ``sigmoid``, this list is not narrowed by
# the subnormal flush: the flush that matters here is on the *input* side at
# float32, and it is accepted on the same policy ``log`` accepted it. See
# MpsSupportsRsqrtDtype.
SERVED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The width measured bit-identical, where the assertion can be equality.
EXACT_DTYPES = [torch.float32]

# The widths the reference has a kernel for and this body declines. float64 has
# no MPSGraph element type; the round-trip is what answers it, so the assertion
# is CPU equality at the operand's own dtype rather than an error.
DECLINED_FLOAT_DTYPES = [torch.float64]

# float16 and bfloat16 are served on a last-ulp rounding difference, so they are
# held to a relative tolerance rather than bit-exactly. The bound is two units in
# the last place at each width rather than a measured point: a one-ulp difference
# at bfloat16 is worth 2^-8 in relative terms, so a bound of one eps would fail
# on a machine that rounds the other way, and a bound tighter than 2*eps would
# not be a claim about the rounding at all. The measured maxima are well inside
# it (9.756e-04 for float16 against 1.953e-03, 7.752e-03 for bfloat16 against
# 1.563e-02).
LOW_PRECISION_RTOL = {
    torch.float16: 2 * torch.finfo(torch.float16).eps,
    torch.bfloat16: 2 * torch.finfo(torch.bfloat16).eps,
}

# The domain's edges, which the composition reproduces exactly. Every one of
# these is a point where a wrong half -- an unsigned sqrt, a fused node with a
# different zero convention -- shows as a wrong *kind* of answer rather than a
# rounded magnitude.
DOMAIN_EDGES = [
    0.0,  # inf
    -0.0,  # -inf
    1.0,  # 1.0, exactly
    4.0,  # 0.5
    -1.0,  # nan
    float("inf"),  # 0.0
    float("-inf"),  # nan
    float("nan"),  # nan
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::rsqrt`` on the host, in the operand's own dtype.

    The op and not a ``1 / x.sqrt()`` composition: the readings differ in their
    rounding, and the reference is what the kernel is held to.
    """
    return torch.ops.aten.rsqrt(host)


def _exact(host: torch.Tensor) -> torch.Tensor:
    """``1 / sqrt(x)`` in float64, the ground truth the roundings are ranked on."""
    return 1.0 / torch.sqrt(host.to(torch.float64))


def _assert_matches_cpu(
    out: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-6
) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(out.cpu().float(), expected.float(), rtol=rtol, atol=0)


class TestRsqrtCorrectness:
    """``torch.rsqrt`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        # A positive operand: rsqrt's domain is the positive half line, and the
        # non-positive half is the edge case below rather than a value the
        # tolerance assertions are shaped for.
        host = torch.rand(4, 64).add(0.5).to(dtype)
        _assert_matches_cpu(
            torch.rsqrt(_to_device(host)),
            _reference(host),
            rtol=1e-6 if dtype == torch.float32 else LOW_PRECISION_RTOL[dtype],
        )

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
        host = torch.rand(*shape).add(0.5)
        _assert_matches_cpu(torch.rsqrt(_to_device(host)), _reference(host))

    @pytest.mark.parametrize("dtype", EXACT_DTYPES)
    @pytest.mark.anyplatform
    def test_the_bit_identical_width_is_bit_identical(self, dtype):
        """The width the measurement found exact, held as equality."""
        host = torch.rand(4, 512).add(0.25).to(dtype)
        out = torch.rsqrt(_to_device(host))
        assert out.cpu().dtype == dtype
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_the_low_precision_widths_never_round_further_from_exact(self, dtype):
        """The increment's finding for the two wide-difference widths.

        Measured, float16 and bfloat16 differ from the reference on a large
        fraction of samples -- 115782 and 122658 of 400001 -- which reads as a
        graph error until the differences are ranked against the exact answer.
        Ranked, the graph is *farther* from ``1 / sqrt(x)`` on zero samples and
        closer on every one that differs, so both roundings are correct and the
        graph's is the better one. What the test holds is that shape rather than
        a boundary: on a wide positive sweep the graph never answers a value
        farther from exact than the reference does.
        """
        host = torch.logspace(-4.0, 6.0, 4096).to(dtype)
        out = torch.rsqrt(_to_device(host)).cpu().to(torch.float64)
        expected = _reference(host).to(torch.float64)
        exact = _exact(host)
        d_out = (out - exact).abs()
        d_ref = (expected - exact).abs()
        assert not bool((d_out > d_ref).any()), (
            f"{dtype}: graph rounds farther from exact than the reference at "
            f"{int((d_out > d_ref).sum())} points"
        )
        # The sweep is what keeps the invariant above from being vacuous: it
        # spans twelve decades of strictly positive input, so both distances
        # are measured against a moving exact value rather than a constant.
        # What it deliberately does *not* claim is that the two sides part
        # company somewhere. Whether the graph rounds differently from the
        # reference at all is a property of the GPU underneath -- a
        # paravirtualized device answers every sample here exactly as the
        # reference does -- so requiring a difference would assert the host,
        # not the backend.
        # float16 saturates the top of the sweep to ``inf`` -- ``rsqrt(inf) = 0``
        # is the domain edge above, not a rounding under test -- so the width
        # is asserted on the input side, where the twelve decades are intact.
        assert host.numel() == 4096, "the sweep must be the wide one"
        assert host.min() > 0, "the sweep must stay inside the domain"
        assert host.max() > host.min().float() * 1e4, "the sweep must be wide"

    @pytest.mark.anyplatform
    def test_the_domain_edges_are_exact(self):
        """The composition reaches every edge through two nodes.

        Measured, every one of these matches the reference bit for bit at all
        three widths. The signed zero is the edge a naive port gets wrong: a
        body that folded ``-0.0`` to ``0.0`` before the sqrt, or that took the
        reciprocal of ``+0`` unconditionally, would answer ``+inf`` where the
        reference answers ``-inf``. A wrong answer at a domain edge is a wrong
        *kind* of answer rather than a rounding difference, which is why these
        are asserted exactly rather than under a tolerance.
        """
        for dtype in SERVED_DTYPES:
            host = torch.tensor(DOMAIN_EDGES, dtype=dtype)
            out = torch.rsqrt(_to_device(host)).cpu()
            expected = _reference(host)
            assert out[0].item() == float("inf"), dtype
            assert out[1].item() == float("-inf"), dtype
            assert out[2].item() == 1.0, dtype
            assert out[3].item() == 0.5, dtype
            assert torch.isnan(out[4]), dtype
            assert out[5].item() == 0.0, dtype
            assert torch.isnan(out[6]), dtype
            assert torch.isnan(out[7]), dtype
            # ... and the same compared value by value, so a body that answered
            # a large finite number instead of an infinity would fail here too.
            for i, x in enumerate(DOMAIN_EDGES):
                got, want = out[i].item(), expected[i].item()
                if want != want:  # nan
                    assert got != got, f"{dtype} rsqrt({x!r}) = {got!r}, expected nan"
                else:
                    assert got == want, (
                        f"{dtype} rsqrt({x!r}) = {got!r}, expected {want!r}"
                    )

    @pytest.mark.anyplatform
    def test_the_reciprocal_family_reaches_every_representable_answer(self):
        """The middle of the curve, where a wrong node would show as a scale.

        A body that used ``x ** -0.5`` or a square-then-reciprocal pair in the
        wrong order would still pass the edges above; what separates it is the
        ordinary middle, which is what a normalisation actually evaluates.
        """
        host = torch.tensor([1e-6, 1e-3, 0.25, 0.5, 1.0, 2.0, 4.0, 1e3, 1e6])
        out = torch.rsqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert torch.equal(out, expected)
        assert out[4].item() == 1.0
        assert out[3].item() > out[4].item() > out[5].item()

    @pytest.mark.anyplatform
    def test_the_subnormal_input_band_is_where_the_platform_flushes(self):
        """The one measured divergence, pinned so it cannot widen unnoticed.

        This SDK flushes a subnormal *input* to zero before the sqrt, and
        ``rsqrt(0)`` is ``inf``, so a positive input below the smallest normal
        comes back ``inf`` where the reference answers a finite (and large)
        number: measured, ``1.1754944e-38`` -- the smallest normal float32 --
        agrees with the reference while ``1.1754942e-38``, the largest
        subnormal, flushes to ``inf``. This is the same shape as the subnormal
        policy the ``log``, ``sigmoid``, ``div``, ``silu`` and ``exp`` bodies
        record, with the band on the input side.

        Where the band starts is the *hardware's* decision, not this body's, so
        the case asserts the shape of the policy rather than a boundary: every
        input at or above the smallest normal agrees exactly, and the flush never
        reaches one. What would be a regression is the flush moving into the
        normal range, and that is what is held.
        """
        tiny = torch.finfo(torch.float32).tiny
        # 1e-20 and tiny*2 are normal *inputs* whose answers are ordinary
        # numbers; the values in between are the band the platform flushes.
        host = torch.tensor([1e-20, tiny, tiny * 2.0, 1e-38, 5e-39, 1e-45, 0.0])
        out = torch.rsqrt(_to_device(host)).cpu()
        expected = _reference(host)
        for x, got, want in zip(host.tolist(), out.tolist(), expected.tolist()):
            if x == 0.0:
                assert got == want == float("inf")
            elif x < tiny:
                # Inside the band the reference answers a large finite number.
                # The graph may answer it or flush the input and answer inf, and
                # both are this policy; what it may not do is answer anything
                # else.
                assert got in (want, float("inf")), (
                    f"rsqrt({x!r}) = {got!r} against the reference's {want!r}"
                )
            else:
                # At or above the smallest normal the two agree, and this is the
                # invariant that would break if the flush widened.
                assert got == want, f"rsqrt({x!r}) = {got!r}, expected {want!r}"
                assert got != float("inf")

    @pytest.mark.anyplatform
    def test_the_flush_never_reaches_a_normal_input(self):
        """The same policy read as an inequality rather than a point.

        The case above names values, and the ones inside the band are the
        hardware's to choose; what must hold on any machine is that a *normal*
        input never flushes. This walks the float32 normals down from the
        smallest one and asserts every answer stays finite, which is the
        property the case above depends on.
        """
        tiny = torch.finfo(torch.float32).tiny
        normals = torch.tensor(
            [tiny * f for f in (1.0, 1.5, 2.0, 4.0, 1e3, 1e6, 1e10, 1e20)]
        )
        assert (normals >= tiny).all()
        out = torch.rsqrt(_to_device(normals)).cpu()
        expected = _reference(normals)
        assert torch.isfinite(out).all(), out
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=0)
        # And the other end of the same statement: below the smallest normal the
        # graph either agrees or answers the inf a flushed operand produces,
        # never anything else.
        below = torch.tensor([tiny, tiny / 2.0, tiny / 1e6, 0.0])
        out_below = torch.rsqrt(_to_device(below)).cpu()
        for x, got in zip(below.tolist(), out_below.tolist()):
            assert got == float("inf") or abs(got) > 1e18, (x, got)

    @pytest.mark.anyplatform
    def test_float16_does_not_flush_its_subnormal_inputs(self):
        """The contrast that keeps the flush above from being a blanket rule.

        Measured, a float16 ``rsqrt`` answers a subnormal input exactly
        (``rsqrt(3.051758e-05) = 181.0`` and ``rsqrt(1.525879e-05) = 256.0``),
        so the input-side flush the float32 case pins is a property of that
        width rather than of the platform. If a body ever generalised the
        float32 boundary to every width, this case would fail with an ``inf``
        where the reference has a finite number.
        """
        host = torch.tensor(
            [3.051758e-05, 1.525879e-05, 6.103516e-05, 0.5], dtype=torch.float16
        )
        out = torch.rsqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert (host < torch.finfo(torch.float16).tiny).any(), "the band is covered"
        assert torch.isfinite(out).all(), out
        assert torch.equal(out, expected)

    @pytest.mark.anyplatform
    def test_the_result_side_band_is_not_reachable(self):
        """The half a composition was expected to risk, held as unreachable.

        A reciprocal whose *result* is subnormal is the band ``sigmoid``
        declined float16 for, and this body is a reciprocal one node away from
        that band. Measured, the band is not reachable: at float32 and bfloat16
        ``rsqrt`` is subnormal only for an input beyond the largest representable
        value, and at float16 the curve saturates at ``rsqrt(inf) = 0`` before it
        goes subnormal. So the test sweeps the top of each width's range and
        asserts no reference answer in that sweep is a *nonzero* subnormal,
        which is the premise the gate's width list rests on.

        The exact zeros are the top of the same sweep: a bfloat16 or float32
        value near the largest representable rounds up to ``inf`` when it is
        cast, and ``rsqrt(inf) = 0``. A zero is not a denormal and the platform
        does not flush it, so the two are distinguished here rather than lumped
        together by one magnitude test.
        """
        for dtype, top in (
            (torch.float32, 3.4e38),
            (torch.float16, 6.5e4),
            (torch.bfloat16, 3.4e38),
        ):
            host = torch.linspace(top * 1e-3, top, 4096, dtype=torch.float64).to(dtype)
            host = host[host > 0]
            expected = _reference(host)
            tiny = torch.finfo(dtype).tiny
            denormal = (expected != 0.0) & (expected.abs() < tiny)
            assert not bool(denormal.any()), (
                f"{dtype}: the sweep reached a subnormal result, so the band is "
                "no longer unreachable and the gate's reasoning needs revisiting"
            )
            assert torch.isfinite(expected).all(), dtype

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        host = torch.empty(0)
        out = torch.rsqrt(_to_device(host))
        assert out.shape == (0,)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        host = torch.tensor(2.0)
        _assert_matches_cpu(torch.rsqrt(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.rand(8, 8).add(0.5).t()
        assert not host.is_contiguous()
        _assert_matches_cpu(torch.rsqrt(_to_device(host)), _reference(host))


class TestRsqrtDtypeGate:
    """The declined widths are declined, and the round-trip answers them exactly."""

    @pytest.mark.parametrize("dtype", DECLINED_FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_declined_widths_round_trip_at_their_own_dtype(self, dtype):
        host = torch.rand(4, 32).add(0.5).to(dtype)
        out = torch.rsqrt(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        # A graph at the wrong width would come back float32, so the dtype is
        # the assertion that the round-trip is what answered.
        assert out.cpu().dtype == dtype

    @pytest.mark.anyplatform
    def test_float64_matches_the_reference_bit_for_bit(self):
        """The round-trip's answer is the reference, not a 1e-7 approximation."""
        host = torch.logspace(-30.0, 30.0, 513, dtype=torch.float64)
        out = torch.rsqrt(_to_device(host))
        assert out.cpu().dtype == torch.float64
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.anyplatform
    def test_float64_keeps_the_domain_edges(self):
        """The edges belong to the reference, so a declined width gets them too."""
        host = torch.tensor(DOMAIN_EDGES, dtype=torch.float64)
        out = torch.rsqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[0].item() == float("inf") and out[1].item() == float("-inf")
        assert out[2].item() == 1.0 and out[3].item() == 0.5
        for i in (4, 6, 7):
            assert torch.isnan(out[i]) and torch.isnan(expected[i])
        assert out[5].item() == 0.0

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8]
    )
    @pytest.mark.anyplatform
    def test_integral_operand_promotes_exactly_like_cpu(self, dtype):
        """An integral operand is promoted to float32, and the body declines it.

        ``at::rsqrt`` has no integral kernel and does not raise: measured, it
        returns float32. The body declines the integral width, so the round-trip
        is what performs the promotion -- and the assertion is the promoted
        dtype and the values, because a graph that had served it would have had
        to allocate from the operand's own options.
        """
        host = torch.arange(1, 6).to(dtype)
        out = torch.rsqrt(_to_device(host))
        expected = _reference(host)
        assert expected.dtype == torch.float32
        _assert_matches_cpu(out, expected)

    @pytest.mark.anyplatform
    def test_bool_operand_promotes_exactly_like_cpu(self):
        """The promoted path keeps the domain edge an integral operand reaches."""
        host = torch.tensor([True, False])
        out = torch.rsqrt(_to_device(host)).cpu()
        expected = _reference(host)
        assert expected.dtype == torch.float32
        # rsqrt(1) = 1 and rsqrt(0) = inf: the edge the promotion must not round.
        assert out[0].item() == 1.0 == expected[0].item()
        assert out[1].item() == float("inf") == expected[1].item()


class TestRsqrtPlanReuse:
    """A plan is per (dtype, shape), and interleaving them shares nothing."""

    @pytest.mark.anyplatform
    def test_interleaved_signatures_never_share_a_plan(self):
        cases = [
            torch.rand(4, 64).add(0.5),
            torch.rand(64, 4).add(0.5),
            torch.rand(4096).add(0.5),
            torch.rand(4, 64).add(0.5).bfloat16(),
        ]
        held = []
        for host in cases:
            out = torch.rsqrt(_to_device(host))
            held.append((out.cpu(), _reference(host), host.dtype))
        for got, wanted, dtype in held:
            assert got.dtype == dtype
            if dtype in LOW_PRECISION_RTOL:
                torch.testing.assert_close(
                    got.float(),
                    wanted.float(),
                    rtol=LOW_PRECISION_RTOL[dtype],
                    atol=0,
                )
            else:
                torch.testing.assert_close(got, wanted, rtol=1e-6, atol=0)

    @pytest.mark.anyplatform
    def test_repeated_calls_reuse_one_plan_and_stay_correct(self):
        host = torch.rand(8, 16).add(0.5)
        wanted = _reference(host)
        for _ in range(8):
            _assert_matches_cpu(torch.rsqrt(_to_device(host)), wanted)


class TestRsqrtDispatch:
    """``rsqrt`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_rsqrt_to_mps(self):
        assert routed_backend_or_none("rsqrt") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_rsqrt_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``rsqrt`` reached
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
                "a = torch.rand(4, 4, device='flagos:0') + 0.5; torch.rsqrt(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] rsqrt -> mps" in result.stderr

    @pytest.mark.main_ops
    def test_a_declined_width_stays_served(self):
        """A declined width is the body's round-trip, not a ``none`` route.

        The conf route is what the log names, so a float64 call still logs
        ``rsqrt -> mps`` even though the body hands it to the reference. This
        holds that the decline is inside the body, and that the conf has not
        been narrowed to the served widths.

        The operand is materialised on the device from a host tensor rather than
        through an arithmetic op, so the only dispatch the census can see is
        ``rsqrt``'s own: a served or round-tripped ``rsqrt`` logs once either
        way, while a ``none`` route would log it against a different backend.
        """
        source = (
            "import torch, torch_fl; "
            "a = torch.rand(4, 4, dtype=torch.float64).add(0.5)"
            ".to('flagos:0'); "
            "torch.rsqrt(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
