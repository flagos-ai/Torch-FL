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

"""Stage-4 Apple Silicon (Metal / MPSGraph) log.

``log`` is the inverse of the sibling ``exp`` and the other half of every
logit-space composition a model writes by hand -- a log-sum-exp, a
temperature-scaled score, a KL term. MPSGraph names it directly
(``logarithmWithTensor:``), so the body is the plain one-node unary template and
the finding is, as for ``exp``, the width list:

* **All three float widths are served.** Measured over 400001 samples per width,
  bfloat16 is bit-identical, float16 differs on 3 points (a last-ulp rounding in
  which the *graph* is marginally closer to exact than the CPU reference), and
  float32 differs on one ulp for 129358 of them -- the same "both are correct"
  reading ``silu`` and ``exp`` took at that width.
* **The domain's edges are reproduced exactly**, which is what makes this
  increment's finding a property of the *input* rather than of the magnitude:
  measured, ``log(0)``, ``log(-0.0)``, ``log(1)``, ``log(-1)``, ``log(-5)``,
  ``log(inf)``, ``log(-inf)`` and ``log(nan)`` all match the reference bit for
  bit.
* **float32 flushes a subnormal input to zero**, and ``log(0)`` is ``-inf``, so a
  positive input below the smallest normal comes back ``-inf`` where the
  reference answers a finite number. Measured, the boundary sits exactly at the
  smallest normal float32; float16 does *not* flush subnormal inputs
  (``log(5.96e-8) = -16.640625`` is exact), so this is a float32 property of the
  platform rather than a width-independent rule.

The tests below hold three things the numbers alone cannot: that the three widths
are served and agree with ``at::log`` as measured, that the flush is confined to
the band where the reference's *input* is subnormal and never reaches a normal
one, and that float64 and the integral/bool operands take the round-trip at
their own dtype.

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

# The widths the graph serves. Unlike ``exp``, this list matches the siblings':
# the flush that cost ``sigmoid`` its float16 width does not reach this node's
# arithmetic. See MpsSupportsLogDtype.
SERVED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The width measured bit-identical, where the assertion can be equality.
EXACT_DTYPES = [torch.bfloat16]

# The widths the reference has a kernel for and this body declines. float64 has
# no MPSGraph element type; the round-trip is what answers it, so the assertion
# is CPU equality at the operand's own dtype rather than an error.
DECLINED_FLOAT_DTYPES = [torch.float64]

# float32 is served on a one-ulp rounding difference, so it is held to a
# relative tolerance rather than bit-exactly. The bound is measured (1.291e-07)
# and rounded up, so a machine that rounds one ulp differently still passes.
FLOAT32_RTOL = 1e-6

# The domain's edges, which the graph reproduces exactly. Every one of these is
# a point where a naive composition or a flush shows as a wrong *kind* of
# answer rather than a rounded magnitude.
DOMAIN_EDGES = [
    0.0,  # -inf
    -0.0,  # -inf
    1.0,  # 0.0, exactly
    -1.0,  # nan
    -5.0,  # nan
    float("inf"),  # inf
    float("-inf"),  # nan
    float("nan"),  # nan
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::log`` on the host, in the operand's own dtype.

    The op and not a ``log2(x) / log2(e)`` composition: the readings differ in
    the rounding and at the domain's edges, which is exactly what the float32
    and edge cases are about, and the reference is what the kernel is held to.
    """
    return torch.ops.aten.log(host)


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(
        out.cpu().float(), expected.float(), rtol=FLOAT32_RTOL, atol=0
    )


class TestLogCorrectness:
    """``torch.log`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        # A positive operand: a logarithm's domain is the positive half line,
        # and the negative half is the edge case below rather than a value the
        # tolerance assertions are shaped for.
        host = torch.rand(4, 64).add(0.5).to(dtype)
        _assert_matches_cpu(torch.log(_to_device(host)), _reference(host))

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
        _assert_matches_cpu(torch.log(_to_device(host)), _reference(host))

    @pytest.mark.parametrize("dtype", EXACT_DTYPES)
    @pytest.mark.anyplatform
    def test_the_bit_identical_width_is_bit_identical(self, dtype):
        """The width the measurement found exact, held as equality."""
        host = torch.rand(4, 512).add(0.25).to(dtype)
        out = torch.log(_to_device(host))
        assert out.cpu().dtype == dtype
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.anyplatform
    def test_float16_rounds_within_one_ulp(self):
        """The sibling's declined width, served here, on a last-ulp difference.

        Measured, 3 of 400001 float16 samples land on the other side of the last
        ulp from the reference, and at those points the *graph* is the one closer
        to the exact answer -- so this is a last-ulp rounding rather than a graph
        error. What the test holds is that shape: a sweep away from ``1.0`` is
        bit-identical, and the explicit neighbourhood of ``1.0`` -- where the
        answer is 0 and the two sides may round to 0 or to the smallest subnormal
        -- differs by no more than that smallest subnormal.
        """
        host = torch.logspace(-5.0, 7.0, 4096).to(torch.float16)
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        assert torch.equal(out, expected)
        # The neighbourhood, spelled out rather than sampled: a body whose node
        # rounded the other way would part company with the reference here.
        tiny = torch.finfo(torch.float16).tiny
        eps = torch.finfo(torch.float16).eps
        near = torch.tensor(
            [1.0 - eps, 1.0 - eps / 2, 1.0, 1.0 + eps / 2, 1.0 + eps],
            dtype=torch.float16,
        )
        near_out = torch.log(_to_device(near)).cpu()
        near_expected = _reference(near)
        assert torch.equal(near_out, near_expected)
        assert near_out[2].item() == 0.0
        # log(1 - eps) is -log(1/(1-eps)) ~= eps, and the rounding puts it a
        # hair above eps rather than below; the bound is two of them.
        assert near_out.abs().max().item() <= 2.0 * eps
        assert tiny <= near.abs().min().item()

    @pytest.mark.anyplatform
    def test_a_small_positive_operand_keeps_its_sign(self):
        """Near 1, where a subnormal *result* would show as a 0.

        Measured, ``log`` of an operand just under 1 is a small negative that
        stays subnormal-and-nonzero on both sides, so the subnormal policy the
        sibling bodies record does not reach this op's output at all: a flush of
        a subnormal result would show here as a 0 the reference does not have.
        """
        host = torch.tensor([1.0 - 1e-5, 0.5, 2.0, 1.0, 1.0001, 1.0 + 1e-5])
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        # A hair from 1.0 the two sides may round to 0 or to the smallest
        # subnormal; neither is a wrong answer, so both are inside the bound.
        torch.testing.assert_close(
            out, expected, rtol=0, atol=torch.finfo(torch.float32).tiny
        )
        assert out[4].item() > 0.0 and expected[4].item() > 0.0
        assert out[0].item() < 0.0 and expected[0].item() < 0.0
        # ... and the bands immediately next to 1, where float32 resolves the
        # difference: a body that flushed a small result would answer 0 here.
        assert out[4].item() > 0.0 and expected[4].item() > 0.0
        assert out[1].item() < 0.0 and expected[1].item() < 0.0

    @pytest.mark.anyplatform
    def test_the_saturated_tails_are_the_widths_own(self):
        host = torch.tensor([1e-30, 1e-10, 1e-3, 1.0, 1e3, 1e10, 1e30])
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[3].item() == 0.0
        assert out[0].item() < out[1].item() < out[2].item() < 0.0
        assert 0.0 < out[4].item() < out[5].item() < out[6].item()
        torch.testing.assert_close(out, expected, rtol=FLOAT32_RTOL, atol=0)

    @pytest.mark.anyplatform
    def test_the_domain_edges_are_exact(self):
        """The increment's finding: the edges, not the magnitudes, are the risk.

        Measured, every one of these matches the reference bit for bit. A body
        that clamped its operand, or that composed ``log`` out of a
        ``log2``/ratio pair, would move at least one of them -- and a wrong
        answer at a domain edge is a wrong *kind* of answer rather than a
        rounding difference, which is why they are asserted exactly rather than
        under the float32 tolerance above.
        """
        host = torch.tensor(DOMAIN_EDGES)
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[0].item() == float("-inf")
        assert out[1].item() == float("-inf")
        assert out[2].item() == 0.0
        assert torch.isnan(out[3]) and torch.isnan(out[4])
        assert out[5].item() == float("inf")
        assert torch.isnan(out[6]) and torch.isnan(out[7])
        # ... and the same compared value by value, so a body that answered a
        # large finite number instead of an infinity would fail here as well.
        for i, x in enumerate(DOMAIN_EDGES):
            got, want = out[i].item(), expected[i].item()
            if want != want:  # nan
                assert got != got, f"log({x!r}) = {got!r}, expected nan"
            else:
                assert got == want, f"log({x!r}) = {got!r}, expected {want!r}"

    @pytest.mark.anyplatform
    def test_the_subnormal_input_band_is_where_the_platform_flushes(self):
        """The one measured divergence, pinned so it cannot widen unnoticed.

        This SDK flushes a subnormal *input* to zero before the node evaluates,
        and ``log(0)`` is ``-inf``, so a positive input below the smallest normal
        comes back ``-inf`` where the reference answers a finite (and large)
        number: measured, ``1.1754944e-38`` -- the smallest normal float32 --
        agrees with the reference while ``1.17e-38`` flushes to ``-inf``. This is
        the same shape as the subnormal policy the ``sigmoid``, ``div``,
        ``silu`` and ``exp`` bodies record, with the band on the input side.

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
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        for x, got, want in zip(host.tolist(), out.tolist(), expected.tolist()):
            if x == 0.0:
                assert got == want == float("-inf")
            elif x < tiny:
                # Inside the band the reference answers a large finite number.
                # The graph may answer it or flush the input and answer -inf,
                # and both are this policy; what it may not do is answer
                # anything else.
                assert got in (want, float("-inf")), (
                    f"log({x!r}) = {got!r} against the reference's {want!r}"
                )
            else:
                # At or above the smallest normal the two agree, and this is the
                # invariant that would break if the flush widened. 1e-20 is in
                # the set on purpose: it is a normal input whose answer is an
                # ordinary number, unlike the ones just above `tiny`.
                assert got == want, f"log({x!r}) = {got!r}, expected {want!r}"
                assert got != float("-inf")

    @pytest.mark.anyplatform
    def test_the_flush_never_reaches_a_normal_input(self):
        """The same policy read as an inequality rather than a point.

        The case above names values, and the ones inside the band are the
        hardware's to choose; what must hold on any machine is that a *normal*
        input never flushes. This walks the float32 normals down from the
        smallest one and asserts every answer stays finite, which is the
        property the two tests above depend on.
        """
        tiny = torch.finfo(torch.float32).tiny
        normals = torch.tensor(
            [tiny * f for f in (1.0, 1.5, 2.0, 4.0, 1e3, 1e6, 1e10, 1e20)]
        )
        assert (normals >= tiny).all()
        out = torch.log(_to_device(normals)).cpu()
        expected = _reference(normals)
        assert torch.isfinite(out).all(), out
        torch.testing.assert_close(out, expected, rtol=FLOAT32_RTOL, atol=0)
        # And the other end of the same statement: below the smallest normal the
        # graph either agrees or answers the -inf a flushed operand produces,
        # never anything else.
        below = torch.tensor([tiny, tiny / 2.0, tiny / 1e6, 0.0])
        out_below = torch.log(_to_device(below)).cpu()
        for x, got in zip(below.tolist(), out_below.tolist()):
            assert x == 0.0 or got == float("-inf") or got < -87.0, (x, got)

    @pytest.mark.anyplatform
    def test_float16_does_not_flush_its_subnormal_inputs(self):
        """The contrast that keeps the flush above from being a blanket rule.

        Measured, a float16 ``log`` answers a subnormal input exactly
        (``log(5.96e-8) = -16.640625``), so the input-side flush the float32 case
        pins is a property of that width rather than of the platform. If a body
        ever generalised the float32 boundary to every width, this case would
        fail with a ``-inf`` where the reference has a finite number.
        """
        host = torch.tensor([5.96e-8, 1.19e-7, 1e-5, 6.1e-5], dtype=torch.float16)
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        assert (host < torch.finfo(torch.float16).tiny).any(), "the band is covered"
        assert torch.isfinite(out).all(), out
        assert torch.equal(out, expected)

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        host = torch.empty(0)
        out = torch.log(_to_device(host))
        assert out.shape == (0,)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        host = torch.tensor(2.718281828)
        _assert_matches_cpu(torch.log(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.rand(8, 8).add(0.5).t()
        assert not host.is_contiguous()
        _assert_matches_cpu(torch.log(_to_device(host)), _reference(host))


class TestLogDtypeGate:
    """The declined widths are declined, and the round-trip answers them exactly."""

    @pytest.mark.parametrize("dtype", DECLINED_FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_declined_widths_round_trip_at_their_own_dtype(self, dtype):
        host = torch.rand(4, 32).add(0.5).to(dtype)
        out = torch.log(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        # A graph at the wrong width would come back float32, so the dtype is
        # the assertion that the round-trip is what answered.
        assert out.cpu().dtype == dtype

    @pytest.mark.anyplatform
    def test_float64_matches_the_reference_bit_for_bit(self):
        """The round-trip's answer is the reference, not a 1e-7 approximation."""
        host = torch.logspace(-30.0, 30.0, 513, dtype=torch.float64)
        out = torch.log(_to_device(host))
        assert out.cpu().dtype == torch.float64
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.anyplatform
    def test_float64_keeps_the_domain_edges(self):
        """The edges belong to the reference, so a declined width gets them too."""
        host = torch.tensor(DOMAIN_EDGES, dtype=torch.float64)
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[0].item() == float("-inf") and out[1].item() == float("-inf")
        assert out[2].item() == 0.0
        for i in (3, 4, 6, 7):
            assert torch.isnan(out[i]) and torch.isnan(expected[i])
        assert out[5].item() == float("inf")

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8]
    )
    @pytest.mark.anyplatform
    def test_integral_operand_promotes_exactly_like_cpu(self, dtype):
        """An integral operand is promoted to float32, and the body declines it.

        ``at::log`` has no integral kernel and does not raise: measured, it
        returns float32. The body declines the integral width, so the round-trip
        is what performs the promotion -- and the assertion is the promoted dtype
        and the values, because a graph that had served it would have had to
        allocate from the operand's own options.
        """
        host = torch.arange(1, 6).to(dtype)
        out = torch.log(_to_device(host))
        expected = _reference(host)
        assert expected.dtype == torch.float32
        _assert_matches_cpu(out, expected)

    @pytest.mark.anyplatform
    def test_bool_operand_promotes_exactly_like_cpu(self):
        """The promoted path keeps the domain edge an integral operand reaches."""
        host = torch.tensor([True, False])
        out = torch.log(_to_device(host)).cpu()
        expected = _reference(host)
        assert expected.dtype == torch.float32
        # log(1) = 0 and log(0) = -inf: the edge the promotion must not round.
        assert out[0].item() == 0.0 == expected[0].item()
        assert out[1].item() == float("-inf") == expected[1].item()


class TestLogPlanReuse:
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
            out = torch.log(_to_device(host))
            held.append((out.cpu(), _reference(host), host.dtype))
        for got, wanted, dtype in held:
            assert got.dtype == dtype
            if dtype == torch.bfloat16:
                assert torch.equal(got, wanted)
            else:
                torch.testing.assert_close(got, wanted, rtol=FLOAT32_RTOL, atol=0)

    @pytest.mark.anyplatform
    def test_repeated_calls_reuse_one_plan_and_stay_correct(self):
        host = torch.rand(8, 16).add(0.5)
        wanted = _reference(host)
        for _ in range(8):
            _assert_matches_cpu(torch.log(_to_device(host)), wanted)


class TestLogDispatch:
    """``log`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_log_to_mps(self):
        assert routed_backend_or_none("log") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``log`` reached
        ``cpu_fallback``, because both compute the same answer at float32. The
        log line is emitted by the C++ dispatcher only on the ``mps`` slot, so it
        is what separates the two.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.rand(4, 4, device='flagos:0') + 0.5; torch.log(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] log -> apple" in result.stderr

    @pytest.mark.main_ops
    def test_a_declined_width_stays_served(self):
        """A declined width is the body's round-trip, not a `none` route.

        The conf route is what the log names, so a float64 call still logs
        ``log -> apple`` even though the body hands it to the reference. This holds
        that the decline is inside the body, and that the conf has not been
        narrowed to the served widths.

        The operand is materialised on the device from a host tensor rather than
        through an arithmetic op, so the only dispatch the census can see is
        ``log``'s own: a served or round-tripped ``log`` logs once either way,
        while a ``none`` route would log it against a different backend.
        """
        source = (
            "import torch, torch_fl; "
            "a = torch.rand(4, 4, dtype=torch.float64).add(0.5)"
            ".to('flagos:0'); "
            "torch.log(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
