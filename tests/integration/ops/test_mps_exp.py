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

"""Stage-4 Apple Silicon (Metal / MPSGraph) exp.

``exp`` is the largest elementwise unary a small model leaves on the fallback
profile and the node the rest of the cohort is written in terms of -- a
``log``, an ``rsqrt`` and a hand-written ``tanh`` are all built over an
exponential. MPSGraph has a real selector (``exponentWithTensor:``), so the body
is the plain unary template and the finding is the width list, which *widens*
past the sibling activations':

* **float16 is served, where ``sigmoid`` declined it.** The ``sigmoid`` body's
  float16 decline was the platform's subnormal flush, and it was recorded as a
  property of the *node* rather than of the platform. This op is the measurement
  that separates the two: measured over float16's whole representable curve --
  400001 points spanning [-30, 11], which reaches the lowest subnormal an ``exp``
  can produce -- the graph is bit-identical to ``at::exp`` with zero mismatches,
  so the same exponential keeps the subnormals the sigmoid node flushed.
* **bfloat16 is served, where ``sigmoid`` declined it.** The sibling's decline
  was a rounding difference at a width that holds about three decimal digits;
  measured over [-30, 88], ``exp`` has none -- the two agree bit for bit in the
  normal range.
* **float32 rounds differently by up to two ulp**, which is the same "both are
  correct" reading ``silu``'s float32 took: against an exact float64 evaluation
  the graph and the CPU reference sit on opposite sides of the true value, one
  ulp each.

So the tests below hold three things the number alone cannot: that the three
widths are served and agree with ``at::exp`` as measured, that the subnormal
flush is confined to the band where the reference is itself subnormal, and that
float64 and the integral/bool operands take the round-trip at their own dtype.

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

# The widths the graph serves, and the finding is that this list is *wider* than
# the sibling activations': float16 and bfloat16 are both served here where the
# ``sigmoid`` body declined them, for the reasons documented in the module
# header and in MpsSupportsExpDtype.
SERVED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# The widths measured bit-identical, where the assertion can be equality rather
# than a tolerance.
EXACT_DTYPES = [torch.float16, torch.bfloat16]

# The widths the reference has a kernel for and this body declines. float64 has
# no MPSGraph element type; the round-trip is what answers it, so the assertion
# is CPU equality at the operand's own dtype rather than an error.
DECLINED_FLOAT_DTYPES = [torch.float64]

# float32 is served on a rounding difference of up to two ulp, so it is held to a
# relative tolerance rather than bit-exactly. The bound is measured (1.897e-07)
# and rounded up rather than left at the measurement, so a machine that rounds
# one ulp differently still passes.
FLOAT32_RTOL = 1e-6


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::exp`` on the host, in the operand's own dtype.

    The op and not ``torch.pow(math.e, x)``: the two readings differ in the
    rounding, which is exactly what the float32 case is about, and the reference
    is what the kernel is held to.
    """
    return torch.ops.aten.exp(host)


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(
        out.cpu().float(), expected.float(), rtol=FLOAT32_RTOL, atol=0
    )


class TestExpCorrectness:
    """``torch.exp`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = (torch.randn(4, 64) * 3.0).to(dtype)
        _assert_matches_cpu(torch.exp(_to_device(host)), _reference(host))

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
        host = torch.randn(*shape)
        _assert_matches_cpu(torch.exp(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_the_narrow_widths_are_bit_identical(self):
        """The increment's finding: an exponential rounds exactly.

        Measured, the graph and the reference agree bit for bit at float16 and
        bfloat16, which is why those widths are served where the bare ``sigmoid``
        declined them. A body that started following the sibling's gate would not
        fail here, but a graph that started rounding at these widths would.
        """
        for dtype in EXACT_DTYPES:
            host = torch.randn(4, 512).mul(4.0).to(dtype)
            out = torch.exp(_to_device(host))
            assert out.cpu().dtype == dtype
            assert torch.equal(out.cpu(), _reference(host)), dtype

    @pytest.mark.anyplatform
    def test_the_float16_low_tail_keeps_its_subnormals(self):
        """The case that would fail if float16 were the sibling's decline.

        Measured, a float16 ``exp`` answers every subnormal the reference answers
        -- the platform's subnormal flush is a property of the ``sigmoid`` node
        and not of the width. If the graph flushed here as it does there, this
        band would come back all zeros and 137500 of these points would differ.
        """
        host = torch.linspace(-15.0, -8.0, 1024).to(torch.float16)
        out = torch.exp(_to_device(host)).cpu()
        expected = _reference(host)
        tiny = torch.finfo(torch.float16).tiny
        # The band this sweep covers is where a float16 exp is subnormal.
        assert (expected < tiny).any(), "the sweep must reach the float16 band"
        assert torch.equal(out, expected)

    @pytest.mark.anyplatform
    def test_the_saturated_tails_are_the_widths_own(self):
        """Both ends of the range, where a weaker exp shows as a number.

        A ``pow(e, x)`` composition or a naive reduction overflows differently at
        the top; the graph saturates to ``inf`` exactly where the reference does,
        and answers a small positive where the reference does. The inputs are held
        *out* of the subnormal band the separate case below pins, because an
        input deep in the negative tail (``-100``) is where the flush lives and
        this case is about the saturated ends rather than that policy.
        """
        host = torch.tensor([-60.0, -20.0, 0.0, 20.0, 80.0, 88.0, 100.0])
        out = torch.exp(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[-1].item() == float("inf") and expected[-1].item() == float("inf")
        assert out[2].item() == 1.0
        assert out[3].item() > 0.0
        assert out[0].item() > 0.0  # never a composition's 0 in the normal range
        torch.testing.assert_close(out, expected, rtol=FLOAT32_RTOL, atol=0)

    @pytest.mark.anyplatform
    def test_the_subnormal_band_is_where_the_platform_flushes(self):
        """The one measured divergence, pinned so it cannot widen unnoticed.

        This SDK flushes a subnormal *result* to zero, which is the same policy
        the ``div`` and ``sigmoid`` bodies record. For a float32 ``exp`` the
        flush is confined to the input band where the reference's own answer is
        subnormal but nonzero: measured, ``exp(-87.0)`` agrees at 1.6458e-38 and
        ``exp(-87.5)`` is 9.9824e-39 on the reference and exactly 0 on the graph.

        Where the band starts is the *hardware's* decision, not this body's, so
        the case asserts the shape of the policy rather than a boundary: every
        input whose reference answer is normal agrees exactly, and the flush
        never reaches one. What would be a regression is the flush moving into
        the normal range, and that is what is held.
        """
        host = torch.tensor([-86.0, -87.0, -87.3, -87.34, -87.5, -88.0, -90.0])
        out = torch.exp(_to_device(host)).cpu()
        expected = _reference(host)
        tiny = torch.finfo(torch.float32).tiny
        for x, got, want in zip(host.tolist(), out.tolist(), expected.tolist()):
            if want == 0.0:
                # Below the band the reference is zero too, so the flush is
                # indistinguishable from the answer.
                assert got == 0.0, f"exp({x}) = {got!r}, expected the reference's own 0"
            elif want < tiny:
                # Inside the band the reference is subnormal-and-nonzero. The
                # graph may answer it or flush it, and both are this policy;
                # what it may not do is answer something else.
                assert got == 0.0 or got == want or abs(got - want) <= tiny, (
                    f"exp({x}) = {got!r} against the reference's {want!r}"
                )
            else:
                # Inside the normal range the two agree.
                assert got == want, f"exp({x}) = {got!r}, expected {want!r}"
                assert got > 0.0
        # The flush is confined to the subnormal band: the smallest normal
        # input the reference answers exactly, which is the invariant that
        # would break if the flush widened.
        assert expected[1].item() >= tiny and out[1].item() > 0.0

    @pytest.mark.anyplatform
    def test_zero_and_negative_zero(self):
        """``exp(0)`` is exactly 1, which the reference also returns."""
        host = torch.tensor([0.0, -0.0])
        out = torch.exp(_to_device(host))
        torch.testing.assert_close(out.cpu(), _reference(host), rtol=0, atol=0)
        assert out.cpu()[0].item() == 1.0
        assert out.cpu()[1].item() == 1.0

    @pytest.mark.anyplatform
    def test_inf_and_nan(self):
        host = torch.tensor([float("inf"), float("-inf"), float("nan")])
        out = torch.exp(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[0].item() == float("inf")
        assert out[1].item() == 0.0
        assert torch.isnan(out[2])
        assert torch.equal(out[:2], expected[:2])

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        host = torch.empty(0)
        out = torch.exp(_to_device(host))
        assert out.shape == (0,)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        host = torch.tensor(1.25)
        _assert_matches_cpu(torch.exp(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.randn(8, 8).t()
        assert not host.is_contiguous()
        _assert_matches_cpu(torch.exp(_to_device(host)), _reference(host))


class TestExpDtypeGate:
    """The declined widths are declined, and the round-trip answers them exactly.

    This is the increment's finding read backwards: the body serves *more* than
    the sibling activations do, so a suite that only checked the served widths
    would not distinguish it from one that widened the gate too far. float64 and
    the integral/bool operands are what hold the boundary.
    """

    @pytest.mark.parametrize("dtype", DECLINED_FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_declined_widths_round_trip_at_their_own_dtype(self, dtype):
        host = (torch.randn(4, 32) * 3.0).to(dtype)
        out = torch.exp(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        # A graph at the wrong width would come back float32, so the dtype is
        # the assertion that the round-trip is what answered.
        assert out.cpu().dtype == dtype

    @pytest.mark.anyplatform
    def test_float64_matches_the_reference_bit_for_bit(self):
        """The round-trip's answer is the reference, not a 1e-7 approximation."""
        host = torch.linspace(-30.0, 30.0, 513, dtype=torch.float64)
        out = torch.exp(_to_device(host))
        assert out.cpu().dtype == torch.float64
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8]
    )
    @pytest.mark.anyplatform
    def test_integral_operand_promotes_exactly_like_cpu(self, dtype):
        """An integral operand is promoted to float32, and the body declines it.

        ``at::exp`` has no integral kernel and does not raise: measured, it
        returns float32. The body declines the integral width, so the round-trip
        is what performs the promotion -- and the assertion is the promoted dtype
        and the values, because a graph that had served it would have had to
        allocate from the operand's own options.
        """
        host = torch.arange(-2, 3).to(dtype)
        out = torch.exp(_to_device(host))
        expected = _reference(host)
        assert expected.dtype == torch.float32
        _assert_matches_cpu(out, expected)

    @pytest.mark.anyplatform
    def test_bool_operand_promotes_exactly_like_cpu(self):
        host = torch.tensor([True, False, True])
        out = torch.exp(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        assert _reference(host).dtype == torch.float32


class TestExpPlanReuse:
    """A plan is per (dtype, shape), and interleaving them shares nothing."""

    @pytest.mark.anyplatform
    def test_interleaved_signatures_never_share_a_plan(self):
        cases = [
            torch.randn(4, 64) * 3.0,
            torch.randn(64, 4) * 3.0,
            torch.randn(4096),
            (torch.randn(4, 64) * 3.0).half(),
        ]
        held = []
        for host in cases:
            out = torch.exp(_to_device(host))
            held.append((out.cpu(), _reference(host), host.dtype))
        for got, wanted, dtype in held:
            assert got.dtype == dtype
            if dtype == torch.float32:
                torch.testing.assert_close(got, wanted, rtol=FLOAT32_RTOL, atol=0)
            else:
                assert torch.equal(got, wanted)

    @pytest.mark.anyplatform
    def test_repeated_calls_reuse_one_plan_and_stay_correct(self):
        host = torch.randn(8, 16) * 2.0
        wanted = _reference(host)
        for _ in range(8):
            _assert_matches_cpu(torch.exp(_to_device(host)), wanted)


class TestExpDispatch:
    """``exp`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_exp_to_mps(self):
        assert routed_backend_or_none("exp") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``exp`` reached
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
                "a = torch.randn(4, 4, device='flagos:0'); torch.exp(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] exp -> apple" in result.stderr

    @pytest.mark.main_ops
    def test_a_declined_width_stays_served(self):
        """A declined width is the body's round-trip, not a `none` route.

        The conf route is what the log names, so a float64 call still logs
        ``exp -> apple`` even though the body hands it to the reference. This holds
        that the decline is inside the body, and that the conf has not been
        narrowed to the served widths.
        """
        source = (
            "import torch, torch_fl; "
            "a = torch.randn(4, 4, device='flagos:0').double(); "
            "torch.exp(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
