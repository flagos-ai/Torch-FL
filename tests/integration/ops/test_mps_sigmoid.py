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

"""Stage-4 Apple Silicon (Metal / MPSGraph) sigmoid.

``sigmoid`` is the activation ``silu`` is composed from, and an op a model
reaches for on its own: a hand-written attention mask, a gating projection, a
``torch.sigmoid`` in a loss. MPSGraph has a real selector
(``sigmoidWithTensor:``) and it is the *same* selector the routed ``silu`` body
already builds, so the node is not the finding here. The width list is.

``silu`` serves float32, float16 and bfloat16 (measured bit-identical on the two
narrow widths, see ``test_mps_silu.py``). This op serves **float32 alone**, and
the reason is worth pinning because it inverts the obvious reading -- two bodies
that build the same node, with different dtype gates:

* **float16 is declined for the platform's subnormal flush, not the node's
  arithmetic.** A ``silu`` tail is ``x * sigmoid(x)`` where both factors are
  subnormal, so the flush this SDK applies to a subnormal *result* produces the
  zero the reference produces too, and the two agree bit for bit. A bare
  ``sigmoid``'s left tail has no such partner: the graph flushes 151465 of
  400001 samples over [-40, 40] to zero where the reference keeps them, which is
  3.088e-02 max abs and 6.2e-02 relative -- a wrong answer.
* **bfloat16 is declined for rounding at a width that holds about three decimal
  digits.** Nothing there is subnormal (bfloat16's exponent range is float32's),
  so the 8.289e-03 absolute difference over the same operand is plain rounding
  -- and still a difference this backend declines to claim, by the same rule
  that keeps ``softmax`` off bfloat16.

So the tests below hold three things the number alone cannot: that float32 is
served and agrees with ``at::sigmoid``, that the two narrow widths are declined
and take the round-trip *correctly and at their own dtype*, and that the graph
is not silently answering the declined calls.

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

# The one width the graph serves exactly, and the three it declines. float64 has
# no MPSGraph element type at all; float16 and bfloat16 are the measured
# declines documented in the module header.
SERVED_DTYPES = [torch.float32]

# The widths the reference has a kernel for and this body declines. The
# round-trip is what answers them, so the assertion is CPU equality at the
# operand's own dtype rather than an error.
DECLINED_FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float64]

TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _reference(host: torch.Tensor) -> torch.Tensor:
    """``at::sigmoid`` on the host, in the operand's own dtype.

    The op and not ``1 / (1 + exp(-x))``: the two readings differ on the
    saturated tails, which is where a composition would be caught, and the
    reference is what the kernel is held to.
    """
    return torch.ops.aten.sigmoid(host)


def _assert_matches_cpu(out: torch.Tensor, expected: torch.Tensor) -> None:
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    torch.testing.assert_close(out.cpu().float(), expected.float(), **TOLERANCE)


class TestSigmoidCorrectness:
    """``torch.sigmoid`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("dtype", SERVED_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype):
        host = (torch.randn(4, 64) * 3.0).to(dtype)
        _assert_matches_cpu(torch.sigmoid(_to_device(host)), _reference(host))

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
        _assert_matches_cpu(torch.sigmoid(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_the_saturated_tails_are_the_widths_own(self):
        """Both ends of the range, where a weaker sigmoid shows as a number.

        A ``1 / (1 + exp(-x))`` composition overflows to ``inf`` at the negative
        end and answers 0 where the reference answers a tiny positive. The graph
        does not: it saturates the positive tail to exactly 1 and keeps the
        negative tail a tiny positive, which is what is held here. The values are
        held to the served width's own rounding rather than bit-exactly, because
        the graph and the reference round differently by up to one ulp in the
        middle of the range (measured: 1 ulp at x = -20) -- that is the ordinary
        float32 difference the gate already accepts, and the separate cases below
        are what hold saturation and the subnormal band exactly.
        """
        host = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0, 88.0])
        out = torch.sigmoid(_to_device(host))
        torch.testing.assert_close(out.cpu(), _reference(host), **TOLERANCE)
        assert out.cpu()[4].item() == 1.0 and out.cpu()[5].item() == 1.0
        assert out.cpu()[0].item() > 0.0

    @pytest.mark.anyplatform
    def test_the_negative_tail_never_answers_the_compositions_zero(self):
        """The failure a ``1 / (1 + exp(-x))`` composition would show.

        Measured, the graph's tail agrees with the reference down to the smallest
        normal float and never produces the composition's ``inf``/``0`` at a
        finite input where the reference is normal.
        """
        host = torch.tensor([-87.0, -60.0, -40.0])
        out = torch.sigmoid(_to_device(host)).cpu()
        expected = _reference(host)
        assert torch.equal(out, expected)
        assert torch.isfinite(out).all()
        assert (out > 0).all()

    @pytest.mark.anyplatform
    def test_the_subnormal_band_is_where_the_platform_flushes(self):
        """The one measured divergence, pinned so it cannot widen unnoticed.

        This SDK flushes a subnormal *result* to zero, which is the same policy
        the ``div`` body records. For a float32 sigmoid the flush is confined to
        the input band where the reference's own answer is subnormal but nonzero:
        measured, ``sigmoid(-87.0)`` agrees at 1.6458e-38, ``sigmoid(-88.0)`` is
        6.0546e-39 on the reference and exactly 0 on the graph, and by -89.0 the
        reference is zero as well.

        Where the band starts is the *hardware's* decision, not this body's, so
        the case asserts the shape of the policy rather than a boundary: every
        input whose reference answer is normal agrees exactly, and the flush
        never reaches one. Measured, the first flushed input differs by
        generation -- ``-88.0`` on an M5, ``-87.5`` on CI's paravirtualized
        part -- and any point boundary here would be a test that fails on the
        other machine. What would be a regression is the flush moving into the
        normal range, and that is what is held.
        """
        host = torch.tensor([-86.0, -87.0, -87.5, -88.0, -88.5, -89.0, -100.0])
        out = torch.sigmoid(_to_device(host)).cpu()
        expected = _reference(host)
        tiny = torch.finfo(torch.float32).tiny
        for x, got, want in zip(host.tolist(), out.tolist(), expected.tolist()):
            if want == 0.0:
                # Below the band the reference is zero too, so the flush is
                # indistinguishable from the answer.
                assert got == 0.0, (
                    f"sigmoid({x}) = {got!r}, expected the reference's own 0"
                )
            elif want < tiny:
                # Inside the band the reference is subnormal-and-nonzero. The
                # graph may answer it or flush it, and both are this policy;
                # what it may not do is answer something else.
                assert got == 0.0 or got == want or abs(got - want) <= tiny, (
                    f"sigmoid({x}) = {got!r} against the reference's {want!r}"
                )
            else:
                # Inside the normal range the two agree.
                assert got == want, f"sigmoid({x}) = {got!r}, expected {want!r}"
                assert got > 0.0
        # The flush is confined to the subnormal band: the smallest normal
        # input the reference answers exactly, which is the invariant that
        # would break if the flush widened.
        assert expected[1].item() >= tiny and out[1].item() > 0.0

    @pytest.mark.anyplatform
    def test_zero_and_signed_zero(self):
        """``sigmoid(0)`` is exactly 0.5, which the reference also returns."""
        host = torch.tensor([0.0, -0.0])
        out = torch.sigmoid(_to_device(host))
        torch.testing.assert_close(out.cpu(), _reference(host), rtol=0, atol=0)
        assert out.cpu()[0].item() == 0.5

    @pytest.mark.anyplatform
    def test_the_extremes_are_the_widths_own(self):
        host = torch.tensor([30.0, 40.0, -40.0])
        out = torch.sigmoid(_to_device(host))
        assert out.cpu()[1].item() == 1.0  # saturates to the reference's 1
        assert out.cpu()[2].item() > 0.0  # never the composition's 0
        torch.testing.assert_close(out.cpu(), _reference(host), rtol=0, atol=0)

    @pytest.mark.anyplatform
    def test_inf_and_nan(self):
        host = torch.tensor([float("inf"), float("-inf"), float("nan")])
        out = torch.sigmoid(_to_device(host)).cpu()
        expected = _reference(host)
        assert out[0].item() == 1.0
        assert out[1].item() == 0.0
        assert torch.isnan(out[2])
        assert torch.equal(out[:2], expected[:2])

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        host = torch.empty(0)
        out = torch.sigmoid(_to_device(host))
        assert out.shape == (0,)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        host = torch.tensor(1.25)
        _assert_matches_cpu(torch.sigmoid(_to_device(host)), _reference(host))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.randn(8, 8).t()
        assert not host.is_contiguous()
        _assert_matches_cpu(torch.sigmoid(_to_device(host)), _reference(host))


class TestSigmoidDtypeGate:
    """The narrow widths are declined, and the round-trip answers them exactly.

    This is the increment's finding rather than a failure path: the body shares
    a node with ``silu`` and still declines what ``silu`` serves, so a suite that
    only checked the served width would not distinguish this op from one that
    silently followed the sibling's gate.
    """

    @pytest.mark.parametrize("dtype", DECLINED_FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_declined_widths_round_trip_at_their_own_dtype(self, dtype):
        host = (torch.randn(4, 32) * 3.0).to(dtype)
        out = torch.sigmoid(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        # A graph at the wrong width would come back float32, so the dtype is
        # the assertion that the round-trip is what answered.
        assert out.cpu().dtype == dtype

    @pytest.mark.anyplatform
    def test_float16_tail_is_declined_rather_than_flushed(self):
        """The failure the float16 decline exists for, held as a case.

        Served through the graph, the left tail of a float16 sigmoid comes back
        as zeros where the reference keeps subnormals -- 3.088e-02 absolute over
        [-40, 40], measured on an M5. The case asserts the reference's own
        nonzero tail, so a body that started serving float16 fails here.
        """
        host = torch.linspace(-20.0, -8.0, 257, dtype=torch.float16)
        out = torch.sigmoid(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        assert int((out.cpu() > 0).sum()) == int((_reference(host) > 0).sum())

    @pytest.mark.anyplatform
    def test_bfloat16_matches_the_reference_bit_for_bit(self):
        """The round-trip's answer is the reference, not a 1e-2 approximation.

        Served through the graph, bfloat16 differed from the reference by
        8.289e-03 -- enough to fail a bit-exact comparison, which is the point
        of measuring the decline this way rather than with a tolerance.
        """
        host = torch.linspace(-8.0, 8.0, 513).to(torch.bfloat16)
        out = torch.sigmoid(_to_device(host))
        assert torch.equal(out.cpu(), _reference(host))

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8]
    )
    @pytest.mark.anyplatform
    def test_integral_operand_promotes_exactly_like_cpu(self, dtype):
        """An integral operand is promoted to float32, and the body declines it.

        ``at::sigmoid`` has no integral kernel and does not raise: measured, it
        returns float32. The body declines the integral width, so the round-trip
        is what performs the promotion -- and the assertion is the promoted dtype
        and the values, because a graph that had served it would have had to
        allocate from the operand's own options.
        """
        host = torch.arange(-2, 3).to(dtype)
        out = torch.sigmoid(_to_device(host))
        expected = _reference(host)
        assert expected.dtype == torch.float32
        _assert_matches_cpu(out, expected)

    @pytest.mark.anyplatform
    def test_bool_operand_promotes_exactly_like_cpu(self):
        host = torch.tensor([True, False, True])
        out = torch.sigmoid(_to_device(host))
        _assert_matches_cpu(out, _reference(host))
        assert _reference(host).dtype == torch.float32


class TestSigmoidPlanReuse:
    """A plan is per (dtype, shape), and interleaving them shares nothing."""

    @pytest.mark.anyplatform
    def test_interleaved_signatures_never_share_a_plan(self):
        cases = [
            (torch.randn(4, 64) * 3.0, torch.randn(4, 64) * 3.0),
            (torch.randn(64, 4) * 3.0, torch.randn(64, 4) * 3.0),
            (torch.randn(4, 64) * 3.0, torch.randn(4, 64).abs() + 4.0),
            (
                torch.randn(
                    4096,
                ),
                torch.randn(
                    4096,
                ),
            ),
        ]
        held = []
        for host, other in cases:
            out = torch.sigmoid(_to_device(host))
            held.append((out.cpu(), _reference(host), other))
        for got, wanted, _ in held:
            torch.testing.assert_close(got, wanted, **TOLERANCE)

    @pytest.mark.anyplatform
    def test_repeated_calls_reuse_one_plan_and_stay_correct(self):
        host = torch.randn(8, 16) * 2.0
        wanted = _reference(host)
        for _ in range(8):
            _assert_matches_cpu(torch.sigmoid(_to_device(host)), wanted)


class TestSigmoidDispatch:
    """``sigmoid`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_sigmoid_to_mps(self):
        assert routed_backend_or_none("sigmoid") == "mps"

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernel ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or ``sigmoid`` reached
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
                "a = torch.randn(4, 4, device='flagos:0'); torch.sigmoid(a)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] sigmoid -> apple" in result.stderr

    @pytest.mark.main_ops
    def test_a_declined_width_stays_served(self):
        """A declined width is the body's round-trip, not a `none` route.

        The conf route is what the log names, so a float16 call still logs
        ``sigmoid -> apple`` even though the body hands it to the reference. This
        holds that the decline is inside the body, and that the conf has not
        been narrowed to the served width.
        """
        source = (
            "import torch, torch_fl; "
            "a = torch.randn(4, 4, device='flagos:0').half(); "
            "torch.sigmoid(a); "
            "torch_fl._C._synchronize()"
        )
        assert dispatched_to_mps(source) == 1
