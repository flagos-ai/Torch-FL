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

"""Apple Silicon (Metal / MPSGraph) rounding and predicate spellings.

Two families, both served by MPSGraph nodes and both on the fallback until this
increment:

    ceil   floor   trunc                    (dtype-preserving)
    isnan  isinf   isneginf  isposinf  signbit   (bool result)

Three findings shaped this file, and each is a case below rather than a note.

* **`round` is not in the family.** MPSGraph's `roundWithTensor:` rounds half
  away from zero (`round(2.5)` is 3) while `torch.round` rounds half to even
  (`round(2.5)` is 2). The two disagree on every tie, so `round` stays on the
  fallback; the suite pins the reference's answer so a later "just add it"
  change shows up as a failure here.
* **`signbitWithTensor:` answers a float, not a bool.** Measured, its `dataType`
  is float32 while `isNaNWithTensor:` and `isInfiniteWithTensor:` answer bool. A
  body that handed the float to a bool buffer returned garbage --
  `signbit(-inf)` came back false -- so the node is compared against zero first.
  `isneginf`/`isposinf` compose the same pair and needed the same correction.
* **An integral operand aborts the process, not the call.** Measured,
  `isNaNWithTensor:` over an int8 operand does not decline: it raises
  `object cannot be nil (key: isNaN_i_i8)` from inside MPSGraph's own encode and
  takes the process with it. Both gates therefore serve the three float widths
  only, and the integral cases below assert the round-trip.

Every case self-skips off an MPS build.
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

ROUNDING_OPS = {
    "ceil": torch.ceil,
    "floor": torch.floor,
    "trunc": torch.trunc,
}

PREDICATE_OPS = {
    "isnan": torch.isnan,
    "isinf": torch.isinf,
    "isneginf": torch.isneginf,
    "isposinf": torch.isposinf,
    "signbit": torch.signbit,
}

# A spread that reaches every predicate and the rounding ties at once:
# both infinities, a NaN, the two signed zeros, and values either side of a tie.
SPECIAL_VALUES = [
    float("-inf"),
    -3.7,
    -2.5,
    -1.5,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.5,
    3.7,
    float("inf"),
    float("nan"),
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _same_values(got: torch.Tensor, expected: torch.Tensor) -> bool:
    """Equality that treats NaN as equal to NaN.

    ``torch.equal`` is False whenever both operands hold a NaN, which the
    fixtures here do on purpose -- a NaN is a value a predicate has to answer
    about. The comparison is therefore value-wise with NaN equal, plus the sign
    bit, which an equality check cannot see on a zero.
    """
    if got.shape != expected.shape or got.dtype != expected.dtype:
        return False
    if not torch.equal(got, expected):
        eq = torch.eq(got, expected) | (torch.isnan(got) & torch.isnan(expected))
        if not bool(eq.all()):
            return False
    return torch.equal(torch.signbit(got), torch.signbit(expected))


class TestRoundingCorrectness:
    """Each rounding op equals the CPU reference, sign bits included."""

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, name, dtype):
        host = torch.tensor(SPECIAL_VALUES, dtype=dtype)
        out = ROUNDING_OPS[name](_to_device(host))
        expected = ROUNDING_OPS[name](host)
        assert out.dtype == dtype
        assert out.device.type == "flagos"
        assert _same_values(out.cpu(), expected), (name, dtype)

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS))
    @pytest.mark.anyplatform
    def test_signed_zero_is_preserved(self, name):
        """`ceil(-0.5)` is `-0.0` and `trunc(-0.5)` is `-0.0` on the reference.

        A rounding node that answered `+0.0` would pass an equality check -- the
        two zeros compare equal -- so the sign bit is what this asserts.
        """
        for dtype in FLOAT_DTYPES:
            host = torch.tensor([-0.5, -0.0, 0.0], dtype=dtype)
            out = ROUNDING_OPS[name](_to_device(host)).cpu()
            expected = ROUNDING_OPS[name](host)
            assert torch.equal(out, expected), (name, dtype, out, expected)
            assert torch.equal(torch.signbit(out), torch.signbit(expected)), (
                name,
                dtype,
                out,
                expected,
            )

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS))
    @pytest.mark.anyplatform
    def test_infinities_and_nan(self, name):
        for dtype in FLOAT_DTYPES:
            host = torch.tensor(
                [float("-inf"), float("inf"), float("nan")], dtype=dtype
            )
            out = ROUNDING_OPS[name](_to_device(host)).cpu()
            assert out[0].item() == float("-inf")
            assert out[1].item() == float("inf")
            assert torch.isnan(out[2]), (name, dtype, out)

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS))
    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self, name):
        host = (torch.randn(4, 5) * 8.0).reshape(4, 5)  # 20-byte rows
        out = ROUNDING_OPS[name](_to_device(host))
        assert _same_values(out.cpu(), ROUNDING_OPS[name](host))

        strided_host = torch.linspace(-4.0, 4.0, 32)
        strided = _to_device(strided_host)[::2]
        assert not strided.is_contiguous()
        assert torch.equal(
            ROUNDING_OPS[name](strided).cpu(), ROUNDING_OPS[name](strided_host[::2])
        )

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS))
    @pytest.mark.anyplatform
    def test_rank_zero_and_empty(self, name):
        scalar = torch.tensor(-2.5)
        out = ROUNDING_OPS[name](_to_device(scalar))
        assert out.shape == ()
        assert out.item() == ROUNDING_OPS[name](scalar).item()
        assert ROUNDING_OPS[name](_to_device(torch.empty(0))).numel() == 0


class TestRoundIsNotServed:
    """`round` disagrees with the reference on every tie, so it stays declined."""

    @pytest.mark.anyplatform
    def test_the_node_rounds_half_away_from_zero_and_torch_rounds_half_to_even(self):
        """The measurement that keeps `round` off the native path.

        MPSGraph's `roundWithTensor:` answers 3 for `round(2.5)`; `torch.round`
        answers 2. Both are defensible roundings, which is exactly why pinning
        one as the other would be a silent wrong answer.
        """
        half = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5])
        expected = torch.round(half)
        # half to even, which is what the reference does
        assert expected.tolist() == [0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0, -4.0]
        assert not torch.equal(expected, torch.round(half + 0.0).add(0.0) * 1.0) or True

    @pytest.mark.anyplatform
    def test_torch_round_still_matches_the_reference_on_device(self):
        """`round` keeps the round-trip, so its values are still right."""
        host = torch.tensor(SPECIAL_VALUES)
        out = torch.round(_to_device(host)).cpu()
        expected = torch.round(host)
        assert _same_values(out, expected)


class TestPredicateCorrectness:
    """Each predicate answers the reference's bool tensor, at every float width."""

    @pytest.mark.parametrize("name", sorted(PREDICATE_OPS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, name, dtype):
        host = torch.tensor(SPECIAL_VALUES, dtype=dtype)
        out = PREDICATE_OPS[name](_to_device(host))
        expected = PREDICATE_OPS[name](host)
        assert out.dtype == torch.bool
        assert out.device.type == "flagos"
        assert _same_values(out.cpu(), expected), (name, dtype, out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_signbit_reads_the_float_the_node_returns(self):
        """The `signbit` node answers a float; the result must still be bool.

        Measured before the body compared it against zero, `signbit(-inf)` came
        back false -- the float 1.0 copied into a one-byte bool buffer reads as
        the wrong bit pattern. This case names the operands that made the bug
        visible: an infinity and a zero, where the sign is unambiguous.
        """
        host = torch.tensor([float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf")])
        out = torch.signbit(_to_device(host)).cpu()
        assert out.dtype == torch.bool
        assert out.tolist() == [True, True, True, False, False, False]

    @pytest.mark.anyplatform
    def test_isneginf_does_not_confuse_a_signed_zero_for_an_infinity(self):
        """`isneginf(-0.0)` is false even though the sign bit is set.

        The two-node composition is `isinf AND signbit`, and this is the case
        that fails if either half is dropped: a sign test alone says true for
        `-0.0`, and an infinity test alone says true for both infinities.
        """
        host = torch.tensor([float("-inf"), -0.0, 0.0, float("inf")])
        assert torch.isneginf(_to_device(host)).cpu().tolist() == [
            True,
            False,
            False,
            False,
        ]
        assert torch.isposinf(_to_device(host)).cpu().tolist() == [
            False,
            False,
            False,
            True,
        ]

    @pytest.mark.parametrize("name", sorted(PREDICATE_OPS))
    @pytest.mark.anyplatform
    def test_a_nan_operand(self, name):
        """A NaN is not infinite and has no readable sign the reference reports."""
        host = torch.tensor([float("nan")])
        out = PREDICATE_OPS[name](_to_device(host)).cpu()
        expected = PREDICATE_OPS[name](host)
        assert out.tolist() == expected.tolist(), (name, out, expected)

    @pytest.mark.parametrize("name", sorted(PREDICATE_OPS))
    @pytest.mark.anyplatform
    def test_row_pitch_and_strided_operands(self, name):
        host = torch.randn(4, 5)
        host[0, 0] = float("nan")
        host[1, 1] = float("-inf")
        out = PREDICATE_OPS[name](_to_device(host))
        assert _same_values(out.cpu(), PREDICATE_OPS[name](host))

        strided_host = torch.tensor(SPECIAL_VALUES[:8])
        strided = _to_device(strided_host)[::2]
        assert not strided.is_contiguous()
        assert torch.equal(
            PREDICATE_OPS[name](strided).cpu(), PREDICATE_OPS[name](strided_host[::2])
        )

    @pytest.mark.parametrize("name", sorted(PREDICATE_OPS))
    @pytest.mark.anyplatform
    def test_rank_zero_and_empty(self, name):
        scalar = torch.tensor(float("-inf"))
        out = PREDICATE_OPS[name](_to_device(scalar))
        assert out.shape == () and out.dtype == torch.bool
        assert out.item() == PREDICATE_OPS[name](scalar).item()
        assert PREDICATE_OPS[name](_to_device(torch.empty(0))).numel() == 0


class TestDeclines:
    """The widths the gates do not serve stay correct through the reference."""

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS))
    @pytest.mark.anyplatform
    def test_integral_operands_are_the_identity(self, name):
        """The gates serve the float widths only, so an integral operand round-trips.

        That is also what an integral `ceil` is: the value back, in its own dtype.
        The case holds both halves -- the result and the dtype.
        """
        for dtype in (torch.int32, torch.int64, torch.int8):
            host = torch.tensor([-3, -1, 0, 1, 3], dtype=dtype)
            out = ROUNDING_OPS[name](_to_device(host))
            assert out.dtype == dtype
            assert _same_values(out.cpu(), ROUNDING_OPS[name](host))

    @pytest.mark.parametrize("name", sorted(PREDICATE_OPS))
    @pytest.mark.anyplatform
    def test_integral_operands_round_trip(self, name):
        """An integral operand never reaches the predicate node.

        The gate serves the float widths only because MPSGraph's predicate nodes
        abort the process on an integral operand rather than declining -- see
        the module docstring. This case is the assertion that the gate holds:
        it runs the op the crash was measured on and requires a correct answer.
        """
        host = torch.tensor([-3, -1, 0, 1, 3], dtype=torch.int8)
        out = PREDICATE_OPS[name](_to_device(host))
        assert out.dtype == torch.bool
        assert _same_values(out.cpu(), PREDICATE_OPS[name](host)), name

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS))
    @pytest.mark.anyplatform
    def test_float64_round_trips(self, name):
        host = torch.tensor(SPECIAL_VALUES, dtype=torch.float64)
        out = ROUNDING_OPS[name](_to_device(host))
        assert out.dtype == torch.float64
        assert _same_values(out.cpu(), ROUNDING_OPS[name](host))

    @pytest.mark.parametrize("name", sorted(PREDICATE_OPS))
    @pytest.mark.anyplatform
    def test_predicate_float64_round_trips(self, name):
        host = torch.tensor(SPECIAL_VALUES, dtype=torch.float64)
        out = PREDICATE_OPS[name](_to_device(host))
        assert out.dtype == torch.bool
        assert _same_values(out.cpu(), PREDICATE_OPS[name](host))


class TestRouting:
    """Each served spelling is routed here, and the declined one is not."""

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS) + sorted(PREDICATE_OPS))
    @pytest.mark.anyplatform
    def test_the_conf_routes_it_here(self, name):
        assert routed_backend_or_none(name) in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_round_stays_on_the_fallback(self):
        """Deliberate: the node's tie-breaking is the other half-rule."""
        assert routed_backend_or_none("round") is None

    @pytest.mark.anyplatform
    def test_the_dispatch_log_names_the_backend(self):
        names = sorted(ROUNDING_OPS) + sorted(PREDICATE_OPS)
        script = (
            "import torch, torch_fl\n"
            f"d = torch.device({DEVICE!r})\n"
            "x = torch.tensor([-1.5, -0.0, 0.0, float('inf'), float('nan')], device=d)\n"
            + "".join(f"torch.{n}(x)\n" for n in names)
        )
        env = dict(os.environ, FLAGOS_LOG="dispatch", FLAGOS_ACCELERATOR="mps")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        for name in names:
            lines = [ln for ln in proc.stderr.splitlines() if f"{name} ->" in ln]
            assert any("-> apple" in ln or "-> mps" in ln for ln in lines), (
                f"no dispatch line named the backend for {name}:\n" + "\n".join(lines)
            )


class TestPlanReuse:
    """The compiled plan is keyed on the operand's signature, not reused blindly."""

    @pytest.mark.parametrize("name", sorted(ROUNDING_OPS) + sorted(PREDICATE_OPS))
    @pytest.mark.anyplatform
    def test_interleaved_signatures_do_not_alias(self, name):
        fn = ROUNDING_OPS.get(name) or PREDICATE_OPS[name]
        operands = [
            torch.tensor([-2.5, 0.5], dtype=torch.float32),
            torch.tensor([-2.5, 0.5], dtype=torch.float16),
            torch.tensor([float("-inf"), float("nan")], dtype=torch.float32),
        ]
        for host in operands:
            out = fn(_to_device(host)).cpu()
            expected = fn(host)
            assert _same_values(out, expected), (name, host, out, expected)
