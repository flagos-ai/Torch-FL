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

"""Stage-3 Apple Silicon (Metal / MPSGraph) ``pow`` with a host exponent.

``aten.pow.Tensor_Scalar`` -- a tensor raised to a Python or host scalar, which
is ``x ** 2`` in a model and the square a variance is built from. It is the
tensor form of the ``binary_scalar`` family: the exponent is a host value, so it
becomes a one-element constant the graph broadcasts rather than a second
operand.

Three properties are specific to this op.

* **One selector, not a composition.** ``powerWithPrimaryTensor:secondaryTensor:``
  is a real power rather than ``exp(log(x) * e)``, which is what keeps the
  reference's answers on the cases a composition gets wrong: a negative base to a
  fractional exponent is NaN, a negative base to an odd integer is negative, and
  a base of zero to a negative exponent is infinity. The cases below pin all
  three.
* **Which dtypes are served.** A float operand takes any real exponent; an
  integral one does not, because aten raises on a negative integer exponent and
  promotes to float on a fractional one. So the integral widths go to the
  round-trip, where both answers come from the reference in its own words, and
  bfloat16 goes with them for the reason the softmax gate declines it.
* **The exponent is part of the plan key.** ``x ** 2`` and ``x ** 3`` are two
  graphs over the same operand; a key that named only the op would let the first
  one serve the second.

Every case self-skips off an MPS build, like the stage-1 and stage-2 suites.
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

FLOAT_DTYPES = [torch.float32, torch.float16]

TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}

# float16 carries ~3 decimal digits, so a graph and a reference that each round
# once can differ by an ulp or two of half -- 5e-4 relative -- which is not a
# defect and is well past the float32 tolerance.
HALF_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _tolerance(dtype) -> dict:
    return TOLERANCE if dtype == torch.float32 else HALF_TOLERANCE


class TestPowCorrectness:
    """``pow(tensor, exponent)`` on the flagos device equals the CPU reference."""

    @pytest.mark.parametrize("exponent", [2.0, 3.0, 0.5, 2.5, -1.0, -2.0, 0.0, 1.0])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_exponent(self, exponent, dtype):
        host = torch.rand(4, 8, dtype=dtype) + 0.5  # a positive base throughout
        out = torch.pow(_to_device(host), exponent)
        assert out.dtype == dtype
        torch.testing.assert_close(
            out.cpu(), torch.pow(host, exponent), **_tolerance(dtype)
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_integer_exponent_matches_a_repeated_product(self, dtype):
        """``x ** 2`` against ``x * x``, so the case is checked outside softmax."""
        host = torch.rand(64, 8, dtype=dtype) + 0.1
        out = torch.pow(_to_device(host), 2.0)
        torch.testing.assert_close(out.cpu(), (host * host), **_tolerance(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_negative_base_to_a_fractional_exponent_is_nan(self, dtype):
        """The case a naive ``exp(log(x) * e)`` composition gets wrong.

        The reference produces NaN for a negative base raised to a fractional
        power, because ``log`` of a negative number is not real. A kernel built
        on ``exp``/``log`` would have to reproduce that on purpose; the real
        power selector does it by itself, and this is what holds the choice.
        """
        host = torch.tensor([-4.0, -1.0, 1.0, 4.0], dtype=dtype)
        out = torch.pow(_to_device(host), 0.5)
        reference = torch.pow(host, 0.5)
        assert torch.isnan(out.cpu()[:2]).all()
        assert torch.equal(torch.isnan(out.cpu()), torch.isnan(reference))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_negative_base_to_an_odd_integer_stays_negative(self, dtype):
        host = torch.tensor([-3.0, -2.0, 0.0, 2.0], dtype=dtype)
        out = torch.pow(_to_device(host), 3.0)
        torch.testing.assert_close(out.cpu(), torch.pow(host, 3.0), **_tolerance(dtype))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_zero_to_a_negative_exponent_is_infinity(self, dtype):
        host = torch.tensor([0.0, 1.0], dtype=dtype)
        out = torch.pow(_to_device(host), -1.0)
        assert torch.equal(out.cpu(), torch.pow(host, -1.0))

    @pytest.mark.parametrize(
        "shape",
        [(4, 5), (3, 7), (5, 4), (2, 3, 5), (4, 8, 16)],
    )
    @pytest.mark.anyplatform
    def test_rows_narrower_than_a_16_byte_multiple(self, shape):
        """The shapes MPSNDArray's row pitch would silently corrupt (see abs)."""
        host = torch.rand(*shape) + 0.5
        out = torch.pow(_to_device(host), 2.0)
        torch.testing.assert_close(out.cpu(), torch.pow(host, 2.0), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_exponent_one_short_circuits_to_the_operand(self):
        """``x ** 1`` is ``x``, NaN and signed zero included, on both sides.

        Compared with ``equal_nan`` because the NaN in the operand is the case a
        short-circuit has to preserve: ``NaN ** 1`` is NaN, not an error and not
        a number, and a kernel that built the graph anyway would have to agree.
        """
        host = torch.tensor([0.0, -0.0, 1.5, float("nan")])
        out = torch.pow(_to_device(host), 1.0)
        reference = torch.pow(host, 1.0)
        assert torch.allclose(out.cpu(), reference, equal_nan=True)
        assert torch.equal(torch.signbit(out.cpu()), torch.signbit(reference))

    @pytest.mark.anyplatform
    def test_exponent_zero_is_one_everywhere(self):
        """``x ** 0`` is 1 for every finite base."""
        host = torch.tensor([0.0, -0.0, 2.0, -3.0])
        out = torch.pow(_to_device(host), 0.0)
        assert torch.equal(out.cpu(), torch.ones(4))

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        """A non-contiguous operand takes the same answer as the CPU."""
        host = torch.rand(8, 10) + 0.5
        a = _to_device(host)[:, ::2]
        assert not a.is_contiguous()
        out = torch.pow(a, 2.0)
        torch.testing.assert_close(out.cpu(), torch.pow(host[:, ::2], 2.0), **TOLERANCE)

    @pytest.mark.anyplatform
    def test_empty_tensor(self):
        a = _to_device(torch.empty(0, 4))
        out = torch.pow(a, 2.0)
        assert out.shape == (0, 4)
        assert out.device.type == "flagos"


class TestPowDeclinedDtypes:
    """The widths the graph does not serve stay correct through the round-trip."""

    @pytest.mark.anyplatform
    def test_bfloat16_is_correct_despite_being_declined(self):
        """The gate declines bfloat16; the answer still has to be right."""
        host = torch.rand(4, 8, dtype=torch.bfloat16) + 0.5
        out = torch.pow(_to_device(host), 2.0)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out.cpu(), torch.pow(host, 2.0))

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.anyplatform
    def test_integral_base_promotes_or_raises_exactly_like_cpu(self, dtype):
        """An integral base is the reference's calls to make, not the graph's.

        ``int ** 2`` stays integral while ``int ** 0.5`` promotes to float and
        ``int ** -1`` raises -- three answers from one dtype that no single
        element type can express, so the kernel declines the operand entirely and
        the round-trip produces all three.
        """
        host = torch.tensor([1, 2, 3, 4], dtype=dtype)
        out = torch.pow(_to_device(host), 2)
        assert out.dtype == dtype
        assert torch.equal(out.cpu(), torch.pow(host, 2))

        promoted = torch.pow(_to_device(host), 0.5)
        assert promoted.dtype == torch.float32
        torch.testing.assert_close(promoted.cpu(), torch.pow(host, 0.5), **TOLERANCE)

        with pytest.raises(RuntimeError):
            torch.pow(_to_device(host), -1)

    @pytest.mark.anyplatform
    def test_float64_is_correct(self):
        host = torch.rand(4, 8, dtype=torch.float64) + 0.5
        out = torch.pow(_to_device(host), 2.5)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), torch.pow(host, 2.5), **TOLERANCE)


class TestPowScheduleAndPlanReuse:
    """The marshalling contract every MPS kernel shares (mps_common.mm)."""

    @pytest.mark.anyplatform
    def test_interleaved_exponents_and_dtypes_never_share_a_plan(self):
        """Each exponent and each dtype gets its own compiled graph.

        The exponent is a host value the body bakes in as a constant, so a plan
        key that named only the op would let the first graph built -- ``x ** 2``
        -- answer a later ``x ** 3``. Rotating through the signatures instead of
        grouping them puts a different exponent and dtype on a warm cache every
        step.
        """
        host = torch.rand(32, 8) + 0.5
        half = host.half()
        cases = [
            (half, 2.0),
            (half, 3.0),
            (host, 0.5),
            (host, 2.0),
            (half, -1.0),
            (host, 3.0),
            (half, 2.0),
        ]
        on_device = [_to_device(operand) for operand, _ in cases]
        wanted = [torch.pow(operand, exponent) for operand, exponent in cases]

        for round_ in range(3):
            for offset in range(len(cases)):
                index = (offset + round_) % len(cases)
                dtype = cases[index][0].dtype
                out = torch.pow(on_device[index], cases[index][1])
                assert out.dtype == wanted[index].dtype
                torch.testing.assert_close(
                    out.cpu(), wanted[index], **_tolerance(dtype)
                )

    @pytest.mark.anyplatform
    def test_results_survive_block_reuse_and_immediate_readback(self):
        """Nothing waits for the GPU, so both ends of the call are checked here.

        An operand's block goes back to the pool as soon as the caller drops it,
        while the graph reading it has only been scheduled; and the value the
        caller reads back is produced by work the runtime has not awaited. Both
        rest on the sentinel command buffer the runtime commits behind the graph
        (see mps_common.h), so a call that returned before its work was visible
        to StreamQuery would show up here as a wrong result.
        """
        held = []
        for _ in range(32):
            src = torch.rand(1 << 12, device=DEVICE) + 0.5
            wanted = torch.pow(src.cpu(), 2.0)
            out = torch.pow(src, 2.0)
            del src  # recycled while the graph above may still be reading it
            for _ in range(4):  # blocks the allocator is free to hand back out
                torch.pow(torch.rand(1 << 12, device=DEVICE) + 0.5, 2.0)
            held.append((out.cpu(), wanted))  # read back without a synchronize

        for got, wanted in held:
            torch.testing.assert_close(got, wanted, **TOLERANCE)


class TestPowDispatch:
    """``pow.Tensor_Scalar`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_pow_to_mps(self):
        assert routed_backend_or_none("pow.Tensor_Scalar") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_other_pow_spellings_on_the_fallback(self):
        """Tensor-tensor ``pow`` is a different op and has no kernel yet."""
        assert routed_backend_or_none("pow.Tensor_Tensor") is None
        assert routed_backend_or_none("pow.Scalar") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The log line is the proof the kernel ran, not a fallback."""
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.rand(4, 4, device='flagos:0'); torch.pow(a, 2.0)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] pow.Tensor_Scalar -> apple" in result.stderr
