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

"""Stage-2 Apple Silicon (Metal / MPSGraph) reduction kernels.

Stage 2 of the MPSGraph operator layer (see
docs/superpowers/specs/2026-10-14-apple-mps-native-backend-design.md section 7)
adds the ``reduce`` category -- ``sum.dim_IntList`` and ``mean.dim`` -- and this
is where the category's exit criterion is measured, the way
tests/integration/ops/test_mps_unary.py measures the ``unary`` one: every case
compares the flagos result against the CPU reference, and a dispatch-log
assertion proves the ``mps`` kernel ran rather than ``cpu_fallback``.

Four things are specific to this category and are what the cases below pin down.

* **The axes list is aten's, normalised here.** ``sum(x)``, ``sum(x, dim=None)``,
  ``sum(x, [])`` and ``sum(x, ())`` are all the full reduction, a negative
  position resolves against the rank, and the two inputs aten rejects -- a
  repeated axis and one outside the rank -- are declined to the CPU round-trip
  so the reference raises its own error rather than a graph reducing twice.
* **``mean`` has no MPSGraph selector.** It is the sum over the reduced axes
  divided by how many elements those axes covered, as a one-element constant the
  graph broadcasts, which is why the ``keepdim`` and multi-axis cases are worth
  running for it too.
* **The dtype gate is narrower than the binary one.** Only the float widths, and
  only with no ``dtype=`` that moves the result off the operand's dtype: aten
  promotes every integer sum to int64 and ``mean`` of an integer tensor has no
  result dtype without an explicit one, so both round-trip and keep the
  reference's dtype.
* **An empty reduction is not an empty tensor.** MPSGraph has no empty tensor at
  all, so an empty operand round-trips; the arithmetic it needs is aten's (a sum
  over an empty axis is zeros, a mean is NaN), and the reference already has it.

Every case self-skips off an MPS build, like the stage-1 suite.
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

# f32 is bit-identical to the reference; the two half widths accumulate in f32
# inside MPSGraph the way the CPU kernel does but round at different points, so
# they are compared in f32 at a tolerance a few orders of magnitude above the
# 8e-6 that was measured for a 64x64 reduction.
LOW_PRECISION_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}

DIM_CASES = [
    (0, False),
    (1, False),
    (1, True),
    (-1, False),
    (-2, True),
    ([0, 1], False),
    ([1, 0], False),  # unsorted: the kernel sorts before it builds axes
    ([-1, -2], True),
    ([0], True),
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another
    # case left behind.
    return t.clone().to(DEVICE)


def _assert_matches_cpu(out, expected, dtype):
    if dtype == torch.float32:
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(
            out.cpu().float(), expected.float(), **LOW_PRECISION_TOLERANCE
        )


class TestSumCorrectness:
    """``sum.dim_IntList`` on the flagos device equals the CPU reference."""

    @pytest.mark.anyplatform
    def test_full_reduction_spellings_are_all_the_full_reduction(self):
        """``sum(x)``, ``dim=None`` and an empty dim list are one reduction.

        aten defines all four spellings as "every axis", and the kernel
        normalises them to the same axes list; a nullopt/empty dims list is the
        only place where the two sides could disagree about what was asked.
        """
        host = torch.randn(3, 4, 5)
        a = _to_device(host)
        expected = torch.sum(host)
        for out in (
            torch.sum(a),
            torch.sum(a, dim=None),
            torch.sum(a, dim=[]),
            torch.sum(a, dim=()),
        ):
            assert out.shape == expected.shape
            assert out.device.type == "flagos"
            torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("dim,keepdim", DIM_CASES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_dim_keepdim(self, dim, keepdim):
        """Positive, negative, single, multiple and unsorted axis lists.

        The reference is asked the same question, so the case is really about
        the kernel producing the same shape (``keepdim`` and the dropped axes
        both change it) and the same numbers for axes named in either order.
        """
        host = torch.randn(3, 4, 5)
        out = torch.sum(_to_device(host), dim=dim, keepdim=keepdim)
        expected = torch.sum(host, dim=dim, keepdim=keepdim)
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, dtype):
        host = torch.randn(16, 8, dtype=dtype) * 3.0
        a = _to_device(host)
        for dim in (None, 0, 1, [0, 1]):
            _assert_matches_cpu(torch.sum(a, dim=dim), torch.sum(host, dim=dim), dtype)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 5),  # 20-byte rows
            (3, 7),
            (5, 4),  # controls: rows already a multiple of 16 bytes
            (4, 8, 16),
            (2, 3, 5),
            (128, 3),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu_at_row_widths_that_are_not_a_16_byte_multiple(self, shape):
        """The shapes MPSNDArray's row pitch would silently corrupt.

        See the row-pitch note in ``mps_common.h``: a reduction reads every
        element of a tensor whose innermost row need not be 16-byte aligned, and
        an aten-shaped descriptor would take every row after the first from the
        wrong offset. Both a full reduction and an axis reduction are checked,
        because they flatten the operand differently. The aligned shapes in the
        list are the controls that show the padding was the variable.
        """
        host = torch.randn(*shape) * 3.0
        a = _to_device(host)
        torch.testing.assert_close(torch.sum(a).cpu(), torch.sum(host))
        dim = host.dim() - 1
        torch.testing.assert_close(
            torch.sum(a, dim=dim).cpu(), torch.sum(host, dim=dim)
        )

    @pytest.mark.anyplatform
    def test_rank_zero_and_single_element_operands(self):
        """There is nothing to reduce, and that is not a special case.

        A rank-0 operand gets an empty axes list, which is MPSGraph's no-op
        reduction -- the right answer, since a rank-0 tensor is already its own
        sum. A 1-element tensor is the same arithmetic with a shape to keep.
        """
        scalar = torch.tensor(3.5)
        out = torch.sum(_to_device(scalar))
        assert out.shape == ()
        torch.testing.assert_close(out.cpu(), torch.sum(scalar))

        single = torch.tensor([[7.0]])
        torch.testing.assert_close(
            torch.sum(_to_device(single), dim=0).cpu(), torch.sum(single, dim=0)
        )
        torch.testing.assert_close(
            torch.sum(_to_device(single), dim=0, keepdim=True).cpu(),
            torch.sum(single, dim=0, keepdim=True),
        )

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        """A non-contiguous operand takes the same answer as the CPU.

        MPSGraph reads dense memory, so the kernel materialises the operand
        first; the axis reduction is the case that would read the strided buffer
        as if it were dense.
        """
        host = torch.randn(4, 8)
        operand = _to_device(host)[:, ::2]
        assert not operand.is_contiguous()
        torch.testing.assert_close(
            torch.sum(operand, dim=1).cpu(), torch.sum(host[:, ::2], dim=1)
        )

    @pytest.mark.anyplatform
    def test_empty_operand_matches_cpu(self):
        """The arithmetic over nothing is aten's, and it round-trips.

        MPSGraph rejects an empty tensor, so an empty operand is declined; the
        sum over an empty axis is zeros and the full sum of nothing is 0, both
        of which the reference computes.
        """
        host = torch.empty(0, 4)
        a = _to_device(host)
        torch.testing.assert_close(torch.sum(a).cpu(), torch.sum(host))
        torch.testing.assert_close(torch.sum(a, dim=0).cpu(), torch.sum(host, dim=0))
        torch.testing.assert_close(torch.sum(a, dim=1).cpu(), torch.sum(host, dim=1))

    @pytest.mark.parametrize(
        "dtype", [torch.int8, torch.int32, torch.int64, torch.bool]
    )
    @pytest.mark.anyplatform
    def test_integer_and_bool_operands_take_the_int64_promotion(self, dtype):
        """aten promotes every integer sum to int64; the graph would not.

        MPSGraph's reduction keeps the operand's element type, so an integer sum
        is declined and the reference produces the int64 result. The dtype is
        the part a graph of the operand's own element type would get wrong.
        """
        host = torch.tensor([[True, False, True], [False, False, True]])
        if dtype != torch.bool:
            host = host.to(dtype)
        out = torch.sum(_to_device(host), dim=1)
        expected = torch.sum(host, dim=1)
        assert out.dtype == torch.int64 == expected.dtype
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_float64_operand_round_trips(self):
        """float64 has no MPSGraph element type; the value is still right."""
        host = torch.tensor([1.5, -2.25, 4.0], dtype=torch.float64)
        out = torch.sum(_to_device(host))
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), torch.sum(host))

    @pytest.mark.anyplatform
    def test_explicit_dtype_that_moves_off_the_operand_round_trips(self):
        """``dtype=`` widens the result, so the kernel declines.

        A half operand summed into float32 is the common mixed-precision call;
        the element type of the graph's result is the operand's, so the kernel
        takes the round-trip and the reference's dtype is what comes back.
        """
        host = torch.randn(8, dtype=torch.float16) * 3.0
        out = torch.sum(_to_device(host), dtype=torch.float32)
        expected = torch.sum(host, dtype=torch.float32)
        assert out.dtype == torch.float32 == expected.dtype
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.anyplatform
    def test_repeated_axis_raises_exactly_like_cpu(self):
        """``dim=[0, 0]`` is declined so the reference raises its own error."""
        a = _to_device(torch.randn(2, 3))
        with pytest.raises(RuntimeError, match="appears multiple times"):
            torch.sum(a, dim=[0, 0])

    @pytest.mark.anyplatform
    def test_out_of_range_axis_raises_exactly_like_cpu(self):
        """An axis outside the rank is declined, and aten reports it."""
        a = _to_device(torch.randn(2, 3))
        with pytest.raises(IndexError, match="Dimension out of range"):
            torch.sum(a, dim=5)


class TestMeanCorrectness:
    """``mean.dim`` -- a sum divided by the reduced element count."""

    @pytest.mark.parametrize("dim,keepdim", DIM_CASES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_dim_keepdim(self, dim, keepdim):
        """The divisor is the product of the reduced axes, not the tensor size.

        A single-axis or keepdim mean is where a divisor taken from the wrong
        extent would show: the axis reduction is the case that distinguishes
        ``size(axis)`` from ``numel()``.
        """
        host = torch.randn(3, 4, 5)
        out = torch.mean(_to_device(host), dim=dim, keepdim=keepdim)
        expected = torch.mean(host, dim=dim, keepdim=keepdim)
        assert out.shape == expected.shape
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_for_every_supported_dtype(self, dtype):
        host = torch.randn(16, 8, dtype=dtype) * 3.0
        a = _to_device(host)
        for dim in (None, 0, 1, [0, 1]):
            _assert_matches_cpu(
                torch.mean(a, dim=dim), torch.mean(host, dim=dim), dtype
            )

    @pytest.mark.anyplatform
    def test_single_element_and_rank_zero(self):
        """A mean of one element is that element, at any rank."""
        scalar = torch.tensor(2.5)
        torch.testing.assert_close(
            torch.mean(_to_device(scalar)).cpu(), torch.mean(scalar)
        )
        single = torch.tensor([[4.0]])
        torch.testing.assert_close(
            torch.mean(_to_device(single), dim=0).cpu(), torch.mean(single, dim=0)
        )

    @pytest.mark.anyplatform
    def test_strided_operand_is_materialised(self):
        host = torch.randn(4, 8)
        operand = _to_device(host)[:, ::2]
        assert not operand.is_contiguous()
        torch.testing.assert_close(
            torch.mean(operand, dim=1).cpu(), torch.mean(host[:, ::2], dim=1)
        )

    @pytest.mark.anyplatform
    def test_empty_operand_is_nan_like_cpu(self):
        """A mean over nothing is NaN, and the reference computes it.

        The empty operand is declined for the reason the empty sum is -- there
        is no MPSGraph tensor for zero elements -- but the answer is not the
        sum's: it has to be NaN along a reduced empty axis, and 0/0 in the graph
        would only be right by accident of what MPSGraph does with it.
        """
        host = torch.empty(0, 4)
        a = _to_device(host)
        out = torch.mean(a).cpu()
        assert torch.isnan(out) and torch.isnan(torch.mean(host))
        assert torch.isnan(torch.mean(a, dim=0).cpu()).all()

    @pytest.mark.anyplatform
    def test_integer_operand_without_dtype_raises_exactly_like_cpu(self):
        """``mean`` of an integer tensor has no result dtype in aten either."""
        a = _to_device(torch.tensor([1, 2, 3]))
        with pytest.raises(RuntimeError, match="could not infer output dtype"):
            torch.mean(a)

    @pytest.mark.anyplatform
    def test_integer_operand_with_an_explicit_dtype_round_trips(self):
        """With a float result dtype the reference computes the mean."""
        host = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = torch.mean(_to_device(host), dim=1, dtype=torch.float32)
        expected = torch.mean(host, dim=1, dtype=torch.float32)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out.cpu(), expected)


class TestReduceDispatch:
    """``sum`` and ``mean`` are claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.parametrize("op", ["sum.dim_IntList", "mean.dim"])
    @pytest.mark.main_ops
    def test_conf_routes_op_to_mps(self, op):
        assert routed_backend_or_none(op) == "mps"

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The dispatch log is the proof the kernels ran, not a fallback.

        A numeric-only check passes whether MPSGraph ran or the op reached
        ``cpu_fallback``, because both compute the same answer. The log line is
        emitted by the C++ dispatcher only on the ``mps`` slot. A full reduction
        and a dimmed one are both exercised, since aten spells them the same
        op. Run in a subprocess because ``LogEnabled`` latches the environment
        on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.randn(4, 4, device='flagos:0'); "
                "a.sum(); a.sum(dim=0); a.mean(); a.mean(dim=[0, 1])",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[flagos dispatch] sum.dim_IntList -> apple" in result.stderr
        assert "[flagos dispatch] mean.dim -> apple" in result.stderr
