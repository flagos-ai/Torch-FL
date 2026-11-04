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

"""Apple Silicon (Metal / MPSGraph) extrema and product reductions.

`sum.dim_IntList`, `mean.dim`, `all` and `argmax` were already served; `amax`,
`amin` and `prod.dim_int` were on `cpu_fallback`. They are three different
signatures over the same reduction nodes:

    amax / amin      (self, int[1] dim, bool keepdim)            -> int[1] list
    prod.dim_int     (self, int dim, bool keepdim, *, dtype?)    -> one dimension

The case that decided the implementation, and the one worth reading first:

**`amax`/`amin` must use the `PropagateNaN` selectors.** MPSGraph has two
families -- `reductionMaximumWithTensor:axes:name:` and
`reductionMaximumPropagateNaNWithTensor:axes:name:` -- and the plain one drops a
NaN where the reference propagates it. Measured, the plain node answers `1.0` for
`amax([nan, 1.0])` while `torch.amax` answers `nan`. The PropagateNaN form is
what matches, and `test_amax_and_amin_propagate_nan` is the case that fails if
someone routes these two at the plain selectors.

The rest of the cases cover the reduction arithmetics that differ from a plain
sum: an empty dimension (a product over nothing is 1, a maximum over nothing is
undefined), `dim=[]` (reduce over every axis), `keepdim`, negative dimensions,
the `dtype` override on `prod`, and the row pitch of a non-contiguous operand.

Every case self-skips off an MPS build.
"""

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

# The int[1]-dimension reductions.
LIST_REDUCTIONS = {
    "amax": torch.amax,
    "amin": torch.amin,
}

# (dim, keepdim) combinations that exercise every path in the body.
DIM_CASES = [
    ([0], False),
    ([1], False),
    ([1], True),
    ([0, 2], False),
    ([0, 2], True),
    ([2], False),
    ([-1], False),
    ([-2], True),
    ([], False),
]


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _tolerance(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"rtol": 1e-5, "atol": 1e-6}
    # bfloat16 carries ~3 decimal digits, and a product of fifteen of them
    # accumulates that error; the reference and the graph associate the same
    # multiplication in the same order but differ by the rounding of each step.
    return {"rtol": 2e-2, "atol": 2e-2}


class TestExtremaCorrectness:
    """`amax`/`amin` equal the reference over every dimension case."""

    @pytest.mark.parametrize("name", sorted(LIST_REDUCTIONS))
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_over_every_dim(self, name, dtype):
        fn = LIST_REDUCTIONS[name]
        host = torch.randn(3, 4, 5).to(dtype)
        for dims, keepdim in DIM_CASES:
            out = fn(_to_device(host), dims, keepdim=keepdim)
            expected = fn(host, dims, keepdim=keepdim)
            assert out.dtype == dtype
            assert out.shape == expected.shape, (dims, keepdim, out.shape)
            torch.testing.assert_close(
                out.cpu().float(), expected.float(), **_tolerance(dtype)
            )

    @pytest.mark.parametrize("name", sorted(LIST_REDUCTIONS))
    @pytest.mark.anyplatform
    def test_amax_and_amin_propagate_nan(self, name):
        """The reason these two take the PropagateNaN selectors.

        The plain `reductionMaximumWithTensor:` drops a NaN and answers the
        finite operand; `torch.amax` propagates it. A routing at the plain nodes
        would pass every finite case above and fail this one.
        """
        fn = LIST_REDUCTIONS[name]
        host = torch.tensor([[float("nan"), 1.0, 5.0], [2.0, 3.0, 4.0]])
        out = fn(_to_device(host), [1]).cpu()
        expected = fn(host, [1])
        assert torch.isnan(out).equal(torch.isnan(expected)), (name, out, expected)
        assert torch.isnan(out[0]), (name, out)

    @pytest.mark.parametrize("name", sorted(LIST_REDUCTIONS))
    @pytest.mark.anyplatform
    def test_infinities(self, name):
        fn = LIST_REDUCTIONS[name]
        host = torch.tensor([[-float("inf"), 0.0], [float("inf"), 1.0]])
        out = fn(_to_device(host), [1]).cpu()
        expected = fn(host, [1])
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize("name", sorted(LIST_REDUCTIONS))
    @pytest.mark.anyplatform
    def test_a_repeated_dimension_is_rejected(self, name):
        """`amax(x, [0, 0])` raises; the graph path must not answer instead."""
        fn = LIST_REDUCTIONS[name]
        with pytest.raises((RuntimeError, IndexError)):
            fn(_to_device(torch.randn(3, 4)), [0, 0])

    @pytest.mark.parametrize("name", sorted(LIST_REDUCTIONS))
    @pytest.mark.anyplatform
    def test_strided_and_row_pitch_operands(self, name):
        fn = LIST_REDUCTIONS[name]
        host = torch.randn(4, 5)  # 20-byte float32 rows
        torch.testing.assert_close(fn(_to_device(host), [1]).cpu(), fn(host, [1]))
        strided_host = torch.randn(8, 4)
        strided = _to_device(strided_host)[::2]
        assert not strided.is_contiguous()
        torch.testing.assert_close(fn(strided, [1]).cpu(), fn(strided_host[::2], [1]))


class TestProductCorrectness:
    """`prod.dim_int` reduces one dimension and takes an optional dtype."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu_over_every_dim(self, dtype):
        host = (torch.randn(3, 4, 5).abs() * 1.5 + 0.5).to(dtype)
        for dim, keepdim in (
            (0, False),
            (1, False),
            (1, True),
            (2, False),
            (-1, False),
        ):
            out = torch.prod(_to_device(host), dim, keepdim=keepdim)
            expected = torch.prod(host, dim, keepdim=keepdim)
            assert out.dtype == dtype
            assert out.shape == expected.shape, (dim, keepdim, out.shape)
            torch.testing.assert_close(
                out.cpu().float(), expected.float(), **_tolerance(dtype)
            )

    @pytest.mark.anyplatform
    def test_the_dtype_override(self):
        """`prod(x, dim, dtype=float64)` answers float64, not the operand's width.

        The dtype is folded into the plan key, so a float32 and a float64 plan
        for the same shape cannot be answered by one another.
        """
        host = torch.randn(3, 4).abs() + 0.5
        out = torch.prod(_to_device(host), 1, dtype=torch.float64)
        expected = torch.prod(host, 1, dtype=torch.float64)
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), expected)
        # And the same operand again without the override still answers float32.
        plain = torch.prod(_to_device(host), 1)
        assert plain.dtype == torch.float32
        torch.testing.assert_close(plain.cpu(), torch.prod(host, 1))

    @pytest.mark.anyplatform
    def test_a_zero_in_the_reduced_axis(self):
        """A product over a zero is zero, and over nothing is one."""
        host = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
        torch.testing.assert_close(
            torch.prod(_to_device(host), 1).cpu(), torch.prod(host, 1)
        )

    @pytest.mark.anyplatform
    def test_signs_survive(self):
        """An odd number of negatives answers negative."""
        host = torch.tensor([[-1.0, 2.0], [3.0, -4.0], [-1.0, -1.0]])
        torch.testing.assert_close(
            torch.prod(_to_device(host), 1).cpu(), torch.prod(host, 1)
        )
        odd = torch.tensor([[-1.0, 2.0, 3.0]])
        torch.testing.assert_close(
            torch.prod(_to_device(odd), 1).cpu(), torch.prod(odd, 1)
        )


class TestDeclines:
    """The cases the gates do not serve stay correct through the reference."""

    @pytest.mark.parametrize("name", sorted(LIST_REDUCTIONS))
    @pytest.mark.anyplatform
    def test_an_empty_operand_round_trips(self, name):
        """A reduction over nothing is not nothing, so it takes the reference.

        `amax` of an empty tensor raises while `prod` answers 1, and the body
        does not try to express either in the graph.
        """
        fn = LIST_REDUCTIONS[name]
        # IndexError comes from the reference (`Expected reduction dim 0 to have
        # non-zero size`); what this holds is that no answer is produced.
        with pytest.raises((RuntimeError, IndexError)):
            fn(_to_device(torch.empty(0, 3)), [0])

    @pytest.mark.anyplatform
    def test_prod_of_an_empty_axis_is_one(self):
        out = torch.prod(_to_device(torch.empty(0, 3)), 0)
        expected = torch.prod(torch.empty(0, 3), 0)
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("name", sorted(LIST_REDUCTIONS))
    @pytest.mark.anyplatform
    def test_float64_round_trips(self, name):
        fn = LIST_REDUCTIONS[name]
        host = torch.randn(3, 4, dtype=torch.float64)
        out = fn(_to_device(host), [1])
        assert out.dtype == torch.float64
        torch.testing.assert_close(out.cpu(), fn(host, [1]))


class TestRouting:
    """Each op is routed here rather than to the fallback."""

    @pytest.mark.anyplatform
    def test_amax_and_amin_are_routed(self):
        assert routed_backend_or_none("amax") in ("apple", "mps")
        assert routed_backend_or_none("amin") in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_prod_dim_int_is_routed(self):
        assert routed_backend_or_none("prod.dim_int") in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_the_scalar_prod_is_not_claimed(self):
        """`prod` with no dimension is a different signature and stays out."""
        assert routed_backend_or_none("prod") is None
