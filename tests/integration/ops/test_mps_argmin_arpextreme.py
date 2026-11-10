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

"""Apple Silicon (Metal / MPSGraph) argmin, max.dim and min.dim.

These three share one index construction: reduce the values with a NaN-aware
extremum, keep the positions equal to it, name the first NaN if the slice holds
one, and reduce the surviving positions with `minimum`. `argmin` had been the
one direction of the pair left on the reference while `argmax` was served, and
`max.dim`/`min.dim` had never been served at all -- the same half-served shape
that `tril` and `cumprod` were in.

Correctness is compared against the reference on the CPU for every shape, tie
and NaN placement. Every case self-skips off an MPS build.
"""

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"
NAN = float("nan")

# The capability check lives in the marker, not in a module-level call: the
# per-platform skip table in ops/conftest.py turns `mps` into the skip on every
# other lane, and tests/integration/test_integration_source_contract.py holds new
# platform files to that shape.
pytestmark = pytest.mark.mps


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


def _close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """NaN-aware compare -- a NaN result is correct when the reference is NaN."""
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape}"
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype}"
    if not torch.allclose(actual, expected, equal_nan=True):
        raise AssertionError(f"{label}: {actual.tolist()} != {expected.tolist()}")


class TestArgminParity:
    """`argmin` must answer what the served `argmax` answers, mirrored."""

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([3.0, 1.0, 4.0], 1),
            ([1.0, 2.0, 3.0], 0),
            ([3.0, 2.0, 1.0], 2),
            ([2.0, 2.0, 2.0], 0),  # ties name the first
            ([-3.0, -1.0, -2.0], 0),
        ],
    )
    @pytest.mark.anyplatform
    def test_argmin_names_the_first_minimum(self, values, expected):
        host = torch.tensor([values])
        assert torch.argmin(host, 1).item() == expected
        got = torch.argmin(_to_device(host), 1)
        assert got.dtype == torch.int64
        assert got.cpu().item() == expected

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([NAN, 1.0, 2.0], 0),
            ([1.0, NAN, 3.0], 1),
            ([1.0, 2.0, NAN], 2),
            ([NAN, 1.0, NAN, 3.0], 0),
            ([1.0, NAN, NAN], 1),
            ([NAN, NAN, 3.0], 0),
        ],
    )
    @pytest.mark.anyplatform
    def test_argmin_names_the_first_nan(self, values, expected):
        """A NaN wins over any finite minimum, in both directions."""
        host = torch.tensor([values])
        assert torch.argmin(host, 1).item() == expected
        assert torch.argmin(_to_device(host), 1).cpu().item() == expected

    @pytest.mark.anyplatform
    def test_argmin_matches_the_reference_over_shapes(self):
        for shape, dim in (
            ((3, 4), 0),
            ((3, 4), 1),
            ((3, 4), -1),
            ((2, 3, 4), 2),
            ((2, 3, 4), 0),
            ((17, 5), 1),
        ):
            for keepdim in (False, True):
                host = torch.randn(*shape)
                label = f"{shape} d={dim} kd={keepdim}"
                expected = torch.argmin(host, dim, keepdim=keepdim)
                actual = torch.argmin(_to_device(host), dim, keepdim=keepdim).cpu()
                assert actual.equal(expected), label

    @pytest.mark.anyplatform
    def test_argmin_and_argmax_disagree_where_they_should(self):
        """The two directions must not share a hard-coded comparison."""
        host = torch.tensor([[1.0, 9.0, 5.0], [9.0, 1.0, 5.0]])
        dev = _to_device(host)
        assert torch.argmax(dev, 1).cpu().tolist() == [1, 0]
        assert torch.argmin(dev, 1).cpu().tolist() == [0, 1]


class TestExtremePair:
    """`max.dim` / `min.dim` return the (values, indices) pair."""

    @pytest.mark.parametrize("op", ["max", "min"])
    @pytest.mark.anyplatform
    def test_the_pair_matches_the_reference(self, op):
        fn = getattr(torch, op)
        for shape, dim in (
            ((3, 4), 0),
            ((3, 4), 1),
            ((3, 4), -1),
            ((2, 3, 4), 2),
            ((2, 3, 4), 0),
            ((17, 5), 1),
        ):
            for keepdim in (False, True):
                host = torch.randn(*shape)
                expected = fn(host, dim, keepdim=keepdim)
                actual = fn(_to_device(host), dim, keepdim=keepdim)
                label = f"{op} {shape} d={dim} kd={keepdim}"
                _close(actual.values.cpu(), expected.values, label + " values")
                assert actual.indices.cpu().equal(expected.indices), f"{label} indices"

    @pytest.mark.parametrize(
        "values,max_index,min_index",
        [
            # All equal: both directions name position 0.
            ([2.0, 2.0, 2.0], 0, 0),
            # A tie at the top only, then at the bottom only.
            ([1.0, 5.0, 5.0], 1, 0),
            ([-3.0, -1.0, -2.0], 1, 0),
            ([5.0, 1.0, 1.0], 0, 1),
            ([7.0, 7.0, 3.0], 0, 2),
        ],
    )
    @pytest.mark.anyplatform
    def test_a_tie_names_the_first_position(self, values, max_index, min_index):
        """The two directions tie in different places, so their expectations are
        stated separately -- a shared expectation would pass only if one of them
        were computed wrongly."""
        host = torch.tensor([values])
        for op, index in (("max", max_index), ("min", min_index)):
            expected = getattr(torch, op)(host, 1)
            actual = getattr(torch, op)(_to_device(host), 1)
            _close(actual.values.cpu(), expected.values, op)
            assert actual.indices.cpu().item() == expected.indices.item() == index

    @pytest.mark.parametrize(
        "values,index",
        [
            ([NAN, 1.0, 2.0], 0),
            ([1.0, NAN, 3.0], 1),
            ([1.0, 2.0, NAN], 2),
            ([NAN, 1.0, NAN, 3.0], 0),
            ([1.0, NAN, NAN], 1),
        ],
    )
    @pytest.mark.anyplatform
    def test_a_nan_is_the_value_and_its_position_the_index(self, values, index):
        """Both halves of the pair have to agree: the value is NaN and the index
        is where that NaN sits. A pair whose halves disagree is worse than a
        wrong value, because the caller picks one to index with."""
        host = torch.tensor([values])
        for op in ("max", "min"):
            expected = getattr(torch, op)(host, 1)
            actual = getattr(torch, op)(_to_device(host), 1)
            _close(actual.values.cpu(), expected.values, op + " values")
            assert torch.isnan(actual.values.cpu()).all(), (op, actual.values)
            assert actual.indices.cpu().item() == index, (op, values)


class TestTheIndexStaysInRange:
    """The marker a non-matching position contributes must never be returned."""

    @pytest.mark.parametrize("op", ["argmax", "argmin"])
    @pytest.mark.anyplatform
    def test_no_index_reaches_the_axis_length(self, op):
        fn = getattr(torch, op)
        for shape, dim in (((4, 7), 1), ((7, 4), 0), ((2, 3, 5), 2)):
            host = torch.randn(*shape)
            host[0].fill_(NAN)
            got = fn(_to_device(host), dim).cpu()
            assert (got >= 0).all(), got
            assert (got < shape[dim]).all(), (shape, dim, got)

    @pytest.mark.parametrize("op", ["max", "min"])
    @pytest.mark.anyplatform
    def test_the_pair_index_stays_in_range(self, op):
        fn = getattr(torch, op)
        host = torch.randn(4, 7)
        host[2, 3] = NAN
        got = fn(_to_device(host), 1).indices.cpu()
        assert (got >= 0).all() and (got < 7).all(), got


class TestDtypeGate:
    """Only float32 is served; every other dtype must still answer correctly
    through the reference, not raise or answer a marker."""

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.int32, torch.int64]
    )
    @pytest.mark.anyplatform
    def test_an_unserved_dtype_still_answers_correctly(self, dtype):
        host = torch.tensor([[3, 1, 4], [1, 5, 9]], dtype=dtype)
        for op in ("argmax", "argmin"):
            expected = getattr(torch, op)(host, 1)
            actual = getattr(torch, op)(_to_device(host), 1).cpu()
            assert actual.equal(expected), (op, dtype)
        for op in ("max", "min"):
            expected = getattr(torch, op)(host, 1)
            actual = getattr(torch, op)(_to_device(host), 1)
            assert actual.values.cpu().equal(expected.values), (op, dtype)
            assert actual.indices.cpu().equal(expected.indices), (op, dtype)

    @pytest.mark.anyplatform
    def test_an_empty_axis_raises_like_the_reference(self):
        for op in ("argmax", "argmin"):
            with pytest.raises(IndexError):
                getattr(torch, op)(torch.empty(2, 0, device=DEVICE), 1)
        for op in ("max", "min"):
            with pytest.raises(IndexError):
                getattr(torch, op)(torch.empty(2, 0, device=DEVICE), 1)
