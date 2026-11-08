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

"""Apple Silicon (Metal / MPSGraph) argmax over a NaN.

The reference names the **first** NaN in the slice. The graph this backend builds
compares every position against a NaN-aware reduction and reduces the matching
coordinates, which needs the NaN candidate to include the coordinate it compares
against.

It did not. `has_nan` was the *last* NaN's coordinate and the test was a strict
`>`, so that coordinate was excluded from its own candidate set -- a slice whose
only NaN sat at position 0 produced no candidate, and the reduction answered the
`no_match` sentinel, which is the axis length:

    argmax([nan, 1, 2])    reference 0    device 3   (before)

For a length-4 slice, `3` is also a valid index, so the wrong answer looked like a
plausible position rather than an error.

The existing cases missed it for the usual reason: every one of them used finite
values. This file is the intersection -- `argmax` and NaN -- that none of them
covered.

Every case self-skips off an MPS build.
"""

import pytest
import torch
import torch_fl  # noqa: F401

from platform_support import detect_platform


DEVICE = "flagos:0"

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        detect_platform() != "mps",
        reason="Apple Silicon (Metal) backend only",
    ),
]

NAN = float("nan")


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


class TestSingleNaN:
    """One NaN in the slice, at each position."""

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([NAN, 1.0, 2.0], 0),
            ([1.0, NAN, 3.0], 1),
            ([1.0, 2.0, NAN], 2),
            ([NAN], 0),
            ([NAN, NAN], 0),
        ],
    )
    @pytest.mark.anyplatform
    def test_the_nan_position_is_named(self, values, expected):
        host = torch.tensor([values])
        assert torch.argmax(host, 1).item() == expected
        got = torch.argmax(_to_device(host), 1)
        assert got.shape == (1,)
        assert got.dtype == torch.int64
        assert got.cpu().item() == expected, got


class TestSeveralNaNs:
    """The first NaN wins, whatever the later ones do."""

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([NAN, 1.0, NAN, 3.0], 0),
            ([1.0, NAN, NAN], 1),
            ([NAN, NAN, 3.0], 0),
            ([1.0, NAN, 2.0, NAN], 1),
            ([NAN, NAN, NAN], 0),
        ],
    )
    @pytest.mark.anyplatform
    def test_the_first_nan_is_named(self, values, expected):
        host = torch.tensor([values])
        assert torch.argmax(host, 1).item() == expected
        assert torch.argmax(_to_device(host), 1).cpu().item() == expected


class TestNoRegressionOnFiniteSlices:
    """The fix must not disturb a slice without a NaN."""

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([1.0, 2.0, 3.0], 2),
            ([3.0, 2.0, 1.0], 0),
            ([2.0, 2.0, 2.0], 0),
            ([-3.0, -1.0, -2.0], 1),
            ([1.0, 5.0, 5.0], 1),
        ],
    )
    @pytest.mark.anyplatform
    def test_finite_slices_unchanged(self, values, expected):
        host = torch.tensor([values])
        assert torch.argmax(host, 1).item() == expected
        assert torch.argmax(_to_device(host), 1).cpu().item() == expected

    @pytest.mark.anyplatform
    def test_a_sentinel_value_never_leaks(self):
        """The old defect answered `size(axis)`; that is a valid index for a
        longer slice, so check no result can reach it."""
        host = torch.tensor([[NAN, 1.0, 2.0, 3.0]])
        got = torch.argmax(_to_device(host), 1).cpu()
        assert got.item() != host.size(1), "answered the no-match sentinel"
        assert 0 <= got.item() < host.size(1)


class TestOtherAxes:
    """The same behaviour along an axis that is not the last one."""

    @pytest.mark.anyplatform
    def test_a_middle_axis(self):
        # Column 0 holds the NaN and answers 0; column 1 is [1, 3, 5] and
        # answers 2.
        host = torch.tensor([[NAN, 1.0], [2.0, 3.0], [4.0, 5.0]])
        expected = torch.argmax(host, 0)
        assert expected.tolist() == [0, 2]
        assert torch.argmax(_to_device(host), 0).cpu().tolist() == expected.tolist()

    @pytest.mark.anyplatform
    def test_keepdim(self):
        host = torch.tensor([[1.0, NAN, 3.0]])
        assert torch.argmax(_to_device(host), 1, keepdim=True).shape == (1, 1)
        assert torch.argmax(_to_device(host), 1, keepdim=True).cpu().item() == 1

    @pytest.mark.anyplatform
    def test_a_two_dimensional_slice(self):
        host = torch.randn(2, 3, 4)
        host[0, 1, 2] = NAN
        host[1, 0, 0] = NAN
        expected = torch.argmax(host, 2)
        assert torch.argmax(_to_device(host), 2).cpu().equal(expected)
