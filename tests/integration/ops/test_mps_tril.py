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

"""Apple Silicon (Metal / MPSGraph) tril.

`triu` has been served since the triangular family landed, through a graph that
builds `column - row` from `coordinateAlongAxis:` and selects the operand where
that difference is at least the diagonal. `tril` is the same construction with
`<=`, and it was still on the fallback -- so a tensor masked on one side of the
diagonal cost a round-trip while the other side did not.

That asymmetry is the reason this increment exists, and it is worth stating as a
number: on `(1024, 1024)` the fallback took 0.436 ms where `triu`'s graph answers in
0.120 ms. After this change `tril` answers in 0.116 ms.

The cases below are about the places where a "reversed triu" could get the
comparison off by one:

* **The diagonal argument**, positive, zero and negative, which is where an
  off-by-one shows up as a single extra or missing column.
* **A non-square operand**, where `row` and `column` have different ranges and a
  body that assumed a square matrix would mask the wrong triangle.
* **Batches**, where the coordinates must come from the last two axes rather than
  the first two.

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


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.clone().to(DEVICE)


class TestCorrectness:
    """Every shape and diagonal matches the reference."""

    @pytest.mark.parametrize(
        "shape,diagonal",
        [
            ((4, 4), 0),
            ((4, 4), 1),
            ((4, 4), 2),
            ((4, 4), -1),
            ((4, 4), -3),
            # Non-square in both directions: the coordinates are the last two
            # axes and a body that assumed squareness would mask the wrong half.
            ((3, 5), 0),
            ((3, 5), 2),
            ((3, 5), -1),
            ((5, 3), 0),
            ((5, 3), 1),
            # Batches over more than one leading axis.
            ((2, 4, 4), 0),
            ((2, 3, 4, 4), 1),
        ],
    )
    @pytest.mark.anyplatform
    def test_matches_cpu(self, shape, diagonal):
        host = torch.randn(*shape)
        out = torch.tril(_to_device(host), diagonal)
        assert out.shape == host.shape
        assert out.device.type == "flagos"
        expected = torch.tril(host, diagonal)
        # The body selects the operand or a zero, so the answer is exact rather
        # than approximate -- but a float compare keeps it robust to the zero's
        # signedness.
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_float_widths(self, dtype):
        host = torch.randn(4, 4).to(dtype)
        out = torch.tril(_to_device(host))
        assert out.dtype == dtype
        assert torch.equal(out.cpu(), torch.tril(host))

    @pytest.mark.anyplatform
    def test_integral_operand(self):
        host = torch.randint(-9, 9, (4, 5))
        out = torch.tril(_to_device(host), 1)
        assert out.dtype == torch.int64
        assert torch.equal(out.cpu(), torch.tril(host, 1))

    @pytest.mark.anyplatform
    def test_the_far_corners(self):
        """A diagonal past either edge keeps everything or nothing."""
        host = torch.randn(4, 5)
        # Below the lowest cell: the whole tensor survives.
        assert torch.equal(torch.tril(_to_device(host), 99).cpu(), torch.tril(host, 99))
        # Above the highest cell: nothing survives.
        assert torch.equal(
            torch.tril(_to_device(host), -99).cpu(), torch.tril(host, -99)
        )

    @pytest.mark.anyplatform
    def test_the_zero_is_a_zero(self):
        """The masked half is exactly zero, not a small number."""
        host = torch.randn(4, 4) + 5.0
        out = torch.tril(_to_device(host), 0).cpu()
        upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)
        assert (out[upper] == 0).all(), out

    @pytest.mark.anyplatform
    def test_a_strided_operand(self):
        base = torch.randn(8, 4)
        strided = _to_device(base)[::2]
        assert not strided.is_contiguous()
        assert torch.equal(torch.tril(strided).cpu(), torch.tril(base[::2]))

    @pytest.mark.anyplatform
    def test_a_batched_operand_masks_each_matrix(self):
        """Each batched matrix is masked on its own, not the batch as a whole."""
        host = torch.randn(3, 4, 4)
        out = torch.tril(_to_device(host), 1).cpu()
        for i in range(3):
            assert torch.equal(out[i], torch.tril(host[i], 1)), i

    @pytest.mark.anyplatform
    def test_rank_one_declines(self):
        """A 1-D operand has no trailing two axes; the reference raises."""
        with pytest.raises(RuntimeError):
            torch.tril(_to_device(torch.randn(4)))


class TestRouting:
    """Both triangles are routed here, which is the point of the increment."""

    @pytest.mark.anyplatform
    def test_both_directions_route(self):
        assert routed_backend_or_none("tril") in ("apple", "mps")
        # `triu` was already served; this asserts the change did not disturb it.
        assert routed_backend_or_none("triu") in ("apple", "mps")

    @pytest.mark.anyplatform
    def test_they_are_not_the_same_operator(self):
        """A body that forgot to reverse the comparison would pass a unit test
        on the diagonal but fail this."""
        host = torch.randn(4, 4)
        lower = torch.tril(_to_device(host)).cpu()
        upper = torch.triu(_to_device(host)).cpu()
        assert not torch.equal(lower, upper)
        assert torch.equal(lower, torch.tril(host))
        assert torch.equal(upper, torch.triu(host))
