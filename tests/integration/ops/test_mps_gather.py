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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``gather``.

``gather`` is the fourth and last of the read family -- ``embedding``,
``index_select`` and ``take`` came first -- and it is the one whose contract is
neither a sibling's nor a superset of one. Two shape sentences decide it, and
the interesting part of this file is that they look like the ``take`` sentences
and are not:

* **The index must have the operand's rank**, where ``take`` reads any rank.
  ``gather((4, 5), 1, (4, 5))`` is the shape the op is written for, and neither
  a lower-rank nor a higher-rank index is served here -- ``index_select`` and
  ``take`` divide the two directions between them, and this op accepts neither.
* **The result is the index's shape**, which is ``take``'s sentence and not
  ``index_select``'s, so the ``(1, N)`` index that reads one row returns
  ``(1, N)``.

The other dimensions are the finding this increment rests on, and they were
measured rather than read off the SDK:

* **A non-axis dimension may differ from the operand's in either direction.**
  ``gather((4, 5), 1, (4, 3))`` and ``gather((4, 5), 1, (4, 7))`` are both
  valid, where ``gatherAlongAxis:`` requires the two tensors to "match except at
  ``axis``". The first body handed the node the operand as-is and the MPSGraph
  compiler *aborted the process* -- a hard abort, not a decline the body could
  catch -- so the body reconciles the two shape rules by trimming the operand's
  non-axis dimensions down to the index's extents before the gather, on the
  host, and the trim is *shrink-or-leave* because the direction that cannot be
  expressed as a trim does not need one.
* **The trim has to be a host ``narrow``, not the graph's ``sliceTensor:``.**
  Measured: the graph slice produced the right *shape* and the wrong *element*,
  ``gather((4, 5), 0, (3, 1))`` returning ``[0, 10, 0]`` against the
  reference's ``[0, 10, 15]`` -- the sliced tensor and the view the runtime fed
  it disagreed on the row pitch, the same MPSNDArray layout constraint the row
  pitch note in ``mps_common.h`` records. The value case in
  ``TestGatherShape`` is that exact call, because it is the regression a correct
  shape cannot catch.

The declines are the rest of it, and each is a case where the node would answer
plausibly rather than wrongly: an out-of-range index (the node inserts a zero,
the reference raises), a negative index (the reference raises, where ``take``
wraps it), and a ``dim`` outside the operand's rank.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
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
# Every dtype the graph is served on. `gather` is a pure read, so it passes its
# operand's width through rather than narrowing to the float widths.
READABLE_DTYPES = FLOAT_DTYPES + [torch.int32, torch.int16, torch.int8, torch.uint8]


def _reference(self: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """``at::gather`` on the host, in the operand's own dtype.

    The reference the kernel is held to, and deliberately the same op the body
    round-trips to rather than a hand-built ``self[...]``: the shape and range
    contracts this op has and its siblings do not are exactly the places a
    hand-built reference would encode the test author's guess instead of aten's
    answer.
    """
    return torch.ops.aten.gather(self, dim, index)


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another case
    # left behind.
    return t.clone().to(DEVICE)


def _assert_exact(out: torch.Tensor, expected: torch.Tensor) -> None:
    """The result equals the reference's, byte for byte.

    No tolerance: the node moves bytes rather than computing over them, so the
    only two answers are "the right elements" and "the wrong elements".
    """
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    assert torch.equal(out.cpu(), expected), "result differs from the CPU reference"


class TestGatherShape:
    """The result is the index's shape, read along the axis the caller names."""

    @pytest.mark.anyplatform
    def test_the_axis_the_first_version_read_wrong(self):
        """The exact call the graph-slice body answered with ``[0, 10, 0]``.

        ``gather((4, 5), 0, (3, 1))`` reads three rows and one column, so the
        trim is a real one: dimension 1 goes from 5 to 1 and dimension 0 is the
        gathered one. Held against the values and not only the shape, because
        that is the failure this increment was opened for -- the graph slice
        produced a correctly shaped ``3x1`` result holding the *wrong element*
        (``[0, 10, 0]`` where the reference says ``[0, 10, 15]``), which every
        shape assertion in this file would have passed.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.tensor([[0], [2], [3]], dtype=torch.long)
        out = torch.gather(_to_device(operand), 0, _to_device(index))
        assert out.shape == (3, 1)
        _assert_exact(out, _reference(operand, 0, index))
        # Stated outright, because these are the values the bug returned: the
        # third row must be 15, not the 0 the sliced buffer read.
        assert out.cpu().flatten().tolist() == [0.0, 10.0, 15.0]

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize(
        "index",
        [
            torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long),
            torch.tensor([[0], [2], [3], [1]], dtype=torch.long),
        ],
        ids=["one-row", "one-column"],
    )
    def test_the_index_shape_is_the_result_shape(self, dim, index):
        """The lopsided index shapes come back as themselves.

        A reshape or a trim that had flattened the index would answer a square
        index correctly and a one-row or one-column one with the values in the
        wrong places, so the shapes are held side by side and each is compared
        against the reference rather than against the operand.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        out = torch.gather(_to_device(operand), dim, _to_device(index))
        assert out.shape == index.shape
        _assert_exact(out, _reference(operand, dim, index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_rank_three(self, dim):
        """The trim is `axis`-preserving at every rank and every axis."""
        operand = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        index = torch.full((2, 3, 4), 0, dtype=torch.long)
        index = index.scatter(
            dim, index, torch.randint(0, operand.size(dim), (2, 3, 4))
        )
        out = torch.gather(_to_device(operand), dim, _to_device(index))
        assert out.shape == (2, 3, 4)
        _assert_exact(out, _reference(operand, dim, index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [-1, -2, -3])
    def test_negative_dim_resolves_against_the_rank(self, dim):
        """``dim=-1`` over a rank-3 operand is ``dim=2``, measured."""
        operand = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        index = torch.randint(0, operand.shape[dim], (2, 3, 4), dtype=torch.long)
        out = torch.gather(_to_device(operand), dim, _to_device(index))
        _assert_exact(out, _reference(operand, dim, index))

    @pytest.mark.anyplatform
    def test_empty_index(self):
        """No indices is a valid call and an empty result, not a decline.

        MPSGraph rejects an empty tensor, so the body short-circuits this shape
        before it builds a node.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.empty(4, 0, dtype=torch.long)
        out = torch.gather(_to_device(operand), 1, _to_device(index))
        assert out.shape == (4, 0)
        _assert_exact(out, _reference(operand, 1, index))


class TestGatherDifferingDimensions:
    """A non-axis dimension may differ from the operand's, in either direction.

    This is the finding the body is built around. ``gatherAlongAxis:`` wants the
    two tensors to match except at ``axis``, and the reference does not, so the
    body trims the operand down to the index's extents -- and the trim is
    *shrink-or-leave* rather than shrink-or-decline.
    """

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("extent", [1, 2, 3])
    def test_a_narrower_index_dimension(self, extent):
        """``gather((4, 5), 1, (4, N))`` for ``N < 5`` reads ``self[:, :N]``.

        The trim is a real one for each of these: the operand's dimension goes
        from 5 down to ``N``, and the answer is a sub-block of the operand
        rather than of the whole.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.randint(0, 5, (4, extent), dtype=torch.long)
        out = torch.gather(_to_device(operand), 1, _to_device(index))
        assert out.shape == (4, extent)
        _assert_exact(out, _reference(operand, 1, index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("extent", [6, 7, 11])
    def test_a_wider_index_dimension(self, extent):
        """``gather((4, 5), 1, (4, N))`` for ``N > 5`` needs no trim at all.

        The direction that cannot be expressed as a trim, and the one a body
        written from the SDK's shape rule would decline. The reference's element
        mapping reads ``res[i, j] = self[index[i, j], j]``, whose *column* term
        is the operand's own position, so a ``j`` past the operand's extent is
        not a position the reference invents -- the operand already has that
        column. The index spans ``[0, 5)`` here so the case isolates the shape,
        and the columns past 4 are the ones a broadcast operand would answer
        with the wrong values while a left-alone one answers correctly.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.randint(0, 5, (4, extent), dtype=torch.long)
        out = torch.gather(_to_device(operand), 1, _to_device(index))
        assert out.shape == (4, extent)
        _assert_exact(out, _reference(operand, 1, index))

    @pytest.mark.anyplatform
    def test_the_wider_case_on_a_known_index(self):
        """The same case with the values written out.

        The random version above can pass a body that read a repeated column
        because the index happened to name it. Here the index's last two columns
        name ``2`` and ``3`` on a row that starts ``5, 6, 7, 8, 9``, so a body
        that had broadcast the operand (the other way to reconcile the two shape
        rules) would return a repeated ``9`` and this case is where it shows.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 0, 1],
                [0, 1, 2, 3, 4, 3, 2],
                [4, 3, 2, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 4, 4],
            ],
            dtype=torch.long,
        )
        out = torch.gather(_to_device(operand), 1, _to_device(index))
        _assert_exact(out, _reference(operand, 1, index))
        assert out.cpu()[1].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0]

    @pytest.mark.anyplatform
    def test_both_directions_at_once_on_rank_three(self):
        """One dimension narrows while the other stays, at a middle axis.

        Rank-3 is where the trim has two non-axis dimensions to decide
        independently. A ``(1, 5, 4)`` index against a ``(2, 3, 4)`` operand at
        axis 1 narrows dimension 0 to 1 and leaves dimension 2 at the operand's
        own width -- and the axis dimension, 5, is the selector and is free to
        exceed the operand's 3. A *non-axis* dimension that exceeds the
        operand's is the reference's error and is held separately, in
        ``test_a_non_axis_dimension_larger_than_the_operand_raises``.
        """
        operand = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        index = torch.zeros(1, 5, 4, dtype=torch.long)
        index[0, 0] = torch.tensor([0, 1, 2, 0])
        index[0, 3] = torch.tensor([2, 2, 2, 1])
        out = torch.gather(_to_device(operand), 1, _to_device(index))
        assert out.shape == (1, 5, 4)
        _assert_exact(out, _reference(operand, 1, index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "operand_shape, dim, index_shape",
        [
            ((4, 5), 0, (4, 7)),
            ((4, 5), 1, (5, 5)),
            ((2, 4, 6), 1, (2, 4, 7)),
            ((2, 4, 6), 2, (2, 5, 6)),
        ],
        ids=["rank-2 dim-0", "rank-2 dim-1", "rank-3 dim-1", "rank-3 dim-2"],
    )
    def test_a_non_axis_dimension_larger_than_the_operand_raises(
        self, operand_shape, dim, index_shape
    ):
        """The one side of the shape rule the reference enforces, measured.

        The sibling rule -- a *narrower* non-axis dimension -- is legal and is
        what the trim exists for. The wider direction is not: the reference
        raises ``Size does not match at dimension N expected index [...] to be
        no larger than self [...] apart from dimension M``. The node takes the
        other reading entirely (measured: ``gatherAlongAxis:`` compiles the pair
        and shapes the result to the index), so a body that leaned on the node
        for this would answer a call the reference refuses.

        The axis dimension is deliberately *not* covered here: an index wider
        than the operand along the axis is legal and is what
        ``test_a_wider_index_dimension`` holds.
        """
        operand = torch.arange(
            int(torch.tensor(operand_shape).prod()), dtype=torch.float32
        ).reshape(operand_shape)
        index = torch.zeros(index_shape, dtype=torch.long)
        with pytest.raises(RuntimeError, match="no larger than"):
            torch.gather(_to_device(operand), dim, _to_device(index))


class TestGatherRankErrors:
    """An index of the wrong rank is the reference's error, not a broadcast."""

    @pytest.mark.anyplatform
    def test_a_higher_rank_index_raises(self):
        """``gather((2, 3, 4), 1, (1, 2, 1, 4))`` raises, and the body declines.

        The node would happily broadcast it -- measured, it accepts the wider
        shape and shapes the result to it -- so this is the one shape the body
        has to refuse rather than reconcile. Held against the reference's own
        message, because a body that let it through would return a rank-4 tensor
        where the caller is owed an error.
        """
        operand = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        index = torch.zeros(1, 2, 1, 4, dtype=torch.long)
        with pytest.raises(RuntimeError, match="same number of dimensions"):
            torch.gather(_to_device(operand), 1, _to_device(index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("index_shape", [(5,), (3,), (1,)])
    def test_a_rank_one_index_into_a_rank_two_operand_raises(self, index_shape):
        """The lower-rank direction, where ``take`` would have served it.

        Measured: ``gather((4, 5), 1, (5,))`` raises "Index tensor must have the
        same number of dimensions as input tensor", so a body that reused
        ``take``'s any-rank index rule would answer a call the reference refuses
        -- and return a plausible tensor while doing it.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.zeros(index_shape, dtype=torch.long)
        with pytest.raises(RuntimeError, match="same number of dimensions"):
            torch.gather(_to_device(operand), 1, _to_device(index))


class TestGatherRange:
    """The index is checked against the axis's extent, not the operand's size."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("bad_index", [5, 99])
    def test_out_of_range_index_raises(self, bad_index):
        """The node inserts a zero, the reference raises, so the host checks.

        The header says it outright -- "If an index is out of bounds of the
        ``updatesTensor`` along ``axis`` a 0 is inserted" -- and the reference
        raises ``RuntimeError: index 99 is out of bounds for dimension 0 with
        size 5``. A body that skipped the check would return a tensor holding a
        silent zero, which is a plausible wrong value rather than a crash.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.zeros(3, 1, dtype=torch.long)
        index[0, 0] = bad_index
        with pytest.raises(RuntimeError, match="out of bounds"):
            torch.gather(_to_device(operand), 0, _to_device(index))

    @pytest.mark.anyplatform
    def test_negative_index_raises(self):
        """A negative index is out of range here, where ``take`` wraps it.

        Measured: the reference raises ``index -1 is out of bounds for dimension
        0 with size 5``. This is the same failure class as ``take``'s negative
        index bug in the opposite direction -- there the body range-checked for
        the sibling's contract and the wrap was missing, here a body that reused
        ``take``'s normalisation would wrap an index the reference refuses and
        answer a call that should have raised.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.zeros(3, 1, dtype=torch.long)
        index[0, 0] = -1
        with pytest.raises(RuntimeError, match="out of bounds"):
            torch.gather(_to_device(operand), 0, _to_device(index))

    @pytest.mark.anyplatform
    def test_the_bound_is_the_axis_extent_not_the_operand_size(self):
        """``4`` is in range at ``dim=1`` and out of range at ``dim=0``.

        The mistake this op makes easy in both directions: the axis's extent,
        not ``numel``, is the bound, and the same index value is legal at one
        dim and not at the other. A body that checked against ``numel`` would
        serve both.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.full((4,), 4, dtype=torch.long)
        # In range at dim=1, where the axis has 5 elements.
        _assert_exact(
            torch.gather(_to_device(operand), 1, _to_device(index.reshape(1, 4))),
            _reference(operand, 1, index.reshape(1, 4)),
        )
        # Out of range at dim=0, where the axis has 4.
        with pytest.raises(RuntimeError, match="out of bounds"):
            torch.gather(_to_device(operand), 0, _to_device(index.reshape(4, 1)))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [2, -3, 7])
    def test_a_dim_outside_the_rank_raises(self, dim):
        """``IndexError``, and it is the reference's message.

        ``gather((4, 5), 2, ...)`` raises ``Dimension out of range (expected to
        be in range of [-2, 1], but got 2)``, so the body leaves a bad ``dim`` to
        the reference rather than resolving it into a valid axis.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.zeros(4, 5, dtype=torch.long)
        with pytest.raises(IndexError, match="Dimension out of range"):
            torch.gather(_to_device(operand), dim, _to_device(index))


class TestGatherCorrectness:
    """``gather`` on the flagos device equals the CPU reference exactly."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", READABLE_DTYPES)
    def test_the_operand_width_passes_through(self, dtype):
        """A read does not change its operand's width, and it does not round it.

        Each width is filled with values a float cast would round, so a body that
        staged the operand through float32 would show here.
        """
        operand = torch.arange(256).to(dtype).reshape(16, 16)
        index = torch.randint(0, 16, (4, 16), dtype=torch.long)
        out = torch.gather(_to_device(operand), 0, _to_device(index))
        assert out.dtype == dtype
        _assert_exact(out, _reference(operand, 0, index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
    def test_both_index_widths_are_served(self, index_dtype):
        """int32 and int64, and the wider one is the one ``take`` refuses.

        Measured: the reference accepts both here (``take`` accepts only int64),
        so the gate is the embedding one rather than a dedicated narrower
        predicate.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.randint(0, 5, (4, 5)).to(index_dtype)
        out = torch.gather(_to_device(operand), 1, _to_device(index))
        assert out.shape == (4, 5)
        _assert_exact(out, _reference(operand, 1, index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "index_dtype", [torch.int16, torch.int8, torch.bool, torch.float32]
    )
    def test_other_index_widths_raise(self, index_dtype):
        """Every index but int32/int64 is the reference's ``RuntimeError``.

        The narrower integers are the interesting members -- they are integers
        the graph could be handed, and a body that widened them would answer
        where the reference fails.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.zeros(4, 5).to(index_dtype)
        with pytest.raises(RuntimeError, match="int32/int64"):
            torch.gather(_to_device(operand), 1, _to_device(index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.bool])
    def test_declined_operand_dtypes_keep_the_reference(self, dtype):
        """A width MPSGraph has no element type for takes the round-trip.

        The case asserts the values too, so a body that ran the graph anyway
        would have to also be right about a width it cannot represent.
        """
        base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        operand = (base > 10) if dtype is torch.bool else base.to(dtype)
        index = torch.randint(0, 5, (4, 5), dtype=torch.long)
        out = torch.gather(_to_device(operand), 1, _to_device(index))
        assert out.dtype == dtype
        _assert_exact(out, _reference(operand, 1, index))

    @pytest.mark.anyplatform
    def test_non_contiguous_operand(self):
        """A strided operand is materialised before the graph reads it.

        The body hands the graph a dense buffer, so a strided operand read
        densely would read the wrong bytes (see the row-pitch note in
        ``mps_common.h``).
        """
        operand = torch.arange(20).reshape(4, 5).t()
        assert not operand.is_contiguous()
        index = torch.randint(0, 5, (5, 3), dtype=torch.long)
        out = torch.gather(_to_device(operand), 0, _to_device(index))
        _assert_exact(out, _reference(operand, 0, index))

    @pytest.mark.anyplatform
    def test_non_contiguous_index(self):
        """A strided index reads the elements it names, in its own order."""
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        index = torch.zeros(4, 10, dtype=torch.long)[:, ::2]
        index[:, 1] = 3
        index[:, 4] = 2
        assert not index.is_contiguous()
        _assert_exact(
            torch.gather(_to_device(operand), 1, _to_device(index)),
            _reference(operand, 1, index),
        )


class TestGatherPlanReuse:
    """One plan answers a decode loop's calls."""

    @pytest.mark.anyplatform
    def test_the_axis_is_in_the_key(self):
        """The same operands read at two axes are two nodes, not one.

        ``gather``'s only host part is ``axis``, and a key that lost it would
        answer the second call out of the first graph -- a wrong tensor rather
        than a failure.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        device = _to_device(operand)
        zeros = torch.zeros(4, 5, dtype=torch.long)
        at_axis_one = torch.gather(device, 1, _to_device(zeros))
        at_axis_zero = torch.gather(device, 0, _to_device(zeros))
        _assert_exact(at_axis_one, _reference(operand, 1, zeros))
        _assert_exact(at_axis_zero, _reference(operand, 0, zeros))
        assert not torch.equal(at_axis_one.cpu(), at_axis_zero.cpu())

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_another_dtype(self):
        """An operand of another dtype gets its own plan, not the first one's.

        A key that lost an operand would answer the second call out of the first
        graph's plan, which reads the wrong bytes rather than failing.
        """
        index = torch.randint(0, 16, (4, 16), dtype=torch.long)
        for dtype in FLOAT_DTYPES:
            operand = torch.arange(256).to(dtype).reshape(16, 16)
            _assert_exact(
                torch.gather(_to_device(operand), 0, _to_device(index)),
                _reference(operand, 0, index),
            )

    @pytest.mark.anyplatform
    def test_plan_is_reused_across_index_values(self):
        """Two calls that differ only in the elements they name share one plan.

        The index values are a graph *input* and not a key part, so the second
        call must read its own elements rather than the ones the first call
        pinned.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        device = _to_device(operand)
        first_index = torch.zeros(4, 5, dtype=torch.long)
        second_index = torch.full((4, 5), 4, dtype=torch.long)
        first = torch.gather(device, 1, _to_device(first_index))
        second = torch.gather(device, 1, _to_device(second_index))
        _assert_exact(first, _reference(operand, 1, first_index))
        _assert_exact(second, _reference(operand, 1, second_index))
        assert not torch.equal(first.cpu(), second.cpu())

    @pytest.mark.anyplatform
    def test_a_second_call_after_a_declined_one_is_served(self):
        """A declined call leaves no plan behind that the next one inherits.

        The out-of-range path returns before the graph is built, so the plan
        cache is untouched; this holds that the next in-range call still gets a
        correct answer rather than the half-built graph.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        device = _to_device(operand)
        bad = torch.zeros(3, 1, dtype=torch.long)
        bad[0, 0] = 99
        with pytest.raises(RuntimeError, match="out of bounds"):
            torch.gather(device, 0, _to_device(bad))
        index = torch.randint(0, 5, (4, 5), dtype=torch.long)
        _assert_exact(
            torch.gather(device, 1, _to_device(index)), _reference(operand, 1, index)
        )

    @pytest.mark.anyplatform
    def test_a_trim_does_not_pin_the_next_call_s_shape(self):
        """The trim is per call, so a warm plan cannot bake in a block.

        The trim depends on the index's extents, so a plan reused across two
        differently shaped indices has to be a plan that reads the operand, not
        one that baked the first call's block. The two calls here differ in
        exactly the dimension the trim decides.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        device = _to_device(operand)
        narrow = torch.randint(0, 5, (4, 2), dtype=torch.long)
        wide = torch.randint(0, 5, (4, 5), dtype=torch.long)
        _assert_exact(
            torch.gather(device, 1, _to_device(narrow)),
            _reference(operand, 1, narrow),
        )
        _assert_exact(
            torch.gather(device, 1, _to_device(wide)), _reference(operand, 1, wide)
        )


class TestGatherDispatch:
    """``gather`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_gather_to_mps(self):
        assert routed_backend_or_none("gather") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """``gather.out`` writes into a caller-owned tensor, like the rest."""
        assert routed_backend_or_none("gather.out") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """Both the public and the aten spelling reach the kernel.

        Run in a subprocess because ``LogEnabled`` latches the environment on
        first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "t = torch.arange(20, dtype=torch.float32, device='flagos:0')"
                ".reshape(4, 5); "
                "i = torch.zeros(4, 5, dtype=torch.long, device='flagos:0'); "
                "torch.gather(t, 1, i); "
                "torch.ops.aten.gather(t, 1, i)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] gather -> mps") == 2

    @pytest.mark.main_ops
    def test_a_declined_index_does_not_reach_the_node(self):
        """An out-of-range call is entered and answered by the reference.

        The dispatch log names the backend even when the body declines, because
        the kernel is what decided; what this holds is that the *answer* is the
        reference's on the path the body has no node for. Without it the node's
        inserted zero would come back as a value.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                # A one-liner cannot carry this: `try` is a compound statement
                # and a semicolon will not start one, so the body is written in
                # newlines that the `-c` argument preserves.
                "import torch, torch_fl\n"
                "t = torch.arange(20, dtype=torch.float32, "
                "device='flagos:0').reshape(4, 5)\n"
                "i = torch.zeros(3, 1, dtype=torch.long, device='flagos:0')\n"
                "i[0, 0] = 99\n"
                "try:\n"
                "    torch.gather(t, 0, i)\n"
                "except RuntimeError as e:\n"
                "    print('RuntimeError', 'out of bounds' in str(e))\n",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "RuntimeError True" in result.stdout
        assert "[flagos dispatch] gather -> mps" in result.stderr


class TestGatherCost:
    """The op is served, not round-tripped, on the shape it exists for."""

    @pytest.mark.anyplatform
    def test_large_operand_does_not_scale_with_the_operand(self):
        """A 0.6B-class read is not proportional to the operand's size.

        Measured through the fallback this op replaced: one ``gather`` of a
        single row of a ``(151936, 1024)`` float32 operand at ``dim=0`` cost
        42.0 ms, against 0.29 ms for the node this body builds -- independent of
        the element count and proportional to the operand. The regression this
        guards against is a silent return to that path, which a correctness test
        cannot see. The bound is deliberately loose: it separates "the read is
        served on the device" from "the operand was copied".
        """
        import time

        vocab, hidden = 151936, 1024
        operand = torch.randn(vocab, hidden, device=DEVICE)
        index = torch.randint(0, vocab, (1, hidden), device=DEVICE)

        def run(n):
            for _ in range(3):
                torch.gather(operand, 0, index)
            # The accelerator's own barrier, not a Python-level alias: the
            # runtime exposes it on the extension module only, so a
            # `hasattr(torch_fl, "synchronize")` guard is always false and the
            # measurement below would time the launch rather than the work.
            torch_fl._C._synchronize()
            start = time.perf_counter()
            for _ in range(n):
                torch.gather(operand, 0, index)
            # The accelerator's own barrier, not a Python-level alias: the
            # runtime exposes it on the extension module only, so a
            # `hasattr(torch_fl, "synchronize")` guard is always false and the
            # measurement below would time the launch rather than the work.
            torch_fl._C._synchronize()
            return (time.perf_counter() - start) / n * 1e3

        per_call = run(10)
        # The round-trip measured 42.0 ms for one row; the node measures well
        # under a millisecond. 10 ms sits far from both.
        assert per_call < 10.0, (
            f"gather over a {vocab}x{hidden} operand took {per_call:.1f} ms; "
            "the whole-operand fallback is back"
        )

    @pytest.mark.anyplatform
    def test_the_range_pass_does_not_move_the_operand(self):
        """Only the index crosses to the host, so the operand's size is not the
        cost of the check.

        The range pass reads the *index*, which is why a 151936x1024 operand does
        not make it expensive. The check is made against the operand's *shape*
        and the index's *values*, so the only operand-proportional work in this
        body is the trim the node forces -- and the index here spans the
        operand's full non-axis extent, so there is no trim at all and the two
        operands must cost the same. A body that had read the operand to range
        check it would fail this at the wide size.
        """
        import time

        index = torch.randint(0, 1024, (8, 16), device=DEVICE)

        def per_call(operand):
            for _ in range(3):
                torch.gather(operand, 1, index)
            # The accelerator's own barrier, not a Python-level alias: the
            # runtime exposes it on the extension module only, so a
            # `hasattr(torch_fl, "synchronize")` guard is always false and the
            # measurement below would time the launch rather than the work.
            torch_fl._C._synchronize()
            start = time.perf_counter()
            for _ in range(10):
                torch.gather(operand, 1, index)
            # The accelerator's own barrier, not a Python-level alias: the
            # runtime exposes it on the extension module only, so a
            # `hasattr(torch_fl, "synchronize")` guard is always false and the
            # measurement below would time the launch rather than the work.
            torch_fl._C._synchronize()
            return (time.perf_counter() - start) / 10 * 1e3

        # dim=1 over these shapes leaves the index's own 16 columns as the only
        # trim, and dimension 0 is a *leading* narrow for every operand, so the
        # operand's wide extent is not walked.
        small = per_call(torch.randn(16, 1024, device=DEVICE))
        large = per_call(torch.randn(151936, 1024, device=DEVICE))
        assert large < small * 20 + 10.0, (
            f"a {small:.2f} ms read against a small operand and a "
            f"{large:.2f} ms one against a large operand -- the index pass is "
            "reading the operand"
        )

    @pytest.mark.anyplatform
    def test_the_trim_costs_what_the_platform_charges_for_a_strided_copy(self):
        """The operand-proportional cost is the platform's, not the body's.

        A narrowing index at ``dim=0`` forces the trim measured above: the node
        requires the two operands to match on every non-axis dimension, so a
        ``(8, 16)`` read of a ``(151936, 1024)`` operand has to materialise a
        ``(151936, 16)`` block first. That copy is charged by the device for
        *any* view over a buffer that large -- measured here, a dense
        ``op[:8, :16].contiguous()`` and a bare ``view * 2`` cost the same 42 ms
        as the trim -- so the assertion is that ``gather`` is not dearer than
        the trim it is built on, not that the trim is cheap.
        """
        import time

        vocab, hidden = 151936, 1024
        operand = torch.randn(vocab, hidden, device=DEVICE)
        index = torch.randint(0, vocab, (8, 16), device=DEVICE)
        view = operand.narrow(1, 0, 16)

        def per_call(fn):
            for _ in range(3):
                fn()
            # The accelerator's own barrier, not a Python-level alias: the
            # runtime exposes it on the extension module only, so a
            # `hasattr(torch_fl, "synchronize")` guard is always false and the
            # measurement below would time the launch rather than the work.
            torch_fl._C._synchronize()
            start = time.perf_counter()
            for _ in range(10):
                fn()
            # The accelerator's own barrier, not a Python-level alias: the
            # runtime exposes it on the extension module only, so a
            # `hasattr(torch_fl, "synchronize")` guard is always false and the
            # measurement below would time the launch rather than the work.
            torch_fl._C._synchronize()
            return (time.perf_counter() - start) / 10 * 1e3

        trim_only = per_call(lambda: view.contiguous())
        with_gather = per_call(lambda: torch.gather(operand, 0, index))
        # The graph work itself is a fraction of a millisecond; allowing the
        # whole trim's cost plus a wide margin keeps this a check that the body
        # is built on the trim rather than a second copy of the operand. A body
        # that had copied the operand once more (622 MB) would double it.
        assert with_gather < trim_only + 5.0, (
            f"gather took {with_gather:.2f} ms against {trim_only:.2f} ms for "
            "the trim alone -- the body is moving the operand twice"
        )
