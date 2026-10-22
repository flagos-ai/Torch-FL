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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``take``.

``take`` is the third member of the ``embedding`` read family, and the reason
this file is not a copy of its two siblings' is that the op looks similar and is
not. ``index_select`` replaces one axis and keeps the operand's shape; ``take``
discards the operand's structure entirely and adopts the *index's* shape. That
one sentence is what every case below is built around, because the two ops part
company on it in three separate places:

* **The index's rank is the result's rank.** ``index_select`` rejects a
  higher-rank index outright -- ``Index is supposed to be a vector`` -- where
  ``take`` reads the same ``(2, 2)`` index against a flat buffer and returns a
  ``(2, 2)`` tensor. The two cases are held side by side for that reason: a body
  that reused ``index_select``'s shape rule would answer one of them wrongly.
* **A negative index is in range.** Measured, ``take(t, [-1])`` reads the last
  element where ``index_select`` raises ``IndexError``. This was a real bug in
  the first version of this body, and it is the one finding worth recording: the
  range check was written for ``index_select``'s contract (`[0, numel)`), so it
  let ``-1`` through to the graph, and the graph read a negative index as the
  *first* element rather than the last -- `take(t, [-1, 0, 19])` returned
  `[0, 0, 19]` against the reference's `[19, 0, 19]`. A plausible wrong value,
  not a crash, and it survives every positive-index case. The fix carries two
  halves that the cases below separate: the index is range-checked against
  `[-numel, numel)` *and* normalised to non-negative on the host before the graph
  ever sees it. A body that did only the first half passes
  `test_negative_index_is_in_range` and fails
  `test_negative_index_reads_the_element_the_reference_reads`.
* **The index dtype is int64 alone.** ``aten::take`` raises ``RuntimeError:
  take(): Expected a long tensor for index, but got Int`` for an int32 index
  where the two siblings accept it, which is why this body has its own dtype
  predicate rather than reusing the embedding one.

The rest of the file holds what the family shares: the read is exact, the range
contract is the reference's, an unsupported dtype takes the round-trip, one plan
answers a decode loop's shapes, and the op is served on the shape it exists for.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
"""

import os
import subprocess
import sys

import pytest
from device_init import torch

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

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Every dtype the graph is served on. `take` is a pure read, so unlike the
# arithmetic families it passes its operand's width through rather than
# narrowing to the float widths.
READABLE_DTYPES = FLOAT_DTYPES + [torch.int32, torch.int16, torch.int8, torch.uint8]


def _reference(self: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``at::take`` on the host, in the operand's own dtype.

    The reference the kernel is held to, and deliberately the same op the body
    round-trips to rather than a hand-built ``self.flatten()[index]``: the two
    host passes the body makes exist to hand the caller *this op's* answer, so a
    test written against anything else would be holding a different contract for
    the negative-index and out-of-range cases.
    """
    return torch.ops.aten.take(self, index)


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


class TestTakeShape:
    """The result is the index's shape, over a flat reading of the operand."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("operand_shape", [(20,), (4, 5), (2, 2, 5)])
    def test_the_operand_structure_is_discarded(self, operand_shape):
        """A rank-1, rank-2 and rank-3 operand of one size read identically.

        ``take`` flattens its operand, so these three calls name the same 20
        elements and must return the same values. This is where the op parts
        company with ``index_select``, which would keep the operand's trailing
        dimensions; a body that had reused that shape rule answers this one with
        a rank-3 tensor.
        """
        operand = torch.arange(20, dtype=torch.float32).reshape(operand_shape)
        index = torch.tensor([0, 7, 19, 3], dtype=torch.long)
        out = torch.take(_to_device(operand), _to_device(index))
        # The reference, not the operand: `aten::take` returns `(4,)` for all
        # three operand ranks because the result's shape is the index's.
        assert out.shape == (4,)
        _assert_exact(out, _reference(operand, index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "index",
        [
            torch.tensor([0, 7, 19, 3, 1, 2], dtype=torch.long),
            torch.tensor([[0, 7], [19, 3]], dtype=torch.long),
            torch.tensor([[[0, 7], [19, 3]]], dtype=torch.long),
            torch.tensor([[0, 7, 19, 3, 1, 2]], dtype=torch.long),
            torch.tensor([[[0], [7]], [[19], [3]], [[1], [2]]], dtype=torch.long),
        ],
        ids=["rank1", "rank2", "rank3", "row", "column"],
    )
    def test_the_index_shape_is_the_result_shape(self, index):
        """Every index rank comes back as itself, including the rank-2 case
        ``index_select`` rejects.

        ``index_select`` takes a vector and says so, where ``take`` reads any
        index shape as a flat list. The body builds that shape with a second
        reshape whose input carries the operand's trailing 1, so an index rank a
        reshape dropped or reordered would show here as a shape mismatch rather
        than as a wrong value.

        The six elements are shuffled rather than ``torch.arange`` and the two
        lopsided cases (one row, one column) are here because a reshape that
        had flattened and rebuilt the shape would answer a square index
        correctly and a lopsided one with the values in the wrong places.
        """
        operand = torch.arange(20, dtype=torch.float32)
        out = torch.take(_to_device(operand), _to_device(index))
        assert out.shape == index.shape
        _assert_exact(out, _reference(operand, index))

    @pytest.mark.anyplatform
    def test_the_rank_the_two_siblings_disagree_about(self):
        """A ``(2, 2)`` index is a batch for one op and the shape for the other.

        The single case that separates the two bodies' shape rules, held because
        a future refactor that folded the three read ops onto one template would
        pass every other case in this file and fail here.
        """
        operand = torch.arange(6, dtype=torch.float32)
        index = torch.tensor([[5, 0], [2, 1]], dtype=torch.long)
        taken = torch.take(_to_device(operand), _to_device(index))
        assert taken.shape == (2, 2)
        _assert_exact(taken, _reference(operand, index))
        with pytest.raises(IndexError, match="vector"):
            torch.index_select(_to_device(operand), 0, _to_device(index))

    @pytest.mark.anyplatform
    def test_rank_zero_operand(self):
        """A scalar operand has one element and the index names it.

        ``take`` reads ``numel`` elements, so a rank-0 operand is one element and
        is not an error -- unlike ``index_select``, which rejects a rank-0
        operand because it has no axis 0.
        """
        operand = torch.tensor(7.0)
        index = torch.tensor([0], dtype=torch.long)
        out = torch.take(_to_device(operand), _to_device(index))
        assert out.shape == (1,)
        _assert_exact(out, _reference(operand, index))

    @pytest.mark.anyplatform
    def test_empty_index(self):
        """No indices is a valid call and an empty result, not a decline.

        MPSGraph rejects an empty tensor, so the body short-circuits this shape
        before it builds a node; the result is a read of zero elements.
        """
        operand = torch.arange(20, dtype=torch.float32)
        index = torch.empty(0, dtype=torch.long)
        out = torch.take(_to_device(operand), _to_device(index))
        assert out.shape == (0,)
        _assert_exact(out, _reference(operand, index))

    @pytest.mark.anyplatform
    def test_repeated_and_reordered_indices(self):
        """The index is a list, not a permutation."""
        operand = torch.arange(20, dtype=torch.float32)
        index = torch.tensor([19, 0, 19, 7, 7, 1], dtype=torch.long)
        _assert_exact(
            torch.take(_to_device(operand), _to_device(index)),
            _reference(operand, index),
        )


class TestTakeCorrectness:
    """``take`` on the flagos device equals the CPU reference exactly."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matches_cpu_over_the_float_widths(self, dtype):
        operand = torch.randn(32, 8).to(dtype)
        index = torch.randint(0, 256, (6,)).to(torch.long)
        _assert_exact(
            torch.take(_to_device(operand), _to_device(index)),
            _reference(operand, index),
        )

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", READABLE_DTYPES)
    def test_the_operand_width_passes_through(self, dtype):
        """A read does not change its operand's width, and it does not round it.

        The gate here is deliberately wider than the embedding gate: ``take``
        has no arithmetic for a width to change under, so an int8 or uint8
        operand computed on the graph is servable -- and is what a caller's
        ``embedding_bag_forward`` style index path may well hold. Each width is
        filled with values a float cast would round, so a body that staged the
        operand through float32 would show here.
        """
        operand = torch.arange(256).to(dtype)
        index = torch.tensor([0, 1, 200, 255], dtype=torch.long)
        out = torch.take(_to_device(operand), _to_device(index))
        assert out.dtype == dtype
        _assert_exact(out, _reference(operand, index))

    @pytest.mark.anyplatform
    def test_non_contiguous_operand(self):
        """A strided operand is materialised before the graph reads it.

        The body flattens its operand for the graph, so a buffer MpsRunGraph read
        densely would read the wrong bytes for a strided operand (see the
        row-pitch note in ``mps_common.h``). ``(4, 5).t()`` is a non-contiguous
        ``(5, 4)`` -- and because ``take`` flattens, this is the one read op where
        the *logical* order of that buffer, not its storage order, decides the
        answer.
        """
        operand = torch.arange(20).reshape(4, 5).t()
        index = torch.tensor([0, 5, 19, 12], dtype=torch.long)
        assert not operand.is_contiguous()
        out = torch.take(_to_device(operand), _to_device(index))
        _assert_exact(out, _reference(operand, index))

    @pytest.mark.anyplatform
    def test_non_contiguous_index(self):
        """A strided index list reads the elements it names, in its own order."""
        operand = torch.arange(20, dtype=torch.float32)
        index = torch.arange(20).flip(0)[::3]
        assert not index.is_contiguous()
        _assert_exact(
            torch.take(_to_device(operand), _to_device(index)),
            _reference(operand, index),
        )

    @pytest.mark.anyplatform
    def test_last_element_is_in_range(self):
        """The boundary the normalisation is written against is inclusive."""
        operand = torch.arange(20, dtype=torch.float32)
        index = torch.tensor([19, 0], dtype=torch.long)
        _assert_exact(
            torch.take(_to_device(operand), _to_device(index)),
            _reference(operand, index),
        )


class TestTakeNegativeIndex:
    """A negative index is in range, and it must read what the reference reads."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "index",
        [[-1], [-20], [-1, -20, 0], [-1, 19, -2], [-20, -1, -19, 3]],
    )
    def test_negative_index_is_in_range(self, index):
        """A negative index does not raise, unlike ``index_select``'s.

        The first half of the fix, and on its own the weaker half: the check has
        to admit ``[-numel, numel)`` rather than ``[0, numel)``. Held against the
        reference rather than against "it did not raise", so a body that admitted
        the value and then dropped it would still fail.
        """
        operand = torch.arange(20, dtype=torch.float32)
        names = torch.tensor(index, dtype=torch.long)
        _assert_exact(
            torch.take(_to_device(operand), _to_device(names)),
            _reference(operand, names),
        )

    @pytest.mark.anyplatform
    def test_negative_index_reads_the_element_the_reference_reads(self):
        """The case the first version of this body failed.

        Measured through that version: ``take(t, [-1, 0, 19])`` over a
        20-element operand returned ``[0, 0, 19]`` where the reference returns
        ``[19, 0, 19]`` -- the graph read the negative index as the *first*
        element. The check admitted the index (correctly) and the graph was handed
        it unchanged (incorrectly), so the bug is not in the range test and this
        is not the same case as the one above: a body that range-checks against
        ``[-numel, numel)`` and passes the original index through passes that case
        and fails this one. What makes it fail is the normalisation, and what makes
        it visible is asking for an element that is not the first.

        The index is deliberately ``[-1]`` alone rather than ``[-1, ...]``: the
        wrong answer and the right one differ in the very first element, so an
        off-by-one in the wrap cannot be masked by a later element agreeing.
        """
        operand = torch.arange(20, dtype=torch.float32)
        index = torch.tensor([-1], dtype=torch.long)
        out = torch.take(_to_device(operand), _to_device(index))
        _assert_exact(out, _reference(operand, index))
        # Stated outright, because this is the value the bug returned: the
        # reference and therefore the kernel must say 19, not 0.
        assert out.cpu().item() == 19.0

    @pytest.mark.anyplatform
    def test_a_negative_index_and_its_positive_spelling_are_one_read(self):
        """``-1`` and ``19`` are the same element, and the body agrees.

        The wrap is a normalisation on the host, so the two spellings reach the
        graph as the same index buffer. This holds that the *values* agree, which
        a body that wrapped on the wrong side of the `numel` would break --
        ``-1 + 20`` is 19 and ``-20 + 20`` is 0, so the two ends of the range are
        the two cases a wrap with an off-by-one gets wrong.
        """
        operand = torch.arange(20, dtype=torch.float32)
        negative = torch.tensor([-1, -20], dtype=torch.long)
        positive = torch.tensor([19, 0], dtype=torch.long)
        at_negative = torch.take(_to_device(operand), _to_device(negative))
        at_positive = torch.take(_to_device(operand), _to_device(positive))
        assert torch.equal(at_negative.cpu(), at_positive.cpu())
        _assert_exact(at_negative, _reference(operand, negative))

    @pytest.mark.anyplatform
    def test_the_wrap_covers_the_int8_widths_operand_too(self):
        """The normalisation reads the operand's size, not its dtype.

        ``numel`` is a property of the shape, so the two ends of a byte-valued
        operand wrap the same way a float one does. Held because the range pass
        and the operand pass are separate statements in the body, and a
        ``numel`` taken from the wrong tensor -- the index's, say -- is exactly
        the kind of rewrite that reads correctly.
        """
        operand = torch.arange(6).to(torch.int8)
        index = torch.tensor([-1, -6, 5, 0], dtype=torch.long)
        _assert_exact(
            torch.take(_to_device(operand), _to_device(index)),
            _reference(operand, index),
        )


class TestTakeRangeCheck:
    """An index the operand cannot answer reaches the reference's own error."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("bad_index", [20, 100, -21, -100])
    def test_out_of_range_index_raises(self, bad_index):
        """``IndexError``, not the graph's silent first element.

        The bound is two-sided: ``20`` is past the end, ``-21`` is past the
        beginning, and the reference's own message names the value
        (``out of range: tried to access index -21 on a tensor of 20 elements``).
        The negative spelling is the one a check written for ``index_select``
        would get wrong by raising where the reference is happy, and the positive
        one is what a check written only for the negative case would miss.
        """
        operand = torch.arange(20, dtype=torch.float32)
        index = torch.tensor([bad_index], dtype=torch.long)
        with pytest.raises(IndexError):
            torch.take(_to_device(operand), _to_device(index))

    @pytest.mark.anyplatform
    def test_one_bad_index_among_good_ones_raises(self):
        """The check is over the whole tensor, not just the first element.

        ``[19, 20, 0]`` is one past-the-end index inside a valid list, and one
        negative member of the same list is also checked.
        """
        operand = torch.arange(20, dtype=torch.float32)
        with pytest.raises(IndexError):
            torch.take(
                _to_device(operand),
                _to_device(torch.tensor([19, 20], dtype=torch.long)),
            )
        with pytest.raises(IndexError):
            torch.take(
                _to_device(operand),
                _to_device(torch.tensor([0, -21], dtype=torch.long)),
            )

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "index_dtype",
        [
            torch.int32,
            torch.int16,
            torch.int8,
            torch.bool,
            torch.float32,
            torch.float64,
        ],
    )
    def test_a_non_int64_index_raises(self, index_dtype):
        """Every index but int64 is the reference's ``RuntimeError``.

        This is the op's sharpest difference from its two siblings: ``take``
        accepts *only* int64 (``take(): Expected a long tensor for index, but got
        Int``), where ``index_select`` and ``embedding`` both take int32. The
        narrower integers are the interesting members -- they are integers the
        graph could be handed, and a body that reused the embedding gate would
        widen them and answer where the reference fails. The check is the
        dedicated predicate, so the message the caller sees is aten's own.
        """
        operand = torch.arange(20, dtype=torch.float32)
        index = torch.tensor([1, 0]).to(index_dtype)
        with pytest.raises(RuntimeError, match="long tensor"):
            torch.take(_to_device(operand), _to_device(index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.bool])
    def test_declined_operand_dtypes_keep_the_reference(self, dtype):
        """A width MPSGraph has no element type for takes the round-trip.

        float64, int64 and bool are the three the graph cannot describe, and each
        is a dtype ``take`` passes through unchanged. The case asserts the values
        too, so a body that ran the graph anyway would have to also be right about
        a width it cannot represent.
        """
        base = torch.arange(20, dtype=torch.float32)
        operand = (base > 10) if dtype is torch.bool else base.to(dtype)
        index = torch.tensor([0, 19, 3], dtype=torch.long)
        out = torch.take(_to_device(operand), _to_device(index))
        assert out.dtype == dtype
        _assert_exact(out, _reference(operand, index))


class TestTakePlanReuse:
    """One plan answers the shapes a decode loop presents."""

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_a_different_dtype(self):
        """An operand of another dtype gets its own plan, not the first one's.

        ``take`` has no host constant at all -- its key is the operand signature
        MpsRunGraph folds in -- so the dtype is the only thing separating a
        float32 call from a float16 one. A key that lost an operand would answer
        the second call out of the first graph's plan, which reads the wrong bytes
        rather than failing.
        """
        index = torch.tensor([0, 5, 11, 14], dtype=torch.long)
        for dtype in FLOAT_DTYPES:
            operand = torch.arange(16).to(dtype)
            _assert_exact(
                torch.take(_to_device(operand), _to_device(index)),
                _reference(operand, index),
            )

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_another_index_shape(self):
        """The index's shape is in the key, so a reshaped index is a new plan.

        A ``(4,)`` index and a ``(2, 2)`` one over the same operand are two
        different results with two different shapes, and the operands cannot say
        which was asked for.
        """
        operand = torch.arange(16)
        device = _to_device(operand)
        flat = torch.tensor([0, 5, 11, 14], dtype=torch.long)
        square = torch.tensor([[0, 5], [11, 14]], dtype=torch.long)
        at_flat = torch.take(device, _to_device(flat))
        at_square = torch.take(device, _to_device(square))
        assert at_flat.shape == (4,)
        assert at_square.shape == (2, 2)
        _assert_exact(at_flat, _reference(operand, flat))
        _assert_exact(at_square, _reference(operand, square))

    @pytest.mark.anyplatform
    def test_plan_is_reused_across_index_values(self):
        """Two calls that differ only in the elements they name share one plan.

        The index values are a graph *input* and not a key part, so the second
        call must read its own elements rather than the ones the first call
        pinned. Both a positive and a negative spelling are used, since the
        normalisation happens per call and must not be cached with the plan.
        """
        operand = torch.arange(20, dtype=torch.float32)
        device = _to_device(operand)
        first_index = torch.tensor([1, 2, 3], dtype=torch.long)
        second_index = torch.tensor([19, -20, -1], dtype=torch.long)
        first = torch.take(device, _to_device(first_index))
        second = torch.take(device, _to_device(second_index))
        _assert_exact(first, _reference(operand, first_index))
        _assert_exact(second, _reference(operand, second_index))
        assert not torch.equal(first.cpu(), second.cpu())

    @pytest.mark.anyplatform
    def test_a_second_call_after_a_declined_one_is_served(self):
        """A declined call leaves no plan behind that the next one inherits.

        The out-of-range path returns before the graph is built, so the plan
        cache is untouched; this holds that the next in-range call still gets a
        correct answer rather than the half-built graph or the previous index.
        """
        operand = torch.arange(20, dtype=torch.float32)
        device = _to_device(operand)
        with pytest.raises(IndexError):
            torch.take(device, _to_device(torch.tensor([20], dtype=torch.long)))
        index = torch.tensor([-1, 3], dtype=torch.long)
        _assert_exact(torch.take(device, _to_device(index)), _reference(operand, index))


class TestTakeDispatch:
    """``take`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_take_to_mps(self):
        assert routed_backend_or_none("take") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """``take.out`` writes into a caller-owned tensor, like the rest."""
        assert routed_backend_or_none("take.out") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """Both the public and the aten spelling reach the kernel.

        ``torch.take`` is what a caller writes and ``aten.take`` is what a
        decomposition would lower to; both are held so a future change that made
        the public form composite would show up here. Run in a subprocess because
        ``LogEnabled`` latches the environment on first use. The index is
        negative on purpose: the log is a *dispatch* record and must be present
        for a call the body normalises rather than declines.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from device_init import torch, torch_fl; "
                "t = torch.arange(32, dtype=torch.float32, device='flagos:0'); "
                "i = torch.tensor([-1, 0, 31], device='flagos:0'); "
                "torch.take(t, i); "
                "torch.ops.aten.take(t, i)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] take -> mps") == 2

    @pytest.mark.main_ops
    def test_a_declined_index_does_not_reach_the_graph(self):
        """An out-of-range call is entered and answered by the reference.

        The dispatch log names the backend even when the body declines, because
        the kernel is what decided; what this holds is that the *answer* is the
        reference's on the path the body has no node for.
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
                "from device_init import torch, torch_fl\n"
                "t = torch.arange(20, dtype=torch.float32, device='flagos:0')\n"
                "i = torch.tensor([20], device='flagos:0')\n"
                "try:\n"
                "    torch.take(t, i)\n"
                "except IndexError as e:\n"
                "    print('IndexError', 'out of range' in str(e))\n",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "IndexError True" in result.stdout
        assert "[flagos dispatch] take -> mps" in result.stderr


class TestTakeCost:
    """The op is served, not round-tripped, on the shape it exists for."""

    @pytest.mark.anyplatform
    def test_large_operand_does_not_scale_with_the_operand(self):
        """A 0.6B-class read is served, ten times over, not round-tripped once.

        The whole-operand round trip this op replaced copies a ``(151936, 1024)``
        float32 operand to the host and back on every call -- measured at 45.6 ms
        for a *single* element and 46.3 ms for 512, i.e. independent of the
        element count and proportional to the operand, against 0.29 ms for the
        nodes this body builds. The regression this guards against is a silent
        return to that path, which a correctness test cannot see: a round-tripped
        read returns the right elements.

        What catches it is the dispatch log rather than the clock. The log line
        names the backend this module routed to and is emitted once per call, so
        ten reads that were all served log exactly ten ``-> mps`` lines. The
        timing bound this replaced measured 52 ms on CI against 0.3 ms locally
        *for calls that were being served* -- it failed on the runner for work
        the log count proves correct, and it would still miss a fast fallback.
        """
        vocab, hidden = 151936, 1024
        reads = 10
        served = dispatched_to_mps(
            """
from device_init import torch, torch_fl
operand = torch.randn(151936, 1024, device='flagos:0')
index = torch.randint(0, 151936, (512,), device='flagos:0')
torch.take(operand, index)
torch_fl._C._synchronize()
for _ in range(10):
    torch.take(operand, index)
torch_fl._C._synchronize()
""",
        )
        # The warmup logs too: it is the call a fallback would have answered by
        # copying the operand, so it is counted rather than subtracted.
        assert served == reads + 1, (
            f"take over a {vocab}x{hidden} operand logged {served} of "
            f"{reads + 1} reads onto the mps slot; the whole-operand fallback is back"
        )
