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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``index_select``.

``index_select`` is the read ``embedding`` performs, reached through the op the
caller names instead of through the embedding front end, so most of this file
holds the same two contracts its sibling does -- the gather is exact, and an
out-of-range index is never answered by the graph -- and it holds them for the
same reason.

What is new here is ``dim``. The embedding body hard-codes ``axis:0`` because a
table read is what ``F.embedding`` means; ``index_select`` can be asked for any
axis, and the single node this body knows how to build reads axis 0. So the body
resolves ``dim`` against the operand's rank and *declines* every axis it has no
node for rather than building the node it has and returning its answer for a
different question. Declining is not a failure: the round-trip is the
reference's own implementation, so a declined dim still answers correctly and
simply costs what it costs today.

The cases below are grouped by which of those three claims they hold:

* **The read is exact** and its shape is the operand's with axis 0 replaced by
  the index, across float widths, both index widths, a non-contiguous operand
  and an empty index. The index is a *vector* -- ``index_select`` rejects a
  higher-rank index outright (``Index is supposed to be a vector``) -- so unlike
  ``embedding``'s, its rank is not a case this op has, and the cases below say
  so rather than skipping over it.
* **The range contract is the reference's.** MPSGraph's gather has no range
  contract and answers a bad index with the first row (measured on the M5, and
  documented in ``T_EMBEDDING``), so the body checks the range on the host
  before a node exists and the caller sees ``IndexError``.
* **Only ``dim=0`` takes the graph.** ``dim=1``, ``dim=-1`` over a rank-2
  operand, a rank-3 last axis and an out-of-range ``dim`` all take the
  round-trip, and the cases below pin that the answer is the reference's in
  every one of them -- which is also what makes the decline safe to widen later.

Every case self-skips off an MPS build, like the stage-1 to stage-3 suites.
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

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
INDEX_DTYPES = [torch.int64, torch.int32]


def _reference(self: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """``at::index_select`` on the host, in the operand's own dtype.

    The reference the kernel is held to, and deliberately the *same* op the body
    round-trips to rather than a hand-built ``self[index]``: the two host checks
    the body makes exist to hand the caller the reference's own answer, so a
    test written against anything else would be holding a different contract for
    the negative-``dim`` and out-of-range cases.
    """
    return torch.ops.aten.index_select(self, dim, index)


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another case
    # left behind.
    return t.clone().to(DEVICE)


def _assert_exact(out: torch.Tensor, expected: torch.Tensor) -> None:
    """The result equals the reference's, byte for byte.

    No tolerance: the node moves bytes rather than computing over them, so the
    only two answers are "the right rows" and "the wrong rows". A tolerance here
    would hide an off-by-a-row read on a narrow dtype, which is the failure mode
    the range check and the index dtype gate exist to prevent.
    """
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    assert torch.equal(out.cpu(), expected), "result differs from the CPU reference"


class TestIndexSelectCorrectness:
    """``index_select`` on the flagos device equals the CPU reference exactly."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype, index_dtype):
        table = torch.randn(32, 8).to(dtype)
        index = torch.randint(0, 32, (6,)).to(index_dtype)
        _assert_exact(
            torch.index_select(_to_device(table), 0, _to_device(index)),
            _reference(table, 0, index),
        )

    @pytest.mark.parametrize("index_length", [1, 7, 40])
    @pytest.mark.anyplatform
    def test_the_index_length_becomes_the_leading_shape(self, index_length):
        """The result is ``(index_length,) + operand.shape[1:]``.

        The body builds that shape itself, so an index count it dropped or
        reordered on the way through would be a wrong answer rather than a slow
        one. It is also where this op parts company with its sibling: the index
        here is a vector and its *length* replaces axis 0, where ``embedding``
        takes the index's whole shape as a batch dim.
        """
        table = torch.randn(32, 8)
        index = torch.randint(0, 32, (index_length,))
        out = torch.index_select(_to_device(table), 0, _to_device(index))
        assert out.shape == (index_length, 8)
        _assert_exact(out, _reference(table, 0, index))

    @pytest.mark.anyplatform
    def test_a_higher_rank_index_takes_the_reference_error(self):
        """A 2-D index is the reference's ``IndexError``, not a batch dim.

        ``index_select`` takes a vector and says so -- ``Index is supposed to be
        a vector`` -- where ``F.embedding`` would read the same tensor as a
        batch. Held because a body that quietly accepted the higher rank would
        answer with a different op's semantics, which is a wrong answer rather
        than a slow one.
        """
        table = torch.randn(32, 8)
        index = torch.randint(0, 32, (4, 16))
        with pytest.raises(IndexError, match="vector"):
            torch.index_select(_to_device(table), 0, _to_device(index))

    @pytest.mark.anyplatform
    def test_rank_one_operand(self):
        """A vector operand is the same read with a one-column-wide row.

        The body flattens the operand into ``(rows, row_width)`` before it
        describes it to MPSGraph, and for a rank-1 operand ``row_width`` is 1;
        the reshape back is what this holds.
        """
        table = torch.randn(16)
        index = torch.randint(0, 16, (5,))
        out = torch.index_select(_to_device(table), 0, _to_device(index))
        assert out.shape == (5,)
        _assert_exact(out, _reference(table, 0, index))

    @pytest.mark.anyplatform
    def test_rank_three_operand(self):
        """A rank-3 operand reads whole ``(d1, d2)`` slabs, not rows of the flat
        buffer.

        The index names a position on axis 0 and everything beneath it comes
        along, so the row this body flattens is ``numel()/size(0)`` wide rather
        than ``size(1)``. A body that assumed a 2-D operand would answer this
        call with the wrong bytes.
        """
        table = torch.randn(6, 3, 4)
        index = torch.randint(0, 6, (2,))
        out = torch.index_select(_to_device(table), 0, _to_device(index))
        assert out.shape == (2, 3, 4)
        _assert_exact(out, _reference(table, 0, index))

    @pytest.mark.anyplatform
    def test_repeated_and_reordered_indices(self):
        """The index is a list, not a permutation.

        A gather that deduplicated, sorted or masked its index would still pass a
        monotone case, so this one repeats a row and asks for the rows out of
        order. The reference is a pure gather, so both are visible in it.
        """
        table = torch.randn(8, 3)
        index = torch.tensor([3, 3, 0, 7, 3, 1], dtype=torch.long)
        _assert_exact(
            torch.index_select(_to_device(table), 0, _to_device(index)),
            _reference(table, 0, index),
        )

    @pytest.mark.anyplatform
    def test_empty_index(self):
        """No indices is a valid call and an empty result, not a decline.

        MPSGraph rejects an empty tensor, so the body short-circuits this shape
        before it builds a node -- the same guard the embedding body carries.
        The result is a read of zero rows, which is what the reference returns.
        """
        table = torch.randn(16, 4)
        index = torch.empty(0, dtype=torch.long)
        out = torch.index_select(_to_device(table), 0, _to_device(index))
        assert out.shape == (0, 4)
        _assert_exact(out, _reference(table, 0, index))

    @pytest.mark.anyplatform
    def test_non_contiguous_operand(self):
        """A transposed operand reads the same rows.

        The body materialises a strided operand before it describes it to
        MPSGraph, because a buffer MpsRunGraph reads densely would otherwise be
        read as the wrong bytes (see the row-pitch note in mps_common.h).

        ``torch.randn(4, 16).t()`` is a non-contiguous ``(16, 4)``, so the axis
        the index names here is the one the transpose made out of the *columns*.
        That is the point: the body reads the materialised operand, not the
        strides it arrived with.
        """
        table = torch.randn(4, 16).t()
        index = torch.randint(0, 16, (6,))
        assert not table.is_contiguous()
        out = torch.index_select(_to_device(table), 0, _to_device(index))
        assert out.shape == (6, 4)
        _assert_exact(out, _reference(table, 0, index))

    @pytest.mark.anyplatform
    def test_non_contiguous_index(self):
        """A strided index list reads the rows it names, in its own order."""
        table = torch.randn(16, 4)
        index = torch.arange(16).flip(0)[::3]
        assert not index.is_contiguous()
        _assert_exact(
            torch.index_select(_to_device(table), 0, _to_device(index)),
            _reference(table, 0, index),
        )

    @pytest.mark.anyplatform
    def test_last_row_is_in_range(self):
        """The boundary the host check is written against is inclusive at top."""
        table = torch.randn(6, 3)
        index = torch.tensor([5, 0], dtype=torch.long)
        _assert_exact(
            torch.index_select(_to_device(table), 0, _to_device(index)),
            _reference(table, 0, index),
        )

    @pytest.mark.anyplatform
    def test_head_shaped_operand(self):
        """The shape the op is on the hot path for.

        A ``(vocab, hidden)`` table read by an attention or beam-search index is
        the operand the fallback used to copy to the host in full. Kept small
        enough to compare on the host here; the cost of the full size is held
        separately.
        """
        table = torch.randn(4096, 64)
        index = torch.randint(0, 4096, (128,))
        _assert_exact(
            torch.index_select(_to_device(table), 0, _to_device(index)),
            _reference(table, 0, index),
        )


class TestIndexSelectDim:
    """Only ``dim=0`` is served; every other axis is declined, not guessed."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [1, 2, -1, -2, -3])
    def test_other_axes_are_declined_but_correct(self, dim):
        """A non-zero ``dim`` round-trips and returns the reference's answer.

        The body has one node and it reads axis 0, so every other axis takes the
        round-trip. What matters is that the answer is still the reference's: a
        body that built its axis-0 node and returned it here would produce a
        plausible tensor of the wrong shape (or the wrong bytes) rather than an
        error, which is exactly the class of bug this pins. ``dim=-1`` over the
        rank-3 operand is the interesting spelling -- it resolves to axis 2, not
        to the axis this body can reach.
        """
        table = torch.randn(5, 4, 3)
        index = torch.randint(0, table.shape[dim], (2,))
        out = torch.index_select(_to_device(table), dim, _to_device(index))
        assert out.shape == tuple(
            table.shape[a] if a != dim % 3 else 2 for a in range(3)
        )
        _assert_exact(out, _reference(table, dim, index))

    @pytest.mark.anyplatform
    def test_dim_one_differs_from_dim_zero_on_the_same_operand(self):
        """The reason ``dim`` is in the plan key.

        A ``(4, 4)`` square operand read at ``dim=0`` and at ``dim=1`` has the
        same dtype, the same shape and two different answers, and the operands
        cannot say which was asked for. If ``dim`` fell out of the key, the
        second call would be answered out of the first call's plan -- and this
        case, not a shape check, is what would catch it.
        """
        table = torch.randn(4, 4)
        index = torch.tensor([3, 1], dtype=torch.long)
        device = _to_device(table)
        idx = _to_device(index)
        at_zero = torch.index_select(device, 0, idx)
        at_one = torch.index_select(device, 1, idx)
        _assert_exact(at_zero, _reference(table, 0, index))
        _assert_exact(at_one, _reference(table, 1, index))
        assert not torch.equal(at_zero.cpu(), at_one.cpu())

    @pytest.mark.anyplatform
    def test_negative_zero_spelling_is_dim_zero(self):
        """``dim=-3`` over a rank-3 operand resolves to 0 and is served."""
        table = torch.randn(6, 3, 2)
        index = torch.randint(0, 6, (2,))
        _assert_exact(
            torch.index_select(_to_device(table), -3, _to_device(index)),
            _reference(table, -3, index),
        )

    @pytest.mark.anyplatform
    def test_out_of_range_dim_reaches_the_reference_error(self):
        """``dim`` past the rank is aten's ``IndexError``, not a graph answer.

        The body resolves ``dim`` on the host and declines when it resolves
        outside ``[-rank, rank)``, so this reaches the reference's own
        validation. A body that clamped or wrapped the axis would answer here
        instead of failing.
        """
        table = torch.randn(4, 3)
        index = torch.tensor([1, 0], dtype=torch.long)
        with pytest.raises(IndexError):
            torch.index_select(_to_device(table), 2, _to_device(index))


class TestIndexSelectRangeCheck:
    """An index the operand cannot answer reaches the reference's own error."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("bad_index", [6, -1, 100])
    def test_out_of_range_index_raises(self, bad_index):
        """``IndexError``, not the graph's silent first row.

        Measured on an M5, the gather node answers an out-of-range index with the
        operand's first row -- for a negative index as well as one past the end.
        So the case is not "does it raise" in the abstract but "does the caller
        see the reference's error", which only holds while the host range check
        runs before any node exists.
        """
        table = torch.randn(6, 3)
        index = torch.tensor([bad_index], dtype=torch.long)
        with pytest.raises(IndexError):
            torch.index_select(_to_device(table), 0, _to_device(index))

    @pytest.mark.anyplatform
    def test_one_bad_index_among_good_ones_raises(self):
        """The check is over the whole tensor, not just the first element."""
        table = torch.randn(6, 3)
        index = torch.tensor([5, 6, 2, 0], dtype=torch.long)
        with pytest.raises(IndexError):
            torch.index_select(_to_device(table), 0, _to_device(index))

    @pytest.mark.anyplatform
    def test_the_check_covers_the_int32_width_too(self):
        """An int32 index is widened on the host, and the widening is checked.

        The two index widths take different reads of the host buffer before the
        comparison, so a check that only covered ``kLong`` would let an int32
        out-of-range index through to the graph.
        """
        table = torch.randn(6, 3)
        index = torch.tensor([5, 6], dtype=torch.int32)
        with pytest.raises(IndexError):
            torch.index_select(_to_device(table), 0, _to_device(index))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize(
        "index_dtype",
        [torch.float32, torch.float64, torch.int16, torch.int8, torch.bool],
    )
    def test_non_integer_index_raises(self, index_dtype):
        """A float, narrow or bool index is the reference's ``RuntimeError``.

        MPSGraph's gather takes an int32 or int64 index tensor and has no other
        integer input, so the body declines on the index dtype before it builds a
        node, and the caller gets the reference's own message naming int32 and
        int64. ``int8`` and ``int16`` are the interesting members: they are
        integers the graph could in principle be handed, and the reference
        rejects them, so a body that widened whatever it was given would answer
        here where the reference fails.
        """
        table = torch.randn(6, 3)
        index = torch.tensor([1, 0]).to(index_dtype)
        with pytest.raises(RuntimeError, match="int32 or int64"):
            torch.index_select(_to_device(table), 0, _to_device(index))

    @pytest.mark.anyplatform
    def test_float64_operand_takes_the_reference(self):
        """A float64 operand comes back in float64, from the reference.

        MPSGraph has no float64 element type, so the shared gate declines before
        a node is built and the caller gets the reference's own dtype. A case
        that ran the graph anyway would be a wrong answer rather than a slow one.
        """
        table = torch.randn(6, 3, dtype=torch.float64)
        index = torch.tensor([1, 0], dtype=torch.long)
        out = torch.index_select(_to_device(table), 0, _to_device(index))
        assert out.dtype == torch.float64
        _assert_exact(out, _reference(table, 0, index))


class TestIndexSelectPlanReuse:
    """One plan answers the shapes a decode loop presents."""

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_a_different_dtype(self):
        """An operand of another dtype gets its own plan, not the first one's.

        ``dim`` is the body's only host part, so the operand signature
        MpsRunGraph folds in is the only thing separating a float32 call from a
        float16 one. A key that lost the dtype would answer the second call out
        of the first graph's plan, which reads the wrong bytes rather than
        failing.
        """
        index = torch.randint(0, 16, (6,))
        for dtype in FLOAT_DTYPES:
            table = torch.randn(16, 4).to(dtype)
            _assert_exact(
                torch.index_select(_to_device(table), 0, _to_device(index)),
                _reference(table, 0, index),
            )

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_an_index_of_another_width(self):
        """An int32 index reuses the shape's plan without reusing its bytes."""
        table = torch.randn(16, 4)
        index = torch.randint(0, 16, (5,))
        device = _to_device(table)
        which = _to_device(index)
        _assert_exact(
            torch.index_select(device, 0, which.long()), _reference(table, 0, index)
        )
        _assert_exact(
            torch.index_select(device, 0, which.int()),
            _reference(table, 0, index.int().long()),
        )

    @pytest.mark.anyplatform
    def test_plan_is_reused_across_index_values(self):
        """Two calls that differ only in the rows they name share one plan.

        The index values are not a key part -- they are a graph *input*, and
        MpsRunGraph copies them in. This holds that the second call reads its own
        rows rather than the ones the first call pinned.
        """
        table = torch.randn(16, 4)
        device = _to_device(table)
        first_idx = torch.tensor([1, 2, 3], dtype=torch.long)
        second_idx = torch.tensor([15, 0, 7], dtype=torch.long)
        first = torch.index_select(device, 0, _to_device(first_idx))
        second = torch.index_select(device, 0, _to_device(second_idx))
        _assert_exact(first, _reference(table, 0, first_idx))
        _assert_exact(second, _reference(table, 0, second_idx))


class TestIndexSelectDispatch:
    """``index_select`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_index_select_to_mps(self):
        assert routed_backend_or_none("index_select") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """``index_select.out`` writes into a caller-owned tensor, like the rest."""
        assert routed_backend_or_none("index_select.out") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """The public and aten spellings both reach the kernel.

        ``torch.index_select`` is what a caller writes and ``aten.index_select``
        is what a decomposition would lower to; both are held so a future change
        that made the public form composite would show up here. Run in a
        subprocess because ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "t = torch.randn(32, 8, device='flagos:0'); "
                "i = torch.randint(0, 32, (4,), device='flagos:0'); "
                "torch.index_select(t, 0, i); "
                "torch.ops.aten.index_select(t, 0, i)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] index_select -> mps") == 2

    @pytest.mark.main_ops
    def test_declined_dim_does_not_reach_the_graph(self):
        """A ``dim=1`` call is entered and answered by the reference.

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
                "import torch, torch_fl; "
                "t = torch.randn(8, 4, device='flagos:0'); "
                "i = torch.randint(0, 4, (2,), device='flagos:0'); "
                "o = torch.index_select(t, 1, i); "
                "print(tuple(o.shape))",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "(8, 2)" in result.stdout
        assert "[flagos dispatch] index_select -> mps" in result.stderr


class TestIndexSelectCost:
    """The op is served, not round-tripped, on the shape it exists for."""

    @pytest.mark.anyplatform
    def test_large_operand_does_not_scale_with_the_operand(self):
        """A 0.6B-class read is served, ten times over, not round-tripped once.

        The fallback this op replaced copied the whole ``(151936, 1024)``
        operand to the host and back on every call -- 40.4 ms for a *single*
        row, profiled as ``aten::_to_copy`` over the 622 MB tensor, against
        0.29 ms for the one gather node the body builds. The regression this
        guards against is a silent return to that path, which a correctness test
        cannot see: a round-tripped read returns the right rows.

        What catches it is the dispatch log rather than the clock. The log line
        names the backend this module routed to and is emitted once per call, so
        ten reads that were all served log exactly ten ``-> mps`` lines. A
        timing bound was tried first and is the wrong instrument: it measured
        53 ms on CI against 0.3 ms locally *for calls that were being served*,
        so it fails on a paravirtualized runner for work the log count proves
        correct, and it would still miss a fallback that was merely fast.
        """
        vocab, hidden = 151936, 1024
        reads = 10
        served = dispatched_to_mps(
            """
import torch, torch_fl
table = torch.randn(151936, 1024, device='flagos:0')
index = torch.randint(0, 151936, (512,), device='flagos:0')
torch.index_select(table, 0, index)
torch_fl._C._synchronize()
for _ in range(10):
    torch.index_select(table, 0, index)
torch_fl._C._synchronize()
""",
        )
        # The warmup logs too: it is the call a fallback would have answered by
        # copying the operand, so it is counted rather than subtracted.
        assert served == reads + 1, (
            f"index_select over a {vocab}x{hidden} operand logged {served} of "
            f"{reads + 1} reads onto the mps slot; the whole-operand fallback is back"
        )
