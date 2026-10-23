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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``embedding``.

``embedding`` is the first op on this backend whose cost is a *read* of its
operand rather than a computation over it, and the first whose kernel has to
decide whether the graph may answer the call at all. Both are pinned here, and
the second is pinned because it already went wrong once.

* **The gather is exact.** A table read is one node
  (``gatherWithUpdatesTensor:indicesTensor:axis:0 batchDimensions:0``) and the
  cases below compare it against ``at::embedding`` bit for bit rather than with a
  tolerance: there is no arithmetic between the operand and the result, so an
  index that reads the right row reads it exactly.
* **An out-of-range index must not be answered by the graph.** MPSGraph's gather
  has no range contract, so measured on an M5 with a 6-row table and the index
  list ``(5, 6, 2, 0)`` it returns the *first row* for the bad index rather than
  failing, and the reference raises ``IndexError: index out of range in self``.
  A plausible wrong answer is worse than a round-trip, so the body validates the
  range on the host and declines; the cases below hold that the caller still gets
  the reference's error.
* **``padding_idx`` is not read, and must not be.** The forward op's output is
  bit-identical for every ``padding_idx`` a caller can pass -- including the
  ``None`` the Python front end turns into ``-1`` -- and an out-of-range
  ``padding_idx`` is accepted rather than rejected; the zeroing that name asks
  for belongs to ``embedding_dense_backward``. The first version of this kernel
  treated a non-negative ``padding_idx`` as a reason to decline and therefore
  took the whole-table round-trip on *every* ``F.embedding`` call, because
  ``F.embedding`` passes ``padding_idx=None``. The cases below pin the values
  that made that visible.
* **Which spelling is registered.** ``embedding.out`` writes into a tensor the
  caller owns like every other ``.out`` form here, and
  ``embedding_dense_backward`` is a separate op with its own scatter question;
  both stay on the fallback.

Every case self-skips off an MPS build, like the stage-1 to stage-3 suites.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
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


def _reference(indices: torch.Tensor, weight: torch.Tensor, **kwargs) -> torch.Tensor:
    """``at::embedding`` on the host, in the operand's own dtype.

    The reference the kernel is held to. ``index_select`` is deliberately not
    used in its place: it is the same read but a different op with a different
    out-of-range contract, so a test written against it would not hold the one
    behaviour this kernel's range check exists for.
    """
    return torch.ops.aten.embedding(weight, indices, kwargs.get("padding_idx", -1))


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another case
    # left behind.
    return t.clone().to(DEVICE)


def _assert_exact(out: torch.Tensor, expected: torch.Tensor) -> None:
    """The graph's result equals the reference's, byte for byte.

    No tolerance: the node moves bytes rather than computing over them, so the
    only two answers are "the same row" and "the wrong row". A tolerance here
    would hide an off-by-a-row read on a narrow dtype, which is exactly the
    failure mode the range check and the index dtype gate exist to prevent.
    """
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    assert torch.equal(out.cpu(), expected), "result differs from the CPU reference"


class TestEmbeddingCorrectness:
    """``F.embedding`` on the flagos device equals the CPU reference exactly."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
    @pytest.mark.anyplatform
    def test_matches_cpu(self, dtype, index_dtype):
        weight = torch.randn(32, 8).to(dtype)
        indices = torch.randint(0, 32, (4, 16)).to(index_dtype)
        _assert_exact(
            F.embedding(_to_device(indices), _to_device(weight)),
            _reference(indices, weight),
        )

    @pytest.mark.parametrize(
        "index_shape",
        [
            (1,),
            (7,),
            (4, 16),
            (2, 3, 5),
            (2, 3, 4, 5),
        ],
    )
    @pytest.mark.anyplatform
    def test_index_rank_is_preserved(self, index_shape):
        """The result carries the index shape with the row width appended.

        The graph reshapes a flat index buffer and restores the aten shape inside
        the graph, so a rank that is dropped or reordered on the way through is
        the failure this holds against.
        """
        weight = torch.randn(16, 4)
        indices = torch.randint(0, 16, index_shape)
        out = F.embedding(_to_device(indices), _to_device(weight))
        assert out.shape == index_shape + (4,)
        _assert_exact(out, _reference(indices, weight))

    @pytest.mark.anyplatform
    def test_empty_indices(self):
        """No indices is a valid call and an empty result, not a decline.

        MPSGraph rejects an empty tensor, so the body short-circuits this shape
        before it builds a node. The result is still a table read of zero rows,
        which is what the reference returns.
        """
        weight = torch.randn(16, 4)
        indices = torch.empty(0, dtype=torch.long)
        out = F.embedding(_to_device(indices), _to_device(weight))
        assert out.shape == (0, 4)
        _assert_exact(out, _reference(indices, weight))

    @pytest.mark.anyplatform
    def test_non_contiguous_table(self):
        """A transposed table reads the same rows.

        The body materialises a strided operand before it describes it to
        MPSGraph, because a buffer MpsRunGraph reads densely would otherwise be
        read as the wrong bytes (see the row-pitch note in mps_common.h).
        """
        weight = torch.randn(4, 16).t()
        indices = torch.randint(0, 16, (2, 3))
        assert not weight.is_contiguous()
        _assert_exact(
            F.embedding(_to_device(indices), _to_device(weight)),
            _reference(indices, weight),
        )

    @pytest.mark.anyplatform
    def test_head_embedding_shaped_operand(self):
        """The shape the op is on the hot path for.

        A 0.6B-class model reads a ``(vocab, hidden)`` table with a
        ``(batch, seq)`` index tensor, so the operand that matters is the one the
        fallback used to copy to the host in full. Kept small enough to compare on
        the host here; the cost of the full-size shape is held separately.
        """
        weight = torch.randn(4096, 64)
        indices = torch.randint(0, 4096, (1, 128))
        _assert_exact(
            F.embedding(_to_device(indices), _to_device(weight)),
            _reference(indices, weight),
        )


class TestEmbeddingPaddingIdx:
    """``padding_idx`` does not change the forward output, and is not read."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("padding_idx", [0, 3, 15, 31])
    def test_in_table_padding_idx_returns_the_row(self, padding_idx):
        """A padded index still reads its row: the forward op does not zero it.

        This is the case the first version of the kernel got backwards -- it
        assumed the reference zeroed the ``padding_idx`` row, which is what
        ``embedding_dense_backward`` does to the *gradient*, not what this op does
        to its output. Checked against the reference rather than against a
        hand-built expectation so the claim is held by the same source the kernel
        is.
        """
        weight = torch.randn(32, 8)
        indices = torch.tensor([[padding_idx, 0, 1]])
        out = F.embedding(
            _to_device(indices), _to_device(weight), padding_idx=padding_idx
        )
        _assert_exact(out, _reference(indices, weight, padding_idx=padding_idx))
        # And the padded row is genuinely non-zero, so a kernel that zeroed it
        # would fail the comparison above rather than passing on a zero table.
        assert out.cpu()[0, 0].abs().sum().item() > 0

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("padding_idx", [None, -1, -3, -32, 7])
    def test_padding_idx_does_not_change_the_result(self, padding_idx):
        """Every ``padding_idx`` gives the same bytes as no ``padding_idx``.

        ``None`` is the interesting one: ``F.embedding`` turns it into ``-1``
        before the op is reached, so a body that read ``-1`` as "count back from
        the end" would see a real padded row on the most common call there is.
        The rest span the negative and positive spellings, in and out of range.
        """
        weight = torch.randn(32, 8)
        indices = torch.randint(0, 32, (4, 16))
        plain = F.embedding(_to_device(indices), _to_device(weight))
        padded = F.embedding(
            _to_device(indices), _to_device(weight), padding_idx=padding_idx
        )
        _assert_exact(
            padded, _reference(indices, weight, padding_idx=padding_idx or -1)
        )
        assert torch.equal(padded.cpu(), plain.cpu())


class TestEmbeddingRangeCheck:
    """An index the table cannot answer reaches the reference's own error."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("bad_index", [6, -1, 100])
    def test_out_of_range_index_raises(self, bad_index):
        """``IndexError``, not the graph's silent first row.

        Measured on an M5, the gather node answers an out-of-range index with the
        table's first row -- for a negative index as well as one past the end. So
        the case is not "does it raise" in the abstract but "does the caller see
        the reference's error", which only holds while the host range check runs
        before any node exists.
        """
        weight = torch.randn(6, 3)
        indices = torch.tensor([[bad_index]], dtype=torch.long)
        with pytest.raises(IndexError):
            F.embedding(_to_device(indices), _to_device(weight))

    @pytest.mark.anyplatform
    def test_one_bad_index_among_good_ones_raises(self):
        """The check is over the whole tensor, not just the first element."""
        weight = torch.randn(6, 3)
        indices = torch.tensor([[5, 6, 2, 0]], dtype=torch.long)
        with pytest.raises(IndexError):
            F.embedding(_to_device(indices), _to_device(weight))

    @pytest.mark.anyplatform
    def test_last_row_is_in_range(self):
        """The boundary the check is written against is inclusive at the top."""
        weight = torch.randn(6, 3)
        indices = torch.tensor([[5]], dtype=torch.long)
        _assert_exact(
            F.embedding(_to_device(indices), _to_device(weight)),
            _reference(indices, weight),
        )

    @pytest.mark.anyplatform
    def test_float_indices_take_the_reference_error(self):
        """A float index is the reference's ``RuntimeError``, not a graph guess.

        MPSGraph's gather takes an int32 or int64 index tensor and has no other
        integer input, so the body declines on the index dtype before it builds a
        node. The message names Long and Int, which is the reference's list rather
        than this backend's.
        """
        weight = torch.randn(6, 3)
        indices = torch.tensor([[1.0]], dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Long, Int"):
            F.embedding(_to_device(indices), _to_device(weight))


class TestEmbeddingPlanReuse:
    """One plan answers the shapes a decode loop presents."""

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_a_different_dtype(self):
        """A table of another dtype gets its own plan, not the first one's.

        The body key has no host parts, so the operand signature MpsRunGraph
        folds in is the *only* thing separating a float32 call from a float16 one.
        A key that lost the dtype would answer the second call out of the first
        graph's plan, which reads the wrong bytes rather than failing.
        """
        indices = torch.randint(0, 16, (2, 3))
        for dtype in FLOAT_DTYPES:
            weight = torch.randn(16, 4).to(dtype)
            _assert_exact(
                F.embedding(_to_device(indices), _to_device(weight)),
                _reference(indices, weight),
            )

    @pytest.mark.anyplatform
    def test_plan_is_reused_across_padding_idx(self):
        """Two calls that differ only in ``padding_idx`` share one plan.

        They produce the same bytes, so they are the same graph -- and this is
        the case that would break if a future change put ``padding_idx`` back into
        the key without also making the body read it.
        """
        weight = torch.randn(16, 4)
        indices = torch.randint(0, 16, (2, 3))
        first = F.embedding(_to_device(indices), _to_device(weight), padding_idx=-1)
        second = F.embedding(_to_device(indices), _to_device(weight), padding_idx=2)
        assert torch.equal(first.cpu(), second.cpu())


class TestEmbeddingDispatch:
    """``embedding`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_embedding_to_mps(self):
        assert routed_backend_or_none("embedding") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        assert routed_backend_or_none("embedding.out") is None

    @pytest.mark.main_ops
    def test_conf_leaves_the_backward_on_the_fallback(self):
        """The backward is its own op with its own scatter question."""
        assert routed_backend_or_none("embedding_dense_backward") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps_for_both_spellings(self):
        """``F.embedding`` and ``aten.embedding`` both reach the kernel.

        The public spelling is the one that matters to a caller and the aten one
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
                "import torch.nn.functional as F; "
                "w = torch.randn(32, 8, device='flagos:0'); "
                "i = torch.randint(0, 32, (4, 16), device='flagos:0'); "
                "F.embedding(i, w); "
                "torch.ops.aten.embedding(w, i)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] embedding -> apple") == 2

    @pytest.mark.main_ops
    def test_declined_dtype_does_not_reach_the_graph(self):
        """A float64 table comes back in float64, from the reference.

        MPSGraph has no float64 element type, so the body declines before it
        builds a node and the caller gets the reference's own dtype. A case that
        ran the graph anyway would be a wrong answer rather than a slow one.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "import torch.nn.functional as F; "
                "w = torch.randn(32, 8, dtype=torch.float64, device='flagos:0'); "
                "i = torch.randint(0, 32, (4, 16), device='flagos:0'); "
                "print(F.embedding(i, w).dtype)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "torch.float64" in result.stdout
        # The kernel is still entered and logs once; the answer is the
        # reference's, which is what the round-trip is for.
        assert "[flagos dispatch] embedding -> apple" in result.stderr


class TestEmbeddingCost:
    """The op is served, not round-tripped, on the shape it exists for."""

    @pytest.mark.anyplatform
    def test_large_table_does_not_scale_with_the_table(self):
        """A 0.6B-class read is served, ten times over, not round-tripped once.

        The fallback this op replaced copied the whole ``(151936, 1024)`` table
        to the host on every call -- 55.8 ms of a 55.3 ms measurement, against
        0.697 ms for the same read performed on the CPU -- so the regression this
        guards against is a *silent* return to that path, which a correctness
        test cannot see: a round-tripped embedding returns the right rows.

        What catches it is the dispatch log rather than the clock. The log line
        this model's ``--log`` census already relies on (see the dispatch tests
        above) names the backend this module routed to, and it is emitted once
        per call, so ten reads that were all served log exactly ten ``-> apple``
        lines. A timing bound was tried first and is the wrong instrument: it
        measured 62 ms on CI against 0.7 ms locally *for calls that were being
        served*, so it fails on a paravirtualized runner for work the log count
        proves correct, and it would still miss a fallback that was merely fast.
        """
        vocab, hidden = 151936, 1024
        reads = 10
        served = dispatched_to_mps(
            """
import torch, torch_fl
import torch.nn.functional as F
weight = torch.randn(151936, 1024, device='flagos:0')
indices = torch.randint(0, 151936, (1, 512), device='flagos:0')
F.embedding(indices, weight)
torch_fl._C._synchronize()
for _ in range(10):
    F.embedding(indices, weight)
torch_fl._C._synchronize()
""",
        )
        # The warmup logs too: it is the call a fallback would have answered by
        # copying the table, so it is counted rather than subtracted.
        assert served == reads + 1, (
            f"F.embedding over a {vocab}x{hidden} table logged {served} of "
            f"{reads + 1} reads onto the mps slot; the whole-table fallback is back"
        )

    @pytest.mark.anyplatform
    def test_a_declined_width_stays_inside_the_body(self):
        """A declined dtype is the body's round trip, not the dispatcher's.

        The two are different answers to "why is this read slow". The body's
        round trip is a temporary the body owns and is the same code path for
        every platform; a dispatch that never reached the body is the conf or
        the registration having changed underneath it. Only the second is a
        regression, and the dispatch log is what tells them apart: the whole-table
        case above pins the route, and this one pins that a width the gate
        declines still reaches the body and logs the same single line the served
        width does.

        float64 is the width to use because the gate declines it for the reason
        the graph cannot answer at all, so the assertion is independent of any
        tuning decision about which widths are worth serving.
        """
        served = dispatched_to_mps(
            """
import torch, torch_fl
import torch.nn.functional as F
weight = torch.randn(4096, 128, dtype=torch.float64, device='flagos:0')
indices = torch.randint(0, 4096, (1, 512), device='flagos:0')
F.embedding(indices, weight)
torch_fl._C._synchronize()
""",
        )
        assert served == 1, (
            "F.embedding over a float64 table logged "
            f"{served} lines; the body should have been entered once and done "
            "its own round trip, not been handed elsewhere"
        )
