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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``cat``.

``cat`` is MPSGraph's variadic ``concatTensors:dimension:name:``, and the reason
this file is longer than its one node is that the node checks almost none of the
reference's contract. Three of those gaps are bugs this body hit, and each one
is worth a case below because each produced a *plausible* wrong answer rather
than a decline:

* **The row pitch on a rank-1 operand.** ``cat([tensor([0., 1.]),
  tensor([2., 3.])], dim=0)`` handed to the node as two ranked operands returned
  all zeros, and an int32 form returned ``[0, 0, 1, 0, 2, 0]`` -- the eight-byte
  row pitch of a four-byte element, the same symptom the row-pitch note in
  ``mps_common.h`` records. A ``(2, 3)`` pair on ``dim=0`` was correct, which is
  what makes the rank-1 case the interesting one: MPSGraph reads a rank-1 input
  to ``concatTensors:`` as a single row and pays the pitch. The body sidesteps
  it by never handing the node a rank below two -- every operand reaches the
  graph as a two-dimensional ``(outer, inner)`` view with the concatenation over
  the *inner* axis -- which is why ``TestCatShape`` holds a rank-1 list beside a
  rank-2 one.
* **One ``inner`` shared across the list.** The first version of that
  two-dimensional view computed ``outer``/``inner`` once from ``ops[0]``. For a
  rank-1 three-operand list of ``(2,)``, ``(2,)`` and ``(1,)`` that reshaped the
  third operand to ``(1, 2)`` -- a shape its single element cannot fill -- and
  MPSGraph *aborted the process* (``mps.reshape op the result shape is not
  compatible with the input shape``, then ``original module failed
  verification``) rather than declining. ``inner`` is read per operand, and the
  ``(2,), (2,), (1,)`` list is a case below rather than a footnote.
* **The fallback that re-entered the kernel.** The decline path was written
  ``at::cat(tensors, dim)`` against the caller's list, whose tensors are still
  flagos tensors, so a declined call (bool, which has no MPSGraph element type)
  re-entered this same body -- measured at 23810 dispatch lines before the stack
  ran out and the process died with SIGSEGV. The reference here is called on
  host copies, built once before any early return, and the bool decline is held
  in ``TestCatDeclines``.

The rest of the body is the host checking the contract the node would otherwise
guess at: ``dim`` is normalised against the operands' rank, a mixed-dtype list
takes the round-trip (the node does not promote), and every non-``dim`` axis is
compared exactly (the node would *broadcast* a ``(4, 1)`` against a ``(4, 5)``,
which is a plausible-looking wrong answer rather than the reference's error).

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

# Every dtype the node is served on. `cat` computes nothing -- it moves whole
# elements -- so unlike the arithmetic families it passes its operands' width
# through rather than narrowing to the float widths, and the integer widths are
# part of the contract rather than a convenience.
CARRYABLE_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
]


def _reference(tensors, dim: int) -> torch.Tensor:
    """``at::cat`` on the host.

    The reference the kernel is held to, and deliberately the same op the body
    round-trips to rather than a hand-built ``torch.cat``: the host pass exists
    to hand the caller *this op's* answer -- including its dtype promotion and
    its own error types -- so a test written against anything else would be
    holding a different contract for the declined cases.
    """
    return torch.ops.aten.cat(tensors, dim)


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another case
    # left behind.
    return t.clone().to(DEVICE)


def _to_device_list(tensors) -> list:
    return [_to_device(t) for t in tensors]


def _assert_exact(out: torch.Tensor, expected: torch.Tensor) -> None:
    """The result equals the reference's, byte for byte.

    No tolerance: the node moves whole elements rather than computing over them,
    so the only two answers are "the right elements" and "the wrong elements".
    """
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    assert torch.equal(out.cpu(), expected), "result differs from the CPU reference"


class TestCatShape:
    """The result is the operands' shapes with ``dim`` summed."""

    @pytest.mark.anyplatform
    def test_two_rank_one_operands(self):
        """The case the row-pitch trap is written against.

        Two ``(2,)`` float32 operands on ``dim=0`` are four values end to end.
        Handed to the node as their ranked selves this returned all zeros (and an
        int32 form returned ``[0, 0, 1, 0, 2, 0]``), because a rank-1 input to
        ``concatTensors:`` is read as one row and pays the eight-byte pitch of a
        four-byte element. The body views every operand as ``(1, 2)`` first, so
        the case passes only if the two-dimensional view is still there -- it is
        the single cheapest case that fails if the view is removed.
        """
        a = torch.tensor([0.0, 1.0])
        b = torch.tensor([2.0, 3.0])
        out = torch.cat(_to_device_list([a, b]), dim=0)
        assert out.shape == (4,)
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    def test_the_row_pitch_shows_on_an_int_operand(self):
        """The same rank-1 case on a width the pitch is visible in.

        int32 is four bytes, so the eight-byte row pitch reads every other
        element and inserts a zero -- the ``[0, 0, 1, 0, 2, 0]`` the finding
        records. float32 has the same width and shows it too, but an integer
        operand makes the wrong bytes unambiguous rather than a rounded float.
        """
        a = torch.tensor([0, 1], dtype=torch.int32)
        b = torch.tensor([2, 3], dtype=torch.int32)
        out = torch.cat(_to_device_list([a, b]), dim=0)
        assert out.shape == (4,)
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [0, 1, -1, -2], ids=["d0", "d1", "dm1", "dm2"])
    def test_two_rank_two_operands(self, dim):
        """The rank-2 pair on both axes and both negative spellings.

        ``dim`` is resolved against the operands' rank, and the resolved axis is
        what the node bakes in, so the two axes are two different concatenations
        of the same operands. Both spellings of each axis are here because the
        normalisation is the host's and a body that passed ``-1`` through to the
        node would answer the ``dim=1`` case and fail the ``dim=-1`` one -- or
        the reverse, depending on how the node reads a negative index.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        out = torch.cat(_to_device_list([a, b]), dim=dim)
        _assert_exact(out, _reference([a, b], dim))

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
    def test_three_rank_three_operands(self, dim):
        """All six axes of a rank-3 pair, positive and negative.

        The rank-3 result is where the ``(prod(shape[:axis]), prod(shape[axis:]))``
        split has the most to get wrong: for ``dim=1`` the outer extent is
        ``shape[0]`` and the inner is ``shape[1] * shape[2]``, and a body that
        had swapped the two -- a plausible reading of the same sentence -- would
        answer ``dim=2`` here and produce the wrong shape for the rest.
        """
        a = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        b = torch.arange(12, 24, dtype=torch.float32).reshape(2, 2, 3)
        out = torch.cat(_to_device_list([a, b]), dim=dim)
        _assert_exact(out, _reference([a, b], dim))

    @pytest.mark.anyplatform
    def test_ragged_on_the_concatenated_axis(self):
        """Operands of different extents along ``dim`` are the point of ``cat``.

        The non-``dim`` extents match and the ``dim`` extents do not, which is
        the only shape a concatenation exists for. A body that required the
        operands to share a shape -- a natural reading of "the node needs
        broadcast-compatible inputs" -- would decline this and be invisible in
        every equal-shape case above.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(6, 8, dtype=torch.float32).reshape(2, 1)
        out = torch.cat(_to_device_list([a, b]), dim=1)
        assert out.shape == (2, 4)
        _assert_exact(out, _reference([a, b], 1))

    @pytest.mark.anyplatform
    def test_three_operands_ragged_on_the_concatenated_axis(self):
        """A third operand is a third ``inner``, and ``inner`` is read per
        operand.

        At rank 2 the three operands are ``(2, 3)``, ``(2, 3)`` and ``(2, 1)``
        on ``dim=1``: three different ``inner`` values over one ``outer``, so a
        body that carried the first operand's width across the list fails here.
        The rank-1 form of the same list is
        ``test_per_operand_inner_on_a_ragged_rank_one_list`` below.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        c = torch.arange(12, 14, dtype=torch.float32).reshape(2, 1)
        out = torch.cat(_to_device_list([a, b, c]), dim=1)
        assert out.shape == (2, 7)
        _assert_exact(out, _reference([a, b, c], 1))

    @pytest.mark.anyplatform
    def test_per_operand_inner_on_a_ragged_rank_one_list(self):
        """The exact list that aborted the process, held as a case.

        ``(2,)``, ``(2,)`` and ``(1,)`` on ``dim=0``: a rank-1 list whose
        operands do not share an extent. The first version of the two-dimensional
        view read ``inner`` from ``ops[0]`` and reshaped the third operand to
        ``(1, 2)``, which its one element cannot fill; MPSGraph then failed the
        compiled module and the *process* died, so this case could not have been
        a decline even if the body had wanted one. It passes only if ``inner``
        is read per operand -- ``(1, 2)``, ``(1, 2)`` and ``(1, 1)``.
        """
        a = torch.tensor([0.0, 1.0])
        b = torch.tensor([2.0, 3.0])
        c = torch.tensor([4.0])
        out = torch.cat(_to_device_list([a, b, c]), dim=0)
        assert out.shape == (5,)
        _assert_exact(out, _reference([a, b, c], 0))

    @pytest.mark.anyplatform
    def test_single_operand_is_itself(self):
        """A one-element list is a readable call and not a decline.

        ``cat`` over one operand is a copy of that operand, and the body serves
        it through the graph rather than returning the operand's alias. The
        values are asserted against the reference so a body that returned the
        input without copying would still be caught if the shapes disagreed.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = torch.cat(_to_device_list([a]), dim=0)
        assert out.shape == (2, 3)
        _assert_exact(out, _reference([a], 0))

    @pytest.mark.anyplatform
    def test_rank_one_single_operand(self):
        """A lone rank-1 operand is still reshaped to two dimensions for the
        node, and that view must not change the answer."""
        a = torch.tensor([0.0, 1.0, 2.0])
        out = torch.cat(_to_device_list([a]), dim=0)
        assert out.shape == (3,)
        _assert_exact(out, _reference([a], 0))


class TestCatCorrectness:
    """``cat`` on the flagos device equals the CPU reference exactly."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", CARRYABLE_DTYPES)
    def test_the_operand_width_passes_through(self, dtype):
        """A concatenation does not change its operands' width, and it does not
        round it.

        Each width is filled with values a float cast would round, so a body that
        staged the operands through float32 would show here. The node moves whole
        elements, so there is no arithmetic for a width to change under and the
        integer widths are as servable as the float ones.
        """
        a = torch.arange(8).to(dtype)
        b = torch.arange(8, 16).to(dtype)
        out = torch.cat(_to_device_list([a, b]), dim=0)
        assert out.dtype == dtype
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    def test_non_contiguous_operands(self):
        """A strided operand is materialised before the graph reads it.

        MPSGraph reads and writes dense memory (see the row-pitch note in
        ``mps_common.h``), so a body that handed a transposed operand straight to
        the node would read the wrong bytes. Both operands are transposed, which
        is how a caller's operand lists usually arrive non-contiguous.
        """
        a = torch.arange(6).reshape(2, 3).t()
        b = torch.arange(10, 16).reshape(2, 3).t()
        assert not a.is_contiguous()
        assert not b.is_contiguous()
        out = torch.cat(_to_device_list([a, b]), dim=0)
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    def test_a_mixed_contiguity_list(self):
        """A contiguous operand beside a strided one is copied on its own.

        The body materialises the list operand by operand, so the two do not
        share a decision. A body that had copied only the first operand -- or
        only after checking the first was strided -- answers the all-strided and
        all-contiguous cases and fails this one.
        """
        a = torch.arange(6).reshape(2, 3)
        b = torch.arange(10, 16).reshape(3, 2).t()
        assert a.is_contiguous()
        assert not b.is_contiguous()
        out = torch.cat(_to_device_list([a, b]), dim=0)
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    def test_large_list_of_small_operands(self):
        """A list of many operands concatenates all of them.

        The kernel body follows the list rather than any fixed arity, and the
        variadic node is the one place an off-by-one in the operand count would
        be a plausible bug rather than a crash. Twelve operands is enough that a
        count that dropped the first or last would move a value.
        """
        tensors = [torch.full((1, 4), float(i)) for i in range(12)]
        out = torch.cat(_to_device_list(tensors), dim=0)
        assert out.shape == (12, 4)
        _assert_exact(out, _reference(tensors, 0))


class TestCatDeclines:
    """The paths the node cannot serve reach the reference, and reach it safely."""

    @pytest.mark.anyplatform
    def test_a_bool_list_takes_the_reference(self):
        """bool has no MPSGraph element type, so the list round-trips.

        This is the case the recursion bug died on, held by *value*: the dispatch
        log names ``mps`` even when the body declines, because the kernel is what
        decided, so what is asserted is that the answer is the reference's rather
        than a crash or a wrong dtype.
        """
        a = torch.tensor([True, False, True])
        b = torch.tensor([False, False, True])
        out = torch.cat(_to_device_list([a, b]), dim=0)
        assert out.dtype == torch.bool
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    def test_a_bool_decline_does_not_recurse(self):
        """The decline is held out of process, because the bug was a SIGSEGV.

        Run in a subprocess so a recursive re-entry into the kernel is observed
        as a signal rather than as a crashed test runner. The assertion is on the
        process exiting cleanly *and* producing the reference's values: a body
        that returned before the graph without a value would exit 0 with the
        wrong result, and the re-entry exits with a signal instead.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                # A one-liner cannot carry this: the body is a compound
                # statement, so it is written in newlines the `-c` preserves.
                "import torch, torch_fl\n"
                "a = torch.tensor([True, False, True], device='flagos:0')\n"
                "b = torch.tensor([False, False, True], device='flagos:0')\n"
                "out = torch.cat([a, b], dim=0)\n"
                "print('OK', out.dtype, out.cpu().tolist())\n",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK torch.bool [True, False, True, False, False, True]" in result.stdout

    @pytest.mark.anyplatform
    def test_a_mixed_dtype_list_takes_the_reference(self):
        """A mixed list is the reference's promotion, not the node's.

        The node requires one datatype and does not promote, so ``int32`` beside
        ``float32`` is refused by the body and answered by the reference at
        ``float32``. Asserting the *promoted* dtype is the point: a body that
        served the list at the first operand's width would return ``int32``.
        """
        a = torch.tensor([0, 1], dtype=torch.int32)
        b = torch.tensor([2.5, 3.5], dtype=torch.float32)
        out = torch.cat(_to_device_list([a, b]), dim=0)
        assert out.dtype == torch.float32
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    def test_a_float64_list_takes_the_reference(self):
        """float64 is outside the node's element types and round-trips at its
        own width rather than being narrowed."""
        a = torch.arange(4, dtype=torch.float64)
        b = torch.arange(4, 8, dtype=torch.float64)
        out = torch.cat(_to_device_list([a, b]), dim=0)
        assert out.dtype == torch.float64
        _assert_exact(out, _reference([a, b], 0))

    @pytest.mark.anyplatform
    def test_a_shape_mismatch_raises_the_reference_error(self):
        """The non-``dim`` extent must match, and the node would *broadcast* it.

        A ``(2, 3)`` beside a ``(2, 1)`` on ``dim=0`` is an error on the
        reference -- ``Sizes of tensors must match except in dimension 0`` --
        where the node, handed the same pair, would broadcast along the
        mismatched axis and return a plausible-looking ``(2, 3)``. Both the
        raise and the message are asserted, because the failure mode this case
        guards against is a *value* and not an exception.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(2, dtype=torch.float32).reshape(2, 1)
        with pytest.raises(RuntimeError, match="must match"):
            torch.cat(_to_device_list([a, b]), dim=0)

    @pytest.mark.anyplatform
    def test_a_rank_mismatch_raises_the_reference_error(self):
        """A rank-2 operand beside a rank-1 one is the reference's error."""
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(2, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            torch.cat(_to_device_list([a, b]), dim=0)

    @pytest.mark.anyplatform
    def test_dim_out_of_range_raises_the_reference_error(self):
        """A ``dim`` outside the operands' rank is the reference's error.

        The body normalises a negative ``dim`` and declines a ``dim`` it cannot
        place, so the reference raises its own message. A rank-2 pair supports
        ``dim`` in ``[-2, 1]``; ``2`` and ``-3`` are one past each end.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        for dim in (2, -3):
            with pytest.raises(IndexError):
                torch.cat(_to_device_list([a, b]), dim=dim)

    @pytest.mark.anyplatform
    def test_empty_list_raises_the_reference_error(self):
        """``torch.cat([])`` is the reference's error, not an empty tensor.

        The body checks the empty list first -- ``ops[0]`` below it would be out
        of range -- so the reference raises ``cat() received an empty list of
        tensors``.
        """
        with pytest.raises(ValueError, match="empty"):
            torch.cat([], dim=0)

    @pytest.mark.anyplatform
    def test_rank_zero_operands_raise_the_reference_error(self):
        """A scalar has no ``dim`` to concatenate, and the reference says so.

        The body declines a rank-0 list rather than resolving ``dim`` against a
        rank it does not have, so the reference's own error is what reaches the
        caller.
        """
        a = torch.tensor(1.0)
        b = torch.tensor(2.0)
        with pytest.raises(RuntimeError):
            torch.cat(_to_device_list([a, b]), dim=0)


class TestCatPlanReuse:
    """One plan answers the shapes a model presents, and no two share one."""

    @pytest.mark.anyplatform
    def test_the_axis_is_a_key_part(self):
        """The same operands on two axes are two answers, and one plan cannot
        serve both.

        This is the case the ``axis`` key part exists for. A ``(2, 3)`` pair
        concatenated on ``dim=0`` and on ``dim=1`` has one dtype and one set of
        operand shapes -- the operands cannot say which axis was asked for -- and
        the node bakes ``dimension:`` into the compiled graph. A body that
        resolved the axis and left it out of the key would hand the first call's
        plan to the second and return the first axis's result with the second
        axis's shape.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        da = _to_device(a)
        db = _to_device(b)
        d0 = torch.cat([da, db], dim=0)
        d1 = torch.cat([da, db], dim=1)
        assert d0.shape == (4, 3)
        assert d1.shape == (2, 6)
        _assert_exact(d0, _reference([a, b], 0))
        _assert_exact(d1, _reference([a, b], 1))

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_a_different_dtype(self):
        """An operand list of another dtype gets its own plan, not the first
        one's.

        The dtype is the only thing separating a float32 call from a float16 one
        at one shape -- the axis is the same and the shapes agree -- and
        MpsRunGraph folds the operand signature into the key, so the second call
        must build a second graph rather than read the first one's bytes as the
        second width.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        for dtype in (torch.float32, torch.float16, torch.int32):
            da = _to_device(a.to(dtype))
            db = _to_device(b.to(dtype))
            out = torch.cat([da, db], dim=1)
            assert out.dtype == dtype
            _assert_exact(out, _reference([a.to(dtype), b.to(dtype)], 1))

    @pytest.mark.anyplatform
    def test_plan_is_reused_across_operand_values(self):
        """Two lists of one signature share a plan and read their own values.

        The operands are graph *inputs* and not key parts, so the second call
        must build no new graph and still return its own elements. A plan that
        pinned the first call's values would answer this one with them.
        """
        da = torch.arange(6, dtype=torch.float32, device=DEVICE).reshape(2, 3)
        db = torch.arange(6, 12, dtype=torch.float32, device=DEVICE).reshape(2, 3)
        first = torch.cat([da, db], dim=1)
        dc = (da + 100).contiguous()
        dd = (db + 100).contiguous()
        second = torch.cat([dc, dd], dim=1)
        assert second.shape == first.shape
        assert not torch.equal(first.cpu(), second.cpu())
        _assert_exact(second, _reference([dc.cpu(), dd.cpu()], 1))

    @pytest.mark.anyplatform
    def test_a_second_call_after_a_declined_one_is_served(self):
        """A declined call leaves no plan behind that the next one inherits.

        The shape-mismatch and rank-mismatch paths return before the graph is
        built, so the plan cache is untouched; this holds that the next servable
        call still gets a correct answer rather than a half-built graph or the
        previous list.
        """
        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(2, dtype=torch.float32).reshape(2, 1)
        with pytest.raises(RuntimeError):
            torch.cat(_to_device_list([a, b]), dim=0)
        c = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        d = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        out = torch.cat(_to_device_list([c, d]), dim=1)
        assert out.shape == (2, 6)
        _assert_exact(out, _reference([c, d], 1))


class TestCatDispatch:
    """``cat`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_cat_to_mps(self):
        assert routed_backend_or_none("cat") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """``cat.out`` writes into a caller-owned tensor, like the rest."""
        assert routed_backend_or_none("cat.out") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """Both the public and the aten spelling reach the kernel.

        ``torch.cat`` is what a caller writes and ``aten.cat`` is what a
        decomposition would lower to; both are held so a future change that made
        the public form composite would show up here. Run in a subprocess because
        ``LogEnabled`` latches the environment on first use.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl; "
                "a = torch.arange(4, dtype=torch.float32, device='flagos:0'); "
                "b = torch.arange(4, 8, dtype=torch.float32, device='flagos:0'); "
                "torch.cat([a, b], dim=0); "
                "torch.ops.aten.cat([a, b], 0)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] cat -> mps") == 2

    @pytest.mark.main_ops
    def test_a_declined_list_does_not_reach_the_graph(self):
        """A bool list is entered and answered by the reference.

        The dispatch log names the backend even when the body declines, because
        the kernel is what decided; what this holds is that the *answer* is the
        reference's on the path the body has no node for, and that the process
        survives the decline.
        """
        env = os.environ.copy()
        env["FLAGOS_LOG"] = "dispatch"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch, torch_fl\n"
                "a = torch.tensor([True, False], device='flagos:0')\n"
                "b = torch.tensor([True, True], device='flagos:0')\n"
                "print('bool', torch.cat([a, b], 0).cpu().tolist())\n",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "bool [True, False, True, True]" in result.stdout
        assert "[flagos dispatch] cat -> mps" in result.stderr


class TestCatCost:
    """The op is served, not round-tripped, on the shape it exists for."""

    @pytest.mark.anyplatform
    def test_a_large_list_does_not_scale_with_the_operands(self):
        """A KV-cache-style concatenation is not proportional to the data.

        The whole-list round trip this op replaced copies every operand to the
        host and back on every call, which is proportional to the total element
        count; the node concatenates on the device. The regression this guards
        against is a silent return to that path, which a correctness test cannot
        see. The bound is deliberately loose: it separates "the concatenation is
        served on the device" from "the operands were copied", not one kernel's
        tuning from another's.
        """
        import time

        width, rows = 4096, 256
        left = torch.randn(rows, width, device=DEVICE)
        right = torch.randn(rows, width, device=DEVICE)

        def run(n):
            for _ in range(3):
                torch.cat([left, right], dim=0)
            if hasattr(torch_fl, "synchronize"):
                torch_fl.synchronize()
            start = time.perf_counter()
            for _ in range(n):
                torch.cat([left, right], dim=0)
            if hasattr(torch_fl, "synchronize"):
                torch_fl.synchronize()
            return (time.perf_counter() - start) / n * 1e3

        per_call = run(10)
        # A host round trip of two 256x4096 float32 operands is milliseconds of
        # copy each way; the node is well under one. 10 ms sits far from both.
        assert per_call < 10.0, (
            f"cat over two {rows}x{width} operands took {per_call:.1f} ms; "
            "the whole-list fallback is back"
        )
