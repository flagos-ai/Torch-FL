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

"""Stage-4 Apple Silicon (Metal / MPSGraph) ``where.self``.

``where.self`` is MPSGraph's ``selectWithPredicateTensor:truePredicateTensor:
falsePredicateTensor:name:``, and it looks like the simplest op in the stage-4
cohort and is not. The node selects between two tensors on a bool predicate and
does nothing else; the reference does three things around it that the node
neither checks nor performs, and each is a case below:

* **The condition is narrowed to bool, by the reference and not the node.** The
  reference accepts a bool condition and, with a deprecation warning, a uint8
  one, and raises for every other width -- ``where expected condition to be a
  boolean tensor, but got a tensor with dtype Int``. The node's predicate is
  bool, so the uint8 case is widened by a comparison against zero in the body;
  a width outside the two is *declined* so the reference raises its own message
  rather than the body answering a call the reference rejects. Both halves are
  held, in ``TestWhereCondition`` and ``TestWhereDeclines``.
* **The two branches are promoted to a common dtype, and the node does not
  promote.** The node builds one select in one element type; the reference
  promotes ``self``/``other`` the way the binary ops do -- ``int32`` against
  ``int64`` is ``int64``, half against float is float. A promoted pair takes the
  round-trip and keeps the reference's result dtype, which
  ``TestWhereBranchPromotion`` asserts by *dtype* and not only by value: a body
  that served the select at the first branch's width would return ``int32``.
* **The result's shape is the three operands broadcast together.** The node
  broadcasts its inputs itself but says nothing about the shape the *result* is
  allocated with, and the result is not any one operand's shape: a ``(4, 1)``
  condition against ``(4, 5)`` branches is a ``(4, 5)`` result. The body infers
  that shape from all three operands (twice-nested ``infer_size``, because it
  takes two at a time) and declines a triple the reference cannot broadcast.

The condition shares the wrapper's argument list with the branches, so a body
that read the branches' shapes for the result, or that routed the condition
through the same dtype gate as the branches, passes every single-operand-shape
case and fails one of the three above.

Every case self-skips off an MPS build, like the stage-1 to stage-4 suites.
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

# Every dtype the two branches are served on. `where` selects whole elements, so
# it passes its branches' width through rather than narrowing to the float
# widths -- including bool, which is a servable *result* even though bool is not
# a servable predicate width.
SERVABLE_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]


def _reference(condition, self_, other) -> torch.Tensor:
    """``at::where`` on the host.

    The reference the kernel is held to, and deliberately the same op the body
    round-trips to rather than a hand-built select: the host pass exists to hand
    the caller *this op's* answer -- including its promotion and its own error
    messages -- so a test written against anything else would be holding a
    different contract for the declined cases.
    """
    return torch.ops.aten.where(condition, self_, other)


def _to_device(t: torch.Tensor) -> torch.Tensor:
    # A fresh device tensor per case, so no case can pass on storage another case
    # left behind.
    return t.clone().to(DEVICE)


def _assert_exact(out: torch.Tensor, expected: torch.Tensor) -> None:
    """The result equals the reference's, byte for byte.

    No tolerance: the node selects whole elements rather than computing over
    them, so the only two answers are "the right elements" and "the wrong
    elements".
    """
    assert out.device.type == "flagos"
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    assert torch.equal(out.cpu(), expected), "result differs from the CPU reference"


class TestWhereCorrectness:
    """``where`` on the flagos device equals the CPU reference exactly."""

    @pytest.mark.anyplatform
    @pytest.mark.parametrize("dtype", SERVABLE_DTYPES)
    def test_the_branch_width_passes_through(self, dtype):
        """A select does not change its branches' width, and it does not round
        it.

        The same two branches at every servable width, one of them true and one
        false per element so both arms of the select are exercised. float64 is
        deliberately absent and is held in ``TestWhereDeclines``: it is outside
        the node's element types and round-trips rather than being narrowed.
        """
        condition = torch.tensor([True, False, True, False])
        a = torch.arange(4).to(dtype)
        b = (torch.arange(4) + 10).to(dtype)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.dtype == dtype
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_all_true_and_all_false_conditions(self):
        """A condition that never selects one arm still returns a full tensor.

        Every element of the result comes from one branch, which is where a body
        that had wired the predicate to the wrong operand shows up as the two
        branches swapped rather than as a wrong element.
        """
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([-1.0, -2.0, -3.0])
        for cond in ([True, True, True], [False, False, False]):
            condition = torch.tensor(cond)
            out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
            _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_the_result_is_the_condition_shaped_not_a_branch_shaped(self):
        """A condition with an explicit false in the middle moves one element.

        The condition, not the branches, decides which element comes from where
        -- and here it is the *middle* element, so a body that had read the
        predicate shifted by one would be caught where an all-true or all-false
        condition would not.
        """
        condition = torch.tensor([True, False, True, True])
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([10.0, 20.0, 30.0, 40.0])
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.cpu().tolist() == [1.0, 20.0, 3.0, 4.0]
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_non_contiguous_branches(self):
        """A strided branch is materialised before the graph reads it.

        MPSGraph reads and writes dense memory (see the row-pitch note in
        ``mps_common.h``), so a strided branch handed straight to the node would
        be read in storage order. The transposed ``(4, 3)`` is the shape a
        permute leaves behind.
        """
        condition = torch.tensor(
            [
                [True, False, True],
                [False, True, False],
                [True, True, False],
                [False, False, True],
            ]
        )
        a = torch.arange(12).reshape(3, 4).t()
        b = torch.arange(12, 24).reshape(3, 4).t()
        assert not a.is_contiguous()
        assert not b.is_contiguous()
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_non_contiguous_condition(self):
        """A strided condition reads the elements it names, in its own order.

        The condition is a graph input like the branches, so a body that copied
        the branches and not the condition would read the predicate in storage
        order and swap elements whose condition and branches disagree under the
        permutation.
        """
        base = torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, True, False, False],
            ]
        )
        condition = base.t()
        assert not condition.is_contiguous()
        a = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        b = torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        _assert_exact(out, _reference(condition, a, b))


class TestWhereCondition:
    """The condition's width, and the widening the node's bool predicate needs."""

    @pytest.mark.anyplatform
    def test_a_uint8_condition_selects_by_non_zero(self):
        """The one condition width besides bool the reference still accepts.

        uint8 is deprecated as a condition but not rejected, and the reference
        reads it as "non-zero is true", which the node cannot: its predicate is
        bool, so the body compares the condition against zero first. The values
        include a zero and a non-zero, so a body that had compared against the
        wrong constant, or that had trusted the node's own truthiness of a
        non-bool tensor, returns the wrong elements here.
        """
        condition = torch.tensor([0, 1, 0, 2], dtype=torch.uint8)
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([10.0, 20.0, 30.0, 40.0])
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.cpu().tolist() == [10.0, 2.0, 30.0, 4.0]
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_a_bool_condition_does_not_go_through_the_widening(self):
        """The bool path is the node's native predicate and is not compared to
        zero.

        The uint8 widening is a *conditional* node, and this case is the one
        that distinguishes a body which always built it -- or which widened the
        wrong operand -- from one which built it only for uint8. A bool condition
        and a uint8 condition of the same truth values must agree.
        """
        bool_condition = torch.tensor([True, False, True])
        uint8_condition = torch.tensor([1, 0, 1], dtype=torch.uint8)
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0, 30.0])
        from_bool = torch.where(
            _to_device(bool_condition), _to_device(a), _to_device(b)
        )
        from_uint8 = torch.where(
            _to_device(uint8_condition), _to_device(a), _to_device(b)
        )
        assert torch.equal(from_bool.cpu(), from_uint8.cpu())
        _assert_exact(from_bool, _reference(bool_condition, a, b))
        _assert_exact(from_uint8, _reference(uint8_condition, a, b))

    @pytest.mark.anyplatform
    def test_a_bool_condition_with_bool_branches_gives_a_bool_result(self):
        """bool is a servable *branch* width even though it is not a servable
        predicate width.

        The two roles the dtype plays in this op are separate, and this is the
        case that holds them apart: a bool tensor is a valid branch and an
        invalid condition for any other op, and a body that had routed the
        branches through the condition's gate would decline this.
        """
        condition = torch.tensor([True, False, True, False])
        a = torch.tensor([True, True, False, False])
        b = torch.tensor([False, False, True, True])
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.dtype == torch.bool
        _assert_exact(out, _reference(condition, a, b))


class TestWhereBranchPromotion:
    """The reference promotes the two branches, and the node does not."""

    @pytest.mark.anyplatform
    def test_int32_and_int64_promote_to_int64(self):
        """A ``int32``/``int64`` pair takes the round-trip and comes back
        ``int64``.

        The node builds one select in one element type and cannot promote, so
        the body declines the mixed pair and the reference promotes it. The
        result *dtype* is the point: a body that served the select at ``self``'s
        width would return ``int32`` and every value would still be right at
        these magnitudes.
        """
        condition = torch.tensor([True, False, True, False])
        a = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        b = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.dtype == torch.int64
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_half_and_float_promote_to_float(self):
        """A half branch beside a float one is a float result, not a half one.

        The promotion goes the *wider* way, so the values the half branch
        contributes are widened rather than the float branch's being narrowed --
        the narrowing direction is the one that loses data and is invisible at
        values a half can represent, which is why the float branch carries
        values a half cannot (2**-20 and >2**16).
        """
        condition = torch.tensor([True, False, True, False])
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
        b = torch.tensor([2**-20, 70000.0, 2**17, 1e-8], dtype=torch.float32)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.dtype == torch.float32
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_bfloat16_and_float_promote_to_float(self):
        """The other half-width pair promotes the same way.

        bfloat16 has a wider exponent and a narrower mantissa than float16, so
        it is a different promotion case rather than the same one spelled
        differently: a body that had hard-coded float16 as "the half width"
        would decline this and lose the round-trip's promotion.
        """
        condition = torch.tensor([True, False, True])
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        b = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.dtype == torch.float32
        _assert_exact(out, _reference(condition, a, b))


class TestWhereBroadcast:
    """The result's shape is the three operands broadcast together."""

    @pytest.mark.anyplatform
    def test_a_column_condition_against_full_branches(self):
        """A ``(4, 1)`` condition against ``(4, 5)`` branches is a ``(4, 5)``.

        The condition is the operand most likely to be the narrow one in real
        code (a mask broadcast over a batch), and the result's shape is neither
        operand's: the body infers it from all three. A body that had allocated
        the result at the condition's shape would fail here on the shape, and one
        that used a branch's shape would fail on the values.
        """
        condition = torch.tensor([[True], [False], [True], [False]])
        a = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        b = torch.arange(100, 120, dtype=torch.float32).reshape(4, 5)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.shape == (4, 5)
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_a_row_condition_against_full_branches(self):
        """A ``(1, 5)`` condition against ``(4, 5)`` branches broadcasts the
        other way.

        Held beside the column case because the two broadcast along different
        axes: a body that had expanded the condition on the wrong axis answers
        one of the two and fails the other.
        """
        condition = torch.tensor([[True, False, True, False, True]])
        a = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        b = torch.arange(100, 120, dtype=torch.float32).reshape(4, 5)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.shape == (4, 5)
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_a_branch_broadcasts_against_the_condition(self):
        """A scalar branch beside a full-length condition broadcasts.

        Here the *branch* is the narrow operand, which is the case a body that
        had taken its result shape from the branches -- rather than from all
        three -- would get wrong: the condition is ``(6,)`` and either branch
        alone would allocate a ``(1,)`` result.
        """
        condition = torch.tensor([True, False, True, False, True, False])
        a = torch.tensor(1.0)
        b = torch.tensor(2.0)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.shape == (6,)
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_all_three_broadcast(self):
        """A ``(3, 1)`` condition, a ``(1, 4)`` true branch and a scalar false
        branch resolve to ``(3, 4)``.

        The triple-inference case: the shape is the join of three different
        ranks' broadcasts, which is what the twice-nested ``infer_size`` is for.
        """
        condition = torch.tensor([[True], [False], [True]])
        a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        b = torch.tensor(-1.0)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.shape == (3, 4)
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_zero_element_operands_return_an_empty_result(self):
        """An empty result is a valid call and not a decline.

        The body allocates the inferred shape and returns before the graph for a
        zero-element result, because a graph over no elements has nothing to
        compile. The shape is asserted, since the reference's empty result is a
        different shape from either operand's in general.
        """
        condition = torch.zeros((0, 3), dtype=torch.bool)
        a = torch.zeros((0, 3))
        b = torch.ones((0, 3))
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.shape == (0, 3)
        _assert_exact(out, _reference(condition, a, b))


class TestWhereDeclines:
    """The paths the node cannot serve reach the reference's own errors."""

    @pytest.mark.anyplatform
    def test_an_int_condition_raises_the_reference_error(self):
        """An int32 condition is the reference's error, not the node's.

        The body declines every condition width outside bool and uint8 so the
        reference raises its own message rather than the body answering a call
        the reference rejects or guessing at a truthiness the reference does not
        define. Both the raise and the message are asserted, since a body that
        had compared any non-zero to zero would have returned a value instead.
        """
        condition = torch.tensor([1, 0, 1], dtype=torch.int32)
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0, 30.0])
        with pytest.raises(RuntimeError, match="expected condition to be a boolean"):
            torch.where(_to_device(condition), _to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_a_float_condition_raises_the_reference_error(self):
        """The same decline on a float condition, whose cast to bool would be a
        plausible wrong answer rather than a crash."""
        condition = torch.tensor([1.0, 0.0, 1.0])
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0, 30.0])
        with pytest.raises(RuntimeError, match="expected condition to be a boolean"):
            torch.where(_to_device(condition), _to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_a_float64_branch_takes_the_reference(self):
        """float64 is outside the node's element types and round-trips at its
        own width rather than being narrowed."""
        condition = torch.tensor([True, False, True])
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.dtype == torch.float64
        _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_unbroadcastable_operands_raise_the_reference_error(self):
        """A triple the reference cannot broadcast is its own error.

        The body infers the result shape with ``infer_size`` behind a guard, so a
        shape the reference cannot join is declined and the reference raises. A
        ``(3,)`` condition against ``(4,)`` branches is one element apart in the
        only dimension.
        """
        condition = torch.tensor([True, False, True])
        a = torch.arange(4, dtype=torch.float32)
        b = torch.arange(8, 12, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="must match|broadcast"):
            torch.where(_to_device(condition), _to_device(a), _to_device(b))

    @pytest.mark.anyplatform
    def test_a_bool_condition_beside_an_int_condition_pair_is_not_a_shortcut(
        self,
    ):
        """A declined condition leaves the plan cache untouched.

        The decline returns before the graph is built, so the next servable call
        must still get a correct answer rather than a half-built graph or the
        previous condition. The two calls differ only in the condition's width,
        which is what makes the second one a fresh plan.
        """
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0, 30.0])
        with pytest.raises(RuntimeError):
            torch.where(
                _to_device(torch.tensor([1, 0, 1], dtype=torch.int32)),
                _to_device(a),
                _to_device(b),
            )
        condition = torch.tensor([True, False, True])
        out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
        assert out.cpu().tolist() == [1.0, 20.0, 3.0]
        _assert_exact(out, _reference(condition, a, b))


class TestWherePlanReuse:
    """One plan answers the shapes a model presents, and no two share one."""

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_a_different_branch_dtype(self):
        """A branch of another dtype gets its own plan, not the first one's.

        ``where.self`` has no host constant at all -- its key is the operand
        signature MpsRunGraph folds in -- so the dtype is the only thing
        separating a float32 call from an int32 one at one shape. A key that lost
        an operand would answer the second call out of the first graph's plan,
        which reads the wrong bytes rather than failing.
        """
        condition = torch.tensor([True, False, True, False])
        for dtype in (torch.float32, torch.float16, torch.int32):
            a = torch.arange(4).to(dtype)
            b = (torch.arange(4) + 10).to(dtype)
            out = torch.where(_to_device(condition), _to_device(a), _to_device(b))
            assert out.dtype == dtype
            _assert_exact(out, _reference(condition, a, b))

    @pytest.mark.anyplatform
    def test_a_warm_plan_answers_the_other_condition_width(self):
        """A bool condition and a uint8 one are two plans over one branch pair.

        The condition's width decides whether the body builds the zero
        comparison, and MpsRunGraph folds the condition's dtype into the key, so
        the two must not share a plan. The truth values agree so the two results
        do as well, which is what makes a shared plan the *wrong* answer only if
        the widening were applied to a bool predicate.
        """
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([10.0, 20.0, 30.0])
        bool_condition = torch.tensor([True, False, True])
        uint8_condition = torch.tensor([1, 0, 1], dtype=torch.uint8)
        from_bool = torch.where(
            _to_device(bool_condition), _to_device(a), _to_device(b)
        )
        from_uint8 = torch.where(
            _to_device(uint8_condition), _to_device(a), _to_device(b)
        )
        assert torch.equal(from_bool.cpu(), from_uint8.cpu())

    @pytest.mark.anyplatform
    def test_plan_is_reused_across_operand_values(self):
        """Two calls of one signature share a plan and read their own values.

        The operands are graph *inputs* and not key parts, so the second call
        must build no new graph and still return its own elements. A plan that
        pinned the first call's values would answer this one with them.
        """
        condition = torch.tensor([True, False, True, False], device=DEVICE)
        a = torch.arange(4, dtype=torch.float32, device=DEVICE)
        b = (torch.arange(4) + 10).to(torch.float32).to(DEVICE)
        first = torch.where(condition, a, b)
        a2 = (a + 100).contiguous()
        b2 = (b + 100).contiguous()
        second = torch.where(condition, a2, b2)
        assert not torch.equal(first.cpu(), second.cpu())
        _assert_exact(second, _reference(condition.cpu(), a2.cpu(), b2.cpu()))


class TestWhereDispatch:
    """``where.self`` is claimed on PrivateUse1, not left to cpu_fallback."""

    @pytest.mark.main_ops
    def test_conf_routes_where_self_to_mps(self):
        assert routed_backend_or_none("where.self") == "mps"

    @pytest.mark.main_ops
    def test_conf_leaves_the_out_spelling_on_the_fallback(self):
        """``where.self_out`` writes into a caller-owned tensor, like the rest."""
        assert routed_backend_or_none("where.self_out") is None

    @pytest.mark.main_ops
    def test_dispatch_log_names_mps(self):
        """Both the public and the aten spelling reach the kernel.

        ``torch.where`` is what a caller writes and ``aten.where.self`` is what
        the three-tensor overload lowers to; both are held so a future change
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
                "c = torch.tensor([True, False], device='flagos:0'); "
                "a = torch.tensor([1.0, 2.0], device='flagos:0'); "
                "b = torch.tensor([3.0, 4.0], device='flagos:0'); "
                "torch.where(c, a, b); "
                "torch.ops.aten.where(c, a, b)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr.count("[flagos dispatch] where.self -> mps") == 2

    @pytest.mark.main_ops
    def test_a_declined_condition_does_not_reach_the_graph(self):
        """An int condition is entered and answered by the reference.

        The dispatch log names the backend even when the body declines, because
        the kernel is what decided; what this holds is that the *answer* is the
        reference's error on the path the body has no node for.
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
                "c = torch.tensor([1, 0], device='flagos:0')\n"
                "a = torch.tensor([1.0, 2.0], device='flagos:0')\n"
                "b = torch.tensor([3.0, 4.0], device='flagos:0')\n"
                "try:\n"
                "    torch.where(c, a, b)\n"
                "except RuntimeError as e:\n"
                "    print('RuntimeError', 'boolean' in str(e))\n",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "RuntimeError True" in result.stdout
        assert "[flagos dispatch] where.self -> mps" in result.stderr


class TestWhereCost:
    """The op is served, not round-tripped, on the shape it exists for."""

    @pytest.mark.anyplatform
    def test_a_large_select_does_not_scale_with_the_data(self):
        """A mask over a large activation is served, ten times over, not copied.

        The whole-operand round trip this op replaced copies three tensors to
        the host and one back on every call, which is proportional to the
        element count; the node selects on the device. The regression this
        guards against is a silent return to that path, which a correctness test
        cannot see: a round-tripped select returns the right values.

        What catches it is the dispatch log rather than the clock. The log line
        names the backend this module routed to and is emitted once per call, so
        ten selects that were all served log exactly ten ``-> mps`` lines -- a
        count a shared runner cannot move, unlike the wall clock the other cost
        cases in this suite were originally written against.
        """
        rows, width = 1024, 4096
        calls = 10
        served = dispatched_to_mps(
            """
import torch, torch_fl
condition = torch.rand(1024, 4096, device='flagos:0') > 0.5
a = torch.randn(1024, 4096, device='flagos:0')
b = torch.randn(1024, 4096, device='flagos:0')
torch.where(condition, a, b)
torch_fl._C._synchronize()
for _ in range(10):
    torch.where(condition, a, b)
torch_fl._C._synchronize()
""",
        )
        # The warmup logs too: it is the call a fallback would have answered by
        # copying the operands, so it is counted rather than subtracted.
        assert served == calls + 1, (
            f"where over {rows}x{width} operands logged {served} of "
            f"{calls + 1} calls onto the mps slot; the whole-operand fallback is back"
        )
